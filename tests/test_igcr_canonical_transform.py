import numpy as np

import xquces.gcr.igcr as legacy_igcr
from xquces.gcr import (
    IGCRAnsatz,
    IGCR2Ansatz,
    IGCR2SpinRestrictedSpec,
    IGCR3Ansatz,
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedParameterization,
    IGCR3SpinRestrictedSpec,
    IGCR4Ansatz,
    IGCR4SpinRestrictedSpec,
)
from xquces.gcr.canonical_transform import (
    relabel_igcr_ansatz_orbitals,
    transport_igcr_ansatz_orbitals,
)


def _random_unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    phases = np.diag(r) / np.abs(np.diag(r))
    return q * phases


def _assert_same_generic_ansatz(left: IGCRAnsatz, right: IGCRAnsatz):
    assert left.order == right.order
    assert left.n_layers == right.n_layers
    assert left.norb == right.norb
    assert left.nocc == right.nocc
    for d_left, d_right in zip(left.diagonals, right.diagonals):
        np.testing.assert_allclose(d_left.full_double(), d_right.full_double())
        np.testing.assert_allclose(d_left.pair_values, d_right.pair_values)
        np.testing.assert_allclose(d_left.tau_matrix(), d_right.tau_matrix())
        np.testing.assert_allclose(d_left.omega_vector(), d_right.omega_vector())
        np.testing.assert_allclose(d_left.eta_vector(), d_right.eta_vector())
        np.testing.assert_allclose(d_left.rho_vector(), d_right.rho_vector())
        np.testing.assert_allclose(d_left.sigma_vector(), d_right.sigma_vector())
    for u_left, u_right in zip(left.rotations, right.rotations):
        np.testing.assert_allclose(u_left, u_right)


def test_legacy_igcr2_relabel_delegates_to_canonical_transform():
    rng = np.random.default_rng(10)
    norb = 4
    pair = rng.normal(scale=0.1, size=(norb, norb))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    ansatz = IGCR2Ansatz(
        diagonal=IGCR2SpinRestrictedSpec(pair=pair),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=2,
    )
    old_for_new = np.array([2, 0, 3, 1])
    phases = np.array([1.0, -1.0, 1.0, -1.0])

    legacy = legacy_igcr.relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
    canonical = relabel_igcr_ansatz_orbitals(ansatz, old_for_new, phases, order=2)

    _assert_same_generic_ansatz(legacy.to_generic(), canonical)


def test_legacy_igcr3_layered_relabel_delegates_to_canonical_transform():
    rng = np.random.default_rng(20)
    norb = 4
    diagonals = tuple(
        IGCR3SpinRestrictedSpec(
            double_params=rng.normal(scale=0.1, size=norb),
            pair_values=rng.normal(scale=0.1, size=6),
            tau=rng.normal(scale=0.1, size=(norb, norb)),
            omega_values=rng.normal(scale=0.1, size=4),
        )
        for _ in range(2)
    )
    ansatz = IGCR3LayeredAnsatz(
        diagonals=diagonals,
        rotations=tuple(_random_unitary(rng, norb) for _ in range(3)),
        nocc=2,
    )
    old_for_new = np.array([1, 3, 0, 2])

    legacy = legacy_igcr.relabel_igcr3_ansatz_orbitals(ansatz, old_for_new)
    canonical = relabel_igcr_ansatz_orbitals(ansatz, old_for_new, order=3)

    _assert_same_generic_ansatz(legacy.to_generic(), canonical)


def test_legacy_igcr4_transport_delegates_to_canonical_transform():
    rng = np.random.default_rng(30)
    norb = 4
    ansatz = IGCR4Ansatz(
        diagonal=IGCR4SpinRestrictedSpec(
            double_params=rng.normal(scale=0.1, size=norb),
            pair_values=rng.normal(scale=0.1, size=6),
            tau=rng.normal(scale=0.1, size=(norb, norb)),
            omega_values=rng.normal(scale=0.1, size=4),
            eta_values=rng.normal(scale=0.1, size=6),
            rho_values=rng.normal(scale=0.1, size=12),
            sigma_values=rng.normal(scale=0.1, size=1),
        ),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=2,
    )
    basis_change = _random_unitary(rng, norb)

    legacy = legacy_igcr.transport_igcr4_ansatz_orbitals(ansatz, basis_change)
    canonical = transport_igcr_ansatz_orbitals(ansatz, basis_change, order=4)

    _assert_same_generic_ansatz(legacy.to_generic(), canonical)


def test_parameter_transfer_uses_patched_legacy_transform_path():
    param = IGCR3SpinRestrictedParameterization(norb=4, nocc=2, layers=2)
    previous = IGCR3SpinRestrictedParameterization(norb=4, nocc=2, layers=1)
    previous_params = np.zeros(previous.n_params)

    transferred = param.transfer_parameters_from(
        previous_params,
        previous_parameterization=previous,
        old_for_new=np.array([2, 0, 3, 1]),
    )
    ansatz = param.ansatz_from_parameters(transferred)

    assert isinstance(ansatz, IGCR3LayeredAnsatz)
    assert ansatz.to_generic().n_layers == 2
