import numpy as np
import pytest

from xquces.gcr import (
    IGCRAnsatz,
    IGCR2Ansatz,
    IGCR2SpinBalancedParameterization,
    IGCR2SpinRestrictedSpec,
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedSpec,
    IGCR4Ansatz,
    IGCR4SpinRestrictedSpec,
)
from xquces.gcr.utils import (
    relabel_igcr_ansatz_orbitals,
    transport_igcr_ansatz_orbitals,
)
from xquces.states import hartree_fock_state


def _random_unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    phases = np.diag(r) / np.abs(np.diag(r))
    return q * phases


def _igcr4_spec(rng, norb):
    return IGCR4SpinRestrictedSpec(
        double_params=rng.normal(scale=0.03, size=norb),
        pair_values=rng.normal(scale=0.03, size=6),
        tau=rng.normal(scale=0.03, size=(norb, norb)),
        omega_values=rng.normal(scale=0.03, size=4),
        eta_values=rng.normal(scale=0.03, size=6),
        rho_values=rng.normal(scale=0.03, size=12),
        sigma_values=rng.normal(scale=0.03, size=1),
    )


def test_canonical_relabel_preserves_shape_metadata_for_one_layer_order4():
    rng = np.random.default_rng(100)
    norb = 4
    ansatz = IGCR4Ansatz(
        diagonal=_igcr4_spec(rng, norb),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=2,
    ).to_generic()

    out = relabel_igcr_ansatz_orbitals(
        ansatz,
        old_for_new=np.array([2, 0, 3, 1]),
        phases=np.array([1.0, -1.0, 1.0, -1.0]),
    )

    assert isinstance(out, IGCRAnsatz)
    assert out.order == ansatz.order
    assert out.nocc == ansatz.nocc
    assert out.n_layers == ansatz.n_layers
    assert out.norb == ansatz.norb


def test_canonical_relabel_matches_legacy_adapter_for_multilayer_order3_state():
    rng = np.random.default_rng(200)
    norb = 4
    nocc = 2
    nelec = (2, 2)
    ansatz = IGCR3LayeredAnsatz(
        diagonals=tuple(
            IGCR3SpinRestrictedSpec(
                double_params=rng.normal(scale=0.02, size=norb),
                pair_values=rng.normal(scale=0.02, size=6),
                tau=rng.normal(scale=0.02, size=(norb, norb)),
                omega_values=rng.normal(scale=0.02, size=4),
            )
            for _ in range(3)
        ),
        rotations=tuple(_random_unitary(rng, norb) for _ in range(4)),
        nocc=nocc,
    )
    old_for_new = np.array([1, 3, 0, 2])
    generic = ansatz.to_generic()

    canonical = relabel_igcr_ansatz_orbitals(generic, old_for_new)
    legacy = relabel_igcr_ansatz_orbitals(ansatz, old_for_new, order=3)

    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        canonical.apply(ref, nelec),
        legacy.apply(ref, nelec),
        atol=1e-12,
    )
    assert canonical.order == 3
    assert canonical.n_layers == 3


def test_canonical_transport_preserves_diagonals_and_layer_count_order2():
    rng = np.random.default_rng(300)
    norb = 4
    pair = rng.normal(scale=0.02, size=(norb, norb))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    ansatz = IGCR2Ansatz(
        diagonal=IGCR2SpinRestrictedSpec(pair=pair),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=2,
    ).to_generic()
    basis_change = _random_unitary(rng, norb)

    out = transport_igcr_ansatz_orbitals(ansatz, basis_change)

    assert out.order == 2
    assert out.n_layers == 1
    np.testing.assert_allclose(out.diagonals[0].pair_values, ansatz.diagonals[0].pair_values)
    np.testing.assert_allclose(out.rotations[0], basis_change.conj().T @ ansatz.rotations[0])
    np.testing.assert_allclose(out.rotations[1], ansatz.rotations[1])


def test_generic_relabel_transport_rejects_spin_balanced_legacy_input():
    param = IGCR2SpinBalancedParameterization(norb=4, nocc=2)
    ansatz = param.ansatz_from_parameters(np.zeros(param.n_params))

    with pytest.raises(TypeError, match="spin-restricted"):
        relabel_igcr_ansatz_orbitals(ansatz, np.array([0, 1, 2, 3]), order=2)

    with pytest.raises(TypeError, match="spin-restricted"):
        transport_igcr_ansatz_orbitals(ansatz, np.eye(4, dtype=np.complex128), order=2)
