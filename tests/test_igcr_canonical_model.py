import numpy as np

from xquces.gcr import (
    IGCRAnsatz,
    IGCRDiagonalCoefficients,
    IGCR3Ansatz,
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedParameterization,
    IGCR3SpinRestrictedSpec,
)
from xquces.states import hartree_fock_state


def _random_unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    phases = np.diag(r) / np.abs(np.diag(r))
    return q * phases


def test_diagonal_coefficients_roundtrip_igcr3_spec():
    norb = 4
    spec = IGCR3SpinRestrictedSpec(
        double_params=np.array([0.1, -0.2, 0.3, -0.4]),
        pair_values=np.arange(6, dtype=float) * 0.01,
        tau=np.arange(norb * norb, dtype=float).reshape(norb, norb) * 0.001,
        omega_values=np.arange(4, dtype=float) * 0.02,
    )

    coeffs = IGCRDiagonalCoefficients.from_igcr3_spec(spec)
    out = coeffs.to_igcr3_spec()

    np.testing.assert_allclose(out.full_double(), spec.full_double())
    np.testing.assert_allclose(out.pair_values, spec.pair_values)
    np.testing.assert_allclose(out.tau_matrix(), spec.tau_matrix())
    np.testing.assert_allclose(out.omega_vector(), spec.omega_vector())


def test_igcr3_legacy_ansatz_roundtrip_through_generic_preserves_state():
    rng = np.random.default_rng(1234)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    spec = IGCR3SpinRestrictedSpec(
        double_params=rng.normal(scale=0.01, size=norb),
        pair_values=rng.normal(scale=0.01, size=6),
        tau=rng.normal(scale=0.01, size=(norb, norb)),
        omega_values=rng.normal(scale=0.01, size=4),
    )
    ansatz = IGCR3Ansatz(
        diagonal=spec,
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=nocc,
    )

    generic = ansatz.to_generic()
    assert isinstance(generic, IGCRAnsatz)
    assert generic.order == 3
    assert generic.n_layers == 1

    legacy = IGCR3Ansatz.from_generic(generic)
    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        legacy.apply(ref, nelec),
        generic.apply(ref, nelec),
        atol=1e-12,
    )


def test_igcr3_parameterization_returns_legacy_via_generic_builders():
    param = IGCR3SpinRestrictedParameterization(norb=4, nocc=2, layers=2)
    params = np.zeros(param.n_params)
    ansatz = param.ansatz_from_parameters(params)

    assert isinstance(ansatz, IGCR3LayeredAnsatz)
    generic = ansatz.to_generic()
    assert isinstance(generic, IGCRAnsatz)
    assert generic.order == 3
    assert generic.n_layers == 2
