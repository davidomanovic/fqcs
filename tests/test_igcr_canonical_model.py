import numpy as np
import pytest

from xquces.gcr import (
    IGCRAnsatz,
    IGCRDiagonalCoefficients,
    IGCR2Ansatz,
    IGCR2LayeredAnsatz,
    IGCR2SpinBalancedParameterization,
    IGCR2SpinRestrictedParameterization,
    IGCR2SpinRestrictedSpec,
    IGCR3Ansatz,
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedParameterization,
    IGCR3SpinRestrictedSpec,
    IGCR4Ansatz,
    IGCR4LayeredAnsatz,
    IGCR4SpinRestrictedParameterization,
    IGCR4SpinRestrictedSpec,
)
from xquces.gcr.canonical import (
    as_layered_igcr_ansatz,
    lift_igcr2_to_igcr3,
    lift_igcr2_to_igcr4,
    lift_igcr3_to_igcr4,
)
from xquces.gcr.igcr import (
    igcr3_from_igcr2_ansatz,
    igcr4_from_igcr2_ansatz,
    igcr4_from_igcr3_ansatz,
)
from xquces.states import hartree_fock_state


def _random_unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    phases = np.diag(r) / np.abs(np.diag(r))
    return q * phases


def test_diagonal_coefficients_roundtrip_igcr2_spec():
    pair = np.array(
        [
            [0.0, 0.1, -0.2, 0.3],
            [0.1, 0.0, 0.4, -0.5],
            [-0.2, 0.4, 0.0, 0.6],
            [0.3, -0.5, 0.6, 0.0],
        ]
    )
    spec = IGCR2SpinRestrictedSpec(pair=pair)

    coeffs = IGCRDiagonalCoefficients.from_igcr2_spec(spec)
    out = coeffs.to_igcr2_spec()

    np.testing.assert_allclose(out.to_standard().pair_params, spec.to_standard().pair_params)
    assert coeffs.order == 2
    assert coeffs.omega_values.size == 4
    assert coeffs.eta_values.size == 6
    assert not np.any(coeffs.tau)
    assert not np.any(coeffs.omega_values)
    assert not np.any(coeffs.eta_values)
    assert not np.any(coeffs.rho_values)
    assert not np.any(coeffs.sigma_values)


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


def test_diagonal_coefficients_roundtrip_igcr4_spec():
    norb = 4
    spec = IGCR4SpinRestrictedSpec(
        double_params=np.array([0.1, -0.2, 0.3, -0.4]),
        pair_values=np.arange(6, dtype=float) * 0.01,
        tau=np.arange(norb * norb, dtype=float).reshape(norb, norb) * 0.001,
        omega_values=np.arange(4, dtype=float) * 0.02,
        eta_values=np.arange(6, dtype=float) * 0.003,
        rho_values=np.arange(12, dtype=float) * 0.004,
        sigma_values=np.array([0.05]),
    )

    coeffs = IGCRDiagonalCoefficients.from_igcr4_spec(spec)
    out = coeffs.to_igcr4_spec()

    np.testing.assert_allclose(out.full_double(), spec.full_double())
    np.testing.assert_allclose(out.pair_values, spec.pair_values)
    np.testing.assert_allclose(out.tau_matrix(), spec.tau_matrix())
    np.testing.assert_allclose(out.omega_vector(), spec.omega_vector())
    np.testing.assert_allclose(out.eta_vector(), spec.eta_vector())
    np.testing.assert_allclose(out.rho_vector(), spec.rho_vector())
    np.testing.assert_allclose(out.sigma_vector(), spec.sigma_vector())


def test_igcr2_legacy_ansatz_roundtrip_through_generic_preserves_state():
    rng = np.random.default_rng(2468)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    pair = rng.normal(scale=0.01, size=(norb, norb))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    ansatz = IGCR2Ansatz(
        diagonal=IGCR2SpinRestrictedSpec(pair=pair),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=nocc,
    )

    generic = ansatz.to_generic()
    assert isinstance(generic, IGCRAnsatz)
    assert generic.order == 2
    assert generic.n_layers == 1

    legacy = IGCR2Ansatz.from_generic(generic)
    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        legacy.apply(ref, nelec),
        generic.apply(ref, nelec),
        atol=1e-12,
    )


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


def test_igcr4_legacy_ansatz_roundtrip_through_generic_preserves_state():
    rng = np.random.default_rng(4321)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    spec = IGCR4SpinRestrictedSpec(
        double_params=rng.normal(scale=0.01, size=norb),
        pair_values=rng.normal(scale=0.01, size=6),
        tau=rng.normal(scale=0.01, size=(norb, norb)),
        omega_values=rng.normal(scale=0.01, size=4),
        eta_values=rng.normal(scale=0.01, size=6),
        rho_values=rng.normal(scale=0.01, size=12),
        sigma_values=rng.normal(scale=0.01, size=1),
    )
    ansatz = IGCR4Ansatz(
        diagonal=spec,
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=nocc,
    )

    generic = ansatz.to_generic()
    assert isinstance(generic, IGCRAnsatz)
    assert generic.order == 4
    assert generic.n_layers == 1

    legacy = IGCR4Ansatz.from_generic(generic)
    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        legacy.apply(ref, nelec),
        generic.apply(ref, nelec),
        atol=1e-12,
    )


def test_canonical_layering_splits_one_layer_diagonal_and_preserves_state():
    rng = np.random.default_rng(111)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    spec = IGCR4SpinRestrictedSpec(
        double_params=rng.normal(scale=0.01, size=norb),
        pair_values=rng.normal(scale=0.01, size=6),
        tau=rng.normal(scale=0.01, size=(norb, norb)),
        omega_values=rng.normal(scale=0.01, size=4),
        eta_values=rng.normal(scale=0.01, size=6),
        rho_values=rng.normal(scale=0.01, size=12),
        sigma_values=rng.normal(scale=0.01, size=1),
    )
    ansatz = IGCR4Ansatz(
        diagonal=spec,
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=nocc,
    ).to_generic()

    layered = as_layered_igcr_ansatz(ansatz, 3, order=4)
    assert layered.n_layers == 3
    for diagonal in layered.diagonals:
        np.testing.assert_allclose(diagonal.full_double(), ansatz.diagonals[0].full_double() / 3)
        np.testing.assert_allclose(diagonal.pair_values, ansatz.diagonals[0].pair_values / 3)
        np.testing.assert_allclose(diagonal.tau_matrix(), ansatz.diagonals[0].tau_matrix() / 3)
        np.testing.assert_allclose(diagonal.omega_vector(), ansatz.diagonals[0].omega_vector() / 3)
        np.testing.assert_allclose(diagonal.eta_vector(), ansatz.diagonals[0].eta_vector() / 3)
        np.testing.assert_allclose(diagonal.rho_vector(), ansatz.diagonals[0].rho_vector() / 3)
        np.testing.assert_allclose(diagonal.sigma_vector(), ansatz.diagonals[0].sigma_vector() / 3)

    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        layered.apply(ref, nelec),
        ansatz.apply(ref, nelec),
        atol=1e-12,
    )


def test_canonical_layering_pads_existing_layered_ansatz_before_final_rotation():
    rng = np.random.default_rng(222)
    norb = 4
    nocc = 2
    identity = np.eye(norb, dtype=np.complex128)
    first = IGCRDiagonalCoefficients.from_igcr3_spec(
        IGCR3SpinRestrictedSpec(
            double_params=rng.normal(scale=0.01, size=norb),
            pair_values=rng.normal(scale=0.01, size=6),
            tau=rng.normal(scale=0.01, size=(norb, norb)),
            omega_values=rng.normal(scale=0.01, size=4),
        )
    )
    generic = IGCRAnsatz(
        order=3,
        diagonals=(first, first),
        rotations=(identity, identity, _random_unitary(rng, norb)),
        nocc=nocc,
    )

    padded = as_layered_igcr_ansatz(generic, 4, order=3)
    assert padded.n_layers == 4
    assert padded.rotations[-1] is not padded.rotations[-2]
    np.testing.assert_allclose(padded.rotations[-1], generic.rotations[-1])
    for diagonal in padded.diagonals[2:]:
        assert not np.any(diagonal.full_double())
        assert not np.any(diagonal.pair_values)
        assert not np.any(diagonal.tau)
        assert not np.any(diagonal.omega_values)


@pytest.mark.parametrize(
    "parameterization_cls, layered_cls",
    [
        (IGCR2SpinRestrictedParameterization, IGCR2LayeredAnsatz),
        (IGCR3SpinRestrictedParameterization, IGCR3LayeredAnsatz),
        (IGCR4SpinRestrictedParameterization, IGCR4LayeredAnsatz),
    ],
)
def test_parameters_from_one_layer_ansatz_uses_canonical_layering_adapter(
    parameterization_cls,
    layered_cls,
):
    one_layer = parameterization_cls(norb=4, nocc=2, layers=1)
    target = parameterization_cls(norb=4, nocc=2, layers=3)
    one_layer_ansatz = one_layer.ansatz_from_parameters(np.zeros(one_layer.n_params))

    params = target.parameters_from_ansatz(one_layer_ansatz)
    embedded = target.ansatz_from_parameters(params)

    assert isinstance(embedded, layered_cls)
    assert embedded.layers == 3
    assert embedded.to_generic().n_layers == 3


def test_igcr2_spin_restricted_parameterization_returns_legacy_via_generic_builders():
    param = IGCR2SpinRestrictedParameterization(norb=4, nocc=2, layers=2)
    params = np.zeros(param.n_params)
    ansatz = param.ansatz_from_parameters(params)

    assert isinstance(ansatz, IGCR2LayeredAnsatz)
    generic = ansatz.to_generic()
    assert isinstance(generic, IGCRAnsatz)
    assert generic.order == 2
    assert generic.n_layers == 2


def test_igcr2_spin_balanced_parameterization_remains_legacy_only():
    param = IGCR2SpinBalancedParameterization(norb=4, nocc=2)
    params = np.zeros(param.n_params)
    ansatz = param.ansatz_from_parameters(params)

    assert isinstance(ansatz, IGCR2Ansatz)
    assert ansatz.is_spin_balanced


def test_canonical_lift_igcr2_to_igcr3_matches_legacy_constructor():
    rng = np.random.default_rng(303)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    pair = rng.normal(scale=0.02, size=(norb, norb))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    ansatz = IGCR2Ansatz(
        diagonal=IGCR2SpinRestrictedSpec(pair=pair),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=nocc,
    )

    lifted = lift_igcr2_to_igcr3(
        ansatz.to_generic(),
        tau_scale=0.7,
        omega_scale=-0.3,
    )
    expected = IGCR3Ansatz.from_igcr2_ansatz(
        ansatz,
        tau_scale=0.7,
        omega_scale=-0.3,
    )
    public = igcr3_from_igcr2_ansatz(
        ansatz,
        tau_scale=0.7,
        omega_scale=-0.3,
    )

    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        lifted.apply(ref, nelec),
        expected.apply(ref, nelec),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        public.apply(ref, nelec),
        expected.apply(ref, nelec),
        atol=1e-12,
    )


def test_canonical_lift_igcr3_to_igcr4_matches_legacy_constructor():
    rng = np.random.default_rng(404)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    ansatz = IGCR3Ansatz(
        diagonal=IGCR3SpinRestrictedSpec(
            double_params=rng.normal(scale=0.01, size=norb),
            pair_values=rng.normal(scale=0.01, size=6),
            tau=rng.normal(scale=0.01, size=(norb, norb)),
            omega_values=rng.normal(scale=0.01, size=4),
        ),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=nocc,
    )

    lifted = lift_igcr3_to_igcr4(
        ansatz.to_generic(),
        eta_scale=0.5,
        rho_scale=-0.25,
        sigma_scale=0.125,
    )
    expected = IGCR4Ansatz.from_igcr3_ansatz(
        ansatz,
        eta_scale=0.5,
        rho_scale=-0.25,
        sigma_scale=0.125,
    )
    public = igcr4_from_igcr3_ansatz(
        ansatz,
        eta_scale=0.5,
        rho_scale=-0.25,
        sigma_scale=0.125,
    )

    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        lifted.apply(ref, nelec),
        expected.apply(ref, nelec),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        public.apply(ref, nelec),
        expected.apply(ref, nelec),
        atol=1e-12,
    )


def test_canonical_lift_igcr2_to_igcr4_matches_legacy_constructor():
    rng = np.random.default_rng(505)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    pair = rng.normal(scale=0.02, size=(norb, norb))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    ansatz = IGCR2Ansatz(
        diagonal=IGCR2SpinRestrictedSpec(pair=pair),
        left=_random_unitary(rng, norb),
        right=_random_unitary(rng, norb),
        nocc=nocc,
    )

    lifted = lift_igcr2_to_igcr4(
        ansatz.to_generic(),
        tau_scale=0.7,
        omega_scale=-0.3,
        eta_scale=0.5,
        rho_scale=-0.25,
        sigma_scale=0.125,
    )
    expected = IGCR4Ansatz.from_igcr2_ansatz(
        ansatz,
        tau_scale=0.7,
        omega_scale=-0.3,
        eta_scale=0.5,
        rho_scale=-0.25,
        sigma_scale=0.125,
    )
    public = igcr4_from_igcr2_ansatz(
        ansatz,
        tau_scale=0.7,
        omega_scale=-0.3,
        eta_scale=0.5,
        rho_scale=-0.25,
        sigma_scale=0.125,
    )

    ref = hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        lifted.apply(ref, nelec),
        expected.apply(ref, nelec),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        public.apply(ref, nelec),
        expected.apply(ref, nelec),
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


def test_igcr4_parameterization_returns_legacy_via_generic_builders():
    param = IGCR4SpinRestrictedParameterization(norb=4, nocc=2, layers=2)
    params = np.zeros(param.n_params)
    ansatz = param.ansatz_from_parameters(params)

    assert isinstance(ansatz, IGCR4LayeredAnsatz)
    generic = ansatz.to_generic()
    assert isinstance(generic, IGCRAnsatz)
    assert generic.order == 4
    assert generic.n_layers == 2
