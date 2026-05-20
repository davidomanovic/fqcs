import numpy as np
import pytest

from xquces.ansatz import (
    GateSequenceParameterization,
    make_state_jacobian,
    make_state_subspace_jacobian,
    make_state_vjp,
)
from xquces.gcr import IGCRSpinRestrictedParameterization
from xquces.gcr.references import CompositeReferenceAnsatzParameterization
from xquces.gcr.restricted_jacobian import (
    make_restricted_gcr_jacobian,
    make_restricted_gcr_subspace_jacobian,
    make_restricted_gcr_vjp,
)
from xquces.jacobian.restricted_igcr import (
    make_restricted_gcr_jacobian as make_restricted_gcr_jacobian_backend,
)
from xquces.states import hartree_fock_state


class _PhaseReferenceParameterization:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.complex128)
        self.n_params = 1

    def state_from_parameters(self, params):
        params = np.asarray(params, dtype=np.float64)
        return np.exp(1j * params[0]) * self.state

    def state_jacobian_from_parameters(self, params):
        return (1j * self.state_from_parameters(params))[:, None]


@pytest.mark.parametrize("order", [2, 3, 4])
def test_canonical_dispatch_matches_restricted_backend(order):
    parameterization = IGCRSpinRestrictedParameterization(norb=4, nocc=2, order=order)
    nelec = (2, 2)
    reference = hartree_fock_state(4, nelec)
    rng = np.random.default_rng(125 + order)
    params = rng.normal(scale=1.0e-3, size=parameterization.n_params)
    directions = rng.normal(size=(parameterization.n_params, 3))
    cotangent = rng.normal(size=reference.size) + 1j * rng.normal(size=reference.size)

    np.testing.assert_allclose(
        make_state_jacobian(parameterization, reference, nelec)(params),
        make_restricted_gcr_jacobian(parameterization, reference, nelec)(params),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        make_state_subspace_jacobian(parameterization, reference, nelec)(
            params,
            directions,
        ),
        make_restricted_gcr_subspace_jacobian(parameterization, reference, nelec)(
            params,
            directions,
        ),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        make_state_vjp(parameterization, reference, nelec)(params, cotangent),
        make_restricted_gcr_vjp(parameterization, reference, nelec)(
            params,
            cotangent,
        ),
        atol=1.0e-12,
    )


def test_sequence_dispatch_uses_canonical_owner_backend():
    parameterization = IGCRSpinRestrictedParameterization(
        norb=4,
        nocc=2,
        order=3,
        layers=2,
    )
    sequence = parameterization.to_gate_sequence()
    nelec = (2, 2)
    reference = hartree_fock_state(4, nelec)
    rng = np.random.default_rng(129)
    params = rng.normal(scale=1.0e-3, size=parameterization.n_params)

    np.testing.assert_allclose(
        make_state_jacobian(sequence, reference, nelec)(params),
        make_state_jacobian(parameterization, reference, nelec)(params),
        atol=1.0e-12,
    )


def test_composite_dispatch_uses_reference_ansatz_path():
    ansatz_parameterization = IGCRSpinRestrictedParameterization(
        norb=4,
        nocc=2,
        order=2,
    )
    nelec = (2, 2)
    reference = _PhaseReferenceParameterization(hartree_fock_state(4, nelec))
    composite = CompositeReferenceAnsatzParameterization(
        reference_parameterization=reference,
        ansatz_parameterization=ansatz_parameterization,
        nelec=nelec,
    )
    rng = np.random.default_rng(130)
    params = rng.normal(scale=1.0e-3, size=composite.n_params)

    jacobian = make_state_jacobian(composite, None, None)(params)

    assert jacobian.shape == (reference.state.size, composite.n_params)
    np.testing.assert_allclose(
        composite.state_jacobian_from_parameters(params),
        jacobian,
        atol=1.0e-12,
    )


def test_unsupported_sequence_jacobian_has_clear_error():
    sequence = GateSequenceParameterization(gates=())

    with pytest.raises(NotImplementedError, match="Gate-sequence state Jacobian"):
        make_state_jacobian(sequence, np.ones(1, dtype=np.complex128), (0, 0))


def test_legacy_restricted_jacobian_facade_points_to_backend():
    assert make_restricted_gcr_jacobian is make_restricted_gcr_jacobian_backend
