import ffsim
import numpy as np

from xquces.gcr.igcr import (
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedParameterization,
    IGCR4LayeredAnsatz,
    IGCR4SpinRestrictedParameterization,
    IGCRSpinRestrictedParameterization,
)
from xquces.gcr.restricted_jacobian import make_restricted_gcr_vjp


def test_high_order_layered_embedding_preserves_state():
    rng = np.random.default_rng(1234)
    ref = ffsim.hartree_fock_state(4, (2, 2))

    for cls, layered_cls in [
        (IGCR3SpinRestrictedParameterization, IGCR3LayeredAnsatz),
        (IGCR4SpinRestrictedParameterization, IGCR4LayeredAnsatz),
    ]:
        one_layer = cls(norb=4, nocc=2, layers=1)
        two_layer = cls(norb=4, nocc=2, layers=2)
        params = rng.normal(scale=1e-3, size=one_layer.n_params)

        ansatz = one_layer.ansatz_from_parameters(params)
        lifted = two_layer.ansatz_from_parameters(
            two_layer.parameters_from_ansatz(ansatz)
        )

        assert isinstance(lifted, layered_cls)
        actual = lifted.apply(ref, (2, 2), copy=True)
        expected = ansatz.apply(ref, (2, 2), copy=True)
        assert np.linalg.norm(actual - expected) < 1e-12


def test_high_order_layered_jacobian_matches_finite_difference():
    rng = np.random.default_rng(5678)
    ref = ffsim.hartree_fock_state(4, (2, 2))
    eps = 1e-6

    for cls in [
        IGCR3SpinRestrictedParameterization,
        IGCR4SpinRestrictedParameterization,
    ]:
        param = cls(norb=4, nocc=2, layers=2)
        fixed = param.apply(ref, (2, 2))
        params = rng.normal(scale=2e-3, size=param.n_params)
        jac = fixed.state_jacobian_from_parameters(params)

        fd_cols = []
        for idx in range(param.n_params):
            plus = params.copy()
            minus = params.copy()
            plus[idx] += eps
            minus[idx] -= eps
            fd_cols.append(
                (fixed.state_from_parameters(plus) - fixed.state_from_parameters(minus))
                / (2.0 * eps)
            )
        fd = np.column_stack(fd_cols)
        assert np.max(np.abs(jac - fd)) < 3e-8

        v = rng.normal(size=ref.size) + 1j * rng.normal(size=ref.size)
        actual = make_restricted_gcr_vjp(param, ref, (2, 2))(params, v)
        expected = 2.0 * (jac.conj().T @ v).real
        assert np.max(np.abs(actual - expected)) < 1e-10


def test_order_selecting_facade_accepts_high_order_layers():
    for order in (3, 4):
        param = IGCRSpinRestrictedParameterization(
            norb=4,
            nocc=2,
            order=order,
            layers=2,
            shared_diagonal=True,
        )
        impl = param.implementation
        assert impl.layers == 2
        assert impl.shared_diagonal
