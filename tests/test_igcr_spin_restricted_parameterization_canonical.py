import numpy as np
import pytest

from xquces.gcr import (
    IGCRAnsatz,
    IGCRSpinRestrictedParameterization,
    IGCR2Ansatz,
    IGCR2LayeredAnsatz,
    IGCR2SpinRestrictedParameterization,
    IGCR3Ansatz,
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedParameterization,
    IGCR4Ansatz,
    IGCR4LayeredAnsatz,
    IGCR4SpinRestrictedParameterization,
)


def _zero_ansatz(parameterization):
    return parameterization.ansatz_from_parameters(np.zeros(parameterization.n_params))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_generic_spin_restricted_parameterization_returns_canonical_ansatz(order):
    parameterization = IGCRSpinRestrictedParameterization(
        norb=4,
        nocc=2,
        order=order,
        layers=2,
    )

    ansatz = _zero_ansatz(parameterization)

    assert isinstance(ansatz, IGCRAnsatz)
    assert ansatz.order == order
    assert ansatz.n_layers == 2
    assert ansatz.nocc == 2


@pytest.mark.parametrize(
    "wrapper_cls, legacy_types, order, scale",
    [
        (IGCR2SpinRestrictedParameterization, (IGCR2Ansatz, IGCR2LayeredAnsatz), 2, 1.0),
        (IGCR3SpinRestrictedParameterization, (IGCR3Ansatz, IGCR3LayeredAnsatz), 3, 3.0),
        (IGCR4SpinRestrictedParameterization, (IGCR4Ansatz, IGCR4LayeredAnsatz), 4, 3.0),
    ],
)
def test_order_specific_wrappers_return_legacy_ansatz(wrapper_cls, legacy_types, order, scale):
    generic = IGCRSpinRestrictedParameterization(
        norb=4,
        nocc=2,
        order=order,
        layers=2,
        right_orbital_chart_override=None,
        left_right_ov_relative_scale=scale,
    )
    wrapper = wrapper_cls(norb=4, nocc=2, layers=2)

    assert wrapper.order == order
    assert wrapper.n_params == generic.n_params

    ansatz = _zero_ansatz(wrapper)
    assert isinstance(ansatz, legacy_types)
    assert ansatz.to_generic().order == order
    assert ansatz.to_generic().n_layers == 2


def test_generic_parameterization_no_longer_exposes_facade_forwarding():
    parameterization = IGCRSpinRestrictedParameterization(norb=4, nocc=2, order=3)

    assert not hasattr(parameterization, "implementation")
    with pytest.raises(AttributeError):
        getattr(parameterization, "definitely_not_a_forwarded_attribute")


@pytest.mark.parametrize("order", [2, 3, 4])
def test_generic_parameterization_inverts_its_own_canonical_ansatz(order):
    parameterization = IGCRSpinRestrictedParameterization(
        norb=4,
        nocc=2,
        order=order,
        layers=2,
    )
    params = np.zeros(parameterization.n_params)
    ansatz = parameterization.ansatz_from_parameters(params)

    recovered = parameterization.parameters_from_ansatz(ansatz)

    np.testing.assert_allclose(recovered, params, atol=1e-12)
