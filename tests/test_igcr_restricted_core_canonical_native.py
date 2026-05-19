import numpy as np

from xquces.gcr import (
    IGCRAnsatz,
    IGCR2LayeredAnsatz,
    IGCR2SpinRestrictedParameterization,
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedParameterization,
    IGCR4LayeredAnsatz,
    IGCR4SpinRestrictedParameterization,
)


def test_spin_restricted_core_builds_canonical_ansatz():
    cases = [
        (2, IGCR2SpinRestrictedParameterization, IGCR2LayeredAnsatz),
        (3, IGCR3SpinRestrictedParameterization, IGCR3LayeredAnsatz),
        (4, IGCR4SpinRestrictedParameterization, IGCR4LayeredAnsatz),
    ]
    for order, param_cls, legacy_cls in cases:
        param = param_cls(norb=4, nocc=2, layers=2)
        params = np.zeros(param.n_params)

        generic = param._layered_core.ansatz_from_parameters(params)
        assert isinstance(generic, IGCRAnsatz)
        assert generic.order == order
        assert generic.n_layers == 2
        assert generic.nocc == 2
        assert generic.norb == 4

        legacy = param.ansatz_from_parameters(params)
        assert isinstance(legacy, legacy_cls)
        assert legacy.to_generic().order == order
        assert legacy.to_generic().n_layers == 2


def test_spin_restricted_core_inverts_canonical_ansatz():
    cases = [
        IGCR2SpinRestrictedParameterization,
        IGCR3SpinRestrictedParameterization,
        IGCR4SpinRestrictedParameterization,
    ]
    for param_cls in cases:
        param = param_cls(norb=4, nocc=2, layers=2)
        params = np.zeros(param.n_params)
        generic = param._layered_core.ansatz_from_parameters(params)

        roundtrip_params = param.parameters_from_ansatz(generic)
        roundtrip = param._layered_core.ansatz_from_parameters(roundtrip_params)

        assert isinstance(roundtrip, IGCRAnsatz)
        assert roundtrip.order == generic.order
        assert roundtrip.n_layers == generic.n_layers
        assert roundtrip.nocc == generic.nocc
        assert roundtrip.norb == generic.norb
