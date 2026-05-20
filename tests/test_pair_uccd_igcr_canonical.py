from __future__ import annotations

import numpy as np

from xquces.gcr.igcr import IGCRSpinRestrictedParameterization
from xquces.gcr.pair_uccd_reference import (
    GCR2PairUCCDParameterization,
    GCR2ProductPairUCCDParameterization,
    GCR3PairUCCDParameterization,
    GCR3ProductPairUCCDParameterization,
    GCR4PairUCCDParameterization,
    GCR4ProductPairUCCDParameterization,
    GCRPairUCCDParameterization,
    PairUCCDIGCRParameterization,
)


def _block_metadata(parameterization):
    return tuple(
        (block.name, block.start, block.stop, block.shape, block.kind)
        for block in parameterization._composite.parameter_blocks()
    )


def test_pair_uccd_igcr_wrappers_use_canonical_order_parameterization():
    cases = (
        (GCR2PairUCCDParameterization, 2, "exponential", {}),
        (GCR3PairUCCDParameterization, 3, "exponential", {}),
        (GCR4PairUCCDParameterization, 4, "exponential", {}),
        (GCR2ProductPairUCCDParameterization, 2, "product", {"layers": 2}),
        (GCR3ProductPairUCCDParameterization, 3, "product", {}),
        (GCR4ProductPairUCCDParameterization, 4, "product", {}),
    )

    rng = np.random.default_rng(240)
    for cls, order, reference_kind, kwargs in cases:
        wrapper = cls(norb=4, nocc=2, **kwargs)
        canonical = PairUCCDIGCRParameterization(
            norb=4,
            nocc=2,
            order=order,
            reference_kind=reference_kind,
            **kwargs,
        )

        assert isinstance(wrapper, PairUCCDIGCRParameterization)
        assert isinstance(wrapper.ansatz_parameterization, IGCRSpinRestrictedParameterization)
        assert wrapper.ansatz_parameterization.order == order
        assert wrapper.n_params == canonical.n_params
        assert _block_metadata(wrapper) == _block_metadata(canonical)

        params = rng.normal(scale=1.0e-3, size=wrapper.n_params)
        np.testing.assert_allclose(
            wrapper.state_from_parameters(params),
            canonical.state_from_parameters(params),
            atol=1.0e-14,
            rtol=0.0,
        )

        _, ansatz_params = wrapper.split_parameters(params)
        legacy_ansatz = wrapper.ansatz_from_parameters(ansatz_params)
        assert not legacy_ansatz.__class__.__name__.startswith("IGCRAnsatz")


def test_gcr_pair_uccd_facade_still_selects_compatibility_wrapper():
    facade = GCRPairUCCDParameterization(
        norb=4,
        nocc=2,
        order=3,
        reference_kind="product",
    )

    implementation = facade.implementation

    assert isinstance(implementation, GCR3ProductPairUCCDParameterization)
    assert isinstance(implementation, PairUCCDIGCRParameterization)
    assert implementation.ansatz_parameterization.order == 3
