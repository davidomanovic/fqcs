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
from xquces.gcr.restricted_model import IGCR3Ansatz, IGCR4Ansatz
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz, UCJLayer


def _small_t2(nocc: int = 2, nvirt: int = 3, seed: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t2 = rng.standard_normal((nocc, nocc, nvirt, nvirt)) * 0.1
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    return t2


def _random_unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    phases = np.diag(r) / np.abs(np.diag(r))
    return q * phases


def _random_spin_restricted_ucj(norb: int, rng) -> UCJAnsatz:
    pair = rng.normal(scale=0.02, size=(norb, norb))
    pair = 0.5 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)
    return UCJAnsatz(
        layers=(
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=rng.normal(scale=0.02, size=norb),
                    pair_params=pair,
                ),
                orbital_rotation=_random_unitary(rng, norb),
            ),
        ),
        final_orbital_rotation=_random_unitary(rng, norb),
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


def test_pair_uccd_igcr_t_amplitude_seed_uses_canonical_lifts(monkeypatch):
    def blocked(*args, **kwargs):  # pragma: no cover - exercised only on failure
        raise AssertionError("legacy high-order lift constructor was called")

    monkeypatch.setattr(IGCR3Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr3_ansatz", classmethod(blocked))

    t2 = _small_t2()
    for order in (3, 4):
        param = PairUCCDIGCRParameterization(norb=5, nocc=2, order=order)
        x = param.parameters_from_t_amplitudes(t2)
        assert x.shape == (param.n_params,)


def test_pair_uccd_igcr_ucj_seed_uses_canonical_lifts(monkeypatch):
    def blocked(*args, **kwargs):  # pragma: no cover - exercised only on failure
        raise AssertionError("legacy high-order UCJ lift constructor was called")

    monkeypatch.setattr(IGCR3Ansatz, "from_ucj_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_ucj_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR3Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr3_ansatz", classmethod(blocked))

    ucj = _random_spin_restricted_ucj(5, np.random.default_rng(27))
    for reference_kind in ("exponential", "product"):
        for order in (3, 4):
            param = PairUCCDIGCRParameterization(
                norb=5,
                nocc=2,
                order=order,
                reference_kind=reference_kind,
            )
            x = param.parameters_from_ucj_ansatz(ucj)
            assert x.shape == (param.n_params,)


def test_product_pair_uccd_igcr_t_amplitude_seed_uses_canonical_lifts(monkeypatch):
    def blocked(*args, **kwargs):  # pragma: no cover - exercised only on failure
        raise AssertionError("legacy high-order lift constructor was called")

    monkeypatch.setattr(IGCR3Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr3_ansatz", classmethod(blocked))

    t2 = _small_t2()
    for order in (3, 4):
        param = PairUCCDIGCRParameterization(
            norb=5,
            nocc=2,
            order=order,
            reference_kind="product",
        )
        x = param.parameters_from_t_amplitudes(t2)
        assert x.shape == (param.n_params,)


def test_pair_uccd_igcr_nested_seed_uses_canonical_lifts(monkeypatch):
    def blocked(*args, **kwargs):  # pragma: no cover - exercised only on failure
        raise AssertionError("legacy high-order lift constructor was called")

    monkeypatch.setattr(IGCR3Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr2_ansatz", classmethod(blocked))
    monkeypatch.setattr(IGCR4Ansatz, "from_igcr3_ansatz", classmethod(blocked))

    t2 = _small_t2(seed=26)
    base = GCR2PairUCCDParameterization(norb=5, nocc=2)
    x2 = base.parameters_from_t_amplitudes(t2)

    target3 = GCR3PairUCCDParameterization(norb=5, nocc=2)
    x3 = target3.nested_lift_parameters_from(
        x2,
        base,
        optimize_weights=False,
    )
    assert x3.shape == (target3.n_params,)

    target4 = GCR4PairUCCDParameterization(norb=5, nocc=2)
    x4 = target4.nested_lift_parameters_from(
        x3,
        target3,
        optimize_weights=False,
    )
    assert x4.shape == (target4.n_params,)
