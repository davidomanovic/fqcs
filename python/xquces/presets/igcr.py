from __future__ import annotations

from typing import Literal

from xquces.gcr.charts import GCR2TraceFixedFullUnitaryChart
from xquces.gcr.igcr import (
    IGCR2SpinBalancedParameterization,
    IGCRSpinRestrictedParameterization,
)
from xquces.gcr.product_pair_uccd import (
    PairUCCDStateParameterization,
    ProductPairUCCDStateParameterization,
    SlaterPairUCCDStateParameterization,
)
from xquces.gcr.references import CompositeReferenceAnsatzParameterization


def IGCR(
    order: int,
    norb: int,
    nocc: int,
    *,
    layers: int = 1,
    spin: Literal["restricted", "balanced"] = "restricted",
    reduced: bool = True,
    shared_diagonal: bool = False,
    left_chart=None,
    middle_chart=None,
    right_chart=None,
    real_right_orbital_chart: bool = False,
    backend: Literal["legacy", "sequence"] = "legacy",
    **kwargs,
):
    order = int(order)
    if backend not in {"legacy", "sequence"}:
        raise ValueError("backend must be 'legacy' or 'sequence'")
    if spin == "balanced":
        if backend == "sequence":
            raise NotImplementedError(
                "sequence backend currently supports spin='restricted'"
            )
        if order != 2:
            raise ValueError("spin-balanced iGCR presets currently support order=2")
        if layers != 1:
            raise ValueError("spin-balanced iGCR presets currently support layers=1")
        if right_chart is not None or real_right_orbital_chart:
            raise ValueError(
                "spin-balanced iGCR presets currently use the default right chart"
            )
        options = {
            "norb": norb,
            "nocc": nocc,
            **kwargs,
        }
        if left_chart is not None:
            options["left_orbital_chart"] = left_chart
        return IGCR2SpinBalancedParameterization(**options)

    if spin != "restricted":
        raise ValueError("spin must be 'restricted' or 'balanced'")

    options = {
        "norb": norb,
        "nocc": nocc,
        "order": order,
        "layers": layers,
        "shared_diagonal": shared_diagonal,
        "reduce_cubic_gauge": bool(reduced),
        "reduce_quartic_gauge": bool(reduced),
        "real_right_orbital_chart": real_right_orbital_chart,
        **kwargs,
    }
    if left_chart is not None:
        options["left_orbital_chart"] = left_chart
    if middle_chart is not None:
        options["middle_orbital_chart"] = middle_chart
    if right_chart is not None:
        options["right_orbital_chart_override"] = right_chart
    parameterization = IGCRSpinRestrictedParameterization(**options)
    if backend == "sequence":
        return parameterization.to_gate_sequence()
    return parameterization


def PairUCCD_GCR(
    order: int,
    norb: int,
    nocc: int,
    *,
    reference_kind: Literal["exponential", "product", "slater"] = "exponential",
    nelec: tuple[int, int] | None = None,
    spin: Literal["restricted", "balanced"] = "restricted",
    right_chart=None,
    **kwargs,
) -> CompositeReferenceAnsatzParameterization:
    """Compose a pair-UCCD reference parameterization with an iGCR ansatz."""

    if spin != "restricted":
        raise ValueError("PairUCCD_GCR currently composes spin-restricted iGCR")

    if nelec is None:
        nelec = (int(nocc), int(nocc))
    else:
        nelec = tuple(int(x) for x in nelec)

    reference_types = {
        "exponential": PairUCCDStateParameterization,
        "product": ProductPairUCCDStateParameterization,
        "slater": SlaterPairUCCDStateParameterization,
    }
    try:
        reference_type = reference_types[reference_kind]
    except KeyError as exc:
        raise ValueError(
            "reference_kind must be 'exponential', 'product', or 'slater'"
        ) from exc

    reference = reference_type(norb=norb, nelec=nelec)
    ansatz = IGCR(
        order,
        norb,
        nocc,
        spin=spin,
        right_chart=(
            GCR2TraceFixedFullUnitaryChart() if right_chart is None else right_chart
        ),
        **kwargs,
    )
    return CompositeReferenceAnsatzParameterization(reference, ansatz, nelec)
