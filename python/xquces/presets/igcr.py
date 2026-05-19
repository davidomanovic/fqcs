from __future__ import annotations

from typing import Literal

from xquces.ansatz import (
    DiagonalCorrelatorGate,
    GateSequenceParameterization,
    OrbitalRotationGate,
)
from xquces.gcr.charts import GCR2TraceFixedFullUnitaryChart
from xquces.gcr.igcr import (
    IGCR2SpinBalancedParameterization,
    IGCRSpinRestrictedParameterization,
    _right_unitary_from_left_and_final,
)
from xquces.gcr.restricted_model import (
    IGCR2Ansatz,
    IGCR2SpinRestrictedSpec,
    IGCR3Ansatz,
    IGCR3SpinRestrictedSpec,
    IGCR4Ansatz,
    IGCR4SpinRestrictedSpec,
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
    """Construct an iGCR parameterization without naming a concrete class."""

    order = int(order)
    if backend not in {"legacy", "sequence"}:
        raise ValueError("backend must be 'legacy' or 'sequence'")
    if spin == "balanced":
        if backend == "sequence":
            raise ValueError("sequence backend currently supports spin='restricted'")
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

    if backend == "sequence":
        if layers != 1:
            raise ValueError("sequence backend currently supports layers=1")
        if shared_diagonal:
            raise ValueError("sequence backend does not use shared_diagonal")
        legacy = IGCR(
            order,
            norb,
            nocc,
            layers=layers,
            spin=spin,
            reduced=reduced,
            shared_diagonal=shared_diagonal,
            left_chart=left_chart,
            middle_chart=middle_chart,
            right_chart=right_chart,
            real_right_orbital_chart=real_right_orbital_chart,
            backend="legacy",
            **kwargs,
        )
        return _igcr_spin_restricted_gate_sequence(order, legacy)

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
    return IGCRSpinRestrictedParameterization(**options)


def _igcr_spin_restricted_gate_sequence(
    order: int,
    legacy_parameterization,
) -> GateSequenceParameterization:
    order = int(order)
    if order == 2:

        def build(instances):
            left, diagonal, right = instances
            return IGCR2Ansatz(
                diagonal=IGCR2SpinRestrictedSpec(pair=diagonal.pair),
                left=left,
                right=right,
                nocc=legacy_parameterization.nocc,
            )

    elif order == 3:

        def build(instances):
            left, diagonal, final = instances
            right = _right_unitary_from_left_and_final(
                left, final, legacy_parameterization.nocc
            )
            return IGCR3Ansatz(
                diagonal=IGCR3SpinRestrictedSpec(
                    double_params=diagonal.double_params,
                    pair_values=diagonal.pair_values,
                    tau=diagonal.tau,
                    omega_values=diagonal.omega_values,
                ),
                left=left,
                right=right,
                nocc=legacy_parameterization.nocc,
            )

    elif order == 4:

        def build(instances):
            left, diagonal, final = instances
            right = _right_unitary_from_left_and_final(
                left, final, legacy_parameterization.nocc
            )
            return IGCR4Ansatz(
                diagonal=IGCR4SpinRestrictedSpec(
                    double_params=diagonal.double_params,
                    pair_values=diagonal.pair_values,
                    tau=diagonal.tau,
                    omega_values=diagonal.omega_values,
                    eta_values=diagonal.eta_values,
                    rho_values=diagonal.rho_values,
                    sigma_values=diagonal.sigma_values,
                ),
                left=left,
                right=right,
                nocc=legacy_parameterization.nocc,
            )

    else:
        raise ValueError("order must be 2, 3, or 4")

    return GateSequenceParameterization(
        gates=(
            OrbitalRotationGate(
                "left",
                legacy_parameterization.left_orbital_chart,
                legacy_parameterization.norb,
            ),
            DiagonalCorrelatorGate(
                "diagonal",
                legacy_parameterization.diagonal_chart,
            ),
            OrbitalRotationGate(
                "right",
                legacy_parameterization.right_orbital_chart,
                legacy_parameterization.norb,
            ),
        ),
        ansatz_builder=build,
        native_parameters_from_public=legacy_parameterization._native_parameters_from_public,
        ansatz_parameters_from_instance=legacy_parameterization.parameters_from_ansatz,
        default_nelec=(legacy_parameterization.nocc, legacy_parameterization.nocc),
    )


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
