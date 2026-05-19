from __future__ import annotations

import numpy as np

from xquces.ansatz import (
    DiagonalCorrelatorGate,
    GateSequenceParameterization,
    OrbitalRotationGate,
)
from xquces.gcr.igcr import (
    IGCR2Ansatz,
    IGCR2SpinRestrictedSpec,
    IGCR2SpinBalancedParameterization,
    IGCR2SpinRestrictedParameterization,
    IGCR3SpinRestrictedParameterization,
    parameter_blocks,
    parameter_view,
)
from xquces.gcr.product_pair_uccd import PairUCCDStateParameterization
from xquces.gcr.references import CompositeReferenceAnsatzParameterization
from xquces.charts.reductions import (
    IGCR3CubicReduction as ChartIGCR3CubicReduction,
    IGCR4QuarticReduction as ChartIGCR4QuarticReduction,
)
from xquces.charts.diagonal import (
    RestrictedPairChart,
    RestrictedCubicChart,
    RestrictedQuarticChart,
)
from xquces.presets import IGCR, PairUCCD_GCR


def _blocks_by_name(parameterization):
    return {block.name: block for block in parameter_blocks(parameterization)}


def test_layered_igcr3_parameter_blocks_have_shapes_and_kinds():
    parameterization = IGCR3SpinRestrictedParameterization(
        norb=4,
        nocc=2,
        layers=2,
    )

    blocks = _blocks_by_name(parameterization)

    assert tuple(blocks) == ("left", "pair", "cubic", "middle", "right")
    assert blocks["left"].shape == (12,)
    assert blocks["left"].kind == "orbital"
    assert blocks["pair"].shape == (2, 6)
    assert blocks["pair"].kind == "diagonal"
    assert blocks["cubic"].shape == (2, 6)
    assert blocks["middle"].shape == (1, 12)
    assert blocks["right"].shape == (8,)
    assert blocks["right"].stop == parameterization.n_params


def test_parameter_view_slices_and_updates_named_blocks():
    parameterization = IGCR3SpinRestrictedParameterization(
        norb=4,
        nocc=2,
        layers=2,
    )
    params = np.arange(parameterization.n_params, dtype=np.float64)
    view = parameter_view(parameterization, params)
    blocks = _blocks_by_name(parameterization)

    np.testing.assert_array_equal(
        view["pair"],
        params[blocks["pair"].slice()].reshape(2, 6),
    )
    updated = view.updated(pair=np.zeros((2, 6), dtype=np.float64))

    np.testing.assert_array_equal(updated[blocks["pair"].slice()], np.zeros(12))
    np.testing.assert_array_equal(updated[blocks["left"].slice()], view.flat("left"))


def test_spin_balanced_igcr2_exposes_diagonal_subblocks():
    parameterization = IGCR2SpinBalancedParameterization(norb=4, nocc=2)

    blocks = parameter_blocks(parameterization)
    assert tuple(block.name for block in blocks) == (
        "left",
        "same_diag",
        "double",
        "same_spin",
        "mixed_spin",
        "right",
    )
    assert sum(block.size for block in blocks) == parameterization.n_params
    assert all(block.kind == "diagonal" for block in blocks[1:-1])


def test_composite_reference_ansatz_blocks_include_reference_prefix():
    parameterization = CompositeReferenceAnsatzParameterization(
        PairUCCDStateParameterization(norb=4, nelec=(2, 2)),
        IGCR3SpinRestrictedParameterization(norb=4, nocc=2),
        nelec=(2, 2),
    )

    blocks = parameterization.parameter_blocks()
    assert blocks[0].name == "reference"
    assert blocks[0].shape == (4,)
    assert blocks[0].kind == "reference"
    assert blocks[-1].stop == parameterization.n_params

    params = np.arange(parameterization.n_params, dtype=np.float64)
    view = parameterization.parameter_view(params)
    np.testing.assert_array_equal(view["reference"], params[:4])


def test_igcr_preset_hides_order_specific_class_names():
    parameterization = IGCR(order=4, norb=4, nocc=2, layers=2)

    assert parameterization.order == 4
    assert tuple(block.name for block in parameterization.parameter_blocks()) == (
        "left",
        "pair",
        "cubic",
        "quartic",
        "middle",
        "right",
    )


def test_restricted_pair_chart_roundtrips_igcr2_diagonal_block():
    parameterization = IGCR2SpinRestrictedParameterization(norb=4, nocc=2)
    chart = parameterization.diagonal_chart
    assert isinstance(chart, RestrictedPairChart)

    params = np.linspace(-0.02, 0.03, chart.n_params)
    coeffs = chart.coefficients_from_parameters(params)
    roundtrip, phase = chart.parameters_from_coefficients(coeffs)

    assert tuple(block.name for block in chart.blocks("diagonal")) == (
        "diagonal.pair",
    )
    assert roundtrip.shape == params.shape
    assert phase.shape == (4,)
    np.testing.assert_allclose(roundtrip, params, atol=1.0e-12)


def test_reductions_are_available_from_chart_module_and_legacy_gcr_module():
    from xquces.gcr.igcr import IGCR3CubicReduction, IGCR4QuarticReduction

    assert IGCR3CubicReduction is ChartIGCR3CubicReduction
    assert IGCR4QuarticReduction is ChartIGCR4QuarticReduction
    assert ChartIGCR3CubicReduction(norb=4, nocc=2).n_params == 6
    assert ChartIGCR4QuarticReduction(norb=4, nocc=2).n_params == 3


def test_restricted_cubic_chart_roundtrips_igcr3_diagonal_block():
    parameterization = IGCR3SpinRestrictedParameterization(norb=4, nocc=2)
    chart = parameterization.diagonal_chart
    assert isinstance(chart, RestrictedCubicChart)

    params = np.linspace(-0.02, 0.03, chart.n_params)
    coeffs = chart.coefficients_from_parameters(params)
    roundtrip, phase = chart.parameters_from_coefficients(coeffs)

    assert tuple(block.name for block in chart.blocks("diagonal")) == (
        "diagonal.pair",
        "diagonal.cubic",
    )
    assert roundtrip.shape == params.shape
    assert phase.shape == (4,)
    np.testing.assert_allclose(roundtrip, params, atol=1.0e-10)


def test_restricted_quartic_chart_roundtrips_igcr4_diagonal_block():
    parameterization = IGCR(order=4, norb=4, nocc=2).implementation
    chart = parameterization.diagonal_chart
    assert isinstance(chart, RestrictedQuarticChart)

    params = np.linspace(-0.02, 0.03, chart.n_params)
    coeffs = chart.coefficients_from_parameters(params)
    roundtrip, phase = chart.parameters_from_coefficients(coeffs)

    assert tuple(block.name for block in chart.blocks("diagonal")) == (
        "diagonal.pair",
        "diagonal.cubic",
        "diagonal.quartic",
    )
    assert roundtrip.shape == params.shape
    assert phase.shape == (4,)
    np.testing.assert_allclose(roundtrip, params, atol=1.0e-10)


def test_pair_uccd_gcr_preset_is_composite_parameterization():
    parameterization = PairUCCD_GCR(order=3, norb=4, nocc=2)

    assert isinstance(parameterization, CompositeReferenceAnsatzParameterization)
    assert parameterization.n_reference_params == 4
    assert tuple(block.name for block in parameterization.parameter_blocks()) == (
        "reference",
        "left",
        "pair",
        "cubic",
        "right",
    )


def test_igcr2_pack_unpack_uses_named_blocks_for_layered_cases():
    for parameterization in [
        IGCR2SpinRestrictedParameterization(norb=4, nocc=2, layers=1),
        IGCR2SpinRestrictedParameterization(norb=4, nocc=2, layers=2),
        IGCR2SpinRestrictedParameterization(
            norb=4,
            nocc=2,
            layers=2,
            shared_diagonal=True,
        ),
        IGCR2SpinBalancedParameterization(norb=4, nocc=2),
    ]:
        params = np.linspace(-0.01, 0.01, parameterization.n_params)
        ansatz = parameterization.ansatz_from_parameters(params)
        roundtrip = parameterization.parameters_from_ansatz(ansatz)

        assert roundtrip.shape == params.shape
        assert np.all(np.isfinite(roundtrip))
        assert parameterization.ansatz_from_parameters(roundtrip).norb == 4


def test_gate_sequence_can_reproduce_one_layer_igcr2_ansatz():
    legacy = IGCR2SpinRestrictedParameterization(norb=4, nocc=2)

    def build(instances):
        left, diagonal, right = instances
        return IGCR2Ansatz(
            diagonal=IGCR2SpinRestrictedSpec(pair=diagonal.pair),
            left=left,
            right=right,
            nocc=legacy.nocc,
        )

    sequence = GateSequenceParameterization(
        gates=(
            OrbitalRotationGate("left", legacy.left_orbital_chart, legacy.norb),
            DiagonalCorrelatorGate("diagonal", legacy.diagonal_chart),
            OrbitalRotationGate("right", legacy.right_orbital_chart, legacy.norb),
        ),
        ansatz_builder=build,
        native_parameters_from_public=legacy._native_parameters_from_public,
    )
    params = np.linspace(-0.01, 0.01, legacy.n_params)
    legacy_ansatz = legacy.ansatz_from_parameters(params)
    sequence_ansatz = sequence.ansatz_from_parameters(params)

    assert tuple(block.name for block in sequence.parameter_blocks()) == (
        "left",
        "diagonal.pair",
        "right",
    )
    np.testing.assert_allclose(sequence_ansatz.left, legacy_ansatz.left)
    np.testing.assert_allclose(sequence_ansatz.right, legacy_ansatz.right)
    np.testing.assert_allclose(
        sequence_ansatz.diagonal.pair,
        legacy_ansatz.diagonal.pair,
    )
