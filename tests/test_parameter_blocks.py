from __future__ import annotations

import numpy as np
import pytest

from xquces.ansatz import (
    DiagonalCorrelatorGate,
    GateSequenceParameterization,
    OrbitalRotationGate,
    parameter_blocks as ansatz_parameter_blocks,
    random_parameters as ansatz_random_parameters,
)
from xquces.ansatz.blocks import parameter_view as ansatz_parameter_view
from xquces.ansatz.parameters import ParameterBlock
from xquces.gcr.igcr import (
    IGCR2Ansatz,
    IGCR2SpinRestrictedSpec,
    IGCR2SpinBalancedParameterization,
    IGCR2SpinRestrictedParameterization,
    IGCR3SpinRestrictedParameterization,
    IGCRSpinRestrictedParameterization,
    parameter_blocks,
    parameter_view,
)
from xquces.gcr.product_pair_uccd import PairUCCDStateParameterization
from xquces.gcr.references import CompositeReferenceAnsatzParameterization
from xquces.states import hartree_fock_state
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


def _assert_allclose_up_to_phase(actual, expected, *, atol=1.0e-12):
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    phase = np.vdot(expected, actual)
    if abs(phase) > 1.0e-14:
        actual = actual * phase.conjugate() / abs(phase)
    np.testing.assert_allclose(actual, expected, atol=atol)


def _legacy_block_name(sequence_block_name: str) -> str:
    return {
        "diagonal.pair": "pair",
        "diagonal.cubic": "cubic",
        "diagonal.quartic": "quartic",
    }.get(sequence_block_name, sequence_block_name)


def _block_metadata(blocks):
    return tuple(
        (block.name, block.start, block.stop, block.shape, block.kind)
        for block in blocks
    )


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
    parameterization = IGCR(order=4, norb=4, nocc=2)
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


def test_igcr2_shared_diagonal_layered_roundtrip_preserves_state():
    parameterization = IGCR2SpinRestrictedParameterization(
        norb=4,
        nocc=2,
        layers=2,
        shared_diagonal=True,
    )
    reference = hartree_fock_state(4, (2, 2))
    params = np.linspace(-1.0e-3, 2.0e-3, parameterization.n_params)

    ansatz = parameterization.ansatz_from_parameters(params)
    roundtrip = parameterization.parameters_from_ansatz(ansatz)
    actual = parameterization.ansatz_from_parameters(roundtrip).apply(
        reference,
        nelec=(2, 2),
        copy=True,
    )
    expected = ansatz.apply(reference, nelec=(2, 2), copy=True)

    _assert_allclose_up_to_phase(actual, expected, atol=1.0e-12)


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


def test_igcr_sequence_backend_matches_legacy_one_layer_blocks_and_state():
    reference = hartree_fock_state(4, (2, 2))
    rng = np.random.default_rng(1234)
    for order in (2, 3, 4):
        legacy = IGCR(order=order, norb=4, nocc=2)
        sequence = IGCR(order=order, norb=4, nocc=2, backend="sequence")
        zero_params = np.zeros(legacy.n_params, dtype=np.float64)
        random_params = rng.normal(scale=1.0e-3, size=legacy.n_params)

        assert sequence.n_params == legacy.n_params
        sequence_blocks = sequence.parameter_blocks()
        legacy_blocks = {block.name: block for block in parameter_blocks(legacy)}
        assert tuple(block.name for block in sequence_blocks) == (
            "left",
            "diagonal.pair",
            *(() if order == 2 else ("diagonal.cubic",)),
            *(() if order < 4 else ("diagonal.quartic",)),
            "right",
        )
        for sequence_block in sequence_blocks:
            legacy_block = legacy_blocks[_legacy_block_name(sequence_block.name)]
            assert sequence_block.start == legacy_block.start
            assert sequence_block.stop == legacy_block.stop
            assert sequence_block.size == legacy_block.size
            assert sequence_block.shape == legacy_block.shape
            assert sequence_block.kind == legacy_block.kind

        for params in (zero_params, random_params):
            legacy_state = legacy.ansatz_from_parameters(params).apply(
                reference,
                nelec=(2, 2),
                copy=True,
            )
            sequence_state = sequence.ansatz_from_parameters(params).apply(
                reference,
                nelec=(2, 2),
                copy=True,
            )
            _assert_allclose_up_to_phase(sequence_state, legacy_state, atol=1.0e-12)
            _assert_allclose_up_to_phase(
                sequence.params_to_vec(reference)(params),
                legacy_state,
                atol=1.0e-12,
            )
            _assert_allclose_up_to_phase(
                sequence.apply(reference).params_to_vec()(params),
                legacy_state,
                atol=1.0e-12,
            )


def test_igcr_sequence_backend_parameters_from_ansatz_matches_legacy_gauge():
    reference = hartree_fock_state(4, (2, 2))
    rng = np.random.default_rng(5678)

    for order in (2, 3, 4):
        legacy = IGCR(order=order, norb=4, nocc=2)
        sequence = IGCR(order=order, norb=4, nocc=2, backend="sequence")
        params = rng.normal(scale=1.0e-3, size=sequence.n_params)

        roundtrip = sequence.parameters_from_ansatz(
            sequence.ansatz_from_parameters(params)
        )
        legacy_roundtrip = legacy.parameters_from_ansatz(
            legacy.ansatz_from_parameters(params)
        )

        assert roundtrip.shape == params.shape
        assert np.all(np.isfinite(roundtrip))
        np.testing.assert_allclose(roundtrip, legacy_roundtrip, atol=1.0e-10)
        _assert_allclose_up_to_phase(
            sequence.params_to_vec(reference)(roundtrip),
            sequence.params_to_vec(reference)(params),
            atol=1.0e-10,
        )


def test_igcr_sequence_backend_is_opt_in_and_rejects_unsupported_cases():
    default_parameterization = IGCR(order=3, norb=4, nocc=2)

    assert isinstance(default_parameterization, IGCRSpinRestrictedParameterization)
    assert not isinstance(default_parameterization, GateSequenceParameterization)
    assert isinstance(
        IGCR(order=3, norb=4, nocc=2, backend="sequence"),
        GateSequenceParameterization,
    )

    with pytest.raises(NotImplementedError, match="layers=1"):
        IGCR(order=2, norb=4, nocc=2, layers=2, backend="sequence")
    with pytest.raises(NotImplementedError, match="shared_diagonal"):
        IGCR(order=2, norb=4, nocc=2, shared_diagonal=True, backend="sequence")
    with pytest.raises(NotImplementedError, match="spin='restricted'"):
        IGCR(order=2, norb=4, nocc=2, spin="balanced", backend="sequence")
    with pytest.raises(ValueError, match="order must be 2, 3, or 4"):
        IGCR(order=5, norb=4, nocc=2, backend="sequence")


def test_igcr_spin_restricted_to_gate_sequence_is_preset_sequence_source():
    parameterization = IGCR(order=4, norb=4, nocc=2)
    sequence = parameterization.to_gate_sequence()

    assert isinstance(sequence, GateSequenceParameterization)
    assert tuple(block.name for block in sequence.parameter_blocks()) == (
        "left",
        "diagonal.pair",
        "diagonal.cubic",
        "diagonal.quartic",
        "right",
    )


def test_igcr_sequence_preset_uses_canonical_gate_sequence_method(monkeypatch):
    calls = []
    original = IGCRSpinRestrictedParameterization.to_gate_sequence

    def spy(self):
        calls.append(self.order)
        return original(self)

    monkeypatch.setattr(IGCRSpinRestrictedParameterization, "to_gate_sequence", spy)

    sequence = IGCR(order=3, norb=4, nocc=2, backend="sequence")

    assert isinstance(sequence, GateSequenceParameterization)
    assert calls == [3]


def test_ansatz_blocks_preserve_legacy_igcr_block_metadata():
    expected = {
        2: (
            ("left", 0, 12, (12,), "orbital"),
            ("pair", 12, 18, (6,), "diagonal"),
            ("right", 18, 26, (8,), "orbital"),
        ),
        3: (
            ("left", 0, 12, (12,), "orbital"),
            ("pair", 12, 18, (6,), "diagonal"),
            ("cubic", 18, 24, (6,), "diagonal"),
            ("right", 24, 32, (8,), "orbital"),
        ),
        4: (
            ("left", 0, 12, (12,), "orbital"),
            ("pair", 12, 18, (6,), "diagonal"),
            ("cubic", 18, 24, (6,), "diagonal"),
            ("quartic", 24, 27, (3,), "diagonal"),
            ("right", 27, 35, (8,), "orbital"),
        ),
    }

    for order, metadata in expected.items():
        parameterization = IGCR(order=order, norb=4, nocc=2)

        assert _block_metadata(ansatz_parameter_blocks(parameterization)) == metadata
        assert _block_metadata(parameter_blocks(parameterization)) == metadata
        assert parameter_blocks(parameterization)[-1].stop == parameterization.n_params


def test_ansatz_blocks_support_sequence_backend_consistently():
    for order in (2, 3, 4):
        sequence = IGCR(order=order, norb=4, nocc=2, backend="sequence")
        blocks = ansatz_parameter_blocks(sequence)

        assert blocks == sequence.parameter_blocks()
        assert blocks[-1].stop == sequence.n_params
        assert sum(block.size for block in blocks) == sequence.n_params
        assert all(block.start < block.stop for block in blocks)


def test_composite_blocks_use_ansatz_block_module_directly():
    composite = CompositeReferenceAnsatzParameterization(
        PairUCCDStateParameterization(norb=4, nelec=(2, 2)),
        IGCR(order=3, norb=4, nocc=2, backend="sequence"),
        nelec=(2, 2),
    )

    assert (
        CompositeReferenceAnsatzParameterization.parameter_blocks.__globals__[
            "parameter_blocks"
        ].__module__
        == "xquces.ansatz.blocks"
    )
    assert (
        CompositeReferenceAnsatzParameterization.parameter_view.__globals__[
            "parameter_view"
        ].__module__
        == "xquces.ansatz.blocks"
    )
    assert tuple(block.name for block in composite.parameter_blocks()) == (
        "reference",
        "left",
        "diagonal.pair",
        "diagonal.cubic",
        "right",
    )
    assert composite.parameter_blocks()[-1].stop == composite.n_params


def test_block_compatibility_imports_delegate_to_ansatz_blocks():
    import xquces.gcr as gcr
    import xquces.gcr.igcr as igcr

    parameterization = IGCR(order=4, norb=4, nocc=2)

    assert igcr.GCRParameterBlock is ParameterBlock
    assert _block_metadata(igcr.parameter_blocks(parameterization)) == _block_metadata(
        ansatz_parameter_blocks(parameterization)
    )
    assert _block_metadata(gcr.parameter_blocks(parameterization)) == _block_metadata(
        ansatz_parameter_blocks(parameterization)
    )
    assert igcr.random_parameters(parameterization, seed=1).shape == (
        parameterization.n_params,
    )
    assert gcr.random_parameters(parameterization, seed=1).shape == (
        parameterization.n_params,
    )


def test_ansatz_parameter_view_works_for_legacy_and_sequence_backends():
    for parameterization in (
        IGCR(order=3, norb=4, nocc=2),
        IGCR(order=3, norb=4, nocc=2, backend="sequence"),
    ):
        params = np.arange(parameterization.n_params, dtype=np.float64)
        view = ansatz_parameter_view(parameterization, params)

        assert view.names[0] == "left"
        np.testing.assert_array_equal(view["left"], params[: view.block("left").size])
        assert view.blocks[-1].stop == parameterization.n_params


def test_ansatz_random_parameters_can_target_named_blocks():
    for parameterization, kept_name in (
        (IGCR(order=4, norb=4, nocc=2), "quartic"),
        (IGCR(order=4, norb=4, nocc=2, backend="sequence"), "diagonal.quartic"),
    ):
        params = ansatz_random_parameters(
            parameterization,
            seed=99,
            blocks={kept_name},
        )
        kept = ansatz_parameter_blocks(parameterization)

        assert np.count_nonzero(params) == next(
            block.size for block in kept if block.name == kept_name
        )
        for block in kept:
            if block.name == kept_name:
                assert np.count_nonzero(params[block.slice()]) == block.size
            else:
                np.testing.assert_array_equal(
                    params[block.slice()],
                    np.zeros(block.size, dtype=np.float64),
                )
