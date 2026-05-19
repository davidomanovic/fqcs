from __future__ import annotations

import numpy as np

from xquces.gcr.igcr import (
    IGCR2Ansatz,
    IGCR2SpinRestrictedParameterization,
    IGCR3Ansatz,
    IGCR3SpinRestrictedParameterization,
    embed_ansatz_parameters as legacy_embed_ansatz_parameters,
    parameters_from_t2 as legacy_parameters_from_t2,
)
from xquces.presets import IGCR
from xquces.seeds import (
    embed_ansatz_parameters as public_embed_ansatz_parameters,
    parameters_from_t2 as public_parameters_from_t2,
)
from xquces.seeds.dispatch import (
    embed_ansatz_parameters,
    parameters_from_t2,
)
from xquces.states import hartree_fock_state


def _small_t_amplitudes(seed: int = 900):
    rng = np.random.default_rng(seed)
    nocc = 2
    nvirt = 2
    t1 = 0.02 * rng.standard_normal((nocc, nvirt))
    t2 = 0.02 * rng.standard_normal((nocc, nocc, nvirt, nvirt))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    return t1, t2


def test_seed_dispatch_imports_and_compatibility_wrappers_work():
    assert callable(embed_ansatz_parameters)
    assert callable(parameters_from_t2)
    assert callable(legacy_embed_ansatz_parameters)
    assert callable(legacy_parameters_from_t2)
    assert callable(public_embed_ansatz_parameters)
    assert callable(public_parameters_from_t2)


def test_embed_ansatz_parameters_matches_legacy_wrapper_for_lower_order_lift():
    source = IGCR2SpinRestrictedParameterization(norb=4, nocc=2)
    target = IGCR3SpinRestrictedParameterization(norb=4, nocc=2)
    params = np.linspace(-1.0e-3, 1.0e-3, source.n_params)
    ansatz = source.ansatz_from_parameters(params)

    direct = embed_ansatz_parameters(target, ansatz)
    legacy = legacy_embed_ansatz_parameters(target, ansatz)
    method = target.parameters_from_igcr2_ansatz(ansatz)

    np.testing.assert_allclose(direct, method, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(legacy, method, atol=1.0e-14, rtol=0.0)


def test_embed_ansatz_parameters_uses_sequence_inverter_owner_for_lifts():
    source = IGCR2SpinRestrictedParameterization(norb=4, nocc=2)
    sequence = IGCR(order=3, norb=4, nocc=2, backend="sequence")
    legacy_target = IGCR(order=3, norb=4, nocc=2)
    params = np.linspace(-1.0e-3, 1.0e-3, source.n_params)
    ansatz = source.ansatz_from_parameters(params)

    direct = embed_ansatz_parameters(sequence, ansatz)
    expected = legacy_target.parameters_from_igcr2_ansatz(ansatz)

    np.testing.assert_allclose(direct, expected, atol=1.0e-14, rtol=0.0)


def test_parameters_from_t2_matches_wrappers_facade_and_circuit():
    t1, t2 = _small_t_amplitudes(901)
    facade = IGCR(order=3, norb=4, nocc=2)
    options = {
        "source_order": 3,
        "t1": t1,
        "strategy": "zero_embed",
        "igcr2_strategy": "ucj",
    }

    direct = parameters_from_t2(facade, t2, **options)
    legacy = legacy_parameters_from_t2(facade, t2, **options)
    public = public_parameters_from_t2(facade, t2, **options)
    method = facade.parameters_from_t2(t2, **options)
    circuit = facade.circuit(hartree_fock_state(4, (2, 2)), (2, 2))
    circuit_params = circuit.parameters_from_t2(t2, **options)

    np.testing.assert_allclose(legacy, direct, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(public, direct, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(method, direct, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(circuit_params, direct, atol=1.0e-14, rtol=0.0)


def test_parameters_from_t2_infers_sequence_order_from_blocks():
    t1, t2 = _small_t_amplitudes(902)
    sequence = IGCR(order=2, norb=4, nocc=2, backend="sequence")

    direct = parameters_from_t2(sequence, t2, t1=t1)
    expected = embed_ansatz_parameters(
        sequence,
        IGCR2Ansatz.from_t_restricted(t2, t1=t1),
    )

    np.testing.assert_allclose(direct, expected, atol=1.0e-14, rtol=0.0)


def test_parameters_from_t2_embeds_sequence_high_order_ansatz():
    _, t2 = _small_t_amplitudes(903)
    sequence = IGCR(order=3, norb=4, nocc=2, backend="sequence")

    direct = parameters_from_t2(sequence, t2, n_reps=1)
    expected = embed_ansatz_parameters(
        sequence,
        IGCR3Ansatz.from_t_restricted(t2, n_reps=1),
    )

    np.testing.assert_allclose(direct, expected, atol=1.0e-14, rtol=0.0)
