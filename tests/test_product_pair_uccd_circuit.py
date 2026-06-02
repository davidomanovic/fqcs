from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from xquces.gcr.pair_uccd_igcr import PairUCCDIGCRParameterization
from xquces.qiskit.gates import (
    gcr_product_pair_uccd_stateprep_jw_circuit,
    product_pair_uccd_stateprep_jw_circuit,
)


def _state(circuit):
    return Statevector.from_label("0" * circuit.num_qubits).evolve(circuit).data


def test_pair_register_swap_network_matches_direct_pair_register():
    norb = 6
    nelec = (3, 3)
    rng = np.random.default_rng(1234)
    params = rng.normal(scale=0.2, size=9)

    optimized = product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        strategy="pair_register",
    )
    direct = product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        strategy="pair_register_direct",
    )

    assert np.allclose(_state(optimized), _state(direct), atol=1e-12)


def test_pair_register_swap_network_restores_logical_order():
    norb = 8
    nelec = (4, 4)
    params = np.linspace(-0.1, 0.1, nelec[0] * (norb - nelec[0]))

    optimized = product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        strategy="pair_register",
    )
    direct = product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        strategy="pair_register_direct",
    )

    assert np.allclose(_state(optimized), _state(direct), atol=1e-12)


def test_spin_orbital_stateprep_matches_direct_pair_register():
    norb = 5
    nelec = (2, 2)
    rng = np.random.default_rng(13579)
    params = rng.normal(scale=0.2, size=nelec[0] * (norb - nelec[0]))

    spin_orbital = product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        strategy="spin_orbital",
    )
    direct = product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        strategy="pair_register_direct",
    )

    assert np.allclose(_state(spin_orbital), _state(direct), atol=1e-12)


def test_gcr_pair_register_permutation_matches_direct_pair_register():
    norb = 6
    nocc = 3
    rng = np.random.default_rng(5678)
    param = PairUCCDIGCRParameterization(
        norb=norb,
        nocc=nocc,
        order=2,
        reference_kind="product",
        layers=1,
    )
    params = rng.normal(scale=0.05, size=param.n_params)

    optimized = gcr_product_pair_uccd_stateprep_jw_circuit(
        param,
        params,
        puccd_strategy="pair_register",
    )
    direct = gcr_product_pair_uccd_stateprep_jw_circuit(
        param,
        params,
        puccd_strategy="pair_register_direct",
    )

    phase = np.vdot(_state(direct), _state(optimized))
    assert abs(abs(phase) - 1.0) < 1e-12
    assert np.allclose(_state(optimized), phase * _state(direct), atol=1e-12)
