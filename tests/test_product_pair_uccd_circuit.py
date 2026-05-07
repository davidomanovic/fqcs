from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from xquces.qiskit.gates import product_pair_uccd_stateprep_jw_circuit


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
