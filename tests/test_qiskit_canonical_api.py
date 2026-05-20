from __future__ import annotations

import numpy as np

from xquces.gcr import IGCRSpinRestrictedParameterization, PairUCCDIGCRParameterization
from xquces.gcr.canonical import IGCRAnsatz
from xquces.qiskit.gates import (
    gcr_product_pair_uccd_stateprep_jw_circuit,
    igcr2_jw_circuit,
    igcr2_stateprep_jw_circuit,
    igcr3_jw_circuit,
    igcr3_stateprep_jw_circuit,
    igcr4_jw_circuit,
    igcr4_stateprep_jw_circuit,
    product_pair_uccd_igcr_stateprep_jw_circuit,
)


def test_canonical_igcr_ansatz_builds_qiskit_circuits():
    builders = {
        2: (igcr2_jw_circuit, igcr2_stateprep_jw_circuit),
        3: (igcr3_jw_circuit, igcr3_stateprep_jw_circuit),
        4: (igcr4_jw_circuit, igcr4_stateprep_jw_circuit),
    }
    rng = np.random.default_rng(1234)
    for order, order_builders in builders.items():
        parameterization = IGCRSpinRestrictedParameterization(
            norb=4,
            nocc=2,
            order=order,
            layers=2,
        )
        params = rng.normal(scale=0.02, size=parameterization.n_params)
        ansatz = parameterization.ansatz_from_parameters(params)
        assert isinstance(ansatz, IGCRAnsatz)

        for builder in order_builders:
            circuit = builder(ansatz)
            assert circuit.num_qubits == 2 * ansatz.norb


def test_product_pair_uccd_igcr_qiskit_accepts_canonical_ansatz():
    rng = np.random.default_rng(5678)
    for order in (2, 3, 4):
        ansatz_parameterization = IGCRSpinRestrictedParameterization(
            norb=4,
            nocc=2,
            order=order,
            layers=1,
        )
        ansatz_params = rng.normal(scale=0.02, size=ansatz_parameterization.n_params)
        ansatz = ansatz_parameterization.ansatz_from_parameters(ansatz_params)
        reference_params = rng.normal(scale=0.02, size=4)

        circuit = product_pair_uccd_igcr_stateprep_jw_circuit(
            ansatz,
            reference_params,
            puccd_strategy="pair_register",
        )
        assert circuit.num_qubits == 2 * ansatz.norb


def test_product_pair_uccd_igcr_parameterization_builds_qiskit_circuit():
    rng = np.random.default_rng(9012)
    for order in (2, 3, 4):
        parameterization = PairUCCDIGCRParameterization(
            norb=4,
            nocc=2,
            order=order,
            reference_kind="product",
        )
        params = rng.normal(scale=0.02, size=parameterization.n_params)
        circuit = gcr_product_pair_uccd_stateprep_jw_circuit(parameterization, params)
        assert circuit.num_qubits == 2 * parameterization.norb
