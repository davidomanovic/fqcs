from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from xquces.gcr.igcr import (
    IGCR3Ansatz,
    IGCR3SpinRestrictedParameterization,
    IGCR3SpinRestrictedSpec,
    IGCR4Ansatz,
    IGCR4SpinRestrictedParameterization,
    IGCR4SpinRestrictedSpec,
)
from xquces.gcr.utils import (
    _default_eta_indices,
    _default_pair_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_triple_indices,
)
from xquces.qiskit.gates import (
    Diag3SpinRestrictedJW,
    Diag4SpinRestrictedJW,
    PhasePolynomialJW,
    add_number_product_phase,
    igcr3_jw_circuit,
    igcr3_stateprep_jw_circuit,
    igcr4_jw_circuit,
    igcr4_stateprep_jw_circuit,
)
from xquces.qiskit.gates.diag_3 import _yield_number_product_phase


def _assert_equivalent_up_to_global_phase(
    lhs: QuantumCircuit,
    rhs: QuantumCircuit,
    *,
    atol: float = 1e-10,
) -> None:
    lhs_data = Operator(lhs).data
    rhs_data = Operator(rhs).data
    overlap = np.vdot(rhs_data.ravel(), lhs_data.ravel())
    phase = overlap / abs(overlap)
    assert np.allclose(lhs_data, phase * rhs_data, atol=atol)


def _assert_same_state_up_to_global_phase(
    lhs: QuantumCircuit,
    rhs: QuantumCircuit,
    *,
    atol: float = 1e-10,
) -> None:
    lhs_data = Statevector.from_instruction(lhs).data
    rhs_data = Statevector.from_instruction(rhs).data
    overlap = np.vdot(rhs_data, lhs_data)
    phase = overlap / abs(overlap)
    assert np.allclose(lhs_data, phase * rhs_data, atol=atol)


def _naive_number_product_circuit(
    nqubits: int,
    theta: float,
    indices: tuple[int, ...],
) -> QuantumCircuit:
    circuit = QuantumCircuit(nqubits)
    for instruction in _yield_number_product_phase(circuit.qubits, theta, indices):
        circuit.append(instruction)
    return circuit


def _phase_polynomial_number_product_circuit(
    nqubits: int,
    theta: float,
    indices: tuple[int, ...],
    *,
    synthesis: str = "parity_gadgets",
) -> QuantumCircuit:
    coeffs: dict[tuple[int, ...], float] = {}
    add_number_product_phase(coeffs, theta, indices)
    circuit = QuantumCircuit(nqubits)
    circuit.append(PhasePolynomialJW(coeffs, nqubits, synthesis=synthesis), circuit.qubits)
    return circuit


def _phase_polynomial_circuit(
    coeffs: dict[tuple[int, ...], float],
    nqubits: int,
    *,
    synthesis: str,
) -> QuantumCircuit:
    circuit = QuantumCircuit(nqubits)
    circuit.append(PhasePolynomialJW(coeffs, nqubits, synthesis=synthesis), circuit.qubits)
    return circuit


def _symmetric_random_matrix(
    rng: np.random.Generator,
    norb: int,
    *,
    scale: float,
) -> np.ndarray:
    mat = rng.normal(scale=scale, size=(norb, norb))
    mat = np.triu(mat, 1)
    return mat + mat.T


def _random_diag3_params(
    rng: np.random.Generator,
    norb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    double = rng.normal(scale=0.2, size=norb)
    pair = _symmetric_random_matrix(rng, norb, scale=0.2)
    tau = rng.normal(scale=0.2, size=(norb, norb))
    np.fill_diagonal(tau, 0.0)
    omega = rng.normal(scale=0.2, size=len(_default_triple_indices(norb)))
    return double, pair, tau, omega


def _random_diag4_params(
    rng: np.random.Generator,
    norb: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    double, pair, tau, omega = _random_diag3_params(rng, norb)
    eta = rng.normal(scale=0.2, size=len(_default_eta_indices(norb)))
    rho = rng.normal(scale=0.2, size=len(_default_rho_indices(norb)))
    sigma = rng.normal(scale=0.2, size=len(_default_sigma_indices(norb)))
    return double, pair, tau, omega, eta, rho, sigma


def _diag3_circuit(
    norb: int,
    params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    synthesis: str,
) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * norb)
    circuit.append(
        Diag3SpinRestrictedJW(
            norb,
            *params,
            time=0.73,
            synthesis=synthesis,
        ),
        circuit.qubits,
    )
    return circuit


def _diag4_circuit(
    norb: int,
    params: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    synthesis: str,
) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * norb)
    circuit.append(
        Diag4SpinRestrictedJW(
            norb,
            *params,
            time=0.73,
            synthesis=synthesis,
        ),
        circuit.qubits,
    )
    return circuit


def _igcr3_ansatz(norb: int = 3, nocc: int = 1) -> IGCR3Ansatz:
    return IGCR3Ansatz(
        diagonal=IGCR3SpinRestrictedSpec(
            double_params=np.zeros(norb),
            pair_values=np.zeros(len(_default_pair_indices(norb))),
            tau=np.zeros((norb, norb)),
            omega_values=np.zeros(len(_default_triple_indices(norb))),
        ),
        left=np.eye(norb, dtype=np.complex128),
        right=np.eye(norb, dtype=np.complex128),
        nocc=nocc,
    )


def _igcr4_ansatz(norb: int = 4, nocc: int = 2) -> IGCR4Ansatz:
    return IGCR4Ansatz(
        diagonal=IGCR4SpinRestrictedSpec(
            double_params=np.zeros(norb),
            pair_values=np.zeros(len(_default_pair_indices(norb))),
            tau=np.zeros((norb, norb)),
            omega_values=np.zeros(len(_default_triple_indices(norb))),
            eta_values=np.zeros(len(_default_eta_indices(norb))),
            rho_values=np.zeros(len(_default_rho_indices(norb))),
            sigma_values=np.zeros(len(_default_sigma_indices(norb))),
        ),
        left=np.eye(norb, dtype=np.complex128),
        right=np.eye(norb, dtype=np.complex128),
        nocc=nocc,
    )


def test_number_product_phase_polynomial_identity():
    rng = np.random.default_rng(1234)
    supports = ((0,), (0, 2), (0, 2, 3), (0, 1, 3, 4))
    for support in supports:
        theta = float(rng.normal(scale=0.5))
        nqubits = max(support) + 1
        _assert_equivalent_up_to_global_phase(
            _phase_polynomial_number_product_circuit(nqubits, theta, support),
            _naive_number_product_circuit(nqubits, theta, support),
        )


def test_number_product_balanced_parity_gadgets_identity():
    rng = np.random.default_rng(4321)
    supports = ((0,), (0, 2), (0, 2, 3), (0, 1, 3, 4))
    for support in supports:
        theta = float(rng.normal(scale=0.5))
        nqubits = max(support) + 1
        _assert_equivalent_up_to_global_phase(
            _phase_polynomial_number_product_circuit(
                nqubits,
                theta,
                support,
                synthesis="balanced_parity_gadgets",
            ),
            _naive_number_product_circuit(nqubits, theta, support),
        )


def test_parity_network_alias_matches_parity_gadgets_for_collected_polynomial():
    coeffs = {
        (0, 1, 2): 0.11,
        (0, 1, 3): -0.07,
        (0, 1, 2, 3): 0.13,
        (1, 2, 3): -0.17,
        (0,): 0.19,
    }
    _assert_equivalent_up_to_global_phase(
        _phase_polynomial_circuit(coeffs, 4, synthesis="parity_network"),
        _phase_polynomial_circuit(coeffs, 4, synthesis="parity_gadgets"),
    )


def test_balanced_parity_gadgets_do_not_increase_cx_count():
    coeffs = {
        (0, 1, 2): 0.11,
        (0, 1, 3): -0.07,
        (0, 1, 2, 3): 0.13,
        (1, 2, 3): -0.17,
    }
    gadgets = PhasePolynomialJW(coeffs, 4, synthesis="parity_gadgets").definition
    balanced = PhasePolynomialJW(
        coeffs,
        4,
        synthesis="balanced_parity_gadgets",
    ).definition
    assert balanced.count_ops().get("cx", 0) <= gadgets.count_ops().get("cx", 0)


def test_diag3_naive_and_phase_polynomial_match():
    rng = np.random.default_rng(2024)
    norb = 3
    params = _random_diag3_params(rng, norb)

    _assert_equivalent_up_to_global_phase(
        _diag3_circuit(norb, params, synthesis="phase_polynomial"),
        _diag3_circuit(norb, params, synthesis="naive"),
    )


def test_diag3_naive_and_parity_network_match():
    rng = np.random.default_rng(2026)
    norb = 3
    params = _random_diag3_params(rng, norb)

    _assert_equivalent_up_to_global_phase(
        _diag3_circuit(norb, params, synthesis="parity_network"),
        _diag3_circuit(norb, params, synthesis="naive"),
    )


def test_diag4_naive_and_phase_polynomial_match():
    rng = np.random.default_rng(2025)
    norb = 4
    params = _random_diag4_params(rng, norb)

    _assert_equivalent_up_to_global_phase(
        _diag4_circuit(norb, params, synthesis="phase_polynomial"),
        _diag4_circuit(norb, params, synthesis="naive"),
    )


def test_diag4_naive_and_parity_network_match():
    rng = np.random.default_rng(2027)
    norb = 4
    params = _random_diag4_params(rng, norb)

    _assert_equivalent_up_to_global_phase(
        _diag4_circuit(norb, params, synthesis="parity_network"),
        _diag4_circuit(norb, params, synthesis="naive"),
    )


def test_igcr3_igcr4_public_api_still_accepts_existing_calls():
    igcr3 = _igcr3_ansatz()
    igcr4 = _igcr4_ansatz()

    assert igcr3_jw_circuit(igcr3).num_qubits == 2 * igcr3.norb
    assert igcr3_stateprep_jw_circuit(igcr3).num_qubits == 2 * igcr3.norb
    assert igcr4_jw_circuit(igcr4).num_qubits == 2 * igcr4.norb
    assert igcr4_stateprep_jw_circuit(igcr4).num_qubits == 2 * igcr4.norb


def test_igcr3_igcr4_public_api_accepts_naive_diagonal_synthesis():
    igcr3 = _igcr3_ansatz()
    igcr4 = _igcr4_ansatz()

    assert (
        igcr3_stateprep_jw_circuit(igcr3, diagonal_synthesis="naive").num_qubits
        == 2 * igcr3.norb
    )
    assert (
        igcr4_stateprep_jw_circuit(igcr4, diagonal_synthesis="naive").num_qubits
        == 2 * igcr4.norb
    )


def test_igcr3_igcr4_public_api_accepts_parity_network_diagonal_synthesis():
    igcr3 = _igcr3_ansatz()
    igcr4 = _igcr4_ansatz()

    assert (
        igcr3_stateprep_jw_circuit(igcr3, diagonal_synthesis="parity_network").num_qubits
        == 2 * igcr3.norb
    )
    assert (
        igcr4_stateprep_jw_circuit(igcr4, diagonal_synthesis="parity_network").num_qubits
        == 2 * igcr4.norb
    )


def test_layered_igcr3_stateprep_phase_polynomial_matches_naive():
    rng = np.random.default_rng(31415)
    param = IGCR3SpinRestrictedParameterization(
        norb=3,
        nocc=1,
        layers=2,
        reduce_cubic_gauge=False,
    )
    ansatz = param.ansatz_from_parameters(
        rng.normal(scale=0.05, size=param.n_params)
    )

    _assert_same_state_up_to_global_phase(
        igcr3_stateprep_jw_circuit(ansatz, diagonal_synthesis="phase_polynomial"),
        igcr3_stateprep_jw_circuit(ansatz, diagonal_synthesis="naive"),
    )


def test_layered_igcr4_stateprep_phase_polynomial_matches_naive():
    rng = np.random.default_rng(27182)
    param = IGCR4SpinRestrictedParameterization(
        norb=4,
        nocc=2,
        layers=2,
        reduce_cubic_gauge=False,
        reduce_quartic_gauge=False,
    )
    ansatz = param.ansatz_from_parameters(
        rng.normal(scale=0.04, size=param.n_params)
    )

    _assert_same_state_up_to_global_phase(
        igcr4_stateprep_jw_circuit(ansatz, diagonal_synthesis="phase_polynomial"),
        igcr4_stateprep_jw_circuit(ansatz, diagonal_synthesis="naive"),
    )


def test_phase_polynomial_threshold_skips_all_terms():
    coeffs = {
        (0,): 0.1,
        (0, 1): -0.2,
        (1, 2, 3): 0.3,
    }
    gate = PhasePolynomialJW(coeffs, 4, threshold=1.0)
    ops = gate.definition.count_ops()
    assert ops.get("cx", 0) == 0
    assert ops.get("rz", 0) == 0


def test_parity_network_threshold_skips_all_terms():
    coeffs = {
        (0,): 0.1,
        (0, 1): -0.2,
        (1, 2, 3): 0.3,
    }
    gate = PhasePolynomialJW(coeffs, 4, threshold=1.0, synthesis="parity_network")
    ops = gate.definition.count_ops()
    assert ops.get("cx", 0) == 0
    assert ops.get("rz", 0) == 0
