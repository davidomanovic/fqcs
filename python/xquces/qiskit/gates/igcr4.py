from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from ffsim.qiskit.gates import PrepareSlaterDeterminantJW
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)

from xquces.gcr.canonical import IGCRAnsatz
from xquces.gcr.restricted_model import IGCR4Ansatz, IGCR4LayeredAnsatz
from xquces.qiskit.gates.diag_4 import Diag4SpinRestrictedJW
from xquces.qiskit.gates.orbital_rotations import OrbitalRotationJW

IGCR4CircuitAnsatz = IGCR4Ansatz | IGCR4LayeredAnsatz | IGCRAnsatz


def _as_igcr4_circuit_ansatz(
    ansatz: IGCR4CircuitAnsatz,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    if isinstance(ansatz, IGCRAnsatz):
        if ansatz.order != 4:
            raise TypeError("expected a canonical iGCR-4 ansatz")
        return ansatz.to_igcr4_ansatz()
    return ansatz


class IGCR4JW(Gate):
    """Full spin-restricted iGCR-4 ansatz gate under Jordan-Wigner."""

    def __init__(
        self,
        ansatz: IGCR4CircuitAnsatz,
        *,
        label: str | None = None,
        validate_orbital_rotations: bool = True,
        diagonal_synthesis: str = "phase_polynomial",
        diagonal_threshold: float = 0.0,
    ):
        ansatz = _as_igcr4_circuit_ansatz(ansatz)
        self.ansatz = ansatz
        self.validate_orbital_rotations = bool(validate_orbital_rotations)
        self.diagonal_synthesis = diagonal_synthesis
        self.diagonal_threshold = float(diagonal_threshold)
        super().__init__("igcr4_jw", 2 * ansatz.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _igcr4_jw(
                qubits,
                self.ansatz,
                validate_orbital_rotations=self.validate_orbital_rotations,
                diagonal_synthesis=self.diagonal_synthesis,
                diagonal_threshold=self.diagonal_threshold,
            ),
            qubits=qubits,
            name=self.name,
        )


def igcr4_jw_circuit(
    ansatz: IGCR4CircuitAnsatz,
    *,
    validate_orbital_rotations: bool = True,
    diagonal_synthesis: str = "phase_polynomial",
    diagonal_threshold: float = 0.0,
) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * ansatz.norb)
    circuit.append(
        IGCR4JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
            diagonal_synthesis=diagonal_synthesis,
            diagonal_threshold=diagonal_threshold,
        ),
        circuit.qubits,
    )
    return circuit


def igcr4_stateprep_jw_circuit(
    ansatz: IGCR4CircuitAnsatz,
    *,
    validate_orbital_rotations: bool = True,
    diagonal_synthesis: str = "phase_polynomial",
    diagonal_threshold: float = 0.0,
) -> QuantumCircuit:
    ansatz = _as_igcr4_circuit_ansatz(ansatz)
    circuit = QuantumCircuit(2 * ansatz.norb)
    for instruction in _igcr4_stateprep_jw(
        circuit.qubits,
        ansatz,
        validate_orbital_rotations=validate_orbital_rotations,
        diagonal_synthesis=diagonal_synthesis,
        diagonal_threshold=diagonal_threshold,
    ):
        circuit.append(instruction)
    return circuit


def _igcr4_jw(
    qubits: Sequence[Qubit],
    ansatz: IGCR4CircuitAnsatz,
    *,
    validate_orbital_rotations: bool,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> Iterator[CircuitInstruction]:
    ansatz = _as_igcr4_circuit_ansatz(ansatz)
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    rotations, diagonals = _igcr4_circuit_layers(ansatz)
    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            rotations[-1],
            validate=validate_orbital_rotations,
        ),
        qubits,
    )
    for layer in range(len(diagonals) - 1, -1, -1):
        yield _igcr4_diagonal_instruction(
            ansatz.norb,
            diagonals[layer],
            qubits,
            diagonal_synthesis=diagonal_synthesis,
            diagonal_threshold=diagonal_threshold,
        )
        yield CircuitInstruction(
            OrbitalRotationJW(
                ansatz.norb,
                rotations[layer],
                validate=validate_orbital_rotations,
            ),
            qubits,
        )


def _igcr4_stateprep_jw(
    qubits: Sequence[Qubit],
    ansatz: IGCR4CircuitAnsatz,
    *,
    validate_orbital_rotations: bool,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> Iterator[CircuitInstruction]:
    ansatz = _as_igcr4_circuit_ansatz(ansatz)
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    rotations, diagonals = _igcr4_circuit_layers(ansatz)
    occupied = (range(ansatz.nocc), range(ansatz.nocc))
    yield CircuitInstruction(
        PrepareSlaterDeterminantJW(
            ansatz.norb,
            occupied,
            orbital_rotation=rotations[-1],
            validate=validate_orbital_rotations,
        ),
        qubits,
    )
    for layer in range(len(diagonals) - 1, -1, -1):
        yield _igcr4_diagonal_instruction(
            ansatz.norb,
            diagonals[layer],
            qubits,
            diagonal_synthesis=diagonal_synthesis,
            diagonal_threshold=diagonal_threshold,
        )
        yield CircuitInstruction(
            OrbitalRotationJW(
                ansatz.norb,
                rotations[layer],
                validate=validate_orbital_rotations,
            ),
            qubits,
        )


def _igcr4_circuit_layers(ansatz: IGCR4CircuitAnsatz):
    ansatz = _as_igcr4_circuit_ansatz(ansatz)
    if isinstance(ansatz, IGCR4Ansatz):
        return (
            (
                np.asarray(ansatz.left, dtype=np.complex128),
                np.asarray(ansatz.right, dtype=np.complex128),
            ),
            (ansatz.diagonal,),
        )
    if isinstance(ansatz, IGCR4LayeredAnsatz):
        return (
            tuple(
                np.asarray(rotation, dtype=np.complex128)
                for rotation in ansatz.rotations
            ),
            tuple(ansatz.diagonals),
        )
    raise TypeError("ansatz must be an IGCR4Ansatz or IGCR4LayeredAnsatz")


def _igcr4_diagonal_instruction(
    norb: int,
    diagonal,
    qubits: Sequence[Qubit],
    *,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> CircuitInstruction:
    return CircuitInstruction(
        Diag4SpinRestrictedJW(
            norb,
            diagonal.full_double(),
            diagonal.pair_matrix(),
            diagonal.tau_matrix(),
            diagonal.omega_vector(),
            diagonal.eta_vector(),
            diagonal.rho_vector(),
            diagonal.sigma_vector(),
            synthesis=diagonal_synthesis,
            threshold=diagonal_threshold,
        ),
        qubits,
    )
