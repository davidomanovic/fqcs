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
from xquces.gcr.restricted_model import IGCR3Ansatz, IGCR3LayeredAnsatz
from xquces.qiskit.gates.diag_3 import Diag3SpinRestrictedJW
from xquces.qiskit.gates.orbital_rotations import OrbitalRotationJW

IGCR3CircuitAnsatz = IGCR3Ansatz | IGCR3LayeredAnsatz | IGCRAnsatz


def _as_igcr3_circuit_ansatz(
    ansatz: IGCR3CircuitAnsatz,
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    if isinstance(ansatz, IGCRAnsatz):
        if ansatz.order != 3:
            raise TypeError("expected a canonical iGCR-3 ansatz")
        return ansatz.to_igcr3_ansatz()
    return ansatz


class IGCR3JW(Gate):
    """Full spin-restricted iGCR-3 ansatz gate under Jordan-Wigner."""

    def __init__(
        self,
        ansatz: IGCR3CircuitAnsatz,
        *,
        label: str | None = None,
        validate_orbital_rotations: bool = True,
        diagonal_synthesis: str = "phase_polynomial",
        diagonal_threshold: float = 0.0,
    ):
        ansatz = _as_igcr3_circuit_ansatz(ansatz)
        self.ansatz = ansatz
        self.validate_orbital_rotations = bool(validate_orbital_rotations)
        self.diagonal_synthesis = diagonal_synthesis
        self.diagonal_threshold = float(diagonal_threshold)
        super().__init__("igcr3_jw", 2 * ansatz.norb, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _igcr3_jw(
                qubits,
                self.ansatz,
                validate_orbital_rotations=self.validate_orbital_rotations,
                diagonal_synthesis=self.diagonal_synthesis,
                diagonal_threshold=self.diagonal_threshold,
            ),
            qubits=qubits,
            name=self.name,
        )


def igcr3_jw_circuit(
    ansatz: IGCR3CircuitAnsatz,
    *,
    validate_orbital_rotations: bool = True,
    diagonal_synthesis: str = "phase_polynomial",
    diagonal_threshold: float = 0.0,
) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * ansatz.norb)
    circuit.append(
        IGCR3JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
            diagonal_synthesis=diagonal_synthesis,
            diagonal_threshold=diagonal_threshold,
        ),
        circuit.qubits,
    )
    return circuit


def igcr3_stateprep_jw_circuit(
    ansatz: IGCR3CircuitAnsatz,
    *,
    validate_orbital_rotations: bool = True,
    diagonal_synthesis: str = "phase_polynomial",
    diagonal_threshold: float = 0.0,
) -> QuantumCircuit:
    ansatz = _as_igcr3_circuit_ansatz(ansatz)
    circuit = QuantumCircuit(2 * ansatz.norb)
    for instruction in _igcr3_stateprep_jw(
        circuit.qubits,
        ansatz,
        validate_orbital_rotations=validate_orbital_rotations,
        diagonal_synthesis=diagonal_synthesis,
        diagonal_threshold=diagonal_threshold,
    ):
        circuit.append(instruction)
    return circuit


def _igcr3_jw(
    qubits: Sequence[Qubit],
    ansatz: IGCR3CircuitAnsatz,
    *,
    validate_orbital_rotations: bool,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> Iterator[CircuitInstruction]:
    ansatz = _as_igcr3_circuit_ansatz(ansatz)
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    rotations, diagonals = _igcr3_circuit_layers(ansatz)
    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            rotations[-1],
            validate=validate_orbital_rotations,
        ),
        qubits,
    )
    for layer in range(len(diagonals) - 1, -1, -1):
        yield _igcr3_diagonal_instruction(
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


def _igcr3_stateprep_jw(
    qubits: Sequence[Qubit],
    ansatz: IGCR3CircuitAnsatz,
    *,
    validate_orbital_rotations: bool,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> Iterator[CircuitInstruction]:
    ansatz = _as_igcr3_circuit_ansatz(ansatz)
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    rotations, diagonals = _igcr3_circuit_layers(ansatz)
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
        yield _igcr3_diagonal_instruction(
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


def _igcr3_circuit_layers(ansatz: IGCR3CircuitAnsatz):
    ansatz = _as_igcr3_circuit_ansatz(ansatz)
    if isinstance(ansatz, IGCR3Ansatz):
        return (
            (
                np.asarray(ansatz.left, dtype=np.complex128),
                np.asarray(ansatz.right, dtype=np.complex128),
            ),
            (ansatz.diagonal,),
        )
    if isinstance(ansatz, IGCR3LayeredAnsatz):
        return (
            tuple(
                np.asarray(rotation, dtype=np.complex128)
                for rotation in ansatz.rotations
            ),
            tuple(ansatz.diagonals),
        )
    raise TypeError("ansatz must be an IGCR3Ansatz or IGCR3LayeredAnsatz")


def _igcr3_diagonal_instruction(
    norb: int,
    diagonal,
    qubits: Sequence[Qubit],
    *,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> CircuitInstruction:
    return CircuitInstruction(
        Diag3SpinRestrictedJW(
            norb,
            diagonal.full_double(),
            diagonal.pair_matrix(),
            diagonal.tau_matrix(),
            diagonal.omega_vector(),
            synthesis=diagonal_synthesis,
            threshold=diagonal_threshold,
        ),
        qubits,
    )
