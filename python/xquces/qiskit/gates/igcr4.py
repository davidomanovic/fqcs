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

from xquces.gcr.igcr import IGCR4Ansatz
from xquces.qiskit.gates.diag_4 import Diag4SpinRestrictedJW
from xquces.qiskit.gates.orbital_rotations import OrbitalRotationJW


class IGCR4JW(Gate):
    """Full spin-restricted iGCR-4 ansatz gate under Jordan-Wigner."""

    def __init__(
        self,
        ansatz: IGCR4Ansatz,
        *,
        label: str | None = None,
        validate_orbital_rotations: bool = True,
        diagonal_synthesis: str = "phase_polynomial",
        diagonal_threshold: float = 0.0,
    ):
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
    ansatz: IGCR4Ansatz,
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
    ansatz: IGCR4Ansatz,
    *,
    validate_orbital_rotations: bool = True,
    diagonal_synthesis: str = "phase_polynomial",
    diagonal_threshold: float = 0.0,
) -> QuantumCircuit:
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
    ansatz: IGCR4Ansatz,
    *,
    validate_orbital_rotations: bool,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    d = ansatz.diagonal
    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            np.asarray(ansatz.right, dtype=np.complex128),
            validate=validate_orbital_rotations,
        ),
        qubits,
    )
    yield CircuitInstruction(
        Diag4SpinRestrictedJW(
            ansatz.norb,
            d.full_double(),
            d.pair_matrix(),
            d.tau_matrix(),
            d.omega_vector(),
            d.eta_vector(),
            d.rho_vector(),
            d.sigma_vector(),
            synthesis=diagonal_synthesis,
            threshold=diagonal_threshold,
        ),
        qubits,
    )
    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            np.asarray(ansatz.left, dtype=np.complex128),
            validate=validate_orbital_rotations,
        ),
        qubits,
    )


def _igcr4_stateprep_jw(
    qubits: Sequence[Qubit],
    ansatz: IGCR4Ansatz,
    *,
    validate_orbital_rotations: bool,
    diagonal_synthesis: str,
    diagonal_threshold: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * ansatz.norb:
        raise ValueError("Expected 2 * ansatz.norb qubits.")

    d = ansatz.diagonal
    occupied = (range(ansatz.nocc), range(ansatz.nocc))
    yield CircuitInstruction(
        PrepareSlaterDeterminantJW(
            ansatz.norb,
            occupied,
            orbital_rotation=np.asarray(ansatz.right, dtype=np.complex128),
            validate=validate_orbital_rotations,
        ),
        qubits,
    )
    yield CircuitInstruction(
        Diag4SpinRestrictedJW(
            ansatz.norb,
            d.full_double(),
            d.pair_matrix(),
            d.tau_matrix(),
            d.omega_vector(),
            d.eta_vector(),
            d.rho_vector(),
            d.sigma_vector(),
            synthesis=diagonal_synthesis,
            threshold=diagonal_threshold,
        ),
        qubits,
    )
    yield CircuitInstruction(
        OrbitalRotationJW(
            ansatz.norb,
            np.asarray(ansatz.left, dtype=np.complex128),
            validate=validate_orbital_rotations,
        ),
        qubits,
    )
