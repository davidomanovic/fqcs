from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, MutableMapping, Sequence

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import CXGate, RZGate

from xquces.gcr.utils import (
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
)


PauliZCoefficients = dict[tuple[int, ...], float]


def _validate_threshold(threshold: float) -> float:
    out = float(threshold)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError("threshold must be a non-negative finite float")
    return out


def _sorted_distinct_indices(indices: Sequence[int]) -> tuple[int, ...]:
    out = tuple(sorted(int(i) for i in indices))
    if len(set(out)) != len(out):
        raise ValueError("number-product phase indices must be distinct")
    return out


def number_product_pauli_z_coefficients(
    indices: Sequence[int],
    theta: float,
) -> PauliZCoefficients:
    """Return Pauli-Z coefficients for ``exp(i theta prod_i n_i)``.

    The empty Pauli string contributes only a global phase and is omitted.
    """
    coeffs: PauliZCoefficients = {}
    add_number_product_phase(coeffs, theta, indices)
    return coeffs


def add_number_product_phase(
    coeffs: MutableMapping[tuple[int, ...], float],
    theta: float,
    indices: Sequence[int],
) -> None:
    """Accumulate a number-product phase into a Pauli-Z phase polynomial."""
    support = _sorted_distinct_indices(indices)
    if not support:
        return

    theta = float(theta)
    if theta == 0.0:
        return

    scale = theta / float(2**len(support))
    for size in range(1, len(support) + 1):
        sign = -1.0 if size % 2 else 1.0
        for subset in itertools.combinations(support, size):
            coeffs[subset] = float(coeffs.get(subset, 0.0)) + sign * scale


def synthesize_pauli_z_phase_polynomial(
    coeffs: Mapping[tuple[int, ...], float],
    qubits: Sequence[Qubit],
    *,
    threshold: float = 0.0,
) -> Iterator[CircuitInstruction]:
    """Yield parity-gadget instructions for ``prod_B exp(i c_B Z_B)``."""
    threshold = _validate_threshold(threshold)
    nqubits = len(qubits)

    for support, coeff0 in sorted(coeffs.items()):
        support = tuple(int(i) for i in support)
        if not support:
            continue
        if len(set(support)) != len(support):
            raise ValueError("Pauli-Z supports must contain distinct qubit indices")
        if support != tuple(sorted(support)):
            raise ValueError("Pauli-Z supports must be sorted")
        if support[0] < 0 or support[-1] >= nqubits:
            raise ValueError("Pauli-Z support index out of range")

        coeff = float(coeff0)
        if abs(coeff) <= threshold:
            continue

        target = support[-1]
        if len(support) == 1:
            yield CircuitInstruction(RZGate(-2.0 * coeff), (qubits[target],))
            continue

        for control in support[:-1]:
            yield CircuitInstruction(CXGate(), (qubits[control], qubits[target]))
        yield CircuitInstruction(RZGate(-2.0 * coeff), (qubits[target],))
        for control in reversed(support[:-1]):
            yield CircuitInstruction(CXGate(), (qubits[control], qubits[target]))


class PhasePolynomialJW(Gate):
    """Collected Pauli-Z phase polynomial synthesized with parity gadgets."""

    def __init__(
        self,
        coeffs: Mapping[tuple[int, ...], float],
        nqubits: int,
        *,
        threshold: float = 0.0,
        label: str | None = None,
    ):
        self.nqubits = int(nqubits)
        if self.nqubits < 0:
            raise ValueError("nqubits must be non-negative")
        self.threshold = _validate_threshold(threshold)
        self.coeffs = self._normalize_coefficients(coeffs)
        super().__init__("phase_polynomial_jw", self.nqubits, [], label=label)

    def _normalize_coefficients(
        self,
        coeffs: Mapping[tuple[int, ...], float],
    ) -> PauliZCoefficients:
        out: PauliZCoefficients = {}
        for support0, coeff0 in coeffs.items():
            support = tuple(sorted(int(i) for i in support0))
            if not support:
                continue
            if len(set(support)) != len(support):
                raise ValueError("Pauli-Z supports must contain distinct qubit indices")
            if support[0] < 0 or support[-1] >= self.nqubits:
                raise ValueError("Pauli-Z support index out of range")
            coeff = float(coeff0)
            if coeff == 0.0:
                continue
            out[support] = out.get(support, 0.0) + coeff
        return out

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in synthesize_pauli_z_phase_polynomial(
            self.coeffs,
            qubits,
            threshold=self.threshold,
        ):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "PhasePolynomialJW":
        return PhasePolynomialJW(
            {support: -coeff for support, coeff in self.coeffs.items()},
            self.nqubits,
            threshold=self.threshold,
            label=self.label,
        )


def add_spin_restricted_diag2_number_products(
    coeffs: MutableMapping[tuple[int, ...], float],
    norb: int,
    double_params: np.ndarray,
    pair_params: np.ndarray,
    *,
    time: float = 1.0,
) -> None:
    double = np.asarray(double_params, dtype=np.float64)
    pair = np.asarray(pair_params, dtype=np.float64)

    for p in range(norb):
        add_number_product_phase(coeffs, float(time) * double[p], (p, norb + p))

    for p, q in itertools.combinations(range(norb), 2):
        theta = float(time) * pair[p, q]
        add_number_product_phase(coeffs, theta, (p, q))
        add_number_product_phase(coeffs, theta, (norb + p, norb + q))
        add_number_product_phase(coeffs, theta, (p, norb + q))
        add_number_product_phase(coeffs, theta, (q, norb + p))


def add_spin_restricted_diag3_number_products(
    coeffs: MutableMapping[tuple[int, ...], float],
    norb: int,
    tau_params: np.ndarray,
    omega_values: np.ndarray,
    *,
    time: float = 1.0,
) -> None:
    tau = np.asarray(tau_params, dtype=np.float64)
    omega = np.asarray(omega_values, dtype=np.float64)

    for p in range(norb):
        for q in range(norb):
            if p == q:
                continue
            theta = float(time) * tau[p, q]
            add_number_product_phase(coeffs, theta, (p, norb + p, q))
            add_number_product_phase(coeffs, theta, (p, norb + p, norb + q))

    for theta0, (p, q, r) in zip(omega, itertools.combinations(range(norb), 3)):
        theta = float(time) * float(theta0)
        for p_spin in (p, norb + p):
            for q_spin in (q, norb + q):
                for r_spin in (r, norb + r):
                    add_number_product_phase(coeffs, theta, (p_spin, q_spin, r_spin))


def add_spin_restricted_diag4_number_products(
    coeffs: MutableMapping[tuple[int, ...], float],
    norb: int,
    eta_values: np.ndarray,
    rho_values: np.ndarray,
    sigma_values: np.ndarray,
    *,
    time: float = 1.0,
) -> None:
    eta = np.asarray(eta_values, dtype=np.float64)
    rho = np.asarray(rho_values, dtype=np.float64)
    sigma = np.asarray(sigma_values, dtype=np.float64)

    for theta0, (p, q) in zip(eta, _default_eta_indices(norb)):
        theta = float(time) * float(theta0)
        add_number_product_phase(coeffs, theta, (p, norb + p, q, norb + q))

    for theta0, (p, q, r) in zip(rho, _default_rho_indices(norb)):
        theta = float(time) * float(theta0)
        for q_spin in (q, norb + q):
            for r_spin in (r, norb + r):
                add_number_product_phase(
                    coeffs,
                    theta,
                    (p, norb + p, q_spin, r_spin),
                )

    for theta0, (p, q, r, s) in zip(sigma, _default_sigma_indices(norb)):
        theta = float(time) * float(theta0)
        for p_spin, q_spin, r_spin, s_spin in itertools.product(
            (p, norb + p),
            (q, norb + q),
            (r, norb + r),
            (s, norb + s),
        ):
            add_number_product_phase(
                coeffs,
                theta,
                (p_spin, q_spin, r_spin, s_spin),
            )
