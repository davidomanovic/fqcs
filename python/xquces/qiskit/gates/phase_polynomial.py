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

_PHASE_POLYNOMIAL_SYNTHESIS_MODES = frozenset(
    {"parity_gadgets", "balanced_parity_gadgets", "parity_network"}
)


def _validate_threshold(threshold: float) -> float:
    out = float(threshold)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError("threshold must be a non-negative finite float")
    return out


def _validate_phase_polynomial_synthesis(synthesis: str) -> str:
    if synthesis not in _PHASE_POLYNOMIAL_SYNTHESIS_MODES:
        allowed = ", ".join(sorted(_PHASE_POLYNOMIAL_SYNTHESIS_MODES))
        raise ValueError(f"synthesis must be one of {{{allowed}}}")
    return synthesis


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


def _active_pauli_z_terms(
    coeffs: Mapping[tuple[int, ...], float],
    nqubits: int,
    threshold: float,
) -> list[tuple[tuple[int, ...], float]]:
    out: PauliZCoefficients = {}
    for support0, coeff0 in coeffs.items():
        support = tuple(sorted(int(i) for i in support0))
        if not support:
            continue
        if len(set(support)) != len(support):
            raise ValueError("Pauli-Z supports must contain distinct qubit indices")
        if support[0] < 0 or support[-1] >= nqubits:
            raise ValueError("Pauli-Z support index out of range")
        coeff = float(coeff0)
        if coeff == 0.0:
            continue
        out[support] = out.get(support, 0.0) + coeff
    return [
        (support, coeff)
        for support, coeff in sorted(out.items())
        if abs(coeff) > threshold
    ]


def _yield_ladder_phase_gadget(
    support: tuple[int, ...],
    coeff: float,
    qubits: Sequence[Qubit],
) -> Iterator[CircuitInstruction]:
    target = support[-1]
    if len(support) == 1:
        yield CircuitInstruction(RZGate(-2.0 * coeff), (qubits[target],))
        return

    for control in support[:-1]:
        yield CircuitInstruction(CXGate(), (qubits[control], qubits[target]))
    yield CircuitInstruction(RZGate(-2.0 * coeff), (qubits[target],))
    for control in reversed(support[:-1]):
        yield CircuitInstruction(CXGate(), (qubits[control], qubits[target]))


def _yield_balanced_phase_gadget(
    support: tuple[int, ...],
    coeff: float,
    qubits: Sequence[Qubit],
) -> Iterator[CircuitInstruction]:
    if len(support) == 1:
        yield CircuitInstruction(RZGate(-2.0 * coeff), (qubits[support[0]],))
        return

    active = list(support)
    history: list[tuple[int, int]] = []
    while len(active) > 1:
        next_active: list[int] = []
        for i in range(0, len(active) - 1, 2):
            control = active[i]
            target = active[i + 1]
            yield CircuitInstruction(CXGate(), (qubits[control], qubits[target]))
            history.append((control, target))
            next_active.append(target)
        if len(active) % 2:
            next_active.append(active[-1])
        active = next_active

    target = active[0]
    yield CircuitInstruction(RZGate(-2.0 * coeff), (qubits[target],))
    for control, target in reversed(history):
        yield CircuitInstruction(CXGate(), (qubits[control], qubits[target]))


def synthesize_pauli_z_phase_polynomial(
    coeffs: Mapping[tuple[int, ...], float],
    qubits: Sequence[Qubit],
    *,
    threshold: float = 0.0,
    synthesis: str = "parity_gadgets",
) -> Iterator[CircuitInstruction]:
    """Yield instructions for ``prod_B exp(i c_B Z_B)``."""
    threshold = _validate_threshold(threshold)
    synthesis = _validate_phase_polynomial_synthesis(synthesis)
    nqubits = len(qubits)
    terms = _active_pauli_z_terms(coeffs, nqubits, threshold)

    if synthesis in {"balanced_parity_gadgets", "parity_network"}:
        yield from synthesize_pauli_z_phase_polynomial_balanced_parity_gadgets(
            dict(terms),
            qubits,
            threshold=threshold,
        )
        return

    for support, coeff in terms:
        yield from _yield_ladder_phase_gadget(support, coeff, qubits)


def synthesize_pauli_z_phase_polynomial_balanced_parity_gadgets(
    coeffs: Mapping[tuple[int, ...], float],
    qubits: Sequence[Qubit],
    *,
    threshold: float = 0.0,
) -> Iterator[CircuitInstruction]:
    """Yield all-to-all balanced parity-gadget instructions."""
    threshold = _validate_threshold(threshold)
    nqubits = len(qubits)
    terms = _active_pauli_z_terms(coeffs, nqubits, threshold)
    for support, coeff in terms:
        yield from _yield_balanced_phase_gadget(support, coeff, qubits)


def synthesize_pauli_z_phase_polynomial_parity_network(
    coeffs: Mapping[tuple[int, ...], float],
    qubits: Sequence[Qubit],
    *,
    threshold: float = 0.0,
) -> Iterator[CircuitInstruction]:
    """Yield all-to-all balanced parity-gadget instructions."""
    yield from synthesize_pauli_z_phase_polynomial_balanced_parity_gadgets(
        coeffs,
        qubits,
        threshold=threshold,
    )


class PhasePolynomialJW(Gate):
    """Collected Pauli-Z phase polynomial."""

    def __init__(
        self,
        coeffs: Mapping[tuple[int, ...], float],
        nqubits: int,
        *,
        threshold: float = 0.0,
        synthesis: str = "parity_gadgets",
        label: str | None = None,
    ):
        self.nqubits = int(nqubits)
        if self.nqubits < 0:
            raise ValueError("nqubits must be non-negative")
        self.threshold = _validate_threshold(threshold)
        self.synthesis = _validate_phase_polynomial_synthesis(synthesis)
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
            synthesis=self.synthesis,
        ):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self) -> "PhasePolynomialJW":
        return PhasePolynomialJW(
            {support: -coeff for support, coeff in self.coeffs.items()},
            self.nqubits,
            threshold=self.threshold,
            synthesis=self.synthesis,
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