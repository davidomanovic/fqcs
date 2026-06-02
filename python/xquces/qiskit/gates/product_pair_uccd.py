from __future__ import annotations

import cmath
import math
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import CircuitInstruction, Gate, QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library import CXGate, SwapGate, UnitaryGate, XGate, XXPlusYYGate

from xquces.gcr.canonical import IGCRAnsatz
from xquces.gcr.utils import relabel_legacy_igcr_ansatz_orbitals
from xquces.gcr.restricted_model import (
    IGCR2Ansatz,
    IGCR2LayeredAnsatz,
    IGCR3Ansatz,
    IGCR3LayeredAnsatz,
    IGCR4Ansatz,
    IGCR4LayeredAnsatz,
)
from xquces.gcr.product_pair_uccd import (
    _pair_uccd_ov_pairs,
    slater_pair_orbital_rotation_from_parameters,
)
from xquces.qiskit.gates.igcr2 import IGCR2JW
from xquces.qiskit.gates.igcr3 import IGCR3JW
from xquces.qiskit.gates.igcr4 import IGCR4JW

IGCRCircuitAnsatz = (
    IGCR2Ansatz
    | IGCR2LayeredAnsatz
    | IGCR3Ansatz
    | IGCR3LayeredAnsatz
    | IGCR4Ansatz
    | IGCR4LayeredAnsatz
    | IGCRAnsatz
)


def _as_legacy_igcr_ansatz(ansatz: IGCRCircuitAnsatz):
    if isinstance(ansatz, IGCRAnsatz):
        return ansatz.to_legacy()
    return ansatz


def _normalize_nelec(nelec: tuple[int, int] | Sequence[int]) -> tuple[int, int]:
    if len(nelec) != 2:
        raise ValueError("nelec must contain (n_alpha, n_beta)")
    out = (int(nelec[0]), int(nelec[1]))
    if out[0] != out[1]:
        raise ValueError("Product pair-UCCD requires n_alpha == n_beta")
    return out


def _validate_product_pair_uccd_params(
    norb: int,
    nelec: tuple[int, int] | Sequence[int],
    params: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray]:
    nelec = _normalize_nelec(nelec)
    if nelec[0] < 0 or nelec[0] > int(norb):
        raise ValueError("invalid electron count")
    out = np.asarray(params, dtype=np.float64)
    expected = len(_pair_uccd_ov_pairs(int(norb), nelec[0]))
    if out.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {out.shape}.")
    return nelec, out


def _pair_uccd_rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    out = np.eye(16, dtype=np.complex128)
    old_pair = 0b0101
    new_pair = 0b1010
    out[old_pair, old_pair] = c
    out[new_pair, new_pair] = c
    out[new_pair, old_pair] = s
    out[old_pair, new_pair] = -s
    return out


class PairUCCDRotationJW(Gate):
    def __init__(self, theta: float, *, label: str | None = None):
        self.theta = float(theta)
        super().__init__("pair_uccd_rotation_jw", 4, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        circuit.append(UnitaryGate(_pair_uccd_rotation_matrix(self.theta)), qubits)
        self.definition = circuit

    def inverse(self) -> "PairUCCDRotationJW":
        return PairUCCDRotationJW(-self.theta, label=self.label)


class PairRegisterUCCDGivensJW(Gate):
    def __init__(self, theta: float, *, label: str | None = None):
        self.theta = float(theta)
        super().__init__("pair_register_uccd_givens_jw", 2, [], label=label)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        circuit.append(XXPlusYYGate(2.0 * self.theta, 0.5 * np.pi), qubits)
        self.definition = circuit

    def inverse(self) -> "PairRegisterUCCDGivensJW":
        return PairRegisterUCCDGivensJW(-self.theta, label=self.label)


class ProductPairUCCDJW(Gate):
    def __init__(
        self,
        norb: int,
        nelec: tuple[int, int] | Sequence[int],
        params: np.ndarray,
        *,
        time: float = 1.0,
        label: str | None = None,
    ):
        self.norb = int(norb)
        self.nelec, self.params_array = _validate_product_pair_uccd_params(
            self.norb,
            nelec,
            params,
        )
        self.time = float(time)
        super().__init__("product_pair_uccd_jw", 2 * self.norb, [], label=label)

    @property
    def nocc(self) -> int:
        return self.nelec[0]

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return _pair_uccd_ov_pairs(self.norb, self.nocc)

    def _define(self) -> None:
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _product_pair_uccd_jw(
                qubits,
                self.norb,
                self.nelec,
                self.params_array,
                time=self.time,
            ),
            qubits=qubits,
            name=self.name,
        )

    def inverse(self) -> "ProductPairUCCDJW":
        return ProductPairUCCDJW(
            self.norb,
            self.nelec,
            self.params_array,
            time=-self.time,
            label=self.label,
        )


def product_pair_uccd_jw_circuit(
    norb: int,
    nelec: tuple[int, int] | Sequence[int],
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * int(norb))
    circuit.append(ProductPairUCCDJW(norb, nelec, params, time=time), circuit.qubits)
    return circuit


def product_pair_uccd_stateprep_jw_circuit(
    norb: int,
    nelec: tuple[int, int] | Sequence[int],
    params: np.ndarray,
    *,
    time: float = 1.0,
    strategy: str = "pair_register",
) -> QuantumCircuit:
    nelec, params = _validate_product_pair_uccd_params(norb, nelec, params)
    circuit = QuantumCircuit(2 * int(norb))
    strategy = _normalize_stateprep_strategy(strategy)
    if strategy == "pair_register_direct":
        instructions = _product_pair_uccd_pair_register_direct_stateprep_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
        )
    elif strategy == "pair_register_slater":
        instructions = _product_pair_uccd_pair_register_slater_stateprep_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
        )
    elif strategy == "pair_register_swap_network":
        instructions, _ = _product_pair_uccd_pair_register_instructions_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
            restore_order=True,
        )
    elif strategy == "pair_register_permuted":
        instructions, _ = _product_pair_uccd_pair_register_instructions_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
            restore_order=False,
        )
    elif strategy == "spin_orbital":
        for p in range(nelec[0]):
            circuit.x(p)
        for p in range(nelec[1]):
            circuit.x(int(norb) + p)
        circuit.append(ProductPairUCCDJW(norb, nelec, params, time=time), circuit.qubits)
        return circuit
    else:
        raise AssertionError("unreachable")
    for instruction in instructions:
        circuit.append(instruction)
    return circuit


def product_pair_uccd_pair_register_stateprep_jw_circuit(
    norb: int,
    nelec: tuple[int, int] | Sequence[int],
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> QuantumCircuit:
    return product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        time=time,
        strategy="pair_register",
    )


def product_pair_uccd_igcr_stateprep_jw_circuit(
    ansatz: IGCRCircuitAnsatz,
    reference_params: np.ndarray,
    *,
    nelec: tuple[int, int] | Sequence[int] | None = None,
    time: float = 1.0,
    validate_orbital_rotations: bool = True,
    sparsify_diagonal: bool = True,
    sparsify_atol: float = 1e-12,
    puccd_strategy: str = "pair_register",
) -> QuantumCircuit:
    ansatz = _as_legacy_igcr_ansatz(ansatz)
    if nelec is None:
        nelec = (ansatz.nocc, ansatz.nocc)
    nelec = _normalize_nelec(nelec)
    if nelec != (ansatz.nocc, ansatz.nocc):
        raise ValueError("nelec must match ansatz.nocc for product pair-UCCD")

    strategy = _normalize_stateprep_strategy(puccd_strategy)
    if strategy == "pair_register_permuted" and isinstance(
        ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)
    ):
        circuit = QuantumCircuit(2 * ansatz.norb)
        instructions, old_for_new = _product_pair_uccd_pair_register_instructions_jw(
            circuit.qubits,
            ansatz.norb,
            nelec,
            np.asarray(reference_params, dtype=np.float64),
            time=time,
            restore_order=False,
        )
        for instruction in instructions:
            circuit.append(instruction)
        relabeled = relabel_legacy_igcr_ansatz_orbitals(
            ansatz,
            np.asarray(old_for_new, dtype=np.int64),
            order=2,
        )
        circuit.append(
            _igcr_gate_from_ansatz(
                relabeled,
                validate_orbital_rotations=validate_orbital_rotations,
                sparsify_diagonal=sparsify_diagonal,
                sparsify_atol=sparsify_atol,
            ),
            circuit.qubits,
        )
        return circuit

    circuit = product_pair_uccd_stateprep_jw_circuit(
        ansatz.norb,
        nelec,
        reference_params,
        time=time,
        strategy=strategy,
    )
    circuit.append(
        _igcr_gate_from_ansatz(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
            sparsify_diagonal=sparsify_diagonal,
            sparsify_atol=sparsify_atol,
        ),
        circuit.qubits,
    )
    return circuit


def gcr_product_pair_uccd_stateprep_jw_circuit(
    parameterization,
    params: np.ndarray,
    *,
    time: float = 1.0,
    validate_orbital_rotations: bool = True,
    sparsify_diagonal: bool = True,
    sparsify_atol: float = 1e-12,
    puccd_strategy: str = "pair_register",
) -> QuantumCircuit:
    if not hasattr(parameterization, "split_parameters"):
        raise TypeError("parameterization must implement split_parameters")
    if not hasattr(parameterization, "ansatz_from_parameters"):
        raise TypeError("parameterization must implement ansatz_from_parameters")
    if not hasattr(parameterization, "norb") or not hasattr(parameterization, "nocc"):
        raise TypeError("parameterization must expose norb and nocc")

    reference_params, ansatz_params = parameterization.split_parameters(params)
    ansatz = parameterization.ansatz_from_parameters(ansatz_params)
    return product_pair_uccd_igcr_stateprep_jw_circuit(
        ansatz,
        reference_params,
        nelec=(parameterization.nocc, parameterization.nocc),
        time=time,
        validate_orbital_rotations=validate_orbital_rotations,
        sparsify_diagonal=sparsify_diagonal,
        sparsify_atol=sparsify_atol,
        puccd_strategy=puccd_strategy,
    )


def _product_pair_uccd_jw(
    qubits: Sequence[Qubit],
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")
    for theta, (i, a) in zip(time * params, _pair_uccd_ov_pairs(norb, nelec[0])):
        if theta == 0.0:
            continue
        yield CircuitInstruction(
            PairUCCDRotationJW(float(theta)),
            (qubits[i], qubits[a], qubits[norb + i], qubits[norb + a]),
        )


def _product_pair_uccd_pair_register_stateprep_jw(
    qubits: Sequence[Qubit],
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float,
) -> Iterator[CircuitInstruction]:
    yield from _product_pair_uccd_pair_register_direct_stateprep_jw(
        qubits,
        norb,
        nelec,
        params,
        time=time,
    )


def _product_pair_uccd_pair_register_slater_stateprep_jw(
    qubits: Sequence[Qubit],
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")
    nocc = nelec[0]
    pair_qubits = tuple(qubits[:norb])
    occupied_orbitals = _pair_register_occupied_orbitals(
        norb,
        nocc,
        params,
        time=time,
    )
    yield from _prepare_spinless_slater_occupied_orbitals_jw(
        pair_qubits,
        occupied_orbitals,
    )
    for p in range(norb):
        yield CircuitInstruction(CXGate(), (qubits[p], qubits[norb + p]))


def _pair_register_occupied_orbitals(
    norb: int,
    nocc: int,
    params: np.ndarray,
    *,
    time: float,
) -> np.ndarray:
    orbital_rotation = _pair_register_orbital_rotation(norb, nocc, params, time=time)
    return orbital_rotation[:, :nocc]


def _prepare_spinless_slater_occupied_orbitals_jw(
    qubits: Sequence[Qubit],
    occupied_orbitals: np.ndarray,
) -> Iterator[CircuitInstruction]:
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=np.complex128)
    norb = len(qubits)
    if occupied_orbitals.ndim != 2 or occupied_orbitals.shape[0] != norb:
        raise ValueError(
            "occupied_orbitals must have shape (norb, n_particles), "
            f"got {occupied_orbitals.shape}."
        )
    n_particles = occupied_orbitals.shape[1]
    for p in range(n_particles):
        yield CircuitInstruction(XGate(), (qubits[p],))
    if n_particles == norb:
        return
    for c, s, i, j in _slater_givens_rotations(occupied_orbitals.T):
        theta = math.acos(max(-1.0, min(1.0, c)))
        if abs(s.imag) <= 1e-12:
            if s.real > 0:
                theta = -theta
            yield CircuitInstruction(
                PairRegisterUCCDGivensJW(theta),
                (qubits[i], qubits[j]),
            )
        else:
            yield CircuitInstruction(
                XXPlusYYGate(2.0 * theta, cmath.phase(s) - 0.5 * math.pi),
                (qubits[i], qubits[j]),
            )


def _slater_givens_rotations(
    orbital_coeffs: np.ndarray,
    *,
    atol: float = 1e-12,
) -> list[tuple[float, complex, int, int]]:
    n_particles, norb = orbital_coeffs.shape
    current = np.array(orbital_coeffs, dtype=np.complex128, copy=True)
    rotations: list[tuple[float, complex, int, int]] = []

    for col in reversed(range(norb - n_particles + 1, norb)):
        for row in range(n_particles - norb + col):
            if abs(current[row, col]) <= atol:
                continue
            c, s = _givens_zero_second(current[row + 1, col], current[row, col])
            lower = np.array(current[row + 1], copy=True)
            upper = np.array(current[row], copy=True)
            current[row + 1] = c * lower + s * upper
            current[row] = c * upper - np.conjugate(s) * lower

    for row in range(n_particles):
        for col in range(norb - n_particles + row, row, -1):
            if abs(current[row, col]) <= atol:
                continue
            c, s = _givens_zero_second(current[row, col - 1], current[row, col])
            rotations.append((c, s, col, col - 1))
            left = np.array(current[:, col - 1], copy=True)
            right = np.array(current[:, col], copy=True)
            current[:, col - 1] = c * left + s * right
            current[:, col] = c * right - np.conjugate(s) * left

    return rotations[::-1]


def _givens_zero_second(a: complex, b: complex) -> tuple[float, complex]:
    if b == 0:
        return 1.0, 0.0j
    norm = math.hypot(abs(a), abs(b))
    if norm == 0.0:
        return 1.0, 0.0j
    if a == 0:
        return 0.0, np.conjugate(b) / abs(b)
    c = abs(a) / norm
    s = c * np.conjugate(b / a)
    return float(c), complex(s)


def _pair_register_orbital_rotation(
    norb: int,
    nocc: int,
    params: np.ndarray,
    *,
    time: float,
) -> np.ndarray:
    return slater_pair_orbital_rotation_from_parameters(
        norb,
        (nocc, nocc),
        params,
        time=time,
    )


def _product_pair_uccd_pair_register_instructions_jw(
    qubits: Sequence[Qubit],
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float,
    restore_order: bool,
) -> tuple[list[CircuitInstruction], tuple[int, ...]]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")
    nocc = nelec[0]
    instructions: list[CircuitInstruction] = []
    for p in range(nocc):
        instructions.append(CircuitInstruction(XGate(), (qubits[p],)))
    pair_qubits = qubits[:norb]
    pair_sites = list(range(norb))
    for theta, (i, a) in zip(time * params, _pair_uccd_ov_pairs(norb, nocc)):
        if theta == 0.0:
            continue
        instructions.extend(_move_pair_site_next_to(pair_qubits, pair_sites, i, a))
        pos_i = pair_sites.index(i)
        pos_a = pair_sites.index(a)
        instructions.append(
            CircuitInstruction(
                PairRegisterUCCDGivensJW(float(theta)),
                (pair_qubits[pos_i], pair_qubits[pos_a]),
            )
        )
    if restore_order:
        instructions.extend(_restore_pair_site_order(pair_qubits, pair_sites))
    for pos in range(norb):
        instructions.append(CircuitInstruction(CXGate(), (qubits[pos], qubits[norb + pos])))
    return instructions, tuple(pair_sites)


def _product_pair_uccd_pair_register_direct_stateprep_jw(
    qubits: Sequence[Qubit],
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float,
) -> Iterator[CircuitInstruction]:
    if len(qubits) != 2 * norb:
        raise ValueError("Expected 2 * norb qubits.")
    nocc = nelec[0]
    for p in range(nocc):
        yield CircuitInstruction(XGate(), (qubits[p],))
    for theta, (i, a) in zip(time * params, _pair_uccd_ov_pairs(norb, nocc)):
        if theta == 0.0:
            continue
        yield CircuitInstruction(
            PairRegisterUCCDGivensJW(float(theta)),
            (qubits[i], qubits[a]),
        )
    for p in range(norb):
        yield CircuitInstruction(CXGate(), (qubits[p], qubits[norb + p]))


def _move_pair_site_next_to(
    qubits: Sequence[Qubit],
    sites: list[int],
    mover: int,
    target: int,
) -> Iterator[CircuitInstruction]:
    while True:
        pos_mover = sites.index(mover)
        pos_target = sites.index(target)
        if abs(pos_mover - pos_target) <= 1:
            return
        step = 1 if pos_mover < pos_target else -1
        pos_next = pos_mover + step
        yield _swap_pair_sites(qubits, sites, pos_mover, pos_next)


def _restore_pair_site_order(
    qubits: Sequence[Qubit],
    sites: list[int],
) -> Iterator[CircuitInstruction]:
    target = list(range(len(sites)))
    while sites != target:
        for left in range(len(sites) - 1):
            right = left + 1
            if sites[left] > sites[right]:
                yield _swap_pair_sites(qubits, sites, left, right)
                break
        else:
            raise RuntimeError("Could not restore pair-register qubit order.")


def _swap_pair_sites(
    qubits: Sequence[Qubit],
    sites: list[int],
    left: int,
    right: int,
) -> CircuitInstruction:
    if abs(left - right) != 1:
        raise ValueError("Pair-register swaps must be nearest-neighbor.")
    lo, hi = sorted((left, right))
    sites[lo], sites[hi] = sites[hi], sites[lo]
    return CircuitInstruction(SwapGate(), (qubits[lo], qubits[hi]))


def _normalize_stateprep_strategy(strategy: str) -> str:
    key = str(strategy).lower().replace("-", "_")
    if key in {"pair", "pair_register", "logical_pair", "logical_pairs", "pair_register_direct", "logical_pair_direct", "dense_pair_register", "direct_pair_register"}:
        return "pair_register_direct"
    if key in {"pair_register_slater", "slater_pair_register", "fermionic_pair_register", "pair_register_fermionic"}:
        return "pair_register_slater"
    if key in {"swap_network", "pair_register_swap_network"}:
        return "pair_register_swap_network"
    if key in {"pair_register_permuted", "permuted_pair_register", "no_restore_pair_register", "pair_register_no_restore"}:
        return "pair_register_permuted"
    if key in {"spin_orbital", "full", "unitary", "four_qubit", "naive"}:
        return "spin_orbital"
    raise ValueError("strategy must be 'pair_register', 'pair_register_slater', 'pair_register_swap_network', 'pair_register_permuted', or 'spin_orbital'")


def _igcr_gate_from_ansatz(
    ansatz: IGCRCircuitAnsatz,
    *,
    validate_orbital_rotations: bool,
    sparsify_diagonal: bool,
    sparsify_atol: float,
) -> Gate:
    ansatz = _as_legacy_igcr_ansatz(ansatz)
    if isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
        return IGCR2JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
            sparsify_diagonal=sparsify_diagonal,
            sparsify_atol=sparsify_atol,
        )
    if isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
        return IGCR3JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
        )
    if isinstance(ansatz, (IGCR4Ansatz, IGCR4LayeredAnsatz)):
        return IGCR4JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
        )
    raise TypeError(
        "ansatz must be a canonical IGCRAnsatz or a legacy iGCR ansatz"
    )
