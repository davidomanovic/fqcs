from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from ffsim.qiskit.gates import PrepareHartreeFockJW, PrepareSlaterDeterminantSpinlessJW
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import CXGate, SwapGate, UnitaryGate, XGate, XXPlusYYGate

from xquces.gcr.igcr import (
    IGCR2Ansatz,
    IGCR2LayeredAnsatz,
    IGCR3Ansatz,
    IGCR4Ansatz,
    relabel_igcr2_ansatz_orbitals,
)
from xquces.gcr.product_pair_uccd import _pair_uccd_ov_pairs
from xquces.qiskit.gates.igcr2 import IGCR2JW
from xquces.qiskit.gates.igcr3 import IGCR3JW
from xquces.qiskit.gates.igcr4 import IGCR4JW


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
    def __init__(
        self,
        theta: float,
        *,
        label: str | None = None,
    ):
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
    def __init__(
        self,
        theta: float,
        *,
        label: str | None = None,
    ):
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
    circuit.append(
        ProductPairUCCDJW(norb, nelec, params, time=time),
        circuit.qubits,
    )
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
    if strategy == "pair_register_slater":
        for instruction in _product_pair_uccd_pair_register_slater_stateprep_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
        ):
            circuit.append(instruction)
    elif strategy == "pair_register_swap_network":
        instructions, _ = _product_pair_uccd_pair_register_instructions_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
            restore_order=True,
        )
        for instruction in instructions:
            circuit.append(instruction)
    elif strategy == "pair_register_permuted":
        instructions, _ = _product_pair_uccd_pair_register_instructions_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
            restore_order=False,
        )
        for instruction in instructions:
            circuit.append(instruction)
    elif strategy == "pair_register_direct":
        for instruction in _product_pair_uccd_pair_register_direct_stateprep_jw(
            circuit.qubits,
            int(norb),
            nelec,
            params,
            time=time,
        ):
            circuit.append(instruction)
    elif strategy == "spin_orbital":
        circuit.append(PrepareHartreeFockJW(int(norb), nelec), circuit.qubits)
        circuit.append(
            ProductPairUCCDJW(norb, nelec, params, time=time),
            circuit.qubits,
        )
    else:
        raise AssertionError("unreachable")
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
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz | IGCR3Ansatz | IGCR4Ansatz,
    reference_params: np.ndarray,
    *,
    nelec: tuple[int, int] | Sequence[int] | None = None,
    time: float = 1.0,
    validate_orbital_rotations: bool = True,
    sparsify_diagonal: bool = True,
    sparsify_atol: float = 1e-12,
    puccd_strategy: str = "pair_register",
) -> QuantumCircuit:
    if nelec is None:
        nelec = (ansatz.nocc, ansatz.nocc)
    nelec = _normalize_nelec(nelec)
    if nelec != (ansatz.nocc, ansatz.nocc):
        raise ValueError("nelec must match ansatz.nocc for product pair-UCCD")

    strategy = _normalize_stateprep_strategy(puccd_strategy)
    if strategy == "pair_register_permuted" and isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
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
        relabeled = relabel_igcr2_ansatz_orbitals(
            ansatz,
            np.asarray(old_for_new, dtype=np.int64),
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
            (
                qubits[i],
                qubits[a],
                qubits[norb + i],
                qubits[norb + a],
            ),
        )


def _product_pair_uccd_pair_register_stateprep_jw(
    qubits: Sequence[Qubit],
    norb: int,
    nelec: tuple[int, int],
    params: np.ndarray,
    *,
    time: float,
) -> Iterator[CircuitInstruction]:
    yield from _product_pair_uccd_pair_register_slater_stateprep_jw(
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
    orbital_rotation = _pair_register_orbital_rotation(norb, nocc, params, time=time)
    yield CircuitInstruction(
        PrepareSlaterDeterminantSpinlessJW(
            norb,
            range(nocc),
            orbital_rotation=orbital_rotation,
        ),
        tuple(qubits[:norb]),
    )
    for p in range(norb):
        yield CircuitInstruction(CXGate(), (qubits[p], qubits[norb + p]))


def _pair_register_orbital_rotation(
    norb: int,
    nocc: int,
    params: np.ndarray,
    *,
    time: float,
) -> np.ndarray:
    orbital_rotation = np.eye(norb, dtype=np.complex128)
    for theta, (i, a) in zip(time * params, _pair_uccd_ov_pairs(norb, nocc)):
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        row_i = np.array(orbital_rotation[i], copy=True)
        row_a = np.array(orbital_rotation[a], copy=True)
        orbital_rotation[i] = c * row_i - s * row_a
        orbital_rotation[a] = s * row_i + c * row_a
    return orbital_rotation


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
    if key in {"pair", "pair_register", "logical_pair", "logical_pairs", "pair_register_slater", "slater_pair_register"}:
        return "pair_register_slater"
    if key in {"swap_network", "pair_register_swap_network"}:
        return "pair_register_swap_network"
    if key in {"pair_register_permuted", "permuted_pair_register", "no_restore_pair_register", "pair_register_no_restore"}:
        return "pair_register_permuted"
    if key in {"pair_register_direct", "logical_pair_direct", "dense_pair_register", "direct_pair_register"}:
        return "pair_register_direct"
    if key in {"spin_orbital", "full", "unitary", "four_qubit", "naive"}:
        return "spin_orbital"
    raise ValueError("strategy must be 'pair_register', 'pair_register_swap_network', 'pair_register_permuted', 'pair_register_direct', or 'spin_orbital'")


def _igcr_gate_from_ansatz(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz | IGCR3Ansatz | IGCR4Ansatz,
    *,
    validate_orbital_rotations: bool,
    sparsify_diagonal: bool,
    sparsify_atol: float,
) -> Gate:
    if isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
        return IGCR2JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
            sparsify_diagonal=sparsify_diagonal,
            sparsify_atol=sparsify_atol,
        )
    if isinstance(ansatz, IGCR3Ansatz):
        return IGCR3JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
        )
    if isinstance(ansatz, IGCR4Ansatz):
        return IGCR4JW(
            ansatz,
            validate_orbital_rotations=validate_orbital_rotations,
        )
    raise TypeError(
        "ansatz must be an IGCR2Ansatz, IGCR2LayeredAnsatz, IGCR3Ansatz, or IGCR4Ansatz"
    )
