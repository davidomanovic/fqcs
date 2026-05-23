from __future__ import annotations

from dataclasses import dataclass

import ffsim
import numpy as np

from xquces.ansatz.blocks import parameter_blocks


_SUBSPACE_JACOBIAN_DIRECTION_CHUNK = 256


def _solve_real_tikhonov(
    columns: np.ndarray, target: np.ndarray, damping: float
) -> np.ndarray:
    columns = np.asarray(columns, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    A = np.vstack([columns.real, columns.imag])
    b = np.concatenate([target.real, target.imag], axis=0)
    if damping > 0.0:
        n = columns.shape[1]
        A = np.vstack([A, np.sqrt(float(damping)) * np.eye(n)])
        if target.ndim == 1:
            zeros = np.zeros(n, dtype=np.float64)
        else:
            zeros = np.zeros((n, target.shape[1]), dtype=np.float64)
        b = np.concatenate([b, zeros], axis=0)
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(coeff, dtype=np.float64)


@dataclass(frozen=True)
class CCSDResidualSeedInfo:
    """Diagnostics for the non-variational CCSD state-residual seed."""

    params: np.ndarray
    active_blocks: tuple[str, ...]
    raw_delta_norms: tuple[float, ...]
    delta_norms: tuple[float, ...]
    jacobian_ranks: tuple[int, ...]
    scales: tuple[float, ...]
    overlap_before: float
    overlap_after: float


def _as_restricted_t1(
    t1: np.ndarray | None,
    nocc: int,
    nvirt: int,
) -> np.ndarray:
    if t1 is None:
        return np.zeros((nocc, nvirt), dtype=np.float64)
    arr = np.asarray(t1, dtype=np.float64)
    if arr.shape != (nocc, nvirt):
        raise ValueError(f"Expected t1 shape {(nocc, nvirt)}, got {arr.shape}.")
    return arr


def _as_restricted_t2(t2: np.ndarray, nocc: int, nvirt: int) -> np.ndarray:
    arr = np.asarray(t2, dtype=np.float64)
    if arr.shape != (nocc, nocc, nvirt, nvirt):
        raise ValueError(
            f"Expected t2 shape {(nocc, nocc, nvirt, nvirt)}, got {arr.shape}."
        )
    return arr


def _ccsd_target_state_from_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None,
    norb: int,
    nelec: tuple[int, int],
    *,
    max_power: int = 4,
) -> np.ndarray:
    """Build normalize[(1 + T + T^2/2! + ...) |HF>] from CCSD amplitudes."""
    nocc = int(nelec[0])
    nvirt = int(norb) - nocc
    t1 = _as_restricted_t1(t1, nocc, nvirt)
    t2 = _as_restricted_t2(t2, nocc, nvirt)
    reference = ffsim.hartree_fock_state(norb, nelec)

    t1_op = ffsim.linear_operator(
        ffsim.singles_excitations_restricted(t1),
        norb=norb,
        nelec=nelec,
    )
    t2_op = ffsim.linear_operator(
        ffsim.doubles_excitations_restricted(t2),
        norb=norb,
        nelec=nelec,
    )

    def apply_t(vec: np.ndarray) -> np.ndarray:
        return np.asarray(t1_op @ vec + t2_op @ vec, dtype=np.complex128)

    psi = np.array(reference, copy=True, dtype=np.complex128)
    term = np.array(reference, copy=True, dtype=np.complex128)
    for k in range(1, int(max_power) + 1):
        term = apply_t(term) / float(k)
        psi += term

    norm = float(np.linalg.norm(psi))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("CCSD target construction produced a zero or non-finite state")
    return psi / norm


def _phase_align_target(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    overlap = np.vdot(base, target)
    if abs(overlap) > 1.0e-14:
        return np.asarray(target, dtype=np.complex128) * np.exp(
            -1j * np.angle(overlap)
        )
    return np.asarray(target, dtype=np.complex128)


def _state_overlap(target: np.ndarray, state: np.ndarray) -> float:
    return float(abs(np.vdot(target, state)))


def _real_stacked(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    return np.vstack([arr.real, arr.imag])


def _block_column_indices(
    parameterization: object,
    active_blocks: tuple[str, ...] | list[str] | set[str] | None,
) -> np.ndarray:
    if active_blocks is None:
        return np.arange(int(parameterization.n_params), dtype=np.int64)
    active_set = set(active_blocks)
    indices: list[int] = []
    seen: set[str] = set()
    for block in parameter_blocks(parameterization):
        if block.name in active_set:
            indices.extend(range(block.start, block.stop))
            seen.add(block.name)
    missing = active_set - seen
    if missing:
        raise ValueError(f"Unknown parameter block(s): {sorted(missing)!r}")
    return np.asarray(indices, dtype=np.int64)


def _active_state_jacobian_columns(
    parameterization: object,
    reference: np.ndarray,
    nelec: tuple[int, int],
    fixed: object,
    params: np.ndarray,
    columns: np.ndarray,
    state_dim: int,
) -> np.ndarray:
    columns = np.asarray(columns, dtype=np.int64)
    n_params = int(parameterization.n_params)
    if columns.size == n_params and np.array_equal(columns, np.arange(n_params)):
        return np.asarray(
            fixed.state_jacobian_from_parameters(params),
            dtype=np.complex128,
        )
    if columns.size == 0:
        return np.zeros((state_dim, 0), dtype=np.complex128)

    try:
        from xquces.ansatz.jacobian import make_state_subspace_jacobian

        subspace_jacobian = make_state_subspace_jacobian(
            parameterization,
            reference,
            nelec,
        )
    except NotImplementedError:
        jacobian = fixed.state_jacobian_from_parameters(params)
        return np.asarray(jacobian[:, columns], dtype=np.complex128)

    out = np.empty((state_dim, columns.size), dtype=np.complex128)
    for start in range(0, columns.size, _SUBSPACE_JACOBIAN_DIRECTION_CHUNK):
        stop = min(start + _SUBSPACE_JACOBIAN_DIRECTION_CHUNK, columns.size)
        chunk = columns[start:stop]
        directions = np.zeros((n_params, chunk.size), dtype=np.float64)
        directions[chunk, np.arange(chunk.size)] = 1.0
        out[:, start:stop] = np.asarray(
            subspace_jacobian(params, directions),
            dtype=np.complex128,
        )
    return out


def _default_high_order_residual_blocks(
    parameterization: object,
    reduced_name: str,
    full_names: tuple[str, ...],
) -> tuple[str, ...]:
    available = {block.name for block in parameter_blocks(parameterization)}
    if reduced_name in available:
        return (reduced_name,)
    return tuple(name for name in full_names if name in available)


def _state_residual_match_parameters(
    parameterization: object,
    reference: np.ndarray,
    nelec: tuple[int, int],
    x_base: np.ndarray,
    target_state: np.ndarray,
    *,
    active_blocks: tuple[str, ...] | list[str] | set[str] | None = None,
    damping: float = 1.0e-8,
    max_step_norm: float = 0.1,
    scale_scan: tuple[float, ...] | list[float] | None = (
        0.0,
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
    ),
    n_iter: int = 3,
    min_step_norm: float = 0.0,
    min_overlap_gain: float = 0.0,
    compute_jacobian_rank: bool = True,
) -> CCSDResidualSeedInfo:
    """Gauss-Newton state matching to a CCSD target without using a Hamiltonian."""
    x = np.asarray(x_base, dtype=np.float64).copy()
    fixed = parameterization.apply(reference, nelec)
    raw_delta_norms: list[float] = []
    delta_norms: list[float] = []
    jacobian_ranks: list[int] = []
    scales: list[float] = []

    psi0 = fixed.state_from_parameters(x)
    target = _phase_align_target(target_state, psi0)
    overlap_before = _state_overlap(target, psi0)
    best_overlap = overlap_before
    best_x = x.copy()
    columns = _block_column_indices(parameterization, active_blocks)
    active_block_names = (
        tuple(block.name for block in parameter_blocks(parameterization))
        if active_blocks is None
        else tuple(active_blocks)
    )

    for _ in range(int(n_iter)):
        psi = fixed.state_from_parameters(x)
        target = _phase_align_target(target_state, psi)
        current_overlap = _state_overlap(target, psi)
        residual = target - psi
        residual -= psi * np.vdot(psi, residual)

        jacobian_block = _active_state_jacobian_columns(
            parameterization,
            reference,
            nelec,
            fixed,
            x,
            columns,
            psi.size,
        )
        jacobian_block -= psi[:, None] * (psi.conj() @ jacobian_block)[None, :]

        delta = _solve_real_tikhonov(jacobian_block, residual, damping)
        raw_delta_norm = float(np.linalg.norm(delta))
        raw_state_step_norm = float(np.linalg.norm(jacobian_block @ delta))
        delta_norm = raw_delta_norm
        if raw_state_step_norm > max_step_norm:
            delta *= float(max_step_norm) / raw_state_step_norm
            delta_norm = float(np.linalg.norm(delta))

        step = np.zeros_like(x, dtype=np.float64)
        step[columns] = delta
        if compute_jacobian_rank:
            rank = int(np.linalg.matrix_rank(_real_stacked(jacobian_block)))
        else:
            rank = -1

        scale = 1.0
        psi_next = None
        if scale_scan is not None:
            scale = 0.0
            local_best_overlap = current_overlap
            local_best_state = psi
            for candidate in scale_scan:
                candidate = float(candidate)
                if candidate == 0.0:
                    candidate_state = psi
                else:
                    candidate_x = x + candidate * step
                    candidate_state = fixed.state_from_parameters(candidate_x)
                candidate_overlap = _state_overlap(target, candidate_state)
                if candidate_overlap > local_best_overlap + 1.0e-14:
                    local_best_overlap = candidate_overlap
                    scale = candidate
                    local_best_state = candidate_state
            psi_next = local_best_state

        x_next = x + scale * step
        if psi_next is None:
            psi_next = fixed.state_from_parameters(x_next)
        overlap_next = _state_overlap(
            _phase_align_target(target_state, psi_next),
            psi_next,
        )
        accepted_delta_norm = float(abs(scale) * delta_norm)
        overlap_gain = float(overlap_next - current_overlap)

        raw_delta_norms.append(raw_delta_norm)
        delta_norms.append(accepted_delta_norm)
        jacobian_ranks.append(rank)
        scales.append(float(scale))

        if overlap_next > best_overlap + 1.0e-14:
            best_overlap = overlap_next
            best_x = x_next.copy()

        if scale == 0.0:
            break
        if min_step_norm > 0.0 and accepted_delta_norm <= min_step_norm:
            break
        if min_overlap_gain > 0.0 and overlap_gain <= min_overlap_gain:
            break
        x = x_next

    return CCSDResidualSeedInfo(
        params=best_x,
        active_blocks=active_block_names,
        raw_delta_norms=tuple(raw_delta_norms),
        delta_norms=tuple(delta_norms),
        jacobian_ranks=tuple(jacobian_ranks),
        scales=tuple(scales),
        overlap_before=overlap_before,
        overlap_after=best_overlap,
    )


def _parameters_from_ccsd_residual_seed(
    parameterization: object,
    t2: np.ndarray,
    t1: np.ndarray | None,
    x_base: np.ndarray,
    *,
    nelec: tuple[int, int] | None = None,
    active_blocks: tuple[str, ...] | list[str] | set[str] | None = None,
    target_max_power: int = 4,
    damping: float = 1.0e-8,
    max_step_norm: float = 0.1,
    scale_scan: tuple[float, ...] | list[float] | None = (
        0.0,
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
    ),
    n_iter: int = 3,
    min_step_norm: float = 0.0,
    min_overlap_gain: float = 0.0,
    compute_jacobian_rank: bool = True,
    return_info: bool = False,
) -> np.ndarray | CCSDResidualSeedInfo:
    norb = int(parameterization.norb)
    nocc = int(parameterization.nocc)
    if nelec is None:
        nelec = (nocc, nocc)
    nelec = tuple(int(x) for x in nelec)
    reference = ffsim.hartree_fock_state(norb, nelec)
    target = _ccsd_target_state_from_t_amplitudes(
        t2,
        t1,
        norb,
        nelec,
        max_power=target_max_power,
    )
    info = _state_residual_match_parameters(
        parameterization,
        reference,
        nelec,
        x_base,
        target,
        active_blocks=active_blocks,
        damping=damping,
        max_step_norm=max_step_norm,
        scale_scan=scale_scan,
        n_iter=n_iter,
        min_step_norm=min_step_norm,
        min_overlap_gain=min_overlap_gain,
        compute_jacobian_rank=compute_jacobian_rank,
    )
    return info if return_info else info.params
