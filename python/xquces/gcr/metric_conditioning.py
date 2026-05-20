from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from xquces.gcr.igcr import IGCR2SpinRestrictedParameterization
from xquces.states import hartree_fock_state


@dataclass(frozen=True)
class MetricConditioningInfo:
    cond: float
    n_soft: int
    state_loss: float
    eigvals: np.ndarray
    eigvecs: np.ndarray


def _normalize_state(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(out))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("state norm is zero or non-finite")
    return out / norm


def _state(parameterization: Any, params: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ansatz = parameterization.ansatz_from_parameters(params)
    nelec = (parameterization.nocc, parameterization.nocc)
    return _normalize_state(ansatz.apply(reference, nelec, copy=True))


def _state_loss(state: np.ndarray, target: np.ndarray) -> float:
    overlap = abs(np.vdot(target, state))
    return float(max(0.0, 1.0 - min(1.0, overlap * overlap)))


def _finite_difference_jacobian(
    parameterization: Any,
    reference: np.ndarray,
    finite_difference_step: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        psi = _state(parameterization, params, reference)
        out = np.empty((psi.size, params.size), dtype=np.complex128)
        for idx in range(params.size):
            step = finite_difference_step * max(1.0, abs(float(params[idx])))
            plus = params.copy()
            minus = params.copy()
            plus[idx] += step
            minus[idx] -= step
            out[:, idx] = (_state(parameterization, plus, reference) - _state(parameterization, minus, reference)) / (2.0 * step)
        return out

    return jac


def _jacobian_factory(
    parameterization: Any,
    reference: np.ndarray,
    finite_difference_step: float,
) -> Callable[[np.ndarray], np.ndarray]:
    nelec = (parameterization.nocc, parameterization.nocc)
    try:
        from xquces.ansatz.jacobian import make_state_jacobian

        return make_state_jacobian(parameterization, reference, nelec)
    except Exception:
        return _finite_difference_jacobian(parameterization, reference, finite_difference_step)


def _metric_from_jacobian(jacobian: np.ndarray, state: np.ndarray) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=np.complex128)
    projected = jac - state[:, None] * (state.conj() @ jac)[None, :]
    metric = np.real(projected.conj().T @ projected)
    return 0.5 * (metric + metric.T)


def _metric_info(
    parameterization: Any,
    params: np.ndarray,
    reference: np.ndarray,
    target_state: np.ndarray,
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    target_cond: float,
    state: np.ndarray | None = None,
) -> MetricConditioningInfo:
    psi = _state(parameterization, params, reference) if state is None else state
    metric = _metric_from_jacobian(jacobian_fn(params), psi)
    eigvals, eigvecs = np.linalg.eigh(metric)
    eigvals = np.maximum(eigvals, 0.0)
    if eigvals.size == 0:
        return MetricConditioningInfo(float("inf"), 0, _state_loss(psi, target_state), eigvals, eigvecs)
    maxeig = float(eigvals[-1])
    if maxeig == 0.0 or not np.isfinite(maxeig):
        return MetricConditioningInfo(float("inf"), eigvals.size, _state_loss(psi, target_state), eigvals, eigvecs)
    absolute_floor = max(1e-14, 1e-14 * maxeig)
    positive = eigvals[eigvals > absolute_floor]
    cond = float("inf") if positive.size == 0 else float(maxeig / positive[0])
    soft_floor = max(absolute_floor, maxeig / float(target_cond))
    n_soft = int(np.count_nonzero(eigvals <= soft_floor))
    return MetricConditioningInfo(cond, n_soft, _state_loss(psi, target_state), eigvals, eigvecs)


def _block_indices(parameterization: Any, name: str) -> np.ndarray:
    n_left = parameterization.n_left_orbital_rotation_params
    n_pair = parameterization.n_pair_params
    n_middle = parameterization.n_middle_orbital_rotation_params
    n_right = parameterization.n_right_orbital_rotation_params
    if name == "left":
        start, stop = 0, n_left
    elif name == "pair":
        start, stop = n_left, n_left + n_pair
    elif name == "middle":
        start, stop = n_left + n_pair, n_left + n_pair + n_middle
    elif name == "right":
        start, stop = n_left + n_pair + n_middle, n_left + n_pair + n_middle + n_right
    elif name == "nonright":
        start, stop = 0, n_left + n_pair + n_middle
    elif name == "all":
        start, stop = 0, n_left + n_pair + n_middle + n_right
    else:
        raise ValueError(name)
    return np.arange(start, stop, dtype=np.int64)


def _unit_direction(n_params: int, idx: int) -> np.ndarray:
    out = np.zeros(n_params, dtype=np.float64)
    out[idx] = 1.0
    return out


def _normalized(values: np.ndarray) -> np.ndarray | None:
    out = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(out))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return out / norm


def _random_block_direction(rng: np.random.Generator, n_params: int, indices: np.ndarray) -> np.ndarray | None:
    if indices.size == 0:
        return None
    out = np.zeros(n_params, dtype=np.float64)
    out[indices] = rng.normal(size=indices.size)
    return _normalized(out)


def _ramp_block_direction(n_params: int, indices: np.ndarray) -> np.ndarray | None:
    if indices.size == 0:
        return None
    out = np.zeros(n_params, dtype=np.float64)
    k = np.arange(1, indices.size + 1, dtype=np.float64)
    out[indices] = np.sin(np.sqrt(2.0) * k) + 0.5 * np.cos(np.sqrt(3.0) * k)
    return _normalized(out)


def _soft_directions(info: MetricConditioningInfo, target_cond: float, max_soft_directions: int) -> list[np.ndarray]:
    eigvals = info.eigvals
    eigvecs = info.eigvecs
    if eigvals.size == 0:
        return []
    maxeig = float(eigvals[-1])
    floor = max(1e-14, 1e-14 * maxeig, maxeig / float(target_cond))
    indices = np.flatnonzero(eigvals <= floor)
    if indices.size == 0:
        indices = np.arange(min(max_soft_directions, eigvals.size), dtype=np.int64)
    out = []
    for idx in indices[:max_soft_directions]:
        direction = _normalized(eigvecs[:, idx])
        if direction is not None:
            out.append(direction)
    return out


def _direction_candidates(
    parameterization: Any,
    info: MetricConditioningInfo,
    target_cond: float,
    max_soft_directions: int,
    random_directions: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    n_params = parameterization.n_params
    pair = _block_indices(parameterization, "pair")
    middle = _block_indices(parameterization, "middle")
    nonright = _block_indices(parameterization, "nonright")
    all_indices = _block_indices(parameterization, "all")
    for direction in (_ramp_block_direction(n_params, pair), _ramp_block_direction(n_params, middle), _ramp_block_direction(n_params, nonright)):
        if direction is not None:
            yield direction
    for idx in pair:
        yield _unit_direction(n_params, int(idx))
    for idx in middle:
        yield _unit_direction(n_params, int(idx))
    for direction in _soft_directions(info, target_cond, max_soft_directions):
        yield direction
    for _ in range(random_directions):
        for indices in (pair, middle, nonright, all_indices):
            direction = _random_block_direction(rng, n_params, indices)
            if direction is not None:
                yield direction


def _finite_cond_value(value: float) -> float:
    if np.isfinite(value):
        return float(value)
    return 1e300


def _better_metric(candidate: MetricConditioningInfo, best: MetricConditioningInfo, max_state_loss: float, cond_improvement: float) -> bool:
    if candidate.state_loss > max_state_loss:
        return False
    if candidate.n_soft < best.n_soft:
        return True
    if candidate.n_soft > best.n_soft:
        return False
    best_cond = _finite_cond_value(best.cond)
    candidate_cond = _finite_cond_value(candidate.cond)
    return candidate_cond < cond_improvement * best_cond


def condition_igcr2_parameterization_metric(
    parameterization: Any,
    params: np.ndarray,
    *,
    target_cond: float = 1e8,
    max_state_loss: float = 1e-3,
    finite_difference_step: float = 1e-5,
    max_soft_directions: int = 8,
    max_trials: int = 240,
    max_rounds: int = 4,
    random_directions: int = 16,
    cond_improvement: float = 0.8,
    displacement_scales: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 7e-1),
    random_seed: int = 271828,
    verbose: bool = True,
) -> np.ndarray:
    x0 = np.asarray(params, dtype=np.float64)
    if x0.shape != (parameterization.n_params,):
        raise ValueError(f"Expected {(parameterization.n_params,)}, got {x0.shape}.")
    if target_cond <= 1.0 or not np.isfinite(target_cond):
        raise ValueError("target_cond must be finite and greater than one")
    if max_state_loss <= 0.0 or not np.isfinite(max_state_loss):
        raise ValueError("max_state_loss must be finite and positive")
    reference = hartree_fock_state(parameterization.norb, (parameterization.nocc, parameterization.nocc))
    target_state = _state(parameterization, x0, reference)
    jacobian_fn = _jacobian_factory(parameterization, reference, finite_difference_step)
    initial = _metric_info(parameterization, x0, reference, target_state, jacobian_fn, target_cond, target_state)
    best_x = x0.copy()
    best_info = initial
    trials = 0
    rng = np.random.default_rng(random_seed)
    for _ in range(max_rounds):
        if best_info.n_soft == 0 and best_info.cond <= target_cond:
            break
        round_x = best_x.copy()
        round_info = best_info
        for direction in _direction_candidates(parameterization, best_info, target_cond, max_soft_directions, random_directions, rng):
            if trials >= max_trials:
                break
            for scale in displacement_scales:
                if trials >= max_trials:
                    break
                for sign in (1.0, -1.0):
                    if trials >= max_trials:
                        break
                    trials += 1
                    candidate_x = best_x + sign * float(scale) * direction
                    candidate_state = _state(parameterization, candidate_x, reference)
                    if _state_loss(candidate_state, target_state) > max_state_loss:
                        continue
                    candidate_info = _metric_info(parameterization, candidate_x, reference, target_state, jacobian_fn, target_cond, candidate_state)
                    if _better_metric(candidate_info, round_info, max_state_loss, cond_improvement):
                        round_x = candidate_x
                        round_info = candidate_info
                    if round_info.n_soft == 0 and round_info.cond <= target_cond:
                        break
        if not _better_metric(round_info, best_info, max_state_loss, cond_improvement):
            break
        best_x = round_x
        best_info = round_info
    accepted = _better_metric(best_info, initial, max_state_loss, cond_improvement)
    if verbose:
        print(
            f"iGCR2 metric-conditioned seed: cond {initial.cond:.3e} -> {best_info.cond:.3e}, "
            f"soft {initial.n_soft} -> {best_info.n_soft}, state_loss {initial.state_loss:.3e} -> {best_info.state_loss:.3e}, "
            f"accepted {accepted}, trials {trials}"
        )
    return best_x if accepted else x0


def _metric_conditioned_parameters_from_t_amplitudes(
    self: IGCR2SpinRestrictedParameterization,
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    metric_conditioning: bool = False,
    condition_metric: bool = False,
    **options: Any,
) -> np.ndarray:
    conditioning_options = {}
    aliases = {
        "metric_conditioning_target_cond": "target_cond",
        "metric_conditioning_max_state_loss": "max_state_loss",
        "max_conditioning_iter": "max_trials",
        "metric_conditioning_max_trials": "max_trials",
    }
    for old, new in aliases.items():
        if old in options:
            conditioning_options[new] = options.pop(old)
    conditioning_keys = {
        "target_cond",
        "max_state_loss",
        "finite_difference_step",
        "max_soft_directions",
        "max_trials",
        "max_rounds",
        "random_directions",
        "cond_improvement",
        "displacement_scales",
        "random_seed",
        "verbose",
    }
    for key in list(options):
        if key in conditioning_keys:
            conditioning_options[key] = options.pop(key)
    mode = options.pop("metric_conditioning_mode", "escape")
    for key in list(options):
        if key.startswith("metric_conditioning_") or key.startswith("conditioning_"):
            options.pop(key)
    if mode not in ("escape", None):
        raise ValueError("only escape metric conditioning is supported")
    original = getattr(self, "_xquces_unconditioned_parameters_from_t_amplitudes")
    params = original(t2, t1=t1, **options)
    if not (metric_conditioning or condition_metric):
        return params
    return condition_igcr2_parameterization_metric(self, params, **conditioning_options)


def install_metric_conditioned_initializer() -> None:
    cls = IGCR2SpinRestrictedParameterization
    if hasattr(cls, "_xquces_unconditioned_parameters_from_t_amplitudes"):
        return
    setattr(cls, "_xquces_unconditioned_parameters_from_t_amplitudes", cls.parameters_from_t_amplitudes)
    setattr(cls, "parameters_from_t_amplitudes", _metric_conditioned_parameters_from_t_amplitudes)
