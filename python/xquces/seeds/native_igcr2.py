from __future__ import annotations

import ffsim
import numpy as np
import scipy.linalg

from xquces.basis import occ_indicator_rows, reshape_state
from xquces.orbitals import apply_orbital_rotation
from xquces.seeds.residual import _as_restricted_t1, _solve_real_tikhonov


def _dense_ov_mixer(nvirt: int, nocc: int) -> np.ndarray:
    a = np.arange(nvirt, dtype=np.float64)[:, None]
    i = np.arange(nocc, dtype=np.float64)[None, :]
    mixer = 1.0 / (a + i + np.sqrt(2.0) + 1.0)
    norm = np.linalg.norm(mixer)
    if norm:
        mixer /= norm
    return mixer


def _dense_pair_mixer(pair_indices: list[tuple[int, int]]) -> np.ndarray:
    vals = np.sin((np.arange(1, len(pair_indices) + 1) * 1.61803398875))
    norm = np.linalg.norm(vals)
    if norm:
        vals /= norm
    return np.asarray(vals, dtype=np.float64)


def _reference_ov_unitary_from_z(z: np.ndarray, nocc: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    nvirt, nocc_z = z.shape
    if nocc_z != nocc:
        raise ValueError("Z has inconsistent occupied dimension")
    norb = nocc + nvirt
    kappa = np.zeros((norb, norb), dtype=np.float64)
    kappa[:nocc, nocc:] = -z.T
    kappa[nocc:, :nocc] = z
    return np.asarray(scipy.linalg.expm(kappa), dtype=np.complex128)


def _pair_values(
    norb: int, nelec: tuple[int, int], pair_indices: list[tuple[int, int]]
) -> np.ndarray:
    occ_a = occ_indicator_rows(norb, nelec[0]).astype(np.float64)
    occ_b = occ_indicator_rows(norb, nelec[1]).astype(np.float64)
    occ = occ_a[:, None, :] + occ_b[None, :, :]
    out = np.empty((*occ.shape[:2], len(pair_indices)), dtype=np.float64)
    for col, (p, q) in enumerate(pair_indices):
        out[..., col] = occ[..., p] * occ[..., q]
    return out.reshape((-1, len(pair_indices)))


def _columns_metric_info(columns: np.ndarray) -> dict[str, float | int]:
    if columns.size == 0:
        return {"rank": 0, "n_zero": 0, "n_soft": 0, "cond": np.inf}
    S = (columns.conj().T @ columns).real
    S = 0.5 * (S + S.T)
    vals = np.maximum(np.linalg.eigvalsh(S), 0.0)
    max_val = float(vals[-1]) if vals.size else 0.0
    if max_val == 0.0:
        return {
            "rank": 0,
            "n_zero": int(vals.size),
            "n_soft": int(vals.size),
            "cond": np.inf,
        }
    zero_cutoff = 1.0e-12 * max_val
    soft_cutoff = 1.0e-8 * max_val
    nonzero = vals[vals > zero_cutoff]
    return {
        "rank": int(nonzero.size),
        "n_zero": int(np.count_nonzero(vals <= zero_cutoff)),
        "n_soft": int(np.count_nonzero(vals < soft_cutoff)),
        "cond": float(max_val / nonzero[0]) if nonzero.size else np.inf,
    }


def _pair_tangent_matrix(
    state: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    pair_indices: list[tuple[int, int]],
) -> np.ndarray:
    flat = reshape_state(state, norb, nelec).reshape(-1)
    values = _pair_values(norb, nelec, pair_indices)
    prob = np.abs(flat) ** 2
    prob /= float(np.sum(prob))
    mean = prob @ values
    return 1j * flat[:, None] * (values - mean)


def _restricted_ccsd_first_order_state(
    phi0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    singles: bool,
    doubles: bool,
) -> np.ndarray:
    out = np.zeros_like(phi0, dtype=np.complex128)
    if singles and t1.size and np.any(t1):
        op = ffsim.singles_excitations_restricted(t1)
        out += ffsim.linear_operator(op, norb=norb, nelec=nelec) @ phi0
    if doubles and t2.size and np.any(t2):
        op = ffsim.doubles_excitations_restricted(t2)
        out += ffsim.linear_operator(op, norb=norb, nelec=nelec) @ phi0
    return out


def _left_tangent_matrix(
    chart: object,
    chi: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    step: float = 1.0e-6,
) -> np.ndarray:
    n_params = chart.n_params(norb)
    columns = np.empty((chi.size, n_params), dtype=np.complex128)
    zero = np.zeros(n_params, dtype=np.float64)
    for idx in range(n_params):
        params = zero.copy()
        params[idx] = step
        left = chart.unitary_from_parameters(params, norb)
        columns[:, idx] = (
            apply_orbital_rotation(chi, left, norb=norb, nelec=nelec) - chi
        ) / step
    return columns


def _full_projective_metric_info(
    parameterization: object,
    params: np.ndarray,
    phi0: np.ndarray,
    nelec: tuple[int, int],
) -> dict[str, float | int]:
    fixed = parameterization.apply(phi0, nelec)
    psi = fixed.state_from_parameters(params)
    jac = fixed.state_jacobian_from_parameters(params)
    jac = jac - psi.reshape((-1, 1)) * (psi.conj() @ jac).reshape((1, -1))
    return _columns_metric_info(jac)


def native_igcr2_seed_from_ccsd_t_amplitudes(
    parameterization: object,
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    right_mixing_eps: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8),
    target_scales: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
    j_mixing_scales: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2),
    j_damping: float = 1.0e-8,
    left_damping: float = 1.0e-8,
    max_soft: int = 0,
    cond_j_max: float = 1.0e12,
    cond_s_max: float = 1.0e12,
    hamiltonian: object | None = None,
    verbose: bool = False,
):
    from xquces.gcr.igcr import IGCR2Ansatz, IGCR2SpinRestrictedSpec

    if parameterization.layers != 1:
        raise ValueError("native iGCR2 t-amplitude seed is implemented for one layer")

    t2 = np.asarray(t2, dtype=np.float64)
    nocc = parameterization.nocc
    nvirt = parameterization.norb - nocc
    if t2.shape != (nocc, nocc, nvirt, nvirt):
        raise ValueError(f"Expected t2 shape {(nocc, nocc, nvirt, nvirt)}, got {t2.shape}.")
    t1 = _as_restricted_t1(t1, nocc, nvirt)

    norb = parameterization.norb
    nelec = (nocc, nocc)
    phi0 = ffsim.hartree_fock_state(norb, nelec)
    pair_indices = parameterization.pair_indices
    n_pair = len(pair_indices)
    mixer = _dense_ov_mixer(nvirt, nocc)
    pair_mixer = _dense_pair_mixer(pair_indices)
    t2_state = _restricted_ccsd_first_order_state(
        phi0, t1, t2, norb, nelec, singles=False, doubles=True
    )
    first_order_state = _restricted_ccsd_first_order_state(
        phi0, t1, t2, norb, nelec, singles=True, doubles=True
    )

    candidates = []
    for eps in right_mixing_eps:
        z = t1.T + float(eps) * mixer
        right = _reference_ov_unitary_from_z(z, nocc)
        right_state = apply_orbital_rotation(phi0, right, norb=norb, nelec=nelec)
        pair_columns = _pair_tangent_matrix(right_state, norb, nelec, pair_indices)
        j_metric = _columns_metric_info(pair_columns)
        if j_metric["rank"] < n_pair or j_metric["cond"] > cond_j_max:
            if verbose:
                print(
                    "native iGCR2 seed rejected right eps="
                    f"{eps:g} J_rank={j_metric['rank']}/{n_pair} "
                    f"J_cond={j_metric['cond']:.6g}",
                    flush=True,
                )
            continue

        for scale in target_scales:
            j_fit = _solve_real_tikhonov(
                pair_columns, float(scale) * t2_state, j_damping
            )
            for j_mix_scale in j_mixing_scales:
                j_values = j_fit + float(j_mix_scale) * pair_mixer
                pair = np.zeros((norb, norb), dtype=np.float64)
                for value, (p, q) in zip(j_values, pair_indices):
                    pair[p, q] = pair[q, p] = value

                no_left = IGCR2Ansatz(
                    diagonal=IGCR2SpinRestrictedSpec(pair=pair),
                    left=np.eye(norb, dtype=np.complex128),
                    right=right,
                    nocc=nocc,
                )
                chi = no_left.apply(phi0, nelec, copy=True)
                target = phi0 + float(scale) * first_order_state
                residual = target - chi
                left_columns = _left_tangent_matrix(
                    parameterization._left_orbital_chart, chi, norb, nelec
                )
                left_values = _solve_real_tikhonov(left_columns, residual, left_damping)
                left = parameterization._left_orbital_chart.unitary_from_parameters(
                    left_values, norb
                )
                ansatz = IGCR2Ansatz(
                    diagonal=IGCR2SpinRestrictedSpec(pair=pair),
                    left=left,
                    right=right,
                    nocc=nocc,
                )
                try:
                    params = parameterization.parameters_from_ansatz(ansatz)
                except Exception:
                    if verbose:
                        print(
                            f"native iGCR2 seed skipped eps={eps:g} scale={scale:g} "
                            f"j_mix={j_mix_scale:g}: outside parameter chart",
                            flush=True,
                        )
                    continue

                psi = ansatz.apply(phi0, nelec, copy=True)
                residual_norm = float(np.linalg.norm(target - psi))
                metric = _full_projective_metric_info(
                    parameterization, params, phi0, nelec
                )
                energy = (
                    float(np.vdot(psi, hamiltonian @ psi).real)
                    if hamiltonian is not None
                    else np.nan
                )
                accepted = (
                    metric["n_soft"] <= max_soft
                    and metric["cond"] <= cond_s_max
                    and j_metric["cond"] <= cond_j_max
                )
                score_value = energy if hamiltonian is not None else residual_norm
                candidates.append(
                    {
                        "ansatz": ansatz,
                        "accepted": bool(accepted),
                        "score": float(score_value),
                        "residual_norm": residual_norm,
                        "metric": metric,
                        "j_metric": j_metric,
                        "eps": float(eps),
                        "scale": float(scale),
                        "j_mix": float(j_mix_scale),
                    }
                )
                if verbose:
                    print(
                        f"native iGCR2 seed eps={eps:g} scale={scale:g} "
                        f"j_mix={j_mix_scale:g} score={score_value:.12g} "
                        f"n_soft={metric['n_soft']} cond={metric['cond']:.6g} "
                        f"J_cond={j_metric['cond']:.6g}",
                        flush=True,
                    )

    if not candidates:
        raise RuntimeError("native iGCR2 seed produced no valid candidates")

    accepted = [row for row in candidates if row["accepted"]]
    if accepted:
        best = min(
            accepted,
            key=lambda row: (row["score"], row["metric"]["cond"], row["residual_norm"]),
        )
    else:
        best = min(
            candidates,
            key=lambda row: (
                row["metric"]["n_soft"],
                row["j_metric"]["n_soft"],
                row["score"],
                row["metric"]["cond"],
            ),
        )
    if verbose:
        metric = best["metric"]
        print(
            f"native iGCR2 seed selected eps={best['eps']:g} "
            f"scale={best['scale']:g} j_mix={best['j_mix']:g} "
            f"n_soft={metric['n_soft']} "
            f"cond={metric['cond']:.6g}",
            flush=True,
        )
    return best["ansatz"]


__all__ = [
    "native_igcr2_seed_from_ccsd_t_amplitudes",
]
