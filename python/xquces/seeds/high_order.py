from __future__ import annotations

import itertools

import numpy as np

from xquces.seeds.residual import (
    _default_high_order_residual_blocks,
    _parameters_from_ccsd_residual_seed,
)


def _default_triple_indices(norb: int) -> list[tuple[int, int, int]]:
    return list(itertools.combinations(range(norb), 3))


def _default_eta_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _default_rho_indices(norb: int) -> list[tuple[int, int, int]]:
    return [
        (p, q, r)
        for p in range(norb)
        for q in range(norb)
        if q != p
        for r in range(q + 1, norb)
        if r != p
    ]


def _default_sigma_indices(norb: int) -> list[tuple[int, int, int, int]]:
    return list(itertools.combinations(range(norb), 4))


def _triples_seed_from_pair_matrix(
    pair_params: np.ndarray,
    nocc: int,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    if pair.ndim != 2 or pair.shape[0] != pair.shape[1]:
        raise ValueError("pair_params must be a square matrix")
    norb = pair.shape[0]
    denom = max(2 * int(nocc) - 2, 1)

    tau = np.zeros((norb, norb), dtype=np.float64)
    if tau_scale != 0.0:
        for p in range(norb):
            for q in range(norb):
                if p != q:
                    tau[p, q] = float(tau_scale) * pair[p, q] / denom

    omega = np.zeros(len(_default_triple_indices(norb)), dtype=np.float64)
    if omega_scale != 0.0:
        for k, (p, q, r) in enumerate(_default_triple_indices(norb)):
            omega[k] = (
                float(omega_scale)
                * (pair[p, q] + pair[p, r] + pair[q, r])
                / (3.0 * denom)
            )
    return tau, omega


def _quartic_seed_from_pair_matrix(
    pair_params: np.ndarray,
    nocc: int,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    if pair.ndim != 2 or pair.shape[0] != pair.shape[1]:
        raise ValueError("pair_params must be a square matrix")
    norb = pair.shape[0]
    denom = max(2 * int(nocc) - 3, 1)

    eta = np.zeros(len(_default_eta_indices(norb)), dtype=np.float64)
    if eta_scale != 0.0:
        for k, (p, q) in enumerate(_default_eta_indices(norb)):
            eta[k] = float(eta_scale) * 0.5 * pair[p, q] / denom

    rho = np.zeros(len(_default_rho_indices(norb)), dtype=np.float64)
    if rho_scale != 0.0:
        for k, (p, q, r) in enumerate(_default_rho_indices(norb)):
            rho[k] = (
                float(rho_scale)
                * (pair[p, q] + pair[p, r] + pair[q, r])
                / (3.0 * denom)
            )

    sigma = np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64)
    if sigma_scale != 0.0:
        for k, (p, q, r, s) in enumerate(_default_sigma_indices(norb)):
            avg = (
                pair[p, q]
                + pair[p, r]
                + pair[p, s]
                + pair[q, r]
                + pair[q, s]
                + pair[r, s]
            ) / 6.0
            sigma[k] = float(sigma_scale) * avg / denom
    return eta, rho, sigma


def igcr3_parameters_from_t_amplitudes(
    parameterization: object,
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    **seed_options,
) -> np.ndarray:
    """Seed iGCR3 parameters from CCSD amplitudes."""
    from xquces.gcr.igcr import IGCR2SpinRestrictedParameterization

    strategy = seed_options.pop(
        "strategy",
        seed_options.pop("seed_strategy", "ccsd_residual"),
    )
    igcr2_param = IGCR2SpinRestrictedParameterization(
        norb=parameterization.norb,
        nocc=parameterization.nocc,
        layers=parameterization.layers,
        shared_diagonal=parameterization.shared_diagonal,
        interaction_pairs=parameterization.interaction_pairs,
        left_orbital_chart=parameterization.left_orbital_chart,
        middle_orbital_chart=parameterization.middle_orbital_chart,
        right_orbital_chart_override=parameterization.right_orbital_chart_override,
        real_right_orbital_chart=parameterization.real_right_orbital_chart,
        left_right_ov_relative_scale=parameterization.left_right_ov_relative_scale,
    )
    igcr2_strategy = seed_options.pop("igcr2_strategy", "ucj")
    igcr2_options = dict(seed_options.pop("igcr2_options", {}))
    igcr2_params = igcr2_param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy=igcr2_strategy,
        **igcr2_options,
    )
    igcr2_ansatz = igcr2_param.ansatz_from_parameters(igcr2_params)
    x_base = parameterization.parameters_from_igcr2_ansatz(
        igcr2_ansatz,
        tau_scale=0.0,
        omega_scale=0.0,
    )
    if strategy in {"igcr2", "zero_embed", "ucj", "ucj_lift", "ucj-t"}:
        return x_base
    if strategy not in {"ccsd_residual", "state_residual", "residual"}:
        raise ValueError(f"Unknown iGCR3 t-amplitude seed strategy: {strategy!r}")
    active_blocks = seed_options.pop("active_blocks", None)
    if active_blocks is None:
        active_blocks = _default_high_order_residual_blocks(
            parameterization,
            "cubic",
            ("tau", "omega"),
        )
    return _parameters_from_ccsd_residual_seed(
        parameterization,
        t2,
        t1,
        x_base,
        active_blocks=active_blocks,
        target_max_power=seed_options.pop("target_max_power", 4),
        damping=seed_options.pop("damping", 1.0e-8),
        max_step_norm=seed_options.pop("max_step_norm", 0.1),
        scale_scan=seed_options.pop(
            "scale_scan",
            (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
        ),
        n_iter=seed_options.pop("n_iter", 3),
        min_step_norm=seed_options.pop("min_step_norm", 0.0),
        min_overlap_gain=seed_options.pop("min_overlap_gain", 0.0),
        compute_jacobian_rank=seed_options.pop("compute_jacobian_rank", True),
        return_info=seed_options.pop("return_info", False),
    )


def igcr4_parameters_from_t_amplitudes(
    parameterization: object,
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    **seed_options,
) -> np.ndarray:
    """Seed iGCR4 parameters from CCSD amplitudes."""
    from xquces.gcr.igcr import IGCR3SpinRestrictedParameterization

    strategy = seed_options.pop(
        "strategy",
        seed_options.pop("seed_strategy", "ccsd_residual"),
    )
    igcr3_param = IGCR3SpinRestrictedParameterization(
        norb=parameterization.norb,
        nocc=parameterization.nocc,
        layers=parameterization.layers,
        shared_diagonal=parameterization.shared_diagonal,
        interaction_pairs=parameterization.interaction_pairs,
        tau_indices_=parameterization.tau_indices_,
        omega_indices_=parameterization.omega_indices_,
        reduce_cubic_gauge=parameterization.reduce_cubic_gauge,
        left_orbital_chart=parameterization.left_orbital_chart,
        middle_orbital_chart=parameterization.middle_orbital_chart,
        right_orbital_chart_override=parameterization.right_orbital_chart_override,
        real_right_orbital_chart=parameterization.real_right_orbital_chart,
        left_right_ov_relative_scale=parameterization.left_right_ov_relative_scale,
    )
    igcr3_strategy = seed_options.pop("igcr3_strategy", "ccsd_residual")
    igcr3_options = dict(seed_options.pop("igcr3_options", {}))
    if "igcr2_strategy" not in seed_options and "igcr2_strategy" not in igcr3_options:
        igcr3_options["igcr2_strategy"] = "ucj"
    if "igcr2_strategy" in seed_options and "igcr2_strategy" not in igcr3_options:
        igcr3_options["igcr2_strategy"] = seed_options["igcr2_strategy"]
    if "igcr2_options" in seed_options and "igcr2_options" not in igcr3_options:
        igcr3_options["igcr2_options"] = seed_options["igcr2_options"]
    for key in (
        "target_max_power",
        "damping",
        "max_step_norm",
        "scale_scan",
        "n_iter",
        "min_step_norm",
        "min_overlap_gain",
        "compute_jacobian_rank",
    ):
        if key in seed_options and key not in igcr3_options:
            igcr3_options[key] = seed_options[key]
    igcr3_params = seed_options.pop("igcr3_params", None)
    if igcr3_params is None:
        igcr3_params = igcr3_param.parameters_from_t_amplitudes(
            t2,
            t1=t1,
            strategy=igcr3_strategy,
            **igcr3_options,
        )
    else:
        igcr3_params = getattr(igcr3_params, "params", igcr3_params)
        igcr3_params = np.asarray(igcr3_params, dtype=np.float64)
        if igcr3_params.shape != (igcr3_param.n_params,):
            raise ValueError(
                f"igcr3_params must have shape {(igcr3_param.n_params,)}, "
                f"got {igcr3_params.shape}."
            )
    igcr3_ansatz = igcr3_param.ansatz_from_parameters(igcr3_params)
    x_base = parameterization.parameters_from_igcr3_ansatz(
        igcr3_ansatz,
        eta_scale=0.0,
        rho_scale=0.0,
        sigma_scale=0.0,
    )
    if strategy in {"igcr3", "zero_embed", "ucj", "ucj_lift", "ucj-t"}:
        return x_base
    if strategy not in {"ccsd_residual", "state_residual", "residual"}:
        raise ValueError(f"Unknown iGCR4 t-amplitude seed strategy: {strategy!r}")
    active_blocks = seed_options.pop("active_blocks", None)
    if active_blocks is None:
        active_blocks = _default_high_order_residual_blocks(
            parameterization,
            "quartic",
            ("eta", "rho", "sigma"),
        )
    return _parameters_from_ccsd_residual_seed(
        parameterization,
        t2,
        t1,
        x_base,
        active_blocks=active_blocks,
        target_max_power=seed_options.pop("target_max_power", 4),
        damping=seed_options.pop("damping", 1.0e-8),
        max_step_norm=seed_options.pop("max_step_norm", 0.1),
        scale_scan=seed_options.pop(
            "scale_scan",
            (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
        ),
        n_iter=seed_options.pop("n_iter", 3),
        min_step_norm=seed_options.pop("min_step_norm", 0.0),
        min_overlap_gain=seed_options.pop("min_overlap_gain", 0.0),
        compute_jacobian_rank=seed_options.pop("compute_jacobian_rank", True),
        return_info=seed_options.pop("return_info", False),
    )
