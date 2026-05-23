from __future__ import annotations

import numpy as np

from xquces.seeds.residual import (
    _default_high_order_residual_blocks,
    _parameters_from_ccsd_residual_seed,
)


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
