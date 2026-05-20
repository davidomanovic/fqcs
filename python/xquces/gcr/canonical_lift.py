from __future__ import annotations

import numpy as np

from xquces.gcr.canonical import IGCRAnsatz, IGCRDiagonalCoefficients
from xquces.gcr.utils import (
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_triple_indices,
)


def _as_spin_restricted_generic(ansatz: IGCRAnsatz | object, *, order: int) -> IGCRAnsatz:
    if isinstance(ansatz, IGCRAnsatz):
        generic = ansatz
    else:
        if hasattr(ansatz, "is_spin_restricted") and not ansatz.is_spin_restricted:
            raise TypeError(
                "canonical iGCR lifts are currently implemented only for "
                "spin-restricted seeds"
            )
        generic = IGCRAnsatz.from_legacy(ansatz, order=order)
    if generic.order != order:
        raise TypeError(f"expected an iGCR-{order} ansatz, got order {generic.order!r}")
    return generic


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


def lift_igcr2_to_igcr3(
    ansatz: IGCRAnsatz | object,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> IGCRAnsatz:
    """Lift a canonical spin-restricted iGCR-2 ansatz to iGCR-3."""

    generic = _as_spin_restricted_generic(ansatz, order=2)
    diagonals = []
    for diagonal in generic.diagonals:
        d2 = diagonal.to_order(2)
        tau, omega = _triples_seed_from_pair_matrix(
            d2.pair_matrix(),
            generic.nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )
        diagonals.append(
            IGCRDiagonalCoefficients(
                order=3,
                norb=d2.norb,
                double_params=d2.full_double(),
                pair_values=d2.pair_values,
                tau=tau,
                omega_values=omega,
                eta_values=d2.eta_vector(),
                rho_values=d2.rho_vector(),
                sigma_values=d2.sigma_vector(),
            )
        )
    return IGCRAnsatz(
        order=3,
        diagonals=tuple(diagonals),
        rotations=generic.rotations,
        nocc=generic.nocc,
    )


def lift_igcr3_to_igcr4(
    ansatz: IGCRAnsatz | object,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCRAnsatz:
    """Lift a canonical spin-restricted iGCR-3 ansatz to iGCR-4."""

    generic = _as_spin_restricted_generic(ansatz, order=3)
    diagonals = []
    for diagonal in generic.diagonals:
        d3 = diagonal.to_order(3)
        eta, rho, sigma = _quartic_seed_from_pair_matrix(
            d3.pair_matrix(),
            generic.nocc,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )
        diagonals.append(
            IGCRDiagonalCoefficients(
                order=4,
                norb=d3.norb,
                double_params=d3.full_double(),
                pair_values=d3.pair_values,
                tau=d3.tau_matrix(),
                omega_values=d3.omega_vector(),
                eta_values=eta,
                rho_values=rho,
                sigma_values=sigma,
            )
        )
    return IGCRAnsatz(
        order=4,
        diagonals=tuple(diagonals),
        rotations=generic.rotations,
        nocc=generic.nocc,
    )


def lift_igcr2_to_igcr4(
    ansatz: IGCRAnsatz | object,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCRAnsatz:
    """Lift a canonical spin-restricted iGCR-2 ansatz directly to iGCR-4."""

    igcr3 = lift_igcr2_to_igcr3(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
    )
    return lift_igcr3_to_igcr4(
        igcr3,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    )

