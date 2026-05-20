from __future__ import annotations

from functools import cache

import numpy as np

from xquces.basis import occ_rows
from xquces.gcr.utils import (
    _default_eta_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_tau_indices,
    _default_triple_indices,
)


@cache
def _number_arrays(norb: int, nelec: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    dim_a = len(occ_a)
    dim_b = len(occ_b)
    n_a = np.zeros((dim_a, norb), dtype=np.float64)
    n_b = np.zeros((dim_b, norb), dtype=np.float64)
    if occ_a.size:
        n_a[np.arange(dim_a)[:, None], occ_a] = 1.0
    if occ_b.size:
        n_b[np.arange(dim_b)[:, None], occ_b] = 1.0
    n = n_a[:, None, :] + n_b[None, :, :]
    d = n_a[:, None, :] * n_b[None, :, :]
    return n.reshape(dim_a * dim_b, norb), d.reshape(dim_a * dim_b, norb)


def _igcr2_feature_matrix(
    parameterization: object,
    nelec: tuple[int, int],
) -> np.ndarray:
    n, _ = _number_arrays(parameterization.norb, nelec)
    if not parameterization.pair_indices:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    rows, cols = zip(*parameterization.pair_indices)
    return n[:, rows] * n[:, cols]


def _igcr3_feature_matrix(
    parameterization: object,
    nelec: tuple[int, int],
) -> np.ndarray:
    n, d = _number_arrays(parameterization.norb, nelec)
    blocks = []
    if parameterization.pair_indices:
        rows, cols = zip(*parameterization.pair_indices)
        blocks.append(n[:, rows] * n[:, cols])
    if parameterization.uses_reduced_cubic_chart:
        tau_idx = _default_tau_indices(parameterization.norb)
        omega_idx = _default_triple_indices(parameterization.norb)
        tau_rows, tau_cols = zip(*tau_idx)
        tau_feat = d[:, tau_rows] * n[:, tau_cols]
        p, q, r = zip(*omega_idx)
        omega_feat = n[:, p] * n[:, q] * n[:, r]
        full = np.concatenate([tau_feat, omega_feat], axis=1)
        blocks.append(full @ parameterization.cubic_reduction.physical_cubic_basis)
    else:
        if parameterization.tau_indices:
            rows, cols = zip(*parameterization.tau_indices)
            blocks.append(d[:, rows] * n[:, cols])
        if parameterization.omega_indices:
            p, q, r = zip(*parameterization.omega_indices)
            blocks.append(n[:, p] * n[:, q] * n[:, r])
    if not blocks:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    return np.concatenate(blocks, axis=1)


def _igcr4_feature_matrix(
    parameterization: object,
    nelec: tuple[int, int],
) -> np.ndarray:
    n, d = _number_arrays(parameterization.norb, nelec)
    blocks = []
    if parameterization.pair_indices:
        rows, cols = zip(*parameterization.pair_indices)
        blocks.append(n[:, rows] * n[:, cols])
    if parameterization.uses_reduced_cubic_chart:
        tau_idx = _default_tau_indices(parameterization.norb)
        omega_idx = _default_triple_indices(parameterization.norb)
        tau_rows, tau_cols = zip(*tau_idx)
        tau_feat = d[:, tau_rows] * n[:, tau_cols]
        p, q, r = zip(*omega_idx)
        omega_feat = n[:, p] * n[:, q] * n[:, r]
        full_cubic = np.concatenate([tau_feat, omega_feat], axis=1)
        blocks.append(
            full_cubic @ parameterization.cubic_reduction.physical_cubic_basis
        )
    else:
        if parameterization.tau_indices:
            rows, cols = zip(*parameterization.tau_indices)
            blocks.append(d[:, rows] * n[:, cols])
        if parameterization.omega_indices:
            p, q, r = zip(*parameterization.omega_indices)
            blocks.append(n[:, p] * n[:, q] * n[:, r])
    if parameterization.uses_reduced_quartic_chart:
        eta_idx = _default_eta_indices(parameterization.norb)
        rho_idx = _default_rho_indices(parameterization.norb)
        sigma_idx = _default_sigma_indices(parameterization.norb)
        eta_p, eta_q = zip(*eta_idx)
        eta_feat = d[:, eta_p] * d[:, eta_q]
        rho_p, rho_q, rho_r = zip(*rho_idx)
        rho_feat = d[:, rho_p] * n[:, rho_q] * n[:, rho_r]
        sigma_p, sigma_q, sigma_r, sigma_s = zip(*sigma_idx)
        sigma_feat = n[:, sigma_p] * n[:, sigma_q] * n[:, sigma_r] * n[:, sigma_s]
        full_quartic = np.concatenate([eta_feat, rho_feat, sigma_feat], axis=1)
        blocks.append(
            full_quartic @ parameterization.quartic_reduction.physical_quartic_basis
        )
    else:
        if parameterization.eta_indices:
            p, q = zip(*parameterization.eta_indices)
            blocks.append(d[:, p] * d[:, q])
        if parameterization.rho_indices:
            p, q, r = zip(*parameterization.rho_indices)
            blocks.append(d[:, p] * n[:, q] * n[:, r])
        if parameterization.sigma_indices:
            p, q, r, s = zip(*parameterization.sigma_indices)
            blocks.append(n[:, p] * n[:, q] * n[:, r] * n[:, s])
    if not blocks:
        return np.zeros((n.shape[0], 0), dtype=np.float64)
    return np.concatenate(blocks, axis=1)


def _diag_feature_matrix(
    parameterization: object,
    nelec: tuple[int, int],
) -> np.ndarray:
    order = getattr(parameterization, "order", None)
    if order == 2:
        return _igcr2_feature_matrix(parameterization, nelec)
    if order == 3:
        return _igcr3_feature_matrix(parameterization, nelec)
    if order == 4:
        return _igcr4_feature_matrix(parameterization, nelec)
    raise TypeError(type(parameterization).__name__)


__all__ = [
    "_diag_feature_matrix",
    "_igcr2_feature_matrix",
    "_igcr3_feature_matrix",
    "_igcr4_feature_matrix",
    "_number_arrays",
]
