from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from xquces.gcr.utils import (
    _default_eta_indices,
    _default_pair_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_tau_indices,
    _default_triple_indices,
)


@dataclass(frozen=True)
class IGCR3CubicReduction:
    """Quotient the restricted cubic diagonal basis by fixed-N identities."""

    norb: int
    nocc: int

    def __post_init__(self):
        if self.norb < 0:
            raise ValueError("norb must be nonnegative")
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")

    @property
    def pair_indices(self):
        return _default_pair_indices(self.norb)

    @property
    def tau_indices(self):
        return _default_tau_indices(self.norb)

    @property
    def omega_indices(self):
        return _default_triple_indices(self.norb)

    @property
    def n_pair_full(self):
        return len(self.pair_indices)

    @property
    def n_cubic_full(self):
        return len(self.tau_indices) + len(self.omega_indices)

    @property
    def gauge_pair_matrix(self):
        return _igcr3_cubic_reduction_matrices(self.norb, self.nocc)[0]

    @property
    def gauge_cubic_matrix(self):
        return _igcr3_cubic_reduction_matrices(self.norb, self.nocc)[1]

    @property
    def physical_cubic_basis(self):
        return _igcr3_cubic_reduction_matrices(self.norb, self.nocc)[2]

    @property
    def n_params(self):
        return self.physical_cubic_basis.shape[1]

    def full_from_reduced(self, params):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self.physical_cubic_basis @ params

    def reduce_full(self, pair_values, cubic_values):
        pair_values = np.asarray(pair_values, dtype=np.float64)
        cubic_values = np.asarray(cubic_values, dtype=np.float64)
        if pair_values.shape != (self.n_pair_full,):
            raise ValueError(
                f"Expected pair shape {(self.n_pair_full,)}, got {pair_values.shape}."
            )
        if cubic_values.shape != (self.n_cubic_full,):
            raise ValueError(
                f"Expected cubic shape {(self.n_cubic_full,)}, got {cubic_values.shape}."
            )

        basis = self.physical_cubic_basis
        reduced = basis.T @ cubic_values
        residual = cubic_values - basis @ reduced
        gauge_coeff, *_ = np.linalg.lstsq(
            self.gauge_cubic_matrix,
            residual,
            rcond=None,
        )
        pair_reduced = pair_values - self.gauge_pair_matrix @ gauge_coeff
        onebody_phase = np.zeros(self.norb, dtype=np.float64)
        if self.norb:
            nelec_total = 2 * int(self.nocc)
            onebody_phase[:] = (
                0.5 * (nelec_total - 2) * (nelec_total - 1) * gauge_coeff[: self.norb]
            )
        return pair_reduced, reduced, onebody_phase


@cache
def _igcr3_cubic_reduction_matrices(norb: int, nocc: int):
    pair_indices = _default_pair_indices(norb)
    tau_indices = _default_tau_indices(norb)
    omega_indices = _default_triple_indices(norb)
    n_pair = len(pair_indices)
    n_tau = len(tau_indices)
    n_omega = len(omega_indices)
    n_identity = norb + n_pair
    nelec_total = 2 * int(nocc)

    pair_index = {pair: i for i, pair in enumerate(pair_indices)}
    tau_index = {pair: i for i, pair in enumerate(tau_indices)}
    omega_index = {triple: i for i, triple in enumerate(omega_indices)}

    gauge_pair = np.zeros((n_pair, n_identity), dtype=np.float64)
    gauge_cubic = np.zeros((n_tau + n_omega, n_identity), dtype=np.float64)

    for p in range(norb):
        col = p
        for q in range(norb):
            if p == q:
                continue
            pair = (p, q) if p < q else (q, p)
            gauge_pair[pair_index[pair], col] += 0.5 * (nelec_total - 2)
            gauge_cubic[tau_index[(p, q)], col] += 1.0

    for k, (p, q) in enumerate(pair_indices):
        col = norb + k
        gauge_pair[pair_index[(p, q)], col] -= nelec_total - 2
        gauge_cubic[tau_index[(p, q)], col] += 2.0
        gauge_cubic[tau_index[(q, p)], col] += 2.0
        for r in range(norb):
            if r == p or r == q:
                continue
            triple = tuple(sorted((p, q, r)))
            gauge_cubic[n_tau + omega_index[triple], col] += 1.0

    if gauge_cubic.size == 0:
        physical = np.zeros((0, 0), dtype=np.float64)
        return gauge_pair, gauge_cubic, physical

    u, s, _ = np.linalg.svd(gauge_cubic, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        rank = int(np.sum(s > max(gauge_cubic.shape) * np.finfo(float).eps * s[0]))
    physical = np.array(u[:, rank:], copy=True, dtype=np.float64)
    for j in range(physical.shape[1]):
        col = physical[:, j]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            physical[:, j] *= -1.0
    return gauge_pair, gauge_cubic, physical


@dataclass(frozen=True)
class IGCR4QuarticReduction:
    norb: int
    nocc: int

    def __post_init__(self):
        if self.norb < 0:
            raise ValueError("norb must be nonnegative")
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")

    @property
    def tau_indices(self):
        return _default_tau_indices(self.norb)

    @property
    def omega_indices(self):
        return _default_triple_indices(self.norb)

    @property
    def eta_indices(self):
        return _default_eta_indices(self.norb)

    @property
    def rho_indices(self):
        return _default_rho_indices(self.norb)

    @property
    def sigma_indices(self):
        return _default_sigma_indices(self.norb)

    @property
    def n_cubic_full(self):
        return len(self.tau_indices) + len(self.omega_indices)

    @property
    def n_quartic_full(self):
        return len(self.eta_indices) + len(self.rho_indices) + len(self.sigma_indices)

    @property
    def gauge_cubic_matrix(self):
        return _igcr4_quartic_reduction_matrices(self.norb, self.nocc)[0]

    @property
    def gauge_quartic_matrix(self):
        return _igcr4_quartic_reduction_matrices(self.norb, self.nocc)[1]

    @property
    def physical_quartic_basis(self):
        return _igcr4_quartic_reduction_matrices(self.norb, self.nocc)[2]

    @property
    def n_params(self):
        return max(self.n_quartic_full - self.n_cubic_full, 0)

    def full_from_reduced(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        if params.size == 0 or np.max(np.abs(params)) <= 1e-14:
            return np.zeros(self.n_quartic_full, dtype=np.float64)
        return self.physical_quartic_basis @ params

    def reduce_full(
        self,
        cubic_values: np.ndarray,
        quartic_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cubic_values = np.asarray(cubic_values, dtype=np.float64)
        quartic_values = np.asarray(quartic_values, dtype=np.float64)
        if cubic_values.shape != (self.n_cubic_full,):
            raise ValueError(
                f"Expected cubic shape {(self.n_cubic_full,)}, got {cubic_values.shape}."
            )
        if quartic_values.shape != (self.n_quartic_full,):
            raise ValueError(
                f"Expected quartic shape {(self.n_quartic_full,)}, got {quartic_values.shape}."
            )

        if quartic_values.size == 0 or np.max(np.abs(quartic_values)) <= 1e-14:
            return np.array(cubic_values, copy=True), np.zeros(
                self.n_params,
                dtype=np.float64,
            )

        basis = self.physical_quartic_basis
        reduced = basis.T @ quartic_values
        residual = quartic_values - basis @ reduced
        gauge_coeff, *_ = np.linalg.lstsq(
            self.gauge_quartic_matrix,
            residual,
            rcond=None,
        )
        cubic_reduced = cubic_values - self.gauge_cubic_matrix @ gauge_coeff
        return cubic_reduced, reduced


@cache
def _igcr4_quartic_reduction_matrices(norb: int, nocc: int):
    tau_indices = _default_tau_indices(norb)
    omega_indices = _default_triple_indices(norb)
    eta_indices = _default_eta_indices(norb)
    rho_indices = _default_rho_indices(norb)
    sigma_indices = _default_sigma_indices(norb)

    n_tau = len(tau_indices)
    n_omega = len(omega_indices)
    n_eta = len(eta_indices)
    n_rho = len(rho_indices)
    n_sigma = len(sigma_indices)
    nelec_total = 2 * int(nocc)

    n_id_tau = n_tau
    n_id_omega = n_omega
    n_id = n_id_tau + n_id_omega

    tau_index = {pair: i for i, pair in enumerate(tau_indices)}
    omega_index = {triple: i for i, triple in enumerate(omega_indices)}
    eta_index = {pair: i for i, pair in enumerate(eta_indices)}
    rho_index = {triple: i for i, triple in enumerate(rho_indices)}
    sigma_index = {quad: i for i, quad in enumerate(sigma_indices)}

    gauge_cubic = np.zeros((n_tau + n_omega, n_id), dtype=np.float64)
    gauge_quartic = np.zeros((n_eta + n_rho + n_sigma, n_id), dtype=np.float64)

    for col, (p, q) in enumerate(tau_indices):
        gauge_cubic[tau_index[(p, q)], col] -= nelec_total - 3
        gauge_quartic[eta_index[(p, q) if p < q else (q, p)], col] += 2.0
        for r in range(norb):
            if r == p or r == q:
                continue
            a, b = (q, r) if q < r else (r, q)
            gauge_quartic[n_eta + rho_index[(p, a, b)], col] += 1.0

    for local_col, (p, q, r) in enumerate(omega_indices):
        col = n_id_tau + local_col
        gauge_cubic[n_tau + omega_index[(p, q, r)], col] -= nelec_total - 3
        gauge_quartic[n_eta + rho_index[(p, q, r)], col] += 2.0
        gauge_quartic[n_eta + rho_index[(q, p, r) if p < r else (q, r, p)], col] += 2.0
        gauge_quartic[n_eta + rho_index[(r, p, q) if p < q else (r, q, p)], col] += 2.0
        for s in range(norb):
            if s == p or s == q or s == r:
                continue
            quad = tuple(sorted((p, q, r, s)))
            gauge_quartic[n_eta + n_rho + sigma_index[quad], col] += 1.0

    if gauge_quartic.size == 0:
        physical = np.zeros((0, 0), dtype=np.float64)
        return gauge_cubic, gauge_quartic, physical

    u, s, _ = np.linalg.svd(gauge_quartic, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        rank = int(np.sum(s > max(gauge_quartic.shape) * np.finfo(float).eps * s[0]))
    physical = np.array(u[:, rank:], copy=True, dtype=np.float64)
    for j in range(physical.shape[1]):
        col = physical[:, j]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            physical[:, j] *= -1.0
    return gauge_cubic, gauge_quartic, physical

