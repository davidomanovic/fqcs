from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xquces.ansatz.parameters import ParameterBlock
from xquces.charts.reductions import IGCR3CubicReduction, IGCR4QuarticReduction
from xquces.gcr.utils import (
    _default_eta_indices,
    _default_pair_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_tau_indices,
    _default_triple_indices,
    _ordered_matrix_from_values,
    _restricted_irreducible_pair_matrix,
    _restricted_left_phase_vector,
    _symmetric_matrix_from_values,
    _validate_ordered_pairs,
    _validate_pairs,
    _validate_rho_indices,
    _validate_sigma_indices,
    _validate_triples,
    _values_from_ordered_matrix,
)


def _prefixed(prefix: str, name: str) -> str:
    return name if prefix == "" else f"{prefix}.{name}"


@dataclass(frozen=True)
class RestrictedPairCoefficients:
    pair: np.ndarray


@dataclass(frozen=True)
class RestrictedCubicCoefficients:
    double_params: np.ndarray
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray


@dataclass(frozen=True)
class RestrictedPairChart:
    """Parameter chart for the spin-restricted iGCR-2 diagonal correlator."""

    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def n_params(self) -> int:
        return len(self.pair_indices)

    def blocks(self, prefix: str = "") -> tuple[ParameterBlock, ...]:
        return (
            ParameterBlock(
                _prefixed(prefix, "pair"),
                0,
                self.n_params,
                shape=(self.n_params,),
                kind="diagonal",
            ),
        )

    def coefficients_from_parameters(
        self,
        params: np.ndarray,
    ) -> RestrictedPairCoefficients:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return RestrictedPairCoefficients(
            pair=_symmetric_matrix_from_values(params, self.norb, self.pair_indices)
        )

    def parameters_from_coefficients(
        self,
        coefficients: RestrictedPairCoefficients,
    ) -> tuple[np.ndarray, np.ndarray]:
        pair = np.asarray(coefficients.pair, dtype=np.float64)
        if pair.shape != (self.norb, self.norb):
            raise ValueError("pair must have shape (norb, norb)")
        params = np.asarray(
            [pair[p, q] for p, q in self.pair_indices],
            dtype=np.float64,
        )
        return params, np.zeros(self.norb, dtype=np.float64)


@dataclass(frozen=True)
class RestrictedQuarticCoefficients:
    double_params: np.ndarray
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray
    eta_values: np.ndarray
    rho_values: np.ndarray
    sigma_values: np.ndarray


@dataclass(frozen=True)
class RestrictedCubicChart:
    """Parameter chart for the spin-restricted iGCR-3 diagonal correlator."""

    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    reduce_cubic_gauge: bool = True

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        _validate_ordered_pairs(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _validate_ordered_pairs(self.tau_indices_, self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _validate_triples(self.omega_indices_, self.norb)

    @property
    def uses_reduced_cubic_chart(self) -> bool:
        return (
            self.reduce_cubic_gauge
            and self.pair_indices == _default_pair_indices(self.norb)
            and self.tau_indices == _default_tau_indices(self.norb)
            and self.omega_indices == _default_triple_indices(self.norb)
        )

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        return IGCR3CubicReduction(self.norb, self.nocc)

    @property
    def n_pair_params(self) -> int:
        return len(self.pair_indices)

    @property
    def n_tau_params(self) -> int:
        if self.uses_reduced_cubic_chart:
            return self.cubic_reduction.n_params
        return len(self.tau_indices)

    @property
    def n_omega_params(self) -> int:
        if self.uses_reduced_cubic_chart:
            return 0
        return len(self.omega_indices)

    @property
    def n_params(self) -> int:
        return self.n_pair_params + self.n_tau_params + self.n_omega_params

    def blocks(self, prefix: str = "") -> tuple[ParameterBlock, ...]:
        blocks: list[ParameterBlock] = []
        start = 0
        stop = start + self.n_pair_params
        blocks.append(
            ParameterBlock(
                _prefixed(prefix, "pair"),
                start,
                stop,
                shape=(self.n_pair_params,),
                kind="diagonal",
            )
        )
        start = stop
        if self.uses_reduced_cubic_chart:
            stop = start + self.n_tau_params
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "cubic"),
                    start,
                    stop,
                    shape=(self.n_tau_params,),
                    kind="diagonal",
                )
            )
            return tuple(blocks)
        stop = start + self.n_tau_params
        if stop > start:
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "tau"),
                    start,
                    stop,
                    shape=(self.n_tau_params,),
                    kind="diagonal",
                )
            )
        start = stop
        stop = start + self.n_omega_params
        if stop > start:
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "omega"),
                    start,
                    stop,
                    shape=(self.n_omega_params,),
                    kind="diagonal",
                )
            )
        return tuple(blocks)

    def coefficients_from_parameters(
        self,
        params: np.ndarray,
    ) -> RestrictedCubicCoefficients:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0

        n = self.n_pair_params
        pair_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
        pair_sparse = _symmetric_matrix_from_values(
            pair_sparse_values,
            self.norb,
            self.pair_indices,
        )
        pair_values = np.asarray(
            [pair_sparse[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            cubic = self.cubic_reduction.full_from_reduced(params[idx : idx + n])
            n_tau_full = len(_default_tau_indices(self.norb))
            tau = _ordered_matrix_from_values(
                cubic[:n_tau_full],
                self.norb,
                _default_tau_indices(self.norb),
            )
            omega_values = np.asarray(cubic[n_tau_full:], dtype=np.float64)
            idx += n
        else:
            n = self.n_tau_params
            tau = _ordered_matrix_from_values(
                params[idx : idx + n], self.norb, self.tau_indices
            )
            idx += n

            n = self.n_omega_params
            omega_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
            omega_sparse = {
                triple: value
                for triple, value in zip(self.omega_indices, omega_sparse_values)
            }
            omega_values = np.asarray(
                [
                    omega_sparse.get(triple, 0.0)
                    for triple in _default_triple_indices(self.norb)
                ],
                dtype=np.float64,
            )
            idx += n

        if idx != self.n_params:
            raise ValueError("diagonal parameter block has inconsistent length")
        return RestrictedCubicCoefficients(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=pair_values,
            tau=tau,
            omega_values=omega_values,
        )

    def parameters_from_coefficients(
        self,
        coefficients: RestrictedCubicCoefficients,
    ) -> tuple[np.ndarray, np.ndarray]:
        double = np.asarray(coefficients.double_params, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double_params has inconsistent shape")
        pair_values = np.asarray(coefficients.pair_values, dtype=np.float64)
        if pair_values.shape != (len(_default_pair_indices(self.norb)),):
            raise ValueError("pair_values has inconsistent shape")
        tau = np.asarray(coefficients.tau, dtype=np.float64)
        if tau.shape != (self.norb, self.norb):
            raise ValueError("tau must have shape (norb, norb)")
        tau = np.array(tau, copy=True, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)
        omega = np.asarray(coefficients.omega_values, dtype=np.float64)
        if omega.shape != (len(_default_triple_indices(self.norb)),):
            raise ValueError("omega_values has inconsistent shape")

        pair_matrix = _symmetric_matrix_from_values(
            pair_values,
            self.norb,
            _default_pair_indices(self.norb),
        )
        pair_eff = _restricted_irreducible_pair_matrix(double, pair_matrix)
        full_pair_values = np.asarray(
            [pair_eff[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        full_cubic = np.concatenate(
            [
                _values_from_ordered_matrix(tau, _default_tau_indices(self.norb)),
                omega,
            ]
        )
        reduced_pair_values, reduced_cubic_values, cubic_onebody_phase = (
            self.cubic_reduction.reduce_full(full_pair_values, full_cubic)
        )

        phase_vec = (
            _restricted_left_phase_vector(double, self.nocc) + cubic_onebody_phase
        )

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        pair_reduced_matrix = _symmetric_matrix_from_values(
            reduced_pair_values, self.norb, _default_pair_indices(self.norb)
        )
        n = self.n_pair_params
        out[idx : idx + n] = np.asarray(
            [pair_reduced_matrix[p, q] for p, q in self.pair_indices], dtype=np.float64
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            out[idx : idx + n] = reduced_cubic_values
            idx += n
        else:
            full_cubic_adjusted = self.cubic_reduction.full_from_reduced(
                reduced_cubic_values
            )
            n_tau_full = len(_default_tau_indices(self.norb))
            tau_adjusted = _ordered_matrix_from_values(
                full_cubic_adjusted[:n_tau_full],
                self.norb,
                _default_tau_indices(self.norb),
            )
            omega_adjusted = {
                triple: val
                for triple, val in zip(
                    _default_triple_indices(self.norb),
                    full_cubic_adjusted[n_tau_full:],
                )
            }
            n = self.n_tau_params
            out[idx : idx + n] = _values_from_ordered_matrix(
                tau_adjusted, self.tau_indices
            )
            idx += n

            n = self.n_omega_params
            out[idx : idx + n] = np.asarray(
                [omega_adjusted[t] for t in self.omega_indices], dtype=np.float64
            )
            idx += n

        if idx != self.n_params:
            raise ValueError("diagonal parameter block has inconsistent length")
        return out, phase_vec


@dataclass(frozen=True)
class RestrictedQuarticChart:
    """Parameter chart for the spin-restricted iGCR-4 diagonal correlator."""

    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    eta_indices_: list[tuple[int, int]] | None = None
    rho_indices_: list[tuple[int, int, int]] | None = None
    sigma_indices_: list[tuple[int, int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
    reduce_quartic_gauge: bool = True

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        _validate_ordered_pairs(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)
        _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)
        _validate_rho_indices(self.rho_indices_, self.norb)
        _validate_sigma_indices(self.sigma_indices_, self.norb)

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _validate_ordered_pairs(self.tau_indices_, self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _validate_triples(self.omega_indices_, self.norb)

    @property
    def eta_indices(self) -> list[tuple[int, int]]:
        return _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)

    @property
    def rho_indices(self) -> list[tuple[int, int, int]]:
        return _validate_rho_indices(self.rho_indices_, self.norb)

    @property
    def sigma_indices(self) -> list[tuple[int, int, int, int]]:
        return _validate_sigma_indices(self.sigma_indices_, self.norb)

    @property
    def uses_reduced_cubic_chart(self) -> bool:
        return (
            self.reduce_cubic_gauge
            and self.pair_indices == _default_pair_indices(self.norb)
            and self.tau_indices == _default_tau_indices(self.norb)
            and self.omega_indices == _default_triple_indices(self.norb)
        )

    @property
    def uses_reduced_quartic_chart(self) -> bool:
        return (
            self.reduce_quartic_gauge
            and self.tau_indices == _default_tau_indices(self.norb)
            and self.omega_indices == _default_triple_indices(self.norb)
            and self.eta_indices == _default_eta_indices(self.norb)
            and self.rho_indices == _default_rho_indices(self.norb)
            and self.sigma_indices == _default_sigma_indices(self.norb)
        )

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        return IGCR3CubicReduction(self.norb, self.nocc)

    @property
    def quartic_reduction(self) -> IGCR4QuarticReduction:
        return IGCR4QuarticReduction(self.norb, self.nocc)

    @property
    def n_pair_params(self) -> int:
        return len(self.pair_indices)

    @property
    def n_tau_params(self) -> int:
        if self.uses_reduced_cubic_chart:
            return self.cubic_reduction.n_params
        return len(self.tau_indices)

    @property
    def n_omega_params(self) -> int:
        if self.uses_reduced_cubic_chart:
            return 0
        return len(self.omega_indices)

    @property
    def n_eta_params(self) -> int:
        if self.uses_reduced_quartic_chart:
            return 0
        return len(self.eta_indices)

    @property
    def n_rho_params(self) -> int:
        if self.uses_reduced_quartic_chart:
            return self.quartic_reduction.n_params
        return len(self.rho_indices)

    @property
    def n_sigma_params(self) -> int:
        if self.uses_reduced_quartic_chart:
            return 0
        return len(self.sigma_indices)

    @property
    def n_params(self) -> int:
        return (
            self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma_params
        )

    def blocks(self, prefix: str = "") -> tuple[ParameterBlock, ...]:
        blocks: list[ParameterBlock] = []
        start = 0
        stop = start + self.n_pair_params
        blocks.append(
            ParameterBlock(
                _prefixed(prefix, "pair"),
                start,
                stop,
                shape=(self.n_pair_params,),
                kind="diagonal",
            )
        )
        start = stop
        if self.uses_reduced_cubic_chart:
            stop = start + self.n_tau_params
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "cubic"),
                    start,
                    stop,
                    shape=(self.n_tau_params,),
                    kind="diagonal",
                )
            )
            start = stop
        else:
            stop = start + self.n_tau_params
            if stop > start:
                blocks.append(
                    ParameterBlock(
                        _prefixed(prefix, "tau"),
                        start,
                        stop,
                        shape=(self.n_tau_params,),
                        kind="diagonal",
                    )
                )
            start = stop
            stop = start + self.n_omega_params
            if stop > start:
                blocks.append(
                    ParameterBlock(
                        _prefixed(prefix, "omega"),
                        start,
                        stop,
                        shape=(self.n_omega_params,),
                        kind="diagonal",
                    )
                )
            start = stop

        if self.uses_reduced_quartic_chart:
            stop = start + self.n_rho_params
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "quartic"),
                    start,
                    stop,
                    shape=(self.n_rho_params,),
                    kind="diagonal",
                )
            )
            return tuple(blocks)

        stop = start + self.n_eta_params
        if stop > start:
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "eta"),
                    start,
                    stop,
                    shape=(self.n_eta_params,),
                    kind="diagonal",
                )
            )
        start = stop
        stop = start + self.n_rho_params
        if stop > start:
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "rho"),
                    start,
                    stop,
                    shape=(self.n_rho_params,),
                    kind="diagonal",
                )
            )
        start = stop
        stop = start + self.n_sigma_params
        if stop > start:
            blocks.append(
                ParameterBlock(
                    _prefixed(prefix, "sigma"),
                    start,
                    stop,
                    shape=(self.n_sigma_params,),
                    kind="diagonal",
                )
            )
        return tuple(blocks)

    def coefficients_from_parameters(
        self,
        params: np.ndarray,
    ) -> RestrictedQuarticCoefficients:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0

        n = self.n_pair_params
        pair_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
        pair_sparse = _symmetric_matrix_from_values(
            pair_sparse_values,
            self.norb,
            self.pair_indices,
        )
        pair_values = np.asarray(
            [pair_sparse[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            cubic = self.cubic_reduction.full_from_reduced(params[idx : idx + n])
            n_tau_full = len(_default_tau_indices(self.norb))
            tau = _ordered_matrix_from_values(
                cubic[:n_tau_full],
                self.norb,
                _default_tau_indices(self.norb),
            )
            omega_values = np.asarray(cubic[n_tau_full:], dtype=np.float64)
            idx += n
        else:
            n = self.n_tau_params
            tau = _ordered_matrix_from_values(
                params[idx : idx + n], self.norb, self.tau_indices
            )
            idx += n

            n = self.n_omega_params
            omega_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
            omega_sparse = {
                triple: value
                for triple, value in zip(self.omega_indices, omega_sparse_values)
            }
            omega_values = np.asarray(
                [
                    omega_sparse.get(triple, 0.0)
                    for triple in _default_triple_indices(self.norb)
                ],
                dtype=np.float64,
            )
            idx += n

        if self.uses_reduced_quartic_chart:
            n = self.n_rho_params
            quartic = self.quartic_reduction.full_from_reduced(params[idx : idx + n])
            n_eta_full = len(_default_eta_indices(self.norb))
            n_rho_full = len(_default_rho_indices(self.norb))
            eta_values = np.asarray(quartic[:n_eta_full], dtype=np.float64)
            rho_values = np.asarray(
                quartic[n_eta_full : n_eta_full + n_rho_full], dtype=np.float64
            )
            sigma_values = np.asarray(
                quartic[n_eta_full + n_rho_full :], dtype=np.float64
            )
            idx += n
        else:
            n = self.n_eta_params
            eta_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
            eta_sparse = {
                pair: value for pair, value in zip(self.eta_indices, eta_sparse_values)
            }
            eta_values = np.asarray(
                [eta_sparse.get(pair, 0.0) for pair in _default_eta_indices(self.norb)],
                dtype=np.float64,
            )
            idx += n

            n = self.n_rho_params
            rho_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
            rho_sparse = {
                triple: value
                for triple, value in zip(self.rho_indices, rho_sparse_values)
            }
            rho_values = np.asarray(
                [
                    rho_sparse.get(triple, 0.0)
                    for triple in _default_rho_indices(self.norb)
                ],
                dtype=np.float64,
            )
            idx += n

            n = self.n_sigma_params
            sigma_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
            sigma_sparse = {
                quad: value
                for quad, value in zip(self.sigma_indices, sigma_sparse_values)
            }
            sigma_values = np.asarray(
                [
                    sigma_sparse.get(quad, 0.0)
                    for quad in _default_sigma_indices(self.norb)
                ],
                dtype=np.float64,
            )
            idx += n

        if idx != self.n_params:
            raise ValueError("diagonal parameter block has inconsistent length")
        return RestrictedQuarticCoefficients(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=pair_values,
            tau=tau,
            omega_values=omega_values,
            eta_values=eta_values,
            rho_values=rho_values,
            sigma_values=sigma_values,
        )

    def parameters_from_coefficients(
        self,
        coefficients: RestrictedQuarticCoefficients,
    ) -> tuple[np.ndarray, np.ndarray]:
        double = np.asarray(coefficients.double_params, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double_params has inconsistent shape")
        pair_values = np.asarray(coefficients.pair_values, dtype=np.float64)
        if pair_values.shape != (len(_default_pair_indices(self.norb)),):
            raise ValueError("pair_values has inconsistent shape")
        tau = np.asarray(coefficients.tau, dtype=np.float64)
        if tau.shape != (self.norb, self.norb):
            raise ValueError("tau must have shape (norb, norb)")
        tau = np.array(tau, copy=True, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)
        omega = np.asarray(coefficients.omega_values, dtype=np.float64)
        if omega.shape != (len(_default_triple_indices(self.norb)),):
            raise ValueError("omega_values has inconsistent shape")
        eta = np.asarray(coefficients.eta_values, dtype=np.float64)
        if eta.shape != (len(_default_eta_indices(self.norb)),):
            raise ValueError("eta_values has inconsistent shape")
        rho = np.asarray(coefficients.rho_values, dtype=np.float64)
        if rho.shape != (len(_default_rho_indices(self.norb)),):
            raise ValueError("rho_values has inconsistent shape")
        sigma = np.asarray(coefficients.sigma_values, dtype=np.float64)
        if sigma.shape != (len(_default_sigma_indices(self.norb)),):
            raise ValueError("sigma_values has inconsistent shape")

        pair_matrix = _symmetric_matrix_from_values(
            pair_values,
            self.norb,
            _default_pair_indices(self.norb),
        )
        pair_eff = _restricted_irreducible_pair_matrix(double, pair_matrix)
        full_pair_values = np.asarray(
            [pair_eff[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        full_cubic = np.concatenate(
            [
                _values_from_ordered_matrix(tau, _default_tau_indices(self.norb)),
                omega,
            ]
        )
        full_quartic = np.concatenate([eta, rho, sigma])

        if self.uses_reduced_quartic_chart:
            full_cubic, reduced_quartic_values = self.quartic_reduction.reduce_full(
                full_cubic,
                full_quartic,
            )
        else:
            reduced_quartic_values = None

        reduced_pair_values, reduced_cubic_values, cubic_onebody_phase = (
            self.cubic_reduction.reduce_full(full_pair_values, full_cubic)
        )

        phase_vec = (
            _restricted_left_phase_vector(double, self.nocc) + cubic_onebody_phase
        )

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_pair_params
        pair_reduced_matrix = _symmetric_matrix_from_values(
            reduced_pair_values, self.norb, _default_pair_indices(self.norb)
        )
        out[idx : idx + n] = np.asarray(
            [pair_reduced_matrix[p, q] for p, q in self.pair_indices], dtype=np.float64
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params
            out[idx : idx + n] = reduced_cubic_values
            idx += n
        else:
            full_cubic_adjusted = self.cubic_reduction.full_from_reduced(
                reduced_cubic_values
            )
            n_tau_full = len(_default_tau_indices(self.norb))
            tau_adjusted = _ordered_matrix_from_values(
                full_cubic_adjusted[:n_tau_full],
                self.norb,
                _default_tau_indices(self.norb),
            )
            omega_adjusted = {
                triple: val
                for triple, val in zip(
                    _default_triple_indices(self.norb),
                    full_cubic_adjusted[n_tau_full:],
                )
            }
            n = self.n_tau_params
            out[idx : idx + n] = _values_from_ordered_matrix(
                tau_adjusted, self.tau_indices
            )
            idx += n

            n = self.n_omega_params
            out[idx : idx + n] = np.asarray(
                [omega_adjusted[t] for t in self.omega_indices], dtype=np.float64
            )
            idx += n

        if self.uses_reduced_quartic_chart:
            n = self.n_rho_params
            if reduced_quartic_values is None:
                raise ValueError("reduced quartic values were not computed")
            out[idx : idx + n] = reduced_quartic_values
            idx += n
        else:
            n = self.n_eta_params
            full_eta = {pair: value for value, pair in zip(eta, _default_eta_indices(self.norb))}
            out[idx : idx + n] = np.asarray(
                [full_eta[t] for t in self.eta_indices], dtype=np.float64
            )
            idx += n

            n = self.n_rho_params
            full_rho = {
                triple: value
                for value, triple in zip(rho, _default_rho_indices(self.norb))
            }
            out[idx : idx + n] = np.asarray(
                [full_rho[t] for t in self.rho_indices], dtype=np.float64
            )
            idx += n

            n = self.n_sigma_params
            full_sigma = {
                quad: value
                for value, quad in zip(sigma, _default_sigma_indices(self.norb))
            }
            out[idx : idx + n] = np.asarray(
                [full_sigma[t] for t in self.sigma_indices], dtype=np.float64
            )
            idx += n

        if idx != self.n_params:
            raise ValueError("diagonal parameter block has inconsistent length")
        return out, phase_vec


__all__ = [
    "RestrictedPairChart",
    "RestrictedPairCoefficients",
    "RestrictedCubicChart",
    "RestrictedCubicCoefficients",
    "RestrictedQuarticChart",
    "RestrictedQuarticCoefficients",
]
