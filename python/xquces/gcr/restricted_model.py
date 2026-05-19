from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xquces._lib import (
    apply_igcr3_spin_restricted_in_place_num_rep,
    apply_igcr4_spin_restricted_in_place_num_rep,
)
from xquces.basis import flatten_state, occ_indicator_rows, reshape_state
from xquces.gates import apply_gcr_spin_balanced, apply_igcr2_spin_restricted
from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.gcr.utils import (
    _assert_square_matrix,
    _default_eta_indices,
    _default_pair_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_tau_indices,
    _default_triple_indices,
    _diag_unitary,
    _restricted_left_phase_vector,
    _symmetric_matrix_from_values,
    exact_reference_ov_unitary,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.init import UCJBalancedDFSeed, UCJRestrictedProjectedDFSeed
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz


@dataclass(frozen=True)
class IGCR2SpinRestrictedSpec:
    pair: np.ndarray

    @property
    def norb(self):
        return self.pair.shape[0]

    def full_double(self):
        return np.zeros(self.norb, dtype=np.float64)

    def to_standard(self):
        pair = np.array(self.pair, copy=True, dtype=np.float64)
        np.fill_diagonal(pair, 0.0)
        return SpinRestrictedSpec(double_params=self.full_double(), pair_params=pair)


def reduce_spin_restricted(diag: SpinRestrictedSpec):
    pair = np.asarray(diag.pair_params, dtype=np.float64).copy()
    b = np.asarray(diag.double_params, dtype=np.float64)
    shift = 0.5 * (b[:, None] + b[None, :])
    mask = ~np.eye(pair.shape[0], dtype=bool)
    pair[mask] -= shift[mask]
    np.fill_diagonal(pair, 0.0)
    return IGCR2SpinRestrictedSpec(pair=pair)


@dataclass(frozen=True)
class IGCR2Ansatz:
    diagonal: object
    left: np.ndarray
    right: np.ndarray
    nocc: int

    @property
    def norb(self):
        return self.diagonal.norb

    @property
    def is_spin_restricted(self):
        return isinstance(self.diagonal, IGCR2SpinRestrictedSpec)

    @property
    def is_spin_balanced(self):
        return not self.is_spin_restricted

    def apply(self, vec, nelec, copy=True):
        if self.is_spin_restricted:
            return apply_igcr2_spin_restricted(
                vec,
                self.diagonal.pair,
                self.norb,
                nelec,
                left_orbital_rotation=self.left,
                right_orbital_rotation=self.right,
                copy=copy,
            )
        d = self.diagonal.to_standard()
        return apply_gcr_spin_balanced(
            vec,
            d.same_spin_params,
            d.mixed_spin_params,
            self.norb,
            nelec,
            left_orbital_rotation=self.left,
            right_orbital_rotation=self.right,
            copy=copy,
        )

    @classmethod
    def from_gcr_ansatz(cls, ansatz: GCRAnsatz, nocc: int):
        right_ov = exact_reference_ov_unitary(ansatz.right_orbital_rotation, nocc)
        if ansatz.is_spin_restricted:
            diag = reduce_spin_restricted(ansatz.diagonal)
            b = np.asarray(ansatz.diagonal.double_params, dtype=np.float64)
            phase_vec = _restricted_left_phase_vector(b, nocc)
            left = np.asarray(
                ansatz.left_orbital_rotation, dtype=np.complex128
            ) @ _diag_unitary(phase_vec)
        else:
            from xquces.gcr.igcr import reduce_spin_balanced

            diag = reduce_spin_balanced(ansatz.diagonal)
            left = np.asarray(ansatz.left_orbital_rotation, dtype=np.complex128)
        return cls(
            diagonal=diag,
            left=left,
            right=np.asarray(right_ov, dtype=np.complex128),
            nocc=nocc,
        )

    @classmethod
    def from_t_amplitudes(cls, t2, t1=None, **seed_options) -> "IGCR2Ansatz":
        """Build a one-layer iGCR-2 ansatz from CCSD amplitudes natively."""
        from xquces.gcr.igcr import (
            IGCR2SpinRestrictedParameterization,
            _native_igcr2_seed_from_ccsd_t_amplitudes,
        )

        nocc = np.asarray(t2).shape[0]
        nvirt = np.asarray(t2).shape[2]
        parameterization = IGCR2SpinRestrictedParameterization(
            norb=nocc + nvirt, nocc=nocc, layers=1
        )
        result = _native_igcr2_seed_from_ccsd_t_amplitudes(
            parameterization, t2, t1=t1, **seed_options
        )
        assert isinstance(result, cls)
        return result

    @classmethod
    def from_ucj_t_amplitudes(cls, t2, t1=None, **df_options) -> "IGCR2Ansatz":
        """Build a one-layer iGCR-2 ansatz by lifting ffsim's UCJ seed."""
        from xquces.seeds.ucj import layered_igcr2_from_ucj_t_amplitudes

        nocc = np.asarray(t2).shape[0]
        result = layered_igcr2_from_ucj_t_amplitudes(
            t2, t1=t1, layers=1, nocc=nocc, **df_options
        )
        assert isinstance(result, cls)
        return result

    @classmethod
    def from_ucj(cls, ucj: UCJAnsatz, nocc: int):
        raise NotImplementedError(
            "IGCR2Ansatz.from_ucj was removed. "
            "Use IGCR2Ansatz.from_t_amplitudes(t2, t1=t1) instead."
        )

    @classmethod
    def from_ucj_ansatz(cls, ansatz: UCJAnsatz, nocc: int):
        raise NotImplementedError(
            "IGCR2Ansatz.from_ucj_ansatz was removed. "
            "Use IGCR2Ansatz.from_t_amplitudes(t2, t1=t1) instead."
        )

    @classmethod
    def from_t_balanced(cls, t2, **kwargs):
        ucj = UCJBalancedDFSeed(t2=t2, **kwargs).build_ansatz()
        gcr = gcr_from_ucj_ansatz(ucj)
        return cls.from_gcr_ansatz(gcr, nocc=t2.shape[0])

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        t1 = kwargs.pop("t1", None)
        return cls.from_t_amplitudes(t2, t1=t1, **kwargs)


@dataclass(frozen=True)
class IGCR2LayeredAnsatz:
    diagonals: tuple[object, ...]
    rotations: tuple[np.ndarray, ...]
    nocc: int

    def __post_init__(self):
        if len(self.diagonals) == 0:
            raise ValueError("at least one diagonal layer is required")
        if len(self.rotations) != len(self.diagonals) + 1:
            raise ValueError("rotations must contain one more entry than diagonals")
        norb = self.diagonals[0].norb
        diag_type = type(self.diagonals[0])
        fixed_diagonals = []
        for diagonal in self.diagonals:
            if diagonal.norb != norb:
                raise ValueError("all diagonal layers must have the same norb")
            if type(diagonal) is not diag_type:
                raise ValueError("all diagonal layers must have the same spin type")
            fixed_diagonals.append(diagonal)
        fixed_rotations = []
        for rotation in self.rotations:
            u = np.asarray(rotation, dtype=np.complex128)
            if u.shape != (norb, norb):
                raise ValueError("rotation has wrong shape")
            if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
                raise ValueError("rotation must be unitary")
            fixed_rotations.append(u)
        object.__setattr__(self, "diagonals", tuple(fixed_diagonals))
        object.__setattr__(self, "rotations", tuple(fixed_rotations))

    @property
    def norb(self):
        return self.diagonals[0].norb

    @property
    def layers(self):
        return len(self.diagonals)

    @property
    def is_spin_restricted(self):
        return isinstance(self.diagonals[0], IGCR2SpinRestrictedSpec)

    @property
    def is_spin_balanced(self):
        return not self.is_spin_restricted

    def apply(self, vec, nelec, copy=True):
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.rotations[-1],
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        for idx in range(self.layers - 1, -1, -1):
            diagonal = self.diagonals[idx]
            if isinstance(diagonal, IGCR2SpinRestrictedSpec):
                arr = apply_igcr2_spin_restricted(
                    arr,
                    diagonal.pair,
                    self.norb,
                    nelec,
                    copy=False,
                )
            else:
                d = diagonal.to_standard()
                arr = apply_gcr_spin_balanced(
                    arr,
                    d.same_spin_params,
                    d.mixed_spin_params,
                    self.norb,
                    nelec,
                    copy=False,
                )
            arr = apply_orbital_rotation(
                arr,
                self.rotations[idx],
                norb=self.norb,
                nelec=nelec,
                copy=False,
            )
        return arr


def spin_restricted_triples_seed_from_pair_params(
    pair_params: np.ndarray,
    nocc: int,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    _assert_square_matrix(pair, "pair_params")
    norb = pair.shape[0]
    nelec_total = 2 * int(nocc)
    denom = max(nelec_total - 2, 1)

    tau = np.zeros((norb, norb), dtype=np.float64)
    if tau_scale != 0.0:
        for p in range(norb):
            for q in range(norb):
                if p != q:
                    tau[p, q] = float(tau_scale) * pair[p, q] / denom

    omega = np.zeros(len(_default_triple_indices(norb)), dtype=np.float64)
    if omega_scale != 0.0:
        for k, (p, q, r) in enumerate(_default_triple_indices(norb)):
            avg_pair = (pair[p, q] + pair[p, r] + pair[q, r]) / 3.0
            omega[k] = float(omega_scale) * avg_pair / denom
    return tau, omega


@dataclass(frozen=True)
class IGCR3SpinRestrictedSpec:
    double_params: np.ndarray
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray

    @property
    def norb(self) -> int:
        return int(np.asarray(self.double_params, dtype=np.float64).shape[0])

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _default_pair_indices(self.norb)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _default_tau_indices(self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _default_triple_indices(self.norb)

    def full_double(self) -> np.ndarray:
        double = np.asarray(self.double_params, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double_params has inconsistent shape")
        return double

    def pair_matrix(self) -> np.ndarray:
        return _symmetric_matrix_from_values(
            np.asarray(self.pair_values, dtype=np.float64),
            self.norb,
            self.pair_indices,
        )

    def tau_matrix(self) -> np.ndarray:
        tau = np.asarray(self.tau, dtype=np.float64)
        if tau.shape != (self.norb, self.norb):
            raise ValueError("tau must have shape (norb, norb)")
        tau = np.array(tau, copy=True, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)
        return tau

    def omega_vector(self) -> np.ndarray:
        omega = np.asarray(self.omega_values, dtype=np.float64)
        if omega.shape != (len(self.omega_indices),):
            raise ValueError("omega_values has inconsistent shape")
        return omega

    def phase_from_occupations(
        self,
        occ_alpha: np.ndarray,
        occ_beta: np.ndarray,
    ) -> float:
        n = np.zeros(self.norb, dtype=np.float64)
        n[np.asarray(occ_alpha, dtype=np.int64)] += 1.0
        n[np.asarray(occ_beta, dtype=np.int64)] += 1.0
        d = np.zeros(self.norb, dtype=np.float64)
        d[np.intersect1d(occ_alpha, occ_beta, assume_unique=True)] = 1.0
        return self.phase_from_number_arrays(n, d)

    def phase_from_number_arrays(self, n: np.ndarray, d: np.ndarray) -> float:
        n = np.asarray(n, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        if n.shape != (self.norb,) or d.shape != (self.norb,):
            raise ValueError("n and d must have shape (norb,)")

        phase = float(np.dot(self.full_double(), d))
        pair = self.pair_matrix()
        for p, q in self.pair_indices:
            phase += pair[p, q] * n[p] * n[q]
        tau = self.tau_matrix()
        for p, q in self.tau_indices:
            phase += tau[p, q] * d[p] * n[q]
        omega = self.omega_vector()
        for value, (p, q, r) in zip(omega, self.omega_indices):
            phase += value * n[p] * n[q] * n[r]
        return float(phase)

    def to_igcr2_diagonal(self) -> IGCR2SpinRestrictedSpec:
        return reduce_spin_restricted(
            SpinRestrictedSpec(
                double_params=self.full_double(),
                pair_params=self.pair_matrix(),
            )
        )

    @classmethod
    def from_igcr2_diagonal(
        cls,
        diagonal: IGCR2SpinRestrictedSpec,
        *,
        tau: np.ndarray | None = None,
        omega_values: np.ndarray | None = None,
    ) -> "IGCR3SpinRestrictedSpec":
        double = diagonal.full_double()
        norb = double.shape[0]
        pair = diagonal.to_standard().pair_params
        pair_values = np.asarray(
            [pair[p, q] for p, q in _default_pair_indices(norb)],
            dtype=np.float64,
        )
        if tau is None:
            tau = np.zeros((norb, norb), dtype=np.float64)
        if omega_values is None:
            omega_values = np.zeros(
                len(_default_triple_indices(norb)), dtype=np.float64
            )
        return cls(
            double_params=double,
            pair_values=pair_values,
            tau=np.asarray(tau, dtype=np.float64),
            omega_values=np.asarray(omega_values, dtype=np.float64),
        )


def apply_igcr3_spin_restricted_diagonal(
    vec: np.ndarray,
    diagonal: IGCR3SpinRestrictedSpec,
    norb: int,
    nelec: tuple[int, int],
    *,
    time: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    state2 = reshape_state(arr, norb, nelec)
    occ_alpha = occ_indicator_rows(norb, nelec[0])
    occ_beta = occ_indicator_rows(norb, nelec[1])
    double = np.asarray(diagonal.full_double(), dtype=np.float64) * time
    pair = np.asarray(diagonal.pair_matrix(), dtype=np.float64) * time
    tau = np.asarray(diagonal.tau_matrix(), dtype=np.float64) * time
    omega = np.asarray(diagonal.omega_vector(), dtype=np.float64) * time
    apply_igcr3_spin_restricted_in_place_num_rep(
        state2,
        double,
        pair,
        tau,
        omega,
        norb,
        occ_alpha,
        occ_beta,
    )
    return flatten_state(state2)


@dataclass(frozen=True)
class IGCR3Ansatz:
    diagonal: IGCR3SpinRestrictedSpec
    left: np.ndarray
    right: np.ndarray
    nocc: int

    @property
    def norb(self) -> int:
        return self.diagonal.norb

    def apply(self, vec, nelec, copy=True):
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.right,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        arr = apply_igcr3_spin_restricted_diagonal(
            arr,
            self.diagonal,
            self.norb,
            nelec,
            copy=False,
        )
        arr = apply_orbital_rotation(
            arr,
            self.left,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        return arr

    def to_igcr2_ansatz(self) -> IGCR2Ansatz:
        if np.linalg.norm(self.diagonal.tau_matrix()) > 1e-14:
            raise ValueError("cannot convert nonzero tau sector to iGCR-2")
        if np.linalg.norm(self.diagonal.omega_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero omega sector to iGCR-2")
        return IGCR2Ansatz(
            diagonal=self.diagonal.to_igcr2_diagonal(),
            left=np.asarray(self.left, dtype=np.complex128),
            right=np.asarray(self.right, dtype=np.complex128),
            nocc=self.nocc,
        )

    @classmethod
    def from_igcr2_ansatz(
        cls,
        ansatz: IGCR2Ansatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        if not ansatz.is_spin_restricted:
            raise TypeError(
                "iGCR-3 is currently implemented only for spin-restricted seeds"
            )
        d = ansatz.diagonal.to_standard()
        tau, omega = spin_restricted_triples_seed_from_pair_params(
            d.pair_params,
            ansatz.nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )
        diagonal = IGCR3SpinRestrictedSpec.from_igcr2_diagonal(
            ansatz.diagonal,
            tau=tau,
            omega_values=omega,
        )
        return cls(
            diagonal=diagonal,
            left=np.asarray(ansatz.left, dtype=np.complex128),
            right=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

    @classmethod
    def from_ucj_ansatz(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        igcr2 = IGCR2Ansatz.from_gcr_ansatz(gcr_from_ucj_ansatz(ansatz), nocc=nocc)
        return cls.from_igcr2_ansatz(igcr2, tau_scale=tau_scale, omega_scale=omega_scale)

    @classmethod
    def from_ucj(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        return cls.from_ucj_ansatz(ansatz, nocc, tau_scale=tau_scale, omega_scale=omega_scale)

    @classmethod
    def from_gcr_ansatz(
        cls,
        ansatz: GCRAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> "IGCR3Ansatz":
        return cls.from_igcr2_ansatz(
            IGCR2Ansatz.from_gcr_ansatz(ansatz, nocc=nocc),
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        tau_scale = kwargs.pop("tau_scale", 0.0)
        omega_scale = kwargs.pop("omega_scale", 0.0)
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj_ansatz(
            ucj,
            nocc=t2.shape[0],
            tau_scale=tau_scale,
            omega_scale=omega_scale,
        )


@dataclass(frozen=True)
class IGCR3LayeredAnsatz:
    diagonals: tuple[IGCR3SpinRestrictedSpec, ...]
    rotations: tuple[np.ndarray, ...]
    nocc: int

    def __post_init__(self):
        if len(self.diagonals) == 0:
            raise ValueError("at least one diagonal layer is required")
        if len(self.rotations) != len(self.diagonals) + 1:
            raise ValueError("rotations must contain one more entry than diagonals")
        norb = self.diagonals[0].norb
        fixed_diagonals = []
        for diagonal in self.diagonals:
            if diagonal.norb != norb:
                raise ValueError("all diagonal layers must have the same norb")
            fixed_diagonals.append(diagonal)
        fixed_rotations = []
        for rotation in self.rotations:
            u = np.asarray(rotation, dtype=np.complex128)
            if u.shape != (norb, norb):
                raise ValueError("rotation has wrong shape")
            if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
                raise ValueError("rotation must be unitary")
            fixed_rotations.append(u)
        object.__setattr__(self, "diagonals", tuple(fixed_diagonals))
        object.__setattr__(self, "rotations", tuple(fixed_rotations))

    @property
    def norb(self) -> int:
        return self.diagonals[0].norb

    @property
    def layers(self) -> int:
        return len(self.diagonals)

    def apply(self, vec, nelec, copy=True):
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.rotations[-1],
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        for idx in range(self.layers - 1, -1, -1):
            arr = apply_igcr3_spin_restricted_diagonal(
                arr,
                self.diagonals[idx],
                self.norb,
                nelec,
                copy=False,
            )
            arr = apply_orbital_rotation(
                arr,
                self.rotations[idx],
                norb=self.norb,
                nelec=nelec,
                copy=False,
            )
        return arr


def spin_restricted_quartic_seed_from_pair_params(
    pair_params: np.ndarray,
    nocc: int,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = np.asarray(pair_params, dtype=np.float64)
    _assert_square_matrix(pair, "pair_params")
    norb = pair.shape[0]
    nelec_total = 2 * int(nocc)
    denom = max(nelec_total - 3, 1)

    eta = np.zeros(len(_default_eta_indices(norb)), dtype=np.float64)
    if eta_scale != 0.0:
        for k, (p, q) in enumerate(_default_eta_indices(norb)):
            eta[k] = float(eta_scale) * 0.5 * (pair[p, q]) / denom

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


@dataclass(frozen=True)
class IGCR4SpinRestrictedSpec:
    double_params: np.ndarray
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray
    eta_values: np.ndarray
    rho_values: np.ndarray
    sigma_values: np.ndarray

    @property
    def norb(self) -> int:
        return int(np.asarray(self.double_params, dtype=np.float64).shape[0])

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
    def eta_indices(self):
        return _default_eta_indices(self.norb)

    @property
    def rho_indices(self):
        return _default_rho_indices(self.norb)

    @property
    def sigma_indices(self):
        return _default_sigma_indices(self.norb)

    def full_double(self) -> np.ndarray:
        double = np.asarray(self.double_params, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double_params has inconsistent shape")
        return double

    def pair_matrix(self) -> np.ndarray:
        return _symmetric_matrix_from_values(
            np.asarray(self.pair_values, dtype=np.float64),
            self.norb,
            self.pair_indices,
        )

    def tau_matrix(self) -> np.ndarray:
        tau = np.asarray(self.tau, dtype=np.float64)
        if tau.shape != (self.norb, self.norb):
            raise ValueError("tau must have shape (norb, norb)")
        tau = np.array(tau, copy=True, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)
        return tau

    def omega_vector(self) -> np.ndarray:
        omega = np.asarray(self.omega_values, dtype=np.float64)
        if omega.shape != (len(self.omega_indices),):
            raise ValueError("omega_values has inconsistent shape")
        return omega

    def eta_vector(self) -> np.ndarray:
        eta = np.asarray(self.eta_values, dtype=np.float64)
        if eta.shape != (len(self.eta_indices),):
            raise ValueError("eta_values has inconsistent shape")
        return eta

    def rho_vector(self) -> np.ndarray:
        rho = np.asarray(self.rho_values, dtype=np.float64)
        if rho.shape != (len(self.rho_indices),):
            raise ValueError("rho_values has inconsistent shape")
        return rho

    def sigma_vector(self) -> np.ndarray:
        sigma = np.asarray(self.sigma_values, dtype=np.float64)
        if sigma.shape != (len(self.sigma_indices),):
            raise ValueError("sigma_values has inconsistent shape")
        return sigma

    def phase_from_occupations(
        self,
        occ_alpha: np.ndarray,
        occ_beta: np.ndarray,
    ) -> float:
        n = np.zeros(self.norb, dtype=np.float64)
        n[np.asarray(occ_alpha, dtype=np.int64)] += 1.0
        n[np.asarray(occ_beta, dtype=np.int64)] += 1.0
        d = np.zeros(self.norb, dtype=np.float64)
        d[np.intersect1d(occ_alpha, occ_beta, assume_unique=True)] = 1.0
        return self.phase_from_number_arrays(n, d)

    def phase_from_number_arrays(self, n: np.ndarray, d: np.ndarray) -> float:
        n = np.asarray(n, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        if n.shape != (self.norb,) or d.shape != (self.norb,):
            raise ValueError("n and d must have shape (norb,)")

        phase = float(np.dot(self.full_double(), d))
        pair = self.pair_matrix()
        for p, q in self.pair_indices:
            phase += pair[p, q] * n[p] * n[q]
        tau = self.tau_matrix()
        for p, q in self.tau_indices:
            phase += tau[p, q] * d[p] * n[q]
        omega = self.omega_vector()
        for value, (p, q, r) in zip(omega, self.omega_indices):
            phase += value * n[p] * n[q] * n[r]
        eta = self.eta_vector()
        for value, (p, q) in zip(eta, self.eta_indices):
            phase += value * d[p] * d[q]
        rho = self.rho_vector()
        for value, (p, q, r) in zip(rho, self.rho_indices):
            phase += value * d[p] * n[q] * n[r]
        sigma = self.sigma_vector()
        for value, (p, q, r, s) in zip(sigma, self.sigma_indices):
            phase += value * n[p] * n[q] * n[r] * n[s]
        return float(phase)

    def to_igcr3_diagonal(self) -> IGCR3SpinRestrictedSpec:
        return IGCR3SpinRestrictedSpec(
            double_params=self.full_double(),
            pair_values=self.pair_values,
            tau=self.tau,
            omega_values=self.omega_values,
        )

    def to_igcr2_diagonal(self) -> IGCR2SpinRestrictedSpec:
        return reduce_spin_restricted(
            SpinRestrictedSpec(
                double_params=self.full_double(),
                pair_params=self.pair_matrix(),
            )
        )

    @classmethod
    def from_igcr3_diagonal(
        cls,
        diagonal: IGCR3SpinRestrictedSpec,
        *,
        eta_values: np.ndarray | None = None,
        rho_values: np.ndarray | None = None,
        sigma_values: np.ndarray | None = None,
    ) -> "IGCR4SpinRestrictedSpec":
        norb = diagonal.norb
        if eta_values is None:
            eta_values = np.zeros(len(_default_eta_indices(norb)), dtype=np.float64)
        if rho_values is None:
            rho_values = np.zeros(len(_default_rho_indices(norb)), dtype=np.float64)
        if sigma_values is None:
            sigma_values = np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64)
        return cls(
            double_params=np.asarray(diagonal.full_double(), dtype=np.float64),
            pair_values=np.asarray(diagonal.pair_values, dtype=np.float64),
            tau=np.asarray(diagonal.tau, dtype=np.float64),
            omega_values=np.asarray(diagonal.omega_values, dtype=np.float64),
            eta_values=np.asarray(eta_values, dtype=np.float64),
            rho_values=np.asarray(rho_values, dtype=np.float64),
            sigma_values=np.asarray(sigma_values, dtype=np.float64),
        )


def apply_igcr4_spin_restricted_diagonal(
    vec: np.ndarray,
    diagonal: IGCR4SpinRestrictedSpec,
    norb: int,
    nelec: tuple[int, int],
    *,
    time: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    state2 = reshape_state(arr, norb, nelec)
    occ_alpha = occ_indicator_rows(norb, nelec[0])
    occ_beta = occ_indicator_rows(norb, nelec[1])
    apply_igcr4_spin_restricted_in_place_num_rep(
        state2,
        np.asarray(diagonal.full_double(), dtype=np.float64) * time,
        np.asarray(diagonal.pair_matrix(), dtype=np.float64) * time,
        np.asarray(diagonal.tau_matrix(), dtype=np.float64) * time,
        np.asarray(diagonal.omega_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.eta_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.rho_vector(), dtype=np.float64) * time,
        np.asarray(diagonal.sigma_vector(), dtype=np.float64) * time,
        norb,
        occ_alpha,
        occ_beta,
    )
    return flatten_state(state2)


@dataclass(frozen=True)
class IGCR4Ansatz:
    diagonal: IGCR4SpinRestrictedSpec
    left: np.ndarray
    right: np.ndarray
    nocc: int

    @property
    def norb(self) -> int:
        return self.diagonal.norb

    def apply(self, vec, nelec, copy=True):
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.right,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        arr = apply_igcr4_spin_restricted_diagonal(
            arr,
            self.diagonal,
            self.norb,
            nelec,
            copy=False,
        )
        arr = apply_orbital_rotation(
            arr,
            self.left,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        return arr

    def to_igcr3_ansatz(self) -> IGCR3Ansatz:
        if np.linalg.norm(self.diagonal.eta_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero eta sector to iGCR-3")
        if np.linalg.norm(self.diagonal.rho_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero rho sector to iGCR-3")
        if np.linalg.norm(self.diagonal.sigma_vector()) > 1e-14:
            raise ValueError("cannot convert nonzero sigma sector to iGCR-3")
        return IGCR3Ansatz(
            diagonal=self.diagonal.to_igcr3_diagonal(),
            left=np.asarray(self.left, dtype=np.complex128),
            right=np.asarray(self.right, dtype=np.complex128),
            nocc=self.nocc,
        )

    @classmethod
    def from_igcr3_ansatz(
        cls,
        ansatz: IGCR3Ansatz,
        *,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        d3 = ansatz.diagonal
        eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(
            d3.pair_matrix(),
            ansatz.nocc,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )
        diagonal = IGCR4SpinRestrictedSpec.from_igcr3_diagonal(
            d3,
            eta_values=eta,
            rho_values=rho,
            sigma_values=sigma,
        )
        return cls(
            diagonal=diagonal,
            left=np.asarray(ansatz.left, dtype=np.complex128),
            right=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=ansatz.nocc,
        )

    @classmethod
    def from_igcr2_ansatz(
        cls,
        ansatz: IGCR2Ansatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        return cls.from_igcr3_ansatz(
            IGCR3Ansatz.from_igcr2_ansatz(
                ansatz,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            ),
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_ucj_ansatz(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        igcr2 = IGCR2Ansatz.from_gcr_ansatz(gcr_from_ucj_ansatz(ansatz), nocc=nocc)
        return cls.from_igcr2_ansatz(
            igcr2,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_ucj(
        cls,
        ansatz: UCJAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        return cls.from_ucj_ansatz(
            ansatz,
            nocc,
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_gcr_ansatz(
        cls,
        ansatz: GCRAnsatz,
        nocc: int,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> "IGCR4Ansatz":
        return cls.from_igcr2_ansatz(
            IGCR2Ansatz.from_gcr_ansatz(ansatz, nocc=nocc),
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )

    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        tau_scale = kwargs.pop("tau_scale", 0.0)
        omega_scale = kwargs.pop("omega_scale", 0.0)
        eta_scale = kwargs.pop("eta_scale", 0.0)
        rho_scale = kwargs.pop("rho_scale", 0.0)
        sigma_scale = kwargs.pop("sigma_scale", 0.0)
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj_ansatz(
            ucj,
            nocc=t2.shape[0],
            tau_scale=tau_scale,
            omega_scale=omega_scale,
            eta_scale=eta_scale,
            rho_scale=rho_scale,
            sigma_scale=sigma_scale,
        )


@dataclass(frozen=True)
class IGCR4LayeredAnsatz:
    diagonals: tuple[IGCR4SpinRestrictedSpec, ...]
    rotations: tuple[np.ndarray, ...]
    nocc: int

    def __post_init__(self):
        if len(self.diagonals) == 0:
            raise ValueError("at least one diagonal layer is required")
        if len(self.rotations) != len(self.diagonals) + 1:
            raise ValueError("rotations must contain one more entry than diagonals")
        norb = self.diagonals[0].norb
        fixed_diagonals = []
        for diagonal in self.diagonals:
            if diagonal.norb != norb:
                raise ValueError("all diagonal layers must have the same norb")
            fixed_diagonals.append(diagonal)
        fixed_rotations = []
        for rotation in self.rotations:
            u = np.asarray(rotation, dtype=np.complex128)
            if u.shape != (norb, norb):
                raise ValueError("rotation has wrong shape")
            if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
                raise ValueError("rotation must be unitary")
            fixed_rotations.append(u)
        object.__setattr__(self, "diagonals", tuple(fixed_diagonals))
        object.__setattr__(self, "rotations", tuple(fixed_rotations))

    @property
    def norb(self) -> int:
        return self.diagonals[0].norb

    @property
    def layers(self) -> int:
        return len(self.diagonals)

    def apply(self, vec, nelec, copy=True):
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.rotations[-1],
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        for idx in range(self.layers - 1, -1, -1):
            arr = apply_igcr4_spin_restricted_diagonal(
                arr,
                self.diagonals[idx],
                self.norb,
                nelec,
                copy=False,
            )
            arr = apply_orbital_rotation(
                arr,
                self.rotations[idx],
                norb=self.norb,
                nelec=nelec,
                copy=False,
            )
        return arr


__all__ = [
    "IGCR2Ansatz",
    "IGCR2LayeredAnsatz",
    "IGCR2SpinRestrictedSpec",
    "IGCR3Ansatz",
    "IGCR3LayeredAnsatz",
    "IGCR3SpinRestrictedSpec",
    "IGCR4Ansatz",
    "IGCR4LayeredAnsatz",
    "IGCR4SpinRestrictedSpec",
    "apply_igcr3_spin_restricted_diagonal",
    "apply_igcr4_spin_restricted_diagonal",
    "reduce_spin_restricted",
    "spin_restricted_quartic_seed_from_pair_params",
    "spin_restricted_triples_seed_from_pair_params",
]
