from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import ffsim
import numpy as np
import scipy.linalg

from xquces._lib import (
    apply_igcr3_spin_restricted_in_place_num_rep,
    apply_igcr4_spin_restricted_in_place_num_rep,
)
from xquces.basis import flatten_state, occ_indicator_rows, reshape_state
from xquces.gates import (
    apply_gcr_spin_balanced,
    apply_gcr_spin_restricted,
    apply_igcr2_spin_restricted,
)
from xquces.gcr.charts import (
    GCR2FullUnitaryChart,
    GCR2TraceFixedFullUnitaryChart,
    IGCR2BlockDiagLeftUnitaryChart,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
)
from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.gcr.utils import (
    _assert_square_matrix,
    _balanced_irreducible_pair_matrices,
    _balanced_left_phase_vector,
    _default_eta_indices,
    _default_pair_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_tau_indices,
    _default_triple_indices,
    _diag_unitary,
    _final_unitary_from_left_and_right,
    _left_right_ov_adapted_to_native,
    _native_to_left_right_ov_adapted,
    _orbital_relabeling_unitary,
    _ordered_matrix_from_values,
    _parameters_from_zero_diag_antihermitian,
    _restricted_irreducible_pair_matrix,
    _restricted_left_phase_vector,
    _right_unitary_from_left_and_final,
    _symmetric_matrix_from_values,
    _validate_ordered_pairs,
    _validate_pairs,
    _validate_rho_indices,
    _validate_sigma_indices,
    _validate_triples,
    _values_from_ordered_matrix,
    _zero_diag_antihermitian_from_parameters,
    exact_reference_ov_params_from_unitary,
    exact_reference_ov_unitary,
    orbital_relabeling_from_overlap,
    orbital_transport_unitary_from_overlap,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.init import (
    CCSDDoubleFactorization,
    UCJBalancedDFSeed,
    UCJRestrictedProjectedDFSeed,
    factorize_ccsd_t_amplitudes,
)
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz, UCJLayer


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


@dataclass(frozen=True)
class IGCR2SpinBalancedSpec:
    same_diag: np.ndarray
    same: np.ndarray
    mixed: np.ndarray
    double: np.ndarray

    @property
    def norb(self):
        return int(np.asarray(self.double, dtype=np.float64).shape[0])

    def full_double(self):
        double = np.asarray(self.double, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double has inconsistent shape")
        return double

    def to_standard(self):
        same = np.array(self.same, copy=True, dtype=np.float64)
        mixed = np.array(self.mixed, copy=True, dtype=np.float64)
        np.fill_diagonal(same, np.asarray(self.same_diag, dtype=np.float64))
        np.fill_diagonal(mixed, self.full_double())
        return SpinBalancedSpec(same_spin_params=same, mixed_spin_params=mixed)


def reduce_spin_restricted(diag: SpinRestrictedSpec):
    pair = np.asarray(diag.pair_params, dtype=np.float64).copy()
    b = np.asarray(diag.double_params, dtype=np.float64)
    shift = 0.5 * (b[:, None] + b[None, :])
    mask = ~np.eye(pair.shape[0], dtype=bool)
    pair[mask] -= shift[mask]
    np.fill_diagonal(pair, 0.0)
    return IGCR2SpinRestrictedSpec(pair=pair)


def reduce_spin_balanced(diag: SpinBalancedSpec):
    same = np.asarray(diag.same_spin_params, dtype=np.float64).copy()
    mixed = np.asarray(diag.mixed_spin_params, dtype=np.float64).copy()
    same_diag = np.diag(same).copy()
    double = np.diag(mixed).copy()
    np.fill_diagonal(same, 0.0)
    np.fill_diagonal(mixed, 0.0)
    return IGCR2SpinBalancedSpec(
        same_diag=same_diag, same=same, mixed=mixed, double=double
    )


def _left_right_ov_transform_scale_for(right_chart: object, scale: float | None):
    if scale is None:
        return None
    if isinstance(right_chart, IGCR2ReferenceOVUnitaryChart):
        return scale
    return None


@dataclass(frozen=True)
class IGCR2Ansatz:
    diagonal: IGCR2SpinRestrictedSpec | IGCR2SpinBalancedSpec
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
        return isinstance(self.diagonal, IGCR2SpinBalancedSpec)

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
        nocc = np.asarray(t2).shape[0]
        t1 = kwargs.pop("t1", None)
        return cls.from_t_amplitudes(t2, t1=t1, **kwargs)


@dataclass(frozen=True)
class IGCR2LayeredAnsatz:
    diagonals: tuple[IGCR2SpinRestrictedSpec | IGCR2SpinBalancedSpec, ...]
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
        return isinstance(self.diagonals[0], IGCR2SpinBalancedSpec)

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


def relabel_igcr2_ansatz_orbitals(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR2Ansatz | IGCR2LayeredAnsatz:
    if ansatz.norb != len(old_for_new):
        raise ValueError("orbital permutation length must match ansatz.norb")
    relabel = _orbital_relabeling_unitary(old_for_new, phases)
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if isinstance(ansatz, IGCR2LayeredAnsatz):
        diagonals = tuple(
            _relabel_igcr2_diagonal(diagonal, old_for_new)
            for diagonal in ansatz.diagonals
        )
        rotations = tuple(relabel.conj().T @ rot @ relabel for rot in ansatz.rotations)
        return IGCR2LayeredAnsatz(
            diagonals=diagonals,
            rotations=rotations,
            nocc=ansatz.nocc,
        )
    diagonal = _relabel_igcr2_diagonal(ansatz.diagonal, old_for_new)
    return IGCR2Ansatz(
        diagonal=diagonal,
        left=relabel.conj().T @ ansatz.left @ relabel,
        right=relabel.conj().T @ ansatz.right @ relabel,
        nocc=ansatz.nocc,
    )


def _relabel_igcr2_diagonal(
    diagonal: IGCR2SpinRestrictedSpec | IGCR2SpinBalancedSpec,
    old_for_new: np.ndarray,
) -> IGCR2SpinRestrictedSpec | IGCR2SpinBalancedSpec:
    if isinstance(diagonal, IGCR2SpinRestrictedSpec):
        pair = diagonal.pair[np.ix_(old_for_new, old_for_new)]
        return IGCR2SpinRestrictedSpec(pair=pair)
    d = diagonal.to_standard()
    diag = SpinBalancedSpec(
        same_spin_params=d.same_spin_params[np.ix_(old_for_new, old_for_new)],
        mixed_spin_params=d.mixed_spin_params[np.ix_(old_for_new, old_for_new)],
    )
    return reduce_spin_balanced(diag)


def transport_igcr2_ansatz_orbitals(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz, basis_change: np.ndarray
) -> IGCR2Ansatz | IGCR2LayeredAnsatz:
    basis_change = np.asarray(basis_change, dtype=np.complex128)
    if basis_change.shape != (ansatz.norb, ansatz.norb):
        raise ValueError(
            f"basis_change must have shape {(ansatz.norb, ansatz.norb)}, "
            f"got {basis_change.shape}."
        )
    if not np.allclose(
        basis_change.conj().T @ basis_change,
        np.eye(ansatz.norb),
        atol=1e-10,
    ):
        raise ValueError("basis_change must be unitary")
    if isinstance(ansatz, IGCR2LayeredAnsatz):
        rotations = list(ansatz.rotations)
        rotations[0] = basis_change.conj().T @ rotations[0]
        return IGCR2LayeredAnsatz(
            diagonals=ansatz.diagonals,
            rotations=tuple(rotations),
            nocc=ansatz.nocc,
        )
    return IGCR2Ansatz(
        diagonal=ansatz.diagonal,
        left=basis_change.conj().T @ np.asarray(ansatz.left, dtype=np.complex128),
        right=np.asarray(ansatz.right, dtype=np.complex128),
        nocc=ansatz.nocc,
    )


def _zero_igcr2_spin_restricted_spec(norb: int) -> IGCR2SpinRestrictedSpec:
    return IGCR2SpinRestrictedSpec(pair=np.zeros((norb, norb), dtype=np.float64))


def _as_layered_igcr2_spin_restricted_ansatz(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    layers: int,
) -> IGCR2LayeredAnsatz:
    if isinstance(ansatz, IGCR2LayeredAnsatz):
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted ansatz")
        if ansatz.layers == layers:
            return ansatz
        if ansatz.layers > layers:
            raise ValueError(
                "cannot exactly embed an IGCR2 ansatz with more layers than the "
                "target parameterization"
            )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        diagonals = list(ansatz.diagonals)
        rotations = list(ansatz.rotations)
        for _ in range(layers - ansatz.layers):
            diagonals.append(_zero_igcr2_spin_restricted_spec(ansatz.norb))
            rotations.insert(-1, identity)
        return IGCR2LayeredAnsatz(
            diagonals=tuple(diagonals),
            rotations=tuple(rotations),
            nocc=ansatz.nocc,
        )
    if ansatz.norb <= 0:
        raise ValueError("ansatz norb must be positive")
    if not ansatz.is_spin_restricted:
        raise TypeError("expected a spin-restricted ansatz")
    identity = np.eye(ansatz.norb, dtype=np.complex128)
    if layers == 1:
        diagonals = [ansatz.diagonal]
    else:
        pair = np.asarray(ansatz.diagonal.pair, dtype=np.float64) / float(layers)
        diagonals = [
            IGCR2SpinRestrictedSpec(pair=pair.copy()) for _ in range(layers)
        ]
    rotations = [ansatz.left, *[identity for _ in range(layers - 1)], ansatz.right]
    return IGCR2LayeredAnsatz(
        diagonals=tuple(diagonals),
        rotations=tuple(rotations),
        nocc=ansatz.nocc,
    )


def _igcr2_layered_spin_restricted_ansatz_from_ucj(
    ansatz: UCJAnsatz,
    nocc: int,
    layers: int,
) -> IGCR2LayeredAnsatz:
    if not ansatz.is_spin_restricted:
        raise TypeError("expected a spin-restricted UCJ ansatz")
    if ansatz.n_layers > layers:
        raise ValueError(
            "UCJ seed has more layers than the IGCR2 parameterization; "
            "increase layers or use a shallower UCJ seed"
        )
    norb = ansatz.norb
    identity = np.eye(norb, dtype=np.complex128)
    final = (
        identity
        if ansatz.final_orbital_rotation is None
        else np.asarray(ansatz.final_orbital_rotation, dtype=np.complex128)
    )

    diagonals = []
    rotations = []
    active = list(reversed(ansatz.layers))
    previous_base = final
    for layer in active:
        if not isinstance(layer.diagonal, SpinRestrictedSpec):
            raise TypeError("expected a spin-restricted UCJ layer")
        diagonal = reduce_spin_restricted(layer.diagonal)
        phase_vec = _restricted_left_phase_vector(layer.diagonal.double_params, nocc)
        base = np.asarray(layer.orbital_rotation, dtype=np.complex128)
        diagonals.append(diagonal)
        rotations.append(previous_base @ base @ _diag_unitary(phase_vec))
        previous_base = base.conj().T

    rotations.append(exact_reference_ov_unitary(previous_base, nocc))
    while len(diagonals) < layers:
        diagonals.append(_zero_igcr2_spin_restricted_spec(norb))
        rotations.append(identity)

    return IGCR2LayeredAnsatz(
        diagonals=tuple(diagonals),
        rotations=tuple(rotations),
        nocc=nocc,
    )


def layered_igcr2_from_ucj_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    layers: int = 1,
    nocc: int | None = None,
    **df_options,
) -> "IGCR2Ansatz | IGCR2LayeredAnsatz":
    """Build an iGCR-2 ansatz by lifting ffsim's UCJ t-amplitude seed.

    Calls ffsim's spin-restricted UCJ seed with n_reps=layers to obtain L
    double-factorization terms (U_1, J_1), ..., (U_L, J_L) and the final
    orbital rotation U_F from t1.
    Each diagonal J_l is reduced independently (iGCR-2 redundancy removal).
    The resulting layered iGCR-2 seed is ordered so it prepares the same state
    as the corresponding UCJ seed, up to a global phase on the reference.

    Returns IGCR2Ansatz for layers=1, IGCR2LayeredAnsatz for layers>1.
    Extra keyword arguments are forwarded to
    UCJOpSpinRestricted.from_t_amplitudes.
    """
    t2 = np.asarray(t2, dtype=np.float64)
    if nocc is None:
        nocc = t2.shape[0]

    ucj_op = ffsim.variational.UCJOpSpinRestricted.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=layers,
        **df_options,
    )

    ucj_layers = []
    for J_l, U_l in zip(ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations):
        pair_l = np.array(J_l, dtype=np.float64, copy=True)
        np.fill_diagonal(pair_l, 0.0)
        ucj_layers.append(
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=np.diag(J_l).copy(),
                    pair_params=pair_l,
                ),
                orbital_rotation=np.asarray(U_l, dtype=np.complex128),
            )
        )
    ucj = UCJAnsatz(
        tuple(ucj_layers),
        final_orbital_rotation=None
        if ucj_op.final_orbital_rotation is None
        else np.asarray(ucj_op.final_orbital_rotation, dtype=np.complex128),
    )
    seeded = _igcr2_layered_spin_restricted_ansatz_from_ucj(ucj, nocc, layers)

    if layers == 1:
        return IGCR2Ansatz(
            diagonal=seeded.diagonals[0],
            left=seeded.rotations[0],
            right=seeded.rotations[1],
            nocc=nocc,
        )
    return seeded


def layered_igcr2_from_ccsd_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    layers: int = 1,
    nocc: int | None = None,
    **df_options,
) -> "IGCR2Ansatz | IGCR2LayeredAnsatz":
    """Compatibility alias for the UCJ-lift t-amplitude seed.

    New iGCR-2-native code should use
    :meth:`IGCR2SpinRestrictedParameterization.parameters_from_t_amplitudes`
    for the direct one-layer seed, or
    :meth:`IGCR2SpinRestrictedParameterization.parameters_from_ucj_t_amplitudes`
    when the UCJ-lift representative is intended.
    """
    return layered_igcr2_from_ucj_t_amplitudes(
        t2, t1=t1, layers=layers, nocc=nocc, **df_options
    )


def _as_restricted_t1(t1: np.ndarray | None, nocc: int, nvirt: int) -> np.ndarray:
    if t1 is None:
        return np.zeros((nocc, nvirt), dtype=np.float64)
    out = np.asarray(t1, dtype=np.float64)
    if out.shape != (nocc, nvirt):
        raise ValueError(f"Expected t1 shape {(nocc, nvirt)}, got {out.shape}.")
    return out


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


def _solve_real_tikhonov(
    columns: np.ndarray, target: np.ndarray, damping: float
) -> np.ndarray:
    columns = np.asarray(columns, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    A = np.vstack([columns.real, columns.imag])
    b = np.concatenate([target.real, target.imag])
    if damping > 0.0:
        n = columns.shape[1]
        A = np.vstack([A, np.sqrt(float(damping)) * np.eye(n)])
        b = np.concatenate([b, np.zeros(n, dtype=np.float64)])
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
        residual = target - psi
        residual -= psi * np.vdot(psi, residual)

        jacobian = fixed.state_jacobian_from_parameters(x)
        jacobian_block = np.array(jacobian[:, columns], copy=True, dtype=np.complex128)
        jacobian_block -= psi[:, None] * (psi.conj() @ jacobian_block)[None, :]

        delta = _solve_real_tikhonov(jacobian_block, residual, damping)
        raw_delta_norm = float(np.linalg.norm(delta))
        delta_norm = raw_delta_norm
        if delta_norm > max_step_norm:
            delta *= float(max_step_norm) / delta_norm
            delta_norm = float(max_step_norm)

        step = np.zeros_like(x, dtype=np.float64)
        step[columns] = delta
        rank = int(np.linalg.matrix_rank(_real_stacked(jacobian_block)))

        scale = 1.0
        if scale_scan is not None:
            scale = 0.0
            local_best_overlap = _state_overlap(target, psi)
            for candidate in scale_scan:
                candidate = float(candidate)
                candidate_x = x + candidate * step
                candidate_state = fixed.state_from_parameters(candidate_x)
                candidate_overlap = _state_overlap(target, candidate_state)
                if candidate_overlap > local_best_overlap + 1.0e-14:
                    local_best_overlap = candidate_overlap
                    scale = candidate

        x_next = x + scale * step
        psi_next = fixed.state_from_parameters(x_next)
        overlap_next = _state_overlap(_phase_align_target(target_state, psi_next), psi_next)

        raw_delta_norms.append(raw_delta_norm)
        delta_norms.append(float(abs(scale) * delta_norm))
        jacobian_ranks.append(rank)
        scales.append(float(scale))

        if overlap_next > best_overlap + 1.0e-14:
            best_overlap = overlap_next
            best_x = x_next.copy()

        if scale == 0.0:
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
    )
    return info if return_info else info.params


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
    parameterization: "IGCR2SpinRestrictedParameterization",
    params: np.ndarray,
    phi0: np.ndarray,
    nelec: tuple[int, int],
) -> dict[str, float | int]:
    fixed = parameterization.apply(phi0, nelec)
    psi = fixed.state_from_parameters(params)
    jac = fixed.state_jacobian_from_parameters(params)
    jac = jac - psi.reshape((-1, 1)) * (psi.conj() @ jac).reshape((1, -1))
    return _columns_metric_info(jac)


def _native_igcr2_seed_from_ccsd_t_amplitudes(
    parameterization: "IGCR2SpinRestrictedParameterization",
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
) -> IGCR2Ansatz:
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


@dataclass(frozen=True)
class IGCR2SpinRestrictedParameterization:
    norb: int
    nocc: int
    layers: int = 1
    shared_diagonal: bool = False
    interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    middle_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = 1.0

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        if int(self.layers) != self.layers or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        object.__setattr__(self, "layers", int(self.layers))
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        if self.left_right_ov_relative_scale is not None and (
            not np.isfinite(float(self.left_right_ov_relative_scale))
            or self.left_right_ov_relative_scale <= 0
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

    @property
    def pair_indices(self):
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def right_orbital_chart(self):
        if self.right_orbital_chart_override is not None:
            return self.right_orbital_chart_override
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart

    @property
    def _middle_orbital_chart(self):
        return self.middle_orbital_chart

    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)

    @property
    def n_middle_orbital_rotation_params_per_layer(self):
        return self._middle_orbital_chart.n_params(self.norb)

    @property
    def n_middle_orbital_rotation_params(self):
        return max(0, self.layers - 1) * self.n_middle_orbital_rotation_params_per_layer

    @property
    def n_double_params(self):
        return 0

    @property
    def n_pair_params(self):
        if self.shared_diagonal:
            return len(self.pair_indices)
        return self.layers * len(self.pair_indices)

    @property
    def n_pair_params_per_layer(self):
        return len(self.pair_indices)

    @property
    def n_diag_params_per_layer(self):
        return self.n_pair_params_per_layer

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_middle_orbital_rotation_params
        )

    @property
    def _middle_orbital_rotation_start(self):
        return self.n_left_orbital_rotation_params + self.n_pair_params

    @property
    def _left_right_ov_transform_scale(self):
        return _left_right_ov_transform_scale_for(
            self.right_orbital_chart,
            self.left_right_ov_relative_scale,
        )

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    @property
    def n_params(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_middle_orbital_rotation_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        idx += n
        n_pair = self.n_pair_params_per_layer
        if self.shared_diagonal:
            pair = _symmetric_matrix_from_values(
                params[idx : idx + n_pair], self.norb, self.pair_indices
            )
            pairs = [pair.copy() for _ in range(self.layers)]
            idx += n_pair
        else:
            pairs = []
            for _ in range(self.layers):
                pairs.append(
                    _symmetric_matrix_from_values(
                        params[idx : idx + n_pair], self.norb, self.pair_indices
                    )
                )
                idx += n_pair
        middle_rotations = []
        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for _ in range(self.layers - 1):
            middle_rotations.append(
                self._middle_orbital_chart.unitary_from_parameters(
                    params[idx : idx + n_middle], self.norb
                )
            )
            idx += n_middle
        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        right = final
        if self.layers == 1:
            return IGCR2Ansatz(
                diagonal=IGCR2SpinRestrictedSpec(pair=pairs[0]),
                left=left,
                right=right,
                nocc=self.nocc,
            )
        return IGCR2LayeredAnsatz(
            diagonals=tuple(IGCR2SpinRestrictedSpec(pair=pair) for pair in pairs),
            rotations=tuple([left, *middle_rotations, right]),
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz):
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        layered = _as_layered_igcr2_spin_restricted_ansatz(ansatz, self.layers)
        if layered.nocc != self.nocc:
            raise ValueError("ansatz nocc does not match parameterization")

        rotations = [np.asarray(u, dtype=np.complex128) for u in layered.rotations]
        rotation_params = []
        for idx in range(self.layers):
            chart = self._left_orbital_chart if idx == 0 else self._middle_orbital_chart
            if idx == 0:
                expected_n_params = self.n_left_orbital_rotation_params
            else:
                expected_n_params = self.n_middle_orbital_rotation_params_per_layer
            if hasattr(chart, "parameters_and_right_phase_from_unitary"):
                params_i, right_phase = chart.parameters_and_right_phase_from_unitary(
                    rotations[idx]
                )
            else:
                params_i = chart.parameters_from_unitary(rotations[idx])
                right_phase = np.zeros(self.norb, dtype=np.float64)
            if params_i.shape != (expected_n_params,):
                raise ValueError(
                    "orbital chart returned the wrong number of parameters; "
                    f"expected {(expected_n_params,)}, got {params_i.shape}"
                )
            rotation_params.append(np.asarray(params_i, dtype=np.float64))
            rotations[idx + 1] = _diag_unitary(right_phase) @ rotations[idx + 1]

        pair_mats = [
            np.asarray(diagonal.pair, dtype=np.float64)
            for diagonal in layered.diagonals
        ]
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = rotation_params[0]
        idx += n
        pair_indices = self.pair_indices
        n_pair = self.n_pair_params_per_layer
        if self.shared_diagonal:
            pair_eff = np.mean(np.stack(pair_mats, axis=0), axis=0)
            out[idx : idx + n_pair] = np.asarray(
                [pair_eff[p, q] for p, q in pair_indices], dtype=np.float64
            )
            idx += n_pair
        else:
            for pair_eff in pair_mats:
                out[idx : idx + n_pair] = np.asarray(
                    [pair_eff[p, q] for p, q in pair_indices], dtype=np.float64
                )
                idx += n_pair

        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for params_i in rotation_params[1:]:
            out[idx : idx + n_middle] = params_i
            idx += n_middle

        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(
            rotations[-1]
        )
        return self._public_parameters_from_native(out)

    def parameters_from_t_amplitudes(
        self,
        t2: np.ndarray,
        t1: np.ndarray | None = None,
        **seed_options,
    ) -> np.ndarray:
        """Seed parameters from CCSD amplitudes through the UCJ lift.

        The default path is the UCJ-lift seed so that iGCR2 initialized from
        CCSD amplitudes matches the corresponding UCJ state.  The older native
        one-layer construction remains available with ``strategy="direct"``.
        """
        strategy = seed_options.pop("strategy", seed_options.pop("seed_strategy", "ucj"))
        if strategy in {"ccsd_residual", "state_residual", "residual"}:
            raise ValueError("CCSD-residual seeding is only defined for iGCR3/iGCR4")
        if strategy in {"ucj", "ucj_lift", "ucj-t"} or self.layers != 1:
            return self.parameters_from_ucj_t_amplitudes(t2, t1=t1, **seed_options)
        if strategy not in {"direct", "native"}:
            raise ValueError(f"Unknown iGCR2 t-amplitude seed strategy: {strategy!r}")
        for key in ("optimize", "regularization", "options"):
            seed_options.pop(key, None)
        ansatz = _native_igcr2_seed_from_ccsd_t_amplitudes(
            self, t2, t1=t1, **seed_options
        )
        return self.parameters_from_ansatz(ansatz)

    def parameters_from_ucj_t_amplitudes(
        self,
        t2: np.ndarray,
        t1: np.ndarray | None = None,
        **df_options,
    ) -> np.ndarray:
        """Seed parameters by lifting ffsim's UCJ t-amplitude initializer."""
        ansatz = layered_igcr2_from_ucj_t_amplitudes(
            t2, t1=t1, layers=self.layers, nocc=self.nocc, **df_options
        )
        return self.parameters_from_ansatz(ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz):
        seeded = _igcr2_layered_spin_restricted_ansatz_from_ucj(
            ansatz,
            self.nocc,
            self.layers,
        )
        return self.parameters_from_ansatz(seeded)

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCR2SpinRestrictedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError(
                "previous ansatz nocc does not match this parameterization"
            )
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError(
                    "Pass either orbital_overlap or explicit relabeling, not both."
                )
            basis_change = orbital_transport_unitary_from_overlap(
                orbital_overlap,
                nocc=self.nocc,
                block_diagonal=block_diagonal,
            )
            ansatz = transport_igcr2_ansatz_orbitals(ansatz, basis_change)
        elif old_for_new is not None:
            ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
        return self.parameters_from_ansatz(ansatz)

    def apply(
        self,
        reference: object,
        nelec: tuple[int, int] | None = None,
    ):
        from dataclasses import replace

        from xquces.gcr.charts import GCR2FullUnitaryChart
        from xquces.gcr.references import (
            apply_ansatz_parameterization,
            reference_is_hartree_fock_state,
        )

        if nelec is None:
            nelec = (self.nocc, self.nocc)
        nelec = tuple(int(x) for x in nelec)
        parameterization = self
        use_full_right = (
            self.right_orbital_chart_override is None
            and not reference_is_hartree_fock_state(reference, self.norb, nelec)
        )
        if use_full_right:
            parameterization = replace(
                self,
                right_orbital_chart_override=GCR2FullUnitaryChart(),
            )
        return apply_ansatz_parameterization(parameterization, reference, nelec)

    def params_to_vec(
        self, reference_vec: np.ndarray, nelec: tuple[int, int]
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec, nelec=nelec, copy=True
            )

        return func


@dataclass(frozen=True)
class IGCR2SpinBalancedParameterization:
    norb: int
    nocc: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    left_right_ov_relative_scale: float | None = 3.0

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(
            self.same_spin_interaction_pairs, self.norb, allow_diagonal=False
        )
        _validate_pairs(
            self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False
        )
        if self.left_right_ov_relative_scale is not None and (
            not np.isfinite(float(self.left_right_ov_relative_scale))
            or self.left_right_ov_relative_scale <= 0
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

    @property
    def same_spin_indices(self):
        return _validate_pairs(
            self.same_spin_interaction_pairs, self.norb, allow_diagonal=False
        )

    @property
    def mixed_spin_indices(self):
        return _validate_pairs(
            self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False
        )

    @property
    def right_orbital_chart(self):
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart

    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)

    @property
    def n_same_diag_params(self):
        return self.norb

    @property
    def n_double_params(self):
        return self.norb

    @property
    def n_same_spin_params(self):
        return len(self.same_spin_indices)

    @property
    def n_mixed_spin_params(self):
        return len(self.mixed_spin_indices)

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_same_diag_params
            + self.n_double_params
            + self.n_same_spin_params
            + self.n_mixed_spin_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return _left_right_ov_transform_scale_for(
            self.right_orbital_chart,
            self.left_right_ov_relative_scale,
        )

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    @property
    def n_params(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_same_diag_params
            + self.n_double_params
            + self.n_same_spin_params
            + self.n_mixed_spin_params
            + self.n_right_orbital_rotation_params
        )

    def ansatz_from_parameters(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        idx += n
        same_diag = np.asarray(
            params[idx : idx + self.n_same_diag_params], dtype=np.float64
        )
        idx += self.n_same_diag_params
        double = np.asarray(params[idx : idx + self.n_double_params], dtype=np.float64)
        idx += self.n_double_params
        same = _symmetric_matrix_from_values(
            np.asarray(params[idx : idx + self.n_same_spin_params], dtype=np.float64),
            self.norb,
            self.same_spin_indices,
        )
        idx += self.n_same_spin_params
        mixed = _symmetric_matrix_from_values(
            np.asarray(params[idx : idx + self.n_mixed_spin_params], dtype=np.float64),
            self.norb,
            self.mixed_spin_indices,
        )
        idx += self.n_mixed_spin_params
        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        right = final
        return IGCR2Ansatz(
            diagonal=IGCR2SpinBalancedSpec(
                same_diag=same_diag, same=same, mixed=mixed, double=double
            ),
            left=left,
            right=right,
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCR2Ansatz):
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        d = ansatz.diagonal.to_standard()
        same_mat = np.asarray(d.same_spin_params, dtype=np.float64).copy()
        mixed_mat = np.asarray(d.mixed_spin_params, dtype=np.float64).copy()
        same_diag = np.diag(same_mat).copy()
        mixed_double = np.diag(mixed_mat).copy()
        np.fill_diagonal(same_mat, 0.0)
        np.fill_diagonal(mixed_mat, 0.0)
        same_full = np.asarray(
            [same_mat[p, q] for p, q in self.same_spin_indices], dtype=np.float64
        )
        mixed_full = np.asarray(
            [mixed_mat[p, q] for p, q in self.mixed_spin_indices], dtype=np.float64
        )
        left_chart = self._left_orbital_chart
        if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
            left_params, right_phase = (
                left_chart.parameters_and_right_phase_from_unitary(
                    np.asarray(ansatz.left, dtype=np.complex128)
                )
            )
        else:
            left_params = left_chart.parameters_from_unitary(
                np.asarray(ansatz.left, dtype=np.complex128)
            )
            right_phase = np.zeros(self.norb, dtype=np.float64)
        right_eff = _diag_unitary(right_phase) @ np.asarray(
            ansatz.right, dtype=np.complex128
        )
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        out[idx : idx + self.n_left_orbital_rotation_params] = left_params
        idx += self.n_left_orbital_rotation_params
        out[idx : idx + self.n_same_diag_params] = same_diag
        idx += self.n_same_diag_params
        out[idx : idx + self.n_double_params] = mixed_double
        idx += self.n_double_params
        out[idx : idx + self.n_same_spin_params] = same_full
        idx += self.n_same_spin_params
        out[idx : idx + self.n_mixed_spin_params] = mixed_full
        idx += self.n_mixed_spin_params
        n = self.n_right_orbital_rotation_params
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(right_eff)
        return self._public_parameters_from_native(out)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz):
        gcr = gcr_from_ucj_ansatz(ansatz)
        return self.parameters_from_ansatz(IGCR2Ansatz.from_gcr_ansatz(gcr, self.nocc))

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCR2SpinBalancedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError(
                "previous ansatz nocc does not match this parameterization"
            )
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError(
                    "Pass either orbital_overlap or explicit relabeling, not both."
                )
            old_for_new, phases = orbital_relabeling_from_overlap(
                orbital_overlap, nocc=self.nocc, block_diagonal=block_diagonal
            )
        if old_for_new is not None:
            ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
        return self.parameters_from_ansatz(ansatz)

    def params_to_vec(
        self, reference_vec: np.ndarray, nelec: tuple[int, int]
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec, nelec=nelec, copy=True
            )

        return func

@dataclass(frozen=True)
class IGCR3CubicReduction:
    """Quotient the restricted cubic diagonal basis by fixed-N identities.

    The identities used here are, modulo constants and one-body N_p phases,

        sum_{q != p} D_p N_q
          + 1/2 (Ne - 2) sum_{q != p} N_p N_q == 0

    and, for p < q,

        sum_{r != p,q} N_p N_q N_r
          + 2 D_p N_q + 2 D_q N_p - (Ne - 2) N_p N_q == 0.

    They are exact on the fixed-(N_alpha, N_beta) sector and remove the
    persistent iGCR-3 cubic nullspace without changing the represented state.
    """

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


def _zero_igcr3_spin_restricted_spec(norb: int) -> IGCR3SpinRestrictedSpec:
    return IGCR3SpinRestrictedSpec(
        double_params=np.zeros(norb, dtype=np.float64),
        pair_values=np.zeros(len(_default_pair_indices(norb)), dtype=np.float64),
        tau=np.zeros((norb, norb), dtype=np.float64),
        omega_values=np.zeros(len(_default_triple_indices(norb)), dtype=np.float64),
    )


def _scale_igcr3_spin_restricted_spec(
    diagonal: IGCR3SpinRestrictedSpec,
    scale: float,
) -> IGCR3SpinRestrictedSpec:
    return IGCR3SpinRestrictedSpec(
        double_params=np.asarray(diagonal.full_double(), dtype=np.float64) * scale,
        pair_values=np.asarray(diagonal.pair_values, dtype=np.float64) * scale,
        tau=np.asarray(diagonal.tau_matrix(), dtype=np.float64) * scale,
        omega_values=np.asarray(diagonal.omega_vector(), dtype=np.float64) * scale,
    )


def _as_layered_igcr3_spin_restricted_ansatz(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    layers: int,
) -> IGCR3LayeredAnsatz:
    if isinstance(ansatz, IGCR3LayeredAnsatz):
        if ansatz.layers == layers:
            return ansatz
        if ansatz.layers > layers:
            raise ValueError(
                "cannot exactly embed an IGCR3 ansatz with more layers than the "
                "target parameterization"
            )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        diagonals = list(ansatz.diagonals)
        rotations = list(ansatz.rotations)
        for _ in range(layers - ansatz.layers):
            diagonals.append(_zero_igcr3_spin_restricted_spec(ansatz.norb))
            rotations.insert(-1, identity)
        return IGCR3LayeredAnsatz(
            diagonals=tuple(diagonals),
            rotations=tuple(rotations),
            nocc=ansatz.nocc,
        )
    if ansatz.norb <= 0:
        raise ValueError("ansatz norb must be positive")
    identity = np.eye(ansatz.norb, dtype=np.complex128)
    if layers == 1:
        diagonals = [ansatz.diagonal]
    else:
        scale = 1.0 / float(layers)
        diagonals = [
            _scale_igcr3_spin_restricted_spec(ansatz.diagonal, scale)
            for _ in range(layers)
        ]
    rotations = [ansatz.left, *[identity for _ in range(layers - 1)], ansatz.right]
    return IGCR3LayeredAnsatz(
        diagonals=tuple(diagonals),
        rotations=tuple(rotations),
        nocc=ansatz.nocc,
    )


def _igcr3_ansatz_from_igcr2_any(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    if isinstance(ansatz, IGCR2LayeredAnsatz):
        if not ansatz.is_spin_restricted:
            raise TypeError(
                "iGCR-3 is currently implemented only for spin-restricted seeds"
            )
        diagonals = []
        for diagonal in ansatz.diagonals:
            d = diagonal.to_standard()
            tau, omega = spin_restricted_triples_seed_from_pair_params(
                d.pair_params,
                ansatz.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
            diagonals.append(
                IGCR3SpinRestrictedSpec.from_igcr2_diagonal(
                    diagonal,
                    tau=tau,
                    omega_values=omega,
                )
            )
        return IGCR3LayeredAnsatz(
            diagonals=tuple(diagonals),
            rotations=ansatz.rotations,
            nocc=ansatz.nocc,
        )
    return IGCR3Ansatz.from_igcr2_ansatz(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
    )


def _relabel_igcr3_diagonal(
    diagonal: IGCR3SpinRestrictedSpec,
    old_for_new: np.ndarray,
) -> IGCR3SpinRestrictedSpec:
    d = diagonal
    norb = d.norb
    double = d.full_double()[old_for_new]
    pair = d.pair_matrix()[np.ix_(old_for_new, old_for_new)]
    tau = d.tau_matrix()[np.ix_(old_for_new, old_for_new)]
    pair_values = np.asarray(
        [pair[p, q] for p, q in _default_pair_indices(norb)],
        dtype=np.float64,
    )
    omega_old = {
        (p, q, r): value for value, (p, q, r) in zip(d.omega_vector(), d.omega_indices)
    }
    omega_values = np.asarray(
        [
            omega_old[
                tuple(
                    sorted(
                        (
                            int(old_for_new[p]),
                            int(old_for_new[q]),
                            int(old_for_new[r]),
                        )
                    )
                )
            ]
            for p, q, r in _default_triple_indices(norb)
        ],
        dtype=np.float64,
    )
    return IGCR3SpinRestrictedSpec(
        double_params=double,
        pair_values=pair_values,
        tau=tau,
        omega_values=omega_values,
    )


def relabel_igcr3_ansatz_orbitals(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    if ansatz.norb != len(old_for_new):
        raise ValueError("orbital permutation length must match ansatz.norb")
    relabel = _orbital_relabeling_unitary(old_for_new, phases)
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if isinstance(ansatz, IGCR3LayeredAnsatz):
        return IGCR3LayeredAnsatz(
            diagonals=tuple(
                _relabel_igcr3_diagonal(diagonal, old_for_new)
                for diagonal in ansatz.diagonals
            ),
            rotations=tuple(relabel.conj().T @ rot @ relabel for rot in ansatz.rotations),
            nocc=ansatz.nocc,
        )
    diagonal = _relabel_igcr3_diagonal(ansatz.diagonal, old_for_new)
    return IGCR3Ansatz(
        diagonal=diagonal,
        left=relabel.conj().T @ ansatz.left @ relabel,
        right=relabel.conj().T @ ansatz.right @ relabel,
        nocc=ansatz.nocc,
    )


def transport_igcr3_ansatz_orbitals(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz, basis_change: np.ndarray
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    basis_change = np.asarray(basis_change, dtype=np.complex128)
    if basis_change.shape != (ansatz.norb, ansatz.norb):
        raise ValueError(
            f"basis_change must have shape {(ansatz.norb, ansatz.norb)}, "
            f"got {basis_change.shape}."
        )
    if not np.allclose(
        basis_change.conj().T @ basis_change,
        np.eye(ansatz.norb),
        atol=1e-10,
    ):
        raise ValueError("basis_change must be unitary")
    if isinstance(ansatz, IGCR3LayeredAnsatz):
        rotations = list(ansatz.rotations)
        rotations[0] = basis_change.conj().T @ rotations[0]
        return IGCR3LayeredAnsatz(
            diagonals=ansatz.diagonals,
            rotations=tuple(rotations),
            nocc=ansatz.nocc,
        )
    return IGCR3Ansatz(
        diagonal=ansatz.diagonal,
        left=basis_change.conj().T @ np.asarray(ansatz.left, dtype=np.complex128),
        right=np.asarray(ansatz.right, dtype=np.complex128),
        nocc=ansatz.nocc,
    )


@dataclass(frozen=True)
class IGCR3SpinRestrictedParameterization:
    norb: int
    nocc: int
    layers: int = 1
    shared_diagonal: bool = False
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    middle_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = 3.0

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        if int(self.layers) != self.layers or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        object.__setattr__(self, "layers", int(self.layers))
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        _validate_ordered_pairs(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)
        if self.left_right_ov_relative_scale is not None and (
            not np.isfinite(float(self.left_right_ov_relative_scale))
            or self.left_right_ov_relative_scale <= 0
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

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
    def right_orbital_chart(self):
        if self.right_orbital_chart_override is not None:
            return self.right_orbital_chart_override
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart

    @property
    def _middle_orbital_chart(self):
        return self.middle_orbital_chart

    @property
    def _right_depends_on_prefix(self) -> bool:
        return True

    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)

    @property
    def n_middle_orbital_rotation_params_per_layer(self):
        return self._middle_orbital_chart.n_params(self.norb)

    @property
    def n_middle_orbital_rotation_params(self):
        return max(0, self.layers - 1) * self.n_middle_orbital_rotation_params_per_layer

    @property
    def n_double_params(self):
        return 0

    @property
    def n_pair_params(self):
        if self.shared_diagonal:
            return self.n_pair_params_per_layer
        return self.layers * self.n_pair_params_per_layer

    @property
    def n_pair_params_per_layer(self):
        return len(self.pair_indices)

    @property
    def n_tau_params(self):
        if self.shared_diagonal:
            return self.n_tau_params_per_layer
        return self.layers * self.n_tau_params_per_layer

    @property
    def n_tau_params_per_layer(self):
        if self.uses_reduced_cubic_chart:
            return self.cubic_reduction.n_params
        return len(self.tau_indices)

    @property
    def n_omega_params(self):
        if self.shared_diagonal:
            return self.n_omega_params_per_layer
        return self.layers * self.n_omega_params_per_layer

    @property
    def n_omega_params_per_layer(self):
        if self.uses_reduced_cubic_chart:
            return 0
        return len(self.omega_indices)

    @property
    def n_diag_params_per_layer(self):
        return (
            self.n_pair_params_per_layer
            + self.n_tau_params_per_layer
            + self.n_omega_params_per_layer
        )

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_middle_orbital_rotation_params
        )

    @property
    def _middle_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return None

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    @property
    def n_params(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_middle_orbital_rotation_params
            + self.n_right_orbital_rotation_params
        )

    def sector_sizes(self) -> dict[str, int]:
        return {
            "left": self.n_left_orbital_rotation_params,
            "double": self.n_double_params,
            "pair": self.n_pair_params,
            "tau": 0 if self.uses_reduced_cubic_chart else self.n_tau_params,
            "omega": self.n_omega_params,
            "cubic": self.n_tau_params
            if self.uses_reduced_cubic_chart
            else (self.n_tau_params + self.n_omega_params),
            "middle": self.n_middle_orbital_rotation_params,
            "right": self.n_right_orbital_rotation_params,
            "total": self.n_params,
        }

    def _diagonal_from_native_parameters(
        self,
        params: np.ndarray,
    ) -> IGCR3SpinRestrictedSpec:
        idx = 0

        n = self.n_pair_params_per_layer
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
            n = self.n_tau_params_per_layer
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
            n = self.n_tau_params_per_layer
            tau = _ordered_matrix_from_values(
                params[idx : idx + n], self.norb, self.tau_indices
            )
            idx += n

            n = self.n_omega_params_per_layer
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

        if idx != self.n_diag_params_per_layer:
            raise ValueError("diagonal parameter block has inconsistent length")
        return IGCR3SpinRestrictedSpec(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=pair_values,
            tau=tau,
            omega_values=omega_values,
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0

        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        idx += n

        n_diag = self.n_diag_params_per_layer
        if self.shared_diagonal:
            diagonal_params = [params[idx : idx + n_diag]] * self.layers
            idx += n_diag
        else:
            diagonal_params = []
            for _ in range(self.layers):
                diagonal_params.append(params[idx : idx + n_diag])
                idx += n_diag
        diagonals = tuple(
            self._diagonal_from_native_parameters(block)
            for block in diagonal_params
        )

        middle_rotations = []
        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for _ in range(self.layers - 1):
            middle_rotations.append(
                self._middle_orbital_chart.unitary_from_parameters(
                    params[idx : idx + n_middle], self.norb
                )
            )
            idx += n_middle

        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        prefix = np.asarray(left, dtype=np.complex128)
        for rotation in middle_rotations:
            prefix = prefix @ np.asarray(rotation, dtype=np.complex128)
        right = _right_unitary_from_left_and_final(prefix, final, self.nocc)

        if self.layers == 1:
            return IGCR3Ansatz(
                diagonal=diagonals[0],
                left=left,
                right=right,
                nocc=self.nocc,
            )
        return IGCR3LayeredAnsatz(
            diagonals=diagonals,
            rotations=tuple([left, *middle_rotations, right]),
            nocc=self.nocc,
        )

    def _native_parameters_from_diagonal(
        self,
        diagonal: IGCR3SpinRestrictedSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        d = diagonal
        pair_eff = _restricted_irreducible_pair_matrix(d.full_double(), d.pair_matrix())
        tau = d.tau_matrix()
        omega = d.omega_vector()

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
            _restricted_left_phase_vector(d.full_double(), self.nocc)
            + cubic_onebody_phase
        )

        out = np.zeros(self.n_diag_params_per_layer, dtype=np.float64)
        idx = 0
        pair_reduced_matrix = _symmetric_matrix_from_values(
            reduced_pair_values, self.norb, _default_pair_indices(self.norb)
        )
        n = self.n_pair_params_per_layer
        out[idx : idx + n] = np.asarray(
            [pair_reduced_matrix[p, q] for p, q in self.pair_indices], dtype=np.float64
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params_per_layer
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
            n = self.n_tau_params_per_layer
            out[idx : idx + n] = _values_from_ordered_matrix(
                tau_adjusted, self.tau_indices
            )
            idx += n

            n = self.n_omega_params_per_layer
            out[idx : idx + n] = np.asarray(
                [omega_adjusted[t] for t in self.omega_indices], dtype=np.float64
            )
            idx += n

        return out, phase_vec

    def parameters_from_ansatz(
        self,
        ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    ) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        layered = _as_layered_igcr3_spin_restricted_ansatz(ansatz, self.layers)
        if layered.nocc != self.nocc:
            raise ValueError("ansatz nocc does not match parameterization")

        rotations = [np.asarray(u, dtype=np.complex128) for u in layered.rotations]
        diag_params = []
        for layer_idx, diagonal in enumerate(layered.diagonals):
            params_i, phase_vec = self._native_parameters_from_diagonal(diagonal)
            diag_params.append(params_i)
            rotations[layer_idx] = rotations[layer_idx] @ _diag_unitary(phase_vec)

        rotation_params = []
        for layer_idx in range(self.layers):
            chart = (
                self._left_orbital_chart
                if layer_idx == 0
                else self._middle_orbital_chart
            )
            expected = (
                self.n_left_orbital_rotation_params
                if layer_idx == 0
                else self.n_middle_orbital_rotation_params_per_layer
            )
            if hasattr(chart, "parameters_and_right_phase_from_unitary"):
                params_i, right_phase = chart.parameters_and_right_phase_from_unitary(
                    rotations[layer_idx]
                )
            else:
                params_i = chart.parameters_from_unitary(rotations[layer_idx])
                right_phase = np.zeros(self.norb, dtype=np.float64)
            if params_i.shape != (expected,):
                raise ValueError(
                    "orbital chart returned the wrong number of parameters; "
                    f"expected {(expected,)}, got {params_i.shape}"
                )
            rotation_params.append(np.asarray(params_i, dtype=np.float64))
            rotations[layer_idx + 1] = _diag_unitary(right_phase) @ rotations[layer_idx + 1]

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = rotation_params[0]
        idx += n

        n_diag = self.n_diag_params_per_layer
        if self.shared_diagonal:
            out[idx : idx + n_diag] = np.mean(np.stack(diag_params, axis=0), axis=0)
            idx += n_diag
        else:
            for params_i in diag_params:
                out[idx : idx + n_diag] = params_i
                idx += n_diag

        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for params_i in rotation_params[1:]:
            out[idx : idx + n_middle] = params_i
            idx += n_middle

        n = self.n_right_orbital_rotation_params
        prefix = np.eye(self.norb, dtype=np.complex128)
        for layer_idx, params_i in enumerate(rotation_params):
            chart = (
                self._left_orbital_chart
                if layer_idx == 0
                else self._middle_orbital_chart
            )
            prefix = prefix @ chart.unitary_from_parameters(params_i, self.norb)
        final_eff = _final_unitary_from_left_and_right(
            prefix,
            rotations[-1],
            self.nocc,
            project_reference_ov=self.right_orbital_chart_override is None,
        )
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(final_eff)
        return self._public_parameters_from_native(out)

    def parameters_from_igcr2_ansatz(
        self,
        ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            _igcr3_ansatz_from_igcr2_any(
                ansatz,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
        )

    def parameters_from_t_amplitudes(
        self,
        t2: np.ndarray,
        t1: np.ndarray | None = None,
        **seed_options,
    ) -> np.ndarray:
        """Seed iGCR3 from CCSD amplitudes by non-variational state matching.

        ``strategy="ccsd_residual"`` constructs a CCSD target state from
        ``exp(T1 + T2)|HF>`` truncated at ``target_max_power`` and projects the
        residual onto this parameterization's tangent space.  No Hamiltonian or
        energy minimization is used by the initializer.
        """
        strategy = seed_options.pop(
            "strategy",
            seed_options.pop("seed_strategy", "ccsd_residual"),
        )
        igcr2_param = IGCR2SpinRestrictedParameterization(
            norb=self.norb,
            nocc=self.nocc,
            layers=self.layers,
            shared_diagonal=self.shared_diagonal,
            interaction_pairs=self.interaction_pairs,
            left_orbital_chart=self.left_orbital_chart,
            middle_orbital_chart=self.middle_orbital_chart,
            right_orbital_chart_override=self.right_orbital_chart_override,
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
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
        x_base = self.parameters_from_igcr2_ansatz(
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
                self,
                "cubic",
                ("tau", "omega"),
            )
        return _parameters_from_ccsd_residual_seed(
            self,
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
            return_info=seed_options.pop("return_info", False),
        )

    def parameters_from_ucj_ansatz(
        self,
        ansatz: UCJAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR3Ansatz.from_ucj_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
        )

    def parameters_from_gcr_ansatz(
        self,
        ansatz: GCRAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR3Ansatz.from_gcr_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
            )
        )

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCR3SpinRestrictedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError(
                "previous ansatz nocc does not match this parameterization"
            )
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError(
                    "Pass either orbital_overlap or explicit relabeling, not both."
                )
            basis_change = orbital_transport_unitary_from_overlap(
                orbital_overlap,
                nocc=self.nocc,
                block_diagonal=block_diagonal,
            )
            if isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
                ansatz = transport_igcr3_ansatz_orbitals(ansatz, basis_change)
            elif isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
                ansatz = transport_igcr2_ansatz_orbitals(ansatz, basis_change)
            else:
                raise TypeError(
                    f"Unsupported ansatz type for transfer: {type(ansatz)!r}"
                )
        elif old_for_new is not None:
            if isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
                ansatz = relabel_igcr3_ansatz_orbitals(ansatz, old_for_new, phases)
            elif isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
                ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
            else:
                raise TypeError(
                    f"Unsupported ansatz type for transfer: {type(ansatz)!r}"
                )
        if isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
            return self.parameters_from_ansatz(ansatz)
        if isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
            return self.parameters_from_igcr2_ansatz(ansatz)
        raise TypeError(f"Unsupported ansatz type for transfer: {type(ansatz)!r}")

    def apply(
        self,
        reference: object,
        nelec: tuple[int, int] | None = None,
    ):
        from dataclasses import replace

        from xquces.gcr.charts import GCR2FullUnitaryChart
        from xquces.gcr.references import (
            apply_ansatz_parameterization,
            reference_is_hartree_fock_state,
        )

        if nelec is None:
            nelec = (self.nocc, self.nocc)
        nelec = tuple(int(x) for x in nelec)
        parameterization = self
        use_full_right = (
            self.right_orbital_chart_override is None
            and not reference_is_hartree_fock_state(reference, self.norb, nelec)
        )
        if use_full_right:
            parameterization = replace(
                self,
                right_orbital_chart_override=GCR2FullUnitaryChart(),
            )
        return apply_ansatz_parameterization(parameterization, reference, nelec)

    def params_to_vec(
        self, reference_vec: np.ndarray, nelec: tuple[int, int]
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec, nelec=nelec, copy=True
            )

        return func


def igcr3_from_igcr2_ansatz(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    return _igcr3_ansatz_from_igcr2_any(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
    )

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
            ansatz, nocc,
            tau_scale=tau_scale, omega_scale=omega_scale,
            eta_scale=eta_scale, rho_scale=rho_scale, sigma_scale=sigma_scale,
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


def _zero_igcr4_spin_restricted_spec(norb: int) -> IGCR4SpinRestrictedSpec:
    return IGCR4SpinRestrictedSpec(
        double_params=np.zeros(norb, dtype=np.float64),
        pair_values=np.zeros(len(_default_pair_indices(norb)), dtype=np.float64),
        tau=np.zeros((norb, norb), dtype=np.float64),
        omega_values=np.zeros(len(_default_triple_indices(norb)), dtype=np.float64),
        eta_values=np.zeros(len(_default_eta_indices(norb)), dtype=np.float64),
        rho_values=np.zeros(len(_default_rho_indices(norb)), dtype=np.float64),
        sigma_values=np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64),
    )


def _scale_igcr4_spin_restricted_spec(
    diagonal: IGCR4SpinRestrictedSpec,
    scale: float,
) -> IGCR4SpinRestrictedSpec:
    return IGCR4SpinRestrictedSpec(
        double_params=np.asarray(diagonal.full_double(), dtype=np.float64) * scale,
        pair_values=np.asarray(diagonal.pair_values, dtype=np.float64) * scale,
        tau=np.asarray(diagonal.tau_matrix(), dtype=np.float64) * scale,
        omega_values=np.asarray(diagonal.omega_vector(), dtype=np.float64) * scale,
        eta_values=np.asarray(diagonal.eta_vector(), dtype=np.float64) * scale,
        rho_values=np.asarray(diagonal.rho_vector(), dtype=np.float64) * scale,
        sigma_values=np.asarray(diagonal.sigma_vector(), dtype=np.float64) * scale,
    )


def _as_layered_igcr4_spin_restricted_ansatz(
    ansatz: IGCR4Ansatz | IGCR4LayeredAnsatz,
    layers: int,
) -> IGCR4LayeredAnsatz:
    if isinstance(ansatz, IGCR4LayeredAnsatz):
        if ansatz.layers == layers:
            return ansatz
        if ansatz.layers > layers:
            raise ValueError(
                "cannot exactly embed an IGCR4 ansatz with more layers than the "
                "target parameterization"
            )
        identity = np.eye(ansatz.norb, dtype=np.complex128)
        diagonals = list(ansatz.diagonals)
        rotations = list(ansatz.rotations)
        for _ in range(layers - ansatz.layers):
            diagonals.append(_zero_igcr4_spin_restricted_spec(ansatz.norb))
            rotations.insert(-1, identity)
        return IGCR4LayeredAnsatz(
            diagonals=tuple(diagonals),
            rotations=tuple(rotations),
            nocc=ansatz.nocc,
        )
    if ansatz.norb <= 0:
        raise ValueError("ansatz norb must be positive")
    identity = np.eye(ansatz.norb, dtype=np.complex128)
    if layers == 1:
        diagonals = [ansatz.diagonal]
    else:
        scale = 1.0 / float(layers)
        diagonals = [
            _scale_igcr4_spin_restricted_spec(ansatz.diagonal, scale)
            for _ in range(layers)
        ]
    rotations = [ansatz.left, *[identity for _ in range(layers - 1)], ansatz.right]
    return IGCR4LayeredAnsatz(
        diagonals=tuple(diagonals),
        rotations=tuple(rotations),
        nocc=ansatz.nocc,
    )


def _igcr4_ansatz_from_igcr3_any(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    if isinstance(ansatz, IGCR3LayeredAnsatz):
        diagonals = []
        for diagonal in ansatz.diagonals:
            eta, rho, sigma = spin_restricted_quartic_seed_from_pair_params(
                diagonal.pair_matrix(),
                ansatz.nocc,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
            diagonals.append(
                IGCR4SpinRestrictedSpec.from_igcr3_diagonal(
                    diagonal,
                    eta_values=eta,
                    rho_values=rho,
                    sigma_values=sigma,
                )
            )
        return IGCR4LayeredAnsatz(
            diagonals=tuple(diagonals),
            rotations=ansatz.rotations,
            nocc=ansatz.nocc,
        )
    return IGCR4Ansatz.from_igcr3_ansatz(
        ansatz,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    )


def _igcr4_ansatz_from_igcr2_any(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    igcr3 = _igcr3_ansatz_from_igcr2_any(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
    )
    return _igcr4_ansatz_from_igcr3_any(
        igcr3,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    )


def _relabel_igcr4_diagonal(
    diagonal: IGCR4SpinRestrictedSpec,
    old_for_new: np.ndarray,
) -> IGCR4SpinRestrictedSpec:
    d = diagonal
    norb = d.norb
    double = d.full_double()[old_for_new]
    pair = d.pair_matrix()[np.ix_(old_for_new, old_for_new)]
    tau = d.tau_matrix()[np.ix_(old_for_new, old_for_new)]

    pair_values = np.asarray(
        [pair[p, q] for p, q in _default_pair_indices(norb)],
        dtype=np.float64,
    )

    omega_old = {idx: val for idx, val in zip(d.omega_indices, d.omega_vector())}
    omega_values = np.asarray(
        [
            omega_old[
                tuple(
                    sorted(
                        (int(old_for_new[p]), int(old_for_new[q]), int(old_for_new[r]))
                    )
                )
            ]
            for p, q, r in _default_triple_indices(norb)
        ],
        dtype=np.float64,
    )

    eta_old = {idx: val for idx, val in zip(d.eta_indices, d.eta_vector())}
    eta_values = np.asarray(
        [
            eta_old[
                (int(old_for_new[p]), int(old_for_new[q]))
                if old_for_new[p] < old_for_new[q]
                else (int(old_for_new[q]), int(old_for_new[p]))
            ]
            for p, q in _default_eta_indices(norb)
        ],
        dtype=np.float64,
    )

    rho_old = {idx: val for idx, val in zip(d.rho_indices, d.rho_vector())}
    rho_values = np.asarray(
        [
            rho_old[
                (
                    int(old_for_new[p]),
                    min(int(old_for_new[q]), int(old_for_new[r])),
                    max(int(old_for_new[q]), int(old_for_new[r])),
                )
            ]
            for p, q, r in _default_rho_indices(norb)
        ],
        dtype=np.float64,
    )

    sigma_old = {idx: val for idx, val in zip(d.sigma_indices, d.sigma_vector())}
    sigma_values = np.asarray(
        [
            sigma_old[
                tuple(
                    sorted(
                        (
                            int(old_for_new[p]),
                            int(old_for_new[q]),
                            int(old_for_new[r]),
                            int(old_for_new[s]),
                        )
                    )
                )
            ]
            for p, q, r, s in _default_sigma_indices(norb)
        ],
        dtype=np.float64,
    )

    return IGCR4SpinRestrictedSpec(
        double_params=double,
        pair_values=pair_values,
        tau=tau,
        omega_values=omega_values,
        eta_values=eta_values,
        rho_values=rho_values,
        sigma_values=sigma_values,
    )


def relabel_igcr4_ansatz_orbitals(
    ansatz: IGCR4Ansatz | IGCR4LayeredAnsatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    if ansatz.norb != len(old_for_new):
        raise ValueError("orbital permutation length must match ansatz.norb")
    relabel = _orbital_relabeling_unitary(old_for_new, phases)
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if isinstance(ansatz, IGCR4LayeredAnsatz):
        return IGCR4LayeredAnsatz(
            diagonals=tuple(
                _relabel_igcr4_diagonal(diagonal, old_for_new)
                for diagonal in ansatz.diagonals
            ),
            rotations=tuple(relabel.conj().T @ rot @ relabel for rot in ansatz.rotations),
            nocc=ansatz.nocc,
        )
    diagonal = _relabel_igcr4_diagonal(ansatz.diagonal, old_for_new)

    return IGCR4Ansatz(
        diagonal=diagonal,
        left=relabel.conj().T @ ansatz.left @ relabel,
        right=relabel.conj().T @ ansatz.right @ relabel,
        nocc=ansatz.nocc,
    )


def transport_igcr4_ansatz_orbitals(
    ansatz: IGCR4Ansatz | IGCR4LayeredAnsatz, basis_change: np.ndarray
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    basis_change = np.asarray(basis_change, dtype=np.complex128)
    if basis_change.shape != (ansatz.norb, ansatz.norb):
        raise ValueError(
            f"basis_change must have shape {(ansatz.norb, ansatz.norb)}, "
            f"got {basis_change.shape}."
        )
    if not np.allclose(
        basis_change.conj().T @ basis_change,
        np.eye(ansatz.norb),
        atol=1e-10,
    ):
        raise ValueError("basis_change must be unitary")
    if isinstance(ansatz, IGCR4LayeredAnsatz):
        rotations = list(ansatz.rotations)
        rotations[0] = basis_change.conj().T @ rotations[0]
        return IGCR4LayeredAnsatz(
            diagonals=ansatz.diagonals,
            rotations=tuple(rotations),
            nocc=ansatz.nocc,
        )
    return IGCR4Ansatz(
        diagonal=ansatz.diagonal,
        left=basis_change.conj().T @ np.asarray(ansatz.left, dtype=np.complex128),
        right=np.asarray(ansatz.right, dtype=np.complex128),
        nocc=ansatz.nocc,
    )


@dataclass(frozen=True)
class IGCR4SpinRestrictedParameterization:
    norb: int
    nocc: int
    layers: int = 1
    shared_diagonal: bool = False
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    eta_indices_: list[tuple[int, int]] | None = None
    rho_indices_: list[tuple[int, int, int]] | None = None
    sigma_indices_: list[tuple[int, int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
    reduce_quartic_gauge: bool = True
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    middle_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = 3.0

    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        if int(self.layers) != self.layers or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        object.__setattr__(self, "layers", int(self.layers))
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        _validate_ordered_pairs(self.tau_indices_, self.norb)
        _validate_triples(self.omega_indices_, self.norb)
        _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)
        _validate_rho_indices(self.rho_indices_, self.norb)
        _validate_sigma_indices(self.sigma_indices_, self.norb)
        if self.left_right_ov_relative_scale is not None and (
            not np.isfinite(float(self.left_right_ov_relative_scale))
            or self.left_right_ov_relative_scale <= 0
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

    @property
    def pair_indices(self):
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)

    @property
    def tau_indices(self):
        return _validate_ordered_pairs(self.tau_indices_, self.norb)

    @property
    def omega_indices(self):
        return _validate_triples(self.omega_indices_, self.norb)

    @property
    def eta_indices(self):
        return _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)

    @property
    def rho_indices(self):
        return _validate_rho_indices(self.rho_indices_, self.norb)

    @property
    def sigma_indices(self):
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
    def right_orbital_chart(self):
        if self.right_orbital_chart_override is not None:
            return self.right_orbital_chart_override
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)

    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart

    @property
    def _middle_orbital_chart(self):
        return self.middle_orbital_chart

    @property
    def _right_depends_on_prefix(self) -> bool:
        return True

    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)

    @property
    def n_middle_orbital_rotation_params_per_layer(self):
        return self._middle_orbital_chart.n_params(self.norb)

    @property
    def n_middle_orbital_rotation_params(self):
        return max(0, self.layers - 1) * self.n_middle_orbital_rotation_params_per_layer

    @property
    def n_pair_params(self):
        if self.shared_diagonal:
            return self.n_pair_params_per_layer
        return self.layers * self.n_pair_params_per_layer

    @property
    def n_pair_params_per_layer(self):
        return len(self.pair_indices)

    @property
    def n_tau_params(self):
        if self.shared_diagonal:
            return self.n_tau_params_per_layer
        return self.layers * self.n_tau_params_per_layer

    @property
    def n_tau_params_per_layer(self):
        if self.uses_reduced_cubic_chart:
            return self.cubic_reduction.n_params
        return len(self.tau_indices)

    @property
    def n_omega_params(self):
        if self.shared_diagonal:
            return self.n_omega_params_per_layer
        return self.layers * self.n_omega_params_per_layer

    @property
    def n_omega_params_per_layer(self):
        if self.uses_reduced_cubic_chart:
            return 0
        return len(self.omega_indices)

    @property
    def n_eta_params(self):
        if self.shared_diagonal:
            return self.n_eta_params_per_layer
        return self.layers * self.n_eta_params_per_layer

    @property
    def n_eta_params_per_layer(self):
        if self.uses_reduced_quartic_chart:
            return 0
        return len(self.eta_indices)

    @property
    def n_rho_params(self):
        if self.shared_diagonal:
            return self.n_rho_params_per_layer
        return self.layers * self.n_rho_params_per_layer

    @property
    def n_rho_params_per_layer(self):
        if self.uses_reduced_quartic_chart:
            return self.quartic_reduction.n_params
        return len(self.rho_indices)

    @property
    def n_sigma_params(self):
        if self.shared_diagonal:
            return self.n_sigma_params_per_layer
        return self.layers * self.n_sigma_params_per_layer

    @property
    def n_sigma_params_per_layer(self):
        if self.uses_reduced_quartic_chart:
            return 0
        return len(self.sigma_indices)

    @property
    def n_diag_params_per_layer(self):
        return (
            self.n_pair_params_per_layer
            + self.n_tau_params_per_layer
            + self.n_omega_params_per_layer
            + self.n_eta_params_per_layer
            + self.n_rho_params_per_layer
            + self.n_sigma_params_per_layer
        )

    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def _right_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma_params
            + self.n_middle_orbital_rotation_params
        )

    @property
    def _middle_orbital_rotation_start(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return None

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self._left_right_ov_transform_scale,
        )

    @property
    def n_params(self):
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_tau_params
            + self.n_omega_params
            + self.n_eta_params
            + self.n_rho_params
            + self.n_sigma_params
            + self.n_middle_orbital_rotation_params
            + self.n_right_orbital_rotation_params
        )

    def sector_sizes(self) -> dict[str, int]:
        return {
            "left": self.n_left_orbital_rotation_params,
            "pair": self.n_pair_params,
            "tau": 0 if self.uses_reduced_cubic_chart else self.n_tau_params,
            "omega": self.n_omega_params,
            "eta": 0 if self.uses_reduced_quartic_chart else self.n_eta_params,
            "rho": self.n_rho_params,
            "sigma": 0 if self.uses_reduced_quartic_chart else self.n_sigma_params,
            "middle": self.n_middle_orbital_rotation_params,
            "right": self.n_right_orbital_rotation_params,
            "total": self.n_params,
        }

    def _diagonal_from_native_parameters(
        self,
        params: np.ndarray,
    ) -> IGCR4SpinRestrictedSpec:
        idx = 0

        n = self.n_pair_params_per_layer
        pair_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
        pair_sparse = _symmetric_matrix_from_values(
            pair_sparse_values, self.norb, self.pair_indices
        )
        pair_values = np.asarray(
            [pair_sparse[p, q] for p, q in _default_pair_indices(self.norb)],
            dtype=np.float64,
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params_per_layer
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
            n = self.n_tau_params_per_layer
            tau = _ordered_matrix_from_values(
                params[idx : idx + n], self.norb, self.tau_indices
            )
            idx += n

            n = self.n_omega_params_per_layer
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
            n = self.n_rho_params_per_layer
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
            n = self.n_eta_params_per_layer
            eta_sparse_values = np.asarray(params[idx : idx + n], dtype=np.float64)
            eta_sparse = {
                pair: value for pair, value in zip(self.eta_indices, eta_sparse_values)
            }
            eta_values = np.asarray(
                [eta_sparse.get(pair, 0.0) for pair in _default_eta_indices(self.norb)],
                dtype=np.float64,
            )
            idx += n

            n = self.n_rho_params_per_layer
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

            n = self.n_sigma_params_per_layer
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

        if idx != self.n_diag_params_per_layer:
            raise ValueError("diagonal parameter block has inconsistent length")
        return IGCR4SpinRestrictedSpec(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=pair_values,
            tau=tau,
            omega_values=omega_values,
            eta_values=eta_values,
            rho_values=rho_values,
            sigma_values=sigma_values,
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0

        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        idx += n

        n_diag = self.n_diag_params_per_layer
        if self.shared_diagonal:
            diagonal_params = [params[idx : idx + n_diag]] * self.layers
            idx += n_diag
        else:
            diagonal_params = []
            for _ in range(self.layers):
                diagonal_params.append(params[idx : idx + n_diag])
                idx += n_diag
        diagonals = tuple(
            self._diagonal_from_native_parameters(block)
            for block in diagonal_params
        )

        middle_rotations = []
        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for _ in range(self.layers - 1):
            middle_rotations.append(
                self._middle_orbital_chart.unitary_from_parameters(
                    params[idx : idx + n_middle], self.norb
                )
            )
            idx += n_middle

        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        prefix = np.asarray(left, dtype=np.complex128)
        for rotation in middle_rotations:
            prefix = prefix @ np.asarray(rotation, dtype=np.complex128)
        right = _right_unitary_from_left_and_final(prefix, final, self.nocc)

        if self.layers == 1:
            return IGCR4Ansatz(
                diagonal=diagonals[0],
                left=left,
                right=right,
                nocc=self.nocc,
            )
        return IGCR4LayeredAnsatz(
            diagonals=diagonals,
            rotations=tuple([left, *middle_rotations, right]),
            nocc=self.nocc,
        )

    def _native_parameters_from_diagonal(
        self,
        diagonal: IGCR4SpinRestrictedSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        d = diagonal
        pair_eff = _restricted_irreducible_pair_matrix(d.full_double(), d.pair_matrix())
        tau = d.tau_matrix()
        omega = d.omega_vector()
        eta = d.eta_vector()
        rho = d.rho_vector()
        sigma = d.sigma_vector()

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
            _restricted_left_phase_vector(d.full_double(), self.nocc)
            + cubic_onebody_phase
        )

        out = np.zeros(self.n_diag_params_per_layer, dtype=np.float64)
        idx = 0

        n = self.n_pair_params_per_layer
        pair_reduced_matrix = _symmetric_matrix_from_values(
            reduced_pair_values, self.norb, _default_pair_indices(self.norb)
        )
        out[idx : idx + n] = np.asarray(
            [pair_reduced_matrix[p, q] for p, q in self.pair_indices], dtype=np.float64
        )
        idx += n

        if self.uses_reduced_cubic_chart:
            n = self.n_tau_params_per_layer
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
            n = self.n_tau_params_per_layer
            out[idx : idx + n] = _values_from_ordered_matrix(
                tau_adjusted, self.tau_indices
            )
            idx += n

            n = self.n_omega_params_per_layer
            out[idx : idx + n] = np.asarray(
                [omega_adjusted[t] for t in self.omega_indices], dtype=np.float64
            )
            idx += n

        if self.uses_reduced_quartic_chart:
            n = self.n_rho_params_per_layer
            out[idx : idx + n] = reduced_quartic_values
            idx += n
        else:
            n = self.n_eta_params_per_layer
            full_eta = {pair: value for value, pair in zip(eta, d.eta_indices)}
            out[idx : idx + n] = np.asarray(
                [full_eta[t] for t in self.eta_indices], dtype=np.float64
            )
            idx += n

            n = self.n_rho_params_per_layer
            full_rho = {triple: value for value, triple in zip(rho, d.rho_indices)}
            out[idx : idx + n] = np.asarray(
                [full_rho[t] for t in self.rho_indices], dtype=np.float64
            )
            idx += n

            n = self.n_sigma_params_per_layer
            full_sigma = {quad: value for value, quad in zip(sigma, d.sigma_indices)}
            out[idx : idx + n] = np.asarray(
                [full_sigma[t] for t in self.sigma_indices], dtype=np.float64
            )
            idx += n

        return out, phase_vec

    def parameters_from_ansatz(
        self,
        ansatz: IGCR4Ansatz | IGCR4LayeredAnsatz,
    ) -> np.ndarray:
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        layered = _as_layered_igcr4_spin_restricted_ansatz(ansatz, self.layers)
        if layered.nocc != self.nocc:
            raise ValueError("ansatz nocc does not match parameterization")

        rotations = [np.asarray(u, dtype=np.complex128) for u in layered.rotations]
        diag_params = []
        for layer_idx, diagonal in enumerate(layered.diagonals):
            params_i, phase_vec = self._native_parameters_from_diagonal(diagonal)
            diag_params.append(params_i)
            rotations[layer_idx] = rotations[layer_idx] @ _diag_unitary(phase_vec)

        rotation_params = []
        for layer_idx in range(self.layers):
            chart = (
                self._left_orbital_chart
                if layer_idx == 0
                else self._middle_orbital_chart
            )
            expected = (
                self.n_left_orbital_rotation_params
                if layer_idx == 0
                else self.n_middle_orbital_rotation_params_per_layer
            )
            if hasattr(chart, "parameters_and_right_phase_from_unitary"):
                params_i, right_phase = chart.parameters_and_right_phase_from_unitary(
                    rotations[layer_idx]
                )
            else:
                params_i = chart.parameters_from_unitary(rotations[layer_idx])
                right_phase = np.zeros(self.norb, dtype=np.float64)
            if params_i.shape != (expected,):
                raise ValueError(
                    "orbital chart returned the wrong number of parameters; "
                    f"expected {(expected,)}, got {params_i.shape}"
                )
            rotation_params.append(np.asarray(params_i, dtype=np.float64))
            rotations[layer_idx + 1] = _diag_unitary(right_phase) @ rotations[layer_idx + 1]

        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = rotation_params[0]
        idx += n

        n_diag = self.n_diag_params_per_layer
        if self.shared_diagonal:
            out[idx : idx + n_diag] = np.mean(np.stack(diag_params, axis=0), axis=0)
            idx += n_diag
        else:
            for params_i in diag_params:
                out[idx : idx + n_diag] = params_i
                idx += n_diag

        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for params_i in rotation_params[1:]:
            out[idx : idx + n_middle] = params_i
            idx += n_middle

        n = self.n_right_orbital_rotation_params
        prefix = np.eye(self.norb, dtype=np.complex128)
        for layer_idx, params_i in enumerate(rotation_params):
            chart = (
                self._left_orbital_chart
                if layer_idx == 0
                else self._middle_orbital_chart
            )
            prefix = prefix @ chart.unitary_from_parameters(params_i, self.norb)
        final_eff = _final_unitary_from_left_and_right(
            prefix,
            rotations[-1],
            self.nocc,
            project_reference_ov=self.right_orbital_chart_override is None,
        )
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(final_eff)

        return self._public_parameters_from_native(out)

    def parameters_from_igcr3_ansatz(
        self,
        ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
        *,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            _igcr4_ansatz_from_igcr3_any(
                ansatz,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def parameters_from_igcr2_ansatz(
        self,
        ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            _igcr4_ansatz_from_igcr2_any(
                ansatz,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def parameters_from_t_amplitudes(
        self,
        t2: np.ndarray,
        t1: np.ndarray | None = None,
        **seed_options,
    ) -> np.ndarray:
        """Seed iGCR4 from CCSD amplitudes by non-variational state matching."""
        strategy = seed_options.pop(
            "strategy",
            seed_options.pop("seed_strategy", "ccsd_residual"),
        )
        igcr3_param = IGCR3SpinRestrictedParameterization(
            norb=self.norb,
            nocc=self.nocc,
            layers=self.layers,
            shared_diagonal=self.shared_diagonal,
            interaction_pairs=self.interaction_pairs,
            tau_indices_=self.tau_indices_,
            omega_indices_=self.omega_indices_,
            reduce_cubic_gauge=self.reduce_cubic_gauge,
            left_orbital_chart=self.left_orbital_chart,
            middle_orbital_chart=self.middle_orbital_chart,
            right_orbital_chart_override=self.right_orbital_chart_override,
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
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
        ):
            if key in seed_options and key not in igcr3_options:
                igcr3_options[key] = seed_options[key]
        igcr3_params = igcr3_param.parameters_from_t_amplitudes(
            t2,
            t1=t1,
            strategy=igcr3_strategy,
            **igcr3_options,
        )
        igcr3_ansatz = igcr3_param.ansatz_from_parameters(igcr3_params)
        x_base = self.parameters_from_igcr3_ansatz(
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
                self,
                "quartic",
                ("eta", "rho", "sigma"),
            )
        return _parameters_from_ccsd_residual_seed(
            self,
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
            return_info=seed_options.pop("return_info", False),
        )

    def parameters_from_ucj_ansatz(
        self,
        ansatz: UCJAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_ucj_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def parameters_from_gcr_ansatz(
        self,
        ansatz: GCRAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_gcr_ansatz(
                ansatz,
                self.nocc,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCR4SpinRestrictedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError(
                "previous ansatz nocc does not match this parameterization"
            )
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError(
                    "Pass either orbital_overlap or explicit relabeling, not both."
                )
            basis_change = orbital_transport_unitary_from_overlap(
                orbital_overlap,
                nocc=self.nocc,
                block_diagonal=block_diagonal,
            )
            if isinstance(ansatz, (IGCR4Ansatz, IGCR4LayeredAnsatz)):
                ansatz = transport_igcr4_ansatz_orbitals(ansatz, basis_change)
            elif isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
                ansatz = transport_igcr3_ansatz_orbitals(ansatz, basis_change)
            elif isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
                ansatz = transport_igcr2_ansatz_orbitals(ansatz, basis_change)
            else:
                raise TypeError(
                    f"Unsupported ansatz type for transfer: {type(ansatz)!r}"
                )
        elif old_for_new is not None:
            if isinstance(ansatz, (IGCR4Ansatz, IGCR4LayeredAnsatz)):
                ansatz = relabel_igcr4_ansatz_orbitals(ansatz, old_for_new, phases)
            elif isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
                ansatz = relabel_igcr3_ansatz_orbitals(ansatz, old_for_new, phases)
            elif isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
                ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
            else:
                raise TypeError(
                    f"Unsupported ansatz type for transfer: {type(ansatz)!r}"
                )
        if isinstance(ansatz, (IGCR4Ansatz, IGCR4LayeredAnsatz)):
            return self.parameters_from_ansatz(ansatz)
        if isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
            return self.parameters_from_igcr3_ansatz(ansatz)
        if isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
            return self.parameters_from_igcr2_ansatz(ansatz)
        raise TypeError(f"Unsupported ansatz type for transfer: {type(ansatz)!r}")

    def apply(
        self,
        reference: object,
        nelec: tuple[int, int] | None = None,
    ):
        from dataclasses import replace

        from xquces.gcr.charts import GCR2FullUnitaryChart
        from xquces.gcr.references import (
            apply_ansatz_parameterization,
            reference_is_hartree_fock_state,
        )

        if nelec is None:
            nelec = (self.nocc, self.nocc)
        nelec = tuple(int(x) for x in nelec)
        parameterization = self
        use_full_right = (
            self.right_orbital_chart_override is None
            and not reference_is_hartree_fock_state(reference, self.norb, nelec)
        )
        if use_full_right:
            parameterization = replace(
                self,
                right_orbital_chart_override=GCR2FullUnitaryChart(),
            )
        return apply_ansatz_parameterization(parameterization, reference, nelec)

    def params_to_vec(
        self, reference_vec: np.ndarray, nelec: tuple[int, int]
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec, nelec=nelec, copy=True
            )

        return func


def igcr4_from_igcr3_ansatz(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    return _igcr4_ansatz_from_igcr3_any(
        ansatz,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    )


def igcr4_from_igcr2_ansatz(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    return _igcr4_ansatz_from_igcr2_any(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    )

@dataclass(frozen=True)
class GCRParameterBlock:
    name: str
    start: int
    stop: int
    frozen: bool = False

    @property
    def size(self) -> int:
        return self.stop - self.start

    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class IGCRVariationalCircuit:
    parameterization: object
    reference: object | None = None
    nelec: tuple[int, int] | None = None
    frozen_blocks: tuple[str, ...] = ()
    base_parameters: np.ndarray | None = None

    @property
    def n_params(self) -> int:
        return int(self.parameterization.n_params)

    @property
    def parameter_blocks(self) -> tuple[GCRParameterBlock, ...]:
        return parameter_blocks(self.parameterization, frozen=self.frozen_blocks)

    @property
    def active_mask(self) -> np.ndarray:
        mask = np.ones(self.n_params, dtype=bool)
        for block in self.parameter_blocks:
            if block.frozen:
                mask[block.slice()] = False
        return mask

    @property
    def n_active_params(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    def full_parameters_from_active(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        mask = self.active_mask
        if params.shape == (self.n_params,):
            return np.array(params, copy=True)
        if params.shape != (self.n_active_params,):
            raise ValueError(f"Expected {(self.n_active_params,)} active parameters, got {params.shape}.")
        base = (
            np.zeros(self.n_params, dtype=np.float64)
            if self.base_parameters is None
            else np.asarray(self.base_parameters, dtype=np.float64).copy()
        )
        if base.shape != (self.n_params,):
            raise ValueError(f"base_parameters must have shape {(self.n_params,)}, got {base.shape}.")
        base[mask] = params
        return base

    def active_parameters_from_full(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return params[self.active_mask]

    def ansatz_from_parameters(self, params: np.ndarray):
        return self.parameterization.ansatz_from_parameters(
            self.full_parameters_from_active(params)
        )

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        full_params = self.full_parameters_from_active(params)
        if self.reference is None and hasattr(
            self.parameterization, "state_from_parameters"
        ):
            return self.parameterization.state_from_parameters(full_params)
        if self.reference is None:
            raise ValueError("reference is required to build a state vector")
        if self.nelec is None:
            raise ValueError("nelec is required to build a state vector")
        ansatz = self.parameterization.ansatz_from_parameters(full_params)
        return ansatz.apply(self.reference, self.nelec, copy=True)

    def with_frozen(self, *blocks: str, base_parameters: np.ndarray | None = None):
        return IGCRVariationalCircuit(
            parameterization=self.parameterization,
            reference=self.reference,
            nelec=self.nelec,
            frozen_blocks=tuple(blocks),
            base_parameters=self.base_parameters if base_parameters is None else base_parameters,
        )

    def random_parameters(
        self,
        scale: float = 1e-3,
        *,
        seed: int | np.random.Generator | None = None,
        active_only: bool = False,
    ) -> np.ndarray:
        params = random_parameters(self.parameterization, scale=scale, seed=seed)
        if active_only:
            return self.active_parameters_from_full(params)
        return params

    def parameters_from_t2(
        self,
        t2: np.ndarray,
        *,
        source_order: int | None = None,
        active_only: bool = False,
        **kwargs,
    ) -> np.ndarray:
        params = parameters_from_t2(
            self.parameterization,
            t2,
            source_order=source_order,
            **kwargs,
        )
        if active_only:
            return self.active_parameters_from_full(params)
        return params


def _block_sizes(parameterization: object) -> list[tuple[str, int]]:
    sizes = []
    if hasattr(parameterization, "n_reference_params") and hasattr(
        parameterization, "ansatz_parameterization"
    ):
        n_reference = int(getattr(parameterization, "n_reference_params", 0))
        if n_reference:
            sizes.append(("reference", n_reference))
        sizes.extend(_block_sizes(parameterization.ansatz_parameterization))
        return sizes
    ordered_attrs = [
        ("left", "n_left_orbital_rotation_params"),
        ("pair", "n_pair_params"),
    ]
    if hasattr(parameterization, "n_tau_params"):
        if getattr(parameterization, "uses_reduced_cubic_chart", False):
            ordered_attrs.append(("cubic", "n_tau_params"))
        else:
            ordered_attrs.extend(
                [
                    ("tau", "n_tau_params"),
                    ("omega", "n_omega_params"),
                ]
            )
    if hasattr(parameterization, "n_eta_params"):
        if getattr(parameterization, "uses_reduced_quartic_chart", False):
            ordered_attrs.append(("quartic", "n_rho_params"))
        else:
            ordered_attrs.extend(
                [
                    ("eta", "n_eta_params"),
                    ("rho", "n_rho_params"),
                    ("sigma", "n_sigma_params"),
                ]
            )
    ordered_attrs.extend(
        [
            ("middle", "n_middle_orbital_rotation_params"),
            ("right", "n_right_orbital_rotation_params"),
        ]
    )
    for name, attr in ordered_attrs:
        size = int(getattr(parameterization, attr, 0))
        if size:
            sizes.append((name, size))
    return sizes


def parameter_blocks(
    parameterization: object,
    *,
    frozen: tuple[str, ...] | list[str] | set[str] = (),
) -> tuple[GCRParameterBlock, ...]:
    frozen_set = set(frozen)
    blocks = []
    start = 0
    for name, size in _block_sizes(parameterization):
        stop = start + size
        blocks.append(GCRParameterBlock(name, start, stop, name in frozen_set))
        start = stop
    if start != int(parameterization.n_params):
        raise ValueError(
            "parameter block sizes do not sum to n_params; "
            f"got {start}, expected {parameterization.n_params}"
        )
    return tuple(blocks)


def random_parameters(
    parameterization: object,
    scale: float = 1e-3,
    *,
    seed: int | np.random.Generator | None = None,
    blocks: tuple[str, ...] | list[str] | set[str] | None = None,
) -> np.ndarray:
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    params = rng.normal(0.0, float(scale), int(parameterization.n_params))
    if blocks is not None:
        keep = set(blocks)
        mask = np.zeros(int(parameterization.n_params), dtype=bool)
        for block in parameter_blocks(parameterization):
            if block.name in keep:
                mask[block.slice()] = True
        params = np.where(mask, params, 0.0)
    return params.astype(np.float64, copy=False)


def embed_ansatz_parameters(parameterization: object, ansatz: object) -> np.ndarray:
    if isinstance(ansatz, (IGCR4Ansatz, IGCR4LayeredAnsatz)) and hasattr(
        parameterization, "parameters_from_ansatz"
    ):
        try:
            return parameterization.parameters_from_ansatz(ansatz)
        except TypeError:
            pass
    if isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
        if hasattr(parameterization, "parameters_from_igcr3_ansatz"):
            return parameterization.parameters_from_igcr3_ansatz(ansatz)
        return parameterization.parameters_from_ansatz(ansatz)
    if isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
        if hasattr(parameterization, "parameters_from_igcr2_ansatz"):
            return parameterization.parameters_from_igcr2_ansatz(ansatz)
        return parameterization.parameters_from_ansatz(ansatz)
    return parameterization.parameters_from_ansatz(ansatz)


def parameters_from_t2(
    parameterization: object,
    t2: np.ndarray,
    *,
    source_order: int | None = None,
    **kwargs,
) -> np.ndarray:
    order = int(source_order or getattr(parameterization, "order", 0) or 0)
    if order == 0:
        if isinstance(parameterization, IGCR4SpinRestrictedParameterization):
            order = 4
        elif isinstance(parameterization, IGCR3SpinRestrictedParameterization):
            order = 3
        else:
            order = 2
    if order == 2:
        target = getattr(parameterization, "implementation", parameterization)
        if isinstance(target, IGCR2SpinRestrictedParameterization):
            t1 = kwargs.pop("t1", None)
            return target.parameters_from_t_amplitudes(t2, t1=t1, **kwargs)
        ansatz = IGCR2Ansatz.from_t_restricted(t2, **kwargs)
    elif order == 3:
        target = getattr(parameterization, "implementation", parameterization)
        if isinstance(target, IGCR3SpinRestrictedParameterization):
            t1 = kwargs.pop("t1", None)
            return target.parameters_from_t_amplitudes(t2, t1=t1, **kwargs)
        ansatz = IGCR3Ansatz.from_t_restricted(t2, **kwargs)
    elif order == 4:
        target = getattr(parameterization, "implementation", parameterization)
        if isinstance(target, IGCR4SpinRestrictedParameterization):
            t1 = kwargs.pop("t1", None)
            return target.parameters_from_t_amplitudes(t2, t1=t1, **kwargs)
        ansatz = IGCR4Ansatz.from_t_restricted(t2, **kwargs)
    else:
        raise ValueError("source_order must be 2, 3, or 4")
    return embed_ansatz_parameters(parameterization, ansatz)


_AUTO_RIGHT_CHART = "auto"


@dataclass(frozen=True)
class IGCRSpinRestrictedParameterization:
    """Order-selecting facade for spin-restricted iGCR ansatz parameterizations."""

    norb: int
    nocc: int
    order: int = 2
    layers: int = 1
    shared_diagonal: bool = False
    interaction_pairs: list[tuple[int, int]] | None = None
    tau_indices_: list[tuple[int, int]] | None = None
    omega_indices_: list[tuple[int, int, int]] | None = None
    eta_indices_: list[tuple[int, int]] | None = None
    rho_indices_: list[tuple[int, int, int]] | None = None
    sigma_indices_: list[tuple[int, int, int, int]] | None = None
    reduce_cubic_gauge: bool = True
    reduce_quartic_gauge: bool = True
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    middle_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None | str = _AUTO_RIGHT_CHART
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = None

    def __post_init__(self):
        if self.order not in {2, 3, 4}:
            raise ValueError("order must be 2, 3, or 4")
        if int(self.layers) != self.layers or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        object.__setattr__(self, "layers", int(self.layers))

    @property
    def implementation(self):
        return self._implementation(full_right=False)

    def _implementation(self, *, full_right: bool):
        right_chart = self.right_orbital_chart_override
        if isinstance(right_chart, str) and right_chart == _AUTO_RIGHT_CHART:
            right_chart = IGCR2LeftUnitaryChart() if full_right else None

        common = {
            "norb": self.norb,
            "nocc": self.nocc,
            "interaction_pairs": self.interaction_pairs,
            "left_orbital_chart": self.left_orbital_chart,
            "right_orbital_chart_override": right_chart,
            "real_right_orbital_chart": self.real_right_orbital_chart,
            "left_right_ov_relative_scale": self.left_right_ov_relative_scale,
        }
        if self.order == 2:
            return IGCR2SpinRestrictedParameterization(
                **common,
                layers=self.layers,
                shared_diagonal=self.shared_diagonal,
                middle_orbital_chart=self.middle_orbital_chart,
            )
        if self.order == 3:
            return IGCR3SpinRestrictedParameterization(
                **common,
                layers=self.layers,
                shared_diagonal=self.shared_diagonal,
                middle_orbital_chart=self.middle_orbital_chart,
                tau_indices_=self.tau_indices_,
                omega_indices_=self.omega_indices_,
                reduce_cubic_gauge=self.reduce_cubic_gauge,
            )
        return IGCR4SpinRestrictedParameterization(
            **common,
            layers=self.layers,
            shared_diagonal=self.shared_diagonal,
            middle_orbital_chart=self.middle_orbital_chart,
            tau_indices_=self.tau_indices_,
            omega_indices_=self.omega_indices_,
            eta_indices_=self.eta_indices_,
            rho_indices_=self.rho_indices_,
            sigma_indices_=self.sigma_indices_,
            reduce_cubic_gauge=self.reduce_cubic_gauge,
            reduce_quartic_gauge=self.reduce_quartic_gauge,
        )

    def _uses_full_right_for_reference(
        self,
        reference: object,
        nelec: tuple[int, int],
    ) -> bool:
        from xquces.gcr.references import reference_is_hartree_fock_state

        if not (
            isinstance(self.right_orbital_chart_override, str)
            and self.right_orbital_chart_override == _AUTO_RIGHT_CHART
        ):
            return False
        return not reference_is_hartree_fock_state(reference, self.norb, nelec)

    def apply(
        self,
        reference: object,
        nelec: tuple[int, int] | None = None,
    ):
        if nelec is None:
            nelec = (self.nocc, self.nocc)
        nelec = tuple(int(x) for x in nelec)
        from xquces.gcr.references import apply_ansatz_parameterization

        parameterization = self._implementation(
            full_right=self._uses_full_right_for_reference(reference, nelec)
        )
        return apply_ansatz_parameterization(parameterization, reference, nelec)

    def params_to_vec(
        self, reference_vec: np.ndarray, nelec: tuple[int, int] | None = None
    ):
        return self.apply(reference_vec, nelec).params_to_vec()

    def circuit(
        self,
        reference: object | None = None,
        nelec: tuple[int, int] | None = None,
        *,
        frozen_blocks: tuple[str, ...] | list[str] | set[str] = (),
        base_parameters: np.ndarray | None = None,
    ) -> IGCRVariationalCircuit:
        if nelec is None and reference is not None:
            nelec = (self.nocc, self.nocc)
        full_right = (
            False
            if reference is None or nelec is None
            else self._uses_full_right_for_reference(reference, tuple(nelec))
        )
        return IGCRVariationalCircuit(
            parameterization=self._implementation(full_right=full_right),
            reference=reference,
            nelec=None if nelec is None else tuple(int(x) for x in nelec),
            frozen_blocks=tuple(frozen_blocks),
            base_parameters=base_parameters,
        )

    def parameter_blocks(
        self,
        *,
        frozen: tuple[str, ...] | list[str] | set[str] = (),
    ) -> tuple[GCRParameterBlock, ...]:
        return parameter_blocks(self.implementation, frozen=frozen)

    def random_parameters(
        self,
        scale: float = 1e-3,
        *,
        seed: int | np.random.Generator | None = None,
        blocks: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> np.ndarray:
        return random_parameters(
            self.implementation,
            scale=scale,
            seed=seed,
            blocks=blocks,
        )

    def parameters_from_t2(
        self,
        t2: np.ndarray,
        *,
        source_order: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        return parameters_from_t2(
            self.implementation,
            t2,
            source_order=source_order,
            **kwargs,
        )

    def __getattr__(self, name: str):
        return getattr(self.implementation, name)

__all__ = [
    "GCR2FullUnitaryChart",
    "GCR2TraceFixedFullUnitaryChart",
    "GCRParameterBlock",
    "CCSDResidualSeedInfo",
    "IGCR2Ansatz",
    "IGCR2LayeredAnsatz",
    "IGCR2BlockDiagLeftUnitaryChart",
    "IGCR2LeftUnitaryChart",
    "IGCR2RealReferenceOVUnitaryChart",
    "IGCR2ReferenceOVUnitaryChart",
    "IGCR2SpinBalancedParameterization",
    "IGCR2SpinBalancedSpec",
    "IGCR2SpinRestrictedParameterization",
    "IGCR2SpinRestrictedSpec",
    "IGCR3Ansatz",
    "IGCR3LayeredAnsatz",
    "IGCR3CubicReduction",
    "IGCR3SpinRestrictedParameterization",
    "IGCR3SpinRestrictedSpec",
    "IGCR4Ansatz",
    "IGCR4LayeredAnsatz",
    "IGCR4QuarticReduction",
    "IGCR4SpinRestrictedParameterization",
    "IGCR4SpinRestrictedSpec",
    "IGCRSpinRestrictedParameterization",
    "IGCRVariationalCircuit",
    "apply_igcr3_spin_restricted_diagonal",
    "apply_igcr4_spin_restricted_diagonal",
    "embed_ansatz_parameters",
    "exact_reference_ov_params_from_unitary",
    "exact_reference_ov_unitary",
    "igcr3_from_igcr2_ansatz",
    "igcr4_from_igcr2_ansatz",
    "igcr4_from_igcr3_ansatz",
    "layered_igcr2_from_ucj_t_amplitudes",
    "layered_igcr2_from_ccsd_t_amplitudes",
    "orbital_relabeling_from_overlap",
    "orbital_transport_unitary_from_overlap",
    "parameter_blocks",
    "parameters_from_t2",
    "random_parameters",
    "reduce_spin_balanced",
    "reduce_spin_restricted",
    "relabel_igcr2_ansatz_orbitals",
    "relabel_igcr3_ansatz_orbitals",
    "relabel_igcr4_ansatz_orbitals",
    "spin_restricted_quartic_seed_from_pair_params",
    "spin_restricted_triples_seed_from_pair_params",
    "transport_igcr2_ansatz_orbitals",
    "transport_igcr3_ansatz_orbitals",
    "transport_igcr4_ansatz_orbitals",
]
