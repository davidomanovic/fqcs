from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import ffsim
import numpy as np

from xquces.ansatz.blocks import (
    _block_kind,
    _block_shape,
    _block_sizes,
    _block_specs,
    _layered_block_shape,
    parameter_blocks as _ansatz_parameter_blocks,
    parameter_view as _ansatz_parameter_view,
    random_parameters as _ansatz_random_parameters,
)
from xquces.ansatz.parameters import ParameterBlock, ParameterView
from xquces.charts.diagonal import (
    RestrictedPairChart,
    RestrictedPairCoefficients,
    RestrictedCubicChart,
    RestrictedCubicCoefficients,
    RestrictedQuarticChart,
    RestrictedQuarticCoefficients,
)
from xquces.charts.reductions import IGCR3CubicReduction, IGCR4QuarticReduction
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
from xquces.seeds.residual import CCSDResidualSeedInfo
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
    from xquces.seeds.ucj import _igcr2_layered_spin_restricted_ansatz_from_ucj as impl

    return impl(ansatz, nocc, layers)


def layered_igcr2_from_ucj_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    layers: int = 1,
    nocc: int | None = None,
    **df_options,
) -> "IGCR2Ansatz | IGCR2LayeredAnsatz":
    """Build an iGCR-2 ansatz by lifting ffsim's UCJ t-amplitude seed."""
    from xquces.seeds.ucj import layered_igcr2_from_ucj_t_amplitudes as impl

    return impl(t2, t1=t1, layers=layers, nocc=nocc, **df_options)


def layered_igcr2_from_ccsd_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    layers: int = 1,
    nocc: int | None = None,
    **df_options,
) -> "IGCR2Ansatz | IGCR2LayeredAnsatz":
    """Compatibility alias for the UCJ-lift t-amplitude seed."""
    from xquces.seeds.ucj import layered_igcr2_from_ccsd_t_amplitudes as impl

    return impl(t2, t1=t1, layers=layers, nocc=nocc, **df_options)


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
    from xquces.seeds.native_igcr2 import native_igcr2_seed_from_ccsd_t_amplitudes

    return native_igcr2_seed_from_ccsd_t_amplitudes(
        parameterization,
        t2,
        t1=t1,
        right_mixing_eps=right_mixing_eps,
        target_scales=target_scales,
        j_mixing_scales=j_mixing_scales,
        j_damping=j_damping,
        left_damping=left_damping,
        max_soft=max_soft,
        cond_j_max=cond_j_max,
        cond_s_max=cond_s_max,
        hamiltonian=hamiltonian,
        verbose=verbose,
    )


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
    def diagonal_chart(self) -> RestrictedPairChart:
        return RestrictedPairChart(
            norb=self.norb,
            nocc=self.nocc,
            interaction_pairs=self.interaction_pairs,
        )

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
            return self.n_pair_params_per_layer
        return self.layers * self.n_pair_params_per_layer

    @property
    def n_pair_params_per_layer(self):
        return self.diagonal_chart.n_params

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
        view = parameter_view(self, params)
        left = self._left_orbital_chart.unitary_from_parameters(
            view.flat("left"), self.norb
        )
        n_pair = self.n_pair_params_per_layer
        if self.shared_diagonal:
            pair = self.diagonal_chart.coefficients_from_parameters(
                view.flat("pair")
            ).pair
            pairs = [pair.copy() for _ in range(self.layers)]
        else:
            pair_values_by_layer = view.flat("pair").reshape(self.layers, n_pair)
            pairs = [
                self.diagonal_chart.coefficients_from_parameters(pair_values).pair
                for pair_values in pair_values_by_layer
            ]
        middle_rotations = []
        n_middle = self.n_middle_orbital_rotation_params_per_layer
        middle_values = (
            view.flat("middle").reshape(self.layers - 1, n_middle)
            if self.layers > 1
            else np.zeros((0, n_middle), dtype=np.float64)
        )
        for middle_params in middle_values:
            middle_rotations.append(
                self._middle_orbital_chart.unitary_from_parameters(
                    middle_params, self.norb
                )
            )
        final = self.right_orbital_chart.unitary_from_parameters(
            view.flat("right"), self.norb
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
        view = parameter_view(self, out)
        view.set("left", rotation_params[0])
        n_pair = self.n_pair_params_per_layer
        if self.shared_diagonal:
            pair_eff = np.mean(np.stack(pair_mats, axis=0), axis=0)
            pair_params, _ = self.diagonal_chart.parameters_from_coefficients(
                RestrictedPairCoefficients(pair=pair_eff)
            )
            view.set("pair", pair_params)
        else:
            pair_values = []
            for pair_eff in pair_mats:
                pair_params, _ = self.diagonal_chart.parameters_from_coefficients(
                    RestrictedPairCoefficients(pair=pair_eff)
                )
                pair_values.append(pair_params)
            pair_values = np.asarray(pair_values, dtype=np.float64)
            view.set("pair", pair_values.reshape(view.block("pair").shape))

        n_middle = self.n_middle_orbital_rotation_params_per_layer
        if self.layers > 1:
            view.set(
                "middle",
                np.asarray(rotation_params[1:], dtype=np.float64).reshape(
                    self.layers - 1, n_middle
                ),
            )

        view.set(
            "right",
            self.right_orbital_chart.parameters_from_unitary(rotations[-1]),
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
        view = parameter_view(self, params)
        left = self._left_orbital_chart.unitary_from_parameters(
            view.flat("left"), self.norb
        )
        same_diag = np.asarray(view["same_diag"], dtype=np.float64)
        double = np.asarray(view["double"], dtype=np.float64)
        same = _symmetric_matrix_from_values(
            view.flat("same_spin"),
            self.norb,
            self.same_spin_indices,
        )
        mixed = _symmetric_matrix_from_values(
            view.flat("mixed_spin"),
            self.norb,
            self.mixed_spin_indices,
        )
        final = self.right_orbital_chart.unitary_from_parameters(
            view.flat("right"), self.norb
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
        view = parameter_view(self, out)
        view.set("left", left_params)
        view.set("same_diag", same_diag)
        view.set("double", mixed_double)
        view.set("same_spin", same_full)
        view.set("mixed_spin", mixed_full)
        view.set("right", self.right_orbital_chart.parameters_from_unitary(right_eff))
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
    def diagonal_chart(self) -> RestrictedCubicChart:
        return RestrictedCubicChart(
            norb=self.norb,
            nocc=self.nocc,
            interaction_pairs=self.interaction_pairs,
            tau_indices_=self.tau_indices_,
            omega_indices_=self.omega_indices_,
            reduce_cubic_gauge=self.reduce_cubic_gauge,
        )

    @property
    def uses_reduced_cubic_chart(self) -> bool:
        return self.diagonal_chart.uses_reduced_cubic_chart

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        return self.diagonal_chart.cubic_reduction

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
        return self.diagonal_chart.n_pair_params

    @property
    def n_tau_params(self):
        if self.shared_diagonal:
            return self.n_tau_params_per_layer
        return self.layers * self.n_tau_params_per_layer

    @property
    def n_tau_params_per_layer(self):
        return self.diagonal_chart.n_tau_params

    @property
    def n_omega_params(self):
        if self.shared_diagonal:
            return self.n_omega_params_per_layer
        return self.layers * self.n_omega_params_per_layer

    @property
    def n_omega_params_per_layer(self):
        return self.diagonal_chart.n_omega_params

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
        coeffs = self.diagonal_chart.coefficients_from_parameters(params)
        return IGCR3SpinRestrictedSpec(
            double_params=coeffs.double_params,
            pair_values=coeffs.pair_values,
            tau=coeffs.tau,
            omega_values=coeffs.omega_values,
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
        return self.diagonal_chart.parameters_from_coefficients(
            RestrictedCubicCoefficients(
                double_params=diagonal.full_double(),
                pair_values=np.asarray(diagonal.pair_values, dtype=np.float64),
                tau=diagonal.tau_matrix(),
                omega_values=diagonal.omega_vector(),
            )
        )

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
        from xquces.seeds.high_order import igcr3_parameters_from_t_amplitudes

        return igcr3_parameters_from_t_amplitudes(self, t2, t1=t1, **seed_options)

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
    def diagonal_chart(self) -> RestrictedQuarticChart:
        return RestrictedQuarticChart(
            norb=self.norb,
            nocc=self.nocc,
            interaction_pairs=self.interaction_pairs,
            tau_indices_=self.tau_indices_,
            omega_indices_=self.omega_indices_,
            eta_indices_=self.eta_indices_,
            rho_indices_=self.rho_indices_,
            sigma_indices_=self.sigma_indices_,
            reduce_cubic_gauge=self.reduce_cubic_gauge,
            reduce_quartic_gauge=self.reduce_quartic_gauge,
        )

    @property
    def uses_reduced_cubic_chart(self) -> bool:
        return self.diagonal_chart.uses_reduced_cubic_chart

    @property
    def uses_reduced_quartic_chart(self) -> bool:
        return self.diagonal_chart.uses_reduced_quartic_chart

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        return self.diagonal_chart.cubic_reduction

    @property
    def quartic_reduction(self) -> IGCR4QuarticReduction:
        return self.diagonal_chart.quartic_reduction

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
        return self.diagonal_chart.n_pair_params

    @property
    def n_tau_params(self):
        if self.shared_diagonal:
            return self.n_tau_params_per_layer
        return self.layers * self.n_tau_params_per_layer

    @property
    def n_tau_params_per_layer(self):
        return self.diagonal_chart.n_tau_params

    @property
    def n_omega_params(self):
        if self.shared_diagonal:
            return self.n_omega_params_per_layer
        return self.layers * self.n_omega_params_per_layer

    @property
    def n_omega_params_per_layer(self):
        return self.diagonal_chart.n_omega_params

    @property
    def n_eta_params(self):
        if self.shared_diagonal:
            return self.n_eta_params_per_layer
        return self.layers * self.n_eta_params_per_layer

    @property
    def n_eta_params_per_layer(self):
        return self.diagonal_chart.n_eta_params

    @property
    def n_rho_params(self):
        if self.shared_diagonal:
            return self.n_rho_params_per_layer
        return self.layers * self.n_rho_params_per_layer

    @property
    def n_rho_params_per_layer(self):
        return self.diagonal_chart.n_rho_params

    @property
    def n_sigma_params(self):
        if self.shared_diagonal:
            return self.n_sigma_params_per_layer
        return self.layers * self.n_sigma_params_per_layer

    @property
    def n_sigma_params_per_layer(self):
        return self.diagonal_chart.n_sigma_params

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
        coeffs = self.diagonal_chart.coefficients_from_parameters(params)
        return IGCR4SpinRestrictedSpec(
            double_params=coeffs.double_params,
            pair_values=coeffs.pair_values,
            tau=coeffs.tau,
            omega_values=coeffs.omega_values,
            eta_values=coeffs.eta_values,
            rho_values=coeffs.rho_values,
            sigma_values=coeffs.sigma_values,
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
        return self.diagonal_chart.parameters_from_coefficients(
            RestrictedQuarticCoefficients(
                double_params=diagonal.full_double(),
                pair_values=np.asarray(diagonal.pair_values, dtype=np.float64),
                tau=diagonal.tau_matrix(),
                omega_values=diagonal.omega_vector(),
                eta_values=diagonal.eta_vector(),
                rho_values=diagonal.rho_vector(),
                sigma_values=diagonal.sigma_vector(),
            )
        )

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
        from xquces.seeds.high_order import igcr4_parameters_from_t_amplitudes

        return igcr4_parameters_from_t_amplitudes(self, t2, t1=t1, **seed_options)

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

GCRParameterBlock = ParameterBlock


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

    def parameter_view(self, params: np.ndarray, *, copy: bool = False) -> ParameterView:
        return parameter_view(
            self.parameterization,
            self.full_parameters_from_active(params),
            frozen=self.frozen_blocks,
            copy=copy,
        )

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
        from xquces.seeds.dispatch import parameters_from_t2 as dispatch_parameters_from_t2

        params = dispatch_parameters_from_t2(
            self.parameterization,
            t2,
            source_order=source_order,
            **kwargs,
        )
        if active_only:
            return self.active_parameters_from_full(params)
        return params


def parameter_blocks(
    parameterization: object,
    *,
    frozen: tuple[str, ...] | list[str] | set[str] = (),
) -> tuple[GCRParameterBlock, ...]:
    return _ansatz_parameter_blocks(parameterization, frozen=frozen)


def parameter_view(
    parameterization: object,
    params: np.ndarray,
    *,
    frozen: tuple[str, ...] | list[str] | set[str] = (),
    copy: bool = False,
) -> ParameterView:
    return _ansatz_parameter_view(
        parameterization,
        params,
        frozen=frozen,
        copy=copy,
    )


def random_parameters(
    parameterization: object,
    scale: float = 1e-3,
    *,
    seed: int | np.random.Generator | None = None,
    blocks: tuple[str, ...] | list[str] | set[str] | None = None,
) -> np.ndarray:
    return _ansatz_random_parameters(
        parameterization,
        scale=scale,
        seed=seed,
        blocks=blocks,
    )


def embed_ansatz_parameters(parameterization: object, ansatz: object) -> np.ndarray:
    from xquces.seeds.dispatch import embed_ansatz_parameters as dispatch_embed_ansatz_parameters

    return dispatch_embed_ansatz_parameters(parameterization, ansatz)


def parameters_from_t2(
    parameterization: object,
    t2: np.ndarray,
    *,
    source_order: int | None = None,
    **kwargs,
) -> np.ndarray:
    from xquces.seeds.dispatch import parameters_from_t2 as dispatch_parameters_from_t2

    return dispatch_parameters_from_t2(
        parameterization,
        t2,
        source_order=source_order,
        **kwargs,
    )


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

    def parameter_view(self, params: np.ndarray, *, copy: bool = False) -> ParameterView:
        return parameter_view(self.implementation, params, copy=copy)

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
        from xquces.seeds.dispatch import parameters_from_t2 as dispatch_parameters_from_t2

        return dispatch_parameters_from_t2(
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
    "ParameterBlock",
    "ParameterView",
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
    "parameter_view",
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
