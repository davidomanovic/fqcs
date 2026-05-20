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
from xquces.gates import (
    apply_gcr_spin_balanced,
    apply_gcr_spin_restricted,
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
from xquces.gcr.restricted_core import (
    SpinRestrictedLayeredDiagonalParameterizationCore,
)
from xquces.gcr.restricted_model import (
    IGCR2Ansatz,
    IGCR2LayeredAnsatz,
    IGCR2SpinRestrictedSpec,
    IGCR3Ansatz,
    IGCR3LayeredAnsatz,
    IGCR3SpinRestrictedSpec,
    IGCR4Ansatz,
    IGCR4LayeredAnsatz,
    IGCR4SpinRestrictedSpec,
    apply_igcr3_spin_restricted_diagonal,
    apply_igcr4_spin_restricted_diagonal,
    reduce_spin_restricted,
    spin_restricted_quartic_seed_from_pair_params,
    spin_restricted_triples_seed_from_pair_params,
)
from xquces.gcr.canonical import IGCRAnsatz, IGCRDiagonalCoefficients
from xquces.gcr.canonical_layering import as_legacy_layered_igcr_ansatz
from xquces.gcr.canonical_lift import (
    lift_igcr2_to_igcr3,
    lift_igcr2_to_igcr4,
    lift_igcr3_to_igcr4,
)
from xquces.gcr.canonical_transform import (
    relabel_igcr_ansatz_orbitals,
    relabel_legacy_igcr_ansatz_orbitals,
    transport_igcr_ansatz_orbitals,
    transport_legacy_igcr_ansatz_orbitals,
)
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


def relabel_igcr2_ansatz_orbitals(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR2Ansatz | IGCR2LayeredAnsatz:
    return relabel_legacy_igcr_ansatz_orbitals(
        ansatz,
        old_for_new,
        phases,
        order=2,
    )




def transport_igcr2_ansatz_orbitals(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz, basis_change: np.ndarray
) -> IGCR2Ansatz | IGCR2LayeredAnsatz:
    return transport_legacy_igcr_ansatz_orbitals(
        ansatz,
        basis_change,
        order=2,
    )




def _as_layered_igcr2_spin_restricted_ansatz(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    layers: int,
) -> IGCR2LayeredAnsatz:
    return as_legacy_layered_igcr_ansatz(ansatz, layers, order=2)


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



_AUTO_RIGHT_CHART = "auto"


@dataclass(frozen=True)
class RestrictedIGCRDiagonalAdapter:
    """Order-specific diagonal chart adapter for canonical spin-restricted iGCR."""

    order: int
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
        if self.order not in {2, 3, 4}:
            raise ValueError("order must be 2, 3, or 4")

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
    def diagonal_chart(self):
        if self.order == 2:
            return RestrictedPairChart(
                norb=self.norb,
                nocc=self.nocc,
                interaction_pairs=self.interaction_pairs,
            )
        if self.order == 3:
            return RestrictedCubicChart(
                norb=self.norb,
                nocc=self.nocc,
                interaction_pairs=self.interaction_pairs,
                tau_indices_=self.tau_indices_,
                omega_indices_=self.omega_indices_,
                reduce_cubic_gauge=self.reduce_cubic_gauge,
            )
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
        return bool(
            self.order >= 3
            and getattr(self.diagonal_chart, "uses_reduced_cubic_chart", False)
        )

    @property
    def uses_reduced_quartic_chart(self) -> bool:
        return bool(
            self.order >= 4
            and getattr(self.diagonal_chart, "uses_reduced_quartic_chart", False)
        )

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        if self.order < 3:
            raise AttributeError("order-2 iGCR has no cubic reduction")
        return self.diagonal_chart.cubic_reduction

    @property
    def quartic_reduction(self) -> IGCR4QuarticReduction:
        if self.order < 4:
            raise AttributeError("order-2/3 iGCR has no quartic reduction")
        return self.diagonal_chart.quartic_reduction

    @property
    def n_pair_params_per_layer(self) -> int:
        return int(self.diagonal_chart.n_pair_params if self.order >= 3 else self.diagonal_chart.n_params)

    @property
    def n_tau_params_per_layer(self) -> int:
        return int(self.diagonal_chart.n_tau_params) if self.order >= 3 else 0

    @property
    def n_omega_params_per_layer(self) -> int:
        return int(self.diagonal_chart.n_omega_params) if self.order >= 3 else 0

    @property
    def n_eta_params_per_layer(self) -> int:
        return int(self.diagonal_chart.n_eta_params) if self.order >= 4 else 0

    @property
    def n_rho_params_per_layer(self) -> int:
        return int(self.diagonal_chart.n_rho_params) if self.order >= 4 else 0

    @property
    def n_sigma_params_per_layer(self) -> int:
        return int(self.diagonal_chart.n_sigma_params) if self.order >= 4 else 0

    @property
    def n_diag_params_per_layer(self) -> int:
        return int(self.diagonal_chart.n_params)

    def diagonal_from_native_parameters(
        self,
        params: np.ndarray,
    ) -> IGCRDiagonalCoefficients:
        coeffs = self.diagonal_chart.coefficients_from_parameters(params)
        if self.order == 2:
            return IGCRDiagonalCoefficients.from_igcr2_spec(
                IGCR2SpinRestrictedSpec(pair=coeffs.pair)
            )
        if self.order == 3:
            return IGCRDiagonalCoefficients.from_igcr3_spec(
                IGCR3SpinRestrictedSpec(
                    double_params=coeffs.double_params,
                    pair_values=coeffs.pair_values,
                    tau=coeffs.tau,
                    omega_values=coeffs.omega_values,
                )
            )
        return IGCRDiagonalCoefficients.from_igcr4_spec(
            IGCR4SpinRestrictedSpec(
                double_params=coeffs.double_params,
                pair_values=coeffs.pair_values,
                tau=coeffs.tau,
                omega_values=coeffs.omega_values,
                eta_values=coeffs.eta_values,
                rho_values=coeffs.rho_values,
                sigma_values=coeffs.sigma_values,
            )
        )

    def native_parameters_from_diagonal(
        self,
        diagonal: IGCRDiagonalCoefficients,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.order == 2:
            spec = diagonal.to_igcr2_spec()
            return self.diagonal_chart.parameters_from_coefficients(
                RestrictedPairCoefficients(pair=np.asarray(spec.pair, dtype=np.float64))
            )
        if self.order == 3:
            spec = diagonal.to_igcr3_spec()
            return self.diagonal_chart.parameters_from_coefficients(
                RestrictedCubicCoefficients(
                    double_params=spec.full_double(),
                    pair_values=np.asarray(spec.pair_values, dtype=np.float64),
                    tau=spec.tau_matrix(),
                    omega_values=spec.omega_vector(),
                )
            )
        spec = diagonal.to_igcr4_spec()
        return self.diagonal_chart.parameters_from_coefficients(
            RestrictedQuarticCoefficients(
                double_params=spec.full_double(),
                pair_values=np.asarray(spec.pair_values, dtype=np.float64),
                tau=spec.tau_matrix(),
                omega_values=spec.omega_vector(),
                eta_values=spec.eta_vector(),
                rho_values=spec.rho_vector(),
                sigma_values=spec.sigma_vector(),
            )
        )

    def sector_sizes(self, owner: "IGCRSpinRestrictedParameterization") -> dict[str, int]:
        if self.order == 2:
            return {
                "left": owner.n_left_orbital_rotation_params,
                "pair": owner.n_pair_params,
                "middle": owner.n_middle_orbital_rotation_params,
                "right": owner.n_right_orbital_rotation_params,
                "total": owner.n_params,
            }
        if self.order == 3:
            return {
                "left": owner.n_left_orbital_rotation_params,
                "double": owner.n_double_params,
                "pair": owner.n_pair_params,
                "tau": 0 if owner.uses_reduced_cubic_chart else owner.n_tau_params,
                "omega": owner.n_omega_params,
                "cubic": owner.n_tau_params
                if owner.uses_reduced_cubic_chart
                else (owner.n_tau_params + owner.n_omega_params),
                "middle": owner.n_middle_orbital_rotation_params,
                "right": owner.n_right_orbital_rotation_params,
                "total": owner.n_params,
            }
        return {
            "left": owner.n_left_orbital_rotation_params,
            "pair": owner.n_pair_params,
            "tau": 0 if owner.uses_reduced_cubic_chart else owner.n_tau_params,
            "omega": owner.n_omega_params,
            "eta": 0 if owner.uses_reduced_quartic_chart else owner.n_eta_params,
            "rho": owner.n_rho_params,
            "sigma": 0 if owner.uses_reduced_quartic_chart else owner.n_sigma_params,
            "middle": owner.n_middle_orbital_rotation_params,
            "right": owner.n_right_orbital_rotation_params,
            "total": owner.n_params,
        }


@dataclass(frozen=True)
class IGCRSpinRestrictedParameterization:
    """Canonical spin-restricted iGCR parameterization with order as data."""

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
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        if int(self.layers) != self.layers or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        object.__setattr__(self, "layers", int(self.layers))
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        if self.order >= 3:
            _validate_ordered_pairs(self.tau_indices_, self.norb)
            _validate_triples(self.omega_indices_, self.norb)
        if self.order >= 4:
            _validate_pairs(self.eta_indices_, self.norb, allow_diagonal=False)
            _validate_rho_indices(self.rho_indices_, self.norb)
            _validate_sigma_indices(self.sigma_indices_, self.norb)
        if self.left_right_ov_relative_scale is not None and (
            not np.isfinite(float(self.left_right_ov_relative_scale))
            or self.left_right_ov_relative_scale <= 0
        ):
            raise ValueError("left_right_ov_relative_scale must be positive or None")

    @property
    def diagonal_adapter(self) -> RestrictedIGCRDiagonalAdapter:
        return RestrictedIGCRDiagonalAdapter(
            order=self.order,
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
    def pair_indices(self):
        return self.diagonal_adapter.pair_indices

    @property
    def tau_indices(self):
        return self.diagonal_adapter.tau_indices

    @property
    def omega_indices(self):
        return self.diagonal_adapter.omega_indices

    @property
    def eta_indices(self):
        return self.diagonal_adapter.eta_indices

    @property
    def rho_indices(self):
        return self.diagonal_adapter.rho_indices

    @property
    def sigma_indices(self):
        return self.diagonal_adapter.sigma_indices

    @property
    def diagonal_chart(self):
        return self.diagonal_adapter.diagonal_chart

    @property
    def uses_reduced_cubic_chart(self) -> bool:
        return self.diagonal_adapter.uses_reduced_cubic_chart

    @property
    def uses_reduced_quartic_chart(self) -> bool:
        return self.diagonal_adapter.uses_reduced_quartic_chart

    @property
    def cubic_reduction(self) -> IGCR3CubicReduction:
        return self.diagonal_adapter.cubic_reduction

    @property
    def quartic_reduction(self) -> IGCR4QuarticReduction:
        return self.diagonal_adapter.quartic_reduction

    @property
    def _right_override_is_auto(self) -> bool:
        return self.right_orbital_chart_override is None or (
            isinstance(self.right_orbital_chart_override, str)
            and self.right_orbital_chart_override == _AUTO_RIGHT_CHART
        )

    @property
    def right_orbital_chart(self):
        if not self._right_override_is_auto:
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
        return self.order >= 3

    @property
    def _project_final_reference_ov(self) -> bool:
        return self.order >= 3 and self._right_override_is_auto

    @property
    def _layered_core(self) -> SpinRestrictedLayeredDiagonalParameterizationCore:
        adapter = self.diagonal_adapter
        return SpinRestrictedLayeredDiagonalParameterizationCore(
            order=self.order,
            norb=self.norb,
            nocc=self.nocc,
            layers=self.layers,
            shared_diagonal=self.shared_diagonal,
            left_orbital_chart=self._left_orbital_chart,
            middle_orbital_chart=self._middle_orbital_chart,
            right_orbital_chart=self.right_orbital_chart,
            n_diag_params_per_layer=adapter.n_diag_params_per_layer,
            diagonal_from_parameters=adapter.diagonal_from_native_parameters,
            parameters_from_diagonal=adapter.native_parameters_from_diagonal,
            right_depends_on_prefix=self._right_depends_on_prefix,
            project_final_reference_ov=self._project_final_reference_ov,
            left_right_ov_transform_scale=self._left_right_ov_transform_scale,
        )

    @property
    def n_left_orbital_rotation_params(self):
        return self._layered_core.n_left_orbital_rotation_params

    @property
    def n_middle_orbital_rotation_params_per_layer(self):
        return self._layered_core.n_middle_orbital_rotation_params_per_layer

    @property
    def n_middle_orbital_rotation_params(self):
        return self._layered_core.n_middle_orbital_rotation_params

    @property
    def n_double_params(self):
        return 0

    def _scaled_layer_count(self, per_layer: int) -> int:
        return int(per_layer if self.shared_diagonal else self.layers * per_layer)

    @property
    def n_pair_params_per_layer(self):
        return self.diagonal_adapter.n_pair_params_per_layer

    @property
    def n_pair_params(self):
        return self._scaled_layer_count(self.n_pair_params_per_layer)

    @property
    def n_tau_params_per_layer(self):
        return self.diagonal_adapter.n_tau_params_per_layer

    @property
    def n_tau_params(self):
        return self._scaled_layer_count(self.n_tau_params_per_layer)

    @property
    def n_omega_params_per_layer(self):
        return self.diagonal_adapter.n_omega_params_per_layer

    @property
    def n_omega_params(self):
        return self._scaled_layer_count(self.n_omega_params_per_layer)

    @property
    def n_eta_params_per_layer(self):
        return self.diagonal_adapter.n_eta_params_per_layer

    @property
    def n_eta_params(self):
        return self._scaled_layer_count(self.n_eta_params_per_layer)

    @property
    def n_rho_params_per_layer(self):
        return self.diagonal_adapter.n_rho_params_per_layer

    @property
    def n_rho_params(self):
        return self._scaled_layer_count(self.n_rho_params_per_layer)

    @property
    def n_sigma_params_per_layer(self):
        return self.diagonal_adapter.n_sigma_params_per_layer

    @property
    def n_sigma_params(self):
        return self._scaled_layer_count(self.n_sigma_params_per_layer)

    @property
    def n_diag_params_per_layer(self):
        return self._layered_core.n_diag_params_per_layer

    @property
    def n_right_orbital_rotation_params(self):
        return self._layered_core.n_right_orbital_rotation_params

    @property
    def _right_orbital_rotation_start(self):
        return self._layered_core._right_orbital_rotation_start

    @property
    def _middle_orbital_rotation_start(self):
        return self._layered_core._middle_orbital_rotation_start

    @property
    def _left_right_ov_transform_scale(self):
        if self.order >= 3:
            return None
        return _left_right_ov_transform_scale_for(
            self.right_orbital_chart,
            self.left_right_ov_relative_scale,
        )

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return self._layered_core.native_parameters_from_public(params)

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return self._layered_core.public_parameters_from_native(params)

    @property
    def n_params(self):
        return self._layered_core.n_params

    def sector_sizes(self) -> dict[str, int]:
        return self.diagonal_adapter.sector_sizes(self)

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCRAnsatz:
        return self._layered_core.ansatz_from_parameters(params)

    def parameters_from_ansatz(self, ansatz: IGCRAnsatz | object) -> np.ndarray:
        return self._layered_core.parameters_from_ansatz(ansatz)

    def parameters_from_igcr2_ansatz(
        self,
        ansatz: IGCRAnsatz | IGCR2Ansatz | IGCR2LayeredAnsatz,
        *,
        tau_scale: float = 0.0,
        omega_scale: float = 0.0,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        generic = ansatz if isinstance(ansatz, IGCRAnsatz) else ansatz.to_generic()
        if generic.order != 2:
            raise TypeError("expected an iGCR-2 ansatz")
        legacy = generic.to_igcr2_ansatz()
        if self.order == 2:
            return self.parameters_from_ansatz(generic)
        if self.order == 3:
            return self.parameters_from_ansatz(
                _igcr3_ansatz_from_igcr2_any(
                    legacy,
                    tau_scale=tau_scale,
                    omega_scale=omega_scale,
                )
            )
        return self.parameters_from_ansatz(
            _igcr4_ansatz_from_igcr2_any(
                legacy,
                tau_scale=tau_scale,
                omega_scale=omega_scale,
                eta_scale=eta_scale,
                rho_scale=rho_scale,
                sigma_scale=sigma_scale,
            )
        )

    def parameters_from_igcr3_ansatz(
        self,
        ansatz: IGCRAnsatz | IGCR3Ansatz | IGCR3LayeredAnsatz,
        *,
        eta_scale: float = 0.0,
        rho_scale: float = 0.0,
        sigma_scale: float = 0.0,
    ) -> np.ndarray:
        generic = ansatz if isinstance(ansatz, IGCRAnsatz) else ansatz.to_generic()
        if generic.order != 3:
            raise TypeError("expected an iGCR-3 ansatz")
        if self.order == 3:
            return self.parameters_from_ansatz(generic)
        if self.order != 4:
            raise TypeError("cannot embed an iGCR-3 ansatz into iGCR-2")
        return self.parameters_from_ansatz(
            _igcr4_ansatz_from_igcr3_any(
                generic.to_igcr3_ansatz(),
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
        if self.order == 2:
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
        if self.order == 3:
            from xquces.seeds.high_order import igcr3_parameters_from_t_amplitudes

            return igcr3_parameters_from_t_amplitudes(self, t2, t1=t1, **seed_options)
        from xquces.seeds.high_order import igcr4_parameters_from_t_amplitudes

        return igcr4_parameters_from_t_amplitudes(self, t2, t1=t1, **seed_options)

    def parameters_from_ucj_t_amplitudes(
        self,
        t2: np.ndarray,
        t1: np.ndarray | None = None,
        **df_options,
    ) -> np.ndarray:
        if self.order != 2:
            raise ValueError("UCJ-lift t-amplitude seeding is only defined for iGCR2")
        ansatz = layered_igcr2_from_ucj_t_amplitudes(
            t2, t1=t1, layers=self.layers, nocc=self.nocc, **df_options
        )
        return self.parameters_from_ansatz(ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz, **scales) -> np.ndarray:
        if self.order == 2:
            seeded = _igcr2_layered_spin_restricted_ansatz_from_ucj(
                ansatz,
                self.nocc,
                self.layers,
            )
            return self.parameters_from_ansatz(seeded)
        if self.order == 3:
            return self.parameters_from_ansatz(
                IGCR3Ansatz.from_ucj_ansatz(ansatz, self.nocc, **scales)
            )
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_ucj_ansatz(ansatz, self.nocc, **scales)
        )

    def parameters_from_gcr_ansatz(self, ansatz: GCRAnsatz, **scales) -> np.ndarray:
        if self.order == 2:
            return self.parameters_from_ansatz(
                IGCR2Ansatz.from_gcr_ansatz(ansatz, self.nocc)
            )
        if self.order == 3:
            return self.parameters_from_ansatz(
                IGCR3Ansatz.from_gcr_ansatz(ansatz, self.nocc, **scales)
            )
        return self.parameters_from_ansatz(
            IGCR4Ansatz.from_gcr_ansatz(ansatz, self.nocc, **scales)
        )

    def _canonical_from_transfer_ansatz(self, ansatz: object) -> IGCRAnsatz:
        if isinstance(ansatz, IGCRAnsatz):
            return ansatz
        if getattr(ansatz, "is_spin_restricted", True) is False:
            raise TypeError("expected a spin-restricted iGCR ansatz")
        return ansatz.to_generic() if hasattr(ansatz, "to_generic") else IGCRAnsatz.from_legacy(ansatz)

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "IGCRSpinRestrictedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        generic = self._canonical_from_transfer_ansatz(ansatz)
        if generic.nocc != self.nocc:
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
            generic = transport_igcr_ansatz_orbitals(generic, basis_change)
        elif old_for_new is not None:
            generic = relabel_igcr_ansatz_orbitals(generic, old_for_new, phases)

        if generic.order == self.order:
            return self.parameters_from_ansatz(generic)
        if generic.order == 2:
            return self.parameters_from_igcr2_ansatz(generic)
        if generic.order == 3:
            return self.parameters_from_igcr3_ansatz(generic)
        raise TypeError(f"Unsupported ansatz order for transfer: {generic.order!r}")

    def _uses_full_right_for_reference(
        self,
        reference: object,
        nelec: tuple[int, int],
    ) -> bool:
        from xquces.gcr.references import reference_is_hartree_fock_state

        if not self._right_override_is_auto:
            return False
        return not reference_is_hartree_fock_state(reference, self.norb, nelec)

    def apply(
        self,
        reference: object,
        nelec: tuple[int, int] | None = None,
    ):
        from dataclasses import replace

        from xquces.gcr.charts import GCR2FullUnitaryChart
        from xquces.gcr.references import apply_ansatz_parameterization

        if nelec is None:
            nelec = (self.nocc, self.nocc)
        nelec = tuple(int(x) for x in nelec)
        parameterization = self
        if self._uses_full_right_for_reference(reference, nelec):
            parameterization = replace(
                self,
                right_orbital_chart_override=GCR2FullUnitaryChart(),
            )
        return apply_ansatz_parameterization(parameterization, reference, nelec)

    def params_to_vec(
        self, reference_vec: np.ndarray, nelec: tuple[int, int] | None = None
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        if nelec is None:
            nelec = (self.nocc, self.nocc)
        nelec = tuple(int(x) for x in nelec)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec, nelec=nelec, copy=True
            )

        return func

    def circuit(
        self,
        reference: object | None = None,
        nelec: tuple[int, int] | None = None,
        *,
        frozen_blocks: tuple[str, ...] | list[str] | set[str] = (),
        base_parameters: np.ndarray | None = None,
    ) -> IGCRVariationalCircuit:
        from dataclasses import replace

        from xquces.gcr.charts import GCR2FullUnitaryChart

        parameterization = self
        if reference is not None:
            if nelec is None:
                nelec = (self.nocc, self.nocc)
            nelec = tuple(int(x) for x in nelec)
            if self._uses_full_right_for_reference(reference, nelec):
                parameterization = replace(
                    self,
                    right_orbital_chart_override=GCR2FullUnitaryChart(),
                )
        return IGCRVariationalCircuit(
            parameterization=parameterization,
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
        return parameter_blocks(self, frozen=frozen)

    def parameter_view(self, params: np.ndarray, *, copy: bool = False) -> ParameterView:
        return parameter_view(self, params, copy=copy)

    def random_parameters(
        self,
        scale: float = 1e-3,
        *,
        seed: int | np.random.Generator | None = None,
        blocks: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> np.ndarray:
        return random_parameters(
            self,
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
            self,
            t2,
            source_order=source_order,
            **kwargs,
        )


@dataclass(frozen=True)
class IGCR2SpinRestrictedParameterization(IGCRSpinRestrictedParameterization):
    """Compatibility wrapper for canonical order-2 spin-restricted iGCR."""

    order: int = field(default=2, init=False)
    right_orbital_chart_override: object | None | str = None
    left_right_ov_relative_scale: float | None = 1.0

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR2Ansatz | IGCR2LayeredAnsatz:
        return super().ansatz_from_parameters(params).to_legacy()




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





def _as_layered_igcr3_spin_restricted_ansatz(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    layers: int,
) -> IGCR3LayeredAnsatz:
    return as_legacy_layered_igcr_ansatz(ansatz, layers, order=3)


def _igcr3_ansatz_from_igcr2_any(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    return lift_igcr2_to_igcr3(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
    ).to_legacy()




def relabel_igcr3_ansatz_orbitals(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    return relabel_legacy_igcr_ansatz_orbitals(
        ansatz,
        old_for_new,
        phases,
        order=3,
    )


def transport_igcr3_ansatz_orbitals(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz, basis_change: np.ndarray
) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
    return transport_legacy_igcr_ansatz_orbitals(
        ansatz,
        basis_change,
        order=3,
    )



@dataclass(frozen=True)
class IGCR3SpinRestrictedParameterization(IGCRSpinRestrictedParameterization):
    """Compatibility wrapper for canonical order-3 spin-restricted iGCR."""

    order: int = field(default=3, init=False)
    right_orbital_chart_override: object | None | str = None
    left_right_ov_relative_scale: float | None = 3.0

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR3Ansatz | IGCR3LayeredAnsatz:
        return super().ansatz_from_parameters(params).to_legacy()




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





def _as_layered_igcr4_spin_restricted_ansatz(
    ansatz: IGCR4Ansatz | IGCR4LayeredAnsatz,
    layers: int,
) -> IGCR4LayeredAnsatz:
    return as_legacy_layered_igcr_ansatz(ansatz, layers, order=4)


def _igcr4_ansatz_from_igcr3_any(
    ansatz: IGCR3Ansatz | IGCR3LayeredAnsatz,
    *,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    return lift_igcr3_to_igcr4(
        ansatz,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    ).to_legacy()


def _igcr4_ansatz_from_igcr2_any(
    ansatz: IGCR2Ansatz | IGCR2LayeredAnsatz,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
    eta_scale: float = 0.0,
    rho_scale: float = 0.0,
    sigma_scale: float = 0.0,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    return lift_igcr2_to_igcr4(
        ansatz,
        tau_scale=tau_scale,
        omega_scale=omega_scale,
        eta_scale=eta_scale,
        rho_scale=rho_scale,
        sigma_scale=sigma_scale,
    ).to_legacy()




def relabel_igcr4_ansatz_orbitals(
    ansatz: IGCR4Ansatz | IGCR4LayeredAnsatz,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    return relabel_legacy_igcr_ansatz_orbitals(
        ansatz,
        old_for_new,
        phases,
        order=4,
    )


def transport_igcr4_ansatz_orbitals(
    ansatz: IGCR4Ansatz | IGCR4LayeredAnsatz, basis_change: np.ndarray
) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
    return transport_legacy_igcr_ansatz_orbitals(
        ansatz,
        basis_change,
        order=4,
    )



@dataclass(frozen=True)
class IGCR4SpinRestrictedParameterization(IGCRSpinRestrictedParameterization):
    """Compatibility wrapper for canonical order-4 spin-restricted iGCR."""

    order: int = field(default=4, init=False)
    right_orbital_chart_override: object | None | str = None
    left_right_ov_relative_scale: float | None = 3.0

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4Ansatz | IGCR4LayeredAnsatz:
        return super().ansatz_from_parameters(params).to_legacy()




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
