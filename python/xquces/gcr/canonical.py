from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from xquces.gcr.utils import (
    _default_eta_indices,
    _default_pair_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_tau_indices,
    _default_triple_indices,
    _symmetric_matrix_from_values,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.model import SpinRestrictedSpec


@dataclass(frozen=True)
class IGCRDiagonalCoefficients:
    """Canonical coefficients for one spin-restricted iGCR diagonal layer.

    The order is data, not a class name.  Missing higher-order sectors are stored
    as consistently-sized zero arrays so that order-2, order-3, and order-4
    layers can share one representation.
    """

    order: int
    norb: int
    double_params: np.ndarray
    pair_values: np.ndarray
    tau: np.ndarray
    omega_values: np.ndarray
    eta_values: np.ndarray
    rho_values: np.ndarray
    sigma_values: np.ndarray

    SUPPORTED_ORDERS: ClassVar[tuple[int, ...]] = (2, 3, 4)

    def __post_init__(self) -> None:
        order = int(self.order)
        norb = int(self.norb)
        if order not in self.SUPPORTED_ORDERS:
            raise ValueError("order must be 2, 3, or 4")
        if norb < 0:
            raise ValueError("norb must be non-negative")

        double = np.asarray(self.double_params, dtype=np.float64)
        pair = np.asarray(self.pair_values, dtype=np.float64)
        tau = np.asarray(self.tau, dtype=np.float64)
        omega = np.asarray(self.omega_values, dtype=np.float64)
        eta = np.asarray(self.eta_values, dtype=np.float64)
        rho = np.asarray(self.rho_values, dtype=np.float64)
        sigma = np.asarray(self.sigma_values, dtype=np.float64)

        expected = {
            "double_params": (norb,),
            "pair_values": (len(_default_pair_indices(norb)),),
            "tau": (norb, norb),
            "omega_values": (len(_default_triple_indices(norb)),),
            "eta_values": (len(_default_eta_indices(norb)),),
            "rho_values": (len(_default_rho_indices(norb)),),
            "sigma_values": (len(_default_sigma_indices(norb)),),
        }
        actual = {
            "double_params": double.shape,
            "pair_values": pair.shape,
            "tau": tau.shape,
            "omega_values": omega.shape,
            "eta_values": eta.shape,
            "rho_values": rho.shape,
            "sigma_values": sigma.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")

        tau = np.array(tau, copy=True, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)

        if order < 3 and (np.any(tau) or np.any(omega)):
            raise ValueError("order-2 iGCR diagonals cannot contain cubic sectors")
        if order < 4 and (np.any(eta) or np.any(rho) or np.any(sigma)):
            raise ValueError("order-2/3 iGCR diagonals cannot contain quartic sectors")

        object.__setattr__(self, "order", order)
        object.__setattr__(self, "norb", norb)
        object.__setattr__(self, "double_params", np.array(double, copy=True))
        object.__setattr__(self, "pair_values", np.array(pair, copy=True))
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "omega_values", np.array(omega, copy=True))
        object.__setattr__(self, "eta_values", np.array(eta, copy=True))
        object.__setattr__(self, "rho_values", np.array(rho, copy=True))
        object.__setattr__(self, "sigma_values", np.array(sigma, copy=True))

    @property
    def pair_indices(self) -> list[tuple[int, int]]:
        return _default_pair_indices(self.norb)

    @property
    def tau_indices(self) -> list[tuple[int, int]]:
        return _default_tau_indices(self.norb)

    @property
    def omega_indices(self) -> list[tuple[int, int, int]]:
        return _default_triple_indices(self.norb)

    @property
    def eta_indices(self) -> list[tuple[int, int]]:
        return _default_eta_indices(self.norb)

    @property
    def rho_indices(self) -> list[tuple[int, int, int]]:
        return _default_rho_indices(self.norb)

    @property
    def sigma_indices(self) -> list[tuple[int, int, int, int]]:
        return _default_sigma_indices(self.norb)

    @classmethod
    def zeros(cls, norb: int, order: int) -> "IGCRDiagonalCoefficients":
        return cls(
            order=order,
            norb=norb,
            double_params=np.zeros(norb, dtype=np.float64),
            pair_values=np.zeros(len(_default_pair_indices(norb)), dtype=np.float64),
            tau=np.zeros((norb, norb), dtype=np.float64),
            omega_values=np.zeros(len(_default_triple_indices(norb)), dtype=np.float64),
            eta_values=np.zeros(len(_default_eta_indices(norb)), dtype=np.float64),
            rho_values=np.zeros(len(_default_rho_indices(norb)), dtype=np.float64),
            sigma_values=np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64),
        )

    @classmethod
    def from_igcr2_spec(cls, spec: object) -> "IGCRDiagonalCoefficients":
        pair_matrix = np.asarray(spec.to_standard().pair_params, dtype=np.float64)
        norb = int(spec.norb)
        return cls(
            order=2,
            norb=norb,
            double_params=np.zeros(norb, dtype=np.float64),
            pair_values=np.asarray(
                [pair_matrix[p, q] for p, q in _default_pair_indices(norb)],
                dtype=np.float64,
            ),
            tau=np.zeros((norb, norb), dtype=np.float64),
            omega_values=np.zeros(len(_default_triple_indices(norb)), dtype=np.float64),
            eta_values=np.zeros(len(_default_eta_indices(norb)), dtype=np.float64),
            rho_values=np.zeros(len(_default_rho_indices(norb)), dtype=np.float64),
            sigma_values=np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64),
        )

    @classmethod
    def from_igcr3_spec(cls, spec: object) -> "IGCRDiagonalCoefficients":
        norb = int(spec.norb)
        return cls(
            order=3,
            norb=norb,
            double_params=np.asarray(spec.full_double(), dtype=np.float64),
            pair_values=np.asarray(spec.pair_values, dtype=np.float64),
            tau=np.asarray(spec.tau_matrix(), dtype=np.float64),
            omega_values=np.asarray(spec.omega_vector(), dtype=np.float64),
            eta_values=np.zeros(len(_default_eta_indices(norb)), dtype=np.float64),
            rho_values=np.zeros(len(_default_rho_indices(norb)), dtype=np.float64),
            sigma_values=np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64),
        )

    @classmethod
    def from_igcr4_spec(cls, spec: object) -> "IGCRDiagonalCoefficients":
        norb = int(spec.norb)
        return cls(
            order=4,
            norb=norb,
            double_params=np.asarray(spec.full_double(), dtype=np.float64),
            pair_values=np.asarray(spec.pair_values, dtype=np.float64),
            tau=np.asarray(spec.tau_matrix(), dtype=np.float64),
            omega_values=np.asarray(spec.omega_vector(), dtype=np.float64),
            eta_values=np.asarray(spec.eta_vector(), dtype=np.float64),
            rho_values=np.asarray(spec.rho_vector(), dtype=np.float64),
            sigma_values=np.asarray(spec.sigma_vector(), dtype=np.float64),
        )

    @classmethod
    def from_legacy_spec(cls, spec: object) -> "IGCRDiagonalCoefficients":
        if hasattr(spec, "eta_vector"):
            return cls.from_igcr4_spec(spec)
        if hasattr(spec, "omega_vector"):
            return cls.from_igcr3_spec(spec)
        return cls.from_igcr2_spec(spec)

    def full_double(self) -> np.ndarray:
        return np.asarray(self.double_params, dtype=np.float64)

    def pair_matrix(self) -> np.ndarray:
        return _symmetric_matrix_from_values(
            np.asarray(self.pair_values, dtype=np.float64),
            self.norb,
            self.pair_indices,
        )

    def tau_matrix(self) -> np.ndarray:
        tau = np.asarray(self.tau, dtype=np.float64).copy()
        np.fill_diagonal(tau, 0.0)
        return tau

    def omega_vector(self) -> np.ndarray:
        return np.asarray(self.omega_values, dtype=np.float64)

    def eta_vector(self) -> np.ndarray:
        return np.asarray(self.eta_values, dtype=np.float64)

    def rho_vector(self) -> np.ndarray:
        return np.asarray(self.rho_values, dtype=np.float64)

    def sigma_vector(self) -> np.ndarray:
        return np.asarray(self.sigma_values, dtype=np.float64)

    def to_order(self, order: int) -> "IGCRDiagonalCoefficients":
        order = int(order)
        if order < self.order:
            if order < 4 and (
                np.linalg.norm(self.eta_vector()) > 1e-14
                or np.linalg.norm(self.rho_vector()) > 1e-14
                or np.linalg.norm(self.sigma_vector()) > 1e-14
            ):
                raise ValueError("cannot drop nonzero quartic sectors")
            if order < 3 and (
                np.linalg.norm(self.tau_matrix()) > 1e-14
                or np.linalg.norm(self.omega_vector()) > 1e-14
            ):
                raise ValueError("cannot drop nonzero cubic sectors")
        eta = self.eta_vector() if order >= 4 else np.zeros_like(self.eta_vector())
        rho = self.rho_vector() if order >= 4 else np.zeros_like(self.rho_vector())
        sigma = self.sigma_vector() if order >= 4 else np.zeros_like(self.sigma_vector())
        tau = self.tau_matrix() if order >= 3 else np.zeros_like(self.tau_matrix())
        omega = self.omega_vector() if order >= 3 else np.zeros_like(self.omega_vector())
        return IGCRDiagonalCoefficients(
            order=order,
            norb=self.norb,
            double_params=self.full_double(),
            pair_values=self.pair_values,
            tau=tau,
            omega_values=omega,
            eta_values=eta,
            rho_values=rho,
            sigma_values=sigma,
        )

    def to_igcr2_spec(self):
        from xquces.gcr.restricted_model import reduce_spin_restricted

        return reduce_spin_restricted(
            SpinRestrictedSpec(
                double_params=self.full_double(),
                pair_params=self.pair_matrix(),
            )
        )

    def to_igcr3_spec(self):
        from xquces.gcr.restricted_model import IGCR3SpinRestrictedSpec

        d = self.to_order(3)
        return IGCR3SpinRestrictedSpec(
            double_params=d.full_double(),
            pair_values=np.asarray(d.pair_values, dtype=np.float64),
            tau=d.tau_matrix(),
            omega_values=d.omega_vector(),
        )

    def to_igcr4_spec(self):
        from xquces.gcr.restricted_model import IGCR4SpinRestrictedSpec

        d = self.to_order(4)
        return IGCR4SpinRestrictedSpec(
            double_params=d.full_double(),
            pair_values=np.asarray(d.pair_values, dtype=np.float64),
            tau=d.tau_matrix(),
            omega_values=d.omega_vector(),
            eta_values=d.eta_vector(),
            rho_values=d.rho_vector(),
            sigma_values=d.sigma_vector(),
        )

    def to_legacy_spec(self, order: int | None = None):
        order = self.order if order is None else int(order)
        if order == 2:
            return self.to_igcr2_spec()
        if order == 3:
            return self.to_igcr3_spec()
        if order == 4:
            return self.to_igcr4_spec()
        raise ValueError("order must be 2, 3, or 4")


@dataclass(frozen=True)
class IGCRAnsatz:
    """Canonical spin-restricted iGCR ansatz.

    The ansatz is always layered.  A single-layer ansatz is represented as
    ``diagonals=(D,)`` and ``rotations=(left, right)``.
    """

    order: int
    diagonals: tuple[IGCRDiagonalCoefficients, ...]
    rotations: tuple[np.ndarray, ...]
    nocc: int

    def __post_init__(self) -> None:
        order = int(self.order)
        if order not in IGCRDiagonalCoefficients.SUPPORTED_ORDERS:
            raise ValueError("order must be 2, 3, or 4")
        if len(self.diagonals) < 1:
            raise ValueError("at least one diagonal layer is required")
        if len(self.rotations) != len(self.diagonals) + 1:
            raise ValueError("rotations must contain one more entry than diagonals")
        norb = self.diagonals[0].norb
        diagonals: list[IGCRDiagonalCoefficients] = []
        for diagonal in self.diagonals:
            if diagonal.norb != norb:
                raise ValueError("all diagonal layers must have the same norb")
            if diagonal.order > order:
                raise ValueError("all diagonal layer orders must be <= ansatz order")
            diagonals.append(diagonal.to_order(order))
        rotations: list[np.ndarray] = []
        for rotation in self.rotations:
            u = np.asarray(rotation, dtype=np.complex128)
            if u.shape != (norb, norb):
                raise ValueError("rotation has wrong shape")
            if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
                raise ValueError("rotation must be unitary")
            rotations.append(u)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "diagonals", tuple(diagonals))
        object.__setattr__(self, "rotations", tuple(rotations))
        object.__setattr__(self, "nocc", int(self.nocc))

    @property
    def n_layers(self) -> int:
        return len(self.diagonals)

    @property
    def layers(self) -> int:
        return self.n_layers

    @property
    def norb(self) -> int:
        return self.diagonals[0].norb

    @classmethod
    def from_legacy(cls, ansatz: object, *, order: int | None = None) -> "IGCRAnsatz":
        if hasattr(ansatz, "diagonals") and hasattr(ansatz, "rotations"):
            diagonals = tuple(
                IGCRDiagonalCoefficients.from_legacy_spec(d) for d in ansatz.diagonals
            )
            inferred_order = max(d.order for d in diagonals)
            return cls(
                order=inferred_order if order is None else order,
                diagonals=diagonals,
                rotations=tuple(np.asarray(u, dtype=np.complex128) for u in ansatz.rotations),
                nocc=ansatz.nocc,
            )
        diagonal = IGCRDiagonalCoefficients.from_legacy_spec(ansatz.diagonal)
        return cls(
            order=diagonal.order if order is None else order,
            diagonals=(diagonal,),
            rotations=(
                np.asarray(ansatz.left, dtype=np.complex128),
                np.asarray(ansatz.right, dtype=np.complex128),
            ),
            nocc=ansatz.nocc,
        )

    def apply(self, vec, nelec, copy: bool = True):
        from xquces.gcr.restricted_model import (
            apply_igcr3_spin_restricted_diagonal,
            apply_igcr4_spin_restricted_diagonal,
        )
        from xquces.gates import apply_igcr2_spin_restricted

        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.rotations[-1],
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        for idx in range(self.n_layers - 1, -1, -1):
            diagonal = self.diagonals[idx]
            if self.order == 2:
                arr = apply_igcr2_spin_restricted(
                    arr,
                    diagonal.to_igcr2_spec().pair,
                    self.norb,
                    nelec,
                    copy=False,
                )
            elif self.order == 3:
                arr = apply_igcr3_spin_restricted_diagonal(
                    arr,
                    diagonal.to_igcr3_spec(),
                    self.norb,
                    nelec,
                    copy=False,
                )
            elif self.order == 4:
                arr = apply_igcr4_spin_restricted_diagonal(
                    arr,
                    diagonal.to_igcr4_spec(),
                    self.norb,
                    nelec,
                    copy=False,
                )
            else:  # pragma: no cover; guarded in __post_init__
                raise ValueError("unsupported iGCR order")
            arr = apply_orbital_rotation(
                arr,
                self.rotations[idx],
                norb=self.norb,
                nelec=nelec,
                copy=False,
            )
        return arr

    def to_igcr2_ansatz(self):
        from xquces.gcr.restricted_model import IGCR2Ansatz, IGCR2LayeredAnsatz

        if self.n_layers == 1:
            return IGCR2Ansatz(
                diagonal=self.diagonals[0].to_igcr2_spec(),
                left=self.rotations[0],
                right=self.rotations[1],
                nocc=self.nocc,
            )
        return IGCR2LayeredAnsatz(
            diagonals=tuple(d.to_igcr2_spec() for d in self.diagonals),
            rotations=self.rotations,
            nocc=self.nocc,
        )

    def to_igcr3_ansatz(self):
        from xquces.gcr.restricted_model import IGCR3Ansatz, IGCR3LayeredAnsatz

        if self.n_layers == 1:
            return IGCR3Ansatz(
                diagonal=self.diagonals[0].to_igcr3_spec(),
                left=self.rotations[0],
                right=self.rotations[1],
                nocc=self.nocc,
            )
        return IGCR3LayeredAnsatz(
            diagonals=tuple(d.to_igcr3_spec() for d in self.diagonals),
            rotations=self.rotations,
            nocc=self.nocc,
        )

    def to_igcr4_ansatz(self):
        from xquces.gcr.restricted_model import IGCR4Ansatz, IGCR4LayeredAnsatz

        if self.n_layers == 1:
            return IGCR4Ansatz(
                diagonal=self.diagonals[0].to_igcr4_spec(),
                left=self.rotations[0],
                right=self.rotations[1],
                nocc=self.nocc,
            )
        return IGCR4LayeredAnsatz(
            diagonals=tuple(d.to_igcr4_spec() for d in self.diagonals),
            rotations=self.rotations,
            nocc=self.nocc,
        )

    def to_legacy(self):
        if self.order == 2:
            return self.to_igcr2_ansatz()
        if self.order == 3:
            return self.to_igcr3_ansatz()
        if self.order == 4:
            return self.to_igcr4_ansatz()
        raise ValueError("order must be 2, 3, or 4")


def _as_spin_restricted_igcr_ansatz(
    ansatz: IGCRAnsatz | object,
    *,
    order: int,
) -> IGCRAnsatz:
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


def lift_igcr2_to_igcr3(
    ansatz: IGCRAnsatz | object,
    *,
    tau_scale: float = 0.0,
    omega_scale: float = 0.0,
) -> IGCRAnsatz:
    """Lift a canonical spin-restricted iGCR-2 ansatz to iGCR-3."""
    from xquces.seeds.high_order import _triples_seed_from_pair_matrix

    generic = _as_spin_restricted_igcr_ansatz(ansatz, order=2)
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
    from xquces.seeds.high_order import _quartic_seed_from_pair_matrix

    generic = _as_spin_restricted_igcr_ansatz(ansatz, order=3)
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


def scale_igcr_diagonal(
    diagonal: IGCRDiagonalCoefficients,
    scale: float,
) -> IGCRDiagonalCoefficients:
    """Scale all active coefficient sectors of one canonical diagonal layer."""
    return IGCRDiagonalCoefficients(
        order=diagonal.order,
        norb=diagonal.norb,
        double_params=diagonal.full_double() * float(scale),
        pair_values=np.asarray(diagonal.pair_values, dtype=np.float64) * float(scale),
        tau=diagonal.tau_matrix() * float(scale),
        omega_values=diagonal.omega_vector() * float(scale),
        eta_values=diagonal.eta_vector() * float(scale),
        rho_values=diagonal.rho_vector() * float(scale),
        sigma_values=diagonal.sigma_vector() * float(scale),
    )


def as_layered_igcr_ansatz(
    ansatz: IGCRAnsatz | object,
    layers: int,
    *,
    order: int | None = None,
) -> IGCRAnsatz:
    if int(layers) != layers or layers < 1:
        raise ValueError("layers must be a positive integer")
    layers = int(layers)

    generic = ansatz if isinstance(ansatz, IGCRAnsatz) else IGCRAnsatz.from_legacy(ansatz, order=order)
    if order is not None and generic.order != int(order):
        generic = IGCRAnsatz(
            order=int(order),
            diagonals=generic.diagonals,
            rotations=generic.rotations,
            nocc=generic.nocc,
        )

    if generic.n_layers == layers:
        return generic
    if generic.n_layers > layers:
        raise ValueError(
            f"cannot exactly embed an iGCR ansatz with {generic.n_layers} layers "
            f"into {layers} layers"
        )

    identity = np.eye(generic.norb, dtype=np.complex128)
    if generic.n_layers == 1:
        scale = 1.0 / float(layers)
        diagonal = generic.diagonals[0]
        return IGCRAnsatz(
            order=generic.order,
            diagonals=tuple(
                scale_igcr_diagonal(diagonal, scale) for _ in range(layers)
            ),
            rotations=tuple(
                [
                    generic.rotations[0],
                    *[identity for _ in range(layers - 1)],
                    generic.rotations[1],
                ]
            ),
            nocc=generic.nocc,
        )

    diagonals = list(generic.diagonals)
    rotations = list(generic.rotations)
    for _ in range(layers - generic.n_layers):
        diagonals.append(IGCRDiagonalCoefficients.zeros(generic.norb, generic.order))
        rotations.insert(-1, identity)
    return IGCRAnsatz(
        order=generic.order,
        diagonals=tuple(diagonals),
        rotations=tuple(rotations),
        nocc=generic.nocc,
    )

