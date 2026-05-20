from __future__ import annotations

import numpy as np

from xquces.gcr.canonical import IGCRAnsatz, IGCRDiagonalCoefficients


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
                [generic.rotations[0], *[identity for _ in range(layers - 1)], generic.rotations[1]]
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


def to_legacy_layered_igcr_ansatz(
    ansatz: IGCRAnsatz,
    *,
    order: int | None = None,
):
    """Convert a canonical iGCR ansatz to the legacy *layered* class."""

    if order is not None and ansatz.order != int(order):
        ansatz = IGCRAnsatz(
            order=int(order),
            diagonals=ansatz.diagonals,
            rotations=ansatz.rotations,
            nocc=ansatz.nocc,
        )

    if ansatz.order == 2:
        from xquces.gcr.restricted_model import IGCR2LayeredAnsatz

        return IGCR2LayeredAnsatz(
            diagonals=tuple(d.to_igcr2_spec() for d in ansatz.diagonals),
            rotations=ansatz.rotations,
            nocc=ansatz.nocc,
        )
    if ansatz.order == 3:
        from xquces.gcr.restricted_model import IGCR3LayeredAnsatz

        return IGCR3LayeredAnsatz(
            diagonals=tuple(d.to_igcr3_spec() for d in ansatz.diagonals),
            rotations=ansatz.rotations,
            nocc=ansatz.nocc,
        )
    if ansatz.order == 4:
        from xquces.gcr.restricted_model import IGCR4LayeredAnsatz

        return IGCR4LayeredAnsatz(
            diagonals=tuple(d.to_igcr4_spec() for d in ansatz.diagonals),
            rotations=ansatz.rotations,
            nocc=ansatz.nocc,
        )
    raise ValueError("order must be 2, 3, or 4")


def as_legacy_layered_igcr_ansatz(
    ansatz: IGCRAnsatz | object,
    layers: int,
    *,
    order: int,
):
    """Canonical layering adapter returning the legacy layered surface."""

    return to_legacy_layered_igcr_ansatz(
        as_layered_igcr_ansatz(ansatz, layers, order=order),
        order=order,
    )


__all__ = [
    "as_layered_igcr_ansatz",
    "as_legacy_layered_igcr_ansatz",
    "scale_igcr_diagonal",
    "to_legacy_layered_igcr_ansatz",
]
