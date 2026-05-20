from __future__ import annotations

import numpy as np

from xquces.gcr.canonical import IGCRAnsatz, IGCRDiagonalCoefficients
from xquces.gcr.utils import (
    _default_eta_indices,
    _default_pair_indices,
    _default_rho_indices,
    _default_sigma_indices,
    _default_triple_indices,
    _orbital_relabeling_unitary,
)


def _values_from_symmetric_matrix(
    matrix: np.ndarray,
    indices: list[tuple[int, int]],
) -> np.ndarray:
    return np.asarray([matrix[p, q] for p, q in indices], dtype=np.float64)


def _as_spin_restricted_igcr_ansatz(
    ansatz: IGCRAnsatz | object,
    *,
    order: int | None = None,
) -> IGCRAnsatz:
    if isinstance(ansatz, IGCRAnsatz):
        if order is not None and ansatz.order != int(order):
            return IGCRAnsatz(
                order=int(order),
                diagonals=ansatz.diagonals,
                rotations=ansatz.rotations,
                nocc=ansatz.nocc,
            )
        return ansatz
    if getattr(ansatz, "is_spin_restricted", True) is False:
        raise TypeError("expected a spin-restricted iGCR ansatz")
    return IGCRAnsatz.from_legacy(ansatz, order=order)


def relabel_igcr_diagonal_coefficients(
    diagonal: IGCRDiagonalCoefficients,
    old_for_new: np.ndarray,
) -> IGCRDiagonalCoefficients:
    """Relabel one canonical spin-restricted iGCR diagonal layer.

    ``old_for_new[p]`` gives the old orbital index occupying new orbital index
    ``p``.  The transformation is purely an index relabeling of diagonal
    coefficient tensors; orbital rotation matrices are handled by
    ``relabel_igcr_ansatz_orbitals``.
    """

    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if old_for_new.shape != (diagonal.norb,):
        raise ValueError("orbital permutation length must match diagonal.norb")
    if sorted(old_for_new.tolist()) != list(range(diagonal.norb)):
        raise ValueError("old_for_new must be a permutation of orbital indices")

    norb = diagonal.norb
    double = diagonal.full_double()[old_for_new]
    pair = diagonal.pair_matrix()[np.ix_(old_for_new, old_for_new)]
    tau = diagonal.tau_matrix()[np.ix_(old_for_new, old_for_new)]

    omega_old = {
        idx: value for idx, value in zip(diagonal.omega_indices, diagonal.omega_vector())
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

    eta_old = {idx: value for idx, value in zip(diagonal.eta_indices, diagonal.eta_vector())}
    eta_values = np.asarray(
        [
            eta_old[
                tuple(sorted((int(old_for_new[p]), int(old_for_new[q]))))
            ]
            for p, q in _default_eta_indices(norb)
        ],
        dtype=np.float64,
    )

    rho_old = {idx: value for idx, value in zip(diagonal.rho_indices, diagonal.rho_vector())}
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

    sigma_old = {
        idx: value for idx, value in zip(diagonal.sigma_indices, diagonal.sigma_vector())
    }
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

    if diagonal.order < 3:
        tau = np.zeros((norb, norb), dtype=np.float64)
        omega_values = np.zeros(len(_default_triple_indices(norb)), dtype=np.float64)
    if diagonal.order < 4:
        eta_values = np.zeros(len(_default_eta_indices(norb)), dtype=np.float64)
        rho_values = np.zeros(len(_default_rho_indices(norb)), dtype=np.float64)
        sigma_values = np.zeros(len(_default_sigma_indices(norb)), dtype=np.float64)

    return IGCRDiagonalCoefficients(
        order=diagonal.order,
        norb=norb,
        double_params=double,
        pair_values=_values_from_symmetric_matrix(pair, _default_pair_indices(norb)),
        tau=tau,
        omega_values=omega_values,
        eta_values=eta_values,
        rho_values=rho_values,
        sigma_values=sigma_values,
    )


def relabel_igcr_ansatz_orbitals(
    ansatz: IGCRAnsatz | object,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
    *,
    order: int | None = None,
) -> IGCRAnsatz:
    """Relabel the orbital basis of a canonical spin-restricted iGCR ansatz."""

    generic = _as_spin_restricted_igcr_ansatz(ansatz, order=order)
    if generic.norb != len(old_for_new):
        raise ValueError("orbital permutation length must match ansatz.norb")
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    relabel = _orbital_relabeling_unitary(old_for_new, phases)
    return IGCRAnsatz(
        order=generic.order,
        diagonals=tuple(
            relabel_igcr_diagonal_coefficients(diagonal, old_for_new)
            for diagonal in generic.diagonals
        ),
        rotations=tuple(
            relabel.conj().T @ rotation @ relabel
            for rotation in generic.rotations
        ),
        nocc=generic.nocc,
    )


def transport_igcr_ansatz_orbitals(
    ansatz: IGCRAnsatz | object,
    basis_change: np.ndarray,
    *,
    order: int | None = None,
) -> IGCRAnsatz:
    """Transport the left orbital frame of a canonical spin-restricted iGCR ansatz."""

    generic = _as_spin_restricted_igcr_ansatz(ansatz, order=order)
    basis_change = np.asarray(basis_change, dtype=np.complex128)
    if basis_change.shape != (generic.norb, generic.norb):
        raise ValueError(
            f"basis_change must have shape {(generic.norb, generic.norb)}, "
            f"got {basis_change.shape}."
        )
    if not np.allclose(
        basis_change.conj().T @ basis_change,
        np.eye(generic.norb),
        atol=1e-10,
    ):
        raise ValueError("basis_change must be unitary")
    rotations = list(generic.rotations)
    rotations[0] = basis_change.conj().T @ rotations[0]
    return IGCRAnsatz(
        order=generic.order,
        diagonals=generic.diagonals,
        rotations=tuple(rotations),
        nocc=generic.nocc,
    )


def relabel_legacy_igcr_ansatz_orbitals(
    ansatz: object,
    old_for_new: np.ndarray,
    phases: np.ndarray | None = None,
    *,
    order: int,
):
    return relabel_igcr_ansatz_orbitals(
        ansatz,
        old_for_new,
        phases,
        order=order,
    ).to_legacy()


def transport_legacy_igcr_ansatz_orbitals(
    ansatz: object,
    basis_change: np.ndarray,
    *,
    order: int,
):
    return transport_igcr_ansatz_orbitals(
        ansatz,
        basis_change,
        order=order,
    ).to_legacy()

