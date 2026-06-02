"""Fixed-particle-number determinant-sector indexing utilities.

This module owns the small amount of shared bookkeeping needed to move between
flat CI vectors and the alpha/beta occupation-sector matrices used by gates,
Hamiltonians, Jacobians, seeds, and reference-state builders.
"""

from __future__ import annotations

import itertools
import math
from functools import cache

import numpy as np


def _occupation_bitstring(occ: tuple[int, ...]) -> int:
    return sum(1 << int(i) for i in occ)


def occ_rows(norb: int, nocc: int) -> np.ndarray:
    """Return occupied-orbital rows for one spin sector.

    Rows follow PySCF's determinant ordering: increasing occupation bitstring.
    """
    if nocc < 0 or nocc > norb:
        raise ValueError("invalid occupation number")
    if nocc == 0:
        return np.zeros((1, 0), dtype=np.uintp)
    rows = sorted(
        itertools.combinations(range(norb), nocc),
        key=_occupation_bitstring,
    )
    return np.asarray(rows, dtype=np.uintp)


@cache
def occ_indicator_rows(norb: int, nocc: int) -> np.ndarray:
    """Return binary occupation indicators for one spin sector."""
    occ = occ_rows(norb, nocc)
    out = np.zeros((len(occ), norb), dtype=np.uint8)
    if occ.size:
        out[np.arange(len(occ))[:, None], occ] = 1
    return out


def sector_dim(norb: int, nocc: int) -> int:
    """Return ``binom(norb, nocc)`` for one spin sector."""
    return math.comb(norb, nocc)


def sector_shape(norb: int, nelec: tuple[int, int]) -> tuple[int, int]:
    """Return the alpha/beta CI matrix shape for ``(n_alpha, n_beta)``."""
    return sector_dim(norb, nelec[0]), sector_dim(norb, nelec[1])


def reshape_state(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """View a flat fixed-sector state as an alpha-by-beta CI matrix."""
    dim_a, dim_b = sector_shape(norb, nelec)
    arr = np.asarray(vec, dtype=np.complex128)
    if arr.size != dim_a * dim_b:
        raise ValueError("state size does not match norb and nelec")
    return np.ascontiguousarray(arr.reshape(dim_a, dim_b))


def flatten_state(mat: np.ndarray) -> np.ndarray:
    """Flatten an alpha-by-beta CI matrix back into a state vector."""
    return np.asarray(mat, dtype=np.complex128).reshape(-1)
