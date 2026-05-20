from __future__ import annotations

from functools import cache

import numpy as np

from xquces.basis import occ_rows


@cache
def _bitstrings(norb: int, nocc: int) -> np.ndarray:
    occ = occ_rows(norb, nocc)
    bits = np.zeros(len(occ), dtype=np.uint64)
    for i, row in enumerate(occ):
        value = 0
        for p in row:
            value |= 1 << int(p)
        bits[i] = value
    return bits


@cache
def _one_body_tensor(norb: int, nocc: int) -> np.ndarray:
    bits = _bitstrings(norb, nocc)
    dim = len(bits)
    index = {int(bit): i for i, bit in enumerate(bits)}
    tensor = np.zeros((norb, norb, dim, dim), dtype=np.complex128)
    for col, bit in enumerate(bits):
        det = int(bit)
        for q in range(norb):
            if ((det >> q) & 1) == 0:
                continue
            sign1 = -1.0 if ((det & ((1 << q) - 1)).bit_count() & 1) else 1.0
            det1 = det ^ (1 << q)
            for p in range(norb):
                if ((det1 >> p) & 1) != 0:
                    continue
                sign2 = (
                    -1.0 if ((det1 & ((1 << p) - 1)).bit_count() & 1) else 1.0
                )
                row = index[det1 | (1 << p)]
                tensor[p, q, row, col] += sign1 * sign2
    return tensor


@cache
def _sector_rep_index(norb: int, nocc: int) -> np.ndarray:
    occ = occ_rows(norb, nocc)
    if nocc == 0:
        return np.zeros((2, 1, 1, 0, 0), dtype=np.int64)
    dim = len(occ)
    rows = np.broadcast_to(occ[:, None, :, None], (dim, dim, nocc, nocc))
    cols = np.broadcast_to(occ[None, :, None, :], (dim, dim, nocc, nocc))
    return np.stack([rows, cols], axis=0)


def _sector_representation(u: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    if nocc == 0:
        return np.ones((1, 1), dtype=np.complex128)
    index = _sector_rep_index(norb, nocc)
    submats = u[index[0], index[1]]
    return np.linalg.det(submats)


def _one_body_batch_to_sector(g_batch: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    if g_batch.shape[0] == 0:
        dim = tensor.shape[-1]
        return np.zeros((0, dim, dim), dtype=np.complex128)
    return np.einsum("jpq,pqmn->jmn", g_batch, tensor, optimize=True)


def _apply_batch_transform(
    left: np.ndarray,
    right: np.ndarray,
    mats: np.ndarray,
) -> np.ndarray:
    if mats.shape[0] == 0:
        return mats
    tmp = np.einsum("am,jmb->jab", left, mats, optimize=True)
    return np.einsum("jab,cb->jac", tmp, right, optimize=True)


def _batch_row_and_col(
    left_batch: np.ndarray,
    right_batch: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    if left_batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    row = np.einsum("jmn,nb->jmb", left_batch, mat, optimize=True)
    col = np.einsum("an,jbn->jab", mat, right_batch, optimize=True)
    return row + col


def _batch_left_multiply(batch: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    return np.einsum("jmn,nb->jmb", batch, mat, optimize=True)


def _batch_right_transpose_multiply(batch: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros((0,) + mat.shape, dtype=np.complex128)
    return np.einsum("an,jbn->jab", mat, batch, optimize=True)


__all__ = [
    "_apply_batch_transform",
    "_batch_left_multiply",
    "_batch_right_transpose_multiply",
    "_batch_row_and_col",
    "_bitstrings",
    "_one_body_batch_to_sector",
    "_one_body_tensor",
    "_sector_rep_index",
    "_sector_representation",
]
