from __future__ import annotations

import itertools

import numpy as np

from xquces.gcr.charts import (
    GCR2FullUnitaryChart,
    GCR2TraceFixedFullUnitaryChart,
    IGCR2BlockDiagLeftUnitaryChart,
    IGCR2LeftUnitaryChart,
    IGCR2RealReferenceOVUnitaryChart,
    IGCR2ReferenceOVUnitaryChart,
)
from xquces.orbitals import ov_generator_from_params


def _antihermitian_basis_from_pairs(
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    basis = np.zeros((2 * len(pairs), norb, norb), dtype=np.complex128)
    for k, (p, q) in enumerate(pairs):
        basis[2 * k, p, q] = 1.0
        basis[2 * k, q, p] = -1.0
        basis[2 * k + 1, p, q] = 1j
        basis[2 * k + 1, q, p] = 1j
    return basis


def _full_antihermitian_basis(norb: int) -> np.ndarray:
    pairs = list(itertools.combinations(range(norb), 2))
    basis = np.zeros((norb * norb, norb, norb), dtype=np.complex128)

    idx = 0
    for p in range(norb):
        basis[idx, p, p] = 1j
        idx += 1

    offdiag = _antihermitian_basis_from_pairs(norb, pairs)
    basis[idx:] = offdiag
    return basis


def _trace_fixed_full_antihermitian_basis(norb: int) -> np.ndarray:
    pairs = list(itertools.combinations(range(norb), 2))
    basis = np.zeros((norb * norb - 1, norb, norb), dtype=np.complex128)

    idx = 0
    for p in range(max(0, norb - 1)):
        basis[idx, p, p] = 1j
        basis[idx, norb - 1, norb - 1] = -1j
        idx += 1

    offdiag = _antihermitian_basis_from_pairs(norb, pairs)
    basis[idx:] = offdiag
    return basis


def _left_chart_basis(chart: object, norb: int) -> np.ndarray:
    if isinstance(chart, GCR2FullUnitaryChart):
        return _full_antihermitian_basis(norb)
    if isinstance(chart, GCR2TraceFixedFullUnitaryChart):
        return _trace_fixed_full_antihermitian_basis(norb)
    if isinstance(chart, IGCR2LeftUnitaryChart):
        pairs = list(itertools.combinations(range(norb), 2))
        return _antihermitian_basis_from_pairs(norb, pairs)
    if isinstance(chart, IGCR2BlockDiagLeftUnitaryChart):
        pairs = list(itertools.combinations(range(chart.nocc), 2))
        pairs += [
            (chart.nocc + p, chart.nocc + q)
            for p, q in itertools.combinations(range(chart.nvirt), 2)
        ]
        return _antihermitian_basis_from_pairs(norb, pairs)
    raise NotImplementedError(type(chart).__name__)


def _left_chart_kappa(
    chart: object,
    params: np.ndarray,
    norb: int,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if params.size == 0:
        return np.zeros((norb, norb), dtype=np.complex128)
    if basis is None:
        basis = _left_chart_basis(chart, norb)
    return np.tensordot(params, basis, axes=(0, 0))


def _right_chart_basis(chart: object, norb: int) -> np.ndarray:
    if isinstance(chart, GCR2FullUnitaryChart):
        return _full_antihermitian_basis(norb)
    if isinstance(chart, GCR2TraceFixedFullUnitaryChart):
        return _trace_fixed_full_antihermitian_basis(norb)
    if isinstance(chart, IGCR2ReferenceOVUnitaryChart):
        nocc = chart.nocc
        nvirt = chart.nvirt
        ncomplex = nocc * nvirt
        basis = np.zeros((2 * ncomplex, norb, norb), dtype=np.complex128)
        for a in range(nvirt):
            for i in range(nocc):
                idx = a * nocc + i
                p = nocc + a
                q = i
                basis[idx, p, q] = 1.0
                basis[idx, q, p] = -1.0
                basis[idx + ncomplex, p, q] = 1j
                basis[idx + ncomplex, q, p] = 1j
        return basis
    if isinstance(chart, IGCR2RealReferenceOVUnitaryChart):
        nocc = chart.nocc
        nvirt = chart.nvirt
        nreal = nocc * nvirt
        basis = np.zeros((nreal, norb, norb), dtype=np.complex128)
        for a in range(nvirt):
            for i in range(nocc):
                idx = a * nocc + i
                p = nocc + a
                q = i
                basis[idx, p, q] = 1.0
                basis[idx, q, p] = -1.0
        return basis
    if isinstance(chart, (IGCR2LeftUnitaryChart, IGCR2BlockDiagLeftUnitaryChart)):
        return _left_chart_basis(chart, norb)
    raise NotImplementedError(type(chart).__name__)


def _right_chart_kappa(chart: object, params: np.ndarray, norb: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if isinstance(chart, GCR2FullUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        basis = _full_antihermitian_basis(norb)
        return np.tensordot(params, basis, axes=(0, 0))
    if isinstance(chart, GCR2TraceFixedFullUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        basis = _trace_fixed_full_antihermitian_basis(norb)
        return np.tensordot(params, basis, axes=(0, 0))
    if isinstance(chart, IGCR2ReferenceOVUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        return ov_generator_from_params(params, norb, chart.nocc)
    if isinstance(chart, IGCR2RealReferenceOVUnitaryChart):
        if params.size == 0:
            return np.zeros((norb, norb), dtype=np.complex128)
        full = np.concatenate([params, np.zeros_like(params)])
        return ov_generator_from_params(full, norb, chart.nocc)
    if isinstance(chart, (IGCR2LeftUnitaryChart, IGCR2BlockDiagLeftUnitaryChart)):
        return _left_chart_kappa(chart, params, norb)
    raise NotImplementedError(type(chart).__name__)


def _generator_batch_from_kappa(
    kappa: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    if basis.shape[0] == 0:
        return np.zeros_like(basis)
    herm = -1j * np.asarray(kappa, dtype=np.complex128)
    eigvals, vecs = np.linalg.eigh(herm)
    delta = 1j * (eigvals[:, None] - eigvals[None, :])
    phi = np.ones_like(delta, dtype=np.complex128)
    mask = np.abs(delta) > 1e-12
    phi[mask] = np.expm1(delta[mask]) / delta[mask]
    basis_eig = np.einsum(
        "pa,jpq,qb->jab",
        vecs.conj(),
        basis,
        vecs,
        optimize=True,
    )
    gen_eig = phi[None, :, :] * basis_eig
    return np.einsum(
        "pa,jab,qb->jpq",
        vecs,
        gen_eig,
        vecs.conj(),
        optimize=True,
    )


__all__ = [
    "_antihermitian_basis_from_pairs",
    "_full_antihermitian_basis",
    "_generator_batch_from_kappa",
    "_left_chart_basis",
    "_left_chart_kappa",
    "_right_chart_basis",
    "_right_chart_kappa",
    "_trace_fixed_full_antihermitian_basis",
]
