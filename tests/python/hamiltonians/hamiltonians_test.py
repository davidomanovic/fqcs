from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import scipy.linalg

from xquces.hamiltonians import CanonicalTransformedHamiltonianLinearOperator


@dataclass(frozen=True)
class _DenseHamiltonian:
    matrix: np.ndarray
    ecore: float = 0.7
    norb: int = 2
    nelec: tuple[int, int] = (1, 1)

    def dense_electronic_matrix(self) -> np.ndarray:
        return np.array(self.matrix, dtype=np.complex128, copy=True)

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(vec, dtype=np.complex128)

    def expectation(self, vec: np.ndarray) -> float:
        arr = np.asarray(vec, dtype=np.complex128)
        return float(np.vdot(arr, self.matrix @ arr).real + self.ecore)


def _random_unitary(dim: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(mat)
    phases = np.diag(r)
    phases = phases / np.abs(phases)
    return q * phases.conj()


def test_canonical_transformed_hamiltonian_from_unitary_matches_dense_formula():
    h_matrix = np.array(
        [
            [1.0, 0.2, 0.0, 0.1j],
            [0.2, -0.4, 0.3, 0.0],
            [0.0, 0.3, 0.7, -0.2],
            [-0.1j, 0.0, -0.2, 0.5],
        ],
        dtype=np.complex128,
    )
    base = _DenseHamiltonian(h_matrix)
    unitary = _random_unitary(4)

    transformed = CanonicalTransformedHamiltonianLinearOperator.from_unitary(
        base,
        unitary,
    )

    expected = unitary.conj().T @ h_matrix @ unitary
    expected = 0.5 * (expected + expected.conj().T)
    np.testing.assert_allclose(transformed.dense_electronic_matrix(), expected)

    vec = np.array([0.3, -0.4j, 0.2, 0.1j], dtype=np.complex128)
    np.testing.assert_allclose(transformed.matvec(vec), expected @ vec)
    assert transformed.expectation(vec) == pytest.approx(
        float(np.vdot(vec, expected @ vec).real + base.ecore)
    )


def test_canonical_transformed_hamiltonian_from_generator_validates_antihermitian():
    base = _DenseHamiltonian(np.eye(4, dtype=np.complex128))

    with pytest.raises(ValueError, match="anti-Hermitian"):
        CanonicalTransformedHamiltonianLinearOperator.from_generator(
            base,
            np.eye(4, dtype=np.complex128),
        )

    generator = np.zeros((4, 4), dtype=np.complex128)
    generator[0, 1] = 0.3
    generator[1, 0] = -0.3
    transformed = CanonicalTransformedHamiltonianLinearOperator.from_generator(
        base,
        generator,
    )
    np.testing.assert_allclose(transformed.unitary, scipy.linalg.expm(generator))


def test_canonical_transformed_hamiltonian_from_dense_matrix_symmetrizes_input():
    base = _DenseHamiltonian(np.eye(4, dtype=np.complex128))
    h_matrix = np.array(
        [
            [1.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="shape"):
        CanonicalTransformedHamiltonianLinearOperator.from_dense_matrix(base, h_matrix)

    full = np.eye(4, dtype=np.complex128)
    full[0, 1] = 2.0
    transformed = CanonicalTransformedHamiltonianLinearOperator.from_dense_matrix(
        base,
        full,
    )
    np.testing.assert_allclose(
        transformed.dense_electronic_matrix(),
        0.5 * (full + full.conj().T),
    )
