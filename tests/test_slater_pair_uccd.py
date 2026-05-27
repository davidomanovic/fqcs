from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector

from xquces.basis import sector_shape
from xquces.gcr.product_pair_uccd import (
    ProductPairUCCDStateParameterization,
    SlaterPairUCCDStateParameterization,
    _pair_uccd_ov_pairs,
    slater_pair_orbital_rotation_from_parameters,
    slater_pair_uccd_state_jacobian,
    slater_pair_uccd_state_vjp,
)
from xquces.qiskit.gates.product_pair_uccd import (
    _pair_register_occupied_orbitals,
    _pair_register_orbital_rotation,
)
from xquces.qiskit.gates import product_pair_uccd_stateprep_jw_circuit
from xquces.states import _doci_spatial_basis, _doci_subspace_indices


def _legacy_pair_register_orbital_rotation(
    norb: int,
    nocc: int,
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    orbital_rotation = np.eye(norb, dtype=np.complex128)
    for theta, (i, a) in zip(time * params, _pair_uccd_ov_pairs(norb, nocc)):
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        row_i = np.array(orbital_rotation[i], copy=True)
        row_a = np.array(orbital_rotation[a], copy=True)
        orbital_rotation[i] = c * row_i - s * row_a
        orbital_rotation[a] = s * row_i + c * row_a
    return orbital_rotation


def _state(circuit):
    return Statevector.from_label("0" * circuit.num_qubits).evolve(circuit).data


def _qiskit_state_to_spin_sector(
    state: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    dim_a, dim_b = sector_shape(norb, nelec)
    out = np.zeros(dim_a * dim_b, dtype=np.complex128)
    doci_indices = _doci_subspace_indices(norb, nelec)
    qiskit_indices = []
    for occ in _doci_spatial_basis(norb, nelec[0]):
        index = 0
        for p in occ:
            index |= 1 << p
            index |= 1 << (norb + p)
        qiskit_indices.append(index)
    out[doci_indices] = state[np.asarray(qiskit_indices, dtype=np.intp)]
    mask = np.ones(state.size, dtype=bool)
    mask[np.asarray(qiskit_indices, dtype=np.intp)] = False
    assert np.linalg.norm(state[mask]) < 1e-12
    return out


def _assert_same_state(actual: np.ndarray, expected: np.ndarray, *, atol: float) -> None:
    phase = np.vdot(expected, actual)
    assert abs(abs(phase) - 1.0) < atol
    assert np.allclose(actual, phase * expected, atol=atol)


def test_state_equals_pair_register_slater_circuit():
    rng = np.random.default_rng(12345)
    for norb, nelec in [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (2, 2)),
        (6, (3, 3)),
    ]:
        param = SlaterPairUCCDStateParameterization(norb, nelec)
        params = rng.normal(scale=0.2, size=param.n_params)

        actual = param.state_from_parameters(params)
        circuit_state = _state(
            product_pair_uccd_stateprep_jw_circuit(
                norb,
                nelec,
                params,
                strategy="pair_register_slater",
            )
        )
        expected = _qiskit_state_to_spin_sector(circuit_state, norb, nelec)

        _assert_same_state(actual, expected, atol=1e-11)


def test_pair_register_slater_uses_ov_sized_givens_product():
    rng = np.random.default_rng(8080)
    norb = 6
    nelec = (3, 3)
    params = rng.normal(scale=0.25, size=len(_pair_uccd_ov_pairs(norb, nelec[0])))

    circuit = product_pair_uccd_stateprep_jw_circuit(
        norb,
        nelec,
        params,
        strategy="pair_register_slater",
    )
    ops = circuit.count_ops()

    n_givens = ops.get("pair_register_uccd_givens_jw", 0) + ops.get("xx_plus_yy", 0)
    assert n_givens == len(_pair_uccd_ov_pairs(norb, nelec[0]))
    assert ops.get("cx", 0) == norb
    assert ops.get("x", 0) == nelec[0]


def test_pair_register_slater_reference_cost_no_worse_than_direct():
    norb = 4
    nelec = (2, 2)
    params = np.zeros(len(_pair_uccd_ov_pairs(norb, nelec[0])))
    basis_gates = ["cx", "rz", "sx", "x"]

    slater = transpile(
        product_pair_uccd_stateprep_jw_circuit(
            norb,
            nelec,
            params,
            strategy="pair_register_slater",
        ),
        basis_gates=basis_gates,
        optimization_level=3,
    )
    direct = transpile(
        product_pair_uccd_stateprep_jw_circuit(
            norb,
            nelec,
            params,
            strategy="pair_register_direct",
        ),
        basis_gates=basis_gates,
        optimization_level=3,
    )

    assert slater.depth() <= direct.depth()
    assert slater.count_ops().get("cx", 0) <= direct.count_ops().get("cx", 0)


def test_orbital_rotation_helper_matches_circuit_convention():
    rng = np.random.default_rng(2468)
    norb = 6
    nelec = (3, 3)
    params = rng.normal(scale=0.3, size=nelec[0] * (norb - nelec[0]))

    expected = _legacy_pair_register_orbital_rotation(
        norb,
        nelec[0],
        params,
        time=0.7,
    )
    actual = slater_pair_orbital_rotation_from_parameters(
        norb,
        nelec,
        params,
        time=0.7,
    )
    wrapped = _pair_register_orbital_rotation(norb, nelec[0], params, time=0.7)
    occupied = _pair_register_occupied_orbitals(norb, nelec[0], params, time=0.7)

    assert np.allclose(actual, expected, atol=1e-14)
    assert np.allclose(wrapped, expected, atol=1e-14)
    assert np.allclose(occupied, expected[:, : nelec[0]], atol=1e-14)


def test_slater_pair_jacobian_matches_finite_difference():
    rng = np.random.default_rng(1357)
    norb = 5
    nelec = (2, 2)
    param = SlaterPairUCCDStateParameterization(norb, nelec)
    params = rng.normal(scale=0.15, size=param.n_params)
    eps = 1e-6

    jac = slater_pair_uccd_state_jacobian(norb, nelec, params)
    for k in range(param.n_params):
        shift = np.zeros_like(params)
        shift[k] = eps
        finite_diff = (
            param.state_from_parameters(params + shift)
            - param.state_from_parameters(params - shift)
        ) / (2 * eps)
        assert np.allclose(jac[:, k], finite_diff, atol=1e-8, rtol=1e-8)


def test_slater_pair_vjp_matches_jacobian_contraction():
    rng = np.random.default_rng(97531)
    norb = 5
    nelec = (2, 2)
    param = SlaterPairUCCDStateParameterization(norb, nelec)
    params = rng.normal(scale=0.15, size=param.n_params)
    dim = int(np.prod(sector_shape(norb, nelec)))
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)

    jac = slater_pair_uccd_state_jacobian(norb, nelec, params)
    expected = 2.0 * np.real(jac.conj().T @ v)
    actual = slater_pair_uccd_state_vjp(norb, nelec, params, v)

    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_slater_pair_energy_gradient_matches_finite_difference():
    rng = np.random.default_rng(8642)
    norb = 4
    nelec = (2, 2)
    param = SlaterPairUCCDStateParameterization(norb, nelec)
    params = rng.normal(scale=0.15, size=param.n_params)
    dim = int(np.prod(sector_shape(norb, nelec)))
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    hamiltonian = mat + mat.conj().T

    psi = param.state_from_parameters(params)
    hpsi = hamiltonian @ psi
    energy = float(np.vdot(psi, hpsi).real)
    residual = hpsi - energy * psi
    grad = slater_pair_uccd_state_vjp(norb, nelec, params, residual)

    eps = 1e-6
    finite_diff = np.zeros(param.n_params)
    for k in range(param.n_params):
        shift = np.zeros_like(params)
        shift[k] = eps
        psi_plus = param.state_from_parameters(params + shift)
        psi_minus = param.state_from_parameters(params - shift)
        e_plus = float(np.vdot(psi_plus, hamiltonian @ psi_plus).real)
        e_minus = float(np.vdot(psi_minus, hamiltonian @ psi_minus).real)
        finite_diff[k] = (e_plus - e_minus) / (2 * eps)

    assert np.allclose(grad, finite_diff, atol=1e-7, rtol=1e-7)


def test_slater_pair_differs_from_unsigned_product_pair():
    rng = np.random.default_rng(5555)
    norb = 4
    nelec = (2, 2)
    params = rng.normal(scale=0.3, size=nelec[0] * (norb - nelec[0]))

    product_state = ProductPairUCCDStateParameterization(
        norb,
        nelec,
    ).state_from_parameters(params)
    slater_state = SlaterPairUCCDStateParameterization(
        norb,
        nelec,
    ).state_from_parameters(params)

    assert np.linalg.norm(product_state - slater_state) > 1e-3
