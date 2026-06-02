from __future__ import annotations

import numpy as np
import pytest

from xquces.basis import sector_shape
from xquces.states import (
    determinant_index,
    determinant_state,
    doci_amplitudes_from_state,
    doci_params_from_state,
    doci_state,
    doci_state_jacobian,
    hartree_fock_state,
    linear_combination_state,
)


def test_determinant_state_places_amplitude_at_determinant_index():
    norb = 4
    nelec = (2, 1)
    alpha_occ = (0, 3)
    beta_occ = (2,)

    state = determinant_state(norb, nelec, alpha_occ, beta_occ)
    index = determinant_index(norb, nelec, alpha_occ, beta_occ)

    assert state.shape == (int(np.prod(sector_shape(norb, nelec))),)
    assert np.count_nonzero(state) == 1
    assert state[index] == 1.0


def test_hartree_fock_state_is_first_determinant():
    state = hartree_fock_state(norb=5, nelec=(2, 3))

    assert state[0] == 1.0
    assert np.count_nonzero(state) == 1
    np.testing.assert_allclose(np.linalg.norm(state), 1.0)


def test_linear_combination_state_normalizes_and_rejects_zero_state():
    state = linear_combination_state(
        norb=3,
        nelec=(1, 1),
        terms=[
            (2.0, (0,), (0,)),
            (1.0j, (1,), (2,)),
        ],
    )

    np.testing.assert_allclose(np.linalg.norm(state), 1.0)

    with pytest.raises(ValueError, match="zero norm"):
        linear_combination_state(2, (1, 1), [(0.0, (0,), (0,))])


def test_doci_state_parameters_roundtrip_and_jacobian_matches_finite_difference():
    norb = 4
    nelec = (2, 2)
    params = np.array([0.2, 0.3, 0.4, 0.1, 0.2])

    state = doci_state(norb, nelec, params=params)
    np.testing.assert_allclose(doci_params_from_state(state, norb, nelec), params)
    amplitudes = doci_amplitudes_from_state(state, norb, nelec)
    np.testing.assert_allclose(
        doci_state(norb, nelec, amplitudes=amplitudes),
        state,
    )

    jac = doci_state_jacobian(norb, nelec, params)
    eps = 1.0e-6
    for j in range(params.size):
        step = np.zeros_like(params)
        step[j] = eps
        finite_difference = (
            doci_state(norb, nelec, params=params + step)
            - doci_state(norb, nelec, params=params - step)
        ) / (2 * eps)
        np.testing.assert_allclose(jac[:, j], finite_difference, atol=1.0e-8)
