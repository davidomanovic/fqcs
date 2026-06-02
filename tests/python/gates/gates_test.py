from __future__ import annotations

import numpy as np
import pytest

from xquces.gates import (
    apply_gcr_spin_restricted,
    apply_igcr2_spin_restricted,
    apply_ucj_spin_restricted,
)
from xquces.states import determinant_state


def test_igcr2_spin_restricted_diagonal_phase_on_determinant():
    norb = 3
    nelec = (2, 1)
    pair_params = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ]
    )
    state = determinant_state(norb, nelec, alpha_occ=(0, 2), beta_occ=(1,))

    out = apply_igcr2_spin_restricted(state, pair_params, norb, nelec)

    phase = np.exp(1j * (pair_params[0, 1] + pair_params[0, 2] + pair_params[1, 2]))
    np.testing.assert_allclose(out, phase * state)


def test_ucj_spin_restricted_zero_parameters_are_identity_and_do_not_mutate_input():
    norb = 4
    nelec = (2, 2)
    state = determinant_state(norb, nelec, alpha_occ=(0, 3), beta_occ=(1, 2))
    original = state.copy()

    out = apply_ucj_spin_restricted(
        state,
        double_params=np.zeros(norb),
        pair_params=np.zeros((norb, norb)),
        norb=norb,
        nelec=nelec,
    )

    np.testing.assert_allclose(out, state)
    np.testing.assert_allclose(state, original)


def test_gcr_spin_restricted_matches_ucj_when_orbital_rotations_are_absent():
    norb = 3
    nelec = (1, 1)
    state = determinant_state(norb, nelec, alpha_occ=(0,), beta_occ=(2,))
    double_params = np.array([0.2, -0.1, 0.3])
    pair_params = np.array(
        [
            [0.0, 0.4, -0.2],
            [0.4, 0.0, 0.5],
            [-0.2, 0.5, 0.0],
        ]
    )

    expected = apply_ucj_spin_restricted(
        state,
        double_params=double_params,
        pair_params=pair_params,
        norb=norb,
        nelec=nelec,
    )
    actual = apply_gcr_spin_restricted(
        state,
        double_params=double_params,
        pair_params=pair_params,
        norb=norb,
        nelec=nelec,
    )

    np.testing.assert_allclose(actual, expected)


def test_ucj_spin_restricted_rejects_bad_parameter_shapes():
    with pytest.raises(ValueError, match="double_params"):
        apply_ucj_spin_restricted(
            np.ones(1),
            double_params=np.zeros(2),
            pair_params=np.zeros((3, 3)),
            norb=3,
            nelec=(0, 0),
        )

    with pytest.raises(ValueError, match="pair_params"):
        apply_ucj_spin_restricted(
            np.ones(1),
            double_params=np.zeros(3),
            pair_params=np.zeros((2, 2)),
            norb=3,
            nelec=(0, 0),
        )
