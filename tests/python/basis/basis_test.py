from __future__ import annotations

import numpy as np
import pytest

from xquces.basis import (
    flatten_state,
    occ_indicator_rows,
    occ_rows,
    reshape_state,
    sector_dim,
    sector_shape,
)


def test_occ_rows_follow_fixed_sector_bitstring_order():
    rows = occ_rows(norb=4, nocc=2)

    np.testing.assert_array_equal(
        rows,
        np.array(
            [
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 3],
                [1, 3],
                [2, 3],
            ],
            dtype=np.uintp,
        ),
    )


def test_occ_indicator_rows_match_occ_rows():
    rows = occ_rows(norb=4, nocc=2)
    indicators = occ_indicator_rows(norb=4, nocc=2)

    assert indicators.shape == (sector_dim(4, 2), 4)
    for row, indicator in zip(rows, indicators):
        expected = np.zeros(4, dtype=np.uint8)
        expected[row] = 1
        np.testing.assert_array_equal(indicator, expected)


def test_sector_shape_and_state_reshape_roundtrip():
    norb = 4
    nelec = (2, 1)
    shape = sector_shape(norb, nelec)
    vec = np.arange(np.prod(shape), dtype=np.float64) + 1j

    mat = reshape_state(vec, norb, nelec)
    assert mat.shape == shape
    np.testing.assert_allclose(flatten_state(mat), vec)


def test_invalid_fixed_sector_inputs_raise():
    with pytest.raises(ValueError, match="invalid occupation"):
        occ_rows(norb=3, nocc=4)

    with pytest.raises(ValueError, match="state size"):
        reshape_state(np.zeros(5), norb=4, nelec=(2, 2))
