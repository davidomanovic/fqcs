from __future__ import annotations

import pytest
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, StagedPassManager

import xquces.qiskit as xq_qiskit
from xquces.qiskit.pass_managers import (
    generate_gcr2_pair_uccd_pass_manager,
)


def _backend(connectivity: str) -> GenericBackendV2:
    if connectivity == "heavy-hex":
        coupling_map = CouplingMap.from_heavy_hex(distance=3, bidirectional=True)
    elif connectivity == "square":
        coupling_map = CouplingMap.from_grid(num_rows=4, num_columns=4)
    else:
        raise AssertionError("unsupported test backend")
    return GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        noise_info=False,
    )


def test_generate_gcr2_pass_manager_can_unpack_like_ffsim():
    result = xq_qiskit.generate_gcr2_pass_manager(
        backend=_backend("heavy-hex"),
        norb=4,
        connectivity="heavy-hex",
        optimization_level=1,
    )

    pass_manager, allowed_pairs_ab = result

    assert isinstance(pass_manager, StagedPassManager)
    assert pass_manager is result.pass_manager
    assert allowed_pairs_ab == list(result.layout.allowed_alpha_beta_pairs)
    assert result.layout.connectivity == "heavy-hex"
    assert result.layout.allowed_alpha_beta_pairs == ((0, 0),)
    assert len(result.layout.initial_layout) == 8


def test_generate_gcr2_pass_manager_square_defaults_all_alpha_beta_pairs():
    result = xq_qiskit.generate_gcr2_pass_manager(
        backend=_backend("square"),
        norb=4,
        connectivity="square",
        optimization_level=1,
    )

    assert result.layout.allowed_alpha_beta_pairs == (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    )


def test_pair_uccd_wrapper_accepts_legacy_topology_keyword():
    result = generate_gcr2_pair_uccd_pass_manager(
        backend=_backend("heavy-hex"),
        norb=4,
        nocc=2,
        topology="heavy-hex",
        optimization_level=1,
    )

    assert result.layout.topology == "heavy-hex"
    assert result.layout.allowed_alpha_beta_pairs == ((0, 0),)


def test_generate_gcr2_pass_manager_ignores_controlled_qiskit_kwargs():
    with pytest.warns(UserWarning, match="Argument ``initial_layout`` is ignored."):
        xq_qiskit.generate_gcr2_pass_manager(
            backend=_backend("square"),
            norb=2,
            connectivity="square",
            optimization_level=1,
            initial_layout=[0, 1, 2, 3],
        )


def test_pairuccd_alias_is_exported():
    assert callable(xq_qiskit.generate_gcr2_pairuccd_pass_manager)
    result = xq_qiskit.generate_gcr2_pairuccd_pass_manager(
        backend=_backend("square"),
        norb=2,
        nocc=1,
        connectivity="square",
        optimization_level=1,
    )

    assert result.layout.allowed_alpha_beta_pairs == ((0, 0), (1, 1))
