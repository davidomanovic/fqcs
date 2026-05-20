from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from xquces.ansatz.parameters import (
    ParameterBlock,
    ParameterView,
    parameter_view as _parameter_view,
)


def _block_kind(name: str) -> str:
    if name == "reference":
        return "reference"
    if name in {"left", "middle", "right"}:
        return "orbital"
    if name in {
        "pair",
        "same_diag",
        "double",
        "same_spin",
        "mixed_spin",
        "tau",
        "omega",
        "cubic",
        "eta",
        "rho",
        "sigma",
        "quartic",
    }:
        return "diagonal"
    return "generic"


def _layered_block_shape(
    parameterization: object,
    name: str,
    size: int,
) -> tuple[int, ...]:
    if size == 0:
        return (0,)
    per_layer_attr = {
        "pair": "n_pair_params_per_layer",
        "tau": "n_tau_params_per_layer",
        "omega": "n_omega_params_per_layer",
        "cubic": "n_tau_params_per_layer",
        "eta": "n_eta_params_per_layer",
        "rho": "n_rho_params_per_layer",
        "quartic": "n_rho_params_per_layer",
        "sigma": "n_sigma_params_per_layer",
        "middle": "n_middle_orbital_rotation_params_per_layer",
    }.get(name)
    if per_layer_attr is None or not hasattr(parameterization, per_layer_attr):
        return (size,)
    per_layer = int(getattr(parameterization, per_layer_attr))
    if per_layer <= 0 or size % per_layer:
        return (size,)
    n_blocks = size // per_layer
    if name == "middle":
        return (n_blocks, per_layer)
    if getattr(parameterization, "shared_diagonal", False) or n_blocks == 1:
        return (per_layer,)
    return (n_blocks, per_layer)


def _block_shape(
    parameterization: object,
    name: str,
    size: int,
) -> tuple[int, ...]:
    if name in {"same_diag", "double"} and hasattr(parameterization, "norb"):
        norb = int(getattr(parameterization, "norb"))
        if size == norb:
            return (norb,)
    return _layered_block_shape(parameterization, name, size)


def _block_specs(parameterization: object) -> list[tuple[str, int, tuple[int, ...], str]]:
    sizes = []
    if hasattr(parameterization, "n_reference_params") and hasattr(
        parameterization, "ansatz_parameterization"
    ):
        n_reference = int(getattr(parameterization, "n_reference_params", 0))
        if n_reference:
            sizes.append(("reference", n_reference, (n_reference,), "reference"))
        sizes.extend(_parameter_block_specs(parameterization.ansatz_parameterization))
        return sizes
    if hasattr(parameterization, "ansatz_parameterization"):
        return _parameter_block_specs(parameterization.ansatz_parameterization)
    if _is_facade_parameterization(parameterization):
        return _block_specs(parameterization.implementation)

    ordered_attrs = [("left", "n_left_orbital_rotation_params")]
    if hasattr(parameterization, "n_same_diag_params"):
        ordered_attrs.extend(
            [
                ("same_diag", "n_same_diag_params"),
                ("double", "n_double_params"),
                ("same_spin", "n_same_spin_params"),
                ("mixed_spin", "n_mixed_spin_params"),
            ]
        )
    else:
        ordered_attrs.append(("pair", "n_pair_params"))
    if hasattr(parameterization, "n_tau_params"):
        if getattr(parameterization, "uses_reduced_cubic_chart", False):
            ordered_attrs.append(("cubic", "n_tau_params"))
        else:
            ordered_attrs.extend(
                [
                    ("tau", "n_tau_params"),
                    ("omega", "n_omega_params"),
                ]
            )
    if hasattr(parameterization, "n_eta_params"):
        if getattr(parameterization, "uses_reduced_quartic_chart", False):
            ordered_attrs.append(("quartic", "n_rho_params"))
        else:
            ordered_attrs.extend(
                [
                    ("eta", "n_eta_params"),
                    ("rho", "n_rho_params"),
                    ("sigma", "n_sigma_params"),
                ]
            )
    ordered_attrs.extend(
        [
            ("middle", "n_middle_orbital_rotation_params"),
            ("right", "n_right_orbital_rotation_params"),
        ]
    )
    for name, attr in ordered_attrs:
        size = int(getattr(parameterization, attr, 0))
        if size:
            sizes.append(
                (
                    name,
                    size,
                    _block_shape(parameterization, name, size),
                    _block_kind(name),
                )
            )
    return sizes


def _parameter_block_specs(
    parameterization: object,
) -> list[tuple[str, int, tuple[int, ...], str]]:
    specs = _block_specs(parameterization)
    if specs:
        return specs
    return [
        (block.name, block.size, block.shape, block.kind)
        for block in _own_parameter_blocks(parameterization)
    ]


def _block_sizes(parameterization: object) -> list[tuple[str, int]]:
    specs = _block_specs(parameterization)
    if specs:
        return [(name, size) for name, size, _, _ in specs]
    return [(block.name, block.size) for block in _own_parameter_blocks(parameterization)]


def parameter_blocks(
    parameterization: object,
    *,
    frozen: tuple[str, ...] | list[str] | set[str] = (),
) -> tuple[ParameterBlock, ...]:
    specs = _block_specs(parameterization)
    if specs:
        return _blocks_from_specs(parameterization, specs, frozen=frozen)
    own_blocks = _own_parameter_blocks(parameterization)
    if own_blocks:
        return _with_frozen(own_blocks, frozen)
    expected = int(getattr(parameterization, "n_params", 0))
    if expected:
        raise ValueError(
            "parameter block sizes do not sum to n_params; "
            f"got 0, expected {expected}"
        )
    return ()


def parameter_view(
    parameterization: object,
    params: np.ndarray,
    *,
    frozen: tuple[str, ...] | list[str] | set[str] = (),
    copy: bool = False,
) -> ParameterView:
    params = np.asarray(params, dtype=np.float64)
    expected = int(parameterization.n_params)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    return _parameter_view(
        params,
        parameter_blocks(parameterization, frozen=frozen),
        copy=copy,
    )


def random_parameters(
    parameterization: object,
    scale: float = 1e-3,
    *,
    seed: int | np.random.Generator | None = None,
    blocks: tuple[str, ...] | list[str] | set[str] | None = None,
) -> np.ndarray:
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    params = rng.normal(0.0, float(scale), int(parameterization.n_params))
    if blocks is not None:
        keep = set(blocks)
        mask = np.zeros(int(parameterization.n_params), dtype=bool)
        for block in parameter_blocks(parameterization):
            if block.name in keep:
                mask[block.slice()] = True
        params = np.where(mask, params, 0.0)
    return params.astype(np.float64, copy=False)


def _blocks_from_specs(
    parameterization: object,
    specs: Iterable[tuple[str, int, tuple[int, ...], str]],
    *,
    frozen: tuple[str, ...] | list[str] | set[str] = (),
) -> tuple[ParameterBlock, ...]:
    frozen_set = set(frozen)
    blocks = []
    start = 0
    for name, size, shape, kind in specs:
        stop = start + size
        blocks.append(
            ParameterBlock(
                name=name,
                start=start,
                stop=stop,
                shape=shape,
                kind=kind,
                frozen=name in frozen_set,
            )
        )
        start = stop
    expected = int(parameterization.n_params)
    if start != expected:
        raise ValueError(
            "parameter block sizes do not sum to n_params; "
            f"got {start}, expected {expected}"
        )
    return tuple(blocks)


def _is_facade_parameterization(parameterization: object) -> bool:
    if getattr(type(parameterization), "implementation", None) is None:
        return False
    return not any(
        hasattr(parameterization, attr)
        for attr in (
            "n_left_orbital_rotation_params",
            "n_pair_params",
            "n_same_diag_params",
        )
    )


def _own_parameter_blocks(parameterization: object) -> tuple[ParameterBlock, ...]:
    method = getattr(parameterization, "parameter_blocks", None)
    if method is None:
        return ()
    owner = getattr(getattr(type(parameterization), "parameter_blocks", None), "__module__", "")
    if owner in {"xquces.gcr.igcr", "xquces.gcr.references"}:
        return ()
    try:
        blocks = method()
    except TypeError:
        return ()
    return tuple(_as_parameter_block(block) for block in blocks)


def _as_parameter_block(block: object) -> ParameterBlock:
    if isinstance(block, ParameterBlock):
        return block
    return ParameterBlock(
        name=getattr(block, "name"),
        start=getattr(block, "start"),
        stop=getattr(block, "stop"),
        shape=getattr(block, "shape", None),
        kind=getattr(block, "kind", "generic"),
        frozen=getattr(block, "frozen", False),
    )


def _with_frozen(
    blocks: tuple[ParameterBlock, ...],
    frozen: tuple[str, ...] | list[str] | set[str] = (),
) -> tuple[ParameterBlock, ...]:
    frozen_set = set(frozen)
    if not frozen_set:
        return tuple(block.with_frozen(False) if block.frozen else block for block in blocks)
    return tuple(block.with_frozen(block.name in frozen_set) for block in blocks)

