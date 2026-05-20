from __future__ import annotations

from typing import Callable

import numpy as np

from xquces.basis import reshape_state
from xquces.gcr.igcr import (
    IGCR2SpinRestrictedParameterization,
    IGCR3SpinRestrictedParameterization,
    IGCR4SpinRestrictedParameterization,
)
from xquces.jacobian.diagonal import _diag_feature_matrix
from xquces.jacobian.orbital import (
    _generator_batch_from_kappa,
    _left_chart_basis,
    _left_chart_kappa,
    _right_chart_basis,
    _right_chart_kappa,
)
from xquces.jacobian.sector import (
    _apply_batch_transform,
    _batch_row_and_col,
    _one_body_batch_to_sector,
    _one_body_tensor,
    _sector_representation,
)


def _public_to_native_matrix(parameterization: object) -> np.ndarray | None:
    scale = getattr(parameterization, "_left_right_ov_transform_scale", None)
    if scale is None:
        return None
    n = parameterization.n_params
    eye = np.eye(n, dtype=np.float64)
    return np.column_stack(
        [parameterization._native_parameters_from_public(eye[:, k]) for k in range(n)]
    )


def _finite_difference_restricted_gcr_jacobian(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
    *,
    step: float = 1e-6,
) -> Callable[[np.ndarray], np.ndarray]:
    reference_vec = np.asarray(reference_vec, dtype=np.complex128)
    dim = reference_vec.size

    def state(params: np.ndarray) -> np.ndarray:
        return parameterization.ansatz_from_parameters(params).apply(
            reference_vec, nelec=nelec, copy=True
        )

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        out = np.empty((dim, parameterization.n_params), dtype=np.complex128)
        for idx in range(parameterization.n_params):
            h = step * max(1.0, abs(float(params[idx])))
            plus = params.copy()
            minus = params.copy()
            plus[idx] += h
            minus[idx] -= h
            out[:, idx] = (state(plus) - state(minus)) / (2.0 * h)
        return out

    return jac


def _finite_difference_restricted_gcr_subspace_jacobian(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
    *,
    step: float = 1e-6,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    reference_vec = np.asarray(reference_vec, dtype=np.complex128)
    dim = reference_vec.size

    def state(params: np.ndarray) -> np.ndarray:
        return parameterization.ansatz_from_parameters(params).apply(
            reference_vec, nelec=nelec, copy=True
        )

    def subspace_jac(params: np.ndarray, directions: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != parameterization.n_params:
            raise ValueError(
                "directions must have shape "
                f"({parameterization.n_params}, m); got {directions.shape}."
            )
        out = np.zeros((dim, directions.shape[1]), dtype=np.complex128)
        param_scale = max(1.0, float(np.linalg.norm(params)))
        for idx in range(directions.shape[1]):
            direction = directions[:, idx]
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm == 0.0:
                continue
            h = step * param_scale / direction_norm
            out[:, idx] = (
                state(params + h * direction) - state(params - h * direction)
            ) / (2.0 * h)
        return out

    return subspace_jac


def _parse_layered_igcr2_native(
    parameterization: object,
    native: np.ndarray,
):
    idx = 0
    n_left = parameterization.n_left_orbital_rotation_params
    left_params = native[idx : idx + n_left]
    idx += n_left

    n_diag = getattr(
        parameterization,
        "n_diag_params_per_layer",
        parameterization.n_pair_params_per_layer,
    )
    if parameterization.shared_diagonal:
        diag_params = [native[idx : idx + n_diag]] * parameterization.layers
        idx += n_diag
    else:
        diag_params = []
        for _ in range(parameterization.layers):
            diag_params.append(native[idx : idx + n_diag])
            idx += n_diag

    n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
    middle_params = []
    for _ in range(parameterization.layers - 1):
        middle_params.append(native[idx : idx + n_middle])
        idx += n_middle

    n_right = parameterization.n_right_orbital_rotation_params
    right_params = native[idx : idx + n_right]
    return left_params, diag_params, middle_params, right_params


def _layered_igcr2_runtime(
    parameterization: object,
    native: np.ndarray,
    reference_mat: np.ndarray,
    nelec: tuple[int, int],
    diag_features: np.ndarray,
    left_chart: object,
    middle_chart: object,
    right_chart: object,
):
    norb = parameterization.norb
    layers = parameterization.layers
    left_params, diag_params, middle_params, right_params = (
        _parse_layered_igcr2_native(parameterization, native)
    )

    prefix_rotations = [
        left_chart.unitary_from_parameters(left_params, norb),
        *[
            middle_chart.unitary_from_parameters(params, norb)
            for params in middle_params
        ],
    ]
    final = right_chart.unitary_from_parameters(right_params, norb)
    prefix_before = []
    prefix = np.eye(norb, dtype=np.complex128)
    for rotation in prefix_rotations:
        prefix_before.append(prefix)
        prefix = prefix @ rotation
    right_depends_on_prefix = bool(
        getattr(parameterization, "_right_depends_on_prefix", False)
    )
    right = prefix.conj().T @ final if right_depends_on_prefix else final
    rotations = [*prefix_rotations, right]

    rep_a = [_sector_representation(u, norb, nelec[0]) for u in rotations]
    rep_b = [_sector_representation(u, norb, nelec[1]) for u in rotations]

    dim_a, dim_b = reference_mat.shape
    phases = [None] * layers
    after_diag = [None] * layers
    after_rotation = [None] * (layers + 1)

    current = rep_a[layers] @ reference_mat @ rep_b[layers].T
    after_rotation[layers] = current
    for idx in range(layers - 1, -1, -1):
        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ diag_params[idx])).reshape(
                dim_a, dim_b
            )
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)
        phases[idx] = phase
        current = phase * current
        after_diag[idx] = current
        current = rep_a[idx] @ current @ rep_b[idx].T
        after_rotation[idx] = current

    return {
        "left_params": left_params,
        "middle_params": middle_params,
        "right_params": right_params,
        "rotations": rotations,
        "prefix_before": prefix_before,
        "prefix_total": prefix,
        "right_depends_on_prefix": right_depends_on_prefix,
        "rep_a": rep_a,
        "rep_b": rep_b,
        "phases": phases,
        "after_diag": after_diag,
        "after_rotation": after_rotation,
    }


def _apply_orbital_generator_batch(
    generator_batch: np.ndarray,
    mat: np.ndarray,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
) -> np.ndarray:
    left = _one_body_batch_to_sector(generator_batch, tensor_a)
    right = _one_body_batch_to_sector(generator_batch, tensor_b)
    return _batch_row_and_col(left, right, mat)


def _propagate_layered_after_rotation(runtime: dict, start: int, mats: np.ndarray):
    out = mats
    for idx in range(start - 1, -1, -1):
        out = runtime["phases"][idx][None, :, :] * out
        out = _apply_batch_transform(
            runtime["rep_a"][idx],
            runtime["rep_b"][idx],
            out,
        )
    return out


def _propagate_layered_after_diagonal(runtime: dict, idx: int, mats: np.ndarray):
    out = _apply_batch_transform(
        runtime["rep_a"][idx],
        runtime["rep_b"][idx],
        mats,
    )
    return _propagate_layered_after_rotation(runtime, idx, out)


def _layered_prefix_rotation_block(
    runtime: dict,
    rot_idx: int,
    generator_batch: np.ndarray,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
) -> np.ndarray:
    if generator_batch.shape[0] == 0:
        dim_a, dim_b = runtime["after_rotation"][0].shape
        return np.zeros((0, dim_a, dim_b), dtype=np.complex128)

    direct = _apply_orbital_generator_batch(
        generator_batch,
        runtime["after_rotation"][rot_idx],
        tensor_a,
        tensor_b,
    )
    out = _propagate_layered_after_rotation(runtime, rot_idx, direct)
    if runtime.get("right_depends_on_prefix", False):
        before = runtime["prefix_before"][rot_idx]
        prefix = runtime["prefix_total"]
        transported = np.einsum(
            "ab,jbc,dc->jad",
            before,
            generator_batch,
            before.conj(),
            optimize=True,
        )
        right_gen = -np.einsum(
            "ab,jbc,cd->jad",
            prefix.conj().T,
            transported,
            prefix,
            optimize=True,
        )
        right_direct = _apply_orbital_generator_batch(
            right_gen,
            runtime["after_rotation"][-1],
            tensor_a,
            tensor_b,
        )
        out += _propagate_layered_after_rotation(
            runtime,
            len(runtime["after_diag"]),
            right_direct,
        )
    return out


def _layered_final_rotation_block(
    runtime: dict,
    generator_batch: np.ndarray,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
) -> np.ndarray:
    if generator_batch.shape[0] == 0:
        dim_a, dim_b = runtime["after_rotation"][0].shape
        return np.zeros((0, dim_a, dim_b), dtype=np.complex128)
    if runtime.get("right_depends_on_prefix", False):
        prefix = runtime["prefix_total"]
        generator_batch = np.einsum(
            "ab,jbc,cd->jad",
            prefix.conj().T,
            generator_batch,
            prefix,
            optimize=True,
        )
    out = _apply_orbital_generator_batch(
        generator_batch,
        runtime["after_rotation"][-1],
        tensor_a,
        tensor_b,
    )
    return _propagate_layered_after_rotation(
        runtime, len(runtime["after_diag"]), out
    )


def make_layered_igcr2_jacobian(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    norb = parameterization.norb
    layers = parameterization.layers
    left_chart = parameterization._left_orbital_chart
    middle_chart = parameterization._middle_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    middle_basis = _left_chart_basis(middle_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    diag_features = _diag_feature_matrix(parameterization, nelec)
    diag_feature_tensor = diag_features.T.reshape(
        diag_features.shape[1], dim_a, dim_b
    )
    transform = _public_to_native_matrix(parameterization)

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        native = parameterization._native_parameters_from_public(params)
        runtime = _layered_igcr2_runtime(
            parameterization,
            native,
            reference_mat,
            nelec,
            diag_features,
            left_chart,
            middle_chart,
            right_chart,
        )

        blocks = []

        n_left = parameterization.n_left_orbital_rotation_params
        if n_left:
            kappa_left = _left_chart_kappa(
                left_chart, runtime["left_params"], norb, basis=left_basis
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            blocks.append(
                _layered_prefix_rotation_block(
                    runtime, 0, gen_left, tensor_a, tensor_b
                ).reshape(n_left, dim_a * dim_b).T
            )

        n_diag = getattr(
            parameterization,
            "n_diag_params_per_layer",
            parameterization.n_pair_params_per_layer,
        )
        if n_diag:
            diag_blocks = []
            for idx in range(layers):
                d_after_diag = (
                    1j
                    * diag_feature_tensor
                    * runtime["after_diag"][idx][None, :, :]
                )
                diag_blocks.append(
                    _propagate_layered_after_diagonal(
                        runtime, idx, d_after_diag
                    )
                )
            if parameterization.shared_diagonal:
                blocks.append(
                    np.sum(diag_blocks, axis=0).reshape(n_diag, dim_a * dim_b).T
                )
            else:
                blocks.extend(
                    block.reshape(n_diag, dim_a * dim_b).T
                    for block in diag_blocks
                )

        n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
        if n_middle:
            for idx, middle_params in enumerate(runtime["middle_params"], start=1):
                kappa_middle = _left_chart_kappa(
                    middle_chart, middle_params, norb, basis=middle_basis
                )
                gen_middle = _generator_batch_from_kappa(
                    kappa_middle, middle_basis
                )
                blocks.append(
                    _layered_prefix_rotation_block(
                        runtime, idx, gen_middle, tensor_a, tensor_b
                    ).reshape(n_middle, dim_a * dim_b).T
                )

        n_right = parameterization.n_right_orbital_rotation_params
        if n_right:
            kappa_final = _right_chart_kappa(
                right_chart, runtime["right_params"], norb
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            blocks.append(
                _layered_final_rotation_block(
                    runtime, gen_final, tensor_a, tensor_b
                ).reshape(n_right, dim_a * dim_b).T
            )

        if blocks:
            out = np.hstack(blocks)
        else:
            out = np.zeros((dim_a * dim_b, 0), dtype=np.complex128)
        if transform is not None:
            out = out @ transform
        return out

    return jac


def make_layered_igcr2_subspace_jacobian(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    norb = parameterization.norb
    layers = parameterization.layers
    left_chart = parameterization._left_orbital_chart
    middle_chart = parameterization._middle_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    middle_basis = _left_chart_basis(middle_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    dim = dim_a * dim_b
    diag_features = _diag_feature_matrix(parameterization, nelec)
    transform = _public_to_native_matrix(parameterization)

    def subspace_jac(params: np.ndarray, directions: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != parameterization.n_params:
            raise ValueError(
                "directions must have shape "
                f"({parameterization.n_params}, m); got {directions.shape}."
            )
        n_dir = directions.shape[1]
        if n_dir == 0:
            return np.zeros((dim, 0), dtype=np.complex128)

        native = parameterization._native_parameters_from_public(params)
        native_dirs = directions if transform is None else transform @ directions
        runtime = _layered_igcr2_runtime(
            parameterization,
            native,
            reference_mat,
            nelec,
            diag_features,
            left_chart,
            middle_chart,
            right_chart,
        )

        d_state = np.zeros((n_dir, dim_a, dim_b), dtype=np.complex128)
        idx = 0

        n_left = parameterization.n_left_orbital_rotation_params
        left_dirs = native_dirs[idx : idx + n_left]
        idx += n_left
        if n_left:
            kappa_left = _left_chart_kappa(
                left_chart, runtime["left_params"], norb, basis=left_basis
            )
            left_basis_dirs = np.einsum(
                "kj,kpq->jpq", left_dirs, left_basis, optimize=True
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis_dirs)
            d_state += _layered_prefix_rotation_block(
                runtime, 0, gen_left, tensor_a, tensor_b
            )

        n_diag = getattr(
            parameterization,
            "n_diag_params_per_layer",
            parameterization.n_pair_params_per_layer,
        )
        if parameterization.shared_diagonal:
            diag_dirs = native_dirs[idx : idx + n_diag]
            idx += n_diag
            if n_diag:
                feature_dirs = diag_features @ diag_dirs
                d_after_diag = [
                    1j
                    * feature_dirs.T.reshape(n_dir, dim_a, dim_b)
                    * runtime["after_diag"][layer][None, :, :]
                    for layer in range(layers)
                ]
                for layer, block in enumerate(d_after_diag):
                    d_state += _propagate_layered_after_diagonal(
                        runtime, layer, block
                    )
        else:
            for layer in range(layers):
                diag_dirs = native_dirs[idx : idx + n_diag]
                idx += n_diag
                if n_diag:
                    feature_dirs = diag_features @ diag_dirs
                    block = (
                        1j
                        * feature_dirs.T.reshape(n_dir, dim_a, dim_b)
                        * runtime["after_diag"][layer][None, :, :]
                    )
                    d_state += _propagate_layered_after_diagonal(
                        runtime, layer, block
                    )

        n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
        for layer, middle_params in enumerate(runtime["middle_params"], start=1):
            middle_dirs = native_dirs[idx : idx + n_middle]
            idx += n_middle
            if n_middle:
                kappa_middle = _left_chart_kappa(
                    middle_chart, middle_params, norb, basis=middle_basis
                )
                middle_basis_dirs = np.einsum(
                    "kj,kpq->jpq",
                    middle_dirs,
                    middle_basis,
                    optimize=True,
                )
                gen_middle = _generator_batch_from_kappa(
                    kappa_middle, middle_basis_dirs
                )
                d_state += _layered_prefix_rotation_block(
                    runtime, layer, gen_middle, tensor_a, tensor_b
                )

        n_right = parameterization.n_right_orbital_rotation_params
        right_dirs = native_dirs[idx : idx + n_right]
        if n_right:
            kappa_final = _right_chart_kappa(
                right_chart, runtime["right_params"], norb
            )
            right_basis_dirs = np.einsum(
                "kj,kpq->jpq", right_dirs, right_basis, optimize=True
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis_dirs)
            d_state += _layered_final_rotation_block(
                runtime, gen_final, tensor_a, tensor_b
            )

        return d_state.reshape(n_dir, dim).T

    return subspace_jac


def _batch_vjp(batch: np.ndarray, v_mat: np.ndarray) -> np.ndarray:
    if batch.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return 2.0 * np.einsum("jab,ab->j", batch.conj(), v_mat, optimize=True).real


def make_layered_igcr2_vjp(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    norb = parameterization.norb
    layers = parameterization.layers
    left_chart = parameterization._left_orbital_chart
    middle_chart = parameterization._middle_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    middle_basis = _left_chart_basis(middle_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    diag_features = _diag_feature_matrix(parameterization, nelec)
    diag_feature_tensor = diag_features.T.reshape(
        diag_features.shape[1], dim_a, dim_b
    )
    transform = _public_to_native_matrix(parameterization)

    def vjp(params: np.ndarray, v: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        v = np.asarray(v, dtype=np.complex128)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        if v.shape != (reference_vec.size,):
            raise ValueError(f"Expected v with shape {(reference_vec.size,)}, got {v.shape}.")
        v_mat = reshape_state(v, norb, nelec)
        native = parameterization._native_parameters_from_public(params)
        runtime = _layered_igcr2_runtime(
            parameterization,
            native,
            reference_mat,
            nelec,
            diag_features,
            left_chart,
            middle_chart,
            right_chart,
        )

        grad_blocks = []

        n_left = parameterization.n_left_orbital_rotation_params
        if n_left:
            kappa_left = _left_chart_kappa(
                left_chart, runtime["left_params"], norb, basis=left_basis
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            block = _layered_prefix_rotation_block(
                runtime, 0, gen_left, tensor_a, tensor_b
            )
            grad_blocks.append(_batch_vjp(block, v_mat))

        n_diag = getattr(
            parameterization,
            "n_diag_params_per_layer",
            parameterization.n_pair_params_per_layer,
        )
        if n_diag:
            diag_blocks = []
            for idx in range(layers):
                d_after_diag = (
                    1j
                    * diag_feature_tensor
                    * runtime["after_diag"][idx][None, :, :]
                )
                diag_blocks.append(
                    _propagate_layered_after_diagonal(
                        runtime, idx, d_after_diag
                    )
                )
            if parameterization.shared_diagonal:
                grad_blocks.append(_batch_vjp(np.sum(diag_blocks, axis=0), v_mat))
            else:
                grad_blocks.extend(_batch_vjp(block, v_mat) for block in diag_blocks)

        n_middle = parameterization.n_middle_orbital_rotation_params_per_layer
        if n_middle:
            for idx, middle_params in enumerate(runtime["middle_params"], start=1):
                kappa_middle = _left_chart_kappa(
                    middle_chart, middle_params, norb, basis=middle_basis
                )
                gen_middle = _generator_batch_from_kappa(
                    kappa_middle, middle_basis
                )
                block = _layered_prefix_rotation_block(
                    runtime, idx, gen_middle, tensor_a, tensor_b
                )
                grad_blocks.append(_batch_vjp(block, v_mat))

        n_right = parameterization.n_right_orbital_rotation_params
        if n_right:
            kappa_final = _right_chart_kappa(
                right_chart, runtime["right_params"], norb
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            block = _layered_final_rotation_block(
                runtime, gen_final, tensor_a, tensor_b
            )
            grad_blocks.append(_batch_vjp(block, v_mat))

        if grad_blocks:
            grad = np.concatenate(grad_blocks)
        else:
            grad = np.zeros(0, dtype=np.float64)
        if transform is not None:
            grad = transform.T @ grad
        return grad

    return vjp


def make_restricted_gcr_vjp(
    parameterization: (
        IGCR2SpinRestrictedParameterization
        | IGCR3SpinRestrictedParameterization
        | IGCR4SpinRestrictedParameterization
    ),
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if (
        isinstance(parameterization, IGCR2SpinRestrictedParameterization)
        or getattr(parameterization, "order", None) == 2
        or getattr(parameterization, "layers", 1) > 1
    ):
        return make_layered_igcr2_vjp(parameterization, reference_vec, nelec)

    jac = make_restricted_gcr_jacobian(parameterization, reference_vec, nelec)

    def vjp(params: np.ndarray, v: np.ndarray) -> np.ndarray:
        J = jac(params)
        return 2.0 * (J.conj().T @ np.asarray(v, dtype=np.complex128)).real

    return vjp


def make_restricted_gcr_jacobian(
    parameterization: (
        IGCR2SpinRestrictedParameterization
        | IGCR3SpinRestrictedParameterization
        | IGCR4SpinRestrictedParameterization
    ),
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    if (
        isinstance(parameterization, IGCR2SpinRestrictedParameterization)
        or getattr(parameterization, "order", None) == 2
        or getattr(parameterization, "layers", 1) > 1
    ):
        return make_layered_igcr2_jacobian(
            parameterization, reference_vec, nelec
        )
    norb = parameterization.norb
    left_chart = parameterization._left_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    diag_features = _diag_feature_matrix(parameterization, nelec)
    transform = _public_to_native_matrix(parameterization)

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        native = parameterization._native_parameters_from_public(params)

        n_left = parameterization.n_left_orbital_rotation_params
        right_start = parameterization._right_orbital_rotation_start
        n_right = parameterization.n_right_orbital_rotation_params

        left_params = native[:n_left]
        diag_params = native[n_left:right_start]
        right_params = native[right_start : right_start + n_right]

        u_left = left_chart.unitary_from_parameters(left_params, norb)
        u_final = right_chart.unitary_from_parameters(right_params, norb)
        u_right = u_left.conj().T @ u_final

        kappa_left = _left_chart_kappa(left_chart, left_params, norb, basis=left_basis)
        kappa_final = _right_chart_kappa(right_chart, right_params, norb)

        rep_left_a = _sector_representation(u_left, norb, nelec[0])
        rep_left_b = _sector_representation(u_left, norb, nelec[1])
        rep_right_a = _sector_representation(u_right, norb, nelec[0])
        rep_right_b = _sector_representation(u_right, norb, nelec[1])

        rotated_right = rep_right_a @ reference_mat @ rep_right_b.T

        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ diag_params)).reshape(dim_a, dim_b)
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)

        diagonalized = phase * rotated_right
        state = rep_left_a @ diagonalized @ rep_left_b.T

        blocks = []

        if n_left:
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis)
            gen_right_from_left = -np.matmul(
                u_left.conj().T,
                np.matmul(gen_left, u_left),
            )

            left_a = _one_body_batch_to_sector(gen_left, tensor_a)
            left_b = _one_body_batch_to_sector(gen_left, tensor_b)
            right_a = _one_body_batch_to_sector(gen_right_from_left, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_left, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _batch_row_and_col(left_a, left_b, state)
            d_state += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_left, dim_a * dim_b).T)

        if diag_params.size:
            d_diagonalized = (
                1j
                * diag_features.T.reshape(diag_params.size, dim_a, dim_b)
                * diagonalized[None, :, :]
            )
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(diag_params.size, dim_a * dim_b).T)

        if n_right:
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis)
            gen_right_from_final = np.matmul(
                u_left.conj().T,
                np.matmul(gen_final, u_left),
            )

            right_a = _one_body_batch_to_sector(gen_right_from_final, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_final, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state = _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            blocks.append(d_state.reshape(n_right, dim_a * dim_b).T)

        if blocks:
            out = np.hstack(blocks)
        else:
            out = np.zeros((dim_a * dim_b, 0), dtype=np.complex128)

        if transform is not None:
            out = out @ transform
        return out

    return jac


def make_restricted_gcr_subspace_jacobian(
    parameterization: (
        IGCR2SpinRestrictedParameterization
        | IGCR3SpinRestrictedParameterization
        | IGCR4SpinRestrictedParameterization
    ),
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a function computing ``J(params) @ directions`` analytically.

    Unlike :func:`make_restricted_gcr_jacobian`, this does not materialise the
    full tangent matrix. Its cost scales with the number of requested directions.
    """
    if (
        isinstance(parameterization, IGCR2SpinRestrictedParameterization)
        or getattr(parameterization, "order", None) == 2
        or getattr(parameterization, "layers", 1) > 1
    ):
        return make_layered_igcr2_subspace_jacobian(
            parameterization, reference_vec, nelec
        )
    norb = parameterization.norb
    left_chart = parameterization._left_orbital_chart
    right_chart = parameterization.right_orbital_chart
    left_basis = _left_chart_basis(left_chart, norb)
    right_basis = _right_chart_basis(right_chart, norb)
    tensor_a = _one_body_tensor(norb, nelec[0])
    tensor_b = _one_body_tensor(norb, nelec[1])
    reference_mat = reshape_state(
        np.asarray(reference_vec, dtype=np.complex128), norb, nelec
    )
    dim_a, dim_b = reference_mat.shape
    dim = dim_a * dim_b
    diag_features = _diag_feature_matrix(parameterization, nelec)
    transform = _public_to_native_matrix(parameterization)

    def subspace_jac(params: np.ndarray, directions: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != parameterization.n_params:
            raise ValueError(
                "directions must have shape "
                f"({parameterization.n_params}, m); got {directions.shape}."
            )
        n_dir = directions.shape[1]
        if n_dir == 0:
            return np.zeros((dim, 0), dtype=np.complex128)

        native = parameterization._native_parameters_from_public(params)
        native_dirs = directions if transform is None else transform @ directions

        n_left = parameterization.n_left_orbital_rotation_params
        right_start = parameterization._right_orbital_rotation_start
        n_right = parameterization.n_right_orbital_rotation_params

        left_params = native[:n_left]
        diag_params = native[n_left:right_start]
        right_params = native[right_start : right_start + n_right]

        left_dirs = native_dirs[:n_left]
        diag_dirs = native_dirs[n_left:right_start]
        right_dirs = native_dirs[right_start : right_start + n_right]

        u_left = left_chart.unitary_from_parameters(left_params, norb)
        u_final = right_chart.unitary_from_parameters(right_params, norb)
        u_right = u_left.conj().T @ u_final

        kappa_left = _left_chart_kappa(left_chart, left_params, norb, basis=left_basis)
        kappa_final = _right_chart_kappa(right_chart, right_params, norb)

        rep_left_a = _sector_representation(u_left, norb, nelec[0])
        rep_left_b = _sector_representation(u_left, norb, nelec[1])
        rep_right_a = _sector_representation(u_right, norb, nelec[0])
        rep_right_b = _sector_representation(u_right, norb, nelec[1])

        rotated_right = rep_right_a @ reference_mat @ rep_right_b.T

        if diag_features.shape[1]:
            phase = np.exp(1j * (diag_features @ diag_params)).reshape(dim_a, dim_b)
        else:
            phase = np.ones((dim_a, dim_b), dtype=np.complex128)

        diagonalized = phase * rotated_right
        state = rep_left_a @ diagonalized @ rep_left_b.T
        d_state = np.zeros((n_dir, dim_a, dim_b), dtype=np.complex128)

        if n_left:
            left_basis_dirs = np.einsum(
                "kj,kpq->jpq",
                left_dirs,
                left_basis,
                optimize=True,
            )
            gen_left = _generator_batch_from_kappa(kappa_left, left_basis_dirs)
            gen_right_from_left = -np.matmul(
                u_left.conj().T,
                np.matmul(gen_left, u_left),
            )

            left_a = _one_body_batch_to_sector(gen_left, tensor_a)
            left_b = _one_body_batch_to_sector(gen_left, tensor_b)
            right_a = _one_body_batch_to_sector(gen_right_from_left, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_left, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_left = _batch_row_and_col(left_a, left_b, state)
            d_left += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)
            d_state += d_left

        if diag_params.size:
            feature_dirs = diag_features @ diag_dirs
            d_diagonalized = (
                1j
                * feature_dirs.T.reshape(n_dir, dim_a, dim_b)
                * diagonalized[None, :, :]
            )
            d_state += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)

        if n_right:
            right_basis_dirs = np.einsum(
                "kj,kpq->jpq",
                right_dirs,
                right_basis,
                optimize=True,
            )
            gen_final = _generator_batch_from_kappa(kappa_final, right_basis_dirs)
            gen_right_from_final = np.matmul(
                u_left.conj().T,
                np.matmul(gen_final, u_left),
            )

            right_a = _one_body_batch_to_sector(gen_right_from_final, tensor_a)
            right_b = _one_body_batch_to_sector(gen_right_from_final, tensor_b)

            d_rotated_right = _batch_row_and_col(right_a, right_b, rotated_right)
            d_diagonalized = phase[None, :, :] * d_rotated_right
            d_state += _apply_batch_transform(rep_left_a, rep_left_b, d_diagonalized)

        return d_state.reshape(n_dir, dim).T

    return subspace_jac

