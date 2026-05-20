from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.gcr.canonical import IGCRAnsatz, IGCRDiagonalCoefficients
from xquces.gcr.canonical_layering import as_layered_igcr_ansatz
from xquces.gcr.utils import (
    _diag_unitary,
    _final_unitary_from_left_and_right,
    _left_right_ov_adapted_to_native,
    _native_to_left_right_ov_adapted,
    _right_unitary_from_left_and_final,
)


@dataclass(frozen=True)
class SpinRestrictedLayeredDiagonalParameterizationCore:
    """Canonical-native mechanics for spin-restricted layered iGCR ansätze."""

    order: int
    norb: int
    nocc: int
    layers: int
    shared_diagonal: bool
    left_orbital_chart: object
    middle_orbital_chart: object
    right_orbital_chart: object
    n_diag_params_per_layer: int
    diagonal_from_parameters: Callable[[np.ndarray], IGCRDiagonalCoefficients]
    parameters_from_diagonal: Callable[
        [IGCRDiagonalCoefficients], tuple[np.ndarray, np.ndarray]
    ]
    right_depends_on_prefix: bool = True
    project_final_reference_ov: bool = True
    left_right_ov_transform_scale: float | None = None

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return int(self.left_orbital_chart.n_params(self.norb))

    @property
    def n_middle_orbital_rotation_params_per_layer(self) -> int:
        return int(self.middle_orbital_chart.n_params(self.norb))

    @property
    def n_middle_orbital_rotation_params(self) -> int:
        return max(0, self.layers - 1) * self.n_middle_orbital_rotation_params_per_layer

    @property
    def n_diag_params(self) -> int:
        if self.shared_diagonal:
            return self.n_diag_params_per_layer
        return self.layers * self.n_diag_params_per_layer

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return int(self.right_orbital_chart.n_params(self.norb))

    @property
    def _middle_orbital_rotation_start(self) -> int:
        return self.n_left_orbital_rotation_params + self.n_diag_params

    @property
    def _right_orbital_rotation_start(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diag_params
            + self.n_middle_orbital_rotation_params
        )

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diag_params
            + self.n_middle_orbital_rotation_params
            + self.n_right_orbital_rotation_params
        )

    def native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self.left_right_ov_transform_scale,
        )

    def public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(
            params,
            self.norb,
            self.nocc,
            self._right_orbital_rotation_start,
            self.left_right_ov_transform_scale,
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCRAnsatz:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self.native_parameters_from_public(params)

        left, diagonal_params, middle_rotations, final = self._parse_native(params)
        diagonals = tuple(
            self.diagonal_from_parameters(block) for block in diagonal_params
        )
        right = self._right_from_final(left, middle_rotations, final)
        return IGCRAnsatz(
            order=self.order,
            diagonals=diagonals,
            rotations=tuple([left, *middle_rotations, right]),
            nocc=self.nocc,
        )

    def parameters_from_ansatz(self, ansatz: IGCRAnsatz | object) -> np.ndarray:
        generic = self._as_canonical_ansatz(ansatz)
        layered = as_layered_igcr_ansatz(generic, self.layers, order=self.order)

        rotations = [
            np.asarray(rotation, dtype=np.complex128)
            for rotation in layered.rotations
        ]
        diag_params = []
        for layer_idx, diagonal in enumerate(layered.diagonals):
            params_i, phase_vec = self.parameters_from_diagonal(diagonal)
            diag_params.append(np.asarray(params_i, dtype=np.float64))
            rotations[layer_idx] = rotations[layer_idx] @ _diag_unitary(phase_vec)

        rotation_params = self._rotation_parameters_from_rotations(rotations)
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = self._pack_left_and_diagonal(out, rotation_params[0], diag_params)
        idx = self._pack_middle(out, idx, rotation_params[1:])
        self._pack_right(out, idx, rotation_params, rotations[-1])
        return self.public_parameters_from_native(out)

    def _as_canonical_ansatz(self, ansatz: IGCRAnsatz | object) -> IGCRAnsatz:
        if isinstance(ansatz, IGCRAnsatz):
            generic = ansatz
        else:
            if getattr(ansatz, "is_spin_restricted", True) is False:
                raise TypeError("expected a spin-restricted iGCR ansatz")
            generic = IGCRAnsatz.from_legacy(ansatz, order=self.order)
        if generic.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if generic.nocc != self.nocc:
            raise ValueError("ansatz nocc does not match parameterization")
        if generic.order != self.order:
            generic = IGCRAnsatz(
                order=self.order,
                diagonals=generic.diagonals,
                rotations=generic.rotations,
                nocc=generic.nocc,
            )
        return generic

    def _parse_native(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self.left_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        idx += n

        n_diag = self.n_diag_params_per_layer
        if self.shared_diagonal:
            diagonal_params = [params[idx : idx + n_diag]] * self.layers
            idx += n_diag
        else:
            diagonal_params = []
            for _ in range(self.layers):
                diagonal_params.append(params[idx : idx + n_diag])
                idx += n_diag

        middle_rotations = []
        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for _ in range(self.layers - 1):
            middle_rotations.append(
                self.middle_orbital_chart.unitary_from_parameters(
                    params[idx : idx + n_middle], self.norb
                )
            )
            idx += n_middle

        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(
            params[idx : idx + n], self.norb
        )
        return left, diagonal_params, middle_rotations, final

    def _right_from_final(
        self,
        left: np.ndarray,
        middle_rotations: list[np.ndarray],
        final: np.ndarray,
    ) -> np.ndarray:
        if not self.right_depends_on_prefix:
            return final
        prefix = np.asarray(left, dtype=np.complex128)
        for rotation in middle_rotations:
            prefix = prefix @ np.asarray(rotation, dtype=np.complex128)
        return _right_unitary_from_left_and_final(prefix, final, self.nocc)

    def _rotation_parameters_from_rotations(
        self,
        rotations: list[np.ndarray],
    ) -> list[np.ndarray]:
        rotation_params = []
        for layer_idx in range(self.layers):
            chart = (
                self.left_orbital_chart
                if layer_idx == 0
                else self.middle_orbital_chart
            )
            expected = (
                self.n_left_orbital_rotation_params
                if layer_idx == 0
                else self.n_middle_orbital_rotation_params_per_layer
            )
            if hasattr(chart, "parameters_and_right_phase_from_unitary"):
                params_i, right_phase = chart.parameters_and_right_phase_from_unitary(
                    rotations[layer_idx]
                )
            else:
                params_i = chart.parameters_from_unitary(rotations[layer_idx])
                right_phase = np.zeros(self.norb, dtype=np.float64)
            if params_i.shape != (expected,):
                raise ValueError(
                    "orbital chart returned the wrong number of parameters; "
                    f"expected {(expected,)}, got {params_i.shape}"
                )
            rotation_params.append(np.asarray(params_i, dtype=np.float64))
            rotations[layer_idx + 1] = _diag_unitary(right_phase) @ rotations[
                layer_idx + 1
            ]
        return rotation_params

    def _pack_left_and_diagonal(
        self,
        out: np.ndarray,
        left_params: np.ndarray,
        diag_params: list[np.ndarray],
    ) -> int:
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx : idx + n] = left_params
        idx += n

        n_diag = self.n_diag_params_per_layer
        if self.shared_diagonal:
            out[idx : idx + n_diag] = np.mean(np.stack(diag_params, axis=0), axis=0)
            return idx + n_diag
        for params_i in diag_params:
            out[idx : idx + n_diag] = params_i
            idx += n_diag
        return idx

    def _pack_middle(
        self,
        out: np.ndarray,
        idx: int,
        middle_params: list[np.ndarray],
    ) -> int:
        n_middle = self.n_middle_orbital_rotation_params_per_layer
        for params_i in middle_params:
            out[idx : idx + n_middle] = params_i
            idx += n_middle
        return idx

    def _pack_right(
        self,
        out: np.ndarray,
        idx: int,
        rotation_params: list[np.ndarray],
        right_rotation: np.ndarray,
    ) -> None:
        n = self.n_right_orbital_rotation_params
        if self.right_depends_on_prefix:
            prefix = np.eye(self.norb, dtype=np.complex128)
            for layer_idx, params_i in enumerate(rotation_params):
                chart = (
                    self.left_orbital_chart
                    if layer_idx == 0
                    else self.middle_orbital_chart
                )
                prefix = prefix @ chart.unitary_from_parameters(params_i, self.norb)
            final = _final_unitary_from_left_and_right(
                prefix,
                right_rotation,
                self.nocc,
                project_reference_ov=self.project_final_reference_ov,
            )
        else:
            final = right_rotation
        out[idx : idx + n] = self.right_orbital_chart.parameters_from_unitary(final)

