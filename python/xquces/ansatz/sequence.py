from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.ansatz.parameters import ParameterBlock, ParameterView, parameter_view


@dataclass(frozen=True)
class GateSequenceParameterization:
    """Flat-parameter sequence of gate parameterizations.

    This is intentionally small: it centralizes block ownership, slicing, and
    instance construction while callers still decide how gate instances are
    assembled into a concrete ansatz object.
    """

    gates: tuple[object, ...]
    ansatz_builder: Callable[[tuple[object, ...]], object] | None = None
    native_parameters_from_public: Callable[[np.ndarray], np.ndarray] | None = None
    ansatz_parameters_from_instance: Callable[[object], np.ndarray] | None = None
    default_nelec: tuple[int, int] | None = None
    parameter_order: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.default_nelec is not None:
            object.__setattr__(
                self,
                "default_nelec",
                tuple(int(x) for x in self.default_nelec),
            )
        if self.parameter_order is None:
            object.__setattr__(
                self,
                "parameter_order",
                tuple(range(len(self.gates))),
            )
        else:
            order = tuple(int(idx) for idx in self.parameter_order)
            expected = tuple(range(len(self.gates)))
            if tuple(sorted(order)) != expected:
                raise ValueError(
                    "parameter_order must be a permutation of gate indices"
                )
            object.__setattr__(self, "parameter_order", order)

    @property
    def n_params(self) -> int:
        return sum(int(gate.n_params) for gate in self.gates)

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        blocks: list[ParameterBlock] = []
        offset = 0
        for gate_idx in self.parameter_order:
            gate = self.gates[gate_idx]
            for block in gate.blocks():
                blocks.append(block.with_offset(offset))
            offset += int(gate.n_params)
        if offset != self.n_params:
            raise ValueError("gate block sizes do not sum to n_params")
        return tuple(blocks)

    def parameter_view(self, params: np.ndarray, *, copy: bool = False) -> ParameterView:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return parameter_view(params, self.parameter_blocks(), copy=copy)

    def gate_instances_from_parameters(self, params: np.ndarray) -> tuple[object, ...]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        if self.native_parameters_from_public is not None:
            params = np.asarray(
                self.native_parameters_from_public(params), dtype=np.float64
            )
            if params.shape != (self.n_params,):
                raise ValueError(
                    "native_parameters_from_public returned shape "
                    f"{params.shape}; expected {(self.n_params,)}."
                )
        instances = [None] * len(self.gates)
        start = 0
        for gate_idx in self.parameter_order:
            gate = self.gates[gate_idx]
            stop = start + int(gate.n_params)
            instances[gate_idx] = gate.instance_from_parameters(params[start:stop])
            start = stop
        return tuple(instances)

    def ansatz_from_parameters(self, params: np.ndarray):
        if self.ansatz_builder is None:
            raise TypeError("ansatz_builder is required to build an ansatz")
        return self.ansatz_builder(self.gate_instances_from_parameters(params))

    def parameters_from_ansatz(self, ansatz) -> np.ndarray:
        if self.ansatz_parameters_from_instance is None:
            raise TypeError(
                "ansatz_parameters_from_instance is required to invert an ansatz"
            )
        params = np.asarray(self.ansatz_parameters_from_instance(ansatz), dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(
                "ansatz_parameters_from_instance returned shape "
                f"{params.shape}; expected {(self.n_params,)}."
            )
        return params

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int] | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        if nelec is None:
            if self.default_nelec is None:
                raise ValueError("nelec is required for a generic gate sequence")
            nelec = self.default_nelec
        nelec = tuple(int(x) for x in nelec)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec,
                nelec=nelec,
                copy=True,
            )

        return func

    def apply(
        self,
        reference: object,
        nelec: tuple[int, int] | None = None,
    ):
        if nelec is None:
            nelec = self.default_nelec
        from xquces.gcr.references import apply_ansatz_parameterization

        resolved = None if nelec is None else tuple(int(x) for x in nelec)
        return apply_ansatz_parameterization(self, reference, resolved)

