from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.ansatz.parameters import ParameterBlock, ParameterView, parameter_view


@dataclass(frozen=True)
class OrbitalRotationGate:
    name: str
    chart: object
    norb: int

    @property
    def n_params(self) -> int:
        return int(self.chart.n_params(self.norb))

    def blocks(self, prefix: str = "") -> tuple[ParameterBlock, ...]:
        name = self.name if prefix == "" else f"{prefix}.{self.name}"
        return (
            ParameterBlock(
                name=name,
                start=0,
                stop=self.n_params,
                shape=(self.n_params,),
                kind="orbital",
            ),
        )

    def instance_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self.chart.unitary_from_parameters(params, self.norb)


@dataclass(frozen=True)
class DiagonalCorrelatorGate:
    name: str
    chart: object

    @property
    def n_params(self) -> int:
        return int(self.chart.n_params)

    def blocks(self, prefix: str = "") -> tuple[ParameterBlock, ...]:
        gate_prefix = self.name if prefix == "" else f"{prefix}.{self.name}"
        return self.chart.blocks(gate_prefix)

    def instance_from_parameters(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self.chart.coefficients_from_parameters(params)


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

    @property
    def n_params(self) -> int:
        return sum(int(gate.n_params) for gate in self.gates)

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        blocks: list[ParameterBlock] = []
        offset = 0
        for gate in self.gates:
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
            params = np.asarray(self.native_parameters_from_public(params), dtype=np.float64)
            if params.shape != (self.n_params,):
                raise ValueError(
                    "native_parameters_from_public returned shape "
                    f"{params.shape}; expected {(self.n_params,)}."
                )
        instances = []
        start = 0
        for gate in self.gates:
            stop = start + int(gate.n_params)
            instances.append(gate.instance_from_parameters(params[start:stop]))
            start = stop
        return tuple(instances)

    def ansatz_from_parameters(self, params: np.ndarray):
        if self.ansatz_builder is None:
            raise TypeError("ansatz_builder is required to build an ansatz")
        return self.ansatz_builder(self.gate_instances_from_parameters(params))


__all__ = [
    "DiagonalCorrelatorGate",
    "GateSequenceParameterization",
    "OrbitalRotationGate",
]
