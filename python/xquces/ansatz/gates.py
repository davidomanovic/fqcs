from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xquces.ansatz.parameters import ParameterBlock


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

