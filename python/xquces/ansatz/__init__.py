"""Shared ansatz building blocks."""

from xquces.ansatz.blocks import parameter_blocks, random_parameters
from xquces.ansatz.gates import DiagonalCorrelatorGate, OrbitalRotationGate
from xquces.ansatz.jacobian import (
    make_state_jacobian,
    make_state_subspace_jacobian,
    make_state_vjp,
)
from xquces.ansatz.parameters import ParameterBlock, ParameterView, parameter_view
from xquces.ansatz.sequence import GateSequenceParameterization

__all__ = [
    "DiagonalCorrelatorGate",
    "GateSequenceParameterization",
    "OrbitalRotationGate",
    "ParameterBlock",
    "ParameterView",
    "make_state_jacobian",
    "make_state_subspace_jacobian",
    "make_state_vjp",
    "parameter_blocks",
    "parameter_view",
    "random_parameters",
]
