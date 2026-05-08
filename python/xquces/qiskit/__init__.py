from __future__ import annotations

from qiskit.transpiler import PassManager

from xquces.qiskit.transpiler_stages import pre_init_passes
from xquces.qiskit.utils import (
    CircuitStats,
    CircuitStatsJob,
    circuit_stats,
    format_count_ops,
    native_backend,
    pretty_print_circuit_stats,
    total_gate_count,
    transpile_to_native,
    two_qubit_gate_count,
)

PRE_INIT = PassManager(list(pre_init_passes()))

from xquces.qiskit.gcr2_pair_uccd_pass_manager import (
    GCR2PairUCCDLayout,
    GCR2PairUCCDPassManagerResult,
    generate_gcr2_pair_uccd_pass_manager,
)

__all__ = [
    "CircuitStats",
    "CircuitStatsJob",
    "GCR2PairUCCDLayout",
    "GCR2PairUCCDPassManagerResult",
    "PRE_INIT",
    "circuit_stats",
    "format_count_ops",
    "generate_gcr2_pair_uccd_pass_manager",
    "native_backend",
    "pre_init_passes",
    "pretty_print_circuit_stats",
    "total_gate_count",
    "transpile_to_native",
    "two_qubit_gate_count",
]
