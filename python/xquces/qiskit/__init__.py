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
"""Pass manager recommended for the Qiskit transpiler ``pre_init`` stage.

See :func:`pre_init_passes` for the transpiler passes included in this pass manager.
"""

from xquces.qiskit.pass_managers import (
    GCR2Layout,
    GCR2PairUCCDLayout,
    GCR2PairUCCDPassManagerResult,
    GCR2PassManagerResult,
    generate_gcr2_pair_uccd_pass_manager,
    generate_gcr2_pairuccd_pass_manager,
    generate_gcr2_pass_manager,
)

__all__ = [
    "CircuitStats",
    "CircuitStatsJob",
    "GCR2Layout",
    "GCR2PairUCCDLayout",
    "GCR2PairUCCDPassManagerResult",
    "GCR2PassManagerResult",
    "PRE_INIT",
    "circuit_stats",
    "format_count_ops",
    "generate_gcr2_pair_uccd_pass_manager",
    "generate_gcr2_pairuccd_pass_manager",
    "generate_gcr2_pass_manager",
    "native_backend",
    "pre_init_passes",
    "pretty_print_circuit_stats",
    "total_gate_count",
    "transpile_to_native",
    "two_qubit_gate_count",
]
