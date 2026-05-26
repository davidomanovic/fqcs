"""Compatibility wrapper for GCR2 pass manager helpers.

Use :mod:`xquces.qiskit.pass_managers` for new code.
"""

from __future__ import annotations

from xquces.qiskit.pass_managers import (
    GCR2PairUCCDLayout,
    GCR2PairUCCDPassManagerResult,
    generate_gcr2_pair_uccd_pass_manager,
    generate_gcr2_pairuccd_pass_manager,
)

__all__ = [
    "GCR2PairUCCDLayout",
    "GCR2PairUCCDPassManagerResult",
    "generate_gcr2_pair_uccd_pass_manager",
    "generate_gcr2_pairuccd_pass_manager",
]
