"""Compatibility exports for the in-tree iGCR-2 seed implementation.

The CCSD-to-iGCR2 seed is implemented directly in :mod:`xquces.gcr.igcr`.
This module intentionally has no import-time monkey-patch side effects.
"""

from __future__ import annotations

from xquces.gcr.igcr import layered_igcr2_from_ccsd_t_amplitudes

__all__ = ["layered_igcr2_from_ccsd_t_amplitudes"]
