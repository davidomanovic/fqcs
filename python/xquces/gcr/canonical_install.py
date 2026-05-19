from __future__ import annotations

from xquces.gcr.canonical import (
    IGCRAnsatz,
    IGCRDiagonalCoefficients,
    install_igcr_legacy_adapters,
)
from xquces.gcr.canonical_layering import as_legacy_layered_igcr_ansatz


def install_igcr_parameterization_adapters() -> None:
    """Install compatibility adapters for the canonical iGCR model.

    Legacy ansatz classes gain ``to_generic``/``from_generic`` adapters.  The
    spin-restricted parameterization builders and ansatz-embedding callbacks now
    route through the canonical always-layered model before returning the legacy
    compatibility objects expected by existing callers.
    """

    install_igcr_legacy_adapters()

    from xquces.gcr import igcr as legacy_igcr

    def _as_layered_igcr2_spin_restricted_ansatz(ansatz, layers):
        if getattr(ansatz, "is_spin_restricted", True) is False:
            raise TypeError("expected a spin-restricted ansatz")
        return as_legacy_layered_igcr_ansatz(ansatz, layers, order=2)

    def _as_layered_igcr3_spin_restricted_ansatz(ansatz, layers):
        return as_legacy_layered_igcr_ansatz(ansatz, layers, order=3)

    def _as_layered_igcr4_spin_restricted_ansatz(ansatz, layers):
        return as_legacy_layered_igcr_ansatz(ansatz, layers, order=4)

    def _igcr2_one_layer_ansatz_from_core(self, diagonal, left, right):
        generic = IGCRAnsatz(
            order=2,
            diagonals=(IGCRDiagonalCoefficients.from_igcr2_spec(diagonal),),
            rotations=(left, right),
            nocc=self.nocc,
        )
        return generic.to_igcr2_ansatz()

    def _igcr2_layered_ansatz_from_core(self, diagonals, rotations):
        generic = IGCRAnsatz(
            order=2,
            diagonals=tuple(
                IGCRDiagonalCoefficients.from_igcr2_spec(diagonal)
                for diagonal in diagonals
            ),
            rotations=rotations,
            nocc=self.nocc,
        )
        return generic.to_igcr2_ansatz()

    def _igcr3_one_layer_ansatz_from_core(self, diagonal, left, right):
        generic = IGCRAnsatz(
            order=3,
            diagonals=(IGCRDiagonalCoefficients.from_igcr3_spec(diagonal),),
            rotations=(left, right),
            nocc=self.nocc,
        )
        return generic.to_igcr3_ansatz()

    def _igcr3_layered_ansatz_from_core(self, diagonals, rotations):
        generic = IGCRAnsatz(
            order=3,
            diagonals=tuple(
                IGCRDiagonalCoefficients.from_igcr3_spec(diagonal)
                for diagonal in diagonals
            ),
            rotations=rotations,
            nocc=self.nocc,
        )
        return generic.to_igcr3_ansatz()

    def _igcr4_one_layer_ansatz_from_core(self, diagonal, left, right):
        generic = IGCRAnsatz(
            order=4,
            diagonals=(IGCRDiagonalCoefficients.from_igcr4_spec(diagonal),),
            rotations=(left, right),
            nocc=self.nocc,
        )
        return generic.to_igcr4_ansatz()

    def _igcr4_layered_ansatz_from_core(self, diagonals, rotations):
        generic = IGCRAnsatz(
            order=4,
            diagonals=tuple(
                IGCRDiagonalCoefficients.from_igcr4_spec(diagonal)
                for diagonal in diagonals
            ),
            rotations=rotations,
            nocc=self.nocc,
        )
        return generic.to_igcr4_ansatz()

    legacy_igcr._as_layered_igcr2_spin_restricted_ansatz = (
        _as_layered_igcr2_spin_restricted_ansatz
    )
    legacy_igcr._as_layered_igcr3_spin_restricted_ansatz = (
        _as_layered_igcr3_spin_restricted_ansatz
    )
    legacy_igcr._as_layered_igcr4_spin_restricted_ansatz = (
        _as_layered_igcr4_spin_restricted_ansatz
    )
    legacy_igcr.IGCR2SpinRestrictedParameterization._one_layer_ansatz_from_core = (
        _igcr2_one_layer_ansatz_from_core
    )
    legacy_igcr.IGCR2SpinRestrictedParameterization._layered_ansatz_from_core = (
        _igcr2_layered_ansatz_from_core
    )
    legacy_igcr.IGCR3SpinRestrictedParameterization._one_layer_ansatz_from_core = (
        _igcr3_one_layer_ansatz_from_core
    )
    legacy_igcr.IGCR3SpinRestrictedParameterization._layered_ansatz_from_core = (
        _igcr3_layered_ansatz_from_core
    )
    legacy_igcr.IGCR4SpinRestrictedParameterization._one_layer_ansatz_from_core = (
        _igcr4_one_layer_ansatz_from_core
    )
    legacy_igcr.IGCR4SpinRestrictedParameterization._layered_ansatz_from_core = (
        _igcr4_layered_ansatz_from_core
    )


__all__ = ["install_igcr_parameterization_adapters"]
