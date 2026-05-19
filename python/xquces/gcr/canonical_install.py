from __future__ import annotations

from xquces.gcr.canonical import (
    IGCRAnsatz,
    IGCRDiagonalCoefficients,
    install_igcr_legacy_adapters,
)


def install_igcr_parameterization_adapters() -> None:
    """Install compatibility adapters for the canonical iGCR model.

    Legacy ansatz classes gain ``to_generic``/``from_generic`` adapters, and the
    routed spin-restricted parameterization core builders construct the
    canonical always-layered model internally before returning the legacy
    compatibility object expected by existing callers.
    """

    install_igcr_legacy_adapters()

    from xquces.gcr import igcr as legacy_igcr

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
