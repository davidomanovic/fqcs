from __future__ import annotations

from xquces.gcr.canonical import (
    IGCRAnsatz,
    IGCRDiagonalCoefficients,
    install_igcr_legacy_adapters,
)


def install_igcr_parameterization_adapters() -> None:
    """Install compatibility adapters for the canonical iGCR model.

    This is intentionally narrow for stage 13: legacy ansatz classes gain
    ``to_generic``/``from_generic`` adapters, and IGCR3 parameterization core
    builders construct the canonical always-layered model internally before
    returning the legacy compatibility object.
    """

    install_igcr_legacy_adapters()

    from xquces.gcr import igcr as legacy_igcr

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

    legacy_igcr.IGCR3SpinRestrictedParameterization._one_layer_ansatz_from_core = (
        _igcr3_one_layer_ansatz_from_core
    )
    legacy_igcr.IGCR3SpinRestrictedParameterization._layered_ansatz_from_core = (
        _igcr3_layered_ansatz_from_core
    )


__all__ = ["install_igcr_parameterization_adapters"]
