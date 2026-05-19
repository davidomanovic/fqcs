from __future__ import annotations

from xquces.gcr.canonical import (
    IGCRAnsatz,
    IGCRDiagonalCoefficients,
    install_igcr_legacy_adapters,
)
from xquces.gcr.canonical_layering import as_legacy_layered_igcr_ansatz
from xquces.gcr.canonical_transform import (
    relabel_legacy_igcr_ansatz_orbitals,
    transport_legacy_igcr_ansatz_orbitals,
)


def install_igcr_parameterization_adapters() -> None:
    """Install compatibility adapters for the canonical iGCR model.

    Legacy ansatz classes gain ``to_generic``/``from_generic`` adapters.  The
    spin-restricted parameterization builders, ansatz-embedding callbacks, and
    relabel/transport wrappers now route through the canonical always-layered
    model before returning the legacy compatibility objects expected by existing
    callers.
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

    def _relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases=None):
        if getattr(ansatz, "is_spin_restricted", True) is False:
            raise TypeError("expected a spin-restricted ansatz")
        return relabel_legacy_igcr_ansatz_orbitals(
            ansatz,
            old_for_new,
            phases,
            order=2,
        )

    def _relabel_igcr3_ansatz_orbitals(ansatz, old_for_new, phases=None):
        return relabel_legacy_igcr_ansatz_orbitals(
            ansatz,
            old_for_new,
            phases,
            order=3,
        )

    def _relabel_igcr4_ansatz_orbitals(ansatz, old_for_new, phases=None):
        return relabel_legacy_igcr_ansatz_orbitals(
            ansatz,
            old_for_new,
            phases,
            order=4,
        )

    def _transport_igcr2_ansatz_orbitals(ansatz, basis_change):
        if getattr(ansatz, "is_spin_restricted", True) is False:
            raise TypeError("expected a spin-restricted ansatz")
        return transport_legacy_igcr_ansatz_orbitals(
            ansatz,
            basis_change,
            order=2,
        )

    def _transport_igcr3_ansatz_orbitals(ansatz, basis_change):
        return transport_legacy_igcr_ansatz_orbitals(
            ansatz,
            basis_change,
            order=3,
        )

    def _transport_igcr4_ansatz_orbitals(ansatz, basis_change):
        return transport_legacy_igcr_ansatz_orbitals(
            ansatz,
            basis_change,
            order=4,
        )

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
    legacy_igcr.relabel_igcr2_ansatz_orbitals = _relabel_igcr2_ansatz_orbitals
    legacy_igcr.relabel_igcr3_ansatz_orbitals = _relabel_igcr3_ansatz_orbitals
    legacy_igcr.relabel_igcr4_ansatz_orbitals = _relabel_igcr4_ansatz_orbitals
    legacy_igcr.transport_igcr2_ansatz_orbitals = _transport_igcr2_ansatz_orbitals
    legacy_igcr.transport_igcr3_ansatz_orbitals = _transport_igcr3_ansatz_orbitals
    legacy_igcr.transport_igcr4_ansatz_orbitals = _transport_igcr4_ansatz_orbitals
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
