from __future__ import annotations

from xquces.gcr import igcr
from xquces.gcr import restricted_model


def test_spin_restricted_model_objects_remain_igcr_reexports():
    names = (
        "IGCR2SpinRestrictedSpec",
        "IGCR2Ansatz",
        "IGCR2LayeredAnsatz",
        "IGCR3SpinRestrictedSpec",
        "IGCR3Ansatz",
        "IGCR3LayeredAnsatz",
        "IGCR4SpinRestrictedSpec",
        "IGCR4Ansatz",
        "IGCR4LayeredAnsatz",
        "apply_igcr3_spin_restricted_diagonal",
        "apply_igcr4_spin_restricted_diagonal",
        "reduce_spin_restricted",
        "spin_restricted_triples_seed_from_pair_params",
        "spin_restricted_quartic_seed_from_pair_params",
    )

    for name in names:
        assert getattr(igcr, name) is getattr(restricted_model, name)
