import xquces

from xquces.gcr.canonical_transform import (
    relabel_igcr_ansatz_orbitals,
    transport_igcr_ansatz_orbitals,
)


def test_top_level_generic_transform_aliases_are_published():
    assert xquces.relabel_igcr_ansatz_orbitals is relabel_igcr_ansatz_orbitals
    assert xquces.transport_igcr_ansatz_orbitals is transport_igcr_ansatz_orbitals
