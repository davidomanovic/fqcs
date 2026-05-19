import numpy as np
import pytest

import xquces
from xquces.gcr import IGCR2SpinBalancedParameterization
from xquces.gcr.canonical_transform import (
    relabel_igcr_ansatz_orbitals,
    transport_igcr_ansatz_orbitals,
)


def test_top_level_generic_transform_aliases_are_published():
    assert xquces.relabel_igcr_ansatz_orbitals is relabel_igcr_ansatz_orbitals
    assert xquces.transport_igcr_ansatz_orbitals is transport_igcr_ansatz_orbitals


def test_top_level_legacy_transform_aliases_are_patched():
    assert xquces.relabel_igcr2_ansatz_orbitals.__module__ == "xquces.gcr.canonical_install"
    assert xquces.relabel_igcr3_ansatz_orbitals.__module__ == "xquces.gcr.canonical_install"
    assert xquces.relabel_igcr4_ansatz_orbitals.__module__ == "xquces.gcr.canonical_install"
    assert xquces.transport_igcr2_ansatz_orbitals.__module__ == "xquces.gcr.canonical_install"
    assert xquces.transport_igcr3_ansatz_orbitals.__module__ == "xquces.gcr.canonical_install"
    assert xquces.transport_igcr4_ansatz_orbitals.__module__ == "xquces.gcr.canonical_install"


def test_spin_balanced_igcr2_relabel_and_transport_are_not_canonicalized():
    param = IGCR2SpinBalancedParameterization(norb=4, nocc=2)
    ansatz = param.ansatz_from_parameters(np.zeros(param.n_params))

    with pytest.raises(TypeError, match="spin-restricted"):
        xquces.relabel_igcr2_ansatz_orbitals(ansatz, np.array([0, 1, 2, 3]))

    with pytest.raises(TypeError, match="spin-restricted"):
        xquces.transport_igcr2_ansatz_orbitals(ansatz, np.eye(4, dtype=np.complex128))
