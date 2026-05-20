from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyscf.fci
import pyscf.lib

from xquces.gcr.igcr import IGCRSpinRestrictedParameterization
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.states import hartree_fock_state
from xquces import utils as xq_utils


BASIS = "sto-3g"
BOND_LENGTHS = np.arange(0.5, 2.5, 0.1)
IGCR_ORDER = 2
N_THREADS = 1

OUT_CSV = Path(__file__).with_name("output") / "h2_igcr2_seed_curve.csv"


pyscf.lib.num_threads(N_THREADS)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

rows = []
for R in BOND_LENGTHS:
    mol = xq_utils.build_diatom("H", "H", float(R), BASIS, symmetry=False)
    mf = xq_utils.run_rhf(mol)
    ccsd = xq_utils.run_rccsd(mf)
    e_fci, _ = pyscf.fci.FCI(mol, mf.mo_coeff).kernel()

    hamiltonian = MolecularHamiltonianLinearOperator.from_scf(mf)
    norb = hamiltonian.norb
    nelec = hamiltonian.nelec

    reference = hartree_fock_state(norb, nelec)
    igcr = IGCRSpinRestrictedParameterization(
        norb=norb,
        nocc=nelec[0],
        order=IGCR_ORDER,
    )
    params = igcr.parameters_from_t_amplitudes(ccsd.t2, t1=ccsd.t1)
    state = igcr.ansatz_from_parameters(params).apply(reference, nelec, copy=True)

    rows.append(
        {
            "R": float(R),
            "E_FCI": float(e_fci),
            "E_HF": float(mf.e_tot),
            "E_CCSD": float(ccsd.e_tot),
            "E_iGCR2_seed": hamiltonian.expectation(state),
        }
    )

header = ("R", "E_FCI", "E_HF", "E_CCSD", "E_iGCR2_seed")
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(",".join(header))
for row in rows:
    print(",".join(f"{row[key]:.12f}" for key in header))
print(f"Wrote {OUT_CSV}")
