from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyscf.fci
import pyscf.lib

from xquces.gcr import PairUCCDIGCRParameterization
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces import utils as xq_utils


BASIS = "sto-3g"
BOND_LENGTHS = np.arange(0.7, 3.0, 0.1)
N_HYDROGEN = 4
IGCR_ORDER = 2
REFERENCE_KIND = "exponential"
N_THREADS = 1

OUT_CSV = Path(__file__).with_name("output") / "h4_pair_uccd_igcr2_seed_curve.csv"


pyscf.lib.num_threads(N_THREADS)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

rows = []
for R in BOND_LENGTHS:
    mol = xq_utils.build_hydrogen_chain(float(R), N_HYDROGEN, BASIS)
    mf = xq_utils.run_rhf(mol)
    ccsd = xq_utils.run_rccsd(mf)
    e_fci, _ = pyscf.fci.FCI(mol, mf.mo_coeff).kernel()

    hamiltonian = MolecularHamiltonianLinearOperator.from_scf(mf)
    norb = hamiltonian.norb
    nelec = hamiltonian.nelec

    pair_uccd_igcr = PairUCCDIGCRParameterization(
        norb=norb,
        nocc=nelec[0],
        order=IGCR_ORDER,
        reference_kind=REFERENCE_KIND,
    )
    params = pair_uccd_igcr.parameters_from_t_amplitudes(ccsd.t2, t1=ccsd.t1)
    state = pair_uccd_igcr.state_from_parameters(params)

    rows.append(
        {
            "R": float(R),
            "E_FCI": float(e_fci),
            "E_HF": float(mf.e_tot),
            "E_CCSD": float(ccsd.e_tot),
            "E_pair_UCCD_iGCR2_seed": hamiltonian.expectation(state),
        }
    )

header = ("R", "E_FCI", "E_HF", "E_CCSD", "E_pair_UCCD_iGCR2_seed")
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(",".join(header))
for row in rows:
    print(",".join(f"{row[key]:.12f}" for key in header))
print(f"Wrote {OUT_CSV}")
