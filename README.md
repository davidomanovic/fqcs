# xquces

Experimental quantum algorithms for electronic-structure calculations.

`xquces` provides state-vector tools for generalized correlated rotations (GCR-k) for orders $k=2,3,4$,
irreducible GCR (iGCR-k), UCJ-style ansatzes, pair-UCCD references, and small
variational workflows for quantum chemistry. The core abstractions are
parameterizations: a parameterization owns a flat real parameter vector and
turns it into an ansatz object that can be applied to a fixed
`(n_alpha, n_beta)` fermionic sector.

The package is research-oriented. APIs are intentionally lightweight and may
change as the ansatzes evolve.

## Features

- Spin-restricted and spin-balanced GCR/iGCR ansatzes.
- Order-2, order-3, and order-4 spin-restricted iGCR parameterizations.
- Layered iGCR ansatzes with optional shared diagonal parameters.
- Pair-UCCD and product pair-UCCD reference parameterizations.
- PySCF-backed molecular Hamiltonian linear operators.
- Qiskit gate and pass-manager utilities for circuit construction.
- Rust-accelerated kernels for diagonal correlators and orbital rotations.

## Installation

This repository builds a Python package backed by a PyO3/Rust extension, so a
Rust toolchain is required.

First, compile the Rust kernel
```bash
maturin develop -r
```

Then
```bash
python3 -m pip install .
```

The molecular examples use PySCF. The notebooks and circuit utilities also use
`ffsim` and `qiskit`.

```bash
python3 -m pip install pyscf ffsim qiskit
```

## Quick Start: GCR-2 For H2

This example builds an H2 Hamiltonian with PySCF, constructs a one-layer
spin-restricted GCR-2/iGCR-2 ansatz, applies it to the Hartree-Fock reference,
and evaluates the energy.

```python
import numpy as np
import pyscf

import xquces

# Build an H2/STO-3G Hamiltonian in the molecular-orbital basis
mol = pyscf.gto.Mole()
mol.build(atom=[["H", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 0.735)]], basis="sto-3g")
scf = pyscf.scf.RHF(mol).run(verbose=0)

hamiltonian = xquces.MolecularHamiltonianLinearOperator.from_scf(scf)
norb, nelec = hamiltonian.norb, hamiltonian.nelec
nocc = nelec[0]

# Construct a one-layer spin-restricted GCR-2 parameterization
gcr2 = xquces.IGCR(order=2, norb=norb, nocc=nocc, layers=1)
print(gcr2.n_params)  # 5

# Pick a small random point in the ansatz and prepare the state
params = gcr2.random_parameters(scale=1.0e-2, seed=1234)
reference = xquces.hartree_fock_state(norb, nelec)
vec = gcr2.ansatz_from_parameters(params).apply(reference, nelec=nelec)

# Evaluate <psi|H|psi>.
linop = hamiltonian.as_linear_operator()
energy = np.vdot(vec, linop @ vec).real
print(energy)  # -1.1161703235258413
```

For optimization code, `params_to_vec` gives the same state-preparation map as a
callable:

```python
params_to_vec = gcr2.params_to_vec(reference, nelec)
vec = params_to_vec(params)
```

Have a look at the `examples/` directory for how to initialize from CCSD amplitudes or how to run VQE to variationally optimize the ansatze.

## Core API

- `xquces.IGCR(order, norb, nocc, ...)` creates the recommended GCR/iGCR
  parameterization. Use `order=2`, `3`, or `4`.
- `parameterization.n_params` is the length of the flat parameter vector.
- `parameterization.parameter_blocks()` returns named slices such as `left`,
  `pair`, `middle`, and `right`.
- `parameterization.random_parameters(...)` creates a small random parameter
  vector, optionally restricted to selected blocks.
- `parameterization.ansatz_from_parameters(params)` returns an ansatz object.
- `ansatz.apply(reference, nelec=...)` applies the ansatz to a state vector.
- `xquces.MolecularHamiltonianLinearOperator.from_scf(scf)` wraps a PySCF RHF
  calculation as a Hamiltonian with `matvec`, `expectation`, and
  `as_linear_operator()`.
- `xquces.PairUCCD_GCR(...)` composes a pair-UCCD-style reference with an iGCR
  ansatz.

## Internal Model

State vectors are stored in fixed alpha/beta electron-number sectors. Orbital
rotations act on those sectors through Givens decompositions, while the
diagonal correlators are applied by the native Rust extension. The public
parameterizations hide the chart details: they map a flat real vector to
orbital rotations plus diagonal coefficients, then package those pieces as an
ansatz object.

For spin-restricted iGCR-2, the diagonal layer is a symmetric pair-coupling
matrix with the one-body double terms absorbed into orbital phases. Higher
orders add cubic and quartic diagonal sectors while preserving the same
parameterization-to-ansatz workflow.

## Development

Run the tests from the repository root after installing the package and test
dependencies:

```bash
python3 -m pytest
```

Useful directories:

- `python/xquces/gcr`: GCR/iGCR ansatz models, charts, and presets.
- `python/xquces/ansatz`: shared flat-parameter block and sequence utilities.
- `python/xquces/qiskit`: Qiskit gates and transpiler helpers.
- `src`: Rust kernels exposed as `xquces._lib`.
- `tests`: regression tests for parameterizations, seeds, circuits, and
  Hamiltonian workflows.
