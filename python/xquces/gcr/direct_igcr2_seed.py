from __future__ import annotations

import numpy as np

from xquces.gcr import igcr as _igcr
from xquces.ucj.model import SpinRestrictedSpec


def layered_igcr2_from_ccsd_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    layers: int = 1,
    nocc: int | None = None,
    **df_options,
) -> _igcr.IGCR2Ansatz | _igcr.IGCR2LayeredAnsatz:
    t2 = np.asarray(t2, dtype=np.float64)
    if nocc is None:
        nocc = t2.shape[0]

    df = _igcr.factorize_ccsd_t_amplitudes(
        t2,
        t1=t1,
        n_reps=layers,
        **df_options,
    )

    if len(df.orbital_rotations) == 0:
        raise ValueError("double factorization returned no orbital rotations")
    if len(df.orbital_rotations) != layers:
        raise ValueError("double factorization returned an unexpected number of layers")
    if len(df.diagonal_coulomb_mats) != layers:
        raise ValueError("double factorization returned an unexpected number of diagonal blocks")

    norb = df.orbital_rotations[0].shape[0]
    identity = np.eye(norb, dtype=np.complex128)
    final = (
        identity
        if df.final_orbital_rotation is None
        else np.asarray(df.final_orbital_rotation, dtype=np.complex128)
    )

    bases = [np.asarray(u, dtype=np.complex128) for u in df.orbital_rotations]
    diagonals = []
    left_factors = []

    for J_l, U_l in zip(df.diagonal_coulomb_mats, bases):
        double_l = np.diag(J_l).copy()
        pair_l = np.array(J_l, dtype=np.float64, copy=True)
        np.fill_diagonal(pair_l, 0.0)
        spec_l = SpinRestrictedSpec(
            double_params=double_l,
            pair_params=pair_l,
        )
        diagonals.append(_igcr.reduce_spin_restricted(spec_l))
        phase_l = _igcr._restricted_left_phase_vector(double_l, nocc)
        left_factors.append(U_l @ _igcr._diag_unitary(phase_l))

    rotations = [left_factors[0]]
    for idx in range(1, layers):
        rotations.append(bases[idx - 1].conj().T @ left_factors[idx])
    rotations.append(bases[-1].conj().T @ final)

    if layers == 1:
        return _igcr.IGCR2Ansatz(
            diagonal=diagonals[0],
            left=rotations[0],
            right=rotations[1],
            nocc=nocc,
        )

    return _igcr.IGCR2LayeredAnsatz(
        diagonals=tuple(diagonals),
        rotations=tuple(rotations),
        nocc=nocc,
    )


def _parameters_from_t_amplitudes(
    self: _igcr.IGCR2SpinRestrictedParameterization,
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    **df_options,
) -> np.ndarray:
    ansatz = layered_igcr2_from_ccsd_t_amplitudes(
        t2,
        t1=t1,
        layers=self.layers,
        nocc=self.nocc,
        **df_options,
    )
    return self.parameters_from_ansatz(ansatz)


def _ansatz_from_t_amplitudes(
    cls,
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    **df_options,
):
    nocc = np.asarray(t2).shape[0]
    ansatz = layered_igcr2_from_ccsd_t_amplitudes(
        t2,
        t1=t1,
        layers=1,
        nocc=nocc,
        **df_options,
    )
    if not isinstance(ansatz, cls):
        raise TypeError("single-layer direct seed did not produce IGCR2Ansatz")
    return ansatz


def _ansatz_from_t_restricted(cls, t2: np.ndarray, **kwargs):
    t1 = kwargs.pop("t1", None)
    return _ansatz_from_t_amplitudes(cls, t2, t1=t1, **kwargs)


_igcr.layered_igcr2_from_ccsd_t_amplitudes = layered_igcr2_from_ccsd_t_amplitudes
_igcr.IGCR2SpinRestrictedParameterization.parameters_from_t_amplitudes = _parameters_from_t_amplitudes
_igcr.IGCR2Ansatz.from_t_amplitudes = classmethod(_ansatz_from_t_amplitudes)
_igcr.IGCR2Ansatz.from_t_restricted = classmethod(_ansatz_from_t_restricted)
