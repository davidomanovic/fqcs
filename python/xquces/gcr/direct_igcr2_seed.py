from __future__ import annotations

import numpy as np

from xquces.gcr import igcr as _igcr


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

    bases = tuple(np.asarray(u, dtype=np.complex128) for u in df.orbital_rotations)
    norb = bases[0].shape[0]
    rt1 = (
        np.eye(norb, dtype=np.complex128)
        if df.final_orbital_rotation is None
        else np.asarray(df.final_orbital_rotation, dtype=np.complex128)
    )

    diagonals: list[_igcr.IGCR2SpinRestrictedSpec] = []
    for j in reversed(df.diagonal_coulomb_mats):
        pair = np.array(j, dtype=np.float64, copy=True)
        np.fill_diagonal(pair, 0.0)
        diagonals.append(_igcr.IGCR2SpinRestrictedSpec(pair=pair))

    rotations: list[np.ndarray] = [bases[-1]]
    for ell in range(layers - 1, 0, -1):
        rotations.append(bases[ell].conj().T @ bases[ell - 1])
    rotations.append(_igcr.exact_reference_ov_unitary(bases[0].conj().T @ rt1, nocc))

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
