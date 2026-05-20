from __future__ import annotations

import ffsim
import numpy as np

from xquces.gcr.restricted_model import (
    IGCR2Ansatz,
    IGCR2LayeredAnsatz,
    IGCR2SpinRestrictedSpec,
    reduce_spin_restricted,
)
from xquces.gcr.utils import (
    _diag_unitary,
    _restricted_left_phase_vector,
    exact_reference_ov_unitary,
)
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz, UCJLayer


def _igcr2_layered_spin_restricted_ansatz_from_ucj(
    ansatz: UCJAnsatz,
    nocc: int,
    layers: int,
) -> IGCR2LayeredAnsatz:
    if not ansatz.is_spin_restricted:
        raise TypeError("expected a spin-restricted UCJ ansatz")
    if ansatz.n_layers > layers:
        raise ValueError(
            "UCJ seed has more layers than the IGCR2 parameterization; "
            "increase layers or use a shallower UCJ seed"
        )
    norb = ansatz.norb
    identity = np.eye(norb, dtype=np.complex128)
    final = (
        identity
        if ansatz.final_orbital_rotation is None
        else np.asarray(ansatz.final_orbital_rotation, dtype=np.complex128)
    )

    diagonals = []
    rotations = []
    active = list(reversed(ansatz.layers))
    previous_base = final
    for layer in active:
        if not isinstance(layer.diagonal, SpinRestrictedSpec):
            raise TypeError("expected a spin-restricted UCJ layer")
        diagonal = reduce_spin_restricted(layer.diagonal)
        phase_vec = _restricted_left_phase_vector(layer.diagonal.double_params, nocc)
        base = np.asarray(layer.orbital_rotation, dtype=np.complex128)
        diagonals.append(diagonal)
        rotations.append(previous_base @ base @ _diag_unitary(phase_vec))
        previous_base = base.conj().T

    rotations.append(exact_reference_ov_unitary(previous_base, nocc))
    while len(diagonals) < layers:
        diagonals.append(
            IGCR2SpinRestrictedSpec(pair=np.zeros((norb, norb), dtype=np.float64))
        )
        rotations.append(identity)

    return IGCR2LayeredAnsatz(
        diagonals=tuple(diagonals),
        rotations=tuple(rotations),
        nocc=nocc,
    )


def layered_igcr2_from_ucj_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    layers: int = 1,
    nocc: int | None = None,
    **df_options,
) -> IGCR2Ansatz | IGCR2LayeredAnsatz:
    """Build an iGCR-2 ansatz by lifting ffsim's UCJ t-amplitude seed.

    Calls ffsim's spin-restricted UCJ seed with n_reps=layers to obtain L
    double-factorization terms (U_1, J_1), ..., (U_L, J_L) and the final
    orbital rotation U_F from t1.
    Each diagonal J_l is reduced independently (iGCR-2 redundancy removal).
    The resulting layered iGCR-2 seed is ordered so it prepares the same state
    as the corresponding UCJ seed, up to a global phase on the reference.

    Returns IGCR2Ansatz for layers=1, IGCR2LayeredAnsatz for layers>1.
    Extra keyword arguments are forwarded to
    UCJOpSpinRestricted.from_t_amplitudes.
    """
    t2 = np.asarray(t2, dtype=np.float64)
    if nocc is None:
        nocc = t2.shape[0]

    ucj_op = ffsim.variational.UCJOpSpinRestricted.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=layers,
        **df_options,
    )

    ucj_layers = []
    for J_l, U_l in zip(ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations):
        pair_l = np.array(J_l, dtype=np.float64, copy=True)
        np.fill_diagonal(pair_l, 0.0)
        ucj_layers.append(
            UCJLayer(
                diagonal=SpinRestrictedSpec(
                    double_params=np.diag(J_l).copy(),
                    pair_params=pair_l,
                ),
                orbital_rotation=np.asarray(U_l, dtype=np.complex128),
            )
        )
    ucj = UCJAnsatz(
        tuple(ucj_layers),
        final_orbital_rotation=None
        if ucj_op.final_orbital_rotation is None
        else np.asarray(ucj_op.final_orbital_rotation, dtype=np.complex128),
    )
    seeded = _igcr2_layered_spin_restricted_ansatz_from_ucj(ucj, nocc, layers)

    if layers == 1:
        return IGCR2Ansatz(
            diagonal=seeded.diagonals[0],
            left=seeded.rotations[0],
            right=seeded.rotations[1],
            nocc=nocc,
        )
    return seeded


def layered_igcr2_from_ccsd_t_amplitudes(
    t2: np.ndarray,
    t1: np.ndarray | None = None,
    *,
    layers: int = 1,
    nocc: int | None = None,
    **df_options,
) -> IGCR2Ansatz | IGCR2LayeredAnsatz:
    """Compatibility alias for the UCJ-lift t-amplitude seed.

    New iGCR-2-native code should use
    :meth:`IGCR2SpinRestrictedParameterization.parameters_from_t_amplitudes`
    for the direct one-layer seed, or
    :meth:`IGCR2SpinRestrictedParameterization.parameters_from_ucj_t_amplitudes`
    when the UCJ-lift representative is intended.
    """
    return layered_igcr2_from_ucj_t_amplitudes(
        t2, t1=t1, layers=layers, nocc=nocc, **df_options
    )

