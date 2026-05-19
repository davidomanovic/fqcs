from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import ffsim
import numpy as np
import pyscf.lib

from xquces.gcr.igcr import (
    CCSDResidualSeedInfo as LegacyCCSDResidualSeedInfo,
    IGCR2SpinRestrictedParameterization,
    IGCR3SpinRestrictedParameterization,
    IGCR4SpinRestrictedParameterization,
    parameter_blocks,
)
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.seeds import CCSDResidualSeedInfo as PublicCCSDResidualSeedInfo
from xquces.seeds.high_order import (
    igcr3_parameters_from_t_amplitudes,
    igcr4_parameters_from_t_amplitudes,
)
from xquces.seeds.residual import (
    CCSDResidualSeedInfo,
    _default_high_order_residual_blocks,
    _parameters_from_ccsd_residual_seed,
)
from xquces.utils import build_hydrogen_chain, run_rccsd, run_rhf


@dataclass(frozen=True)
class _ResidualSeedInfo:
    params: np.ndarray
    raw_delta_norm: float
    delta_norm: float
    jacobian_rank: int
    scale: float


def _as_restricted_t1(
    t1: np.ndarray | None,
    nocc: int,
    nvirt: int,
) -> np.ndarray:
    if t1 is None:
        return np.zeros((nocc, nvirt), dtype=np.float64)
    arr = np.asarray(t1, dtype=np.float64)
    if arr.shape != (nocc, nvirt):
        raise ValueError(f"Expected t1 shape {(nocc, nvirt)}, got {arr.shape}.")
    return arr


def _as_restricted_t2(t2: np.ndarray, nocc: int, nvirt: int) -> np.ndarray:
    arr = np.asarray(t2, dtype=np.float64)
    if arr.shape != (nocc, nocc, nvirt, nvirt):
        raise ValueError(
            f"Expected t2 shape {(nocc, nocc, nvirt, nvirt)}, got {arr.shape}."
        )
    return arr


def ccsd_target_state(
    t1: np.ndarray | None,
    t2: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    max_power: int = 4,
) -> np.ndarray:
    """Build normalize[(1 + T + T^2/2! + ...) |HF>] from CCSD amplitudes."""
    nocc = int(nelec[0])
    nvirt = int(norb) - nocc
    t1 = _as_restricted_t1(t1, nocc, nvirt)
    t2 = _as_restricted_t2(t2, nocc, nvirt)
    reference = ffsim.hartree_fock_state(norb, nelec)

    t1_op = ffsim.linear_operator(
        ffsim.singles_excitations_restricted(t1),
        norb=norb,
        nelec=nelec,
    )
    t2_op = ffsim.linear_operator(
        ffsim.doubles_excitations_restricted(t2),
        norb=norb,
        nelec=nelec,
    )

    def apply_t(vec: np.ndarray) -> np.ndarray:
        return np.asarray(t1_op @ vec + t2_op @ vec, dtype=np.complex128)

    psi = np.array(reference, copy=True, dtype=np.complex128)
    term = np.array(reference, copy=True, dtype=np.complex128)
    for k in range(1, int(max_power) + 1):
        term = apply_t(term) / float(k)
        psi += term

    norm = float(np.linalg.norm(psi))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("CCSD target construction produced a zero or non-finite state")
    return psi / norm


def phase_align_target(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    overlap = np.vdot(base, target)
    if abs(overlap) > 1.0e-14:
        return np.asarray(target, dtype=np.complex128) * np.exp(
            -1j * np.angle(overlap)
        )
    return np.asarray(target, dtype=np.complex128)


def _real_stacked(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    return np.vstack([arr.real, arr.imag])


def _solve_damped_real_least_squares(
    jacobian_block: np.ndarray,
    target: np.ndarray,
    *,
    damping: float,
) -> np.ndarray:
    a = _real_stacked(jacobian_block)
    b = np.concatenate([np.asarray(target).real, np.asarray(target).imag])
    if damping > 0.0:
        n = jacobian_block.shape[1]
        a = np.vstack([a, np.sqrt(float(damping)) * np.eye(n)])
        b = np.concatenate([b, np.zeros(n, dtype=np.float64)])
    delta, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.asarray(delta, dtype=np.float64)


def _block_column_indices(parameterization, active_blocks: Iterable[str]) -> np.ndarray:
    active = tuple(active_blocks)
    active_set = set(active)
    indices: list[int] = []
    seen: set[str] = set()
    for block in parameter_blocks(parameterization):
        if block.name in active_set:
            indices.extend(range(block.start, block.stop))
            seen.add(block.name)
    missing = active_set - seen
    if missing:
        raise ValueError(f"Unknown parameter block(s): {sorted(missing)!r}")
    return np.asarray(indices, dtype=np.int64)


def _state_overlap(target: np.ndarray, state: np.ndarray) -> float:
    return float(abs(np.vdot(target, state)))


def initialize_block_by_state_residual(
    parameterization,
    reference: np.ndarray,
    nelec: tuple[int, int],
    x_base: np.ndarray,
    target_state: np.ndarray,
    active_blocks: Iterable[str],
    *,
    damping: float = 1.0e-8,
    max_step_norm: float = 0.1,
    scale_scan: Iterable[float] | None = (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
) -> _ResidualSeedInfo:
    fixed = parameterization.apply(reference, nelec)
    psi_base = fixed.state_from_parameters(x_base)
    target = phase_align_target(target_state, psi_base)
    residual = target - psi_base
    residual -= psi_base * np.vdot(psi_base, residual)

    jacobian = fixed.state_jacobian_from_parameters(x_base)
    columns = _block_column_indices(parameterization, active_blocks)
    jacobian_block = np.array(jacobian[:, columns], copy=True, dtype=np.complex128)
    jacobian_block -= psi_base[:, None] * (
        psi_base.conj() @ jacobian_block
    )[None, :]

    delta = _solve_damped_real_least_squares(
        jacobian_block,
        residual,
        damping=damping,
    )
    raw_delta_norm = float(np.linalg.norm(delta))
    delta_norm = raw_delta_norm
    if delta_norm > max_step_norm:
        delta *= float(max_step_norm) / delta_norm
        delta_norm = float(max_step_norm)

    step = np.zeros_like(x_base, dtype=np.float64)
    step[columns] = delta
    scale = 1.0
    if scale_scan is not None:
        base_overlap = _state_overlap(target, psi_base)
        best_overlap = base_overlap
        scale = 0.0
        for candidate in scale_scan:
            candidate = float(candidate)
            params_candidate = np.asarray(x_base, dtype=np.float64) + candidate * step
            state_candidate = fixed.state_from_parameters(params_candidate)
            overlap_candidate = _state_overlap(target, state_candidate)
            if overlap_candidate > best_overlap + 1.0e-14:
                best_overlap = overlap_candidate
                scale = candidate

    params = np.asarray(x_base, dtype=np.float64) + scale * step
    rank = int(np.linalg.matrix_rank(_real_stacked(jacobian_block)))
    return _ResidualSeedInfo(
        params=params,
        raw_delta_norm=raw_delta_norm,
        delta_norm=float(abs(scale) * delta_norm),
        jacobian_rank=rank,
        scale=scale,
    )


def _energy(
    hamiltonian: MolecularHamiltonianLinearOperator,
    state: np.ndarray,
) -> float:
    return hamiltonian.expectation(np.asarray(state, dtype=np.complex128))


def _assert_finite_normalized(state: np.ndarray) -> None:
    assert np.all(np.isfinite(state))
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1.0e-10)


def _outside_block_mask(parameterization, active_blocks: tuple[str, ...]) -> np.ndarray:
    mask = np.ones(parameterization.n_params, dtype=bool)
    for block in parameter_blocks(parameterization):
        if block.name in active_blocks:
            mask[block.slice()] = False
    return mask


def _small_high_order_t_amplitudes(seed: int = 700):
    rng = np.random.default_rng(seed)
    norb = 4
    nocc = 2
    t1 = 0.02 * rng.standard_normal((nocc, norb - nocc))
    t2 = 0.02 * rng.standard_normal((nocc, nocc, norb - nocc, norb - nocc))
    return t1, t2


def _assert_residual_seed_info_allclose(actual, expected, *, atol=1.0e-14):
    assert isinstance(actual, CCSDResidualSeedInfo)
    assert isinstance(expected, CCSDResidualSeedInfo)
    np.testing.assert_allclose(actual.params, expected.params, atol=atol, rtol=0.0)
    assert actual.active_blocks == expected.active_blocks
    np.testing.assert_allclose(
        actual.raw_delta_norms,
        expected.raw_delta_norms,
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual.delta_norms,
        expected.delta_norms,
        atol=atol,
        rtol=0.0,
    )
    assert actual.jacobian_ranks == expected.jacobian_ranks
    np.testing.assert_allclose(actual.scales, expected.scales, atol=atol, rtol=0.0)
    assert np.isclose(actual.overlap_before, expected.overlap_before, atol=atol)
    assert np.isclose(actual.overlap_after, expected.overlap_after, atol=atol)


def test_high_order_seed_function_imports_work():
    assert callable(igcr3_parameters_from_t_amplitudes)
    assert callable(igcr4_parameters_from_t_amplitudes)


def test_high_order_seed_module_matches_igcr3_method_zero_embed():
    t1, t2 = _small_high_order_t_amplitudes(701)
    param = IGCR3SpinRestrictedParameterization(norb=4, nocc=2)

    method = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="zero_embed",
        igcr2_strategy="ucj",
    )
    direct = igcr3_parameters_from_t_amplitudes(
        param,
        t2,
        t1=t1,
        strategy="zero_embed",
        igcr2_strategy="ucj",
    )

    np.testing.assert_allclose(direct, method, atol=1.0e-14, rtol=0.0)


def test_high_order_seed_module_matches_igcr3_method_ccsd_residual():
    t1, t2 = _small_high_order_t_amplitudes(702)
    param = IGCR3SpinRestrictedParameterization(norb=4, nocc=2)

    method = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
    )
    direct = igcr3_parameters_from_t_amplitudes(
        param,
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
    )

    np.testing.assert_allclose(direct, method, atol=1.0e-14, rtol=0.0)


def test_high_order_seed_module_matches_igcr3_method_ccsd_residual_info():
    t1, t2 = _small_high_order_t_amplitudes(703)
    param = IGCR3SpinRestrictedParameterization(norb=4, nocc=2)

    method = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
        return_info=True,
    )
    direct = igcr3_parameters_from_t_amplitudes(
        param,
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
        return_info=True,
    )

    _assert_residual_seed_info_allclose(direct, method)
    assert direct.active_blocks == ("cubic",)


def test_high_order_seed_module_matches_igcr4_method_zero_embed():
    t1, t2 = _small_high_order_t_amplitudes(704)
    param = IGCR4SpinRestrictedParameterization(norb=4, nocc=2)

    method = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="zero_embed",
        igcr3_strategy="zero_embed",
        igcr2_strategy="ucj",
    )
    direct = igcr4_parameters_from_t_amplitudes(
        param,
        t2,
        t1=t1,
        strategy="zero_embed",
        igcr3_strategy="zero_embed",
        igcr2_strategy="ucj",
    )

    np.testing.assert_allclose(direct, method, atol=1.0e-14, rtol=0.0)


def test_high_order_seed_module_matches_igcr4_method_ccsd_residual():
    t1, t2 = _small_high_order_t_amplitudes(705)
    param = IGCR4SpinRestrictedParameterization(norb=4, nocc=2)

    method = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr3_strategy="zero_embed",
        igcr2_strategy="ucj",
        n_iter=1,
    )
    direct = igcr4_parameters_from_t_amplitudes(
        param,
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr3_strategy="zero_embed",
        igcr2_strategy="ucj",
        n_iter=1,
    )

    np.testing.assert_allclose(direct, method, atol=1.0e-14, rtol=0.0)


def test_high_order_seed_module_matches_igcr4_method_ccsd_residual_info():
    t1, t2 = _small_high_order_t_amplitudes(706)
    param = IGCR4SpinRestrictedParameterization(norb=4, nocc=2)

    method = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr3_strategy="zero_embed",
        igcr2_strategy="ucj",
        n_iter=1,
        return_info=True,
    )
    direct = igcr4_parameters_from_t_amplitudes(
        param,
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr3_strategy="zero_embed",
        igcr2_strategy="ucj",
        n_iter=1,
        return_info=True,
    )

    _assert_residual_seed_info_allclose(direct, method)
    assert direct.active_blocks == ("quartic",)


def test_ccsd_residual_seed_info_import_compatibility():
    assert LegacyCCSDResidualSeedInfo is CCSDResidualSeedInfo
    assert PublicCCSDResidualSeedInfo is CCSDResidualSeedInfo


def test_library_ccsd_residual_seed_direct_helper_matches_public_method():
    rng = np.random.default_rng(321)
    norb = 4
    nocc = 2
    t1 = 0.03 * rng.standard_normal((nocc, norb - nocc))
    t2 = 0.03 * rng.standard_normal((nocc, nocc, norb - nocc, norb - nocc))
    param = IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)

    x_base = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="zero_embed",
        igcr2_strategy="ucj",
    )
    direct = _parameters_from_ccsd_residual_seed(
        param,
        t2,
        t1,
        x_base,
        active_blocks=("cubic",),
        n_iter=1,
    )
    public = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
    )

    np.testing.assert_allclose(direct, public, atol=1.0e-14, rtol=0.0)


def test_library_ccsd_residual_seed_return_info_fields_are_populated():
    rng = np.random.default_rng(456)
    norb = 4
    nocc = 2
    t1 = 0.02 * rng.standard_normal((nocc, norb - nocc))
    t2 = 0.02 * rng.standard_normal((nocc, nocc, norb - nocc, norb - nocc))
    param = IGCR4SpinRestrictedParameterization(norb=norb, nocc=nocc)

    info = param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
        return_info=True,
    )

    assert isinstance(info, CCSDResidualSeedInfo)
    assert info.params.shape == (param.n_params,)
    assert info.active_blocks == ("quartic",)
    assert len(info.raw_delta_norms) == 1
    assert len(info.delta_norms) == 1
    assert len(info.jacobian_ranks) == 1
    assert len(info.scales) == 1
    assert np.isfinite(info.overlap_before)
    assert np.isfinite(info.overlap_after)


def test_library_ccsd_residual_default_active_blocks_prefer_reduced_blocks():
    igcr3_param = IGCR3SpinRestrictedParameterization(norb=4, nocc=2)
    igcr4_param = IGCR4SpinRestrictedParameterization(norb=4, nocc=2)

    assert _default_high_order_residual_blocks(
        igcr3_param,
        "cubic",
        ("tau", "omega"),
    ) == ("cubic",)
    assert _default_high_order_residual_blocks(
        igcr4_param,
        "quartic",
        ("eta", "rho", "sigma"),
    ) == ("quartic",)


def test_library_ccsd_residual_seed_preserves_lower_order_blocks():
    rng = np.random.default_rng(123)
    norb = 4
    nocc = 2
    t1 = 0.05 * rng.standard_normal((nocc, norb - nocc))
    t2 = 0.05 * rng.standard_normal((nocc, nocc, norb - nocc, norb - nocc))

    igcr3_param = IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    igcr3_zero = igcr3_param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="zero_embed",
        igcr2_strategy="ucj",
    )
    igcr3_info = igcr3_param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
        return_info=True,
    )
    assert igcr3_info.active_blocks == ("cubic",)
    mask = _outside_block_mask(igcr3_param, igcr3_info.active_blocks)
    assert np.allclose(igcr3_info.params[mask], igcr3_zero[mask], atol=1.0e-14)

    igcr4_param = IGCR4SpinRestrictedParameterization(norb=norb, nocc=nocc)
    igcr4_zero = igcr4_param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="zero_embed",
        igcr2_strategy="ucj",
        n_iter=1,
    )
    igcr4_info = igcr4_param.parameters_from_t_amplitudes(
        t2,
        t1=t1,
        strategy="ccsd_residual",
        igcr2_strategy="ucj",
        n_iter=1,
        return_info=True,
    )
    assert igcr4_info.active_blocks == ("quartic",)
    mask = _outside_block_mask(igcr4_param, igcr4_info.active_blocks)
    assert np.allclose(igcr4_info.params[mask], igcr4_zero[mask], atol=1.0e-14)


def test_h4_ccsd_residual_high_order_igcr_initialization():
    pyscf.lib.num_threads(1)
    distances = [0.8, 1.0, 1.5, 2.0, 2.5]
    max_step_norm = 0.1
    cubic_energy_improved = 0
    cubic_energy_worsened = 0
    quartic_energy_improved = 0
    quartic_energy_worsened = 0
    previous_t1 = None
    previous_t2 = None

    print(
        "R E_FCI E_CCSD E_iGCR2_seed E_iGCR3_zero "
        "E_iGCR3_ccsd_residual E_iGCR4_zero E_iGCR4_ccsd_residual "
        "overlap_iGCR2_CCSD overlap_iGCR3_zero_CCSD "
        "overlap_iGCR3_init_CCSD overlap_iGCR4_zero_CCSD "
        "overlap_iGCR4_init_CCSD ||delta_cubic|| ||delta_quartic|| "
        "rank_cubic_jacobian rank_quartic_jacobian"
    )

    for r in distances:
        mol = build_hydrogen_chain(r, 4, "sto-3g")
        mf = run_rhf(mol)
        ccsd = run_rccsd(mf, t1=previous_t1, t2=previous_t2)
        previous_t1 = np.array(ccsd.t1, copy=True)
        previous_t2 = np.array(ccsd.t2, copy=True)

        norb = mf.mo_coeff.shape[1]
        nelec = (mol.nelectron // 2, mol.nelectron // 2)
        nocc = nelec[0]
        nvirt = norb - nocc
        reference = ffsim.hartree_fock_state(norb, nelec)
        hamiltonian = MolecularHamiltonianLinearOperator.from_scf(mf)
        e_fci = float(np.linalg.eigvalsh(hamiltonian.dense_electronic_matrix())[0])
        e_fci += hamiltonian.ecore

        t1 = _as_restricted_t1(ccsd.t1, nocc, nvirt)
        t2 = _as_restricted_t2(ccsd.t2, nocc, nvirt)
        target = ccsd_target_state(t1, t2, norb, nelec, max_power=4)
        _assert_finite_normalized(target)

        igcr2_param = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
        igcr2_params = igcr2_param.parameters_from_t_amplitudes(
            t2,
            t1=t1,
            strategy="ucj",
        )
        igcr2_ansatz = igcr2_param.ansatz_from_parameters(igcr2_params)
        igcr2_state = igcr2_param.apply(reference, nelec).state_from_parameters(
            igcr2_params
        )

        igcr3_param = IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
        igcr3_zero_params = igcr3_param.parameters_from_igcr2_ansatz(
            igcr2_ansatz,
            tau_scale=0.0,
            omega_scale=0.0,
        )
        igcr3_zero_state = igcr3_param.apply(reference, nelec).state_from_parameters(
            igcr3_zero_params
        )
        cubic_info = initialize_block_by_state_residual(
            igcr3_param,
            reference,
            nelec,
            igcr3_zero_params,
            target,
            ("cubic",),
            max_step_norm=max_step_norm,
        )
        igcr3_init_state = igcr3_param.apply(reference, nelec).state_from_parameters(
            cubic_info.params
        )

        igcr3_init_ansatz = igcr3_param.ansatz_from_parameters(cubic_info.params)
        igcr4_param = IGCR4SpinRestrictedParameterization(norb=norb, nocc=nocc)
        igcr4_zero_params = igcr4_param.parameters_from_igcr3_ansatz(
            igcr3_init_ansatz,
            eta_scale=0.0,
            rho_scale=0.0,
            sigma_scale=0.0,
        )
        igcr4_zero_state = igcr4_param.apply(reference, nelec).state_from_parameters(
            igcr4_zero_params
        )
        quartic_info = initialize_block_by_state_residual(
            igcr4_param,
            reference,
            nelec,
            igcr4_zero_params,
            target,
            ("quartic",),
            max_step_norm=max_step_norm,
        )
        igcr4_init_state = igcr4_param.apply(reference, nelec).state_from_parameters(
            quartic_info.params
        )

        _assert_finite_normalized(igcr2_state)
        _assert_finite_normalized(igcr3_zero_state)
        _assert_finite_normalized(igcr3_init_state)
        _assert_finite_normalized(igcr4_zero_state)
        _assert_finite_normalized(igcr4_init_state)

        e_igcr2_seed = _energy(hamiltonian, igcr2_state)
        e_igcr3_zero = _energy(hamiltonian, igcr3_zero_state)
        e_igcr3_init = _energy(hamiltonian, igcr3_init_state)
        e_igcr4_zero = _energy(hamiltonian, igcr4_zero_state)
        e_igcr4_init = _energy(hamiltonian, igcr4_init_state)

        overlap_igcr2 = _state_overlap(target, igcr2_state)
        overlap_igcr3_zero = _state_overlap(target, igcr3_zero_state)
        overlap_igcr3_init = _state_overlap(target, igcr3_init_state)
        overlap_igcr4_zero = _state_overlap(target, igcr4_zero_state)
        overlap_igcr4_init = _state_overlap(target, igcr4_init_state)

        assert np.all(
            np.isfinite(
                [
                    e_fci,
                    ccsd.e_tot,
                    e_igcr2_seed,
                    e_igcr3_zero,
                    e_igcr3_init,
                    e_igcr4_zero,
                    e_igcr4_init,
                    overlap_igcr2,
                    overlap_igcr3_zero,
                    overlap_igcr3_init,
                    overlap_igcr4_zero,
                    overlap_igcr4_init,
                    cubic_info.delta_norm,
                    quartic_info.delta_norm,
                ]
            )
        )
        assert cubic_info.jacobian_rank > 0
        assert quartic_info.jacobian_rank > 0
        assert np.isclose(e_igcr3_zero, e_igcr2_seed, atol=1.0e-10)
        assert np.isclose(e_igcr4_zero, e_igcr3_init, atol=1.0e-10)
        assert overlap_igcr3_init + 1.0e-12 >= overlap_igcr3_zero
        assert overlap_igcr4_init + 1.0e-12 >= overlap_igcr4_zero

        cubic_diff = e_igcr3_init - e_igcr3_zero
        quartic_diff = e_igcr4_init - e_igcr4_zero
        cubic_energy_improved += int(cubic_diff < -1.0e-10)
        cubic_energy_worsened += int(cubic_diff > 1.0e-10)
        quartic_energy_improved += int(quartic_diff < -1.0e-10)
        quartic_energy_worsened += int(quartic_diff > 1.0e-10)

        print(
            f"{r:.2f} {e_fci:.12f} {float(ccsd.e_tot):.12f} "
            f"{e_igcr2_seed:.12f} {e_igcr3_zero:.12f} {e_igcr3_init:.12f} "
            f"{e_igcr4_zero:.12f} {e_igcr4_init:.12f} "
            f"{overlap_igcr2:.12f} {overlap_igcr3_zero:.12f} "
            f"{overlap_igcr3_init:.12f} {overlap_igcr4_zero:.12f} "
            f"{overlap_igcr4_init:.12f} {cubic_info.delta_norm:.6e} "
            f"{quartic_info.delta_norm:.6e} {cubic_info.jacobian_rank:d} "
            f"{quartic_info.jacobian_rank:d}"
        )

    print(
        "summary "
        f"cubic_energy_improved={cubic_energy_improved}/{len(distances)} "
        f"cubic_energy_worsened={cubic_energy_worsened}/{len(distances)} "
        f"quartic_energy_improved={quartic_energy_improved}/{len(distances)} "
        f"quartic_energy_worsened={quartic_energy_worsened}/{len(distances)}"
    )
