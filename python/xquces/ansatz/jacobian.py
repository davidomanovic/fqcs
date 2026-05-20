from __future__ import annotations

from typing import Callable

import numpy as np


def _native_factory(
    parameterization: object,
    names: tuple[str, ...],
    reference_vec: np.ndarray | None,
    nelec: tuple[int, int] | None,
) -> Callable | None:
    for name in names:
        method = getattr(parameterization, name, None)
        if callable(method):
            return method(reference_vec, nelec)
    return None


def _is_gate_sequence(parameterization: object) -> bool:
    from xquces.ansatz.sequence import GateSequenceParameterization

    return isinstance(parameterization, GateSequenceParameterization)


def _gate_sequence_owner(parameterization: object) -> object | None:
    inverter = getattr(parameterization, "ansatz_parameters_from_instance", None)
    owner = getattr(inverter, "__self__", None)
    if owner is None or owner is parameterization:
        return None
    return owner


def _restricted_jacobian_factory(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray] | None:
    from xquces.gcr.igcr import IGCRSpinRestrictedParameterization

    if not isinstance(parameterization, IGCRSpinRestrictedParameterization):
        return None
    from xquces.gcr.restricted_jacobian import make_restricted_gcr_jacobian

    return make_restricted_gcr_jacobian(parameterization, reference_vec, nelec)


def _restricted_vjp_factory(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
    from xquces.gcr.igcr import IGCRSpinRestrictedParameterization

    if not isinstance(parameterization, IGCRSpinRestrictedParameterization):
        return None
    from xquces.gcr.restricted_jacobian import make_restricted_gcr_vjp

    return make_restricted_gcr_vjp(parameterization, reference_vec, nelec)


def _restricted_subspace_jacobian_factory(
    parameterization: object,
    reference_vec: np.ndarray,
    nelec: tuple[int, int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
    from xquces.gcr.igcr import IGCRSpinRestrictedParameterization

    if not isinstance(parameterization, IGCRSpinRestrictedParameterization):
        return None
    from xquces.gcr.restricted_jacobian import make_restricted_gcr_subspace_jacobian

    return make_restricted_gcr_subspace_jacobian(parameterization, reference_vec, nelec)


def _composite_jacobian_factory(
    parameterization: object,
) -> Callable[[np.ndarray], np.ndarray] | None:
    from xquces.gcr.references import (
        CompositeReferenceAnsatzParameterization,
        make_composite_reference_ansatz_jacobian,
    )

    if not isinstance(parameterization, CompositeReferenceAnsatzParameterization):
        return None
    return make_composite_reference_ansatz_jacobian(parameterization)


def _composite_vjp_factory(
    parameterization: object,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
    from xquces.gcr.references import (
        CompositeReferenceAnsatzParameterization,
        make_composite_reference_ansatz_vjp,
    )

    if not isinstance(parameterization, CompositeReferenceAnsatzParameterization):
        return None
    return make_composite_reference_ansatz_vjp(parameterization)


def _composite_subspace_jacobian_factory(
    parameterization: object,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
    from xquces.gcr.references import (
        CompositeReferenceAnsatzParameterization,
        make_composite_reference_ansatz_subspace_jacobian,
    )

    if not isinstance(parameterization, CompositeReferenceAnsatzParameterization):
        return None
    return make_composite_reference_ansatz_subspace_jacobian(parameterization)


def _unsupported(parameterization: object, operation: str) -> NotImplementedError:
    type_name = type(parameterization).__name__
    return NotImplementedError(
        f"{operation} is not implemented for {type_name}. "
        "Provide a native Jacobian backend or use a supported ansatz "
        "parameterization."
    )


def make_state_jacobian(
    parameterization: object,
    reference_vec: np.ndarray | None,
    nelec: tuple[int, int] | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a state Jacobian factory for a supported parameterization.

    This is the canonical dispatch layer. The existing spin-restricted iGCR
    analytic implementation remains in ``xquces.gcr.restricted_jacobian`` as a
    backend, while higher-level callers use this function.
    """

    native = _native_factory(
        parameterization,
        ("make_state_jacobian", "state_jacobian_factory"),
        reference_vec,
        nelec,
    )
    if native is not None:
        return native

    if _is_gate_sequence(parameterization):
        owner = _gate_sequence_owner(parameterization)
        if owner is not None:
            return make_state_jacobian(owner, reference_vec, nelec)
        raise _unsupported(
            parameterization,
            "Gate-sequence state Jacobian",
        )

    if reference_vec is not None and nelec is not None:
        restricted = _restricted_jacobian_factory(
            parameterization,
            np.asarray(reference_vec, dtype=np.complex128),
            tuple(nelec),
        )
        if restricted is not None:
            return restricted

    composite = _composite_jacobian_factory(parameterization)
    if composite is not None:
        return composite

    raise _unsupported(parameterization, "State Jacobian")


def make_state_vjp(
    parameterization: object,
    reference_vec: np.ndarray | None,
    nelec: tuple[int, int] | None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a state VJP factory for a supported parameterization."""

    native = _native_factory(
        parameterization,
        ("make_state_vjp", "state_vjp_factory"),
        reference_vec,
        nelec,
    )
    if native is not None:
        return native

    if _is_gate_sequence(parameterization):
        owner = _gate_sequence_owner(parameterization)
        if owner is not None:
            return make_state_vjp(owner, reference_vec, nelec)
        raise _unsupported(parameterization, "Gate-sequence state VJP")

    if reference_vec is not None and nelec is not None:
        restricted = _restricted_vjp_factory(
            parameterization,
            np.asarray(reference_vec, dtype=np.complex128),
            tuple(nelec),
        )
        if restricted is not None:
            return restricted

    composite = _composite_vjp_factory(parameterization)
    if composite is not None:
        return composite

    raise _unsupported(parameterization, "State VJP")


def make_state_subspace_jacobian(
    parameterization: object,
    reference_vec: np.ndarray | None,
    nelec: tuple[int, int] | None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a subspace state-Jacobian factory for a supported parameterization."""

    native = _native_factory(
        parameterization,
        ("make_state_subspace_jacobian", "state_subspace_jacobian_factory"),
        reference_vec,
        nelec,
    )
    if native is not None:
        return native

    if _is_gate_sequence(parameterization):
        owner = _gate_sequence_owner(parameterization)
        if owner is not None:
            return make_state_subspace_jacobian(owner, reference_vec, nelec)
        raise _unsupported(
            parameterization,
            "Gate-sequence state subspace Jacobian",
        )

    if reference_vec is not None and nelec is not None:
        restricted = _restricted_subspace_jacobian_factory(
            parameterization,
            np.asarray(reference_vec, dtype=np.complex128),
            tuple(nelec),
        )
        if restricted is not None:
            return restricted

    composite = _composite_subspace_jacobian_factory(parameterization)
    if composite is not None:
        return composite

    raise _unsupported(parameterization, "State subspace Jacobian")


__all__ = [
    "make_state_jacobian",
    "make_state_subspace_jacobian",
    "make_state_vjp",
]
