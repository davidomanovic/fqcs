from __future__ import annotations

import numpy as np


def _embedding_target(parameterization: object) -> object:
    inverter = getattr(parameterization, "ansatz_parameters_from_instance", None)
    owner = getattr(inverter, "__self__", None)
    return parameterization if owner is None else owner


def embed_ansatz_parameters(parameterization: object, ansatz: object) -> np.ndarray:
    from xquces.gcr.restricted_model import (
        IGCR2Ansatz,
        IGCR2LayeredAnsatz,
        IGCR3Ansatz,
        IGCR3LayeredAnsatz,
        IGCR4Ansatz,
        IGCR4LayeredAnsatz,
    )

    target = _embedding_target(parameterization)
    if isinstance(ansatz, (IGCR4Ansatz, IGCR4LayeredAnsatz)) and hasattr(
        target, "parameters_from_ansatz"
    ):
        try:
            return target.parameters_from_ansatz(ansatz)
        except TypeError:
            pass
    if isinstance(ansatz, (IGCR3Ansatz, IGCR3LayeredAnsatz)):
        if hasattr(target, "parameters_from_igcr3_ansatz"):
            return target.parameters_from_igcr3_ansatz(ansatz)
        return target.parameters_from_ansatz(ansatz)
    if isinstance(ansatz, (IGCR2Ansatz, IGCR2LayeredAnsatz)):
        if hasattr(target, "parameters_from_igcr2_ansatz"):
            return target.parameters_from_igcr2_ansatz(ansatz)
        return target.parameters_from_ansatz(ansatz)
    return target.parameters_from_ansatz(ansatz)


def _order_from_blocks(parameterization: object) -> int:
    parameter_blocks = getattr(parameterization, "parameter_blocks", None)
    if parameter_blocks is None:
        return 0
    try:
        names = {block.name for block in parameter_blocks()}
    except Exception:
        return 0
    if "diagonal.quartic" in names or "quartic" in names:
        return 4
    if "diagonal.cubic" in names or "cubic" in names:
        return 3
    if "diagonal.pair" in names or "pair" in names:
        return 2
    return 0


def parameters_from_t2(
    parameterization: object,
    t2: np.ndarray,
    *,
    source_order: int | None = None,
    **kwargs,
) -> np.ndarray:
    from xquces.gcr.igcr import (
        IGCR2SpinRestrictedParameterization,
        IGCR3SpinRestrictedParameterization,
        IGCR4SpinRestrictedParameterization,
    )
    from xquces.gcr.restricted_model import IGCR2Ansatz, IGCR3Ansatz, IGCR4Ansatz

    target = getattr(parameterization, "implementation", parameterization)
    order = int(
        source_order
        or getattr(parameterization, "order", 0)
        or getattr(target, "order", 0)
        or _order_from_blocks(parameterization)
        or 0
    )
    if order == 0:
        if isinstance(target, IGCR4SpinRestrictedParameterization):
            order = 4
        elif isinstance(target, IGCR3SpinRestrictedParameterization):
            order = 3
        else:
            order = 2
    if (
        hasattr(target, "parameters_from_t_amplitudes")
        and int(getattr(target, "order", order)) == order
    ):
        t1 = kwargs.pop("t1", None)
        return target.parameters_from_t_amplitudes(t2, t1=t1, **kwargs)
    if order == 2:
        if isinstance(target, IGCR2SpinRestrictedParameterization):
            t1 = kwargs.pop("t1", None)
            return target.parameters_from_t_amplitudes(t2, t1=t1, **kwargs)
        ansatz = IGCR2Ansatz.from_t_restricted(t2, **kwargs)
    elif order == 3:
        if isinstance(target, IGCR3SpinRestrictedParameterization):
            t1 = kwargs.pop("t1", None)
            return target.parameters_from_t_amplitudes(t2, t1=t1, **kwargs)
        ansatz = IGCR3Ansatz.from_t_restricted(t2, **kwargs)
    elif order == 4:
        if isinstance(target, IGCR4SpinRestrictedParameterization):
            t1 = kwargs.pop("t1", None)
            return target.parameters_from_t_amplitudes(t2, t1=t1, **kwargs)
        ansatz = IGCR4Ansatz.from_t_restricted(t2, **kwargs)
    else:
        raise ValueError("source_order must be 2, 3, or 4")
    return embed_ansatz_parameters(parameterization, ansatz)

