from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import rustworkx
from qiskit.circuit import Instruction
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.providers import BackendV2
from qiskit.transpiler import StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import ApplyLayout, VF2PostLayout
from rustworkx import NoEdgeBetweenNodes, PyGraph

from xquces.qiskit import PRE_INIT

Topology = Literal["heavy-hex", "square"]

VF2_CALL_LIMIT = 30_000_000


@dataclass(frozen=True)
class GCR2PairUCCDLayout:
    initial_layout: tuple[int, ...]
    allowed_alpha_beta_pairs: tuple[tuple[int, int], ...]
    topology: Topology


@dataclass(frozen=True)
class GCR2PairUCCDPassManagerResult:
    pass_manager: StagedPassManager
    layout: GCR2PairUCCDLayout


def _normalize_topology(topology: str) -> Topology:
    key = str(topology).lower().replace("_", "-")
    if key not in {"heavy-hex", "square"}:
        raise ValueError("topology must be 'heavy-hex' or 'square'")
    return key


def _linear_pairs(norb: int) -> list[tuple[int, int]]:
    return [(p, p + 1) for p in range(max(0, norb - 1))]


def _default_alpha_beta_pairs(norb: int, topology: Topology) -> list[tuple[int, int]]:
    if topology == "heavy-hex":
        return [(p, p) for p in range(norb) if p % 4 == 0]
    return [(p, p) for p in range(norb)]


def _validate_pairs(
    norb: int,
    name: str,
    pairs: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = [(int(p), int(q)) for p, q in pairs]
    for pair in out:
        if not ((0 <= pair[0] < norb) and (0 <= pair[1] < norb)):
            raise ValueError(f"{name} contains out-of-range orbital pair {pair}")
    return out


def _resolve_interaction_pairs(
    norb: int,
    topology: Topology,
    interaction_pairs: tuple[Sequence[tuple[int, int]], Sequence[tuple[int, int]] | None]
    | tuple[
        Sequence[tuple[int, int]],
        Sequence[tuple[int, int]] | None,
        Sequence[tuple[int, int]],
    ]
    | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    if interaction_pairs is None:
        pairs_aa = _linear_pairs(norb)
        pairs_ab = _default_alpha_beta_pairs(norb, topology)
        pairs_bb = pairs_aa
    elif len(interaction_pairs) == 2:
        raw_aa, raw_ab = interaction_pairs
        pairs_aa = _validate_pairs(norb, "pairs_aa", raw_aa)
        pairs_ab = (
            _default_alpha_beta_pairs(norb, topology)
            if raw_ab is None
            else _validate_pairs(norb, "pairs_ab", raw_ab)
        )
        pairs_bb = list(pairs_aa)
    elif len(interaction_pairs) == 3:
        raw_aa, raw_ab, raw_bb = interaction_pairs
        pairs_aa = _validate_pairs(norb, "pairs_aa", raw_aa)
        pairs_ab = (
            _default_alpha_beta_pairs(norb, topology)
            if raw_ab is None
            else _validate_pairs(norb, "pairs_ab", raw_ab)
        )
        pairs_bb = _validate_pairs(norb, "pairs_bb", raw_bb)
    else:
        raise ValueError("interaction_pairs must be None, length 2, or length 3")
    return pairs_aa, pairs_ab, pairs_bb


def _create_two_chain_layout_graph(
    norb: int,
    pairs_aa: Sequence[tuple[int, int]],
    pairs_bb: Sequence[tuple[int, int]],
) -> PyGraph:
    graph = rustworkx.PyGraph()
    graph.add_nodes_from(range(2 * norb))
    for p, q in pairs_aa:
        graph.add_edge(p, q, None)
    for p, q in pairs_bb:
        graph.add_edge(norb + p, norb + q, None)
    return graph


def _candidate_layout_graph(
    norb: int,
    topology: Topology,
    pairs_aa: Sequence[tuple[int, int]],
    pairs_ab: Sequence[tuple[int, int]],
    pairs_bb: Sequence[tuple[int, int]],
) -> PyGraph:
    graph = _create_two_chain_layout_graph(norb, pairs_aa, pairs_bb)
    if topology == "heavy-hex":
        for index, (p, q) in enumerate(
            sorted(pairs_ab, key=lambda pair: (pair[0], pair[1]))
        ):
            bridge = 2 * norb + index
            graph.add_node(bridge)
            graph.add_edge(p, bridge, None)
            graph.add_edge(bridge, norb + q, None)
    else:
        for p, q in pairs_ab:
            graph.add_edge(p, norb + q, None)
    return graph


def _two_qubit_gate_names(operations: Sequence[Instruction]) -> list[str]:
    excluded = {"barrier", "reset", "measure", "measurement", "delay"}
    return [
        operation.name
        for operation in operations
        if operation.num_qubits == 2 and operation.name not in excluded
    ]


def _make_backend_coupling_graph(
    backend: BackendV2,
    *,
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
) -> PyGraph:
    target = backend.target
    graph = copy.deepcopy(backend.coupling_map.graph)
    if not graph.is_symmetric():
        graph.make_symmetric()
    backend_graph = graph.to_undirected()

    removed_edges: list[set[int]] = []
    for edge in list(backend_graph.edge_list()):
        edge_set = set(edge)
        if edge_set in removed_edges:
            continue
        try:
            backend_graph.remove_edge(edge[0], edge[1])
            removed_edges.append(edge_set)
        except NoEdgeBetweenNodes:
            pass

    try:
        measure_target = target["measure"]
    except KeyError:
        measure_target = None

    if measure_target is not None:
        for node_id in list(backend_graph.node_indices()):
            prop = measure_target[(node_id,)] if (node_id,) in measure_target else None
            if (
                prop is not None
                and prop.error is not None
                and prop.error >= readout_error_threshold
            ):
                backend_graph.remove_node(node_id)

    for gate_name in _two_qubit_gate_names(target.operations):
        gate_target = target[gate_name]
        for edge in list(backend_graph.edge_list()):
            if edge in gate_target:
                found_edge = edge
            elif edge[::-1] in gate_target:
                found_edge = edge[::-1]
            else:
                continue
            prop = gate_target[found_edge]
            if (
                prop is not None
                and prop.error is not None
                and prop.error >= two_qubit_error_threshold
            ):
                backend_graph.remove_edge(edge[0], edge[1])

    return backend_graph


def _is_subgraph_embeddable(backend_graph: PyGraph, layout_graph: PyGraph) -> bool:
    return rustworkx.is_subgraph_isomorphic(
        backend_graph,
        layout_graph,
        call_limit=VF2_CALL_LIMIT,
        id_order=False,
        induced=False,
    )


def _find_layout_graph(
    backend_graph: PyGraph,
    norb: int,
    topology: Topology,
    pairs_aa: Sequence[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: Sequence[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    allowed = list(pairs_ab)
    while True:
        layout_graph = _candidate_layout_graph(
            norb,
            topology,
            pairs_aa,
            allowed,
            pairs_bb,
        )
        if _is_subgraph_embeddable(backend_graph, layout_graph):
            return layout_graph, allowed
        if not allowed:
            raise RuntimeError(
                "No backend layout satisfies the same-spin chain constraints after error pruning."
            )
        removed = allowed.pop()
        warnings.warn(
            "Backend cannot accommodate alpha-beta interaction "
            f"{removed}; dropping it from the layout graph."
        )


def _layout_from_graphs(
    backend_graph: PyGraph,
    layout_graph: PyGraph,
    num_logical_qubits: int,
) -> tuple[int, ...]:
    mappings = rustworkx.vf2_mapping(
        backend_graph,
        layout_graph,
        subgraph=True,
        id_order=False,
        induced=False,
        call_limit=VF2_CALL_LIMIT,
    )
    mapping = next(mappings, None)
    if mapping is None:
        raise RuntimeError("No backend subgraph layout was found.")

    physical_for_layout_node = [-1] * layout_graph.num_nodes()
    for physical_node, layout_node in mapping.items():
        physical_for_layout_node[layout_node] = int(physical_node)

    initial_layout = tuple(physical_for_layout_node[:num_logical_qubits])
    if any(qubit < 0 for qubit in initial_layout):
        raise RuntimeError("Incomplete backend subgraph mapping for logical qubits.")
    return initial_layout


def _placeholder_layout(
    backend: BackendV2,
    norb: int,
    topology: Topology,
    pairs_aa: Sequence[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: Sequence[tuple[int, int]],
    *,
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
) -> GCR2PairUCCDLayout:
    backend_graph = _make_backend_coupling_graph(
        backend,
        two_qubit_error_threshold=two_qubit_error_threshold,
        readout_error_threshold=readout_error_threshold,
    )
    layout_graph, allowed_pairs_ab = _find_layout_graph(
        backend_graph,
        norb,
        topology,
        pairs_aa,
        pairs_ab,
        pairs_bb,
    )
    initial_layout = _layout_from_graphs(backend_graph, layout_graph, 2 * norb)
    return GCR2PairUCCDLayout(
        initial_layout=initial_layout,
        allowed_alpha_beta_pairs=tuple(allowed_pairs_ab),
        topology=topology,
    )


def generate_gcr2_pair_uccd_pass_manager(
    backend: BackendV2,
    norb: int,
    nocc: int | None = None,
    topology: Topology = "heavy-hex",
    interaction_pairs: tuple[Sequence[tuple[int, int]], Sequence[tuple[int, int]] | None]
    | tuple[
        Sequence[tuple[int, int]],
        Sequence[tuple[int, int]] | None,
        Sequence[tuple[int, int]],
    ]
    | None = None,
    two_qubit_error_threshold: float = 1.0,
    readout_error_threshold: float = 0.10,
    **qiskit_pm_kwargs: Any,
) -> GCR2PairUCCDPassManagerResult:
    """Build a backend-aware pass manager for GCR-2 on a product pair-UCCD reference.

    The logical register is assumed to use alpha-first, beta-second Jordan-Wigner
    ordering.  The initial layout places alpha and beta orbitals on two hardware
    paths, then adds prioritized alpha-beta contacts.  On square topologies these
    contacts are direct edges.  On heavy-hex topologies they are embedded through
    physical bridge qubits, matching the device degree constraints while keeping
    the logical circuit on ``2 * norb`` qubits.

    The returned pass manager runs the xquces pre-init stage, then Qiskit's preset
    pipeline with the selected initial layout, and finally re-enables VF2 post-layout
    so that the compiler can still choose a lower-noise isomorphic embedding after
    routing.
    """
    norb = int(norb)
    if norb <= 0:
        raise ValueError("norb must be positive")
    if nocc is not None and not (0 <= int(nocc) <= norb):
        raise ValueError("nocc must satisfy 0 <= nocc <= norb")

    topology = _normalize_topology(topology)
    pairs_aa, pairs_ab, pairs_bb = _resolve_interaction_pairs(
        norb,
        topology,
        interaction_pairs,
    )

    if "initial_layout" in qiskit_pm_kwargs:
        warnings.warn(
            "initial_layout is controlled by generate_gcr2_pair_uccd_pass_manager and is ignored."
        )
        del qiskit_pm_kwargs["initial_layout"]
    if "layout_method" in qiskit_pm_kwargs:
        warnings.warn(
            "layout_method is controlled by generate_gcr2_pair_uccd_pass_manager and is ignored."
        )
        del qiskit_pm_kwargs["layout_method"]

    layout = _placeholder_layout(
        backend,
        norb,
        topology,
        pairs_aa,
        pairs_ab,
        pairs_bb,
        two_qubit_error_threshold=float(two_qubit_error_threshold),
        readout_error_threshold=float(readout_error_threshold),
    )

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        initial_layout=list(layout.initial_layout),
        **qiskit_pm_kwargs,
    )
    pass_manager.pre_init = PRE_INIT

    def _has_post_layout(property_set: dict[str, Any]) -> bool:
        return property_set.get("post_layout") is not None

    pass_manager.routing.append(
        VF2PostLayout(
            target=backend.target,
            strict_direction=False,
            call_limit=VF2_CALL_LIMIT,
        )
    )
    pass_manager.routing.append(
        ConditionalController(ApplyLayout(), condition=_has_post_layout)
    )

    return GCR2PairUCCDPassManagerResult(pass_manager=pass_manager, layout=layout)


__all__ = [
    "GCR2PairUCCDLayout",
    "GCR2PairUCCDPassManagerResult",
    "generate_gcr2_pair_uccd_pass_manager",
]
