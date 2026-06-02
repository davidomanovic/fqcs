"""Preset Qiskit pass managers for xquces ansatz circuits."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import rustworkx
from qiskit.circuit import Instruction
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.providers import BackendV2
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import ApplyLayout, VF2PostLayout
from rustworkx import NoEdgeBetweenNodes, PyGraph

from xquces.qiskit.transpiler_stages import pre_init_passes
from xquces.qiskit.utils import ignore_ibm_fractional_translation_plugin_warning

Connectivity = Literal["heavy-hex", "square"]
Topology = Connectivity

InteractionPairs = (
    tuple[Sequence[tuple[int, int]], Sequence[tuple[int, int]] | None]
    | tuple[
        Sequence[tuple[int, int]],
        Sequence[tuple[int, int]] | None,
        Sequence[tuple[int, int]],
    ]
)

VF2_CALL_LIMIT = 30_000_000


@dataclass(frozen=True)
class GCR2Layout:
    """Layout metadata produced by a GCR2 pass manager generator.

    Attributes:
        initial_layout: Physical qubit indices used as the initial Qiskit layout for
            the ``2 * norb`` logical spin-orbital qubits.
        allowed_alpha_beta_pairs: Subset of requested alpha-beta orbital interaction
            pairs that can be accommodated by the backend layout.
        topology: Connectivity topology used for the layout.
    """

    initial_layout: tuple[int, ...]
    allowed_alpha_beta_pairs: tuple[tuple[int, int], ...]
    topology: Connectivity

    @property
    def connectivity(self) -> Connectivity:
        """Connectivity topology used for the layout."""
        return self.topology


@dataclass(frozen=True)
class GCR2PassManagerResult:
    """Result returned by GCR2 pass manager generators.

    The result can also be unpacked like the ffsim LUCJ helper::

        pass_manager, pairs_ab = generate_gcr2_pass_manager(...)

    Attributes:
        pass_manager: Configured Qiskit preset pass manager.
        layout: Layout metadata used to construct the pass manager.
    """

    pass_manager: StagedPassManager
    layout: GCR2Layout

    @property
    def allowed_alpha_beta_pairs(self) -> tuple[tuple[int, int], ...]:
        """Alpha-beta interaction pairs accommodated by the backend layout."""
        return self.layout.allowed_alpha_beta_pairs

    def __iter__(self) -> Iterator[object]:
        yield self.pass_manager
        yield list(self.layout.allowed_alpha_beta_pairs)


GCR2PairUCCDLayout = GCR2Layout
GCR2PairUCCDPassManagerResult = GCR2PassManagerResult


def _normalize_connectivity(connectivity: str) -> Connectivity:
    key = str(connectivity).lower().replace("_", "-")
    if key == "heavy-hex":
        return "heavy-hex"
    if key == "square":
        return "square"
    raise ValueError("connectivity must be 'heavy-hex' or 'square'")


def _resolve_connectivity_alias(
    connectivity: str,
    qiskit_pm_kwargs: dict[str, Any],
) -> Connectivity:
    if "topology" not in qiskit_pm_kwargs:
        return _normalize_connectivity(connectivity)

    topology = qiskit_pm_kwargs.pop("topology")
    if connectivity != "heavy-hex":
        raise TypeError("Pass only one of 'connectivity' or 'topology'.")
    return _normalize_connectivity(topology)


def _linear_pairs(norb: int) -> list[tuple[int, int]]:
    return [(p, p + 1) for p in range(max(0, norb - 1))]


def _default_alpha_beta_pairs(norb: int, connectivity: Connectivity) -> list[tuple[int, int]]:
    if connectivity == "heavy-hex":
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
    connectivity: Connectivity,
    interaction_pairs: InteractionPairs | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Resolve same-spin and opposite-spin interaction pairs.

    ``None`` chooses the standard heavy-hex/square GCR2 defaults: nearest-neighbor
    same-spin chains and the topology-specific alpha-beta pairs used by the ffsim
    LUCJ layout helper.
    """
    if interaction_pairs is None:
        pairs_aa = _linear_pairs(norb)
        pairs_ab = _default_alpha_beta_pairs(norb, connectivity)
        pairs_bb = list(pairs_aa)
    elif len(interaction_pairs) == 2:
        raw_aa, raw_ab = interaction_pairs
        pairs_aa = _validate_pairs(norb, "pairs_aa", raw_aa)
        pairs_ab = (
            _default_alpha_beta_pairs(norb, connectivity)
            if raw_ab is None
            else _validate_pairs(norb, "pairs_ab", raw_ab)
        )
        pairs_bb = list(pairs_aa)
    elif len(interaction_pairs) == 3:
        raw_aa, raw_ab, raw_bb = interaction_pairs
        pairs_aa = _validate_pairs(norb, "pairs_aa", raw_aa)
        pairs_ab = (
            _default_alpha_beta_pairs(norb, connectivity)
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
    """Create the two same-spin orbital chains used by GCR2 layouts.

    Nodes ``0`` through ``norb - 1`` represent alpha spin orbitals, and nodes
    ``norb`` through ``2 * norb - 1`` represent beta spin orbitals.  Same-spin
    interaction pairs are added as edges within the corresponding chain.
    """
    graph = rustworkx.PyGraph()
    graph.add_nodes_from(range(2 * norb))
    for p, q in pairs_aa:
        graph.add_edge(p, q, None)
    for p, q in pairs_bb:
        graph.add_edge(norb + p, norb + q, None)
    return graph


def _candidate_layout_graph(
    norb: int,
    connectivity: Connectivity,
    pairs_aa: Sequence[tuple[int, int]],
    pairs_ab: Sequence[tuple[int, int]],
    pairs_bb: Sequence[tuple[int, int]],
) -> PyGraph:
    """Build a candidate GCR2 layout graph for a backend topology.

    On square lattices, alpha-beta interactions are represented as direct edges.
    On heavy-hex lattices, each alpha-beta interaction is represented by a bridge
    node between the alpha and beta orbitals, matching the ffsim LUCJ zig-zag
    layout strategy.
    """
    graph = _create_two_chain_layout_graph(norb, pairs_aa, pairs_bb)
    if connectivity == "heavy-hex":
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
    """Return a filtered undirected backend coupling graph.

    Directed coupling maps are symmetrized, duplicate undirected edges are removed,
    and nodes or edges with error rates greater than or equal to the provided
    thresholds are pruned.  Missing instruction properties are treated as usable.
    """
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
    connectivity: Connectivity,
    pairs_aa: Sequence[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: Sequence[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    """Find a topology-compliant layout graph embeddable on the backend.

    If the backend cannot accommodate every requested alpha-beta pair, pairs are
    dropped from the end of the list until a valid subgraph embedding is found.
    List requested pairs in descending order of priority to control which pairs
    are preserved first.
    """
    allowed = list(pairs_ab)
    while True:
        layout_graph = _candidate_layout_graph(
            norb,
            connectivity,
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


def _gcr2_layout(
    backend: BackendV2,
    norb: int,
    connectivity: Connectivity,
    pairs_aa: Sequence[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: Sequence[tuple[int, int]],
    *,
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
) -> GCR2Layout:
    backend_graph = _make_backend_coupling_graph(
        backend,
        two_qubit_error_threshold=two_qubit_error_threshold,
        readout_error_threshold=readout_error_threshold,
    )
    layout_graph, allowed_pairs_ab = _find_layout_graph(
        backend_graph,
        norb,
        connectivity,
        pairs_aa,
        pairs_ab,
        pairs_bb,
    )
    initial_layout = _layout_from_graphs(backend_graph, layout_graph, 2 * norb)
    return GCR2Layout(
        initial_layout=initial_layout,
        allowed_alpha_beta_pairs=tuple(allowed_pairs_ab),
        topology=connectivity,
    )


def _drop_controlled_qiskit_pm_kwargs(qiskit_pm_kwargs: dict[str, Any]) -> None:
    for name in ("initial_layout", "layout_method"):
        if name in qiskit_pm_kwargs:
            warnings.warn(f"Argument ``{name}`` is ignored.")
            del qiskit_pm_kwargs[name]


def _build_pass_manager(
    backend: BackendV2,
    layout: GCR2Layout,
    qiskit_pm_kwargs: dict[str, Any],
) -> StagedPassManager:
    with ignore_ibm_fractional_translation_plugin_warning():
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            initial_layout=list(layout.initial_layout),
            **qiskit_pm_kwargs,
        )
    pass_manager.pre_init = PassManager(list(pre_init_passes()))

    # Supplying an initial layout disables Qiskit's default VF2PostLayout pass.
    # Re-enable it so the transpiler can still choose a lower-noise isomorphic
    # mapping after routing.
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
    return pass_manager


def _generate_gcr2_pass_manager(
    backend: BackendV2,
    norb: int,
    connectivity: str,
    interaction_pairs: InteractionPairs | None,
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
    qiskit_pm_kwargs: dict[str, Any],
) -> GCR2PassManagerResult:
    norb = int(norb)
    if norb <= 0:
        raise ValueError("norb must be positive")

    resolved_connectivity = _resolve_connectivity_alias(connectivity, qiskit_pm_kwargs)
    pairs_aa, pairs_ab, pairs_bb = _resolve_interaction_pairs(
        norb,
        resolved_connectivity,
        interaction_pairs,
    )

    _drop_controlled_qiskit_pm_kwargs(qiskit_pm_kwargs)

    layout = _gcr2_layout(
        backend,
        norb,
        resolved_connectivity,
        pairs_aa,
        pairs_ab,
        pairs_bb,
        two_qubit_error_threshold=float(two_qubit_error_threshold),
        readout_error_threshold=float(readout_error_threshold),
    )
    return GCR2PassManagerResult(
        pass_manager=_build_pass_manager(backend, layout, qiskit_pm_kwargs),
        layout=layout,
    )


def generate_gcr2_pass_manager(
    backend: BackendV2,
    norb: int,
    connectivity: Connectivity = "heavy-hex",
    interaction_pairs: InteractionPairs | None = None,
    two_qubit_error_threshold: float = 1.0,
    readout_error_threshold: float = 0.10,
    **qiskit_pm_kwargs: Any,
) -> GCR2PassManagerResult:
    """Generate a Qiskit preset pass manager for GCR2 ansatz circuits.

    Construct a pass manager that maps a GCR2/iGCR2 circuit onto a target backend
    using the same heavy-hex and square layout strategy used by ffsim's LUCJ pass
    manager. The layout contains two same-spin orbital chains, with alpha-beta
    interactions represented directly on square topologies and through bridge
    qubits on heavy-hex topologies.

    Args:
        backend: Target Qiskit backend.
        norb: Number of spatial orbitals. The ansatz uses ``2 * norb`` logical
            qubits, and heavy-hex transpilation may use additional ancilla qubits.
        connectivity: Backend connectivity topology. Must be ``"heavy-hex"`` or
            ``"square"``.
        interaction_pairs: Optional length-2 or length-3 tuple describing
            alpha-alpha, alpha-beta, and optional beta-beta orbital interaction
            pairs. If omitted, same-spin interactions default to nearest-neighbor
            orbital chains and alpha-beta interactions default to ``[(p, p) for p
            in range(norb) if p % 4 == 0]`` on heavy-hex and ``[(p, p) for p in
            range(norb)]`` on square. If alpha-beta pairs are ``None`` inside an
            explicit tuple, the same topology-specific defaults are used.

            Each pair ``(p, q)`` must satisfy ``0 <= p, q < norb``. Requested
            alpha-beta pairs are treated as a priority-ordered list: if the backend
            cannot accommodate every pair, pairs are dropped from the end until a
            valid layout is found.
        two_qubit_error_threshold: Two-qubit gate error threshold. Edges in the
            backend coupling graph with error rate ``>= two_qubit_error_threshold``
            are removed. The default, ``1.0``, removes only completely faulty edges.
        readout_error_threshold: Readout error threshold. Nodes in the backend
            coupling graph with readout error ``>= readout_error_threshold`` are
            removed.
        **qiskit_pm_kwargs: Additional keyword arguments forwarded to
            :func:`qiskit.transpiler.generate_preset_pass_manager`. The arguments
            ``initial_layout`` and ``layout_method`` are controlled by this helper
            and are ignored with a warning if provided.

    Returns:
        A :class:`GCR2PassManagerResult` containing the configured pass manager and
        the layout metadata. The result can be unpacked as ``(pass_manager,
        allowed_alpha_beta_pairs)`` for compatibility with the ffsim helper style.

    Raises:
        ValueError: If ``norb`` is non-positive, an interaction pair is out of range,
            or ``connectivity`` is not ``"heavy-hex"`` or ``"square"``.
        RuntimeError: If no backend subgraph satisfies the requested same-spin
            constraints after error pruning.

    Note:
        Providing ``initial_layout`` to Qiskit's preset pass manager generator
        normally disables ``VF2PostLayout``. This function re-enables it explicitly
        so the transpiler can search for a better noise-aware isomorphic subgraph
        mapping after routing.
    """
    return _generate_gcr2_pass_manager(
        backend,
        norb,
        connectivity,
        interaction_pairs,
        two_qubit_error_threshold,
        readout_error_threshold,
        qiskit_pm_kwargs,
    )


def generate_gcr2_pair_uccd_pass_manager(
    backend: BackendV2,
    norb: int,
    nocc: int | None = None,
    connectivity: Connectivity = "heavy-hex",
    interaction_pairs: InteractionPairs | None = None,
    two_qubit_error_threshold: float = 1.0,
    readout_error_threshold: float = 0.10,
    **qiskit_pm_kwargs: Any,
) -> GCR2PassManagerResult:
    """Generate a Qiskit preset pass manager for GCR2 pair-UCCD circuits.

    This is the pair-UCCD reference variant of :func:`generate_gcr2_pass_manager`.
    It uses the same GCR2 heavy-hex/square layout engine and keeps ``nocc`` in the
    signature for compatibility with pair-UCCD parameterizations.

    Args:
        backend: Target Qiskit backend.
        norb: Number of spatial orbitals.
        nocc: Optional number of occupied spatial orbitals. If provided, it must
            satisfy ``0 <= nocc <= norb``.
        connectivity: Backend connectivity topology. Must be ``"heavy-hex"`` or
            ``"square"``. The deprecated keyword ``topology`` is also accepted.
        interaction_pairs: Optional length-2 or length-3 tuple describing
            alpha-alpha, alpha-beta, and optional beta-beta orbital interaction
            pairs. See :func:`generate_gcr2_pass_manager` for the default values
            and priority semantics.
        two_qubit_error_threshold: Two-qubit gate error threshold.
        readout_error_threshold: Readout error threshold.
        **qiskit_pm_kwargs: Additional keyword arguments forwarded to
            :func:`qiskit.transpiler.generate_preset_pass_manager`.

    Returns:
        A :class:`GCR2PassManagerResult` containing the configured pass manager and
        layout metadata.
    """
    norb = int(norb)
    if nocc is not None and not (0 <= int(nocc) <= norb):
        raise ValueError("nocc must satisfy 0 <= nocc <= norb")
    return _generate_gcr2_pass_manager(
        backend,
        norb,
        connectivity,
        interaction_pairs,
        two_qubit_error_threshold,
        readout_error_threshold,
        qiskit_pm_kwargs,
    )


def generate_gcr2_pairuccd_pass_manager(
    *args: Any,
    **kwargs: Any,
) -> GCR2PassManagerResult:
    """Alias for :func:`generate_gcr2_pair_uccd_pass_manager`."""
    return generate_gcr2_pair_uccd_pass_manager(*args, **kwargs)
