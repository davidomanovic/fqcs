#qubit topologies
import itertools
import numpy as np

def all_to_all_indices(norb):
    pairs = list(itertools.combinations(range(norb), 2))
    triples = list(itertools.combinations(range(norb), 3))
    quads = list(itertools.combinations(range(norb), 4))
    return sparse_indices_from_sets(norb, pairs, triples, quads)


def square_edges(norb):
    ncols = int(np.ceil(np.sqrt(norb)))
    nrows = int(np.ceil(norb / ncols))
    edges = []

    def idx(i, j):
        return i * ncols + j

    for i in range(nrows):
        for j in range(ncols):
            p = idx(i, j)
            if p >= norb:
                continue

            q = idx(i, j + 1)
            if j + 1 < ncols and q < norb:
                edges.append(tuple(sorted((p, q))))

            q = idx(i + 1, j)
            if i + 1 < nrows and q < norb:
                edges.append(tuple(sorted((p, q))))

    return sorted(set(edges))


def heavy_hex_like_edges(norb):
    edges = [(p, p + 1) for p in range(norb - 1)]

    for p in range(0, norb - 4, 4):
        edges.append((p, p + 4))

    return sorted(set(tuple(sorted(edge)) for edge in edges))


def connected_subsets(norb, edges, size):
    adj = {p: set() for p in range(norb)}

    for p, q in edges:
        adj[p].add(q)
        adj[q].add(p)

    out = []

    for subset in itertools.combinations(range(norb), size):
        allowed = set(subset)
        seen = {subset[0]}
        stack = [subset[0]]

        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in allowed and y not in seen:
                    seen.add(y)
                    stack.append(y)

        if len(seen) == size:
            out.append(subset)

    return out


def directed_pairs(edges):
    return sorted(set([(p, q) for p, q in edges] + [(q, p) for p, q in edges]))


def rho_from_triples(triples):
    rho = []

    for triple in triples:
        for p in triple:
            rest = sorted(q for q in triple if q != p)
            rho.append((p, rest[0], rest[1]))

    return sorted(set(rho))


def sparse_indices_from_sets(norb, pairs, triples, quads):
    pairs = sorted(set(tuple(sorted(pair)) for pair in pairs))
    triples = sorted(set(tuple(sorted(triple)) for triple in triples))
    quads = sorted(set(tuple(sorted(quad)) for quad in quads))

    return {
        "interaction_pairs": pairs,
        "tau_indices_": directed_pairs(pairs),
        "omega_indices_": triples,
        "eta_indices_": pairs,
        "rho_indices_": rho_from_triples(triples),
        "sigma_indices_": quads,
    }


def topology_indices(norb, topology):
    if topology == "all_to_all":
        return all_to_all_indices(norb)

    if topology == "square":
        edges = square_edges(norb)
        triples = connected_subsets(norb, edges, 3)
        quads = connected_subsets(norb, edges, 4)
        return sparse_indices_from_sets(norb, edges, triples, quads)

    if topology == "heavy_hex":
        edges = heavy_hex_like_edges(norb)
        triples = connected_subsets(norb, edges, 3)
        quads = connected_subsets(norb, edges, 4)
        return sparse_indices_from_sets(norb, edges, triples, quads)

    raise ValueError(f"Unknown topology: {topology}")