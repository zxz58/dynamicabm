"""
common.py
=========
Shared building blocks used by every simulation script in this package:

  * generate_er_network  -- builds the same "add random edges until you have
    exactly M of them" Erdos-Renyi-style network as the original C++
    `network()` function in every .cpp file, but returns it as a *sparse*
    adjacency matrix + adjacency list instead of a dense NxN array. The
    physical network is identical (same distribution); only the storage
    representation changes, which is what makes the Python version usable at
    all (a dense N=5000 network means 25,000,000 element lookups per
    timestep in the original C loops).

  * fitness_random_walk -- vectorised version of the "sum twelve uniform(0,1)
    draws, subtract 6" trick the original authors use to approximate a
    Gaussian random walk of pathogen fitness (Psi). Kept exactly as in the
    C++ (rather than swapped for np.random.normal) so the increment
    distribution is identical to the paper's code.

  * pick_random_neighbor / pick_weighted_neighbor -- helpers used when a
    newly infected node must "inherit" the fitness of one of its infected
    neighbours (uniformly at random for the alpha-mutation models, weighted
    by fitness for the beta-mutation models).

All RNG draws go through a numpy.random.Generator instance that the caller
supplies, so runs are reproducible via a seed and there is no dependence on
Python's global random state.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def generate_er_network(N: int, num_edges: int, rng: np.random.Generator):
    """Undirected random graph on N nodes with exactly `num_edges` edges.

    Matches the original C++ `network()` routine: repeatedly draw a random
    pair of distinct nodes and add the (undirected) edge if it is not
    already present, until the target edge count is reached. For the
    parameters used in this project (N=5000, num_edges=37500, i.e. ~1.5%
    density) collisions are rare, so this is fast in plain Python.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (N, N), dtype float64
        Symmetric 0/1 adjacency matrix. Used for fast matrix-vector
        products (A @ infected_mask) that replace the O(N^2) neighbour
        counting loops in the C++ code.
    adj_list : list[np.ndarray]
        adj_list[i] is the array of neighbour indices of node i. Used for
        the (comparatively rare) per-node operation of picking a random
        infected neighbour to inherit fitness from.
    """
    edges: set[tuple[int, int]] = set()
    while len(edges) < num_edges:
        need = num_edges - len(edges)
        # oversample to absorb self-loop / duplicate rejections
        batch = max(int(need * 1.2), 16)
        h = rng.integers(0, N, size=batch)
        hh = rng.integers(0, N, size=batch)
        for a, b in zip(h.tolist(), hh.tolist()):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            if e not in edges:
                edges.add(e)
                if len(edges) == num_edges:
                    break

    rows = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    data = np.ones(all_rows.shape[0], dtype=np.float64)
    A = csr_matrix((data, (all_rows, all_cols)), shape=(N, N))

    adj_list = [A.indices[A.indptr[i]:A.indptr[i + 1]].copy() for i in range(N)]
    return A, adj_list


def fitness_random_walk(psi_values: np.ndarray, sigma: float, X: float,
                         lower: float, upper: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Vectorised version of the C++ "12-uniforms-minus-6" mutation step.

    For each entry in `psi_values`, draws d = (sum of 12 U(0,1)) - 6, an
    Irwin-Hall approximation to a standard normal (mean 0, variance 1),
    then updates psi <- clip(psi + X + d*sigma, lower, upper).
    """
    n = psi_values.shape[0]
    if n == 0:
        return psi_values
    d = rng.random((n, 12)).sum(axis=1) - 6.0
    new_psi = psi_values + X + d * sigma
    np.clip(new_psi, lower, upper, out=new_psi)
    return new_psi


def pick_random_neighbor_index(node_indices: np.ndarray, adj_list, candidate_mask: np.ndarray,
                                weight_values: np.ndarray, rng: np.random.Generator,
                                weighted: bool = False) -> np.ndarray:
    """For every node in `node_indices`, pick one neighbour satisfying
    `candidate_mask` (uniformly, or weighted by `weight_values` when
    weighted=True) and return the CHOSEN NEIGHBOUR'S INDEX (not one of its
    attribute values).

    This is the building block `pick_random_neighbor` below is written on
    top of; it exists separately for models where a newly infected node
    needs to inherit *several* attributes from the same donor neighbour
    (e.g. both a recovery-evasion trait and a transmissibility trait in the
    combined-mutation model) -- you look the donor index up once, then
    index into as many per-node arrays as you need with it, guaranteeing
    every inherited trait comes from the same individual.
    """
    out = np.empty(node_indices.shape[0], dtype=np.int64)
    for pos, i in enumerate(node_indices):
        neigh = adj_list[i]
        cand = neigh[candidate_mask[neigh]]
        if cand.size == 0:
            # Should not happen if `i` was flagged infectable this step, but
            # guard defensively (keeps the node as its own "donor").
            out[pos] = i
            continue
        if weighted:
            w = weight_values[cand].astype(np.float64)
            w_sum = w.sum()
            if w_sum <= 0:
                choice = rng.choice(cand)
            else:
                choice = rng.choice(cand, p=w / w_sum)
        else:
            choice = rng.choice(cand)
        out[pos] = choice
    return out


def pick_random_neighbor(node_indices: np.ndarray, adj_list, infected_mask: np.ndarray,
                          source_values: np.ndarray, rng: np.random.Generator,
                          weighted: bool = False) -> np.ndarray:
    """For every node in `node_indices`, pick one currently-infected neighbour
    (uniformly, or weighted by `source_values` when weighted=True) and return
    the source_values entry belonging to the chosen neighbour.

    This mirrors the original code's "walk over neighbours, find the k-th
    infected one" logic, restricted to a Python loop over only the (usually
    small) set of nodes that actually need a draw this step -- infection
    events are rare relative to N, so this stays cheap even though it isn't
    fully vectorised. Implemented on top of `pick_random_neighbor_index`.
    """
    idx = pick_random_neighbor_index(node_indices, adj_list, infected_mask,
                                      source_values, rng, weighted=weighted)
    return source_values[idx]
