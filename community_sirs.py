"""
community_sirs.py
=================
Focused SIRS agent-based model for endemic virulence evolution on a small
community-structured dynamic contact network.

The model uses continuous-time Gillespie dynamics. Hosts live in two or
three Watts-Strogatz communities. Virulence is a heritable continuous
phenotype assigned at infection and controls transmission, recovery,
disease mortality, behavioral edge removal, and inter-community movement.

The main implementation idea is:

1. Build an initial local contact graph inside each community.
2. Keep a mutable active graph as sets of neighbors, because edges are added
   and removed during the simulation.
3. At each Gillespie step, enumerate all currently possible events and their
   rates, draw the waiting time, then draw exactly one event proportional to
   its rate.
4. Record regularly spaced samples after burn-in so endemic behavior can be
   summarized and plotted.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


STATE_S = 0
STATE_I = 1
STATE_R = 2
STATE_D = 3


@dataclass(frozen=True)
class TradeoffParams:
    """Bounds and shape choices for mapping virulence into event rates.

    The simulation stores virulence as one scalar per host. These parameters
    define how that scalar becomes biologically meaningful event rates:
    beta, alpha, gamma, delta, and phi. Keeping the mapping in one dataclass
    makes sensitivity analysis easier, because a future sweep can vary these
    fields without changing the Gillespie logic.
    """

    v_min: float
    v_max: float
    beta_min: float
    beta_max: float
    alpha_min: float
    alpha_max: float
    gamma_min: float
    gamma_max: float
    delta_min: float
    delta_max: float
    phi_max: float
    beta_shape: str
    alpha_shape: str
    gamma_shape: str
    delta_shape: str
    phi_shape: str


# One row per realization. These are the quantities most useful for endemic
# steady-state comparison across seeds or later sensitivity-analysis grids.
SUMMARY_FIELDS = [
    "realization",
    "seed",
    "N",
    "K",
    "t_final",
    "samples",
    "inter_edge_decay_rate",
    "extinct",
    "mean_prevalence",
    "final_prevalence",
    "mean_s_frac",
    "final_s_frac",
    "mean_r_frac",
    "final_r_frac",
    "mean_d_frac",
    "final_d_frac",
    "mean_infected_virulence",
    "final_infected_virulence",
    "mean_virulence_variance",
    "final_virulence_variance",
    "cumulative_deaths",
    "mean_active_edges",
    "final_active_edges",
    "mean_inter_edges",
    "final_inter_edges",
    "realized_clustering",
]

# Per-sample rows are intentionally richer than summary rows. Plotting and
# snapshot tooling should read these instead of trying to reconstruct state
# from one-row-per-realization summaries.
SAMPLE_FIELDS = [
    "realization",
    "seed",
    "t",
    "inter_edge_decay_rate",
    "S",
    "I",
    "R",
    "D",
    "prevalence",
    "s_frac",
    "r_frac",
    "d_frac",
    "mean_infected_virulence",
    "virulence_variance",
    "active_edges",
    "inter_edges",
    "realized_clustering",
    "cumulative_deaths",
    "extinct",
]


def _edge(u: int, v: int) -> tuple[int, int]:
    """Canonicalize an undirected edge so it can be stored in a set."""
    return (u, v) if u < v else (v, u)


def split_communities(N: int, K: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Assign hosts to K communities as evenly as integer sizes allow.

    Returns both a per-node community_id array and a list of node arrays. The
    first form is convenient for quickly testing whether an edge is local or
    inter-community; the second is convenient when drawing a random movement
    partner from a different community.
    """
    sizes = [N // K + (1 if c < (N % K) else 0) for c in range(K)]
    community_id = np.empty(N, dtype=np.int64)
    communities: list[np.ndarray] = []
    start = 0
    for c, size in enumerate(sizes):
        nodes = np.arange(start, start + size, dtype=np.int64)
        community_id[nodes] = c
        communities.append(nodes)
        start += size
    return community_id, communities


def build_community_graph(N: int, K: int, kbar: int, rewiring_prob: float,
                          seed: int | None):
    """
    Build disjoint Watts-Strogatz communities and adjacency sets.

    N: total number of nodes
    K: number of communities
    kbar: target local degree inside each community. 
        Higher kbar means denser local communities. Lower kbar means sparser communities.
    rewiring_prob: This controls how locally clustered vs random the within-community graph is.

    The graph starts with only within-community contacts. Inter-community
    contacts appear later through movement events, so the model can distinguish
    stable local social structure from transient long-range connectivity.
    """
    community_id, communities = split_communities(N, K)
    # active_adj is the mutable contact network used by Gillespie events;
    # baseline_adj records the initial local social structure for diagnostics
    # and future extensions.
    active_adj = [set() for _ in range(N)]
    baseline_adj = [set() for _ in range(N)]
    rng_seed = seed

    for c, nodes in enumerate(communities):
        size = len(nodes)
        if size <= 1:
            continue
        # NetworkX Watts-Strogatz requires an even k for ring-neighbor wiring.
        k = min(kbar, size - 1)
        if k % 2 == 1:
            k -= 1
        if k < 2:
            k = 2 if size > 2 else 1
        local = nx.watts_strogatz_graph(size, k, rewiring_prob,
                                        seed=None if rng_seed is None else rng_seed + c)
        for u_local, v_local in local.edges():
            u = int(nodes[u_local])
            v = int(nodes[v_local])
            active_adj[u].add(v)
            active_adj[v].add(u)
            baseline_adj[u].add(v)
            baseline_adj[v].add(u)

    return community_id, communities, active_adj, baseline_adj


def active_edges(active_adj: list[set[int]]) -> set[tuple[int, int]]:
    """Return current usable contact edges without double-counting.

    active_adj stores each undirected edge twice, once in each endpoint's
    neighbor set. Most model calculations need each edge once, so this helper
    canonicalizes edges as u < v.
    """
    out = set()
    for u, neigh in enumerate(active_adj):
        for v in neigh:
            if u < v:
                out.add((u, v))
    return out


def realized_clustering(active_adj: list[set[int]], alive_mask: np.ndarray) -> float:
    """Compute clustering on the active network after removing dead hosts.

    Clustering is reported on the graph that disease processes can actually
    use at that time. Dead hosts are excluded because mortality permanently
    removes them and their incident edges from the contact network.
    """
    G = nx.Graph()
    alive_nodes = np.where(alive_mask)[0].tolist()
    G.add_nodes_from(alive_nodes)
    for u, v in active_edges(active_adj):
        if alive_mask[u] and alive_mask[v]:
            G.add_edge(u, v)
    if len(G) < 3:
        return 0.0
    return float(nx.average_clustering(G))


def shape_value(z: np.ndarray | float, shape: str) -> np.ndarray | float:
    """Apply the requested sensitivity shape on normalized virulence z.

    z is always clipped into [0, 1]. The shapes are intentionally simple:
    linear gives a straight trade-off, concave gives diminishing returns, and
    convex gives weak effects at low virulence with stronger effects near the
    upper end of the phenotype range.
    """
    z = np.clip(z, 0.0, 1.0)
    if shape == "linear":
        return z
    if shape == "concave":
        return np.sqrt(z)
    if shape == "convex":
        return z * z
    raise ValueError(f"unknown shape: {shape}")


def bounded_tradeoff(v: np.ndarray | float, params: TradeoffParams, low: float,
                     high: float, shape: str, increasing: bool) -> np.ndarray | float:
    """Convert virulence into a bounded rate with controlled shape/direction.

    This is the shared transformation for every virulence-dependent rate. It
    first normalizes v to the phenotype interval, applies the chosen curve, and
    then maps that curve onto [low, high]. For decreasing trade-offs, the curve
    is flipped so high virulence produces lower rates.
    """
    denom = params.v_max - params.v_min
    if denom <= 0:
        raise ValueError("v_max must be greater than v_min")
    z = (v - params.v_min) / denom
    factor = shape_value(z, shape)
    if not increasing:
        factor = 1.0 - factor
    return low + (high - low) * factor


def beta_of(v, params: TradeoffParams):
    """Per-contact transmission rate; higher virulence transmits better."""
    return bounded_tradeoff(v, params, params.beta_min, params.beta_max,
                            params.beta_shape, increasing=True)


def alpha_of(v, params: TradeoffParams):
    """Disease-induced mortality rate; higher virulence is more lethal."""
    return bounded_tradeoff(v, params, params.alpha_min, params.alpha_max,
                            params.alpha_shape, increasing=True)


def gamma_of(v, params: TradeoffParams):
    """Recovery rate; higher virulence is harder to clear."""
    return bounded_tradeoff(v, params, params.gamma_min, params.gamma_max,
                            params.gamma_shape, increasing=False)


def delta_of(v, params: TradeoffParams):
    """Behavioral edge-removal rate; higher virulence induces more avoidance."""
    return bounded_tradeoff(v, params, params.delta_min, params.delta_max,
                            params.delta_shape, increasing=True)


def phi_of(v, params: TradeoffParams):
    """Inter-community movement rate; higher virulence reduces mobility."""
    return bounded_tradeoff(v, params, 0.0, params.phi_max,
                            params.phi_shape, increasing=False)


def restore_edges(i: int, state: np.ndarray, active_adj: list[set[int]],
                  severed_edges: list[set[tuple[int, int]]]):
    # Recovery restores the contacts lost during this infection episode,
    # unless the other endpoint has died in the meantime.
    for u, v in list(severed_edges[i]):
        if state[u] != STATE_D and state[v] != STATE_D:
            active_adj[u].add(v)
            active_adj[v].add(u)
    severed_edges[i].clear()


def remove_all_edges(i: int, active_adj: list[set[int]],
                     severed_edges: list[set[tuple[int, int]]]):
    # Disease mortality is absorbing: dead hosts leave the active graph.
    for j in list(active_adj[i]):
        active_adj[j].discard(i)
        active_adj[i].discard(j)
    severed_edges[i].clear()


def add_intercommunity_edge(i: int, state: np.ndarray, community_id: np.ndarray,
                            communities: list[np.ndarray], active_adj: list[set[int]],
                            rng: np.random.Generator, max_attempts: int = 25) -> bool:
    """Add one long-range edge from host i to a random host in another community.

    Movement is modeled as edge creation rather than moving a host between
    communities. This keeps community membership fixed while allowing the
    active contact graph to gain cross-community bridges over time.
    """
    if state[i] == STATE_D:
        return False
    other_communities = [c for c in range(len(communities)) if c != community_id[i]]
    # Try a bounded number of random partners; saturated small graphs can make
    # valid cross-community endpoints temporarily hard to find.
    for _ in range(max_attempts):
        c = int(rng.choice(other_communities))
        candidates = communities[c]
        if candidates.size == 0:
            continue
        j = int(rng.choice(candidates))
        if state[j] == STATE_D or j == i or j in active_adj[i]:
            continue
        active_adj[i].add(j)
        active_adj[j].add(i)
        return True
    return False


def remove_edge(edge: tuple[int, int], active_adj: list[set[int]]):
    """Remove an undirected active edge if it is still present."""
    u, v = edge
    active_adj[u].discard(v)
    active_adj[v].discard(u)


def count_inter_edges(active_adj: list[set[int]], community_id: np.ndarray) -> int:
    """Count currently active edges whose endpoints are in different communities."""
    return sum(1 for u, v in active_edges(active_adj) if community_id[u] != community_id[v])


def summarize_state(t: float, state: np.ndarray, virulence: np.ndarray,
                    active_adj: list[set[int]], community_id: np.ndarray) -> dict[str, float]:
    """Compute all per-timepoint metrics used by summaries, plots, and snapshots.

    This function deliberately contains no random draws or state mutation. It
    is safe to call at arbitrary sample or snapshot times and gives the same
    definitions everywhere in the codebase.
    """
    N = state.shape[0]
    infected = state == STATE_I
    s_count = int(np.count_nonzero(state == STATE_S))
    i_count = int(np.count_nonzero(infected))
    r_count = int(np.count_nonzero(state == STATE_R))
    d_count = int(np.count_nonzero(state == STATE_D))
    alive_mask = state != STATE_D
    if i_count:
        infected_v = virulence[infected]
        mean_v = float(np.mean(infected_v))
        var_v = float(np.var(infected_v))
    else:
        mean_v = math.nan
        var_v = math.nan
    return {
        "t": t,
        "S": s_count,
        "I": i_count,
        "R": r_count,
        "D": d_count,
        "prevalence": i_count / N,
        "s_frac": s_count / N,
        "r_frac": r_count / N,
        "d_frac": d_count / N,
        "mean_infected_virulence": mean_v,
        "virulence_variance": var_v,
        "active_edges": float(len(active_edges(active_adj))),
        "inter_edges": float(count_inter_edges(active_adj, community_id)),
        "realized_clustering": realized_clustering(active_adj, alive_mask),
    }


def choose_weighted_event(events: list[tuple[str, tuple, float]],
                          total_rate: float, rng: np.random.Generator):
    """Draw one Gillespie event proportional to its event rate."""
    threshold = rng.random() * total_rate
    acc = 0.0
    for kind, payload, rate in events:
        acc += rate
        if threshold <= acc:
            return kind, payload
    return events[-1][0], events[-1][1]


def collect_events(state: np.ndarray, virulence: np.ndarray, active_adj: list[set[int]],
                   community_id: np.ndarray, params: TradeoffParams, rho: float,
                   inter_edge_decay_rate: float):
    """Enumerate every event currently available to the Gillespie sampler.

    The returned event list is explicit rather than optimized: each tuple is
    (event_kind, payload, rate). That makes the model readable and easy to
    audit. For larger simulations, this is the main place to optimize, because
    it rebuilds the event list from the active graph at every event.
    """
    events: list[tuple[str, tuple, float]] = []

    # Edge-local events: transmission along S-I contacts and behavioral
    # removal of contacts incident to infectious hosts.
    for u, v in active_edges(active_adj):
        is_inter_edge = community_id[u] != community_id[v]
        su = state[u]
        sv = state[v]
        if su == STATE_S and sv == STATE_I:
            events.append(("transmission", (u, v), float(beta_of(virulence[v], params))))
        elif su == STATE_I and sv == STATE_S:
            events.append(("transmission", (v, u), float(beta_of(virulence[u], params))))

        if su == STATE_I:
            events.append(("edge_removal", (u, v, u), float(delta_of(virulence[u], params))))
        if sv == STATE_I:
            events.append(("edge_removal", (u, v, v), float(delta_of(virulence[v], params))))
        # Long-range contacts represent temporary movement/mixing, so they
        # decay independently of disease state. Local Watts-Strogatz edges do
        # not use this event; they are the stable within-community backbone.
        if is_inter_edge:
            events.append(("inter_edge_decay", (u, v), float(inter_edge_decay_rate)))

    # Node-local disease events depend on the infected host's virulence.
    infected_idx = np.where(state == STATE_I)[0]
    for i in infected_idx:
        v = virulence[i]
        events.append(("recovery", (int(i),), float(gamma_of(v, params))))
        events.append(("mortality", (int(i),), float(alpha_of(v, params))))

    # Waning immunity and movement are node events; S/R hosts move at phi_max.
    recovered_idx = np.where(state == STATE_R)[0]
    for i in recovered_idx:
        events.append(("waning", (int(i),), float(rho)))

    alive_idx = np.where(state != STATE_D)[0]
    for i in alive_idx:
        # Infectious hosts move according to their virulence-dependent phi(v).
        # Susceptible and recovered hosts use phi_max, matching the model text.
        rate = float(phi_of(virulence[i], params)) if state[i] == STATE_I else params.phi_max
        events.append(("movement", (int(i),), rate))

    # Zero-rate events are harmless conceptually but would distort the event
    # picker if total_rate and the list disagreed, so filter them once here.
    total_rate = sum(rate for _, _, rate in events if rate > 0.0)
    if total_rate <= 0.0:
        return [], 0.0
    events = [(kind, payload, rate) for kind, payload, rate in events if rate > 0.0]
    return events, total_rate


def snapshot_state(t: float, state: np.ndarray, virulence: np.ndarray,
                   active_adj: list[set[int]], community_id: np.ndarray,
                   cumulative_deaths: int) -> dict:
    """Copy mutable simulation state for later network rendering.

    active_adj is a list of mutable sets, so a shallow copy would continue to
    change as the simulation runs. The deep copy freezes the graph exactly as
    it looked at the requested snapshot time.
    """
    return {
        "t": t,
        "state": state.copy(),
        "virulence": virulence.copy(),
        "active_adj": copy.deepcopy(active_adj),
        "community_id": community_id.copy(),
        "cumulative_deaths": cumulative_deaths,
    }


def add_sample_context(sample: dict[str, float], realization: int,
                       seed_value: int | str, cumulative_deaths: int,
                       inter_edge_decay_rate: float) -> dict:
    """Attach run metadata to one sampled state row.

    The plotting script expects each row to be self-contained: it should know
    which realization it came from, what seed generated that realization, and
    how many deaths had accumulated by that sample time.
    """
    row = {
        "realization": realization,
        "seed": seed_value,
        **sample,
        "inter_edge_decay_rate": inter_edge_decay_rate,
        "cumulative_deaths": cumulative_deaths,
        "extinct": int(sample["I"] == 0),
    }
    return {field: row.get(field, "") for field in SAMPLE_FIELDS}


def run_one_realization(args, realization: int, rng: np.random.Generator,
                        return_samples: bool = False,
                        snapshot_times: list[float] | None = None):
    """Run one stochastic realization and return summary/sample/snapshot data.

    This function is the model's main state machine. It owns the mutable host
    arrays and active graph, while helper functions handle rate construction,
    event selection, and metric calculation.
    """
    params = TradeoffParams(
        v_min=args.v_min, v_max=args.v_max,
        beta_min=args.beta_min, beta_max=args.beta_max,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        delta_min=args.delta_min, delta_max=args.delta_max,
        phi_max=args.phi_max,
        beta_shape=args.beta_shape, alpha_shape=args.alpha_shape,
        gamma_shape=args.gamma_shape, delta_shape=args.delta_shape,
        phi_shape=args.phi_shape,
    )
    graph_seed = None if args.seed is None else args.seed + realization * 1009
    community_id, communities, active_adj, _baseline_adj = build_community_graph(
        args.N, args.K, args.kbar, args.rewiring_prob, graph_seed)

    state = np.full(args.N, STATE_S, dtype=np.int8)
    virulence = np.full(args.N, args.v_init, dtype=np.float64)
    infection_count = np.zeros(args.N, dtype=np.int64)
    severed_edges = [set() for _ in range(args.N)]

    # Seed the endemic simulation with a small infected fraction. Setting
    # --initial-prevalence 0 is allowed for extinction/output sanity checks.
    n_initial = int(round(args.initial_prevalence * args.N))
    if args.initial_prevalence > 0.0:
        n_initial = max(1, n_initial)
    initial = rng.choice(args.N, size=n_initial, replace=False)
    if n_initial:
        state[initial] = STATE_I
        infection_count[initial] = 1

    t = 0.0
    cumulative_deaths = 0
    extinct = 0
    next_sample = args.burn_in_time
    samples: list[dict[str, float]] = []
    sample_rows: list[dict] = []
    snapshots: list[dict] = []
    seed_value = "" if args.seed is None else args.seed + realization
    requested_snapshots = sorted(snapshot_times or [])
    next_snapshot = 0

    def capture_samples_until(limit: float):
        """Record regularly spaced samples up to a continuous-time limit."""
        nonlocal next_sample
        while next_sample <= limit:
            sample = summarize_state(next_sample, state, virulence, active_adj, community_id)
            samples.append(sample)
            if return_samples:
                sample_rows.append(add_sample_context(
                    sample, realization, seed_value, cumulative_deaths,
                    args.inter_edge_decay_rate))
            next_sample += args.sample_interval

    def capture_snapshots_until(limit: float):
        """Freeze graph/host state for all requested snapshots reached so far."""
        nonlocal next_snapshot
        while next_snapshot < len(requested_snapshots) and requested_snapshots[next_snapshot] <= limit:
            snapshots.append(snapshot_state(
                requested_snapshots[next_snapshot], state, virulence,
                active_adj, community_id, cumulative_deaths))
            next_snapshot += 1

    while t < args.t_max:
        # Gillespie step 1: compute the total event rate in the current state.
        events, total_rate = collect_events(
            state, virulence, active_adj, community_id, params, args.rho,
            args.inter_edge_decay_rate)
        if total_rate <= 0.0:
            break
        # Gillespie step 2: draw the waiting time until the next event.
        t_next = t + float(rng.exponential(1.0 / total_rate))
        event_limit = min(t_next, args.t_max)

        # Samples summarize the state just before the next event, after burn-in.
        capture_samples_until(event_limit)
        capture_snapshots_until(event_limit)

        if t_next > args.t_max:
            t = args.t_max
            break
        t = t_next
        # Gillespie step 3: choose which event fires at time t.
        kind, payload = choose_weighted_event(events, total_rate, rng)

        if kind == "transmission":
            susceptible, source = payload
            if state[susceptible] == STATE_S and state[source] == STATE_I:
                # Payload order is (new host, infecting host). Only the source
                # strain determines offspring virulence.
                state[susceptible] = STATE_I
                infection_count[susceptible] += 1
                # Virulence is inherited from the source strain with mutation.
                child_v = virulence[source] + rng.normal(0.0, args.sigma_m)
                virulence[susceptible] = float(np.clip(child_v, args.v_min, args.v_max))
                severed_edges[susceptible].clear()
        elif kind == "recovery":
            (i,) = payload
            if state[i] == STATE_I:
                # Recovered hosts retain immunity until a later waning event.
                state[i] = STATE_R
                restore_edges(i, state, active_adj, severed_edges)
        elif kind == "mortality":
            (i,) = payload
            if state[i] == STATE_I:
                # Deaths are permanent in this version: no birth or replacement
                # process replenishes the host population.
                state[i] = STATE_D
                cumulative_deaths += 1
                remove_all_edges(i, active_adj, severed_edges)
        elif kind == "waning":
            (i,) = payload
            if state[i] == STATE_R:
                # Waning restores susceptibility but does not alter contacts.
                state[i] = STATE_S
        elif kind == "edge_removal":
            u, v, owner = payload
            if v in active_adj[u] and state[owner] == STATE_I:
                remove_edge((u, v), active_adj)
                # Record the owner of the behavioral severing so only that
                # host's recovery restores this contact.
                severed_edges[owner].add(_edge(u, v))
        elif kind == "movement":
            (i,) = payload
            # Failed movement attempts are allowed; they simply mean no new
            # valid cross-community contact was available for this draw.
            add_intercommunity_edge(i, state, community_id, communities, active_adj, rng)
        elif kind == "inter_edge_decay":
            # Temporary inter-community contacts age out through this event.
            # The event is still checked because another process may have
            # removed the edge after the event list was built.
            u, v = payload
            if community_id[u] != community_id[v] and v in active_adj[u]:
                remove_edge((u, v), active_adj)

    # If the process ended early because no positive-rate events remain, still
    # fill requested sample/snapshot times with the final frozen state.
    capture_samples_until(args.t_max)
    capture_snapshots_until(args.t_max)

    final = summarize_state(t, state, virulence, active_adj, community_id)
    if not np.any(state == STATE_I):
        extinct = 1

    def sample_mean(key: str) -> float:
        """Average a metric over post-burn-in samples, ignoring NaNs."""
        vals = np.array([s[key] for s in samples], dtype=np.float64)
        if vals.size == 0:
            return math.nan
        if np.all(np.isnan(vals)):
            return math.nan
        return float(np.nanmean(vals))

    summary = {
        "realization": realization,
        "seed": seed_value,
        "N": args.N,
        "K": args.K,
        "t_final": t,
        "samples": len(samples),
        "inter_edge_decay_rate": args.inter_edge_decay_rate,
        "extinct": extinct,
        "mean_prevalence": sample_mean("prevalence"),
        "final_prevalence": final["prevalence"],
        "mean_s_frac": sample_mean("s_frac"),
        "final_s_frac": final["s_frac"],
        "mean_r_frac": sample_mean("r_frac"),
        "final_r_frac": final["r_frac"],
        "mean_d_frac": sample_mean("d_frac"),
        "final_d_frac": final["d_frac"],
        "mean_infected_virulence": sample_mean("mean_infected_virulence"),
        "final_infected_virulence": final["mean_infected_virulence"],
        "mean_virulence_variance": sample_mean("virulence_variance"),
        "final_virulence_variance": final["virulence_variance"],
        "cumulative_deaths": cumulative_deaths,
        "mean_active_edges": sample_mean("active_edges"),
        "final_active_edges": final["active_edges"],
        "mean_inter_edges": sample_mean("inter_edges"),
        "final_inter_edges": final["inter_edges"],
        "realized_clustering": final["realized_clustering"],
    }
    if return_samples or snapshot_times is not None:
        return summary, sample_rows, snapshots
    return summary


def run_sweep(args):
    """Run independent realizations and collect summary plus optional samples."""
    rng = np.random.default_rng(args.seed)
    rows = []
    all_samples = []
    t_start = _time.time()
    for realization in range(args.realizations):
        result = run_one_realization(args, realization, rng,
                                     return_samples=bool(args.samples_out))
        if args.samples_out:
            row, sample_rows, _snapshots = result
            all_samples.extend(sample_rows)
        else:
            row = result
        rows.append(row)
        if not args.quiet:
            elapsed = _time.time() - t_start
            print(f"[CommunitySIRS] {realization + 1}/{args.realizations} "
                  f"prevalence={row['final_prevalence']:.3f} "
                  f"deaths={row['cumulative_deaths']} "
                  f"elapsed={elapsed:.1f}s", file=sys.stderr)
    return rows, all_samples


def build_arg_parser(prog: str = "CommunitySIRS") -> argparse.ArgumentParser:
    """Define the public CLI shared by simulation and snapshot scripts."""
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--out", default="community_sirs_summary.csv",
                   help="output CSV path")
    p.add_argument("--samples-out", default=None,
                   help="optional per-sample CSV path for plotting steady-state trajectories")
    p.add_argument("--N", type=int, default=100, help="population size")
    p.add_argument("--K", type=int, choices=[2, 3], default=3,
                   help="number of communities; only 2 or 3 are supported")
    p.add_argument("--kbar", type=int, default=8,
                   help="Watts-Strogatz local mean degree target")
    p.add_argument("--rewiring-prob", type=float, default=0.01,
                   help="Watts-Strogatz rewiring probability")
    p.add_argument("--t-max", type=float, default=500.0)
    p.add_argument("--burn-in-time", type=float, default=100.0)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--realizations", type=int, default=10)
    p.add_argument("--initial-prevalence", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="suppress progress output")

    p.add_argument("--v-min", type=float, default=0.0)
    p.add_argument("--v-max", type=float, default=1.0)
    p.add_argument("--v-init", type=float, default=0.3)
    p.add_argument("--sigma-m", type=float, default=0.02,
                   help="per-transmission virulence mutation standard deviation")

    p.add_argument("--beta-min", type=float, default=0.0)
    p.add_argument("--beta-max", type=float, default=0.08)
    p.add_argument("--alpha-min", type=float, default=0.0)
    p.add_argument("--alpha-max", type=float, default=0.01)
    p.add_argument("--gamma-min", type=float, default=0.02)
    p.add_argument("--gamma-max", type=float, default=0.12)
    p.add_argument("--delta-min", type=float, default=0.0)
    p.add_argument("--delta-max", type=float, default=0.03)
    p.add_argument("--phi-max", type=float, default=0.02)
    p.add_argument("--inter-edge-decay-rate", type=float, default=0.08,
                   help="Gillespie removal rate per active inter-community edge. "
                        "Larger values shorten long-range edge lifetimes; tune "
                        "against --phi-max to target about 20-30 inter-community edges.")
    p.add_argument("--rho", type=float, default=0.01,
                   help="waning immunity rate")

    shapes = ["linear", "concave", "convex"]
    p.add_argument("--beta-shape", choices=shapes, default="concave")
    p.add_argument("--alpha-shape", choices=shapes, default="linear")
    p.add_argument("--gamma-shape", choices=shapes, default="linear")
    p.add_argument("--delta-shape", choices=shapes, default="linear")
    p.add_argument("--phi-shape", choices=shapes, default="linear")
    return p


def validate_args(args):
    """Fail fast on parameter combinations that would make the model invalid."""
    if args.N < args.K:
        raise SystemExit("--N must be at least --K")
    if args.v_max <= args.v_min:
        raise SystemExit("--v-max must be greater than --v-min")
    if not (args.v_min <= args.v_init <= args.v_max):
        raise SystemExit("--v-init must lie between --v-min and --v-max")
    if args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be positive")
    if args.t_max <= 0:
        raise SystemExit("--t-max must be positive")
    if args.burn_in_time < 0 or args.burn_in_time > args.t_max:
        raise SystemExit("--burn-in-time must be between 0 and --t-max")
    if not (0.0 <= args.initial_prevalence <= 1.0):
        raise SystemExit("--initial-prevalence must be between 0 and 1")
    if args.rewiring_prob < 0.0 or args.rewiring_prob > 1.0:
        raise SystemExit("--rewiring-prob must be between 0 and 1")
    for name in ["beta", "alpha", "gamma", "delta"]:
        lo = getattr(args, f"{name}_min")
        hi = getattr(args, f"{name}_max")
        if lo < 0 or hi < 0 or hi < lo:
            raise SystemExit(f"--{name}-min/--{name}-max must be nonnegative and ordered")
    if (args.phi_max < 0 or args.inter_edge_decay_rate < 0
            or args.rho < 0 or args.sigma_m < 0):
        raise SystemExit("--phi-max, --inter-edge-decay-rate, --rho, and --sigma-m must be nonnegative")


def main(prog: str = "CommunitySIRS"):
    """Parse CLI args, run simulations, and write requested CSV outputs."""
    parser = build_arg_parser(prog)
    args = parser.parse_args()
    validate_args(args)
    summary_rows, sample_rows = run_sweep(args)
    out_path = Path(args.out)
    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    if args.samples_out:
        samples_path = Path(args.samples_out)
        with samples_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            writer.writerows(sample_rows)


if __name__ == "__main__":
    main("community_sirs")
