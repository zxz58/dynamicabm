"""
community_sirs.py
=================
Focused SIRS agent-based model for endemic virulence evolution on a small
community-structured dynamic contact network.

The model uses continuous-time Gillespie dynamics. Hosts live in two or
three Watts-Strogatz communities. Virulence is a heritable continuous
phenotype assigned at infection and controls transmission, recovery,
disease mortality, behavioral edge removal, and inter-community movement.
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


SUMMARY_FIELDS = [
    "realization",
    "seed",
    "N",
    "K",
    "t_final",
    "samples",
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

SAMPLE_FIELDS = [
    "realization",
    "seed",
    "t",
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
    return (u, v) if u < v else (v, u)


def split_communities(N: int, K: int) -> tuple[np.ndarray, list[np.ndarray]]:
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
    """Return current usable contact edges without double-counting."""
    out = set()
    for u, neigh in enumerate(active_adj):
        for v in neigh:
            if u < v:
                out.add((u, v))
    return out


def realized_clustering(active_adj: list[set[int]], alive_mask: np.ndarray) -> float:
    """Compute clustering on the active network after removing dead hosts."""
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
    """Convert virulence into a bounded rate with controlled shape/direction."""
    denom = params.v_max - params.v_min
    if denom <= 0:
        raise ValueError("v_max must be greater than v_min")
    z = (v - params.v_min) / denom
    factor = shape_value(z, shape)
    if not increasing:
        factor = 1.0 - factor
    return low + (high - low) * factor


def beta_of(v, params: TradeoffParams):
    return bounded_tradeoff(v, params, params.beta_min, params.beta_max,
                            params.beta_shape, increasing=True)


def alpha_of(v, params: TradeoffParams):
    return bounded_tradeoff(v, params, params.alpha_min, params.alpha_max,
                            params.alpha_shape, increasing=True)


def gamma_of(v, params: TradeoffParams):
    return bounded_tradeoff(v, params, params.gamma_min, params.gamma_max,
                            params.gamma_shape, increasing=False)


def delta_of(v, params: TradeoffParams):
    return bounded_tradeoff(v, params, params.delta_min, params.delta_max,
                            params.delta_shape, increasing=True)


def phi_of(v, params: TradeoffParams):
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


def count_inter_edges(active_adj: list[set[int]], community_id: np.ndarray) -> int:
    return sum(1 for u, v in active_edges(active_adj) if community_id[u] != community_id[v])


def summarize_state(t: float, state: np.ndarray, virulence: np.ndarray,
                    active_adj: list[set[int]], community_id: np.ndarray) -> dict[str, float]:
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
    threshold = rng.random() * total_rate
    acc = 0.0
    for kind, payload, rate in events:
        acc += rate
        if threshold <= acc:
            return kind, payload
    return events[-1][0], events[-1][1]


def collect_events(state: np.ndarray, virulence: np.ndarray, active_adj: list[set[int]],
                   params: TradeoffParams, rho: float):
    events: list[tuple[str, tuple, float]] = []

    # Edge-local events: transmission along S-I contacts and behavioral
    # removal of contacts incident to infectious hosts.
    for u, v in active_edges(active_adj):
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
        rate = float(phi_of(virulence[i], params)) if state[i] == STATE_I else params.phi_max
        events.append(("movement", (int(i),), rate))

    total_rate = sum(rate for _, _, rate in events if rate > 0.0)
    if total_rate <= 0.0:
        return [], 0.0
    events = [(kind, payload, rate) for kind, payload, rate in events if rate > 0.0]
    return events, total_rate


def snapshot_state(t: float, state: np.ndarray, virulence: np.ndarray,
                   active_adj: list[set[int]], community_id: np.ndarray,
                   cumulative_deaths: int) -> dict:
    return {
        "t": t,
        "state": state.copy(),
        "virulence": virulence.copy(),
        "active_adj": copy.deepcopy(active_adj),
        "community_id": community_id.copy(),
        "cumulative_deaths": cumulative_deaths,
    }


def add_sample_context(sample: dict[str, float], realization: int,
                       seed_value: int | str, cumulative_deaths: int) -> dict:
    row = {
        "realization": realization,
        "seed": seed_value,
        **sample,
        "cumulative_deaths": cumulative_deaths,
        "extinct": int(sample["I"] == 0),
    }
    return {field: row.get(field, "") for field in SAMPLE_FIELDS}


def run_one_realization(args, realization: int, rng: np.random.Generator,
                        return_samples: bool = False,
                        snapshot_times: list[float] | None = None):
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
        nonlocal next_sample
        while next_sample <= limit:
            sample = summarize_state(next_sample, state, virulence, active_adj, community_id)
            samples.append(sample)
            if return_samples:
                sample_rows.append(add_sample_context(
                    sample, realization, seed_value, cumulative_deaths))
            next_sample += args.sample_interval

    def capture_snapshots_until(limit: float):
        nonlocal next_snapshot
        while next_snapshot < len(requested_snapshots) and requested_snapshots[next_snapshot] <= limit:
            snapshots.append(snapshot_state(
                requested_snapshots[next_snapshot], state, virulence,
                active_adj, community_id, cumulative_deaths))
            next_snapshot += 1

    while t < args.t_max:
        events, total_rate = collect_events(state, virulence, active_adj, params, args.rho)
        if total_rate <= 0.0:
            break
        t_next = t + float(rng.exponential(1.0 / total_rate))
        event_limit = min(t_next, args.t_max)

        # Samples summarize the state just before the next event, after burn-in.
        capture_samples_until(event_limit)
        capture_snapshots_until(event_limit)

        if t_next > args.t_max:
            t = args.t_max
            break
        t = t_next
        kind, payload = choose_weighted_event(events, total_rate, rng)

        if kind == "transmission":
            susceptible, source = payload
            if state[susceptible] == STATE_S and state[source] == STATE_I:
                state[susceptible] = STATE_I
                infection_count[susceptible] += 1
                # Virulence is inherited from the source strain with mutation.
                child_v = virulence[source] + rng.normal(0.0, args.sigma_m)
                virulence[susceptible] = float(np.clip(child_v, args.v_min, args.v_max))
                severed_edges[susceptible].clear()
        elif kind == "recovery":
            (i,) = payload
            if state[i] == STATE_I:
                state[i] = STATE_R
                restore_edges(i, state, active_adj, severed_edges)
        elif kind == "mortality":
            (i,) = payload
            if state[i] == STATE_I:
                state[i] = STATE_D
                cumulative_deaths += 1
                remove_all_edges(i, active_adj, severed_edges)
        elif kind == "waning":
            (i,) = payload
            if state[i] == STATE_R:
                state[i] = STATE_S
        elif kind == "edge_removal":
            u, v, owner = payload
            if v in active_adj[u] and state[owner] == STATE_I:
                active_adj[u].discard(v)
                active_adj[v].discard(u)
                # Record the owner of the behavioral severing so only that
                # host's recovery restores this contact.
                severed_edges[owner].add(_edge(u, v))
        elif kind == "movement":
            (i,) = payload
            add_intercommunity_edge(i, state, community_id, communities, active_adj, rng)

    capture_samples_until(args.t_max)
    capture_snapshots_until(args.t_max)

    final = summarize_state(t, state, virulence, active_adj, community_id)
    if not np.any(state == STATE_I):
        extinct = 1

    def sample_mean(key: str) -> float:
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
    if args.phi_max < 0 or args.rho < 0 or args.sigma_m < 0:
        raise SystemExit("--phi-max, --rho, and --sigma-m must be nonnegative")


def main(prog: str = "CommunitySIRS"):
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
