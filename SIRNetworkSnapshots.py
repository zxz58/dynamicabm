"""
SIRNetworkSnapshots.py
======================
Run one SIR alpha-mutation simulation and save PNG snapshots of the contact
network at selected days.

The epidemic dynamics match the SIR branch of phase_diagram_alpha.py:
fitness Psi affects recovery through alpha/Psi, transmission is governed by
beta with the existing avoidance/distancing/symptomatic_fraction modifiers,
and recovered nodes move to the absorbing -2 state.

Isolation in this tool is visual-only. During the isolation window, rendered
edges incident to currently infected symptomatic nodes are hidden; the
infection and recovery calculations still use the full network and the same
probabilities as phase_diagram_alpha.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time as _time

import numpy as np

from common import generate_er_network, fitness_random_walk, pick_random_neighbor


STATE_SUSCEPTIBLE = -1.0
STATE_INFECTED = 1.0
STATE_RECOVERED = -2.0

NODE_COLORS = {
    "susceptible": "#d9d9d9",
    "asymptomatic_infected": "#f5a623",
    "symptomatic_infected": "#d62728",
    "recovered": "#4c78a8",
}


def parse_days(s: str) -> list[int]:
    days = sorted({int(x) for x in s.split(",") if x.strip() != ""})
    if any(day < 0 for day in days):
        raise ValueError("snapshot days must be non-negative")
    return days


def edge_list_from_sparse(A) -> list[tuple[int, int]]:
    rows, cols = A.nonzero()
    return [(int(i), int(j)) for i, j in zip(rows.tolist(), cols.tolist()) if i < j]


def visible_edges_for_day(edges: list[tuple[int, int]], state: np.ndarray,
                          symptomatic: np.ndarray, day: int,
                          isolation_start: int, isolation_end: int):
    if not (isolation_start <= day <= isolation_end):
        return edges, 0

    isolated = (state > 0) & symptomatic
    visible = [(u, v) for u, v in edges if not (isolated[u] or isolated[v])]
    return visible, len(edges) - len(visible)


def node_colors(state: np.ndarray, symptomatic: np.ndarray) -> list[str]:
    colors = []
    for s, is_symptomatic in zip(state.tolist(), symptomatic.tolist()):
        if s == STATE_SUSCEPTIBLE:
            colors.append(NODE_COLORS["susceptible"])
        elif s == STATE_RECOVERED:
            colors.append(NODE_COLORS["recovered"])
        elif is_symptomatic:
            colors.append(NODE_COLORS["symptomatic_infected"])
        else:
            colors.append(NODE_COLORS["asymptomatic_infected"])
    return colors


def snapshot_counts(day: int, state: np.ndarray, symptomatic: np.ndarray,
                    total_edges: int, hidden_edges: int) -> dict[str, int]:
    infected = state > 0
    symptomatic_infected = infected & symptomatic
    return {
        "day": day,
        "susceptible": int(np.count_nonzero(state == STATE_SUSCEPTIBLE)),
        "infected": int(np.count_nonzero(infected)),
        "recovered": int(np.count_nonzero(state == STATE_RECOVERED)),
        "symptomatic_infected": int(np.count_nonzero(symptomatic_infected)),
        "asymptomatic_infected": int(np.count_nonzero(infected & ~symptomatic)),
        "visible_edges": int(total_edges - hidden_edges),
        "hidden_edges": int(hidden_edges),
    }


def draw_snapshot(G, pos, state: np.ndarray, symptomatic: np.ndarray,
                  visible_edges: list[tuple[int, int]], hidden_edges: int,
                  day: int, out_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_axis_off()

    nx.draw_networkx_edges(
        G, pos, edgelist=visible_edges, ax=ax, edge_color="#eeeeee",
        width=0.45, alpha=0.35)
    nx.draw_networkx_edges(
        G, pos, edgelist=visible_edges, ax=ax, edge_color="#888888",
        width=0.75, alpha=0.75)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors(state, symptomatic),
        node_size=42, linewidths=0.35, edgecolors="#262626")

    counts = snapshot_counts(day, state, symptomatic, G.number_of_edges(), hidden_edges)
    ax.set_title(
        f"SIR alpha network snapshot - day {day}\n"
        f"S={counts['susceptible']}  I={counts['infected']}  "
        f"R={counts['recovered']}  hidden edges={hidden_edges}",
        fontsize=11)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Susceptible",
               markerfacecolor=NODE_COLORS["susceptible"],
               markeredgecolor="#262626", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Asymptomatic infected",
               markerfacecolor=NODE_COLORS["asymptomatic_infected"],
               markeredgecolor="#262626", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Symptomatic infected",
               markerfacecolor=NODE_COLORS["symptomatic_infected"],
               markeredgecolor="#262626", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Recovered",
               markerfacecolor=NODE_COLORS["recovered"],
               markeredgecolor="#262626", markersize=7),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def simulate_and_snapshot(args):
    import networkx as nx

    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = os.path.join(args.out_dir, ".mplconfig")

    snapshot_days = parse_days(args.snapshot_days)
    if not snapshot_days:
        raise ValueError("at least one snapshot day is required")
    if max(snapshot_days) > args.TT:
        raise ValueError("snapshot days cannot exceed TT")
    if args.isolation_end < args.isolation_start:
        raise ValueError("isolation-end must be greater than or equal to isolation-start")
    if not (0.0 <= args.avoidance <= 1.0 and 0.0 <= args.distancing <= 1.0):
        raise ValueError("avoidance and distancing must be between 0 and 1")
    if not (0.0 <= args.symptomatic_fraction <= 1.0):
        raise ValueError("symptomatic-fraction must be between 0 and 1")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    rng = np.random.default_rng(args.seed)
    A, adj_list = generate_er_network(args.N, args.num_edges, rng)
    edges = edge_list_from_sparse(A)
    G = nx.Graph()
    G.add_nodes_from(range(args.N))
    G.add_edges_from(edges)

    layout_seed = args.seed if args.layout_seed is None else args.layout_seed
    pos = nx.spring_layout(G, seed=layout_seed, iterations=args.layout_iterations)

    state = np.full(args.N, STATE_SUSCEPTIBLE, dtype=np.float64)
    psi = np.ones(args.N, dtype=np.float64)
    symptomatic = np.zeros(args.N, dtype=bool)
    n0 = max(args.N // 50, 1)
    state[:n0] = STATE_INFECTED
    symptomatic[:n0] = rng.random(n0) < args.symptomatic_fraction

    beta_effective_sym = args.beta * (1.0 - args.avoidance) * (1.0 - args.distancing)
    beta_effective_asym = args.beta * (1.0 - args.distancing)

    metadata = []
    requested = set(snapshot_days)

    def capture(day: int):
        visible_edges, hidden_edges = visible_edges_for_day(
            edges, state, symptomatic, day, args.isolation_start, args.isolation_end)
        out_path = os.path.join(args.out_dir, f"snapshot_day_{day:04d}.png")
        draw_snapshot(G, pos, state, symptomatic, visible_edges, hidden_edges,
                      day, out_path)
        row = snapshot_counts(day, state, symptomatic, len(edges), hidden_edges)
        row["file"] = os.path.basename(out_path)
        metadata.append(row)

    if 0 in requested:
        capture(0)

    for day in range(1, args.TT + 1):
        infected_mask = state > 0
        symptomatic_infected_mask = infected_mask & symptomatic
        asymptomatic_infected_mask = infected_mask & ~symptomatic
        linum_sym = A.dot(symptomatic_infected_mask.astype(np.float64))
        linum_asym = A.dot(asymptomatic_infected_mask.astype(np.float64))

        susceptible_idx = np.where(state == STATE_SUSCEPTIBLE)[0]
        if susceptible_idx.size:
            escape_prob = (
                (1.0 - beta_effective_sym) ** linum_sym[susceptible_idx]
                * (1.0 - beta_effective_asym) ** linum_asym[susceptible_idx]
            )
            newly_infected = susceptible_idx[
                rng.random(susceptible_idx.shape[0]) < (1.0 - escape_prob)]
        else:
            newly_infected = np.empty(0, dtype=np.int64)

        if newly_infected.size:
            psi[newly_infected] = pick_random_neighbor(
                newly_infected, adj_list, infected_mask, psi, rng, weighted=False)
            symptomatic[newly_infected] = (
                rng.random(newly_infected.size) < args.symptomatic_fraction)
            state[newly_infected] = STATE_INFECTED

        infected_idx = np.where(infected_mask)[0]
        if infected_idx.size:
            with np.errstate(divide="ignore"):
                recovery_prob = args.alpha / psi[infected_idx]
            recovering = infected_idx[rng.random(infected_idx.shape[0]) < recovery_prob]
            state[recovering] = STATE_RECOVERED

        infected_idx2 = np.where(state > 0)[0]
        if infected_idx2.size:
            psi[infected_idx2] = fitness_random_walk(
                psi[infected_idx2], args.sigma, args.X, 0.0, args.psi_cap, rng)

        if day in requested:
            capture(day)
            if not args.quiet:
                print(f"[SIRNetworkSnapshots] saved day {day}", file=sys.stderr)

    metadata_path = os.path.join(args.out_dir, "snapshot_metadata.csv")
    fieldnames = [
        "day", "susceptible", "infected", "recovered",
        "symptomatic_infected", "asymptomatic_infected",
        "visible_edges", "hidden_edges", "file",
    ]
    with open(metadata_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    return metadata_path, len(metadata)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="snapshots")
    p.add_argument("--snapshot-days", default="0,10,20,30,60")
    p.add_argument("--isolation-start", type=int, default=20)
    p.add_argument("--isolation-end", type=int, default=50)
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--TT", type=int, default=3000)
    p.add_argument("--beta", type=float, default=0.008)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--sigma", type=float, default=0.03)
    p.add_argument("--X", type=float, default=0.0)
    p.add_argument("--psi-cap", type=float, default=20.0)
    p.add_argument("--num-edges", type=int, default=37500)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--avoidance", type=float, default=0.0)
    p.add_argument("--distancing", type=float, default=0.0)
    p.add_argument("--symptomatic-fraction", type=float, default=1.0)
    p.add_argument("--layout-seed", type=int, default=None,
                   help="seed for the fixed spring layout; defaults to --seed")
    p.add_argument("--layout-iterations", type=int, default=50,
                   help="spring-layout iterations")
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    t_start = _time.time()
    metadata_path, n_snapshots = simulate_and_snapshot(args)
    if not args.quiet:
        elapsed = _time.time() - t_start
        print(f"[SIRNetworkSnapshots] wrote {n_snapshots} snapshots", file=sys.stderr)
        print(f"[SIRNetworkSnapshots] metadata saved to {metadata_path}", file=sys.stderr)
        print(f"[SIRNetworkSnapshots] elapsed={elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
