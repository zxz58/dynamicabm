"""Save selected-time network snapshots from one CommunitySIRS realization."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

import community_sirs


NODE_COLORS = {
    community_sirs.STATE_S: "#d9d9d9",
    community_sirs.STATE_R: "#4c78a8",
    community_sirs.STATE_D: "#252525",
}


def parse_times(value: str) -> list[float]:
    """Parse selected continuous snapshot times from a comma-separated string."""
    times = sorted({float(x) for x in value.split(",") if x.strip()})
    if any(t < 0 for t in times):
        raise ValueError("snapshot times must be non-negative")
    return times


def default_snapshot_times(args) -> list[float]:
    """Use start, burn-in, and final time when no explicit times are provided."""
    return sorted({0.0, float(args.burn_in_time), float(args.t_max)})


def community_layout(snapshot: dict, seed: int | None, iterations: int):
    """Place communities in fixed macro-positions with local spring layouts."""
    import networkx as nx

    state = snapshot["state"]
    active_adj = snapshot["active_adj"]
    community_id = snapshot["community_id"]
    K = int(np.max(community_id)) + 1
    centers = {
        2: [(-1.3, 0.0), (1.3, 0.0)],
        3: [(0.0, 1.35), (-1.25, -0.75), (1.25, -0.75)],
    }[K]
    pos = {}
    for c in range(K):
        # Layout each community internally, then translate it to its community
        # center so the snapshot clearly shows between-community bridges.
        nodes = np.where(community_id == c)[0].tolist()
        local = nx.Graph()
        local.add_nodes_from(nodes)
        for u, v in community_sirs.active_edges(active_adj):
            if community_id[u] == c and community_id[v] == c:
                local.add_edge(u, v)
        local_pos = nx.spring_layout(local, seed=None if seed is None else seed + c,
                                     iterations=iterations)
        cx, cy = centers[c]
        for node, (x, y) in local_pos.items():
            scale = 0.68
            pos[node] = (cx + scale * x, cy + scale * y)
    return pos


def draw_snapshot(snapshot: dict, out_path: str, layout_seed: int | None,
                  layout_iterations: int):
    """Render one saved simulation state as a network PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    state = snapshot["state"]
    virulence = snapshot["virulence"]
    active_adj = snapshot["active_adj"]
    community_id = snapshot["community_id"]
    t = snapshot["t"]

    G = nx.Graph()
    G.add_nodes_from(range(len(state)))
    edges = list(community_sirs.active_edges(active_adj))
    G.add_edges_from(edges)
    pos = community_layout(snapshot, layout_seed, layout_iterations)

    local_edges = [(u, v) for u, v in edges if community_id[u] == community_id[v]]
    inter_edges = [(u, v) for u, v in edges if community_id[u] != community_id[v]]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_axis_off()
    # Draw local structure first and inter-community bridges on top.
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, ax=ax, edge_color="#9e9e9e",
                           width=0.9, alpha=0.75)
    nx.draw_networkx_edges(G, pos, edgelist=inter_edges, ax=ax, edge_color="#525252",
                           width=1.0, alpha=0.65)

    cmap = plt.get_cmap("plasma")
    vmin = float(np.nanmin(virulence))
    vmax = float(np.nanmax(virulence))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    node_colors = []
    node_sizes = []
    for s, v in zip(state.tolist(), virulence.tolist()):
        # Infected hosts use a continuous virulence colormap; other states use
        # categorical colors so the disease state remains easy to read.
        if s == community_sirs.STATE_I:
            node_colors.append(cmap(norm(v)))
            node_sizes.append(48)
        else:
            node_colors.append(NODE_COLORS[s])
            node_sizes.append(36 if s != community_sirs.STATE_D else 24)

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        linewidths=0.35, edgecolors="#262626")

    summary = community_sirs.summarize_state(t, state, virulence, active_adj, community_id)
    ax.set_title(
        f"Community SIRS network snapshot - t={t:g}\n"
        f"S={summary['S']}  I={summary['I']}  R={summary['R']}  D={summary['D']}  "
        f"inter edges={int(summary['inter_edges'])}",
        fontsize=11)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Susceptible",
               markerfacecolor=NODE_COLORS[community_sirs.STATE_S],
               markeredgecolor="#262626", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Recovered",
               markerfacecolor=NODE_COLORS[community_sirs.STATE_R],
               markeredgecolor="#262626", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Dead",
               markerfacecolor=NODE_COLORS[community_sirs.STATE_D],
               markeredgecolor="#262626", markersize=7),
        Line2D([0], [0], color="#525252", label="Inter-community edge", linewidth=1.6),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, -0.02), fontsize=8)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Infected virulence", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def snapshot_metadata(snapshot: dict, filename: str) -> dict:
    """Create the metadata row that accompanies one snapshot PNG."""
    summary = community_sirs.summarize_state(
        snapshot["t"], snapshot["state"], snapshot["virulence"],
        snapshot["active_adj"], snapshot["community_id"])
    return {
        "t": snapshot["t"],
        "S": summary["S"],
        "I": summary["I"],
        "R": summary["R"],
        "D": summary["D"],
        "prevalence": summary["prevalence"],
        "mean_infected_virulence": summary["mean_infected_virulence"],
        "virulence_variance": summary["virulence_variance"],
        "active_edges": summary["active_edges"],
        "inter_edges": summary["inter_edges"],
        "realized_clustering": summary["realized_clustering"],
        "cumulative_deaths": snapshot["cumulative_deaths"],
        "file": filename,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Reuse the model CLI and add snapshot-specific options."""
    p = community_sirs.build_arg_parser("CommunitySIRSNetworkSnapshots")
    p.set_defaults(out="community_sirs_snapshot_summary.csv", realizations=1)
    p.add_argument("--out-dir", default="community_sirs_snapshots")
    p.add_argument("--snapshot-times", default=None,
                   help="comma-separated continuous times; default is 0,burn-in,t-max")
    p.add_argument("--layout-seed", type=int, default=None,
                   help="seed for fixed community-aware layout; defaults to --seed")
    p.add_argument("--layout-iterations", type=int, default=80)
    return p


def main():
    """Run one realization, render selected snapshots, and write metadata."""
    args = build_arg_parser().parse_args()
    community_sirs.validate_args(args)
    snapshot_times = default_snapshot_times(args) if args.snapshot_times is None else parse_times(args.snapshot_times)
    if max(snapshot_times, default=0.0) > args.t_max:
        raise SystemExit("--snapshot-times cannot exceed --t-max")

    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = os.path.join(args.out_dir, ".mplconfig")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    _summary, _samples, snapshots = community_sirs.run_one_realization(
        args, 0, rng, snapshot_times=snapshot_times)

    metadata = []
    layout_seed = args.seed if args.layout_seed is None else args.layout_seed
    for snapshot in snapshots:
        stamp = str(snapshot["t"]).replace(".", "p")
        filename = f"snapshot_t_{stamp}.png"
        out_path = os.path.join(args.out_dir, filename)
        draw_snapshot(snapshot, out_path, layout_seed, args.layout_iterations)
        metadata.append(snapshot_metadata(snapshot, filename))
        if not args.quiet:
            print(f"[CommunitySIRSNetworkSnapshots] saved {out_path}")

    metadata_path = os.path.join(args.out_dir, "snapshot_metadata.csv")
    fieldnames = [
        "t", "S", "I", "R", "D", "prevalence", "mean_infected_virulence",
        "virulence_variance", "active_edges", "inter_edges",
        "realized_clustering", "cumulative_deaths", "file",
    ]
    with open(metadata_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


if __name__ == "__main__":
    main()
