"""Plot steady-state trajectories from CommunitySIRS sample CSV output."""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_samples(path: str) -> dict[int, list[dict[str, float]]]:
    """Load a CommunitySIRS --samples-out CSV grouped by realization."""
    by_realization: dict[int, list[dict[str, float]]] = defaultdict(list)
    with open(path, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            realization = int(row["realization"])
            parsed = {}
            for key, value in row.items():
                if key in ("realization", "seed"):
                    continue
                parsed[key] = float(value) if value not in ("", "nan") else np.nan
            by_realization[realization].append(parsed)
    for rows in by_realization.values():
        rows.sort(key=lambda item: item["t"])
    return by_realization


def aggregate(by_realization: dict[int, list[dict[str, float]]], metric: str):
    """Compute mean and standard deviation across realizations at each time."""
    times = sorted({row["t"] for rows in by_realization.values() for row in rows})
    means = []
    stds = []
    for t in times:
        vals = [
            row[metric]
            for rows in by_realization.values()
            for row in rows
            if row["t"] == t
        ]
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(np.nanmean(arr)))
            stds.append(float(np.nanstd(arr)))
    return np.array(times), np.array(means), np.array(stds)


def plot_mean_with_band(ax, by_realization, metric: str, label: str, color: str):
    """Draw one metric as a mean trajectory with optional realization spread."""
    t, mean, std = aggregate(by_realization, metric)
    ax.plot(t, mean, label=label, color=color, linewidth=1.8)
    if len(by_realization) > 1:
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)


def make_plot(samples_path: str, out_path: str, burn_in_time: float | None):
    """Create the four-panel steady-state diagnostic figure."""
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_realization = read_samples(samples_path)
    if not by_realization:
        raise ValueError("sample CSV has no rows")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    ax_prev, ax_vir, ax_edges, ax_state = axes.ravel()

    plot_mean_with_band(ax_prev, by_realization, "prevalence", "I/N", "#d62728")
    ax_prev.set_ylabel("Prevalence")

    plot_mean_with_band(ax_vir, by_realization, "mean_infected_virulence",
                        "Mean infected virulence", "#7b3294")
    ax_vir.set_ylabel("Virulence")

    plot_mean_with_band(ax_edges, by_realization, "active_edges", "Active edges", "#4d4d4d")
    plot_mean_with_band(ax_edges, by_realization, "inter_edges", "Inter-community edges", "#1f78b4")
    ax_edges.set_ylabel("Edges")
    ax_edges.legend(frameon=False, fontsize=8)

    plot_mean_with_band(ax_state, by_realization, "s_frac", "S", "#bdbdbd")
    plot_mean_with_band(ax_state, by_realization, "r_frac", "R", "#4c78a8")
    plot_mean_with_band(ax_state, by_realization, "d_frac", "D", "#252525")
    ax_state.set_ylabel("Fraction")
    ax_state.legend(frameon=False, fontsize=8, ncol=3)

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time")
        if burn_in_time is not None:
            ax.axvline(burn_in_time, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.suptitle("Community SIRS steady-state trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the plotting CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True, help="CommunitySIRS --samples-out CSV")
    p.add_argument("--out", default=None, help="output PNG path")
    p.add_argument("--burn-in-time", type=float, default=None,
                   help="optional burn-in marker time")
    return p


def main():
    """Read sample CSV and save the trajectory PNG."""
    args = build_arg_parser().parse_args()
    out_path = args.out
    if out_path is None:
        base, _ext = os.path.splitext(args.samples)
        out_path = base + ".png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.samples, out_path, args.burn_in_time)


if __name__ == "__main__":
    main()
