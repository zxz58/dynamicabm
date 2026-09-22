"""Compare multiple CommunitySIRS sample CSV outputs in one 1x2 figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from PlotCommunitySIRS import (
    aggregate,
    parse_sample_arg,
    read_samples,
    realization_means,
    setup_matplotlib,
)


def draw_virulence_vs_cross_community_transmissions(ax, sample_sets, colors):
    """Draw one realization-level point cloud per scenario.

    The x-axis is the post-burn-in/sample-window mean of the cumulative
    cross-community transmission counter. The y-axis is the corresponding mean
    infected virulence. Both means ignore NaN values, which keeps extinct
    periods from crashing the plot.
    """
    for idx, (label, by_realization) in enumerate(sample_sets):
        color = colors(idx % 10)
        x = realization_means(by_realization, "cumulative_intercommunity_transmissions")
        y = realization_means(by_realization, "mean_infected_virulence")
        ax.scatter(x, y, color=color, alpha=0.65, s=28, label=label)
        if x.size > 1 and not np.all(np.isnan(x)) and not np.all(np.isnan(y)):
            ax.scatter([np.nanmean(x)], [np.nanmean(y)], marker="D", s=64,
                       color=color, edgecolors="#222222", linewidths=0.6)
    ax.set_xlabel("Mean cumulative cross-community transmissions")
    ax.set_ylabel("Mean infected virulence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def draw_virulence_time_comparison(ax, sample_sets, burn_in_time: float | None, colors):
    """Draw mean infected virulence trajectories for all scenarios."""
    for idx, (label, by_realization) in enumerate(sample_sets):
        color = colors(idx % 10)
        t, mean, std = aggregate(by_realization, "mean_infected_virulence")
        ax.plot(t, mean, color=color, linewidth=1.8, label=label)
        if len(by_realization) > 1:
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    if burn_in_time is not None:
        ax.axvline(burn_in_time, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean infected virulence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def make_comparison_plot(sample_sets, out_path: str, burn_in_time: float | None):
    """Create the two-panel multi-scenario comparison figure."""
    if not sample_sets:
        raise ValueError("at least one --samples input is required")
    for label, by_realization in sample_sets:
        if not by_realization:
            raise ValueError(f"sample CSV for {label!r} has no rows")

    plt = setup_matplotlib()
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax_scatter, ax_time = axes

    draw_virulence_vs_cross_community_transmissions(ax_scatter, sample_sets, colors)
    draw_virulence_time_comparison(ax_time, sample_sets, burn_in_time, colors)

    fig.suptitle("Community SIRS scenario comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the comparison plotting CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True, action="append",
                   help="CommunitySIRS --samples-out CSV. Use PATH or LABEL:PATH; repeat for scenarios.")
    p.add_argument("--out", default="comparison.png",
                   help="output PNG path")
    p.add_argument("--burn-in-time", type=float, default=100,
                   help="optional burn-in marker time on the right panel")
    return p


def main():
    """Read sample CSVs and save the comparison figure."""
    args = build_arg_parser().parse_args()
    sample_inputs = [parse_sample_arg(value) for value in args.samples]
    sample_sets = [(label, read_samples(path)) for label, path in sample_inputs]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_comparison_plot(sample_sets, args.out, args.burn_in_time)


if __name__ == "__main__":
    main()
