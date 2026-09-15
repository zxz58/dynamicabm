"""
InfectionTimeSeries.py
=======================
Runs one epidemic scenario (or several, swept over --avoidance and/or
--distancing values) at a *fixed* (beta, sigma) and tracks, at every
timestep, averaged across `--realizations` independent runs:

    cumulative_infections                -- total infection EVENTS so far
    cumulative_symptomatic_infections    -- of which, symptomatic cases
    cumulative_asymptomatic_infections   -- of which, asymptomatic cases
    current_infected                     -- number infected right now
    current_symptomatic_infected         -- of which, symptomatic
    current_asymptomatic_infected        -- of which, asymptomatic
    susceptible_remaining                -- number never yet infected

(For SIR, "cumulative" counts unique individuals ever infected, since
there's no reinfection; for SIS it also counts reinfection events, the
standard "cumulative incidence" measure for a model that allows
reinfection.)

Symptomatic vs. asymptomatic infections
-----------------------------------------
Each new infection is randomly assigned symptomatic (visible) or
asymptomatic (invisible) via --symptomatic-fraction (0-1, default 1.0 =
everyone symptomatic, matching the original model). This is what makes
--avoidance and --distancing behave differently rather than being two
names for the same number:

    --avoidance   only discounts transmission from SYMPTOMATIC infected
                  neighbors (you can't avoid someone you can't tell is sick)
    --distancing  discounts transmission from ALL infected neighbors,
                  symptomatic or not

At symptomatic_fraction=1.0 the two are mathematically interchangeable (as
before). Set it below 1.0 -- e.g. 0.5 or 0.6 -- to see avoidance become
strictly weaker than distancing at the same nominal strength, since
avoidance can never touch the asymptomatic-driven share of transmission.
The symptomatic/asymptomatic breakdown columns above let you see exactly
how much of the remaining spread is running through invisible carriers.

Usage
-----
Single scenario:
    python InfectionTimeSeries.py --beta 0.02 --sigma 0.05 --avoidance 0.5 \\
        --symptomatic-fraction 0.6

Sweep (cartesian product of the two lists, so 3x2=6 scenarios here):
    python InfectionTimeSeries.py --beta 0.02 --sigma 0.05 \\
        --avoidance 0,0.25,0.5 --distancing 0,0.3 --symptomatic-fraction 0.6

Output is a long-format CSV (`--out`, default timeseries.csv) with columns:
day, avoidance, distancing, symptomatic_fraction, scenario, plus a mean and
std column for each of the seven tracked quantities above -- easy to load
with pandas/Excel for your own analysis.

In addition, a PNG plot is saved automatically (one line per scenario,
shaded +/-1 std band across realizations) -- by default next to the CSV
with the same name but a .png extension. Control this with --plot-metric
(which of the tracked quantities to plot), --plot-out (custom path),
--plot-title, or turn it off with --no-plot. This needs matplotlib
(`pip install matplotlib`); if it isn't installed, the script still runs
and just skips the plot with a warning.

Choose which underlying model to run with --model; it must be one already
implemented in this package (SIR_alpha, SIS_alpha, SIR_beta, SIS_beta,
SIR_combined, SIS_combined). For the combined model, use --psi-alpha-cap /
--psi-beta-cap instead of --psi-cap (which only applies to the alpha/beta
models).
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time

import numpy as np

import phase_diagram_alpha
import phase_diagram_beta
import phase_diagram_combined

METRICS = [
    "cumulative_infections",
    "cumulative_symptomatic_infections",
    "cumulative_asymptomatic_infections",
    "current_infected",
    "current_symptomatic_infected",
    "current_asymptomatic_infected",
    "susceptible_remaining",
]


def parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def run_scenario(model: str, N: int, TT: int, beta: float, sigma: float,
                  alpha: float, X: float, psi_cap: float, psi_alpha_cap: float,
                  psi_beta_cap: float, num_edges: int, n_realizations: int,
                  avoidance: float, distancing: float, symptomatic_fraction: float,
                  seed: int | None):
    """Run `n_realizations` realizations of one (avoidance, distancing)
    scenario and return per-day mean/std arrays across realizations, for
    every metric in METRICS. `psi_cap` is used for the alpha/beta models;
    `psi_alpha_cap`/`psi_beta_cap` are used for the combined model.
    """
    rng = np.random.default_rng(seed)
    runs = {m: np.empty((n_realizations, TT), dtype=np.float64) for m in METRICS}

    if model in ("SIR_alpha", "SIS_alpha"):
        engine = phase_diagram_alpha
        dynamics = "SIR" if model == "SIR_alpha" else "SIS"
        for r in range(n_realizations):
            _, _, series = engine.run_one_realization(
                N, TT, beta, alpha, sigma, X, psi_cap, dynamics, 0, num_edges, rng,
                avoidance=avoidance, distancing=distancing,
                symptomatic_fraction=symptomatic_fraction, track_timeseries=True)
            for m in METRICS:
                runs[m][r] = series[m]
    elif model in ("SIR_beta", "SIS_beta"):
        engine = phase_diagram_beta
        dynamics = "SIR" if model == "SIR_beta" else "SIS"
        for r in range(n_realizations):
            _, _, series = engine.run_one_realization(
                N, TT, beta, alpha, sigma, X, psi_cap, dynamics, 0, num_edges, rng,
                avoidance=avoidance, distancing=distancing,
                symptomatic_fraction=symptomatic_fraction, track_timeseries=True)
            for m in METRICS:
                runs[m][r] = series[m]
    elif model in ("SIR_combined", "SIS_combined"):
        dynamics = "SIR" if model == "SIR_combined" else "SIS"
        for r in range(n_realizations):
            _, _, series = phase_diagram_combined.run_one_realization(
                N, TT, beta, alpha, sigma, X, psi_alpha_cap, psi_beta_cap,
                dynamics, 0, num_edges, rng, avoidance=avoidance,
                distancing=distancing, symptomatic_fraction=symptomatic_fraction,
                track_timeseries=True)
            for m in METRICS:
                runs[m][r] = series[m]
    else:
        raise ValueError(f"unknown model {model!r}")

    result = {}
    for m in METRICS:
        result[f"{m}_mean"] = runs[m].mean(axis=0)
        result[f"{m}_std"] = runs[m].std(axis=0)
    return result


def make_plot(days, results_by_label, metric: str, plot_path: str, title: str):
    """Save a PNG line plot of `metric` over time, one line per scenario
    label, with a shaded +/-1 std band across realizations. Silently skips
    (with a warning to stderr) if matplotlib isn't installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed; just write the file
        import matplotlib.pyplot as plt
    except ImportError:
        print("[InfectionTimeSeries] matplotlib not installed -- skipping "
              "plot. Run `pip install matplotlib` to enable it.", file=sys.stderr)
        return

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, result in results_by_label.items():
        mean = result[mean_key]
        ax.plot(days, mean, label=label, linewidth=1.8)
        if std_key in result:
            std = result[std_key]
            ax.fill_between(days, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Simulated day")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(title, fontsize=10, wrap=True)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[InfectionTimeSeries] plot saved to {plot_path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["SIR_alpha", "SIS_alpha", "SIR_beta", "SIS_beta",
                                        "SIR_combined", "SIS_combined"],
                    default="SIR_alpha")
    p.add_argument("--out", default="timeseries.csv")
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--TT", type=int, default=1000)
    p.add_argument("--realizations", type=int, default=20,
                    help="independent realizations averaged per scenario")
    p.add_argument("--beta", type=float, required=True, help="fixed transmission rate")
    p.add_argument("--sigma", type=float, required=True, help="fixed mutation rate")
    p.add_argument("--alpha", type=float, default=0.1, help="baseline recovery rate")
    p.add_argument("--X", type=float, default=0.0)
    p.add_argument("--psi-cap", type=float, default=20.0,
                    help="Psi cap for SIR_alpha/SIS_alpha/SIR_beta/SIS_beta models")
    p.add_argument("--psi-alpha-cap", type=float, default=20.0,
                    help="Psi_alpha (recovery-evasion) cap, combined model only")
    p.add_argument("--psi-beta-cap", type=float, default=10.0,
                    help="Psi_beta (transmissibility) cap, combined model only")
    p.add_argument("--num-edges", type=int, default=37500)
    p.add_argument("--avoidance", type=str, default="0.0",
                    help="comma-separated list of targeted-avoidance values to sweep, e.g. 0,0.25,0.5")
    p.add_argument("--distancing", type=str, default="0.0",
                    help="comma-separated list of blanket-distancing values to sweep, e.g. 0,0.3")
    p.add_argument("--symptomatic-fraction", type=float, default=1.0,
                    help="probability a new infection is symptomatic (visible) "
                         "rather than asymptomatic (invisible); default 1.0 "
                         "makes avoidance and distancing interchangeable, "
                         "set below 1.0 (e.g. 0.6) to make them diverge")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-plot", action="store_true", help="skip generating the PNG plot")
    p.add_argument("--plot-out", default=None,
                    help="path for the PNG plot (default: same name as --out, with .png)")
    p.add_argument("--plot-metric", choices=METRICS, default="cumulative_infections",
                    help="which tracked quantity to plot (default: cumulative_infections)")
    p.add_argument("--plot-title", default=None, help="custom plot title")
    args = p.parse_args()

    avoidance_values = parse_float_list(args.avoidance)
    distancing_values = parse_float_list(args.distancing)
    scenarios = [(a, d) for a in avoidance_values for d in distancing_values]

    t_start = _time.time()
    results_by_label = {}
    with open(args.out, "w") as fp:
        header = ["day", "avoidance", "distancing", "symptomatic_fraction", "scenario"]
        for m in METRICS:
            header += [f"{m}_mean", f"{m}_std"]
        fp.write(",".join(header) + "\n")
        for idx, (av, di) in enumerate(scenarios):
            label = f"avoidance={av:g}_distancing={di:g}"
            result = run_scenario(
                args.model, args.N, args.TT, args.beta, args.sigma, args.alpha,
                args.X, args.psi_cap, args.psi_alpha_cap, args.psi_beta_cap,
                args.num_edges, args.realizations, av, di,
                args.symptomatic_fraction, args.seed)
            results_by_label[label] = result
            for day in range(args.TT):
                row = [str(day), f"{av:g}", f"{di:g}", f"{args.symptomatic_fraction:g}", label]
                for m in METRICS:
                    row.append(f"{result[f'{m}_mean'][day]:.4f}")
                    row.append(f"{result[f'{m}_std'][day]:.4f}")
                fp.write(",".join(row) + "\n")
            fp.flush()
            if not args.quiet:
                elapsed = _time.time() - t_start
                final_cum = result["cumulative_infections_mean"][-1]
                final_asym = result["cumulative_asymptomatic_infections_mean"][-1]
                print(f"[InfectionTimeSeries] {idx + 1}/{len(scenarios)} {label}: "
                      f"final cumulative infections = {final_cum:.0f}/{args.N} "
                      f"(asymptomatic share: {final_asym:.0f}) "
                      f"elapsed={elapsed:.1f}s", file=sys.stderr)

    if not args.no_plot:
        plot_path = args.plot_out
        if plot_path is None:
            base, _ext = os.path.splitext(args.out)
            plot_path = base + ".png"
        title = args.plot_title
        if title is None:
            title = (f"{args.plot_metric.replace('_', ' ')} -- {args.model} "
                      f"(beta={args.beta:g}, sigma={args.sigma:g}, "
                      f"symptomatic_fraction={args.symptomatic_fraction:g}, "
                      f"N={args.N}, mean of {args.realizations} runs)")
        days = list(range(args.TT))
        make_plot(days, results_by_label, args.plot_metric, plot_path, title)


if __name__ == "__main__":
    main()
