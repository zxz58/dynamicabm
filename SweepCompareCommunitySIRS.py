"""Run one-at-a-time CommunitySIRS parameter sweeps and comparison plots."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SWEEP_PARAMETERS = {
    "phi_max": "--phi-max",
    "beta_min": "--beta-min",
    "beta_max": "--beta-max",
    "alpha_min": "--alpha-min",
    "alpha_max": "--alpha-max",
    "gamma_min": "--gamma-min",
    "gamma_max": "--gamma-max",
    "rho": "--rho",
    "eta": "--inter-edge-decay-rate",
}


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of nonnegative float values."""
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("value list must contain at least one number")
    if any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("sweep values must be nonnegative")
    return values


def value_label(value: float) -> str:
    """Create a filesystem- and legend-safe representation of a numeric value."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def scenario_label(param: str, value: float) -> str:
    """Create the scenario label used in filenames and comparison legends."""
    return f"{param}_{value_label(value)}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the generic one-at-a-time sweep CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-param", choices=sorted(SWEEP_PARAMETERS), default="alpha_max",
                   help="parameter to sweep one-at-a-time")
    p.add_argument("--sweep-values", type=parse_float_list, default=parse_float_list("0.25,0.3,0.4,0.6,0.8"),
                   help="comma-separated values for --sweep-param")
    p.add_argument("--phi-values", type=parse_float_list, default=None,
                   help="backward-compatible alias for --sweep-param phi_max --sweep-values")
    p.add_argument("--out-dir", default=None,
                   help="directory for per-scenario CSVs and the comparison figure")
    p.add_argument("--comparison-out", default=None,
                   help="optional comparison PNG path")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable used for subprocess calls")

    p.add_argument("--N", type=int, default=100)
    p.add_argument("--K", type=int, choices=[2, 3], default=3)
    p.add_argument("--t-max", type=float, default=5000.0)
    p.add_argument("--burn-in-time", type=float, default=1000.0)
    p.add_argument("--sample-interval", type=float, default=10.0)
    p.add_argument("--realizations", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--phi-max", type=float, default=0.1)
    p.add_argument("--beta-min", type=float, default=0.001)
    p.add_argument("--beta-max", type=float, default=0.5)
    p.add_argument("--alpha-min", type=float, default=0.0)
    p.add_argument("--alpha-max", type=float, default=0.01)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--gamma-max", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--inter-edge-decay-rate", type=float, default=0.1)
    return p


def fixed_parameter_args(args, sweep_param: str, sweep_value: float) -> list[str]:
    """Build fixed/swept parameter flags for one simulation command."""
    values = {
        "phi_max": args.phi_max,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "gamma_min": args.gamma_min,
        "gamma_max": args.gamma_max,
        "rho": args.rho,
        "eta": args.inter_edge_decay_rate,
    }
    values[sweep_param] = sweep_value

    out = []
    for param, flag in SWEEP_PARAMETERS.items():
        out.extend([flag, str(values[param])])
    return out


def run_command(cmd: list[str], quiet: bool):
    """Run one subprocess and fail fast if it exits nonzero."""
    if not quiet:
        print("[SweepCompareCommunitySIRS]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def main():
    """Run all scenarios for one parameter, then generate the comparison plot."""
    args = build_arg_parser().parse_args()
    sweep_param = args.sweep_param
    sweep_values = args.sweep_values
    if args.phi_values is not None:
        sweep_param = "phi_max"
        sweep_values = args.phi_values

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "sweeps" / sweep_param
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_out = (
        Path(args.comparison_out)
        if args.comparison_out
        else out_dir / f"community_sirs_{sweep_param}_comparison.png"
    )

    repo_dir = Path(__file__).resolve().parent
    sample_args = []
    for value in sweep_values:
        label = scenario_label(sweep_param, value)
        summary_path = out_dir / f"summary_{label}.csv"
        samples_path = out_dir / f"samples_{label}.csv"
        sim_cmd = [
            args.python, str(repo_dir / "CommunitySIRS.py"),
            "--N", str(args.N),
            "--K", str(args.K),
            "--t-max", str(args.t_max),
            "--burn-in-time", str(args.burn_in_time),
            "--sample-interval", str(args.sample_interval),
            "--realizations", str(args.realizations),
            *fixed_parameter_args(args, sweep_param, value),
            "--out", str(summary_path),
            "--samples-out", str(samples_path),
        ]
        if args.seed is not None:
            sim_cmd.extend(["--seed", str(args.seed)])
        if args.quiet:
            sim_cmd.append("--quiet")
        run_command(sim_cmd, args.quiet)
        sample_args.extend(["--samples", f"{label}:{samples_path}"])

    compare_cmd = [
        args.python, str(repo_dir / "CompareCommunitySIRS.py"),
        *sample_args,
        "--out", str(comparison_out),
        "--burn-in-time", str(args.burn_in_time),
    ]
    run_command(compare_cmd, args.quiet)


if __name__ == "__main__":
    main()
