"""
TimingMitigation.py
====================
Python translation of TimingMitigation.cpp.

Same SIR / alpha-mutation model as phase_diagram_alpha.py (fitness affects
recovery rate, uniform inheritance from an infected neighbour), but here
sigma is fixed at 0.03 and beta itself changes mid-simulation: it starts at
0.008 and drops to 0.004 ("mitigation") starting at time step `tR`, for a
sweep of mitigation start times tR = 0, 2, 4, ..., 120.

For each tR, over `n_realizations` runs we track:
  * count2 -- fraction of runs with a major outbreak (R/N >= 0.2)
  * count3 -- fraction of runs in which the population's average pathogen
    fitness among infected individuals ever exceeds a threshold (1.67),
    which the original authors use as a proxy for a fitness-driven second
    wave breaking through the mitigation.

Output columns match the original: tR, beta (final value, always 0.004),
sigma, outbreak_fraction, second_wave_fraction.
"""
from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np

from common import generate_er_network, fitness_random_walk, pick_random_neighbor


def run_one_realization(N: int, TT: int, beta_before: float, beta_after: float,
                         tR: int, alpha: float, sigma: float, X: float,
                         psi_cap: float, fitness_threshold: float, num_edges: int,
                         rng: np.random.Generator) -> tuple[float, float, bool]:
    A, adj_list = generate_er_network(N, num_edges, rng)

    state = -np.ones(N, dtype=np.float64)
    psi = np.ones(N, dtype=np.float64)
    n0 = N // 50
    state[:n0] = 1.0

    judge = False  # becomes True once mean fitness among infected exceeds threshold

    for t in range(TT):
        beta = beta_after if t >= tR else beta_before

        infected_mask = state > 0
        linum = A.dot(infected_mask.astype(np.float64))

        susceptible_idx = np.where(state == -1)[0]
        if susceptible_idx.size:
            p_inf = 1.0 - (1.0 - beta) ** linum[susceptible_idx]
            draws = rng.random(susceptible_idx.shape[0])
            newly_infected = susceptible_idx[draws < p_inf]
        else:
            newly_infected = np.empty(0, dtype=np.int64)

        if newly_infected.size:
            psi[newly_infected] = pick_random_neighbor(
                newly_infected, adj_list, infected_mask, psi, rng, weighted=False)
            state[newly_infected] = 1.0

        infected_idx = np.where(infected_mask)[0]
        if infected_idx.size:
            draws2 = rng.random(infected_idx.shape[0])
            with np.errstate(divide="ignore"):
                recovery_prob = alpha / psi[infected_idx]
            recovering = infected_idx[draws2 < recovery_prob]
            state[recovering] = -2.0

        infected_idx2 = np.where(state > 0)[0]
        if infected_idx2.size:
            psi[infected_idx2] = fitness_random_walk(
                psi[infected_idx2], sigma, X, 0.0, psi_cap, rng)
            mean_fitness = psi[infected_idx2].mean()
            if mean_fitness > fitness_threshold:
                judge = True

    I = float(np.sum(state > 0))
    RR = float(np.sum(state < -1.5))
    return I / N, RR / N, judge


def run_sweep(out_path: str, N: int, TT: int, n_realizations: int, alpha: float,
              X: float, psi_cap: float, sigma: float, beta_before: float,
              beta_after: float, fitness_threshold: float,
              tR_values: np.ndarray, num_edges: int, seed: int | None,
              verbose: bool = True):
    rng = np.random.default_rng(seed)
    t_start = _time.time()
    with open(out_path, "w") as fp:
        for idx, tR in enumerate(tR_values):
            count2 = 0
            count3 = 0
            for _ in range(n_realizations):
                frac_I, frac_R, judge = run_one_realization(
                    N, TT, beta_before, beta_after, int(tR), alpha, sigma, X,
                    psi_cap, fitness_threshold, num_edges, rng)
                if frac_R >= 0.2:
                    count2 += 1
                if judge:
                    count3 += 1
            fp.write(f"{int(tR)}  {beta_after:.6f}  {sigma:.6f}  "
                      f"{count2 / n_realizations:.6f}  {count3 / n_realizations:.6f}\n")
            fp.flush()
            if verbose:
                elapsed = _time.time() - t_start
                print(f"[TimingMitigation] {idx + 1}/{len(tR_values)} "
                      f"tR={int(tR)} outbreak_frac={count2 / n_realizations:.2f} "
                      f"second_wave_frac={count3 / n_realizations:.2f} "
                      f"elapsed={elapsed:.1f}s", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="da.txt")
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--TT", type=int, default=3000)
    p.add_argument("--realizations", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--X", type=float, default=0.0)
    p.add_argument("--psi-cap", type=float, default=20.0)
    p.add_argument("--sigma", type=float, default=0.03)
    p.add_argument("--beta-before", type=float, default=0.008)
    p.add_argument("--beta-after", type=float, default=0.004)
    p.add_argument("--fitness-threshold", type=float, default=1.67)
    p.add_argument("--num-edges", type=int, default=37500)
    p.add_argument("--tR-max", type=int, default=120)
    p.add_argument("--tR-step", type=int, default=2)
    p.add_argument("--tR-values", type=int, default=None,
                    help="if set, override with this many evenly spaced tR "
                         "points between 0 and tR-max (quick test runs)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.tR_values:
        tRs = np.linspace(0, args.tR_max, args.tR_values).round().astype(int)
    else:
        tRs = np.arange(0, args.tR_max + 1, args.tR_step)

    run_sweep(
        out_path=args.out, N=args.N, TT=args.TT, n_realizations=args.realizations,
        alpha=args.alpha, X=args.X, psi_cap=args.psi_cap, sigma=args.sigma,
        beta_before=args.beta_before, beta_after=args.beta_after,
        fitness_threshold=args.fitness_threshold, tR_values=tRs,
        num_edges=args.num_edges, seed=args.seed, verbose=not args.quiet)


if __name__ == "__main__":
    main()
