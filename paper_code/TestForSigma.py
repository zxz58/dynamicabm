"""
TestForSigma.py
================
Python translation of TestForSigma.cpp.

This one has no contact network. It studies how a *within-host* mutation
process (intra-host fitness F[:,0], controlled by sigmaPsi) interacts with
selection driven by a separate, independently-mutating *between-host*
fitness axis F[:,1] (controlled by sigmaPhi, held fixed).

Setup per realization:
  * N "strains" are laid out on an integer axis 0..N-1. Each strain s has
    two randomly drawn fitness values, F[s,0] (intra-host) and F[s,1]
    (inter-host), each ~ 1 + (Irwin-Hall approx. normal) * sigma, clipped
    to [0, 2].
  * M individuals each carry a strain index (all start at the middle
    strain N/2).
  * Each of `rho` iterations:
      - record the mean/std of F[strain,0] and F[strain,1] across the
        population (only the values at the *final* iteration are kept,
        averaged over K realizations, matching the original code -- it
        only accumulates aveK*/stdK* "if (t==rho-1)").
      - each individual's strain index does a +1/-1/stay random walk along
        the strain axis with probability p each direction (mutation
        between strains), clipped to [0, N-1].
      - each individual with strain s "dies" with probability 0.1/F[s,0]
        (lower intra-host fitness -> higher death/turnover probability)
        and, if so, is replaced by copying the strain of a uniformly
        random *surviving* individual (a Moran-process-style death/birth
        step) -- this is what lets higher-inter-host-fitness strains
        spread through the population even though the mutation walk
        itself is undirected.

This is swept over `sigmaPsi` (0, 0.02, ..., 0.58), with K Monte Carlo
realizations per value.

Implementation note vs. the C++ original: the death/replacement step in
the C code is a sequential loop that samples a random individual index and
rejects it (retrying) if that individual has *already* been marked dead
earlier in the same pass -- i.e. it resamples from the "not yet processed
or already replaced" pool at that specific point in the sequential sweep.
Here we instead determine all deaths from the strain assignment at the
*start* of the step, then have every dying individual copy the strain of a
uniformly random *survivor* (an individual not dying this step). This is
the standard, order-independent way to implement a Moran-style death/birth
update and preserves the intended selection mechanism (still true that
lower intra-host fitness strains are replaced more often); it just removes
the C loop's incidental, sequential-order dependence, which is not part of
the model itself.
"""
from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np


def irwin_hall_shift(rng: np.random.Generator, shape) -> np.ndarray:
    """sum of 12 U(0,1) minus 6 -- Irwin-Hall approx. of a standard normal."""
    return rng.random(shape + (12,)).sum(axis=-1) - 6.0


def run_realization(N: int, M: int, rho: int, p: float, sigmaPsi: float,
                     sigmaPhi: float, rng: np.random.Generator):
    # Draw per-strain fitnesses.
    F0 = 1.0 + irwin_hall_shift(rng, (N,)) * sigmaPsi
    np.clip(F0, 0.0, 2.0, out=F0)
    F1 = 1.0 + irwin_hall_shift(rng, (N,)) * sigmaPhi
    np.clip(F1, 0.0, 2.0, out=F1)

    strain = np.full(M, N // 2, dtype=np.int64)

    ave1 = ave2 = std1 = std2 = 0.0

    for t in range(rho):
        f0_vals = F0[strain]
        f1_vals = F1[strain]

        if t == rho - 1:
            ave1 = f0_vals.mean()
            ave2 = f1_vals.mean()
            std1 = f0_vals.std()
            std2 = f1_vals.std()

        # Mutation: +1 w.p. p, -1 w.p. p, else unchanged; clipped to [0, N-1].
        draws = rng.random(M)
        step = np.zeros(M, dtype=np.int64)
        step[draws <= p] = 1
        step[(draws > p) & (draws <= 2 * p)] = -1
        strain = np.clip(strain + step, 0, N - 1)

        # Death / replacement (Moran-style birth-death) driven by intra-host
        # fitness F0 of the strain each individual currently carries.
        with np.errstate(divide="ignore"):
            death_prob = 0.1 / F0[strain]  # F0==0 -> certain death, same as original
        dies = rng.random(M) < death_prob
        n_dead = int(dies.sum())
        if n_dead:
            survivors = np.where(~dies)[0]
            if survivors.size:
                parents = rng.choice(survivors, size=n_dead, replace=True)
                strain[dies] = strain[parents]
            # (if nobody survives this step -- astronomically unlikely for
            # any reasonable p / death rate -- leave strains unchanged)

    return ave1, ave2, std1, std2


def run_sweep(out_path: str, N: int, M: int, K: int, rho: int, p: float,
              sigmaPhi: float, sigmaPsi_values: np.ndarray, seed: int | None,
              verbose: bool = True):
    rng = np.random.default_rng(seed)
    t_start = _time.time()
    with open(out_path, "w") as fp:
        for idx, sigmaPsi in enumerate(sigmaPsi_values):
            aveK1 = aveK2 = stdK1 = stdK2 = 0.0
            for _ in range(K):
                a1, a2, s1, s2 = run_realization(N, M, rho, p, sigmaPsi, sigmaPhi, rng)
                aveK1 += a1 / K
                aveK2 += a2 / K
                stdK1 += s1 / K
                stdK2 += s2 / K
            fp.write(f"{rho}\t{p}\t{sigmaPsi:.6f}\t{sigmaPhi}\t"
                      f"{aveK1:.6f}  {aveK2:.6f}\t{stdK1:.6f}  {stdK2:.6f}\n")
            fp.flush()
            if verbose:
                elapsed = _time.time() - t_start
                print(f"[TestForSigma] {idx + 1}/{len(sigmaPsi_values)} "
                      f"sigmaPsi={sigmaPsi:.3f} aveK1={aveK1:.3f} aveK2={aveK2:.3f} "
                      f"elapsed={elapsed:.1f}s", file=sys.stderr)


def main():
    p_arg = argparse.ArgumentParser(description=__doc__)
    p_arg.add_argument("--out", default="da.txt")
    p_arg.add_argument("--N", type=int, default=200, help="number of strains")
    p_arg.add_argument("--M", type=int, default=2000, help="number of individuals")
    p_arg.add_argument("--K", type=int, default=5000, help="Monte Carlo realizations")
    p_arg.add_argument("--rho", type=int, default=1000, help="iterations per realization")
    p_arg.add_argument("--p", type=float, default=0.02, help="strain mutation rate")
    p_arg.add_argument("--sigmaPhi", type=float, default=0.2,
                        help="inter-host fitness noise (held fixed in the sweep)")
    p_arg.add_argument("--sigmaPsi-max", type=float, default=0.6)
    p_arg.add_argument("--sigmaPsi-step", type=float, default=0.02)
    p_arg.add_argument("--sigmaPsi-values", type=int, default=None,
                        help="if set, override with this many evenly spaced "
                             "points between 0 and sigmaPsi-max (quick test runs)")
    p_arg.add_argument("--seed", type=int, default=None)
    p_arg.add_argument("--quiet", action="store_true")
    args = p_arg.parse_args()

    if args.sigmaPsi_values:
        sigmaPsis = np.linspace(0.0, args.sigmaPsi_max, args.sigmaPsi_values)
    else:
        n = int(round(args.sigmaPsi_max / args.sigmaPsi_step))
        sigmaPsis = args.sigmaPsi_step * np.arange(n)

    run_sweep(
        out_path=args.out, N=args.N, M=args.M, K=args.K, rho=args.rho, p=args.p,
        sigmaPhi=args.sigmaPhi, sigmaPsi_values=sigmaPsis, seed=args.seed,
        verbose=not args.quiet)


if __name__ == "__main__":
    main()
