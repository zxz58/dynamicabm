"""
phase_diagram_alpha.py
=======================
Python translation of PhaseDiagramSIR_alpha.cpp and PhaseDiagramSIS_alpha.cpp.

Model ("alpha-mutation"): pathogen fitness Psi affects the HOST's recovery
rate. An infected node i recovers with probability alpha/Psi[i] each step.
Fitness is inherited uniformly at random from one of the infecting node's
already-infected neighbours, and then undergoes a small Gaussian-like
random walk (see common.fitness_random_walk) every step it stays infected.

state encoding (float array), same as the original:
    -1  susceptible
    +1  infected
    -2  recovered   (SIR only; SIS recycles recovered nodes back to -1)

Two dynamics share this file:
    SIR: recovered nodes go to -2 and never return to susceptible.
         Outbreak criterion: R/N >= 0.2 at the end of the run.
    SIS: recovered nodes go straight back to -1 (susceptible again).
         Outbreak criterion: I/N >= 0.2 at the end of the run.

For each (beta, sigma) pair, `n_realizations` independent runs are
simulated (fresh network + fresh epidemic each time) and the fraction that
produced a major outbreak is written to the output file -- exactly the
quantity plotted in the paper's phase diagrams.

Implementation notes vs. the original C++ (see README.md for the full
rationale):
  * Sparse adjacency + matrix-vector products replace the O(N^2) neighbour
    counting loops.
  * Within one simulated day, all state transitions are computed from the
    state at the *start* of that day (synchronous update) rather than the
    partially-updated, node-order-dependent state the sequential C loop
    sees. This is a standard, mean-field-consistent way to implement this
    exact model and removes an RNG-order artifact of the C code, without
    changing any transition rule.
  * All grids (sigma, beta), N, TT and n_realizations are CLI-configurable
    so you can run a fast, small-scale sanity check before committing to
    the full paper-scale sweep, which is very expensive in *any* language.
"""
from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np

from common import generate_er_network, fitness_random_walk, pick_random_neighbor


def sigma_grid(sigma_min: float = 0.001, sigma_max: float = 10.0,
               growth: float = 10 ** 0.04) -> np.ndarray:
    """Reproduces: for(sigma=0.001; sigma<=10; sigma *= 10**0.04)"""
    vals = []
    s = sigma_min
    while s <= sigma_max:
        vals.append(s)
        s *= growth
    return np.array(vals)


def beta_grid(beta_min: float = 0.0, beta_max: float = 0.01,
              step: float = 0.0002) -> np.ndarray:
    """Reproduces: for(beta=0; beta<=0.01; beta+=0.0002)"""
    n = int(round((beta_max - beta_min) / step)) + 1
    return beta_min + step * np.arange(n)


def run_one_realization(N: int, TT: int, beta: float, alpha: float, sigma: float,
                         X: float, psi_cap: float, dynamics: str, burn_in: int,
                         num_edges: int, rng: np.random.Generator,
                         avoidance: float = 0.0, distancing: float = 0.0,
                         symptomatic_fraction: float = 1.0,
                         track_timeseries: bool = False):
    """Simulate one epidemic realization.

    Returns (final_infected_frac, final_recovered_frac) -- for SIS,
    final_recovered_frac is always 0 since recovered nodes are recycled to
    susceptible immediately. If `track_timeseries=True`, a third element is
    returned: a dict of per-timestep arrays (see below).

    Symptomatic vs. asymptomatic infections
    ----------------------------------------
    Each newly infected node is independently assigned "symptomatic" with
    probability `symptomatic_fraction` (0.0-1.0), drawn once at the moment
    of infection and fixed for the duration of that infection episode.
    This split is what lets the two behavioral toggles below actually
    diverge in effect -- see next section. `symptomatic_fraction=1.0`
    (the default) makes every infection symptomatic, reproducing the
    original behavior exactly (no asymptomatic carriers exist).

    Two independent behavioral toggles, both 0.0-1.0:

      `avoidance`  -- TARGETED precaution: well individuals reduce
                      transmission risk specifically along edges to a
                      currently infected AND SYMPTOMATIC neighbor (you can
                      only avoid someone you can tell is sick). Edges to
                      asymptomatic infected neighbors are NOT discounted by
                      avoidance -- those cases are invisible.
      `distancing` -- BLANKET precaution: a flat reduction applied to every
                      edge to any infected neighbor, symptomatic or not
                      ("everyone stays home more" regardless of whether
                      they can see who's sick).

    beta_effective(symptomatic neighbor)   = beta * (1-avoidance) * (1-distancing)
    beta_effective(asymptomatic neighbor)  = beta * (1-distancing)

    With `symptomatic_fraction < 1.0`, these two parameters are no longer
    mathematically interchangeable: `distancing` suppresses transmission
    from both visible and invisible cases, while `avoidance` can only ever
    suppress the visible (symptomatic) share -- so at a fixed combined
    beta reduction, higher `avoidance` (relative to `distancing`) leaves
    more of the epidemic driven by asymptomatic spread that avoidance
    cannot touch. At `symptomatic_fraction=1.0` the two collapse back to
    being interchangeable, as documented previously.
    """
    beta_effective_sym = beta * (1.0 - avoidance) * (1.0 - distancing)
    beta_effective_asym = beta * (1.0 - distancing)
    A, adj_list = generate_er_network(N, num_edges, rng)

    state = -np.ones(N, dtype=np.float64)   # -1 susceptible
    psi = np.ones(N, dtype=np.float64)
    symptomatic = np.zeros(N, dtype=bool)   # only meaningful while infected
    n0 = N // 50
    state[:n0] = 1.0
    symptomatic[:n0] = rng.random(n0) < symptomatic_fraction

    recover_state = -2.0 if dynamics == "SIR" else -1.0

    if track_timeseries:
        cumulative_infections = np.empty(TT, dtype=np.float64)
        cumulative_symptomatic_infections = np.empty(TT, dtype=np.float64)
        cumulative_asymptomatic_infections = np.empty(TT, dtype=np.float64)
        current_infected = np.empty(TT, dtype=np.float64)
        current_symptomatic_infected = np.empty(TT, dtype=np.float64)
        current_asymptomatic_infected = np.empty(TT, dtype=np.float64)
        susceptible_remaining = np.empty(TT, dtype=np.float64)
        total_infection_events = float(n0)  # count the initial seed infections
        total_symptomatic_events = float(np.count_nonzero(symptomatic[:n0]))
        total_asymptomatic_events = total_infection_events - total_symptomatic_events

    for t in range(TT):
        infected_mask = state > 0
        symptomatic_infected_mask = infected_mask & symptomatic
        asymptomatic_infected_mask = infected_mask & ~symptomatic
        linum_sym = A.dot(symptomatic_infected_mask.astype(np.float64)) # how many symptomatic infected neighbors?
        linum_asym = A.dot(asymptomatic_infected_mask.astype(np.float64)) # how many asymptomatic infected neighbors?

        # Infection Events
        susceptible_idx = np.where(state == -1)[0]
        if susceptible_idx.size:
            escape_prob = ((1.0 - beta_effective_sym) ** linum_sym[susceptible_idx]
                            * (1.0 - beta_effective_asym) ** linum_asym[susceptible_idx])
            p_inf = 1.0 - escape_prob
            draws = rng.random(susceptible_idx.shape[0])
            newly_infected = susceptible_idx[draws < p_inf]
        else:
            newly_infected = np.empty(0, dtype=np.int64)

        if newly_infected.size:
            psi[newly_infected] = pick_random_neighbor(
                newly_infected, adj_list, infected_mask, psi, rng, weighted=False)
            symptomatic[newly_infected] = rng.random(newly_infected.size) < symptomatic_fraction
            state[newly_infected] = 1.0

        # Recovery: only nodes infected at the *start* of this step are
        # eligible (matches the original if/else-if exclusivity per node).
        infected_idx = np.where(infected_mask)[0]
        if infected_idx.size:
            draws2 = rng.random(infected_idx.shape[0])
            with np.errstate(divide="ignore"):
                # Psi==0 -> alpha/Psi == inf -> certain recovery, matching
                # the original C++'s identical (also-undefined-but-consistent)
                # division-by-zero behaviour.
                recovery_prob = alpha / psi[infected_idx]
            recovering = infected_idx[draws2 < recovery_prob]
            state[recovering] = recover_state

        # Fitness random walk applies to everyone infected *after* this
        # step's transitions (matches the original's separate stats loop).
        infected_idx2 = np.where(state > 0)[0]
        if infected_idx2.size:
            psi[infected_idx2] = fitness_random_walk(
                psi[infected_idx2], sigma, X, 0.0, psi_cap, rng)

        if track_timeseries:
            n_new_sym = int(np.count_nonzero(symptomatic[newly_infected])) if newly_infected.size else 0
            n_new_asym = newly_infected.size - n_new_sym
            total_infection_events += newly_infected.size
            total_symptomatic_events += n_new_sym
            total_asymptomatic_events += n_new_asym
            cumulative_infections[t] = total_infection_events
            cumulative_symptomatic_infections[t] = total_symptomatic_events
            cumulative_asymptomatic_infections[t] = total_asymptomatic_events
            current_infected[t] = infected_idx2.size
            current_symptomatic_infected[t] = np.count_nonzero((state > 0) & symptomatic)
            current_asymptomatic_infected[t] = infected_idx2.size - current_symptomatic_infected[t]
            susceptible_remaining[t] = np.count_nonzero(state == -1)

    I = float(np.sum(state > 0))
    RR = float(np.sum(state < -1.5)) if dynamics == "SIR" else 0.0
    if track_timeseries:
        return I / N, RR / N, {
            "cumulative_infections": cumulative_infections,
            "cumulative_symptomatic_infections": cumulative_symptomatic_infections,
            "cumulative_asymptomatic_infections": cumulative_asymptomatic_infections,
            "current_infected": current_infected,
            "current_symptomatic_infected": current_symptomatic_infected,
            "current_asymptomatic_infected": current_asymptomatic_infected,
            "susceptible_remaining": susceptible_remaining,
        }
    return I / N, RR / N


def run_phase_diagram(dynamics: str, out_path: str, N: int, TT: int,
                       n_realizations: int, alpha: float, X: float, psi_cap: float,
                       sigmas: np.ndarray, betas: np.ndarray, num_edges: int,
                       burn_in: int, seed: int | None, verbose: bool = True,
                       avoidance: float = 0.0, distancing: float = 0.0,
                       symptomatic_fraction: float = 1.0):
    assert dynamics in ("SIR", "SIS")
    rng = np.random.default_rng(seed)
    t_start = _time.time()
    total = len(sigmas) * len(betas)
    done = 0
    with open(out_path, "w") as fp:
        for sigma in sigmas:
            for beta in betas:
                count2 = 0
                for _ in range(n_realizations):
                    frac_I, frac_R = run_one_realization(
                        N, TT, beta, alpha, sigma, X, psi_cap, dynamics,
                        burn_in, num_edges, rng, avoidance=avoidance,
                        distancing=distancing, symptomatic_fraction=symptomatic_fraction)
                    outbreak_metric = frac_R if dynamics == "SIR" else frac_I
                    if outbreak_metric >= 0.2:
                        count2 += 1
                fp.write(f"{beta:.6f}  {sigma:.6f}  {count2 / n_realizations:.6f}\n")
                fp.flush()
                done += 1
                if verbose:
                    elapsed = _time.time() - t_start
                    print(f"[{dynamics}_alpha] {done}/{total} "
                          f"(beta={beta:.4f}, sigma={sigma:.4f}) "
                          f"outbreak_frac={count2 / n_realizations:.2f} "
                          f"elapsed={elapsed:.1f}s", file=sys.stderr)


def build_arg_parser(default_psi_cap: float, prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--out", default="da.txt", help="output file (default: da.txt)")
    p.add_argument("--N", type=int, default=5000, help="network size")
    p.add_argument("--TT", type=int, default=3000, help="max simulated time steps")
    p.add_argument("--realizations", type=int, default=50,
                    help="Monte Carlo realizations per (beta, sigma) pair")
    p.add_argument("--alpha", type=float, default=0.1, help="baseline recovery rate")
    p.add_argument("--X", type=float, default=0.0, help="mean drift of fitness random walk")
    p.add_argument("--psi-cap", type=float, default=default_psi_cap,
                    help="upper clip for fitness Psi")
    p.add_argument("--num-edges", type=int, default=37500, help="edges in the ER network")
    p.add_argument("--burn-in", type=int, default=2500,
                    help="steps to discard before averaging (kept for parity; "
                         "the outbreak metric itself only uses the final state)")
    p.add_argument("--sigma-min", type=float, default=0.001)
    p.add_argument("--sigma-max", type=float, default=10.0)
    p.add_argument("--sigma-growth", type=float, default=10 ** 0.04)
    p.add_argument("--sigma-values", type=int, default=None,
                    help="if set, override the sigma grid with this many "
                         "log-spaced points between sigma-min and sigma-max "
                         "(handy for a quick, coarse test run)")
    p.add_argument("--beta-min", type=float, default=0.0)
    p.add_argument("--beta-max", type=float, default=0.01)
    p.add_argument("--beta-step", type=float, default=0.0002)
    p.add_argument("--beta-values", type=int, default=None,
                    help="if set, override the beta grid with this many "
                         "linearly-spaced points (handy for a quick test run)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--avoidance", type=float, default=0.0,
                    help="targeted avoidance: 0-1 strength with which well "
                         "individuals discount transmission specifically "
                         "from currently-infected neighbors")
    p.add_argument("--distancing", type=float, default=0.0,
                    help="blanket distancing: 0-1 strength applied to ALL "
                         "edges regardless of neighbor infection status "
                         "('everyone stays home more')")
    p.add_argument("--symptomatic-fraction", type=float, default=1.0,
                    help="probability that a new infection is symptomatic "
                         "(visible) rather than asymptomatic (invisible). "
                         "avoidance can only discount transmission from "
                         "symptomatic neighbors; distancing discounts both. "
                         "Default 1.0 (all infections symptomatic) "
                         "reproduces the original model, where avoidance "
                         "and distancing are mathematically interchangeable; "
                         "set below 1.0 to make them diverge.")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p


def resolve_grids(args) -> tuple[np.ndarray, np.ndarray]:
    if args.sigma_values:
        sigmas = np.geomspace(args.sigma_min, args.sigma_max, args.sigma_values)
    else:
        sigmas = sigma_grid(args.sigma_min, args.sigma_max, args.sigma_growth)
    if args.beta_values:
        betas = np.linspace(args.beta_min, args.beta_max, args.beta_values)
    else:
        betas = beta_grid(args.beta_min, args.beta_max, args.beta_step)
    return sigmas, betas


def main(dynamics: str, default_psi_cap: float, prog: str):
    parser = build_arg_parser(default_psi_cap, prog)
    args = parser.parse_args()
    sigmas, betas = resolve_grids(args)
    run_phase_diagram(
        dynamics=dynamics, out_path=args.out, N=args.N, TT=args.TT,
        n_realizations=args.realizations, alpha=args.alpha, X=args.X,
        psi_cap=args.psi_cap, sigmas=sigmas, betas=betas,
        num_edges=args.num_edges, burn_in=args.burn_in, seed=args.seed,
        avoidance=args.avoidance, distancing=args.distancing,
        symptomatic_fraction=args.symptomatic_fraction, verbose=not args.quiet)


if __name__ == "__main__":
    main("SIR", default_psi_cap=20.0, prog="phase_diagram_alpha")