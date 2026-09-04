"""
phase_diagram_beta.py
======================
Python translation of PhaseDiagramSIR_beta.cpp and PhaseDiagramSIS_beta.cpp.

Model ("beta-mutation"): pathogen fitness Psi affects TRANSMISSIBILITY
rather than recovery. Each infected neighbour j of a susceptible node i
independently tries to infect i with probability beta*Psi[j] per step, so
the probability i escapes infection this step is prod_j (1 - beta*Psi[j])
over i's infected neighbours j -- exactly the `linum[i]` product computed
in the original C++. If i becomes infected, the infecting neighbour is
chosen with probability proportional to Psi (fitter variants are more
likely to be the one that "wins" the transmission), and i inherits that
neighbour's Psi. Recovery is a plain rate `alpha` (no fitness dependence)
in this variant. Fitness then random-walks exactly as in the alpha model.

Vectorisation trick: instead of forming the per-node product directly, we
sum log(1 - beta*Psi[j]) over infected neighbours via a single sparse
matrix-vector product (A @ vec, where vec[j] = log(1-beta*Psi[j]) if j is
infected else 0), then exponentiate. This is mathematically identical to
the nested loop in the C++ but O(edges) instead of O(N^2) per step.

Everything else (state encoding, SIR vs SIS behaviour, outbreak criterion,
CLI options) mirrors phase_diagram_alpha.py -- see its docstring and
README.md for the full list of implementation notes vs. the C++ original.
"""
from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np

from common import generate_er_network, fitness_random_walk, pick_random_neighbor
from phase_diagram_alpha import sigma_grid, beta_grid, resolve_grids


def run_one_realization(N: int, TT: int, beta: float, alpha: float, sigma: float,
                         X: float, psi_cap: float, dynamics: str, burn_in: int,
                         num_edges: int, rng: np.random.Generator,
                         avoidance: float = 0.0, distancing: float = 0.0,
                         symptomatic_fraction: float = 1.0,
                         track_timeseries: bool = False):
    """See phase_diagram_alpha.run_one_realization's docstring for the full
    explanation of `avoidance`, `distancing`, and `symptomatic_fraction`.
    Here, symptomatic and asymptomatic infected neighbors get separate
    effective per-neighbor transmission factors:
      beta_effective(symptomatic)  = beta * (1-avoidance) * (1-distancing)
      beta_effective(asymptomatic) = beta * (1-distancing)
    each multiplied against that neighbor's fitness Psi[j] as in the base
    beta-mutation model.
    """
    beta_effective_sym = beta * (1.0 - avoidance) * (1.0 - distancing)
    beta_effective_asym = beta * (1.0 - distancing)
    A, adj_list = generate_er_network(N, num_edges, rng)

    state = -np.ones(N, dtype=np.float64)
    psi = np.ones(N, dtype=np.float64)
    symptomatic = np.zeros(N, dtype=bool)
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
        total_infection_events = float(n0)
        total_symptomatic_events = float(np.count_nonzero(symptomatic[:n0]))
        total_asymptomatic_events = total_infection_events - total_symptomatic_events

    for t in range(TT):
        infected_mask = state > 0
        symptomatic_infected_mask = infected_mask & symptomatic
        asymptomatic_infected_mask = infected_mask & ~symptomatic

        # vec[j] = log(1 - beta_effective_{sym/asym}*Psi[j]) for infected j
        # of the corresponding kind, else 0.
        factor_sym = np.clip(1.0 - beta_effective_sym * psi, 1e-12, None)
        factor_asym = np.clip(1.0 - beta_effective_asym * psi, 1e-12, None)
        vec = np.where(symptomatic_infected_mask, np.log(factor_sym),
                        np.where(asymptomatic_infected_mask, np.log(factor_asym), 0.0))
        logprod = A.dot(vec)                 # sum over infected neighbours
        escape_prob = np.exp(logprod)

        susceptible_idx = np.where(state == -1)[0]
        if susceptible_idx.size:
            p_inf = 1.0 - escape_prob[susceptible_idx]
            draws = rng.random(susceptible_idx.shape[0])
            newly_infected = susceptible_idx[draws < p_inf]
        else:
            newly_infected = np.empty(0, dtype=np.int64)

        if newly_infected.size:
            psi[newly_infected] = pick_random_neighbor(
                newly_infected, adj_list, infected_mask, psi, rng, weighted=True)
            symptomatic[newly_infected] = rng.random(newly_infected.size) < symptomatic_fraction
            state[newly_infected] = 1.0

        infected_idx = np.where(infected_mask)[0]
        if infected_idx.size:
            draws2 = rng.random(infected_idx.shape[0])
            recovering = infected_idx[draws2 < alpha]
            state[recovering] = recover_state

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
                    print(f"[{dynamics}_beta] {done}/{total} "
                          f"(beta={beta:.4f}, sigma={sigma:.4f}) "
                          f"outbreak_frac={count2 / n_realizations:.2f} "
                          f"elapsed={elapsed:.1f}s", file=sys.stderr)


def build_arg_parser(default_psi_cap: float, prog: str) -> argparse.ArgumentParser:
    from phase_diagram_alpha import build_arg_parser as _base
    return _base(default_psi_cap, prog)


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
    main("SIR", default_psi_cap=10.0, prog="phase_diagram_beta")
