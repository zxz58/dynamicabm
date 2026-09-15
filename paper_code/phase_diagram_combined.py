"""
phase_diagram_combined.py
==========================
A third mutation model, alongside the original two:

  * "alpha-mutation" (phase_diagram_alpha.py): pathogen fitness Psi affects
    only the HOST's recovery rate (alpha/Psi). Transmission rate beta is
    fixed and doesn't depend on fitness at all.
  * "beta-mutation" (phase_diagram_beta.py): pathogen fitness Psi affects
    only TRANSMISSIBILITY (beta*Psi per contact). Recovery rate alpha is
    fixed and doesn't depend on fitness at all.
  * "combined-mutation" (this file): the pathogen carries TWO independent
    fitness traits that mutate simultaneously:
        Psi_alpha -- recovery evasion, exactly as in the alpha model:
                     recovery_prob = alpha / Psi_alpha[i]
        Psi_beta  -- transmissibility, exactly as in the beta model:
                     each infected neighbor j transmits with a per-step
                     factor derived from beta * Psi_beta[j]
    Both traits undergo the same "12-uniforms-minus-6" random walk each
    step, using the SAME swept `sigma` value (one underlying mutation
    process expressed on two phenotypic axes), but each is clipped to its
    own cap (`--psi-alpha-cap`, default 20, matching the alpha model's
    default; `--psi-beta-cap`, default 10, matching the beta model's
    default) since the two traits have different natural scales.

Inheritance: when a susceptible node is newly infected, BOTH traits are
copied from the SAME donor neighbor (a mutation event carries the whole
phenotype, not a mix of traits from different sources). The donor is
chosen weighted by Psi_beta among all currently-infected neighbors --
i.e. the same logic the beta model uses to decide who "wins" the
transmission event, since Psi_beta is what actually governs transmission
here. (Psi_alpha does not influence who transmits to whom; it only affects
how long the recipient stays infected once infected.)

Everything else -- network model, SIR/SIS dynamics, outbreak criterion,
avoidance/distancing/symptomatic_fraction behavioral toggles, time-series
tracking, synchronous per-step updates -- matches the alpha/beta engines;
see their docstrings for the general implementation notes shared across
this package.
"""
from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np

from common import generate_er_network, fitness_random_walk, pick_random_neighbor_index
from phase_diagram_alpha import sigma_grid, beta_grid


def run_one_realization(N: int, TT: int, beta: float, alpha: float, sigma: float,
                         X: float, psi_alpha_cap: float, psi_beta_cap: float,
                         dynamics: str, burn_in: int, num_edges: int,
                         rng: np.random.Generator, avoidance: float = 0.0,
                         distancing: float = 0.0, symptomatic_fraction: float = 1.0,
                         track_timeseries: bool = False):
    """Simulate one epidemic realization under the combined-mutation model.

    Returns (final_infected_frac, final_recovered_frac), or a third dict
    element of per-timestep arrays if `track_timeseries=True` -- same
    return shape and same tracked metrics as the alpha/beta engines
    (cumulative_infections, cumulative_symptomatic/asymptomatic_infections,
    current_infected, current_symptomatic/asymptomatic_infected,
    susceptible_remaining), so it's a drop-in fourth option for
    InfectionTimeSeries.py.
    """
    beta_effective_sym = beta * (1.0 - avoidance) * (1.0 - distancing)
    beta_effective_asym = beta * (1.0 - distancing)
    A, adj_list = generate_er_network(N, num_edges, rng)

    state = -np.ones(N, dtype=np.float64)
    psi_alpha = np.ones(N, dtype=np.float64)
    psi_beta = np.ones(N, dtype=np.float64)
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

        # Transmission is governed by Psi_beta of the (potentially)
        # infecting neighbor, exactly as in the beta-mutation model, just
        # with separate effective rates for symptomatic vs asymptomatic
        # sources (see phase_diagram_alpha.run_one_realization's docstring
        # for the avoidance/distancing rationale).
        factor_sym = np.clip(1.0 - beta_effective_sym * psi_beta, 1e-12, None)
        factor_asym = np.clip(1.0 - beta_effective_asym * psi_beta, 1e-12, None)
        vec = np.where(symptomatic_infected_mask, np.log(factor_sym),
                        np.where(asymptomatic_infected_mask, np.log(factor_asym), 0.0))
        logprod = A.dot(vec)
        escape_prob = np.exp(logprod)

        susceptible_idx = np.where(state == -1)[0]
        if susceptible_idx.size:
            p_inf = 1.0 - escape_prob[susceptible_idx]
            draws = rng.random(susceptible_idx.shape[0])
            newly_infected = susceptible_idx[draws < p_inf]
        else:
            newly_infected = np.empty(0, dtype=np.int64)

        if newly_infected.size:
            # Pick ONE donor per newly infected node (weighted by the
            # donor's Psi_beta, matching who actually could have
            # transmitted), then inherit BOTH traits from that same donor.
            donor_idx = pick_random_neighbor_index(
                newly_infected, adj_list, infected_mask, psi_beta, rng, weighted=True)
            psi_alpha[newly_infected] = psi_alpha[donor_idx]
            psi_beta[newly_infected] = psi_beta[donor_idx]
            symptomatic[newly_infected] = rng.random(newly_infected.size) < symptomatic_fraction
            state[newly_infected] = 1.0

        # Recovery: driven by the node's own Psi_alpha, same formula as the
        # alpha-mutation model. Only nodes infected at the *start* of this
        # step are eligible (matches the original if/else-if exclusivity).
        infected_idx = np.where(infected_mask)[0]
        if infected_idx.size:
            draws2 = rng.random(infected_idx.shape[0])
            with np.errstate(divide="ignore"):
                recovery_prob = alpha / psi_alpha[infected_idx]
            recovering = infected_idx[draws2 < recovery_prob]
            state[recovering] = recover_state

        # Both traits random-walk for everyone infected *after* this step's
        # transitions, using independent draws but the same sigma.
        infected_idx2 = np.where(state > 0)[0]
        if infected_idx2.size:
            psi_alpha[infected_idx2] = fitness_random_walk(
                psi_alpha[infected_idx2], sigma, X, 0.0, psi_alpha_cap, rng)
            psi_beta[infected_idx2] = fitness_random_walk(
                psi_beta[infected_idx2], sigma, X, 0.0, psi_beta_cap, rng)

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
                       n_realizations: int, alpha: float, X: float,
                       psi_alpha_cap: float, psi_beta_cap: float,
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
                        N, TT, beta, alpha, sigma, X, psi_alpha_cap, psi_beta_cap,
                        dynamics, burn_in, num_edges, rng, avoidance=avoidance,
                        distancing=distancing, symptomatic_fraction=symptomatic_fraction)
                    outbreak_metric = frac_R if dynamics == "SIR" else frac_I
                    if outbreak_metric >= 0.2:
                        count2 += 1
                fp.write(f"{beta:.6f}  {sigma:.6f}  {count2 / n_realizations:.6f}\n")
                fp.flush()
                done += 1
                if verbose:
                    elapsed = _time.time() - t_start
                    print(f"[{dynamics}_combined] {done}/{total} "
                          f"(beta={beta:.4f}, sigma={sigma:.4f}) "
                          f"outbreak_frac={count2 / n_realizations:.2f} "
                          f"elapsed={elapsed:.1f}s", file=sys.stderr)


def build_arg_parser(default_psi_alpha_cap: float, default_psi_beta_cap: float,
                      prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--out", default="da.txt", help="output file (default: da.txt)")
    p.add_argument("--N", type=int, default=5000, help="network size")
    p.add_argument("--TT", type=int, default=3000, help="max simulated time steps")
    p.add_argument("--realizations", type=int, default=50,
                    help="Monte Carlo realizations per (beta, sigma) pair")
    p.add_argument("--alpha", type=float, default=0.1, help="baseline recovery rate")
    p.add_argument("--X", type=float, default=0.0, help="mean drift of fitness random walk")
    p.add_argument("--psi-alpha-cap", type=float, default=default_psi_alpha_cap,
                    help="upper clip for the recovery-evasion trait Psi_alpha")
    p.add_argument("--psi-beta-cap", type=float, default=default_psi_beta_cap,
                    help="upper clip for the transmissibility trait Psi_beta")
    p.add_argument("--num-edges", type=int, default=37500, help="edges in the ER network")
    p.add_argument("--burn-in", type=int, default=2500,
                    help="steps to discard before averaging (kept for parity; "
                         "the outbreak metric itself only uses the final state)")
    p.add_argument("--sigma-min", type=float, default=0.001)
    p.add_argument("--sigma-max", type=float, default=10.0)
    p.add_argument("--sigma-growth", type=float, default=10 ** 0.04)
    p.add_argument("--sigma-values", type=int, default=None,
                    help="if set, override the sigma grid with this many "
                         "log-spaced points between sigma-min and sigma-max")
    p.add_argument("--beta-min", type=float, default=0.0)
    p.add_argument("--beta-max", type=float, default=0.01)
    p.add_argument("--beta-step", type=float, default=0.0002)
    p.add_argument("--beta-values", type=int, default=None,
                    help="if set, override the beta grid with this many "
                         "linearly-spaced points")
    p.add_argument("--avoidance", type=float, default=0.0,
                    help="targeted avoidance: 0-1 strength with which well "
                         "individuals discount transmission specifically "
                         "from currently-infected symptomatic neighbors")
    p.add_argument("--distancing", type=float, default=0.0,
                    help="blanket distancing: 0-1 strength applied to ALL "
                         "edges regardless of neighbor infection status "
                         "('everyone stays home more')")
    p.add_argument("--symptomatic-fraction", type=float, default=1.0,
                    help="probability a new infection is symptomatic (visible) "
                         "rather than asymptomatic (invisible); default 1.0 "
                         "makes avoidance and distancing interchangeable, "
                         "set below 1.0 (e.g. 0.6) to make them diverge")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
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


def main(dynamics: str, default_psi_alpha_cap: float, default_psi_beta_cap: float, prog: str):
    parser = build_arg_parser(default_psi_alpha_cap, default_psi_beta_cap, prog)
    args = parser.parse_args()
    sigmas, betas = resolve_grids(args)
    run_phase_diagram(
        dynamics=dynamics, out_path=args.out, N=args.N, TT=args.TT,
        n_realizations=args.realizations, alpha=args.alpha, X=args.X,
        psi_alpha_cap=args.psi_alpha_cap, psi_beta_cap=args.psi_beta_cap,
        sigmas=sigmas, betas=betas, num_edges=args.num_edges, burn_in=args.burn_in,
        seed=args.seed, avoidance=args.avoidance, distancing=args.distancing,
        symptomatic_fraction=args.symptomatic_fraction, verbose=not args.quiet)


if __name__ == "__main__":
    main("SIR", default_psi_alpha_cap=20.0, default_psi_beta_cap=10.0, prog="phase_diagram_combined")
