"""
ReemergenceImmunityEvasion.py
==============================
Python translation of "ReemergenceImmunity Evasion.cpp" -- the full,
COVID-style compartmental model with immune evasion / reinfection.

Time step = 1/96 of a day (~15 minutes); TT=40000 steps by default (~417
days). States (matching the original's integer codes exactly):

    0 susceptible        5 hospitalized         9  asymptomatic (infectious)
    1 exposed            6 ventilated           10 pre-symptomatic (infectious)
    2 mild symptoms      7 recovered            11 vaccinated (unused; kept
    3 severe symptoms    8 deceased                for parity, never created)
    4 critical symptoms                          12 mild-symptoms-after-reinfection
                                                     (also infectious; see note
                                                     below -- this branch is
                                                     dead code in the original
                                                     C++ too, kept for fidelity)

Contagious/infectious states are {9, 10, 12}. New infections (S->E) happen
at rate beta/96 per contagious neighbour, independent of that neighbour's
immune-evasion ability. REinfection of a recovered/vaccinated node instead
only "sees" the subset of its contagious neighbours whose immune-evasion
score passes a Bernoulli(Psi2) test that step -- this is what lets a more
evasive variant cause a second wave among previously recovered people
(Psi2 = Psi^10 / (pm^10 + Psi^10), a Hill function of raw fitness Psi).

Vectorisation approach: the per-node "count of contagious neighbours" and
"count of *evasion-successful* contagious neighbours" are both computed
with one sparse matrix-vector product / one bincount over the *edge list*
(computed once per realization, since the network doesn't change), instead
of the original's O(N^2) double loop repeated every one of the up to 40000
steps. Neighbour-fitness-donor selection (who a newly exposed or newly
reinfected node "catches" its variant from) is done with a small Python
loop restricted to the handful of nodes that actually transition that
step. See README.md for the general implementation notes shared by all
scripts in this package (synchronous per-step updates, etc.).

The early-exit `if (I<0.5 and E<0.5): stop` from the original C++ is kept
verbatim, including the fact that it does not check the asymptomatic /
pre-symptomatic counts -- this is a property of the original algorithm,
not a bug introduced here.

Because a single realization can take up to 40000 steps and the full grid
is sigma in {0, 0.1, 0.2} x beta in 22 values x 50 realizations, this is by
far the most expensive script in the package (expensive in the original
C++ too). All scales are CLI-configurable so you can sanity-check on a
small network / short horizon before committing to a full run.
"""
from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np

from common import generate_er_network


CONTAGIOUS_STATES = (9, 10, 12)


def hill(psi: np.ndarray, pm: float) -> np.ndarray:
    p10 = np.power(psi, 10)
    return p10 / (pm ** 10 + p10)


def pick_donor(nodes: np.ndarray, adj_list, candidate_mask: np.ndarray,
                psi: np.ndarray, rng: np.random.Generator,
                evasion_prob: np.ndarray | None = None) -> np.ndarray:
    """For each node in `nodes`, pick one qualifying neighbour and return its
    Psi. If `evasion_prob` is given, a candidate neighbour j additionally
    has to pass a fresh Bernoulli(evasion_prob[j]) draw (reinfection case);
    otherwise any neighbour with candidate_mask[j] qualifies (new-infection
    case).
    """
    out = np.empty(nodes.shape[0], dtype=np.float64)
    for pos, i in enumerate(nodes):
        neigh = adj_list[i]
        cand = neigh[candidate_mask[neigh]]
        if evasion_prob is not None and cand.size:
            passed = cand[rng.random(cand.size) < evasion_prob[cand]]
            if passed.size:
                cand = passed
        if cand.size == 0:
            out[pos] = psi[i]  # defensive fallback; should not normally trigger
            continue
        out[pos] = psi[rng.choice(cand)]
    return out


def run_one_realization(N: int, TT: int, beta: float, sigma: float, pm: float,
                         num_edges: int, rng: np.random.Generator) -> float:
    """Returns Re = number of individuals reinfected at least once."""
    A, adj_list = generate_er_network(N, num_edges, rng)
    src, dst = A.nonzero()  # directed edge list, fixed for this realization

    state = np.zeros(N, dtype=np.int8)      # 0 = susceptible
    state2 = np.zeros(N, dtype=np.int8)     # 1 = will be asymptomatic, 2 = symptomatic path
    state3 = -np.ones(N, dtype=np.int8)     # -1 = not yet reinfected, 1 = reinfected once
    psi = np.zeros(N, dtype=np.float64)
    psi2 = hill(psi, pm)

    n_seed = max(N // 500, 1)
    state[:n_seed] = 1  # seed exposures
    seed_path = np.where(rng.random(n_seed) < 0.3, 1, 2)
    state2[:n_seed] = seed_path

    for t in range(TT):
        state0 = state  # decisions this step are based on the state as it
                         # stands at the top of the loop (see module docstring)
        contagious_mask = np.isin(state0, CONTAGIOUS_STATES)

        # --- neighbour-count based quantities, O(edges) not O(N^2) ---
        dst_contagious = contagious_mask[dst]
        src_c = src[dst_contagious]
        linum = np.bincount(src_c, minlength=N).astype(np.float64)
        linum2 = (1.0 - beta / 96.0) ** linum  # escape prob. from new infection

        dst_e = dst[dst_contagious]
        evasion_draws = rng.random(src_c.shape[0]) < psi2[dst_e]
        infe = np.bincount(src_c[evasion_draws], minlength=N).astype(np.float64)
        linum3 = (1.0 - beta / 96.0) ** infe   # escape prob. from reinfection

        # --- S(0) -> E(1) ---
        S_idx = np.where(state0 == 0)[0]
        if S_idx.size:
            newly_exposed = S_idx[rng.random(S_idx.size) < (1.0 - linum2[S_idx])]
            if newly_exposed.size:
                state2[newly_exposed] = np.where(
                    rng.random(newly_exposed.size) < 0.3, 1, 2)
                donor_psi = pick_donor(newly_exposed, adj_list, contagious_mask, psi, rng)
                psi[newly_exposed] = donor_psi
                psi2[newly_exposed] = hill(donor_psi, pm)
                state[newly_exposed] = 1

        # --- E(1) -> asymptomatic(9) / pre-symptomatic(10) ---
        E_idx = np.where(state0 == 1)[0]
        if E_idx.size:
            asym_candidates = E_idx[state2[E_idx] == 1]
            presym_candidates = E_idx[state2[E_idx] == 2]
            if asym_candidates.size:
                trans = asym_candidates[rng.random(asym_candidates.size) < 1.0 / (4.0 * 96)]
                state[trans] = 9
            if presym_candidates.size:
                trans = presym_candidates[rng.random(presym_candidates.size) < 1.0 / (3.0 * 96)]
                state[trans] = 10

        # --- asymptomatic(9) -> recovered(7) ---
        idx = np.where(state0 == 9)[0]
        if idx.size:
            trans = idx[rng.random(idx.size) < 1.0 / (6.0 * 96)]
            state[trans] = 7

        # --- pre-symptomatic(10) -> mild(2) / severe(3) / critical(4) ---
        idx = np.where(state0 == 10)[0]
        if idx.size:
            moving = idx[rng.random(idx.size) < 1.0 / (2.0 * 96)]
            if moving.size:
                b = rng.random(moving.size)
                state[moving[b < 55.0 / 70.0]] = 2
                state[moving[(b >= 55.0 / 70.0) & (b < 65.0 / 70.0)]] = 3
                state[moving[b >= 65.0 / 70.0]] = 4
                # NOTE: the original C++ also has a `c<-1` branch here that
                # would move some "2"s to state 12; that condition is always
                # false (dead code) in the source, so state 12 is never
                # actually created -- kept unreached here too, for fidelity.

        # --- mild(2) / mild-reinfected(12, unreachable) -> recovered(7) ---
        idx = np.where((state0 == 2) | (state0 == 12))[0]
        if idx.size:
            trans = idx[rng.random(idx.size) < 1.0 / (5.0 * 96)]
            state[trans] = 7

        # --- severe(3) -> hospitalized(5) ---
        idx = np.where(state0 == 3)[0]
        if idx.size:
            trans = idx[rng.random(idx.size) < 1.0 / (4.0 * 96)]
            state[trans] = 5

        # --- critical(4) -> ventilated(6) ---
        idx = np.where(state0 == 4)[0]
        if idx.size:
            trans = idx[rng.random(idx.size) < 1.0 / (3.0 * 96)]
            state[trans] = 6

        # --- hospitalized(5) -> recovered(7) / deceased(8) ---
        idx = np.where(state0 == 5)[0]
        if idx.size:
            moving = idx[rng.random(idx.size) < 1.0 / (11.0 * 96)]
            if moving.size:
                b = rng.random(moving.size)
                state[moving[b < 0.85]] = 7
                state[moving[b >= 0.85]] = 8

        # --- ventilated(6) -> recovered(7) / deceased(8) ---
        idx = np.where(state0 == 6)[0]
        if idx.size:
            moving = idx[rng.random(idx.size) < 1.0 / (13.0 * 96)]
            if moving.size:
                b = rng.random(moving.size)
                state[moving[b < 0.5]] = 7
                state[moving[b >= 0.5]] = 8

        # --- recovered(7) / vaccinated(11) -> reinfection ---
        idx = np.where((state0 == 7) | (state0 == 11))[0]
        if idx.size:
            eligible = idx[state3[idx] < 0]
            if eligible.size:
                reinfected = eligible[rng.random(eligible.size) < (1.0 - linum3[eligible])]
                if reinfected.size:
                    state3[reinfected] = 1
                    state2[reinfected] = np.where(
                        rng.random(reinfected.size) < 0.3, 1, 2)
                    donor_psi = pick_donor(reinfected, adj_list, contagious_mask,
                                            psi, rng, evasion_prob=psi2)
                    psi[reinfected] = donor_psi
                    psi2[reinfected] = hill(donor_psi, pm)
                    state[reinfected] = 1

        # --- fitness random walk for every node on the "infectious track" ---
        track_idx = np.where(np.isin(state, (1, 2, 3, 4, 5, 6, 9, 10, 12)))[0]
        if track_idx.size:
            d = rng.random((track_idx.size, 12)).sum(axis=1) - 6.0
            new_psi = psi[track_idx] + (d * sigma) / 96.0
            np.clip(new_psi, 0.0, 1.0, out=new_psi)
            psi[track_idx] = new_psi
            psi2[track_idx] = hill(new_psi, pm)

        # --- original's early-exit heuristic (kept exactly, including its
        # blind spot for asymptomatic/pre-symptomatic carriers) ---
        I_count = np.count_nonzero(np.isin(state, (2, 3, 4, 5, 6, 12)))
        E_count = np.count_nonzero(state == 1)
        if I_count < 0.5 and E_count < 0.5:
            break

    return float(np.count_nonzero(state3 > 0))  # Re


def run_sweep(out_path: str, N: int, TT: int, n_realizations: int, pm: float,
              num_edges: int, sigmas: np.ndarray, betas: np.ndarray,
              reemergence_threshold: float, seed: int | None, verbose: bool = True):
    rng = np.random.default_rng(seed)
    t_start = _time.time()
    total = len(sigmas) * len(betas)
    done = 0
    with open(out_path, "w") as fp:
        for sigma in sigmas:
            for beta in betas:
                count2 = 0
                for _ in range(n_realizations):
                    Re = run_one_realization(N, TT, beta, sigma, pm, num_edges, rng)
                    if Re > reemergence_threshold:
                        count2 += 1
                fp.write(f"{beta:.6f}  {sigma:.6f}  {count2 / n_realizations:.6f}\n")
                fp.flush()
                done += 1
                if verbose:
                    elapsed = _time.time() - t_start
                    print(f"[ReemergenceImmunityEvasion] {done}/{total} "
                          f"(beta={beta:.4f}, sigma={sigma:.4f}) "
                          f"reemergence_frac={count2 / n_realizations:.2f} "
                          f"elapsed={elapsed:.1f}s", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="da.txt")
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--TT", type=int, default=40000)
    p.add_argument("--realizations", type=int, default=50)
    p.add_argument("--pm", type=float, default=0.3, help="reinfection-rate Hill midpoint")
    p.add_argument("--num-edges", type=int, default=37500)
    p.add_argument("--reemergence-threshold", type=float, default=500.0)
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--sigma-max", type=float, default=0.25)
    p.add_argument("--sigma-step", type=float, default=0.1)
    p.add_argument("--sigma-values", type=int, default=None)
    p.add_argument("--beta-min", type=float, default=0.0)
    p.add_argument("--beta-max", type=float, default=0.106)
    p.add_argument("--beta-step", type=float, default=0.005)
    p.add_argument("--beta-values", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.sigma_values:
        sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_values)
    else:
        n = int(round((args.sigma_max - args.sigma_min) / args.sigma_step)) + 1
        sigmas = args.sigma_min + args.sigma_step * np.arange(n)
        sigmas = sigmas[sigmas <= args.sigma_max + 1e-9]

    if args.beta_values:
        betas = np.linspace(args.beta_min, args.beta_max, args.beta_values)
    else:
        n = int(round((args.beta_max - args.beta_min) / args.beta_step))
        betas = args.beta_min + args.beta_step * np.arange(n)
        betas = betas[betas < args.beta_max - 1e-9 + args.beta_step]

    run_sweep(
        out_path=args.out, N=args.N, TT=args.TT, n_realizations=args.realizations,
        pm=args.pm, num_edges=args.num_edges, sigmas=sigmas, betas=betas,
        reemergence_threshold=args.reemergence_threshold, seed=args.seed,
        verbose=not args.quiet)


if __name__ == "__main__":
    main()
