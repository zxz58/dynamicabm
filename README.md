# Epidemic Spreading Under Mutation — Python port

Python translation of the C++ simulation code from *"Epidemic spreading
under mutually independent intra- and inter-host pathogen evolution."*
All seven original programs are here, ported 1:1 in terms of model logic,
plus the accompanying COVID-19 data files (unchanged).

## File correspondence

| Original C++                          | Python file(s)                                   | What it produces |
|----------------------------------------|---------------------------------------------------|-------------------|
| `PhaseDiagramSIR_alpha.cpp`             | `PhaseDiagramSIR_alpha.py` (+ `phase_diagram_alpha.py`) | Fig. 2, Fig. 3a-c, Fig. S4a |
| `PhaseDiagramSIS_alpha.cpp`             | `PhaseDiagramSIS_alpha.py` (+ `phase_diagram_alpha.py`) | Fig. 2, Fig. S5a |
| `PhaseDiagramSIR_beta.cpp`              | `PhaseDiagramSIR_beta.py` (+ `phase_diagram_beta.py`)   | Fig. 3, Fig. S4b |
| `PhaseDiagramSIS_beta.cpp`              | `PhaseDiagramSIS_beta.py` (+ `phase_diagram_beta.py`)   | Fig. S5b |
| `TimingMitigation.cpp`                  | `TimingMitigation.py`                              | Fig. 4 |
| `ReemergenceImmunity Evasion.cpp`       | `ReemergenceImmunityEvasion.py`                    | Fig. 5 |
| `TestForSigma.cpp`                      | `TestForSigma.py`                                  | Fig. S6 |

`common.py` holds the pieces shared by every script: Erdos-Renyi-style
network generation and the "sum of twelve uniforms minus six" fitness
random-walk step.

Each script writes its output to `da.txt` by default (same filename and
column layout as the original), configurable with `--out`.

## Extensions beyond the original C++ (not in the paper)

These were added on request and aren't translations of anything in the
original repository:

| File | What it adds |
|---|---|
| `phase_diagram_combined.py`, `PhaseDiagramSIR_combined.py`, `PhaseDiagramSIS_combined.py` | A third mutation model where the pathogen's recovery-evasion trait (Psi_alpha, as in the alpha model) and transmissibility trait (Psi_beta, as in the beta model) mutate **simultaneously**, rather than only one axis being allowed to evolve at a time. Same (beta, sigma) phase-diagram sweep structure as the alpha/beta scripts; each trait keeps its own clip cap (`--psi-alpha-cap`, `--psi-beta-cap`). New infections inherit *both* traits from the same donor neighbor (weighted by that neighbor's Psi_beta, since that's what governs transmission), so a variant's whole phenotype moves together rather than mixing across sources. |
| `InfectionTimeSeries.py` | Runs one or more scenarios (sweeping `--avoidance`/`--distancing`) at a fixed (beta, sigma) and outputs the *full time course* (cumulative infections, current infected, susceptible remaining, each split into symptomatic/asymptomatic) instead of just an end-of-run outbreak fraction, plus an automatic PNG plot. Works with all four original models and the combined model via `--model`. |
| `avoidance` / `distancing` / `symptomatic_fraction` parameters (added to `phase_diagram_alpha.py`, `phase_diagram_beta.py`, and `phase_diagram_combined.py`) | Behavioral toggles for isolation/social-distancing experiments. `avoidance` discounts transmission specifically from *symptomatic* infected neighbors (targeted precaution); `distancing` discounts transmission from *all* infected neighbors regardless of symptom visibility (blanket precaution). At the default `symptomatic_fraction=1.0` (everyone symptomatic) the two are mathematically interchangeable; set `symptomatic_fraction` below 1.0 to introduce invisible asymptomatic carriers that `avoidance` structurally cannot protect against but `distancing` still can. See the docstring at the top of `phase_diagram_alpha.py` for the full mathematical rationale. |

## Requirements

```
pip install numpy scipy matplotlib

```
(`requirements.txt` included.) No other dependencies.

## Running

Every script is a CLI program with the *same default parameters as the
original C++* (network size, time horizon, parameter grids, number of
realizations). For example, to reproduce `PhaseDiagramSIR_alpha.cpp`
exactly as originally parameterised:

```bash
python PhaseDiagramSIR_alpha.py --out sir_alpha_da.txt
```

Every script also accepts flags to shrink the problem for a quick sanity
check before committing to a full run, e.g.:

```bash
python PhaseDiagramSIR_alpha.py --N 500 --TT 300 --realizations 5 \
    --num-edges 3750 --sigma-values 10 --beta-values 10 --seed 0
```

Run `python <script>.py --help` on any file for its full option list.

## What changed vs. the C++ source, and why

The physical model (state definitions, transition rates, mutation process,
network topology, outbreak criteria, output format) is preserved exactly.
Three implementation choices were made to get this running in Python in a
tractable amount of time; none of them change what is being simulated:

1. **Sparse adjacency instead of a dense NxN array.** The original C++
   stores and loops over a full 5000x5000 array every time step
   (25,000,000 element inspections/step), even though the network only has
   37,500 edges (~1.5% density). All scripts here store the network as a
   `scipy.sparse` matrix and use matrix-vector products / edge-list
   operations, which do the identical calculation (same neighbours, same
   counts) in O(edges) instead of O(N^2). This is what makes full-scale
   runs (N=5000, thousands of time steps, tens of thousands of parameter
   combinations) finish in a practical amount of time instead of taking
   orders of magnitude longer than the C++ version.

2. **Synchronous per-step updates.** The C++ code loops over nodes
   `i = 0..N-1` sequentially and mutates the shared `state[]` array as it
   goes, so a node's decision can, in principle, see a handful of *other*
   nodes' already-updated states from earlier in the same pass (an
   artifact of loop order, not a deliberate feature of the model). Every
   script here instead makes all of one time step's decisions from a
   consistent snapshot of the state at the start of that step, matching
   the model exactly as any textbook or paper description of it would
   read, and removing the sequential-order dependency. Neighbour-fitness
   inheritance ("who did I catch this variant from?") is still handled
   node-by-node with a small, exact Python loop -- it is restricted to the
   handful of nodes that actually change state that step, since infection
   events are always a small fraction of N per step.

3. **The original's own optimizations/quirks were kept where they affect
   results.** For instance, `ReemergenceImmunityEvasion.py` keeps the
   original's early-exit condition (`stop once no one is exposed or
   symptomatic`) verbatim, including the fact that it does not check
   asymptomatic/pre-symptomatic counts -- and the dead "state 12" branch
   in the pre-symptomatic transition (a condition that is always false in
   the original C++, so state 12 is never actually reached) is preserved
   as dead code rather than silently removed or "fixed."

See the module docstring at the top of each `.py` file for the specific
details relevant to that script.

## Performance expectations

Even with the sparse-matrix speedup, the *paper-scale* parameter sweeps
are inherently expensive Monte Carlo studies (that is true of the original
C++ too -- these look like the kind of job the original authors ran on a
cluster over an extended period, not something meant to finish on a laptop
in seconds). Indicative timings measured in this environment (single core,
no special hardware):

| Script | One realization at full scale (N=5000) | Full default grid |
|---|---|---|
| `PhaseDiagramSIR_alpha` / `SIS_alpha` / `SIR_beta` / `SIS_beta` | ~0.5 s (TT=3000) | ~101 sigma x 51 beta x 50 realizations ≈ tens of hours |
| `TimingMitigation` | ~0.5 s (TT=3000) | 61 tR values x 100 realizations ≈ under an hour |
| `ReemergenceImmunityEvasion` | ~2-15 s (TT up to 40000, varies a lot with beta/sigma) | 3 sigma x ~21 beta x 50 realizations ≈ up to half a day |
| `TestForSigma` | no network; K=5000 realizations of a much smaller/faster process | seconds to low minutes |

Use `--sigma-values` / `--beta-values` / `--tR-values` / `--sigmaPsi-values`
(coarser, evenly-spaced overrides of the default grids) plus smaller
`--N` / `--TT` / `--realizations` to get a fast, representative preview of
any phase diagram, then scale up to the full grid for a publication-grade
run (e.g. left running overnight, or split across parameter ranges and run
in parallel processes/machines -- every script is a pure function of its
CLI arguments plus an optional `--seed`, so this parallelises trivially by
running multiple instances with disjoint sigma/beta ranges and
concatenating the output files).

## Data

`data/` is carried over unchanged from the original repository: JHU
COVID-19 time series (used for offline beta/R0 fitting by country, per
Meidan et al. 2021) and `FitBetaByCountry.xlsx`. These are inputs to an
external fitting step described in the original `data/Readme.txt`, not
consumed by any of the simulation scripts themselves.
