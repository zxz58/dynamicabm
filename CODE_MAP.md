# Code Map

This repository is a compact Python port of epidemic mutation simulation
code from the paper "Epidemic spreading under mutually independent intra-
and inter-host pathogen evolution." It is organized as command-line
simulation scripts around a few reusable Monte Carlo engines.

## High-Level Structure

```text
CLI wrapper scripts
  -> shared phase-diagram engines
     -> common network / mutation helpers
        -> NumPy + SciPy sparse simulation loops

Standalone experiment scripts
  -> common helpers where needed
  -> write tabular outputs
```

Most scripts write whitespace-delimited output to `da.txt` by default,
configurable with `--out`. `InfectionTimeSeries.py` instead writes a CSV
and can also create a PNG plot.

## Core Shared Module

### `common.py`

Shared primitives used by the network-based simulations.

- `generate_er_network(N, num_edges, rng)`: builds an undirected random graph
  with exactly `num_edges`, returning both a SciPy CSR adjacency matrix and
  an adjacency list.
- `fitness_random_walk(psi_values, sigma, X, lower, upper, rng)`: applies the
  original code's "sum twelve uniforms minus six" mutation step and clips the
  result.
- `pick_random_neighbor_index(...)`: for each target node, chooses a qualifying
  neighbor index, optionally weighted by a per-node value.
- `pick_random_neighbor(...)`: chooses a donor neighbor and returns one of its
  attributes, such as pathogen fitness `Psi`.

## Phase Diagram Engines

### `phase_diagram_alpha.py`

Reusable engine for alpha-mutation phase diagrams.

Model:

- Pathogen fitness `Psi` affects host recovery.
- States are `-1` susceptible, `1` infected, and `-2` recovered.
- SIR recovery moves nodes to `-2`.
- SIS recovery recycles nodes to `-1`.
- Recovery probability is `alpha / Psi[i]`.
- Transmission uses fixed `beta`, with optional `avoidance`, `distancing`,
  and symptomatic/asymptomatic behavior.

Important functions:

- `sigma_grid(...)`
- `beta_grid(...)`
- `run_one_realization(...)`
- `run_phase_diagram(...)`
- `build_arg_parser(...)`
- `resolve_grids(...)`
- `main(...)`

### `phase_diagram_beta.py`

Reusable engine for beta-mutation phase diagrams.

Model:

- Pathogen fitness `Psi` affects transmissibility.
- Each infected neighbor transmits with probability related to `beta * Psi[j]`.
- Recovery is plain `alpha`.
- Newly infected nodes inherit `Psi` from a donor chosen with fitness weighting.
- Escape probability is computed efficiently by summing log factors through a
  sparse matrix-vector product.

Important functions:

- `run_one_realization(...)`
- `run_phase_diagram(...)`
- `build_arg_parser(...)`
- `main(...)`

It reuses grid and parser helpers from `phase_diagram_alpha.py`.

### `phase_diagram_combined.py`

Reusable engine for combined-mutation phase diagrams.

Model:

- `Psi_alpha` controls recovery evasion through `alpha / Psi_alpha[i]`.
- `Psi_beta` controls transmissibility through `beta * Psi_beta[j]`.
- Both traits mutate simultaneously.
- Newly infected nodes inherit both traits from the same donor neighbor.
- Donor selection is weighted by `Psi_beta`, because transmissibility decides
  which neighbor most likely caused infection.

Important functions:

- `run_one_realization(...)`
- `run_phase_diagram(...)`
- `build_arg_parser(...)`
- `resolve_grids(...)`
- `main(...)`

## Thin CLI Wrappers

These files only select dynamics/model defaults and delegate to the reusable
phase-diagram engines.

- `PhaseDiagramSIR_alpha.py`: calls `phase_diagram_alpha.main("SIR", ...)`
  with `Psi` cap `20`.
- `PhaseDiagramSIS_alpha.py`: calls `phase_diagram_alpha.main("SIS", ...)`
  with `Psi` cap `20`.
- `PhaseDiagramSIR_beta.py`: calls `phase_diagram_beta.main("SIR", ...)`
  with `Psi` cap `10`.
- `PhaseDiagramSIS_beta.py`: calls `phase_diagram_beta.main("SIS", ...)`
  with `Psi` cap `10`.
- `PhaseDiagramSIR_combined.py`: calls `phase_diagram_combined.main("SIR", ...)`
  with `Psi_alpha` cap `20` and `Psi_beta` cap `10`.
- `PhaseDiagramSIS_combined.py`: calls `phase_diagram_combined.main("SIS", ...)`
  with `Psi_alpha` cap `20` and `Psi_beta` cap `10`.

## Time-Series Tool

### `InfectionTimeSeries.py`

Runs one or more fixed `(beta, sigma)` scenarios and records full time courses
instead of only end-of-run outbreak fractions.

Supported models:

- `SIR_alpha`
- `SIS_alpha`
- `SIR_beta`
- `SIS_beta`
- `SIR_combined`
- `SIS_combined`

Tracked metrics:

- `cumulative_infections`
- `cumulative_symptomatic_infections`
- `cumulative_asymptomatic_infections`
- `current_infected`
- `current_symptomatic_infected`
- `current_asymptomatic_infected`
- `susceptible_remaining`

Important functions:

- `parse_float_list(...)`
- `run_scenario(...)`
- `make_plot(...)`
- `main(...)`

Outputs:

- CSV, default `timeseries.csv`
- PNG plot, default path derived from the CSV name

## Standalone Experiments

### `TimingMitigation.py`

SIR alpha-mutation experiment where the transmission rate changes at a
mitigation start time `tR`.

Flow:

```text
for tR in sweep:
  for realization:
    build network
    simulate SIR alpha model
    beta = beta_before before tR
    beta = beta_after after tR
    track outbreak and high-fitness threshold crossing
  write row
```

Output columns:

```text
tR, beta_after, sigma, outbreak_fraction, second_wave_fraction
```

Important functions:

- `run_one_realization(...)`
- `run_sweep(...)`
- `main(...)`

### `ReemergenceImmunityEvasion.py`

Detailed COVID-style compartmental model with immune evasion and reinfection.

State codes:

- `0`: susceptible
- `1`: exposed
- `2`: mild symptoms
- `3`: severe symptoms
- `4`: critical symptoms
- `5`: hospitalized
- `6`: ventilated
- `7`: recovered
- `8`: deceased
- `9`: asymptomatic infectious
- `10`: pre-symptomatic infectious
- `11`: vaccinated, preserved but unused
- `12`: mild symptoms after reinfection, preserved as unreachable original
  dead-code behavior

Important functions:

- `hill(psi, pm)`: transforms raw immune-evasion fitness into evasion
  probability.
- `pick_donor(...)`: donor selection for infection or reinfection.
- `run_one_realization(...)`: returns the number of reinfected individuals.
- `run_sweep(...)`: sweeps `(beta, sigma)` and writes reemergence fractions.
- `main(...)`

### `TestForSigma.py`

Non-network within-host/between-host fitness experiment.

Model:

- `N` strains are arranged along an integer axis.
- `M` individuals each carry one strain index.
- Strain mutation is a `+1`, `-1`, or stay random walk.
- Death/replacement is driven by intra-host fitness.
- The sweep varies `sigmaPsi`, while `sigmaPhi` is fixed by default.

Important functions:

- `irwin_hall_shift(...)`
- `run_realization(...)`
- `run_sweep(...)`
- `main(...)`

## Data And Dependencies

### `requirements.txt`

Runtime dependencies:

- `numpy`
- `scipy`
- `matplotlib`

### `data/`

Contains COVID-19 time series files and `FitBetaByCountry.xlsx`. The project
README states these are carried over from the original repository for external
parameter fitting and are not consumed by the simulation scripts themselves.

## Main Execution Flow

```text
User runs CLI script
  -> argparse parses model, grid, and runtime options
  -> NumPy RNG is created from optional --seed
  -> for each parameter combination:
       -> for each realization:
            -> generate_er_network(...)
            -> initialize states and pathogen fitness arrays
            -> simulate TT timesteps
            -> compute outbreak / reemergence / time-series metrics
       -> write result row(s)
```

## Mental Model

The main reusable simulation kernels are:

- `phase_diagram_alpha.py`: fitness affects recovery.
- `phase_diagram_beta.py`: fitness affects transmission.
- `phase_diagram_combined.py`: both fitness axes mutate together.

Everything else is either a thin wrapper around those kernels, a
special-purpose experiment, or output/data/dependency support.
