# Community SIRS ABM

This branch focuses on a community-structured SIRS agent-based model for
endemic steady-state behavior and virulence evolution. The original paper-port
code has been moved into `paper_code/`; the root-level scripts now center on
the new community model and its figure workflow.

## Community Model Files

- `CommunitySIRS.py`: thin command-line entry point.
- `community_sirs.py`: core Gillespie SIRS engine.
- `PlotCommunitySIRS.py`: four-panel steady-state trajectory plotting from
  sampled model output.
- `CommunitySIRSNetworkSnapshots.py`: selected-time community network snapshot
  PNGs plus metadata.

The model supports 2 or 3 Watts-Strogatz communities, S/I/R/D states, waning
immunity, disease-induced mortality, behavioral edge removal and restoration,
inter-community movement edges, and heritable virulence with mutation at
transmission.

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a small steady-state simulation and save both summary and sample data:

```bash
python3 CommunitySIRS.py \
  --N 90 --K 3 \
  --t-max 20 --burn-in-time 5 --sample-interval 1 \
  --realizations 2 --seed 1 \
  --out community_sirs_summary.csv \
  --samples-out community_sirs_samples.csv
```

Plot the sampled steady-state trajectories:

```bash
python3 PlotCommunitySIRS.py \
  --samples community_sirs_samples.csv \
  --out community_sirs_steady.png \
  --burn-in-time 5
```

Create network snapshots at selected continuous times:

```bash
python3 CommunitySIRSNetworkSnapshots.py \
  --N 90 --K 3 \
  --t-max 20 --burn-in-time 5 \
  --snapshot-times 0,5,20 \
  --seed 1 \
  --out-dir community_sirs_snapshots
```

## Main Outputs

`CommunitySIRS.py` writes one summary row per realization. Important columns
include prevalence, S/R/D fractions, mean infected virulence, virulence
variance, cumulative deaths, active edges, inter-community edges, realized
clustering, and extinction status.

When `--samples-out` is provided, the model also writes one row per sampled
time point. Use this file for trajectory plots and later sensitivity-analysis
figures.

`CommunitySIRSNetworkSnapshots.py` writes PNG snapshots and
`snapshot_metadata.csv`, which records state counts, virulence summaries,
active/inter-community edge counts, clustering, deaths, and the PNG filename.

## Paper Code

The original paper reproduction scripts and documentation are now under
`paper_code/`, including the old phase diagram, time-series, mitigation,
reemergence, and snapshot tools. Keep new community-model development at the
repo root unless a change intentionally belongs to the historical paper port.
