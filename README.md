# Community SIRS ABM

This branch focuses on a community-structured SIRS agent-based model for
endemic steady-state behavior and virulence evolution. The original paper-port
code has been moved into `paper_code/`; the root-level scripts now center on
the new community model and its figure workflow.

## Community Model Files

- `CommunitySIRS.py`: thin command-line entry point.
- `community_sirs.py`: core Gillespie SIRS engine.
- `PlotCommunitySIRS.py`: single-scenario four-panel diagnostic plotting from
  sampled model output.
- `CompareCommunitySIRS.py`: multi-scenario 1x2 comparison figure from
  multiple sampled model outputs.
- `CommunitySIRSNetworkSnapshots.py`: selected-time community network snapshot
  PNGs plus metadata.

The model supports 2 or 3 Watts-Strogatz communities, S/I/R/D states, waning
immunity, disease-induced mortality, behavioral edge removal and restoration,
inter-community movement edges, and heritable virulence with mutation at
transmission.

Inter-community edges are temporary. New long-range contacts are created by
movement events controlled by `--phi-max` and `phi(v)`, while each active
inter-community edge decays at Gillespie rate `--inter-edge-decay-rate`. As a
rough starting point, the expected number of inter-community edges scales like
creation rate divided by decay rate. Increase `--inter-edge-decay-rate` to
shorten edge lifetimes, or increase `--phi-max` to create more long-range
contacts. Tune these together to keep the realized steady-state count around
20-30 edges.

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
  --inter-edge-decay-rate 0.08 \
  --out community_sirs_summary.csv \
  --samples-out community_sirs_samples.csv
```

Plot the sampled steady-state diagnostics. The fourth panel shows all
state-composition fractions together: S/I/R/D.

```bash
python3 PlotCommunitySIRS.py \
  --samples community_sirs_samples.csv \
  --out community_sirs_steady.png \
  --burn-in-time 5
```

Compare virulence across inter-community-connectivity scenarios by generating
multiple sample files and passing `LABEL:PATH` inputs to
`CompareCommunitySIRS.py`. The comparison figure colors both points and lines
by sample-file label. The left panel shows mean infected virulence versus mean
cumulative cross-community transmissions, with one point per realization and a
larger mean marker per scenario when multiple realizations are available. The
right panel shows mean infected virulence over time for each scenario:

```bash
python3 CommunitySIRS.py \
  --phi-max 0.01 --inter-edge-decay-rate 0.08 \
  --samples-out samples_low_phi.csv \
  --out summary_low_phi.csv
python3 CommunitySIRS.py \
  --phi-max 0.03 --inter-edge-decay-rate 0.08 \
  --samples-out samples_high_phi.csv \
  --out summary_high_phi.csv
python3 CompareCommunitySIRS.py \
  --samples low_phi:samples_low_phi.csv \
  --samples high_phi:samples_high_phi.csv \
  --out community_sirs_comparison.png \
  --burn-in-time 100
```

You can compare edge-lifetime scenarios the same way:

```bash
python3 CommunitySIRS.py \
  --phi-max 0.02 --inter-edge-decay-rate 0.04 \
  --samples-out samples_slow_decay.csv \
  --out summary_slow_decay.csv
python3 CommunitySIRS.py \
  --phi-max 0.02 --inter-edge-decay-rate 0.12 \
  --samples-out samples_fast_decay.csv \
  --out summary_fast_decay.csv
python3 CompareCommunitySIRS.py \
  --samples slow_decay:samples_slow_decay.csv \
  --samples fast_decay:samples_fast_decay.csv \
  --out decay_comparison.png \
  --burn-in-time 100
```

If multiple `--samples` inputs are passed to `PlotCommunitySIRS.py`, it still
creates one separate four-panel diagnostic figure per file using label suffixes:

```bash
python3 PlotCommunitySIRS.py \
  --samples low_phi:samples_low_phi.csv \
  --samples high_phi:samples_high_phi.csv \
  --out community_sirs_steady.png \
  --burn-in-time 100
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
variance, cumulative deaths, cumulative cross-community transmissions, active
edges, inter-community edges, realized clustering, `inter_edge_decay_rate`, and
extinction status.

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
