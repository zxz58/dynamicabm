# Community SIRS Handoff

## Current State

The active community-model code lives at the repo root:

- `community_sirs.py` implements the continuous-time Gillespie SIRS ABM.
- `CommunitySIRS.py` is the thin simulation CLI wrapper.
- `PlotCommunitySIRS.py` creates single-scenario four-panel diagnostics.
- `CompareCommunitySIRS.py` creates the multi-scenario 1x2 comparison figure.
- `CommunitySIRSNetworkSnapshots.py` creates selected-time network snapshots.

The old paper-port code has been moved into `paper_code/`. Treat that folder
as historical/reference code unless the task explicitly asks to modify the
paper reproduction scripts.

## Model Summary

The model uses 2 or 3 Watts-Strogatz communities. Hosts are in states
`S=0`, `I=1`, `R=2`, or `D=3`. Events are selected with Gillespie dynamics:
transmission, recovery, waning immunity, disease mortality, behavioral edge
removal, inter-community movement, and inter-community edge decay. Virulence
is inherited at transmission with Gaussian mutation clipped to `[v_min, v_max]`.

Virulence controls:

- `beta(v)`: transmission, increasing.
- `alpha(v)`: disease mortality, increasing.
- `gamma(v)`: recovery, decreasing.
- `delta(v)`: behavioral edge removal, increasing.
- `phi(v)`: movement, decreasing.

Each trade-off supports `linear`, `concave`, and `convex` shape options.

Inter-community edges should be treated as temporary contacts. Movement events
create them; `--inter-edge-decay-rate` removes each active inter-community
edge with an exponential lifetime. A rough tuning heuristic is:

```text
expected inter-community edges ~= creation rate / inter-edge decay rate
```

The realized count also depends on duplicate movement attempts, deaths,
behavioral removals, and infection dynamics. To target 20-30 inter-community
edges at steady state, first tune `--inter-edge-decay-rate` against the chosen
`--phi-max`.

## Figure Workflow

Use sampled trajectory output for steady-state figures:

```bash
python3 CommunitySIRS.py --samples-out community_sirs_samples.csv --out community_sirs_summary.csv
python3 PlotCommunitySIRS.py --samples community_sirs_samples.csv --out community_sirs_steady.png --burn-in-time 100
```

`PlotCommunitySIRS.py` is the single-scenario diagnostic plotter. Its four
panels are prevalence/I fraction, mean infected virulence, active and
inter-community edges, and S/I/R/D state composition.

For comparing connectivity scenarios, generate separate sample files and pass
them to `CompareCommunitySIRS.py` with `LABEL:PATH` inputs. The comparison
figure colors points and lines by sample-file label:

```bash
python3 CommunitySIRS.py --phi-max 0.01 --inter-edge-decay-rate 0.08 --samples-out samples_low_phi.csv --out summary_low_phi.csv
python3 CommunitySIRS.py --phi-max 0.03 --inter-edge-decay-rate 0.08 --samples-out samples_high_phi.csv --out summary_high_phi.csv
python3 CompareCommunitySIRS.py \
  --samples low_phi:samples_low_phi.csv \
  --samples high_phi:samples_high_phi.csv \
  --out community_sirs_comparison.png \
  --burn-in-time 100
```

The left comparison panel uses realized cross-community transmission events,
not the raw number of inter-community edges. It counts infections where the
source and newly infected host are in different communities. The right panel
compares mean infected virulence over time across the supplied scenarios.

Use the snapshot script for selected network states:

```bash
python3 CommunitySIRSNetworkSnapshots.py --snapshot-times 0,100,500 --out-dir community_sirs_snapshots
```

The snapshot layout fixes community positions and uses local spring layouts
inside each community. Local edges and inter-community edges are drawn
differently; infected nodes use a virulence colormap.

## Checks To Run

```bash
python3 -m py_compile community_sirs.py CommunitySIRS.py PlotCommunitySIRS.py CompareCommunitySIRS.py CommunitySIRSNetworkSnapshots.py
python3 CommunitySIRS.py --N 90 --K 3 --t-max 20 --burn-in-time 5 --sample-interval 1 --realizations 2 --seed 1 --samples-out /tmp/community_samples.csv --out /tmp/community_summary.csv --quiet
python3 PlotCommunitySIRS.py --samples /tmp/community_samples.csv --out /tmp/community_steady.png --burn-in-time 5
python3 CompareCommunitySIRS.py --samples base:/tmp/community_samples.csv --out /tmp/community_comparison.png --burn-in-time 5
python3 CommunitySIRSNetworkSnapshots.py --N 90 --K 3 --t-max 20 --burn-in-time 5 --snapshot-times 0,5,20 --seed 1 --out-dir /tmp/community_snapshots --quiet
```

## Likely Next Steps

- Add a sensitivity runner that sweeps one or two parameters and writes
  plotting-friendly CSVs.
- Add heatmap plotting for endemic prevalence, mean virulence, deaths, and
  inter-community edges once parameter sweeps become routine.
- Consider performance improvements if larger `N`, longer `t-max`, or larger
  parameter sweeps become routine.
