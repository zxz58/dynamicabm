# Community SIRS Handoff

## Current State

The active community-model code lives at the repo root:

- `community_sirs.py` implements the continuous-time Gillespie SIRS ABM.
- `CommunitySIRS.py` is the thin simulation CLI wrapper.
- `PlotCommunitySIRS.py` creates the steady-state trajectory figure.
- `CommunitySIRSNetworkSnapshots.py` creates selected-time network snapshots.

The old paper-port code has been moved into `paper_code/`. Treat that folder
as historical/reference code unless the task explicitly asks to modify the
paper reproduction scripts.

## Model Summary

The model uses 2 or 3 Watts-Strogatz communities. Hosts are in states
`S=0`, `I=1`, `R=2`, or `D=3`. Events are selected with Gillespie dynamics:
transmission, recovery, waning immunity, disease mortality, behavioral edge
removal, and inter-community movement. Virulence is inherited at transmission
with Gaussian mutation clipped to `[v_min, v_max]`.

Virulence controls:

- `beta(v)`: transmission, increasing.
- `alpha(v)`: disease mortality, increasing.
- `gamma(v)`: recovery, decreasing.
- `delta(v)`: behavioral edge removal, increasing.
- `phi(v)`: movement, decreasing.

Each trade-off supports `linear`, `concave`, and `convex` shape options.

## Figure Workflow

Use sampled trajectory output for steady-state figures:

```bash
python3 CommunitySIRS.py --samples-out community_sirs_samples.csv --out community_sirs_summary.csv
python3 PlotCommunitySIRS.py --samples community_sirs_samples.csv --out community_sirs_steady.png --burn-in-time 100
```

Use the snapshot script for selected network states:

```bash
python3 CommunitySIRSNetworkSnapshots.py --snapshot-times 0,100,500 --out-dir community_sirs_snapshots
```

The snapshot layout fixes community positions and uses local spring layouts
inside each community. Local edges and inter-community edges are drawn
differently; infected nodes use a virulence colormap.

## Checks To Run

```bash
python3 -m py_compile community_sirs.py CommunitySIRS.py PlotCommunitySIRS.py CommunitySIRSNetworkSnapshots.py
python3 CommunitySIRS.py --N 90 --K 3 --t-max 20 --burn-in-time 5 --sample-interval 1 --realizations 2 --seed 1 --samples-out /tmp/community_samples.csv --out /tmp/community_summary.csv --quiet
python3 PlotCommunitySIRS.py --samples /tmp/community_samples.csv --out /tmp/community_steady.png --burn-in-time 5
python3 CommunitySIRSNetworkSnapshots.py --N 90 --K 3 --t-max 20 --burn-in-time 5 --snapshot-times 0,5,20 --seed 1 --out-dir /tmp/community_snapshots --quiet
```

## Likely Next Steps

- Add a sensitivity runner that sweeps one or two parameters and writes
  plotting-friendly CSVs.
- Add heatmap plotting for endemic prevalence, mean virulence, deaths, and
  inter-community edges.
- Consider performance improvements if larger `N`, longer `t-max`, or larger
  parameter sweeps become routine.
