# Planned synthetic experiment suite

The synthetic experiment suite is defined by:

```text
configs/experiments/experiment_manifest.yaml
configs/experiments/EXPERIMENT_MANIFEST.md
```

## Experiment groups

- Exp1: fixed-mixing iVAE degradation under cross-domain mixing drift.
- Exp2: first-order linear mixing-drift alignment on the linear benchmark.
- Exp3: auxiliary-condition ablation.
- Exp4: boundary analysis under higher-order nonlinear drift.
- Exp5--Exp6: appendix experiments for additional nonlinear and residual-shift settings.

## Commands

List all planned experiments:

```bash
python scripts/run_planned_experiments.py --list
```

Run main experiments:

```bash
python scripts/run_planned_experiments.py --group main --run run/planned_suite --seed 0 --n-sims 5
```

Run appendix experiments:

```bash
python scripts/run_planned_experiments.py --group appendix --run run/planned_suite --seed 0 --n-sims 5
```

Generate figures:

```bash
python plot_synthetic_results.py --run-dir run/planned_suite --out-dir run/planned_suite/figures
```

The generated output directory contains summary CSV files, JSON summaries, checkpoints, and figures for the synthetic benchmark analysis.
