# Synthetic benchmarks for ReMiDA-CLSRNet

This directory contains the synthetic source-recovery experiments used to analyze fixed mixing, first-order linear mixing drift, auxiliary-condition ablations, and higher-order nonlinear drift boundaries.

Run a single experiment with:

```bash
python main.py --config configs/experiments/exp1_vanilla_failure/exp1_00_fixedmix_sanity.yaml --run run/single --doc fixedmix_sanity --seed 0 --n-sims 1
```

Run the planned suite with:

```bash
python scripts/run_planned_experiments.py --group main --run run/planned_suite --seed 0 --n-sims 5
```

Generate figures with:

```bash
python plot_synthetic_results.py --run-dir run/planned_suite --out-dir run/planned_suite/figures
```
