# Synthetic benchmark experiments

This folder contains the synthetic benchmark code used for the ReMiDA-CLSRNet source-recovery experiments. It includes the iVAE implementation, controlled drift data generators, experiment YAML files, batch runners, metric summarization, and figure-generation code.

## Main files

- `main.py`: entry point for a single synthetic experiment.
- `configs/experiments/experiment_manifest.yaml`: machine-readable experiment manifest.
- `configs/experiments/EXPERIMENT_MANIFEST.md`: human-readable experiment list.
- `scripts/run_planned_experiments.py`: batch runner for Exp1--Exp6.
- `plot_synthetic_results.py`: aggregates synthetic summaries and generates figures.

## Run experiments

List the planned experiments:

```bash
python scripts/run_planned_experiments.py --list
```

Run the main synthetic experiments:

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
