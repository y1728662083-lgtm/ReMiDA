#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../synthetic_benchmarks"
python scripts/run_planned_experiments.py --group main --run run/planned_suite --seed 0 --n-sims 5
python plot_synthetic_results.py --run-dir run/planned_suite --out-dir run/planned_suite/figures
