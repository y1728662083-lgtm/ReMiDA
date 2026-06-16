#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
python scripts/run_planned_experiments.py --group main --run run/planned_suite --seed 0 --n-sims 5
python scripts/run_planned_experiments.py --group appendix --run run/planned_suite --seed 0 --n-sims 5
