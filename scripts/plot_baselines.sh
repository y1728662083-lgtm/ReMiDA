#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper_figures"
python plot_baseline_summary.py --csv ../results/baseline_fixed_sub01/overall_mean_std.csv --out-dir ../results/baseline_fixed_sub01/figures
