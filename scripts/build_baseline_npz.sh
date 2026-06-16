#!/usr/bin/env bash
set -euo pipefail
DATASET_ROOT=${DATASET_ROOT:-.}
cd "$(dirname "$0")/../baselines_tcl_ivae"
mkdir -p data_npz
python scripts/build_npz_from_autodl_fif.py   --dataset-root "$DATASET_ROOT"   --derivatives-subdir derivatives   --out data_npz/test_read.npz   --domain-mode subject   --u-source period   --n-periods 20
