#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../baselines_tcl_ivae"
DATA=${DATA:-data_npz/test_read.npz}
mkdir -p runs_fixed_sub01
for TGT in sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10; do
  echo "[iVAE] sub-01 -> ${TGT}"
  python scripts/train_ivae_baseline_paper.py     --data "$DATA"     --out "runs_fixed_sub01/ivae_sub-01_to_${TGT}_paper"     --ref-domain sub-01     --target-domain "$TGT"     --latent-dim 32     --hidden-dim 128     --epochs 120     --batch-size 128     --seed 42     --pca-dim 256     --lr 1e-4     --decoder-var 1.0     --clamp-logvar 6.0     --downstream-inputs all
  echo "[TCL] sub-01 -> ${TGT}"
  python scripts/train_tcl_pytorch_baseline_paper.py     --data "$DATA"     --out "runs_fixed_sub01/tcl_sub-01_to_${TGT}_paper"     --ref-domain sub-01     --target-domain "$TGT"     --split-file "runs_fixed_sub01/ivae_sub-01_to_${TGT}_paper/split_indices.npz"     --latent-dim 32     --hidden-dim 256     --epochs 120     --batch-size 128     --seed 42     --pca-dim 256     --downstream-inputs all
done
python scripts/aggregate_results.py --runs-root runs_fixed_sub01 --out runs_fixed_sub01/aggregate_summary
