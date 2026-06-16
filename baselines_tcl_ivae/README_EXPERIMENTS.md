# TCL and iVAE baselines for the fixed-reference EEG protocol

This folder contains the TCL and iVAE baseline code used for the paper-aligned comparison. It supports the fixed-reference protocol used for the additional real EEG baseline analysis:

```text
reference domain: sub-01
target domains:  sub-02 ... sub-10
reference split: 8:2 -> reftrain/refval
target split:    5:5 -> targetadapt/targettest
```

The baseline comparison reports vector-feature inputs:

- `latent`: iVAE/TCL learned latent representation.
- `raw_pca`: raw EEG after standardization and PCA, used as a vector baseline.
- `raw_plus_latent`: concatenation of `raw_pca` and latent features.

These inputs are not identical to the proposed model's `r1/a1/l1/rL` EEGNet inputs. They are nonlinear-ICA representation baselines evaluated with a shared MLP classifier.

## 1. Build `.npz` from `.fif` derivatives

```bash
cd baselines_tcl_ivae
python scripts/build_npz_from_autodl_fif.py \
  --dataset-root /path/to/dataset/root \
  --derivatives-subdir derivatives \
  --out data_npz/test_read.npz \
  --domain-mode subject \
  --u-source period \
  --n-periods 20
```

## 2. Run fixed-sub-01 baselines

```bash
cd ..
bash scripts/run_baselines_fixed_sub01.sh
```

## 3. Run one pair manually

```bash
python scripts/train_ivae_baseline_paper.py \
  --data data_npz/test_read.npz \
  --out runs_fixed_sub01/ivae_sub-01_to_sub-02_paper \
  --ref-domain sub-01 \
  --target-domain sub-02 \
  --latent-dim 32 \
  --hidden-dim 128 \
  --epochs 120 \
  --batch-size 128 \
  --seed 42 \
  --pca-dim 256 \
  --lr 1e-4 \
  --decoder-var 1.0 \
  --clamp-logvar 6.0 \
  --downstream-inputs all
```

Aggregate existing runs:

```bash
python scripts/aggregate_results.py \
  --runs-root runs_fixed_sub01 \
  --out runs_fixed_sub01/aggregate_summary
```
