# TCL and iVAE baselines

This folder contains the fixed-reference baseline experiments used in the manuscript.

## Input data

Create the baseline `.npz` from the real EEG derivative files:

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

The generated `.npz` contains:

- `X`: EEG epochs, flattened internally when needed.
- `y`: class labels.
- `domain`: subject-level domain labels.
- `u`: shared-period labels.

## Single reference-target run

### iVAE

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

### TCL

```bash
python scripts/train_tcl_pytorch_baseline_paper.py \
  --data data_npz/test_read.npz \
  --out runs_fixed_sub01/tcl_sub-01_to_sub-02_paper \
  --ref-domain sub-01 \
  --target-domain sub-02 \
  --split-file runs_fixed_sub01/ivae_sub-01_to_sub-02_paper/split_indices.npz \
  --latent-dim 32 \
  --hidden-dim 256 \
  --epochs 120 \
  --batch-size 128 \
  --seed 42 \
  --pca-dim 256 \
  --downstream-inputs all
```

The `--split-file` option ensures that TCL and iVAE use the same `reftrain`, `refval`, `targetadapt`, and `targettest` indices.

## Fixed sub-01 protocol

Run all nine target subjects:

```bash
cd ..
bash scripts/run_baselines_fixed_sub01.sh
```

The protocol evaluates:

```text
sub-01 -> sub-02
sub-01 -> sub-03
...
sub-01 -> sub-10
```

## Output files

Each run folder contains:

- `run_meta.json`: reference/target domains and split sizes.
- `split_indices.npz`: exact sample indices for each split.
- `metrics_by_input.json`: metrics for `latent`, `raw_pca`, and `raw_plus_latent`.
- `features_*.npz`: learned baseline features and split metadata.

The aggregate folder contains:

- `per_pair_metrics.csv`
- `overall_mean_std.csv`
- `by_reference_domain_mean_std.csv`
- `by_target_domain_mean_std.csv`
