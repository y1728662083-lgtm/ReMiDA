# ReMiDA-CLSRNet experiment repository

This repository contains the experiment code used for the manuscript:

**Reference-Domain Mixing-Drift Alignment and Condition-Constrained Latent Source Representation for Cross-Domain EEG Signals**

The repository is organized into three experiment blocks:

1. `synthetic_benchmarks/`: controlled synthetic experiments for fixed-mixing degradation, linear mixing-drift correction, auxiliary-condition ablation, and nonlinear-boundary analysis.
2. `real_eeg_remida/`: real EEG ReMiDA-CLSRNet experiments on the Inner Speech derivatives, including the EEGNet evaluation of four input representations.
3. `baselines_tcl_ivae/`: fixed-reference TCL and iVAE baseline experiments aligned with the real EEG protocol.

Raw EEG files, trained checkpoints, exported `.fif` files, and generated latent feature arrays are not included because of size and data-redistribution constraints. The scripts read the dataset from a user-provided derivatives directory and regenerate all intermediate files.

## Repository layout

```text
ReMiDA_CLSRNet_experiments_git_ready/
├── synthetic_benchmarks/      # synthetic benchmark experiments and synthetic figure scripts
├── real_eeg_remida/           # proposed method on real EEG data
├── baselines_tcl_ivae/        # TCL/iVAE baseline experiments
├── paper_figures/             # scripts for summary/result figures
├── results/                   # submitted result records and summaries
├── scripts/                   # top-level run scripts
├── third_party/               # original TCL source kept for provenance/reference
├── README.md
├── REPRODUCIBILITY.md
├── CODE_ORGANIZATION.md
├── FILE_MANIFEST.txt
├── requirements.txt
└── environment.yml
```

## Environment

A Python 3.10 environment is recommended.

```bash
conda env create -f environment.yml
conda activate remida_repro
```

Alternatively:

```bash
conda create -n remida_repro python=3.10 -y
conda activate remida_repro
pip install -r requirements.txt
```

## Expected real EEG data layout

The real EEG and baseline scripts expect the Inner Speech derivative files in the following structure:

```text
${DATASET_ROOT}/derivatives/sub-XX/ses-YY/*_eeg-epo.fif
${DATASET_ROOT}/derivatives/sub-XX/ses-YY/*_events.dat
${DATASET_ROOT}/derivatives/sub-XX/ses-YY/*_report.pkl
```

Set `DATASET_ROOT` to the directory that contains the `derivatives/` folder before running real EEG or baseline scripts.

## A. Synthetic benchmark experiments

Run the main synthetic suite:

```bash
bash scripts/run_synthetic_main.sh
```

The generated summaries and figures are written to:

```text
synthetic_benchmarks/run/planned_suite/figures/
```

To list the planned synthetic experiments without running them:

```bash
cd synthetic_benchmarks
python scripts/run_planned_experiments.py --list
```

## B. Real EEG ReMiDA-CLSRNet experiments

Run the fixed-reference cross-subject protocol with `sub-01` as the reference domain:

```bash
DATASET_ROOT=/path/to/dataset/root DEVICE=cuda bash scripts/run_real_eeg_fixed_sub01.sh
```

Main outputs:

```text
real_eeg_remida/outputs/fixed_sub01_all_subjects_linear/cross_subject/eegnet_metrics_per_run.csv
real_eeg_remida/outputs/fixed_sub01_all_subjects_linear/cross_subject/eegnet_metrics_summary.csv
```

The four input representations evaluated by EEGNet are:

- `raw`: raw-preprocessed observations (`r1` in the manuscript).
- `aligned_raw`: ReMiDA-aligned observations (`a1`).
- `linear_latent`: latent representations (`l1`).
- `raw_plus_linear_latent`: raw-preprocessed observations concatenated with latent representations (`rL`).

## C. TCL/iVAE baseline experiments

Build the compact baseline `.npz` file from the real EEG derivatives:

```bash
DATASET_ROOT=/path/to/dataset/root bash scripts/build_baseline_npz.sh
```

Run the fixed-reference baseline protocol (`sub-01 -> sub-02`, ..., `sub-01 -> sub-10`):

```bash
bash scripts/run_baselines_fixed_sub01.sh
```

Aggregated results are saved to:

```text
baselines_tcl_ivae/runs_fixed_sub01/aggregate_summary/
```

The submitted baseline result records are also stored in:

```text
results/baseline_fixed_sub01/
```

## D. Figure generation

Generate baseline summary figures from the submitted CSV records:

```bash
bash scripts/plot_baselines.sh
```

The figure files are written to:

```text
results/baseline_fixed_sub01/figures/
```

## Notes on baseline implementation

The real EEG epochs have a high-dimensional channel-time representation. Directly fitting iVAE to vectorized epochs can lead to numerical divergence. For a stable and fair baseline implementation, the iVAE baseline uses standardization fitted only on `reftrain + targetadapt`, PCA fitted only on `reftrain + targetadapt`, a smaller learning rate, fixed decoder variance, log-variance clipping, gradient clipping, and non-finite-loss checking. These operations are baseline stabilization settings and are not part of ReMiDA-CLSRNet.

For TCL, the repository includes the original TensorFlow TCL source under `third_party/TCL-master/` for provenance. The submitted baseline run uses the PyTorch period-discrimination implementation in `baselines_tcl_ivae/scripts/train_tcl_pytorch_baseline_paper.py`, avoiding a TensorFlow-1.x dependency while preserving the period-label discrimination objective used for the comparison.
