# Code organization and provenance

## Main components

- `synthetic_benchmarks/`: synthetic source-recovery experiments, including the iVAE implementation, drift benchmarks, experiment manifests, metric computation, and synthetic figure generation.
- `real_eeg_remida/`: real EEG ReMiDA-CLSRNet experiments and EEGNet evaluation code.
- `baselines_tcl_ivae/`: TCL and iVAE baseline runners aligned with the fixed-reference real EEG protocol.
- `third_party/TCL-master/`: original TensorFlow TCL source retained for reference and provenance.

## Cleaning policy

The repository excludes:

- Python cache files and notebook checkpoints.
- Trained checkpoints and generated latent feature arrays.
- Raw EEG data and large exported `.fif` files.
- Temporary local run folders.

The excluded artifacts can be regenerated from the provided scripts and a local copy of the dataset derivatives.

## Baseline code reuse

The iVAE baseline runner imports the iVAE network implementation from `synthetic_benchmarks/models/nets.py`. The TCL comparison used in the submitted baseline run is implemented in PyTorch in `baselines_tcl_ivae/scripts/train_tcl_pytorch_baseline_paper.py`; the original TensorFlow TCL code is kept under `third_party/TCL-master/` for reference.
