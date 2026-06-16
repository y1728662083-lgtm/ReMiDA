# Real EEG ReMiDA-CLSRNet experiments

This folder contains the real EEG experiment code for ReMiDA-CLSRNet. It reads Inner Speech derivative files directly from `.fif`, `events.dat`, and `report.pkl`, then evaluates EEGNet on four input representations:

- `raw`: raw-preprocessed observations (`r1`).
- `aligned_raw`: ReMiDA-aligned observations (`a1`).
- `linear_latent`: latent representations (`l1`).
- `raw_plus_linear_latent`: raw-preprocessed observations concatenated with latent representations (`rL`).

## Fixed-reference cross-subject protocol

Use `configs/fixed_sub01_all_subjects.yaml` when `sub-01` is fixed as the reference domain and each remaining subject is aligned to it.

```bash
python run_cross_subject.py \
  --config configs/fixed_sub01_all_subjects.yaml \
  --dataset-root /path/to/dataset/root \
  --device cuda
```

Expected data layout:

```text
/path/to/dataset/root/derivatives/sub-XX/ses-YY/*_eeg-epo.fif
/path/to/dataset/root/derivatives/sub-XX/ses-YY/*_events.dat
/path/to/dataset/root/derivatives/sub-XX/ses-YY/*_report.pkl
```

Rebuild summary tables after training:

```bash
python rebuild_protocol_summaries.py --run-dir outputs/fixed_sub01_all_subjects_linear
```

Main summary files:

```text
outputs/fixed_sub01_all_subjects_linear/cross_subject/eegnet_metrics_per_run.csv
outputs/fixed_sub01_all_subjects_linear/cross_subject/eegnet_metrics_summary.csv
```
