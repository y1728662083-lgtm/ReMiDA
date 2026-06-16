#!/usr/bin/env bash
set -euo pipefail
DATASET_ROOT=${DATASET_ROOT:-.}
DEVICE=${DEVICE:-cuda}
cd "$(dirname "$0")/../real_eeg_remida"
python run_cross_subject.py --config configs/fixed_sub01_all_subjects.yaml --dataset-root "$DATASET_ROOT" --device "$DEVICE"
python rebuild_protocol_summaries.py --run-dir outputs/fixed_sub01_all_subjects_linear
