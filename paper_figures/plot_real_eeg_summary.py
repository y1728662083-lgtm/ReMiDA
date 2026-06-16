#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _find_metric_col(df, base):
    for c in df.columns:
        if c == base or c == base + '__mean' or c.lower() == base.lower() + '__mean':
            return c
    candidates = [c for c in df.columns if base.lower() in c.lower() and 'mean' in c.lower()]
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser(description='Plot EEGNet four-input summary from real_eeg_remida outputs.')
    ap.add_argument('--csv', required=True, help='Path to eegnet_metrics_summary.csv')
    ap.add_argument('--out-dir', default='figures_real_eeg')
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_col = 'mode' if 'mode' in df.columns else 'input_mode'
    if mode_col not in df.columns:
        raise SystemExit('Cannot find mode/input_mode column.')
    metrics = ['val_acc','val_f1','test_acc','test_f1']
    for base in metrics:
        col = _find_metric_col(df, base)
        if col is None:
            continue
        sdf = df[[mode_col, col]].dropna().copy()
        sdf[col] = sdf[col].astype(float)
        if sdf[col].max() <= 1.0:
            sdf[col] *= 100.0
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.bar(sdf[mode_col].astype(str), sdf[col])
        ax.set_ylabel(base.replace('_', '-').upper() + ' (%)')
        ax.set_title('Real EEG input-representation comparison')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        fig.tight_layout()
        out = out_dir / f'{base}.png'
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix('.pdf'))
        plt.close(fig)
    print(f'[saved] {out_dir}')

if __name__ == '__main__':
    main()
