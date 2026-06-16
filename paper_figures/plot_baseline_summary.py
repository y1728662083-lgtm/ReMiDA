#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ['val_acc_mean', 'val_f1_mean', 'test_acc_mean', 'test_f1_mean']

def main():
    ap = argparse.ArgumentParser(description='Plot fixed-reference TCL/iVAE baseline summary.')
    ap.add_argument('--csv', default='../results/baseline_fixed_sub01/overall_mean_std.csv')
    ap.add_argument('--out-dir', default='../results/baseline_fixed_sub01/figures')
    args = ap.parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['label'] = df['method'].astype(str) + ' / ' + df['input_mode'].astype(str)
    # percentage scale if values are fractions
    plot_df = df.copy()
    for m in METRICS:
        plot_df[m] = plot_df[m].astype(float) * 100.0
    for metric in METRICS:
        sdf = plot_df.sort_values(metric)
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        ax.barh(sdf['label'], sdf[metric])
        ax.set_xlabel(metric.replace('_mean', '').replace('_', '-').upper() + ' (%)')
        ax.set_title('Fixed sub-01 reference baseline summary')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        fig.tight_layout()
        out = out_dir / f'{metric}.png'
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix('.pdf'))
        plt.close(fig)
    print(f'[saved] {out_dir}')

if __name__ == '__main__':
    main()
