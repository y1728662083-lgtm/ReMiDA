import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description='Run one baseline for many reference-target domain pairs.')
    p.add_argument('--data', required=True)
    p.add_argument('--method', required=True, choices=['ivae', 'tcl_pytorch'])
    p.add_argument('--runs-root', default='runs')
    p.add_argument('--refs', nargs='*', default=None, help='Reference domains. Default: all domains.')
    p.add_argument('--targets', nargs='*', default=None, help='Target domains. Default: all domains except current ref.')
    p.add_argument('--within-subject', action='store_true', help='For domains like sub-01::ses-01, only run pairs with the same subject prefix.')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--pca-dim', type=int, default=256)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--downstream-inputs', nargs='+', default=['all'], choices=['all','latent','raw_pca','raw_plus_latent'])
    p.add_argument('--max-pairs', type=int, default=0, help='Debug only. 0 means all pairs.')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def subject_prefix(domain: str) -> str:
    return str(domain).split('::')[0]


def main():
    args = parse_args()
    z = np.load(args.data, allow_pickle=True)
    domains = sorted(set(z['domain'].astype(str)))
    refs = args.refs if args.refs else domains
    targets = args.targets if args.targets else domains
    pairs = []
    for ref in refs:
        for tgt in targets:
            if ref == tgt:
                continue
            if args.within_subject and subject_prefix(ref) != subject_prefix(tgt):
                continue
            pairs.append((ref, tgt))
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    print(f'[pairs] n={len(pairs)}')
    for i, (ref, tgt) in enumerate(pairs, 1):
        safe_ref = ref.replace('::', '_')
        safe_tgt = tgt.replace('::', '_')
        if args.method == 'ivae':
            script = ROOT / 'scripts' / 'train_ivae_baseline_paper.py'
            out = Path(args.runs_root) / f'ivae_{safe_ref}_to_{safe_tgt}_paper'
            cmd = [
                sys.executable, str(script), '--data', args.data, '--out', str(out),
                '--ref-domain', ref, '--target-domain', tgt, '--latent-dim', str(args.latent_dim),
                '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--seed', str(args.seed),
                '--pca-dim', str(args.pca_dim), '--lr', '1e-4', '--decoder-var', '1.0', '--clamp-logvar', '6.0',
                '--downstream-inputs', *args.downstream_inputs,
            ]
        else:
            script = ROOT / 'scripts' / 'train_tcl_pytorch_baseline_paper.py'
            out = Path(args.runs_root) / f'tcl_{safe_ref}_to_{safe_tgt}_paper'
            # Reuse corresponding iVAE split if it exists. If not, this script will make a split itself.
            ivae_split = Path(args.runs_root) / f'ivae_{safe_ref}_to_{safe_tgt}_paper' / 'split_indices.npz'
            cmd = [
                sys.executable, str(script), '--data', args.data, '--out', str(out),
                '--ref-domain', ref, '--target-domain', tgt, '--latent-dim', str(args.latent_dim),
                '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--seed', str(args.seed),
                '--pca-dim', str(args.pca_dim), '--downstream-inputs', *args.downstream_inputs,
            ]
            if ivae_split.exists():
                cmd += ['--split-file', str(ivae_split)]
        if args.skip_existing and (out / 'metrics_by_input.json').exists():
            print(f'[{i}/{len(pairs)}] skip existing {out}')
            continue
        print(f'[{i}/{len(pairs)}] {args.method}: {ref} -> {tgt}')
        print(' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    manifest = {'method': args.method, 'n_pairs': len(pairs), 'pairs': pairs, 'data': args.data}
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.runs_root) / f'{args.method}_pairs_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
