#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = yaml.safe_load(f)
    items = obj.get('experiments', [])
    return sorted(items, key=lambda x: int(x.get('order', 0)))

def filter_items(items, group, experiments, docs, start_order, end_order):
    out = []
    exp_set = {e.lower() for e in experiments} if experiments else None
    doc_set = set(docs) if docs else None
    for item in items:
        sec = str(item.get('section', '')).lower()
        exp = str(item.get('experiment', '')).lower()
        doc = str(item.get('doc', ''))
        order = int(item.get('order', 0))
        if group != 'all' and sec != group:
            continue
        if exp_set is not None and exp not in exp_set:
            continue
        if doc_set is not None and doc not in doc_set:
            continue
        if start_order is not None and order < start_order:
            continue
        if end_order is not None and order > end_order:
            continue
        out.append(item)
    return out

def main():
    ap = argparse.ArgumentParser(description='Run the planned iVAE synthetic experiment suite from the manifest.')
    ap.add_argument('--manifest', type=str, default='configs/experiments/experiment_manifest.yaml')
    ap.add_argument('--run', type=str, default='run/planned_suite')
    ap.add_argument('--group', type=str, default='all', choices=['all','main','appendix'])
    ap.add_argument('--experiment', action='append', default=[], help='Specific experiment id, e.g. Exp3. Repeatable.')
    ap.add_argument('--doc', action='append', default=[], help='Specific doc name. Repeatable.')
    ap.add_argument('--start-order', type=int, default=None)
    ap.add_argument('--end-order', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-sims', type=int, default=5)
    ap.add_argument('--python', type=str, default=sys.executable)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--continue-on-error', action='store_true')
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    items = load_manifest(manifest_path)
    selected = filter_items(items, args.group, args.experiment, args.doc, args.start_order, args.end_order)

    if not selected:
        print('No experiment matched the current filters.')
        return 0
    if args.list:
        for item in selected:
            print(f"[{item['order']:>3}] {item['section']:<8} {item['experiment']:<12} {item['method']:<6} {item['doc']} -> {item['config']}")
        return 0

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for item in selected:
        cfg_rel = str(item['config'])
        doc = str(item['doc'])
        cmd = [args.python, 'main.py', '--config', cfg_rel, '--run', str(run_dir), '--doc', doc, '--seed', str(args.seed), '--n-sims', str(args.n_sims)]
        print(f"\n=== [{item['order']}] {item['experiment']} | {item['method']} | {doc} ===")
        print(' '.join(cmd))
        if args.dry_run:
            continue
        ret = subprocess.run(cmd, cwd=str(repo_root))
        if ret.returncode != 0:
            failures.append((doc, ret.returncode))
            if not args.continue_on_error:
                print(f'Aborting because {doc} failed with code {ret.returncode}.')
                return ret.returncode
    if failures:
        print('\nCompleted with failures:')
        for doc, code in failures:
            print(f'  - {doc}: exit code {code}')
        return 1
    print('\nAll selected experiments finished successfully.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
