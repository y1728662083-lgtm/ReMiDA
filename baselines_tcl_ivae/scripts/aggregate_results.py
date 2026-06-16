import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Aggregate per reference-target baseline results into paper-style means.')
    p.add_argument('--runs-root', default='runs')
    p.add_argument('--pattern', default='*/metrics_by_input.json', help='Glob relative to runs-root.')
    p.add_argument('--out', default='runs/aggregate_summary')
    return p.parse_args()


def mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    return {'mean': float(arr.mean()) if arr.size else None, 'std': float(arr.std()) if arr.size else None, 'n': int(arr.size)}


def main():
    args = parse_args()
    root = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mp in sorted(root.glob(args.pattern)):
        run_dir = mp.parent
        meta_path = run_dir / 'run_meta.json'
        try:
            metrics_by_input = json.loads(mp.read_text())
        except Exception as e:
            print(f'[skip] cannot read {mp}: {e}')
            continue
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        method = meta.get('method', run_dir.name.split('_')[0])
        ref = meta.get('ref_domain', '')
        tgt = meta.get('target_domain', '')
        for input_mode, m in metrics_by_input.items():
            row = {
                'run_dir': str(run_dir),
                'method': method,
                'ref_domain': ref,
                'target_domain': tgt,
                'input_mode': input_mode,
                'val_acc': m.get('val_acc'),
                'val_f1': m.get('val_f1'),
                'test_acc': m.get('test_acc'),
                'test_f1': m.get('test_f1'),
            }
            rows.append(row)
    if not rows:
        raise SystemExit(f'No metrics found under {root} with pattern {args.pattern}')

    csv_path = out / 'per_pair_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    by_ref = defaultdict(list)
    by_tgt = defaultdict(list)
    for r in rows:
        key = (r['method'], r['input_mode'])
        grouped[key].append(r)
        by_ref[(r['method'], r['input_mode'], r['ref_domain'])].append(r)
        by_tgt[(r['method'], r['input_mode'], r['target_domain'])].append(r)

    def summarize_dict(group_dict):
        out = []
        for key, items in sorted(group_dict.items(), key=lambda x: str(x[0])):
            if len(key) == 2:
                method, input_mode = key
                row = {'method': method, 'input_mode': input_mode}
            elif '::REF::' in str(key):
                pass
            else:
                method, input_mode, domain = key
                row = {'method': method, 'input_mode': input_mode, 'domain': domain}
            for metric in ['val_acc', 'val_f1', 'test_acc', 'test_f1']:
                vals = [float(i[metric]) for i in items if i[metric] is not None]
                ms = mean_std(vals)
                row[f'{metric}_mean'] = ms['mean']
                row[f'{metric}_std'] = ms['std']
                row[f'{metric}_n'] = ms['n']
            out.append(row)
        return out

    overall = summarize_dict(grouped)
    ref_summary = summarize_dict(by_ref)
    tgt_summary = summarize_dict(by_tgt)

    def write_csv(path, data):
        if not data:
            return
        keys = sorted({k for row in data for k in row.keys()})
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    write_csv(out / 'overall_mean_std.csv', overall)
    write_csv(out / 'by_reference_domain_mean_std.csv', ref_summary)
    write_csv(out / 'by_target_domain_mean_std.csv', tgt_summary)
    (out / 'overall_mean_std.json').write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'by_reference_domain_mean_std.json').write_text(json.dumps(ref_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'by_target_domain_mean_std.json').write_text(json.dumps(tgt_summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[saved] {csv_path}')
    print(f'[saved] {out / "overall_mean_std.csv"}')
    print(f'[saved] {out / "by_reference_domain_mean_std.csv"}')
    print(f'[saved] {out / "by_target_domain_mean_std.csv"}')


if __name__ == '__main__':
    main()
