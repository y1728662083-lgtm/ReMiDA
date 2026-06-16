#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge linear drift-sweep summaries into a single master CSV/MD + plots.

Inputs (default):
  - results/tables/summary_lin_base_v2.txt
  - results/tables/summary_lin_anchoronly01_v2.txt
  - results/tables/summary_lin_adapter_tether_delta_anchor01_signfix_v2.txt

Outputs:
  - results/tables/lin_sweep_master.csv
  - results/tables/lin_sweep_full_mcc.md
  - results/tables/plot_lin_sweep_full_mcc.png
  - results/tables/plot_lin_sweep_perm_agree.png
  - results/tables/plot_lin_sweep_sign_agree.png
  - results/tables/plot_lin_sweep_probe_acc.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DS2DRIFT = {
    "ds0p0": 0.00,
    "ds0p05": 0.05,
    "ds0p1": 0.10,
    "ds0p2": 0.20,
    "ds0p4": 0.40,
}

DS_RE = re.compile(r"(ds0p05|ds0p0|ds0p1|ds0p2|ds0p4)")
HEADER_RE = re.compile(r"=+\s*(.*?)\s*=+")
PAIR_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_]+_mean,std)\s*\(\s*(?P<mean>[-+0-9.eE]+)\s*,\s*(?P<std>[-+0-9.eE]+)\s*\)"
)
NSEEDS_RE = re.compile(r"\bn_seeds\s+(?P<n>\d+)\b")

def extract_ds_token(s: str) -> Optional[str]:
    m = DS_RE.search(s)
    return m.group(1) if m else None

def safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def parse_summary_file(path: Path, method: str) -> List[Dict]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    records: List[Dict] = []
    cur: Optional[Dict] = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        # require FULL_MCC
        if cur.get("FULL_MCC_mean") is None:
            cur = None
            return
        records.append(cur)
        cur = None

    for line in lines:
        h = HEADER_RE.match(line.strip())
        if h:
            flush()
            logname = h.group(1)
            ds = extract_ds_token(logname)
            if ds is None:
                # header that doesn't include ds token; ignore this block
                cur = None
                continue
            cur = {
                "method": method,
                "ds_tag": ds,
                "drift_strength": DS2DRIFT[ds],
                "log_ref": logname,
                "FULL_MCC_mean": None, "FULL_MCC_std": None,
                "perm_agree_mean": None, "perm_agree_std": None,
                "sign_agree_mean": None, "sign_agree_std": None,
                "probe_acc_mean": None, "probe_acc_std": None,
                "n_seeds": None,
            }
            continue

        if cur is None:
            continue

        m_pair = PAIR_RE.search(line)
        if m_pair:
            name = m_pair.group("name")  # e.g., FULL_MCC_mean,std
            mean = safe_float(m_pair.group("mean"))
            std = safe_float(m_pair.group("std"))
            base = name.replace("_mean,std", "")
            cur[f"{base}_mean"] = mean
            cur[f"{base}_std"] = std
            continue

        m_ns = NSEEDS_RE.search(line)
        if m_ns:
            cur["n_seeds"] = int(m_ns.group("n"))
            continue

    flush()
    return records

def fmt_pm(mean: Optional[float], std: Optional[float], nd: int = 4) -> str:
    if mean is None or std is None:
        return ""
    return f"{mean:.{nd}f} ± {std:.{nd}f}"

def write_master_csv(records: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # stable column order
    fieldnames = [
        "method", "drift_strength", "ds_tag", "log_ref",
        "FULL_MCC_mean", "FULL_MCC_std",
        "perm_agree_mean", "perm_agree_std",
        "sign_agree_mean", "sign_agree_std",
        "probe_acc_mean", "probe_acc_std",
        "n_seeds",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["method"], x["drift_strength"])):
            w.writerow({k: r.get(k, "") for k in fieldnames})

def write_full_mcc_md(records: List[Dict], out_md: Path, method_order: List[str]) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    # drift -> method -> "mean ± std"
    table: Dict[float, Dict[str, str]] = {}
    for r in records:
        d = float(r["drift_strength"])
        table.setdefault(d, {})
        table[d][r["method"]] = fmt_pm(r.get("FULL_MCC_mean"), r.get("FULL_MCC_std"), nd=4)

    drifts = sorted(table.keys())
    cols = method_order

    lines = []
    lines.append("| drift_strength | " + " | ".join(cols) + " |")
    lines.append("|---:|" + "|".join(["---:"] * len(cols)) + "|")
    for d in drifts:
        row = [f"{d:.2f}"]
        for m in cols:
            row.append(table[d].get(m, ""))
        lines.append("| " + " | ".join(row) + " |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

def try_make_plots(records: List[Dict], outdir: Path, method_order: List[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available, skip plots: {e}")
        return

    def plot_metric(mean_key: str, std_key: str, ylabel: str, fname: str):
        plt.figure()
        for method in method_order:
            rows = [r for r in records if r["method"] == method]
            rows = sorted(rows, key=lambda x: x["drift_strength"])
            x = [r["drift_strength"] for r in rows]
            y = [r.get(mean_key) for r in rows]
            yerr = [r.get(std_key) for r in rows]
            # drop None rows
            xx, yy, ee = [], [], []
            for a, b, c in zip(x, y, yerr):
                if b is None or c is None:
                    continue
                xx.append(a); yy.append(b); ee.append(c)
            if not xx:
                continue
            plt.errorbar(xx, yy, yerr=ee, marker="o", linestyle="-", label=method)

        plt.xlabel("drift_strength")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs drift_strength")
        plt.legend()
        plt.tight_layout()
        outpath = outdir / fname
        plt.savefig(outpath, dpi=200)
        plt.close()

    outdir.mkdir(parents=True, exist_ok=True)
    plot_metric("FULL_MCC_mean", "FULL_MCC_std", "FULL_MCC", "plot_lin_sweep_full_mcc.png")
    plot_metric("perm_agree_mean", "perm_agree_std", "perm_agree", "plot_lin_sweep_perm_agree.png")
    plot_metric("sign_agree_mean", "sign_agree_std", "sign_agree", "plot_lin_sweep_sign_agree.png")
    plot_metric("probe_acc_mean", "probe_acc_std", "probe_acc", "plot_lin_sweep_probe_acc.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/tables")
    ap.add_argument("--base", type=str, default="results/tables/summary_lin_base_v2.txt")
    ap.add_argument("--anchor", type=str, default="results/tables/summary_lin_anchoronly01_v2.txt")
    ap.add_argument("--adapter", type=str, default="results/tables/summary_lin_adapter_tether_delta_anchor01_signfix_v2.txt")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    in_base = Path(args.base)
    in_anchor = Path(args.anchor)
    in_adapter = Path(args.adapter)

    method_order = [
        "lin_base",
        "lin_anchoronly01",
        "lin_adapter_tether_delta_anchor01_signfix",
    ]

    if not in_base.exists():
        raise FileNotFoundError(in_base)
    if not in_anchor.exists():
        raise FileNotFoundError(in_anchor)
    if not in_adapter.exists():
        raise FileNotFoundError(in_adapter)

    records: List[Dict] = []
    records += parse_summary_file(in_base, method="lin_base")
    records += parse_summary_file(in_anchor, method="lin_anchoronly01")
    records += parse_summary_file(in_adapter, method="lin_adapter_tether_delta_anchor01_signfix")

    out_csv = outdir / "lin_sweep_master.csv"
    out_md = outdir / "lin_sweep_full_mcc.md"

    write_master_csv(records, out_csv)
    write_full_mcc_md(records, out_md, method_order=method_order)
    try_make_plots(records, outdir, method_order=method_order)

    print("[ok] wrote:")
    print("  ", out_csv)
    print("  ", out_md)
    print("  ", outdir / "plot_lin_sweep_full_mcc.png")
    print("  ", outdir / "plot_lin_sweep_perm_agree.png")
    print("  ", outdir / "plot_lin_sweep_sign_agree.png")
    print("  ", outdir / "plot_lin_sweep_probe_acc.png")

if __name__ == "__main__":
    main()
