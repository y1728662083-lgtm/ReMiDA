from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _aggregate_frames(frames: List[pd.DataFrame], key_cols: Iterable[str]) -> pd.DataFrame:
    valid_frames = [f for f in frames if f is not None and not f.empty]
    if not valid_frames:
        return pd.DataFrame()
    df = pd.concat(valid_frames, axis=0, ignore_index=True)
    key_cols = [c for c in list(key_cols) if c in df.columns]
    key_set = set(key_cols)
    numeric_cols = [c for c in df.columns if c not in key_set and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df[key_cols].drop_duplicates().reset_index(drop=True) if key_cols else pd.DataFrame(index=[0])
    agg = df.groupby(key_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    flat_cols = []
    for col in agg.columns.to_flat_index():
        if isinstance(col, tuple):
            flat_cols.append("__".join([str(x) for x in col if x]))
        else:
            flat_cols.append(str(col))
    agg.columns = flat_cols
    return agg.reset_index(drop=True)


def rebuild(run_dir: Path) -> None:
    top_rows = []
    for protocol_dir in [run_dir / "cross_subject", run_dir / "cross_session"]:
        if not protocol_dir.exists():
            continue
        linear_rows = []
        eegnet_frames = []
        for child in sorted(protocol_dir.iterdir()):
            if not child.is_dir():
                continue
            metrics_test = child / "metrics_test.json"
            eeg_cmp = child / "eegnet_suite" / "comparison.csv"
            if metrics_test.exists():
                data = json.loads(metrics_test.read_text(encoding="utf-8"))
                linear_rows.append({"run_name": child.name, "protocol": protocol_dir.name, **data})
            if eeg_cmp.exists():
                df = pd.read_csv(eeg_cmp)
                df.insert(0, "run_name", child.name)
                df.insert(1, "protocol", protocol_dir.name)
                eegnet_frames.append(df)
        linear_df = pd.DataFrame(linear_rows)
        linear_df.to_csv(protocol_dir / "linear_metrics_per_run.csv", index=False)
        if not linear_df.empty:
            linear_summary = _aggregate_frames([linear_df], key_cols=["protocol"])
            linear_summary.to_csv(protocol_dir / "linear_metrics_summary.csv", index=False)
        eegnet_all = pd.concat(eegnet_frames, axis=0, ignore_index=True) if eegnet_frames else pd.DataFrame()
        eegnet_all.to_csv(protocol_dir / "eegnet_metrics_per_run.csv", index=False)
        if not eegnet_all.empty:
            eegnet_summary = _aggregate_frames([eegnet_all], key_cols=["protocol", "mode"])
            eegnet_summary.to_csv(protocol_dir / "eegnet_metrics_summary.csv", index=False)
            with (protocol_dir / "eegnet_metrics_summary.tex").open("w", encoding="utf-8") as f:
                f.write(eegnet_summary.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
            x = eegnet_summary.copy()
            x.insert(0, "protocol_name", protocol_dir.name)
            top_rows.append(x)
    if top_rows:
        all_summary = pd.concat(top_rows, axis=0, ignore_index=True)
        all_summary.to_csv(run_dir / "all_protocols_eegnet_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to outputs/sub01_sub02_cross_sub_ses_linear")
    args = ap.parse_args()
    rebuild(Path(args.run_dir))


if __name__ == "__main__":
    main()
