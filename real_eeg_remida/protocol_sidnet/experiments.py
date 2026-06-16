from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import ProtocolCollection, ProtocolRunBundle, build_protocol_collection
from .eegnet_experiment import RepresentationStore, run_eegnet_suite
from .linear_trainer import LinearTrainArtifacts, train_linear_protocol_run
from .utils import ensure_dir, write_json


@dataclass
class RunResult:
    bundle: ProtocolRunBundle
    linear: LinearTrainArtifacts
    eegnet_comparison: pd.DataFrame


@dataclass
class ExperimentResultCollection:
    protocol_name: str
    run_results: List[RunResult]
    summary_dir: Path



def _make_representation_store(artifacts: LinearTrainArtifacts) -> RepresentationStore:
    bundle = artifacts.bundle
    return RepresentationStore(
        raw=bundle.x_raw.astype(np.float32),
        aligned_raw=artifacts.aligned_all.astype(np.float32),
        y=bundle.y.astype(np.int64),
        train_idx=bundle.train_idx.astype(np.int64),
        val_idx=bundle.val_idx.astype(np.int64),
        test_idx=bundle.test_idx.astype(np.int64),
        linear_latent=artifacts.latent_all.astype(np.float32),
    )



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



def run_protocol(cfg: ExperimentConfig, protocol_name: str, bundles: List[ProtocolRunBundle]) -> ExperimentResultCollection:
    summary_dir = ensure_dir(cfg.run_dir / protocol_name)
    run_results: List[RunResult] = []
    linear_rows = []
    eegnet_rows = []
    for bundle in bundles:
        print(f"[pipeline] running {bundle.run_name}")
        artifacts = train_linear_protocol_run(cfg, bundle)
        store = _make_representation_store(artifacts)
        eegnet_dir = artifacts.run_dir / "eegnet_suite"
        eegnet_df = run_eegnet_suite(cfg, store, eegnet_dir)
        run_results.append(RunResult(bundle=bundle, linear=artifacts, eegnet_comparison=eegnet_df))

        linear_rows.append({
            "run_name": bundle.run_name,
            "protocol": bundle.protocol_name,
            "reference_entity": bundle.reference_entity,
            **artifacts.metrics_test,
        })
        tmp = eegnet_df.copy()
        tmp.insert(0, "run_name", bundle.run_name)
        tmp.insert(1, "protocol", bundle.protocol_name)
        eegnet_rows.append(tmp)

    linear_df = pd.DataFrame(linear_rows)
    linear_df.to_csv(summary_dir / "linear_metrics_per_run.csv", index=False)
    if not linear_df.empty:
        linear_summary = _aggregate_frames([linear_df], key_cols=["protocol"])
        linear_summary.to_csv(summary_dir / "linear_metrics_summary.csv", index=False)
    eegnet_all = pd.concat(eegnet_rows, axis=0, ignore_index=True) if eegnet_rows else pd.DataFrame()
    eegnet_all.to_csv(summary_dir / "eegnet_metrics_per_run.csv", index=False)
    if not eegnet_all.empty:
        eegnet_summary = _aggregate_frames([eegnet_all], key_cols=["protocol", "mode"])
        eegnet_summary.to_csv(summary_dir / "eegnet_metrics_summary.csv", index=False)
        with (summary_dir / "eegnet_metrics_summary.tex").open("w", encoding="utf-8") as f:
            f.write(eegnet_summary.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))

    write_json(
        {
            "protocol": protocol_name,
            "n_runs": len(run_results),
            "run_names": [r.bundle.run_name for r in run_results],
        },
        summary_dir / "run_summary.json",
    )
    return ExperimentResultCollection(protocol_name=protocol_name, run_results=run_results, summary_dir=summary_dir)



def run_all_protocols(cfg: ExperimentConfig) -> Dict[str, ExperimentResultCollection]:
    collection = build_protocol_collection(cfg)
    outputs: Dict[str, ExperimentResultCollection] = {}
    if collection.cross_subject_runs:
        outputs["cross_subject"] = run_protocol(cfg, "cross_subject", collection.cross_subject_runs)
    if collection.cross_session_runs:
        outputs["cross_session"] = run_protocol(cfg, "cross_session", collection.cross_session_runs)
    top_summary_dir = ensure_dir(cfg.run_dir)
    rows = []
    for protocol_name, result in outputs.items():
        summ_path = result.summary_dir / "eegnet_metrics_summary.csv"
        if summ_path.exists():
            df = pd.read_csv(summ_path)
            df.insert(0, "protocol_name", protocol_name)
            rows.append(df)
    if rows:
        all_summary = pd.concat(rows, axis=0, ignore_index=True)
        all_summary.to_csv(top_summary_dir / "all_protocols_eegnet_summary.csv", index=False)
    return outputs
