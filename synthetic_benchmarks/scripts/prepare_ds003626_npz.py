#!/usr/bin/env python
"""Prepare OpenNeuro ds003626 (Inner Speech) derivatives into a single .npz.

This script converts per-trial epoched EEG (MNE .fif) + events (.dat) into a flat dataset
compatible with iVAE code in this repo.

It follows the dataset structure described by the Inner_Speech_Dataset repo:
  derivatives/sub-XX/ses-0Y/sub-XX_ses-0Y_eeg-epo.fif
  derivatives/sub-XX/ses-0Y/sub-XX_ses-0Y_events.dat

References:
  - Inner_Speech_Dataset README (dataset structure + derivatives filenames)
  - data_processing.filter_by_condition (condition code mapping: pron=0, inner=1, vis=2)

Output (.npz):
  X        (N, D) float32   flattened timepoints
  U        (N, K) float32   one-hot direction label (aux variable)
  y        (N,)   int64     direction id
  z        (N,)   int64     domain/session id
  trial_id (N,)   int64     trial index (global across domains)
  meta_json         json     domain_map etc.

Note:
  - We intentionally keep this script separate from training so MNE is only needed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_list_arg(v: str) -> List[str]:
    # Accept "01,02,03" or "01" etc.
    v = v.strip()
    if not v:
        return []
    if "," in v:
        return [x.strip() for x in v.split(",") if x.strip()]
    return [v]


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels).astype(int).reshape(-1)
    out = np.zeros((labels.size, int(num_classes)), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def _load_epochs_fif(fif_path: Path) -> np.ndarray:
    try:
        import mne
    except Exception as e:
        raise RuntimeError(
            "MNE is required to read .fif epochs. Install with: pip install mne\n"
            f"Original import error: {e}"
        )

    epochs = mne.read_epochs(str(fif_path), verbose="ERROR")
    # epochs.get_data() is the public API; _data also works but is private.
    return epochs.get_data()  # (n_trials, n_channels, n_times)


def _load_events_dat(events_path: Path) -> np.ndarray:
    """Load events .dat which is stored as a pickled pandas object in this dataset."""
    try:
        obj = pd.read_pickle(events_path)
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy()
        return np.asarray(obj)
    except Exception:
        # Fallbacks (in case the file is stored differently)
        try:
            return np.load(events_path, allow_pickle=True)
        except Exception:
            return np.loadtxt(events_path)


def _condition_code(condition: str) -> Optional[int]:
    c = condition.strip().lower()
    if c in {"all", "any"}:
        return None
    if c in {"pron", "pronounced"}:
        return 0
    if c in {"in", "inner", "innerspeech", "inner_speech"}:
        return 1
    if c in {"vis", "visualized", "visualised"}:
        return 2
    raise ValueError(f"Unknown condition={condition}. Use inner|pronounced|visualized|all")


def _map_labels_to_0k(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Map arbitrary integer labels to 0..K-1."""
    y = np.asarray(y).astype(int).reshape(-1)
    uniq = np.unique(y)
    mapping = {int(v): int(i) for i, v in enumerate(sorted([int(x) for x in uniq]))}
    y2 = np.vectorize(lambda t: mapping[int(t)])(y)
    return y2.astype(np.int64), mapping


def build_npz(
    root_dir: Path,
    out_path: Path,
    subjects: List[str],
    sessions: List[str],
    condition: str,
    t_start: float,
    t_end: float,
    fs: int,
    time_stride: int,
    pick_channels: Optional[int],
    max_trials_per_domain: Optional[int],
    pca_dim: Optional[int],
    max_pca_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    cond_code = _condition_code(condition)

    # Build (subject, session) domain list
    domains: List[Tuple[str, str]] = []
    for sub in subjects:
        for ses in sessions:
            domains.append((sub, ses))

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    z_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []

    domain_map = []
    trial_offset = 0

    for dom_id, (sub, ses) in enumerate(domains):
        sub = str(sub)
        ses = str(ses)
        # allow both "1" and "01"
        sub_fmt = sub if len(sub) == 2 else f"{int(sub):02d}"
        ses_fmt = ses if len(ses) == 2 else f"{int(ses):02d}"

        base = root_dir / "derivatives" / f"sub-{sub_fmt}" / f"ses-{ses_fmt}"
        fif_path = base / f"sub-{sub_fmt}_ses-{ses_fmt}_eeg-epo.fif"
        events_path = base / f"sub-{sub_fmt}_ses-{ses_fmt}_events.dat"

        if not fif_path.exists():
            raise FileNotFoundError(f"Missing epochs file: {fif_path}")
        if not events_path.exists():
            raise FileNotFoundError(f"Missing events file: {events_path}")

        X_trials = _load_epochs_fif(fif_path)  # (T, C, L)
        Y = _load_events_dat(events_path)      # (T, >=3)

        if Y.ndim != 2 or Y.shape[0] != X_trials.shape[0]:
            raise RuntimeError(
                f"Events shape {Y.shape} doesn't match epochs trials {X_trials.shape[0]} for {sub_fmt}/{ses_fmt}"
            )

        # Condition is at column 2 in the official processing code
        if cond_code is not None:
            mask = (Y[:, 2].astype(int) == int(cond_code))
            X_trials = X_trials[mask]
            Y = Y[mask]

        # Class label is at column 1 in the official processing code
        y_trials = Y[:, 1].astype(int)

        # Optional cap per domain
        if max_trials_per_domain is not None and X_trials.shape[0] > max_trials_per_domain:
            keep = rng.choice(X_trials.shape[0], size=int(max_trials_per_domain), replace=False)
            keep = np.sort(keep)
            X_trials = X_trials[keep]
            y_trials = y_trials[keep]

        # Crop time window
        start = max(int(round(float(t_start) * fs)), 0)
        end = min(int(round(float(t_end) * fs)), X_trials.shape[2])
        if end <= start:
            raise ValueError(f"Invalid time window: start={start}, end={end}, n_times={X_trials.shape[2]}")
        X_trials = X_trials[:, :, start:end]

        # Optional channel pick
        if pick_channels is not None:
            X_trials = X_trials[:, : int(pick_channels), :]

        # Optional time stride (subsample timepoints)
        if int(time_stride) > 1:
            X_trials = X_trials[:, :, :: int(time_stride)]

        n_trials, n_ch, n_t = X_trials.shape

        # Flatten timepoints into samples: (n_trials*n_t, n_ch)
        X = np.transpose(X_trials, (0, 2, 1)).reshape(n_trials * n_t, n_ch).astype(np.float32)
        y = np.repeat(y_trials, n_t).astype(np.int64)
        z = np.full((X.shape[0],), dom_id, dtype=np.int64)
        trial_id = np.repeat(np.arange(trial_offset, trial_offset + n_trials, dtype=np.int64), n_t)
        trial_offset += n_trials

        X_all.append(X)
        y_all.append(y)
        z_all.append(z)
        trial_all.append(trial_id)

        domain_map.append({"domain_id": dom_id, "subject": f"sub-{sub_fmt}", "session": f"ses-{ses_fmt}"})
        print(f"Loaded domain {dom_id}: sub-{sub_fmt} ses-{ses_fmt} -> trials={n_trials}, samples={X.shape[0]}, D={n_ch}")

    X = np.concatenate(X_all, axis=0)
    y_raw = np.concatenate(y_all, axis=0)
    z = np.concatenate(z_all, axis=0)
    trial_id = np.concatenate(trial_all, axis=0)

    # Map y to 0..K-1 and build one-hot U
    y, y_map = _map_labels_to_0k(y_raw)
    K = int(np.max(y) + 1)
    U = _one_hot(y, K)

    # Optional PCA (fit on subset, transform all)
    if pca_dim is not None:
        from sklearn.decomposition import IncrementalPCA

        pca_dim = int(pca_dim)
        if pca_dim <= 0 or pca_dim > X.shape[1]:
            raise ValueError(f"Invalid pca_dim={pca_dim} for D={X.shape[1]}")

        # Fit on random subset to control memory
        n_fit = min(int(max_pca_samples), X.shape[0])
        fit_idx = rng.choice(X.shape[0], size=n_fit, replace=False)
        ipca = IncrementalPCA(n_components=pca_dim, batch_size=8192)
        ipca.fit(X[fit_idx])

        # Transform in chunks
        X_pca = np.empty((X.shape[0], pca_dim), dtype=np.float32)
        bs = 8192
        for i0 in range(0, X.shape[0], bs):
            i1 = min(i0 + bs, X.shape[0])
            X_pca[i0:i1] = ipca.transform(X[i0:i1]).astype(np.float32)
        X = X_pca
        print(f"Applied IncrementalPCA: new D={X.shape[1]}")

    meta = {
        "dataset": "ds003626",
        "root_dir": str(root_dir),
        "subjects": subjects,
        "sessions": sessions,
        "condition": condition,
        "cond_code": cond_code,
        "t_start": t_start,
        "t_end": t_end,
        "fs": fs,
        "time_stride": time_stride,
        "pick_channels": pick_channels,
        "max_trials_per_domain": max_trials_per_domain,
        "pca_dim": pca_dim,
        "y_map": y_map,
        "domain_map": domain_map,
        "n_domains": int(z.max() + 1),
        "N": int(X.shape[0]),
        "D": int(X.shape[1]),
        "K": int(U.shape[1]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        U=U.astype(np.float32),
        y=y.astype(np.int64),
        z=z.astype(np.int64),
        trial_id=trial_id.astype(np.int64),
        meta_json=json.dumps(meta),
    )
    print(f"Saved: {out_path} (N={X.shape[0]}, D={X.shape[1]}, K={U.shape[1]}, n_domains={meta['n_domains']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to ds003626 dataset root (contains derivatives/)")
    ap.add_argument("--out", type=str, required=True, help="Output .npz path")
    ap.add_argument("--subjects", type=str, default="01", help="Subjects, e.g. 01 or 01,02,03")
    ap.add_argument("--sessions", type=str, default="01,02,03", help="Sessions, e.g. 01,02,03")
    ap.add_argument("--condition", type=str, default="inner", help="inner|pronounced|visualized|all")
    ap.add_argument("--t_start", type=float, default=1.5)
    ap.add_argument("--t_end", type=float, default=3.5)
    ap.add_argument("--fs", type=int, default=256)
    ap.add_argument("--time_stride", type=int, default=4, help="Subsample time axis, e.g. 4 -> 256Hz->64Hz")
    ap.add_argument("--pick_channels", type=int, default=None, help="Keep first N channels (optional)")
    ap.add_argument("--max_trials_per_domain", type=int, default=None, help="Optional cap per domain")
    ap.add_argument("--pca_dim", type=int, default=32, help="Optional PCA output dimension (set 0 to disable)")
    ap.add_argument("--max_pca_samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    pca_dim = None if int(args.pca_dim) <= 0 else int(args.pca_dim)

    build_npz(
        root_dir=Path(args.root),
        out_path=Path(args.out),
        subjects=_parse_list_arg(args.subjects),
        sessions=_parse_list_arg(args.sessions),
        condition=args.condition,
        t_start=float(args.t_start),
        t_end=float(args.t_end),
        fs=int(args.fs),
        time_stride=int(args.time_stride),
        pick_channels=args.pick_channels,
        max_trials_per_domain=args.max_trials_per_domain,
        pca_dim=pca_dim,
        max_pca_samples=int(args.max_pca_samples),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
