#!/usr/bin/env python3
"""Build a ReMiDA-baseline .npz file directly from the AutoDL Inner Speech derivatives tree.

This adapter follows the reading logic in the user's data.py:
- discover derivatives/sub-XX/ses-YY/*_eeg-epo.fif
- read epochs with mne.read_epochs(..., preload=True, proj=False)
- read *_events.dat and *_report.pkl with pickle compatibility fallbacks
- keep class_id / condition_id / subject / session / domain metadata

The generated .npz can be consumed by train_ivae_baseline.py and train_tcl_baseline.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np


def _epochs_get_data_compat(epochs: mne.BaseEpochs) -> np.ndarray:
    try:
        data = epochs.get_data(copy=True)
    except TypeError:  # older MNE versions
        data = epochs.get_data()
    return np.asarray(data, dtype=np.float32).copy()


def _load_pickle_compat(path: Path) -> Any:
    last_err: Optional[Exception] = None
    for encoding in (None, "latin1", "bytes"):
        try:
            with path.open("rb") as f:
                if encoding is None:
                    return pickle.load(f)
                return pickle.load(f, encoding=encoding)
        except Exception as exc:
            last_err = exc
    assert last_err is not None
    raise last_err


def _coerce_pickled_events(obj: Any, path: Path) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
    elif isinstance(obj, dict):
        candidate_keys = [
            "events", "event", "data", "arr", "array", "trials",
            b"events", b"event", b"data", b"arr", b"array", b"trials",
        ]
        arr = None
        for key in candidate_keys:
            if key in obj:
                arr = np.asarray(obj[key])
                break
        if arr is None and len(obj) == 1:
            arr = np.asarray(next(iter(obj.values())))
        if arr is None:
            raise ValueError(
                f"Unsupported pickled events.dat structure from {path}. Dict keys={list(obj.keys())}"
            )
    else:
        raise ValueError(f"Unsupported events.dat object type from {path}: {type(obj)!r}")

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D events array from {path}, got shape {arr.shape}")
    if arr.shape[1] < 3:
        raise ValueError(
            f"Expected events.dat with at least 3 columns [sample, class_id, condition_id], "
            f"got shape {arr.shape} from {path}"
        )
    return arr.astype(np.int64)


def _load_events_dat(path: Path) -> np.ndarray:
    obj = _load_pickle_compat(path)
    return _coerce_pickled_events(obj, path)


def _load_contaminated_trial_indices(path: Optional[Path], n_epochs: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.array([], dtype=np.int64)
    report = _load_pickle_compat(path)
    if not isinstance(report, dict):
        return np.array([], dtype=np.int64)
    emg_trials = report.get("EMG_trials", report.get(b"EMG_trials", []))
    arr = np.asarray(emg_trials, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    # data.py compatibility: convert 1-based trial indices to 0-based if needed.
    if arr.min() >= 1 and arr.max() <= n_epochs:
        arr = arr - 1
    arr = arr[(arr >= 0) & (arr < n_epochs)]
    return np.unique(arr)


def _parse_int_list(values: Optional[Sequence[str]], name: str) -> Optional[List[int]]:
    if values is None or len(values) == 0:
        return None
    out: List[int] = []
    for v in values:
        for part in str(v).replace(",", " ").split():
            if part.strip():
                out.append(int(part))
    if not out:
        return None
    return out


def _discover_subjects(derivatives_root: Path, subjects: Optional[Sequence[str]]) -> List[str]:
    if subjects:
        return list(subjects)
    return sorted(p.name for p in derivatives_root.glob("sub-*") if p.is_dir())


def _discover_sessions(subject_dir: Path, sessions: Optional[Sequence[str]]) -> List[str]:
    if sessions:
        return list(sessions)
    return sorted(p.name for p in subject_dir.glob("ses-*") if p.is_dir())


def _make_domain(subject: str, session: str, mode: str) -> str:
    if mode == "subject":
        return subject
    if mode == "session":
        return session
    if mode == "subject_session":
        return f"{subject}::{session}"
    raise ValueError(f"Unknown domain mode: {mode}")


def _make_period_labels_for_session(n: int, n_periods: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.arange(n)
    labels = np.floor(order * n_periods / max(n, 1)).astype(np.int64)
    return np.clip(labels, 0, n_periods - 1)


def build_npz(
    dataset_root: Path,
    derivatives_subdir: str,
    out_path: Path,
    subjects: Optional[Sequence[str]],
    sessions: Optional[Sequence[str]],
    domain_mode: str,
    u_source: str,
    n_periods: int,
    class_ids: Optional[Sequence[int]],
    condition_ids: Optional[Sequence[int]],
    drop_emg_contaminated: bool,
    require_events_dat: bool,
    flatten: bool,
    verbose: bool,
    max_records: Optional[int],
    compress_npz: bool,
) -> None:
    derivatives_root = dataset_root / derivatives_subdir
    if not derivatives_root.exists():
        raise FileNotFoundError(f"derivatives root does not exist: {derivatives_root}")

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    domain_parts: List[np.ndarray] = []
    subject_parts: List[np.ndarray] = []
    session_parts: List[np.ndarray] = []
    condition_parts: List[np.ndarray] = []
    u_parts: List[np.ndarray] = []
    local_trial_parts: List[np.ndarray] = []
    source_file_parts: List[np.ndarray] = []
    kept_count_by_record = []
    processed_records = 0

    times_ref = None
    ch_names_ref = None
    sfreq_ref = None

    all_subjects = _discover_subjects(derivatives_root, subjects)
    if not all_subjects:
        raise FileNotFoundError(f"No subject folders found under {derivatives_root}")

    for subject in all_subjects:
        subject_dir = derivatives_root / subject
        all_sessions = _discover_sessions(subject_dir, sessions)
        for session in all_sessions:
            session_dir = subject_dir / session
            eeg_files = sorted(session_dir.glob("*_eeg-epo.fif"))
            if not eeg_files:
                continue
            eeg_path = eeg_files[0]
            event_files = sorted(session_dir.glob("*_events.dat"))
            report_files = sorted(session_dir.glob("*_report.pkl"))
            events_path = event_files[0] if event_files else None
            report_path = report_files[0] if report_files else None
            if require_events_dat and events_path is None:
                raise FileNotFoundError(f"Missing events.dat for {subject} {session}: {session_dir}")

            if max_records is not None and processed_records >= max_records:
                if verbose:
                    print(f"[STOP] reached --max-records {max_records}", flush=True)
                break
            if verbose:
                print(f"[READ] {subject}/{session}: {eeg_path}", flush=True)
            epochs = mne.read_epochs(str(eeg_path), preload=True, proj=False, verbose="ERROR")
            data = _epochs_get_data_compat(epochs)  # (n_epochs, n_channels, n_times)
            n_epochs = len(data)
            if verbose:
                print(f"[LOADED] {subject}/{session}: epochs={n_epochs}, shape={data.shape}", flush=True)
            if n_epochs == 0:
                continue

            times = epochs.times.copy()
            ch_names = list(epochs.ch_names)
            sfreq = float(epochs.info["sfreq"])
            if times_ref is None:
                times_ref = times
                ch_names_ref = ch_names
                sfreq_ref = sfreq
            else:
                if len(times_ref) != len(times) or not np.allclose(times_ref, times):
                    raise ValueError(f"Inconsistent time axis in {eeg_path}")
                if list(ch_names_ref) != list(ch_names):
                    raise ValueError(f"Inconsistent channel order in {eeg_path}")
                if not np.isclose(float(sfreq_ref), sfreq):
                    raise ValueError(f"Inconsistent sfreq in {eeg_path}: {sfreq_ref} vs {sfreq}")

            if events_path is not None:
                events_dat = _load_events_dat(events_path)
            else:
                # Fallback only when explicitly allowed. Class labels become 0, condition labels become 0.
                events_dat = np.zeros((n_epochs, 4), dtype=np.int64)
                if getattr(epochs, "events", None) is not None and len(epochs.events) == n_epochs:
                    events_dat[:, 0] = epochs.events[:, 0]
                events_dat[:, 1] = 0
                events_dat[:, 2] = 0
                events_dat[:, 3] = int(session.split("-")[-1]) if session.split("-")[-1].isdigit() else 0

            if len(events_dat) != n_epochs:
                raise ValueError(f"events.dat length {len(events_dat)} != epochs {n_epochs} for {eeg_path}")

            keep = np.ones(n_epochs, dtype=bool)
            if drop_emg_contaminated:
                contaminated = _load_contaminated_trial_indices(report_path, n_epochs=n_epochs)
                if contaminated.size > 0:
                    keep[contaminated] = False
            if class_ids is not None:
                keep &= np.isin(events_dat[:, 1].astype(np.int64), np.asarray(class_ids, dtype=np.int64))
            if condition_ids is not None:
                keep &= np.isin(events_dat[:, 2].astype(np.int64), np.asarray(condition_ids, dtype=np.int64))
            if not np.any(keep):
                kept_count_by_record.append({"subject": subject, "session": session, "kept": 0, "total": int(n_epochs)})
                continue

            data = data[keep]
            ev = events_dat[keep]
            n_kept = len(data)
            domain = _make_domain(subject, session, domain_mode)
            if u_source == "period":
                u = _make_period_labels_for_session(n_kept, n_periods)
            elif u_source == "condition_id":
                u = ev[:, 2].astype(np.int64)
            elif u_source == "session_num":
                if ev.shape[1] >= 4:
                    u = ev[:, 3].astype(np.int64)
                else:
                    u = np.full(n_kept, int(session.split("-")[-1]) if session.split("-")[-1].isdigit() else 0, dtype=np.int64)
            else:
                raise ValueError(f"Unknown u_source: {u_source}")

            X_parts.append(data.astype(np.float32))
            y_parts.append(ev[:, 1].astype(np.int64))
            condition_parts.append(ev[:, 2].astype(np.int64))
            u_parts.append(u.astype(np.int64))
            domain_parts.append(np.full(n_kept, domain, dtype=object))
            subject_parts.append(np.full(n_kept, subject, dtype=object))
            session_parts.append(np.full(n_kept, session, dtype=object))
            local_trial_parts.append(np.arange(n_epochs, dtype=np.int64)[keep])
            source_file_parts.append(np.full(n_kept, str(eeg_path), dtype=object))
            kept_count_by_record.append({"subject": subject, "session": session, "kept": int(n_kept), "total": int(n_epochs)})
            processed_records += 1
            if verbose:
                print(f"[KEEP] {subject}/{session}: kept={n_kept}/{n_epochs}, domain={domain}", flush=True)
        if max_records is not None and processed_records >= max_records:
            break

    if not X_parts:
        raise RuntimeError("No epochs remained after applying filters.")

    X = np.concatenate(X_parts, axis=0)
    if flatten:
        X = X.reshape(X.shape[0], -1).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    domain = np.concatenate(domain_parts, axis=0).astype(str)
    subject_arr = np.concatenate(subject_parts, axis=0).astype(str)
    session_arr = np.concatenate(session_parts, axis=0).astype(str)
    condition = np.concatenate(condition_parts, axis=0)
    u = np.concatenate(u_parts, axis=0)
    local_trial = np.concatenate(local_trial_parts, axis=0)
    source_file = np.concatenate(source_file_parts, axis=0).astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[SAVE] writing {out_path} with X shape={X.shape}; compress={compress_npz}", flush=True)
    save_func = np.savez_compressed if compress_npz else np.savez
    save_func(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        domain=domain,
        u=u.astype(np.int64),
        subject=subject_arr,
        session=session_arr,
        condition_id=condition.astype(np.int64),
        local_trial_index=local_trial.astype(np.int64),
        source_file=source_file,
        sfreq=np.array(float(sfreq_ref), dtype=np.float32),
        ch_names=np.asarray(ch_names_ref, dtype=object),
        times=np.asarray(times_ref, dtype=np.float32),
    )

    summary = {
        "out": str(out_path),
        "dataset_root": str(dataset_root),
        "derivatives_root": str(derivatives_root),
        "n_samples": int(len(X)),
        "X_shape": list(X.shape),
        "class_ids": sorted([int(v) for v in np.unique(y)]),
        "domains": sorted([str(v) for v in np.unique(domain)]),
        "subjects": sorted([str(v) for v in np.unique(subject_arr)]),
        "sessions": sorted([str(v) for v in np.unique(session_arr)]),
        "u_source": u_source,
        "u_values": sorted([int(v) for v in np.unique(u)]),
        "domain_mode": domain_mode,
        "drop_emg_contaminated": bool(drop_emg_contaminated),
        "records": kept_count_by_record,
    }
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved NPZ: {out_path}")
    print(f"Saved summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert AutoDL derivatives FIF/events data into baseline .npz format.")
    p.add_argument("--dataset-root", default=".", help="Root folder containing derivatives/.")
    p.add_argument("--derivatives-subdir", default="derivatives")
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument("--subjects", nargs="*", default=None, help="Optional subject list, e.g. sub-01 sub-02. Default: all sub-*.")
    p.add_argument("--sessions", nargs="*", default=None, help="Optional session list, e.g. ses-01 ses-02. Default: all ses-*.")
    p.add_argument("--domain-mode", choices=["subject", "session", "subject_session"], default="subject")
    p.add_argument("--u-source", choices=["period", "condition_id", "session_num"], default="period")
    p.add_argument("--n-periods", type=int, default=20)
    p.add_argument("--class-ids", nargs="*", default=None, help="Optional class ids, e.g. 0 1 2 3 or 1,2,3,4.")
    p.add_argument("--condition-ids", nargs="*", default=None, help="Optional condition ids to keep.")
    p.add_argument("--drop-emg-contaminated", action="store_true", help="Drop EMG_trials listed in *_report.pkl, matching data.py behavior when enabled.")
    p.add_argument("--allow-missing-events", action="store_true", help="Do not require *_events.dat. Labels will be unavailable if missing.")
    p.add_argument("--flatten", action="store_true", help="Save X as (N,F). If omitted, save as (N,C,T); train scripts flatten automatically.")
    p.add_argument("--verbose", action="store_true", help="Print progress for each FIF file.")
    p.add_argument("--max-records", type=int, default=None, help="Debug only: stop after this many subject/session records.")
    p.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed. Much faster, larger file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_npz(
        dataset_root=Path(args.dataset_root),
        derivatives_subdir=args.derivatives_subdir,
        out_path=Path(args.out),
        subjects=args.subjects,
        sessions=args.sessions,
        domain_mode=args.domain_mode,
        u_source=args.u_source,
        n_periods=args.n_periods,
        class_ids=_parse_int_list(args.class_ids, "class_ids"),
        condition_ids=_parse_int_list(args.condition_ids, "condition_ids"),
        drop_emg_contaminated=bool(args.drop_emg_contaminated),
        require_events_dat=not bool(args.allow_missing_events),
        flatten=bool(args.flatten),
        verbose=bool(args.verbose),
        max_records=args.max_records,
        compress_npz=not bool(args.no_compress),
    )


if __name__ == "__main__":
    main()
