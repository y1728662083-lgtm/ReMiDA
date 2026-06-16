from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle

import mne
import numpy as np
import pandas as pd

from .channel_selection import select_task_related_channels
from .config import ExperimentConfig
from .constants import CLASS_ID_TO_NAME, CONDITION_NAME_TO_ID
from .utils import StandardizerBank, fit_subject_standardizers, subjectwise_standardize_np


@dataclass
class SessionRecord:
    subject: str
    session: str
    eeg_fif_path: Path
    events_dat_path: Optional[Path]
    report_pkl_path: Optional[Path]

    @property
    def domain_key(self) -> str:
        return f"{self.subject}::{self.session}"


@dataclass
class BaseLoadedDataset:
    x_raw: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    times: np.ndarray
    sfreq: float
    ch_names: List[str]
    info_template: Any
    channel_selection_mode: str


@dataclass
class ProtocolRunBundle:
    protocol_name: str
    run_name: str
    reference_entity: str
    x_raw: np.ndarray
    y: np.ndarray
    entity_ids: np.ndarray
    domain_ids: np.ndarray
    metadata: pd.DataFrame
    entity_to_id: Dict[str, int]
    domain_to_id: Dict[str, int]
    id_to_entity: Dict[int, str]
    id_to_domain: Dict[int, str]
    domain_to_entity: Dict[int, int]
    train_idx: np.ndarray
    val_idx: np.ndarray
    adapt_idx: np.ndarray
    test_idx: np.ndarray
    times: np.ndarray
    sfreq: float
    ch_names: List[str]
    info_template: Any
    channel_selection_mode: str

    def fit_standardizers(self) -> StandardizerBank:
        train_entities = self.entity_ids[self.train_idx]
        bank = fit_subject_standardizers(self.x_raw[self.train_idx], train_entities, self.id_to_entity)
        # add adaptation/test entities using their unlabeled statistics when missing
        from .utils import SubjectStandardizer
        for eid, entity in self.id_to_entity.items():
            if entity in bank.by_subject:
                continue
            mask = self.entity_ids == eid
            if np.any(mask):
                bank.by_subject[entity] = SubjectStandardizer.fit_mean_std(self.x_raw[mask])
        return bank

    def standardized_raw(self, bank: StandardizerBank) -> np.ndarray:
        return subjectwise_standardize_np(self.x_raw, self.entity_ids, self.id_to_entity, bank)


@dataclass
class ProtocolCollection:
    base: BaseLoadedDataset
    cross_subject_runs: List[ProtocolRunBundle]
    cross_session_runs: List[ProtocolRunBundle]


def _epochs_get_data_compat(epochs: mne.BaseEpochs) -> np.ndarray:
    try:
        data = epochs.get_data(copy=True)
    except TypeError:
        data = epochs.get_data()
    return np.asarray(data, dtype=np.float32).copy()


def _load_pickle_compat(path: Path):
    last_err = None
    for encoding in (None, "latin1", "bytes"):
        try:
            with path.open("rb") as f:
                if encoding is None:
                    return pickle.load(f)
                return pickle.load(f, encoding=encoding)
        except Exception as exc:
            last_err = exc
    raise last_err


def _coerce_pickled_events(obj, path: Path) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
    elif isinstance(obj, dict):
        candidate_keys = ["events", "event", "data", "arr", "array", "trials", b"events", b"data", b"trials"]
        arr = None
        for key in candidate_keys:
            if key in obj:
                arr = np.asarray(obj[key])
                break
        if arr is None and len(obj) == 1:
            arr = np.asarray(next(iter(obj.values())))
        if arr is None:
            raise ValueError(f"Unsupported events structure in {path}: keys={list(obj.keys())}")
    else:
        raise ValueError(f"Unsupported events object in {path}: {type(obj)!r}")
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"events.dat from {path} must be N x >=3; got {arr.shape}")
    return arr.astype(np.int64)


def _load_events_dat(path: Path) -> np.ndarray:
    return _coerce_pickled_events(_load_pickle_compat(path), path)


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
    if arr.min() >= 1 and arr.max() <= n_epochs:
        arr = arr - 1
    arr = arr[(arr >= 0) & (arr < n_epochs)]
    return np.unique(arr)


def discover_session_records(cfg: ExperimentConfig) -> List[SessionRecord]:
    root = Path(cfg.dataset.dataset_root) / cfg.dataset.derivatives_subdir
    records: List[SessionRecord] = []
    for subject in cfg.dataset.subjects:
        for session in cfg.dataset.sessions:
            session_dir = root / subject / session
            if not session_dir.exists():
                continue
            eeg_files = sorted(session_dir.glob("*_eeg-epo.fif"))
            if not eeg_files:
                continue
            events_files = sorted(session_dir.glob("*_events.dat"))
            report_files = sorted(session_dir.glob("*_report.pkl"))
            records.append(
                SessionRecord(
                    subject=subject,
                    session=session,
                    eeg_fif_path=eeg_files[0],
                    events_dat_path=events_files[0] if events_files else None,
                    report_pkl_path=report_files[0] if report_files else None,
                )
            )
    if not records:
        raise FileNotFoundError(
            f"No session records found under {root}. Expected dataset_root/derivatives/sub-XX/ses-YY/*_eeg-epo.fif"
        )
    return records


def _session_dataframe(subject: str, session: str, n_epochs: int, events_dat: np.ndarray) -> pd.DataFrame:
    if len(events_dat) != n_epochs:
        raise ValueError(f"events length mismatch for {subject} {session}: {len(events_dat)} vs {n_epochs}")
    class_ids = events_dat[:, 1].astype(np.int64)
    condition_ids = events_dat[:, 2].astype(np.int64)
    sample_idx = events_dat[:, 0].astype(np.int64)
    return pd.DataFrame(
        {
            "subject": subject,
            "session": session,
            "domain_key": [f"{subject}::{session}"] * n_epochs,
            "sample_index": sample_idx,
            "class_id": class_ids,
            "class_name": [CLASS_ID_TO_NAME.get(int(x), str(int(x))) for x in class_ids],
            "condition_id": condition_ids,
            "trial_index_within_file": np.arange(n_epochs, dtype=np.int64),
        }
    )


def load_base_dataset(cfg: ExperimentConfig) -> BaseLoadedDataset:
    records = discover_session_records(cfg)
    first_epochs = mne.read_epochs(str(records[0].eeg_fif_path), preload=True, proj=False, verbose="ERROR")
    if cfg.dataset.baseline_tmin is not None or cfg.dataset.baseline_tmax is not None:
        first_epochs.apply_baseline((cfg.dataset.baseline_tmin, cfg.dataset.baseline_tmax))
    candidate_ch_names = list(first_epochs.ch_names)
    if cfg.dataset.use_task_related_channels:
        _, pick_names, pick_mode = select_task_related_channels(
            candidate_ch_names,
            requested=cfg.dataset.keep_channels or None,
            min_required_channels=cfg.dataset.min_required_channels,
        )
    else:
        pick_names = [name for name in candidate_ch_names if all(marker not in name.upper() for marker in ["EOG", "EMG", "EXG", "AUX", "ECG"])]
        pick_mode = "all_eeg_excluding_aux"

    arrays: List[np.ndarray] = []
    frames: List[pd.DataFrame] = []
    reference_info = None
    times = None
    sfreq = None
    for rec in records:
        epochs = mne.read_epochs(str(rec.eeg_fif_path), preload=True, proj=False, verbose="ERROR")
        if cfg.dataset.baseline_tmin is not None or cfg.dataset.baseline_tmax is not None:
            epochs.apply_baseline((cfg.dataset.baseline_tmin, cfg.dataset.baseline_tmax))
        epochs.pick(pick_names)
        data = _epochs_get_data_compat(epochs)
        if cfg.dataset.crop_points is not None:
            data = data[..., : int(cfg.dataset.crop_points)]
        events_dat = _load_events_dat(rec.events_dat_path) if rec.events_dat_path else np.column_stack(
            [epochs.events[:, 0], epochs.events[:, 2] - 1, np.full(len(epochs), CONDITION_NAME_TO_ID[cfg.dataset.condition])]
        )
        df = _session_dataframe(rec.subject, rec.session, len(epochs), events_dat)
        mask_condition = df["condition_id"].to_numpy(dtype=np.int64) == CONDITION_NAME_TO_ID[cfg.dataset.condition]
        mask_class = np.isin(df["class_id"].to_numpy(dtype=np.int64), np.asarray(cfg.dataset.class_ids, dtype=np.int64))
        mask = mask_condition & mask_class
        if cfg.dataset.drop_emg_contaminated:
            contaminated = _load_contaminated_trial_indices(rec.report_pkl_path, len(epochs))
            if contaminated.size:
                contam_mask = np.zeros(len(epochs), dtype=bool)
                contam_mask[contaminated] = True
                mask &= ~contam_mask
        arrays.append(data[mask].astype(np.float32))
        frames.append(df.loc[mask].reset_index(drop=True))
        if reference_info is None:
            reference_info = epochs.copy().pick(pick_names).info
            times = epochs.times[: data.shape[-1]].copy()
            sfreq = float(epochs.info["sfreq"])

    x = np.concatenate(arrays, axis=0).astype(np.float32)
    meta = pd.concat(frames, axis=0, ignore_index=True)
    y = meta["class_id"].to_numpy(dtype=np.int64)
    meta["global_index"] = np.arange(len(meta), dtype=np.int64)
    return BaseLoadedDataset(
        x_raw=x,
        y=y,
        metadata=meta.reset_index(drop=True),
        times=times,
        sfreq=sfreq,
        ch_names=pick_names,
        info_template=reference_info,
        channel_selection_mode=pick_mode,
    )


def _stratified_train_val_split(global_indices: np.ndarray, labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    global_indices = np.asarray(global_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    tr_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for cls in np.unique(labels[global_indices]):
        idx = global_indices[labels[global_indices] == cls].copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * float(val_ratio)))
        n_val = min(max(n_val, 1 if len(idx) >= 5 else 0), max(len(idx) - 1, 0))
        val_parts.append(idx[:n_val])
        tr_parts.append(idx[n_val:])
    tr = np.sort(np.concatenate(tr_parts)) if tr_parts else np.array([], dtype=np.int64)
    val = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=np.int64)
    return tr, val


def _make_local_bundle(
    base: BaseLoadedDataset,
    protocol_name: str,
    run_name: str,
    subset_idx: np.ndarray,
    train_global: np.ndarray,
    val_global: np.ndarray,
    adapt_global: np.ndarray,
    test_global: np.ndarray,
    reference_entity: str,
    entity_key_series: pd.Series,
) -> ProtocolRunBundle:
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    local_of_global = {int(g): i for i, g in enumerate(subset_idx.tolist())}

    x = base.x_raw[subset_idx].astype(np.float32)
    meta = base.metadata.iloc[subset_idx].reset_index(drop=True).copy()
    y = meta["class_id"].to_numpy(dtype=np.int64)
    meta["entity_key"] = entity_key_series.iloc[subset_idx].astype(str).to_numpy()
    unique_entities = sorted(meta["entity_key"].astype(str).unique().tolist())
    entity_to_id = {k: i for i, k in enumerate(unique_entities)}
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    unique_domains = sorted(meta["domain_key"].astype(str).unique().tolist())
    domain_to_id = {k: i for i, k in enumerate(unique_domains)}
    id_to_domain = {v: k for k, v in domain_to_id.items()}
    domain_to_entity = {}
    for dom, did in domain_to_id.items():
        ent = meta.loc[meta["domain_key"] == dom, "entity_key"].iloc[0]
        domain_to_entity[did] = entity_to_id[str(ent)]
    entity_ids = np.asarray([entity_to_id[s] for s in meta["entity_key"].astype(str).tolist()], dtype=np.int64)
    domain_ids = np.asarray([domain_to_id[s] for s in meta["domain_key"].astype(str).tolist()], dtype=np.int64)

    def _localize(idx: np.ndarray) -> np.ndarray:
        return np.asarray([local_of_global[int(g)] for g in np.asarray(idx, dtype=np.int64).tolist()], dtype=np.int64)

    train_idx = _localize(train_global)
    val_idx = _localize(val_global)
    adapt_idx = _localize(adapt_global)
    test_idx = _localize(test_global)

    split = np.array(["unused"] * len(meta), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[adapt_idx] = "adapt"
    split[test_idx] = "test"
    meta["split"] = split
    meta["protocol"] = protocol_name
    meta["run_name"] = run_name

    return ProtocolRunBundle(
        protocol_name=protocol_name,
        run_name=run_name,
        reference_entity=reference_entity,
        x_raw=x,
        y=y,
        entity_ids=entity_ids,
        domain_ids=domain_ids,
        metadata=meta,
        entity_to_id=entity_to_id,
        domain_to_id=domain_to_id,
        id_to_entity=id_to_entity,
        id_to_domain=id_to_domain,
        domain_to_entity=domain_to_entity,
        train_idx=train_idx,
        val_idx=val_idx,
        adapt_idx=adapt_idx,
        test_idx=test_idx,
        times=base.times,
        sfreq=base.sfreq,
        ch_names=base.ch_names,
        info_template=base.info_template,
        channel_selection_mode=base.channel_selection_mode,
    )


def build_cross_subject_runs(base: BaseLoadedDataset, cfg: ExperimentConfig) -> List[ProtocolRunBundle]:
    runs: List[ProtocolRunBundle] = []
    meta = base.metadata
    subjects = cfg.dataset.subjects
    if len(subjects) != 2:
        raise ValueError("Cross-subject protocol currently expects exactly two subjects.")
    for direction in cfg.cross_subject.directions:
        source, target = direction.split("->")
        source = source.strip()
        target = target.strip()
        for session in cfg.cross_subject.sessions:
            subset_mask = meta["subject"].isin([source, target]) & (meta["session"].astype(str) == str(session))
            subset_idx = meta.index[subset_mask].to_numpy(dtype=np.int64)
            src_idx = meta.index[(meta["subject"].astype(str) == source) & (meta["session"].astype(str) == session)].to_numpy(dtype=np.int64)
            tgt_idx = meta.index[(meta["subject"].astype(str) == target) & (meta["session"].astype(str) == session)].to_numpy(dtype=np.int64)
            if len(src_idx) == 0 or len(tgt_idx) == 0:
                continue
            tr, val = _stratified_train_val_split(src_idx, base.y, cfg.cross_subject.source_val_ratio, cfg.cross_subject.seed)
            run_name = f"cross_subject__{source}_to_{target}__{session}"
            entity_keys = meta["subject"].astype(str)
            runs.append(
                _make_local_bundle(
                    base=base,
                    protocol_name="cross_subject",
                    run_name=run_name,
                    subset_idx=subset_idx,
                    train_global=tr,
                    val_global=val,
                    adapt_global=tgt_idx,
                    test_global=tgt_idx,
                    reference_entity=source,
                    entity_key_series=entity_keys,
                )
            )
    return runs


def build_cross_session_runs(base: BaseLoadedDataset, cfg: ExperimentConfig) -> List[ProtocolRunBundle]:
    runs: List[ProtocolRunBundle] = []
    meta = base.metadata
    entity_keys_all = (meta["subject"].astype(str) + "::" + meta["session"].astype(str))
    for subject in cfg.cross_session.subjects:
        for target_session in cfg.cross_session.target_sessions:
            source_sessions = [s for s in cfg.dataset.sessions if s != target_session]
            ref_session = source_sessions[0]
            subset_mask = (meta["subject"].astype(str) == subject) & meta["session"].isin(source_sessions + [target_session])
            subset_idx = meta.index[subset_mask].to_numpy(dtype=np.int64)
            src_idx = meta.index[(meta["subject"].astype(str) == subject) & meta["session"].isin(source_sessions)].to_numpy(dtype=np.int64)
            tgt_idx = meta.index[(meta["subject"].astype(str) == subject) & (meta["session"].astype(str) == target_session)].to_numpy(dtype=np.int64)
            if len(src_idx) == 0 or len(tgt_idx) == 0:
                continue
            tr, val = _stratified_train_val_split(src_idx, base.y, cfg.cross_session.source_val_ratio, cfg.cross_session.seed)
            run_name = f"cross_session__{subject}__holdout_{target_session}"
            runs.append(
                _make_local_bundle(
                    base=base,
                    protocol_name="cross_session",
                    run_name=run_name,
                    subset_idx=subset_idx,
                    train_global=tr,
                    val_global=val,
                    adapt_global=tgt_idx,
                    test_global=tgt_idx,
                    reference_entity=f"{subject}::{ref_session}",
                    entity_key_series=entity_keys_all,
                )
            )
    return runs


def build_protocol_collection(cfg: ExperimentConfig) -> ProtocolCollection:
    base = load_base_dataset(cfg)
    cross_subject_runs = build_cross_subject_runs(base, cfg) if cfg.cross_subject.enabled else []
    cross_session_runs = build_cross_session_runs(base, cfg) if cfg.cross_session.enabled else []
    return ProtocolCollection(base=base, cross_subject_runs=cross_subject_runs, cross_session_runs=cross_session_runs)
