from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class JSONLLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def mean(self) -> float:
        return 0.0 if self.count == 0 else self.total / self.count


@dataclass
class MeterCollection:
    meters: Dict[str, RunningMean] = field(default_factory=dict)

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = RunningMean()
            self.meters[k].update(float(v))

    def as_dict(self) -> Dict[str, float]:
        return {k: m.mean for k, m in self.meters.items()}


class CSVHistoryWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False
        self._fieldnames: Optional[List[str]] = None

    def write_row(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        if not self._header_written:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self._header_written = True
        else:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writerow(row)


@dataclass
class BestCheckpointTracker:
    mode: str = "min"
    best_score: float = math.inf
    best_epoch: int = -1
    best_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(self.mode)
        if self.mode == "max":
            self.best_score = -math.inf

    def update(self, epoch: int, score: float, path: str | Path) -> bool:
        improved = float(score) < self.best_score if self.mode == "min" else float(score) > self.best_score
        if improved:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.best_path = str(path)
            return True
        return False


def torch_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_item(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def flatten_batch_time(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got {x.shape}")
    return np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)


def flatten_torch_batch_time(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B,C,T], got {tuple(x.shape)}")
    return x.permute(1, 0, 2).reshape(x.shape[1], -1)


class SubjectStandardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> None:
        self.mean = mean.astype(np.float32)
        self.scale = np.maximum(scale.astype(np.float32), eps)
        self.eps = float(eps)

    @classmethod
    def fit_mean_std(cls, x: np.ndarray, eps: float = 1e-6) -> "SubjectStandardizer":
        mean = x.mean(axis=(0, 2), keepdims=True)
        std = x.std(axis=(0, 2), keepdims=True)
        return cls(mean=mean, scale=std, eps=eps)

    def transform_np(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse_np(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.mean

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.mean).to(device=x.device, dtype=x.dtype)
        scale = torch.from_numpy(self.scale).to(device=x.device, dtype=x.dtype)
        return (x - mean) / scale

    def inverse_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.mean).to(device=x.device, dtype=x.dtype)
        scale = torch.from_numpy(self.scale).to(device=x.device, dtype=x.dtype)
        return x * scale + mean


class StandardizerBank:
    def __init__(self, by_subject: Dict[str, SubjectStandardizer]) -> None:
        self.by_subject = dict(by_subject)

    def transform_np(self, x: np.ndarray, subject: str) -> np.ndarray:
        return self.by_subject[subject].transform_np(x)

    def inverse_np(self, x: np.ndarray, subject: str) -> np.ndarray:
        return self.by_subject[subject].inverse_np(x)

    def transform_torch(self, x: torch.Tensor, subject: str) -> torch.Tensor:
        return self.by_subject[subject].transform_torch(x)

    def inverse_torch(self, x: torch.Tensor, subject: str) -> torch.Tensor:
        return self.by_subject[subject].inverse_torch(x)


def fit_subject_standardizers(x: np.ndarray, subject_labels: np.ndarray, id_to_subject: Dict[int, str]) -> StandardizerBank:
    by_subject: Dict[str, SubjectStandardizer] = {}
    for sid, subject in id_to_subject.items():
        mask = np.asarray(subject_labels, dtype=np.int64) == int(sid)
        if not np.any(mask):
            continue
        by_subject[str(subject)] = SubjectStandardizer.fit_mean_std(x[mask])
    return StandardizerBank(by_subject)


def subjectwise_standardize_np(x: np.ndarray, subject_labels: np.ndarray, id_to_subject: Dict[int, str], bank: StandardizerBank) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for sid, subject in id_to_subject.items():
        mask = np.asarray(subject_labels, dtype=np.int64) == int(sid)
        if np.any(mask):
            out[mask] = bank.transform_np(x[mask], str(subject))
    return out


def stratified_group_split(groups: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    unique = np.unique(groups)
    for g in unique:
        idx = np.where(groups == g)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_parts.append(idx)
            continue
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_test = min(max(n_test, 1 if n >= 5 else 0), max(n - 2, 0))
        n_val = min(max(n_val, 1 if n >= 8 else 0), max(n - n_test - 1, 0))
        n_train = max(n - n_test - n_val, 1)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train+n_val])
        test_parts.append(idx[n_train+n_val:])
    train_idx = np.sort(np.concatenate(train_parts, axis=0)) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.sort(np.concatenate(val_parts, axis=0)) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts, axis=0)) if test_parts else np.array([], dtype=np.int64)
    return train_idx, val_idx, test_idx


def ridge_probe_accuracy(features: np.ndarray, labels: np.ndarray, alpha: float = 1.0, test_ratio: float = 0.25, seed: int = 42) -> Tuple[float, float]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        chance = 1.0 / max(len(np.unique(labels)), 1)
        return chance, chance
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        n_test = min(n_test, max(len(idx) - 1, 1))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    tr = np.concatenate(train_parts)
    te = np.concatenate(test_parts)
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(features[tr], labels[tr])
    pred = clf.predict(features[te])
    acc = accuracy_score(labels[te], pred)
    chance = np.max(np.bincount(labels[te])) / len(te)
    return float(acc), float(chance)


def multiclass_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def logistic_probe(features: np.ndarray, labels: np.ndarray, seed: int = 42, test_ratio: float = 0.25, max_iter: int = 500) -> Dict[str, float]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        chance = 1.0 / max(len(np.unique(labels)), 1)
        return {"accuracy": chance, "macro_f1": chance, "balanced_accuracy": chance, "chance": chance}
    rng = np.random.default_rng(seed)
    tr_parts, te_parts = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        n_test = min(n_test, max(len(idx) - 1, 1))
        te_parts.append(idx[:n_test])
        tr_parts.append(idx[n_test:])
    tr = np.concatenate(tr_parts)
    te = np.concatenate(te_parts)
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(features[tr], labels[tr])
    pred = clf.predict(features[te])
    out = multiclass_classification_metrics(labels[te], pred)
    out["chance"] = float(np.max(np.bincount(labels[te])) / len(te))
    return out
