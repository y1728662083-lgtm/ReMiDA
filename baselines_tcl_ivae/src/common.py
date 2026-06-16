import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_npz_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["X", "y", "domain"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}. Required keys: {required}")
    out = {k: data[k] for k in data.files}
    X0 = np.asarray(out["X"])
    if X0.ndim < 2:
        raise ValueError("X must have shape (N, F) or (N, C, T).")
    out["X_original_shape"] = np.array(X0.shape)
    if X0.ndim > 2:
        X = X0.reshape(X0.shape[0], -1)
    else:
        X = X0
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out["X"] = X
    if len(out["X"]) != len(out["y"]) or len(out["X"]) != len(out["domain"]):
        raise ValueError("X, y and domain must have the same first dimension.")
    return out


def encode_1d_labels(a: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    le = LabelEncoder()
    enc = le.fit_transform(np.asarray(a).astype(str))
    mapping = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
    return enc.astype(np.int64), mapping


def one_hot(labels: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    labels = labels.astype(np.int64).reshape(-1)
    if n_classes is None:
        n_classes = int(labels.max()) + 1 if labels.size else 0
    y = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
    if labels.size:
        y[np.arange(labels.shape[0]), labels] = 1.0
    return y


def normalize_u(u: Optional[np.ndarray], domains: np.ndarray, n_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer period labels and one-hot period labels."""
    n = len(domains)
    if u is None:
        labels = np.zeros(n, dtype=np.int64)
        for d in np.unique(domains):
            idx = np.where(domains == d)[0]
            order = np.arange(len(idx))
            bins = np.floor(order * n_periods / max(len(idx), 1)).astype(np.int64)
            bins = np.clip(bins, 0, n_periods - 1)
            labels[idx] = bins
        return labels, one_hot(labels, n_periods)
    u = np.asarray(u)
    if u.ndim == 2:
        labels = np.argmax(u, axis=1).astype(np.int64)
        return labels, u.astype(np.float32)
    labels, _ = encode_1d_labels(u)
    return labels, one_hot(labels)


@dataclass
class SplitIndices:
    reftrain: np.ndarray
    refval: np.ndarray
    targetadapt: np.ndarray
    targettest: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "reftrain": np.asarray(self.reftrain, dtype=np.int64),
            "refval": np.asarray(self.refval, dtype=np.int64),
            "targetadapt": np.asarray(self.targetadapt, dtype=np.int64),
            "targettest": np.asarray(self.targettest, dtype=np.int64),
        }


def make_reference_target_split(
    y: np.ndarray,
    domains: np.ndarray,
    ref_domain: str,
    target_domain: str,
    seed: int = 42,
    ref_val_ratio: float = 0.2,
    target_test_ratio: float = 0.5,
) -> SplitIndices:
    ref_mask = domains.astype(str) == str(ref_domain)
    tgt_mask = domains.astype(str) == str(target_domain)
    ref_idx = np.where(ref_mask)[0]
    tgt_idx = np.where(tgt_mask)[0]
    if len(ref_idx) == 0:
        raise ValueError(f"No sample found for ref_domain={ref_domain}. Available={sorted(set(domains.astype(str)))}")
    if len(tgt_idx) == 0:
        raise ValueError(f"No sample found for target_domain={target_domain}. Available={sorted(set(domains.astype(str)))}")

    def _stratified_split(idx: np.ndarray, test_size: float):
        yy = y[idx]
        counts = {c: int(np.sum(yy == c)) for c in np.unique(yy)}
        stratify = yy if counts and min(counts.values()) >= 2 else None
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)

    reftrain, refval = _stratified_split(ref_idx, ref_val_ratio)
    targetadapt, targettest = _stratified_split(tgt_idx, target_test_ratio)
    return SplitIndices(reftrain=np.asarray(reftrain, dtype=np.int64), refval=np.asarray(refval, dtype=np.int64), targetadapt=np.asarray(targetadapt, dtype=np.int64), targettest=np.asarray(targettest, dtype=np.int64))


def save_split(path: str, split: SplitIndices) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.savez_compressed(path, **split.as_dict())


def load_split(path: str) -> SplitIndices:
    z = np.load(path)
    return SplitIndices(z["reftrain"], z["refval"], z["targetadapt"], z["targettest"])


def fit_standardizer(X: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(X[idx])
    Xt = scaler.transform(X).astype(np.float32)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Xt, scaler


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, Z: np.ndarray, y: np.ndarray, split: SplitIndices, device: torch.device) -> Dict[str, float]:
    model.eval()
    pred = {}
    for name, idx in split.as_dict().items():
        logits = model(torch.from_numpy(Z[idx]).float().to(device))
        pred[name] = torch.argmax(logits, dim=1).cpu().numpy()
    return {
        "val_acc": float(accuracy_score(y[split.refval], pred["refval"])),
        "val_f1": float(f1_score(y[split.refval], pred["refval"], average="macro", zero_division=0)),
        "test_acc": float(accuracy_score(y[split.targettest], pred["targettest"])),
        "test_f1": float(f1_score(y[split.targettest], pred["targettest"], average="macro", zero_division=0)),
    }


def train_downstream_classifier(
    Z: np.ndarray,
    y: np.ndarray,
    split: SplitIndices,
    out_dir: str,
    seed: int = 42,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    patience: int = 25,
    device: Optional[str] = None,
    prefix: str = "",
) -> Dict[str, float]:
    set_seed(seed)
    ensure_dir(out_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    n_classes = int(np.max(y)) + 1
    Z = np.asarray(Z, dtype=np.float32)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    model = MLPClassifier(Z.shape[1], n_classes, hidden_dim, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(Z[split.reftrain]).float(), torch.from_numpy(y[split.reftrain]).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_f1 = -1.0
    bad = 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(xb)
        metrics = evaluate_classifier(model, Z, y, split, device)
        history.append({"epoch": ep, "loss": total / max(len(train_ds), 1), **metrics})
        if metrics["val_f1"] > best_val_f1:
            best_val_f1 = metrics["val_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_classifier(model, Z, y, split, device)
    name = f"{prefix}_" if prefix else ""
    torch.save(model.state_dict(), os.path.join(out_dir, f"{name}downstream_mlp.pt"))
    with open(os.path.join(out_dir, f"{name}downstream_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, f"{name}metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def train_downstream_for_input_modes(
    feature_bank: Dict[str, np.ndarray],
    y: np.ndarray,
    split: SplitIndices,
    out_dir: str,
    input_modes: List[str],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    patience: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    """Train one identical MLP downstream classifier per input mode."""
    if len(input_modes) == 1 and input_modes[0] == "all":
        input_modes = ["latent", "raw_pca", "raw_plus_latent"]
    out = {}
    for mode in input_modes:
        if mode not in feature_bank:
            raise KeyError(f"Requested input mode {mode!r}, but available={sorted(feature_bank)}")
        mode_dir = os.path.join(out_dir, f"downstream_{mode}")
        print(f"[downstream] training MLP for input_mode={mode}, dim={feature_bank[mode].shape[1]}", flush=True)
        out[mode] = train_downstream_classifier(
            feature_bank[mode], y, split, mode_dir, seed=seed, epochs=epochs,
            batch_size=batch_size, lr=lr, hidden_dim=hidden_dim, patience=patience,
            device=device, prefix=mode,
        )
    with open(os.path.join(out_dir, "metrics_by_input.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def save_features(path: str, Z: np.ndarray, y: np.ndarray, u_label: np.ndarray, domains: np.ndarray, split: SplitIndices, extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    payload = dict(
        Z=np.asarray(Z, dtype=np.float32),
        y=y.astype(np.int64),
        u=u_label.astype(np.int64),
        domain=domains.astype(str),
        **split.as_dict(),
        z_reftrain=Z[split.reftrain].astype(np.float32),
        y_reftrain=y[split.reftrain].astype(np.int64),
        z_refval=Z[split.refval].astype(np.float32),
        y_refval=y[split.refval].astype(np.int64),
        z_targetadapt=Z[split.targetadapt].astype(np.float32),
        y_targetadapt=y[split.targetadapt].astype(np.int64),
        z_targettest=Z[split.targettest].astype(np.float32),
        y_targettest=y[split.targettest].astype(np.int64),
    )
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
