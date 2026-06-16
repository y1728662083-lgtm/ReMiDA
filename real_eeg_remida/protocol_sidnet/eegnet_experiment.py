from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from .config import ExperimentConfig
from .eegnet_model import EEGNet, categorical_cross_entropy
from .representation_fusion import concat_representations
from .utils import CSVHistoryWriter, BestCheckpointTracker, ensure_dir, multiclass_classification_metrics, set_global_seed, torch_device, write_json


@dataclass
class RepresentationStore:
    raw: np.ndarray
    aligned_raw: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    linear_latent: Optional[np.ndarray] = None


class EEGNetWrapper(torch.nn.Module):
    def __init__(self, n_classes: int, channels: int, samples: int, dropout: float = 0.5):
        super().__init__()
        self.net = EEGNet(n_classes=n_classes, channels=channels, samples=samples, dropoutRate=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _batch_iter(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(len(x), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        take = idx[start:start + batch_size]
        yield x[take], y[take]


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y.astype(np.int64)] = 1.0
    return out


def _standardize(train: np.ndarray, *others: np.ndarray, eps: float = 1e-6):
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + eps
    outs = [(train - mean) / std]
    for arr in others:
        outs.append((arr - mean) / std)
    return tuple(outs)


def _evaluate(model: EEGNetWrapper, x: np.ndarray, y: np.ndarray, device: torch.device):
    model.eval()
    probs_all = []
    with torch.no_grad():
        for xb, _ in _batch_iter(x, y, 64, False, 0):
            xt = torch.from_numpy(xb[:, None, :, :]).to(device)
            probs_all.append(model(xt).detach().cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    pred = probs.argmax(axis=1)
    metrics = multiclass_classification_metrics(y, pred)
    metrics["pred"] = pred
    metrics["probs"] = probs
    return metrics


def build_mode_input(store: RepresentationStore, mode: str) -> np.ndarray:
    samples = store.raw.shape[-1]
    if mode == "raw":
        return store.raw
    if mode == "aligned_raw":
        return store.aligned_raw
    if mode == "linear_latent":
        return concat_representations(None, store.linear_latent, None, samples)
    if mode == "raw_plus_linear_latent":
        return concat_representations(store.raw, store.linear_latent, None, samples)
    raise ValueError(mode)


def run_single_eegnet_mode(cfg: ExperimentConfig, store: RepresentationStore, mode: str, out_dir: str | Path) -> Dict[str, float]:
    set_global_seed(cfg.seed)
    out_dir = ensure_dir(out_dir)
    x_all = build_mode_input(store, mode)
    train_idx, val_idx, test_idx = store.train_idx, store.val_idx, store.test_idx
    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train, y_val, y_test = store.y[train_idx], store.y[val_idx], store.y[test_idx]
    x_train, x_val, x_test = _standardize(x_train, x_val, x_test)
    n_classes = int(np.max(store.y) + 1)
    dev = torch_device(cfg.device)
    model = EEGNetWrapper(n_classes=n_classes, channels=x_train.shape[1], samples=x_train.shape[2], dropout=cfg.eegnet.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.eegnet.lr, weight_decay=cfg.eegnet.weight_decay)
    tracker = BestCheckpointTracker(mode="max")
    history = CSVHistoryWriter(out_dir / "history.csv")
    best_state = None
    wait = 0
    for epoch in range(1, cfg.eegnet.epochs + 1):
        model.train()
        losses = []
        for xb, yb in _batch_iter(x_train, y_train, cfg.eegnet.batch_size, True, cfg.seed + epoch):
            xt = torch.from_numpy(xb[:, None, :, :]).to(dev)
            yt = torch.from_numpy(_one_hot(yb, n_classes)).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = categorical_cross_entropy(model(xt), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_eval = _evaluate(model, x_val, y_val, dev)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), "val_accuracy": float(val_eval["accuracy"]), "val_macro_f1": float(val_eval["macro_f1"]), "val_balanced_accuracy": float(val_eval["balanced_accuracy"])}
        history.write_row(row)
        if epoch == 1 or epoch % 5 == 0:
            print(f"[EEGNet:{mode}] epoch {epoch:03d} train_loss={row['train_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f}")
        if tracker.update(epoch, row["val_macro_f1"], out_dir / "best.pt"):
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.eegnet.patience:
                print(f"[EEGNet:{mode}] early stop at epoch {epoch}, best_epoch={tracker.best_epoch}, best_macro_f1={tracker.best_score:.4f}")
                break
    model.load_state_dict(best_state, strict=True)
    val_eval = _evaluate(model, x_val, y_val, dev)
    test_eval = _evaluate(model, x_test, y_test, dev)
    metrics = {
        "best_epoch": int(tracker.best_epoch),
        "val_accuracy": float(val_eval["accuracy"]),
        "val_macro_f1": float(val_eval["macro_f1"]),
        "val_balanced_accuracy": float(val_eval["balanced_accuracy"]),
        "test_accuracy": float(test_eval["accuracy"]),
        "test_macro_f1": float(test_eval["macro_f1"]),
        "test_balanced_accuracy": float(test_eval["balanced_accuracy"]),
    }
    write_json(metrics, out_dir / "metrics.json")
    cm = confusion_matrix(y_test, test_eval["pred"], labels=np.arange(n_classes, dtype=np.int64))
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(n_classes)], columns=[f"pred_{i}" for i in range(n_classes)]).to_csv(out_dir / "confusion_matrix.csv")
    pred_df = pd.DataFrame({"y_true": y_test.astype(np.int64), "y_pred": test_eval["pred"].astype(np.int64)})
    for i in range(n_classes):
        pred_df[f"prob_{i}"] = test_eval["probs"][:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    return metrics


def run_eegnet_suite(cfg: ExperimentConfig, store: RepresentationStore, out_dir: str | Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    rows = []
    for mode in cfg.eegnet.modes:
        metrics = run_single_eegnet_mode(cfg, store, mode, out_dir / mode)
        rows.append({"mode": mode, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison.csv", index=False)
    with (out_dir / "comparison.tex").open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
    return df
