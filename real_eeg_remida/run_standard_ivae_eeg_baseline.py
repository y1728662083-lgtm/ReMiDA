#!/usr/bin/env python3
"""
Standard iVAE baseline for real EEG protocols built by protocol_sidnet.

What this script does
---------------------
1) Loads the same real EEG protocol bundles used by the SID/ReMiDA pipeline.
2) Trains a standard cleanIVAE baseline on ref_train + target_adapt only.
   - No ReCA anchor loss.
   - No ReLiMDA / drift adapter.
   - No supervised labels are used by iVAE.
3) Converts time-point iVAE latents back to trial-level latents by averaging over time.
4) Runs the same downstream EEGNet-style classifier comparison on:
   - raw
   - standard_ivae_latent
   - raw_plus_standard_ivae_latent
5) Writes per-run and summary CSV/JSON files.

Expected project layout
-----------------------
Put this file in the project root, next to run_cross_subject.py / run_cross_session.py.
No scripts/ folder is required.

The script first tries to import the original `cleanIVAE` from an external iVAE repo.
If that import fails, it automatically uses a self-contained conditional VAE/iVAE fallback
implemented inside this file, so it can still run in the current SID-Net project.

Example
-------
python run_standard_ivae_eeg_baseline.py \
  --config configs/sub01_sub02_cross_sub_ses.yaml \
  --dataset-root /path/to/InnerSpeech2021 \
  --protocol both \
  --latent-dim 32 \
  --u-segments 20 \
  --ivae-epochs 120 \
  --out-dir outputs/standard_ivae_baseline
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Import helpers
# -----------------------------------------------------------------------------


def _add_path(path: Optional[str | Path]) -> None:
    if path is None:
        return
    p = Path(path).expanduser().resolve()
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _ensure_protocol_sidnet_importable(project_root: Optional[str | Path]) -> None:
    """Make protocol_sidnet importable.

    The normal case is a package directory named protocol_sidnet under project_root.
    A fallback is included for development folders that directly contain config.py/data.py
    and use relative imports inside those files.
    """
    if project_root is not None:
        _add_path(project_root)

    try:
        importlib.import_module("protocol_sidnet.config")
        importlib.import_module("protocol_sidnet.data")
        return
    except ModuleNotFoundError:
        pass

    root = Path(project_root or ".").expanduser().resolve()
    if (root / "protocol_sidnet").is_dir():
        _add_path(root)
        return

    # Fallback: synthesize a namespace package whose __path__ points to root.
    # This allows files such as data.py to resolve `from .config import ...`.
    if (root / "config.py").exists() and (root / "data.py").exists():
        import types

        pkg = types.ModuleType("protocol_sidnet")
        pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
        sys.modules.setdefault("protocol_sidnet", pkg)


def _import_protocol_modules(project_root: Optional[str | Path]):
    _ensure_protocol_sidnet_importable(project_root)
    from protocol_sidnet.config import load_config
    from protocol_sidnet.data import ProtocolRunBundle, build_protocol_collection
    from protocol_sidnet.eegnet_model import EEGNet, categorical_cross_entropy

    return load_config, build_protocol_collection, ProtocolRunBundle, EEGNet, categorical_cross_entropy


def _activation_layer(name: str, slope: float = 0.1) -> nn.Module:
    name = str(name).lower()
    if name in {"lrelu", "leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(float(slope))
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name in {"elu"}:
        return nn.ELU()
    if name in {"gelu"}:
        return nn.GELU()
    return nn.LeakyReLU(float(slope))


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, activation: str, slope: float = 0.1) -> nn.Sequential:
    layers: List[nn.Module] = []
    n_layers = max(1, int(n_layers))
    last = int(in_dim)
    for _ in range(n_layers):
        layers.append(nn.Linear(last, int(hidden_dim)))
        layers.append(_activation_layer(activation, slope=slope))
        last = int(hidden_dim)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class FallbackCleanIVAE(nn.Module):
    """Self-contained standard conditional VAE/iVAE fallback.

    It follows the iVAE idea used here for the EEG baseline: q(z|x,u), p(z|u), and p(x|z).
    No labels, ReCA anchors, drift adapters, or domain-specific mixing modules are used.
    The API intentionally matches the original cleanIVAE runner: `elbo(...)` and `forward(...)`.
    """

    def __init__(self, data_dim: int, latent_dim: int, aux_dim: int, hidden_dim: int = 128, n_layers: int = 2, activation: str = "lrelu", slope: float = 0.1):
        super().__init__()
        self.data_dim = int(data_dim)
        self.latent_dim = int(latent_dim)
        self.aux_dim = int(aux_dim)
        self.encoder = _make_mlp(self.data_dim + self.aux_dim, 2 * self.latent_dim, hidden_dim, n_layers, activation, slope=slope)
        self.prior = _make_mlp(self.aux_dim, 2 * self.latent_dim, hidden_dim, n_layers, activation, slope=slope)
        self.decoder = _make_mlp(self.latent_dim, self.data_dim, hidden_dim, n_layers, activation, slope=slope)

    def _posterior(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, u], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(logvar, -8.0, 8.0)

    def _prior(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.prior(u)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(logvar, -8.0, 8.0)

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        q_mu, q_logvar = self._posterior(x, u)
        z = self._sample(q_mu, q_logvar)
        x_mu = self.decoder(z)
        p_mu, p_logvar = self._prior(u)
        return x_mu, q_mu, q_logvar, z, (p_mu, p_logvar)

    def elbo(self, x: torch.Tensor, u: torch.Tensor, N: int, a: float = 1.0, b: float = 1.0, c: float = 0.0, d: float = 1.0):
        x_mu, q_mu, q_logvar, z, prior = self.forward(x, u)
        p_mu, p_logvar = prior
        recon = torch.nn.functional.mse_loss(x_mu, x, reduction="mean")
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * torch.mean(p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / (p_var + 1e-8) - 1.0)
        loss = float(a) * recon + float(b) * kl
        return loss, z


def _import_clean_ivae(ivae_root: Optional[str | Path], require_external: bool = False):
    _add_path(ivae_root)
    try:
        from models import cleanIVAE  # type: ignore
        print("[setup] using external models.cleanIVAE")
        return cleanIVAE
    except Exception as exc:
        if require_external:
            raise ImportError(
                "Cannot import `cleanIVAE` from `models`. Pass --ivae-root pointing to the original iVAE repo "
                "or remove --require-external-cleanivae to use the built-in fallback."
            ) from exc
        print("[setup] external cleanIVAE not found; using built-in FallbackCleanIVAE")
        return FallbackCleanIVAE


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
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


def resolve_device(device: str | torch.device) -> torch.device:
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev)
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def csv_write_row(path: str | Path, row: Dict[str, Any], append: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists or not append:
            writer.writeheader()
        writer.writerow(row)


def one_hot(ids: np.ndarray, n_classes: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    out = np.zeros((len(ids), int(n_classes)), dtype=np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
    }


# -----------------------------------------------------------------------------
# Split handling and standardization
# -----------------------------------------------------------------------------


def stratified_split_indices(
    indices: np.ndarray,
    y: np.ndarray,
    first_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-stratified split of local trial indices into first/second parts."""
    rng = np.random.default_rng(int(seed))
    indices = np.asarray(indices, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    first_parts: List[np.ndarray] = []
    second_parts: List[np.ndarray] = []
    for cls in np.unique(y[indices]):
        idx = indices[y[indices] == cls].copy()
        rng.shuffle(idx)
        n = len(idx)
        if n <= 1:
            first_parts.append(idx)
            continue
        n_first = int(round(n * float(first_ratio)))
        n_first = min(max(n_first, 1), n - 1)
        first_parts.append(idx[:n_first])
        second_parts.append(idx[n_first:])
    first = np.sort(np.concatenate(first_parts)) if first_parts else np.array([], dtype=np.int64)
    second = np.sort(np.concatenate(second_parts)) if second_parts else np.array([], dtype=np.int64)
    return first, second


@dataclass
class EffectiveSplits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    adapt_idx: np.ndarray
    test_idx: np.ndarray
    repaired: bool
    overlap_before: int


def make_effective_splits(bundle: Any, adapt_ratio: float, seed: int, repair_overlap: bool = True) -> EffectiveSplits:
    """Return local train/val/adapt/test indices.

    Some existing protocol builders use the full target set as both adapt and test.
    This function optionally repairs that into a class-stratified target_adapt/target_test split.
    """
    train_idx = np.asarray(bundle.train_idx, dtype=np.int64)
    val_idx = np.asarray(bundle.val_idx, dtype=np.int64)
    adapt_idx = np.asarray(bundle.adapt_idx, dtype=np.int64)
    test_idx = np.asarray(bundle.test_idx, dtype=np.int64)
    overlap = int(len(np.intersect1d(adapt_idx, test_idx)))

    if repair_overlap and overlap > 0:
        pool = np.unique(np.concatenate([adapt_idx, test_idx]))
        adapt_idx, test_idx = stratified_split_indices(pool, bundle.y, first_ratio=adapt_ratio, seed=seed)
        repaired = True
    else:
        repaired = False

    return EffectiveSplits(
        train_idx=train_idx,
        val_idx=val_idx,
        adapt_idx=adapt_idx,
        test_idx=test_idx,
        repaired=repaired,
        overlap_before=overlap,
    )


def splitwise_standardize_trials(
    x: np.ndarray,
    entity_ids: np.ndarray,
    train_idx: np.ndarray,
    adapt_idx: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """Standardize [N,C,T] per entity without using target_test.

    For each entity:
      - if it has ref_train samples, fit mean/std on those;
      - otherwise fit on target_adapt samples;
      - if neither exists, fall back to all samples of that entity and mark fallback=True.
    """
    x = np.asarray(x, dtype=np.float32)
    ent = np.asarray(entity_ids, dtype=np.int64)
    out = np.empty_like(x, dtype=np.float32)
    stats: Dict[int, Dict[str, Any]] = {}
    train_idx = np.asarray(train_idx, dtype=np.int64)
    adapt_idx = np.asarray(adapt_idx, dtype=np.int64)

    for eid in np.unique(ent):
        mask_all = np.where(ent == int(eid))[0]
        idx_train = np.intersect1d(mask_all, train_idx)
        idx_adapt = np.intersect1d(mask_all, adapt_idx)
        fallback = False
        if len(idx_train) > 0:
            fit_idx = idx_train
            fit_split = "train"
        elif len(idx_adapt) > 0:
            fit_idx = idx_adapt
            fit_split = "adapt"
        else:
            fit_idx = mask_all
            fit_split = "all_fallback"
            fallback = True
        mean = x[fit_idx].mean(axis=(0, 2), keepdims=True)
        std = x[fit_idx].std(axis=(0, 2), keepdims=True)
        std = np.maximum(std, eps)
        out[mask_all] = (x[mask_all] - mean) / std
        stats[int(eid)] = {
            "fit_split": fit_split,
            "n_fit_trials": int(len(fit_idx)),
            "fallback": bool(fallback),
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        }
    return out.astype(np.float32), stats


# -----------------------------------------------------------------------------
# Time-point conversion for iVAE
# -----------------------------------------------------------------------------


@dataclass
class TimepointBlock:
    X: np.ndarray
    U: np.ndarray
    trial_id: np.ndarray
    y: np.ndarray
    z: np.ndarray
    segment_id: np.ndarray


def make_time_segments(n_times: int, requested_segments: int) -> Tuple[np.ndarray, int]:
    k = int(max(1, min(int(requested_segments), int(n_times))))
    seg = np.floor(np.arange(int(n_times), dtype=np.float64) * k / float(n_times)).astype(np.int64)
    seg = np.clip(seg, 0, k - 1)
    return seg, k


def trials_to_timepoint_block(
    x_trials: np.ndarray,
    trial_indices: np.ndarray,
    y_trial: np.ndarray,
    domain_trial: np.ndarray,
    n_u_segments: int,
) -> TimepointBlock:
    """Convert [N_trial,C,T] to time-point samples [N_trial*T,C]."""
    trial_indices = np.asarray(trial_indices, dtype=np.int64)
    x_sel = np.asarray(x_trials[trial_indices], dtype=np.float32)
    n_trials, n_channels, n_times = x_sel.shape
    seg, k = make_time_segments(n_times, n_u_segments)

    # [N,C,T] -> [N,T,C] -> [N*T,C]
    X = np.transpose(x_sel, (0, 2, 1)).reshape(n_trials * n_times, n_channels).astype(np.float32)
    segment_id = np.tile(seg, n_trials).astype(np.int64)
    U = one_hot(segment_id, k)
    trial_id = np.repeat(trial_indices, n_times).astype(np.int64)
    y = np.repeat(np.asarray(y_trial[trial_indices], dtype=np.int64), n_times).astype(np.int64)
    z = np.repeat(np.asarray(domain_trial[trial_indices], dtype=np.int64), n_times).astype(np.int64)
    return TimepointBlock(X=X, U=U, trial_id=trial_id, y=y, z=z, segment_id=segment_id)


# -----------------------------------------------------------------------------
# iVAE training and inference
# -----------------------------------------------------------------------------


def train_clean_ivae(
    cleanIVAE_cls: Any,
    X_train: np.ndarray,
    U_train: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    n_layers: int,
    activation: str,
    lr: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
    out_dir: Path,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 0.0,
    d: float = 1.0,
    patience: int = 0,
    grad_clip: float = 0.0,
    num_workers: int = 0,
) -> Tuple[nn.Module, List[float]]:
    set_seed(seed)
    out_dir = ensure_dir(out_dir)
    X_train = np.asarray(X_train, dtype=np.float32)
    U_train = np.asarray(U_train, dtype=np.float32)
    n_samples, data_dim = X_train.shape
    aux_dim = U_train.shape[1]

    model = cleanIVAE_cls(
        data_dim=int(data_dim),
        latent_dim=int(latent_dim),
        aux_dim=int(aux_dim),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        activation=str(activation),
        slope=0.1,
    ).to(device)

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(U_train))
    loader_params = {
        "batch_size": int(batch_size),
        "shuffle": True,
        "drop_last": False,
        "num_workers": int(num_workers) if device.type == "cuda" else 0,
        "pin_memory": bool(device.type == "cuda"),
    }
    loader = DataLoader(ds, **loader_params)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    loss_history: List[float] = []
    best_loss = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: List[float] = []
        for xb, ub in loader:
            xb = xb.to(device)
            ub = ub.to(device)
            opt.zero_grad(set_to_none=True)
            loss, _ = model.elbo(xb, ub, n_samples, a=float(a), b=float(b), c=float(c), d=float(d))
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        loss_history.append(epoch_loss)
        csv_write_row(out_dir / "ivae_history.csv", {"epoch": epoch, "train_loss": epoch_loss}, append=True)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(epochs):
            print(f"[iVAE] epoch {epoch:03d}/{epochs} train_loss={epoch_loss:.6f}")

        if np.isfinite(epoch_loss) and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state": best_state, "train_loss": best_loss}, out_dir / "ivae_best.pt")
            wait = 0
        else:
            wait += 1
            if patience and wait >= int(patience):
                print(f"[iVAE] early stop at epoch {epoch}; best_loss={best_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, loss_history


@torch.no_grad()
def encode_clean_ivae_timepoints(
    model: nn.Module,
    X: np.ndarray,
    U: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    U = np.asarray(U, dtype=np.float32)
    outs: List[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        xb = torch.from_numpy(X[start : start + int(batch_size)]).to(device)
        ub = torch.from_numpy(U[start : start + int(batch_size)]).to(device)
        # cleanIVAE forward API in the existing runner returns (_, _, _, s, _).
        try:
            _, _, _, s, _ = model(xb, ub)
        except ValueError:
            # Fallback for variants returning four values.
            out = model(xb, ub)
            s = out[-2] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
        outs.append(s.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0).astype(np.float32)


def aggregate_timepoint_latents_to_trials(
    s_time: np.ndarray,
    trial_id: np.ndarray,
    n_trials_total: int,
) -> np.ndarray:
    s_time = np.asarray(s_time, dtype=np.float32)
    trial_id = np.asarray(trial_id, dtype=np.int64).reshape(-1)
    if len(s_time) != len(trial_id):
        raise ValueError(f"s_time and trial_id length mismatch: {len(s_time)} vs {len(trial_id)}")
    latent_dim = s_time.shape[1]
    sums = np.zeros((int(n_trials_total), latent_dim), dtype=np.float64)
    counts = np.zeros((int(n_trials_total),), dtype=np.float64)
    np.add.at(sums, trial_id, s_time.astype(np.float64))
    np.add.at(counts, trial_id, 1.0)
    counts = np.maximum(counts, 1.0)
    return (sums / counts[:, None]).astype(np.float32)


# -----------------------------------------------------------------------------
# EEGNet downstream classifier
# -----------------------------------------------------------------------------


def latent_to_pseudo_timeseries(latent: np.ndarray, samples: int) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim == 3:
        latent = latent.reshape(latent.shape[0], -1)
    return np.repeat(latent[:, :, None], repeats=int(samples), axis=2).astype(np.float32)


def build_downstream_input(raw: np.ndarray, latent: np.ndarray, mode: str) -> np.ndarray:
    samples = raw.shape[-1]
    if mode == "raw":
        return np.asarray(raw, dtype=np.float32)
    if mode == "standard_ivae_latent":
        return latent_to_pseudo_timeseries(latent, samples)
    if mode == "raw_plus_standard_ivae_latent":
        return np.concatenate([np.asarray(raw, dtype=np.float32), latent_to_pseudo_timeseries(latent, samples)], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported mode={mode}")


def standardize_eegnet_input(train: np.ndarray, *others: np.ndarray, eps: float = 1e-6):
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + float(eps)
    outs = [(train - mean) / std]
    for arr in others:
        outs.append((arr - mean) / std)
    return tuple(outs)


def iter_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(len(x), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
    for start in range(0, len(idx), int(batch_size)):
        take = idx[start : start + int(batch_size)]
        yield x[take], y[take]


@torch.no_grad()
def evaluate_eegnet(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> Dict[str, Any]:
    model.eval()
    probs_all: List[np.ndarray] = []
    for xb, _ in iter_batches(x, y, batch_size=64, shuffle=False, seed=0):
        xt = torch.from_numpy(xb[:, None, :, :].astype(np.float32)).to(device)
        probs_all.append(model(xt).detach().cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    pred = probs.argmax(axis=1)
    metrics = classification_metrics(y, pred)
    metrics["pred"] = pred
    metrics["probs"] = probs
    return metrics


def run_eegnet_mode(
    EEGNet_cls: Any,
    categorical_cross_entropy_fn: Any,
    raw: np.ndarray,
    latent: np.ndarray,
    y: np.ndarray,
    splits: EffectiveSplits,
    mode: str,
    out_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    dropout: float,
) -> Dict[str, float]:
    set_seed(seed)
    out_dir = ensure_dir(out_dir)
    x_all = build_downstream_input(raw=raw, latent=latent, mode=mode)
    train_idx, val_idx, test_idx = splits.train_idx, splits.val_idx, splits.test_idx
    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    x_train, x_val, x_test = standardize_eegnet_input(x_train, x_val, x_test)

    n_classes = int(np.max(y) + 1)
    model = EEGNet_cls(
        n_classes=n_classes,
        channels=int(x_train.shape[1]),
        samples=int(x_train.shape[2]),
        dropoutRate=float(dropout),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_f1 = -math.inf
    best_epoch = -1
    wait = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: List[float] = []
        for xb, yb in iter_batches(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed + epoch):
            xt = torch.from_numpy(xb[:, None, :, :].astype(np.float32)).to(device)
            yt = torch.from_numpy(one_hot(yb, n_classes)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = categorical_cross_entropy_fn(model(xt), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        val_eval = evaluate_eegnet(model, x_val, y_val, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_accuracy": float(val_eval["accuracy"]),
            "val_macro_f1": float(val_eval["macro_f1"]),
            "val_balanced_accuracy": float(val_eval["balanced_accuracy"]),
        }
        csv_write_row(out_dir / "history.csv", row, append=True)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(epochs):
            print(f"[EEGNet:{mode}] epoch {epoch:03d}/{epochs} val_macro_f1={row['val_macro_f1']:.4f}")

        if row["val_macro_f1"] > best_val_f1:
            best_val_f1 = float(row["val_macro_f1"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state": best_state, "val_macro_f1": best_val_f1}, out_dir / "best.pt")
            wait = 0
        else:
            wait += 1
            if int(patience) > 0 and wait >= int(patience):
                print(f"[EEGNet:{mode}] early stop at epoch {epoch}; best_epoch={best_epoch}; best_val_f1={best_val_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    val_eval = evaluate_eegnet(model, x_val, y_val, device)
    test_eval = evaluate_eegnet(model, x_test, y_test, device)
    metrics = {
        "best_epoch": int(best_epoch),
        "val_accuracy": float(val_eval["accuracy"]),
        "val_macro_f1": float(val_eval["macro_f1"]),
        "val_balanced_accuracy": float(val_eval["balanced_accuracy"]),
        "test_accuracy": float(test_eval["accuracy"]),
        "test_macro_f1": float(test_eval["macro_f1"]),
        "test_balanced_accuracy": float(test_eval["balanced_accuracy"]),
    }
    write_json(metrics, out_dir / "metrics.json")

    labels = np.arange(n_classes, dtype=np.int64)
    cm = confusion_matrix(y_test, test_eval["pred"], labels=labels)
    pd.DataFrame(cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels]).to_csv(out_dir / "confusion_matrix.csv")
    pred_df = pd.DataFrame({"y_true": y_test.astype(np.int64), "y_pred": test_eval["pred"].astype(np.int64)})
    for i in labels:
        pred_df[f"prob_{int(i)}"] = test_eval["probs"][:, int(i)]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    return metrics


# -----------------------------------------------------------------------------
# Per-run orchestration
# -----------------------------------------------------------------------------


def summarize_group(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    key_cols = [c for c in key_cols if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in set(key_cols) and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df[key_cols].drop_duplicates().reset_index(drop=True)
    agg = df.groupby(key_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["__".join([str(x) for x in col if x]) if isinstance(col, tuple) else str(col) for col in agg.columns.to_flat_index()]
    return agg


def run_one_bundle(
    bundle: Any,
    args: argparse.Namespace,
    cfg: Any,
    cleanIVAE_cls: Any,
    EEGNet_cls: Any,
    categorical_cross_entropy_fn: Any,
    device: torch.device,
    out_root: Path,
) -> pd.DataFrame:
    run_dir = ensure_dir(out_root / bundle.protocol_name / bundle.run_name)
    seed = int(args.seed if args.seed is not None else getattr(cfg, "seed", 42))
    print(f"\n[pipeline] {bundle.protocol_name} | {bundle.run_name}")

    splits = make_effective_splits(
        bundle,
        adapt_ratio=float(args.target_adapt_ratio),
        seed=seed,
        repair_overlap=not bool(args.no_repair_adapt_test_overlap),
    )
    if splits.repaired:
        print(
            f"[split] repaired adapt/test overlap={splits.overlap_before}: "
            f"target_adapt={len(splits.adapt_idx)}, target_test={len(splits.test_idx)}"
        )

    x_std, standardizer_info = splitwise_standardize_trials(
        x=bundle.x_raw,
        entity_ids=bundle.entity_ids,
        train_idx=splits.train_idx,
        adapt_idx=splits.adapt_idx,
    )

    train_trials_for_ivae = np.sort(np.unique(np.concatenate([splits.train_idx, splits.adapt_idx]))).astype(np.int64)
    train_block = trials_to_timepoint_block(
        x_trials=x_std,
        trial_indices=train_trials_for_ivae,
        y_trial=bundle.y,
        domain_trial=bundle.domain_ids,
        n_u_segments=int(args.u_segments),
    )
    latent_dim = int(args.latent_dim) if int(args.latent_dim) > 0 else min(32, train_block.X.shape[1])

    write_json(
        {
            "protocol": bundle.protocol_name,
            "run_name": bundle.run_name,
            "reference_entity": bundle.reference_entity,
            "n_trials_total": int(len(bundle.x_raw)),
            "n_train_trials": int(len(splits.train_idx)),
            "n_val_trials": int(len(splits.val_idx)),
            "n_adapt_trials": int(len(splits.adapt_idx)),
            "n_test_trials": int(len(splits.test_idx)),
            "adapt_test_overlap_before": int(splits.overlap_before),
            "adapt_test_repaired": bool(splits.repaired),
            "iVAE_train_trials": int(len(train_trials_for_ivae)),
            "iVAE_train_timepoints": int(len(train_block.X)),
            "data_dim": int(train_block.X.shape[1]),
            "latent_dim": int(latent_dim),
            "aux_dim": int(train_block.U.shape[1]),
            "standardizer_info": standardizer_info,
        },
        run_dir / "run_setup.json",
    )

    model, loss_hist = train_clean_ivae(
        cleanIVAE_cls=cleanIVAE_cls,
        X_train=train_block.X,
        U_train=train_block.U,
        latent_dim=latent_dim,
        hidden_dim=int(args.ivae_hidden_dim),
        n_layers=int(args.ivae_layers),
        activation=str(args.ivae_activation),
        lr=float(args.ivae_lr),
        batch_size=int(args.ivae_batch_size),
        epochs=int(args.ivae_epochs),
        device=device,
        seed=seed,
        out_dir=run_dir / "standard_ivae",
        a=float(args.ivae_a),
        b=float(args.ivae_b),
        c=float(args.ivae_c),
        d=float(args.ivae_d),
        patience=int(args.ivae_patience),
        grad_clip=float(args.ivae_grad_clip),
        num_workers=int(args.num_workers),
    )

    all_trials = np.arange(len(bundle.x_raw), dtype=np.int64)
    full_block = trials_to_timepoint_block(
        x_trials=x_std,
        trial_indices=all_trials,
        y_trial=bundle.y,
        domain_trial=bundle.domain_ids,
        n_u_segments=int(args.u_segments),
    )
    s_time = encode_clean_ivae_timepoints(
        model=model,
        X=full_block.X,
        U=full_block.U,
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    latent_trial = aggregate_timepoint_latents_to_trials(s_time, full_block.trial_id, n_trials_total=len(bundle.x_raw))

    np.save(run_dir / "standard_ivae_latent_trial.npy", latent_trial.astype(np.float32))
    # Save time-point latents only if requested because it can be large.
    if bool(args.save_timepoint_latents):
        np.save(run_dir / "standard_ivae_latent_timepoint.npy", s_time.astype(np.float32))
    manifest = bundle.metadata.copy().reset_index(drop=True)
    manifest["effective_split"] = "unused"
    manifest.loc[splits.train_idx, "effective_split"] = "train"
    manifest.loc[splits.val_idx, "effective_split"] = "val"
    manifest.loc[splits.adapt_idx, "effective_split"] = "adapt"
    manifest.loc[splits.test_idx, "effective_split"] = "test"
    manifest.to_csv(run_dir / "manifest_with_effective_split.csv", index=False)

    modes = ["raw", "standard_ivae_latent", "raw_plus_standard_ivae_latent"]
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        metrics = run_eegnet_mode(
            EEGNet_cls=EEGNet_cls,
            categorical_cross_entropy_fn=categorical_cross_entropy_fn,
            raw=bundle.x_raw.astype(np.float32),
            latent=latent_trial.astype(np.float32),
            y=bundle.y.astype(np.int64),
            splits=splits,
            mode=mode,
            out_dir=run_dir / "eegnet" / mode,
            device=device,
            seed=seed,
            epochs=int(args.eegnet_epochs if args.eegnet_epochs is not None else cfg.eegnet.epochs),
            batch_size=int(args.eegnet_batch_size if args.eegnet_batch_size is not None else cfg.eegnet.batch_size),
            lr=float(args.eegnet_lr if args.eegnet_lr is not None else cfg.eegnet.lr),
            weight_decay=float(args.eegnet_weight_decay if args.eegnet_weight_decay is not None else cfg.eegnet.weight_decay),
            patience=int(args.eegnet_patience if args.eegnet_patience is not None else cfg.eegnet.patience),
            dropout=float(args.eegnet_dropout if args.eegnet_dropout is not None else cfg.eegnet.dropout),
        )
        rows.append({
            "protocol": bundle.protocol_name,
            "run_name": bundle.run_name,
            "reference_entity": bundle.reference_entity,
            "mode": mode,
            **metrics,
        })

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "eegnet_comparison.csv", index=False)
    write_json(
        {
            "loss_history_last": float(loss_hist[-1]) if loss_hist else None,
            "outputs": {
                "latent_trial": "standard_ivae_latent_trial.npy",
                "comparison": "eegnet_comparison.csv",
            },
        },
        run_dir / "standard_ivae_summary.json",
    )
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a standard iVAE real-EEG baseline and downstream EEGNet comparison.")

    p.add_argument("--config", type=str, required=True, help="Path to the SID/protocol YAML config.")
    p.add_argument("--dataset-root", type=str, default=None, help="Override cfg.dataset.dataset_root.")
    p.add_argument("--project-root", type=str, default=".", help="Project root containing protocol_sidnet package. Default: current directory.")
    p.add_argument("--ivae-root", type=str, default=None, help="Optional path to original iVAE repo containing models.cleanIVAE. If omitted/unavailable, a built-in fallback is used.")
    p.add_argument("--require-external-cleanivae", action="store_true", help="Fail instead of using the built-in fallback when external models.cleanIVAE is unavailable.")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory. Default: cfg.output.out_dir/standard_ivae_baseline.")
    p.add_argument("--protocol", choices=["cross_subject", "cross_session", "both"], default="both")
    p.add_argument("--max-runs", type=int, default=0, help="Debug: run only the first N bundles across selected protocols. 0 means all.")
    p.add_argument("--seed", type=int, default=None, help="Override cfg.seed.")
    p.add_argument("--device", type=str, default=None, help="Override cfg.device.")
    p.add_argument("--num-workers", type=int, default=0)

    # Split handling
    p.add_argument("--target-adapt-ratio", type=float, default=0.5, help="Used when repairing overlapping target_adapt/target_test.")
    p.add_argument("--no-repair-adapt-test-overlap", action="store_true", help="Disable repair when adapt_idx and test_idx overlap.")

    # iVAE baseline
    p.add_argument("--latent-dim", type=int, default=32, help="iVAE latent dimension. Use <=0 to default to min(32, channels).")
    p.add_argument("--u-segments", type=int, default=20, help="Number of shared time-bin auxiliary conditions U within each trial.")
    p.add_argument("--ivae-epochs", type=int, default=120)
    p.add_argument("--ivae-batch-size", type=int, default=4096)
    p.add_argument("--ivae-lr", type=float, default=1e-3)
    p.add_argument("--ivae-hidden-dim", type=int, default=128)
    p.add_argument("--ivae-layers", type=int, default=2)
    p.add_argument("--ivae-activation", type=str, default="lrelu")
    p.add_argument("--ivae-a", type=float, default=1.0)
    p.add_argument("--ivae-b", type=float, default=1.0)
    p.add_argument("--ivae-c", type=float, default=0.0)
    p.add_argument("--ivae-d", type=float, default=1.0)
    p.add_argument("--ivae-patience", type=int, default=0, help="0 disables early stopping for iVAE.")
    p.add_argument("--ivae-grad-clip", type=float, default=0.0, help="0 disables gradient clipping for iVAE.")
    p.add_argument("--eval-batch-size", type=int, default=8192)
    p.add_argument("--save-timepoint-latents", action="store_true", help="Also save [N*T, latent_dim] latents. Can be large.")

    # EEGNet downstream classifier; default to config values when None.
    p.add_argument("--eegnet-epochs", type=int, default=None)
    p.add_argument("--eegnet-batch-size", type=int, default=None)
    p.add_argument("--eegnet-lr", type=float, default=None)
    p.add_argument("--eegnet-weight-decay", type=float, default=None)
    p.add_argument("--eegnet-patience", type=int, default=None)
    p.add_argument("--eegnet-dropout", type=float, default=None)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    load_config, build_protocol_collection, _, EEGNet_cls, categorical_cross_entropy_fn = _import_protocol_modules(args.project_root)
    cleanIVAE_cls = _import_clean_ivae(args.ivae_root, require_external=bool(args.require_external_cleanivae))

    overrides: Dict[str, Any] = {}
    if args.dataset_root is not None:
        overrides.setdefault("dataset", {})["dataset_root"] = args.dataset_root
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = int(args.seed)

    cfg = load_config(args.config, overrides=overrides)
    seed = int(args.seed if args.seed is not None else getattr(cfg, "seed", 42))
    set_seed(seed)
    device = resolve_device(args.device if args.device is not None else getattr(cfg, "device", "cuda"))
    cfg.device = str(device)

    out_root = Path(args.out_dir) if args.out_dir is not None else Path(cfg.output.out_dir) / "standard_ivae_baseline"
    out_root = ensure_dir(out_root)
    print(f"[setup] config={args.config}")
    print(f"[setup] dataset_root={cfg.dataset.dataset_root}")
    print(f"[setup] output={out_root}")
    print(f"[setup] device={device}")

    collection = build_protocol_collection(cfg)
    bundles: List[Any] = []
    if args.protocol in {"cross_subject", "both"}:
        bundles.extend(collection.cross_subject_runs)
    if args.protocol in {"cross_session", "both"}:
        bundles.extend(collection.cross_session_runs)
    if int(args.max_runs) > 0:
        bundles = bundles[: int(args.max_runs)]
    if not bundles:
        raise RuntimeError(f"No protocol bundles selected for protocol={args.protocol}")

    write_json(
        {
            "config": args.config,
            "dataset_root": cfg.dataset.dataset_root,
            "protocol": args.protocol,
            "n_bundles": len(bundles),
            "seed": seed,
            "device": str(device),
            "modes": ["raw", "standard_ivae_latent", "raw_plus_standard_ivae_latent"],
            "note": "standard cleanIVAE baseline; no ReCA anchor, no drift adapter, no labels in iVAE training",
        },
        out_root / "experiment_setup.json",
    )

    all_rows: List[pd.DataFrame] = []
    for bundle in bundles:
        df = run_one_bundle(
            bundle=bundle,
            args=args,
            cfg=cfg,
            cleanIVAE_cls=cleanIVAE_cls,
            EEGNet_cls=EEGNet_cls,
            categorical_cross_entropy_fn=categorical_cross_entropy_fn,
            device=device,
            out_root=out_root,
        )
        all_rows.append(df)

    all_df = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    all_df.to_csv(out_root / "eegnet_metrics_per_run.csv", index=False)
    summary = summarize_group(all_df, key_cols=["protocol", "mode"])
    summary.to_csv(out_root / "eegnet_metrics_summary.csv", index=False)
    try:
        with (out_root / "eegnet_metrics_summary.tex").open("w", encoding="utf-8") as f:
            f.write(summary.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
    except Exception as exc:
        print(f"[warn] failed to write LaTeX summary: {exc}")

    print("\n[done] Summary:")
    print(summary.to_string(index=False))
    print(f"[done] outputs saved under: {out_root}")
    print(f"[done] runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
