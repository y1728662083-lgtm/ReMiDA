from __future__ import annotations

import copy
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import ProtocolRunBundle
from .export import save_epochs_fif, save_latent_npy
from .front_linear import covariance_match_initializer_by_domain
from .linear_model import LinearJointSIDNet
from .losses import band_alignment_loss, class_center_loss, kl_normal, smooth_l1_recon
from .metrics import compute_linear_proxy_metrics_for_pair, linear_val_score
from .utils import BestCheckpointTracker, CSVHistoryWriter, JSONLLogger, ensure_dir, set_global_seed, torch_device, write_json


@dataclass
class LinearTrainArtifacts:
    run_dir: Path
    metrics_val: Dict[str, float]
    metrics_test: Dict[str, float]
    aligned_all: np.ndarray
    latent_all: np.ndarray
    bundle: ProtocolRunBundle


class _SourceDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, entity_ids: np.ndarray, domain_ids: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.entity_ids = entity_ids.astype(np.int64)
        self.domain_ids = domain_ids.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
            torch.tensor(int(self.entity_ids[idx]), dtype=torch.long),
            torch.tensor(int(self.domain_ids[idx]), dtype=torch.long),
        )


class _AdaptDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, entity_ids: np.ndarray, domain_ids: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.entity_ids = entity_ids.astype(np.int64)
        self.domain_ids = domain_ids.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(int(self.entity_ids[idx]), dtype=torch.long),
            torch.tensor(int(self.domain_ids[idx]), dtype=torch.long),
        )


def _macro_f1_from_pred(pred: np.ndarray, y: np.ndarray) -> float:
    parts = []
    for cls in np.unique(y):
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        parts.append(0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec))
    return float(np.mean(parts)) if parts else 0.0


def _forward_collect(model: LinearJointSIDNet, x: np.ndarray, domain_ids: np.ndarray, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    outputs = []
    with torch.no_grad():
        bs = 128
        for start in range(0, len(x), bs):
            xb = torch.from_numpy(x[start:start + bs]).to(device)
            db = torch.from_numpy(domain_ids[start:start + bs]).to(device)
            out = model(xb, db, grl_lambda=1.0)
            outputs.append({k: v.detach().cpu().numpy() for k, v in out.items()})
    merged = {k: np.concatenate([o[k] for o in outputs], axis=0) for k in outputs[0].keys()}
    return merged


def _summarize_classification(logits: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = logits.argmax(axis=1)
    return {
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": _macro_f1_from_pred(pred, y),
    }


def _proxy_metrics(bundle: ProtocolRunBundle, aligned: np.ndarray, subset_idx: np.ndarray) -> Dict[str, float]:
    meta = bundle.metadata.iloc[np.asarray(subset_idx, dtype=np.int64)].reset_index(drop=True)
    x = aligned[np.asarray(subset_idx, dtype=np.int64)]
    ent_arr = meta["entity_key"].astype(str).to_numpy()
    ref_mask = ent_arr == str(bundle.reference_entity)
    if not np.any(ref_mask):
        raise ValueError(f"Reference entity {bundle.reference_entity} absent in subset for {bundle.run_name}")
    x_ref = x[ref_mask]
    metrics = []
    for ent in sorted(set(ent_arr.tolist())):
        if ent == bundle.reference_entity:
            continue
        mask = ent_arr == str(ent)
        if np.any(mask):
            metrics.append(compute_linear_proxy_metrics_for_pair(x_ref, x[mask], sfreq=bundle.sfreq, seed=42))
    if not metrics:
        return compute_linear_proxy_metrics_for_pair(x_ref, x_ref, sfreq=bundle.sfreq, seed=42)
    out: Dict[str, float] = {}
    for key in metrics[0].keys():
        out[key] = float(np.mean([m[key] for m in metrics]))
    return out


def _save_checkpoint(path: Path, model: LinearJointSIDNet, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    try:
        torch.save(payload, path)
    except Exception as exc:
        warnings.warn(f"Checkpoint save failed at {path}: {exc}")


def train_linear_protocol_run(cfg: ExperimentConfig, bundle: ProtocolRunBundle) -> LinearTrainArtifacts:
    set_global_seed(cfg.seed)
    out_dir = ensure_dir(cfg.run_dir / bundle.protocol_name / bundle.run_name)
    device = torch_device(cfg.device)

    standardizers = bundle.fit_standardizers()
    x_std = bundle.standardized_raw(standardizers)
    ref_eid = bundle.entity_to_id[bundle.reference_entity]
    ref_train_global = bundle.train_idx[bundle.entity_ids[bundle.train_idx] == ref_eid]
    if len(ref_train_global) == 0:
        ref_train_global = bundle.train_idx
    init = covariance_match_initializer_by_domain(
        x_std=x_std,
        domain_ids=bundle.domain_ids,
        reference_indices=ref_train_global,
        domain_keys=bundle.id_to_domain,
        eps=1e-5,
    )

    cfg_local = copy.deepcopy(cfg)
    cfg_local.dataset.reference_subject = bundle.reference_entity
    model = LinearJointSIDNet(
        cfg=cfg_local,
        n_channels=x_std.shape[1],
        n_times=x_std.shape[2],
        subjects=bundle.entity_to_id,
        domains=bundle.domain_to_id,
        domain_to_subject=bundle.domain_to_entity,
        n_classes=int(np.max(bundle.y) + 1),
    ).to(device)
    model.front.initialize(init)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    source_ds = _SourceDataset(x_std[bundle.train_idx], bundle.y[bundle.train_idx], bundle.entity_ids[bundle.train_idx], bundle.domain_ids[bundle.train_idx])
    adapt_ds = _AdaptDataset(x_std[bundle.adapt_idx], bundle.entity_ids[bundle.adapt_idx], bundle.domain_ids[bundle.adapt_idx])
    source_loader = torch.utils.data.DataLoader(source_ds, batch_size=cfg.training.batch_size_source, shuffle=True, num_workers=cfg.training.num_workers, drop_last=False)
    adapt_loader = torch.utils.data.DataLoader(adapt_ds, batch_size=cfg.training.batch_size_adapt, shuffle=True, num_workers=cfg.training.num_workers, drop_last=False)

    history = CSVHistoryWriter(out_dir / "history.csv")
    jsonl = JSONLLogger(out_dir / "epoch_metrics.jsonl")
    tracker = BestCheckpointTracker(mode="min")
    best_state = None
    wait = 0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        meter = {k: [] for k in ["total", "recon", "kl", "class", "domain", "band", "center", "front_cov", "front_ortho", "front_tether", "front_delta"]}

        src_iter = itertools.cycle(source_loader)
        adp_iter = itertools.cycle(adapt_loader)
        n_steps = max(len(source_loader), len(adapt_loader))
        for _ in range(n_steps):
            xs, ys, es, ds = next(src_iter)
            xa, ea, da = next(adp_iter)
            xb = torch.cat([xs, xa], dim=0).to(device)
            eb = torch.cat([es, ea], dim=0).to(device)
            db = torch.cat([ds, da], dim=0).to(device)
            yb = ys.to(device)
            n_src = xs.shape[0]

            opt.zero_grad(set_to_none=True)
            out = model(xb, db, grl_lambda=1.0)
            recon = smooth_l1_recon(out["x_recon"], out["x_aligned"].detach())
            kl = kl_normal(out["mu"], out["logvar"])
            cls = F.cross_entropy(out["class_logits"][:n_src], yb)
            dom = F.cross_entropy(out["subject_logits"], eb)
            center = class_center_loss(out["z"][:n_src], yb, eb[:n_src])
            ref_mask = eb == ref_eid
            non_mask = ~ref_mask
            if torch.any(ref_mask) and torch.any(non_mask):
                _, front_cov = model.front.mean_cov_alignment_loss(out["x_aligned"][non_mask], out["x_aligned"][ref_mask])
                band = band_alignment_loss(out["x_aligned"][non_mask], out["x_aligned"][ref_mask], sfreq=bundle.sfreq)
            else:
                front_cov = torch.zeros((), device=device)
                band = torch.zeros((), device=device)
            front_ortho = model.front.ortho_regularizer()
            front_tether = model.front.tether_regularizer()
            front_delta = model.front.delta_regularizer()
            total = (
                cfg.linear_model.lambda_recon * recon
                + cfg.linear_model.beta_kl * kl
                + cfg.linear_model.lambda_class * cls
                + cfg.linear_model.lambda_domain_adv * dom
                + cfg.linear_model.lambda_band_alignment * band
                + cfg.linear_model.lambda_latent_center * center
                + cfg.front.lambda_front_cov * front_cov
                + cfg.front.lambda_front_ortho * front_ortho
                + cfg.front.lambda_front_tether * front_tether
                + cfg.front.lambda_front_delta * front_delta
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            opt.step()

            meter["total"].append(float(total.detach().cpu().item()))
            meter["recon"].append(float(recon.detach().cpu().item()))
            meter["kl"].append(float(kl.detach().cpu().item()))
            meter["class"].append(float(cls.detach().cpu().item()))
            meter["domain"].append(float(dom.detach().cpu().item()))
            meter["band"].append(float(band.detach().cpu().item()))
            meter["center"].append(float(center.detach().cpu().item()))
            meter["front_cov"].append(float(front_cov.detach().cpu().item()))
            meter["front_ortho"].append(float(front_ortho.detach().cpu().item()))
            meter["front_tether"].append(float(front_tether.detach().cpu().item()))
            meter["front_delta"].append(float(front_delta.detach().cpu().item()))

        val_collect = _forward_collect(model, x_std[np.concatenate([bundle.val_idx, bundle.adapt_idx])], bundle.domain_ids[np.concatenate([bundle.val_idx, bundle.adapt_idx])], device)
        n_val = len(bundle.val_idx)
        val_logits = val_collect["class_logits"][:n_val]
        val_cls = _summarize_classification(val_logits, bundle.y[bundle.val_idx])
        val_entity_pred = val_collect["subject_logits"].argmax(axis=1)
        val_entity_true = bundle.entity_ids[np.concatenate([bundle.val_idx, bundle.adapt_idx])]
        val_entity_acc = float(np.mean(val_entity_pred == val_entity_true))
        val_entity_chance = float(np.max(np.bincount(val_entity_true)) / len(val_entity_true))
        val_recon_rmse = float(np.sqrt(np.mean((val_collect["x_recon"] - val_collect["x_aligned"]) ** 2)))
        val_score = linear_val_score(val_recon_rmse, val_cls["macro_f1"], val_entity_acc, val_entity_chance)

        row = {
            "epoch": epoch,
            "train_total": float(np.mean(meter["total"])),
            "train_recon": float(np.mean(meter["recon"])),
            "train_kl": float(np.mean(meter["kl"])),
            "train_class": float(np.mean(meter["class"])),
            "train_domain": float(np.mean(meter["domain"])),
            "train_band": float(np.mean(meter["band"])),
            "train_center": float(np.mean(meter["center"])),
            "train_front_cov": float(np.mean(meter["front_cov"])),
            "train_front_ortho": float(np.mean(meter["front_ortho"])),
            "train_front_tether": float(np.mean(meter["front_tether"])),
            "train_front_delta": float(np.mean(meter["front_delta"])),
            "val_score": float(val_score),
            "val_class_macro_f1": float(val_cls["macro_f1"]),
            "val_entity_probe_acc": float(val_entity_acc),
            "val_entity_probe_chance": float(val_entity_chance),
            "val_recon_rmse": float(val_recon_rmse),
        }
        history.write_row(row)
        jsonl.log(row)
        if epoch == 1 or epoch % 5 == 0:
            print(f"[{bundle.run_name}] epoch {epoch:03d} train_total={row['train_total']:.4f} val_score={row['val_score']:.4f} val_macro_f1={row['val_class_macro_f1']:.4f}")
        ckpt_path = out_dir / "checkpoints" / "best.pt"
        if tracker.update(epoch, row["val_score"], ckpt_path):
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            _save_checkpoint(ckpt_path, model, opt, epoch, row)
            wait = 0
        else:
            wait += 1
            if wait >= cfg.training.patience:
                print(f"[{bundle.run_name}] early stop at epoch {epoch}, best_epoch={tracker.best_epoch}, best_score={tracker.best_score:.4f}")
                break
        if cfg.output.save_checkpoints and cfg.training.save_every > 0 and epoch % cfg.training.save_every == 0:
            _save_checkpoint(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, opt, epoch, row)

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    all_collect = _forward_collect(model, x_std, bundle.domain_ids, device)

    source_all_idx = np.sort(np.concatenate([bundle.train_idx, bundle.val_idx]))
    val_metrics = _proxy_metrics(bundle, all_collect["x_aligned"], np.sort(np.concatenate([bundle.val_idx, bundle.adapt_idx])))
    val_metrics.update({
        "class_accuracy": float(_summarize_classification(all_collect["class_logits"][bundle.val_idx], bundle.y[bundle.val_idx])["accuracy"]),
        "class_macro_f1": float(_summarize_classification(all_collect["class_logits"][bundle.val_idx], bundle.y[bundle.val_idx])["macro_f1"]),
        "entity_probe_acc": float(np.mean(all_collect["subject_logits"][np.concatenate([bundle.val_idx, bundle.adapt_idx])].argmax(axis=1) == bundle.entity_ids[np.concatenate([bundle.val_idx, bundle.adapt_idx])])),
        "entity_probe_chance": float(np.max(np.bincount(bundle.entity_ids[np.concatenate([bundle.val_idx, bundle.adapt_idx])])) / len(np.concatenate([bundle.val_idx, bundle.adapt_idx]))),
        "recon_rmse": float(np.sqrt(np.mean((all_collect["x_recon"][np.concatenate([bundle.val_idx, bundle.adapt_idx])] - all_collect["x_aligned"][np.concatenate([bundle.val_idx, bundle.adapt_idx])]) ** 2))),
        "val_score": float(tracker.best_score),
        "best_epoch": int(tracker.best_epoch),
    })

    test_proxy = _proxy_metrics(bundle, all_collect["x_aligned"], np.sort(np.concatenate([source_all_idx, bundle.test_idx])))
    test_cls = _summarize_classification(all_collect["class_logits"][bundle.test_idx], bundle.y[bundle.test_idx])
    all_entity_pred = all_collect["subject_logits"].argmax(axis=1)
    test_metrics = dict(test_proxy)
    test_metrics.update({
        "test_class_accuracy": float(test_cls["accuracy"]),
        "test_class_macro_f1": float(test_cls["macro_f1"]),
        "entity_probe_acc_global": float(np.mean(all_entity_pred == bundle.entity_ids)),
        "entity_probe_chance_global": float(np.max(np.bincount(bundle.entity_ids)) / len(bundle.entity_ids)),
        "recon_rmse_global": float(np.sqrt(np.mean((all_collect["x_recon"] - all_collect["x_aligned"]) ** 2))),
    })

    if cfg.dataset.export_fif:
        save_epochs_fif(all_collect["x_aligned"], bundle.info_template, bundle.times, bundle.ch_names, bundle.metadata, out_dir / "aligned_all-epo.fif")
    if cfg.dataset.export_latents_npy:
        save_latent_npy(all_collect["mu"], bundle.metadata, out_dir / "linear")
    bundle.metadata.to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame([val_metrics]).to_csv(out_dir / "metrics_val.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(out_dir / "metrics_test.csv", index=False)
    write_json(val_metrics, out_dir / "metrics_val.json")
    write_json(test_metrics, out_dir / "metrics_test.json")
    write_json(
        {
            "run_name": bundle.run_name,
            "protocol": bundle.protocol_name,
            "reference_entity": bundle.reference_entity,
            "channel_selection_mode": bundle.channel_selection_mode,
            "n_samples_total": int(len(bundle.x_raw)),
            "n_train": int(len(bundle.train_idx)),
            "n_val": int(len(bundle.val_idx)),
            "n_adapt": int(len(bundle.adapt_idx)),
            "n_test": int(len(bundle.test_idx)),
            "best_epoch": int(tracker.best_epoch),
            "best_score": float(tracker.best_score),
        },
        out_dir / "run_summary.json",
    )
    return LinearTrainArtifacts(
        run_dir=out_dir,
        metrics_val=val_metrics,
        metrics_test=test_metrics,
        aligned_all=all_collect["x_aligned"],
        latent_all=all_collect["mu"],
        bundle=bundle,
    )
