import copy
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import (
    CustomSyntheticDataset,
    SyntheticDataset,
    generate_data_with_mixing_drift,
    generate_hierarchical_data_with_mixing_drift,
)
from metrics import mean_corr_coef as mcc
from metrics.diagnostics import (
    compute_domain_alignment,
    get_domain_scatter_palette,
    hungarian_match_abs,
    identity_consistency_vs_ref,
    pearson_corr_matrix,
    save_corr_heatmap,
    save_corr_heatmap_csv,
    save_palette_csv,
    save_stacked_similarity_3d,
    session_leakage_probe,
)
from models import (
    Discriminator,
    HierarchicalSessionLinearAdapter,
    SubjectResidualNonlinearAdapter,
    UConditionalMomentAligner,
    cleanIVAE,
    cleanVAE,
    iVAE,
    permute_dims,
)
from models.drift_adapter import DomainLinearAdapter
from utils.ica_anchor import compute_fastica_anchors


def _one_hot_int(labels, num_classes: int):
    labels = np.asarray(labels).astype(int).reshape(-1)
    out = np.zeros((labels.size, num_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def _extract_u_raw_id(U_raw: np.ndarray) -> np.ndarray:
    U_raw_np = np.asarray(U_raw)
    if U_raw_np.ndim == 2 and U_raw_np.shape[1] > 1:
        return U_raw_np.argmax(axis=1).astype(np.int64)
    return U_raw_np.reshape(-1).astype(np.int64)


def _make_aux_U(U_raw, z_all, aux_mode: str, n_domains: int, n_seg: int):
    """Build the auxiliary variable U fed into the identifiable source model."""
    aux_mode = str(aux_mode).lower()
    z_all = np.asarray(z_all).astype(int).reshape(-1)
    U_raw_np = np.asarray(U_raw)
    if U_raw_np.ndim == 2 and U_raw_np.shape[1] > 1:
        u_id = U_raw_np.argmax(axis=1).astype(int)
        Uu = U_raw_np.astype(np.float32)
    else:
        u_id = U_raw_np.reshape(-1).astype(int)
        Uu = _one_hot_int(u_id, int(n_seg))
    Uz = _one_hot_int(z_all, int(n_domains))

    if aux_mode == "u":
        return Uu
    if aux_mode == "z":
        return Uz
    if aux_mode in ("joint", "u_z", "uz", "joint_u_z"):
        joint_id = z_all * int(n_seg) + u_id
        return _one_hot_int(joint_id, int(n_domains) * int(n_seg))
    if aux_mode in ("joint_concat", "concat", "u+z"):
        return np.concatenate([Uu, Uz], axis=1).astype(np.float32)
    if aux_mode == "none":
        return np.zeros((z_all.shape[0], 1), dtype=np.float32)
    raise ValueError(f"Unknown aux_mode={aux_mode}")


def _compute_domain_stats_and_winit(X_np, z_np, S_anchor_np=None, ridge=1e-3):
    """Compute per-domain mean/std, and optional W_init via ridge least squares."""
    z_np = np.asarray(z_np).astype(int)
    n_domains = int(z_np.max() + 1)
    d = int(X_np.shape[1])
    mu = np.zeros((n_domains, d), dtype=np.float32)
    std = np.zeros((n_domains, d), dtype=np.float32)
    W_init = None
    if S_anchor_np is not None:
        W_init = np.zeros((n_domains, d, d), dtype=np.float32)

    I = np.eye(d, dtype=np.float64)
    for dom in range(n_domains):
        idx = np.where(z_np == dom)[0]
        Xd = X_np[idx].astype(np.float64)
        mu[dom] = Xd.mean(axis=0).astype(np.float32)
        std_dom = Xd.std(axis=0) + 1e-6
        std[dom] = std_dom.astype(np.float32)
        if W_init is not None:
            Sd = S_anchor_np[idx].astype(np.float64)
            Xs = (Xd - mu[dom].astype(np.float64)) / std_dom
            XtX = Xs.T @ Xs + float(ridge) * I
            W = (Sd.T @ Xs) @ np.linalg.inv(XtX)
            W_init[dom] = W.astype(np.float32)
    return torch.from_numpy(mu), torch.from_numpy(std), (torch.from_numpy(W_init) if W_init is not None else None)


def _build_custom_dataset(
    X: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    U_raw: Optional[np.ndarray] = None,
    u_raw_id: Optional[np.ndarray] = None,
    domain_id: Optional[np.ndarray] = None,
    subject_id: Optional[np.ndarray] = None,
    session_id: Optional[np.ndarray] = None,
    s_anchor: Optional[np.ndarray] = None,
    device: str = "cpu",
):
    dset = CustomSyntheticDataset(
        X=X,
        U=U,
        S=S,
        device=device,
        U_raw=U_raw,
        u_raw_id=u_raw_id,
        domain_id=domain_id,
        subject_id=subject_id,
        session_id=session_id,
    )
    if s_anchor is not None:
        dset.s_anchor = torch.from_numpy(s_anchor).to(device)
    return dset


def _subset_custom_dataset(full_arrays: Dict[str, Any], mask: np.ndarray, device: str = "cpu"):
    out = {}
    for k, v in full_arrays.items():
        if v is None:
            out[k] = None
        else:
            out[k] = np.asarray(v)[mask]
    return _build_custom_dataset(device=device, **out)


def _unpack_batch(batch, dset):
    x, u, s_true = batch[0], batch[1], batch[2]
    ptr = 3
    z_batch = None
    s_anchor = None
    u_raw_id = None
    subject_id = None
    session_id = None
    if getattr(dset, "z", None) is not None:
        z_batch = batch[ptr]
        ptr += 1
    if getattr(dset, "s_anchor", None) is not None:
        s_anchor = batch[ptr]
        ptr += 1
    if getattr(dset, "u_raw_id", None) is not None:
        u_raw_id = batch[ptr]
        ptr += 1
    if getattr(dset, "subject_id", None) is not None:
        subject_id = batch[ptr]
        ptr += 1
    if getattr(dset, "session_id", None) is not None:
        session_id = batch[ptr]
        ptr += 1
    return x, u, s_true, z_batch, s_anchor, u_raw_id, subject_id, session_id


def _resolve_backbone(config):
    backbone = getattr(config, "backbone", None)
    if backbone is not None:
        return str(backbone).lower()
    return "cleanivae" if bool(getattr(config, "ica", True)) else "cleanvae"


def _instantiate_model(config, d_data: int, d_latent: int, d_aux: int):
    backbone = _resolve_backbone(config)
    if backbone == "ivae":
        model = iVAE(
            latent_dim=d_latent,
            data_dim=d_data,
            aux_dim=d_aux,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            activation=config.activation,
            slope=0.1,
            device=config.device,
            anneal=bool(getattr(config, "anneal", False)),
        ).to(config.device)
    elif backbone == "cleanivae":
        model = cleanIVAE(
            data_dim=d_data,
            latent_dim=d_latent,
            aux_dim=d_aux,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            activation=config.activation,
            slope=0.1,
        ).to(config.device)
    elif backbone == "cleanvae":
        model = cleanVAE(
            data_dim=d_data,
            latent_dim=d_latent,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            activation=config.activation,
            slope=0.1,
        ).to(config.device)
    else:
        raise ValueError(f"Unsupported backbone={backbone}")
    return backbone, model


def _compute_model_loss(model, backbone: str, x: torch.Tensor, u: torch.Tensor, N: int,
                        a: float, b: float, c: float, d: float, global_it: int, max_iter: int):
    if backbone == "ivae":
        if bool(getattr(model, "anneal_params", False)):
            model.anneal(N, max_iter=max_iter, it=global_it)
        objective, z = model.elbo(x, u)
        return -objective, z
    if backbone == "cleanivae":
        loss, z = model.elbo(x, u, N, a=a, b=b, c=c, d=d)
        return loss, z
    if backbone == "cleanvae":
        loss, z = model.elbo(x, u, N, a=a, b=b, c=c, d=d)
        return loss, z
    raise ValueError(f"Unsupported backbone={backbone}")


def _eval_latents(model, backbone: str, x: torch.Tensor, u: torch.Tensor):
    if backbone == "ivae":
        _, _, z, _ = model(x, u)
        return z
    if backbone == "cleanivae":
        _, _, _, s, _ = model(x, u)
        return s
    if backbone == "cleanvae":
        _, _, _, s = model(x)
        return s
    raise ValueError(f"Unsupported backbone={backbone}")


def _schedule_beta(epoch: int, start_epoch: int, end_epoch: int, beta_max: float) -> float:
    if epoch < int(start_epoch):
        return 0.0
    if end_epoch <= start_epoch:
        return float(beta_max)
    frac = (float(epoch) - float(start_epoch)) / max(1.0, float(end_epoch) - float(start_epoch))
    frac = min(max(frac, 0.0), 1.0)
    return float(beta_max) * frac


def _freeze_module(module):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = False


def _clone_trainable_params(module):
    if module is None:
        return None
    return {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}


def _module_state(module):
    if module is None:
        return None
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _load_module_state(module, state):
    if module is None or state is None:
        return
    module.load_state_dict(state, strict=True)


def _apply_frontend(x, z_batch, subject_batch, u_raw_batch,
                    session_adapter=None, subject_adapter=None, moment_aligner=None, beta_nonlin: float = 0.0,
                    moment_enabled: bool = False):
    x_after_session = x
    if session_adapter is not None and z_batch is not None:
        x_after_session = session_adapter(x_after_session, z_batch)
    x_after_subject = x_after_session
    if subject_adapter is not None and subject_batch is not None:
        x_after_subject = subject_adapter(x_after_subject, subject_batch, beta=beta_nonlin)
    x_final = x_after_subject
    if moment_aligner is not None and moment_enabled and subject_batch is not None and u_raw_batch is not None:
        x_final = moment_aligner(x_final, subject_batch, u_raw_batch)
    return x_after_session, x_after_subject, x_final


def _fit_moment_aligner(moment_aligner, eval_dset, session_adapter, subject_adapter, beta_nonlin: float, config):
    if moment_aligner is None:
        return
    device = config.device
    with torch.no_grad():
        X = eval_dset.x.to(device)
        Z = None if getattr(eval_dset, "z", None) is None else torch.from_numpy(np.asarray(eval_dset.z).astype(int)).long().to(device)
        SUBJECT = None if getattr(eval_dset, "subject_id", None) is None else torch.from_numpy(np.asarray(eval_dset.subject_id).astype(int)).long().to(device)
        U_RAW = None if getattr(eval_dset, "u_raw_id", None) is None else torch.from_numpy(np.asarray(eval_dset.u_raw_id).astype(int)).long().to(device)
        _, x_after_subject, _ = _apply_frontend(
            X, Z, SUBJECT, U_RAW,
            session_adapter=session_adapter,
            subject_adapter=subject_adapter,
            moment_aligner=None,
            beta_nonlin=beta_nonlin,
            moment_enabled=False,
        )
        moment_aligner.fit(
            x_after_subject,
            SUBJECT,
            U_RAW,
            min_count=int(getattr(config, "dist_align_min_count", 50)),
            shrinkage=float(getattr(config, "dist_align_shrinkage", 0.0)),
        )


def _batch_moment_alignment_loss(x_subject, subject_batch, u_raw_batch, moment_aligner, min_count: int = 4):
    if moment_aligner is None:
        return torch.zeros((), device=x_subject.device)
    losses = []
    ref_subject = int(moment_aligner.ref_subject)
    for subj in torch.unique(subject_batch.long()):
        subj_int = int(subj.item())
        if subj_int == ref_subject:
            continue
        mask_subj = (subject_batch.long() == subj_int)
        for u in torch.unique(u_raw_batch.long()[mask_subj]):
            mask = mask_subj & (u_raw_batch.long() == int(u.item()))
            if int(mask.sum()) < int(min_count):
                continue
            xb = x_subject[mask]
            mu = xb.mean(dim=0)
            sd = xb.std(dim=0, unbiased=False) + 1e-6
            ref_mu = moment_aligner.mean_table[ref_subject, int(u.item())].to(x_subject.device)
            ref_sd = moment_aligner.std_table[ref_subject, int(u.item())].to(x_subject.device)
            losses.append(((mu - ref_mu) ** 2).mean() + ((torch.log(sd ** 2) - torch.log(ref_sd ** 2)) ** 2).mean())
    if len(losses) == 0:
        return torch.zeros((), device=x_subject.device)
    return torch.stack(losses).mean()


def _save_checkpoint(args, doc_name: str, seed: int, tag: str, epoch: int,
                     model, backbone: str,
                     session_adapter=None, subject_adapter=None, moment_aligner=None,
                     metrics: Optional[Dict[str, Any]] = None, config_dict: Optional[Dict[str, Any]] = None):
    ckpt = {
        "epoch": int(epoch),
        "seed": int(seed),
        "backbone": backbone,
        "model_state": model.state_dict(),
        "session_adapter_state": None if session_adapter is None else session_adapter.state_dict(),
        "subject_adapter_state": None if subject_adapter is None else subject_adapter.state_dict(),
        "moment_aligner_state": None if moment_aligner is None else moment_aligner.state_dict(),
        "metrics": metrics,
        "config": config_dict,
    }
    out = os.path.join(args.checkpoints, f"{doc_name}_seed{seed}_{tag}.pt")
    torch.save(ckpt, out)
    return out


def _namespace_to_dict(ns):
    if isinstance(ns, dict):
        return {k: _namespace_to_dict(v) for k, v in ns.items()}
    if hasattr(ns, "__dict__"):
        out = {}
        for k, v in vars(ns).items():
            out[k] = _namespace_to_dict(v)
        return out
    if isinstance(ns, (list, tuple)):
        return [_namespace_to_dict(v) for v in ns]
    if isinstance(ns, (np.integer, np.floating)):
        return ns.item()
    if isinstance(ns, np.ndarray):
        return ns.tolist()
    if torch.is_tensor(ns):
        return ns.detach().cpu().tolist()
    return ns


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_metrics_json(args, doc_name: str, seed: int, metrics: Dict[str, Any]):
    out = os.path.join(args.log, f"{doc_name}_seed{seed}_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)
    return out


def _build_metadata_frame(eval_dset):
    n = int(eval_dset.x.shape[0])
    data = {"sample_index": np.arange(n, dtype=np.int64)}
    z = getattr(eval_dset, "z", None)
    if z is None:
        z = getattr(eval_dset, "domain_id", None)
    if z is not None:
        data["domain_id"] = np.asarray(z).astype(np.int64)
    subject_id = getattr(eval_dset, "subject_id", None)
    if subject_id is not None:
        data["subject_id"] = np.asarray(subject_id).astype(np.int64)
    session_id = getattr(eval_dset, "session_id", None)
    if session_id is not None:
        data["session_id"] = np.asarray(session_id).astype(np.int64)
    u_raw_id = getattr(eval_dset, "u_raw_id", None)
    if u_raw_id is not None:
        data["u_raw_id"] = np.asarray(u_raw_id).astype(np.int64)
    return pd.DataFrame(data)


def _save_feature_table_csv(out_csv: str, metadata_df: pd.DataFrame, features: np.ndarray):
    arr = np.asarray(features, dtype=np.float64)
    df = metadata_df.copy()
    for j in range(arr.shape[1]):
        df[f"dim_{j}"] = arr[:, j]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def _select_plot_indices(n_samples: int, max_points: int, seed: int) -> np.ndarray:
    n_samples = int(n_samples)
    max_points = max(1, int(max_points))
    if n_samples <= max_points:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(n_samples, size=max_points, replace=False).astype(np.int64))


def _normalize_for_embedding(features_subset: np.ndarray):
    X = np.asarray(features_subset, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("features_subset must be a non-empty 2D array")
    mean = X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mean) / scale
    return Xn, mean, scale


def _compute_2d_embedding(features_subset: np.ndarray, seed: int):
    X, _, _ = _normalize_for_embedding(features_subset)
    method = "tsne"
    try:
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, (X.shape[0] - 1) // 3))
        perplexity = min(perplexity, max(1, X.shape[0] - 1))
        try:
            emb = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
                random_state=int(seed),
            ).fit_transform(X)
        except TypeError:
            emb = TSNE(
                n_components=2,
                init="pca",
                perplexity=perplexity,
                random_state=int(seed),
            ).fit_transform(X)
    except Exception:
        method = "pca"
        try:
            from sklearn.decomposition import PCA

            emb = PCA(n_components=2, random_state=int(seed)).fit_transform(X)
        except Exception:
            method = "raw"
            if X.shape[1] >= 2:
                emb = X[:, :2]
            else:
                emb = np.concatenate([X[:, :1], np.zeros((X.shape[0], 1), dtype=X.dtype)], axis=1)
    return np.asarray(emb, dtype=np.float64), method


def _fit_shared_linear_embedding(reference_features: np.ndarray, seed: int):
    X_ref, mean, scale = _normalize_for_embedding(reference_features)
    method = "shared_pca"
    try:
        from sklearn.decomposition import PCA

        model = PCA(n_components=2, random_state=int(seed))
        emb_ref = model.fit_transform(X_ref)
        state = {
            "kind": "pca",
            "method": method,
            "mean": mean,
            "scale": scale,
            "model": model,
        }
    except Exception:
        method = "shared_raw"
        if X_ref.shape[1] >= 2:
            emb_ref = X_ref[:, :2]
        else:
            emb_ref = np.concatenate([X_ref[:, :1], np.zeros((X_ref.shape[0], 1), dtype=X_ref.dtype)], axis=1)
        state = {
            "kind": "raw",
            "method": method,
            "mean": mean,
            "scale": scale,
            "model": None,
        }
    return np.asarray(emb_ref, dtype=np.float64), state


def _apply_shared_linear_embedding(features_subset: np.ndarray, embedding_state):
    X = np.asarray(features_subset, dtype=np.float64)
    mean = np.asarray(embedding_state["mean"], dtype=np.float64)
    scale = np.asarray(embedding_state["scale"], dtype=np.float64)
    Xn = (X - mean) / scale
    if embedding_state.get("kind") == "pca" and embedding_state.get("model") is not None:
        emb = embedding_state["model"].transform(Xn)
    else:
        if Xn.shape[1] >= 2:
            emb = Xn[:, :2]
        else:
            emb = np.concatenate([Xn[:, :1], np.zeros((Xn.shape[0], 1), dtype=Xn.dtype)], axis=1)
    return np.asarray(emb, dtype=np.float64), str(embedding_state.get("method", "shared_raw"))


def _save_embedding_csv(out_csv: str, metadata_df: pd.DataFrame, indices: np.ndarray,
                        embedding: np.ndarray, method: str):
    idx = np.asarray(indices).astype(np.int64)
    emb = np.asarray(embedding, dtype=np.float64)
    df = metadata_df.iloc[idx].copy().reset_index(drop=True)
    df["embed_x"] = emb[:, 0]
    df["embed_y"] = emb[:, 1]
    df["embed_method"] = method
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def _compute_shared_axis_limits(embedding_dfs):
    xs = []
    ys = []
    for df in embedding_dfs:
        if df is None or len(df) == 0:
            continue
        xs.append(np.asarray(df["embed_x"].values, dtype=np.float64))
        ys.append(np.asarray(df["embed_y"].values, dtype=np.float64))
    if len(xs) == 0:
        return None, None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xspan = max(1e-8, xmax - xmin)
    yspan = max(1e-8, ymax - ymin)
    margin_x = 0.08 * xspan
    margin_y = 0.08 * yspan
    return (xmin - margin_x, xmax + margin_x), (ymin - margin_y, ymax + margin_y)


def _draw_domain_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str):
    if len(x) < 3:
        return
    try:
        from matplotlib.patches import Ellipse
    except Exception:
        return
    pts = np.column_stack([x, y])
    if pts.shape[0] < 3:
        return
    cov = np.cov(pts, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width = 2.0 * 1.55 * np.sqrt(vals[0])
    height = 2.0 * 1.55 * np.sqrt(vals[1])
    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.15,
        alpha=0.95,
        zorder=4,
    )
    ax.add_patch(ellipse)


def _plot_domain_scatter(embedding_df: pd.DataFrame, out_png: str, title: str, palette_map: Dict[int, str],
                         xlim=None, ylim=None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    if "domain_id" in embedding_df.columns:
        domains = sorted(int(x) for x in embedding_df["domain_id"].dropna().unique().tolist())
    else:
        domains = [0]
        embedding_df = embedding_df.copy()
        embedding_df["domain_id"] = 0

    for dom in domains:
        sub = embedding_df[embedding_df["domain_id"] == dom]
        x = sub["embed_x"].values
        y = sub["embed_y"].values
        color = palette_map.get(int(dom), "#333333")
        ax.scatter(
            x,
            y,
            s=7,
            alpha=0.70,
            color=color,
            edgecolors="none",
            label=f"D{int(dom)}",
            zorder=2,
        )
        if len(sub) > 0:
            _draw_domain_ellipse(ax, x, y, color)
            ax.scatter(
                [float(np.mean(x))],
                [float(np.mean(y))],
                s=78,
                marker="X",
                color=color,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if len(domains) <= 12:
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(domains)))
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf")
    plt.close(fig)



def _plot_domain_scatter_panel(panel_data, out_png: str, palette_map: Dict[int, str], axis_limits=None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    axis_limits = axis_limits or {}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.8))
    axes = axes.ravel()
    for ax, (title, embedding_df, key) in zip(axes, panel_data):
        if "domain_id" in embedding_df.columns:
            domains = sorted(int(x) for x in embedding_df["domain_id"].dropna().unique().tolist())
        else:
            domains = [0]
            embedding_df = embedding_df.copy()
            embedding_df["domain_id"] = 0
        for dom in domains:
            sub = embedding_df[embedding_df["domain_id"] == dom]
            x = sub["embed_x"].values
            y = sub["embed_y"].values
            color = palette_map.get(int(dom), "#333333")
            ax.scatter(
                x,
                y,
                s=6,
                alpha=0.68,
                color=color,
                edgecolors="none",
                zorder=2,
            )
            if len(sub) > 0:
                _draw_domain_ellipse(ax, x, y, color)
                ax.scatter(
                    [float(np.mean(x))],
                    [float(np.mean(y))],
                    s=70,
                    marker="X",
                    color=color,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=5,
                )
        limits = axis_limits.get(key)
        if limits is not None:
            xlim, ylim = limits
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = []
    labels = []
    if len(panel_data) > 0 and "domain_id" in panel_data[0][1].columns:
        domains = sorted(int(x) for x in panel_data[0][1]["domain_id"].dropna().unique().tolist())
        from matplotlib.lines import Line2D
        for dom in domains[:12]:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                                  markerfacecolor=palette_map.get(dom, "#333333"), markeredgewidth=0))
            labels.append(f"D{dom}")
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=min(6, len(handles)))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, dpi=220)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf")
    plt.close(fig)


def _decode_reconstruction(model, s_est: torch.Tensor):
    if hasattr(model, "decoder"):
        return model.decoder(s_est)
    if hasattr(model, "f"):
        return model.f(s_est)
    raise AttributeError("Model does not expose a decoder or f network for reconstruction.")


def _save_domain_scatter_artifacts(args, config, doc_name: str, seed: int, eval_dset,
                                   raw_x_np: np.ndarray, aligned_x_np: np.ndarray,
                                   latent_s_np: np.ndarray, recon_x_np: np.ndarray):
    if getattr(eval_dset, "z", None) is None and getattr(eval_dset, "domain_id", None) is None:
        return {}

    metadata_df = _build_metadata_frame(eval_dset)
    n_domains = int(metadata_df["domain_id"].max()) + 1 if "domain_id" in metadata_df.columns else 1
    palette = get_domain_scatter_palette(n_domains)
    palette_map = {int(i): palette[i] for i in range(len(palette))}
    palette_csv = os.path.join(args.log, f"{doc_name}_seed{seed}_domain_palette.csv")
    save_palette_csv([f"D{i}" for i in range(n_domains)], palette, palette_csv)

    max_points = int(getattr(config, "scatter_max_points", 4000))
    plot_idx = _select_plot_indices(len(metadata_df), max_points=max_points, seed=int(seed))

    raw_plot = np.asarray(raw_x_np, dtype=np.float64)[plot_idx]
    aligned_plot = np.asarray(aligned_x_np, dtype=np.float64)[plot_idx]
    recon_plot = np.asarray(recon_x_np, dtype=np.float64)[plot_idx]
    latent_plot = np.asarray(latent_s_np, dtype=np.float64)[plot_idx]

    _, observed_embedding_state = _fit_shared_linear_embedding(raw_plot, seed=int(seed))
    raw_emb, raw_method = _apply_shared_linear_embedding(raw_plot, observed_embedding_state)
    aligned_emb, aligned_method = _apply_shared_linear_embedding(aligned_plot, observed_embedding_state)
    recon_emb, recon_method = _apply_shared_linear_embedding(recon_plot, observed_embedding_state)
    latent_emb, latent_state = _fit_shared_linear_embedding(latent_plot, seed=int(seed))
    latent_method = str(latent_state.get("method", "shared_pca")).replace("shared_", "")

    datasets = [
        ("raw_observed", np.asarray(raw_x_np, dtype=np.float64), raw_emb, raw_method, "Raw observed domains"),
        ("aligned_observed", np.asarray(aligned_x_np, dtype=np.float64), aligned_emb, aligned_method, "Mixing-aligned observed domains"),
        ("estimated_latent", np.asarray(latent_s_np, dtype=np.float64), latent_emb, latent_method, "Estimated latent sources"),
        ("reconstructed_observed", np.asarray(recon_x_np, dtype=np.float64), recon_emb, recon_method, "Generator-reconstructed observations"),
    ]

    observed_limits = _compute_shared_axis_limits([
        _save_embedding_csv(os.path.join(args.log, f"{doc_name}_seed{seed}_raw_observed_embedding.csv"), metadata_df, plot_idx, raw_emb, raw_method),
        _save_embedding_csv(os.path.join(args.log, f"{doc_name}_seed{seed}_aligned_observed_embedding.csv"), metadata_df, plot_idx, aligned_emb, aligned_method),
        _save_embedding_csv(os.path.join(args.log, f"{doc_name}_seed{seed}_reconstructed_observed_embedding.csv"), metadata_df, plot_idx, recon_emb, recon_method),
    ])
    # Re-read the already saved observed embedding CSVs so downstream artifact naming stays unchanged.
    raw_emb_df = pd.read_csv(os.path.join(args.log, f"{doc_name}_seed{seed}_raw_observed_embedding.csv"))
    aligned_emb_df = pd.read_csv(os.path.join(args.log, f"{doc_name}_seed{seed}_aligned_observed_embedding.csv"))
    recon_emb_df = pd.read_csv(os.path.join(args.log, f"{doc_name}_seed{seed}_reconstructed_observed_embedding.csv"))
    latent_emb_df = _save_embedding_csv(
        os.path.join(args.log, f"{doc_name}_seed{seed}_estimated_latent_embedding.csv"),
        metadata_df,
        plot_idx,
        latent_emb,
        latent_method,
    )

    emb_df_map = {
        "raw_observed": raw_emb_df,
        "aligned_observed": aligned_emb_df,
        "estimated_latent": latent_emb_df,
        "reconstructed_observed": recon_emb_df,
    }
    limits_map = {
        "raw_observed": observed_limits,
        "aligned_observed": observed_limits,
        "estimated_latent": _compute_shared_axis_limits([latent_emb_df]),
        "reconstructed_observed": observed_limits,
    }

    panel_data = []
    artifact_paths = {"domain_palette_csv": os.path.basename(palette_csv)}
    for name, feat, _, _, title in datasets:
        full_csv = os.path.join(args.log, f"{doc_name}_seed{seed}_{name}_full.csv")
        embedding_csv = os.path.join(args.log, f"{doc_name}_seed{seed}_{name}_embedding.csv")
        plot_png = os.path.join(args.log, f"{doc_name}_seed{seed}_{name}_scatter.png")
        _save_feature_table_csv(full_csv, metadata_df, feat)
        emb_df = emb_df_map[name]
        if not os.path.exists(embedding_csv):
            emb_df.to_csv(embedding_csv, index=False, encoding="utf-8-sig")
        _plot_domain_scatter(
            emb_df,
            plot_png,
            title=title,
            palette_map=palette_map,
            xlim=limits_map[name][0],
            ylim=limits_map[name][1],
        )
        panel_data.append((title, emb_df, name))
        artifact_paths[f"{name}_full_csv"] = os.path.basename(full_csv)
        artifact_paths[f"{name}_embedding_csv"] = os.path.basename(embedding_csv)
        artifact_paths[f"{name}_scatter_png"] = os.path.basename(plot_png)
        artifact_paths[f"{name}_scatter_pdf"] = os.path.basename(os.path.splitext(plot_png)[0] + '.pdf')

    panel_png = os.path.join(args.log, f"{doc_name}_seed{seed}_domain_scatter_panels.png")
    _plot_domain_scatter_panel(panel_data, panel_png, palette_map=palette_map, axis_limits=limits_map)
    artifact_paths["domain_scatter_panel_png"] = os.path.basename(panel_png)
    artifact_paths["domain_scatter_panel_pdf"] = os.path.basename(os.path.splitext(panel_png)[0] + '.pdf')
    return artifact_paths


def _load_corr_matrix_from_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty correlation CSV: {path}")
    pivot = df.pivot(index="true_source", columns="estimated_component", values="abs_corr")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot.to_numpy(dtype=np.float64)


def _save_stacked_similarity_artifacts(args, doc_name: str, seeds, n_domains: int):
    outputs = []
    if n_domains is None:
        return outputs
    for dom in range(int(n_domains)):
        corr_by_seed = {}
        for seed in seeds:
            csv_path = os.path.join(args.log, f"{doc_name}_seed{seed}_corr_heatmap_z{dom}.csv")
            if os.path.exists(csv_path):
                try:
                    corr_by_seed[int(seed)] = _load_corr_matrix_from_csv(csv_path)
                except Exception as e:
                    print(f"[diag] failed to load heatmap csv for stacked plot: {csv_path} | {e}")
        if len(corr_by_seed) == 0:
            continue
        out_png = os.path.join(args.log, f"{doc_name}_stacked_similarity_z{dom}.png")
        out_pdf = os.path.join(args.log, f"{doc_name}_stacked_similarity_z{dom}.pdf")
        out_csv = os.path.join(args.log, f"{doc_name}_stacked_similarity_z{dom}.csv")
        save_stacked_similarity_3d(
            corr_by_seed=corr_by_seed,
            out_png=out_png,
            out_pdf=out_pdf,
            out_csv=out_csv,
            doc=doc_name,
            domain=int(dom),
            title=f"stacked abs corr (z={dom})",
        )
        outputs.append({
            "domain": int(dom),
            "png": os.path.basename(out_png),
            "pdf": os.path.basename(out_pdf),
            "csv": os.path.basename(out_csv),
            "n_layers": int(len(corr_by_seed)),
        })
    return outputs


def _compute_eval_metrics(args, config, seed: int, doc_name: str,
                          eval_dset, s_true_np, s_np,
                          z_all_eval=None, subject_all_eval=None, session_all_eval=None):
    metrics = {
        "seed": int(seed),
        "full_mcc": float(mcc(s_true_np, s_np)),
    }
    if z_all_eval is not None:
        n_domains_cfg = int(np.max(z_all_eval)) + 1
        per_domain = []
        for dom in range(n_domains_cfg):
            mask = (z_all_eval == dom)
            if int(mask.sum()) < 10:
                continue
            per_domain.append({
                "domain": int(dom),
                "mcc": float(mcc(s_true_np[mask], s_np[mask])),
                "n": int(mask.sum()),
            })
        metrics["per_domain"] = per_domain

        if bool(getattr(config, "report_diagnostics", False)):
            aligns = compute_domain_alignment(
                S_true=s_true_np,
                S_est=s_np,
                z=z_all_eval,
                n_domains=n_domains_cfg,
                min_samples=int(getattr(config, "diag_min_samples", 50)),
            )
            metrics["alignment"] = []
            for a in aligns:
                metrics["alignment"].append({
                    "domain": int(a.domain),
                    "score": float(a.score),
                    "perm_est_to_true": a.perm_est_to_true.tolist(),
                    "sign_est": a.sign_est.tolist(),
                })
            if len(aligns) > 0:
                ref_dom = 0
                cons = identity_consistency_vs_ref(aligns, ref_domain=ref_dom)
                metrics["perm_agreement"] = float(cons["perm_agreement"])
                metrics["sign_agreement"] = float(cons["sign_agreement"])
                if bool(getattr(config, "save_heatmaps", True)):
                    metrics["corr_heatmaps"] = []
                    for a in aligns:
                        out_png = os.path.join(args.log, f"{doc_name}_seed{seed}_corr_heatmap_z{a.domain}.png")
                        out_pdf = os.path.join(args.log, f"{doc_name}_seed{seed}_corr_heatmap_z{a.domain}.pdf")
                        out_csv = os.path.join(args.log, f"{doc_name}_seed{seed}_corr_heatmap_z{a.domain}.csv")
                        save_corr_heatmap(a.corr, out_png, title=f"abs corr (z={a.domain})", out_pdf=out_pdf)
                        save_corr_heatmap_csv(a.corr, out_csv, doc=doc_name, seed=int(seed), domain=int(a.domain))
                        metrics["corr_heatmaps"].append({
                            "domain": int(a.domain),
                            "png": os.path.basename(out_png),
                            "pdf": os.path.basename(out_pdf),
                            "csv": os.path.basename(out_csv),
                        })
            try:
                max_samp = int(getattr(config, "probe_max_samples", 20000))
                probe_est = session_leakage_probe(s_np, z_all_eval, max_samples=max_samp, seed=int(seed))
                metrics["probe_acc"] = float(probe_est["acc"])
                metrics["probe_chance"] = float(probe_est["chance"])
            except Exception as e:
                metrics["probe_error"] = str(e)

    if subject_all_eval is not None:
        n_subjects = int(np.max(subject_all_eval)) + 1
        per_subject = []
        for subj in range(n_subjects):
            mask = (subject_all_eval == subj)
            if int(mask.sum()) < 10:
                continue
            per_subject.append({
                "subject": int(subj),
                "mcc": float(mcc(s_true_np[mask], s_np[mask])),
                "n": int(mask.sum()),
            })
        metrics["per_subject"] = per_subject

    if session_all_eval is not None:
        n_sessions = int(np.max(session_all_eval)) + 1
        per_session = []
        for sess in range(n_sessions):
            mask = (session_all_eval == sess)
            if int(mask.sum()) < 10:
                continue
            per_session.append({
                "session": int(sess),
                "mcc": float(mcc(s_true_np[mask], s_np[mask])),
                "n": int(mask.sum()),
            })
        metrics["per_session"] = per_session

    return metrics


def runner(args, config):
    st = time.time()
    print(f"Executing script on: {config.device}\n")

    factor = getattr(config, "gamma", 0) > 0
    if factor and bool(getattr(config, "mix_drift", False)):
        raise NotImplementedError("mix_drift=True currently supports gamma=0 in this runner.")

    eval_dset = None
    z_all = None
    subject_all = None
    session_all = None
    u_raw_id_all = None
    S_anchor_all = None
    reference_domains = None
    subject_of_domain = None

    if bool(getattr(config, "mix_drift", False)):
        if bool(getattr(config, "hierarchical_domains", False)):
            S, X, U_raw, M, L, extra = generate_hierarchical_data_with_mixing_drift(
                n_per_seg=config.nps,
                n_seg=config.ns,
                d_sources=config.dl,
                d_data=config.dd,
                n_layers=config.nl,
                prior=config.p,
                activation=config.act,
                seed=config.s,
                slope=getattr(config, "slope", 0.1),
                repeat_linearity=bool(getattr(config, "repeat_linearity", True)),
                noisy=float(getattr(config, "noisy", 0.0)),
                uncentered=bool(getattr(config, "uncentered", False)),
                n_subjects=int(getattr(config, "n_subjects", 2)),
                sessions_per_subject=getattr(config, "sessions_per_subject", 3),
                subject_mixing_mode=str(getattr(config, "subject_mixing_mode", "perturb")),
                subject_nonlinear_strength=float(getattr(config, "subject_nonlinear_strength", 0.2)),
                session_drift_strength=float(getattr(config, "session_drift_strength", 0.05)),
                share_source_params=bool(getattr(config, "share_source_params", True)),
                source_conditional_shift=bool(getattr(config, "source_conditional_shift", False)),
                source_mean_shift_strength=float(getattr(config, "source_mean_shift_strength", 0.0)),
                source_scale_shift_strength=float(getattr(config, "source_scale_shift_strength", 0.0)),
                return_extra=True,
            )
            z_all = extra.z
            subject_all = extra.subject_id
            session_all = extra.session_id
            u_raw_id_all = extra.u_raw_id
            reference_domains = extra.reference_domains
            subject_of_domain = extra.subject_of_domain
            n_domains_cfg = int(z_all.max()) + 1
        else:
            S, X, U_raw, M, L, extra = generate_data_with_mixing_drift(
                n_per_seg=config.nps,
                n_seg=config.ns,
                d_sources=config.dl,
                d_data=config.dd,
                n_layers=config.nl,
                prior=config.p,
                activation=config.act,
                seed=config.s,
                slope=getattr(config, "slope", 0.1),
                repeat_linearity=bool(getattr(config, "repeat_linearity", True)),
                one_hot_labels=True,
                n_domains=int(getattr(config, "n_domains", 2)),
                drift_mode=getattr(config, "drift_mode", "perturb"),
                drift_strength=float(getattr(config, "drift_strength", 0.1)),
                share_source_params=bool(getattr(config, "share_source_params", True)),
                return_extra=True,
            )
            z_all = extra.z
            u_raw_id_all = _extract_u_raw_id(U_raw)
            subject_all = np.asarray(z_all).copy()
            session_all = np.asarray(z_all).copy()
            reference_domains = np.arange(int(getattr(config, "n_domains", int(z_all.max()) + 1)), dtype=np.int64)
            subject_of_domain = np.arange(int(getattr(config, "n_domains", int(z_all.max()) + 1)), dtype=np.int64)
            n_domains_cfg = int(getattr(config, "n_domains", int(z_all.max()) + 1))

        aux_mode = str(getattr(config, "aux_mode", "u")).lower()
        U = _make_aux_U(
            U_raw=U_raw,
            z_all=z_all,
            aux_mode=aux_mode,
            n_domains=n_domains_cfg,
            n_seg=int(getattr(config, "ns", U_raw.shape[1] if np.asarray(U_raw).ndim == 2 else 1)),
        )
        print(f"[aux] aux_mode={aux_mode} | U_raw={tuple(np.asarray(U_raw).shape)} -> U={tuple(U.shape)}")

        full_arrays = {
            "X": X,
            "U": U,
            "S": S,
            "U_raw": U_raw,
            "u_raw_id": u_raw_id_all,
            "domain_id": z_all,
            "subject_id": subject_all,
            "session_id": session_all,
        }
        full_dset = _build_custom_dataset(device="cpu", **full_arrays)

        train_domains = getattr(config, "train_domains", None)
        test_domains = getattr(config, "test_domains", None)
        if train_domains is not None:
            train_domains = list(train_domains)
            if test_domains is None:
                test_domains = [d for d in range(n_domains_cfg) if d not in train_domains]
            else:
                test_domains = list(test_domains)
            train_mask = np.isin(z_all, np.asarray(train_domains, dtype=int))
            dset = _subset_custom_dataset(full_arrays, train_mask, device="cpu")
            eval_dset = full_dset
        else:
            dset = full_dset
            eval_dset = full_dset

        if bool(getattr(config, "use_ica_anchor", False)):
            try:
                ref_dom = int(getattr(config, "ica_anchor_ref_domain", 0))
                max_samp = getattr(config, "ica_anchor_max_samples_per_domain", 5000)
                max_samp = None if max_samp is None else int(max_samp)
                max_iter = int(getattr(config, "ica_anchor_max_iter", 500))
                S_anchor_all, anchor_info = compute_fastica_anchors(
                    X=X,
                    z=z_all,
                    U=U_raw,
                    n_domains=n_domains_cfg,
                    n_components=int(config.dl),
                    seed=int(getattr(config, "s", 0)),
                    ref_domain=ref_dom,
                    max_samples_per_domain=max_samp,
                    max_iter=max_iter,
                )
                full_dset.s_anchor = torch.from_numpy(S_anchor_all).to("cpu")
                if train_domains is not None:
                    dset.s_anchor = full_dset.s_anchor[train_mask]
                else:
                    dset.s_anchor = full_dset.s_anchor
                for info in anchor_info:
                    perm_str = ",".join([str(int(x)) for x in info.perm_ref_to_dom.tolist()])
                    print(f"[anchor] dom {info.domain}: mean_similarity={info.mean_abs_corr:.3f}, perm_ref->dom=[{perm_str}]")
            except Exception as e:
                print(f"[anchor] failed to compute ICA anchors: {e}")
    else:
        dset = SyntheticDataset(
            args.data_path,
            config.nps,
            config.ns,
            config.dl,
            config.dd,
            config.nl,
            config.s,
            config.p,
            config.act,
            uncentered=config.uncentered,
            noisy=config.noisy,
            double=factor,
        )
        eval_dset = dset

    d_data, d_latent, d_aux = dset.get_dims()

    num_workers = int(getattr(config, "num_workers", 0))
    if torch.cuda.is_available():
        loader_params = {"num_workers": num_workers, "pin_memory": True}
    else:
        loader_params = {"num_workers": 0}
    data_loader = DataLoader(dset, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_params)

    perfs = []
    loss_hists = []
    perf_hists = []
    seed_metric_list = []
    doc_name = args.doc if args.doc else os.path.splitext(args.config)[0]
    config_dict = _namespace_to_dict(config)

    for seed in range(args.seed, args.seed + args.n_sims):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        backbone, model = _instantiate_model(config, d_data=d_data, d_latent=d_latent, d_aux=d_aux)

        session_adapter = None
        subject_adapter = None
        moment_aligner = None
        session_adapter_init_params = None

        if bool(getattr(config, "mix_drift", False)) and bool(getattr(config, "use_drift_adapter", False)):
            init_mode = str(getattr(config, "drift_adapter_init", "identity")).lower()
            ridge = float(getattr(config, "drift_adapter_ridge", 1e-3))
            S_anchor_np = S_anchor_all if (init_mode == "ica" and S_anchor_all is not None) else None
            mu_t, std_t, W_init_t = _compute_domain_stats_and_winit(X_np=X, z_np=z_all, S_anchor_np=S_anchor_np, ridge=ridge)
            if bool(getattr(config, "hierarchical_domains", False)):
                session_adapter = HierarchicalSessionLinearAdapter(
                    n_subjects=int(np.max(subject_all)) + 1,
                    n_domains=int(np.max(z_all)) + 1,
                    d=int(d_data),
                    mu=mu_t,
                    std=std_t,
                    subject_of_domain=torch.from_numpy(np.asarray(subject_of_domain).astype(np.int64)),
                    W_init=W_init_t,
                    reference_domains=torch.from_numpy(np.asarray(reference_domains).astype(np.int64)) if reference_domains is not None else None,
                    use_shared=True,
                ).to(config.device)
            else:
                session_adapter = DomainLinearAdapter(
                    n_domains=int(np.max(z_all)) + 1,
                    d=int(d_data),
                    mu=mu_t,
                    std=std_t,
                    W_init=W_init_t,
                    use_shared=True,
                ).to(config.device)
            if bool(getattr(config, "freeze_adapter", False)):
                _freeze_module(session_adapter)
            session_adapter_init_params = _clone_trainable_params(session_adapter)
            print(f"[adapter] enabled session linear adapter | hierarchical={bool(getattr(config, 'hierarchical_domains', False))}")

        if bool(getattr(config, "mix_drift", False)) and bool(getattr(config, "use_subject_nonlinear_adapter", False)):
            subject_adapter = SubjectResidualNonlinearAdapter(
                n_subjects=int(np.max(subject_all)) + 1,
                d=int(d_data),
                bottleneck_dim=int(getattr(config, "subject_bottleneck_dim", 8)),
                activation=str(getattr(config, "subject_adapter_activation", "xtanh")),
                slope=float(getattr(config, "subject_adapter_slope", 0.1)),
                ref_subject=int(getattr(config, "subject_ref", 0)),
                zero_init=True,
            ).to(config.device)
            print("[adapter] enabled subject nonlinear residual adapter")

        if bool(getattr(config, "mix_drift", False)) and bool(getattr(config, "use_u_conditional_dist_align", False)):
            moment_aligner = UConditionalMomentAligner(
                n_subjects=int(np.max(subject_all)) + 1,
                n_u=int(np.max(u_raw_id_all)) + 1,
                d=int(d_data),
                ref_subject=int(getattr(config, "dist_align_ref_subject", getattr(config, "subject_ref", 0))),
                eta_mean=float(getattr(config, "dist_align_eta_mean", 1.0)),
                eta_var=float(getattr(config, "dist_align_eta_var", 0.3)),
                eps=float(getattr(config, "dist_align_eps", 1e-5)),
            ).to(config.device)
            print("[adapter] enabled same-U_raw cross-subject moment aligner")

        lr_adapter = float(getattr(config, "lr_adapter", config.lr))
        lr_subject = float(getattr(config, "lr_subject_adapter", config.lr))
        param_groups = [{"params": model.parameters(), "lr": config.lr}]
        if session_adapter is not None:
            session_params = [p for p in session_adapter.parameters() if p.requires_grad]
            if len(session_params) > 0:
                param_groups.append({"params": session_params, "lr": lr_adapter})
        if subject_adapter is not None:
            subject_params = [p for p in subject_adapter.parameters() if p.requires_grad]
            if len(subject_params) > 0:
                param_groups.append({"params": subject_params, "lr": lr_subject})
        optimizer = optim.Adam(param_groups)
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0)

        if factor:
            D = Discriminator(d_latent).to(config.device)
            optim_D = optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.9))
        else:
            D = None
            optim_D = None

        best_state = None
        best_train_loss = np.inf
        loss_hist = []
        perf_hist = []
        global_it = 0
        max_iter = int(config.epochs) * max(1, len(data_loader))

        for epoch in range(1, config.epochs + 1):
            model.train()
            if session_adapter is not None:
                session_adapter.train()
            if subject_adapter is not None:
                subject_adapter.train()

            beta_nonlin = _schedule_beta(
                epoch,
                start_epoch=int(getattr(config, "beta_start_epoch", 1)),
                end_epoch=int(getattr(config, "beta_end_epoch", max(2, config.epochs // 2))),
                beta_max=float(getattr(config, "beta_max", 0.0 if subject_adapter is None else 0.1)),
            )
            dist_start = int(getattr(config, "dist_align_start_epoch", config.epochs + 1))
            moment_enabled = moment_aligner is not None and (epoch >= dist_start)
            if moment_aligner is not None and epoch >= dist_start:
                refresh_mode = str(getattr(config, "dist_align_refresh", "epoch")).lower()
                if refresh_mode in {"epoch", "ema", "static"}:
                    if refresh_mode == "static" and int(moment_aligner.fitted.item()) == 1:
                        pass
                    else:
                        _fit_moment_aligner(moment_aligner, eval_dset, session_adapter, subject_adapter, beta_nonlin, config)

            if bool(getattr(config, "anneal", False)):
                a = getattr(config, "a", 100)
                d_coef = getattr(config, "d", 10)
                b = getattr(config, "b", 1)
                c = 0
                if epoch > config.epochs / 1.6:
                    b = 1
                    c = 1
                    d_coef = 1
                    a = 2 * getattr(config, "a", 100)
            else:
                a = getattr(config, "a", 100)
                b = getattr(config, "b", 1)
                c = getattr(config, "c", 0)
                d_coef = getattr(config, "d", 10)

            train_loss = 0.0
            train_perf = 0.0
            train_anchor = 0.0
            n_anchor_batches = 0

            for batch in data_loader:
                global_it += 1
                x, u, s_true, z_batch, s_anchor, u_raw_batch, subject_batch, session_batch = _unpack_batch(batch, dset)
                if factor:
                    raise NotImplementedError("factor/gamma training path is not used in the current cross-domain experiments.")
                x = x.to(config.device)
                u = u.to(config.device)
                z_dev = None if z_batch is None else z_batch.to(config.device).long()
                subject_dev = None if subject_batch is None else subject_batch.to(config.device).long()
                u_raw_dev = None if u_raw_batch is None else u_raw_batch.to(config.device).long()
                if s_anchor is not None:
                    s_anchor = s_anchor.to(config.device)

                optimizer.zero_grad()
                x_after_session, x_after_subject, x_front = _apply_frontend(
                    x,
                    z_dev,
                    subject_dev,
                    u_raw_dev,
                    session_adapter=session_adapter,
                    subject_adapter=subject_adapter,
                    moment_aligner=moment_aligner,
                    beta_nonlin=beta_nonlin,
                    moment_enabled=moment_enabled,
                )
                loss, z_latent = _compute_model_loss(
                    model,
                    backbone,
                    x_front,
                    u,
                    len(dset),
                    a=float(a),
                    b=float(b),
                    c=float(c),
                    d=float(d_coef),
                    global_it=global_it,
                    max_iter=max_iter,
                )

                if session_adapter is not None:
                    lam_init = float(getattr(config, "lambda_adapter_init", 0.0))
                    if lam_init > 0 and session_adapter_init_params is not None:
                        init_reg = torch.zeros((), device=config.device)
                        for n, p in session_adapter.named_parameters():
                            if p.requires_grad:
                                init_reg = init_reg + (p - session_adapter_init_params[n]).pow(2).mean()
                        loss = loss + lam_init * init_reg
                    lam_delta = float(getattr(config, "lambda_drift_adapter", 0.0))
                    lam_ortho = float(getattr(config, "lambda_drift_ortho", 0.0))
                    if lam_delta > 0:
                        loss = loss + lam_delta * session_adapter.delta_reg(z_dev)
                    if lam_ortho > 0:
                        loss = loss + lam_ortho * session_adapter.ortho_reg(z_dev)

                if subject_adapter is not None and subject_dev is not None:
                    lam_res = float(getattr(config, "lambda_subject_residual", 0.0))
                    lam_jac = float(getattr(config, "lambda_subject_jacobian", 0.0))
                    lam_w = float(getattr(config, "lambda_subject_weight", 0.0))
                    if lam_res > 0:
                        loss = loss + lam_res * subject_adapter.residual_reg(x_after_session, subject_dev)
                    if lam_jac > 0:
                        loss = loss + lam_jac * subject_adapter.jacobian_reg(x_after_session, subject_dev, beta=beta_nonlin)
                    if lam_w > 0:
                        loss = loss + lam_w * subject_adapter.weight_reg()

                lam_anchor = float(getattr(config, "lambda_ica_anchor", 0.0))
                if s_anchor is not None and lam_anchor > 0:
                    z_mu = z_latent.mean(dim=0, keepdim=True)
                    z_sd = z_latent.std(dim=0, keepdim=True) + 1e-8
                    a_mu = s_anchor.mean(dim=0, keepdim=True)
                    a_sd = s_anchor.std(dim=0, keepdim=True) + 1e-8
                    z_std = (z_latent - z_mu) / z_sd
                    a_std = (s_anchor - a_mu) / a_sd
                    loss_pos = (z_std - a_std) ** 2
                    loss_neg = (z_std + a_std) ** 2
                    anchor_loss = torch.minimum(loss_pos, loss_neg).mean()
                    loss = loss + lam_anchor * anchor_loss
                    train_anchor += float(anchor_loss.item())
                    n_anchor_batches += 1

                if moment_aligner is not None and moment_enabled and subject_dev is not None and u_raw_dev is not None:
                    lam_batch_moment = float(getattr(config, "lambda_u_conditional_batch_moment", 0.0))
                    if lam_batch_moment > 0:
                        batch_moment_loss = _batch_moment_alignment_loss(
                            x_after_subject,
                            subject_dev,
                            u_raw_dev,
                            moment_aligner,
                            min_count=int(getattr(config, "dist_align_batch_min_count", 4)),
                        )
                        loss = loss + lam_batch_moment * batch_moment_loss

                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                try:
                    perf = mcc(s_true.numpy(), z_latent.detach().cpu().numpy())
                except Exception:
                    perf = 0.0
                train_perf += float(perf)

            train_loss /= max(1, len(data_loader))
            train_perf /= max(1, len(data_loader))
            loss_hist.append(train_loss)
            perf_hist.append(train_perf)
            msg = f"==> Epoch {epoch}/{config.epochs}:\ttrain loss: {train_loss:.6f}\ttrain perf: {train_perf:.6f}\tbeta_nonlin: {beta_nonlin:.4f}"
            if n_anchor_batches > 0:
                msg += f"\tanchor loss: {train_anchor / float(n_anchor_batches):.6f}"
            print(msg)

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "session_adapter": None if session_adapter is None else copy.deepcopy(session_adapter.state_dict()),
                    "subject_adapter": None if subject_adapter is None else copy.deepcopy(subject_adapter.state_dict()),
                    "moment_aligner": None if moment_aligner is None else copy.deepcopy(moment_aligner.state_dict()),
                    "epoch": epoch,
                }
                if bool(getattr(config, "save_best_model", True)):
                    _save_checkpoint(
                        args,
                        doc_name,
                        seed,
                        tag="best",
                        epoch=epoch,
                        model=model,
                        backbone=backbone,
                        session_adapter=session_adapter,
                        subject_adapter=subject_adapter,
                        moment_aligner=moment_aligner,
                        metrics={"train_loss": train_loss, "train_perf": train_perf},
                        config_dict=config_dict,
                    )

            if not bool(getattr(config, "no_scheduler", False)):
                scheduler.step(train_loss)

        if best_state is not None and bool(getattr(config, "evaluate_best_model", True)):
            model.load_state_dict(best_state["model"])
            _load_module_state(session_adapter, best_state["session_adapter"])
            _load_module_state(subject_adapter, best_state["subject_adapter"])
            _load_module_state(moment_aligner, best_state["moment_aligner"])

        beta_eval = _schedule_beta(
            epoch=config.epochs,
            start_epoch=int(getattr(config, "beta_start_epoch", 1)),
            end_epoch=int(getattr(config, "beta_end_epoch", max(2, config.epochs // 2))),
            beta_max=float(getattr(config, "beta_max", 0.0 if subject_adapter is None else 0.1)),
        )
        if moment_aligner is not None and int(getattr(config, "dist_align_start_epoch", config.epochs + 1)) <= config.epochs:
            _fit_moment_aligner(moment_aligner, eval_dset, session_adapter, subject_adapter, beta_eval, config)

        print(f"\ntotal runtime so far: {time.time() - st}")

        model.eval()
        if session_adapter is not None:
            session_adapter.eval()
        if subject_adapter is not None:
            subject_adapter.eval()

        with torch.no_grad():
            Xt = eval_dset.x.to(config.device)
            Ut = eval_dset.u.to(config.device)
            St = eval_dset.s
            z_eval_t = None if getattr(eval_dset, "z", None) is None else torch.from_numpy(np.asarray(eval_dset.z).astype(int)).long().to(config.device)
            subj_eval_t = None if getattr(eval_dset, "subject_id", None) is None else torch.from_numpy(np.asarray(eval_dset.subject_id).astype(int)).long().to(config.device)
            u_raw_eval_t = None if getattr(eval_dset, "u_raw_id", None) is None else torch.from_numpy(np.asarray(eval_dset.u_raw_id).astype(int)).long().to(config.device)

            x_aligned_eval, _, Xt_front = _apply_frontend(
                Xt,
                z_eval_t,
                subj_eval_t,
                u_raw_eval_t,
                session_adapter=session_adapter,
                subject_adapter=subject_adapter,
                moment_aligner=moment_aligner,
                beta_nonlin=beta_eval,
                moment_enabled=(moment_aligner is not None and config.epochs >= int(getattr(config, "dist_align_start_epoch", config.epochs + 1))),
            )
            s_est = _eval_latents(model, backbone, Xt_front, Ut)
            recon_eval = _decode_reconstruction(model, s_est)
            s_np = s_est.detach().cpu().numpy()
            s_true_np = St.numpy()
            raw_x_np = Xt.detach().cpu().numpy()
            aligned_x_np = x_aligned_eval.detach().cpu().numpy()
            recon_x_np = recon_eval.detach().cpu().numpy()

        metrics = _compute_eval_metrics(
            args,
            config,
            seed,
            doc_name,
            eval_dset,
            s_true_np=s_true_np,
            s_np=s_np,
            z_all_eval=None if getattr(eval_dset, "z", None) is None else np.asarray(eval_dset.z),
            subject_all_eval=None if getattr(eval_dset, "subject_id", None) is None else np.asarray(eval_dset.subject_id),
            session_all_eval=None if getattr(eval_dset, "session_id", None) is None else np.asarray(eval_dset.session_id),
        )
        metrics["beta_eval"] = float(beta_eval)
        metrics["backbone"] = backbone
        perfs.append(metrics["full_mcc"])
        seed_metric_list.append(metrics)
        print(f"Final MCC on FULL evaluation set (seed={seed}): {metrics['full_mcc']:.6f}")
        if "perm_agreement" in metrics:
            print(f"[diag] perm_agreement={metrics['perm_agreement']:.3f}, sign_agreement={metrics.get('sign_agreement', float('nan')):.3f}")
        if "probe_acc" in metrics:
            print(f"[diag] probe_acc={metrics['probe_acc']:.3f}")

        if bool(getattr(config, "eval_align_to_anchor", False)) and getattr(eval_dset, "s_anchor", None) is not None and getattr(eval_dset, "z", None) is not None:
            try:
                S_anchor_np = eval_dset.s_anchor.numpy()
                z_eval = np.asarray(eval_dset.z).astype(int)
                n_domains_cfg = int(np.max(z_eval)) + 1
                s_aligned = np.zeros_like(s_np)
                for dom in range(n_domains_cfg):
                    mask = (z_eval == dom)
                    if mask.sum() < 50:
                        continue
                    corr = pearson_corr_matrix(S_anchor_np[mask], s_np[mask])
                    perm_a2e, _, _ = hungarian_match_abs(corr)
                    for a_idx, e_idx in enumerate(perm_a2e):
                        if e_idx < 0:
                            continue
                        sign = 1 if corr[a_idx, e_idx] >= 0 else -1
                        s_aligned[mask, a_idx] = sign * s_np[mask, e_idx]
                metrics["posthoc_anchor_aligned_mcc"] = float(mcc(s_true_np, s_aligned))
                print(f"Final MCC after ICA-anchor post-hoc alignment (seed={seed}): {metrics['posthoc_anchor_aligned_mcc']:.6f}")
            except Exception as e:
                metrics["posthoc_anchor_align_error"] = str(e)
                print(f"[diag] post-hoc anchor alignment failed: {e}")

        if bool(getattr(config, "mix_drift", False)) and (getattr(eval_dset, "z", None) is not None or getattr(eval_dset, "domain_id", None) is not None):
            try:
                scatter_artifacts = _save_domain_scatter_artifacts(
                    args=args,
                    config=config,
                    doc_name=doc_name,
                    seed=int(seed),
                    eval_dset=eval_dset,
                    raw_x_np=raw_x_np,
                    aligned_x_np=aligned_x_np,
                    latent_s_np=s_np,
                    recon_x_np=recon_x_np,
                )
                if scatter_artifacts:
                    metrics.setdefault("saved_artifacts", {}).update(scatter_artifacts)
            except Exception as e:
                metrics["domain_scatter_error"] = str(e)
                print(f"[diag] failed to save domain scatter artifacts: {e}")

        if bool(getattr(config, "save_last_model", True)):
            _save_checkpoint(
                args,
                doc_name,
                seed,
                tag="last",
                epoch=config.epochs,
                model=model,
                backbone=backbone,
                session_adapter=session_adapter,
                subject_adapter=subject_adapter,
                moment_aligner=moment_aligner,
                metrics=metrics,
                config_dict=config_dict,
            )
        _save_metrics_json(args, doc_name, seed, metrics)
        loss_hists.append(loss_hist)
        perf_hists.append(perf_hist)

    summary = {
        "doc": doc_name,
        "n_sims": int(args.n_sims),
        "seeds": list(range(args.seed, args.seed + args.n_sims)),
        "full_mcc_mean": float(np.mean(perfs)) if len(perfs) > 0 else float("nan"),
        "full_mcc_std": float(np.std(perfs)) if len(perfs) > 0 else float("nan"),
        "seed_metrics": seed_metric_list,
    }
    if z_all is not None and bool(getattr(config, "save_heatmaps", True)):
        try:
            summary["stacked_similarity"] = _save_stacked_similarity_artifacts(
                args=args,
                doc_name=doc_name,
                seeds=summary["seeds"],
                n_domains=int(np.max(z_all)) + 1,
            )
        except Exception as e:
            summary["stacked_similarity_error"] = str(e)
            print(f"[diag] failed to save stacked similarity artifacts: {e}")
    with open(os.path.join(args.log, f"{doc_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    return perfs, loss_hists, perf_hists
