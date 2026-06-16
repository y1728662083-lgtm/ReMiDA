from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA, PCA

from .losses import psd_jsd_np
from .utils import logistic_probe, ridge_probe_accuracy


def compute_bandpower_features_np(x: np.ndarray, sfreq: float, band_defs: Dict[str, Iterable[float]] | None = None) -> np.ndarray:
    if band_defs is None:
        band_defs = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    spec = np.fft.rfft(x, axis=-1)
    psd = (spec.real ** 2 + spec.imag ** 2) / max(x.shape[-1], 1)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / float(sfreq))
    feats = []
    for lo, hi in band_defs.values():
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            feats.append(psd[..., mask].mean(axis=-1))
        else:
            feats.append(np.zeros(x.shape[:2], dtype=np.float64))
    return np.concatenate(feats, axis=1).astype(np.float32)


def compute_logspec_flat_features_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    spec = np.fft.rfft(x, axis=-1)
    mag = np.log1p(np.abs(spec))
    return mag.reshape(x.shape[0], -1).astype(np.float32)


def _flatten_batch_time(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0, 2, 1)).reshape(-1, x.shape[1])


class ReferenceICAProxy:
    def __init__(self, n_components: int, mean_: np.ndarray, unmixing_: np.ndarray) -> None:
        self.n_components = int(n_components)
        self.mean_ = np.asarray(mean_, dtype=np.float32)
        self.unmixing_ = np.asarray(unmixing_, dtype=np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xc = x - self.mean_[None, :, None]
        return np.einsum("kc,nct->nkt", self.unmixing_, xc).astype(np.float32)


def fit_reference_ica_proxy(x_ref: np.ndarray, n_components: int, seed: int = 42, max_iter: int = 1000) -> ReferenceICAProxy:
    flat = _flatten_batch_time(np.asarray(x_ref, dtype=np.float64))
    mean = flat.mean(axis=0)
    centered = flat - mean
    n_components = int(min(n_components, centered.shape[1], max(2, centered.shape[0] - 1)))
    try:
        ica = FastICA(n_components=n_components, random_state=seed, whiten="unit-variance", max_iter=max_iter, tol=1e-4)
        ica.fit(centered)
        unmixing = np.asarray(ica.components_, dtype=np.float64)
    except Exception:
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(centered)
        unmixing = np.asarray(pca.components_, dtype=np.float64)
    return ReferenceICAProxy(n_components=n_components, mean_=mean.astype(np.float32), unmixing_=unmixing.astype(np.float32))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _corr_matrix(profile_ref: np.ndarray, profile_non: np.ndarray) -> np.ndarray:
    k = profile_ref.shape[0]
    out = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            out[i, j] = _safe_corr(profile_ref[i], profile_non[j])
    return out


def _source_probe_features(s: np.ndarray) -> np.ndarray:
    mean = s.mean(axis=-1)
    std = s.std(axis=-1)
    rms = np.sqrt(np.mean(s.astype(np.float64) ** 2, axis=-1) + 1e-8)
    return np.concatenate([mean, std, rms], axis=1).astype(np.float32)


def compute_linear_proxy_metrics_for_pair(
    x_ref: np.ndarray,
    x_non: np.ndarray,
    sfreq: float,
    n_proxy_components: int = 32,
    seed: int = 42,
) -> Dict[str, float]:
    proxy = fit_reference_ica_proxy(x_ref, n_components=n_proxy_components, seed=seed)
    s_ref = proxy.transform(x_ref)
    s_non = proxy.transform(x_non)
    prof_ref = s_ref.mean(axis=0)
    prof_non = s_non.mean(axis=0)
    corr = _corr_matrix(prof_ref, prof_non)
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    matched = corr[row_ind, col_ind]
    order = np.argsort(row_ind)
    col_sorted = col_ind[order]
    matched_sorted = matched[order]
    full_mcc = float(np.mean(np.abs(matched)))
    perm_agreement = float(np.mean(col_sorted == np.arange(len(col_sorted))))
    sign_agreement = float(np.mean(matched_sorted > 0.0))
    src_feat = np.concatenate([_source_probe_features(s_ref), _source_probe_features(s_non)], axis=0)
    dom_lbl = np.concatenate([np.zeros(len(s_ref), dtype=np.int64), np.ones(len(s_non), dtype=np.int64)])
    probe_acc, probe_chance = ridge_probe_accuracy(src_feat, dom_lbl, alpha=1e-2, test_ratio=0.30, seed=seed)
    band_ref = compute_bandpower_features_np(x_ref, sfreq=sfreq)
    band_non = compute_bandpower_features_np(x_non, sfreq=sfreq)
    logspec_ref = compute_logspec_flat_features_np(x_ref)
    logspec_non = compute_logspec_flat_features_np(x_non)
    mean_gap = float(np.mean((band_ref.mean(axis=0) - band_non.mean(axis=0)) ** 2))
    x0 = band_ref - band_ref.mean(axis=0, keepdims=True)
    x1 = band_non - band_non.mean(axis=0, keepdims=True)
    cov0 = (x0.T @ x0) / max(len(x0) - 1, 1)
    cov1 = (x1.T @ x1) / max(len(x1) - 1, 1)
    cov_gap = float(np.mean((cov0 - cov1) ** 2))
    z = np.concatenate([band_ref, band_non], axis=0)
    pd2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    vals = pd2[pd2 > 0]
    gamma = 1.0 / max(float(np.median(vals)) if vals.size else 1.0, 1e-6)
    def _rbf(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)
    band_mmd = float(_rbf(band_ref, band_ref).mean() + _rbf(band_non, band_non).mean() - 2.0 * _rbf(band_ref, band_non).mean())
    composite = float(np.mean([1.0 - full_mcc, 1.0 - perm_agreement, 1.0 - sign_agreement, abs(probe_acc - probe_chance), mean_gap, cov_gap, band_mmd]))
    return {
        "full_mcc_proxy": full_mcc,
        "perm_agreement": perm_agreement,
        "sign_agreement": sign_agreement,
        "probe_acc": float(probe_acc),
        "probe_chance": float(probe_chance),
        "band_mean_gap": mean_gap,
        "band_cov_gap": cov_gap,
        "band_mmd": band_mmd,
        "psd_jsd": psd_jsd_np(logspec_ref, logspec_non),
        "composite_score": composite,
    }


def aggregate_linear_metrics_across_subjects(
    x_aligned: np.ndarray,
    metadata: pd.DataFrame,
    sfreq: float,
    reference_subject: str,
    split_indices: np.ndarray,
    seed: int = 42,
) -> Dict[str, float]:
    idx = np.asarray(split_indices, dtype=np.int64)
    meta = metadata.iloc[idx].reset_index(drop=True)
    x = x_aligned[idx]
    subj_arr = meta["subject"].astype(str).to_numpy()
    ref_mask = subj_arr == str(reference_subject)
    if not np.any(ref_mask):
        # fallback for tiny validation subsets: choose the subject with the most samples in this split
        uniq, counts = np.unique(subj_arr, return_counts=True)
        chosen = str(uniq[np.argmax(counts)])
        ref_mask = subj_arr == chosen
    x_ref = x[ref_mask]
    metrics = []
    for subj in sorted(set(meta["subject"].astype(str).tolist())):
        if subj == reference_subject:
            continue
        mask = meta["subject"].astype(str).to_numpy() == str(subj)
        if np.any(mask):
            metrics.append(compute_linear_proxy_metrics_for_pair(x_ref, x[mask], sfreq=sfreq, seed=seed))
    if not metrics:
        return compute_linear_proxy_metrics_for_pair(x_ref, x_ref, sfreq=sfreq, seed=seed)
    out = {}
    for key in metrics[0].keys():
        out[key] = float(np.mean([m[key] for m in metrics]))
    return out


def linear_val_score(recon_rmse: float, class_macro_f1: float, subject_probe_acc: float, subject_probe_chance: float) -> float:
    recon_term = float(recon_rmse) / (1.0 + float(recon_rmse))
    class_term = 1.0 - float(class_macro_f1)
    if subject_probe_chance >= 1.0:
        domain_term = 0.0
    else:
        domain_term = abs(float(subject_probe_acc) - float(subject_probe_chance)) / max(1.0 - float(subject_probe_chance), 1e-6)
    return float(0.40 * class_term + 0.35 * domain_term + 0.25 * recon_term)


def nonlinear_composite_score(recon_rmse: float, class_macro_f1: float, domain_acc: float, domain_chance: float, psd_jsd: float, center_loss: float) -> float:
    recon_term = float(recon_rmse) / (1.0 + float(recon_rmse))
    class_term = 1.0 - float(class_macro_f1)
    if domain_chance >= 1.0:
        domain_term = 0.0
    else:
        domain_term = abs(float(domain_acc) - float(domain_chance)) / max(1.0 - float(domain_chance), 1e-6)
    psd_term = float(psd_jsd) / (1.0 + float(psd_jsd))
    center_term = float(center_loss) / (1.0 + float(center_loss))
    return float(0.30 * class_term + 0.25 * domain_term + 0.20 * recon_term + 0.15 * psd_term + 0.10 * center_term)


def compute_nonlinear_metrics(
    x_in: np.ndarray,
    x_recon: np.ndarray,
    common_latent: np.ndarray,
    private_latent: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    sfreq: float,
    seed: int = 42,
) -> Dict[str, float]:
    recon_rmse = float(np.sqrt(np.mean((np.asarray(x_in, dtype=np.float32) - np.asarray(x_recon, dtype=np.float32)) ** 2)))
    band_in = compute_bandpower_features_np(x_in, sfreq=sfreq)
    band_recon = compute_bandpower_features_np(x_recon, sfreq=sfreq)
    band_mean_gap = float(np.mean((band_in.mean(axis=0) - band_recon.mean(axis=0)) ** 2))
    x0 = band_in - band_in.mean(axis=0, keepdims=True)
    x1 = band_recon - band_recon.mean(axis=0, keepdims=True)
    cov0 = (x0.T @ x0) / max(len(x0) - 1, 1)
    cov1 = (x1.T @ x1) / max(len(x1) - 1, 1)
    cov_gap = float(np.mean((cov0 - cov1) ** 2))
    psd_jsd = float(psd_jsd_np(compute_logspec_flat_features_np(x_in), compute_logspec_flat_features_np(x_recon)))
    class_probe = logistic_probe(common_latent, y, seed=seed)
    domain_probe_common = logistic_probe(common_latent, subject_ids, seed=seed)
    private_subject_probe = logistic_probe(private_latent, subject_ids, seed=seed)
    # class center dispersion
    disp = []
    for cls in np.unique(y):
        mask = y == cls
        if mask.sum() < 2:
            continue
        group = common_latent[mask]
        disp.append(np.mean((group - group.mean(axis=0, keepdims=True)) ** 2))
    class_disp = float(np.mean(disp)) if disp else 0.0
    score = nonlinear_composite_score(
        recon_rmse=recon_rmse,
        class_macro_f1=class_probe["macro_f1"],
        domain_acc=domain_probe_common["accuracy"],
        domain_chance=domain_probe_common["chance"],
        psd_jsd=psd_jsd,
        center_loss=class_disp,
    )
    return {
        "recon_rmse": recon_rmse,
        "band_mean_gap": band_mean_gap,
        "band_cov_gap": cov_gap,
        "psd_jsd": psd_jsd,
        "common_class_probe_acc": float(class_probe["accuracy"]),
        "common_class_probe_macro_f1": float(class_probe["macro_f1"]),
        "domain_probe_acc": float(domain_probe_common["accuracy"]),
        "domain_probe_chance": float(domain_probe_common["chance"]),
        "private_subject_probe_acc": float(private_subject_probe["accuracy"]),
        "private_subject_probe_macro_f1": float(private_subject_probe["macro_f1"]),
        "class_center_dispersion": class_disp,
        "composite_score": score,
    }
