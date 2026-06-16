"""Diagnostics and figure helpers for multi-domain / session drift experiments.

This module provides:
1) Domain-wise identity consistency diagnostics.
2) Session leakage probing.
3) Styled heatmap / stacked-similarity figure utilities.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# Palette from the user's reference heatmap (blue -> white -> red).
SIMILARITY_HEATMAP_COLORS: List[str] = [
    "#1B3B70",
    "#276FAF",
    "#4D9AC7",
    "#99C8E0",
    "#D4E6EF",
    "#F8F4F2",
    "#FBD8C3",
    "#F2A481",
    "#D6604D",
    "#B5202E",
    "#700C22",
]

# Palette from the user's reference scatter figure.
DOMAIN_SCATTER_COLORS: List[str] = [
    "#D5695D",
    "#F5B041",
    "#F6DA65",
    "#52BE80",
    "#91DFD0",
    "#5DADE2",
    "#A469BD",
    "#8A7067",
    "#FFBCA7",
    "#484F98",
    "#FFFF85",
]


def _standardize_cols(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Zero-mean, unit-std per column."""
    X = X.astype(np.float64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return X / (std + eps)


def pearson_corr_matrix(S_true: np.ndarray, S_est: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Pearson correlation matrix between columns of S_true and S_est.

    Returns:
        corr: shape (d_true, d_est)
    """
    A = _standardize_cols(S_true, eps=eps)
    B = _standardize_cols(S_est, eps=eps)
    n = max(1, A.shape[0])
    corr = (A.T @ B) / float(n)
    return corr.astype(np.float64)


@dataclass
class DomainAlignment:
    domain: int
    corr: np.ndarray              # (d_true, d_est)
    perm_true_to_est: np.ndarray  # (d_true,)
    perm_est_to_true: np.ndarray  # (d_est,)
    sign_est: np.ndarray          # (d_est,) sign of each est dim w.r.t its matched true dim
    score: float                  # mean abs corr after matching


def hungarian_match_abs(corr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Hungarian matching maximizing absolute correlation."""
    C = np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(-C)
    d_true, d_est = corr.shape
    perm_true_to_est = -np.ones(d_true, dtype=int)
    perm_est_to_true = -np.ones(d_est, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm_true_to_est[r] = int(c)
        perm_est_to_true[c] = int(r)
    score = float(C[row_ind, col_ind].mean()) if len(row_ind) > 0 else 0.0
    return perm_true_to_est, perm_est_to_true, score


def compute_domain_alignment(
    S_true: np.ndarray,
    S_est: np.ndarray,
    z: np.ndarray,
    n_domains: int,
    min_samples: int = 50,
) -> List[DomainAlignment]:
    """Compute per-domain correlation + matching info."""
    z = np.asarray(z).astype(int)
    out: List[DomainAlignment] = []
    for dom in range(n_domains):
        mask = (z == dom)
        if int(mask.sum()) < min_samples:
            continue
        corr = pearson_corr_matrix(S_true[mask], S_est[mask])
        p_t2e, p_e2t, score = hungarian_match_abs(corr)
        sign_est = np.ones(S_est.shape[1], dtype=int)
        for est_idx, true_idx in enumerate(p_e2t):
            if true_idx < 0:
                continue
            val = corr[true_idx, est_idx]
            sign_est[est_idx] = 1 if val >= 0 else -1
        out.append(
            DomainAlignment(
                domain=dom,
                corr=corr,
                perm_true_to_est=p_t2e,
                perm_est_to_true=p_e2t,
                sign_est=sign_est,
                score=score,
            )
        )
    return out


def identity_consistency_vs_ref(alignments: List[DomainAlignment], ref_domain: int = 0) -> Dict[str, float]:
    """Compute identity consistency (permutation/sign agreement) vs a reference domain."""
    if len(alignments) == 0:
        return {"perm_agreement": float("nan"), "sign_agreement": float("nan")}

    dom_to_align = {a.domain: a for a in alignments}
    ref = dom_to_align.get(ref_domain, alignments[0])

    ref_map = ref.perm_est_to_true
    ref_sign = ref.sign_est

    perm_agrs = []
    sign_agrs = []
    for a in alignments:
        if a.domain == ref.domain:
            continue
        amap = a.perm_est_to_true
        asign = a.sign_est
        matched = (ref_map >= 0) & (amap >= 0)
        if matched.sum() == 0:
            continue
        perm_agrs.append(float((amap[matched] == ref_map[matched]).mean()))
        sign_agrs.append(float((asign[matched] == ref_sign[matched]).mean()))
    if len(perm_agrs) == 0:
        return {"perm_agreement": float("nan"), "sign_agreement": float("nan")}
    return {"perm_agreement": float(np.mean(perm_agrs)), "sign_agreement": float(np.mean(sign_agrs))}


def session_leakage_probe(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 0,
) -> Dict[str, float]:
    """Train a simple logistic regression probe to predict session/domain id."""
    labels = np.asarray(labels).astype(int)
    X = np.asarray(features).astype(np.float64)
    n = X.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        labels = labels[idx]

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    try:
        lr = LogisticRegression(max_iter=2000, n_jobs=1, multi_class="auto", solver="lbfgs")
    except TypeError:
        try:
            lr = LogisticRegression(max_iter=2000, n_jobs=1, solver="lbfgs")
        except TypeError:
            try:
                lr = LogisticRegression(max_iter=2000, solver="lbfgs")
            except TypeError:
                lr = LogisticRegression(max_iter=2000)

    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), lr)
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    n_cls = int(np.unique(labels).size)
    chance = 1.0 / max(1, n_cls)
    return {"acc": acc, "chance": chance, "n_classes": float(n_cls), "n_samples": float(len(labels))}


def get_similarity_cmap():
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:
        return None
    return LinearSegmentedColormap.from_list("domain_similarity_reference", SIMILARITY_HEATMAP_COLORS)


def get_domain_scatter_palette(n_domains: int) -> List[str]:
    n_domains = max(0, int(n_domains))
    if n_domains == 0:
        return []

    # For a small number of domains, choose well-separated colors from the
    # reference palette instead of taking the first consecutive warm colors.
    preferred_idx = {
        1: [0],
        2: [0, 5],
        3: [0, 3, 5],
        4: [0, 3, 5, 9],
        5: [0, 1, 3, 5, 9],
        6: [0, 1, 3, 5, 7, 9],
    }
    if n_domains in preferred_idx:
        return [DOMAIN_SCATTER_COLORS[i] for i in preferred_idx[n_domains]]

    if n_domains <= len(DOMAIN_SCATTER_COLORS):
        idx = np.linspace(0, len(DOMAIN_SCATTER_COLORS) - 1, n_domains)
        idx = np.round(idx).astype(int).tolist()
        return [DOMAIN_SCATTER_COLORS[i] for i in idx]

    out = []
    for i in range(n_domains):
        out.append(DOMAIN_SCATTER_COLORS[i % len(DOMAIN_SCATTER_COLORS)])
    return out


def save_palette_csv(labels: List[str], colors: List[str], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "color_hex"])
        for label, color in zip(labels, colors):
            writer.writerow([label, color])


def save_corr_heatmap_csv(
    corr: np.ndarray,
    out_csv: str,
    doc: Optional[str] = None,
    seed: Optional[int] = None,
    domain: Optional[int] = None,
) -> None:
    """Save abs-correlation heatmap data in long CSV format."""
    C = np.asarray(corr, dtype=np.float64)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["doc", "seed", "domain", "true_source", "estimated_component", "corr", "abs_corr"])
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                writer.writerow([
                    "" if doc is None else doc,
                    "" if seed is None else int(seed),
                    "" if domain is None else int(domain),
                    int(i),
                    int(j),
                    float(C[i, j]),
                    float(abs(C[i, j])),
                ])


def save_corr_heatmap(
    corr: np.ndarray,
    out_png: str,
    title: str = "",
    out_pdf: Optional[str] = None,
) -> None:
    """Save a styled absolute-correlation heatmap image."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    C = np.abs(np.asarray(corr, dtype=np.float64))
    cmap = get_similarity_cmap()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(C, aspect="equal", cmap=cmap, vmin=0.0, vmax=max(1.0, float(C.max())))
    ax.set_xlabel("estimated component")
    ax.set_ylabel("true source")
    ax.set_xticks(np.arange(C.shape[1]))
    ax.set_yticks(np.arange(C.shape[0]))
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("|corr|", rotation=90)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=220)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def save_stacked_similarity_3d(
    corr_by_seed: Dict[int, np.ndarray],
    out_png: str,
    title: str = "",
    out_pdf: Optional[str] = None,
    out_csv: Optional[str] = None,
    doc: Optional[str] = None,
    domain: Optional[int] = None,
) -> None:
    """Save a 3D stacked similarity plot and its long-form CSV.

    Each seed becomes one thin layer. Cell colors encode absolute correlation.
    """
    if len(corr_by_seed) == 0:
        return

    seeds = sorted(int(s) for s in corr_by_seed.keys())
    mats = [np.abs(np.asarray(corr_by_seed[s], dtype=np.float64)) for s in seeds]
    d_true, d_est = mats[0].shape
    for mat in mats[1:]:
        if mat.shape != (d_true, d_est):
            raise ValueError("All correlation matrices must have the same shape for stacked plotting.")

    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["doc", "domain", "seed", "seed_layer", "true_source", "estimated_component", "abs_corr"])
            for layer, (seed, mat) in enumerate(zip(seeds, mats)):
                for i in range(d_true):
                    for j in range(d_est):
                        writer.writerow([
                            "" if doc is None else doc,
                            "" if domain is None else int(domain),
                            int(seed),
                            int(layer),
                            int(i),
                            int(j),
                            float(mat[i, j]),
                        ])

    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return

    cmap = get_similarity_cmap()
    vmin = min(float(mat.min()) for mat in mats)
    vmax = max(float(mat.max()) for mat in mats)
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-12))

    fig = plt.figure(figsize=(8.4, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    dx = 0.86
    dy = 0.86
    dz = 0.06
    for layer, (seed, mat) in enumerate(zip(seeds, mats)):
        z0 = float(layer)
        for i in range(d_true):
            for j in range(d_est):
                color = cmap(norm(float(mat[i, j])))
                ax.bar3d(
                    float(j),
                    float(i),
                    z0,
                    dx,
                    dy,
                    dz,
                    color=color,
                    shade=False,
                    edgecolor="white",
                    linewidth=0.18,
                    alpha=1.0,
                )

    ax.set_xlabel("estimated component", labelpad=8)
    ax.set_ylabel("true source", labelpad=8)
    ax.set_zlabel("seed", labelpad=10)
    ax.set_xticks(np.arange(d_est) + dx / 2.0)
    ax.set_xticklabels([str(i) for i in range(d_est)])
    ax.set_yticks(np.arange(d_true) + dy / 2.0)
    ax.set_yticklabels([str(i) for i in range(d_true)])
    ax.set_zticks(np.arange(len(seeds)) + dz / 2.0)
    ax.set_zticklabels([str(s) for s in seeds])
    ax.view_init(elev=26, azim=-58)
    ax.set_box_aspect((max(1, d_est), max(1, d_true), max(1, len(seeds)) * 0.8))
    if title:
        ax.set_title(title)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(left=0.02, right=0.80, bottom=0.05, top=0.94)
    cax = fig.add_axes([0.84, 0.18, 0.028, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel("|corr|", rotation=90)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=240)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)
