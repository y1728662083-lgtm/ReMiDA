"""ICA anchor utilities for multi-session alignment.

This module computes per-domain ICA sources and
align them into a *canonical* component order (up to a reference domain).

Why this is useful:
  - In multi-session drift experiments, pooled training of iVAE/TCL-like models
    can recover decent sources *within each session*, but the component identity
    (permutation/sign) may vary across sessions.
  - An ICA anchor gives each session a rough coordinate system; aligning ICA
    components across sessions produces a consistent ordering that we can use as
    a weak supervision signal (anchor loss) during training.

The code is designed to be dependency-light. It lazily imports scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from metrics.diagnostics import pearson_corr_matrix, hungarian_match_abs
import numpy as np
from scipy.optimize import linear_sum_assignment

def _uvar_signature(S, U_onehot, eps=1e-6):
    """
    S: (N, d) ICA sources for ONE domain
    U_onehot: (N, n_seg) one-hot labels
    return: sig (d, n_seg) where sig[j,k] = log Var(S[:,j] | u=k)
    """
    if U_onehot.ndim == 1:
        u = U_onehot.astype(int)
        n_seg = int(u.max() + 1)
    else:
        u = U_onehot.argmax(axis=1)
        n_seg = U_onehot.shape[1]

    N, d = S.shape
    sig = np.zeros((d, n_seg), dtype=np.float64)
    for k in range(n_seg):
        idx = (u == k)
        if idx.sum() < 2:
            sig[:, k] = 0.0
        else:
            v = np.var(S[idx], axis=0)  # (d,)
            sig[:, k] = np.log(v + eps)
    # Standardize each component signature to remove scale effects.
    sig = sig - sig.mean(axis=1, keepdims=True)
    sig = sig / (sig.std(axis=1, keepdims=True) + 1e-8)
    return sig

def _match_perm_by_signature(sig_ref, sig_dom):
    """
    sig_ref: (d, n_seg)
    sig_dom: (d, n_seg)
    return perm (length d): ref component i matches dom component perm[i]
    and mean_sim (float)
    """
    # similarity = cosine / dot product because signatures already standardized
    sim = sig_ref @ sig_dom.T / sig_ref.shape[1]  # (d, d)
    # Hungarian: maximize sim => minimize -sim
    row_ind, col_ind = linear_sum_assignment(-sim)
    # row_ind should be [0..d-1] but we sort just in case
    order = np.argsort(row_ind)
    perm = col_ind[order]
    mean_sim = float(sim[row_ind, col_ind].mean())
    return perm, mean_sim

def _standardize_cols(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return X / (std + eps)


@dataclass
class ICAAnchorInfo:
    domain: int
    perm_ref_to_dom: np.ndarray  # (d,)
    sign_ref: np.ndarray         # (d,) sign to apply after permutation
    mean_abs_corr: float


def _fit_fastica(
    X_fit: np.ndarray,
    n_components: int,
    seed: int,
    max_iter: int = 500,
) -> object:
    """Fit a FastICA model. Lazy-import sklearn so that ICA dependencies are loaded only when needed."""
    from sklearn.decomposition import FastICA

    # Use conservative arguments for compatibility across sklearn versions.
    ica = FastICA(n_components=n_components, random_state=seed, max_iter=max_iter)
    ica.fit(X_fit)
    return ica


def compute_fastica_anchors(
    X: np.ndarray,
    z: np.ndarray,
    U: np.ndarray,
    n_domains: int,
    n_components: int,
    seed: int = 0,
    ref_domain: int = 0,
    max_samples_per_domain: Optional[int] = 5000,
    max_iter: int = 500,
) -> Tuple[np.ndarray, List[ICAAnchorInfo]]:
    """Compute per-domain ICA sources and align them into a canonical order.

    Args:
        X: (N, d_x) observations.
        z: (N,) domain/session ids.
        n_domains: number of domains.
        n_components: ICA components (should match latent dim).
        seed: RNG seed.
        ref_domain: which domain defines the canonical component order.
        max_samples_per_domain: if not None, fit ICA on a subsample per domain for speed.
        max_iter: FastICA max iterations.

    Returns:
        S_anchor: (N, n_components) per-sample ICA sources aligned to ref_domain ordering.
        info: list of ICAAnchorInfo per domain.
    """
    z = np.asarray(z).astype(int)
    X = np.asarray(X)
    N, dx = X.shape
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    # Precompute per-domain ICA sources
    S_by_dom: Dict[int, np.ndarray] = {}
    ica_by_dom: Dict[int, object] = {}
    idx_by_dom: Dict[int, np.ndarray] = {}
    U_by_dom: Dict[int, np.ndarray] = {}

    rng_global = np.random.RandomState(seed)
    for dom in range(n_domains):
        idx = np.where(z == dom)[0]
        idx_by_dom[dom] = idx
        U_by_dom[dom] = U[idx]
        if idx.size == 0:
            continue
        X_dom = X[idx]
        X_dom = _standardize_cols(X_dom)

        # Fit on a subsample if requested, then transform all samples
        if max_samples_per_domain is not None and idx.size > max_samples_per_domain:
            sub = rng_global.choice(idx.size, size=int(max_samples_per_domain), replace=False)
            X_fit = X_dom[sub]
        else:
            X_fit = X_dom

        ica = _fit_fastica(X_fit, n_components=n_components, seed=seed + dom, max_iter=max_iter)
        S_dom = ica.transform(X_dom)
        S_dom = _standardize_cols(S_dom)
        S_by_dom[dom] = S_dom
        ica_by_dom[dom] = ica

    if ref_domain not in S_by_dom:
        # fallback to first available
        for dom in range(n_domains):
            if dom in S_by_dom:
                ref_domain = dom
                break
    S_ref = S_by_dom[ref_domain]
    U_ref = U_by_dom[ref_domain]
    sig_ref = _uvar_signature(S_ref, U_ref)
    # Align each domain's ICA sources to ref
    S_anchor = np.zeros((N, n_components), dtype=np.float32)
    info: List[ICAAnchorInfo] = []

    for dom in range(n_domains):
        idx = idx_by_dom.get(dom, None)
        if idx is None or idx.size == 0 or dom not in S_by_dom:
            continue
        S_dom = S_by_dom[dom]
        if dom == ref_domain:
            S_anchor[idx] = S_dom.astype(np.float32)
            info.append(ICAAnchorInfo(domain=dom,
                                      perm_ref_to_dom=np.arange(n_components, dtype=int),
                                      sign_ref=np.ones(n_components, dtype=int),
                                      mean_abs_corr=1.0))
            continue

        U_dom = U_by_dom[dom]
        sig_dom = _uvar_signature(S_dom, U_dom)

        perm_ref_to_dom, mean_sim = _match_perm_by_signature(sig_ref, sig_dom)

        # In the unpaired setting, the signature can reliably align permutation, but not sign.
        # Therefore sign_ref is set to 1; sign ambiguity is handled by a sign-invariant anchor loss.
        # sign_ref = np.ones(n_components, dtype=int)

        # # reorder only
        # S_aligned = S_dom[:, perm_ref_to_dom]
        # First align permutation using signatures.
        perm_ref_to_dom, mean_sim = _match_perm_by_signature(sig_ref, sig_dom)

        # =========================
        # Use ICA components_ to align signs in sensor space.
        # =========================
        try:
            W_ref = ica_by_dom[ref_domain].components_   # (d, dx)
            W_dom = ica_by_dom[dom].components_          # (d, dx)

            W_dom_reord = W_dom[perm_ref_to_dom]         # Reorder according to the reference order.
            dots = (W_dom_reord * W_ref).sum(axis=1)     # Dot product between each component and the reference component.

            sign_ref = np.where(dots >= 0, 1, -1).astype(int)
            sign_ref[sign_ref == 0] = 1
        except Exception as e:
            # Avoid crashing under version-specific sklearn behavior.
            sign_ref = np.ones(n_components, dtype=int)

        # reorder + sign flip
        S_aligned = S_dom[:, perm_ref_to_dom] * sign_ref.reshape(1, -1)
        S_anchor[idx] = S_aligned.astype(np.float32)
        info.append(ICAAnchorInfo(
            domain=dom,
            perm_ref_to_dom=perm_ref_to_dom.astype(int),
            sign_ref=sign_ref.astype(int),
            mean_abs_corr=float(mean_sim),   # Field name is kept for compatibility; here it denotes mean signature similarity.
        ))

    return S_anchor, info
