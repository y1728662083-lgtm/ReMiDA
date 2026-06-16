
"""
Drifted synthetic data generator (session/domain-wise varying mixing).

Goal: create data that violates the "fixed mixing function" assumption used by
TCL / iVAE / IIA-style identifiable nonlinear ICA methods, by letting the mixing
MLP (or linear mixing) vary across sessions/domains.

This file contains the drift-aware data generator used by the synthetic benchmark experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .data import (
    generate_nonstationary_sources,
    generate_mixing_matrix,
    to_one_hot,
    lrelu,
    sigmoid,
)


def _get_activation(activation: str, slope: float, dtype):
    """Same activation options as in the original generator."""
    if activation == 'lrelu':
        return lambda x: lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        return sigmoid
    elif activation == 'xtanh':
        return lambda x: np.tanh(x) + slope * x
    elif activation == 'none':
        return lambda x: x
    else:
        raise ValueError(f'incorrect non linearity: {activation}')


def _sample_sources_given_params(
    n_per_seg: int,
    n_seg: int,
    d: int,
    prior: str,
    m: np.ndarray,
    L: np.ndarray,
    dtype=np.float32,
    seed: Optional[int] = None,
):
    """
    Sample sources using *fixed* segment parameters (m, L).
    This lets us keep the nonstationarity pattern (u -> (m,L)) the same across domains,
    while changing only the mixing.
    """
    if seed is not None:
        np.random.seed(seed)

    n = n_per_seg * n_seg
    labels = np.zeros(n, dtype=dtype)

    # Draw base noise according to prior (same options as original code)
    if prior == 'lap':
        sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
    elif prior == 'hs':
        # Avoid importing scipy here; `generate_nonstationary_sources` already uses scipy.
        # For drift experiments, `gauss` and `lap` are usually sufficient.
        raise ValueError("prior='hs' is not supported in drift_data.py; use 'gauss' or 'lap'.")
    elif prior == 'gauss':
        sources = np.random.randn(n, d).astype(dtype)
    else:
        raise ValueError(f'incorrect prior: {prior}')

    # Apply per-segment scaling and mean shift
    for seg in range(n_seg):
        segID = range(n_per_seg * seg, n_per_seg * (seg + 1))
        sources[segID] *= L[seg]
        sources[segID] += m[seg]
        labels[segID] = seg

    return sources, labels


def _normalize_columns(A: np.ndarray, eps: float = 1e-12):
    norms = np.sqrt((A ** 2).sum(axis=0, keepdims=True))
    return A / (norms + eps)


def _perturb_matrix(A: np.ndarray, strength: float, dtype=np.float32, max_tries: int = 50, cond_max: float = 1e4):
    """
    Add a small random perturbation to a mixing matrix, with column normalization
    and basic conditioning guard.
    """
    if strength <= 0:
        return A.copy()

    for _ in range(max_tries):
        noise = np.random.randn(*A.shape).astype(dtype)
        Ap = A + strength * noise
        Ap = _normalize_columns(Ap)
        # conditioning guard (very loose, only to avoid numerical blow-ups)
        try:
            c = np.linalg.cond(Ap)
        except Exception:
            c = np.inf
        if np.isfinite(c) and c < cond_max:
            return Ap.astype(dtype)

    # fallback: return the last try even if cond is large (still breaks fixed-mixing assumption)
    return Ap.astype(dtype)


def _apply_mixing_mlp(S: np.ndarray, A_layers: List[np.ndarray], act_f):
    """
    Apply a (possibly nonlinear) mixing MLP defined by linear layers A_layers and activation act_f.
    Last layer has no nonlinearity (same as original).
    """
    X = S
    for i, A in enumerate(A_layers):
        X = X @ A
        if i != len(A_layers) - 1:
            X = act_f(X)
    return X


def _generate_repeat_linearity_layers(
    d_sources: int,
    d_data: int,
    n_layers: int,
    lin_type: str,
    n_iter_4_cond: int,
    dtype,
    staircase: bool,
):
    """
    Replicates the `repeat_linearity=True` branch from the original generator:
    - First layer uses A (d_sources -> d_data) with nonlinearity
    - Remaining layers reuse B (d_data -> d_data) with nonlinearity except last
    Returns (A, B).
    """
    assert n_layers > 1
    A = generate_mixing_matrix(d_sources, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype, staircase=staircase)
    if d_sources != d_data:
        B = generate_mixing_matrix(d_data, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
    else:
        B = A
    return A, B


def _apply_repeat_linearity(S: np.ndarray, A: np.ndarray, B: np.ndarray, act_f, n_layers: int):
    X = act_f(S @ A)
    for nl in range(1, n_layers):
        if nl == n_layers - 1:
            X = X @ B
        else:
            X = act_f(X @ B)
    return X


@dataclass
class DriftExtra:
    """Extra information for diagnostics (not needed by baseline models)."""
    z: np.ndarray                      # domain/session label per sample, shape (N,)
    mixing: List[Dict[str, Any]]       # per-domain mixing parameters (A_layers or (A,B))


def generate_data_with_mixing_drift(
    n_per_seg: int,
    n_seg: int,
    d_sources: int,
    d_data: Optional[int] = None,
    n_layers: int = 3,
    prior: str = 'gauss',
    activation: str = 'lrelu',
    seed: int = 10,
    slope: float = 0.1,
    var_bounds: np.ndarray = np.array([0.5, 3]),
    lin_type: str = 'uniform',
    n_iter_4_cond: int = int(1e4),
    dtype=np.float32,
    noisy: float = 0.0,
    uncentered: bool = False,
    centers: Optional[np.ndarray] = None,
    staircase: bool = False,
    discrete: bool = False,
    one_hot_labels: bool = True,
    repeat_linearity: bool = False,
    # --- new (drift) arguments ---
    n_domains: int = 2,
    drift_mode: str = 'perturb',    # 'perturb' or 'independent'
    drift_strength: float = 0.1,    # how strong the per-domain perturbation is
    share_source_params: bool = True,
    return_extra: bool = True,
):
    """
    Generate synthetic data with *domain-wise varying mixing*.

    Key idea:
      - Keep the auxiliary variable u (segment id) and its source modulation pattern the same.
      - Change the mixing across domains/sessions: X_d = f_d(S_d).
    This violates the "single fixed mixing function f" assumption behind iVAE/IIA/TCL identifiability proofs.

    Returns:
      If return_extra is True:
        (S, X, U, M, L, extra)
      Else:
        (S, X, U, M, L)
    """
    if d_data is None:
        d_data = d_sources

    if n_domains < 2:
        raise ValueError("n_domains must be >= 2 to create mixing drift.")

    if seed is not None:
        np.random.seed(seed)

    act_f = _get_activation(activation, slope, dtype)

    # ---------------------------------------------------------
    # 1) Source generation: create a shared nonstationarity pattern (u -> (m,L))
    # ---------------------------------------------------------
    S0, U0, M, L = generate_nonstationary_sources(
        n_per_seg, n_seg, d_sources,
        prior=prior,
        var_bounds=var_bounds,
        dtype=dtype,
        uncentered=uncentered,
        centers=centers,
        staircase=staircase,
    )

    n_base = n_per_seg * n_seg
    S_list = [S0]
    U_list = [U0]
    z_list = [np.zeros(n_base, dtype=np.int64)]

    # Sample sources for other domains with the SAME (M, L) if share_source_params=True
    for dom in range(1, n_domains):
        if share_source_params:
            Sd, Ud = _sample_sources_given_params(
                n_per_seg=n_per_seg,
                n_seg=n_seg,
                d=d_sources,
                prior=prior,
                m=M,
                L=L,
                dtype=dtype,
                seed=seed + dom * 1000,
            )
        else:
            Sd, Ud, _, _ = generate_nonstationary_sources(
                n_per_seg, n_seg, d_sources,
                prior=prior,
                var_bounds=var_bounds,
                dtype=dtype,
                uncentered=uncentered,
                centers=centers,
                staircase=staircase,
            )
        S_list.append(Sd)
        U_list.append(Ud)
        z_list.append(np.full(n_base, dom, dtype=np.int64))

    S = np.vstack(S_list).astype(dtype)
    U = np.concatenate(U_list).astype(dtype)
    z = np.concatenate(z_list)

    # ---------------------------------------------------------
    # 2) Mixing generation + application per domain
    # ---------------------------------------------------------
    mixing_info: List[Dict[str, Any]] = []
    X_domains = []

    # --- base mixing (domain 0) ---
    if repeat_linearity:
        A0, B0 = _generate_repeat_linearity_layers(
            d_sources=d_sources,
            d_data=d_data,
            n_layers=n_layers,
            lin_type=lin_type,
            n_iter_4_cond=n_iter_4_cond,
            dtype=dtype,
            staircase=staircase,
        )
        mixing_info.append({"mode": "repeat_linearity", "A": A0, "B": B0})
    else:
        # general MLP: sample A for each layer (as in original)
        A_layers0 = []
        dim_in = d_sources
        for nl in range(n_layers):
            A = generate_mixing_matrix(dim_in, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype,
                                       staircase=staircase)
            A_layers0.append(A)
            dim_in = d_data
        mixing_info.append({"mode": "mlp", "A_layers": A_layers0})

    # Apply domain 0 mixing
    S_dom0 = S[0:n_base]
    if repeat_linearity:
        X0 = _apply_repeat_linearity(S_dom0, mixing_info[0]["A"], mixing_info[0]["B"], act_f, n_layers)
    else:
        X0 = _apply_mixing_mlp(S_dom0, mixing_info[0]["A_layers"], act_f)
    X_domains.append(X0)

    # --- other domains ---
    for dom in range(1, n_domains):
        if drift_mode not in {"perturb", "independent"}:
            raise ValueError("drift_mode must be 'perturb' or 'independent'.")

        if drift_mode == "independent":
            # sample a fresh mixing for each domain (strongest violation)
            if repeat_linearity:
                Ad, Bd = _generate_repeat_linearity_layers(
                    d_sources=d_sources,
                    d_data=d_data,
                    n_layers=n_layers,
                    lin_type=lin_type,
                    n_iter_4_cond=n_iter_4_cond,
                    dtype=dtype,
                    staircase=staircase,
                )
                mixing_info.append({"mode": "repeat_linearity", "A": Ad, "B": Bd})
            else:
                A_layersd = []
                dim_in = d_sources
                for nl in range(n_layers):
                    A = generate_mixing_matrix(dim_in, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype,
                                               staircase=staircase)
                    A_layersd.append(A)
                    dim_in = d_data
                mixing_info.append({"mode": "mlp", "A_layers": A_layersd})

        else:  # perturb
            np.random.seed(seed + dom * 10000)  # deterministic per domain
            if repeat_linearity:
                Ad = _perturb_matrix(mixing_info[0]["A"], strength=drift_strength, dtype=dtype)
                Bd = _perturb_matrix(mixing_info[0]["B"], strength=drift_strength, dtype=dtype)
                mixing_info.append({"mode": "repeat_linearity", "A": Ad, "B": Bd})
            else:
                A_layersd = [
                    _perturb_matrix(A, strength=drift_strength, dtype=dtype)
                    for A in mixing_info[0]["A_layers"]
                ]
                mixing_info.append({"mode": "mlp", "A_layers": A_layersd})

        Sd = S[dom * n_base:(dom + 1) * n_base]
        if repeat_linearity:
            Xd = _apply_repeat_linearity(Sd, mixing_info[dom]["A"], mixing_info[dom]["B"], act_f, n_layers)
        else:
            Xd = _apply_mixing_mlp(Sd, mixing_info[dom]["A_layers"], act_f)
        X_domains.append(Xd)

    X = np.vstack(X_domains).astype(dtype)

    # add noise:
    if noisy:
        X += noisy * np.random.randn(*X.shape).astype(dtype)

    if discrete:
        X = np.random.binomial(1, sigmoid(X)).astype(dtype)

    # one-hot auxiliary variable u (segment id)
    if one_hot_labels:
        # Keep one-hot labels in float dtype (consistent with the original generator)
        # so the iVAE prior network can consume them without dtype casting issues.
        U = to_one_hot([U], m=n_seg)[0].astype(dtype)

    if return_extra:
        extra = DriftExtra(z=z, mixing=mixing_info)
        return S, X, U, M, L, extra
    return S, X, U, M, L


@dataclass
class HierarchicalDriftExtra:
    """Metadata for hierarchical subject/session drift benchmarks."""

    z: np.ndarray
    subject_id: np.ndarray
    session_id: np.ndarray
    u_raw_id: np.ndarray
    subject_mixing: List[Dict[str, Any]]
    session_drift: List[np.ndarray]
    reference_domains: np.ndarray
    subject_of_domain: np.ndarray



def _parse_sessions_per_subject(sessions_per_subject, n_subjects: int) -> List[int]:
    if isinstance(sessions_per_subject, int):
        return [int(sessions_per_subject)] * int(n_subjects)
    vals = list(sessions_per_subject)
    if len(vals) != int(n_subjects):
        raise ValueError("sessions_per_subject must be int or list of length n_subjects")
    return [int(v) for v in vals]



def _sample_session_linear_drift(d: int, strength: float, dtype=np.float32, max_tries: int = 100, cond_max: float = 1e4):
    if strength <= 0:
        return np.eye(d, dtype=dtype)
    for _ in range(max_tries):
        E = np.random.randn(d, d).astype(dtype)
        R = np.eye(d, dtype=dtype) + float(strength) * E
        try:
            c = np.linalg.cond(R)
        except Exception:
            c = np.inf
        if np.isfinite(c) and c < cond_max:
            return R.astype(dtype)
    return R.astype(dtype)



def _generate_subject_mixing(
    d_sources: int,
    d_data: int,
    n_layers: int,
    lin_type: str,
    n_iter_4_cond: int,
    dtype,
    staircase: bool,
    repeat_linearity: bool,
):
    if repeat_linearity:
        A, B = _generate_repeat_linearity_layers(
            d_sources=d_sources,
            d_data=d_data,
            n_layers=n_layers,
            lin_type=lin_type,
            n_iter_4_cond=n_iter_4_cond,
            dtype=dtype,
            staircase=staircase,
        )
        return {"mode": "repeat_linearity", "A": A, "B": B}

    A_layers = []
    dim_in = d_sources
    for _ in range(n_layers):
        A = generate_mixing_matrix(
            dim_in,
            d_data,
            lin_type=lin_type,
            n_iter_4_cond=n_iter_4_cond,
            dtype=dtype,
            staircase=staircase,
        )
        A_layers.append(A)
        dim_in = d_data
    return {"mode": "mlp", "A_layers": A_layers}



def _perturb_subject_mixing(base: Dict[str, Any], strength: float, dtype=np.float32):
    if base["mode"] == "repeat_linearity":
        return {
            "mode": "repeat_linearity",
            "A": _perturb_matrix(base["A"], strength=strength, dtype=dtype),
            "B": _perturb_matrix(base["B"], strength=strength, dtype=dtype),
        }
    return {
        "mode": "mlp",
        "A_layers": [_perturb_matrix(A, strength=strength, dtype=dtype) for A in base["A_layers"]],
    }



def _apply_subject_mixing(S: np.ndarray, mix: Dict[str, Any], act_f, n_layers: int):
    if mix["mode"] == "repeat_linearity":
        return _apply_repeat_linearity(S, mix["A"], mix["B"], act_f, n_layers)
    return _apply_mixing_mlp(S, mix["A_layers"], act_f)



def generate_hierarchical_data_with_mixing_drift(
    n_per_seg: int,
    n_seg: int,
    d_sources: int,
    d_data: Optional[int] = None,
    n_layers: int = 3,
    prior: str = "gauss",
    activation: str = "xtanh",
    seed: int = 10,
    slope: float = 0.1,
    var_bounds: np.ndarray = np.array([0.5, 3]),
    lin_type: str = "uniform",
    n_iter_4_cond: int = int(1e4),
    dtype=np.float32,
    noisy: float = 0.0,
    uncentered: bool = False,
    centers: Optional[np.ndarray] = None,
    staircase: bool = False,
    discrete: bool = False,
    one_hot_labels: bool = True,
    repeat_linearity: bool = True,
    n_subjects: int = 2,
    sessions_per_subject: Any = 3,
    subject_mixing_mode: str = "perturb",
    subject_nonlinear_strength: float = 0.2,
    session_drift_strength: float = 0.05,
    share_source_params: bool = True,
    source_conditional_shift: bool = False,
    source_mean_shift_strength: float = 0.0,
    source_scale_shift_strength: float = 0.0,
    return_extra: bool = True,
):
    """Generate hierarchical subject/session drift synthetic data.

    Observations follow

        x_{b,j} = R_{b,j} F_b(s_u) + eps,

    where b indexes subject, j indexes session, and u is the shared U_raw.
    Subject-level differences are induced by F_b, while session-level differences
    are induced by near-identity linear drift matrices R_{b,j}.
    """
    if d_data is None:
        d_data = d_sources
    if seed is not None:
        np.random.seed(seed)

    n_subjects = int(n_subjects)
    sess_list = _parse_sessions_per_subject(sessions_per_subject, n_subjects)
    act_f = _get_activation(activation, slope, dtype)

    _, _, M_base, L_base = generate_nonstationary_sources(
        n_per_seg,
        n_seg,
        d_sources,
        prior=prior,
        var_bounds=var_bounds,
        dtype=dtype,
        uncentered=uncentered,
        centers=centers,
        staircase=staircase,
    )

    base_subject_mix = _generate_subject_mixing(
        d_sources=d_sources,
        d_data=d_data,
        n_layers=n_layers,
        lin_type=lin_type,
        n_iter_4_cond=n_iter_4_cond,
        dtype=dtype,
        staircase=staircase,
        repeat_linearity=repeat_linearity,
    )

    subject_mixing: List[Dict[str, Any]] = []
    for subj in range(n_subjects):
        if subj == 0:
            subject_mixing.append(base_subject_mix)
            continue
        if subject_mixing_mode == "independent":
            mix = _generate_subject_mixing(
                d_sources=d_sources,
                d_data=d_data,
                n_layers=n_layers,
                lin_type=lin_type,
                n_iter_4_cond=n_iter_4_cond,
                dtype=dtype,
                staircase=staircase,
                repeat_linearity=repeat_linearity,
            )
        elif subject_mixing_mode == "perturb":
            np.random.seed(seed + 10000 + subj)
            mix = _perturb_subject_mixing(base_subject_mix, strength=float(subject_nonlinear_strength), dtype=dtype)
        else:
            raise ValueError("subject_mixing_mode must be 'perturb' or 'independent'")
        subject_mixing.append(mix)

    X_blocks = []
    S_blocks = []
    U_blocks = []
    u_id_blocks = []
    z_blocks = []
    subj_blocks = []
    sess_blocks = []
    session_drift: List[np.ndarray] = []
    reference_domains = []
    subject_of_domain = []

    dom_counter = 0
    for subj in range(n_subjects):
        n_sess = int(sess_list[subj])
        reference_domains.append(dom_counter)
        for sess in range(n_sess):
            if share_source_params:
                M_subj = np.array(M_base, copy=True)
                L_subj = np.array(L_base, copy=True)
            else:
                _, _, M_subj, L_subj = generate_nonstationary_sources(
                    n_per_seg,
                    n_seg,
                    d_sources,
                    prior=prior,
                    var_bounds=var_bounds,
                    dtype=dtype,
                    uncentered=uncentered,
                    centers=centers,
                    staircase=staircase,
                )

            if source_conditional_shift:
                rng_shift = np.random.RandomState(seed + 20000 + subj)
                if source_mean_shift_strength > 0:
                    M_subj = M_subj + float(source_mean_shift_strength) * rng_shift.randn(*M_subj.shape).astype(dtype)
                if source_scale_shift_strength > 0:
                    scale = np.exp(float(source_scale_shift_strength) * rng_shift.randn(*L_subj.shape)).astype(dtype)
                    L_subj = L_subj * scale

            Sd, Ud = _sample_sources_given_params(
                n_per_seg=n_per_seg,
                n_seg=n_seg,
                d=d_sources,
                prior=prior,
                m=M_subj,
                L=L_subj,
                dtype=dtype,
                seed=seed + subj * 1000 + sess * 31,
            )
            X_subject = _apply_subject_mixing(Sd, subject_mixing[subj], act_f, n_layers=n_layers)
            R = np.eye(d_data, dtype=dtype) if sess == 0 else _sample_session_linear_drift(d_data, session_drift_strength, dtype=dtype)
            Xd = (X_subject @ R).astype(dtype)
            if noisy:
                Xd = Xd + noisy * np.random.randn(*Xd.shape).astype(dtype)
            if discrete:
                Xd = np.random.binomial(1, sigmoid(Xd)).astype(dtype)

            X_blocks.append(Xd)
            S_blocks.append(Sd.astype(dtype))
            U_blocks.append(Ud.astype(dtype))
            u_id_blocks.append(Ud.astype(np.int64))
            z_blocks.append(np.full(Ud.shape[0], dom_counter, dtype=np.int64))
            subj_blocks.append(np.full(Ud.shape[0], subj, dtype=np.int64))
            sess_blocks.append(np.full(Ud.shape[0], sess, dtype=np.int64))
            session_drift.append(R.astype(dtype))
            subject_of_domain.append(subj)
            dom_counter += 1

    X = np.vstack(X_blocks).astype(dtype)
    S = np.vstack(S_blocks).astype(dtype)
    U_id = np.concatenate(U_blocks).astype(np.int64)
    u_raw_id = np.concatenate(u_id_blocks).astype(np.int64)
    z = np.concatenate(z_blocks).astype(np.int64)
    subject_id = np.concatenate(subj_blocks).astype(np.int64)
    session_id = np.concatenate(sess_blocks).astype(np.int64)

    U = U_id
    if one_hot_labels:
        U = to_one_hot([U_id], m=n_seg)[0].astype(dtype)

    if return_extra:
        extra = HierarchicalDriftExtra(
            z=z,
            subject_id=subject_id,
            session_id=session_id,
            u_raw_id=u_raw_id,
            subject_mixing=subject_mixing,
            session_drift=session_drift,
            reference_domains=np.asarray(reference_domains, dtype=np.int64),
            subject_of_domain=np.asarray(subject_of_domain, dtype=np.int64),
        )
        return S, X, U, M_base, L_base, extra
    return S, X, U, M_base, L_base
