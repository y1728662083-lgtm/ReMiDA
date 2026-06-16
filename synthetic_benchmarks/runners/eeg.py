"""Runner for real EEG datasets (e.g., OpenNeuro ds003626) stored as .npz.

Why this runner exists
----------------------
The original ivae runner in this repo assumes synthetic data and requires ground-truth sources S
for MCC evaluation. Real EEG has no ground-truth sources, so we:
  - Train the same cleanIVAE model (optionally with drift adapter + ICA anchor loss)
  - Evaluate with proxy metrics:
      * session leakage probe: predict z (session/domain) from estimated sources
      * cross-domain identity consistency *relative to ICA anchors* (anchor is a weak reference)
      * cross-session direction decoding accuracy (trial-level) from estimated sources

Input dataset format
--------------------
Use scripts/prepare_ds003626_npz.py to produce an .npz containing:
  X (N,D), U (N,K), y (N,), z (N,), trial_id (N,)
Optionally S_anchor (N,dl) if you precomputed anchors.

Config keys used
----------------
Required:
  dataset: ds003626_npz
  eeg_npz_path: path/to/file.npz
  dl, hidden_dim, n_layers, activation, epochs, batch_size, lr

Optional (same names as synthetic drift runner):
  use_ica_anchor: true/false
  lambda_ica_anchor: float
  use_drift_adapter: true/false
  adapter_init: 'ica' (recommended)
  lambda_adapter_init, lambda_drift_adapter, lambda_drift_ortho, lr_adapter

EEG-specific evaluation:
  decode_direction: true/false
  decode_max_iter: 2000
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data.eeg_npz import EEGNPZDataset
from metrics.diagnostics import (
    compute_domain_alignment,
    identity_consistency_vs_ref,
    pearson_corr_matrix,
    hungarian_match_abs,
    save_corr_heatmap,
    session_leakage_probe,
)
from models import cleanIVAE
from models.drift_adapter import DomainLinearAdapter
from utils.ica_anchor import compute_fastica_anchors


def _trial_aggregate(features: np.ndarray, y: np.ndarray, z: np.ndarray, trial_id: np.ndarray):
    """Aggregate timepoint-level features into trial-level by mean."""
    trial_id = np.asarray(trial_id).astype(int).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    z = np.asarray(z).astype(int).reshape(-1)

    uniq = np.unique(trial_id)
    # Map trial_id to 0..T-1 for dense accumulation
    tmap = {int(t): i for i, t in enumerate(uniq.tolist())}
    tid = np.vectorize(lambda t: tmap[int(t)])(trial_id).astype(int)
    T = int(len(uniq))

    F = features.shape[1]
    sums = np.zeros((T, F), dtype=np.float64)
    cnt = np.zeros((T,), dtype=np.float64)
    np.add.at(sums, tid, features.astype(np.float64))
    np.add.at(cnt, tid, 1.0)
    feat_trial = (sums / (cnt[:, None] + 1e-12)).astype(np.float32)

    # Trial labels: take the first occurrence
    y_trial = np.zeros((T,), dtype=np.int64)
    z_trial = np.zeros((T,), dtype=np.int64)
    for t_raw, t_idx in tmap.items():
        first = np.where(trial_id == t_raw)[0][0]
        y_trial[t_idx] = int(y[first])
        z_trial[t_idx] = int(z[first])
    return feat_trial, y_trial, z_trial


def _decode_direction_cross_session(feat_trial: np.ndarray, y_trial: np.ndarray, z_trial: np.ndarray, max_iter: int = 2000):
    """Train on all-but-one sessions and test on the held-out session."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_domains = int(np.max(z_trial) + 1)
    accs = {}

    for dom in range(n_domains):
        tr = z_trial != dom
        te = z_trial == dom
        if te.sum() == 0 or tr.sum() == 0:
            continue

        # Robust default; multi_class arg is ignored/handled depending on sklearn version
        try:
            clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", multi_class="auto")
        except TypeError:
            clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")

        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(feat_trial[tr], y_trial[tr])
        pred = pipe.predict(feat_trial[te])
        acc = float((pred == y_trial[te]).mean())
        accs[int(dom)] = acc

    if len(accs) == 0:
        return {"mean": None, "per_domain": {}}

    return {"mean": float(np.mean(list(accs.values()))), "per_domain": accs}


def runner(args, config):
    st = time.time()

    # ----------------------------
    # Load dataset
    # ----------------------------
    npz_path = getattr(config, "eeg_npz_path", None)
    if hasattr(args, "data") and args.data:
        npz_path = args.data
    if not npz_path:
        raise ValueError("For dataset=ds003626_npz, set eeg_npz_path in YAML or pass --data")

    dset = EEGNPZDataset(npz_path, device="cpu")
    d_data = dset.data_dim
    d_aux = dset.aux_dim
    d_latent = int(getattr(config, "dl", d_data))

    if d_latent <= 0:
        raise ValueError(f"Invalid dl={d_latent}")

    # ----------------------------
    # Optionally compute ICA anchors (weak reference) + attach to dataset
    # ----------------------------
    use_anchor = bool(getattr(config, "use_ica_anchor", True))
    if use_anchor and (dset.s_anchor is None):
        # Use U for u-variance signature; use raw one-hot for signature
        X_np = dset.x.numpy().astype(np.float64)
        U_np = dset.u.numpy().astype(np.float64)
        z_np = dset.z.numpy().astype(int)
        n_domains = int(z_np.max() + 1)
        print(f"[EEG] Computing ICA anchors: n_domains={n_domains}, n_components={d_latent}")
        S_anchor = compute_fastica_anchors(
            X=X_np,
            z=z_np,
            U=U_np,
            n_domains=n_domains,
            n_components=d_latent,
            max_samples_per_domain=int(getattr(config, "ica_anchor_max_samples", 50000)),
            random_state=int(getattr(config, "ica_anchor_seed", 0)),
        ).astype(np.float32)
        dset.s_anchor = torch.from_numpy(S_anchor).float()

    # ----------------------------
    # DataLoader
    # ----------------------------
    num_workers = int(getattr(config, "num_workers", 0))
    if torch.cuda.is_available():
        loader_params = {"num_workers": num_workers, "pin_memory": True}
    else:
        loader_params = {"num_workers": 0}
    data_loader = DataLoader(dset, batch_size=int(config.batch_size), shuffle=True, drop_last=True, **loader_params)

    # ----------------------------
    # Drift adapter (optional)
    # ----------------------------
    adapter = None
    freeze_adapter = bool(getattr(config, "freeze_adapter", False))
    if bool(getattr(config, "use_drift_adapter", False)):
        n_domains = dset.n_domains
        adapter = DomainLinearAdapter(n_domains=n_domains, d=d_data).to(config.device)

        # Adapter init
        adapter_init = str(getattr(config, "adapter_init", "identity")).lower()
        if adapter_init == "ica":
            if dset.s_anchor is None:
                raise RuntimeError("adapter_init=ica requires ICA anchors (use_ica_anchor: true)")
            if d_data != d_latent:
                raise ValueError(
                    f"adapter_init=ica expects data_dim == latent_dim so W_init is square. Got D={d_data}, dl={d_latent}.\n"
                    "Tip: set --pca_dim == dl in prepare_ds003626_npz.py and use dl equal to that pca_dim."
                )
            # Compute domain stats and W_init to map X_std -> S_anchor
            from runners.ivae import _compute_domain_stats_and_winit

            X_np = dset.x.numpy().astype(np.float32)
            z_np = dset.z.numpy().astype(int)
            S_anchor_np = dset.s_anchor.numpy().astype(np.float32)
            mu, std, W_init = _compute_domain_stats_and_winit(X_np, z_np, S_anchor_np=S_anchor_np, ridge=float(getattr(config, "adapter_ridge", 1e-3)))
            adapter.mu.data.copy_(mu)
            adapter.std.data.copy_(std)
            adapter.W.data.copy_(W_init)
            print("[EEG] Adapter initialized from ICA anchors (per-domain).")
        else:
            print(f"[EEG] Adapter init = {adapter_init} (identity).")

        # Snapshot for tether regularization
        adapter_init_params = {n: p.detach().clone() for n, p in adapter.named_parameters() if p.requires_grad}
        adapter_init_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    else:
        adapter_init_params = None
        adapter_init_state = None

    # ----------------------------
    # Train (multi-seed)
    # ----------------------------
    results = {"seeds": [], "npz": npz_path}

    for seed in range(int(args.seed), int(args.seed) + int(args.n_sims)):
        if adapter is not None and adapter_init_state is not None:
            adapter.load_state_dict(adapter_init_state, strict=True)
            adapter_init_params = {n: p.detach().clone() for n, p in adapter.named_parameters() if p.requires_grad}

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = cleanIVAE(
            data_dim=d_data,
            latent_dim=d_latent,
            aux_dim=d_aux,
            hidden_dim=int(config.hidden_dim),
            n_layers=int(config.n_layers),
            activation=str(config.activation),
            slope=0.1,
        ).to(config.device)

        lr_adapter = float(getattr(config, "lr_adapter", float(config.lr)))
        param_groups = [{"params": model.parameters(), "lr": float(config.lr)}]
        if adapter is not None and (not freeze_adapter):
            param_groups.append({"params": adapter.parameters(), "lr": lr_adapter})
        optimizer = optim.Adam(param_groups)

        lam_anchor = float(getattr(config, "lambda_ica_anchor", 0.0))
        lam_delta = float(getattr(config, "lambda_drift_adapter", 0.0))
        lam_ortho = float(getattr(config, "lambda_drift_ortho", 0.0))
        lam_init = float(getattr(config, "lambda_adapter_init", 0.0))

        loss_hist = []
        anchor_hist = []

        for epoch in range(1, int(config.epochs) + 1):
            model.train()
            train_loss = 0.0
            train_anchor = 0.0
            n_anchor_batches = 0

            for batch in data_loader:
                # Without anchors: (x,u,y,z,trial)
                # With anchors:    (x,u,y,z,trial,s_anchor)
                if len(batch) == 5:
                    x, u, _, z_batch, _, = batch
                    s_anchor = None
                elif len(batch) == 6:
                    x, u, _, z_batch, _, s_anchor = batch
                else:
                    raise ValueError(f"Unexpected batch tuple length={len(batch)}")

                x = x.to(config.device)
                u = u.to(config.device)
                z_dev = z_batch.to(config.device).long()
                if s_anchor is not None:
                    s_anchor = s_anchor.to(config.device)

                # Apply adapter
                if adapter is not None:
                    x = adapter(x, z_dev)

                optimizer.zero_grad()
                loss, z_lat = model.elbo(x, u, len(dset), a=float(config.a), b=float(config.b), c=float(config.c), d=float(config.d))

                # Adapter regularization
                if adapter is not None and (not freeze_adapter):
                    if lam_init > 0 and adapter_init_params is not None:
                        init_reg = 0.0
                        for n, p in adapter.named_parameters():
                            if not p.requires_grad:
                                continue
                            init_reg = init_reg + (p - adapter_init_params[n]).pow(2).mean()
                        loss = loss + lam_init * init_reg
                    if lam_delta > 0:
                        loss = loss + lam_delta * adapter.delta_reg(z_dev)
                    if lam_ortho > 0:
                        loss = loss + lam_ortho * adapter.ortho_reg(z_dev)

                # Anchor loss (sign-invariant)
                if (s_anchor is not None) and (lam_anchor > 0):
                    z_mu = z_lat.mean(dim=0, keepdim=True)
                    z_sd = z_lat.std(dim=0, keepdim=True) + 1e-8
                    a_mu = s_anchor.mean(dim=0, keepdim=True)
                    a_sd = s_anchor.std(dim=0, keepdim=True) + 1e-8
                    z_std = (z_lat - z_mu) / z_sd
                    a_std = (s_anchor - a_mu) / a_sd
                    loss_pos = (z_std - a_std) ** 2
                    loss_neg = (z_std + a_std) ** 2
                    anchor_loss = torch.minimum(loss_pos, loss_neg).mean()
                    loss = loss + lam_anchor * anchor_loss
                    train_anchor += float(anchor_loss.item())
                    n_anchor_batches += 1

                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())

            train_loss /= max(len(data_loader), 1)
            loss_hist.append(train_loss)

            if n_anchor_batches > 0:
                anchor_hist.append(train_anchor / float(n_anchor_batches))
                print(f"[seed={seed}] Epoch {epoch}/{int(config.epochs)} loss={train_loss:.6f} anchor={anchor_hist[-1]:.6f}")
            else:
                print(f"[seed={seed}] Epoch {epoch}/{int(config.epochs)} loss={train_loss:.6f}")

        # ----------------------------
        # Evaluate
        # ----------------------------
        model.eval()
        with torch.no_grad():
            X_full = dset.x.to(config.device)
            U_full = dset.u.to(config.device)
            z_full = dset.z.to(config.device).long()
            if adapter is not None:
                X_full = adapter(X_full, z_full)
            _, _, _, s_full, _ = model(X_full, U_full)
            s_np = s_full.cpu().numpy().astype(np.float32)

        # Trial-level aggregation
        feat_trial, y_trial, z_trial = _trial_aggregate(
            features=s_np,
            y=dset.y.cpu().numpy(),
            z=dset.z.cpu().numpy(),
            trial_id=dset.trial_id.cpu().numpy(),
        )

        # Leakage probe (trial-level)
        leak = session_leakage_probe(feat_trial, z_trial, test_size=0.3, random_state=seed)

        # Direction decoding (cross-session)
        decode = None
        if bool(getattr(config, "decode_direction", True)):
            decode = _decode_direction_cross_session(
                feat_trial,
                y_trial,
                z_trial,
                max_iter=int(getattr(config, "decode_max_iter", 2000)),
            )

        # Identity consistency relative to anchor (if anchor exists)
        identity = None
        anchor_match = None
        if dset.s_anchor is not None:
            S_anchor_np = dset.s_anchor.numpy().astype(np.float32)
            z_np = dset.z.numpy().astype(int)
            n_domains = int(z_np.max() + 1)

            aligns = compute_domain_alignment(
                S_true=S_anchor_np,
                S_est=s_np,
                z=z_np,
                n_domains=n_domains,
                min_samples=int(getattr(config, "diag_min_samples", 2000)),
            )
            identity = identity_consistency_vs_ref(aligns, ref_domain=0)

            # Also compute per-domain anchor-match score (mean abs corr after Hungarian)
            per_dom = {}
            for dom in range(n_domains):
                m = z_np == dom
                if m.sum() < 100:
                    continue
                corr = pearson_corr_matrix(S_anchor_np[m], s_np[m])
                perm, _, corr_reord = hungarian_match_abs(corr)
                # corr_reord is |corr| in matched order
                per_dom[int(dom)] = float(np.mean(np.diag(corr_reord)))
            if per_dom:
                anchor_match = {"mean": float(np.mean(list(per_dom.values()))), "per_domain": per_dom}

            # Save one illustrative heatmap (domain 0) if requested
            if bool(getattr(config, "save_heatmaps", False)):
                out_png = os.path.join(args.log, f"corr_anchor_vs_est_seed{seed}.png")
                try:
                    save_corr_heatmap(S_anchor_np, s_np, title=f"anchor vs est (seed={seed})", save_path=out_png)
                except Exception as e:
                    print(f"[EEG] heatmap failed: {e}")

        seed_res = {
            "seed": int(seed),
            "loss_hist": loss_hist,
            "anchor_hist": anchor_hist,
            "session_leakage": leak,
            "decode_direction": decode,
            "identity_vs_anchor": identity,
            "anchor_match": anchor_match,
        }
        results["seeds"].append(seed_res)

        print(f"[EEG] seed={seed} leakage_acc={leak.get('acc', None)} decode_mean={None if decode is None else decode.get('mean', None)}")

    # Save summary json
    os.makedirs(args.log, exist_ok=True)
    out_json = os.path.join(args.log, "eeg_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[EEG] Saved summary: {out_json}")
    print(f"[EEG] total runtime: {time.time() - st:.1f}s")

    return results
