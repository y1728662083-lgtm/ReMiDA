from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambd)


def smooth_l1_recon(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(x_hat, x)


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)


def bandpower_torch(x: torch.Tensor, sfreq: float, bands: Dict[str, tuple[float, float]] | None = None) -> torch.Tensor:
    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
        }
    x = x - x.mean(dim=-1, keepdim=True)
    spec = torch.fft.rfft(x, dim=-1)
    psd = (spec.real ** 2 + spec.imag ** 2) / max(x.shape[-1], 1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(sfreq)).to(x.device)
    feats = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        if torch.any(mask):
            feats.append(psd[..., mask].mean(dim=-1))
        else:
            feats.append(torch.zeros_like(psd[..., 0]))
    return torch.cat(feats, dim=-1)


def band_alignment_loss(x_a: torch.Tensor, x_b: torch.Tensor, sfreq: float) -> torch.Tensor:
    fa = bandpower_torch(x_a, sfreq)
    fb = bandpower_torch(x_b, sfreq)
    return ((fa.mean(dim=0) - fb.mean(dim=0)) ** 2).mean()


def class_center_loss(z: torch.Tensor, y: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2:
        z = z.reshape(z.shape[0], -1)
    losses = []
    classes = torch.unique(y)
    for cls in classes:
        mask_cls = y == cls
        if mask_cls.sum() < 2:
            continue
        zc = z[mask_cls]
        sc = subject_ids[mask_cls]
        subjects = torch.unique(sc)
        if len(subjects) < 2:
            continue
        subject_means = []
        for s in subjects:
            mask_sub = sc == s
            if mask_sub.sum() > 0:
                subject_means.append(zc[mask_sub].mean(dim=0, keepdim=True))
        if len(subject_means) < 2:
            continue
        stack = torch.cat(subject_means, dim=0)
        center = stack.mean(dim=0, keepdim=True)
        losses.append(((stack - center) ** 2).mean())
    if not losses:
        return torch.zeros((), device=z.device)
    return torch.stack(losses).mean()


def psd_jsd_np(x_a: np.ndarray, x_b: np.ndarray, bins: int = 64) -> float:
    xa = np.asarray(x_a, dtype=np.float64).reshape(-1)
    xb = np.asarray(x_b, dtype=np.float64).reshape(-1)
    lo = min(float(xa.min()), float(xb.min()))
    hi = max(float(xa.max()), float(xb.max()))
    if hi <= lo:
        return 0.0
    pa, _ = np.histogram(xa, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(xb, bins=bins, range=(lo, hi), density=True)
    pa = pa / max(pa.sum(), 1e-12)
    pb = pb / max(pb.sum(), 1e-12)
    m = 0.5 * (pa + pb)
    def _kl(p, q):
        mask = (p > 0) & (q > 0)
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    return float(0.5 * _kl(pa, m) + 0.5 * _kl(pb, m))
