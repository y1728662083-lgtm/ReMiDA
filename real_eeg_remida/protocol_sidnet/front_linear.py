from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn

from .utils import flatten_batch_time, flatten_torch_batch_time



def _matrix_sqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, eps, None)
    return (v * np.sqrt(w)[None, :]) @ v.T



def _matrix_inv_sqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, eps, None)
    return (v * (1.0 / np.sqrt(w))[None, :]) @ v.T



def covariance_match_initializer_by_domain(
    x_std: np.ndarray,
    domain_ids: np.ndarray,
    reference_indices: np.ndarray,
    domain_keys: Dict[int, str],
    eps: float = 1e-5,
) -> Dict[str, np.ndarray]:
    ref_flat = flatten_batch_time(x_std[np.asarray(reference_indices, dtype=np.int64)].astype(np.float64))
    ref_cov = np.cov(ref_flat, bias=False) + eps * np.eye(ref_flat.shape[0], dtype=np.float64)
    ref_sqrt = _matrix_sqrt(ref_cov, eps=eps)

    out: Dict[str, np.ndarray] = {}
    domain_ids = np.asarray(domain_ids, dtype=np.int64)
    for did, key in domain_keys.items():
        mask = domain_ids == int(did)
        if not np.any(mask):
            continue
        flat = flatten_batch_time(x_std[mask].astype(np.float64))
        cov = np.cov(flat, bias=False) + eps * np.eye(flat.shape[0], dtype=np.float64)
        out[key] = (ref_sqrt @ _matrix_inv_sqrt(cov, eps=eps)).astype(np.float32)
    return out


class HierarchicalLinearFrontEnd(nn.Module):
    def __init__(
        self,
        subjects: Dict[str, int],
        domains: Dict[str, int],
        domain_to_subject: Dict[int, int],
        n_channels: int,
        reference_subject: str,
        use_bias: bool = False,
        freeze_reference: bool = True,
        use_domain_delta: bool = True,
    ) -> None:
        super().__init__()
        self.subjects = dict(subjects)
        self.domains = dict(domains)
        self.domain_to_subject = dict(domain_to_subject)
        self.n_subjects = len(self.subjects)
        self.n_domains = len(self.domains)
        self.n_channels = int(n_channels)
        self.reference_subject = reference_subject
        self.reference_subject_id = self.subjects[reference_subject]
        self.freeze_reference = bool(freeze_reference)
        self.use_domain_delta = bool(use_domain_delta)

        weight_subject = torch.eye(n_channels, dtype=torch.float32).unsqueeze(0).repeat(self.n_subjects, 1, 1)
        weight_domain = torch.zeros(self.n_domains, n_channels, n_channels, dtype=torch.float32)
        bias_subject = torch.zeros(self.n_subjects, n_channels, dtype=torch.float32)
        bias_domain = torch.zeros(self.n_domains, n_channels, dtype=torch.float32)

        self.weight_subject = nn.Parameter(weight_subject)
        self.weight_domain = nn.Parameter(weight_domain) if self.use_domain_delta else None
        self.bias_subject = nn.Parameter(bias_subject, requires_grad=use_bias)
        self.bias_domain = nn.Parameter(bias_domain, requires_grad=use_bias and self.use_domain_delta)

        self.register_buffer("weight_subject_init", weight_subject.clone())
        self.register_buffer("weight_domain_init", weight_domain.clone())
        self.register_buffer("bias_subject_init", bias_subject.clone())
        self.register_buffer("bias_domain_init", bias_domain.clone())
        self.register_buffer(
            "domain_to_subject_tensor",
            torch.tensor([self.domain_to_subject[i] for i in range(self.n_domains)], dtype=torch.long),
        )
        ref_mask = (self.domain_to_subject_tensor == self.reference_subject_id).float()
        self.register_buffer("reference_domain_mask", ref_mask)

    def initialize(self, init_weights: Dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            device = self.weight_subject.device
            dtype = self.weight_subject.dtype
            per_domain = torch.zeros(
                self.n_domains,
                self.n_channels,
                self.n_channels,
                dtype=dtype,
                device=device,
            )
            eye = torch.eye(self.n_channels, dtype=dtype, device=device)
            domain_to_subject = self.domain_to_subject_tensor.to(device)

            for key, did in self.domains.items():
                if key in init_weights:
                    per_domain[did].copy_(torch.as_tensor(init_weights[key], dtype=dtype, device=device))
                else:
                    per_domain[did].copy_(eye)

            for sid in range(self.n_subjects):
                dom_idx = torch.where(domain_to_subject == sid)[0]
                if sid == self.reference_subject_id:
                    self.weight_subject[sid].copy_(eye)
                else:
                    self.weight_subject[sid].copy_(per_domain.index_select(0, dom_idx).mean(dim=0))
                self.weight_subject_init[sid].copy_(self.weight_subject[sid])

            if self.use_domain_delta and self.weight_domain is not None:
                for did in range(self.n_domains):
                    sid = int(domain_to_subject[did].item())
                    if sid == self.reference_subject_id and self.freeze_reference:
                        self.weight_domain[did].zero_()
                    else:
                        self.weight_domain[did].copy_(per_domain[did] - self.weight_subject[sid])
                self.weight_domain_init.copy_(self.weight_domain.detach())

            self.bias_subject_init.copy_(self.bias_subject.detach())
            if self.bias_domain is not None:
                self.bias_domain_init.copy_(self.bias_domain.detach())

    def effective_weights_all(self) -> torch.Tensor:
        w_sub = self.weight_subject[self.domain_to_subject_tensor]
        if self.use_domain_delta and self.weight_domain is not None:
            w = w_sub + self.weight_domain
        else:
            w = w_sub
        if self.freeze_reference:
            eye = torch.eye(self.n_channels, dtype=w.dtype, device=w.device)
            ref_mask = self.reference_domain_mask.view(-1, 1, 1)
            w = w * (1.0 - ref_mask) + eye.view(1, self.n_channels, self.n_channels) * ref_mask
        return w

    def effective_bias_all(self) -> torch.Tensor:
        b_sub = self.bias_subject[self.domain_to_subject_tensor]
        if self.use_domain_delta and self.bias_domain is not None:
            b = b_sub + self.bias_domain
        else:
            b = b_sub
        if self.freeze_reference:
            ref_mask = self.reference_domain_mask.view(-1, 1)
            b = b * (1.0 - ref_mask)
        return b

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if domain_ids.ndim == 0:
            domain_ids = domain_ids.unsqueeze(0)
        W = self.effective_weights_all()[domain_ids.long()]
        y = torch.einsum("bij,bjt->bit", W, x)
        b = self.effective_bias_all()[domain_ids.long()].unsqueeze(-1)
        return y + b

    def ortho_regularizer(self) -> torch.Tensor:
        W = self.effective_weights_all()
        I = torch.eye(self.n_channels, dtype=W.dtype, device=W.device)
        WWt = torch.einsum("bij,bkj->bik", W, W)
        return ((WWt - I) ** 2).mean()

    def tether_regularizer(self) -> torch.Tensor:
        dWs = self.weight_subject - self.weight_subject_init
        dBs = self.bias_subject - self.bias_subject_init
        if self.freeze_reference:
            mask_subject = torch.ones(self.n_subjects, dtype=dWs.dtype, device=dWs.device)
            mask_subject[self.reference_subject_id] = 0.0
            dWs = dWs * mask_subject.view(-1, 1, 1)
            dBs = dBs * mask_subject.view(-1, 1)
        reg = (dWs ** 2).mean() + (dBs ** 2).mean()
        if self.use_domain_delta and self.weight_domain is not None:
            dWd = self.weight_domain - self.weight_domain_init
            if self.freeze_reference:
                dWd = dWd * (1.0 - self.reference_domain_mask.view(-1, 1, 1))
            reg = reg + (dWd ** 2).mean()
        if self.use_domain_delta and self.bias_domain is not None:
            dBd = self.bias_domain - self.bias_domain_init
            if self.freeze_reference:
                dBd = dBd * (1.0 - self.reference_domain_mask.view(-1, 1))
            reg = reg + (dBd ** 2).mean()
        return reg

    def delta_regularizer(self) -> torch.Tensor:
        if not self.use_domain_delta or self.weight_domain is None:
            return torch.zeros((), device=self.weight_subject.device)
        delta = self.weight_domain
        if self.freeze_reference:
            delta = delta * (1.0 - self.reference_domain_mask.view(-1, 1, 1))
        reg = (delta ** 2).mean()
        if self.bias_domain is not None:
            b = self.bias_domain
            if self.freeze_reference:
                b = b * (1.0 - self.reference_domain_mask.view(-1, 1))
            reg = reg + (b ** 2).mean()
        return reg

    def mean_cov_alignment_loss(self, x_src: torch.Tensor, x_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fs = flatten_torch_batch_time(x_src)
        fr = flatten_torch_batch_time(x_ref)
        mean_gap = ((fs.mean(dim=1) - fr.mean(dim=1)) ** 2).mean()
        fs = fs - fs.mean(dim=1, keepdim=True)
        fr = fr - fr.mean(dim=1, keepdim=True)
        cov_s = fs @ fs.T / max(fs.shape[1] - 1, 1)
        cov_r = fr @ fr.T / max(fr.shape[1] - 1, 1)
        cov_gap = ((cov_s - cov_r) ** 2).mean()
        return mean_gap, cov_gap
