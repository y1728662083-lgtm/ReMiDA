"""Domain-specific linear drift adapter.

Goal:
  - Model session/domain-specific *linear* drift in the observation space.
  - Provide a learnable demixing/alignment matrix W_z per domain z.
  - Optionally parameterize as W_z = W_shared + Δ_z, with regularization on Δ_z.
  - This module implements the first-order non-fixed mixing component used by the synthetic benchmark models.

Typical usage in runners:
  - Build adapter with per-domain mean/std buffers (for standardizing X per domain).
  - Optionally initialize W_z from ICA anchors (least squares on (X_std -> S_anchor)).
  - Apply adapter(x, z_batch) before feeding x into iVAE.
  - Add regularizers: lambda_delta * adapter.delta_reg(z_batch)
                    + lambda_ortho * adapter.ortho_reg(z_batch)
"""

from __future__ import annotations

import torch
from torch import nn


class DomainLinearAdapter(nn.Module):
    """Per-domain linear adapter: y = W_z * ((x - mu_z)/std_z).

    Args:
        n_domains: number of domains/sessions.
        d: feature dimension (must match x dim).
        mu: (n_domains, d) per-domain mean of x (non-trainable buffer).
        std: (n_domains, d) per-domain std of x (non-trainable buffer).
        W_init: optional (n_domains, d, d) initialization for W_z (torch tensor).
        use_shared: if True, parameterize W_z = W_shared + delta_z.
    """

    def __init__(
        self,
        n_domains: int,
        d: int,
        mu: torch.Tensor,
        std: torch.Tensor,
        W_init: torch.Tensor | None = None,
        use_shared: bool = True,
    ) -> None:
        super().__init__()
        assert mu.shape == (n_domains, d), f"mu must be (n_domains,d), got {tuple(mu.shape)}"
        assert std.shape == (n_domains, d), f"std must be (n_domains,d), got {tuple(std.shape)}"
        self.n_domains = int(n_domains)
        self.d = int(d)

        self.register_buffer("mu", mu.float())
        self.register_buffer("std", std.float())

        if use_shared:
            if W_init is None:
                W_shared = torch.eye(d, dtype=torch.float32)
                delta = torch.zeros(n_domains, d, d, dtype=torch.float32)
            else:
                assert W_init.shape == (n_domains, d, d), f"W_init must be (n_domains,d,d), got {tuple(W_init.shape)}"
                W_shared = W_init.mean(dim=0)
                delta = W_init - W_shared.unsqueeze(0)
            self.W_shared = nn.Parameter(W_shared)
            self.delta = nn.Parameter(delta)
        else:
            if W_init is None:
                W = torch.eye(d, dtype=torch.float32).unsqueeze(0).repeat(n_domains, 1, 1)
            else:
                assert W_init.shape == (n_domains, d, d), f"W_init must be (n_domains,d,d), got {tuple(W_init.shape)}"
                W = W_init
            self.W = nn.Parameter(W)

    def _W_all(self) -> torch.Tensor:
        if hasattr(self, "W"):
            return self.W
        return self.W_shared.unsqueeze(0) + self.delta

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply per-sample domain matrix.

        x: (B, d)
        z: (B,) int64 domain ids
        """
        z = z.long()
        mu = self.mu[z]
        std = self.std[z]
        x_std = (x - mu) / (std + 1e-8)
        W = self._W_all()[z]  # (B,d,d)
        y = torch.einsum("bij,bj->bi", W, x_std)
        return y

    def delta_reg(self, z: torch.Tensor | None = None) -> torch.Tensor:
        """L2 regularizer on Δ_z (only meaningful if use_shared=True)."""
        device = self._W_all().device
        if not hasattr(self, "delta"):
            return torch.tensor(0.0, device=device)
        if z is None:
            return (self.delta ** 2).mean()
        z = z.long()
        return (self.delta[z] ** 2).mean()

    def ortho_reg(self, z: torch.Tensor | None = None) -> torch.Tensor:
        """Orthogonality regularizer: ||W W^T - I||_F^2.

        Helpful to prevent degenerate scaling/conditioning early on.
        """
        W = self._W_all()
        if z is not None:
            z = z.long()
            W = W[z]
        I = torch.eye(self.d, device=W.device)
        WWt = torch.einsum("bij,bkj->bik", W, W)
        return ((WWt - I) ** 2).mean()
