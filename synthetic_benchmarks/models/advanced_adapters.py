from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _as_long_tensor(x, device=None):
    if torch.is_tensor(x):
        return x.long().to(device=device) if device is not None else x.long()
    return torch.as_tensor(x, dtype=torch.long, device=device)


class HierarchicalSessionLinearAdapter(nn.Module):
    """Subject-aware session linear adapter.

    Each flattened session/domain `z` belongs to one subject `b = subject_of_domain[z]`.
    The adapter is

        y = W_{b,j} ((x - mu_j) / std_j)

    with parameterization

        W_{b,j} = W_shared[b] + Delta[z]

    and optional reference sessions forced to have zero delta.
    """

    def __init__(
        self,
        n_subjects: int,
        n_domains: int,
        d: int,
        mu: torch.Tensor,
        std: torch.Tensor,
        subject_of_domain: torch.Tensor,
        W_init: Optional[torch.Tensor] = None,
        reference_domains: Optional[torch.Tensor] = None,
        use_shared: bool = True,
    ) -> None:
        super().__init__()
        assert mu.shape == (n_domains, d)
        assert std.shape == (n_domains, d)
        assert subject_of_domain.shape == (n_domains,)
        self.n_subjects = int(n_subjects)
        self.n_domains = int(n_domains)
        self.d = int(d)
        self.use_shared = bool(use_shared)

        self.register_buffer("mu", mu.float())
        self.register_buffer("std", std.float())
        self.register_buffer("subject_of_domain", subject_of_domain.long())
        if reference_domains is None:
            reference_mask = torch.zeros(n_domains, dtype=torch.float32)
        else:
            reference_mask = torch.zeros(n_domains, dtype=torch.float32)
            reference_mask[_as_long_tensor(reference_domains)] = 1.0
        self.register_buffer("reference_mask", reference_mask)

        if use_shared:
            if W_init is None:
                W_shared = torch.eye(d, dtype=torch.float32).unsqueeze(0).repeat(n_subjects, 1, 1)
                delta = torch.zeros(n_domains, d, d, dtype=torch.float32)
            else:
                assert W_init.shape == (n_domains, d, d)
                W_shared = torch.zeros(n_subjects, d, d, dtype=torch.float32)
                delta = torch.zeros(n_domains, d, d, dtype=torch.float32)
                for subj in range(n_subjects):
                    dom_idx = torch.where(subject_of_domain.long() == int(subj))[0]
                    if dom_idx.numel() == 0:
                        W_shared[subj] = torch.eye(d, dtype=torch.float32)
                        continue
                    if reference_domains is not None:
                        ref_in_subj = [int(x) for x in _as_long_tensor(reference_domains).tolist() if int(subject_of_domain[int(x)]) == subj]
                    else:
                        ref_in_subj = []
                    if len(ref_in_subj) > 0:
                        W_shared[subj] = W_init[ref_in_subj[0]].float()
                    else:
                        W_shared[subj] = W_init[dom_idx].float().mean(dim=0)
                    delta[dom_idx] = W_init[dom_idx].float() - W_shared[subj].unsqueeze(0)
                if reference_domains is not None:
                    delta[_as_long_tensor(reference_domains)] = 0.0
            self.W_shared = nn.Parameter(W_shared)
            self.delta = nn.Parameter(delta)
        else:
            if W_init is None:
                W = torch.eye(d, dtype=torch.float32).unsqueeze(0).repeat(n_domains, 1, 1)
            else:
                assert W_init.shape == (n_domains, d, d)
                W = W_init.float()
            self.W = nn.Parameter(W)

    def W_all(self) -> torch.Tensor:
        if hasattr(self, "W"):
            return self.W
        shared = self.W_shared[self.subject_of_domain]
        delta = self.delta.clone()
        if self.reference_mask is not None:
            delta = delta * (1.0 - self.reference_mask.view(-1, 1, 1))
        return shared + delta

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        domain_id = domain_id.long()
        mu = self.mu[domain_id]
        std = self.std[domain_id]
        x_std = (x - mu) / (std + 1e-8)
        W = self.W_all()[domain_id]
        return torch.einsum("bij,bj->bi", W, x_std)

    def delta_reg(self, domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not hasattr(self, "delta"):
            return torch.zeros((), device=self.mu.device)
        delta = self.delta
        if self.reference_mask is not None:
            delta = delta * (1.0 - self.reference_mask.view(-1, 1, 1))
        if domain_id is not None:
            domain_id = domain_id.long()
            delta = delta[domain_id]
        return (delta ** 2).mean()

    def ortho_reg(self, domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.W_all()
        if domain_id is not None:
            W = W[domain_id.long()]
        I = torch.eye(self.d, device=W.device)
        WWt = torch.einsum("bij,bkj->bik", W, W)
        return ((WWt - I) ** 2).mean()


class SubjectResidualNonlinearAdapter(nn.Module):
    """Subject-level weak nonlinear residual adapter.

    y = x + beta * P_b phi(Q_b x + b_q_b) + b_p_b

    Reference subject is fixed to identity by forcing residual to zero.
    """

    def __init__(
        self,
        n_subjects: int,
        d: int,
        bottleneck_dim: int = 8,
        activation: str = "xtanh",
        slope: float = 0.1,
        ref_subject: int = 0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.n_subjects = int(n_subjects)
        self.d = int(d)
        self.bottleneck_dim = int(bottleneck_dim)
        self.activation = str(activation).lower()
        self.slope = float(slope)
        self.ref_subject = int(ref_subject)

        q_scale = 0.02
        self.Q = nn.Parameter(torch.randn(n_subjects, bottleneck_dim, d) * q_scale)
        self.bq = nn.Parameter(torch.zeros(n_subjects, bottleneck_dim))
        self.P = nn.Parameter(torch.zeros(n_subjects, d, bottleneck_dim))
        self.bp = nn.Parameter(torch.zeros(n_subjects, d))
        if not zero_init:
            nn.init.normal_(self.P, mean=0.0, std=q_scale)
        self.register_buffer("ref_mask_subject", torch.zeros(n_subjects, dtype=torch.float32))
        self.ref_mask_subject[self.ref_subject] = 1.0

    def _phi(self, h: torch.Tensor) -> torch.Tensor:
        if self.activation == "xtanh":
            return torch.tanh(h) + self.slope * h
        if self.activation == "tanh":
            return torch.tanh(h)
        if self.activation == "lrelu":
            return torch.where(h >= 0, h, self.slope * h)
        if self.activation == "relu":
            return torch.relu(h)
        if self.activation == "none":
            return h
        raise ValueError(f"Unsupported activation={self.activation}")

    def _phi_prime(self, h: torch.Tensor) -> torch.Tensor:
        if self.activation == "xtanh":
            return 1.0 / (torch.cosh(h) ** 2) + self.slope
        if self.activation == "tanh":
            return 1.0 / (torch.cosh(h) ** 2)
        if self.activation == "lrelu":
            return torch.where(h >= 0, torch.ones_like(h), torch.full_like(h, self.slope))
        if self.activation == "relu":
            return (h >= 0).float()
        if self.activation == "none":
            return torch.ones_like(h)
        raise ValueError(f"Unsupported activation={self.activation}")

    def residual(self, x: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        subject_id = subject_id.long()
        Q = self.Q[subject_id]
        bq = self.bq[subject_id]
        P = self.P[subject_id]
        bp = self.bp[subject_id]
        h = torch.einsum("brd,bd->br", Q, x) + bq
        act = self._phi(h)
        res = torch.einsum("bdr,br->bd", P, act) + bp
        ref_mask = (subject_id == self.ref_subject).float().unsqueeze(1)
        res = res * (1.0 - ref_mask)
        return res

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        res = self.residual(x, subject_id)
        return x + float(beta) * res

    def residual_reg(self, x: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        return (self.residual(x, subject_id) ** 2).mean()

    def weight_reg(self) -> torch.Tensor:
        return (self.P ** 2).mean() + (self.Q ** 2).mean() + (self.bp ** 2).mean() + (self.bq ** 2).mean()

    def jacobian_reg(self, x: torch.Tensor, subject_id: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        subject_id = subject_id.long()
        Q = self.Q[subject_id]
        P = self.P[subject_id]
        h = torch.einsum("brd,bd->br", Q, x) + self.bq[subject_id]
        phi_p = self._phi_prime(h)
        J = torch.einsum("bir,br,brj->bij", P, phi_p, Q)
        ref_mask = (subject_id == self.ref_subject).float().view(-1, 1, 1)
        J = J * (1.0 - ref_mask)
        return (float(beta) ** 2) * (J ** 2).mean()


class UConditionalMomentAligner(nn.Module):
    """Same-U_raw cross-subject moment alignment."""

    def __init__(
        self,
        n_subjects: int,
        n_u: int,
        d: int,
        ref_subject: int = 0,
        eta_mean: float = 1.0,
        eta_var: float = 0.3,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_subjects = int(n_subjects)
        self.n_u = int(n_u)
        self.d = int(d)
        self.ref_subject = int(ref_subject)
        self.eta_mean = float(eta_mean)
        self.eta_var = float(eta_var)
        self.eps = float(eps)
        self.register_buffer("mean_table", torch.zeros(n_subjects, n_u, d))
        self.register_buffer("std_table", torch.ones(n_subjects, n_u, d))
        self.register_buffer("count_table", torch.zeros(n_subjects, n_u))
        self.register_buffer("pooled_mean_table", torch.zeros(n_u, d))
        self.register_buffer("pooled_std_table", torch.ones(n_u, d))
        self.register_buffer("fitted", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
        u_raw_id: torch.Tensor,
        min_count: int = 50,
        shrinkage: float = 0.0,
    ) -> None:
        device = self.mean_table.device
        x = x.detach().to(device)
        subject_id = subject_id.long().to(device)
        u_raw_id = u_raw_id.long().to(device)
        n_subjects, n_u, d = self.n_subjects, self.n_u, self.d

        mean_table = torch.zeros(n_subjects, n_u, d, device=device)
        std_table = torch.ones(n_subjects, n_u, d, device=device)
        count_table = torch.zeros(n_subjects, n_u, device=device)

        pooled_mean = torch.zeros(n_u, d, device=device)
        pooled_std = torch.ones(n_u, d, device=device)

        for u in range(n_u):
            mask_u = (u_raw_id == u)
            if mask_u.any():
                xu = x[mask_u]
                pooled_mean[u] = xu.mean(dim=0)
                pooled_std[u] = xu.std(dim=0, unbiased=False) + self.eps
            for subj in range(n_subjects):
                mask = mask_u & (subject_id == subj)
                cnt = int(mask.sum())
                count_table[subj, u] = float(cnt)
                if cnt > 0:
                    xs = x[mask]
                    mu = xs.mean(dim=0)
                    sd = xs.std(dim=0, unbiased=False) + self.eps
                    alpha = float(shrinkage)
                    if cnt < max(1, int(min_count)):
                        alpha = max(alpha, 1.0 - float(cnt) / float(max(1, min_count)))
                    mean_table[subj, u] = (1.0 - alpha) * mu + alpha * pooled_mean[u]
                    std_table[subj, u] = (1.0 - alpha) * sd + alpha * pooled_std[u]
                else:
                    mean_table[subj, u] = pooled_mean[u]
                    std_table[subj, u] = pooled_std[u]

        self.mean_table.copy_(mean_table)
        self.std_table.copy_(std_table)
        self.count_table.copy_(count_table)
        self.pooled_mean_table.copy_(pooled_mean)
        self.pooled_std_table.copy_(pooled_std)
        self.fitted.fill_(1)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor, u_raw_id: torch.Tensor) -> torch.Tensor:
        if int(self.fitted.item()) == 0:
            return x
        subject_id = subject_id.long()
        u_raw_id = u_raw_id.long()
        mu = self.mean_table[subject_id, u_raw_id]
        sd = self.std_table[subject_id, u_raw_id]
        ref_idx = torch.full_like(subject_id, fill_value=self.ref_subject)
        mu_ref = self.mean_table[ref_idx, u_raw_id]
        sd_ref = self.std_table[ref_idx, u_raw_id]

        mean_tgt = (1.0 - self.eta_mean) * mu + self.eta_mean * mu_ref
        scale = (1.0 - self.eta_var) + self.eta_var * (sd_ref / (sd + self.eps))
        y = mean_tgt + scale * (x - mu)
        ref_mask = (subject_id == self.ref_subject).float().unsqueeze(1)
        return y * (1.0 - ref_mask) + x * ref_mask

    def moment_loss(self, min_count: int = 1) -> torch.Tensor:
        if int(self.fitted.item()) == 0:
            return torch.zeros((), device=self.mean_table.device)
        losses = []
        for subj in range(self.n_subjects):
            if subj == self.ref_subject:
                continue
            valid = (self.count_table[subj] >= float(min_count)) & (self.count_table[self.ref_subject] >= float(min_count))
            if valid.any():
                mu_gap = self.mean_table[subj, valid] - self.mean_table[self.ref_subject, valid]
                sd_gap = torch.log(self.std_table[subj, valid] ** 2 + self.eps) - torch.log(self.std_table[self.ref_subject, valid] ** 2 + self.eps)
                losses.append((mu_gap ** 2).mean() + (sd_gap ** 2).mean())
        if len(losses) == 0:
            return torch.zeros((), device=self.mean_table.device)
        return torch.stack(losses).mean()
