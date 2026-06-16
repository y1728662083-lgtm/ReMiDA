from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .backbones import make_decoder, make_encoder
from .config import ExperimentConfig
from .front_linear import HierarchicalLinearFrontEnd
from .losses import grad_reverse


class LinearJointSIDNet(nn.Module):
    def __init__(self, cfg: ExperimentConfig, n_channels: int, n_times: int, subjects: Dict[str, int], domains: Dict[str, int], domain_to_subject: Dict[int, int], n_classes: int) -> None:
        super().__init__()
        lm = cfg.linear_model
        self.front = HierarchicalLinearFrontEnd(
            subjects=subjects,
            domains=domains,
            domain_to_subject=domain_to_subject,
            n_channels=n_channels,
            reference_subject=cfg.dataset.reference_subject,
            use_bias=cfg.front.use_bias,
            freeze_reference=cfg.front.freeze_reference,
            use_domain_delta=cfg.front.use_domain_delta,
        )
        self.encoder = make_encoder(
            lm.backbone_encoder,
            in_channels=n_channels,
            out_dim=lm.encoder_hidden,
            hidden_dim=lm.encoder_hidden,
            temporal_width=lm.temporal_width,
            temporal_layers=lm.temporal_layers,
            lstm_hidden=lm.lstm_hidden,
            lstm_layers=lm.lstm_layers,
            mamba_dim=lm.mamba_dim,
            dropout=lm.dropout,
        )
        self.mu_head = nn.Sequential(
            nn.Linear(lm.encoder_hidden, lm.reparam_hidden),
            nn.GELU(),
            nn.Dropout(lm.dropout),
            nn.Linear(lm.reparam_hidden, lm.latent_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(lm.encoder_hidden, lm.reparam_hidden),
            nn.GELU(),
            nn.Dropout(lm.dropout),
            nn.Linear(lm.reparam_hidden, lm.latent_dim),
        )
        self.decoder = make_decoder(
            lm.backbone_decoder,
            latent_dim=lm.latent_dim,
            out_channels=n_channels,
            out_times=n_times,
            hidden_dim=lm.decoder_hidden,
            temporal_width=lm.temporal_width,
            temporal_layers=lm.temporal_layers,
            lstm_hidden=lm.lstm_hidden,
            lstm_layers=lm.lstm_layers,
            mamba_dim=lm.mamba_dim,
            dropout=lm.dropout,
        )
        self.class_head = nn.Sequential(
            nn.Linear(lm.latent_dim, lm.reparam_hidden),
            nn.GELU(),
            nn.Dropout(lm.dropout),
            nn.Linear(lm.reparam_hidden, n_classes),
        )
        self.subject_head = nn.Sequential(
            nn.Linear(lm.latent_dim, lm.reparam_hidden),
            nn.GELU(),
            nn.Dropout(lm.dropout),
            nn.Linear(lm.reparam_hidden, len(subjects)),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        x_aligned = self.front(x, domain_ids)
        h = self.encoder(x_aligned)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        class_logits = self.class_head(z)
        subject_logits = self.subject_head(grad_reverse(z, grl_lambda))
        return {
            "x_aligned": x_aligned,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "x_recon": x_recon,
            "class_logits": class_logits,
            "subject_logits": subject_logits,
        }

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        x_aligned = self.front(x, domain_ids)
        h = self.encoder(x_aligned)
        return self.mu_head(h)

    @torch.no_grad()
    def aligned_only(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.front(x, domain_ids)
