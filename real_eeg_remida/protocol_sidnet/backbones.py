from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def _maybe_import_mamba():
    try:
        from mamba_ssm import Mamba  # type: ignore
        return Mamba
    except Exception:
        return None


class TemporalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        y = self.dropout(y)
        return F.gelu(x + y)


class PseudoMambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, conv_kernel: int = 5) -> None:
        super().__init__()
        pad = conv_kernel // 2
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=pad, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,D]
        residual = x
        y = self.norm(x)
        gate, value = self.in_proj(y).chunk(2, dim=-1)
        value = value.transpose(1, 2)
        value = self.dwconv(value).transpose(1, 2)
        y = torch.sigmoid(gate) * torch.tanh(value)
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class OptionalMambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        Mamba = _maybe_import_mamba()
        if Mamba is not None:
            self.impl = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.impl = PseudoMambaBlock(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, hidden_dim: int, temporal_width: int = 7, temporal_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_channels, hidden_dim, temporal_width, padding=temporal_width // 2)
        self.blocks = nn.ModuleList([TemporalResBlock(hidden_dim, temporal_width, dropout) for _ in range(temporal_layers)])
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.stem(x))
        for blk in self.blocks:
            y = blk(y)
        y = self.pool(y)
        return self.fc(y)


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, out_times: int, hidden_dim: int, temporal_width: int = 7, temporal_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.out_times = int(out_times)
        self.hidden_dim = int(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * out_times),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([TemporalResBlock(hidden_dim, temporal_width, dropout) for _ in range(temporal_layers)])
        self.out = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.fc(z).reshape(z.shape[0], self.hidden_dim, self.out_times)
        for blk in self.blocks:
            y = blk(y)
        return self.out(y)


class CNNLSTMEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, hidden_dim: int, lstm_hidden: int = 128, lstm_layers: int = 1, temporal_width: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, temporal_width, padding=temporal_width // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, temporal_width, padding=temporal_width // 2),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x).transpose(1, 2)
        y, _ = self.lstm(y)
        y = y[:, -1]
        return self.fc(y)


class CNNLSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, out_times: int, hidden_dim: int, lstm_hidden: int = 128, lstm_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.out_times = int(out_times)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_times * hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0, bidirectional=True)
        self.out = nn.Linear(lstm_hidden * 2, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.fc(z).reshape(z.shape[0], self.out_times, -1)
        y, _ = self.lstm(y)
        y = self.out(y).transpose(1, 2)
        return y


class CNNMambaEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, hidden_dim: int, mamba_dim: int = 128, temporal_width: int = 7, temporal_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, temporal_width, padding=temporal_width // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, mamba_dim, temporal_width, padding=temporal_width // 2),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([OptionalMambaLayer(mamba_dim, dropout=dropout) for _ in range(temporal_layers)])
        self.norm = nn.LayerNorm(mamba_dim)
        self.fc = nn.Sequential(
            nn.Linear(mamba_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x).transpose(1, 2)
        for layer in self.layers:
            y = layer(y)
        y = self.norm(y).mean(dim=1)
        return self.fc(y)


class CNNMambaDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, out_times: int, hidden_dim: int, mamba_dim: int = 128, temporal_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.out_times = int(out_times)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_times * mamba_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([OptionalMambaLayer(mamba_dim, dropout=dropout) for _ in range(temporal_layers)])
        self.norm = nn.LayerNorm(mamba_dim)
        self.out = nn.Linear(mamba_dim, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.fc(z).reshape(z.shape[0], self.out_times, -1)
        for layer in self.layers:
            y = layer(y)
        y = self.norm(y)
        return self.out(y).transpose(1, 2)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, out_times: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.out_times = int(out_times)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels * out_times),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.net(z)
        return y.reshape(z.shape[0], self.out_channels, self.out_times)


def make_encoder(name: str, in_channels: int, out_dim: int, hidden_dim: int, temporal_width: int, temporal_layers: int, lstm_hidden: int, lstm_layers: int, mamba_dim: int, dropout: float) -> nn.Module:
    name = name.lower()
    if name == "cnn":
        return CNNEncoder(in_channels, out_dim, hidden_dim, temporal_width, temporal_layers, dropout)
    if name == "cnn_lstm":
        return CNNLSTMEncoder(in_channels, out_dim, hidden_dim, lstm_hidden, lstm_layers, temporal_width, dropout)
    if name == "cnn_mamba":
        return CNNMambaEncoder(in_channels, out_dim, hidden_dim, mamba_dim, temporal_width, temporal_layers, dropout)
    raise ValueError(f"Unknown encoder backbone: {name}")


def make_decoder(name: str, latent_dim: int, out_channels: int, out_times: int, hidden_dim: int, temporal_width: int, temporal_layers: int, lstm_hidden: int, lstm_layers: int, mamba_dim: int, dropout: float) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPDecoder(latent_dim, out_channels, out_times, hidden_dim, dropout)
    if name == "cnn":
        return CNNDecoder(latent_dim, out_channels, out_times, hidden_dim, temporal_width, temporal_layers, dropout)
    if name == "cnn_lstm":
        return CNNLSTMDecoder(latent_dim, out_channels, out_times, hidden_dim, lstm_hidden, lstm_layers, dropout)
    if name == "cnn_mamba":
        return CNNMambaDecoder(latent_dim, out_channels, out_times, hidden_dim, mamba_dim, temporal_layers, dropout)
    raise ValueError(f"Unknown decoder backbone: {name}")
