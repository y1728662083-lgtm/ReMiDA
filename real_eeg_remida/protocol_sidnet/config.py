from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    dataset_root: str = ".."
    derivatives_subdir: str = "derivatives"
    subjects: List[str] = field(default_factory=lambda: ["sub-01", "sub-02"])
    sessions: List[str] = field(default_factory=lambda: ["ses-01", "ses-02", "ses-03"])
    condition: str = "inner"
    class_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    class_names: List[str] = field(default_factory=lambda: ["arriba", "abajo", "derecha", "izquierda"])
    use_task_related_channels: bool = False
    keep_channels: List[str] = field(default_factory=list)
    min_required_channels: int = 12
    drop_emg_contaminated: bool = False
    crop_points: Optional[int] = None
    baseline_tmin: Optional[float] = -0.5
    baseline_tmax: Optional[float] = 0.0
    export_fif: bool = True
    export_latents_npy: bool = True


@dataclass
class CrossSubjectConfig:
    enabled: bool = True
    directions: List[str] = field(default_factory=lambda: ["sub-01->sub-02", "sub-02->sub-01"])
    sessions: List[str] = field(default_factory=lambda: ["ses-01", "ses-02", "ses-03"])
    source_val_ratio: float = 0.20
    seed: int = 42


@dataclass
class CrossSessionConfig:
    enabled: bool = True
    subjects: List[str] = field(default_factory=lambda: ["sub-01", "sub-02"])
    target_sessions: List[str] = field(default_factory=lambda: ["ses-01", "ses-02", "ses-03"])
    source_val_ratio: float = 0.20
    seed: int = 42


@dataclass
class FrontConfig:
    use_bias: bool = False
    freeze_reference: bool = True
    use_domain_delta: bool = True
    lambda_front_cov: float = 0.05
    lambda_front_ortho: float = 0.005
    lambda_front_tether: float = 0.01
    lambda_front_delta: float = 0.005


@dataclass
class LinearModelConfig:
    latent_dim: int = 64
    backbone_encoder: str = "cnn"
    backbone_decoder: str = "cnn"
    encoder_hidden: int = 128
    decoder_hidden: int = 128
    temporal_layers: int = 2
    temporal_width: int = 7
    lstm_hidden: int = 128
    lstm_layers: int = 1
    mamba_dim: int = 128
    dropout: float = 0.2
    beta_kl: float = 1e-3
    lambda_recon: float = 1.0
    lambda_class: float = 1.0
    lambda_domain_adv: float = 0.20
    lambda_band_alignment: float = 0.10
    lambda_latent_center: float = 0.10
    reparam_hidden: int = 128


@dataclass
class TrainingConfig:
    epochs: int = 120
    batch_size_source: int = 32
    batch_size_adapt: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 30
    grad_clip: float = 5.0
    num_workers: int = 0
    checkpoint_metric: str = "val_score"
    save_every: int = 10


@dataclass
class EEGNetConfig:
    epochs: int = 120
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    dropout: float = 0.5
    modes: List[str] = field(default_factory=lambda: [
        "raw",
        "aligned_raw",
        "linear_latent",
        "raw_plus_linear_latent",
    ])


@dataclass
class OutputConfig:
    out_dir: str = "./outputs"
    experiment_name: str = "sub01_sub02_cross_protocol_linear"
    save_checkpoints: bool = True


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cross_subject: CrossSubjectConfig = field(default_factory=CrossSubjectConfig)
    cross_session: CrossSessionConfig = field(default_factory=CrossSessionConfig)
    front: FrontConfig = field(default_factory=FrontConfig)
    linear_model: LinearModelConfig = field(default_factory=LinearModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eegnet: EEGNetConfig = field(default_factory=EEGNetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "cuda"
    seed: int = 42

    @property
    def run_dir(self) -> Path:
        return Path(self.output.out_dir) / self.output.experiment_name


def _update_dataclass(dc, values: Dict[str, Any]):
    for key, value in values.items():
        if not hasattr(dc, key):
            continue
        setattr(dc, key, value)
    return dc


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ExperimentConfig()
    if "dataset" in raw:
        cfg.dataset = _update_dataclass(cfg.dataset, raw["dataset"])
    if "cross_subject" in raw:
        cfg.cross_subject = _update_dataclass(cfg.cross_subject, raw["cross_subject"])
    if "cross_session" in raw:
        cfg.cross_session = _update_dataclass(cfg.cross_session, raw["cross_session"])
    if "front" in raw:
        cfg.front = _update_dataclass(cfg.front, raw["front"])
    if "linear_model" in raw:
        cfg.linear_model = _update_dataclass(cfg.linear_model, raw["linear_model"])
    if "training" in raw:
        cfg.training = _update_dataclass(cfg.training, raw["training"])
    if "eegnet" in raw:
        cfg.eegnet = _update_dataclass(cfg.eegnet, raw["eegnet"])
    if "output" in raw:
        cfg.output = _update_dataclass(cfg.output, raw["output"])
    if "device" in raw:
        cfg.device = raw["device"]
    if "seed" in raw:
        cfg.seed = int(raw["seed"])
    if overrides:
        for section, values in overrides.items():
            if hasattr(cfg, section) and isinstance(values, dict):
                _update_dataclass(getattr(cfg, section), values)
            elif hasattr(cfg, section):
                setattr(cfg, section, values)
    return cfg
