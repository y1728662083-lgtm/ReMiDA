"""Utilities for using real EEG datasets stored as .npz.

This repo's original runners were built for synthetic datasets where ground-truth sources S are
available. For real EEG, we typically have:
  - X: (N, D) observations
  - U: (N, K) auxiliary labels (e.g., direction class one-hot)
  - y: (N,) direction class id (0..K-1)
  - z: (N,) domain/session id (0..n_domains-1)
  - trial_id: (N,) trial index for aggregating timepoints into trials

We store these arrays in a single .npz produced by scripts/prepare_ds003626_npz.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EEGNPZMeta:
    """Lightweight metadata container."""

    domain_map: Optional[Any] = None  # can be list of dicts or strings
    info: Optional[Dict[str, Any]] = None


class EEGNPZDataset(Dataset):
    """A torch Dataset wrapping an EEG .npz file.

    Expected keys in the .npz:
      - X: float32/float64, shape (N, D)
      - U: float32, shape (N, K)
      - y: int, shape (N,)
      - z: int, shape (N,)
      - trial_id: int, shape (N,)

    Optional keys:
      - S_anchor: float32, shape (N, d_latent)  (ICA anchor aligned across domains)
      - meta_json: str/bytes  (json for domain map etc.)
    """

    def __init__(self, npz_path: str | Path, device: torch.device | str = "cpu"):
        self.npz_path = str(npz_path)
        arr = np.load(self.npz_path, allow_pickle=True)

        def _req(key: str):
            if key not in arr:
                raise KeyError(f"Missing key '{key}' in {self.npz_path}. Keys={list(arr.keys())}")
            return arr[key]

        X = _req("X")
        U = _req("U")
        y = _req("y")
        z = _req("z")
        trial_id = _req("trial_id")

        self.x = torch.from_numpy(np.asarray(X)).float().to(device)
        self.u = torch.from_numpy(np.asarray(U)).float().to(device)
        self.y = torch.from_numpy(np.asarray(y).astype(np.int64)).long().to(device)
        self.z = torch.from_numpy(np.asarray(z).astype(np.int64)).long().to(device)
        self.trial_id = torch.from_numpy(np.asarray(trial_id).astype(np.int64)).long().to(device)

        self.s_anchor: Optional[torch.Tensor] = None
        if "S_anchor" in arr:
            self.s_anchor = torch.from_numpy(np.asarray(arr["S_anchor"])).float().to(device)

        self.meta = EEGNPZMeta()
        if "meta_json" in arr:
            try:
                import json

                raw = arr["meta_json"].item() if arr["meta_json"].shape == () else arr["meta_json"]
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                meta = json.loads(str(raw))
                self.meta.domain_map = meta.get("domain_map")
                self.meta.info = meta
            except Exception:
                # keep meta empty if parsing fails
                pass

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def data_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def aux_dim(self) -> int:
        return int(self.u.shape[1])

    @property
    def n_domains(self) -> int:
        return int(self.z.max().item() + 1)

    def __getitem__(self, idx: int):
        if self.s_anchor is None:
            return self.x[idx], self.u[idx], self.y[idx], self.z[idx], self.trial_id[idx]
        return self.x[idx], self.u[idx], self.y[idx], self.z[idx], self.trial_id[idx], self.s_anchor[idx]
