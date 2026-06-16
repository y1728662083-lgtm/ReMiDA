from __future__ import annotations

from typing import Optional

import numpy as np


def latent_to_pseudo_timeseries(latent: np.ndarray, samples: int) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim == 3:
        latent = latent.reshape(latent.shape[0], -1)
    return np.repeat(latent[:, :, None], repeats=int(samples), axis=2).astype(np.float32)


def concat_representations(raw: Optional[np.ndarray], linear_latent: Optional[np.ndarray], nonlinear_latent: Optional[np.ndarray], samples: int) -> np.ndarray:
    pieces = []
    if raw is not None:
        pieces.append(np.asarray(raw, dtype=np.float32))
    if linear_latent is not None:
        pieces.append(latent_to_pseudo_timeseries(linear_latent, samples))
    if nonlinear_latent is not None:
        pieces.append(latent_to_pseudo_timeseries(nonlinear_latent, samples))
    if not pieces:
        raise ValueError("No representations provided for concatenation.")
    return np.concatenate(pieces, axis=1).astype(np.float32)
