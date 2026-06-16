from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import mne
import numpy as np
import pandas as pd

from .utils import ensure_dir


def _epochs_get_data_compat(epochs: mne.BaseEpochs) -> np.ndarray:
    try:
        return epochs.get_data(copy=True)
    except TypeError:
        return epochs.get_data()


def save_epochs_fif(
    x: np.ndarray,
    info_template,
    times: np.ndarray,
    ch_names: List[str],
    metadata: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = info_template.copy()
    if list(info["ch_names"]) != list(ch_names):
        info = mne.create_info(ch_names=ch_names, sfreq=float(info["sfreq"]), ch_types="eeg")
    events = np.column_stack(
        [
            np.arange(len(x), dtype=np.int64),
            np.zeros(len(x), dtype=np.int64),
            metadata["class_id"].to_numpy(dtype=np.int64) + 1,
        ]
    )
    epochs = mne.EpochsArray(
        np.asarray(x, dtype=np.float32),
        info=info,
        events=events,
        tmin=float(times[0]) if len(times) else 0.0,
        metadata=metadata.reset_index(drop=True).copy(),
        verbose="ERROR",
    )
    epochs.save(str(path), overwrite=True, verbose="ERROR")
    return path


def save_latent_npy(latent: np.ndarray, metadata: pd.DataFrame, out_prefix: str | Path) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_prefix) + "_latent.npy", np.asarray(latent, dtype=np.float32))
    metadata.reset_index(drop=True).to_csv(str(out_prefix) + "_manifest.csv", index=False)
