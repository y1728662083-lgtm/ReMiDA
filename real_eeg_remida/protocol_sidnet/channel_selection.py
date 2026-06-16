from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .constants import AUX_CHANNEL_MARKERS, DEFAULT_INNER_SPEECH_CHANNELS


def _normalize(name: str) -> str:
    return name.strip().upper().replace(" ", "")


def select_task_related_channels(
    ch_names: Sequence[str],
    requested: Iterable[str] | None = None,
    min_required_channels: int = 12,
) -> Tuple[List[int], List[str], str]:
    requested = list(requested) if requested else list(DEFAULT_INNER_SPEECH_CHANNELS)
    requested_norm = {_normalize(x): x for x in requested}
    picked_idx: List[int] = []
    picked_names: List[str] = []
    for i, name in enumerate(ch_names):
        norm = _normalize(name)
        if norm in requested_norm:
            picked_idx.append(i)
            picked_names.append(name)
    if len(picked_idx) >= min_required_channels:
        return picked_idx, picked_names, "task_related"

    fallback_idx: List[int] = []
    fallback_names: List[str] = []
    for i, name in enumerate(ch_names):
        up = name.upper()
        if any(marker in up for marker in AUX_CHANNEL_MARKERS):
            continue
        fallback_idx.append(i)
        fallback_names.append(name)
    return fallback_idx, fallback_names, "fallback_all_eeg"
