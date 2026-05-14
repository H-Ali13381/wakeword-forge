"""Shared audio cleanup helpers for recordings and training samples."""

from __future__ import annotations

import numpy as np

from wakeword_forge.config import SAMPLE_RATE


def trim_silence_edges(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = 0.006,
    padding_seconds: float = 0.05,
) -> np.ndarray:
    """Remove leading/trailing near-silence while keeping a small context pad.

    Returns the original audio when no sample crosses the threshold, so callers can
    still run their own silence/quality checks instead of silently saving an empty clip.
    """

    if audio.size == 0:
        return audio

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    active = np.flatnonzero(np.abs(mono) >= threshold)
    if active.size == 0:
        return audio

    padding = int(sample_rate * padding_seconds)
    start = max(int(active[0]) - padding, 0)
    stop = min(int(active[-1]) + padding + 1, audio.shape[0])
    return audio[start:stop]
