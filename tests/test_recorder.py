"""tests/test_recorder.py — recording cleanup helpers."""

import numpy as np
import pytest

from wakeword_forge.config import SAMPLE_RATE
from wakeword_forge.recorder import _prepare_recorded_take


def test_prepare_recorded_take_trims_edge_silence():
    speech = np.full(SAMPLE_RATE // 2, 0.2, dtype=np.float32)
    raw = np.concatenate(
        [
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            speech,
            np.zeros(SAMPLE_RATE, dtype=np.float32),
        ]
    )

    prepared = _prepare_recorded_take(raw, SAMPLE_RATE)

    assert speech.shape[0] <= prepared.shape[0] < SAMPLE_RATE


def test_prepare_recorded_take_rejects_silence():
    with pytest.raises(ValueError, match="Mic too quiet"):
        _prepare_recorded_take(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
