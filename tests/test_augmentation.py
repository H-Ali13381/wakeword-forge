"""
tests/test_augmentation.py — smoke tests for augmentation
"""

import numpy as np
import soundfile as sf
import torch

from wakeword_forge.augmentation import (
    add_gaussian_noise,
    speed_perturb,
    time_shift,
    amplitude_scale,
    Augmentor,
    _load_wav,
)
from wakeword_forge.config import SAMPLE_RATE, MAX_SAMPLES


def make_wav(duration=1.0):
    return torch.randn(1, int(duration * SAMPLE_RATE))


def test_gaussian_noise_shape():
    wav = make_wav()
    out = add_gaussian_noise(wav, snr_db=20)
    assert out.shape == wav.shape


def test_speed_perturb_shape():
    wav = make_wav()
    out = speed_perturb(wav, SAMPLE_RATE, factor=1.0)
    # factor=1.0 should be near-identity
    assert out.shape[-1] > 0


def test_time_shift_shape():
    wav = make_wav()
    out = time_shift(wav, max_shift=0.1)
    assert out.shape == wav.shape


def test_amplitude_scale_noclip():
    wav = torch.zeros(1, 1000)
    out = amplitude_scale(wav)
    assert out.shape == wav.shape


def test_augmentor_output_length():
    aug = Augmentor(max_chain=3, p=0.7)
    wav = make_wav(0.8)
    out = aug(wav)
    # Augmentor pads/trims to MAX_SAMPLES
    assert out.shape[-1] == MAX_SAMPLES


def test_load_wav_can_trim_leading_and_trailing_silence(tmp_path):
    speech = np.full(SAMPLE_RATE // 2, 0.2, dtype=np.float32)
    raw = np.concatenate(
        [
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            speech,
            np.zeros(SAMPLE_RATE, dtype=np.float32),
        ]
    )
    path = tmp_path / "trim_me.wav"
    sf.write(path, raw, SAMPLE_RATE)

    untrimmed = _load_wav(path, trim_silence=False)
    trimmed = _load_wav(path, trim_silence=True)

    assert untrimmed.shape[-1] == raw.shape[0]
    assert speech.shape[0] <= trimmed.shape[-1] < SAMPLE_RATE
