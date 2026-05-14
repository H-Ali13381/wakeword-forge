"""
tests/test_augmentation.py — smoke tests for augmentation
"""

import torch

from wakeword_forge.augmentation import (
    add_gaussian_noise, speed_perturb, time_shift, amplitude_scale, Augmentor
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
