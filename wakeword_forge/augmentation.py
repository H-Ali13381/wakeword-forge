"""
augmentation.py — Waveform augmentation utilities.

Standard waveform augmentation utilities for local wake-word training.
Background noise and room impulse-response files are optional; the augmentor
degrades gracefully when they are absent.

Augmentations applied per sample (randomly sampled subset each time):
  - Additive Gaussian noise
  - Speed perturbation  (resample trick)
  - Pitch shifting      (torchaudio)
  - Time shift
  - Amplitude scaling
  - Room impulse response convolution (if IR files available)
  - Background noise mixing (if noise files available)
"""

from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from .config import SAMPLE_RATE, MAX_SAMPLES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_wav(path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a wav/flac/ogg file and resample to target_sr. Returns (1, T) float32 tensor.

    Uses soundfile instead of torchaudio.load to avoid the torchcodec dependency
    introduced in recent torchaudio versions.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    # data shape: (T,) mono or (T, C) multichannel
    if data.ndim == 2:
        data = data.mean(axis=1)           # mix down to mono
    wav = torch.from_numpy(data).unsqueeze(0)   # (1, T)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _pad_or_trim(wav: torch.Tensor, max_samples: int = MAX_SAMPLES) -> torch.Tensor:
    if wav.shape[-1] > max_samples:
        return wav[..., :max_samples]
    if wav.shape[-1] < max_samples:
        pad = max_samples - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, pad))
    return wav


# ── Individual augments ───────────────────────────────────────────────────────

def add_gaussian_noise(wav: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
    if snr_db is None:
        snr_db = random.uniform(10, 40)
    signal_power = wav.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


def speed_perturb(wav: torch.Tensor, sr: int = SAMPLE_RATE, factor: float | None = None) -> torch.Tensor:
    if factor is None:
        factor = random.uniform(0.85, 1.15)
    new_sr = int(sr * factor)
    return torchaudio.functional.resample(wav, new_sr, sr)


def time_shift(wav: torch.Tensor, max_shift: float = 0.2) -> torch.Tensor:
    shift = int(random.uniform(-max_shift, max_shift) * wav.shape[-1])
    return torch.roll(wav, shift, dims=-1)


def amplitude_scale(wav: torch.Tensor) -> torch.Tensor:
    factor = random.uniform(0.5, 1.5)
    return wav * factor


def pitch_shift(wav: torch.Tensor, sr: int = SAMPLE_RATE, n_steps: float | None = None) -> torch.Tensor:
    if n_steps is None:
        n_steps = random.uniform(-3, 3)
    effects = [["pitch", str(n_steps * 100)], ["rate", str(sr)]]
    out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
    return out


def mix_noise(wav: torch.Tensor, noise: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
    """Mix a background noise clip into wav at a given SNR."""
    if snr_db is None:
        snr_db = random.uniform(5, 25)
    # loop noise to match wav length
    n = wav.shape[-1]
    if noise.shape[-1] < n:
        reps = (n // noise.shape[-1]) + 1
        noise = noise.repeat(1, reps)
    offset = random.randint(0, max(0, noise.shape[-1] - n))
    noise = noise[..., offset: offset + n]
    sig_rms = wav.pow(2).mean().sqrt()
    noise_rms = noise.pow(2).mean().sqrt() + 1e-9
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    return wav + noise * (target_noise_rms / noise_rms)


def apply_ir(wav: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    """Convolve wav with an impulse response (simple FFT convolution)."""
    n = wav.shape[-1]
    ir = ir[..., : min(ir.shape[-1], n)]
    padded = torch.nn.functional.pad(wav, (0, ir.shape[-1] - 1))
    out = torch.nn.functional.conv1d(
        padded.unsqueeze(0), ir.flip(-1).unsqueeze(0).unsqueeze(0)
    ).squeeze(0)
    return out[..., :n]


# ── CascadingAugmentor ────────────────────────────────────────────────────────

class Augmentor:
    """
    Applies a random subset of augmentations to a waveform tensor.

    Args:
        noise_dir:  path to a directory of background noise .wav files (optional)
        ir_dir:     path to a directory of impulse response .wav files (optional)
        max_chain:  max number of augmentations to chain per sample
        p:          probability of applying any single augmentation
    """

    BUILTIN_AUGMENTS = ["gaussian", "speed", "pitch", "time_shift", "amplitude"]

    def __init__(
        self,
        noise_dir: Path | None = None,
        ir_dir: Path | None = None,
        max_chain: int = 4,
        p: float = 0.5,
    ) -> None:
        self.max_chain = max_chain
        self.p = p

        self._noise_files: list[Path] = []
        if noise_dir and noise_dir.exists():
            self._noise_files = list(noise_dir.rglob("*.wav"))

        self._ir_files: list[Path] = []
        if ir_dir and ir_dir.exists():
            self._ir_files = list(ir_dir.rglob("*.wav"))

    def __call__(self, wav: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
        augments = random.sample(self.BUILTIN_AUGMENTS, k=random.randint(1, self.max_chain))

        for aug in augments:
            if random.random() > self.p:
                continue
            if aug == "gaussian":
                wav = add_gaussian_noise(wav)
            elif aug == "speed":
                wav = speed_perturb(wav, sr)
            elif aug == "pitch":
                try:
                    wav = pitch_shift(wav, sr)
                except Exception:
                    pass  # sox not available
            elif aug == "time_shift":
                wav = time_shift(wav)
            elif aug == "amplitude":
                wav = amplitude_scale(wav)

        if self._noise_files and random.random() < 0.5:
            noise_path = random.choice(self._noise_files)
            try:
                noise = _load_wav(noise_path, sr)
                wav = mix_noise(wav, noise)
            except Exception:
                pass

        if self._ir_files and random.random() < 0.3:
            ir_path = random.choice(self._ir_files)
            try:
                ir = _load_wav(ir_path, sr)
                wav = apply_ir(wav, ir)
            except Exception:
                pass

        # Normalize and clip
        peak = wav.abs().max()
        if peak > 1e-6:
            wav = wav / peak * 0.95

        return _pad_or_trim(wav)
