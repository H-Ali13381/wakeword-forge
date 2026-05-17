"""
augmentation.py — waveform and spectrogram augmentation utilities.

The training augmentor uses a weighted cascading policy: positives and hard
negatives use a stronger preset, while regular background negatives can use a
lighter preset so already-diverse noise/speech clips are not over-distorted.
Optional local noise and room-response folders are mixed in when configured.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .audio import trim_silence_edges
from .config import MAX_SAMPLES, SAMPLE_RATE

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from .config import ForgeConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_wav(
    path: Path,
    target_sr: int = SAMPLE_RATE,
    *,
    trim_silence: bool = False,
) -> torch.Tensor:
    """Load a wav/flac/ogg file and resample to target_sr. Returns (1, T) float32 tensor.

    Uses soundfile instead of torchaudio.load to avoid the torchcodec dependency
    introduced in recent torchaudio versions.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if trim_silence:
        data = trim_silence_edges(data, sample_rate=sr)
    wav = torch.from_numpy(data).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _pad_or_trim(wav: torch.Tensor, max_samples: int = MAX_SAMPLES) -> torch.Tensor:
    if wav.shape[-1] > max_samples:
        return wav[..., :max_samples]
    if wav.shape[-1] < max_samples:
        pad = max_samples - wav.shape[-1]
        return F.pad(wav, (0, pad))
    return wav


def _normalize_peak(wav: torch.Tensor, target: float = 0.95) -> torch.Tensor:
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak * target
    return wav.clamp(-1.0, 1.0)


# ── Individual waveform augments ──────────────────────────────────────────────


def add_gaussian_noise(wav: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
    if snr_db is None:
        snr_db = random.uniform(10, 40)
    signal_power = wav.pow(2).mean().clamp_min(1e-12)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


def speed_perturb(wav: torch.Tensor, sr: int = SAMPLE_RATE, factor: float | None = None) -> torch.Tensor:
    if factor is None:
        factor = random.uniform(0.85, 1.15)
    new_sr = max(1, int(sr * factor))
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
    n = wav.shape[-1]
    if noise.shape[-1] < n:
        reps = (n // max(1, noise.shape[-1])) + 1
        noise = noise.repeat(1, reps)
    offset = random.randint(0, max(0, noise.shape[-1] - n))
    noise = noise[..., offset : offset + n]
    sig_rms = wav.pow(2).mean().sqrt().clamp_min(1e-9)
    noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-9)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    return wav + noise * (target_noise_rms / noise_rms)


def apply_ir(wav: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    """Convolve wav with an impulse response using grouped 1-D convolution."""
    n = wav.shape[-1]
    ir = ir[..., : min(ir.shape[-1], n)]
    if ir.numel() == 0:
        return wav
    ir = ir / ir.abs().max().clamp_min(1e-6)
    kernel = ir.flip(-1).reshape(1, 1, -1)
    padded = F.pad(wav.reshape(1, 1, -1), (kernel.shape[-1] - 1, 0))
    out = F.conv1d(padded, kernel).reshape(1, -1)
    return out[..., :n]


def _gain_db(wav: torch.Tensor, gain_db: float) -> torch.Tensor:
    return wav * (10 ** (gain_db / 20))


def _gain_transition(wav: torch.Tensor, start_db: float, end_db: float) -> torch.Tensor:
    gains = torch.linspace(
        10 ** (start_db / 20),
        10 ** (end_db / 20),
        wav.shape[-1],
        device=wav.device,
        dtype=wav.dtype,
    )
    return wav * gains.unsqueeze(0)


def _time_mask_waveform(wav: torch.Tensor, max_fraction: float = 0.15) -> torch.Tensor:
    n = wav.shape[-1]
    if n <= 1:
        return wav
    width = random.randint(1, max(1, int(n * max_fraction)))
    start = random.randint(0, max(0, n - width))
    out = wav.clone()
    out[..., start : start + width] = 0.0
    return out


def _polarity_invert(wav: torch.Tensor) -> torch.Tensor:
    return -wav


def _clipping_distortion(wav: torch.Tensor) -> torch.Tensor:
    abs_wav = wav.abs().flatten()
    if abs_wav.numel() == 0:
        return wav
    threshold = torch.quantile(abs_wav, random.uniform(0.90, 0.99)).clamp_min(1e-6)
    return wav.clamp(-float(threshold), float(threshold))


def _safe_filter(wav: torch.Tensor, sr: int, name: str) -> torch.Tensor:
    try:
        if name == "band_pass":
            return torchaudio.functional.bandpass_biquad(
                wav,
                sample_rate=sr,
                central_freq=random.uniform(600, min(4000, sr / 2 - 100)),
                Q=random.uniform(0.5, 1.5),
            )
        if name == "band_stop":
            return torchaudio.functional.bandreject_biquad(
                wav,
                sample_rate=sr,
                central_freq=random.uniform(200, min(4000, sr / 2 - 100)),
                Q=random.uniform(1.0, 5.0),
            )
        if name == "high_pass":
            return torchaudio.functional.highpass_biquad(
                wav,
                sample_rate=sr,
                cutoff_freq=random.uniform(200, min(1000, sr / 2 - 100)),
            )
        if name == "low_pass":
            return torchaudio.functional.lowpass_biquad(
                wav,
                sample_rate=sr,
                cutoff_freq=random.uniform(1500, min(7500, sr / 2 - 100)),
            )
    except Exception:
        return wav
    return wav


@dataclass(frozen=True)
class _TransformSpec:
    name: str
    weight: int
    group: str | None = None


# ── Cascading waveform augmentor ──────────────────────────────────────────────


class CascadingAugmentor:
    """Apply weighted cascades of waveform augmentations.

    ``standard`` is intended for positives and phrase-specific hard negatives.
    ``light`` is intended for broad background negatives that already contain
    natural acoustic variation.
    """

    PRESETS: dict[str, dict[int, float]] = {
        "standard": {1: 1.00, 2: 0.70, 3: 0.50, 4: 0.30, 5: 0.10},
        "light": {1: 0.60, 2: 0.20},
    }
    _PRESET_DEFAULT_MAX_CHAIN: dict[str, int] = {"standard": 5, "light": 2}

    BASE_TRANSFORMS: tuple[_TransformSpec, ...] = (
        _TransformSpec("gaussian", 3, "noise"),
        _TransformSpec("gaussian_snr", 3, "noise"),
        _TransformSpec("band_pass", 3, "filter"),
        _TransformSpec("band_stop", 3, "filter"),
        _TransformSpec("high_pass", 3, "filter"),
        _TransformSpec("low_pass", 3, "filter"),
        _TransformSpec("gain", 2, "gain"),
        _TransformSpec("gain_transition", 2, "gain"),
        _TransformSpec("time_mask", 2, None),
        _TransformSpec("speed", 1, None),
        _TransformSpec("pitch", 1, None),
        _TransformSpec("polarity", 1, None),
        _TransformSpec("clipping", 1, None),
    )

    def __init__(
        self,
        noise_dir: Path | None = None,
        ir_dir: Path | None = None,
        *,
        short_noise_dir: Path | None = None,
        truck_noise_dir: Path | None = None,
        max_chain: int | None = None,
        p: float = 0.5,
        preset: str = "standard",
        regular_negative_preset: str = "light",
        seed: int | None = None,
    ) -> None:
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(self.PRESETS)}")
        if regular_negative_preset not in (*self.PRESETS.keys(), "none"):
            raise ValueError(
                f"Unknown regular-negative preset {regular_negative_preset!r}. "
                f"Choose from: {', '.join((*self.PRESETS.keys(), 'none'))}"
            )
        self.preset = preset
        self.regular_negative_preset = regular_negative_preset
        self.max_chain = min(max_chain or self._PRESET_DEFAULT_MAX_CHAIN[preset], 5)
        self.p = p
        self.rng = random.Random(seed)

        self.noise_dir = Path(noise_dir).expanduser() if noise_dir else None
        self.ir_dir = Path(ir_dir).expanduser() if ir_dir else None
        self.short_noise_dir = Path(short_noise_dir).expanduser() if short_noise_dir else None
        self.truck_noise_dir = Path(truck_noise_dir).expanduser() if truck_noise_dir else None

        self._noise_files = self._collect_wavs(self.noise_dir)
        self._ir_files = self._collect_wavs(self.ir_dir)
        self._short_noise_files = self._collect_wavs(self.short_noise_dir)
        self._truck_noise_files = self._collect_wavs(self.truck_noise_dir)

        self._transforms: list[_TransformSpec] = list(self.BASE_TRANSFORMS)
        if self._noise_files:
            self._transforms.append(_TransformSpec("background_noise", 3, "noise"))
        if self._short_noise_files:
            self._transforms.append(_TransformSpec("short_noise", 3, "noise"))
        if self._truck_noise_files:
            self._transforms.append(_TransformSpec("truck_noise", 3, "noise"))
        if self._ir_files:
            self._transforms.append(_TransformSpec("impulse_response", 3, "filter"))

    @staticmethod
    def _collect_wavs(directory: Path | None) -> list[Path]:
        if directory and directory.exists():
            return sorted(directory.rglob("*.wav"))
        return []

    @property
    def pool_size(self) -> int:
        return len(self._transforms)

    def _sample_cascade_level(self, preset: str | None = None) -> int:
        active = preset or self.preset
        if active == "none":
            return 0
        cascade_probs = self.PRESETS[active]
        max_chain = min(self.max_chain, self._PRESET_DEFAULT_MAX_CHAIN[active])
        level = 0
        for k in range(1, max_chain + 1):
            if self.rng.random() < cascade_probs.get(k, 0.0):
                level = k
            else:
                break
        return level

    def _weighted_sample_without_replacement(self, k: int) -> list[_TransformSpec]:
        candidates = list(self._transforms)
        selected: list[_TransformSpec] = []
        excluded_groups: set[str] = set()
        for _ in range(k):
            eligible = [
                spec
                for spec in candidates
                if spec.group is None or spec.group not in excluded_groups
            ]
            if not eligible:
                break
            weights = [spec.weight for spec in eligible]
            chosen = self.rng.choices(eligible, weights=weights, k=1)[0]
            selected.append(chosen)
            candidates.remove(chosen)
            if chosen.group is not None:
                excluded_groups.add(chosen.group)
        return selected

    def _choose_audio(self, files: Sequence[Path]) -> torch.Tensor | None:
        if not files:
            return None
        try:
            return _load_wav(self.rng.choice(list(files)), SAMPLE_RATE)
        except Exception:
            return None

    def _apply_transform(self, wav: torch.Tensor, sr: int, spec: _TransformSpec) -> torch.Tensor:
        name = spec.name
        if name == "gaussian":
            return add_gaussian_noise(wav, snr_db=self.rng.uniform(10, 40))
        if name == "gaussian_snr":
            return add_gaussian_noise(wav, snr_db=self.rng.uniform(5, 40))
        if name in {"band_pass", "band_stop", "high_pass", "low_pass"}:
            return _safe_filter(wav, sr, name)
        if name == "gain":
            return _gain_db(wav, self.rng.uniform(-3, 12))
        if name == "gain_transition":
            return _gain_transition(wav, self.rng.uniform(-6, 6), self.rng.uniform(-6, 6))
        if name == "time_mask":
            return _time_mask_waveform(wav)
        if name == "speed":
            return speed_perturb(wav, sr, factor=self.rng.uniform(0.8, 1.25))
        if name == "pitch":
            try:
                return pitch_shift(wav, sr, n_steps=self.rng.uniform(-4, 4))
            except Exception:
                return wav
        if name == "polarity":
            return _polarity_invert(wav)
        if name == "clipping":
            return _clipping_distortion(wav)
        if name == "background_noise":
            noise = self._choose_audio(self._noise_files)
            return mix_noise(wav, noise, snr_db=self.rng.uniform(3, 15)) if noise is not None else wav
        if name == "short_noise":
            noise = self._choose_audio(self._short_noise_files)
            return mix_noise(wav, noise, snr_db=self.rng.uniform(3, 15)) if noise is not None else wav
        if name == "truck_noise":
            noise = self._choose_audio(self._truck_noise_files)
            return mix_noise(wav, noise, snr_db=self.rng.uniform(3, 12)) if noise is not None else wav
        if name == "impulse_response":
            ir = self._choose_audio(self._ir_files)
            return apply_ir(wav, ir) if ir is not None else wav
        return wav

    def augment(
        self,
        wav: torch.Tensor,
        sr: int = SAMPLE_RATE,
        *,
        cascade_level: int | None = None,
        preset: str | None = None,
    ) -> torch.Tensor:
        active_preset = preset or self.preset
        if active_preset not in (*self.PRESETS.keys(), "none"):
            raise ValueError(f"Unknown preset {active_preset!r}. Choose from: {', '.join(self.PRESETS)}")
        if active_preset == "none":
            return _pad_or_trim(wav.clone())

        if cascade_level is None:
            cascade_level = self._sample_cascade_level(active_preset)
        cascade_level = min(max(0, int(cascade_level)), self.pool_size)
        if cascade_level == 0:
            return _pad_or_trim(wav.clone())

        result = wav.clone().float()
        for spec in self._weighted_sample_without_replacement(cascade_level):
            if self.rng.random() > self.p:
                continue
            try:
                result = self._apply_transform(result, sr, spec)
            except Exception:
                continue
        return _pad_or_trim(_normalize_peak(result))

    def augment_with_info(
        self,
        wav: torch.Tensor,
        sr: int = SAMPLE_RATE,
        *,
        cascade_level: int | None = None,
        preset: str | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        active_preset = preset or self.preset
        if cascade_level is None:
            cascade_level = self._sample_cascade_level(active_preset)
        selected = self._weighted_sample_without_replacement(min(cascade_level, self.pool_size))
        result = wav.clone().float()
        applied: list[str] = []
        for spec in selected:
            if self.rng.random() > self.p:
                continue
            before = result
            try:
                result = self._apply_transform(result, sr, spec)
            except Exception:
                result = before
                continue
            applied.append(spec.name)
        return _pad_or_trim(_normalize_peak(result)), applied

    def generate_all_levels(self, wav: torch.Tensor, sr: int = SAMPLE_RATE) -> list[tuple[int, torch.Tensor]]:
        return [(level, self.augment(wav, sr, cascade_level=level)) for level in range(self.max_chain + 1)]

    def __call__(self, wav: torch.Tensor, sr: int = SAMPLE_RATE, preset: str | None = None) -> torch.Tensor:
        return self.augment(wav, sr, preset=preset)


class Augmentor(CascadingAugmentor):
    """Backward-compatible name for the default cascading waveform augmentor."""


# ── Spectrogram augmentor ─────────────────────────────────────────────────────


class SpectrogramAugmentor:
    """Apply SpecAugment-style masking, time warping, and noise to mel tensors."""

    def __init__(
        self,
        freq_mask_param: int = 8,
        num_freq_masks: int = 2,
        time_mask_param: int = 40,
        num_time_masks: int = 2,
        time_warp_w: int = 10,
        noise_std: float = 0.01,
        p_freq_mask: float = 0.8,
        p_time_mask: float = 0.8,
        p_time_warp: float = 0.3,
        p_noise: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.num_freq_masks = num_freq_masks
        self.time_mask_param = time_mask_param
        self.num_time_masks = num_time_masks
        self.time_warp_w = time_warp_w
        self.noise_std = noise_std
        self.p_freq_mask = p_freq_mask
        self.p_time_mask = p_time_mask
        self.p_time_warp = p_time_warp
        self.p_noise = p_noise
        self.rng = random.Random(seed)

    def frequency_mask(self, spec: torch.Tensor) -> torch.Tensor:
        result = spec.clone()
        n_mels = result.shape[-2]
        for _ in range(self.num_freq_masks):
            width = self.rng.randint(1, max(1, min(self.freq_mask_param, n_mels)))
            start = self.rng.randint(0, max(n_mels - width, 0))
            result[..., start : start + width, :] = 0.0
        return result

    def time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        result = spec.clone()
        n_time = result.shape[-1]
        for _ in range(self.num_time_masks):
            width = self.rng.randint(1, max(1, min(self.time_mask_param, n_time)))
            start = self.rng.randint(0, max(n_time - width, 0))
            result[..., :, start : start + width] = 0.0
        return result

    def time_warp(self, spec: torch.Tensor) -> torch.Tensor:
        n_time = spec.shape[-1]
        if n_time <= 2 * self.time_warp_w + 1:
            return spec.clone()
        src = self.rng.randint(self.time_warp_w, n_time - self.time_warp_w)
        dst = src + self.rng.randint(-self.time_warp_w, self.time_warp_w)
        if dst <= 0 or dst >= n_time or dst == src:
            return spec.clone()

        left = spec[..., :src]
        right = spec[..., src:]
        orig_shape = spec.shape[:-1]
        left_flat = left.reshape(-1, 1, left.shape[-1]).float()
        right_flat = right.reshape(-1, 1, right.shape[-1]).float()
        left_warped = F.interpolate(left_flat, size=dst, mode="linear", align_corners=False)
        right_warped = F.interpolate(right_flat, size=n_time - dst, mode="linear", align_corners=False)
        warped = torch.cat([left_warped, right_warped], dim=-1)
        return warped.reshape(*orig_shape, n_time).to(spec.dtype)

    def add_noise(self, spec: torch.Tensor) -> torch.Tensor:
        return spec + torch.randn_like(spec) * self.noise_std

    def augment(self, spec: torch.Tensor) -> torch.Tensor:
        result = spec.clone()
        if self.rng.random() < self.p_time_warp:
            result = self.time_warp(result)
        if self.rng.random() < self.p_freq_mask:
            result = self.frequency_mask(result)
        if self.rng.random() < self.p_time_mask:
            result = self.time_mask(result)
        if self.rng.random() < self.p_noise:
            result = self.add_noise(result)
        return result

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        return self.augment(spec)


# ── Config bridge ─────────────────────────────────────────────────────────────


def _optional_dir(raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else path


def build_training_augmentors(config: "ForgeConfig") -> tuple[CascadingAugmentor | None, SpectrogramAugmentor | None]:
    """Build waveform and spectrogram augmentors from persisted project config."""
    if not config.training_augmentation_enabled:
        return None, None

    waveform = CascadingAugmentor(
        noise_dir=_optional_dir(config.augmentation_noise_dir),
        ir_dir=_optional_dir(config.augmentation_ir_dir),
        short_noise_dir=_optional_dir(config.augmentation_short_noise_dir),
        truck_noise_dir=_optional_dir(config.augmentation_truck_noise_dir),
        max_chain=config.training_augmentation_max_chain,
        p=config.training_augmentation_probability,
        preset=config.training_augmentation_preset,
        regular_negative_preset=config.regular_negative_augmentation_preset,
    )
    spectrogram = SpectrogramAugmentor() if config.use_spectrogram_augmentation else None
    return waveform, spectrogram
