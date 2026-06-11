"""
tests/test_augmentation.py — smoke tests for augmentation
"""

import numpy as np
import soundfile as sf
import torch
from pathlib import Path

from forge.augmentation import (
    add_gaussian_noise,
    augmentation_metadata,
    far_field,
    low_gain_mic,
    onset_jitter,
    speed_perturb,
    time_shift,
    amplitude_scale,
    Augmentor,
    CascadingAugmentor,
    SpectrogramAugmentor,
    build_training_augmentors,
    _load_wav,
)
from forge.config import SAMPLE_RATE, MAX_SAMPLES, ForgeConfig


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


def test_low_gain_mic_degrades_snr_instead_of_only_scaling_amplitude():
    torch.manual_seed(0)
    t = torch.linspace(0, 1, SAMPLE_RATE, dtype=torch.float32).unsqueeze(0)
    wav = 0.5 * torch.sin(2 * torch.pi * 440 * t)

    degraded = low_gain_mic(
        wav,
        attenuation_db=-24.0,
        noise_floor_rms=0.02,
        quantization_bits=None,
    )

    attenuated_direct = wav * (10 ** (-24.0 / 20))
    residual = degraded - attenuated_direct

    assert degraded.shape == wav.shape
    assert residual.pow(2).mean().sqrt() > attenuated_direct.pow(2).mean().sqrt() * 0.5


def test_far_field_preserves_shape_and_adds_countable_room_distance_stress():
    torch.manual_seed(0)
    wav = torch.zeros(1, SAMPLE_RATE)
    wav[..., 100:200] = 0.5
    ir = torch.zeros(1, 400)
    ir[..., 0] = 1.0
    ir[..., 120] = 0.35
    background = torch.randn(1, SAMPLE_RATE) * 0.01

    stressed = far_field(
        wav,
        SAMPLE_RATE,
        ir=ir,
        background=background,
        distance_db=-12.0,
        wet_mix=0.4,
        snr_db=6.0,
    )

    assert stressed.shape == wav.shape
    assert not torch.allclose(stressed, wav)


def test_onset_jitter_places_short_clip_inside_model_window_without_wraparound():
    wav = torch.ones(1, 100)

    shifted = onset_jitter(wav, max_samples=300, offset=80)

    assert shifted.shape == (1, 300)
    assert torch.count_nonzero(shifted[..., :80]) == 0
    assert torch.all(shifted[..., 80:180] == 1)
    assert torch.count_nonzero(shifted[..., 180:]) == 0


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


def test_cascading_augmentor_presets_and_level_generation():
    standard = CascadingAugmentor(preset="standard", seed=123)
    light = CascadingAugmentor(preset="light", seed=123)

    assert standard.max_chain == 5
    assert light.max_chain == 2
    assert standard.pool_size >= light.pool_size >= 1

    wav = make_wav(0.5)
    variants = standard.generate_all_levels(wav)

    assert [level for level, _variant in variants] == [0, 1, 2, 3, 4, 5]
    assert all(variant.shape[-1] == MAX_SAMPLES for _level, variant in variants)


def test_cascading_augmentor_policy_includes_audit_robustness_transforms():
    specs = {spec.name: spec for spec in CascadingAugmentor.BASE_TRANSFORMS}

    assert {"low_gain_mic", "far_field", "onset_jitter"}.issubset(specs)
    assert specs["speed"].weight >= 3
    assert CascadingAugmentor.SPEED_RANGE[1] >= 1.35


def test_robust_v1_is_explicit_audit_validation_preset():
    robust = CascadingAugmentor(preset="robust-v1", seed=123)
    specs = {spec.name: spec for spec in robust.BASE_TRANSFORMS}

    assert robust.preset == "robust-v1"
    assert robust.max_chain == 5
    assert robust.PRESETS["robust-v1"] == robust.PRESETS["standard"]
    assert {"low_gain_mic", "far_field", "onset_jitter"}.issubset(specs)
    assert specs["speed"].weight >= 3
    assert robust.SPEED_RANGE == (0.85, 1.35)


def test_build_training_augmentors_accepts_robust_v1_preset(tmp_path):
    config = ForgeConfig(
        project_dir=str(tmp_path),
        training_augmentation_enabled=True,
        training_augmentation_preset="robust-v1",
        regular_negative_augmentation_preset="light",
    )

    waveform, _spectrogram = build_training_augmentors(config)
    metadata = augmentation_metadata(config)

    assert isinstance(waveform, CascadingAugmentor)
    assert waveform.preset == "robust-v1"
    assert metadata["policy_version"] == "robust-v1"
    assert metadata["waveform"]["preset"] == "robust-v1"


def test_cascading_augmentor_rejects_unknown_preset():
    try:
        CascadingAugmentor(preset="extreme")
    except ValueError as exc:
        assert "Unknown preset" in str(exc)
    else:
        raise AssertionError("Unknown augmentation presets should be rejected")


def test_spectrogram_augmentor_preserves_shape_and_changes_copy():
    spec = torch.ones(1, 40, 301)
    augmentor = SpectrogramAugmentor(
        freq_mask_param=8,
        time_mask_param=30,
        p_freq_mask=1.0,
        p_time_mask=1.0,
        p_time_warp=0.0,
        p_noise=0.0,
        seed=7,
    )

    augmented = augmentor.augment(spec)

    assert augmented.shape == spec.shape
    assert augmented.data_ptr() != spec.data_ptr()
    assert torch.count_nonzero(augmented == 0) > 0


def test_build_training_augmentors_respects_config_paths_and_disable(tmp_path):
    noise_dir = tmp_path / "noise"
    ir_dir = tmp_path / "ir"
    short_noise_dir = tmp_path / "short"
    truck_noise_dir = tmp_path / "truck"
    for path in (noise_dir, ir_dir, short_noise_dir, truck_noise_dir):
        path.mkdir()

    disabled = ForgeConfig(project_dir=str(tmp_path), training_augmentation_enabled=False)
    assert build_training_augmentors(disabled) == (None, None)

    config = ForgeConfig(
        project_dir=str(tmp_path),
        training_augmentation_enabled=True,
        training_augmentation_preset="standard",
        regular_negative_augmentation_preset="light",
        augmentation_noise_dir=str(noise_dir),
        augmentation_ir_dir=str(ir_dir),
        augmentation_short_noise_dir=str(short_noise_dir),
        augmentation_truck_noise_dir=str(truck_noise_dir),
        use_spectrogram_augmentation=True,
    )

    waveform, spectrogram = build_training_augmentors(config)

    assert isinstance(waveform, CascadingAugmentor)
    assert waveform.preset == "standard"
    assert waveform.regular_negative_preset == "light"
    assert waveform.noise_dir == noise_dir
    assert waveform.ir_dir == ir_dir
    assert waveform.short_noise_dir == short_noise_dir
    assert waveform.truck_noise_dir == truck_noise_dir
    assert isinstance(spectrogram, SpectrogramAugmentor)


def test_augmentation_metadata_records_policy_assets_and_specaugment(tmp_path):
    noise_dir = tmp_path / "noise"
    ir_dir = tmp_path / "ir"
    short_noise_dir = tmp_path / "short"
    truck_noise_dir = tmp_path / "truck"
    for path in (noise_dir, ir_dir, short_noise_dir, truck_noise_dir):
        path.mkdir()
    sf.write(noise_dir / "fan.wav", np.zeros(160, dtype=np.float32), SAMPLE_RATE)
    sf.write(ir_dir / "room.wav", np.r_[1.0, np.zeros(159)].astype(np.float32), SAMPLE_RATE)

    config = ForgeConfig(
        project_dir=str(tmp_path),
        training_augmentation_enabled=True,
        training_augmentation_preset="standard",
        regular_negative_augmentation_preset="light",
        augmentation_noise_dir=str(noise_dir),
        augmentation_ir_dir=str(ir_dir),
        augmentation_short_noise_dir=str(short_noise_dir),
        augmentation_truck_noise_dir=str(truck_noise_dir),
        use_spectrogram_augmentation=True,
    )

    metadata = augmentation_metadata(config)

    assert metadata["enabled"] is True
    assert metadata["policy_version"]
    assert metadata["waveform"]["preset"] == "standard"
    assert metadata["waveform"]["regular_negative_preset"] == "light"
    assert "low_gain_mic" in metadata["waveform"]["transforms"]
    assert metadata["asset_counts"] == {
        "background_noise": 1,
        "room_impulse_response": 1,
        "short_noise": 0,
        "truck_noise": 0,
    }
    assert metadata["spectrogram"]["enabled"] is True


def test_training_augmentation_docs_do_not_advertise_unused_extra():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    makefile = (root / "Makefile").read_text()
    docs = (root / "README.md").read_text() + (root / "docs/advanced-usage.md").read_text()

    assert "augment = [" not in pyproject
    assert "audiomentations" not in pyproject
    assert "install-augment" not in makefile
    assert "[tts,ui,augment]" not in docs
    assert "--augmentation-preset" in makefile
    assert "training-time acoustic augmentation" in docs
