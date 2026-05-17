"""
tests/test_dscnn_trainer.py — DS-CNN trainer behavior and public backend dispatch.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from wakeword_forge.config import BACKGROUND_NEGATIVE_TARGET, ForgeConfig, SAMPLE_RATE
from wakeword_forge.models.dscnn_trainer import DSCNNDataset, DSCNNTrainer, _build_sampler
from wakeword_forge.review import training_data_fingerprint
from wakeword_forge.trainer import run_training


def _write_wav(path: Path, seconds: float = 0.25, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    wav = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, wav, SAMPLE_RATE)


def test_dscnn_dataset_labels_partials_as_negatives(tmp_path):
    pos = tmp_path / "pos.wav"
    neg = tmp_path / "neg.wav"
    partial = tmp_path / "partial.wav"
    _write_wav(pos, freq=440.0)
    _write_wav(neg, freq=220.0)
    _write_wav(partial, freq=330.0)

    dataset = DSCNNDataset([pos], [neg], partial_files=[partial])

    assert len(dataset) == 3
    assert dataset.labels.tolist() == [1.0, 0.0, 0.0]
    wav, label = dataset[0]
    assert wav.ndim == 1
    assert label == 1.0


def test_dscnn_dataset_uses_standard_and_light_augmentation_profiles(tmp_path):
    pos = tmp_path / "pos.wav"
    neg = tmp_path / "neg.wav"
    partial = tmp_path / "partial.wav"
    _write_wav(pos, freq=440.0)
    _write_wav(neg, freq=220.0)
    _write_wav(partial, freq=330.0)

    class RecordingAugmentor:
        regular_negative_preset = "light"

        def __init__(self):
            self.presets: list[str | None] = []

        def __call__(self, wav, sr=SAMPLE_RATE, preset=None):
            self.presets.append(preset)
            return wav

    augmentor = RecordingAugmentor()
    dataset = DSCNNDataset([pos], [neg], partial_files=[partial], augmentor=augmentor)

    assert len(dataset) == 7
    assert dataset.labels.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    for index in range(len(dataset)):
        dataset[index]

    assert augmentor.presets == ["standard", "standard", "light", "standard"]


def test_build_sampler_requires_both_classes():
    sampler = _build_sampler(torch.tensor([1.0, 0.0, 0.0]))
    assert sampler.num_samples == 3

    try:
        _build_sampler(torch.tensor([1.0, 1.0]))
    except ValueError as exc:
        assert "Need both classes" in str(exc)
    else:
        raise AssertionError("_build_sampler should reject single-class labels")


def test_dscnn_trainer_export_without_training_checkpoint_raises(tmp_path):
    trainer = DSCNNTrainer(ForgeConfig(project_dir=str(tmp_path), wake_phrase="Hey Nova"))

    try:
        trainer.export_onnx()
    except RuntimeError as exc:
        assert "Train before exporting" in str(exc)
    else:
        raise AssertionError("export_onnx should reject missing trained model")


def test_run_training_rejects_sparse_negative_coverage(tmp_path, monkeypatch):
    for i in range(10):
        _write_wav(tmp_path / "samples" / "positives" / f"pos_{i}.wav", freq=440 + i)
    for i in range(5):
        _write_wav(tmp_path / "samples" / "negatives" / f"neg_{i}.wav", freq=220 + i)

    class FakeDSCNNTrainer:
        def __init__(self, _config):
            pass

        def train(self, *_args, **_kwargs):
            raise AssertionError("training should be blocked before backend dispatch")

        def export_onnx(self):
            raise AssertionError("training should be blocked before export")

    fake_module = types.ModuleType("wakeword_forge.models.dscnn_trainer")
    fake_module.DSCNNTrainer = FakeDSCNNTrainer
    monkeypatch.setitem(sys.modules, "wakeword_forge.models.dscnn_trainer", fake_module)

    cfg = ForgeConfig(project_dir=str(tmp_path), wake_phrase="Hey Nova")

    with pytest.raises(ValueError, match="negative coverage"):
        run_training(cfg)


def test_run_training_dispatches_default_dscnn_backend(tmp_path, monkeypatch):
    for i in range(10):
        _write_wav(tmp_path / "samples" / "positives" / f"pos_{i}.wav", freq=440 + i)
    for i in range(BACKGROUND_NEGATIVE_TARGET):
        _write_wav(tmp_path / "samples" / "negatives" / f"neg_{i}.wav", freq=220 + i)

    calls = {}

    class FakeDSCNNTrainer:
        def __init__(self, config):
            calls["backend"] = config.backend
            self._threshold = 0.37
            self._eer = 0.11

        def train(self, pos_files, neg_files, partial_files=None, augmentor=None, spec_augmentor=None):
            calls["pos"] = len(pos_files)
            calls["neg"] = len(neg_files)
            calls["partials"] = len(partial_files or [])
            calls["augmentor"] = type(augmentor).__name__ if augmentor else None
            calls["spec_augmentor"] = type(spec_augmentor).__name__ if spec_augmentor else None

        def export_onnx(self):
            out = tmp_path / "output" / "wakeword.onnx"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake")
            return out

    fake_module = types.ModuleType("wakeword_forge.models.dscnn_trainer")
    fake_module.DSCNNTrainer = FakeDSCNNTrainer
    monkeypatch.setitem(sys.modules, "wakeword_forge.models.dscnn_trainer", fake_module)

    cfg = ForgeConfig(project_dir=str(tmp_path), wake_phrase="Hey Nova", use_tts_augmentation=False)
    exported = run_training(cfg)

    assert exported == tmp_path / "output" / "wakeword.onnx"
    assert calls == {
        "backend": "dscnn",
        "pos": 10,
        "neg": BACKGROUND_NEGATIVE_TARGET,
        "partials": 0,
        "augmentor": "CascadingAugmentor",
        "spec_augmentor": None,
    }
    assert cfg.trained_threshold == 0.37
    assert cfg.trained_eer == 0.11
    assert cfg.trained_sample_fingerprint == training_data_fingerprint(cfg)
