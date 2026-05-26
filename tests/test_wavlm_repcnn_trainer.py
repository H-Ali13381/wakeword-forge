"""
tests/test_wavlm_repcnn_trainer.py — WavLM teacher -> RepCNN student backend behavior.
"""

import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from forge.config import BACKGROUND_NEGATIVE_TARGET, ForgeConfig, SAMPLE_RATE
from forge.models.wavlm_repcnn import (
    RepConvBlock,
    WakewordDataset,
    WavLMRepCNNTrainer,
    export_repcnn_onnx,
)
from forge.review import training_data_fingerprint
from forge.trainer import run_training


def _write_wav(path: Path, seconds: float = 0.25, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    wav = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, wav, SAMPLE_RATE)


def test_wavlm_repcnn_dataset_labels_partials_as_negatives(tmp_path):
    pos = tmp_path / "pos.wav"
    neg = tmp_path / "neg.wav"
    partial = tmp_path / "partial.wav"
    _write_wav(pos, freq=440.0)
    _write_wav(neg, freq=220.0)
    _write_wav(partial, freq=330.0)

    dataset = WakewordDataset([pos], [neg], partial_files=[partial])

    assert len(dataset) == 3
    assert dataset.labels.tolist() == [1.0, 0.0, 0.0]
    wav, label = dataset[0]
    assert wav.ndim == 1
    assert label == 1.0


def test_wavlm_repcnn_trainer_export_without_training_checkpoint_raises(tmp_path):
    trainer = WavLMRepCNNTrainer(ForgeConfig(project_dir=str(tmp_path), wake_phrase="Hey Nova"))

    try:
        trainer.export_onnx()
    except RuntimeError as exc:
        assert "Train before exporting RepCNN" in str(exc)
    else:
        raise AssertionError("export_onnx should reject missing trained model")


def test_export_repcnn_onnx_writes_distillation_metadata(tmp_path):
    from forge.models.wavlm_repcnn import RepCNN

    path = tmp_path / "wakeword.onnx"

    exported = export_repcnn_onnx(
        RepCNN(channels=8, kernel_sizes=(3,), n_branches=1),
        path,
        wake_phrase="Hey Nova",
        threshold=0.42,
        eer=0.12,
        teacher_model="microsoft/wavlm-base",
    )

    assert exported == path
    sidecar = path.with_suffix(".json")
    assert sidecar.exists()
    metadata = __import__("json").loads(sidecar.read_text())
    assert metadata["backend"] == "wavlm-repcnn"
    assert metadata["model_type"] == "repcnn"
    assert metadata["reparameterized"] is True
    assert metadata["repconv_merged"] is True
    assert metadata["teacher_model_type"] == "wavlm"
    assert metadata["teacher_model"] == "microsoft/wavlm-base"
    assert metadata["threshold"] == 0.42

    # ONNX graph must expose the stable runtime contract used by mic-test/quality-check.
    import onnx

    model = onnx.load(path)
    assert [input.name for input in model.graph.input] == ["waveform"]
    assert [output.name for output in model.graph.output] == ["score"]
    initializer_names = [init.name for init in model.graph.initializer]
    assert any("fused_conv" in name for name in initializer_names)
    assert not any("branches" in name for name in initializer_names)


def test_repconv_block_reparameterize_preserves_eval_output():
    import torch

    torch.manual_seed(0)
    block = RepConvBlock(channels=4, kernel_size=5, n_branches=2).eval()
    x = torch.randn(3, 4, 17)

    with torch.no_grad():
        before = block(x)
        block.reparameterize().eval()
        after = block(x)

    assert block.fused_conv is not None
    assert len(block.branches) == 0
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)


def test_run_training_dispatches_default_wavlm_repcnn_backend(tmp_path, monkeypatch):
    for i in range(10):
        _write_wav(tmp_path / "samples" / "positives" / f"pos_{i}.wav", freq=440 + i)
    for i in range(BACKGROUND_NEGATIVE_TARGET):
        _write_wav(tmp_path / "samples" / "negatives" / f"neg_{i}.wav", freq=220 + i)

    calls = {}

    class FakeWavLMRepCNNTrainer:
        def __init__(self, config):
            calls["backend"] = config.backend
            self.threshold = 0.37
            self.eer = 0.11

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

    fake_module = types.ModuleType("forge.models.wavlm_repcnn")
    fake_module.WavLMRepCNNTrainer = FakeWavLMRepCNNTrainer
    monkeypatch.setitem(sys.modules, "forge.models.wavlm_repcnn", fake_module)

    cfg = ForgeConfig(project_dir=str(tmp_path), wake_phrase="Hey Nova", use_tts_augmentation=False)
    exported = run_training(cfg)

    assert exported == tmp_path / "output" / "wakeword.onnx"
    assert calls == {
        "backend": "wavlm-repcnn",
        "pos": 10,
        "neg": BACKGROUND_NEGATIVE_TARGET,
        "partials": 0,
        "augmentor": "CascadingAugmentor",
        "spec_augmentor": None,
    }
    assert cfg.trained_threshold == 0.37
    assert cfg.trained_eer == 0.11
    assert cfg.trained_sample_fingerprint == training_data_fingerprint(cfg)
