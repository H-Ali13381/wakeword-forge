"""
tests/test_dscnn.py — public DS-CNN architecture smoke tests.
"""

import json

import onnx
import torch

from wakeword_forge.config import MAX_FRAMES, N_MELS, SAMPLE_RATE
from wakeword_forge.models.dscnn import DSCNN, DSCNNDetector, LogMelFrontend, export_dscnn_onnx


def test_dscnn_forward_returns_scores():
    model = DSCNN()
    mel = torch.randn(2, N_MELS, MAX_FRAMES)

    out = model(mel)

    assert out.shape == (2,)
    assert (out >= 0).all() and (out <= 1).all()


def test_log_mel_frontend_shape():
    frontend = LogMelFrontend()
    wav = torch.randn(2, SAMPLE_RATE * 2)

    mel = frontend(wav)

    assert mel.shape == (2, N_MELS, MAX_FRAMES)


def test_dscnn_detector_accepts_waveform():
    detector = DSCNNDetector(DSCNN())
    wav = torch.randn(1, SAMPLE_RATE * 2)

    with torch.no_grad():
        out = detector(wav)

    assert out.shape == (1,)
    assert 0.0 <= float(out[0]) <= 1.0


def test_export_dscnn_onnx_writes_stable_artifacts(tmp_path):
    path = tmp_path / "wakeword.onnx"

    exported = export_dscnn_onnx(
        DSCNN(),
        path,
        wake_phrase="Hey Nova",
        threshold=0.42,
        eer=0.12,
    )

    assert exported == path
    assert path.exists()

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["wake_phrase"] == "Hey Nova"
    assert sidecar["sample_rate"] == SAMPLE_RATE
    assert sidecar["threshold"] == 0.42
    assert sidecar["eer"] == 0.12
    assert sidecar["backend"] == "dscnn"
    assert sidecar["model_type"] == "dscnn"
    assert sidecar["model_file"] == "wakeword.onnx"

    metadata = {p.key: p.value for p in onnx.load(path).metadata_props}
    assert metadata["wake_phrase"] == "Hey Nova"
    assert metadata["sample_rate"] == str(SAMPLE_RATE)
    assert metadata["threshold"] == "0.42"
    assert metadata["model_type"] == "dscnn"
