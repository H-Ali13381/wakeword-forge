from __future__ import annotations

from pathlib import Path

import numpy as np

from wakeword_forge.config import SAMPLE_RATE
from wakeword_forge import synthesizer


def _touch_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")


class FakeBackend:
    def synthesize(self, _text: str, **_kwargs):
        return np.zeros(160, dtype=np.float32), SAMPLE_RATE


def test_partial_negative_synthesis_appends_after_existing_numbered_files(monkeypatch, tmp_path):
    monkeypatch.setattr(synthesizer, "build_backend", lambda _engine: FakeBackend())
    _touch_wav(tmp_path / "partial_0000.wav")

    saved = synthesizer.synthesize_partial_negatives(
        "Okay Hermes",
        tmp_path,
        n=1,
        engine="kokoro",
    )

    assert [path.name for path in saved] == ["partial_0001.wav"]
    assert len(list(tmp_path.glob("*.wav"))) == 2
