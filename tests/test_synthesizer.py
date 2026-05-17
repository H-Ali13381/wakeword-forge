from __future__ import annotations

import sys
import tomllib
import types
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
        "Hey Nova",
        tmp_path,
        n=1,
        engine="kokoro",
    )

    assert [path.name for path in saved] == ["partial_0001.wav"]
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_build_backend_accepts_qwentts_engine_aliases(monkeypatch):
    created: list[str] = []

    class StubQwenTTSBackend:
        def __init__(self):
            created.append("qwentts")

    monkeypatch.setattr(synthesizer, "QwenTTSBackend", StubQwenTTSBackend, raising=False)

    for engine in ("qwentts", "qwen-tts", "qwen3-tts"):
        assert isinstance(synthesizer.build_backend(engine), StubQwenTTSBackend)

    assert created == ["qwentts", "qwentts", "qwentts"]


def test_qwentts_backend_uses_baseline_custom_voice_api(monkeypatch):
    calls: dict[str, dict] = {}

    class FakeModel:
        def generate_custom_voice(self, **kwargs):
            calls["generate"] = kwargs
            return ([np.ones(160, dtype=np.float32) * 0.25], 24_000)

    class FakeQwen3TTSModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["load"] = {"model_name": model_name, **kwargs}
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=FakeQwen3TTSModel))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="fake-bfloat16"))

    backend = synthesizer.QwenTTSBackend(
        voice="Ryan",
        instructions=("Speak slowly and clearly.",),
        model_name="fake/qwentts",
        device="cpu",
        use_flash_attn=False,
    )

    audio, sample_rate = backend.synthesize("Hey Nova")

    assert calls["load"]["model_name"] == "fake/qwentts"
    assert calls["load"]["device_map"] == "cpu"
    assert calls["load"]["dtype"] == "fake-bfloat16"
    assert calls["load"]["attn_implementation"] == "sdpa"
    assert calls["generate"] == {
        "text": "Hey Nova",
        "language": "English",
        "speaker": "Ryan",
        "instruct": "Speak slowly and clearly.",
    }
    assert sample_rate == 24_000
    assert audio.dtype == np.float32
    assert audio.shape == (160,)


def test_qwentts_voice_designs_mirror_work_pipeline_config():
    designs = synthesizer.QWENTTS_VOICE_DESIGNS

    assert tuple(designs) == ("english_voices", "french_voices", "additional_voices", "notes")
    assert designs["notes"].startswith("All 9 Qwen3-TTS CustomVoice speakers")

    voices = synthesizer.QWENTTS_BASELINE_VOICES
    assert [voice["name"] for voice in voices] == [
        "Ryan",
        "Aiden",
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ono_Anna",
        "Sohee",
        "Ryan_fr",
        "Aiden_fr",
        "Vivian_en",
        "Serena_en",
    ]
    by_name = {voice["name"]: voice for voice in voices}
    assert by_name["Ryan_fr"] == {
        "name": "Ryan_fr",
        "speaker": "Ryan",
        "language": "French",
        "gender": "male",
        "native_language": "English (American)",
        "description": "Ryan (American male) synthesizing French — yields US-accented French pronunciation for English wake phrases",
    }
    assert by_name["Aiden_fr"] == {
        "name": "Aiden_fr",
        "speaker": "Aiden",
        "language": "French",
        "gender": "male",
        "native_language": "English (American)",
        "description": "Aiden (American male) synthesizing French — yields a second US-accented French rendition",
    }
    assert by_name["Vivian_en"]["speaker"] == "Vivian"
    assert by_name["Serena_en"]["speaker"] == "Serena"

    assert synthesizer.QWENTTS_STYLE_INSTRUCTIONS == (
        "",
        "Speak slowly and clearly.",
        "Speak quickly and casually.",
        "Speak with a whisper.",
        "Speak with enthusiasm and energy.",
        "Speak in a calm, relaxed tone.",
        "Speak as if calling someone from across the room.",
        "Speak as if you just woke up, groggy and half-asleep.",
        "Speak while distracted, casually, as if doing something else.",
        "Speak as if firmly commanding a device.",
        "Speak in a flat, expressionless monotone.",
        "Speak with rising intonation, as if asking a question.",
        "Speak with mild impatience or frustration.",
        "Speak in a bright, cheerful, upbeat manner.",
        "Speak with a British English accent.",
        "Speak with an Indian English accent.",
        "Speak with a noticeable French accent.",
        "Speak with a Québécois French accent.",
        "Say it very fast, blending the words together as one.",
        "Say it with a clear, deliberate pause between each word.",
    )


def test_qwentts_backend_uses_cross_language_voice_design(monkeypatch):
    calls: dict[str, dict] = {}

    class FakeModel:
        def generate_custom_voice(self, **kwargs):
            calls["generate"] = kwargs
            return ([np.ones(160, dtype=np.float32) * 0.25], 24_000)

    class FakeQwen3TTSModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["load"] = {"model_name": model_name, **kwargs}
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=FakeQwen3TTSModel))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="fake-bfloat16"))

    backend = synthesizer.QwenTTSBackend(
        voice="Ryan_fr",
        instructions=("",),
        model_name="fake/qwentts",
        device="cpu",
        use_flash_attn=False,
    )

    backend.synthesize("Hey Nova")

    assert calls["generate"] == {
        "text": "Hey Nova",
        "language": "French",
        "speaker": "Ryan",
    }


def test_qwentts_synthetic_backend_is_documented_and_optional_extra_declared():
    readme = Path("README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "ENGINE=qwentts" in readme
    assert "13 mirrored CustomVoice voice designs" in readme
    extras = pyproject["project"]["optional-dependencies"]
    assert "qwentts" in extras
    assert "qwen-tts" in extras["qwentts"]
