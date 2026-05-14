from __future__ import annotations

from pathlib import Path

from wakeword_forge.negatives import _generate_synthetic_negatives, ensure_negatives


def _touch_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")


def test_generate_synthetic_negatives_appends_after_existing_numbered_files(tmp_path):
    _touch_wav(tmp_path / "synthetic_neg_0000.wav")

    saved = _generate_synthetic_negatives(tmp_path, 1)

    assert [path.name for path in saved] == ["synthetic_neg_0001.wav"]
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_ensure_negatives_reaches_local_target_without_external_sources(tmp_path):
    for i in range(10):
        _touch_wav(tmp_path / f"recorded_neg_{i:04d}.wav")

    all_wavs = ensure_negatives(
        tmp_path,
        target=115,
        use_common_voice=False,
        use_esc50=False,
    )

    assert len(all_wavs) == 115
    assert len(list(tmp_path.glob("*.wav"))) == 115
