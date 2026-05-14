from __future__ import annotations

import numpy as np
import soundfile as sf

from wakeword_forge.config import ForgeConfig, SAMPLE_RATE
from wakeword_forge.project import CONFIG_FILENAME, import_positive_samples, load_or_create_config, reset_project
from wakeword_forge.review import training_data_fingerprint


def _write_wav(path, *, seconds: float = 0.25, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = max(1, int(seconds * sample_rate))
    t = np.linspace(0, seconds, samples, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def test_reset_project_removes_forge_artifacts_without_deleting_project_root(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    cfg.save(tmp_path / CONFIG_FILENAME)
    keep_file = tmp_path / "notes.txt"
    keep_file.write_text("keep me")

    artifact_files = [
        cfg.positives_path / "take_0000.wav",
        cfg.negatives_path / "neg_0000.wav",
        cfg.synthetic_path / "synth_0000.wav",
        cfg.partials_path / "partial_0000.wav",
        cfg.confusables_path / "confusable_0000.wav",
        cfg.output_path / "wakeword.onnx",
        cfg.cache_path / "tmp.bin",
        cfg.confusables_cache,
    ]
    for path in artifact_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    removed = reset_project(cfg)

    assert tmp_path.exists()
    assert keep_file.read_text() == "keep me"
    assert not (tmp_path / CONFIG_FILENAME).exists()
    assert not cfg.samples_path.exists()
    assert not cfg.output_path.exists()
    assert not cfg.cache_path.exists()
    assert not cfg.confusables_cache.exists()
    assert set(artifact_files + [tmp_path / CONFIG_FILENAME]).issubset(set(removed))
    assert load_or_create_config(tmp_path).wake_phrase == ""


def test_import_positive_samples_copies_existing_audio_without_overwriting(tmp_path):
    source = tmp_path / "existing_voice"
    _write_wav(source / "clip_a.wav")
    _write_wav(source / "nested" / "clip_b.wav")
    (source / "notes.txt").write_text("not audio")

    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path / "project"))
    _write_wav(cfg.positives_path / "imported_0000.wav")

    result = import_positive_samples(cfg, source, limit=2)

    assert result.imported_count == 2
    assert result.available_count == 2
    assert [path.name for path in result.imported_paths] == ["imported_0001.wav", "imported_0002.wav"]
    assert (cfg.positives_path / "imported_0000.wav").exists()
    assert all(path.exists() for path in result.imported_paths)


def test_import_positive_samples_invalidates_reviews_and_model_acceptance(tmp_path):
    source = tmp_path / "existing_voice"
    _write_wav(source / "clip_a.wav")

    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path / "project"))
    _write_wav(cfg.positives_path / "take_0000.wav")
    cfg.sample_review_approved = True
    cfg.sample_review_fingerprint = "old-samples"
    cfg.trained_sample_fingerprint = training_data_fingerprint(cfg)
    cfg.quality_check_passed = True
    cfg.model_accepted = True
    cfg.quality_checked_model_path = str(cfg.output_path / "wakeword.onnx")
    cfg.quality_checked_model_fingerprint = "old-model"
    cfg.accepted_model_fingerprint = "old-model"
    cfg.quality_positive_hits = 3
    cfg.quality_positive_trials = 3
    cfg.quality_false_triggers = 0
    cfg.quality_score_min = 0.1
    cfg.quality_score_max = 0.9

    result = import_positive_samples(cfg, source, limit=1)

    assert result.imported_count == 1
    assert cfg.sample_review_approved is False
    assert cfg.sample_review_fingerprint == ""
    assert cfg.quality_check_passed is False
    assert cfg.model_accepted is False
    assert cfg.quality_checked_model_path == ""
    assert cfg.quality_checked_model_fingerprint == ""
    assert cfg.accepted_model_fingerprint == ""
    assert cfg.quality_positive_hits == 0
    assert cfg.quality_positive_trials == 0
    assert cfg.quality_score_min is None
    assert cfg.quality_score_max is None


def test_import_positive_samples_rejects_missing_source_folder(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path / "project"))

    try:
        import_positive_samples(cfg, tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "Existing sample folder" in str(exc)
    else:
        raise AssertionError("missing sample folder should raise FileNotFoundError")
