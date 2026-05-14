from __future__ import annotations

from wakeword_forge.config import ForgeConfig
from wakeword_forge.project import CONFIG_FILENAME, load_or_create_config, reset_project


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
