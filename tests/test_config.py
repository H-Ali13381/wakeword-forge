"""
tests/test_config.py — unit tests for ForgeConfig
"""



from wakeword_forge.config import ForgeConfig


def test_save_load_roundtrip(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    cfg_file = tmp_path / "forge_config.json"
    cfg.save(cfg_file)
    loaded = ForgeConfig.load(cfg_file)
    assert loaded.wake_phrase == "Hey Nova"
    assert loaded.backend == "dscnn"


def test_resolved_paths(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path))
    assert cfg.positives_path == tmp_path / "samples" / "positives"
    assert cfg.negatives_path == tmp_path / "samples" / "negatives"
    assert cfg.output_path == tmp_path / "output"


def test_default_record_duration_allows_extra_margin_for_trimming():
    assert ForgeConfig().record_duration == 4.0


def test_config_phrase_options_dedupe_primary_and_extra_phrases():
    cfg = ForgeConfig(
        wake_phrase="Okay Hermes",
        wake_phrases=["Okay Hermes", "Hey Hermes", "  ", "hey hermes"],
    )

    assert cfg.phrase_options == ("Okay Hermes", "Hey Hermes", "hey hermes")
