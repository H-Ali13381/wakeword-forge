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
