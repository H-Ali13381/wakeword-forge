from __future__ import annotations

from typer.testing import CliRunner

from wakeword_forge.cli import app
from wakeword_forge.config import ForgeConfig


def test_info_displays_zero_eer_as_trained_metric(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path), wake_phrase="Hey Nova")
    cfg.trained_eer = 0.0
    cfg.trained_threshold = 0.3947
    cfg.save(tmp_path / "forge_config.json")

    result = CliRunner().invoke(app, ["info", "--dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Trained EER" in result.output
    assert "0.0000" in result.output
