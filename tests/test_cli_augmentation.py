from typer.testing import CliRunner

from forge.cli import app


def test_train_help_exposes_robust_v1_as_explicit_augmentation_preset():
    result = CliRunner().invoke(app, ["train", "--help"])

    assert result.exit_code == 0, result.output
    assert "robust-v1" in result.output
    assert "augmentation-pres" in result.output
