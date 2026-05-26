from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _dry_run_make_train(*overrides: str) -> str:
    result = subprocess.run(
        ["make", "-n", "train", "DIR=/tmp/wakeword-forge-make-test", *overrides],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _train_command(dry_run: str) -> str:
    commands = [line for line in dry_run.splitlines() if "wakeword-forge train" in line]
    assert len(commands) == 1
    return commands[0]


def test_make_train_respects_saved_config_by_default():
    command = _train_command(_dry_run_make_train())

    assert '--dir "/tmp/wakeword-forge-make-test"' in command
    assert "--backend wavlm-repcnn" in command
    assert "--augmentation" not in command
    assert "--no-augmentation" not in command
    assert "--augmentation-preset" not in command
    assert "--regular-negative-preset" not in command
    assert "--spectrogram-augmentation" not in command
    assert "--no-spectrogram-augmentation" not in command


def test_make_train_can_explicitly_override_training_augmentation_config():
    command = _train_command(
        _dry_run_make_train(
            "AUGMENTATION=--no-augmentation",
            "AUGMENTATION_PRESET=light",
            "REGULAR_NEGATIVE_PRESET=none",
            "SPECTROGRAM_AUGMENTATION=--spectrogram-augmentation",
            "AUGMENTATION_NOISE_DIR=/tmp/noise clips",
        )
    )

    assert "--no-augmentation" in command
    assert '--augmentation-preset "light"' in command
    assert '--regular-negative-preset "none"' in command
    assert "--spectrogram-augmentation" in command
    assert '--augmentation-noise-dir "/tmp/noise clips"' in command
