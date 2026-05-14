"""Project-level helpers shared by the CLI and Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ForgeConfig, MIN_NEGATIVES, MIN_POSITIVES

CONFIG_FILENAME = "forge_config.json"


def load_or_create_config(project_dir: Path | str) -> ForgeConfig:
    """Load ``forge_config.json`` from a project directory, or return defaults."""
    project_path = Path(project_dir).expanduser()
    cfg_path = project_path / CONFIG_FILENAME
    if cfg_path.exists():
        return ForgeConfig.load(cfg_path)
    return ForgeConfig(project_dir=str(project_path))


def save_config(config: ForgeConfig) -> Path:
    """Persist a config next to the project data and return the written path."""
    path = Path(config.project_dir).expanduser() / CONFIG_FILENAME
    config.save(path)
    return path


def ensure_project_dirs(config: ForgeConfig) -> None:
    """Create the project folders used by the guided dashboard and CLI flow."""
    for directory in (
        config.positives_path,
        config.negatives_path,
        config.synthetic_path,
        config.partials_path,
        config.confusables_path,
        config.output_path,
        config.cache_path,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def count_wavs(directory: Path) -> int:
    """Count WAV files below ``directory`` without failing when it is absent."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.wav")))


@dataclass(frozen=True)
class ProjectStatus:
    """Dashboard-friendly summary of a wakeword-forge project."""

    project_dir: Path
    wake_phrase: str
    real_positives: int
    synthetic_positives: int
    negatives: int
    partial_negatives: int
    confusable_negatives: int
    has_model: bool
    trained_eer: float | None
    trained_threshold: float

    @property
    def total_positives(self) -> int:
        return self.real_positives + self.synthetic_positives

    @property
    def total_negatives(self) -> int:
        return self.negatives + self.confusable_negatives

    @property
    def positive_shortfall(self) -> int:
        return max(0, MIN_POSITIVES - self.total_positives)

    @property
    def negative_shortfall(self) -> int:
        return max(0, MIN_NEGATIVES - self.total_negatives)

    @property
    def ready_to_train(self) -> bool:
        return bool(self.wake_phrase) and self.positive_shortfall == 0 and self.negative_shortfall == 0

    @property
    def progress_fraction(self) -> float:
        done = min(self.total_positives, MIN_POSITIVES) + min(self.total_negatives, MIN_NEGATIVES)
        return done / (MIN_POSITIVES + MIN_NEGATIVES)

    @property
    def next_action(self) -> str:
        if not self.wake_phrase:
            return "Choose a wake phrase."
        if self.positive_shortfall:
            return (
                "Record or synthesize at least "
                f"{self.positive_shortfall} more wake-phrase examples."
            )
        if self.negative_shortfall:
            return f"Record or generate at least {self.negative_shortfall} more negative examples."
        if not self.has_model:
            return "Train the detector."
        return "Try a live microphone test."


def inspect_project(config: ForgeConfig) -> ProjectStatus:
    """Return a read-only status snapshot for a project config."""
    return ProjectStatus(
        project_dir=config.project_path,
        wake_phrase=config.wake_phrase,
        real_positives=count_wavs(config.positives_path),
        synthetic_positives=count_wavs(config.synthetic_path),
        negatives=count_wavs(config.negatives_path),
        partial_negatives=count_wavs(config.partials_path),
        confusable_negatives=count_wavs(config.confusables_path),
        has_model=(config.output_path / "wakeword.onnx").exists(),
        trained_eer=config.trained_eer,
        trained_threshold=config.trained_threshold,
    )
