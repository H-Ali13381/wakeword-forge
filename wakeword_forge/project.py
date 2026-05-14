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
    sample_review_approved: bool
    generated_review_approved: bool
    quality_check_passed: bool
    model_accepted: bool
    quality_positive_hits: int
    quality_positive_trials: int
    quality_false_triggers: int
    quality_score_min: float | None
    quality_score_max: float | None

    @property
    def total_positives(self) -> int:
        return self.real_positives + self.synthetic_positives

    @property
    def total_negatives(self) -> int:
        return self.negatives + self.confusable_negatives

    @property
    def generated_audio_count(self) -> int:
        return self.synthetic_positives + self.partial_negatives + self.confusable_negatives

    @property
    def positive_shortfall(self) -> int:
        return max(0, MIN_POSITIVES - self.total_positives)

    @property
    def negative_shortfall(self) -> int:
        return max(0, MIN_NEGATIVES - self.total_negatives)

    @property
    def samples_ready(self) -> bool:
        return bool(self.wake_phrase) and self.positive_shortfall == 0 and self.negative_shortfall == 0

    @property
    def sample_review_required(self) -> bool:
        return self.samples_ready and not self.sample_review_approved

    @property
    def generated_review_required(self) -> bool:
        return self.generated_audio_count > 0 and not self.generated_review_approved

    @property
    def ready_to_train(self) -> bool:
        return (
            self.samples_ready
            and self.sample_review_approved
            and not self.generated_review_required
        )

    @property
    def quality_check_required(self) -> bool:
        return self.has_model and not self.quality_check_passed

    @property
    def model_acceptance_required(self) -> bool:
        return self.has_model and self.quality_check_passed and not self.model_accepted

    @property
    def workflow_stage(self) -> str:
        if not self.samples_ready:
            return "samples needed"
        if self.sample_review_required:
            return "samples ready"
        if self.generated_review_required:
            return "generated review needed"
        if not self.has_model:
            return "sample review approved"
        if self.quality_check_required:
            return "trained"
        if self.model_acceptance_required:
            return "live quality check passed"
        return "model accepted"

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
        if self.sample_review_required:
            return "Review samples before training."
        if self.generated_review_required:
            return "Audit generated audio before training."
        if not self.has_model:
            return "Train the detector."
        if self.quality_check_required:
            return "Run the guided live quality check."
        if self.model_acceptance_required:
            return "Accept the model or collect more samples."
        return "Model accepted. Copy wakeword.onnx into your runtime."


def inspect_project(config: ForgeConfig) -> ProjectStatus:
    """Return a read-only status snapshot for a project config."""
    from .review import (
        generated_review_current,
        model_acceptance_current,
        quality_check_current,
        sample_review_current,
    )

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
        sample_review_approved=sample_review_current(config),
        generated_review_approved=generated_review_current(config),
        quality_check_passed=quality_check_current(config),
        model_accepted=model_acceptance_current(config),
        quality_positive_hits=config.quality_positive_hits,
        quality_positive_trials=config.quality_positive_trials,
        quality_false_triggers=config.quality_false_triggers,
        quality_score_min=config.quality_score_min,
        quality_score_max=config.quality_score_max,
    )
