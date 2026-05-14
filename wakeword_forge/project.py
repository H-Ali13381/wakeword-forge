"""Project-level helpers shared by the CLI and Streamlit dashboard."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import ForgeConfig, MIN_NEGATIVES, MIN_POSITIVES

CONFIG_FILENAME = "forge_config.json"
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg")


@dataclass(frozen=True)
class SampleImportResult:
    """Summary of copying existing audio into the positive sample set."""

    source_dir: Path
    imported_paths: tuple[Path, ...]
    available_count: int
    skipped_paths: tuple[Path, ...] = ()

    @property
    def imported_count(self) -> int:
        return len(self.imported_paths)


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


def _safe_project_target(project_root: Path, target: Path) -> Path:
    """Resolve a reset target and reject anything outside the project root."""

    project_root = project_root.expanduser().resolve()
    resolved = target.expanduser().resolve(strict=False)
    if resolved == project_root or not resolved.is_relative_to(project_root):
        raise ValueError(f"Refusing to reset unsafe path outside project artifacts: {resolved}")
    return resolved


def _existing_paths_under(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*")) + [path]
    return [path]


def reset_project(config: ForgeConfig) -> list[Path]:
    """Delete wakeword-forge artifacts for a project without deleting the project root.

    Removes the persisted config, sample tree, model outputs, cache tree, and generated
    confusable-phrase cache. Unrelated files in the project directory are preserved.
    Returns the concrete files/directories that existed and were removed.
    """

    project_root = config.project_path
    targets = [
        project_root / CONFIG_FILENAME,
        config.samples_path,
        config.output_path,
        config.cache_path,
        config.confusables_cache,
    ]

    removed: list[Path] = []
    seen: set[Path] = set()
    for raw_target in targets:
        target = _safe_project_target(project_root, raw_target)
        if target in seen or not target.exists():
            continue
        seen.add(target)
        removed.extend(_existing_paths_under(target))
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return removed


def count_wavs(directory: Path) -> int:
    """Count WAV files below ``directory`` without failing when it is absent."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.wav")))


def _audio_sample_files(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )
    )


def _next_numbered_sample_path(out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = out_dir / f"{prefix}_{index:04d}.wav"
        if not candidate.exists():
            return candidate
        index += 1


def _write_imported_positive_sample(source_path: Path, target_path: Path) -> None:
    import soundfile as sf

    from .augmentation import _load_wav
    from .config import SAMPLE_RATE

    wav = _load_wav(source_path, SAMPLE_RATE, trim_silence=True)
    audio = wav.squeeze(0).detach().cpu().numpy()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target_path), audio, SAMPLE_RATE, subtype="PCM_16")


def import_positive_samples(
    config: ForgeConfig,
    source_dir: Path | str,
    *,
    limit: int | None = None,
) -> SampleImportResult:
    """Copy existing wake-phrase clips into ``samples/positives`` as normalized WAV files."""

    source_path = Path(source_dir).expanduser()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Existing sample folder is not available: {source_path}")

    audio_files = _audio_sample_files(source_path)
    max_imports = len(audio_files) if limit is None else max(0, int(limit))
    imported: list[Path] = []
    skipped: list[Path] = []

    for source_path in audio_files:
        if len(imported) >= max_imports:
            break
        target_path = _next_numbered_sample_path(config.positives_path, "imported")
        try:
            _write_imported_positive_sample(source_path, target_path)
        except Exception:
            if target_path.exists():
                target_path.unlink()
            skipped.append(source_path)
            continue
        imported.append(target_path)

    if imported:
        from .review import reset_sample_dependent_approvals

        reset_sample_dependent_approvals(config)

    return SampleImportResult(
        source_dir=Path(source_dir).expanduser(),
        imported_paths=tuple(imported),
        available_count=len(audio_files),
        skipped_paths=tuple(skipped),
    )


@dataclass(frozen=True)
class ProjectStatus:
    """Dashboard-friendly summary of a wakeword-forge project."""

    project_dir: Path
    wake_phrase: str
    wake_phrases: tuple[str, ...]
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

    phrases = config.phrase_options
    primary_phrase = phrases[0] if phrases else ""
    return ProjectStatus(
        project_dir=config.project_path,
        wake_phrase=primary_phrase,
        wake_phrases=phrases,
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
