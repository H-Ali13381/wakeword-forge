"""Human-in-the-loop review and quality-check helpers."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import ForgeConfig

ObservationKind = Literal["positive", "near_miss", "silence"]


@dataclass(frozen=True)
class SampleInventory:
    """Sorted WAV inventory grouped by user-facing review category."""

    positives: list[Path]
    negatives: list[Path]
    synthetic: list[Path]
    partials: list[Path]
    confusables: list[Path]

    @property
    def generated(self) -> list[Path]:
        return [*self.synthetic, *self.partials, *self.confusables]

    @property
    def all_samples(self) -> list[Path]:
        return [*self.positives, *self.negatives, *self.generated]


@dataclass(frozen=True)
class QualityObservation:
    """One scored trial in the guided post-training protocol."""

    kind: ObservationKind
    score: float


@dataclass(frozen=True)
class QualityReport:
    """Summary of a guided live quality check."""

    threshold: float
    positive_hits: int
    positive_misses: int
    positive_trials: int
    false_triggers: int
    negative_trials: int
    score_min: float | None
    score_max: float | None

    @property
    def passed(self) -> bool:
        return self.positive_trials > 0 and self.positive_misses == 0 and self.false_triggers == 0


def _wav_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.wav"))


def sample_inventory(config: ForgeConfig) -> SampleInventory:
    """Return sorted WAV groups for review UI and CLI commands."""

    return SampleInventory(
        positives=_wav_files(config.positives_path),
        negatives=_wav_files(config.negatives_path),
        synthetic=_wav_files(config.synthetic_path),
        partials=_wav_files(config.partials_path),
        confusables=_wav_files(config.confusables_path),
    )


def select_generated_audit_samples(
    config: ForgeConfig,
    *,
    limit: int = 12,
    seed: int = 42,
) -> list[Path]:
    """Select a deterministic random subset of generated clips to audit."""

    candidates = sample_inventory(config).generated
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[: max(0, limit)]


def _fingerprint_paths(paths: list[Path], root: Path) -> str:
    """Fingerprint a reviewed file set by relative path, size, and mtime."""

    digest = hashlib.sha256()
    for path in sorted(paths):
        rel = str(path.relative_to(root) if path.is_relative_to(root) else path)
        stat = path.stat()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def sample_review_fingerprint(config: ForgeConfig) -> str:
    inventory = sample_inventory(config)
    return _fingerprint_paths([*inventory.positives, *inventory.negatives], config.project_path)


def generated_review_fingerprint(config: ForgeConfig) -> str:
    return _fingerprint_paths(sample_inventory(config).generated, config.project_path)


def sample_review_current(config: ForgeConfig) -> bool:
    return (
        config.sample_review_approved
        and config.sample_review_fingerprint == sample_review_fingerprint(config)
    )


def generated_review_current(config: ForgeConfig) -> bool:
    return (
        config.generated_review_approved
        and config.generated_review_fingerprint == generated_review_fingerprint(config)
    )


def model_fingerprint(path: Path) -> str:
    """Return a content fingerprint for a trained model artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_project_model(config: ForgeConfig) -> Path:
    return config.output_path / "wakeword.onnx"


def quality_check_current(config: ForgeConfig) -> bool:
    model_path = current_project_model(config)
    return (
        config.quality_check_passed
        and model_path.exists()
        and config.quality_checked_model_fingerprint == model_fingerprint(model_path)
    )


def model_acceptance_current(config: ForgeConfig) -> bool:
    model_path = current_project_model(config)
    return (
        config.model_accepted
        and quality_check_current(config)
        and config.accepted_model_fingerprint == model_fingerprint(model_path)
    )


def delete_samples(paths: list[Path]) -> list[Path]:
    """Delete selected sample files and return the files actually removed."""

    removed: list[Path] = []
    for path in paths:
        if path.exists() and path.suffix.lower() == ".wav":
            path.unlink()
            removed.append(path)
    return removed


def approve_sample_review(config: ForgeConfig) -> None:
    """Mark recorded/base samples as reviewed for this exact file set."""

    config.sample_review_approved = True
    config.sample_review_fingerprint = sample_review_fingerprint(config)
    config.model_accepted = False


def approve_generated_review(config: ForgeConfig) -> None:
    """Mark generated TTS/hard-negative samples as audited for this exact file set."""

    config.generated_review_approved = True
    config.generated_review_fingerprint = generated_review_fingerprint(config)
    config.model_accepted = False


def summarize_quality_observations(
    observations: list[QualityObservation],
    *,
    threshold: float,
) -> QualityReport:
    """Summarize positive/negative guided trial scores against a threshold."""

    positive_scores = [obs.score for obs in observations if obs.kind == "positive"]
    negative_scores = [obs.score for obs in observations if obs.kind in {"near_miss", "silence"}]
    all_scores = [obs.score for obs in observations]
    positive_hits = sum(score >= threshold for score in positive_scores)
    false_triggers = sum(score >= threshold for score in negative_scores)
    return QualityReport(
        threshold=threshold,
        positive_hits=positive_hits,
        positive_misses=len(positive_scores) - positive_hits,
        positive_trials=len(positive_scores),
        false_triggers=false_triggers,
        negative_trials=len(negative_scores),
        score_min=min(all_scores) if all_scores else None,
        score_max=max(all_scores) if all_scores else None,
    )


def record_quality_check(
    config: ForgeConfig,
    report: QualityReport,
    *,
    model_path: Path | None = None,
) -> None:
    """Persist guided quality-check results into the project config."""

    checked_model = model_path or current_project_model(config)
    checked_fingerprint = model_fingerprint(checked_model) if checked_model.exists() else ""
    config.quality_check_passed = report.passed
    config.model_accepted = False
    config.quality_checked_model_path = str(checked_model)
    config.quality_checked_model_fingerprint = checked_fingerprint
    config.accepted_model_fingerprint = ""
    config.quality_positive_hits = report.positive_hits
    config.quality_positive_trials = report.positive_trials
    config.quality_false_triggers = report.false_triggers
    config.quality_score_min = report.score_min
    config.quality_score_max = report.score_max


def accept_model(config: ForgeConfig) -> None:
    """Mark the current trained model as accepted after a passing quality check."""

    model_path = current_project_model(config)
    if not config.quality_check_passed:
        raise ValueError("Run and pass the guided quality check before accepting the model.")
    if not model_path.exists():
        raise ValueError("Cannot accept model: current model artifact is missing.")
    fingerprint = model_fingerprint(model_path)
    if config.quality_checked_model_fingerprint != fingerprint:
        raise ValueError("Run the guided quality check against the current model before accepting it.")
    config.model_accepted = True
    config.accepted_model_fingerprint = fingerprint


def reset_trained_output_approval(config: ForgeConfig) -> None:
    """Invalidate post-training approval whenever a new model is trained."""

    config.quality_check_passed = False
    config.model_accepted = False
    config.quality_checked_model_path = ""
    config.quality_checked_model_fingerprint = ""
    config.accepted_model_fingerprint = ""
    config.quality_positive_hits = 0
    config.quality_positive_trials = 0
    config.quality_false_triggers = 0
    config.quality_score_min = None
    config.quality_score_max = None
