from __future__ import annotations

from pathlib import Path

import pytest

from wakeword_forge.config import ForgeConfig, MIN_NEGATIVES, MIN_POSITIVES
from wakeword_forge.project import inspect_project
from wakeword_forge.review import (
    QualityObservation,
    accept_model,
    approve_generated_review,
    approve_sample_review,
    record_quality_check,
    sample_inventory,
    select_generated_audit_samples,
    summarize_quality_observations,
)


def _touch_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")


def _ready_config(tmp_path: Path) -> ForgeConfig:
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    for i in range(MIN_POSITIVES):
        _touch_wav(cfg.positives_path / f"take_{i:03d}.wav")
    for i in range(MIN_NEGATIVES):
        _touch_wav(cfg.negatives_path / f"neg_{i:03d}.wav")
    return cfg


def test_training_requires_explicit_sample_review_after_counts_are_ready(tmp_path):
    cfg = _ready_config(tmp_path)

    status = inspect_project(cfg)

    assert status.samples_ready is True
    assert status.sample_review_required is True
    assert status.ready_to_train is False
    assert status.workflow_stage == "samples ready"
    assert status.next_action == "Review samples before training."

    approve_sample_review(cfg)
    status = inspect_project(cfg)

    assert status.sample_review_approved is True
    assert status.ready_to_train is True
    assert status.workflow_stage == "sample review approved"
    assert status.next_action == "Train the detector."


def test_generated_audio_requires_audit_before_training(tmp_path):
    cfg = _ready_config(tmp_path)
    _touch_wav(cfg.synthetic_path / "synth_000.wav")
    _touch_wav(cfg.partials_path / "partial_000.wav")
    approve_sample_review(cfg)

    status = inspect_project(cfg)

    assert status.generated_audio_count == 2
    assert status.generated_review_required is True
    assert status.ready_to_train is False
    assert status.workflow_stage == "generated review needed"
    assert status.next_action == "Audit generated audio before training."

    approve_generated_review(cfg)
    status = inspect_project(cfg)

    assert status.generated_review_approved is True
    assert status.ready_to_train is True


def test_sample_inventory_and_generated_audit_selection_are_sorted_and_limited(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    _touch_wav(cfg.positives_path / "take_b.wav")
    _touch_wav(cfg.positives_path / "take_a.wav")
    _touch_wav(cfg.synthetic_path / "synth_b.wav")
    _touch_wav(cfg.synthetic_path / "synth_a.wav")
    _touch_wav(cfg.partials_path / "partial.wav")
    _touch_wav(cfg.confusables_path / "confusable.wav")

    inventory = sample_inventory(cfg)
    audit = select_generated_audit_samples(cfg, limit=3, seed=7)

    assert [p.name for p in inventory.positives] == ["take_a.wav", "take_b.wav"]
    assert [p.name for p in inventory.generated] == [
        "synth_a.wav",
        "synth_b.wav",
        "partial.wav",
        "confusable.wav",
    ]
    assert len(audit) == 3
    assert {p.parent.name for p in audit} <= {"synthetic", "partials", "confusables"}


def test_quality_observations_summarize_hits_misses_false_triggers_and_score_range():
    report = summarize_quality_observations(
        [
            QualityObservation(kind="positive", score=0.91),
            QualityObservation(kind="positive", score=0.42),
            QualityObservation(kind="near_miss", score=0.77),
            QualityObservation(kind="silence", score=0.12),
        ],
        threshold=0.70,
    )

    assert report.positive_hits == 1
    assert report.positive_misses == 1
    assert report.false_triggers == 1
    assert report.negative_trials == 2
    assert report.score_min == 0.12
    assert report.score_max == 0.91
    assert report.passed is False


def test_quality_checkpoint_and_model_acceptance_update_project_stage(tmp_path):
    cfg = _ready_config(tmp_path)
    approve_sample_review(cfg)
    _touch_wav(cfg.output_path / "wakeword.onnx")

    status = inspect_project(cfg)
    assert status.workflow_stage == "trained"
    assert status.next_action == "Run the guided live quality check."

    report = summarize_quality_observations(
        [
            QualityObservation(kind="positive", score=0.95),
            QualityObservation(kind="positive", score=0.90),
            QualityObservation(kind="near_miss", score=0.20),
            QualityObservation(kind="silence", score=0.10),
        ],
        threshold=0.70,
    )
    assert report.passed is True

    record_quality_check(cfg, report)
    status = inspect_project(cfg)
    assert status.workflow_stage == "live quality check passed"
    assert status.next_action == "Accept the model or collect more samples."

    accept_model(cfg)
    status = inspect_project(cfg)
    assert status.model_accepted is True
    assert status.workflow_stage == "model accepted"
    assert status.next_action == "Model accepted. Copy wakeword.onnx into your runtime."


def test_model_acceptance_requires_passing_quality_check(tmp_path):
    cfg = _ready_config(tmp_path)

    with pytest.raises(ValueError, match="quality check"):
        accept_model(cfg)


def test_sample_review_is_invalidated_when_reviewed_files_change(tmp_path):
    cfg = _ready_config(tmp_path)
    approve_sample_review(cfg)
    assert inspect_project(cfg).ready_to_train is True

    _touch_wav(cfg.positives_path / "take_new.wav")
    status = inspect_project(cfg)

    assert cfg.sample_review_approved is True
    assert status.sample_review_approved is False
    assert status.sample_review_required is True
    assert status.ready_to_train is False
    assert status.next_action == "Review samples before training."


def test_generated_review_is_invalidated_when_generated_files_change(tmp_path):
    cfg = _ready_config(tmp_path)
    approve_sample_review(cfg)
    approve_generated_review(cfg)
    assert inspect_project(cfg).ready_to_train is True

    _touch_wav(cfg.synthetic_path / "synth_new.wav")
    status = inspect_project(cfg)

    assert cfg.generated_review_approved is True
    assert status.generated_review_approved is False
    assert status.generated_review_required is True
    assert status.ready_to_train is False
    assert status.next_action == "Audit generated audio before training."


def test_model_acceptance_requires_current_project_model(tmp_path):
    cfg = _ready_config(tmp_path)
    approve_sample_review(cfg)
    model_path = cfg.output_path / "wakeword.onnx"
    _touch_wav(model_path)
    report = summarize_quality_observations(
        [QualityObservation(kind="positive", score=0.9)],
        threshold=0.7,
    )
    record_quality_check(cfg, report, model_path=model_path)
    accept_model(cfg)
    assert inspect_project(cfg).model_accepted is True

    model_path.write_bytes(b"replacement model")
    status = inspect_project(cfg)

    assert cfg.model_accepted is True
    assert status.quality_check_passed is False
    assert status.model_accepted is False
    with pytest.raises(ValueError, match="current model"):
        accept_model(cfg)


def test_quality_check_on_alternate_model_cannot_accept_project_model(tmp_path):
    cfg = _ready_config(tmp_path)
    _touch_wav(cfg.output_path / "wakeword.onnx")
    alternate_model = tmp_path / "scratch.onnx"
    alternate_model.write_bytes(b"other model")
    report = summarize_quality_observations(
        [QualityObservation(kind="positive", score=0.9)],
        threshold=0.7,
    )

    record_quality_check(cfg, report, model_path=alternate_model)

    with pytest.raises(ValueError, match="current model"):
        accept_model(cfg)
