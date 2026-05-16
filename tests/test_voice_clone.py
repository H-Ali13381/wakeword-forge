import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from wakeword_forge.cli import app
from wakeword_forge.config import ForgeConfig, SAMPLE_RATE
from wakeword_forge.project import ensure_project_dirs, inspect_project
from wakeword_forge.voice_clone import (
    SourcePolicyError,
    apply_cloned_sample_decision,
    build_candidate_rows,
    build_qwentts_docker_run_command,
    generate_one_voice_clone_sample,
    list_cloned_review_items,
    load_source_manifest,
    select_reference_candidates,
    stage_cloned_sample_for_review,
    transcript_matches_phrase,
    validate_cloned_audio,
    write_one_sample_qwentts_job,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate)


def _load_qwentts_runner():
    path = Path("docker/qwentts/qwentts_clone_one.py")
    spec = importlib.util.spec_from_file_location("qwentts_clone_one_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_manifest_requires_youtube_opt_in_and_preserves_responsible_use_metadata(tmp_path):
    open_audio = tmp_path / "open.wav"
    youtube_audio = tmp_path / "yt.wav"
    open_audio.write_bytes(b"placeholder")
    youtube_audio.write_bytes(b"placeholder")
    manifest = tmp_path / "sources.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": str(open_audio),
                "speaker_id": "open-speaker",
                "source_type": "open_dataset",
                "license": "CC-BY-4.0",
                "usage_policy": "research_or_license_compatible_use_only",
            },
            {
                "path": str(youtube_audio),
                "speaker_id": "yt-speaker",
                "youtube_url": "https://youtu.be/example",
                "license": "source-rights-required",
                "usage_policy": "personal_use_only_unless_rights_are_cleared",
            },
        ],
    )

    with pytest.raises(SourcePolicyError, match="YouTube"):
        load_source_manifest(manifest, allow_youtube=False)

    rows = load_source_manifest(manifest, allow_youtube=True)

    assert [row["source_type"] for row in rows] == ["open_dataset", "youtube"]
    assert rows[0]["license"] == "CC-BY-4.0"
    assert rows[1]["usage_policy"] == "personal_use_only_unless_rights_are_cleared"
    assert rows[1]["youtube_url"] == "https://youtu.be/example"


def test_build_candidate_rows_transcribes_and_selects_one_best_single_speaker_reference(tmp_path):
    audio = tmp_path / "speaker.wav"
    audio.write_bytes(b"placeholder")
    source_rows = [
        {
            "path": str(audio),
            "speaker_id": "speaker-a",
            "source_id": "open_manifest",
            "dataset_id": "dataset-one",
            "source_type": "open_dataset",
            "license": "CC0-1.0",
            "usage_policy": "open_dataset_ok",
            "whisper_result": {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 4.0,
                        "text": "This is a clean single speaker reference sentence.",
                        "avg_logprob": -0.05,
                        "no_speech_prob": 0.01,
                    },
                    {
                        "start": 5.0,
                        "end": 8.0,
                        "text": "HOST: hello. GUEST: overlap from two people.",
                    },
                ]
            },
        },
        {
            "path": str(audio),
            "speaker_id": "speaker-a",
            "source_id": "open_manifest",
            "dataset_id": "dataset-one",
            "source_type": "open_dataset",
            "whisper_result": {
                "segments": [
                    {
                        "start": 9.0,
                        "end": 13.5,
                        "text": "This is the better reference sample with more reliable speech.",
                        "avg_logprob": -0.01,
                        "no_speech_prob": 0.0,
                    }
                ]
            },
        },
    ]

    candidates = build_candidate_rows(source_rows, min_duration_sec=1.0, max_duration_sec=8.0, min_words=5)
    selected = select_reference_candidates(candidates, max_speakers=5)

    assert len(candidates) == 2
    assert len(selected) == 1
    assert selected[0]["speaker_id"] == "speaker-a"
    assert selected[0]["reference_transcript"] == "This is the better reference sample with more reliable speech."
    assert selected[0]["quality_score"] > 0.7
    assert selected[0]["speaker_hash"]


def test_transcript_matching_accepts_contains_and_fuzzy_wakephrase_matches():
    exact = transcript_matches_phrase("The generated clip says: OK, Hermes!", ["OK Hermes"])
    fuzzy = transcript_matches_phrase("The generated clip says okay hermez now", ["OK Hermes"])
    miss = transcript_matches_phrase("The generated clip says hello there", ["OK Hermes"])
    substring_miss = transcript_matches_phrase("That joke landed well", ["ok"])
    short_fuzzy_miss = transcript_matches_phrase("they are ready", ["hey"])
    prefix_token_miss = transcript_matches_phrase("renova is nearby", ["nova"])
    embedded_multiword_miss = transcript_matches_phrase("awake word detector", ["wake word"])
    masked_short_token_miss = transcript_matches_phrase("the oak hermes clip", ["OK Hermes"])
    masked_token_miss = transcript_matches_phrase("the fake word detector", ["wake word"])
    split_token_miss = transcript_matches_phrase("ok harm is", ["OK Hermes"])

    assert exact.matched is True
    assert exact.method == "contains"
    assert fuzzy.matched is True
    assert fuzzy.method == "fuzzy"
    assert miss.matched is False
    assert substring_miss.matched is False
    assert short_fuzzy_miss.matched is False
    assert prefix_token_miss.matched is False
    assert embedded_multiword_miss.matched is False
    assert masked_short_token_miss.matched is False
    assert masked_token_miss.matched is False
    assert split_token_miss.matched is False


def test_validate_cloned_audio_rejects_silence_and_requires_wakephrase_transcript(tmp_path):
    good = tmp_path / "good.wav"
    silent = tmp_path / "silent.wav"
    t = np.linspace(0.0, 0.5, SAMPLE_RATE // 2, endpoint=False)
    _write_wav(good, 0.2 * np.sin(2 * np.pi * 440 * t))
    _write_wav(silent, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    good_result = validate_cloned_audio(good, ["OK Hermes"], transcript="okay hermes")
    silent_result = validate_cloned_audio(silent, ["OK Hermes"], transcript="okay hermes")
    mismatch_result = validate_cloned_audio(good, ["OK Hermes"], transcript="hello world")

    assert good_result.passed is True
    assert good_result.suggested_label == "positive"
    assert silent_result.passed is False
    assert "silence" in " ".join(silent_result.reasons).lower()
    assert mismatch_result.passed is False
    assert mismatch_result.suggested_label == "negative"


def test_staged_cloned_sample_requires_human_decision_before_entering_training_pool(tmp_path):
    config = ForgeConfig(wake_phrase="OK Hermes", project_dir=str(tmp_path))
    ensure_project_dirs(config)
    config.sample_review_approved = True
    config.sample_review_fingerprint = "stale-approved"
    generated = tmp_path / "generated.wav"
    _write_wav(generated, np.ones(SAMPLE_RATE // 4, dtype=np.float32) * 0.05)
    validation = validate_cloned_audio(generated, ["OK Hermes"], transcript="OK Hermes")

    staged = stage_cloned_sample_for_review(
        config,
        generated,
        validation=validation,
        metadata={
            "source_type": "open_dataset",
            "license": "CC0-1.0",
            "review_status": "positive",
            "suggested_label": "negative",
        },
    )

    assert staged.audio_path.exists()
    assert staged.metadata["review_status"] == "pending"
    assert staged.metadata["suggested_label"] == "positive"
    assert inspect_project(config).real_positives == 0
    assert len(list_cloned_review_items(config)) == 1

    accepted = apply_cloned_sample_decision(config, staged.audio_path, "positive")

    assert accepted is not None
    assert accepted.parent == config.positives_path
    assert accepted.exists()
    assert not staged.audio_path.exists()
    assert not staged.metadata_path.exists()
    assert inspect_project(config).real_positives == 1
    assert config.sample_review_approved is False


def test_staged_cloned_sample_negative_and_unusable_decisions(tmp_path):
    config = ForgeConfig(wake_phrase="OK Hermes", project_dir=str(tmp_path))
    ensure_project_dirs(config)
    generated = tmp_path / "generated.wav"
    _write_wav(generated, np.ones(SAMPLE_RATE // 4, dtype=np.float32) * 0.05)
    validation = validate_cloned_audio(generated, ["OK Hermes"], transcript="hello world")

    staged_negative = stage_cloned_sample_for_review(config, generated, validation=validation, metadata={})
    negative_path = apply_cloned_sample_decision(config, staged_negative.audio_path, "negative")
    assert negative_path is not None
    assert negative_path.parent == config.negatives_path
    assert inspect_project(config).negatives == 1

    staged_unusable = stage_cloned_sample_for_review(config, negative_path, validation=validation, metadata={})
    unusable_path = apply_cloned_sample_decision(config, staged_unusable.audio_path, "unusable")
    assert unusable_path is None
    assert not staged_unusable.audio_path.exists()
    assert not staged_unusable.metadata_path.exists()


def test_cli_exposes_voice_clone_commands_with_responsible_use_language():
    runner = CliRunner()

    generate_help = runner.invoke(app, ["voice-clone-one", "--help"])
    review_help = runner.invoke(app, ["review-cloned-samples", "--help"])

    assert generate_help.exit_code == 0
    assert "--source-manifest" in generate_help.output
    assert "--allow-youtube" in generate_help.output
    assert "responsible" in generate_help.output.lower()
    assert "one sample" in generate_help.output.lower()
    assert review_help.exit_code == 0
    assert "positive" in review_help.output
    assert "negative" in review_help.output
    assert "unusable" in review_help.output


def test_readme_and_makefile_document_qwentts_voice_clone_flow():
    readme = Path("README.md").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")
    provenance = Path("DATA_PROVENANCE.md").read_text(encoding="utf-8")

    assert "voice-clone-one" in readme
    assert "review-cloned-samples" in readme
    assert "responsible" in readme.lower()
    assert "fair use" in readme.lower()
    assert "qwentts-voice-clone-one" in makefile
    assert "review-cloned-samples" in makefile
    assert "YouTube" in provenance
    assert "consent" in provenance.lower()


def test_qwentts_docker_runner_script_is_present_and_job_driven():
    runner = Path("docker/qwentts/qwentts_clone_one.py")
    dockerfile = Path("docker/qwentts/Dockerfile")

    compose = Path("docker/qwentts/docker-compose.yml")

    assert runner.exists()
    assert dockerfile.exists()
    assert compose.exists()
    text = runner.read_text(encoding="utf-8")
    assert "--job" in text
    assert "generate_voice_clone" in text or "clone_voice" in text or "generate_custom_voice" in text
    assert "manifest" not in text.lower()


def test_generate_one_voice_clone_sample_stops_after_first_suitable_source(tmp_path):
    config = ForgeConfig(wake_phrase="OK Hermes", project_dir=str(tmp_path / "project"))
    ensure_project_dirs(config)
    source_audio = tmp_path / "source.wav"
    _write_wav(source_audio, np.ones(SAMPLE_RATE * 4, dtype=np.float32) * 0.05)
    manifest = tmp_path / "sources.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": str(source_audio),
                "speaker_id": "first-speaker",
                "source_type": "open_dataset",
                "license": "CC0-1.0",
                "usage_policy": "open_dataset_ok",
                "whisper_result": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 3.0,
                            "text": "This is a suitable reference voice sample.",
                            "avg_logprob": -0.01,
                            "no_speech_prob": 0.0,
                        }
                    ]
                },
            },
            {"path": str(tmp_path / "missing.wav"), "speaker_id": "should-not-be-touched"},
        ],
    )

    def docker_runner(command: list[str]) -> None:
        job_path = Path(command[command.index("--job") + 1].replace("/jobs/", str((config.cache_path / "voice_clone" / "jobs")) + "/"))
        job = json.loads(job_path.read_text(encoding="utf-8"))
        output = Path(str(job["output_path"]).replace("/project/", str(config.project_path) + "/"))
        _write_wav(output, np.ones(SAMPLE_RATE // 2, dtype=np.float32) * 0.05)

    class FakeTranscriber:
        def transcribe(self, audio_path: Path) -> dict:
            return {"text": "OK Hermes"}

    result = generate_one_voice_clone_sample(
        config,
        source_manifest=manifest,
        docker_runner=docker_runner,
        transcriber=FakeTranscriber(),  # type: ignore[arg-type]
    )

    assert result.reference_candidate["speaker_id"] == "first-speaker"
    assert result.validation.suggested_label == "positive"
    assert result.staged_item.audio_path.exists()
    assert result.staged_item.metadata["review_status"] == "pending"


def test_generate_one_voice_clone_sample_validates_against_phrase_override(tmp_path):
    config = ForgeConfig(
        wake_phrase="OK Hermes",
        wake_phrases=["Hey Nova"],
        project_dir=str(tmp_path / "project"),
    )
    ensure_project_dirs(config)
    source_audio = tmp_path / "source.wav"
    _write_wav(source_audio, np.ones(SAMPLE_RATE * 4, dtype=np.float32) * 0.05)
    manifest = tmp_path / "sources.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": str(source_audio),
                "speaker_id": "override-speaker",
                "source_type": "open_dataset",
                "license": "CC0-1.0",
                "usage_policy": "open_dataset_ok",
                "whisper_result": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 3.0,
                            "text": "This is a suitable reference voice sample.",
                            "avg_logprob": -0.01,
                            "no_speech_prob": 0.0,
                        }
                    ]
                },
            }
        ],
    )

    def docker_runner(command: list[str]) -> None:
        job_arg = command[command.index("--job") + 1]
        jobs_dir = config.cache_path / "voice_clone" / "jobs"
        job_path = Path(job_arg.replace("/jobs/", str(jobs_dir) + "/"))
        job = json.loads(job_path.read_text(encoding="utf-8"))
        output = Path(str(job["output_path"]).replace("/project/", str(config.project_path) + "/"))
        _write_wav(output, np.ones(SAMPLE_RATE // 2, dtype=np.float32) * 0.05)

    class FakeTranscriber:
        def transcribe(self, audio_path: Path) -> dict:
            return {"text": "Custom Wake"}

    result = generate_one_voice_clone_sample(
        config,
        source_manifest=manifest,
        phrase="Custom Wake",
        docker_runner=docker_runner,
        transcriber=FakeTranscriber(),  # type: ignore[arg-type]
    )

    assert json.loads(result.job_file.read_text(encoding="utf-8"))["text"] == "Custom Wake"
    assert result.validation.suggested_label == "positive"
    assert result.validation.match.phrase == "Custom Wake"


def test_qwentts_docker_job_is_one_sample_at_a_time_and_uses_project_mounts(tmp_path):
    project_dir = tmp_path / "project"
    output_dir = project_dir / "samples" / "cloned_review"
    job_file = tmp_path / "jobs" / "job.json"

    write_one_sample_qwentts_job(
        job_file,
        text="OK Hermes",
        reference_audio=Path("/project/cache/reference.wav"),
        reference_text="This is the source voice.",
        output_path=Path("/project/samples/cloned_review/clone.wav"),
        language="English",
        metadata={"speaker_hash": "abc123", "source_type": "open_dataset"},
    )
    command = build_qwentts_docker_run_command(
        job_file=job_file,
        project_dir=project_dir,
        output_dir=output_dir,
        image="wakeword-forge-qwentts:test",
    )
    job = json.loads(job_file.read_text(encoding="utf-8"))

    assert job["text"] == "OK Hermes"
    assert job["reference_audio"] == "/project/cache/reference.wav"
    assert job["output_path"] == "/project/samples/cloned_review/clone.wav"
    assert job["metadata"]["source_type"] == "open_dataset"
    assert command[:3] == ["docker", "run", "--rm"]
    assert command.count("--job") == 1
    assert command[command.index("--job") + 1] == f"/jobs/{job_file.name}"
    assert "wakeword-forge-qwentts:test" in command
    assert all("manifest" not in part.lower() for part in command)


def test_qwentts_runner_calls_reference_audio_signature(monkeypatch):
    runner = _load_qwentts_runner()
    calls = []

    class FakeModel:
        def generate_voice_clone(self, text, language, ref_audio, ref_text):
            calls.append(
                {
                    "text": text,
                    "language": language,
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                }
            )
            return {"audio": np.ones(SAMPLE_RATE // 4, dtype=np.float32), "sample_rate": SAMPLE_RATE}

    monkeypatch.setattr(runner, "_load_model", lambda job: FakeModel())
    audio, sample_rate = runner._synthesize(
        {
            "text": "OK Hermes",
            "language": "English",
            "reference_audio": "/project/reference.wav",
            "reference_text": "This is the reference voice.",
            "output_path": "/project/output.wav",
        }
    )

    assert sample_rate == SAMPLE_RATE
    assert audio.size == SAMPLE_RATE // 4
    assert calls == [
        {
            "text": "OK Hermes",
            "language": "English",
            "ref_audio": "/project/reference.wav",
            "ref_text": "This is the reference voice.",
        }
    ]


def test_qwentts_runner_rejects_clone_api_without_reference_audio(monkeypatch):
    runner = _load_qwentts_runner()

    class TextOnlyCloneModel:
        def generate_voice_clone(self, text, language):
            return {
                "audio": np.ones(SAMPLE_RATE // 4, dtype=np.float32),
                "sample_rate": SAMPLE_RATE,
            }

    monkeypatch.setattr(runner, "_load_model", lambda job: TextOnlyCloneModel())

    with pytest.raises(RuntimeError, match="reference audio"):
        runner._synthesize(
            {
                "text": "OK Hermes",
                "language": "English",
                "reference_audio": "/project/reference.wav",
                "reference_text": "This is the reference voice.",
                "output_path": "/project/output.wav",
            }
        )


def test_qwentts_runner_rejects_speaker_only_custom_voice_api(monkeypatch):
    runner = _load_qwentts_runner()

    class SpeakerOnlyModel:
        def generate_custom_voice(self, text, language, speaker):
            return {
                "audio": np.ones(SAMPLE_RATE // 4, dtype=np.float32),
                "sample_rate": SAMPLE_RATE,
            }

    monkeypatch.setattr(runner, "_load_model", lambda job: SpeakerOnlyModel())

    with pytest.raises(RuntimeError, match="reference audio"):
        runner._synthesize(
            {
                "text": "OK Hermes",
                "language": "English",
                "reference_audio": "/project/reference.wav",
                "reference_text": "This is the reference voice.",
                "output_path": "/project/output.wav",
            }
        )
