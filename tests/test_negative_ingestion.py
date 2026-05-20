from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from wakeword_forge.cli import app
from wakeword_forge.config import SAMPLE_RATE
from wakeword_forge.negative_ingestion import import_negative_audio
from wakeword_forge.project import inspect_project
from wakeword_forge.config import ForgeConfig


def _write_wav(path: Path, *, sample_rate: int = SAMPLE_RATE, seconds: float = 1.0, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    mono = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.stack([mono, mono * 0.5], axis=1) if channels == 2 else mono
    sf.write(path, audio, sample_rate)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_import_negative_manifest_chunks_normalizes_caps_and_preserves_provenance(tmp_path):
    project_dir = tmp_path / "project"
    source_dir = tmp_path / "dataset"
    audio_a = source_dir / "speaker_a.wav"
    audio_b = source_dir / "speaker_b.wav"
    _write_wav(audio_a, sample_rate=8_000, seconds=2.4, channels=2)
    _write_wav(audio_b, sample_rate=16_000, seconds=2.4)
    manifest = tmp_path / "manifests" / "external_negatives.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": str(audio_a),
                "label": "neg",
                "source_dataset": "toy_speech",
                "category": "speech",
                "license": "CC0-1.0",
                "transcript_filter_required": True,
                "transcript_exclusion_terms": ["hey nova"],
                "transcript": "ordinary background speech",
            },
            {
                "path": str(audio_b),
                "label": "neg",
                "source_dataset": "toy_speech",
                "category": "speech",
                "license": "CC0-1.0",
            },
        ],
    )
    cfg = ForgeConfig(project_dir=str(project_dir), wake_phrase="Hey Nova")

    result = import_negative_audio(
        cfg,
        manifest=manifest,
        kind="background",
        chunk_duration=1.0,
        min_chunk_duration=0.5,
        max_chunks_per_file=1,
        limit_per_source=1,
        limit=10,
    )

    assert result.available_count == 2
    assert result.imported_count == 1
    imported = result.imported_paths[0]
    assert imported.parent == cfg.negatives_path
    info = sf.info(imported)
    assert info.samplerate == SAMPLE_RATE
    assert round(info.duration, 2) == 1.0

    sidecar = imported.with_suffix(imported.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["source_dataset"] == "toy_speech"
    assert metadata["category"] == "speech"
    assert metadata["license"] == "CC0-1.0"
    assert metadata["target_kind"] == "background"
    assert metadata["chunk_index"] == 0
    assert metadata["source_manifest"] == str(manifest)
    assert result.manifest_path == cfg.negatives_path / "negative_import_manifest.jsonl"
    assert result.manifest_path.exists()
    assert inspect_project(cfg).negatives == 1


def test_import_negative_manifest_stratifies_by_category_quotas_independent_of_manifest_order(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path / "project"), wake_phrase="Hey Nova")
    source_dir = tmp_path / "dataset"
    rows = []
    for category, count in (("speech", 5), ("noise", 3), ("silence", 2), ("music", 2)):
        for index in range(count):
            audio = source_dir / category / f"{category}_{index}.wav"
            _write_wav(audio, seconds=1.0)
            rows.append(
                {
                    "path": str(audio),
                    "label": "neg",
                    "source_dataset": f"toy_{category}",
                    "category": category,
                    "license": "CC0-1.0",
                }
            )
    manifest = tmp_path / "external_negatives.jsonl"
    _write_jsonl(manifest, rows)

    result = import_negative_audio(
        cfg,
        manifest=manifest,
        kind="background",
        chunk_duration=1.0,
        max_chunks_per_file=1,
        limit=6,
        strata={"speech": 2, "noise": 2, "silence": 2},
    )

    assert result.imported_count == 6
    categories = [
        json.loads(path.with_suffix(".wav.json").read_text(encoding="utf-8"))["category"]
        for path in result.imported_paths
    ]
    assert categories.count("speech") == 2
    assert categories.count("noise") == 2
    assert categories.count("silence") == 2
    assert "music" not in categories
    assert result.strata_counts == {"speech": 2, "noise": 2, "silence": 2}


def test_import_negative_manifest_filters_positive_labels_and_transcript_contamination(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path / "project"), wake_phrase="Hey Nova")
    keep = tmp_path / "keep.wav"
    contaminated = tmp_path / "contaminated.wav"
    positive = tmp_path / "positive.wav"
    for path in (keep, contaminated, positive):
        _write_wav(path, seconds=1.0)
    manifest = tmp_path / "external_negatives.jsonl"
    _write_jsonl(
        manifest,
        [
            {"path": str(contaminated), "label": "neg", "source_dataset": "speech", "transcript_exclusion_terms": ["hey nova"], "transcript": "someone says Hey Nova here"},
            {"path": str(positive), "label": "pos", "source_dataset": "speech"},
            {"path": str(keep), "label": "neg", "source_dataset": "speech", "transcript": "harmless conversation"},
        ],
    )

    result = import_negative_audio(cfg, manifest=manifest, kind="background", chunk_duration=1.0)

    assert result.imported_count == 1
    assert len(result.skipped_paths) == 2
    metadata = json.loads(result.imported_paths[0].with_suffix(".wav.json").read_text(encoding="utf-8"))
    assert metadata["source_path"] == str(keep)


def test_import_negative_source_directory_can_target_partials(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path / "project"), wake_phrase="Hey Nova")
    source = tmp_path / "partial_source"
    _write_wav(source / "okay_only.flac", seconds=0.75)

    result = import_negative_audio(cfg, source_dir=source, kind="partial", chunk_duration=1.0)

    assert result.imported_count == 1
    assert result.imported_paths[0].parent == cfg.partials_path
    assert inspect_project(cfg).partial_negatives == 1
    metadata = json.loads(result.imported_paths[0].with_suffix(".wav.json").read_text(encoding="utf-8"))
    assert metadata["target_kind"] == "partial"
    assert metadata["source_dataset"] == source.name


def test_cli_import_negatives_applies_strata_quotas(tmp_path):
    runner = CliRunner()
    project_dir = tmp_path / "project"
    source_dir = tmp_path / "dataset"
    rows = []
    for category, count in (("speech", 3), ("noise", 2), ("silence", 2)):
        for index in range(count):
            audio = source_dir / category / f"{category}_{index}.wav"
            _write_wav(audio, seconds=1.0)
            rows.append(
                {
                    "path": str(audio),
                    "label": "neg",
                    "source_dataset": f"toy_{category}",
                    "category": category,
                    "license": "CC0-1.0",
                }
            )
    manifest = tmp_path / "external_negatives.jsonl"
    _write_jsonl(manifest, rows)

    result = runner.invoke(
        app,
        [
            "import-negatives",
            "--dir",
            str(project_dir),
            "--manifest",
            str(manifest),
            "--kind",
            "background",
            "--chunk-duration",
            "1.0",
            "--max-chunks-per-file",
            "1",
            "--limit",
            "3",
            "--strata",
            "speech=1,noise=1,silence=1",
        ],
    )

    assert result.exit_code == 0, result.output
    metadata = [
        json.loads(path.with_suffix(".wav.json").read_text(encoding="utf-8"))
        for path in sorted((project_dir / "samples" / "negatives").glob("*.wav"))
    ]
    assert [row["category"] for row in metadata].count("speech") == 1
    assert [row["category"] for row in metadata].count("noise") == 1
    assert [row["category"] for row in metadata].count("silence") == 1
    assert all(row["stratify_by"] == "category" for row in metadata)


def test_cli_documents_import_negatives_workflow():
    runner = CliRunner()

    result = runner.invoke(app, ["import-negatives", "--help"])

    assert result.exit_code == 0
    assert "--source-dir" in result.output
    assert "--manifest" in result.output
    assert "--kind" in result.output
    assert "--limit-per-source" in result.output
    assert "--strata" in result.output
    assert "--stratify-by" in result.output
    assert "negative" in result.output.lower()


def test_documentation_and_makefile_document_negative_ingestion():
    docs = Path("README.md").read_text(encoding="utf-8") + Path("docs/advanced-usage.md").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")
    provenance = Path("DATA_PROVENANCE.md").read_text(encoding="utf-8")

    assert "import-negatives" in docs
    assert "make import-negatives" in docs
    assert "manifest" in docs.lower()
    assert "limit-per-source" in docs
    assert "--strata" in docs
    assert "speech=50,noise=50,silence=50" in docs
    assert "import-negatives" in makefile
    assert "NEG_STRATA" in makefile
    assert "--strata" in makefile
    assert "negative_import_manifest.jsonl" in provenance
