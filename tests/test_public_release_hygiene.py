from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines() if line]


def test_public_release_tree_has_no_bundled_runtime_artifacts():
    forbidden_suffixes = {
        ".ckpt",
        ".db",
        ".flac",
        ".mp3",
        ".ogg",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
        ".sqlite",
        ".sqlite3",
        ".wav",
    }
    forbidden_names = {".env", "credentials.json", "forge_config.json", "secrets.json"}
    forbidden_dirs = {
        "benchmark_results",
        "legacy",
        "mobile_review_imports",
        "mobile_review_packs",
        "output",
        "processing_logs",
        "projects",
        "samples",
        "wakeword_projects",
    }
    hits: list[str] = []

    for path in _tracked_files():
        rel = path.relative_to(ROOT)
        parts = set(rel.parts)
        if path.suffix.lower() in forbidden_suffixes:
            hits.append(str(rel))
        if path.name in forbidden_names:
            hits.append(str(rel))
        if parts & forbidden_dirs:
            hits.append(str(rel))

    assert hits == []


def test_public_docs_use_exported_runtime_metadata_filename():
    """Docs should match the sidecar written next to output/wakeword.onnx."""

    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "advanced-usage.md",
        ROOT / "DATA_PROVENANCE.md",
        ROOT / "SECURITY.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "architecture.mmd",
    ]
    stale_mentions = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        stale_runtime_names = (
            "output/config.json",
            "config.json • threshold • metadata",
            "`config.json`",
        )
        if any(name in text for name in stale_runtime_names):
            stale_mentions.append(str(path.relative_to(ROOT)))

    assert stale_mentions == []
    assert "output/wakeword.json" in (ROOT / "README.md").read_text(encoding="utf-8")
