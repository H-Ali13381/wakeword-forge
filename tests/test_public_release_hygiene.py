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
    forbidden_names = {".env", "credentials.json", "secrets.json"}
    forbidden_dirs = {"benchmark_results", "output", "samples"}
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
