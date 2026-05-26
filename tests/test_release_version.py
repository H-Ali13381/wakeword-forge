from __future__ import annotations

import re
import tomllib
from pathlib import Path

from typer.testing import CliRunner

import wakeword_forge
from wakeword_forge.cli import app

ROOT = Path(__file__).resolve().parents[1]
SEMVER_RE = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")


def test_release_version_is_semver_and_matches_project_metadata():
    project_metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project_metadata["project"]["version"]

    assert SEMVER_RE.fullmatch(version)
    assert wakeword_forge.__version__ == version


def test_cli_exposes_release_version():
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == f"wakeword-forge {wakeword_forge.__version__}"


def test_changelog_tracks_current_release_version():
    changelog = ROOT / "CHANGELOG.md"

    assert changelog.exists()
    text = changelog.read_text(encoding="utf-8")
    assert f"## [{wakeword_forge.__version__}]" in text
    assert (
        f"https://github.com/H-Ali13381/wakeword-forge/releases/tag/v{wakeword_forge.__version__}"
        in text
    )


def test_releasing_guide_documents_tagged_github_release_flow():
    releasing = ROOT / "RELEASING.md"

    assert releasing.exists()
    text = releasing.read_text(encoding="utf-8")
    assert "make release-check" in text
    assert f"git tag -a v{wakeword_forge.__version__}" in text
    assert f"gh release create v{wakeword_forge.__version__}" in text
