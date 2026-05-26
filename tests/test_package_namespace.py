from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_internal_package_namespace_is_forge() -> None:
    forge = importlib.import_module("forge")

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert forge.__version__ == metadata["project"]["version"]
    assert metadata["project"]["scripts"] == {
        "wakeword-forge": "forge.cli:app",
        "wakeword-forge-dashboard": "forge.dashboard:main",
    }
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["forge"]
    assert (ROOT / "forge" / "__init__.py").is_file()
    assert not (ROOT / "wakeword_forge").exists()


def test_readme_distinguishes_repo_source_package_and_workspace() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "repo and CLI are named `wakeword-forge`" in readme
    assert "source code lives in `forge/`" in readme
    assert "local training workspace" in readme
