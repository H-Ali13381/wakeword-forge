from __future__ import annotations

import tomllib
from pathlib import Path

from wakeword_forge.config import ForgeConfig, MIN_NEGATIVES, MIN_POSITIVES
import wakeword_forge.dashboard as dashboard
from wakeword_forge.dashboard import make_command
from wakeword_forge.project import inspect_project


def _touch_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")


def test_project_status_counts_samples_and_reports_next_action(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        record_positives=MIN_POSITIVES,
        record_negatives=MIN_NEGATIVES,
        tts_variants=2,
    )
    _touch_wav(cfg.positives_path / "take_001.wav")
    _touch_wav(cfg.synthetic_path / "synth_001.wav")
    _touch_wav(cfg.negatives_path / "neg_001.wav")

    status = inspect_project(cfg)

    assert status.real_positives == 1
    assert status.synthetic_positives == 1
    assert status.total_positives == 2
    assert status.negatives == 1
    assert status.ready_to_train is False
    assert status.next_action == f"Record or synthesize at least {MIN_POSITIVES - 2} more wake-phrase examples."
    assert status.progress_fraction == 3 / (MIN_POSITIVES + MIN_NEGATIVES)


def test_project_status_marks_training_ready_when_minimums_are_met(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Computer",
        project_dir=str(tmp_path),
        record_positives=MIN_POSITIVES,
        record_negatives=MIN_NEGATIVES,
    )
    for i in range(MIN_POSITIVES):
        _touch_wav(cfg.positives_path / f"take_{i}.wav")
    for i in range(MIN_NEGATIVES):
        _touch_wav(cfg.negatives_path / f"neg_{i}.wav")

    status = inspect_project(cfg)

    assert status.samples_ready is True
    assert status.ready_to_train is False
    assert status.next_action == "Review samples before training."
    assert status.progress_fraction == 1.0


def test_make_command_outputs_copy_pasteable_cli_fallbacks(tmp_path):
    project_dir = tmp_path / "wakeword demo"

    assert make_command("dashboard", project_dir) == f"make dashboard DIR='{project_dir}'"
    assert make_command("info", project_dir) == f"make info DIR='{project_dir}'"
    assert (
        make_command("record", project_dir, phrase="Hey Nova", n=5)
        == f"make record DIR='{project_dir}' PHRASE='Hey Nova' N=5"
    )
    assert (
        make_command("synth", project_dir, phrase="Hey Nova", n=12, engine="kokoro")
        == f"make synth DIR='{project_dir}' PHRASE='Hey Nova' N=12 ENGINE=kokoro"
    )


def test_dashboard_main_uses_dir_arg_when_streamlit_is_already_running(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(dashboard, "_running_inside_streamlit", lambda: True)
    monkeypatch.setattr(dashboard, "run_app", lambda project_dir: calls.append(project_dir))

    dashboard.main(["--dir", str(tmp_path)])

    assert calls == [str(tmp_path)]


def test_sidebar_config_preserves_review_fingerprints(tmp_path):
    class FakeSidebar:
        def header(self, *_args, **_kwargs):
            pass

        def subheader(self, *_args, **_kwargs):
            pass

        def text_input(self, _label, value):
            return value

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def toggle(self, _label, value):
            return value

        def selectbox(self, _label, *, options, index):
            return options[index]

        def button(self, *_args, **_kwargs):
            return False

    class FakeSt:
        sidebar = FakeSidebar()

    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        sample_review_fingerprint="samples123",
        generated_review_fingerprint="generated123",
        quality_checked_model_path="/tmp/model.onnx",
        quality_checked_model_fingerprint="quality123",
        accepted_model_fingerprint="accepted123",
    )

    updated = dashboard._sidebar_config(FakeSt(), cfg)

    assert updated.sample_review_fingerprint == "samples123"
    assert updated.generated_review_fingerprint == "generated123"
    assert updated.quality_checked_model_path == "/tmp/model.onnx"
    assert updated.quality_checked_model_fingerprint == "quality123"
    assert updated.accepted_model_fingerprint == "accepted123"


def test_makefile_defaults_to_dashboard_with_cli_fallback():
    makefile = Path("Makefile").read_text()

    assert "dashboard" in makefile
    assert "cli-run" in makefile
    assert "start: dashboard" in makefile
    assert "$(FORGE) dashboard --dir \"$(DIR)\"" in makefile
    assert "$(FORGE) run --dir \"$(DIR)\"" in makefile
    assert "review-samples" in makefile
    assert "audit-generated" in makefile
    assert "quality-check" in makefile
    assert "accept-model" in makefile


def test_pyproject_declares_streamlit_ui_extra_and_dashboard_script():
    data = tomllib.loads(Path("pyproject.toml").read_text())

    optional = data["project"]["optional-dependencies"]
    assert any(dep.startswith("streamlit>=") for dep in optional["ui"])
    assert "wakeword-forge-dashboard" in data["project"]["scripts"]
    assert data["project"]["scripts"]["wakeword-forge-dashboard"] == "wakeword_forge.dashboard:main"
