from __future__ import annotations

import runpy
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


def test_dashboard_script_loads_when_executed_by_streamlit_runner():
    namespace = runpy.run_path("wakeword_forge/dashboard.py")

    assert namespace["DEFAULT_PROJECT_DIR"].name == "wakeword_forge_project"


def test_dashboard_main_uses_dir_arg_when_streamlit_is_already_running(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(dashboard, "_running_inside_streamlit", lambda: True)
    monkeypatch.setattr(dashboard, "run_app", lambda project_dir: calls.append(project_dir))

    dashboard.main(["--dir", str(tmp_path)])

    assert calls == [str(tmp_path)]


def test_workspace_step_shows_only_workspace_fields_and_preserves_review_fingerprints(tmp_path):
    class ForbiddenSidebar:
        def __getattr__(self, name):
            raise AssertionError(f"wizard settings should not render in sidebar: {name}")

    class FakeSt:
        sidebar = ForbiddenSidebar()

        def __init__(self):
            self.text_labels: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def text_input(self, label, value):
            self.text_labels.append(label)
            return value

        def text_area(self, label, value, **_kwargs):
            self.text_labels.append(label)
            return "Okay Hermes\nHey Hermes"

        def number_input(self, *_args, **_kwargs):
            raise AssertionError("recording parameters leaked into workspace step")

        def toggle(self, *_args, **_kwargs):
            raise AssertionError("augmentation parameters leaked into workspace step")

        def selectbox(self, *_args, **_kwargs):
            raise AssertionError("augmentation parameters leaked into workspace step")

        def button(self, *_args, **_kwargs):
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        sample_review_fingerprint="samples123",
        generated_review_fingerprint="generated123",
        quality_checked_model_path="/tmp/model.onnx",
        quality_checked_model_fingerprint="quality123",
        accepted_model_fingerprint="accepted123",
    )
    fake = FakeSt()

    updated = dashboard._render_workspace_step(fake, cfg)

    assert fake.text_labels == ["Project directory", "Wake phrases (one per line)"]
    assert updated.wake_phrase == "Okay Hermes"
    assert updated.wake_phrases == ["Okay Hermes", "Hey Hermes"]
    assert updated.sample_review_fingerprint == "samples123"
    assert updated.generated_review_fingerprint == "generated123"
    assert updated.quality_checked_model_path == "/tmp/model.onnx"
    assert updated.quality_checked_model_fingerprint == "quality123"
    assert updated.accepted_model_fingerprint == "accepted123"


def test_recording_step_shows_only_recording_parameters(tmp_path):
    class FakeSt:
        def __init__(self):
            self.number_labels: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def number_input(self, label, **kwargs):
            self.number_labels.append(label)
            return kwargs["value"]

        def text_input(self, *_args, **_kwargs):
            raise AssertionError("workspace parameters leaked into recording step")

        def toggle(self, *_args, **_kwargs):
            raise AssertionError("augmentation parameters leaked into recording step")

        def selectbox(self, *_args, **_kwargs):
            raise AssertionError("augmentation parameters leaked into recording step")

        def button(self, *_args, **_kwargs):
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    dashboard._render_recording_step(fake, cfg)

    assert fake.number_labels == [
        "Target positive recordings",
        "Target negative recordings",
        "Seconds per take",
    ]


class CaptureFakeSt:
    def __init__(self, pressed: set[str] | None = None):
        self.pressed = pressed or set()
        self.session_state: dict[str, object] = {}
        self.buttons: list[str] = []
        self.button_kwargs: dict[str, dict] = {}
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.successes: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.codes: list[tuple[str, dict]] = []
        self.audios: list[tuple[object, dict]] = []

    def subheader(self, *_args, **_kwargs):
        pass

    def caption(self, text, **_kwargs):
        self.captions.append(str(text))

    def markdown(self, text, **_kwargs):
        self.markdowns.append(str(text))

    def success(self, text, **_kwargs):
        self.successes.append(str(text))

    def warning(self, text, **_kwargs):
        self.warnings.append(str(text))

    def error(self, text, **_kwargs):
        self.errors.append(str(text))

    def code(self, text, **kwargs):
        self.codes.append((str(text), kwargs))

    def audio(self, data, **kwargs):
        self.audios.append((data, kwargs))

    def button(self, label, **kwargs):
        label = str(label)
        self.buttons.append(label)
        self.button_kwargs[label] = kwargs
        return label in self.pressed

    def columns(self, count):
        return [self for _ in range(count)]

    def spinner(self, *_args, **_kwargs):
        return self

    def rerun(self):
        self.session_state["rerun_requested"] = True

    def exception(self, exc):
        self.errors.append(str(exc))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_default_wizard_starts_on_intro_until_user_begins(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path))
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    assert dashboard._default_wizard_step(status) == "intro"
    assert dashboard._current_wizard_step(fake, status) == "intro"

    fake.session_state[dashboard.DASHBOARD_STEP_KEY] = "workspace"
    assert dashboard._current_wizard_step(fake, status) == "workspace"


def test_intro_step_renders_title_card_and_begin_button(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path))
    fake = CaptureFakeSt(pressed={"Begin"})

    dashboard._render_intro_step(fake, cfg)

    rendered = "\n".join(fake.markdowns + fake.buttons)
    assert "Train the trigger. Keep the voice." in rendered
    assert "Begin" in fake.buttons
    assert fake.button_kwargs["Begin"]["type"] == "secondary"
    assert fake.session_state[dashboard.DASHBOARD_STEP_KEY] == "workspace"
    assert fake.session_state["rerun_requested"] is True


def test_update_notice_warns_when_github_has_new_commits():
    from wakeword_forge.update_check import UpdateRecommendation

    fake = CaptureFakeSt()
    recommendation = UpdateRecommendation(
        status="update_available",
        message="Update available: GitHub main is 2 commits ahead of this checkout.",
        update_command="git pull --ff-only origin main",
        repo_url="https://github.com/H-Ali13381/wakeword-forge",
        local_ref="old-sha",
        remote_ref="new-sha",
        remote_ahead_by=2,
        detail_url="https://github.com/H-Ali13381/wakeword-forge/compare/old...main",
    )

    dashboard._render_update_notice(fake, recommendation)

    assert fake.warnings == ["Update available: GitHub main is 2 commits ahead of this checkout."]
    assert fake.codes == [("git pull --ff-only origin main", {"language": "bash"})]
    assert any("compare/old...main" in caption for caption in fake.captions)


def test_update_notice_stays_quiet_when_checkout_is_current():
    from wakeword_forge.update_check import UpdateRecommendation

    fake = CaptureFakeSt()
    recommendation = UpdateRecommendation(
        status="current",
        message="wakeword-forge is up to date with GitHub main.",
        update_command="git pull --ff-only origin main",
        repo_url="https://github.com/H-Ali13381/wakeword-forge",
    )

    dashboard._render_update_notice(fake, recommendation)

    assert fake.warnings == []
    assert fake.codes == []


def test_update_recommendation_is_cached_in_session_state():
    from wakeword_forge.update_check import UpdateRecommendation

    fake = CaptureFakeSt()
    recommendation = UpdateRecommendation(
        status="current",
        message="wakeword-forge is up to date with GitHub main.",
        update_command="git pull --ff-only origin main",
        repo_url="https://github.com/H-Ali13381/wakeword-forge",
    )
    calls = 0

    def checker():
        nonlocal calls
        calls += 1
        return recommendation

    assert dashboard._cached_update_recommendation(fake, checker) is recommendation
    assert dashboard._cached_update_recommendation(fake, checker) is recommendation
    assert calls == 1


def test_capture_step_prompts_multiple_phrases_in_rotation(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Okay Hermes",
        wake_phrases=["Okay Hermes", "Hey Hermes"],
        project_dir=str(tmp_path),
        record_positives=20,
        record_negatives=10,
    )
    _touch_wav(cfg.positives_path / "take_0000.wav")
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    rendered = "\n".join(fake.captions)
    assert "Hey Hermes" in rendered


def test_capture_step_shows_one_wake_phrase_record_button_instead_of_bulk_recording(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path), record_positives=20, record_negatives=10)
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    rendered = "\n".join(fake.buttons + fake.captions + fake.markdowns)
    assert "0 / 20 wake-phrase takes saved" in rendered
    assert "Record wake-phrase take 1 of 20" in fake.buttons
    assert "Record 20 wake-phrase takes" not in fake.buttons


def test_capture_step_replays_last_saved_take_and_offers_next_recording(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path), record_positives=20, record_negatives=10)
    status = inspect_project(cfg)
    last_take = cfg.positives_path / "take_0000.wav"
    _touch_wav(last_take)
    fake = CaptureFakeSt()
    fake.session_state[dashboard.LAST_CAPTURED_TAKE_KEY] = {
        "path": str(last_take),
        "kind": "wake-phrase",
        "saved_count": 1,
        "target_count": 20,
    }

    dashboard._render_capture_step(fake, cfg, status)

    assert "Replay last wake-phrase take" in fake.buttons
    assert "Next recording" in fake.buttons
    assert fake.audios, "saved takes should render an audio player for immediate replay"
    assert any("Saved wake-phrase take 1/20" in message for message in fake.successes)


def test_capture_step_record_button_records_exactly_one_take(monkeypatch, tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path), record_positives=20, record_negatives=10)
    status = inspect_project(cfg)
    saved_path = cfg.positives_path / "take_0000.wav"
    calls: list[tuple[str, Path, float, str]] = []

    def fake_record_one_take(*, phrase: str, out_dir: Path, duration: float, prefix: str) -> Path:
        calls.append((phrase, out_dir, duration, prefix))
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        _touch_wav(saved_path)
        return saved_path

    monkeypatch.setattr(dashboard, "_record_one_take", fake_record_one_take)
    fake = CaptureFakeSt(pressed={"Record wake-phrase take 1 of 20"})

    dashboard._render_capture_step(fake, cfg, status)

    assert calls == [("Hey Nova", cfg.positives_path, cfg.record_duration, "take")]
    assert fake.session_state[dashboard.LAST_CAPTURED_TAKE_KEY]["path"] == str(saved_path)
    assert fake.session_state[dashboard.LAST_CAPTURED_TAKE_KEY]["saved_count"] == 1


def test_capture_step_shows_visible_generation_targets_after_required_takes(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Okay Hermes",
        project_dir=str(tmp_path),
        record_positives=20,
        record_negatives=10,
        tts_variants=300,
    )
    for i in range(20):
        _touch_wav(cfg.positives_path / f"take_{i:04d}.wav")
    for i in range(10):
        _touch_wav(cfg.negatives_path / f"neg_{i:04d}.wav")
    for i in range(300):
        _touch_wav(cfg.synthetic_path / f"synth_{i:05d}.wav")
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    rendered = "\n".join(fake.buttons + fake.captions + fake.markdowns + fake.successes)
    assert "Hard negatives: 0/100 partials, 0/0 confusables" in rendered
    assert "Background negatives: 10/150" in rendered
    assert "Generate 100 hard negatives" in fake.buttons
    assert "Fill 140 background negatives" in fake.buttons


def test_capture_step_disables_noop_hard_negative_button_when_targets_are_satisfied(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Okay Hermes",
        project_dir=str(tmp_path),
        record_positives=20,
        record_negatives=10,
        tts_variants=300,
    )
    for i in range(20):
        _touch_wav(cfg.positives_path / f"take_{i:04d}.wav")
    for i in range(150):
        _touch_wav(cfg.negatives_path / f"neg_{i:04d}.wav")
    for i in range(300):
        _touch_wav(cfg.synthetic_path / f"synth_{i:05d}.wav")
    for i in range(100):
        _touch_wav(cfg.partials_path / f"partial_{i:04d}.wav")
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    rendered = "\n".join(fake.buttons + fake.captions + fake.markdowns + fake.successes)
    assert "Hard negatives: 100/100 partials, 0/0 confusables" in rendered
    assert "No confusable phrase cache found" in rendered
    assert "Hard negatives ready" in fake.buttons
    assert fake.button_kwargs["Hard negatives ready"]["disabled"] is True
    assert "Background negatives ready" in fake.buttons
    assert fake.button_kwargs["Background negatives ready"]["disabled"] is True


class ResetFakeSt(CaptureFakeSt):
    def __init__(self, *, checked: bool, pressed: set[str] | None = None):
        super().__init__(pressed=pressed)
        self.checked = checked
        self.checkboxes: list[str] = []
        self.button_kwargs: dict[str, dict] = {}
        self.dividers = 0

    def checkbox(self, label, **_kwargs):
        self.checkboxes.append(str(label))
        return self.checked

    def button(self, label, **kwargs):
        label = str(label)
        self.button_kwargs[label] = kwargs
        return super().button(label, **kwargs)

    def divider(self):
        self.dividers += 1


def test_start_over_controls_require_confirmation_before_wiping(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    fake = ResetFakeSt(checked=False)

    dashboard._render_start_over_controls(fake, cfg)

    assert "I understand this deletes local samples, generated audio, model output, and config" in fake.checkboxes
    assert fake.button_kwargs["Wipe configs and start from scratch"]["disabled"] is True


def test_start_over_controls_wipe_project_and_clear_dashboard_state(monkeypatch, tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    calls: list[ForgeConfig] = []

    def fake_reset_project(config: ForgeConfig):
        calls.append(config)
        return [config.project_path / "forge_config.json", config.samples_path]

    monkeypatch.setattr(dashboard, "reset_project", fake_reset_project)
    fake = ResetFakeSt(checked=True, pressed={"Wipe configs and start from scratch"})
    fake.session_state[dashboard.DASHBOARD_STEP_KEY] = "capture"
    fake.session_state[dashboard.LAST_CAPTURED_TAKE_KEY] = {"path": "old.wav"}

    dashboard._render_start_over_controls(fake, cfg)

    assert calls == [cfg]
    assert dashboard.DASHBOARD_STEP_KEY not in fake.session_state
    assert dashboard.LAST_CAPTURED_TAKE_KEY not in fake.session_state
    assert fake.session_state["rerun_requested"] is True
    assert any("Project reset" in message for message in fake.successes)


def test_dashboard_defaults_to_one_step_at_a_time_for_first_time_user(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path))
    status = inspect_project(cfg)

    assert dashboard._default_wizard_step(status) == "intro"


def test_sidebar_progress_tracks_workflow_without_settings_inputs(tmp_path):
    class FakeSidebar:
        def __init__(self):
            self.calls: list[tuple[str, tuple, dict]] = []

        def _record(self, name, *args, **kwargs):
            self.calls.append((name, args, kwargs))

        def header(self, *args, **kwargs):
            self._record("header", *args, **kwargs)

        def subheader(self, *args, **kwargs):
            self._record("subheader", *args, **kwargs)

        def caption(self, *args, **kwargs):
            self._record("caption", *args, **kwargs)

        def progress(self, *args, **kwargs):
            self._record("progress", *args, **kwargs)

        def metric(self, *args, **kwargs):
            self._record("metric", *args, **kwargs)

        def markdown(self, *args, **kwargs):
            self._record("markdown", *args, **kwargs)

        def divider(self, *args, **kwargs):
            self._record("divider", *args, **kwargs)

        def text_input(self, *_args, **_kwargs):
            raise AssertionError("settings input leaked into sidebar")

        def number_input(self, *_args, **_kwargs):
            raise AssertionError("settings input leaked into sidebar")

        def toggle(self, *_args, **_kwargs):
            raise AssertionError("settings input leaked into sidebar")

        def selectbox(self, *_args, **_kwargs):
            raise AssertionError("settings input leaked into sidebar")

    class FakeSt:
        sidebar = FakeSidebar()

    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    status = inspect_project(cfg)

    dashboard._render_progress_sidebar(FakeSt(), status)

    rendered = "\n".join(str(arg) for _name, args, _kwargs in FakeSt.sidebar.calls for arg in args)
    assert "Progress" in rendered
    assert "Next step" in rendered
    assert "Start" in rendered
    assert "1. Name the trigger" in rendered
    assert "4. Capture examples" in rendered
    assert "5. Review samples" in rendered
    assert "6. Train and test" in rendered
    assert "forge-step-box" in rendered
    assert "forge-step-active" in rendered
    assert "forge-step-pending" in rendered
    assert "✅" not in rendered
    assert "⏳" not in rendered


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
