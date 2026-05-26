from __future__ import annotations

from dataclasses import replace
import runpy
import tomllib
from pathlib import Path

from forge.config import (
    BACKGROUND_NEGATIVE_TARGET,
    ForgeConfig,
    MIN_NEGATIVES,
    MIN_POSITIVES,
    PARTIAL_NEGATIVE_TARGET,
)
import forge.dashboard as dashboard
from forge.dashboard import make_command
from forge.project import inspect_project


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
    assert status.progress_fraction == 3 / (MIN_POSITIVES + BACKGROUND_NEGATIVE_TARGET + PARTIAL_NEGATIVE_TARGET)


def test_project_status_marks_training_ready_when_minimums_are_met(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Computer",
        project_dir=str(tmp_path),
        record_positives=MIN_POSITIVES,
        record_negatives=MIN_NEGATIVES,
    )
    for i in range(MIN_POSITIVES):
        _touch_wav(cfg.positives_path / f"take_{i}.wav")
    for i in range(BACKGROUND_NEGATIVE_TARGET):
        _touch_wav(cfg.negatives_path / f"neg_{i}.wav")

    status = inspect_project(cfg)

    assert status.samples_ready is True
    assert status.ready_to_train is False
    assert status.next_action == "Review samples before training."
    assert status.progress_fraction == 1.0


def test_project_status_blocks_training_until_negative_coverage_targets_are_met(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        record_positives=MIN_POSITIVES,
        record_negatives=MIN_NEGATIVES,
    )
    for i in range(MIN_POSITIVES):
        _touch_wav(cfg.positives_path / f"take_{i}.wav")
    for i in range(MIN_NEGATIVES):
        _touch_wav(cfg.negatives_path / f"neg_{i}.wav")

    status = inspect_project(cfg)

    assert status.samples_ready is False
    assert status.negative_coverage_ready is False
    assert status.background_negative_shortfall == 145
    assert status.partial_negative_shortfall == 100
    assert status.workflow_stage == "negative coverage needed"
    assert status.next_action == (
        "Generate or import at least 145 more background negatives and "
        "100 more partial hard negatives before training."
    )


def test_make_command_outputs_copy_pasteable_cli_fallbacks(tmp_path):
    project_dir = tmp_path / "wakeword project"

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
    namespace = runpy.run_path("forge/dashboard.py")

    default_dir = namespace["DEFAULT_PROJECT_DIR"]
    assert default_dir.parts[-2:] == ("projects", "default")


def test_dashboard_main_uses_dir_arg_when_streamlit_is_already_running(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(dashboard, "_running_inside_streamlit", lambda: True)
    monkeypatch.setattr(dashboard, "run_app", lambda project_dir: calls.append(project_dir))

    dashboard.main(["--dir", str(tmp_path)])

    assert calls == [str(tmp_path)]


def test_workspace_step_confirms_only_project_directory_and_preserves_review_fingerprints(tmp_path):
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

        def text_input(self, label, value, **_kwargs):
            self.text_labels.append(label)
            return value

        def text_area(self, *_args, **_kwargs):
            raise AssertionError("wake phrase textarea should not appear in project-directory step")

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
        wake_phrases=["Hello Nova"],
        project_dir=str(tmp_path),
        sample_review_fingerprint="samples123",
        generated_review_fingerprint="generated123",
        quality_checked_model_path="/tmp/model.onnx",
        quality_checked_model_fingerprint="quality123",
        accepted_model_fingerprint="accepted123",
    )
    fake = FakeSt()

    updated = dashboard._render_workspace_step(fake, cfg)

    assert fake.text_labels == ["Project directory"]
    assert updated.wake_phrase == "Hey Nova"
    assert updated.wake_phrases == ["Hello Nova"]
    assert updated.sample_review_fingerprint == "samples123"
    assert updated.generated_review_fingerprint == "generated123"
    assert updated.quality_checked_model_path == "/tmp/model.onnx"
    assert updated.quality_checked_model_fingerprint == "quality123"
    assert updated.accepted_model_fingerprint == "accepted123"


def test_phrase_step_uses_single_line_primary_phrase_and_alias_button(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.text_labels: list[str] = []
            self.buttons: list[str] = []
            self.button_kwargs: dict[str, dict] = {}

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def text_input(self, label, value="", **_kwargs):
            self.text_labels.append(label)
            if label == "Primary wake phrase":
                return "Hey Nova"
            return value

        def text_area(self, *_args, **_kwargs):
            raise AssertionError("wake phrases should not use a one-per-line textarea")

        def button(self, label, **kwargs):
            self.buttons.append(str(label))
            self.button_kwargs[str(label)] = kwargs
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(project_dir=str(tmp_path))

    updated = dashboard._render_phrase_step(fake, cfg)

    assert fake.text_labels == ["Primary wake phrase"]
    assert "Add another phrase" in fake.buttons
    assert "Confirm wake phrase" in fake.buttons
    assert fake.button_kwargs["Confirm wake phrase"]["type"] == "primary"
    assert updated.wake_phrase == "Hey Nova"
    assert updated.wake_phrases == []


def test_phrase_step_renders_optional_alias_rows(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state = {dashboard.PHRASE_ALIAS_COUNT_KEY: 1}
            self.text_labels: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def text_input(self, label, value="", **_kwargs):
            self.text_labels.append(label)
            if label == "Primary wake phrase":
                return "Hey Nova"
            if label == "Alias 1":
                return "Hello Nova"
            return value

        def text_area(self, *_args, **_kwargs):
            raise AssertionError("aliases should render as single-line inputs")

        def button(self, *_args, **_kwargs):
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(project_dir=str(tmp_path))

    updated = dashboard._render_phrase_step(fake, cfg)

    assert fake.text_labels == ["Primary wake phrase", "Alias 1"]
    assert updated.wake_phrase == "Hey Nova"
    assert updated.wake_phrases == ["Hello Nova"]


def test_recording_step_shows_only_recording_parameters(tmp_path):
    class FakeSt:
        def __init__(self):
            self.number_labels: list[str] = []
            self.text_labels: list[str] = []
            self.select_labels: list[str] = []
            self.buttons: list[str] = []
            self.button_kwargs: dict[str, dict] = {}

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def number_input(self, label, **kwargs):
            self.number_labels.append(label)
            return kwargs["value"]

        def text_input(self, label, **kwargs):
            self.text_labels.append(label)
            return kwargs.get("value", "")

        def toggle(self, *_args, **_kwargs):
            raise AssertionError("augmentation parameters leaked into recording step")

        def selectbox(self, label, **kwargs):
            self.select_labels.append(label)
            return kwargs["options"][kwargs["index"]]

        def button(self, label, **kwargs):
            self.buttons.append(str(label))
            self.button_kwargs[str(label)] = kwargs
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

    assert fake.select_labels == ["Positive sample source", "Negative sample source"]
    assert fake.text_labels == []
    assert fake.number_labels == [
        "Target positive examples",
        "Target negative recordings",
        "Seconds per take",
    ]
    assert fake.buttons[-2:] == ["Back", "Confirm recording plan"]
    assert fake.button_kwargs["Back"]["type"] == "secondary"
    assert fake.button_kwargs["Confirm recording plan"]["type"] == "primary"


def test_augmentation_step_offers_qwentts_engine(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.select_options: dict[str, list[str]] = {}
            self.select_help: dict[str, str] = {}
            self.text_labels: list[str] = []
            self.buttons: list[str] = []
            self.captions: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *args, **_kwargs):
            if args:
                self.captions.append(str(args[0]))

        def toggle(self, label, **kwargs):
            if label == "Use SpecAugment-style mel masking":
                return False
            return kwargs["value"]

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            self.select_options[str(label)] = list(kwargs["options"])
            self.select_help[str(label)] = str(kwargs.get("help", ""))
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(str(label))
            return kwargs["value"]

        def button(self, label, **_kwargs):
            self.buttons.append(str(label))
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path), tts_engine="qwentts")

    updated = dashboard._render_augmentation_step(fake, cfg)

    assert fake.select_options["TTS engine"] == ["qwentts", "kokoro", "piper", "none"]
    engine_help = fake.select_help["TTS engine"]
    assert "QwenTTS" in engine_help
    assert "most natural" in engine_help
    assert "compatible hardware" in engine_help
    assert "slower" in engine_help
    assert "Kokoro" in engine_help
    assert "Piper" in engine_help
    assert fake.select_options["Training augmentation preset"] == ["standard", "light"]
    assert fake.select_options["Background negative augmentation"] == ["light", "standard", "none"]
    assert fake.select_options["Background noise data source"] == [
        "Use recommended open-source data",
        "Use my own local folder",
        "Skip external background data",
    ]
    assert fake.select_options["Advanced acoustic folder source"] == [
        "Use recommended open-source data",
        "Use my own local folders",
        "Skip advanced acoustic folders",
    ]
    assert fake.text_labels == []
    assert "Import recommended open-source data" not in fake.buttons
    assert updated.tts_engine == "qwentts"
    assert updated.training_augmentation_enabled is True


def test_augmentation_step_groups_controls_and_uses_dropdown_for_advanced_folders(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.markdowns: list[str] = []
            self.captions: list[str] = []
            self.text_labels: list[str] = []
            self.select_options: dict[str, list[str]] = {}
            self.buttons: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def markdown(self, text, **_kwargs):
            self.markdowns.append(str(text))

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def toggle(self, label, **kwargs):
            if label == "Use SpecAugment-style mel masking":
                return False
            return kwargs["value"]

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            self.select_options[str(label)] = list(kwargs["options"])
            if label == "Background noise data source":
                return "Use recommended open-source data"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(str(label))
            return kwargs["value"]

        def button(self, label, **_kwargs):
            self.buttons.append(str(label))
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    dashboard._render_augmentation_step(fake, cfg)

    rendered = "\n".join(fake.markdowns + fake.captions)
    assert "Generated positives" in rendered
    assert "Optional synthetic wake-phrase clips" in rendered
    assert "Training-time robustness" in rendered
    assert "Augment reviewed samples during training" in rendered
    assert "Background noise source" in rendered
    assert "Choose the noise pool used by background mixing" in rendered
    assert "Advanced acoustic folders" in rendered
    assert "Optional room, transient, and low-frequency acoustic assets" in rendered
    assert fake.select_options["Advanced acoustic folder source"] == [
        "Use recommended open-source data",
        "Use my own local folders",
        "Skip advanced acoustic folders",
    ]
    assert fake.text_labels == []


def test_augmentation_step_defaults_advanced_folders_to_manual_when_any_value_is_set(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.select_indices: dict[str, int] = {}
            self.text_labels: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def markdown(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def toggle(self, label, **kwargs):
            if label == "Use SpecAugment-style mel masking":
                return False
            return kwargs["value"]

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            self.select_indices[str(label)] = int(kwargs["index"])
            if label == "Background noise data source":
                return "Use recommended open-source data"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(str(label))
            return kwargs["value"]

        def button(self, *_args, **_kwargs):
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        augmentation_ir_dir=str(tmp_path / "impulses"),
    )

    dashboard._render_augmentation_step(fake, cfg)

    assert fake.select_indices["Advanced acoustic folder source"] == 1
    assert fake.text_labels[-3:] == [
        "Room impulse response folder",
        "Short noise folder",
        "Low-frequency noise folder",
    ]


def test_augmentation_step_can_select_recommended_open_data_for_advanced_folders(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.select_options: dict[str, list[str]] = {}
            self.text_labels: list[str] = []
            self.markdowns: list[str] = []
            self.captions: list[str] = []
            self.buttons: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def markdown(self, text, **_kwargs):
            self.markdowns.append(str(text))

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def toggle(self, label, **kwargs):
            if label == "Use SpecAugment-style mel masking":
                return False
            return kwargs["value"]

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            self.select_options[str(label)] = list(kwargs["options"])
            if label in {"Background noise data source", "Advanced acoustic folder source"}:
                return "Use recommended open-source data"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(str(label))
            return kwargs["value"]

        def button(self, label, **_kwargs):
            self.buttons.append(str(label))
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    updated = dashboard._render_augmentation_step(fake, cfg)

    dirs = dashboard._recommended_advanced_acoustic_dirs(cfg)
    assert fake.select_options["Advanced acoustic folder source"] == [
        "Use recommended open-source data",
        "Use my own local folders",
        "Skip advanced acoustic folders",
    ]
    assert "Room impulse response folder" not in fake.text_labels
    assert updated.augmentation_ir_dir == str(dirs["ir"])
    assert updated.augmentation_short_noise_dir == str(dirs["short_noise"])
    assert updated.augmentation_truck_noise_dir == str(dirs["low_frequency"])
    rendered = "\n".join(fake.markdowns + fake.captions)
    assert "Recommended advanced acoustic data will be installed" in rendered
    assert "Recommended advanced acoustic folders" not in rendered
    assert str(dirs["ir"]) in rendered
    assert "Import recommended advanced acoustic data" in fake.buttons


def test_augmentation_step_opens_recommended_advanced_acoustic_confirmation_in_modal(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.dialog_titles: list[str] = []
            self.buttons: list[str] = []
            self.markdowns: list[str] = []
            self.checkboxes: list[str] = []
            self.warnings: list[str] = []
            self.captions: list[str] = []

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def markdown(self, text, **_kwargs):
            self.markdowns.append(str(text))

        def warning(self, text, **_kwargs):
            self.warnings.append(str(text))

        def checkbox(self, label, **_kwargs):
            self.checkboxes.append(str(label))
            return False

        def button(self, label, **_kwargs):
            label = str(label)
            self.buttons.append(label)
            return label == "Import recommended advanced acoustic data"

        def dialog(self, title):
            self.dialog_titles.append(str(title))

            def decorator(fn):
                def wrapped(*args, **kwargs):
                    return fn(*args, **kwargs)

                return wrapped

            return decorator

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    selected_dirs = dashboard._render_recommended_advanced_acoustic_import(fake, cfg)

    assert fake.session_state[dashboard.ADVANCED_DATA_CONFIRM_KEY] is True
    assert fake.dialog_titles == ["Import recommended advanced acoustic data"]
    rendered_modal = "\n".join(fake.markdowns + fake.warnings + fake.checkboxes + fake.buttons + fake.captions)
    assert "Recommended advanced acoustic data may include" in rendered_modal
    assert "Confirm and install recommended acoustic data" in fake.buttons
    assert selected_dirs["ir"] == str(dashboard._recommended_advanced_acoustic_dirs(cfg)["ir"])


def test_augmentation_step_installs_recommended_advanced_acoustic_data_with_progress(monkeypatch, tmp_path):
    calls: list[ForgeConfig] = []

    def fake_import_recommended_advanced_acoustic_data(config: ForgeConfig, *, progress_callback=None):
        calls.append(config)
        if progress_callback is not None:
            progress_callback("Preparing recommended acoustic folders", 0, 3)
            progress_callback("Installing room and noise assets", 1, 3)
            progress_callback("Recommended acoustic assets ready", 3, 3)
        dirs = dashboard._recommended_advanced_acoustic_dirs(config)
        imported = []
        for name, directory in dirs.items():
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{name}_0001.wav"
            path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            imported.append(path)
        return imported

    monkeypatch.setattr(
        dashboard,
        "import_recommended_advanced_acoustic_data",
        fake_import_recommended_advanced_acoustic_data,
    )

    class Progress:
        def __init__(self, owner):
            self.owner = owner

        def progress(self, value, *, text=""):
            self.owner.progress_updates.append((value, text))

    class Spinner:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeSt:
        def __init__(self):
            self.session_state = {dashboard.ADVANCED_DATA_CONFIRM_KEY: True}
            self.progress_updates: list[tuple[float, str]] = []
            self.successes: list[str] = []

        def caption(self, *_args, **_kwargs):
            pass

        def markdown(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def checkbox(self, *_args, **_kwargs):
            return True

        def button(self, label, **_kwargs):
            return label == "Confirm and install recommended acoustic data"

        def progress(self, value, *, text=""):
            self.progress_updates.append((value, text))
            return Progress(self)

        def spinner(self, *_args, **_kwargs):
            return Spinner()

        def success(self, text):
            self.successes.append(str(text))

        def rerun(self):
            self.session_state["rerun_requested"] = True

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    selected_dirs = dashboard._render_recommended_advanced_acoustic_import(fake, cfg)

    assert calls
    saved = ForgeConfig.load(tmp_path / "forge_config.json")
    assert saved.augmentation_ir_dir == selected_dirs["ir"]
    assert saved.augmentation_short_noise_dir == selected_dirs["short_noise"]
    assert saved.augmentation_truck_noise_dir == selected_dirs["low_frequency"]
    assert fake.progress_updates[0] == (0.0, "Preparing recommended acoustic folders")
    assert fake.progress_updates[-1] == (1.0, "Recommended acoustic assets ready")
    assert any("Installed 3 recommended advanced acoustic audio files" in message for message in fake.successes)
    assert fake.session_state[dashboard.ADVANCED_DATA_CONFIRM_KEY] is False
    assert fake.session_state["rerun_requested"] is True

def test_augmentation_step_recommends_open_source_background_data_with_license_disclaimer(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.select_options: dict[str, list[str]] = {}
            self.text_labels: list[str] = []
            self.buttons: list[str] = []
            self.captions: list[str] = []
            self.markdowns: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def markdown(self, text, **_kwargs):
            self.markdowns.append(str(text))

        def toggle(self, label, **kwargs):
            if label == "Use SpecAugment-style mel masking":
                return False
            return kwargs["value"]

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            self.select_options[str(label)] = list(kwargs["options"])
            if label == "Background noise data source":
                return "Use recommended open-source data"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(str(label))
            return kwargs["value"]

        def button(self, label, **_kwargs):
            self.buttons.append(str(label))
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    updated = dashboard._render_augmentation_step(fake, cfg)

    assert fake.select_options["Background noise data source"] == [
        "Use recommended open-source data",
        "Use my own local folder",
        "Skip external background data",
    ]
    assert "Background noise folder" not in fake.text_labels
    assert fake.select_options["Advanced acoustic folder source"] == [
        "Use recommended open-source data",
        "Use my own local folders",
        "Skip advanced acoustic folders",
    ]
    assert fake.text_labels == []
    rendered = "\n".join(fake.captions + fake.markdowns)
    assert "Mozilla Common Voice" in rendered
    assert "ESC-50" in rendered
    assert "CC BY-NC 3.0" in rendered
    assert "verify the dataset licenses" in rendered
    assert "Import recommended open-source data" in fake.buttons
    assert updated.augmentation_noise_dir == str(dashboard._recommended_open_data_dir(cfg))


def test_augmentation_step_requires_license_confirmation_before_downloading_recommended_data(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state = {dashboard.OPEN_DATA_CONFIRM_KEY: True}
            self.button_kwargs: dict[str, dict] = {}
            self.checkboxes: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def markdown(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def toggle(self, label, **kwargs):
            if label == "Use SpecAugment-style mel masking":
                return False
            return kwargs["value"]

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            if label == "Background noise data source":
                return "Use recommended open-source data"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, _label, **kwargs):
            return kwargs["value"]

        def checkbox(self, label, **_kwargs):
            self.checkboxes.append(str(label))
            return False

        def button(self, label, **kwargs):
            self.button_kwargs[str(label)] = kwargs
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    dashboard._render_augmentation_step(fake, cfg)

    assert any("license" in label.lower() for label in fake.checkboxes)
    assert fake.button_kwargs["Confirm and download recommended data"]["disabled"] is True


def test_augmentation_step_opens_recommended_data_confirmation_in_modal(tmp_path):
    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.dialog_titles: list[str] = []
            self.buttons: list[str] = []
            self.markdowns: list[str] = []
            self.checkboxes: list[str] = []
            self.warnings: list[str] = []

        def caption(self, *_args, **_kwargs):
            pass

        def markdown(self, text, **_kwargs):
            self.markdowns.append(str(text))

        def warning(self, text, **_kwargs):
            self.warnings.append(str(text))

        def checkbox(self, label, **_kwargs):
            self.checkboxes.append(str(label))
            return False

        def button(self, label, **_kwargs):
            label = str(label)
            self.buttons.append(label)
            return label == "Import recommended open-source data"

        def dialog(self, title):
            self.dialog_titles.append(str(title))

            def decorator(fn):
                def wrapped(*args, **kwargs):
                    return fn(*args, **kwargs)

                return wrapped

            return decorator

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    dashboard._render_recommended_open_data_import(fake, cfg)

    assert fake.session_state[dashboard.OPEN_DATA_CONFIRM_KEY] is True
    assert fake.dialog_titles == ["Import recommended open-source data"]
    rendered_modal = "\n".join(fake.markdowns + fake.warnings + fake.checkboxes + fake.buttons)
    assert "Recommended open-source data may include" in rendered_modal
    assert "Confirm and download recommended data" in fake.buttons


def test_augmentation_step_makes_imported_recommended_data_state_obvious(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    recommended_dir = dashboard._recommended_open_data_dir(cfg)
    _touch_wav(recommended_dir / "recommended_0001.wav")
    _touch_wav(recommended_dir / "recommended_0002.wav")
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        augmentation_noise_dir=str(recommended_dir),
    )

    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.captions: list[str] = []
            self.successes: list[str] = []
            self.markdowns: list[str] = []
            self.buttons: list[str] = []

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def success(self, text, **_kwargs):
            self.successes.append(str(text))

        def markdown(self, text, **_kwargs):
            self.markdowns.append(str(text))

        def button(self, label, **_kwargs):
            self.buttons.append(str(label))
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()

    selected_dir = dashboard._render_recommended_open_data_import(fake, cfg)

    rendered = "\n".join(fake.successes + fake.captions + fake.markdowns)
    assert selected_dir == str(recommended_dir)
    assert "Recommended background data" not in rendered
    assert "forge-data-source-kicker" not in rendered
    assert "Active · 2 audio files" in rendered
    assert str(recommended_dir) in rendered


def test_augmentation_step_replaces_import_cta_with_repair_action_when_recommended_data_active(
    tmp_path,
):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    recommended_dir = dashboard._recommended_open_data_dir(cfg)
    _touch_wav(recommended_dir / "recommended_0001.wav")
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        augmentation_noise_dir=str(recommended_dir),
    )

    class FakeSt:
        def __init__(self):
            self.session_state: dict[str, object] = {}
            self.captions: list[str] = []
            self.successes: list[str] = []
            self.markdowns: list[str] = []
            self.buttons: list[str] = []

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def success(self, text, **_kwargs):
            self.successes.append(str(text))

        def markdown(self, text, **_kwargs):
            self.markdowns.append(str(text))

        def button(self, label, **_kwargs):
            self.buttons.append(str(label))
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()

    dashboard._render_recommended_open_data_import(fake, cfg)

    rendered = "\n".join(fake.successes + fake.captions + fake.markdowns)
    assert "forge-data-source-card" in rendered
    assert "Active · 1 audio file" in rendered
    assert "This folder is selected for training-time background-noise augmentation." not in rendered
    assert "Only needed if files are missing" not in rendered
    assert "Recommended import includes" not in rendered
    assert "Re-import or repair recommended data" in fake.buttons
    assert "Import recommended open-source data" not in fake.buttons


def test_augmentation_step_downloads_recommended_open_source_data_with_progress(monkeypatch, tmp_path):
    calls: list[ForgeConfig] = []

    def fake_import_recommended_open_audio(config: ForgeConfig, *, progress_callback=None):
        calls.append(config)
        if progress_callback is not None:
            progress_callback("Preparing recommended data folders", 0, 3)
            progress_callback("Downloading open-source audio", 1, 3)
            progress_callback("Recommended audio ready", 3, 3)
        out_dir = dashboard._recommended_open_data_dir(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        imported = out_dir / "recommended_0001.wav"
        imported.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        return [imported]

    monkeypatch.setattr(dashboard, "import_recommended_open_audio", fake_import_recommended_open_audio)

    class Progress:
        def __init__(self, owner):
            self.owner = owner

        def progress(self, value, *, text=""):
            self.owner.progress_updates.append((value, text))

    class Spinner:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeSt:
        def __init__(self):
            self.session_state = {dashboard.OPEN_DATA_CONFIRM_KEY: True}
            self.progress_updates: list[tuple[float, str]] = []
            self.successes: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

        def markdown(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def toggle(self, label, **kwargs):
            if label == "Use SpecAugment-style mel masking":
                return False
            return kwargs["value"]

        def number_input(self, _label, **kwargs):
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            if label == "Background noise data source":
                return "Use recommended open-source data"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, _label, **kwargs):
            return kwargs["value"]

        def checkbox(self, *_args, **_kwargs):
            return True

        def button(self, label, **_kwargs):
            return label == "Confirm and download recommended data"

        def progress(self, value, *, text=""):
            self.progress_updates.append((value, text))
            return Progress(self)

        def spinner(self, *_args, **_kwargs):
            return Spinner()

        def success(self, text):
            self.successes.append(str(text))

        def rerun(self):
            self.session_state["rerun_requested"] = True

        def columns(self, count):
            return [self for _ in range(count)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = FakeSt()
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))

    updated = dashboard._render_augmentation_step(fake, cfg)

    assert calls == [updated]
    assert fake.progress_updates[0] == (0.0, "Preparing recommended data folders")
    assert fake.progress_updates[-1] == (1.0, "Recommended audio ready")
    assert any("Imported 1 recommended open-source audio file" in message for message in fake.successes)
    assert fake.session_state[dashboard.OPEN_DATA_CONFIRM_KEY] is False
    assert fake.session_state["rerun_requested"] is True


def test_recording_step_can_choose_existing_positive_sample_folder(tmp_path):
    source = tmp_path / "existing voice"
    _touch_wav(source / "okay_hermes_0001.wav")
    _touch_wav(source / "nested" / "hey_hermes_0002.wav")
    (source / "notes.txt").write_text("not audio")

    class FakeSt:
        def __init__(self):
            self.text_labels: list[str] = []
            self.number_labels: list[str] = []
            self.captions: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def number_input(self, label, **kwargs):
            self.number_labels.append(str(label))
            assert label != "Target positive examples"
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            if label == "Positive sample source":
                return "Import existing folder"
            if label == "Negative sample source":
                return "Record with microphone"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(label)
            assert kwargs["help"].startswith("Folder containing existing wake-phrase")
            return str(source)

        def toggle(self, *_args, **_kwargs):
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

    updated = dashboard._render_recording_step(fake, cfg)

    assert fake.text_labels == ["Existing positive sample folder"]
    assert fake.number_labels == ["Target negative recordings", "Seconds per take"]
    assert any("Found 2 existing wake-phrase audio files" in caption for caption in fake.captions)
    assert updated.record_positives == 2
    assert updated.sample_source_dir == str(source)


def test_recording_step_can_choose_existing_negative_sample_folder(tmp_path):
    negative_source = tmp_path / "existing negatives"
    _touch_wav(negative_source / "room_noise_0001.wav")
    _touch_wav(negative_source / "nested" / "speech_0002.ogg")
    (negative_source / "notes.txt").write_text("not audio")

    class FakeSt:
        def __init__(self):
            self.text_labels: list[str] = []
            self.number_labels: list[str] = []
            self.select_labels: list[str] = []
            self.captions: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def number_input(self, label, **kwargs):
            self.number_labels.append(str(label))
            assert label != "Target negative recordings"
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            self.select_labels.append(str(label))
            if label == "Positive sample source":
                return "Record with microphone"
            if label == "Negative sample source":
                return "Import existing folder"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(str(label))
            assert kwargs["help"].startswith("Folder containing existing non-wakeword")
            return str(negative_source)

        def toggle(self, *_args, **_kwargs):
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

    updated = dashboard._render_recording_step(fake, cfg)

    assert fake.select_labels == ["Positive sample source", "Negative sample source"]
    assert fake.text_labels == ["Existing negative sample folder"]
    assert fake.number_labels == ["Target positive examples", "Seconds per take"]
    assert any("Found 2 existing negative audio files" in caption for caption in fake.captions)
    assert updated.record_negatives == 2
    assert updated.negative_source_dir == str(negative_source)


def test_recording_step_hides_seconds_per_take_when_both_sources_are_imported(tmp_path):
    positive_source = tmp_path / "existing positives"
    negative_source = tmp_path / "existing negatives"
    _touch_wav(positive_source / "okay_hermes_0001.wav")
    _touch_wav(negative_source / "room_noise_0001.wav")

    class FakeSt:
        def __init__(self):
            self.text_labels: list[str] = []
            self.number_labels: list[str] = []
            self.select_labels: list[str] = []
            self.captions: list[str] = []

        def subheader(self, *_args, **_kwargs):
            pass

        def caption(self, text, **_kwargs):
            self.captions.append(str(text))

        def number_input(self, label, **kwargs):
            self.number_labels.append(str(label))
            return kwargs["value"]

        def selectbox(self, label, **kwargs):
            self.select_labels.append(str(label))
            if label in {"Positive sample source", "Negative sample source"}:
                return "Import existing folder"
            return kwargs["options"][kwargs["index"]]

        def text_input(self, label, **kwargs):
            self.text_labels.append(str(label))
            if label == "Existing positive sample folder":
                return str(positive_source)
            if label == "Existing negative sample folder":
                return str(negative_source)
            return kwargs.get("value", "")

        def toggle(self, *_args, **_kwargs):
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
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path), record_duration=5.25)

    updated = dashboard._render_recording_step(fake, cfg)

    assert fake.select_labels == ["Positive sample source", "Negative sample source"]
    assert fake.text_labels == ["Existing positive sample folder", "Existing negative sample folder"]
    assert fake.number_labels == []
    assert updated.record_duration == 5.25


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

    fake.session_state[dashboard.DASHBOARD_STEP_KEY] = "phrase"
    assert dashboard._current_wizard_step(fake, status) == "phrase"


def test_intro_step_renders_title_card_and_begin_button(tmp_path):
    cfg = ForgeConfig(project_dir=str(tmp_path))
    fake = CaptureFakeSt(pressed={"Begin"})

    dashboard._render_intro_step(fake, cfg)

    rendered = "\n".join(fake.markdowns + fake.buttons)
    assert "Train the trigger. Keep the voice." in rendered
    assert "Begin" in fake.buttons
    assert fake.button_kwargs["Begin"]["type"] == "primary"
    assert fake.session_state[dashboard.DASHBOARD_STEP_KEY] == "workspace"
    assert fake.session_state["rerun_requested"] is True


def test_step_back_navigation_moves_to_previous_step_and_clears_capture_replay():
    fake = CaptureFakeSt(pressed={"Back"})
    fake.session_state[dashboard.DASHBOARD_STEP_KEY] = "capture"
    fake.session_state[dashboard.LAST_CAPTURED_TAKE_KEY] = {"path": "old.wav"}

    dashboard._render_step_back_navigation(fake, "capture")

    assert "Back" in fake.buttons
    assert fake.session_state[dashboard.DASHBOARD_STEP_KEY] == "augmentation"
    assert dashboard.LAST_CAPTURED_TAKE_KEY not in fake.session_state
    assert fake.session_state["rerun_requested"] is True


def test_wizard_action_row_places_gray_back_next_to_blue_primary_action():
    fake = CaptureFakeSt()

    pressed = dashboard._render_wizard_action_row(fake, "recording", "Confirm recording plan")

    assert pressed is False
    assert fake.buttons[-2:] == ["Back", "Confirm recording plan"]
    assert fake.button_kwargs["Back"]["type"] == "secondary"
    assert fake.button_kwargs["Confirm recording plan"]["type"] == "primary"
    assert fake.button_kwargs["Back"]["use_container_width"] is True
    assert fake.button_kwargs["Confirm recording plan"]["use_container_width"] is True


def test_step_change_invalidates_downstream_checkpoints_only_when_values_changed(tmp_path):
    reviewed = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        sample_review_approved=True,
        generated_review_approved=True,
        sample_review_fingerprint="samples123",
        generated_review_fingerprint="generated123",
        trained_sample_fingerprint="training123",
        trained_eer=0.12,
        quality_check_passed=True,
        model_accepted=True,
        quality_checked_model_path="/tmp/model.onnx",
        quality_checked_model_fingerprint="quality123",
        accepted_model_fingerprint="accepted123",
        quality_positive_hits=3,
        quality_positive_trials=3,
        quality_false_triggers=0,
        quality_score_min=0.7,
        quality_score_max=0.9,
    )

    unchanged = dashboard._apply_step_change_invalidations(reviewed, replace(reviewed), "phrase")
    changed = dashboard._apply_step_change_invalidations(
        reviewed,
        replace(reviewed, wake_phrase="Computer"),
        "phrase",
    )

    assert unchanged.sample_review_approved is True
    assert unchanged.generated_review_approved is True
    assert unchanged.quality_check_passed is True
    assert unchanged.model_accepted is True
    assert changed.sample_review_approved is False
    assert changed.sample_review_fingerprint == ""
    assert changed.generated_review_approved is False
    assert changed.generated_review_fingerprint == ""
    assert changed.trained_sample_fingerprint == ""
    assert changed.trained_eer is None
    assert changed.quality_check_passed is False
    assert changed.model_accepted is False
    assert changed.quality_checked_model_path == ""
    assert changed.quality_checked_model_fingerprint == ""
    assert changed.accepted_model_fingerprint == ""
    assert changed.quality_positive_hits == 0
    assert changed.quality_positive_trials == 0


def test_augmentation_plan_change_keeps_sample_review_but_invalidates_generated_and_training(tmp_path):
    reviewed = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        sample_review_approved=True,
        generated_review_approved=True,
        sample_review_fingerprint="samples123",
        generated_review_fingerprint="generated123",
        trained_sample_fingerprint="training123",
        trained_eer=0.12,
        quality_check_passed=True,
        model_accepted=True,
        quality_checked_model_path="/tmp/model.onnx",
        quality_checked_model_fingerprint="quality123",
        accepted_model_fingerprint="accepted123",
    )

    changed = dashboard._apply_step_change_invalidations(
        reviewed,
        replace(reviewed, tts_engine="kokoro"),
        "augmentation",
    )

    assert changed.sample_review_approved is True
    assert changed.sample_review_fingerprint == "samples123"
    assert changed.generated_review_approved is False
    assert changed.generated_review_fingerprint == ""
    assert changed.trained_sample_fingerprint == ""
    assert changed.quality_check_passed is False
    assert changed.model_accepted is False


def test_intro_cards_have_equal_height_layout_css():
    css = dashboard._css()

    assert ".forge-card" in css
    assert "box-sizing: border-box;" in css
    assert "height: 9.5rem;" in css
    assert "display: flex;" in css
    assert "flex-direction: column;" in css
    assert "forge-card-grid" in css


def test_intro_card_html_is_not_markdown_indented_code():
    card = dashboard._card("Guided", "Wizard", "One decision at a time, then explicit review.")

    assert card.startswith('<div class="forge-card">')
    assert "\n    <div" not in card
    assert card.endswith("</div>")


def test_step_guidance_is_compact_and_neutral():
    fake = CaptureFakeSt()

    dashboard._render_step_guidance(fake, "phrase")

    rendered = "\n".join(fake.captions + fake.markdowns)
    assert "Use one primary phrase" in rendered
    assert "forge-guide-card" not in rendered
    assert "Next move" not in rendered
    assert "What" not in rendered
    assert "Why" not in rendered


def test_dashboard_css_removes_gap_between_source_selectbox_and_active_data_card():
    css = dashboard._css()

    assert ".forge-data-source-card" in css
    assert "margin: -0.25rem 0 0.65rem;" in css


def test_dashboard_css_uses_blue_for_primary_actions_and_amber_only_for_state():
    css = dashboard._css()

    assert "--forge-primary" in css
    assert "#58c7ff" in css
    assert "button[kind=\"primary\"]" in css
    assert "linear-gradient(135deg, var(--forge-primary)" in css
    assert "button[kind=\"secondary\"]" in css
    assert "background: rgba(255, 242, 223, 0.06);" in css
    assert "forge-step-active" in css
    assert "var(--forge-active)" in css
    assert "forge-next-action" not in css


def test_dashboard_dark_theme_forces_readable_main_text_and_inputs():
    css = dashboard._css()

    assert ".stApp {" in css
    assert "color: var(--forge-text);" in css
    assert "[data-testid=\"stSidebar\"]" in css
    assert "[data-testid=\"stSidebar\"] *" in css
    assert "[data-testid=\"stWidgetLabel\"]" in css
    assert "[data-testid=\"stTextInput\"] input" in css
    assert "background: rgba(17, 19, 23, 0.82);" in css
    assert "caret-color: var(--forge-primary);" in css


def test_dashboard_css_styles_streamlit_toggles_blue_when_checked():
    css = dashboard._css()

    assert "label[data-baseweb=\"checkbox\"]:has(input:checked) > div:first-child" in css
    assert "background: var(--forge-primary-strong) !important;" in css
    assert "border-color: var(--forge-primary-strong) !important;" in css
    assert "label[data-baseweb=\"checkbox\"]:has(input:focus-visible) > div:first-child" in css


def test_dashboard_css_applies_same_blue_outline_to_baseweb_inputs_and_selects():
    css = dashboard._css()

    assert 'div[data-baseweb="input"],' in css
    assert 'div[data-baseweb="select"]' in css
    assert "border-color: rgba(88, 199, 255, 0.24)" in css
    assert 'div[data-baseweb="input"]:focus-within' in css
    assert 'div[data-baseweb="select"]:focus-within' in css
    assert "box-shadow: 0 0 0 0.14rem rgba(88, 199, 255, 0.32)" in css


def test_dashboard_css_forces_baseweb_select_inner_surface_dark():
    css = dashboard._css()

    assert 'div[data-baseweb="select"] > div {' in css
    assert "background: rgba(17, 19, 23, 0.82) !important;" in css
    assert 'div[data-baseweb="select"] [role="combobox"]' in css


def test_dashboard_css_forces_number_input_inner_surfaces_dark():
    css = dashboard._css()

    assert 'div[data-baseweb="input"] [data-baseweb="base-input"]' in css
    assert '[data-testid="stNumberInputStepDown"]' in css
    assert '[data-testid="stNumberInputStepUp"]' in css
    assert "background: rgba(17, 19, 23, 0.82) !important;" in css
    assert "color: var(--forge-text) !important;" in css


def test_current_step_renderer_does_not_emit_redundant_next_move_card(tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_current_wizard_step(fake, cfg, status, "done")

    rendered = "\n".join(fake.markdowns + fake.captions)
    assert "Next move" not in rendered
    assert "forge-next-action" not in rendered


def test_update_notice_warns_when_github_has_new_commits():
    from forge.update_check import UpdateRecommendation

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
    from forge.update_check import UpdateRecommendation

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
    from forge.update_check import UpdateRecommendation

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
        wake_phrase="Hey Nova",
        wake_phrases=["Hey Nova", "Hello Nova"],
        project_dir=str(tmp_path),
        record_positives=20,
        record_negatives=10,
    )
    _touch_wav(cfg.positives_path / "take_0000.wav")
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    rendered = "\n".join(fake.captions)
    assert "Hello Nova" in rendered


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


def test_capture_step_recommends_strong_negative_utterances(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path),
        record_positives=20,
        record_negatives=10,
    )
    for i in range(20):
        _touch_wav(cfg.positives_path / f"take_{i:04d}.wav")
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    rendered = "\n".join(fake.captions + fake.markdowns)
    assert "confusable near-misses" in rendered
    assert "partial utterances" in rendered
    assert "repeated fragments" in rendered
    for example in (
        "hey novah",
        "hey",
        "hey-",
        "no",
        "va",
        "nova",
        "heyheyheyhey",
        "nonono",
        "vavava",
        "nova-nova",
    ):
        assert example in rendered


def test_negative_guidance_filters_full_trigger_aliases():
    cfg = ForgeConfig(wake_phrase="Nova", wake_phrases=["Hey Nova"])

    guidance = dashboard._negative_example_guidance(cfg)

    assert "`nova`" not in guidance
    assert "`nova-nova`" not in guidance
    assert "`hey nova`" not in guidance
    assert "novah" in guidance
    assert "nonono" in guidance


def test_negative_guidance_filters_separatorless_repeated_short_triggers():
    cases = (
        (ForgeConfig(wake_phrase="OK"), ("`ok`", "`okokok`", "`okokokok`", "`ok-ok`")),
        (ForgeConfig(wake_phrase="Go"), ("`go`", "`gogogo`", "`gogogogo`", "`go-go`")),
        (ForgeConfig(wake_phrase="Hey", wake_phrases=["Hey Nova"]), ("`hey`", "`heyheyhey`", "`heyheyheyhey`")),
        (ForgeConfig(wake_phrase="OK", wake_phrases=["OK Nova"]), ("`ok novah`",)),
        (ForgeConfig(wake_phrase="Okay", wake_phrases=["Hey Nova"]), ("`okay novah`",)),
    )

    for cfg, forbidden_examples in cases:
        guidance = dashboard._negative_example_guidance(cfg)
        for example in forbidden_examples:
            assert example not in guidance


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


def test_capture_step_imports_existing_samples_instead_of_recording(monkeypatch, tmp_path):
    source = tmp_path / "existing voice"
    source.mkdir()
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path / "project"),
        record_positives=3,
        record_negatives=10,
        sample_source_dir=str(source),
    )
    _touch_wav(cfg.positives_path / "take_0000.wav")
    status = inspect_project(cfg)
    calls: list[tuple[ForgeConfig, Path, int]] = []

    class Result:
        imported_count = 2
        available_count = 4
        imported_paths = [cfg.positives_path / "imported_0000.wav", cfg.positives_path / "imported_0001.wav"]

    def fake_import_positive_samples(config: ForgeConfig, source_dir: Path, *, limit: int):
        calls.append((config, source_dir, limit))
        return Result()

    monkeypatch.setattr(dashboard, "import_positive_samples", fake_import_positive_samples)
    fake = CaptureFakeSt(pressed={"Import 2 existing wake-phrase samples"})

    dashboard._render_capture_step(fake, cfg, status)

    assert "Record wake-phrase take 2 of 3" not in fake.buttons
    assert calls == [(cfg, source, 2)]
    assert any("Imported 2 existing wake-phrase samples" in message for message in fake.successes)
    assert fake.session_state["rerun_requested"] is True


def test_capture_step_disables_import_when_existing_folder_is_missing(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path / "project"),
        record_positives=3,
        record_negatives=10,
        sample_source_dir=str(tmp_path / "missing"),
    )
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    assert "Import 3 existing wake-phrase samples" in fake.buttons
    assert fake.button_kwargs["Import 3 existing wake-phrase samples"]["disabled"] is True
    assert any("Existing positive sample folder is not available" in caption for caption in fake.captions)


def test_capture_step_imports_existing_negative_samples_instead_of_recording(monkeypatch, tmp_path):
    source = tmp_path / "existing negatives"
    source.mkdir()
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path / "project"),
        record_positives=3,
        record_negatives=4,
        negative_source_dir=str(source),
    )
    for i in range(3):
        _touch_wav(cfg.positives_path / f"take_{i:04d}.wav")
    _touch_wav(cfg.negatives_path / "neg_0000.wav")
    status = inspect_project(cfg)
    calls: list[tuple[ForgeConfig, Path, int]] = []

    class Result:
        imported_count = 3
        available_count = 5
        skipped_paths = ()
        imported_paths = [cfg.negatives_path / f"external_neg_{i:04d}.wav" for i in range(3)]

    def fake_import_negative_audio(config: ForgeConfig, *, source_dir: Path, kind: str, limit: int, **kwargs):
        calls.append((config, source_dir, limit))
        assert kind == "background"
        return Result()

    monkeypatch.setattr(dashboard, "import_negative_audio", fake_import_negative_audio)
    fake = CaptureFakeSt(pressed={"Import 3 existing negative samples"})

    dashboard._render_capture_step(fake, cfg, status)

    assert "Record counter-example take 2 of 4" not in fake.buttons
    assert calls == [(cfg, source, 3)]
    assert any("Imported 3 existing negative samples" in message for message in fake.successes)
    assert fake.session_state["rerun_requested"] is True


def test_capture_step_disables_negative_import_when_existing_folder_is_missing(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
        project_dir=str(tmp_path / "project"),
        record_positives=3,
        record_negatives=4,
        negative_source_dir=str(tmp_path / "missing"),
    )
    for i in range(3):
        _touch_wav(cfg.positives_path / f"take_{i:04d}.wav")
    status = inspect_project(cfg)
    fake = CaptureFakeSt()

    dashboard._render_capture_step(fake, cfg, status)

    assert "Import 4 existing negative samples" in fake.buttons
    assert fake.button_kwargs["Import 4 existing negative samples"]["disabled"] is True
    assert any("Existing negative sample folder is not available" in caption for caption in fake.captions)


def test_capture_step_shows_visible_generation_targets_after_required_takes(tmp_path):
    cfg = ForgeConfig(
        wake_phrase="Hey Nova",
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
        wake_phrase="Hey Nova",
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
    assert fake.button_kwargs["Wipe local project data"]["disabled"] is True


def test_dashboard_progress_reset_clears_wizard_state_without_wiping(monkeypatch):
    calls: list[ForgeConfig] = []

    def fake_reset_project(config: ForgeConfig):
        calls.append(config)
        return []

    monkeypatch.setattr(dashboard, "reset_project", fake_reset_project)
    fake = ResetFakeSt(checked=False, pressed={"Reset dashboard progress"})
    fake.session_state[dashboard.DASHBOARD_STEP_KEY] = "capture"
    fake.session_state[dashboard.LAST_CAPTURED_TAKE_KEY] = {"path": "old.wav"}

    dashboard._render_dashboard_progress_reset(fake)

    assert calls == []
    assert dashboard.DASHBOARD_STEP_KEY not in fake.session_state
    assert dashboard.LAST_CAPTURED_TAKE_KEY not in fake.session_state
    assert "Project files were not deleted" in fake.session_state[dashboard.RESET_MESSAGE_KEY]
    assert fake.session_state["rerun_requested"] is True


def test_start_over_controls_wipe_project_and_clear_dashboard_state(monkeypatch, tmp_path):
    cfg = ForgeConfig(wake_phrase="Hey Nova", project_dir=str(tmp_path))
    calls: list[ForgeConfig] = []

    def fake_reset_project(config: ForgeConfig):
        calls.append(config)
        return [config.project_path / "forge_config.json", config.samples_path]

    monkeypatch.setattr(dashboard, "reset_project", fake_reset_project)
    fake = ResetFakeSt(checked=True, pressed={"Wipe local project data"})
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
    assert "Progress" not in rendered
    assert "Next step" not in rendered
    assert "Run checklist" in rendered
    assert "Start" in rendered
    assert "1. Project folder" in rendered
    assert "2. Wake phrase" in rendered
    assert "5. Capture examples" in rendered
    assert "6. Review samples" in rendered
    assert "7. Train and test" in rendered
    assert "forge-step-box" in rendered
    assert "forge-step-active" in rendered
    assert "forge-step-pending" in rendered
    assert "✅" not in rendered
    assert "⏳" not in rendered


def test_makefile_defaults_to_dashboard_with_cli_fallback():
    makefile = Path("Makefile").read_text()

    assert "ENGINE ?= qwentts" in makefile
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
    assert data["project"]["scripts"]["wakeword-forge-dashboard"] == "forge.dashboard:main"
