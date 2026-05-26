"""Streamlit dashboard for wakeword-forge.

The module is intentionally import-light: tests can import helper functions without
requiring Streamlit, while the app imports Streamlit only when launched.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import math
import shlex
import subprocess
import sys
import wave
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Callable, cast

from forge.config import (
    CONFUSABLE_NEGATIVE_TARGET,
    ForgeConfig,
    MIN_NEGATIVES,
    MIN_POSITIVES,
    normalize_phrases,
)
from forge.negative_ingestion import import_negative_audio
from forge.project import (
    SUPPORTED_AUDIO_EXTENSIONS,
    ensure_project_dirs,
    import_positive_samples,
    inspect_project,
    load_or_create_config,
    reset_project,
    save_config,
)
from forge.update_check import UpdateRecommendation, check_for_updates

DEFAULT_PROJECT_DIR = Path.cwd() / "projects" / "default"
OPEN_DATA_CONFIRM_KEY = "forge_open_data_license_confirmation"
ADVANCED_DATA_CONFIRM_KEY = "forge_advanced_acoustic_license_confirmation"
RECOMMENDED_OPEN_DATA_TARGET = 200
RECOMMENDED_OPEN_DATA_MODE = "Use recommended open-source data"
MANUAL_OPEN_DATA_MODE = "Use my own local folder"
SKIP_OPEN_DATA_MODE = "Skip external background data"
MANUAL_ADVANCED_ACOUSTIC_MODE = "Use my own local folders"
SKIP_ADVANCED_ACOUSTIC_MODE = "Skip advanced acoustic folders"
BACKGROUND_NOISE_DATA_SOURCE_OPTIONS = [
    RECOMMENDED_OPEN_DATA_MODE,
    MANUAL_OPEN_DATA_MODE,
    SKIP_OPEN_DATA_MODE,
]
ADVANCED_ACOUSTIC_DATA_SOURCE_OPTIONS = [
    RECOMMENDED_OPEN_DATA_MODE,
    MANUAL_ADVANCED_ACOUSTIC_MODE,
    SKIP_ADVANCED_ACOUSTIC_MODE,
]
RECOMMENDED_OPEN_DATA_LICENSE_NOTICE = """Recommended open-source data may include:

- Mozilla Common Voice speech clips via Hugging Face/datasets. Common Voice clips are distributed under a public-domain/CC0-style license, but you should still review Mozilla's current dataset terms before redistribution or commercial use.
- ESC-50 environmental sound clips. ESC-50 is CC BY-NC 3.0, which requires attribution and is non-commercial.
- Locally generated synthetic silence/noise clips with no third-party dataset license.

Only import datasets you are allowed to use for your project. wakeword-forge preserves this as local project data; you are responsible: verify the dataset licenses for your deployment context.
""".strip()
RECOMMENDED_ADVANCED_ACOUSTIC_LICENSE_NOTICE = """Recommended advanced acoustic data may include:

- Locally generated synthetic room impulse responses for small-room/reverb simulation.
- Locally generated short transient clips for click, tap, and brief household-noise robustness.
- Locally generated low-frequency rumble clips for fan, vehicle, and machinery-style robustness.

These generated assets are stored inside this project and are not uploaded by default. If you replace them with third-party/open datasets later, verify the dataset licenses before redistribution or deployment.
""".strip()
TTS_ENGINE_HELP = """TTS means text-to-speech: generated wake-phrase clips used to add voice variety.

- QwenTTS: most natural speech and the recommended choice when you have compatible hardware, but it is slower and needs heavier model dependencies.
- Kokoro: lighter fallback that is easier to run on typical machines, with less natural prosody.
- Piper: small offline fallback for fast/local generation; quality is more robotic.
- none: skip generated speech and train only from recorded/imported samples.
""".strip()


def make_command(
    action: str,
    project_dir: Path | str,
    *,
    phrase: str | None = None,
    n: int | None = None,
    engine: str | None = None,
) -> str:
    """Return a copy-pasteable Make command for users who prefer the CLI."""
    command = f"make {action} DIR={shlex.quote(str(project_dir))}"
    if phrase is not None:
        command += f" PHRASE={shlex.quote(phrase)}"
    if n is not None:
        command += f" N={n}"
    if engine is not None:
        command += f" ENGINE={shlex.quote(engine)}"
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wakeword-forge Streamlit dashboard")
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_PROJECT_DIR),
        help="Project directory to inspect or create.",
    )
    return parser.parse_args(argv)


def dashboard_script_path() -> Path:
    return Path(__file__).resolve()


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    except Exception:
        return False
    return get_script_run_ctx() is not None


def launch_streamlit(project_dir: Path | str = DEFAULT_PROJECT_DIR) -> int:
    """Launch this file through Streamlit's runner."""
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            "Streamlit is not installed. Install it with `pip install wakeword-forge[ui]` "
            "or use `make install` from a checkout."
        )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_script_path()),
        "--",
        "--dir",
        str(project_dir),
    ]
    return subprocess.call(cmd)


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entrypoint: launch Streamlit, or render when already inside it."""
    args = parse_args(argv)
    if _running_inside_streamlit():
        run_app(args.dir)
        return
    raise SystemExit(launch_streamlit(args.dir))


def _css() -> str:
    return """
    <style>
    :root {
        --forge-primary: #58c7ff;
        --forge-primary-strong: #2ea8ff;
        --forge-success: #40d689;
        --forge-active: #f5c654;
        --forge-danger: #ff5858;
        --forge-ink: #111317;
        --forge-text: #fff2df;
        --forge-muted: rgba(255, 242, 223, 0.66);
        --forge-panel: rgba(25, 28, 34, 0.74);
    }
    .stApp {
        color: var(--forge-text);
        background:
            radial-gradient(circle at top left, rgba(88, 199, 255, 0.14), transparent 34rem),
            linear-gradient(135deg, #0f1115 0%, #171a20 55%, #0d0f12 100%);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp label, .stApp span {
        color: var(--forge-text);
    }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {
        color: var(--forge-muted);
    }
    [data-testid="stHeader"] {
        background: #111317;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111317 0%, #191b22 100%);
        border-right: 1px solid rgba(88, 199, 255, 0.16);
    }
    [data-testid="stSidebar"] * {
        color: var(--forge-text);
    }
    [data-testid="stTextInput"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        background: rgba(17, 19, 23, 0.82);
        border-color: rgba(88, 199, 255, 0.24);
        caret-color: var(--forge-primary);
        color: var(--forge-text);
    }
    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        background: rgba(17, 19, 23, 0.82) !important;
        border-color: rgba(88, 199, 255, 0.24) !important;
        color: var(--forge-text);
    }
    div[data-baseweb="input"] [data-baseweb="base-input"] {
        background: rgba(17, 19, 23, 0.82) !important;
        border-color: rgba(88, 199, 255, 0.24) !important;
        color: var(--forge-text) !important;
    }
    div[data-baseweb="input"] [data-baseweb="base-input"] input {
        background: transparent !important;
        color: var(--forge-text) !important;
    }
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {
        background: rgba(17, 19, 23, 0.82) !important;
        border-color: rgba(88, 199, 255, 0.24) !important;
        color: var(--forge-text) !important;
    }
    [data-testid="stNumberInputStepDown"] svg,
    [data-testid="stNumberInputStepUp"] svg {
        fill: var(--forge-text) !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within {
        border-color: var(--forge-primary-strong) !important;
        box-shadow: 0 0 0 0.14rem rgba(88, 199, 255, 0.32) !important;
    }
    div[data-baseweb="select"] > div {
        background: rgba(17, 19, 23, 0.82) !important;
        border-color: rgba(88, 199, 255, 0.24) !important;
        color: var(--forge-text) !important;
    }
    div[data-baseweb="select"] [role="combobox"] {
        background: transparent !important;
        color: var(--forge-text) !important;
    }
    div[data-baseweb="select"] svg {
        color: var(--forge-text) !important;
        fill: var(--forge-text) !important;
    }
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
        background: var(--forge-primary-strong) !important;
        border-color: var(--forge-primary-strong) !important;
    }
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-child > div {
        background: var(--forge-text) !important;
    }
    [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:focus-visible) > div:first-child {
        box-shadow: 0 0 0 0.14rem rgba(88, 199, 255, 0.32) !important;
    }
    .forge-hero {
        border: 1px solid rgba(88, 199, 255, 0.22);
        border-radius: 24px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(88,199,255,0.10), rgba(17,19,23,0.92));
        box-shadow: 0 18px 70px rgba(0,0,0,0.34);
    }
    .forge-kicker {
        color: var(--forge-primary);
        font-size: 0.8rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .forge-title {
        color: #fff2df;
        font-size: clamp(2.1rem, 5vw, 4.4rem);
        line-height: 0.92;
        font-weight: 900;
        margin: 0;
    }
    .forge-subtitle {
        color: rgba(255, 242, 223, 0.76);
        font-size: 1.05rem;
        margin-top: 0.7rem;
        max-width: 58rem;
    }
    .forge-card-grid {
        align-items: stretch;
        display: grid;
        gap: 1rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 1rem 0;
    }
    .forge-card {
        background: rgba(17, 19, 23, 0.76);
        border: 1px solid rgba(210, 168, 93, 0.22);
        border-radius: 18px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        height: 9.5rem;
        padding: 1rem;
        width: 100%;
    }
    .forge-card-label {
        color: rgba(255, 242, 223, 0.58);
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .forge-card-value {
        color: #fff2df;
        font-size: 2rem;
        font-weight: 850;
        margin-top: 0.2rem;
    }
    .forge-card-note {
        color: rgba(255, 242, 223, 0.65);
        font-size: 0.88rem;
        margin-top: 0.3rem;
    }
    .forge-data-source-card {
        border: 1px solid rgba(64, 214, 137, 0.34);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(64, 214, 137, 0.15), rgba(17, 19, 23, 0.76));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        margin: -0.25rem 0 0.65rem;
        padding: 0.9rem 1rem;
    }
    .forge-data-source-title {
        color: var(--forge-text);
        font-size: 1.02rem;
        font-weight: 830;
        margin-top: 0;
    }
    .forge-data-source-path {
        color: rgba(255, 242, 223, 0.76);
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.8rem;
        line-height: 1.55;
        margin-top: 0.38rem;
        overflow-wrap: anywhere;
    }
    .forge-subsection {
        border-top: 1px solid rgba(255, 242, 223, 0.10);
        margin: 1.1rem 0 0.45rem;
        padding-top: 0.82rem;
    }
    .forge-subsection-title {
        color: var(--forge-text);
        font-size: 1.02rem;
        font-weight: 850;
        letter-spacing: 0.01em;
    }
    .forge-subsection-note {
        color: var(--forge-muted);
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 0.18rem;
    }
    .forge-step-box {
        border: 1px solid rgba(255, 242, 223, 0.16);
        border-left-width: 0.38rem;
        border-radius: 13px;
        padding: 0.72rem 0.82rem;
        margin: 0.46rem 0;
        background: rgba(255, 242, 223, 0.055);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .forge-step-label {
        color: #fff2df;
        font-weight: 800;
        font-size: 0.92rem;
    }
    .forge-step-state {
        display: inline-block;
        margin-top: 0.22rem;
        color: rgba(255, 242, 223, 0.74);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .forge-step-note {
        color: rgba(255, 242, 223, 0.66);
        font-size: 0.78rem;
        margin-top: 0.22rem;
    }
    .forge-step-done {
        border-color: color-mix(in srgb, var(--forge-success) 72%, transparent);
        background: rgba(64, 214, 137, 0.13);
    }
    .forge-step-active {
        border-color: color-mix(in srgb, var(--forge-active) 82%, transparent);
        background: rgba(245, 198, 84, 0.14);
    }
    .forge-step-pending {
        border-color: rgba(143, 151, 164, 0.54);
        background: rgba(143, 151, 164, 0.10);
    }
    .forge-step-issue {
        border-color: rgba(255, 88, 88, 0.82);
        background: rgba(255, 88, 88, 0.13);
    }
    .forge-step-hint {
        color: var(--forge-muted);
        font-size: 0.94rem;
        margin: -0.2rem 0 1rem;
    }
    .stButton > button[kind="secondary"] {
        border-color: rgba(255, 242, 223, 0.22);
        background: rgba(255, 242, 223, 0.06);
        color: var(--forge-text);
    }
    .stButton > button[kind="primary"] {
        border-color: var(--forge-primary-strong);
        background: linear-gradient(135deg, var(--forge-primary), var(--forge-primary-strong));
        color: #111317;
        font-weight: 850;
    }
    </style>
    """


def _card(label: str, value: str, note: str) -> str:
    return (
        '<div class="forge-card">'
        f'<div class="forge-card-label">{html.escape(label)}</div>'
        f'<div class="forge-card-value">{html.escape(value)}</div>'
        f'<div class="forge-card-note">{html.escape(note)}</div>'
        "</div>"
    )


def _render_subsection(st, title: str, note: str) -> None:
    markup = (
        '<div class="forge-subsection">'
        f'<div class="forge-subsection-title">{html.escape(title)}</div>'
        f'<div class="forge-subsection-note">{html.escape(note)}</div>'
        "</div>"
    )
    markdown = getattr(st, "markdown", None)
    if callable(markdown):
        markdown(markup, unsafe_allow_html=True)
    else:
        st.caption(f"{title}: {note}")


def _run_blocking_action(label: str, action: Callable[[], object]) -> None:
    import streamlit as st

    try:
        with st.spinner(label):
            action()
    except Exception as exc:  # pragma: no cover - UI error display path.
        st.error(str(exc))
        st.exception(exc)
        return
    st.success(f"Finished: {label}")
    st.rerun()


DASHBOARD_STEP_KEY = "forge_dashboard_step"
LAST_CAPTURED_TAKE_KEY = "forge_last_captured_take"
PHRASE_ALIAS_COUNT_KEY = "forge_phrase_alias_count"
RESET_MESSAGE_KEY = "forge_reset_message"
UPDATE_CHECK_STATE_KEY = "forge_update_check"
WIZARD_STEPS = (
    "intro",
    "workspace",
    "phrase",
    "recording",
    "augmentation",
    "capture",
    "review",
    "train",
    "done",
)
WIZARD_STEP_LABELS = {
    "intro": "Start",
    "workspace": "1. Project folder",
    "phrase": "2. Wake phrase",
    "recording": "3. Recording plan",
    "augmentation": "4. Augmentation plan",
    "capture": "5. Capture examples",
    "review": "6. Review samples",
    "train": "7. Train and test",
    "done": "8. Accept model",
}
STEP_HINTS = {
    "workspace": "Confirm where wakeword-forge stores this project's config, samples, generated audio, and model output.",
    "phrase": "Use one primary phrase. Add aliases only for variants you actually say.",
    "recording": "Choose capture targets and whether positives come from the mic or an existing folder.",
    "augmentation": "Pick generated speech and acoustic variation before creating extra examples.",
    "capture": "Record or import examples one item at a time, then fill hard/background negatives.",
    "review": "Spot-check samples before training; approvals go stale if files change.",
    "train": "Train only after review gates are current, then run a live quality check.",
    "done": "Use the accepted model and run a final mic test in your runtime environment.",
}
WORKSPACE_FIELDS = ("project_dir",)
PHRASE_FIELDS = ("wake_phrase", "wake_phrases")
RECORDING_PLAN_FIELDS = (
    "record_positives",
    "record_negatives",
    "record_duration",
    "sample_source_dir",
    "negative_source_dir",
)
GENERATED_AUGMENTATION_FIELDS = (
    "use_tts_augmentation",
    "tts_variants",
    "tts_engine",
)
TRAINING_AUGMENTATION_FIELDS = (
    "training_augmentation_enabled",
    "training_augmentation_preset",
    "regular_negative_augmentation_preset",
    "use_spectrogram_augmentation",
    "augmentation_noise_dir",
    "augmentation_ir_dir",
    "augmentation_short_noise_dir",
    "augmentation_truck_noise_dir",
)


def _fields_changed(before: ForgeConfig, after: ForgeConfig, fields: tuple[str, ...]) -> bool:
    return any(getattr(before, field) != getattr(after, field) for field in fields)


def _clear_training_checkpoints(config: ForgeConfig) -> None:
    """Clear trained-model approval state after upstream wizard choices change."""

    config.trained_eer = None
    config.trained_sample_fingerprint = ""
    config.quality_check_passed = False
    config.model_accepted = False
    config.quality_checked_model_path = ""
    config.quality_checked_model_fingerprint = ""
    config.accepted_model_fingerprint = ""
    config.quality_positive_hits = 0
    config.quality_positive_trials = 0
    config.quality_false_triggers = 0
    config.quality_score_min = None
    config.quality_score_max = None


def _clear_generated_review_checkpoints(config: ForgeConfig) -> None:
    config.generated_review_approved = False
    config.generated_review_fingerprint = ""
    _clear_training_checkpoints(config)


def _clear_all_review_checkpoints(config: ForgeConfig) -> None:
    config.sample_review_approved = False
    config.sample_review_fingerprint = ""
    _clear_generated_review_checkpoints(config)


def _apply_step_change_invalidations(
    previous_config: ForgeConfig,
    updated_config: ForgeConfig,
    edited_step: str,
) -> ForgeConfig:
    """Return updated_config with stale downstream approvals cleared after a wizard edit."""

    updated = replace(updated_config)
    if edited_step == "workspace" and _fields_changed(previous_config, updated, WORKSPACE_FIELDS):
        _clear_all_review_checkpoints(updated)
    elif edited_step == "phrase" and _fields_changed(previous_config, updated, PHRASE_FIELDS):
        _clear_all_review_checkpoints(updated)
    elif edited_step == "recording" and _fields_changed(previous_config, updated, RECORDING_PLAN_FIELDS):
        _clear_all_review_checkpoints(updated)
    elif edited_step == "augmentation":
        if _fields_changed(previous_config, updated, GENERATED_AUGMENTATION_FIELDS):
            _clear_generated_review_checkpoints(updated)
        if _fields_changed(previous_config, updated, TRAINING_AUGMENTATION_FIELDS):
            _clear_training_checkpoints(updated)
    return updated


def _cached_update_recommendation(
    st,
    checker: Callable[[], UpdateRecommendation] = check_for_updates,
) -> UpdateRecommendation:
    cached = st.session_state.get(UPDATE_CHECK_STATE_KEY)
    if isinstance(cached, UpdateRecommendation):
        return cached
    recommendation = checker()
    st.session_state[UPDATE_CHECK_STATE_KEY] = recommendation
    return recommendation


def _count_supported_audio_files(source_path: Path) -> int:
    return len(
        [
            path
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ]
    )


def _render_update_notice(st, recommendation: UpdateRecommendation | None = None) -> None:
    recommendation = recommendation or _cached_update_recommendation(st)
    if not recommendation.needs_update:
        return

    st.warning(recommendation.message)
    st.caption("Update before starting a new training run if you want the latest fixes and workflow changes.")
    st.code(recommendation.update_command, language="bash")
    if recommendation.detail_url:
        st.caption(f"Compare: {recommendation.detail_url}")


def _set_wizard_step(st, step: str) -> None:
    if step not in WIZARD_STEPS:
        raise ValueError(f"Unknown dashboard wizard step: {step}")
    st.session_state[DASHBOARD_STEP_KEY] = step
    if step != "capture":
        st.session_state.pop(LAST_CAPTURED_TAKE_KEY, None)


def _navigate_to_wizard_step(st, step: str) -> None:
    _set_wizard_step(st, step)
    st.rerun()


def _previous_wizard_step(step: str) -> str | None:
    if step not in WIZARD_STEPS:
        return None
    index = WIZARD_STEPS.index(step)
    if index <= 0:
        return None
    return WIZARD_STEPS[index - 1]


def _render_step_back_navigation(st, current_step: str) -> None:
    previous_step = _previous_wizard_step(current_step)
    if previous_step is None:
        return
    st.caption(
        "Need to fix an earlier choice? Go back, adjust it, then confirm again. "
        "Changed plans refresh stale review/training checkpoints without deleting files."
    )
    label = "Back"
    if st.button(label, type="secondary", use_container_width=False):
        _navigate_to_wizard_step(st, previous_step)


def _render_wizard_action_row(
    st,
    current_step: str,
    primary_label: str,
    *,
    disabled: bool = False,
    primary_type: str = "primary",
) -> bool:
    """Render the bottom [Back | primary action] row for wizard steps."""

    previous_step = _previous_wizard_step(current_step)
    if previous_step is None:
        return bool(
            st.button(
                primary_label,
                type=primary_type,
                disabled=disabled,
                use_container_width=True,
            )
        )

    back_col, primary_col = st.columns(2)
    with back_col:
        back_label = "Back"
        if st.button(back_label, type="secondary", use_container_width=True):
            _navigate_to_wizard_step(st, previous_step)
            return False
    with primary_col:
        return bool(
            st.button(
                primary_label,
                type=primary_type,
                disabled=disabled,
                use_container_width=True,
            )
        )


def _confirm_step(
    st,
    config: ForgeConfig,
    next_step: str,
    message: str,
    *,
    previous_config: ForgeConfig | None = None,
    edited_step: str | None = None,
) -> None:
    if previous_config is not None and edited_step is not None:
        config = _apply_step_change_invalidations(previous_config, config, edited_step)
    ensure_project_dirs(config)
    save_config(config)
    _set_wizard_step(st, next_step)
    st.success(message)
    st.rerun()


def _configured_primary_and_aliases(config: ForgeConfig) -> tuple[str, list[str]]:
    phrases = list(config.phrase_options)
    if not phrases:
        return "", []
    return phrases[0], phrases[1:]


def _render_step_guidance(st, step: str) -> None:
    """Render compact current-step help without competing with the checklist."""

    hint = STEP_HINTS.get(step)
    if not hint:
        return
    st.caption(hint)


def _render_intro_step(st, config: ForgeConfig) -> None:
    """Render the landing card before asking for project setup details."""

    st.markdown(
        """
        <div class="forge-hero">
          <div class="forge-kicker">local wake-word forge</div>
          <h1 class="forge-title">Train the trigger. Keep the voice.</h1>
          <div class="forge-subtitle">
            Dashboard-first workflow for recording, synthesis, hard negatives,
            training, export, and local mic testing. No account. No cloud upload.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    cards = [
        _card("Private", "Local", "Recordings stay in the project folder."),
        _card("Guided", "Wizard", "One decision at a time, then explicit review."),
        _card("Flexible", "Aliases", "Train one detector for one or more trigger phrases."),
    ]
    st.markdown(
        f'<div class="forge-card-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Begin", type="primary", use_container_width=True):
        _set_wizard_step(st, "workspace")
        st.rerun()


def _render_workspace_step(st, config: ForgeConfig) -> ForgeConfig:
    """Render only the project directory confirmation step."""

    st.subheader("1. Project folder")
    st.caption("Confirm where this project stores config, samples, generated audio, cache, and model output.")
    project_dir = st.text_input(
        "Project directory",
        value=str(config.project_path),
        help="Use a disposable project folder for experiments. Existing unrelated files are preserved by dashboard reset actions.",
    )
    updated = replace(config, project_dir=project_dir.strip() or str(DEFAULT_PROJECT_DIR))
    if _render_wizard_action_row(st, "workspace", "Confirm project folder"):
        _confirm_step(
            st,
            updated,
            "phrase",
            "Project folder confirmed.",
            previous_config=config,
            edited_step="workspace",
        )
    return updated


def _render_phrase_step(st, config: ForgeConfig) -> ForgeConfig:
    """Render the primary wake phrase plus optional alias inputs."""

    st.subheader("2. Wake phrase")
    st.caption("Start with the exact phrase you will say. Add aliases only if you truly use alternate wording.")
    primary, aliases = _configured_primary_and_aliases(config)
    alias_count = max(len(aliases), int(st.session_state.get(PHRASE_ALIAS_COUNT_KEY, len(aliases))))

    primary_text = st.text_input(
        "Primary wake phrase",
        value=primary,
        help="The main phrase that should trigger the detector.",
    )
    alias_inputs: list[str] = []
    for index in range(alias_count):
        default = aliases[index] if index < len(aliases) else ""
        alias_inputs.append(
            st.text_input(
                f"Alias {index + 1}",
                value=default,
                help="Optional alternate phrase, such as another way you naturally call the same assistant.",
            )
        )

    if st.button("Add another phrase", type="secondary", use_container_width=False):
        st.session_state[PHRASE_ALIAS_COUNT_KEY] = alias_count + 1
        st.rerun()
        return config

    phrases = normalize_phrases((primary_text, *alias_inputs))
    updated = replace(
        config,
        wake_phrase=phrases[0] if phrases else "",
        wake_phrases=list(phrases[1:]),
    )
    if _render_wizard_action_row(
        st,
        "phrase",
        "Confirm wake phrase",
        disabled=not updated.wake_phrase,
    ):
        _confirm_step(
            st,
            updated,
            "recording",
            "Wake phrase confirmed.",
            previous_config=config,
            edited_step="phrase",
        )
    return updated


def _render_recording_step(st, config: ForgeConfig) -> ForgeConfig:
    """Render only recording/import parameters for the second wizard step."""

    st.subheader("2. Recording plan")
    st.caption("Use a microphone or import existing folders of wake-phrase and non-wakeword clips.")
    source_options = ("Record with microphone", "Import existing folder")
    source_mode = st.selectbox(
        "Positive sample source",
        options=source_options,
        index=1 if config.sample_source_dir else 0,
    )
    sample_source_dir = ""
    record_positives = max(config.record_positives, MIN_POSITIVES)
    if source_mode == "Import existing folder":
        sample_source_dir = st.text_input(
            "Existing positive sample folder",
            value=config.sample_source_dir,
            help="Folder containing existing wake-phrase audio clips. Supported formats: WAV, FLAC, OGG.",
        ).strip()
        source_path = Path(sample_source_dir).expanduser() if sample_source_dir else None
        if source_path is not None and source_path.is_dir():
            record_positives = _count_supported_audio_files(source_path)
            st.caption(f"Found {record_positives} existing wake-phrase audio files in that folder.")
        elif source_path is not None:
            st.caption("The positive target will come from that folder once it is available.")
    else:
        record_positives = st.number_input(
            "Target positive examples",
            min_value=MIN_POSITIVES,
            max_value=200,
            value=record_positives,
        )

    negative_source_mode = st.selectbox(
        "Negative sample source",
        options=source_options,
        index=1 if config.negative_source_dir else 0,
    )
    negative_source_dir = ""
    record_negatives = max(config.record_negatives, MIN_NEGATIVES)
    if negative_source_mode == "Import existing folder":
        negative_source_dir = st.text_input(
            "Existing negative sample folder",
            value=config.negative_source_dir,
            help="Folder containing existing non-wakeword audio clips. Supported formats: WAV, FLAC, OGG.",
        ).strip()
        negative_source_path = Path(negative_source_dir).expanduser() if negative_source_dir else None
        if negative_source_path is not None and negative_source_path.is_dir():
            record_negatives = _count_supported_audio_files(negative_source_path)
            st.caption(f"Found {record_negatives} existing negative audio files in that folder.")
        elif negative_source_path is not None:
            st.caption("The negative target will come from that folder once it is available.")
    else:
        record_negatives = st.number_input(
            "Target negative recordings",
            min_value=MIN_NEGATIVES,
            max_value=200,
            value=record_negatives,
        )
    record_duration = float(config.record_duration)
    if source_mode == "Record with microphone" or negative_source_mode == "Record with microphone":
        record_duration = st.number_input(
            "Seconds per take",
            min_value=1.0,
            max_value=8.0,
            value=record_duration,
            step=0.25,
        )

    updated = replace(
        config,
        record_positives=int(record_positives),
        record_negatives=int(record_negatives),
        record_duration=float(record_duration),
        sample_source_dir=sample_source_dir,
        negative_source_dir=negative_source_dir,
    )
    if _render_wizard_action_row(st, "recording", "Confirm recording plan"):
        _confirm_step(
            st,
            updated,
            "augmentation",
            "Recording plan confirmed.",
            previous_config=config,
            edited_step="recording",
        )
    return updated


def _recommended_open_data_dir(config: ForgeConfig) -> Path:
    return config.project_path / "sources" / "recommended_open_audio" / "background_noise"


def _recommended_advanced_acoustic_dirs(config: ForgeConfig) -> dict[str, Path]:
    base = config.project_path / "sources" / "recommended_open_audio"
    return {
        "ir": base / "room_impulse_responses",
        "short_noise": base / "short_noises",
        "low_frequency": base / "low_frequency_noises",
    }


def _format_markdown_path(path: Path) -> str:
    escaped = str(path).replace("`", "\\`")
    return f"`{escaped}`"


def _format_markdown_path_list(paths: Sequence[tuple[str, Path]], *, heading: str | None = "**Target folders:**") -> str:
    lines = [heading] if heading else []
    for label, path in paths:
        lines.append(f"- **{label}:**\n  {_format_markdown_path(path)}")
    return "\n".join(lines)


def _background_noise_data_mode(config: ForgeConfig) -> str:
    noise_dir = str(config.augmentation_noise_dir).strip()
    if not noise_dir:
        return SKIP_OPEN_DATA_MODE
    if Path(noise_dir).expanduser() == _recommended_open_data_dir(config):
        return RECOMMENDED_OPEN_DATA_MODE
    return MANUAL_OPEN_DATA_MODE


def _advanced_acoustic_data_mode(config: ForgeConfig) -> str:
    values = (
        str(config.augmentation_ir_dir).strip(),
        str(config.augmentation_short_noise_dir).strip(),
        str(config.augmentation_truck_noise_dir).strip(),
    )
    if not any(values):
        return SKIP_ADVANCED_ACOUSTIC_MODE

    recommended_dirs = _recommended_advanced_acoustic_dirs(config)
    if (
        Path(values[0]).expanduser() == recommended_dirs["ir"]
        and Path(values[1]).expanduser() == recommended_dirs["short_noise"]
        and Path(values[2]).expanduser() == recommended_dirs["low_frequency"]
    ):
        return RECOMMENDED_OPEN_DATA_MODE
    return MANUAL_ADVANCED_ACOUSTIC_MODE


def _recommended_advanced_acoustic_state(config: ForgeConfig) -> tuple[bool, dict[str, int], int]:
    recommended_dirs = _recommended_advanced_acoustic_dirs(config)
    is_active = (
        Path(str(config.augmentation_ir_dir)).expanduser() == recommended_dirs["ir"]
        and Path(str(config.augmentation_short_noise_dir)).expanduser() == recommended_dirs["short_noise"]
        and Path(str(config.augmentation_truck_noise_dir)).expanduser() == recommended_dirs["low_frequency"]
    )
    counts = {
        key: _count_supported_audio_files(directory) if directory.is_dir() else 0
        for key, directory in recommended_dirs.items()
    }
    return is_active, counts, sum(counts.values())


def _render_recommended_advanced_acoustic_status(
    st,
    config: ForgeConfig,
    *,
    selected: bool = False,
) -> dict[str, Path]:
    recommended_dirs = _recommended_advanced_acoustic_dirs(config)
    is_active, counts, imported_count = _recommended_advanced_acoustic_state(config)
    if is_active and imported_count:
        plural = "file" if imported_count == 1 else "files"
        st.markdown(
            '<div class="forge-data-source-card">'
            f'<div class="forge-data-source-title">Active · {imported_count} advanced acoustic audio {plural}</div>'
            '<div class="forge-data-source-path">'
            f'Room impulses ({counts["ir"]}): {html.escape(str(recommended_dirs["ir"]))}<br>'
            f'Short transients ({counts["short_noise"]}): {html.escape(str(recommended_dirs["short_noise"]))}<br>'
            f'Low-frequency rumble ({counts["low_frequency"]}): {html.escape(str(recommended_dirs["low_frequency"]))}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif is_active or selected:
        st.warning(
            "Recommended advanced acoustic data is selected, but no audio files were found yet. "
            "Use the import button below to install the project-local acoustic assets.\n\n"
            + _format_markdown_path_list(
                (
                    ("Room impulses", recommended_dirs["ir"]),
                    ("Short transients", recommended_dirs["short_noise"]),
                    ("Low-frequency rumble", recommended_dirs["low_frequency"]),
                )
            )
        )
    else:
        st.caption(
            "Recommended advanced acoustic data will be installed when selected.\n\n"
            + _format_markdown_path_list(
                (
                    ("Room impulses", recommended_dirs["ir"]),
                    ("Short transients", recommended_dirs["short_noise"]),
                    ("Low-frequency rumble", recommended_dirs["low_frequency"]),
                )
            )
        )
    return recommended_dirs


def _write_pcm16_wav(path: Path, samples: Sequence[float], sample_rate: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            clamped = max(-1.0, min(1.0, float(sample)))
            frames.extend(int(clamped * 32767).to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))
    return path


def import_recommended_advanced_acoustic_data(
    config: ForgeConfig,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[Path]:
    """Install generated recommended room/transient/rumble acoustic assets."""

    recommended_dirs = _recommended_advanced_acoustic_dirs(config)
    for directory in recommended_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback("Preparing recommended acoustic folders", 0, 3)

    sample_rate = 16_000
    impulse_samples = [0.0] * int(sample_rate * 0.35)
    for index in range(len(impulse_samples)):
        if index == 0:
            impulse_samples[index] = 1.0
        else:
            impulse_samples[index] = 0.45 * math.exp(-index / 1_250) * math.sin(index * 0.19)

    transient_samples = [0.0] * int(sample_rate * 0.35)
    for center in (900, 2_400, 3_900):
        for offset in range(-80, 81):
            index = center + offset
            if 0 <= index < len(transient_samples):
                envelope = max(0.0, 1.0 - abs(offset) / 81)
                transient_samples[index] += 0.42 * envelope * math.sin(index * 1.73)

    rumble_samples = [
        0.34 * math.sin(2 * math.pi * 55 * index / sample_rate)
        + 0.12 * math.sin(2 * math.pi * 93 * index / sample_rate)
        for index in range(int(sample_rate * 3.0))
    ]

    if progress_callback is not None:
        progress_callback("Installing room and noise assets", 1, 3)
    imported = [
        _write_pcm16_wav(recommended_dirs["ir"] / "synthetic_room_impulse_0001.wav", impulse_samples),
        _write_pcm16_wav(recommended_dirs["short_noise"] / "synthetic_short_transients_0001.wav", transient_samples),
        _write_pcm16_wav(recommended_dirs["low_frequency"] / "synthetic_low_frequency_rumble_0001.wav", rumble_samples),
    ]
    if progress_callback is not None:
        progress_callback("Recommended acoustic assets ready", 3, 3)
    return imported


def import_recommended_open_audio(
    config: ForgeConfig,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[Path]:
    """Import the recommended open-source background-data bundle into the project."""

    out_dir = _recommended_open_data_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback("Preparing recommended data folders", 0, 3)

    from forge.negatives import ensure_negatives

    if progress_callback is not None:
        progress_callback("Downloading open-source audio", 1, 3)
    files = ensure_negatives(
        out_dir=out_dir,
        target=RECOMMENDED_OPEN_DATA_TARGET,
        use_common_voice=True,
        use_esc50=True,
    )
    if progress_callback is not None:
        progress_callback("Recommended audio ready", 3, 3)
    return list(files)


def _recommended_open_data_state(config: ForgeConfig, recommended_dir: Path) -> tuple[bool, int]:
    is_active = Path(str(config.augmentation_noise_dir)).expanduser() == recommended_dir
    imported_count = _count_supported_audio_files(recommended_dir) if recommended_dir.is_dir() else 0
    return is_active, imported_count


def _render_recommended_open_data_status(st, config: ForgeConfig, recommended_dir: Path) -> None:
    is_active, imported_count = _recommended_open_data_state(config, recommended_dir)
    if is_active and imported_count:
        plural = "file" if imported_count == 1 else "files"
        st.markdown(
            '<div class="forge-data-source-card">'
            f'<div class="forge-data-source-title">Active · {imported_count} audio {plural}</div>'
            f'<div class="forge-data-source-path">{html.escape(str(recommended_dir))}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    elif is_active:
        st.warning(
            "Recommended background data is selected, but no audio files were found yet.\n\n"
            f"**Target folder:**\n{_format_markdown_path(recommended_dir)}"
        )
    else:
        st.caption(
            "Recommended background data will be stored here when selected.\n\n"
            f"**Target folder:**\n{_format_markdown_path(recommended_dir)}"
        )


def _render_recommended_open_data_confirmation(st, config: ForgeConfig, recommended_dir: Path) -> None:
    st.warning("Confirm the license notice before downloading or generating recommended background data.")
    st.markdown(RECOMMENDED_OPEN_DATA_LICENSE_NOTICE)
    st.caption(f"Import target:\n{_format_markdown_path(recommended_dir)}")
    accepted = st.checkbox(
        "I understand the dataset licenses and will verify they fit my use case before redistribution or deployment."
    )

    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", type="secondary", use_container_width=True):
            st.session_state[OPEN_DATA_CONFIRM_KEY] = False
            st.rerun()
    with confirm_col:
        confirmed = st.button(
            "Confirm",
            type="primary",
            disabled=not accepted,
            use_container_width=True,
        )
    if not confirmed:
        return

    progress_bar = st.progress(0.0, text="Preparing recommended data folders")

    def progress_callback(label: str, step: int, total: int) -> None:
        denominator = max(total, 1)
        value = max(0.0, min(1.0, step / denominator))
        progress_bar.progress(value, text=label)

    updated = replace(config, augmentation_noise_dir=str(recommended_dir))
    with st.spinner("Downloading recommended open-source background data"):
        imported = import_recommended_open_audio(updated, progress_callback=progress_callback)
    save_config(updated)
    plural = "file" if len(imported) == 1 else "files"
    st.success(f"Imported {len(imported)} recommended open-source audio {plural} into {recommended_dir}.")
    st.session_state[OPEN_DATA_CONFIRM_KEY] = False
    st.rerun()


def _render_recommended_open_data_dialog(st, config: ForgeConfig, recommended_dir: Path) -> None:
    dialog = getattr(st, "dialog", None)
    if callable(dialog):
        dialog = cast(Callable[[str], Callable[[Callable[[], None]], Callable[[], None]]], dialog)

        @dialog("Import recommended open-source data")
        def confirmation_dialog() -> None:
            _render_recommended_open_data_confirmation(st, config, recommended_dir)

        confirmation_dialog()
        return

    _render_recommended_open_data_confirmation(st, config, recommended_dir)


def _render_recommended_open_data_import(st, config: ForgeConfig) -> str:
    recommended_dir = _recommended_open_data_dir(config)
    is_active, imported_count = _recommended_open_data_state(config, recommended_dir)
    _render_recommended_open_data_status(st, config, recommended_dir)
    if is_active and imported_count:
        action_label = "Re-import or repair recommended data"
    else:
        action_label = "Import recommended open-source data"
        st.warning(
            "Recommended import includes Mozilla Common Voice, ESC-50 (CC BY-NC 3.0), "
            "and local synthetic clips; verify the dataset licenses in the confirmation popup."
        )
    if st.button(action_label, type="secondary", use_container_width=True):
        st.session_state[OPEN_DATA_CONFIRM_KEY] = True

    if st.session_state.get(OPEN_DATA_CONFIRM_KEY):
        _render_recommended_open_data_dialog(st, config, recommended_dir)
    return str(recommended_dir)


def _render_recommended_advanced_acoustic_confirmation(
    st,
    config: ForgeConfig,
    recommended_dirs: dict[str, Path],
) -> None:
    st.warning("Confirm the notice before installing recommended advanced acoustic data.")
    st.markdown(RECOMMENDED_ADVANCED_ACOUSTIC_LICENSE_NOTICE)
    st.caption(
        "Import targets:\n"
        + _format_markdown_path_list(
            (
                ("Room impulses", recommended_dirs["ir"]),
                ("Short transients", recommended_dirs["short_noise"]),
                ("Low-frequency rumble", recommended_dirs["low_frequency"]),
            ),
            heading=None,
        )
    )
    accepted = st.checkbox(
        "I understand what will be installed and will verify any replacement dataset licenses before redistribution or deployment."
    )

    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", type="secondary", use_container_width=True):
            st.session_state[ADVANCED_DATA_CONFIRM_KEY] = False
            st.rerun()
    with confirm_col:
        confirmed = st.button(
            "Confirm",
            type="primary",
            disabled=not accepted,
            use_container_width=True,
        )
    if not confirmed:
        return

    progress_bar = st.progress(0.0, text="Preparing recommended acoustic folders")

    def progress_callback(label: str, step: int, total: int) -> None:
        denominator = max(total, 1)
        value = max(0.0, min(1.0, step / denominator))
        progress_bar.progress(value, text=label)

    updated = replace(
        config,
        augmentation_ir_dir=str(recommended_dirs["ir"]),
        augmentation_short_noise_dir=str(recommended_dirs["short_noise"]),
        augmentation_truck_noise_dir=str(recommended_dirs["low_frequency"]),
    )
    with st.spinner("Installing recommended advanced acoustic data"):
        imported = import_recommended_advanced_acoustic_data(updated, progress_callback=progress_callback)
    save_config(updated)
    plural = "file" if len(imported) == 1 else "files"
    st.success(f"Installed {len(imported)} recommended advanced acoustic audio {plural}.")
    st.session_state[ADVANCED_DATA_CONFIRM_KEY] = False
    st.rerun()


def _render_recommended_advanced_acoustic_dialog(
    st,
    config: ForgeConfig,
    recommended_dirs: dict[str, Path],
) -> None:
    dialog = getattr(st, "dialog", None)
    if callable(dialog):
        dialog = cast(Callable[[str], Callable[[Callable[[], None]], Callable[[], None]]], dialog)

        @dialog("Import recommended advanced acoustic data")
        def confirmation_dialog() -> None:
            _render_recommended_advanced_acoustic_confirmation(st, config, recommended_dirs)

        confirmation_dialog()
        return

    _render_recommended_advanced_acoustic_confirmation(st, config, recommended_dirs)


def _render_recommended_advanced_acoustic_import(st, config: ForgeConfig) -> dict[str, str]:
    recommended_dirs = _render_recommended_advanced_acoustic_status(st, config, selected=True)
    is_active, _counts, imported_count = _recommended_advanced_acoustic_state(config)
    if is_active and imported_count:
        action_label = "Re-import or repair recommended advanced acoustic data"
    else:
        action_label = "Import recommended advanced acoustic data"
        st.warning(
            "Recommended acoustic import installs project-local room impulse, short transient, "
            "and low-frequency rumble clips; review the confirmation popup first."
        )
    if st.button(action_label, type="secondary", use_container_width=True):
        st.session_state[ADVANCED_DATA_CONFIRM_KEY] = True

    if st.session_state.get(ADVANCED_DATA_CONFIRM_KEY):
        _render_recommended_advanced_acoustic_dialog(st, config, recommended_dirs)

    return {key: str(value) for key, value in recommended_dirs.items()}


def _render_augmentation_step(st, config: ForgeConfig) -> ForgeConfig:
    """Render only generated-audio parameters for the third wizard step."""

    st.subheader("3. Augmentation plan")
    st.caption("Decide whether to add generated positives and hard negatives before capture/training.")

    _render_subsection(st, "Generated positives", "Optional synthetic wake-phrase clips to widen voice coverage.")
    use_tts = st.toggle("Use TTS augmentation", value=config.use_tts_augmentation)
    tts_variants = st.number_input(
        "Synthetic positive target",
        min_value=0,
        max_value=2_000,
        value=int(config.tts_variants),
        step=25,
        disabled=not use_tts,
    )
    tts_engine_options = ["qwentts", "kokoro", "piper", "none"]
    tts_engine = st.selectbox(
        "TTS engine",
        options=tts_engine_options,
        index=tts_engine_options.index(config.tts_engine)
        if config.tts_engine in set(tts_engine_options)
        else 0,
        disabled=not use_tts,
        help=TTS_ENGINE_HELP,
    )

    _render_subsection(
        st,
        "Training-time robustness",
        "Augment reviewed samples during training so the detector handles real rooms and noise.",
    )
    use_training_aug = st.toggle(
        "Use training-time acoustic augmentation",
        value=config.training_augmentation_enabled,
    )
    training_preset_options = ["standard", "light"]
    training_preset = st.selectbox(
        "Training augmentation preset",
        options=training_preset_options,
        index=training_preset_options.index(config.training_augmentation_preset)
        if config.training_augmentation_preset in set(training_preset_options)
        else 0,
        disabled=not use_training_aug,
    )
    negative_preset_options = ["light", "standard", "none"]
    regular_negative_preset = st.selectbox(
        "Background negative augmentation",
        options=negative_preset_options,
        index=negative_preset_options.index(config.regular_negative_augmentation_preset)
        if config.regular_negative_augmentation_preset in set(negative_preset_options)
        else 0,
        disabled=not use_training_aug,
    )
    use_spec_aug = st.toggle(
        "Use SpecAugment-style mel masking",
        value=config.use_spectrogram_augmentation,
        disabled=not use_training_aug,
    )

    _render_subsection(st, "Background noise source", "Choose the noise pool used by background mixing.")
    open_data_mode = st.selectbox(
        "Background noise data source",
        options=BACKGROUND_NOISE_DATA_SOURCE_OPTIONS,
        index=BACKGROUND_NOISE_DATA_SOURCE_OPTIONS.index(_background_noise_data_mode(config)),
        disabled=not use_training_aug,
        help="Use recommended open-source data, point at your own local folder, or skip external background noise mixing.",
    )
    if not use_training_aug:
        noise_dir = ""
    elif open_data_mode == RECOMMENDED_OPEN_DATA_MODE:
        noise_dir = _render_recommended_open_data_import(st, config)
    elif open_data_mode == MANUAL_OPEN_DATA_MODE:
        noise_dir = st.text_input(
            "Background noise folder",
            value=config.augmentation_noise_dir,
            disabled=not use_training_aug,
            help="Folder of local WAV noise clips to mix into augmented samples.",
        )
    else:
        noise_dir = ""
        st.caption("No external background-noise folder will be used. Synthetic waveform augmentation can still run.")

    _render_subsection(
        st,
        "Advanced acoustic folders",
        "Optional room, transient, and low-frequency acoustic assets for stronger training-time variation.",
    )
    advanced_mode = st.selectbox(
        "Advanced acoustic folder source",
        options=ADVANCED_ACOUSTIC_DATA_SOURCE_OPTIONS,
        index=ADVANCED_ACOUSTIC_DATA_SOURCE_OPTIONS.index(_advanced_acoustic_data_mode(config)),
        disabled=not use_training_aug,
        help="Use project-local recommended folders, point at your own local folders, or skip these optional acoustic assets.",
    )
    if not use_training_aug:
        ir_dir = ""
        short_noise_dir = ""
        truck_noise_dir = ""
    elif advanced_mode == RECOMMENDED_OPEN_DATA_MODE:
        advanced_dirs = _render_recommended_advanced_acoustic_import(st, config)
        ir_dir = advanced_dirs["ir"]
        short_noise_dir = advanced_dirs["short_noise"]
        truck_noise_dir = advanced_dirs["low_frequency"]
    elif advanced_mode == MANUAL_ADVANCED_ACOUSTIC_MODE:
        st.caption("Optional local folders for impulse responses, short transients, and low-frequency rumble.")
        ir_dir = st.text_input(
            "Room impulse response folder",
            value=config.augmentation_ir_dir,
            disabled=not use_training_aug,
            help="Optional folder of local WAV room impulse responses.",
        )
        short_noise_dir = st.text_input(
            "Short noise folder",
            value=config.augmentation_short_noise_dir,
            disabled=not use_training_aug,
            help="Optional folder of local WAV transient noises.",
        )
        truck_noise_dir = st.text_input(
            "Low-frequency noise folder",
            value=config.augmentation_truck_noise_dir,
            disabled=not use_training_aug,
            help="Optional folder of local WAV rumble or machinery noise.",
        )
    else:
        ir_dir = ""
        short_noise_dir = ""
        truck_noise_dir = ""
        st.caption("Advanced acoustic folders will be skipped; base waveform augmentation can still run.")

    updated = replace(
        config,
        use_tts_augmentation=bool(use_tts),
        tts_variants=int(tts_variants) if use_tts else 0,
        tts_engine=str(tts_engine) if use_tts else "none",
        training_augmentation_enabled=bool(use_training_aug),
        training_augmentation_preset=str(training_preset),
        regular_negative_augmentation_preset=str(regular_negative_preset),
        use_spectrogram_augmentation=bool(use_spec_aug) if use_training_aug else False,
        augmentation_noise_dir=str(noise_dir).strip() if use_training_aug else "",
        augmentation_ir_dir=str(ir_dir).strip() if use_training_aug else "",
        augmentation_short_noise_dir=str(short_noise_dir).strip() if use_training_aug else "",
        augmentation_truck_noise_dir=str(truck_noise_dir).strip() if use_training_aug else "",
    )
    if _render_wizard_action_row(st, "augmentation", "Confirm augmentation plan"):
        _confirm_step(
            st,
            updated,
            "capture",
            "Augmentation plan confirmed.",
            previous_config=config,
            edited_step="augmentation",
        )
    return updated


def _next_recording_path(out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = out_dir / f"{prefix}_{index:04d}.wav"
        if not candidate.exists():
            return candidate
        index += 1


def _record_one_take(*, phrase: str, out_dir: Path, duration: float, prefix: str) -> Path:
    """Record and save exactly one take for dashboard-driven capture."""

    import soundfile as sf

    from forge.config import SAMPLE_RATE
    from forge.recorder import _prepare_recorded_take, _record_take

    _ = phrase  # Phrase is displayed by the dashboard; keep it in the call contract.
    audio = _prepare_recorded_take(_record_take(duration, SAMPLE_RATE), SAMPLE_RATE)

    out_path = _next_recording_path(out_dir, prefix)
    sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
    return out_path


def _captured_take_state(path: Path, kind: str, saved_count: int, target_count: int) -> dict[str, object]:
    return {
        "path": str(path),
        "kind": kind,
        "saved_count": saved_count,
        "target_count": target_count,
    }


def _last_captured_take(st) -> tuple[Path, str, int, int] | None:
    state = getattr(st, "session_state", {}).get(LAST_CAPTURED_TAKE_KEY)
    if not isinstance(state, dict):
        return None
    try:
        path = Path(str(state["path"]))
        kind = str(state["kind"])
        saved_count = int(state["saved_count"])
        target_count = int(state["target_count"])
    except (KeyError, TypeError, ValueError):
        return None
    if not path.exists():
        return None
    return path, kind, saved_count, target_count


def _render_last_captured_take(st) -> bool:
    last_take = _last_captured_take(st)
    if last_take is None:
        return False

    path, kind, saved_count, target_count = last_take
    st.success(f"Saved {kind} take {saved_count}/{target_count}: {path.name}")
    st.caption("Replay it now. If it sounds right, move to the next recording.")
    st.button(f"Replay last {kind} take", use_container_width=True)
    st.audio(str(path), format="audio/wav")
    if st.button("Next recording", type="secondary", use_container_width=True):
        st.session_state.pop(LAST_CAPTURED_TAKE_KEY, None)
        st.rerun()
    return True


def _render_one_take_recorder(
    st,
    *,
    kind: str,
    phrase: str,
    current_count: int,
    target_count: int,
    out_dir: Path,
    duration: float,
    prefix: str,
    guidance: str | None = None,
) -> None:
    st.markdown(f"**{current_count} / {target_count} {kind} takes saved**")
    if current_count >= target_count:
        st.success(f"All {target_count} {kind} takes are recorded.")
        return

    next_take = current_count + 1
    st.caption(f"Take {next_take}: press record, speak `{phrase}`, then replay before moving on.")
    if guidance:
        st.caption(guidance)
    if st.button(
        f"Record {kind} take {next_take} of {target_count}",
        disabled=not bool(phrase),
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(f"Recording {kind} take {next_take}/{target_count}"):
                saved_path = _record_one_take(
                    phrase=phrase,
                    out_dir=out_dir,
                    duration=duration,
                    prefix=prefix,
                )
        except Exception as exc:  # pragma: no cover - UI error display path.
            st.error(str(exc))
            st.exception(exc)
            return

        st.session_state[LAST_CAPTURED_TAKE_KEY] = _captured_take_state(
            saved_path,
            kind,
            saved_count=next_take,
            target_count=target_count,
        )
        st.rerun()


def _positive_phrase_for_take(config: ForgeConfig, current_count: int) -> str:
    phrases = config.phrase_options
    if not phrases:
        return config.wake_phrase
    return phrases[current_count % len(phrases)]


_CONFUSABLE_WORD_HINTS = {
    "nova": ("novah",),
}


def _simple_words(text: str) -> tuple[str, ...]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return tuple(word for word in normalized.split() if word)


def _dedupe_ordered(items: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return tuple(deduped)


def _matches_full_trigger_example(example: str, trigger_options: tuple[tuple[str, ...], ...]) -> bool:
    example_words = _simple_words(example)
    if not example_words:
        return False
    for trigger_words in trigger_options:
        if not trigger_words:
            continue
        if example_words == trigger_words:
            return True
        if len(example_words) >= len(trigger_words):
            for start in range(0, len(example_words) - len(trigger_words) + 1):
                if example_words[start : start + len(trigger_words)] == trigger_words:
                    return True
        if len(example_words) % len(trigger_words) == 0:
            repeats = len(example_words) // len(trigger_words)
            if repeats > 1 and example_words == trigger_words * repeats:
                return True
        trigger_text = "".join(trigger_words)
        example_text = "".join(example_words)
        if len(example_text) % len(trigger_text) == 0:
            repeats = len(example_text) // len(trigger_text)
            if repeats > 1 and example_text == trigger_text * repeats:
                return True
    return False


def _filter_full_trigger_examples(examples: tuple[str, ...], trigger_phrases: tuple[str, ...]) -> tuple[str, ...]:
    trigger_options = tuple(_simple_words(phrase) for phrase in trigger_phrases)
    return tuple(
        example
        for example in examples
        if not _matches_full_trigger_example(example, trigger_options)
    )


def _negative_examples_for_phrase(phrase: str) -> tuple[str, ...]:
    words = list(_simple_words(phrase))
    if not words:
        return ()

    examples: list[str] = []
    for index, word in enumerate(words):
        for replacement in _CONFUSABLE_WORD_HINTS.get(word, ()):  # near-miss words users can say.
            near_miss = list(words)
            near_miss[index] = replacement
            examples.append(" ".join(near_miss))

    if len(words) > 1:
        examples.extend(" ".join(words[:index]) for index in range(1, len(words)))

    first_word = words[0]
    if first_word in {"okay", "ok"}:
        first_fragment = "ok"
        examples.append("ok-")
    else:
        first_fragment = first_word[: max(1, min(3, len(first_word)))]
        examples.append(f"{first_fragment}-")
    examples.append(first_fragment * 4)

    tail_words = words[1:] if len(words) > 1 else words
    for word in tail_words:
        if len(word) >= 4:
            midpoint = max(2, len(word) // 2)
            left = word[:midpoint]
            right = word[midpoint:]
            examples.extend([left, right, word, left * 3, right * 3, f"{word}-{word}"])
        else:
            examples.extend([word, word * 3, f"{word}-{word}"])

    return _filter_full_trigger_examples(_dedupe_ordered(examples), (phrase,))


def _negative_example_guidance(config: ForgeConfig) -> str:
    phrases = _phrase_list_for_generation(config)
    examples = _filter_full_trigger_examples(
        _dedupe_ordered([example for phrase in phrases for example in _negative_examples_for_phrase(phrase)]),
        phrases,
    )
    example_text = ", ".join(f"`{example}`" for example in examples[:14])
    if example_text:
        return (
            "Good counter-examples: confusable near-misses, partial utterances, "
            f"repeated fragments, and normal speech that is not the trigger. Try: {example_text}."
        )
    return (
        "Good counter-examples: confusable near-misses, partial utterances, repeated fragments, "
        "and normal speech that is not the trigger."
    )


def _phrase_list_for_generation(config: ForgeConfig) -> tuple[str, ...]:
    return config.phrase_options or normalize_phrases((config.wake_phrase,))


def _render_positive_sample_import(st, config: ForgeConfig, current_count: int, target_count: int) -> None:
    st.markdown(f"**{current_count} / {target_count} wake-phrase samples imported**")
    missing = max(0, target_count - current_count)
    source_dir = Path(config.sample_source_dir).expanduser() if config.sample_source_dir else None
    source_available = source_dir is not None and source_dir.is_dir()
    if source_dir is None:
        st.caption("Choose an existing positive sample folder in the recording plan.")
    elif not source_available:
        st.caption(f"Existing positive sample folder is not available: {source_dir}")
    else:
        st.caption(f"Import wake-phrase clips from `{source_dir}`. Source files are copied; originals stay untouched.")

    if st.button(
        f"Import {missing} existing wake-phrase samples",
        disabled=missing == 0 or not source_available,
        type="secondary",
        use_container_width=True,
    ):
        try:
            assert source_dir is not None
            with st.spinner(f"Importing {missing} existing wake-phrase samples"):
                result = import_positive_samples(config, source_dir, limit=missing)
        except Exception as exc:  # pragma: no cover - UI error display path.
            st.error(str(exc))
            st.exception(exc)
            return

        if result.imported_count:
            save_config(config)
            st.success(
                f"Imported {result.imported_count} existing wake-phrase samples "
                f"from {result.available_count} available audio files."
            )
        else:
            st.warning("No existing wake-phrase samples were imported from that folder.")
        skipped_paths = getattr(result, "skipped_paths", ())
        if skipped_paths:
            st.caption(f"Skipped {len(skipped_paths)} unreadable audio file(s).")
        st.rerun()


def _render_negative_sample_import(st, config: ForgeConfig, current_count: int, target_count: int) -> None:
    st.markdown(f"**{current_count} / {target_count} negative samples imported**")
    missing = max(0, target_count - current_count)
    source_dir = Path(config.negative_source_dir).expanduser() if config.negative_source_dir else None
    source_available = source_dir is not None and source_dir.is_dir()
    if source_dir is None:
        st.caption("Choose an existing negative sample folder in the recording plan.")
    elif not source_available:
        st.caption(f"Existing negative sample folder is not available: {source_dir}")
    else:
        st.caption(f"Import non-wakeword clips from `{source_dir}`. Source files are copied; originals stay untouched.")

    if st.button(
        f"Import {missing} existing negative samples",
        disabled=missing == 0 or not source_available,
        type="secondary",
        use_container_width=True,
    ):
        try:
            assert source_dir is not None
            with st.spinner(f"Importing {missing} existing negative samples"):
                result = import_negative_audio(
                    config,
                    source_dir=source_dir,
                    kind="background",
                    limit=missing,
                    limit_per_source=None,
                    max_chunks_per_file=1,
                )
        except Exception as exc:  # pragma: no cover - UI error display path.
            st.error(str(exc))
            st.exception(exc)
            return

        if result.imported_count:
            save_config(config)
            st.success(
                f"Imported {result.imported_count} existing negative samples "
                f"from {result.available_count} available audio files."
            )
        else:
            st.warning("No existing negative samples were imported from that folder.")
        skipped_paths = getattr(result, "skipped_paths", ())
        if skipped_paths:
            st.caption(f"Skipped {len(skipped_paths)} unreadable audio file(s).")
        st.rerun()


def _split_count(total: int, buckets: int) -> list[int]:
    if total <= 0 or buckets <= 0:
        return []
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _render_capture_step(st, config: ForgeConfig, status) -> None:
    st.subheader("4. Capture examples")
    st.caption("Capture one clip at a time: record, replay, then move to the next take.")

    if _render_last_captured_take(st):
        return

    if status.real_positives < config.record_positives:
        if config.sample_source_dir:
            _render_positive_sample_import(
                st,
                config,
                current_count=status.real_positives,
                target_count=config.record_positives,
            )
        else:
            _render_one_take_recorder(
                st,
                kind="wake-phrase",
                phrase=_positive_phrase_for_take(config, status.real_positives),
                current_count=status.real_positives,
                target_count=config.record_positives,
                out_dir=config.positives_path,
                duration=config.record_duration,
                prefix="take",
            )
        return

    if status.negatives < config.record_negatives:
        if config.negative_source_dir:
            _render_negative_sample_import(
                st,
                config,
                current_count=status.negatives,
                target_count=config.record_negatives,
            )
        else:
            _render_one_take_recorder(
                st,
                kind="counter-example",
                phrase="a near-miss, partial phrase, repeated fragment, or anything except the full trigger",
                current_count=status.negatives,
                target_count=config.record_negatives,
                out_dir=config.negatives_path,
                duration=config.record_duration,
                prefix="neg",
                guidance=_negative_example_guidance(config),
            )
        return

    if config.use_tts_augmentation and config.tts_engine != "none":
        synth_missing = max(0, config.tts_variants - status.synthetic_positives)
        if synth_missing > 0:
            if st.button(f"Generate {synth_missing} TTS positives", use_container_width=True):
                from forge.synthesizer import synthesize_positive_phrases

                _run_blocking_action(
                    "Generating synthetic positives",
                    lambda: synthesize_positive_phrases(
                        phrases=_phrase_list_for_generation(config),
                        out_dir=config.synthetic_path,
                        n=synth_missing,
                        engine=config.tts_engine,
                    ),
                )
            return

        st.success(f"Synthetic positives ready: {status.synthetic_positives}/{config.tts_variants}.")

        from forge.synthesizer import load_confusable_phrases

        phrases = _phrase_list_for_generation(config)
        partial_phrases = tuple(phrase for phrase in phrases if len(phrase.split()) >= 2)
        partial_target = status.partial_negative_target
        confusable_phrases = load_confusable_phrases(config.confusables_cache)
        confusable_target = CONFUSABLE_NEGATIVE_TARGET if confusable_phrases else 0
        partial_missing = max(0, partial_target - status.partial_negatives)
        confusable_missing = max(0, confusable_target - status.confusable_negatives)
        hard_missing = partial_missing + confusable_missing
        st.caption(
            f"Hard negatives: {status.partial_negatives}/{partial_target} partials, "
            f"{status.confusable_negatives}/{confusable_target} confusables"
        )
        if not confusable_phrases:
            st.caption(
                "No confusable phrase cache found; add confusable_variants.txt to generate "
                "confusable hard negatives."
            )
        hard_button_label = (
            f"Generate {hard_missing} hard negatives" if hard_missing else "Hard negatives ready"
        )
        if st.button(
            hard_button_label,
            disabled=hard_missing == 0,
            use_container_width=True,
        ):
            from forge.synthesizer import synthesize_confusable_negatives, synthesize_partial_negatives

            def hard_negatives() -> None:
                if partial_missing:
                    for phrase, count in zip(partial_phrases, _split_count(partial_missing, len(partial_phrases))):
                        if count:
                            synthesize_partial_negatives(
                                phrase=phrase,
                                out_dir=config.partials_path,
                                n=count,
                                engine=config.tts_engine,
                            )
                if confusable_missing:
                    synthesize_confusable_negatives(
                        phrase=config.wake_phrase,
                        out_dir=config.confusables_path,
                        cache_file=config.confusables_cache,
                        n_variants=confusable_missing,
                        engine=config.tts_engine,
                    )

            _run_blocking_action("Generating hard negatives", hard_negatives)

    background_target = status.background_negative_target
    background_missing = max(0, background_target - status.negatives)
    st.caption(f"Background negatives: {status.negatives}/{background_target}")
    background_button_label = (
        f"Fill {background_missing} background negatives"
        if background_missing
        else "Background negatives ready"
    )
    if st.button(
        background_button_label,
        disabled=background_missing == 0,
        use_container_width=True,
    ):
        from forge.negatives import ensure_negatives

        _run_blocking_action(
            "Creating background negatives",
            lambda: ensure_negatives(
                out_dir=config.negatives_path,
                target=background_target,
                use_common_voice=False,
                use_esc50=False,
            ),
        )
    if st.button("Continue to review", disabled=not status.samples_ready, type="secondary", use_container_width=True):
        _set_wizard_step(st, "review")
        st.rerun()


def _render_train_step(st, config: ForgeConfig, status) -> None:
    st.subheader("6. Train and test")
    st.caption("Train only after review gates are current. Quality checking stays explicit.")
    if _render_wizard_action_row(st, "train", "Train detector", disabled=not status.ready_to_train):
        from forge.trainer import run_training

        _run_blocking_action("Training detector", lambda: run_training(config))
    if status.has_model:
        st.code(make_command("quality-check", config.project_path), language="bash")
        st.code(make_command("accept-model", config.project_path), language="bash")
    else:
        st.caption("The quality-check and accept steps unlock after training exports wakeword.onnx.")


def _render_done_step(st, config: ForgeConfig) -> None:
    st.subheader("Model accepted")
    st.success("This project has a checked and accepted wakeword.onnx.")
    st.code(make_command("mic-test", config.project_path), language="bash")


def _default_wizard_step(status) -> str:
    if not status.wake_phrase:
        return "intro"
    if not status.samples_ready:
        return "recording"
    if status.sample_review_required or status.generated_review_required:
        return "review"
    if not status.has_model or status.quality_check_required or status.model_acceptance_required:
        return "train"
    return "done"


def _current_wizard_step(st, status) -> str:
    step = getattr(st, "session_state", {}).get(DASHBOARD_STEP_KEY)
    if step not in WIZARD_STEPS:
        step = _default_wizard_step(status)
    if not status.wake_phrase and step not in {"intro", "workspace", "phrase"}:
        return "intro"
    return step


def _render_current_wizard_step(st, config: ForgeConfig, status, current_step: str | None = None) -> None:
    step = current_step or _current_wizard_step(st, status)
    if step != "intro":
        if step not in {"workspace", "phrase", "recording", "augmentation", "review", "train"}:
            _render_step_back_navigation(st, step)
        _render_step_guidance(st, step)
    if step == "intro":
        _render_intro_step(st, config)
    elif step == "workspace":
        _render_workspace_step(st, config)
    elif step == "phrase":
        _render_phrase_step(st, config)
    elif step == "recording":
        _render_recording_step(st, config)
    elif step == "augmentation":
        _render_augmentation_step(st, config)
    elif step == "capture":
        _render_capture_step(st, config, status)
    elif step == "review":
        st.subheader("5. Review samples")
        st.caption("Approve the current sample set before training. If files change, approvals go stale.")
        _render_review_checkpoints(st, config, status)
        if _render_wizard_action_row(
            st,
            "review",
            "Continue to training",
            disabled=not status.ready_to_train,
        ):
            _set_wizard_step(st, "train")
            st.rerun()
    elif step == "train":
        _render_train_step(st, config, status)
    else:
        _render_done_step(st, config)


def _workflow_progress_fraction(status) -> float:
    if not status.wake_phrase:
        return 0.0
    if not status.samples_ready:
        return max(0.08, min(status.progress_fraction, 1.0) * 0.30)
    if status.sample_review_required:
        return 0.38
    if status.generated_review_required:
        return 0.48
    if not status.has_model:
        return 0.62
    if status.quality_check_required:
        return 0.76
    if status.model_acceptance_required:
        return 0.88
    return 1.0


def _workflow_step_box(label: str, state: str, note: str = "") -> str:
    state_labels = {
        "done": "Done",
        "active": "In progress",
        "pending": "Not started",
        "issue": "Issue",
    }
    safe_label = html.escape(label)
    safe_state = html.escape(state_labels[state])
    safe_note = f'<div class="forge-step-note">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="forge-step-box forge-step-{state}">'
        f'<div class="forge-step-label">{safe_label}</div>'
        f'<span class="forge-step-state">{safe_state}</span>'
        f"{safe_note}"
        "</div>"
    )


def _workflow_step_state(step: str, current_step: str, issue: bool = False) -> str:
    if issue:
        return "issue"
    current_index = WIZARD_STEPS.index(current_step) if current_step in WIZARD_STEPS else 0
    step_index = WIZARD_STEPS.index(step)
    if step_index < current_index:
        return "done"
    if step_index == current_index:
        return "active"
    return "pending"


def _render_progress_sidebar(st, status, current_step: str | None = None, config: ForgeConfig | None = None) -> None:
    """Render the sidebar as a compact workflow progress rail."""

    current_step = current_step or _default_wizard_step(status)
    sidebar = st.sidebar
    sidebar.subheader("Run checklist")
    step_notes = {
        "intro": "overview and start",
        "workspace": "confirm project folder",
        "phrase": f"{len(status.wake_phrases)} phrase(s)" if status.wake_phrases else "choose wake phrase",
        "recording": f"targets: {MIN_POSITIVES}+ positives / {MIN_NEGATIVES}+ negatives",
        "augmentation": f"{status.synthetic_positives} synthetic clips",
        "capture": f"{status.total_positives} positives / {status.total_negatives} negatives",
        "review": "sample + generated-audio gates",
        "train": "export ONNX and live quality check",
        "done": "accepted runtime model",
    }
    for step in WIZARD_STEPS:
        sidebar.markdown(
            _workflow_step_box(
                WIZARD_STEP_LABELS[step],
                _workflow_step_state(step, current_step),
                step_notes.get(step, ""),
            ),
            unsafe_allow_html=True,
        )
    sidebar.divider()

    sidebar.subheader("Snapshot")
    sidebar.metric("Positives", status.total_positives)
    sidebar.metric("Negatives", status.total_negatives)
    sidebar.caption(f"Project: {status.project_dir}")

    if config is not None:
        sidebar.divider()
        sidebar.subheader("Recovery")
        _render_dashboard_progress_reset(st, sidebar)
        with sidebar.expander("Wipe local project data", expanded=False):
            _render_start_over_controls(st, config, include_divider=False)


def _render_cli_fallbacks(st, config: ForgeConfig) -> None:
    st.subheader("Pure CLI fallback")
    st.caption("Every dashboard action has a boring Make command. Copy these if Streamlit is not ideal.")
    commands = [
        make_command("dashboard", config.project_path),
        make_command("cli-run", config.project_path),
        make_command("info", config.project_path),
        make_command(
            "record",
            config.project_path,
            phrase=config.wake_phrase or "Hey Nova",
            n=config.record_positives,
        ),
        make_command(
            "synth",
            config.project_path,
            phrase=config.wake_phrase or "Hey Nova",
            n=config.tts_variants,
            engine=config.tts_engine,
        ),
        make_command("review", config.project_path),
        make_command("audit", config.project_path),
        make_command("train", config.project_path),
        make_command("quality-check", config.project_path),
        make_command("accept-model", config.project_path),
        make_command("mic-test", config.project_path),
    ]
    st.code("\n".join(commands), language="bash")


def _clear_reset_session_state(st) -> None:
    st.session_state.pop(DASHBOARD_STEP_KEY, None)
    st.session_state.pop(LAST_CAPTURED_TAKE_KEY, None)
    st.session_state.pop(PHRASE_ALIAS_COUNT_KEY, None)


def _render_dashboard_progress_reset(st, container=None) -> None:
    """Render a non-destructive dashboard-only reset action."""

    ui = container or st
    ui.caption("Stuck or want to revisit choices? Reset only the dashboard step state.")
    if ui.button(
        "Reset dashboard progress",
        type="secondary",
        use_container_width=True,
        help="Clears wizard progress and the last-take replay panel. It does not delete project files.",
    ):
        _clear_reset_session_state(st)
        message = "Dashboard progress reset. Project files were not deleted."
        st.session_state[RESET_MESSAGE_KEY] = message
        ui.success(message)
        st.rerun()


def _render_start_over_controls(st, config: ForgeConfig, *, include_divider: bool = True) -> None:
    if include_divider:
        st.divider()
    st.subheader("Wipe project data")
    st.caption(
        "Destructive reset: deletes this project's forge_config.json, samples, generated audio, "
        f"model output, and cache. Unrelated files in `{config.project_path}` are preserved."
    )
    confirmed = st.checkbox(
        "I understand this deletes local samples, generated audio, model output, and config"
    )
    if st.button(
        "Wipe local project data",
        disabled=not confirmed,
        type="secondary",
        use_container_width=True,
    ):
        removed = reset_project(config)
        _clear_reset_session_state(st)
        message = f"Project reset. Removed {len(removed)} project artifact(s)."
        st.session_state[RESET_MESSAGE_KEY] = message
        st.success(message)
        st.rerun()


def _status_badge(value: bool) -> str:
    return "✅ approved" if value else "⏳ pending"


def _render_review_checkpoints(st, config: ForgeConfig, status) -> None:
    from forge.review import (
        accept_model,
        approve_generated_review,
        approve_sample_review,
        sample_inventory,
        select_generated_audit_samples,
    )

    expanded = status.samples_ready or status.has_model
    with st.expander("Human review checkpoints", expanded=expanded):
        st.markdown(f"**Workflow stage:** `{status.workflow_stage}`")
        gate_cols = st.columns(4)
        gate_cols[0].metric("Sample review", _status_badge(status.sample_review_approved))
        gate_cols[1].metric("Generated audit", _status_badge(status.generated_review_approved))
        gate_cols[2].metric("Quality check", "✅ passed" if status.quality_check_passed else "⏳ pending")
        gate_cols[3].metric("Model accepted", "✅ yes" if status.model_accepted else "⏳ no")

        inventory = sample_inventory(config)
        st.subheader("Pre-training sample review")
        st.caption("Spot-check clips, delete/re-record bad takes from the CLI if needed, then approve.")
        sample_cols = st.columns(2)
        with sample_cols[0]:
            st.write(f"Recorded positives: {len(inventory.positives)}")
            for path in inventory.positives[:3]:
                st.caption(path.relative_to(config.project_path))
                st.audio(str(path))
        with sample_cols[1]:
            st.write(f"Recorded negatives: {len(inventory.negatives)}")
            for path in inventory.negatives[:3]:
                st.caption(path.relative_to(config.project_path))
                st.audio(str(path))
        if st.button(
            "Approve sample review",
            disabled=not status.samples_ready or status.sample_review_approved,
            use_container_width=True,
        ):
            approve_sample_review(config)
            save_config(config)
            st.success("Sample review approved.")
            st.rerun()

        st.subheader("Generated-output audit")
        audit_paths = select_generated_audit_samples(config, limit=6)
        if audit_paths:
            for path in audit_paths:
                st.caption(path.relative_to(config.project_path))
                st.audio(str(path))
        else:
            st.caption("No generated audio to audit yet.")
        if st.button(
            "Approve generated-audio audit",
            disabled=status.generated_audio_count == 0 or status.generated_review_approved,
            use_container_width=True,
        ):
            approve_generated_review(config)
            save_config(config)
            st.success("Generated-audio audit approved.")
            st.rerun()

        st.subheader("Post-training quality checkpoint")
        st.markdown(
            "Run a guided protocol: say the wake phrase, say near misses/confusables, "
            "then stay silent/background-only. The CLI records detections, misses, false triggers, "
            "and score range."
        )
        st.code(make_command("quality-check", config.project_path), language="bash")
        if status.quality_positive_trials:
            st.write(
                f"Last check: {status.quality_positive_hits}/{status.quality_positive_trials} positives, "
                f"{status.quality_false_triggers} false triggers."
            )
            if status.quality_score_min is not None and status.quality_score_max is not None:
                st.write(f"Score range: {status.quality_score_min:.3f}–{status.quality_score_max:.3f}")
        if st.button(
            "Accept current model",
            disabled=not status.quality_check_passed or status.model_accepted,
            type="primary",
            use_container_width=True,
        ):
            try:
                accept_model(config)
            except ValueError as exc:
                st.error(str(exc))
            else:
                save_config(config)
                st.success("Model accepted.")
                st.rerun()


def run_app(project_dir: Path | str = DEFAULT_PROJECT_DIR) -> None:
    """Render the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="wakeword-forge",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_css(), unsafe_allow_html=True)

    config = load_or_create_config(Path(project_dir))

    reset_message = st.session_state.pop(RESET_MESSAGE_KEY, None)
    if reset_message:
        st.success(reset_message)
    _render_update_notice(st)
    ensure_project_dirs(config)
    status = inspect_project(config)
    current_step = _current_wizard_step(st, status)
    _render_progress_sidebar(st, status, current_step, config)

    _render_current_wizard_step(st, config, status, current_step)

    with st.expander("Advanced status and CLI fallback", expanded=False):
        st.json(
            {
                "project_dir": str(status.project_dir),
                "wake_phrase": status.wake_phrase,
                "wake_phrases": list(status.wake_phrases),
                "workflow_stage": status.workflow_stage,
                "next_action": status.next_action,
                "ready_to_train": status.ready_to_train,
                "has_model": status.has_model,
                "review": {
                    "sample_review_approved": status.sample_review_approved,
                    "generated_review_approved": status.generated_review_approved,
                    "quality_check_passed": status.quality_check_passed,
                    "model_accepted": status.model_accepted,
                },
                "counts": {
                    "real_positives": status.real_positives,
                    "synthetic_positives": status.synthetic_positives,
                    "negatives": status.negatives,
                    "partial_negatives": status.partial_negatives,
                    "confusable_negatives": status.confusable_negatives,
                },
            }
        )
        _render_cli_fallbacks(st, config)


if __name__ == "__main__":
    main()
