"""Streamlit dashboard for wakeword-forge.

The module is intentionally import-light: tests can import helper functions without
requiring Streamlit, while the app imports Streamlit only when launched.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Callable

from wakeword_forge.config import ForgeConfig, MIN_NEGATIVES, MIN_POSITIVES, normalize_phrases
from wakeword_forge.project import (
    ensure_project_dirs,
    inspect_project,
    load_or_create_config,
    reset_project,
    save_config,
)

DEFAULT_PROJECT_DIR = Path.home() / "wakeword_forge_project"


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
        --forge-ember: #ff7a2f;
        --forge-brass: #d2a85d;
        --forge-ink: #111317;
        --forge-panel: rgba(25, 28, 34, 0.74);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 122, 47, 0.18), transparent 34rem),
            linear-gradient(135deg, #0f1115 0%, #171a20 55%, #0d0f12 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111317 0%, #191b22 100%);
        border-right: 1px solid rgba(255, 122, 47, 0.18);
    }
    .forge-hero {
        border: 1px solid rgba(255, 122, 47, 0.25);
        border-radius: 24px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(255,122,47,0.16), rgba(17,19,23,0.92));
        box-shadow: 0 18px 70px rgba(0,0,0,0.34);
    }
    .forge-kicker {
        color: var(--forge-brass);
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
    .forge-card {
        border: 1px solid rgba(210, 168, 93, 0.22);
        border-radius: 18px;
        padding: 1rem;
        background: rgba(17, 19, 23, 0.76);
        min-height: 8.2rem;
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
        border-color: rgba(64, 214, 137, 0.72);
        background: rgba(64, 214, 137, 0.13);
    }
    .forge-step-active {
        border-color: rgba(245, 198, 84, 0.82);
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
    .stButton > button[kind="secondary"] {
        border-color: rgba(210, 168, 93, 0.55);
        background: rgba(210, 168, 93, 0.13);
        color: #fff2df;
    }
    </style>
    """


def _card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="forge-card">
      <div class="forge-card-label">{label}</div>
      <div class="forge-card-value">{value}</div>
      <div class="forge-card-note">{note}</div>
    </div>
    """


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


DASHBOARD_STEP_KEY = "wakeword_forge_dashboard_step"
LAST_CAPTURED_TAKE_KEY = "wakeword_forge_last_captured_take"
RESET_MESSAGE_KEY = "wakeword_forge_reset_message"
WIZARD_STEPS = ("intro", "workspace", "recording", "augmentation", "capture", "review", "train", "done")
WIZARD_STEP_LABELS = {
    "intro": "Start",
    "workspace": "1. Name the trigger",
    "recording": "2. Recording plan",
    "augmentation": "3. Augmentation plan",
    "capture": "4. Capture examples",
    "review": "5. Review samples",
    "train": "6. Train and test",
    "done": "7. Accept model",
}


def _set_wizard_step(st, step: str) -> None:
    st.session_state[DASHBOARD_STEP_KEY] = step


def _confirm_step(st, config: ForgeConfig, next_step: str, message: str) -> None:
    ensure_project_dirs(config)
    save_config(config)
    _set_wizard_step(st, next_step)
    st.success(message)
    st.rerun()


def _phrase_text(config: ForgeConfig) -> str:
    return "\n".join(config.phrase_options or normalize_phrases((config.wake_phrase,)))


def _parse_phrase_text(text: str) -> tuple[str, ...]:
    return normalize_phrases(tuple(text.splitlines()))


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
    cols = st.columns(3)
    cards = [
        _card("Private", "Local", "Recordings stay in the project folder."),
        _card("Guided", "Wizard", "One decision at a time, then explicit review."),
        _card("Flexible", "Aliases", "Train one detector for one or more trigger phrases."),
    ]
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)
    if st.button("Begin", type="secondary", use_container_width=True):
        _set_wizard_step(st, "workspace")
        st.rerun()


def _render_workspace_step(st, config: ForgeConfig) -> ForgeConfig:
    """Render only workspace/phrase fields for the first wizard step."""

    st.subheader("1. Name the trigger")
    st.caption("Choose where this project lives and list every phrase that should trigger the detector.")
    project_dir = st.text_input("Project directory", value=str(config.project_path))
    phrase_text = st.text_area(
        "Wake phrases (one per line)",
        value=_phrase_text(config),
        help="Use one phrase per line. The first phrase is the primary display name; all lines become positive trigger aliases.",
    )
    phrases = _parse_phrase_text(phrase_text)

    updated = replace(
        config,
        project_dir=project_dir.strip() or str(DEFAULT_PROJECT_DIR),
        wake_phrase=phrases[0] if phrases else "",
        wake_phrases=list(phrases),
    )
    can_continue = bool(updated.phrase_options)
    if st.button("Confirm phrases", type="secondary", disabled=not can_continue, use_container_width=True):
        _confirm_step(st, updated, "recording", "Phrases confirmed.")
    return updated


def _render_recording_step(st, config: ForgeConfig) -> ForgeConfig:
    """Render only recording parameters for the second wizard step."""

    st.subheader("2. Recording plan")
    st.caption("Set the human-recorded examples. No synthesis or training controls here.")
    record_positives = st.number_input(
        "Target positive recordings",
        min_value=MIN_POSITIVES,
        max_value=200,
        value=max(config.record_positives, MIN_POSITIVES),
    )
    record_negatives = st.number_input(
        "Target negative recordings",
        min_value=MIN_NEGATIVES,
        max_value=200,
        value=max(config.record_negatives, MIN_NEGATIVES),
    )
    record_duration = st.number_input(
        "Seconds per take",
        min_value=1.0,
        max_value=8.0,
        value=float(config.record_duration),
        step=0.25,
    )

    updated = replace(
        config,
        record_positives=int(record_positives),
        record_negatives=int(record_negatives),
        record_duration=float(record_duration),
    )
    if st.button("Confirm recording plan", type="secondary", use_container_width=True):
        _confirm_step(st, updated, "augmentation", "Recording plan confirmed.")
    return updated


def _render_augmentation_step(st, config: ForgeConfig) -> ForgeConfig:
    """Render only generated-audio parameters for the third wizard step."""

    st.subheader("3. Augmentation plan")
    st.caption("Decide whether to add generated positives and hard negatives before capture/training.")
    use_tts = st.toggle("Use TTS augmentation", value=config.use_tts_augmentation)
    tts_variants = st.number_input(
        "Synthetic positive target",
        min_value=0,
        max_value=2_000,
        value=int(config.tts_variants),
        step=25,
        disabled=not use_tts,
    )
    tts_engine = st.selectbox(
        "TTS engine",
        options=["kokoro", "piper", "none"],
        index=["kokoro", "piper", "none"].index(config.tts_engine)
        if config.tts_engine in {"kokoro", "piper", "none"}
        else 0,
        disabled=not use_tts,
    )

    updated = replace(
        config,
        use_tts_augmentation=bool(use_tts),
        tts_variants=int(tts_variants) if use_tts else 0,
        tts_engine=str(tts_engine) if use_tts else "none",
    )
    if st.button("Confirm augmentation plan", type="secondary", use_container_width=True):
        _confirm_step(st, updated, "capture", "Augmentation plan confirmed.")
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

    from wakeword_forge.config import SAMPLE_RATE
    from wakeword_forge.recorder import _prepare_recorded_take, _record_take

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
) -> None:
    st.markdown(f"**{current_count} / {target_count} {kind} takes saved**")
    if current_count >= target_count:
        st.success(f"All {target_count} {kind} takes are recorded.")
        return

    next_take = current_count + 1
    st.caption(f"Take {next_take}: press record, speak `{phrase}`, then replay before moving on.")
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


def _phrase_list_for_generation(config: ForgeConfig) -> tuple[str, ...]:
    return config.phrase_options or normalize_phrases((config.wake_phrase,))


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
        _render_one_take_recorder(
            st,
            kind="counter-example",
            phrase=f"anything except {' / '.join(_phrase_list_for_generation(config)) or 'the wake phrase'}",
            current_count=status.negatives,
            target_count=config.record_negatives,
            out_dir=config.negatives_path,
            duration=config.record_duration,
            prefix="neg",
        )
        return

    if config.use_tts_augmentation and config.tts_engine != "none":
        synth_missing = max(0, config.tts_variants - status.synthetic_positives)
        if synth_missing > 0:
            if st.button(f"Generate {synth_missing} TTS positives", use_container_width=True):
                from wakeword_forge.synthesizer import synthesize_positive_phrases

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

        from wakeword_forge.synthesizer import load_confusable_phrases

        phrases = _phrase_list_for_generation(config)
        partial_phrases = tuple(phrase for phrase in phrases if len(phrase.split()) >= 2)
        partial_target = 100 if partial_phrases else 0
        confusable_phrases = load_confusable_phrases(config.confusables_cache)
        confusable_target = 50 if confusable_phrases else 0
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
            from wakeword_forge.synthesizer import synthesize_confusable_negatives, synthesize_partial_negatives

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

    background_target = max(150, config.record_negatives)
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
        from wakeword_forge.negatives import ensure_negatives

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
    if st.button("Train detector", type="primary", disabled=not status.ready_to_train, use_container_width=True):
        from wakeword_forge.trainer import run_training

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
    if not status.wake_phrase and step not in {"intro", "workspace"}:
        return "intro"
    return step


def _render_current_wizard_step(st, config: ForgeConfig, status, current_step: str | None = None) -> None:
    step = current_step or _current_wizard_step(st, status)
    if step == "intro":
        _render_intro_step(st, config)
    elif step == "workspace":
        _render_workspace_step(st, config)
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
        if st.button("Continue to training", disabled=not status.ready_to_train, type="secondary", use_container_width=True):
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


def _render_progress_sidebar(st, status, current_step: str | None = None) -> None:
    """Render the sidebar as a compact workflow progress rail."""

    current_step = current_step or _default_wizard_step(status)
    sidebar = st.sidebar
    sidebar.header("Progress")
    sidebar.progress(_workflow_progress_fraction(status), text=status.workflow_stage)
    sidebar.caption(f"Next step: {status.next_action}")
    sidebar.divider()

    sidebar.subheader("Run checklist")
    step_notes = {
        "intro": "overview and start",
        "workspace": f"{len(status.wake_phrases)} phrase(s)" if status.wake_phrases else "choose phrases",
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


def _render_start_over_controls(st, config: ForgeConfig) -> None:
    st.divider()
    st.subheader("Start over")
    st.caption(
        "Deletes this project's forge_config.json, samples, generated audio, model output, "
        f"and cache. Unrelated files in `{config.project_path}` are preserved."
    )
    confirmed = st.checkbox(
        "I understand this deletes local samples, generated audio, model output, and config"
    )
    if st.button(
        "Wipe configs and start from scratch",
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
    from wakeword_forge.review import (
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
    ensure_project_dirs(config)
    status = inspect_project(config)
    current_step = _current_wizard_step(st, status)
    _render_progress_sidebar(st, status, current_step)

    if current_step != "intro":
        st.info(f"Next: {status.next_action}")
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
        _render_start_over_controls(st, config)


if __name__ == "__main__":
    main()
