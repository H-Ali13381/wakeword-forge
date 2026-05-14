"""Streamlit dashboard for wakeword-forge.

The module is intentionally import-light: tests can import helper functions without
requiring Streamlit, while the app imports Streamlit only when launched.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from .config import ForgeConfig, MIN_NEGATIVES, MIN_POSITIVES
from .project import ensure_project_dirs, inspect_project, load_or_create_config, save_config

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


def _sidebar_config(st, config: ForgeConfig) -> ForgeConfig:
    st.sidebar.header("Project settings")
    project_dir = st.sidebar.text_input("Project directory", value=str(config.project_path))
    wake_phrase = st.sidebar.text_input("Wake phrase", value=config.wake_phrase)

    st.sidebar.subheader("Recording")
    record_positives = st.sidebar.number_input(
        "Target positive recordings",
        min_value=MIN_POSITIVES,
        max_value=200,
        value=max(config.record_positives, MIN_POSITIVES),
    )
    record_negatives = st.sidebar.number_input(
        "Target negative recordings",
        min_value=MIN_NEGATIVES,
        max_value=200,
        value=max(config.record_negatives, MIN_NEGATIVES),
    )
    record_duration = st.sidebar.number_input(
        "Seconds per take",
        min_value=1.0,
        max_value=8.0,
        value=float(config.record_duration),
        step=0.25,
    )

    st.sidebar.subheader("Augmentation")
    use_tts = st.sidebar.toggle("Use TTS augmentation", value=config.use_tts_augmentation)
    tts_variants = st.sidebar.number_input(
        "Synthetic positive target",
        min_value=0,
        max_value=2_000,
        value=int(config.tts_variants),
        step=25,
    )
    tts_engine = st.sidebar.selectbox(
        "TTS engine",
        options=["kokoro", "piper", "none"],
        index=["kokoro", "piper", "none"].index(config.tts_engine)
        if config.tts_engine in {"kokoro", "piper", "none"}
        else 0,
    )

    updated = ForgeConfig(
        wake_phrase=wake_phrase.strip(),
        project_dir=project_dir.strip() or str(DEFAULT_PROJECT_DIR),
        record_positives=int(record_positives),
        record_negatives=int(record_negatives),
        record_duration=float(record_duration),
        use_tts_augmentation=bool(use_tts),
        tts_variants=int(tts_variants),
        tts_engine=str(tts_engine),
        backend=config.backend,
        max_epochs=config.max_epochs,
        contribute_samples=config.contribute_samples,
        samples_dir=config.samples_dir,
        output_dir=config.output_dir,
        cache_dir=config.cache_dir,
        trained_threshold=config.trained_threshold,
        trained_eer=config.trained_eer,
        sample_review_approved=config.sample_review_approved,
        generated_review_approved=config.generated_review_approved,
        sample_review_fingerprint=config.sample_review_fingerprint,
        generated_review_fingerprint=config.generated_review_fingerprint,
        quality_check_passed=config.quality_check_passed,
        model_accepted=config.model_accepted,
        quality_checked_model_path=config.quality_checked_model_path,
        quality_checked_model_fingerprint=config.quality_checked_model_fingerprint,
        accepted_model_fingerprint=config.accepted_model_fingerprint,
        quality_positive_hits=config.quality_positive_hits,
        quality_positive_trials=config.quality_positive_trials,
        quality_false_triggers=config.quality_false_triggers,
        quality_score_min=config.quality_score_min,
        quality_score_max=config.quality_score_max,
    )

    if st.sidebar.button("Save settings", type="primary", use_container_width=True):
        ensure_project_dirs(updated)
        save_config(updated)
        st.sidebar.success("Saved forge_config.json")
        st.rerun()

    return updated


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


def _status_badge(value: bool) -> str:
    return "✅ approved" if value else "⏳ pending"


def _render_review_checkpoints(st, config: ForgeConfig, status) -> None:
    from .review import (
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
    config = _sidebar_config(st, config)
    ensure_project_dirs(config)
    status = inspect_project(config)

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

    st.write("")
    st.progress(status.progress_fraction, text=status.next_action)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        _card("positives", str(status.total_positives), f"{status.real_positives} real / {status.synthetic_positives} TTS"),
        unsafe_allow_html=True,
    )
    col2.markdown(
        _card("negatives", str(status.total_negatives), f"{status.negatives} recorded / {status.confusable_negatives} confusable"),
        unsafe_allow_html=True,
    )
    col3.markdown(
        _card("hard partials", str(status.partial_negatives), "multi-word false-positive guard"),
        unsafe_allow_html=True,
    )
    model_note = "exported" if status.has_model else "not trained yet"
    col4.markdown(_card("model", "ONNX" if status.has_model else "—", model_note), unsafe_allow_html=True)

    st.write("")
    st.info(f"Next: {status.next_action}")
    _render_review_checkpoints(st, config, status)

    actions_left, actions_mid, actions_right = st.columns(3)

    with actions_left:
        st.subheader("1. Capture")
        pos_missing = max(1, config.record_positives - status.real_positives)
        neg_missing = max(1, config.record_negatives - status.negatives)
        if st.button(
            f"Record {pos_missing} wake-phrase takes",
            disabled=not bool(config.wake_phrase),
            use_container_width=True,
        ):
            from .recorder import record_session

            _run_blocking_action(
                "Recording positive examples",
                lambda: record_session(
                    phrase=config.wake_phrase,
                    n_takes=pos_missing,
                    out_dir=config.positives_path,
                    duration=config.record_duration,
                    label="positives",
                ),
            )
        if st.button(
            f"Record {neg_missing} counter-examples",
            disabled=not bool(config.wake_phrase),
            use_container_width=True,
        ):
            from .recorder import record_session

            _run_blocking_action(
                "Recording negative examples",
                lambda: record_session(
                    phrase=f"anything except '{config.wake_phrase}'",
                    n_takes=neg_missing,
                    out_dir=config.negatives_path,
                    duration=config.record_duration,
                    label="negatives",
                ),
            )

    with actions_mid:
        st.subheader("2. Augment")
        synth_missing = max(0, config.tts_variants - status.synthetic_positives)
        synth_disabled = (
            not config.wake_phrase
            or not config.use_tts_augmentation
            or config.tts_engine == "none"
            or synth_missing == 0
        )
        if st.button(
            f"Generate {synth_missing} TTS positives",
            disabled=synth_disabled,
            use_container_width=True,
        ):
            from .synthesizer import synthesize_positives

            _run_blocking_action(
                "Generating synthetic positives",
                lambda: synthesize_positives(
                    phrase=config.wake_phrase,
                    out_dir=config.synthetic_path,
                    n=synth_missing,
                    engine=config.tts_engine,
                ),
            )
        if st.button(
            "Generate hard negatives",
            disabled=not config.wake_phrase or config.tts_engine == "none",
            use_container_width=True,
        ):
            from .synthesizer import synthesize_confusable_negatives, synthesize_partial_negatives

            def hard_negatives() -> None:
                if len(config.wake_phrase.split()) >= 2:
                    synthesize_partial_negatives(
                        phrase=config.wake_phrase,
                        out_dir=config.partials_path,
                        n=max(0, 100 - status.partial_negatives),
                        engine=config.tts_engine,
                    )
                synthesize_confusable_negatives(
                    phrase=config.wake_phrase,
                    out_dir=config.confusables_path,
                    cache_file=config.confusables_cache,
                    n_variants=max(0, 50 - status.confusable_negatives),
                    engine=config.tts_engine,
                )

            _run_blocking_action("Generating hard negatives", hard_negatives)
        if st.button("Fill background negatives", use_container_width=True):
            from .negatives import ensure_negatives

            _run_blocking_action(
                "Creating background negatives",
                lambda: ensure_negatives(
                    out_dir=config.negatives_path,
                    target=max(150, config.record_negatives),
                    use_common_voice=False,
                    use_esc50=False,
                ),
            )

    with actions_right:
        st.subheader("3. Train & test")
        if st.button(
            "Train detector",
            type="primary",
            disabled=not status.ready_to_train,
            use_container_width=True,
        ):
            from .trainer import run_training

            _run_blocking_action("Training detector", lambda: run_training(config))
        st.caption("Live mic test is still CLI-first because it needs a foreground audio stream.")
        st.code(make_command("mic-test", config.project_path), language="bash")
        if status.trained_eer is not None:
            st.metric("Validation EER", f"{status.trained_eer:.3f}")
        st.metric("Threshold", f"{status.trained_threshold:.4f}")

    with st.expander("Status details", expanded=False):
        st.json(
            {
                "project_dir": str(status.project_dir),
                "wake_phrase": status.wake_phrase,
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
