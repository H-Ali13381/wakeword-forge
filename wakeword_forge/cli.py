"""
cli.py — Typer-based CLI entrypoint for wakeword-forge.

Commands:
    wakeword-forge dashboard        Streamlit dashboard-first workflow
    wakeword-forge run              Full interactive CLI pipeline
    wakeword-forge record           Recording session only
    wakeword-forge synth            TTS synthesis only
    wakeword-forge train            Training only (uses existing samples)
    wakeword-forge test             Real-time threshold test against mic input
    wakeword-forge info             Show project status
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .config import MIN_NEGATIVES, MIN_POSITIVES
from .project import count_wavs, load_or_create_config, save_config

app = typer.Typer(
    name="wakeword-forge",
    help="Train a personal wake-word detector. No cloud. No account.",
    add_completion=False,
)
console = Console()


# ── run ───────────────────────────────────────────────────────────────────────

@app.command()
def run(
    project_dir: Path = typer.Option(
        Path.cwd() / "wakeword_project",
        "--dir", "-d",
        help="Project directory (created if it doesn't exist)",
    ),
) -> None:
    """Full interactive pipeline: record → synth → train → export."""

    console.print(Panel.fit(
        "[bold cyan]wakeword-forge[/bold cyan]\n"
        "Train a personal wake-word detector from your voice.",
        border_style="cyan",
    ))

    project_dir.mkdir(parents=True, exist_ok=True)
    config = load_or_create_config(project_dir)

    # ── Step 1: Wake phrase ───────────────────────────────────────────────────
    if not config.wake_phrase:
        config.wake_phrase = Prompt.ask(
            "\n[bold]What is your wake-phrase?[/bold]  "
            "(e.g. 'Hey Nova', 'Okay Atlas', 'Computer')"
        ).strip()
        if not config.wake_phrase:
            console.print("[red]Wake phrase cannot be empty.[/red]")
            raise typer.Exit(1)

    console.print(f"\n[green]Wake phrase:[/green] [bold]{config.wake_phrase}[/bold]")
    save_config(config)

    # ── Step 2: Record positives ──────────────────────────────────────────────
    n_pos = count_wavs(config.positives_path)
    if n_pos < MIN_POSITIVES:
        console.print(
            f"\n[bold]Step 1 of 3:[/bold] Record your wake-phrase  "
            f"[dim]({n_pos} saved so far, need {config.record_positives})[/dim]"
        )
        console.print(
            f'  Say [bold white]"{config.wake_phrase}"[/bold white] clearly '
            f"each time you hear the prompt.\n"
        )
        if Confirm.ask("Start recording positives now?", default=True):
            from .recorder import record_session
            record_session(
                phrase=config.wake_phrase,
                n_takes=config.record_positives - n_pos,
                out_dir=config.positives_path,
                duration=config.record_duration,
                label="positives",
            )

    # ── Step 3: Record negatives ──────────────────────────────────────────────
    n_neg = count_wavs(config.negatives_path)
    if n_neg < MIN_NEGATIVES:
        console.print(
            f"\n[bold]Step 2 of 3:[/bold] Record counter-examples  "
            f"[dim]({n_neg} saved so far, need {config.record_negatives})[/dim]"
        )
        console.print(
            "  Say similar-sounding words, ambient speech, or anything\n"
            "  that is [red]not[/red] your wake-phrase.\n"
        )
        if Confirm.ask("Start recording counter-examples now?", default=True):
            from .recorder import record_session
            record_session(
                phrase=f"(anything but '{config.wake_phrase}')",
                n_takes=config.record_negatives - n_neg,
                out_dir=config.negatives_path,
                duration=config.record_duration,
                label="negatives",
            )

    # ── Step 4: TTS synthesis ─────────────────────────────────────────────────
    n_synth = count_wavs(config.synthetic_path)
    if config.use_tts_augmentation and config.tts_engine != "none" and n_synth < config.tts_variants:
        console.print(
            f"\n[bold]Step 3a:[/bold] Generate synthetic positives  "
            f"[dim]({n_synth} done, target {config.tts_variants})[/dim]"
        )
        if Confirm.ask(
            f"Generate {config.tts_variants - n_synth} synthetic variants with {config.tts_engine}?",
            default=True,
        ):
            from .synthesizer import synthesize_positives
            synthesize_positives(
                phrase=config.wake_phrase,
                out_dir=config.synthetic_path,
                n=config.tts_variants - n_synth,
                engine=config.tts_engine,
            )

    # ── Step 4b: Partial-phrase negatives ─────────────────────────────────────
    # Only for multi-word phrases. These are the core fix for temporal location.
    n_partials = count_wavs(config.partials_path)
    words = config.wake_phrase.split()
    if len(words) >= 2 and config.tts_engine != "none" and n_partials < 100:
        console.print(
            f"\n[bold]Step 3b:[/bold] Generate partial-phrase negatives  "
            f"[dim]('{'  '.join(words[:-1])}' only — prevents partial-match false positives)[/dim]"
        )
        if Confirm.ask("Generate partial-phrase hard negatives?", default=True):
            from .synthesizer import synthesize_partial_negatives
            synthesize_partial_negatives(
                phrase=config.wake_phrase,
                out_dir=config.partials_path,
                n=100 - n_partials,
                engine=config.tts_engine,
            )

    # ── Step 3c: Confusable negatives from editable phrase cache ──────────────
    n_confusables = count_wavs(config.confusables_path)
    if config.tts_engine != "none" and n_confusables < 50:
        from .synthesizer import synthesize_confusable_negatives
        synthesize_confusable_negatives(
            phrase=config.wake_phrase,
            out_dir=config.confusables_path,
            cache_file=config.confusables_cache,
            n_variants=50 - n_confusables,
            engine=config.tts_engine,
        )

    # ── Step 4c: Background negatives ────────────────────────────────────────
    n_neg_total = count_wavs(config.negatives_path)
    if n_neg_total < 50:
        console.print(
            f"\n[bold]Step 3c:[/bold] Background negatives  "
            f"[dim]({n_neg_total} recorded — supplementing with synthetic background)[/dim]"
        )
        from .negatives import ensure_negatives
        ensure_negatives(
            out_dir=config.negatives_path,
            target=150,
            use_common_voice=False,  # off by default; user can enable via config
            use_esc50=False,
        )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    console.print("\n[bold]Training...[/bold]")
    from .trainer import run_training
    try:
        onnx_path = run_training(config)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # ── Step 6: Privacy opt-in ────────────────────────────────────────────────
    if not config.contribute_samples:
        config.contribute_samples = Confirm.ask(
            "\nWould you like to contribute your anonymized samples "
            "to improve the shared community model?",
            default=False,
        )
        if config.contribute_samples:
            console.print("[dim]Sample sharing is not yet implemented. Coming soon.[/dim]")
        save_config(config)

    # ── Done ──────────────────────────────────────────────────────────────────
    console.print(Panel.fit(
        f"[bold green]Done![/bold green]\n\n"
        f"Model saved to: [cyan]{onnx_path}[/cyan]\n"
        f"Config:         [cyan]{onnx_path.with_suffix('.json')}[/cyan]\n\n"
        f"EER: {config.trained_eer:.3f}  |  Threshold: {config.trained_threshold:.4f}"
        if config.trained_eer else
        f"[bold green]Done![/bold green]\n\nModel: [cyan]{onnx_path}[/cyan]",
        border_style="green",
    ))


# ── dashboard ─────────────────────────────────────────────────────────────────

@app.command()
def dashboard(
    project_dir: Path = typer.Option(
        Path.cwd() / "wakeword_project",
        "--dir", "-d",
        help="Project directory (created if it doesn't exist)",
    ),
) -> None:
    """Launch the Streamlit dashboard-first workflow."""
    from .dashboard import launch_streamlit

    raise typer.Exit(launch_streamlit(project_dir))


# ── record ────────────────────────────────────────────────────────────────────

@app.command()
def record(
    phrase: str = typer.Argument(..., help="Wake phrase to record"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    n: int = typer.Option(20, "--n", help="Number of takes"),
    label: str = typer.Option("positives", "--label"),
) -> None:
    """Standalone recording session."""
    from .recorder import record_session
    record_session(phrase=phrase, n_takes=n, out_dir=out, label=label)


# ── synth ─────────────────────────────────────────────────────────────────────

@app.command()
def synth(
    phrase: str = typer.Argument(..., help="Wake phrase to synthesize"),
    out: Path = typer.Option(..., "--out", "-o"),
    n: int = typer.Option(300, "--n"),
    engine: str = typer.Option("kokoro", "--engine", "-e"),
) -> None:
    """Generate synthetic positive samples using TTS."""
    from .synthesizer import synthesize_positives
    synthesize_positives(phrase=phrase, out_dir=out, n=n, engine=engine)


# ── train ─────────────────────────────────────────────────────────────────────

@app.command()
def train(
    project_dir: Path = typer.Option(Path.cwd() / "wakeword_project", "--dir", "-d"),
    backend: str = typer.Option("dscnn", "--backend", "-b", help="Supported backend: dscnn"),
) -> None:
    """Train using existing samples in the project directory."""
    config = load_or_create_config(project_dir)
    config.backend = backend
    from .trainer import run_training
    run_training(config)


# ── test ──────────────────────────────────────────────────────────────────────

@app.command()
def test(
    model: Path = typer.Argument(..., help="Path to .onnx model"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    chunk_ms: int = typer.Option(300, "--chunk-ms", help="Audio chunk size in ms"),
) -> None:
    """
    Live microphone test — streams audio and prints scores in real-time.
    Press Ctrl+C to stop.
    """
    import time
    import numpy as np
    import onnxruntime as ort
    import sounddevice as sd
    from .config import SAMPLE_RATE

    cfg_path = config or model.with_suffix(".json")
    threshold = 0.5
    wake_phrase = "?"
    if cfg_path.exists():
        cfg_data = json.loads(cfg_path.read_text())
        threshold = cfg_data.get("threshold", 0.5)
        wake_phrase = cfg_data.get("wake_phrase", "?")

    console.print(f"Wake phrase: [bold]{wake_phrase}[/bold]")
    console.print(f"Threshold:   [yellow]{threshold:.4f}[/yellow]")
    console.print("[dim]Streaming... Ctrl+C to stop.[/dim]\n")

    # Load model once — never inside the callback
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    chunk_samples = int(chunk_ms / 1000 * SAMPLE_RATE)
    window: list[np.ndarray] = []
    window_max_samples = 2 * SAMPLE_RATE  # 2s rolling window

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        mono = indata[:, 0].astype(np.float32)
        window.append(mono)
        # Trim to 2s
        total = sum(len(w) for w in window)
        while total > window_max_samples and window:
            removed = window.pop(0)
            total -= len(removed)
        buf = np.concatenate(window) if window else np.zeros(chunk_samples, np.float32)
        score = float(sess.run(None, {input_name: buf[None]})[0][0])
        bar_len = 30
        filled = int(score * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        color = "green" if score >= threshold else "white"
        triggered = "  [bold red]<<< WAKE[/bold red]" if score >= threshold else ""
        console.print(f"\r[{color}]{bar}[/{color}] {score:.3f}{triggered}          ", end="")

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        callback=callback, blocksize=chunk_samples,
    ):
        try:
            while True:
                time.sleep(0.05)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")


# ── info ──────────────────────────────────────────────────────────────────────

@app.command()
def info(
    project_dir: Path = typer.Option(Path.cwd() / "wakeword_project", "--dir", "-d"),
) -> None:
    """Show project status."""
    config = load_or_create_config(project_dir)

    t = Table(title="Project Status", show_header=False, box=None, padding=(0, 2))
    t.add_column("Key", style="bold cyan")
    t.add_column("Value")

    t.add_row("Wake phrase",    config.wake_phrase or "[dim]not set[/dim]")
    t.add_row("Project dir",    str(project_dir))
    t.add_row("Positives",      str(count_wavs(config.positives_path)))
    t.add_row("Synthetic pos",  str(count_wavs(config.synthetic_path)))
    t.add_row("Negatives",      str(count_wavs(config.negatives_path)))
    t.add_row("Backend",        config.backend)
    t.add_row("TTS engine",     config.tts_engine)
    t.add_row("Trained EER",    f"{config.trained_eer:.4f}" if config.trained_eer else "[dim]not trained[/dim]")
    t.add_row("Threshold",      f"{config.trained_threshold:.4f}" if config.trained_threshold != 0.5 else "[dim]default[/dim]")

    console.print(t)


if __name__ == "__main__":
    app()
