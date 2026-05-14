"""
recorder.py — Interactive microphone recording with rich terminal UI.

Guides the user through N takes of their wake-phrase, with:
  - Visual countdown before each take
  - Waveform amplitude bar during recording
  - Instant playback after each take
  - Accept / retry / skip per take

Usage (standalone test):
    python -m wakeword_forge.recorder --phrase "Hey Nova" --n 20 --out samples/positives
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.prompt import Confirm, Prompt

from .config import SAMPLE_RATE, MAX_DURATION

console = Console()


def _record_take(duration: float, sample_rate: int) -> np.ndarray:
    """Block-record a single take and return the float32 waveform."""
    frames = int(duration * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def _play(audio: np.ndarray, sample_rate: int) -> None:
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def _amplitude_bar(audio: np.ndarray, width: int = 40) -> str:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    level = min(int(rms * width * 8), width)
    bar = "█" * level + "░" * (width - level)
    db = 20 * np.log10(rms + 1e-9)
    return f"[{'green' if level > 4 else 'red'}]{bar}[/] {db:+.1f} dB"


def record_session(
    phrase: str,
    n_takes: int,
    out_dir: Path,
    duration: float = MAX_DURATION,
    sample_rate: int = SAMPLE_RATE,
    label: str = "positives",
) -> list[Path]:
    """
    Interactively record n_takes of phrase, saving accepted takes to out_dir.
    Returns list of saved file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    existing = sorted(out_dir.glob("*.wav"))
    start_idx = len(existing)

    console.print(f"\n[bold cyan]Recording session[/bold cyan] — [yellow]{label}[/yellow]")
    console.print(f"Phrase: [bold white]{phrase}[/bold white]")
    console.print(f"Takes needed: [bold]{n_takes}[/bold]  |  Duration: {duration:.1f}s each")
    console.print("[dim]Press Enter to start each take. Type 'q' to quit early.[/dim]\n")

    take_num = 0
    while len(saved) < n_takes:
        remaining = n_takes - len(saved)
        console.print(
            f"[bold]Take {len(saved) + 1}/{n_takes}[/bold]"
            f"  [dim]({remaining} remaining)[/dim]"
        )

        # Prompt to start
        action = Prompt.ask(
            "  [green]Enter[/green] to record  |  [yellow]s[/yellow] to skip  |  [red]q[/red] to quit",
            default="",
            show_default=False,
        ).strip().lower()

        if action == "q":
            console.print("[yellow]Quitting early.[/yellow]")
            break
        if action == "s":
            console.print("[dim]Skipped.[/dim]")
            continue

        # Countdown
        for i in range(3, 0, -1):
            console.print(f"  [bold yellow]{i}...[/bold yellow]", end="\r")
            time.sleep(0.7)
        console.print("  [bold green]GO — speak now![/bold green]   ", end="\r")

        audio = _record_take(duration, sample_rate)

        console.print(f"\n  Level: {_amplitude_bar(audio)}")

        # Quick silence check
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            console.print("  [red]Mic too quiet — check your microphone.[/red]")
            continue

        # Playback option
        if Confirm.ask("  Play back?", default=False):
            _play(audio, sample_rate)

        if Confirm.ask("  [bold green]Keep this take?[/bold green]", default=True):
            out_path = out_dir / f"take_{start_idx + take_num:04d}.wav"
            sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
            saved.append(out_path)
            console.print(f"  [green]Saved → {out_path.name}[/green]")
        else:
            console.print("  [dim]Discarded.[/dim]")

        take_num += 1

    console.print(
        f"\n[bold green]Session complete.[/bold green] "
        f"Saved {len(saved)}/{n_takes} takes to [cyan]{out_dir}[/cyan]\n"
    )
    return saved


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--phrase", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=MAX_DURATION)
    args = p.parse_args()

    record_session(
        phrase=args.phrase,
        n_takes=args.n,
        out_dir=Path(args.out),
        duration=args.duration,
    )
