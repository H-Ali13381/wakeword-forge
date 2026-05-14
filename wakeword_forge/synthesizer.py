"""
synthesizer.py — Generate synthetic positive examples using TTS.

Supports:
  - kokoro-onnx (default, Apache-2.0, CPU-capable, ~82M params)
  - piper (CPU, very fast, lower quality)
  - none  (skip synthesis, use recorded samples only)

The synthesizer generates N variants of the wake-phrase with randomized:
  - Voice / speaker
  - Speaking rate
  - Text spelling variant (e.g. "Hey Nova" / "hey nova" / "Hey Nová")

Variants are saved as WAV files under output_dir/synthetic/positives/.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import track

from .config import SAMPLE_RATE

console = Console()


# ── Text variants ─────────────────────────────────────────────────────────────

def _text_variants(phrase: str) -> list[str]:
    """Generate surface-form spelling variants of the wake-phrase."""
    variants = {phrase, phrase.lower(), phrase.upper(), phrase.title()}
    # common shorthand mappings
    for src, dst in [("okay", "ok"), ("ok", "okay"), ("hey", "hey"), ("hi", "hi")]:
        if src in phrase.lower():
            variants.add(phrase.lower().replace(src, dst))
    return list(variants)


def _partial_variants(phrase: str) -> list[str]:
    """
    Generate partial versions of a multi-word wake phrase.
    These become hard negative examples — the model must NOT fire on them.

    e.g. "Hey Nova"     →  ["Hey", "hey", "HEY"]
         "Okay Atlas"   →  ["Okay", "okay", "OKAY"]
         "Computer"     →  []  (single-word phrase, no partials)
    """
    words = phrase.split()
    if len(words) < 2:
        return []
    partials: set[str] = set()
    # All strict prefixes (not the full phrase)
    for n in range(1, len(words)):
        prefix = " ".join(words[:n])
        partials.update({prefix, prefix.lower(), prefix.upper(), prefix.title()})
    return list(partials)


# ── Backend protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class TTSBackend(Protocol):
    def synthesize(self, text: str, **kwargs) -> tuple[np.ndarray, int]:
        """Return (waveform float32, sample_rate)."""
        ...


# ── Kokoro backend ────────────────────────────────────────────────────────────

class KokoroBackend:
    """
    Thin wrapper around kokoro-onnx.
    Install with: pip install kokoro-onnx
    Model files are auto-downloaded on first use (~85 MB total).
    """

    VOICES = [
        "af_heart", "af_alloy", "af_aoede", "af_bella",
        "am_adam", "am_echo", "am_eric", "am_fenrir",
        "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
    ]

    MODEL_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro  # type: ignore
        model_path  = self._ensure_file("kokoro-v1.0.onnx",  self.MODEL_URL)
        voices_path = self._ensure_file("voices-v1.0.bin",   self.VOICES_URL)
        self._kokoro = Kokoro(str(model_path), str(voices_path))

    @staticmethod
    def _ensure_file(filename: str, url: str) -> Path:
        """Download file into ~/.cache/wakeword-forge/ if not already present."""
        import urllib.request
        cache_dir = Path.home() / ".cache" / "wakeword-forge"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / filename
        if not dest.exists():
            console.print(f"[dim]Downloading {filename}...[/dim]")
            urllib.request.urlretrieve(url, dest)
            console.print(f"[dim]  → {dest}[/dim]")
        return dest

    def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0) -> tuple[np.ndarray, int]:
        voice = voice or random.choice(self.VOICES)
        samples, sr = self._kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        return samples.astype(np.float32), sr


# ── Piper backend ─────────────────────────────────────────────────────────────

class PiperBackend:
    """
    Wrapper around piper-tts.
    Requires: pip install piper-tts
    And a downloaded voice model, e.g.:
      piper --download-dir ~/.local/share/piper-voices en_US-amy-medium
    """

    def __init__(self, voice: str = "en_US-amy-medium", data_dir: str | None = None) -> None:
        from piper import PiperVoice  # type: ignore
        self._Voice = PiperVoice
        self._voice_name = voice
        self._data_dir = data_dir
        self._piper = PiperVoice.load(voice, data_path=data_dir)

    def synthesize(self, text: str, **_) -> tuple[np.ndarray, int]:
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._piper.synthesize(text, wf)
        buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32")
        return audio, sr


# ── No-op backend ─────────────────────────────────────────────────────────────

class NoneBackend:
    def synthesize(self, text: str, **_) -> tuple[np.ndarray, int]:
        raise RuntimeError("TTS engine set to 'none' — skipping synthesis.")


# ── Factory ───────────────────────────────────────────────────────────────────

def build_backend(engine: str) -> TTSBackend:
    if engine == "kokoro":
        return KokoroBackend()
    elif engine == "piper":
        return PiperBackend()
    elif engine == "none":
        return NoneBackend()
    else:
        raise ValueError(f"Unknown TTS engine: {engine!r}")


# ── Main synthesize function ──────────────────────────────────────────────────

def synthesize_positives(
    phrase: str,
    out_dir: Path,
    n: int = 300,
    engine: str = "kokoro",
    seed: int = 42,
) -> list[Path]:
    """
    Generate n synthetic positive examples of phrase and save to out_dir.
    Returns list of saved file paths.
    """
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Synthesizing {n} positive samples[/bold cyan] using [yellow]{engine}[/yellow]")

    try:
        backend = build_backend(engine)
    except ImportError as e:
        console.print(f"[red]TTS engine '{engine}' not installed: {e}[/red]")
        console.print("[yellow]Skipping synthesis — using recorded samples only.[/yellow]")
        return []

    variants = _text_variants(phrase)
    saved: list[Path] = []

    speeds = [0.85, 0.9, 0.95, 1.0, 1.0, 1.05, 1.1, 1.15]

    for i in track(range(n), description="Synthesizing..."):
        text = random.choice(variants)
        speed = random.choice(speeds)

        try:
            audio, sr = backend.synthesize(text, speed=speed)
        except Exception as e:
            console.print(f"[red]Synthesis failed on sample {i}: {e}[/red]")
            continue

        # Resample to 16 kHz if needed
        if sr != SAMPLE_RATE:
            import torchaudio
            import torch
            wav_t = torch.from_numpy(audio).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
            audio = wav_t.squeeze(0).numpy()

        out_path = out_dir / f"synth_{i:05d}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        saved.append(out_path)

    console.print(
        f"[bold green]Synthesis complete.[/bold green] "
        f"{len(saved)} samples saved to [cyan]{out_dir}[/cyan]\n"
    )
    return saved


def load_confusable_phrases(cache_file: Path) -> list[str]:
    """Load user-reviewed confusable phrases from an editable plain-text cache."""
    if not cache_file.exists():
        return []

    phrases: list[str] = []
    for raw_line in cache_file.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            phrases.append(line)
    return phrases


def synthesize_confusable_negatives(
    phrase: str,
    out_dir: Path,
    cache_file: Path,
    n_variants: int = 50,
    engine: str = "kokoro",
    seed: int = 44,
) -> list[Path]:
    """
    Generate synthetic hard-negative clips from phonetically confusable phrases.

    Confusables are loaded from cache_file so users can review and edit the
    phrase list before generating audio.

    Silently skips if:
      - cache_file does not exist
      - cache_file exists but has no non-comment phrases

    Returns list of saved .wav paths (may be empty).
    """
    confusables = load_confusable_phrases(cache_file)
    if not confusables:
        console.print(
            f"[dim]No confusable phrase cache at {cache_file}; skipping confusable negatives.[/dim]"
        )
        return []

    console.print(
        f"\n[bold cyan]Synthesizing {n_variants} confusable negatives[/bold cyan] "
        f"from {len(confusables)} phrases"
    )

    try:
        backend = build_backend(engine)
    except (ImportError, Exception) as e:
        console.print(f"[yellow]TTS not available ({e}), skipping confusable synthesis.[/yellow]")
        return []

    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    speeds = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    saved: list[Path] = []

    for i in track(range(n_variants), description="Confusables..."):
        text = random.choice(confusables)
        speed = random.choice(speeds)
        try:
            audio, sr = backend.synthesize(text, speed=speed)
        except Exception as e:
            console.print(f"[red]Synthesis error on sample {i}: {e}[/red]")
            continue

        if sr != SAMPLE_RATE:
            import torch
            import torchaudio
            wav_t = torch.from_numpy(audio).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
            audio = wav_t.squeeze(0).numpy()

        out_path = out_dir / f"confusable_{i:04d}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        saved.append(out_path)

    console.print(
        f"[bold green]Confusables done.[/bold green] "
        f"{len(saved)} saved to [cyan]{out_dir}[/cyan]\n"
    )
    return saved


def synthesize_partial_negatives(
    phrase: str,
    out_dir: Path,
    n: int = 100,
    engine: str = "kokoro",
    seed: int = 43,
) -> list[Path]:
    """
    Generate synthetic hard-negative examples using only partial phrases.

    For a phrase like "Okay Atlas", this generates clips of just "Okay",
    "okay", etc.
    These are the trickiest false positives: acoustically similar to the real
    wakeword but missing the second word.  CTC loss handles this structurally
    but having them in the training set accelerates convergence.

    Returns [] if the phrase is a single word (no partials possible).
    """
    partials = _partial_variants(phrase)
    if not partials:
        return []

    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold cyan]Synthesizing {n} partial-phrase negatives[/bold cyan] "
        f"for [yellow]{phrase!r}[/yellow]"
    )
    console.print(f"  Partial variants: {partials}")

    try:
        backend = build_backend(engine)
    except ImportError as e:
        console.print(f"[yellow]TTS not available ({e}), skipping partial synthesis.[/yellow]")
        return []

    speeds = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    saved: list[Path] = []

    for i in track(range(n), description="Partial negatives..."):
        text = random.choice(partials)
        speed = random.choice(speeds)
        try:
            audio, sr = backend.synthesize(text, speed=speed)
        except Exception as e:
            console.print(f"[red]Synthesis error on sample {i}: {e}[/red]")
            continue

        if sr != SAMPLE_RATE:
            import torch
            import torchaudio
            wav_t = torch.from_numpy(audio).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
            audio = wav_t.squeeze(0).numpy()

        out_path = out_dir / f"partial_{i:04d}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        saved.append(out_path)

    console.print(
        f"[bold green]Partial negatives done.[/bold green] "
        f"{len(saved)} saved to [cyan]{out_dir}[/cyan]\n"
    )
    return saved
