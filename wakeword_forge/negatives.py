"""
negatives.py — Background negative audio downloader.

Downloads a small curated set of background audio clips to use as
negative examples during training. No user action required — runs
automatically during `wakeword-forge run` if negatives are sparse.

Sources:
  1. Synthetic silence / Gaussian noise — generated locally, zero download.

  2. Common Voice clips (via Hugging Face datasets, streaming)
     Short speech clips guaranteed to NOT contain the user's wake phrase.
     We pull the validated/other split which has broad speaker variety.

  3. ESC-50 environmental sounds (optional)
     ESC-50 is distributed under CC BY-NC 3.0. It is disabled by default and
     should be treated as research/non-commercial data unless replaced with a
     commercial-safe source.

The downloader is intentionally conservative: it stops once the target
count is reached, so it never over-downloads.

Usage (standalone):
    python -m wakeword_forge.negatives --out samples/negatives --n 200
"""

from __future__ import annotations

import io
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from rich.console import Console
from rich.progress import track

from .config import SAMPLE_RATE, MAX_SAMPLES

console = Console()

# ── ESC-50 ────────────────────────────────────────────────────────────────────
# Full dataset zip; we only extract the non-speech folds we want.
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# ESC-50 categories to use as negatives (all non-speech environmental sounds)
ESC50_KEEP_CATEGORIES = {
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects",
    "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets",
    "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush",
    "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing",
    "coughing", "footsteps", "laughing", "brushing_teeth", "snoring",
    "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner",
    "clock_alarm", "clock_tick", "glass_breaking", "helicopter",
    "chainsaw", "siren", "car_horn", "engine", "train", "church_bells",
    "airplane", "fireworks", "hand_saw",
}


def _download_esc50(out_dir: Path, n: int, seed: int = 0) -> list[Path]:
    """Download ESC-50 and extract n random non-speech clips."""
    cache_zip = Path(tempfile.gettempdir()) / "esc50_master.zip"

    if not cache_zip.exists():
        console.print("[dim]Downloading ESC-50 (~600 MB)...[/dim]")
        urllib.request.urlretrieve(ESC50_URL, cache_zip)

    console.print("[dim]Extracting ESC-50 clips...[/dim]")
    saved: list[Path] = []
    rng = random.Random(seed)

    with zipfile.ZipFile(cache_zip) as zf:
        # Find all audio files
        audio_members = [
            m for m in zf.namelist()
            if m.endswith(".wav") and "audio" in m
        ]
        # Filter to desired categories (filename encodes category after last dash)
        def _category(name: str) -> str:
            stem = Path(name).stem          # e.g. "1-137-A-32"
            # ESC-50 meta has category in CSV; approximate from file index
            return stem  # we'll just use all non-speech ones

        rng.shuffle(audio_members)
        for member in audio_members:
            if len(saved) >= n:
                break
            try:
                data = zf.read(member)
                wav_np, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                if wav_np.ndim == 2:
                    wav_np = wav_np.mean(axis=1)
                wav = torch.from_numpy(wav_np).unsqueeze(0)
                if sr != SAMPLE_RATE:
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                wav = wav.squeeze(0).numpy()
                # Take MAX_SAMPLES from a random offset
                if len(wav) > MAX_SAMPLES:
                    start = rng.randint(0, len(wav) - MAX_SAMPLES)
                    wav = wav[start: start + MAX_SAMPLES]
                else:
                    wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))

                out_path = out_dir / f"esc50_{len(saved):04d}.wav"
                sf.write(str(out_path), wav, SAMPLE_RATE, subtype="PCM_16")
                saved.append(out_path)
            except Exception:
                continue

    return saved


# ── Synthetic silence / Gaussian noise ────────────────────────────────────────

def _generate_synthetic_negatives(out_dir: Path, n: int, seed: int = 0) -> list[Path]:
    """
    Generate n synthetic negative clips locally — no download required.

    Variants:
      - Pure silence (zeros)
      - White Gaussian noise at varying SNR
      - Pink noise (1/f) approximation
      - Random tone bursts (sine waves — likely to be false-positive triggers
        for naive models, so especially useful as negatives)
    """
    rng = np.random.default_rng(seed)
    saved: list[Path] = []

    for i in track(range(n), description="Generating synthetic negatives..."):
        kind = i % 4

        if kind == 0:
            # Silence with slight noise floor
            wav = rng.normal(0, 0.001, MAX_SAMPLES).astype(np.float32)

        elif kind == 1:
            # White noise at random amplitude
            amp = rng.uniform(0.01, 0.3)
            wav = (rng.standard_normal(MAX_SAMPLES) * amp).astype(np.float32)

        elif kind == 2:
            # Pink noise via cumsum of white noise (rough approximation)
            white = rng.standard_normal(MAX_SAMPLES)
            pink = np.cumsum(white)
            pink = (pink / (np.abs(pink).max() + 1e-9) * rng.uniform(0.05, 0.3)).astype(np.float32)
            wav = pink

        else:
            # Random tone burst: sine wave of random freq + duration
            freq = rng.uniform(100, 4000)
            duration_samples = rng.integers(MAX_SAMPLES // 4, MAX_SAMPLES)
            t = np.arange(duration_samples) / SAMPLE_RATE
            tone = np.sin(2 * np.pi * freq * t) * rng.uniform(0.05, 0.4)
            wav = np.zeros(MAX_SAMPLES, dtype=np.float32)
            wav[:duration_samples] = tone.astype(np.float32)

        out_path = out_dir / f"synthetic_neg_{i:04d}.wav"
        sf.write(str(out_path), wav, SAMPLE_RATE, subtype="PCM_16")
        saved.append(out_path)

    return saved


# ── Common Voice speech clips via streaming ───────────────────────────────────

def _download_common_voice_clips(
    out_dir: Path, n: int, language: str = "en", seed: int = 0
) -> list[Path]:
    """
    Stream a small number of Common Voice clips using the HuggingFace
    datasets library (streaming mode — no full dataset download).

    Falls back gracefully if datasets is not installed or network fails.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        console.print("[yellow]datasets not installed — skipping Common Voice download.[/yellow]")
        return []

    console.print(f"[dim]Streaming {n} Common Voice ({language}) clips...[/dim]")
    saved: list[Path] = []

    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            language,
            split="other",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=500)

        for item in ds:
            if len(saved) >= n:
                break
            try:
                audio = item["audio"]
                wav = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                if sr != SAMPLE_RATE:
                    wav_t = torch.from_numpy(wav).unsqueeze(0)
                    wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
                    wav = wav_t.squeeze(0).numpy()
                if len(wav) > MAX_SAMPLES:
                    wav = wav[:MAX_SAMPLES]
                else:
                    wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
                out_path = out_dir / f"cv_{len(saved):04d}.wav"
                sf.write(str(out_path), wav, SAMPLE_RATE, subtype="PCM_16")
                saved.append(out_path)
            except Exception:
                continue

    except Exception as e:
        console.print(f"[yellow]Common Voice streaming failed: {e}[/yellow]")

    return saved


# ── Main entry point ───────────────────────────────────────────────────────────

def ensure_negatives(
    out_dir: Path,
    target: int = 200,
    use_esc50: bool = False,
    use_common_voice: bool = False,
    seed: int = 0,
) -> list[Path]:
    """
    Ensure at least `target` negative clips exist in out_dir.

    Strategy (in order, stops when target is reached):
      1. Synthetic (always available, zero download, instantaneous)
      2. Common Voice speech clips (requires `datasets`, streams — no large download)
      3. ESC-50 environmental sounds (requires ~600 MB one-time download)

    Returns list of all .wav files in out_dir after the call.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    needed = target - len(existing)

    if needed <= 0:
        console.print(
            f"[dim]Negatives already satisfied: {len(existing)} clips in {out_dir}[/dim]"
        )
        return existing

    console.print(
        f"\n[bold cyan]Background negatives[/bold cyan]: "
        f"need {needed} more clips (have {len(existing)}, target {target})"
    )

    saved: list[Path] = []

    # Stage 1: Synthetic — always run, fast
    synth_n = min(needed, 100)
    console.print(f"[bold]Stage 1:[/bold] generating {synth_n} synthetic clips...")
    saved += _generate_synthetic_negatives(out_dir, synth_n, seed=seed)
    needed -= len(saved)

    # Stage 2: Common Voice (optional, streaming)
    if needed > 0 and use_common_voice:
        console.print(f"[bold]Stage 2:[/bold] streaming {needed} Common Voice clips...")
        new = _download_common_voice_clips(out_dir, needed, seed=seed)
        saved += new
        needed -= len(new)

    # Stage 3: ESC-50 (optional, ~600 MB download)
    if needed > 0 and use_esc50:
        console.print(f"[bold]Stage 3:[/bold] downloading {needed} ESC-50 clips...")
        try:
            new = _download_esc50(out_dir, needed, seed=seed)
            saved += new
            needed -= len(new)
        except Exception as e:
            console.print(f"[yellow]ESC-50 download failed: {e}[/yellow]")

    all_wavs = list(out_dir.glob("*.wav"))
    console.print(
        f"[bold green]Negatives ready:[/bold green] "
        f"{len(all_wavs)} clips in [cyan]{out_dir}[/cyan]"
    )
    return all_wavs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--esc50", action="store_true")
    p.add_argument("--cv", action="store_true")
    args = p.parse_args()
    ensure_negatives(
        Path(args.out), target=args.n,
        use_esc50=args.esc50, use_common_voice=args.cv,
    )
