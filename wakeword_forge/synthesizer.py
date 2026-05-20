"""
synthesizer.py — Generate synthetic positive examples using TTS.

Supports:
  - qwentts (recommended GPU Qwen3-TTS CustomVoice baseline speakers; optional extra)
  - kokoro-onnx (CPU-capable fallback, Apache-2.0, ~82M params)
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
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import track

from .config import SAMPLE_RATE

console = Console()


def _next_numbered_index(out_dir: Path, prefix: str) -> int:
    """Return the next free numeric suffix for files like ``prefix_0001.wav``."""

    highest = -1
    stem_prefix = f"{prefix}_"
    for path in out_dir.glob(f"{stem_prefix}*.wav"):
        suffix = path.stem.removeprefix(stem_prefix)
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


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

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, int]:
        voice = voice or random.choice(self.VOICES)
        samples, sr = self._kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        return samples.astype(np.float32), sr


# ── Piper backend ─────────────────────────────────────────────────────────────

class PiperBackend:
    """
    Wrapper around piper-tts.
    Requires: pip install piper-tts
    Voice files are downloaded on first use into ~/.local/share/piper-voices/.
    """

    def __init__(self, voice: str = "en_US-amy-medium", data_dir: str | None = None) -> None:
        from piper import PiperVoice  # type: ignore

        self._Voice = PiperVoice
        self._voice_name = voice
        self._data_dir = data_dir
        model_path = self._ensure_voice_model(voice, data_dir=data_dir)
        self._piper = PiperVoice.load(
            model_path,
            config_path=Path(f"{model_path}.json"),
            download_dir=model_path.parent,
        )

    @staticmethod
    def _ensure_voice_model(voice: str, data_dir: str | None = None) -> Path:
        """Resolve or download a Piper .onnx voice model."""

        requested = Path(voice).expanduser()
        if requested.suffix == ".onnx" or requested.exists():
            model_path = requested
        else:
            import subprocess
            import sys

            model_dir = Path(data_dir).expanduser() if data_dir else Path.home() / ".local" / "share" / "piper-voices"
            model_path = model_dir / f"{voice}.onnx"
            config_path = Path(f"{model_path}.json")
            if not model_path.exists() or not config_path.exists():
                model_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "piper.download_voices",
                        "--download-dir",
                        str(model_dir),
                        voice,
                    ],
                    check=True,
                )
        config_path = Path(f"{model_path}.json")
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Piper voice model/config not found for {voice!r}")
        return model_path

    def synthesize(self, text: str, **_) -> tuple[np.ndarray, int]:
        chunks = list(self._piper.synthesize(text))
        if not chunks:
            raise RuntimeError("Piper returned no audio chunks")
        sample_rate = int(chunks[0].sample_rate)
        audio = np.concatenate([np.asarray(chunk.audio_float_array, dtype=np.float32) for chunk in chunks])
        return audio, sample_rate


# ── QwenTTS baseline CustomVoice backend ──────────────────────────────────────

DEFAULT_QWENTTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

QWENTTS_VOICE_GROUPS = ("english_voices", "french_voices", "additional_voices")

QWENTTS_VOICE_DESIGNS: dict[str, tuple[dict[str, str], ...] | str] = {
    "english_voices": (
        {
            "name": "Ryan",
            "language": "English",
            "gender": "male",
            "description": "Dynamic male voice with strong rhythmic drive",
        },
        {
            "name": "Aiden",
            "language": "English",
            "gender": "male",
            "description": "Sunny American male voice with a clear midrange",
        },
    ),
    "french_voices": (
        {
            "name": "Vivian",
            "language": "French",
            "gender": "female",
            "native_language": "Chinese",
            "description": "Bright, slightly edgy young female voice",
        },
        {
            "name": "Serena",
            "language": "French",
            "gender": "female",
            "native_language": "Chinese",
            "description": "Warm, gentle young female voice",
        },
    ),
    "additional_voices": (
        {
            "name": "Uncle_Fu",
            "language": "English",
            "gender": "male",
            "native_language": "Chinese",
            "description": "Seasoned male voice with a low, mellow timbre",
        },
        {
            "name": "Dylan",
            "language": "English",
            "gender": "male",
            "native_language": "Chinese (Beijing Dialect)",
            "description": "Youthful Beijing male voice with a clear, natural timbre",
        },
        {
            "name": "Eric",
            "language": "English",
            "gender": "male",
            "native_language": "Chinese (Sichuan Dialect)",
            "description": "Lively Chengdu male voice with a slightly husky brightness",
        },
        {
            "name": "Ono_Anna",
            "language": "English",
            "gender": "female",
            "native_language": "Japanese",
            "description": "Playful Japanese female voice with a light, nimble timbre",
        },
        {
            "name": "Sohee",
            "language": "English",
            "gender": "female",
            "native_language": "Korean",
            "description": "Warm Korean female voice with rich emotion",
        },
        {
            "name": "Ryan_fr",
            "speaker": "Ryan",
            "language": "French",
            "gender": "male",
            "native_language": "English (American)",
            "description": "Ryan (American male) synthesizing French — yields US-accented French pronunciation for English wake phrases",
        },
        {
            "name": "Aiden_fr",
            "speaker": "Aiden",
            "language": "French",
            "gender": "male",
            "native_language": "English (American)",
            "description": "Aiden (American male) synthesizing French — yields a second US-accented French rendition",
        },
        {
            "name": "Vivian_en",
            "speaker": "Vivian",
            "language": "English",
            "gender": "female",
            "native_language": "Chinese",
            "description": "Vivian (Chinese-native French voice) synthesizing English — adds Chinese-accented English variety",
        },
        {
            "name": "Serena_en",
            "speaker": "Serena",
            "language": "English",
            "gender": "female",
            "native_language": "Chinese",
            "description": "Serena (Chinese-native French voice) synthesizing English — adds a second Chinese-accented English variety",
        },
    ),
    "notes": "All 9 Qwen3-TTS CustomVoice speakers can speak any of the 10 supported languages. english_voices and french_voices cover the primary synthesis targets. additional_voices add accent and cross-language diversity: the *_fr entries run English CustomVoice speakers in French (US-accented French), while *_en entries run French CustomVoice speakers in English (Chinese-accented English). The optional 'speaker' field specifies the CustomVoice API name when it differs from the unique 'name' identifier used for file naming.",
}


def _flatten_qwentts_voice_designs(
    designs: dict[str, tuple[dict[str, str], ...] | str] = QWENTTS_VOICE_DESIGNS,
) -> tuple[dict[str, str], ...]:
    voices: list[dict[str, str]] = []
    for group in QWENTTS_VOICE_GROUPS:
        raw_group = designs.get(group, ())
        if isinstance(raw_group, str):
            continue
        for voice in raw_group:
            voices.append(dict(voice))
    return tuple(voices)


QWENTTS_BASELINE_VOICES: tuple[dict[str, str], ...] = _flatten_qwentts_voice_designs()

QWENTTS_STYLE_INSTRUCTIONS: tuple[str, ...] = (
    "",
    "Speak slowly and clearly.",
    "Speak quickly and casually.",
    "Speak with a whisper.",
    "Speak with enthusiasm and energy.",
    "Speak in a calm, relaxed tone.",
    "Speak as if calling someone from across the room.",
    "Speak as if you just woke up, groggy and half-asleep.",
    "Speak while distracted, casually, as if doing something else.",
    "Speak as if firmly commanding a device.",
    "Speak in a flat, expressionless monotone.",
    "Speak with rising intonation, as if asking a question.",
    "Speak with mild impatience or frustration.",
    "Speak in a bright, cheerful, upbeat manner.",
    "Speak with a British English accent.",
    "Speak with an Indian English accent.",
    "Speak with a noticeable French accent.",
    "Speak with a Québécois French accent.",
    "Say it very fast, blending the words together as one.",
    "Say it with a clear, deliberate pause between each word.",
)


def _first_mapping_value(result: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = result.get(key)
        if value is not None:
            return value
    return None


def _unwrap_qwentts_audio(result: Any) -> tuple[np.ndarray, int]:
    if isinstance(result, tuple) and len(result) == 2:
        audio, sample_rate = result
    elif isinstance(result, dict):
        audio = _first_mapping_value(result, ("wavs", "audio", "waveform"))
        sample_rate = _first_mapping_value(result, ("sample_rate", "sampling_rate", "sr"))
    else:
        raise RuntimeError(f"Unsupported QwenTTS result type: {type(result).__name__}")
    if isinstance(audio, (list, tuple)):
        if not audio:
            raise RuntimeError("QwenTTS returned no audio arrays")
        audio = audio[0]
    if audio is None or sample_rate is None:
        raise RuntimeError("QwenTTS result must include audio and sample_rate")
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(
            axis=0 if audio_array.shape[0] <= audio_array.shape[-1] else 1
        )
    return audio_array.astype(np.float32), int(sample_rate)


def _post_process_qwentts_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if not audio.size:
        return audio
    audio = audio - float(audio.mean())
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


class QwenTTSBackend:
    """Baseline Qwen3-TTS CustomVoice source for ordinary synthetic samples.

    This is distinct from the voice-cloning flow: it uses Qwen's built-in
    CustomVoice speakers and optional natural-language style instructions, not
    external reference audio.
    """

    VOICES = QWENTTS_BASELINE_VOICES
    INSTRUCTIONS = QWENTTS_STYLE_INSTRUCTIONS

    def __init__(
        self,
        *,
        voice: str | None = None,
        instructions: Sequence[str] | None = None,
        model_name: str = DEFAULT_QWENTTS_MODEL,
        device: str = "cuda:0",
        dtype: str | Any = "bfloat16",
        use_flash_attn: bool = True,
    ) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore

        self._model_cls = Qwen3TTSModel
        self._torch = torch
        self.voice = voice
        self.instructions = tuple(self.INSTRUCTIONS if instructions is None else instructions)
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.use_flash_attn = use_flash_attn
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        import importlib.util
        import os

        attn_implementation = "flash_attention_2" if self.use_flash_attn and importlib.util.find_spec("flash_attn") else "sdpa"
        self._model = self._model_cls.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=self.dtype,
            attn_implementation=attn_implementation,
            token=os.environ.get("HF_TOKEN"),
        )
        return self._model

    def _select_voice(self, voice: str | None = None) -> dict[str, str]:
        requested = voice or self.voice
        if requested:
            for voice_info in self.VOICES:
                if requested in {voice_info["name"], voice_info.get("speaker", voice_info["name"])}:
                    return voice_info
            raise ValueError(f"Unknown QwenTTS baseline voice: {requested!r}")
        return random.choice(self.VOICES)

    def _select_instruction(self, instruct: str | None = None) -> str:
        if instruct is not None:
            return instruct
        return random.choice(self.instructions) if self.instructions else ""

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        instruct: str | None = None,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, int]:
        _ = speed  # QwenTTS controls delivery through natural-language instructions.
        voice_info = self._select_voice(voice)
        speaker = voice_info.get("speaker", voice_info["name"])
        kwargs = {
            "text": text,
            "language": voice_info.get("language", "English"),
            "speaker": speaker,
        }
        instruction = self._select_instruction(instruct)
        if instruction:
            kwargs["instruct"] = instruction
        audio, sample_rate = _unwrap_qwentts_audio(self._get_model().generate_custom_voice(**kwargs))
        return _post_process_qwentts_audio(audio), sample_rate


# ── No-op backend ─────────────────────────────────────────────────────────────

class NoneBackend:
    def synthesize(self, text: str, **_) -> tuple[np.ndarray, int]:
        raise RuntimeError("TTS engine set to 'none' — skipping synthesis.")


# ── Factory ───────────────────────────────────────────────────────────────────

def build_backend(engine: str) -> TTSBackend:
    normalized = engine.strip().lower().replace("_", "-")
    if normalized == "kokoro":
        return KokoroBackend()
    elif normalized == "piper":
        return PiperBackend()
    elif normalized in {"qwen", "qwentts", "qwen-tts", "qwen3-tts"}:
        return QwenTTSBackend()
    elif normalized == "none":
        return NoneBackend()
    else:
        raise ValueError(f"Unknown TTS engine: {engine!r}")


# ── Main synthesize function ──────────────────────────────────────────────────

def synthesize_positives(
    phrase: str,
    out_dir: Path,
    n: int = 300,
    engine: str = "qwentts",
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
    start_index = _next_numbered_index(out_dir, "synth")

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

        out_path = out_dir / f"synth_{start_index + len(saved):05d}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        saved.append(out_path)

    console.print(
        f"[bold green]Synthesis complete.[/bold green] "
        f"{len(saved)} samples saved to [cyan]{out_dir}[/cyan]\n"
    )
    return saved


def _clean_phrase_sequence(phrases: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        value = " ".join(str(phrase).strip().split())
        if value and value not in seen:
            cleaned.append(value)
            seen.add(value)
    return tuple(cleaned)


def _split_count(total: int, buckets: int) -> list[int]:
    if total <= 0 or buckets <= 0:
        return []
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def synthesize_positive_phrases(
    phrases: Sequence[str],
    out_dir: Path,
    n: int = 300,
    engine: str = "qwentts",
    seed: int = 42,
) -> list[Path]:
    """Generate synthetic positives distributed across one or more trigger phrases."""

    phrase_list = _clean_phrase_sequence(phrases)
    if not phrase_list or n <= 0:
        return []
    if len(phrase_list) == 1:
        return synthesize_positives(phrase_list[0], out_dir, n=n, engine=engine, seed=seed)

    saved: list[Path] = []
    for index, (phrase, count) in enumerate(zip(phrase_list, _split_count(n, len(phrase_list)))):
        if count:
            saved.extend(synthesize_positives(phrase, out_dir, n=count, engine=engine, seed=seed + index))
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
    engine: str = "qwentts",
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
    start_index = _next_numbered_index(out_dir, "confusable")

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

        out_path = out_dir / f"confusable_{start_index + len(saved):04d}.wav"
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
    engine: str = "qwentts",
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
    start_index = _next_numbered_index(out_dir, "partial")

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

        out_path = out_dir / f"partial_{start_index + len(saved):04d}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        saved.append(out_path)

    console.print(
        f"[bold green]Partial negatives done.[/bold green] "
        f"{len(saved)} saved to [cyan]{out_dir}[/cyan]\n"
    )
    return saved
