"""Voice-cloned sample sourcing, validation, and review staging.

This module deliberately keeps QwenTTS itself outside the host Python process.
Host-side code plans exactly one clone job, invokes a Dockerized QwenTTS runner,
transcribes/validates the result, then stages the WAV for explicit human review.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE, ForgeConfig

ReviewDecision = Literal["positive", "negative", "unusable"]
SourceType = Literal["open_dataset", "youtube", "local", "other"]

YOUTUBE_SOURCE_TYPES = {"youtube", "yt", "youtube_video", "youtube-video"}
YOUTUBE_URL_KEYS = ("youtube_url", "source_url", "url")
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus")
DEFAULT_QWENTTS_IMAGE = "wakeword-forge-qwentts:latest"
DEFAULT_QWENTTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


class SourcePolicyError(ValueError):
    """Raised when source audio is disabled by responsible-use policy."""


@dataclass(frozen=True)
class PhraseMatch:
    """Wake-phrase transcript match result."""

    matched: bool
    phrase: str = ""
    score: float = 0.0
    method: str = "none"


@dataclass(frozen=True)
class CloneValidation:
    """Audio + transcript validation result for one generated clone."""

    passed: bool
    transcript: str
    match: PhraseMatch
    duration_sec: float
    peak: float
    rms: float
    reasons: tuple[str, ...]
    suggested_label: ReviewDecision


@dataclass(frozen=True)
class ClonedReviewItem:
    """One pending human-review item for a voice-cloned sample."""

    audio_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VoiceClonePipelineResult:
    """Summary of generating and staging exactly one cloned sample."""

    source_row: dict[str, Any]
    reference_candidate: dict[str, Any]
    job_file: Path
    generated_path: Path
    validation: CloneValidation
    staged_item: ClonedReviewItem


class WhisperTranscriber:
    """Lazy OpenAI Whisper transcriber wrapper used for source/output STT."""

    def __init__(self, *, model_name: str = "base", device: str | None = None, language: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.language = language
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import whisper  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Voice-clone transcription requires openai-whisper. Install the voice extra "
                "or run in an environment with the `whisper` Python package."
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = whisper.load_model(self.model_name, **kwargs)
        return self._model

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"word_timestamps": False}
        if self.language:
            kwargs["language"] = self.language
        return self._get_model().transcribe(str(audio_path), **kwargs)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_source_type(source_type: Any) -> str:
    normalized = str(source_type or "open_dataset").strip().lower()
    return "youtube" if normalized in YOUTUBE_SOURCE_TYPES else normalized


def looks_like_youtube_url(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return "youtube.com" in text or "youtu.be" in text


def infer_source_type(row: dict[str, Any], default: Any = "open_dataset") -> str:
    if any(looks_like_youtube_url(row.get(key)) for key in YOUTUBE_URL_KEYS):
        return "youtube"
    explicit = row.get("source_type") or row.get("type")
    return normalize_source_type(explicit or default)


def require_youtube_opt_in(source_type: Any, *, allow_youtube: bool) -> str:
    normalized = normalize_source_type(source_type)
    if normalized == "youtube" and not allow_youtube:
        raise SourcePolicyError(
            "YouTube source audio is disabled by default. Pass --allow-youtube only for "
            "personal-use experiments, fair-use contexts you have evaluated, or rights-cleared data."
        )
    return normalized


def _non_empty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _speaker_id(row: dict[str, Any], fallback_path: Path | None = None) -> str:
    for key in ("speaker_id", "source_speaker_id", "speaker", "client_id", "reader_id", "creator_id"):
        if _non_empty(row.get(key)):
            return str(row[key])
    if fallback_path is not None:
        return fallback_path.stem
    return "unknown-speaker"


def _resolve_manifest_path(value: Any, *, manifest_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((manifest_dir / path).resolve())


def load_source_manifest(
    manifest_path: Path | str,
    *,
    allow_youtube: bool = False,
    default_source_id: str = "voice_clone_sources",
    default_source_type: SourceType = "open_dataset",
) -> list[dict[str, Any]]:
    """Load source-audio rows and enforce responsible-use source policy.

    Rows may point at local audio via ``path`` or remote audio via ``url`` /
    ``youtube_url``. YouTube rows are always opt-in because copyright, consent,
    and platform terms vary by use case.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for raw in _read_jsonl(manifest):
        row = dict(raw)
        source_type = require_youtube_opt_in(
            infer_source_type(row, row.get("source_type") or default_source_type),
            allow_youtube=allow_youtube,
        )
        row["source_type"] = source_type
        row["source_id"] = str(row.get("source_id") or default_source_id)

        if _non_empty(row.get("path")):
            row["path"] = _resolve_manifest_path(row["path"], manifest_dir=manifest.parent)
        if _non_empty(row.get("transcript_json")):
            row["transcript_json"] = _resolve_manifest_path(row["transcript_json"], manifest_dir=manifest.parent)

        fallback_path = Path(row["path"]) if _non_empty(row.get("path")) else None
        speaker = _speaker_id(row, fallback_path=fallback_path)
        row["speaker_id"] = speaker
        row.setdefault("source_speaker_id", speaker)
        row.setdefault("license", "")
        row.setdefault("usage_policy", "")
        rows.append(row)
    return rows


def _safe_download_name(row: dict[str, Any], suffix: str = ".wav") -> str:
    token = json.dumps(
        {key: row.get(key) for key in ("url", "youtube_url", "speaker_id", "source_id")},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(token.encode("utf-8", errors="replace")).hexdigest()[:12]
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(row.get("speaker_id") or "source")).strip("_")
    return f"{stem or 'source'}_{digest}{suffix}"


def download_source_audio(
    row: dict[str, Any],
    dest_dir: Path,
    *,
    allow_youtube: bool = False,
    downloader: Callable[[dict[str, Any], Path], Path] | None = None,
) -> Path:
    """Materialize one source-audio row as a local file.

    The function processes exactly one row per call so callers can keep the
    source→clone→validate→review loop strictly one sample at a time.
    """

    source_type = require_youtube_opt_in(infer_source_type(row, row.get("source_type")), allow_youtube=allow_youtube)
    if _non_empty(row.get("path")):
        path = Path(str(row["path"])).expanduser().resolve()
        if path.exists():
            return path
    if downloader is not None:
        return downloader(row, dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = str(row.get("youtube_url") or row.get("url") or row.get("source_url") or "").strip()
    if not url:
        raise FileNotFoundError(f"No local source audio exists and no URL is available for {row.get('speaker_id')}")

    if source_type == "youtube":  # pragma: no cover - exercised through command contract tests
        output = dest_dir / _safe_download_name(row)
        cmd = ["yt-dlp"]
        cookie_browser = str(row.get("cookies_from_browser") or "").strip() or None
        cookie_browser = cookie_browser or os.environ.get("WAKEWORD_FORGE_YT_COOKIES_BROWSER")
        cookies_file = str(row.get("cookies_file") or "").strip() or None
        cookies_file = cookies_file or os.environ.get("WAKEWORD_FORGE_YT_COOKIES_FILE")
        if cookie_browser:
            cmd.extend(["--cookies-from-browser", cookie_browser])
        elif cookies_file:
            cmd.extend(["--cookies", cookies_file])
        node = shutil.which("node") or shutil.which("nodejs")
        if node:
            cmd.extend(["--js-runtimes", f"node:{node}"])
        cmd.extend([
            "--no-playlist",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            str(output.with_suffix(".%(ext)s")),
            url,
        ])
        completed = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"yt-dlp failed for {url}: {completed.stderr.strip()}")
        if not output.exists():
            matches = sorted(dest_dir.glob(output.with_suffix("").name + ".*"))
            if matches:
                return matches[0]
        return output

    suffix = Path(urllib.parse.urlparse(url).path).suffix
    if suffix.lower() not in AUDIO_EXTENSIONS:
        suffix = ".wav"
    output = dest_dir / _safe_download_name(row, suffix=suffix)
    urllib.request.urlretrieve(url, output)  # noqa: S310 - explicit user-provided dataset URL
    return output


def _word_count(text: str) -> int:
    return len(re.findall(r"[\wÀ-ÿ']+", text, flags=re.UNICODE))


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


SPEAKER_LABEL_RE = re.compile(
    r"(?:^|[\n.!?]\s*)(?:speaker\s*\d+|host|guest|interviewer|interviewee|man|woman|a|b)\s*:",
    re.IGNORECASE,
)
CROSSTALK_MARKERS = ("[crosstalk", "[overlap", "[music", "(crosstalk", "(overlap")


def _looks_single_speaker(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    lower = cleaned.lower()
    if any(marker in lower for marker in CROSSTALK_MARKERS):
        return False
    return len(SPEAKER_LABEL_RE.findall(cleaned)) <= 1


def score_reference_candidate(
    *,
    text: str,
    duration_sec: float,
    avg_logprob: float | None = None,
    no_speech_prob: float | None = None,
) -> dict[str, Any]:
    """Return heuristic quality components for a source voice-clone reference."""

    words = _word_count(text)
    word_score = _clamp01((words - 2) / 10.0)
    duration_score = _clamp01(1.0 - abs(duration_sec - 5.0) / 8.0)
    confidence_score = 0.7 if avg_logprob is None else _clamp01((float(avg_logprob) + 1.5) / 1.5)
    speech_score = 0.7 if no_speech_prob is None else _clamp01(1.0 - float(no_speech_prob))
    single_speaker_score = 1.0 if _looks_single_speaker(text) else 0.0
    quality_score = _clamp01(
        0.30 * word_score
        + 0.25 * duration_score
        + 0.20 * confidence_score
        + 0.15 * speech_score
        + 0.10 * single_speaker_score
    )
    return {
        "quality_score": round(quality_score, 4),
        "quality_components": {
            "word_score": round(word_score, 4),
            "duration_score": round(duration_score, 4),
            "whisper_confidence_score": round(confidence_score, 4),
            "speech_score": round(speech_score, 4),
            "single_speaker_score": round(single_speaker_score, 4),
        },
    }


def _load_precomputed_transcript(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("whisper_result"), dict):
        return row["whisper_result"]
    if _non_empty(row.get("transcript_json")):
        path = Path(str(row["transcript_json"])).expanduser()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected transcript JSON object in {path}")
        return data
    return None


def build_candidate_rows(
    source_rows: Iterable[dict[str, Any]],
    *,
    transcriber: WhisperTranscriber | None = None,
    min_duration_sec: float = 2.0,
    max_duration_sec: float = 12.0,
    min_words: int = 5,
    min_quality_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Transcribe source audio and return candidate reference snippets."""

    candidates: list[dict[str, Any]] = []
    lazy_transcriber = transcriber
    for source_row in source_rows:
        transcript = _load_precomputed_transcript(source_row)
        if transcript is None:
            if lazy_transcriber is None:
                lazy_transcriber = WhisperTranscriber()
            transcript = lazy_transcriber.transcribe(Path(str(source_row["path"])))

        for seg in transcript.get("segments", []):
            text = str(seg.get("text", "")).strip()
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
            except (TypeError, ValueError):
                continue
            duration = round(end - start, 4)
            if duration < min_duration_sec or duration > max_duration_sec:
                continue
            if _word_count(text) < min_words or not _looks_single_speaker(text):
                continue
            score = score_reference_candidate(
                text=text,
                duration_sec=duration,
                avg_logprob=seg.get("avg_logprob"),
                no_speech_prob=seg.get("no_speech_prob"),
            )
            if score["quality_score"] < min_quality_score:
                continue
            row = {
                "path": source_row["path"],
                "speaker_id": str(source_row.get("speaker_id") or Path(str(source_row["path"])).stem),
                "source_speaker_id": str(
                    source_row.get("source_speaker_id")
                    or source_row.get("speaker_id")
                    or Path(str(source_row["path"])).stem
                ),
                "source_id": source_row.get("source_id", "voice_clone_sources"),
                "dataset_id": source_row.get("dataset_id", ""),
                "source_type": normalize_source_type(source_row.get("source_type", "open_dataset")),
                "language": source_row.get("language") or transcript.get("language") or "English",
                "license": source_row.get("license", ""),
                "usage_policy": source_row.get("usage_policy", ""),
                "start_sec": round(start, 4),
                "end_sec": round(end, 4),
                "duration_sec": duration,
                "reference_transcript": text,
                "whisper_transcript": text,
                "single_speaker": True,
                "quality_score": score["quality_score"],
                "quality_components": score["quality_components"],
            }
            for key in ("youtube_url", "source_url", "source_description", "runtime_path"):
                if key in source_row:
                    row[key] = source_row[key]
            candidates.append(row)
    return candidates


def _speaker_hash(source_id: str, dataset_id: str, speaker_id: str) -> str:
    token = f"{source_id}::{dataset_id}::{speaker_id}".encode("utf-8", errors="replace")
    return hashlib.sha1(token).hexdigest()[:12]


def _candidate_quality(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("quality_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def select_reference_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Keep at most one best reference snippet per source/dataset/speaker identity."""

    best_by_speaker: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in candidates:
        candidate = dict(raw)
        key = (
            str(candidate.get("source_id", "voice_clone_sources")),
            str(candidate.get("dataset_id", "")),
            str(candidate.get("source_speaker_id") or candidate.get("speaker_id") or "speaker"),
        )
        previous = best_by_speaker.get(key)
        if previous is None or (
            _candidate_quality(candidate), len(str(candidate.get("reference_transcript", "")))
        ) > (_candidate_quality(previous), len(str(previous.get("reference_transcript", "")))):
            candidate["speaker_hash"] = _speaker_hash(*key)
            best_by_speaker[key] = candidate
    selected = sorted(
        best_by_speaker.values(),
        key=lambda item: (-_candidate_quality(item), str(item.get("source_id", "")), str(item.get("speaker_id", ""))),
    )
    return selected[:max_speakers] if max_speakers is not None else selected


def normalize_transcript(text: str) -> str:
    lowered = text.lower().replace("okay", "ok")
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def _phrase_token_windows(tokens: list[str], phrase_len: int) -> Iterable[list[str]]:
    for start in range(0, len(tokens) - phrase_len + 1):
        yield tokens[start : start + phrase_len]


def _embedded_substring_mismatch(expected: str, actual: str) -> bool:
    return expected != actual and (expected in actual or actual in expected)


def _fuzzy_token_score(
    phrase_tokens: list[str],
    window_tokens: list[str],
    *,
    token_threshold: float,
) -> float:
    if len(phrase_tokens) != len(window_tokens) or len(phrase_tokens) < 2:
        return 0.0
    scores: list[float] = []
    for expected, actual in zip(phrase_tokens, window_tokens, strict=True):
        if len(expected) <= 2 or len(actual) <= 2:
            if expected != actual:
                return 0.0
            scores.append(1.0)
            continue
        if _embedded_substring_mismatch(expected, actual):
            return 0.0
        score = SequenceMatcher(None, expected, actual).ratio()
        if score < token_threshold:
            return 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def transcript_matches_phrase(
    transcript: str,
    wake_phrases: Iterable[str],
    *,
    fuzzy_threshold: float = 0.78,
) -> PhraseMatch:
    """Return whether a transcript contains or fuzzily matches any wake phrase."""

    normalized_transcript = normalize_transcript(transcript)
    tokens = normalized_transcript.split()
    best = PhraseMatch(False)
    for phrase in wake_phrases:
        normalized_phrase = normalize_transcript(phrase)
        if not normalized_phrase:
            continue
        phrase_tokens = normalized_phrase.split()
        phrase_len = len(phrase_tokens)
        exact_match = any(
            tokens[start : start + phrase_len] == phrase_tokens
            for start in range(0, len(tokens) - phrase_len + 1)
        )
        if exact_match:
            return PhraseMatch(True, phrase=phrase, score=1.0, method="contains")
        for window_tokens in _phrase_token_windows(tokens, phrase_len):
            score = _fuzzy_token_score(
                phrase_tokens,
                window_tokens,
                token_threshold=fuzzy_threshold,
            )
            if score > best.score:
                best = PhraseMatch(
                    score >= fuzzy_threshold,
                    phrase=phrase,
                    score=round(score, 4),
                    method="fuzzy",
                )
    return best if best.matched else PhraseMatch(False, phrase=best.phrase, score=best.score, method="none")


def _transcript_text(result: dict[str, Any]) -> str:
    if _non_empty(result.get("text")):
        return str(result["text"])
    segments = result.get("segments")
    if isinstance(segments, list):
        return " ".join(str(seg.get("text", "")).strip() for seg in segments if isinstance(seg, dict)).strip()
    return ""


def validate_cloned_audio(
    audio_path: Path | str,
    wake_phrases: Iterable[str],
    *,
    transcript: str | None = None,
    transcriber: WhisperTranscriber | None = None,
    min_duration_sec: float = 0.15,
    max_duration_sec: float = 3.5,
    silence_peak_threshold: float = 0.006,
    silence_rms_threshold: float = 0.002,
) -> CloneValidation:
    """Validate one generated clone by audio energy and STT/fuzzy wake-phrase match."""

    path = Path(audio_path).expanduser()
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    duration_sec = len(audio) / float(sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0

    if transcript is None:
        if transcriber is None:
            transcriber = WhisperTranscriber()
        transcript = _transcript_text(transcriber.transcribe(path))
    match = transcript_matches_phrase(transcript or "", wake_phrases)

    reasons: list[str] = []
    if duration_sec < min_duration_sec:
        reasons.append(f"duration too short: {duration_sec:.3f}s")
    if duration_sec > max_duration_sec:
        reasons.append(f"duration too long: {duration_sec:.3f}s")
    if peak < silence_peak_threshold or rms < silence_rms_threshold:
        reasons.append("audio is silence or near-silence")
    if not match.matched:
        reasons.append("transcript does not contain the wake phrase")

    suggested_label: ReviewDecision
    if peak < silence_peak_threshold or rms < silence_rms_threshold or duration_sec < min_duration_sec:
        suggested_label = "unusable"
    elif match.matched:
        suggested_label = "positive"
    else:
        suggested_label = "negative"

    return CloneValidation(
        passed=not reasons,
        transcript=transcript or "",
        match=match,
        duration_sec=round(duration_sec, 4),
        peak=round(peak, 6),
        rms=round(rms, 6),
        reasons=tuple(reasons),
        suggested_label=suggested_label,
    )


def _metadata_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(audio_path.suffix + ".json")


def _next_numbered_path(directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = directory / f"{prefix}_{index:04d}.wav"
        if not candidate.exists():
            return candidate
        index += 1


def _clone_review_dir(config: ForgeConfig) -> Path:
    return config.samples_path / "cloned_review"


def stage_cloned_sample_for_review(
    config: ForgeConfig,
    audio_path: Path | str,
    *,
    validation: CloneValidation,
    metadata: dict[str, Any],
) -> ClonedReviewItem:
    """Copy one generated clone into the pending human-review queue."""

    review_dir = config.cloned_review_path if hasattr(config, "cloned_review_path") else _clone_review_dir(config)
    target = _next_numbered_path(review_dir, "cloned")
    shutil.copy2(Path(audio_path).expanduser(), target)
    payload = {
        **metadata,
        "review_status": "pending",
        "suggested_label": validation.suggested_label,
        "validation": asdict(validation),
        "responsible_use": {
            "requires_human_review": True,
            "note": "Use only voices and source audio you have rights, consent, or a defensible fair-use basis to process.",
        },
    }
    meta_path = _metadata_path(target)
    _write_json(meta_path, payload)
    return ClonedReviewItem(audio_path=target, metadata_path=meta_path, metadata=payload)


def list_cloned_review_items(config: ForgeConfig) -> list[ClonedReviewItem]:
    review_dir = config.cloned_review_path if hasattr(config, "cloned_review_path") else _clone_review_dir(config)
    items: list[ClonedReviewItem] = []
    if not review_dir.exists():
        return items
    for audio_path in sorted(review_dir.glob("*.wav")):
        meta_path = _metadata_path(audio_path)
        metadata: dict[str, Any] = {}
        if meta_path.exists():
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        items.append(ClonedReviewItem(audio_path=audio_path, metadata_path=meta_path, metadata=metadata))
    return items


def _resolve_review_audio(config: ForgeConfig, sample: Path | str) -> Path:
    review_dir = (config.cloned_review_path if hasattr(config, "cloned_review_path") else _clone_review_dir(config)).resolve()
    raw = Path(sample).expanduser()
    path = raw if raw.is_absolute() else review_dir / raw
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(review_dir):
        raise ValueError(f"Refusing to review sample outside cloned review queue: {resolved}")
    return resolved


def apply_cloned_sample_decision(
    config: ForgeConfig,
    sample: Path | str,
    decision: ReviewDecision,
) -> Path | None:
    """Apply a human decision: move to training positives/negatives, or delete."""

    if decision not in {"positive", "negative", "unusable"}:
        raise ValueError("decision must be one of: positive, negative, unusable")
    audio_path = _resolve_review_audio(config, sample)
    if not audio_path.exists():
        raise FileNotFoundError(f"Pending cloned sample not found: {audio_path}")
    meta_path = _metadata_path(audio_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if decision == "unusable":
        audio_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        return None

    target_dir = config.positives_path if decision == "positive" else config.negatives_path
    target = _next_numbered_path(target_dir, f"cloned_{decision}")
    shutil.move(str(audio_path), target)
    metadata["review_status"] = decision
    metadata["training_label"] = decision
    metadata["training_path"] = str(target)
    _write_json(_metadata_path(target), metadata)
    if meta_path.exists():
        meta_path.unlink()

    from .review import reset_sample_dependent_approvals

    reset_sample_dependent_approvals(config)
    return target


def write_one_sample_qwentts_job(
    job_file: Path | str,
    *,
    text: str,
    reference_audio: Path,
    reference_text: str,
    output_path: Path,
    language: str = "English",
    instruct: str = "",
    model_name: str = DEFAULT_QWENTTS_MODEL,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one QwenTTS clone job JSON file."""

    payload = {
        "text": text,
        "language": language,
        "reference_audio": str(reference_audio),
        "reference_text": reference_text,
        "instruct": instruct,
        "output_path": str(output_path),
        "model_name": model_name,
        "metadata": metadata or {},
    }
    path = Path(job_file).expanduser()
    _write_json(path, payload)
    return path


def build_qwentts_docker_run_command(
    *,
    job_file: Path | str,
    project_dir: Path | str,
    output_dir: Path | str,
    image: str = DEFAULT_QWENTTS_IMAGE,
    gpus: str = "all",
) -> list[str]:
    """Return a Docker command that runs exactly one QwenTTS job."""

    job_path = Path(job_file).expanduser().resolve()
    project_path = Path(project_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        gpus,
        "-e",
        "HF_TOKEN",
        "-v",
        f"{project_path}:/project:rw",
        "-v",
        f"{job_path.parent}:/jobs:ro",
        "-v",
        f"{output_path}:/outputs:rw",
        image,
        "python3",
        "/workspace/qwentts_clone_one.py",
        "--job",
        f"/jobs/{job_path.name}",
    ]


def run_qwentts_docker_job(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"QwenTTS Docker job failed with exit code {completed.returncode}")


def _candidate_reference_source(candidate: dict[str, Any]) -> Path:
    return Path(str(candidate.get("path"))).expanduser()


def _ffmpeg_clip(source: Path, target: Path, start: float, end: float) -> None:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{max(0.001, end - start):.6f}",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(target),
    ]
    completed = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to cut reference clip: {completed.stderr.strip()}")


def materialize_reference_clip(candidate: dict[str, Any], output_dir: Path) -> Path:
    """Write the selected source segment as one reference WAV for QwenTTS."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source = _candidate_reference_source(candidate)
    start = candidate.get("start_sec")
    end = candidate.get("end_sec")
    fingerprint = hashlib.sha1(json.dumps(candidate, sort_keys=True, default=str).encode()).hexdigest()[:12]
    target = output_dir / f"reference_{fingerprint}.wav"
    if target.exists():
        return target
    if start is None or end is None:
        shutil.copy2(source, target)
        return target
    start_f = float(start)
    end_f = float(end)
    try:
        info = sf.info(str(source))
        sr = int(info.samplerate)
        audio, read_sr = sf.read(
            str(source),
            start=max(0, int(start_f * sr)),
            stop=max(1, int(end_f * sr)),
            dtype="float32",
            always_2d=False,
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sf.write(str(target), audio, int(read_sr), subtype="PCM_16")
    except (RuntimeError, OSError, ValueError):
        _ffmpeg_clip(source, target, start_f, end_f)
    return target


def generate_one_voice_clone_sample(
    config: ForgeConfig,
    *,
    source_manifest: Path,
    allow_youtube: bool = False,
    phrase: str | None = None,
    transcriber: WhisperTranscriber | None = None,
    docker_runner: Callable[[list[str]], None] | None = None,
    image: str = DEFAULT_QWENTTS_IMAGE,
) -> VoiceClonePipelineResult:
    """Run source→STT→select→Docker QwenTTS→STT validation→review staging once."""

    wake_phrase = phrase or (config.phrase_options[0] if config.phrase_options else config.wake_phrase)
    if not wake_phrase:
        raise ValueError("A wake phrase is required before generating cloned samples.")
    project_dir = config.project_path
    clone_cache = config.cache_path / "voice_clone"
    source_rows = load_source_manifest(source_manifest, allow_youtube=allow_youtube)
    selected_source: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    source_errors: list[str] = []
    for row in source_rows:
        try:
            downloaded = {**row, "path": str(download_source_audio(row, clone_cache / "sources", allow_youtube=allow_youtube))}
            candidates = select_reference_candidates(
                build_candidate_rows([downloaded], transcriber=transcriber),
                max_speakers=1,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            source_errors.append(f"{row.get('speaker_id', 'unknown')}: {exc}")
            continue
        if candidates:
            selected_source = downloaded
            candidate = candidates[0]
            break
    if selected_source is None or candidate is None:
        detail = f" Tried sources: {'; '.join(source_errors[:3])}" if source_errors else ""
        raise ValueError(f"No suitable single-speaker reference snippet found in the selected sources.{detail}")
    reference_clip = materialize_reference_clip(candidate, clone_cache / "references")
    generated_dir = clone_cache / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = _next_numbered_path(generated_dir, "qwentts_clone")
    project_output_path = Path("/project") / generated_path.relative_to(project_dir)
    project_reference_path = Path("/project") / reference_clip.relative_to(project_dir)
    job_file = clone_cache / "jobs" / f"{generated_path.stem}.json"
    write_one_sample_qwentts_job(
        job_file,
        text=wake_phrase,
        reference_audio=project_reference_path,
        reference_text=str(candidate.get("reference_transcript", "")),
        output_path=project_output_path,
        language=str(candidate.get("language", "English")),
        metadata={
            "source_type": candidate.get("source_type", ""),
            "source_id": candidate.get("source_id", ""),
            "speaker_hash": candidate.get("speaker_hash", ""),
            "license": candidate.get("license", ""),
            "usage_policy": candidate.get("usage_policy", ""),
        },
    )
    command = build_qwentts_docker_run_command(
        job_file=job_file,
        project_dir=project_dir,
        output_dir=generated_dir,
        image=image,
    )
    (docker_runner or run_qwentts_docker_job)(command)
    if not generated_path.exists():
        raise FileNotFoundError(f"QwenTTS job completed but did not write {generated_path}")
    validation_phrases = (wake_phrase,) if phrase else (config.phrase_options or (wake_phrase,))
    validation = validate_cloned_audio(generated_path, validation_phrases, transcriber=transcriber)
    staged = stage_cloned_sample_for_review(
        config,
        generated_path,
        validation=validation,
        metadata={
            "voice_clone_source": candidate,
            "qwentts_job": str(job_file),
            "source_policy_disclaimer": (
                "Only clone voices from sources you have rights, consent, or a defensible fair-use "
                "basis to process; keep YouTube use opt-in and provenance-tracked."
            ),
        },
    )
    return VoiceClonePipelineResult(
        source_row=selected_source,
        reference_candidate=candidate,
        job_file=job_file,
        generated_path=generated_path,
        validation=validation,
        staged_item=staged,
    )


__all__ = [
    "CloneValidation",
    "ClonedReviewItem",
    "PhraseMatch",
    "SourcePolicyError",
    "VoiceClonePipelineResult",
    "WhisperTranscriber",
    "apply_cloned_sample_decision",
    "build_candidate_rows",
    "build_qwentts_docker_run_command",
    "download_source_audio",
    "generate_one_voice_clone_sample",
    "infer_source_type",
    "list_cloned_review_items",
    "load_source_manifest",
    "materialize_reference_clip",
    "normalize_source_type",
    "normalize_transcript",
    "require_youtube_opt_in",
    "run_qwentts_docker_job",
    "select_reference_candidates",
    "stage_cloned_sample_for_review",
    "transcript_matches_phrase",
    "validate_cloned_audio",
    "write_one_sample_qwentts_job",
]
