#!/usr/bin/env python3
"""Run one QwenTTS custom-voice generation job inside Docker."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TARGET_PEAK = 0.9
REFERENCE_AUDIO_KWARGS = ("ref_audio", "reference_audio", "prompt_audio", "audio_prompt")
REFERENCE_TEXT_KWARGS = ("ref_text", "reference_text", "prompt_text")


def _load_job(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"text", "reference_audio", "reference_text", "output_path"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Job file missing required keys: {', '.join(missing)}")
    return data


def _call_with_supported_kwargs(method: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(method)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return method(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    missing_required = [
        name
        for name, param in signature.parameters.items()
        if name not in supported
        and param.default is inspect.Parameter.empty
        and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    ]
    if missing_required:
        method_name = getattr(method, "__name__", "QwenTTS method")
        raise RuntimeError(
            f"{method_name} requires unsupported argument(s): {', '.join(missing_required)}"
        )
    return method(**supported)


def _supports_any_kwarg(method: Any, names: tuple[str, ...]) -> bool:
    signature = inspect.signature(method)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return True
    return any(name in signature.parameters for name in names)


def _require_reference_audio_support(method: Any, method_name: str) -> None:
    if _supports_any_kwarg(method, REFERENCE_AUDIO_KWARGS):
        return
    raise RuntimeError(
        f"QwenTTS {method_name} does not support reference audio; "
        "this runner only accepts APIs that consume the one-sample reference voice."
    )


def _first_mapping_value(result: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = result.get(key)
        if value is not None:
            return value
    return None


def _unwrap_audio(result: Any) -> tuple[np.ndarray, int]:
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
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _post_process(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0 if audio.shape[0] <= audio.shape[-1] else 1)
    if not audio.size:
        return audio
    audio = audio - float(audio.mean())
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio * (TARGET_PEAK / peak)
    return audio.astype(np.float32)


def _load_model(job: dict[str, Any]) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel  # type: ignore

    dtype_name = str(job.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)
    return Qwen3TTSModel.from_pretrained(
        str(job.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")),
        device_map=str(job.get("device", "cuda:0")),
        dtype=dtype,
        attn_implementation=str(job.get("attn_implementation", "flash_attention_2")),
        token=os.environ.get("HF_TOKEN"),
    )


def _synthesize(job: dict[str, Any]) -> tuple[np.ndarray, int]:
    model = _load_model(job)
    reference_audio = str(job["reference_audio"])
    reference_text = str(job["reference_text"])
    kwargs = {
        "text": str(job["text"]),
        "language": str(job.get("language", "English")),
        "ref_audio": reference_audio,
        "reference_audio": reference_audio,
        "prompt_audio": reference_audio,
        "audio_prompt": reference_audio,
        "ref_text": reference_text,
        "reference_text": reference_text,
        "prompt_text": reference_text,
        "instruct": str(job.get("instruct", "")),
    }
    if not kwargs["instruct"]:
        kwargs.pop("instruct")
    for method_name in ("generate_voice_clone", "generate_zero_shot", "clone_voice"):
        method = getattr(model, method_name, None)
        if method is not None:
            _require_reference_audio_support(method, method_name)
            return _unwrap_audio(_call_with_supported_kwargs(method, kwargs))
    custom = getattr(model, "generate_custom_voice", None)
    if custom is not None:
        _require_reference_audio_support(custom, "generate_custom_voice")
        return _unwrap_audio(_call_with_supported_kwargs(custom, kwargs))
    raise RuntimeError("QwenTTS model does not expose generate_voice_clone, clone_voice, or generate_custom_voice")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="One-sample job JSON file.")
    args = parser.parse_args()

    job = _load_job(args.job)
    output_path = Path(str(job["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = _synthesize(job)
    sf.write(str(output_path), _post_process(audio), sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()
