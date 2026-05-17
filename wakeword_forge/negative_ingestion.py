"""Manifest-driven ingestion for external negative audio.

The importer keeps external negative datasets balanced and auditable:

- manifest or source-directory inputs
- per-source and per-file caps so one corpus/file cannot dominate
- 16 kHz mono PCM WAV output
- fixed-duration chunking
- transcript exclusion checks when manifests provide transcripts
- per-sample and aggregate provenance records
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from .config import MAX_DURATION, SAMPLE_RATE
from .project import SUPPORTED_AUDIO_EXTENSIONS
from .config import ForgeConfig

NEGATIVE_IMPORT_MANIFEST = "negative_import_manifest.jsonl"
_NEGATIVE_LABELS = {"", "neg", "negative", "0", 0, False}
_TARGET_DIR_BY_KIND = {
    "background": "negatives_path",
    "negative": "negatives_path",
    "negatives": "negatives_path",
    "partial": "partials_path",
    "partials": "partials_path",
    "confusable": "confusables_path",
    "confusables": "confusables_path",
}


@dataclass(frozen=True)
class NegativeImportResult:
    """Summary of negative audio imported into a forge project."""

    source: Path
    target_dir: Path
    manifest_path: Path
    imported_paths: tuple[Path, ...]
    available_count: int
    skipped_paths: tuple[Path, ...] = ()
    strata_counts: dict[str, int] = field(default_factory=dict)
    strata_limits: dict[str, int] = field(default_factory=dict)

    @property
    def imported_count(self) -> int:
        return len(self.imported_paths)


def _target_dir(config: ForgeConfig, kind: str) -> Path:
    attr = _TARGET_DIR_BY_KIND.get(kind.strip().lower())
    if attr is None:
        valid = ", ".join(sorted(_TARGET_DIR_BY_KIND))
        raise ValueError(f"Unknown negative import kind: {kind!r}. Valid options: {valid}")
    return getattr(config, attr)


def _audio_files(source_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )
    )


def _records_from_source_dir(source_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "label": "neg",
            "source_dataset": source_dir.name,
            "category": "directory_import",
            "license": "unknown",
            "notes": "Imported from a local source directory.",
        }
        for path in _audio_files(source_dir)
    ]


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _records_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {manifest_path}:{line_number}: {exc}") from exc
            raw_path = record.get("path")
            if not raw_path:
                raise ValueError(f"Manifest row missing path at {manifest_path}:{line_number}")
            enriched = dict(record)
            enriched["_resolved_path"] = str(_resolve_manifest_path(str(raw_path), manifest_path))
            enriched["_source_manifest"] = str(manifest_path)
            records.append(enriched)
    return records


def _source_id(record: dict[str, Any], audio_path: Path) -> str:
    return str(
        record.get("source_dataset")
        or record.get("dataset_id")
        or record.get("display_name")
        or record.get("source")
        or audio_path.parent.name
        or "unknown"
    )


def _normalize_stratum(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _normalize_strata(strata: Mapping[str, int] | None) -> dict[str, int] | None:
    if strata is None:
        return None
    normalized: dict[str, int] = {}
    for key, raw_limit in strata.items():
        name = _normalize_stratum(key)
        limit = max(0, int(raw_limit))
        normalized[name] = normalized.get(name, 0) + limit
    return normalized


def parse_strata_quotas(raw: str | None) -> dict[str, int] | None:
    """Parse CLI strata quotas without eval.

    Accepts either JSON objects (``{"speech": 50}``) or compact comma/
    semicolon separated pairs (``speech=50,noise=50,silence=20``).
    """

    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    if text.startswith("{"):
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError("Strata quotas JSON must be an object.")
        items = loaded.items()
    else:
        parsed_items: list[tuple[str, str]] = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            separator = "=" if "=" in part else ":" if ":" in part else ""
            if not separator:
                raise ValueError(
                    "Strata quotas must use name=count pairs, e.g. speech=50,noise=50,silence=20."
                )
            name, value = part.split(separator, 1)
            parsed_items.append((name, value))
        items = parsed_items

    quotas: dict[str, int] = {}
    for key, raw_limit in items:
        raw_name = str(key).strip()
        if not raw_name:
            raise ValueError("Stratum names cannot be blank.")
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid quota for stratum {raw_name!r}: {raw_limit!r}") from exc
        if limit < 0:
            raise ValueError(f"Quota for stratum {raw_name!r} must be non-negative.")
        name = _normalize_stratum(raw_name)
        quotas[name] = quotas.get(name, 0) + limit
    return quotas or None


def _record_stratum(record: dict[str, Any], stratify_by: str) -> str:
    return _normalize_stratum(record.get(stratify_by))


def _is_negative_label(record: dict[str, Any]) -> bool:
    label = record.get("label", "neg")
    if isinstance(label, str):
        return label.strip().lower() in _NEGATIVE_LABELS
    return label in _NEGATIVE_LABELS


def _transcript_text(record: dict[str, Any]) -> str:
    parts = []
    for key in ("transcript", "text", "sentence", "normalized_text"):
        value = record.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts).lower()


def _has_transcript_contamination(record: dict[str, Any]) -> bool:
    terms = record.get("transcript_exclusion_terms") or []
    if not terms:
        return False
    text = _transcript_text(record)
    if not text:
        return False
    return any(str(term).strip().lower() in text for term in terms if str(term).strip())


def _resample_linear(data: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return data.astype(np.float32, copy=False)
    duration = len(data) / src_sr
    n_out = max(1, int(round(duration * target_sr)))
    old_x = np.linspace(0.0, duration, num=len(data), endpoint=False)
    new_x = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(new_x, old_x, data).astype(np.float32)


def _resample(data: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return data.astype(np.float32, copy=False)
    try:
        import soxr  # type: ignore
    except ImportError:
        return _resample_linear(data, src_sr, target_sr)
    return soxr.resample(data, src_sr, target_sr).astype(np.float32)


def _load_audio_segment(audio_path: Path, record: dict[str, Any]) -> np.ndarray:
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        mono = data.mean(axis=1)
    else:
        mono = data[:, 0]

    start_sec = record.get("start_sec")
    end_sec = record.get("end_sec")
    if start_sec is not None or end_sec is not None:
        start = max(0, int(float(start_sec or 0.0) * sr))
        end = len(mono) if end_sec is None else max(start, int(float(end_sec) * sr))
        mono = mono[start:end]

    return _resample(mono, sr, SAMPLE_RATE)


def _iter_chunks(
    audio: np.ndarray,
    *,
    chunk_duration: float,
    min_chunk_duration: float,
    max_chunks: int,
) -> Iterable[np.ndarray]:
    chunk_samples = max(1, int(round(chunk_duration * SAMPLE_RATE)))
    min_samples = max(1, int(round(min_chunk_duration * SAMPLE_RATE)))
    offset = 0
    emitted = 0
    while offset < len(audio) and emitted < max_chunks:
        segment = audio[offset : offset + chunk_samples]
        if len(segment) < min_samples:
            break
        if len(segment) < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - len(segment)))
        yield np.clip(segment.astype(np.float32, copy=False), -1.0, 1.0)
        emitted += 1
        offset += chunk_samples


def _next_import_path(target_dir: Path, prefix: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    highest = -1
    stem_prefix = f"{prefix}_"
    for path in target_dir.glob(f"{stem_prefix}*.wav"):
        suffix = path.stem.removeprefix(stem_prefix)
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return target_dir / f"{prefix}_{highest + 1:04d}.wav"


def _metadata_for_import(
    *,
    source_record: dict[str, Any],
    audio_path: Path,
    output_path: Path,
    target_kind: str,
    chunk_index: int,
    source_manifest: Path | None,
) -> dict[str, Any]:
    passthrough_keys = (
        "source_dataset",
        "display_name",
        "category",
        "license",
        "languages",
        "source_url",
        "notes",
        "recommended_use_cases",
        "contamination_assumption",
        "recommended_scale",
        "start_sec",
        "end_sec",
        "duration_sec",
    )
    metadata: dict[str, Any] = {
        "label": "negative",
        "target_kind": target_kind,
        "imported_path": str(output_path),
        "source_path": str(audio_path),
        "source_manifest": str(source_manifest) if source_manifest is not None else None,
        "source_dataset": _source_id(source_record, audio_path),
        "sample_rate": SAMPLE_RATE,
        "chunk_index": chunk_index,
    }
    for key in passthrough_keys:
        if key in source_record:
            metadata[key] = source_record[key]
    return metadata


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def import_negative_audio(
    config: ForgeConfig,
    *,
    source_dir: Path | str | None = None,
    manifest: Path | str | None = None,
    kind: str = "background",
    limit: int | None = None,
    limit_per_source: int | None = 100,
    max_chunks_per_file: int = 20,
    chunk_duration: float = MAX_DURATION,
    min_chunk_duration: float = 0.5,
    prefix: str = "external_neg",
    strata: Mapping[str, int] | None = None,
    stratify_by: str = "category",
) -> NegativeImportResult:
    """Import external negative audio into a project with provenance.

    Exactly one of ``source_dir`` or ``manifest`` must be provided. Manifest rows
    should include ``path``, ``label``, ``source_dataset``, ``category``,
    ``license``, optional chunk offsets, and optional transcript exclusion fields.
    """

    if (source_dir is None) == (manifest is None):
        raise ValueError("Pass exactly one of source_dir or manifest.")
    target_dir = _target_dir(config, kind)
    aggregate_manifest = target_dir / NEGATIVE_IMPORT_MANIFEST

    source_manifest: Path | None = None
    if manifest is not None:
        source_manifest = Path(manifest).expanduser()
        if not source_manifest.is_file():
            raise FileNotFoundError(f"Negative manifest is not available: {source_manifest}")
        records = _records_from_manifest(source_manifest)
        source = source_manifest
    else:
        source = Path(source_dir).expanduser()  # type: ignore[arg-type]
        if not source.is_dir():
            raise FileNotFoundError(f"Negative source folder is not available: {source}")
        records = _records_from_source_dir(source)

    imported: list[Path] = []
    skipped: list[Path] = []
    aggregate_rows: list[dict[str, Any]] = []
    chunks_per_source: Counter[str] = Counter()
    chunks_per_stratum: Counter[str] = Counter()
    seen_record_keys: set[tuple[str, Any, Any]] = set()
    total_limit = None if limit is None else max(0, int(limit))
    source_limit = None if limit_per_source is None else max(0, int(limit_per_source))
    per_file_limit = max(1, int(max_chunks_per_file))
    strata_limits = _normalize_strata(strata)
    stratify_field = stratify_by.strip() or "category"

    for record in records:
        if total_limit is not None and len(imported) >= total_limit:
            break
        raw_path = record.get("_resolved_path") or record.get("path")
        audio_path = Path(str(raw_path)).expanduser()
        record_key = (str(audio_path.resolve(strict=False)), record.get("start_sec"), record.get("end_sec"))
        if record_key in seen_record_keys:
            continue
        seen_record_keys.add(record_key)

        source_id = _source_id(record, audio_path)
        stratum = _record_stratum(record, stratify_field)
        if strata_limits is not None:
            stratum_limit = strata_limits.get(stratum)
            if stratum_limit is None or chunks_per_stratum[stratum] >= stratum_limit:
                continue
        if source_limit is not None and chunks_per_source[source_id] >= source_limit:
            continue
        if not _is_negative_label(record) or _has_transcript_contamination(record):
            skipped.append(audio_path)
            continue
        if not audio_path.is_file():
            skipped.append(audio_path)
            continue

        try:
            audio = _load_audio_segment(audio_path, record)
        except Exception:
            skipped.append(audio_path)
            continue

        file_remaining = per_file_limit
        for chunk_index, chunk in enumerate(
            _iter_chunks(
                audio,
                chunk_duration=chunk_duration,
                min_chunk_duration=min_chunk_duration,
                max_chunks=file_remaining,
            )
        ):
            if total_limit is not None and len(imported) >= total_limit:
                break
            if strata_limits is not None and chunks_per_stratum[stratum] >= strata_limits[stratum]:
                break
            if source_limit is not None and chunks_per_source[source_id] >= source_limit:
                break
            out_path = _next_import_path(target_dir, prefix)
            sf.write(str(out_path), chunk, SAMPLE_RATE, subtype="PCM_16")
            metadata = _metadata_for_import(
                source_record=record,
                audio_path=audio_path,
                output_path=out_path,
                target_kind=kind,
                chunk_index=chunk_index,
                source_manifest=source_manifest,
            )
            if strata_limits is not None:
                metadata["stratify_by"] = stratify_field
                metadata["stratum"] = stratum
                metadata["stratum_quota"] = strata_limits[stratum]
            out_path.with_suffix(out_path.suffix + ".json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            aggregate_rows.append(metadata)
            imported.append(out_path)
            chunks_per_source[source_id] += 1
            chunks_per_stratum[stratum] += 1

    _append_jsonl(aggregate_manifest, aggregate_rows)

    if imported:
        from .review import reset_sample_dependent_approvals

        reset_sample_dependent_approvals(config)

    return NegativeImportResult(
        source=source,
        target_dir=target_dir,
        manifest_path=aggregate_manifest,
        imported_paths=tuple(imported),
        available_count=len(records),
        skipped_paths=tuple(dict.fromkeys(skipped)),
        strata_counts=dict(chunks_per_stratum),
        strata_limits=strata_limits or {},
    )


__all__ = ["NEGATIVE_IMPORT_MANIFEST", "NegativeImportResult", "import_negative_audio", "parse_strata_quotas"]
