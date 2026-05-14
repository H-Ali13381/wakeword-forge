"""
config.py — shared constants and the user-facing Config dataclass.
Every other module imports from here; nothing else carries magic numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# ── Audio constants (must stay in sync with model export) ────────────────────

SAMPLE_RATE: int = 16_000
MAX_DURATION: float = 3.0          # seconds — clips longer than this are trimmed/padded
MAX_SAMPLES: int = int(MAX_DURATION * SAMPLE_RATE)   # 48 000

# Mel-spectrogram params (used by compact CNN wakeword backends)
N_FFT: int = 400       # 25 ms window
HOP_LENGTH: int = 160  # 10 ms hop
N_MELS: int = 40
MAX_FRAMES: int = int(MAX_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1  # 301

# Training defaults
TARGET_FAR: float = 0.01    # 1 % false-alarm budget for FRR@FAR objective

# Minimum recordings required before training will proceed
MIN_POSITIVES: int = 10
MIN_NEGATIVES: int = 5

# Partial-phrase negatives: clips of only the first word(s) of a multi-word
# wake phrase.  These are the hardest false positives to suppress and must
# be in the training set.
# The synthesizer generates these automatically for multi-word phrases.


# ── User-facing project config ────────────────────────────────────────────────

@dataclass
class ForgeConfig:
    """Persisted per-project configuration."""

    wake_phrase: str = ""
    project_dir: str = ""

    # Recording
    record_positives: int = 20
    record_negatives: int = 10
    record_duration: float = 2.5   # seconds per take

    # Synthesis
    use_tts_augmentation: bool = True
    tts_variants: int = 300        # synthetic positive samples to generate
    tts_engine: str = "kokoro"     # "kokoro" | "piper" | "none"

    # Training
    backend: str = "dscnn"            # public v0.1 backend
    max_epochs: int = 40

    # Privacy
    contribute_samples: bool = False

    # Paths (relative to project_dir)
    samples_dir: str = "samples"
    output_dir: str = "output"
    cache_dir: str = ".cache"

    # Runtime (filled in after training)
    trained_threshold: float = 0.5
    trained_eer: float | None = None

    # Human review checkpoints
    sample_review_approved: bool = False
    generated_review_approved: bool = False
    sample_review_fingerprint: str = ""
    generated_review_fingerprint: str = ""
    quality_check_passed: bool = False
    model_accepted: bool = False
    quality_checked_model_path: str = ""
    quality_checked_model_fingerprint: str = ""
    accepted_model_fingerprint: str = ""
    quality_positive_hits: int = 0
    quality_positive_trials: int = 0
    quality_false_triggers: int = 0
    quality_score_min: float | None = None
    quality_score_max: float | None = None

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "ForgeConfig":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ── Resolved absolute paths ───────────────────────────────────────────────

    @property
    def project_path(self) -> Path:
        return Path(self.project_dir).expanduser().resolve()

    @property
    def samples_path(self) -> Path:
        return self.project_path / self.samples_dir

    @property
    def positives_path(self) -> Path:
        return self.samples_path / "positives"

    @property
    def negatives_path(self) -> Path:
        return self.samples_path / "negatives"

    @property
    def synthetic_path(self) -> Path:
        return self.samples_path / "synthetic"

    @property
    def partials_path(self) -> Path:
        """Synthetic partial-phrase negatives from multi-word wake phrases."""
        return self.samples_path / "partials"

    @property
    def confusables_path(self) -> Path:
        """Synthetic confusable-phrase negatives from an editable phrase cache."""
        return self.samples_path / "confusables"

    @property
    def confusables_cache(self) -> Path:
        """Cached list of confusable phrases (editable plain text)."""
        return self.project_path / "confusable_variants.txt"

    @property
    def output_path(self) -> Path:
        return self.project_path / self.output_dir

    @property
    def cache_path(self) -> Path:
        return self.project_path / self.cache_dir
