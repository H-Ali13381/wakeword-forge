"""
trainer.py — High-level DS-CNN training orchestrator.

The public v0.1 flow intentionally supports one backend: ``dscnn``. Keeping the
training path narrow makes the CLI easier to understand and keeps heavyweight
research backends out of the default open-source package.

Data collected:
  - positives: user recordings + synthetic TTS variants
  - negatives: user recordings + background clips
  - partials:  synthetic partial-phrase clips from multi-word wake phrases
               These are the hardest false positives to eliminate and must
               be in training.  synthesize_partial_negatives() generates
               them automatically for multi-word phrases.
"""

from __future__ import annotations

from pathlib import Path
from rich.console import Console

from .config import ForgeConfig, MIN_POSITIVES, MIN_NEGATIVES
from .augmentation import Augmentor

console = Console()

SUPPORTED_BACKENDS = {"dscnn"}


def validate_backend(backend: str) -> str:
    """Return a supported backend name or raise a user-facing error."""
    if backend not in SUPPORTED_BACKENDS:
        valid = " | ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"Unknown backend: {backend!r}. Valid options: {valid}")
    return backend


def _collect_wavs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.wav"))


def run_training(config: ForgeConfig) -> Path:
    """
    Collect data, train, and export.
    Returns path to the exported ONNX model.
    """
    backend = validate_backend(config.backend)

    pos_files = _collect_wavs(config.positives_path)
    neg_files = _collect_wavs(config.negatives_path)
    synth_files = _collect_wavs(config.synthetic_path)
    partial_files = _collect_wavs(config.partials_path)
    confusable_files = _collect_wavs(config.confusables_path)

    all_pos = pos_files + synth_files
    all_neg = neg_files + confusable_files

    console.print(
        f"\n[bold]Data summary[/bold]\n"
        f"  Real positives:      {len(pos_files)}\n"
        f"  Synthetic positives: {len(synth_files)}\n"
        f"  Negatives:           {len(neg_files)}\n"
        f"  Confusable negs:     {len(confusable_files)}\n"
        f"  Partial negatives:   {len(partial_files)}"
        + ("  [dim](multi-word hard negatives)[/dim]" if partial_files else
           "  [dim](none — single-word phrase or TTS disabled)[/dim]")
        + "\n"
    )

    if len(all_pos) < MIN_POSITIVES:
        raise ValueError(
            f"Not enough positive samples ({len(all_pos)} < {MIN_POSITIVES}). "
            f"Record more examples or enable TTS synthesis."
        )
    if len(all_neg) < MIN_NEGATIVES:
        raise ValueError(
            f"Not enough negative samples ({len(all_neg)} < {MIN_NEGATIVES}). "
            f"Record more counter-examples."
        )

    augmentor = Augmentor(max_chain=4, p=0.6)

    if backend == "dscnn":
        from .models.dscnn_trainer import DSCNNTrainer
        from .review import reset_trained_output_approval, training_data_fingerprint

        trainer = DSCNNTrainer(config)
        trainer.train(all_pos, all_neg, partial_files=partial_files, augmentor=augmentor)
        onnx_path = trainer.export_onnx()
        config.trained_threshold = trainer._threshold
        config.trained_eer = trainer._eer
        config.trained_sample_fingerprint = training_data_fingerprint(config)
        reset_trained_output_approval(config)
    else:  # pragma: no cover - validate_backend guards this branch.
        raise AssertionError(f"Unsupported backend escaped validation: {backend}")

    config.save(Path(config.project_dir) / "forge_config.json")
    return onnx_path
