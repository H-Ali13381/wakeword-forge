"""
models/dscnn_trainer.py — straightforward trainer for the public DS-CNN backend.

The trainer keeps the user-facing flow simple: load waveform files, balance the
binary classes, train a compact DS-CNN detector, pick a validation threshold,
and export an ONNX runtime artifact.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from ..augmentation import Augmentor, _load_wav, _pad_or_trim
from ..config import MAX_SAMPLES, TARGET_FAR, ForgeConfig
from .dscnn import DSCNN, DSCNNDetector, export_dscnn_onnx

console = Console()


class DSCNNDataset(Dataset):
    """Waveform dataset with positives, negatives, and partial-phrase negatives."""

    def __init__(
        self,
        pos_files: list[Path],
        neg_files: list[Path],
        partial_files: list[Path] | None = None,
        augmentor: Augmentor | None = None,
        n_aug_variants: int = 2,
    ) -> None:
        self.augmentor = augmentor
        items: list[tuple[Path, float, bool]] = []

        for path in pos_files:
            items.append((Path(path), 1.0, False))
            if augmentor:
                for _ in range(n_aug_variants):
                    items.append((Path(path), 1.0, True))

        for path in neg_files:
            items.append((Path(path), 0.0, False))

        for path in partial_files or []:
            items.append((Path(path), 0.0, False))
            if augmentor:
                items.append((Path(path), 0.0, True))

        self.items = items
        self.labels = torch.tensor([label for _, label, _ in items], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        path, label, use_aug = self.items[idx]
        wav = _load_wav(path, trim_silence=True)
        wav = self.augmentor(wav) if use_aug and self.augmentor else _pad_or_trim(wav)
        return wav.squeeze(0), float(label)


def _collate(batch: list[tuple[torch.Tensor, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    wavs, labels = zip(*batch)
    max_len = min(max(w.shape[-1] for w in wavs), MAX_SAMPLES)
    padded = []
    for wav in wavs:
        wav = wav[:max_len]
        padded.append(F.pad(wav, (0, max_len - wav.shape[-1])))
    return torch.stack(padded), torch.tensor(labels, dtype=torch.float32)


def _build_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes. Got {n_pos} pos, {n_neg} neg.")
    pos_weight = n_neg / n_pos
    weights = torch.where(labels == 1, torch.full_like(labels, pos_weight), torch.ones_like(labels))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _stratified_split(labels: torch.Tensor, val_fraction: float = 0.15) -> tuple[list[int], list[int]]:
    """Create deterministic train/validation indices while preserving both classes."""
    pos = torch.nonzero(labels == 1, as_tuple=False).flatten().tolist()
    neg = torch.nonzero(labels == 0, as_tuple=False).flatten().tolist()
    if not pos or not neg:
        raise ValueError("Need both positive and negative examples for training.")

    generator = torch.Generator().manual_seed(42)
    pos = torch.tensor(pos)[torch.randperm(len(pos), generator=generator)].tolist()
    neg = torch.tensor(neg)[torch.randperm(len(neg), generator=generator)].tolist()

    n_pos_val = min(max(1, int(round(len(pos) * val_fraction))), len(pos) - 1) if len(pos) > 1 else 1
    n_neg_val = min(max(1, int(round(len(neg) * val_fraction))), len(neg) - 1) if len(neg) > 1 else 1

    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]
    if not train:
        train = val[:]
    return train, val


def _validation_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, float, float]:
    """Return (eer, threshold, combined_score) for binary validation scores."""
    if len(np.unique(labels)) < 2:
        return None, 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer_idx = int(np.argmin(np.abs(fpr - (1 - tpr))))
    eer = float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2)
    threshold = float(thresholds[eer_idx])

    eligible = np.where(fpr <= TARGET_FAR)[0]
    frr_at_far = float(1 - tpr[eligible[-1]]) if len(eligible) else 1.0
    combined = (1 - eer) * 0.5 + (1 - frr_at_far) * 0.5
    return eer, threshold, combined


class DSCNNTrainer:
    """Train and export the public DS-CNN backend."""

    def __init__(self, config: ForgeConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._threshold: float = 0.5
        self._eer: float | None = None
        self._best_model: DSCNN | None = None

    def train(
        self,
        pos_files: list[Path],
        neg_files: list[Path],
        partial_files: list[Path] | None = None,
        augmentor: Augmentor | None = None,
    ) -> dict:
        console.print(
            f"\n[bold cyan]Training DS-CNN[/bold cyan] on [green]{self.device}[/green]\n"
            f"  Positives: {len(pos_files)}  "
            f"Negatives: {len(neg_files)}  "
            f"Partial: {len(partial_files or [])}"
        )

        dataset = DSCNNDataset(pos_files, neg_files, partial_files, augmentor)
        train_idx, val_idx = _stratified_split(dataset.labels)
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_labels = dataset.labels[train_idx]
        sampler = _build_sampler(train_labels)

        batch_size = max(1, min(32, len(train_ds)))
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=_collate)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

        detector = DSCNNDetector(DSCNN()).to(self.device)
        opt = torch.optim.AdamW(detector.model.parameters(), lr=1e-3, weight_decay=1e-3)

        best_score = -float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        best_metrics: tuple[float | None, float] = (None, 0.5)

        for _epoch in range(max(1, self.config.max_epochs)):
            detector.train()
            for wavs, labels in train_dl:
                wavs = wavs.to(self.device)
                labels = labels.to(self.device)
                opt.zero_grad()
                logits = detector.forward_logits(wavs)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(detector.model.parameters(), max_norm=1.0)
                opt.step()

            detector.eval()
            val_losses: list[float] = []
            scores: list[float] = []
            labels_np: list[float] = []
            with torch.no_grad():
                for wavs, labels in val_dl:
                    wavs = wavs.to(self.device)
                    labels = labels.to(self.device)
                    logits = detector.forward_logits(wavs)
                    val_losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).cpu()))
                    batch_scores = torch.sigmoid(logits)
                    scores.extend(batch_scores.cpu().tolist())
                    labels_np.extend(labels.cpu().tolist())

            mean_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            eer, threshold, combined = _validation_metrics(np.array(labels_np), np.array(scores))
            score = combined if combined > 0 else -mean_loss
            if score > best_score or (score == best_score and mean_loss < best_loss):
                best_score = score
                best_loss = mean_loss
                best_metrics = (eer, threshold)
                best_state = copy.deepcopy(detector.model.state_dict())

        if best_state is None:
            raise RuntimeError("Training did not produce a DS-CNN checkpoint.")

        best_model = DSCNN()
        best_model.load_state_dict(best_state)
        self._best_model = best_model.cpu().eval()
        self._eer, self._threshold = best_metrics
        return {
            "backend": "dscnn",
            "threshold": self._threshold,
            "eer": self._eer,
            "val_loss": best_loss,
        }

    def export_onnx(self) -> Path:
        if self._best_model is None:
            raise RuntimeError("Train before exporting DS-CNN to ONNX.")
        return export_dscnn_onnx(
            self._best_model,
            self.config.output_path / "wakeword.onnx",
            wake_phrase=self.config.wake_phrase,
            threshold=self._threshold,
            eer=self._eer,
        )
