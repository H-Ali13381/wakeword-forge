"""
models/wavlm_repcnn.py — WavLM teacher -> RepCNN student wakeword backend.

Training flow:
  1. Load positive wake-phrase clips plus background/confusable/partial negatives.
  2. Train a WavLM teacher classifier on waveforms.
  3. Distill that teacher into a compact RepCNN student on log-mel frames.
  4. Export only the RepCNN student as ONNX with stable ``waveform`` -> ``score`` IO.

The WavLM teacher is a training-only artifact; runtime remains a small CNN graph.
"""

from __future__ import annotations

import copy
import gc
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from rich.console import Console
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from ..augmentation import (
    Augmentor,
    SpectrogramAugmentor,
    _load_wav,
    _pad_or_trim,
    augmentation_metadata as build_augmentation_metadata,
)
from ..config import HOP_LENGTH, MAX_FRAMES, MAX_SAMPLES, N_FFT, N_MELS, SAMPLE_RATE, TARGET_FAR, ForgeConfig

console = Console()

DEFAULT_WAVLM_TEACHER = "microsoft/wavlm-base"


class WakewordDataset(Dataset):
    """Waveform dataset with positives, background negatives, and partial negatives."""

    def __init__(
        self,
        pos_files: list[Path],
        neg_files: list[Path],
        partial_files: list[Path] | None = None,
        augmentor: Augmentor | None = None,
        n_aug_variants: int = 2,
    ) -> None:
        self.augmentor = augmentor
        items: list[tuple[Path, float, str | None]] = []
        hard_preset = getattr(augmentor, "preset", "standard") if augmentor else "standard"
        regular_preset = getattr(augmentor, "regular_negative_preset", "none") if augmentor else "none"

        for path in pos_files:
            items.append((Path(path), 1.0, None))
            if augmentor:
                for _ in range(n_aug_variants):
                    items.append((Path(path), 1.0, hard_preset))

        for path in neg_files:
            items.append((Path(path), 0.0, None))
            if augmentor and regular_preset != "none":
                items.append((Path(path), 0.0, regular_preset))

        for path in partial_files or []:
            items.append((Path(path), 0.0, None))
            if augmentor:
                items.append((Path(path), 0.0, hard_preset))

        self.items = items
        self.labels = torch.tensor([label for _, label, _ in items], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        path, label, preset = self.items[idx]
        wav = _load_wav(path, trim_silence=True)
        wav = self.augmentor(wav, preset=preset) if preset and self.augmentor else _pad_or_trim(wav)
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


class MelFrontend(nn.Module):
    """Convert raw mono waveform batches to normalized log-mel frames."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        max_frames: int = MAX_FRAMES,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=float(sample_rate) / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=None,
            mel_scale="htk",
        )
        self.register_buffer("mel_fb", mel_fb, persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if torch.onnx.is_in_onnx_export():
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=False,
            )
            power = spec.pow(2).sum(dim=-1)
        else:
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
            power = spec.abs().pow(2)
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)
        mel = torch.log(mel + 1e-9)
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True) + 1e-9
        mel = (mel - mean) / std

        frames = mel.shape[-1]
        if frames < self.max_frames:
            mel = F.pad(mel, (0, self.max_frames - frames))
        else:
            mel = mel[..., : self.max_frames]
        return mel


class RepConvBlock(nn.Module):
    """Rep-style temporal block with parallel depthwise branches."""

    def __init__(self, channels: int, kernel_size: int = 5, n_branches: int = 2) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                )
                for _ in range(max(1, n_branches))
            ]
        )
        self.skip = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.fused_conv: nn.Conv1d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_conv is not None:
            return self.relu(self.fused_conv(x))
        out = self.skip(x)
        for branch in self.branches:
            out = out + branch(x)
        return self.relu(out)

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
        weight = conv.weight
        bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=weight.device, dtype=weight.dtype)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_weight = weight * scale.reshape(-1, 1, 1)
        fused_bias = bn.bias + (bias - bn.running_mean) * scale
        return fused_weight, fused_bias

    @staticmethod
    def _fuse_identity_bn(
        bn: nn.BatchNorm1d,
        channels: int,
        kernel_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        identity = torch.zeros(channels, 1, kernel_size, device=device, dtype=dtype)
        identity[:, 0, kernel_size // 2] = 1.0
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_weight = identity * scale.reshape(-1, 1, 1)
        fused_bias = bn.bias + (torch.zeros_like(bn.running_mean) - bn.running_mean) * scale
        return fused_weight, fused_bias

    def reparameterize(self) -> "RepConvBlock":
        """Merge RepConv branches + skip BatchNorm into one depthwise Conv1d."""

        if self.fused_conv is not None:
            return self
        if not self.branches:
            raise RuntimeError("Cannot reparameterize RepConvBlock without branches.")

        first_conv = self.branches[0][0]
        if not isinstance(first_conv, nn.Conv1d):  # pragma: no cover - architecture guard.
            raise TypeError("RepConvBlock branch[0] must start with Conv1d.")
        channels = int(first_conv.out_channels)
        kernel_size = int(first_conv.kernel_size[0])
        device = first_conv.weight.device
        dtype = first_conv.weight.dtype

        fused_weight = torch.zeros(channels, 1, kernel_size, device=device, dtype=dtype)
        fused_bias = torch.zeros(channels, device=device, dtype=dtype)
        for branch in self.branches:
            conv, bn = branch[0], branch[1]
            if not isinstance(conv, nn.Conv1d) or not isinstance(bn, nn.BatchNorm1d):  # pragma: no cover
                raise TypeError("RepConvBlock branches must be Conv1d + BatchNorm1d.")
            w, b = self._fuse_conv_bn(conv, bn)
            fused_weight = fused_weight + w
            fused_bias = fused_bias + b

        skip_w, skip_b = self._fuse_identity_bn(self.skip, channels, kernel_size, device=device, dtype=dtype)
        fused_weight = fused_weight + skip_w
        fused_bias = fused_bias + skip_b

        fused = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            fused.weight.copy_(fused_weight)
            fused.bias.copy_(fused_bias)

        self.fused_conv = fused
        self.branches = nn.ModuleList()
        self.skip = nn.Identity()
        return self


class RepCNN(nn.Module):
    """Compact RepCNN wakeword detector. Input: ``(B, n_mels, T)``; output logits ``(B,)``."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        channels: int = 80,
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
        n_branches: int = 4,
        dropout: float = 0.05,
        stem_kernel_size: int = 5,
        stem_stride: int = 2,
    ) -> None:
        super().__init__()
        self.params = {
            "n_mels": n_mels,
            "channels": channels,
            "kernel_sizes": list(kernel_sizes),
            "n_branches": n_branches,
            "dropout": dropout,
            "stem_kernel_size": stem_kernel_size,
            "stem_stride": stem_stride,
        }
        self.stem = nn.Sequential(
            nn.Conv1d(
                n_mels,
                channels,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=stem_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    RepConvBlock(channels, kernel_size=k, n_branches=n_branches),
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                )
                for k in kernel_sizes
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    @property
    def feature_dim(self) -> int:
        return int(self.head.in_features)

    def forward_features(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel)
        for stage in self.stages:
            x = stage(x)
        return self.pool(x).squeeze(-1)

    def forward_logits_and_features(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(mel)
        logits = self.head(self.dropout(features)).squeeze(-1)
        return logits, features

    def forward_logits(self, mel: torch.Tensor) -> torch.Tensor:
        logits, _features = self.forward_logits_and_features(mel)
        return logits

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(mel))

    def reparameterize(self) -> "RepCNN":
        """Fold all RepConv branches into fused depthwise convs for deployment/export."""

        for stage in self.stages:
            block = stage[0]
            if isinstance(block, RepConvBlock):
                block.reparameterize()
        return self


class RepCNNDetector(nn.Module):
    """Full runtime graph: waveform -> log-mel -> RepCNN -> score."""

    def __init__(self, model: RepCNN, frontend: MelFrontend | None = None) -> None:
        super().__init__()
        self.frontend = frontend or MelFrontend()
        self.model = model

    def forward_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model.forward_logits(self.frontend(waveform))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(waveform))


class WavLMTeacher(nn.Module):
    """Training-only WavLM waveform classifier used as the RepCNN distillation teacher."""

    def __init__(
        self,
        model_name: str = DEFAULT_WAVLM_TEACHER,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        unfreeze_layers: int = 1,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import WavLMModel
        except ImportError as exc:  # pragma: no cover - depends on optional environment.
            raise RuntimeError(
                "The WavLM->RepCNN backend requires transformers. Install dependencies with "
                "`pip install -e .` after the updated pyproject, or `pip install transformers`."
            ) from exc

        self.model_name = model_name
        self.backbone = WavLMModel.from_pretrained(model_name)
        self.unfreeze_layers = max(0, int(unfreeze_layers))
        if gradient_checkpointing and self.unfreeze_layers > 0 and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "config"):
                self.backbone.config.use_cache = False
        self.embedding_dim = int(self.backbone.config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._set_trainable_backbone_layers(self.unfreeze_layers)

    def _set_trainable_backbone_layers(self, unfreeze_layers: int) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze_layers <= 0:
            return
        layers = getattr(getattr(self.backbone, "encoder", None), "layers", None)
        if not layers:
            return
        for layer in layers[-int(unfreeze_layers):]:
            for param in layer.parameters():
                param.requires_grad = True

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(waveforms)
        hidden = outputs.last_hidden_state
        mean_emb = hidden.mean(dim=1)
        max_emb = hidden.max(dim=1).values
        return (mean_emb + max_emb) / 2

    def forward_logits(self, waveforms: torch.Tensor) -> torch.Tensor:
        emb = self.encode(waveforms)
        return self.head(emb).squeeze(-1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(waveforms))


class _FeatureProjector(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(student_dim, teacher_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _teacher_epoch_count(config: ForgeConfig) -> int:
    explicit = getattr(config, "wavlm_teacher_epochs", None)
    if explicit is not None:
        return max(1, int(explicit))
    return max(1, min(8, int(config.max_epochs)))


def _teacher_batch_size(config: ForgeConfig, dataset_size: int) -> int:
    explicit = getattr(config, "wavlm_batch_size", None)
    batch_size = int(explicit) if explicit else 1
    return max(1, min(batch_size, dataset_size))


def _teacher_inference_batch_size(config: ForgeConfig, batch_size: int) -> int:
    explicit = getattr(config, "wavlm_inference_batch_size", None)
    if explicit is None:
        explicit = getattr(config, "wavlm_batch_size", 1)
    return max(1, min(int(explicit) if explicit else 1, batch_size))


def _student_batch_size(config: ForgeConfig, dataset_size: int) -> int:
    explicit = getattr(config, "repcnn_batch_size", None)
    batch_size = int(explicit) if explicit else 32
    return max(1, min(batch_size, dataset_size))


class WavLMRepCNNTrainer:
    """Train a WavLM teacher, distill into RepCNN, and export the RepCNN ONNX artifact."""

    def __init__(self, config: ForgeConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold: float = 0.5
        self.eer: float | None = None
        self.teacher_model_name: str = getattr(config, "wavlm_teacher_model", DEFAULT_WAVLM_TEACHER)
        self._best_model: RepCNN | None = None
        self._teacher: WavLMTeacher | None = None

    def _train_teacher(self, train_dl: DataLoader, val_dl: DataLoader) -> WavLMTeacher:
        teacher = WavLMTeacher(
            model_name=self.teacher_model_name,
            unfreeze_layers=int(getattr(self.config, "wavlm_unfrozen_layers", 1)),
            gradient_checkpointing=bool(getattr(self.config, "wavlm_gradient_checkpointing", True)),
        ).to(self.device)
        params = [p for p in teacher.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=float(getattr(self.config, "wavlm_learning_rate", 2e-5)),
            weight_decay=1e-3,
        )

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(_teacher_epoch_count(self.config)):
            teacher.train()
            for wavs, labels in train_dl:
                wavs = wavs.to(self.device)
                labels = labels.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = teacher.forward_logits(wavs)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()

            teacher.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for wavs, labels in val_dl:
                    wavs = wavs.to(self.device)
                    labels = labels.to(self.device)
                    logits = teacher.forward_logits(wavs)
                    val_losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).cpu()))
            mean_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            console.print(f"  WavLM teacher epoch {epoch + 1}: val_loss={mean_loss:.4f}")
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if best_state is not None:
            teacher.load_state_dict(best_state)
        teacher.eval()
        return teacher

    def _teacher_forward_chunks(self, teacher: WavLMTeacher, wavs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run WavLM teacher inference in RAM-safe chunks during RepCNN distillation."""

        chunk_size = _teacher_inference_batch_size(self.config, int(wavs.shape[0]))
        embs: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for chunk in wavs.split(chunk_size):
            emb = teacher.encode(chunk)
            logit = teacher.head(emb).squeeze(-1)
            embs.append(emb)
            logits.append(logit)
        return torch.cat(embs, dim=0), torch.cat(logits, dim=0)

    def _train_student(
        self,
        teacher: WavLMTeacher,
        train_dl: DataLoader,
        val_dl: DataLoader,
        spec_augmentor: SpectrogramAugmentor | None = None,
    ) -> tuple[RepCNN, float | None, float, float]:
        detector = RepCNNDetector(RepCNN()).to(self.device)
        projector = _FeatureProjector(detector.model.feature_dim, teacher.embedding_dim).to(self.device)
        opt = torch.optim.AdamW(
            list(detector.model.parameters()) + list(projector.parameters()),
            lr=float(getattr(self.config, "repcnn_learning_rate", 1e-3)),
            weight_decay=1e-3,
        )
        temperature = float(getattr(self.config, "distill_temperature", 2.0))
        logit_weight = float(getattr(self.config, "distill_logit_weight", 0.5))
        feature_weight = float(getattr(self.config, "distill_feature_weight", 0.25))
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        best_score = -float("inf")
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: tuple[float | None, float] = (None, 0.5)

        for epoch in range(max(1, int(self.config.max_epochs))):
            detector.train()
            projector.train()
            for wavs, labels in train_dl:
                wavs = wavs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    teacher_embs, teacher_logits = self._teacher_forward_chunks(teacher, wavs)

                mels = detector.frontend(wavs)
                if spec_augmentor:
                    mels = spec_augmentor(mels)
                student_logits, student_features = detector.model.forward_logits_and_features(mels)

                hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
                teacher_probs = torch.sigmoid(teacher_logits / temperature)
                student_probs = torch.sigmoid(student_logits / temperature)
                eps = 1e-8
                kl = (
                    teacher_probs * (torch.log(teacher_probs + eps) - torch.log(student_probs + eps))
                    + (1 - teacher_probs)
                    * (torch.log(1 - teacher_probs + eps) - torch.log(1 - student_probs + eps))
                ).mean() * (temperature**2)
                projected_features = projector(student_features)
                cosine = 1 - F.cosine_similarity(projected_features, teacher_embs, dim=1).mean()
                loss = hard_loss + logit_weight * kl + feature_weight * cosine

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(detector.model.parameters()) + list(projector.parameters()), max_norm=1.0)
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
                    scores.extend(torch.sigmoid(logits).cpu().tolist())
                    labels_np.extend(labels.cpu().tolist())

            mean_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            eer, threshold, combined = _validation_metrics(np.array(labels_np), np.array(scores))
            score = combined if combined > 0 else -mean_loss
            console.print(
                f"  RepCNN student epoch {epoch + 1}: val_loss={mean_loss:.4f} "
                f"eer={eer if eer is not None else 'n/a'} threshold={threshold:.4f}"
            )
            if score > best_score or (score == best_score and mean_loss < best_loss):
                best_score = score
                best_loss = mean_loss
                best_metrics = (eer, threshold)
                best_state = {k: v.detach().cpu().clone() for k, v in detector.model.state_dict().items()}
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if best_state is None:
            raise RuntimeError("Training did not produce a RepCNN checkpoint.")
        model = RepCNN()
        model.load_state_dict(best_state)
        model.eval()
        eer, threshold = best_metrics
        return model, eer, threshold, best_loss

    def train(
        self,
        pos_files: list[Path],
        neg_files: list[Path],
        partial_files: list[Path] | None = None,
        augmentor: Augmentor | None = None,
        spec_augmentor: SpectrogramAugmentor | None = None,
    ) -> dict[str, Any]:
        console.print(
            f"\n[bold cyan]Training WavLM → RepCNN[/bold cyan] on [green]{self.device}[/green]\n"
            f"  Positives: {len(pos_files)}  "
            f"Negatives: {len(neg_files)}  "
            f"Partial: {len(partial_files or [])}\n"
            f"  Teacher: {self.teacher_model_name}"
        )

        dataset = WakewordDataset(pos_files, neg_files, partial_files, augmentor)
        train_idx, val_idx = _stratified_split(dataset.labels)
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_labels = dataset.labels[train_idx]
        sampler = _build_sampler(train_labels)

        teacher_bs = _teacher_batch_size(self.config, len(train_ds))
        student_bs = _student_batch_size(self.config, len(train_ds))
        train_dl_teacher = DataLoader(train_ds, batch_size=teacher_bs, sampler=sampler, collate_fn=_collate)
        train_dl_student = DataLoader(train_ds, batch_size=student_bs, sampler=sampler, collate_fn=_collate)
        val_dl_teacher = DataLoader(val_ds, batch_size=teacher_bs, shuffle=False, collate_fn=_collate)
        val_dl_student = DataLoader(val_ds, batch_size=student_bs, shuffle=False, collate_fn=_collate)

        teacher = self._train_teacher(train_dl_teacher, val_dl_teacher)
        model, eer, threshold, best_loss = self._train_student(
            teacher,
            train_dl_student,
            val_dl_student,
            spec_augmentor=spec_augmentor,
        )

        self._teacher = teacher.cpu().eval()
        self._best_model = model.cpu().eval()
        self.eer = eer
        self.threshold = threshold

        ckpt_path = self.config.output_path / "repcnn_student.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._best_model.state_dict(),
                "params": copy.deepcopy(self._best_model.params),
                "backend": "wavlm-repcnn",
                "teacher_model": self.teacher_model_name,
                "threshold": self.threshold,
                "eer": self.eer,
            },
            ckpt_path,
        )

        return {
            "backend": "wavlm-repcnn",
            "teacher_model": self.teacher_model_name,
            "threshold": self.threshold,
            "eer": self.eer,
            "val_loss": best_loss,
            "checkpoint": str(ckpt_path),
        }

    def export_onnx(self) -> Path:
        if self._best_model is None:
            raise RuntimeError("Train before exporting RepCNN to ONNX.")
        return export_repcnn_onnx(
            self._best_model,
            self.config.output_path / "wakeword.onnx",
            wake_phrase=self.config.wake_phrase,
            threshold=self.threshold,
            eer=self.eer,
            teacher_model=self.teacher_model_name,
            augmentation_metadata=build_augmentation_metadata(self.config),
        )


def _metadata_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, bool, int, float)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _set_value_info_shape(value_info: Any, dims: tuple[int, ...]) -> None:
    shape = value_info.type.tensor_type.shape
    del shape.dim[:]
    for dim in dims:
        shape.dim.add().dim_value = int(dim)


def _write_onnx_metadata(path: Path, metadata: dict[str, object]) -> None:
    import onnx

    proto = onnx.load(path)
    if proto.graph.input:
        _set_value_info_shape(proto.graph.input[0], (1, MAX_SAMPLES))
    if proto.graph.output:
        _set_value_info_shape(proto.graph.output[0], (1,))
    del proto.metadata_props[:]
    for key, value in metadata.items():
        prop = proto.metadata_props.add()
        prop.key = str(key)
        prop.value = _metadata_value(value)
    onnx.save(proto, path)


def export_repcnn_onnx(
    model: RepCNN,
    output_path: Path,
    wake_phrase: str = "",
    threshold: float = 0.5,
    eer: float | None = None,
    teacher_model: str = DEFAULT_WAVLM_TEACHER,
    augmentation_metadata: dict[str, object] | None = None,
) -> Path:
    """Export a distilled RepCNN detector with stable ``waveform``/``score`` IO names."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    export_model = copy.deepcopy(model).eval()
    export_model.reparameterize()
    detector = RepCNNDetector(export_model)
    detector.eval()
    dummy = torch.zeros(1, MAX_SAMPLES)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="stft with return_complex=False is deprecated.*")
        warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean.*")
        warnings.filterwarnings("ignore", message="Constant folding - Only steps=1.*")
        torch.onnx.export(
            detector,
            dummy,
            str(output_path),
            input_names=["waveform"],
            output_names=["score"],
            opset_version=17,
            dynamo=False,
        )

    metadata = {
        "wake_phrase": wake_phrase,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "max_frames": MAX_FRAMES,
        "threshold": threshold,
        "eer": eer,
        "backend": "wavlm-repcnn",
        "model_type": "repcnn",
        "reparameterized": True,
        "repconv_merged": True,
        "teacher_model_type": "wavlm",
        "teacher_model": teacher_model,
        "model_file": output_path.name,
    }
    if augmentation_metadata is not None:
        metadata["augmentation"] = augmentation_metadata
    _write_onnx_metadata(output_path, metadata)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return output_path
