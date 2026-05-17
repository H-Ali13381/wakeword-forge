"""
models/dscnn.py — public DS-CNN wakeword backend.

This module implements a small depthwise-separable convolutional keyword
spotting model on normalized log-mel frames. The public runtime boundary is a
single ONNX graph with input ``waveform`` and output ``score``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ..config import HOP_LENGTH, MAX_FRAMES, N_FFT, N_MELS, SAMPLE_RATE


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise temporal convolution followed by pointwise channel mixing."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x + residual


class DSCNN(nn.Module):
    """
    Depthwise-separable CNN classifier for wakeword log-mel frames.

    Input:  ``(B, N_MELS, T)`` normalized log-mel frames.
    Output: ``(B,)`` sigmoid wakeword scores in ``[0, 1]``.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        channels: int = 48,
        n_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.params = {
            "n_mels": n_mels,
            "channels": channels,
            "n_blocks": n_blocks,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[
                DepthwiseSeparableConv1d(
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

    def forward_logits(self, mel: torch.Tensor) -> torch.Tensor:
        """Return raw pre-sigmoid logits for BCE-style training losses."""
        x = self.stem(mel)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(mel))


class LogMelFrontend(nn.Module):
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


class DSCNNDetector(nn.Module):
    """Full waveform → log-mel → DS-CNN → score pipeline."""

    def __init__(self, model: DSCNN, frontend: LogMelFrontend | None = None) -> None:
        super().__init__()
        self.frontend = frontend or LogMelFrontend()
        self.model = model

    def forward_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model.forward_logits(self.frontend(waveform))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(waveform))


def _write_onnx_metadata(path: Path, metadata: dict[str, object]) -> None:
    import onnx

    proto = onnx.load(path)
    del proto.metadata_props[:]
    for key, value in metadata.items():
        prop = proto.metadata_props.add()
        prop.key = str(key)
        prop.value = "" if value is None else str(value)
    onnx.save(proto, path)


def export_dscnn_onnx(
    model: DSCNN,
    output_path: Path,
    wake_phrase: str = "",
    threshold: float = 0.5,
    eer: float | None = None,
) -> Path:
    """Export a DS-CNN detector with stable ``waveform``/``score`` IO names."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    detector = DSCNNDetector(model)
    detector.eval()
    dummy = torch.zeros(1, SAMPLE_RATE * 2)

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
            dynamic_axes={"waveform": {0: "batch", 1: "time"}, "score": {0: "batch"}},
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
        "backend": "dscnn",
        "model_type": "dscnn",
        "model_file": output_path.name,
    }
    _write_onnx_metadata(output_path, metadata)

    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return output_path
