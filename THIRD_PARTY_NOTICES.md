# Third-party notices

This file summarizes third-party software, model, dataset, and service dependencies used or referenced by wakeword-forge. It is a release checklist, not a substitute for legal review.

## Python packages

Runtime dependencies declared in `pyproject.toml`:

- PyTorch / TorchAudio: tensor and audio processing. Upstream: https://pytorch.org/
- sounddevice: microphone capture. Upstream: https://python-sounddevice.readthedocs.io/
- SoundFile / libsndfile: audio file IO. Upstream: https://python-soundfile.readthedocs.io/
- NumPy: numerical arrays. Upstream: https://numpy.org/
- ONNX: model export format. Upstream: https://onnx.ai/
- ONNX Runtime: model inference runtime. Upstream: https://onnxruntime.ai/
- Rich: terminal UI. Upstream: https://github.com/Textualize/rich
- Typer: CLI. Upstream: https://typer.tiangolo.com/
- Streamlit: optional dashboard UI. Upstream: https://streamlit.io/
- scikit-learn: metrics. Upstream: https://scikit-learn.org/
- Kokoro ONNX: optional TTS augmentation backend. Upstream: https://github.com/thewh1teagle/kokoro-onnx

For any release that vendors dependencies, publishes prebuilt model artifacts, or redistributes generated datasets, verify exact package versions and license metadata from the artifacts actually shipped or installed.

## Model backends

- DS-CNN backend: implemented in this repository for compact local wake-word detection.
- Kokoro TTS assets: optional generated-speech backend. Verify code, model, voice assets, and generated-output terms separately.

## Optional datasets

- Synthetic silence/noise: generated locally by wakeword-forge.
- Common Voice: optional streaming source for speech negatives when enabled. Verify the exact corpus version, language split, and license before publishing derived artifacts.
- ESC-50: optional environmental-sound source when enabled. ESC-50 is CC BY-NC 3.0; keep it disabled by default for commercial-safe workflows and do not publish models trained on it as commercial-safe without replacement data or legal review.

## User-generated artifacts

User recordings, generated audio, exported ONNX models, and `config.json` files belong to the user who created them, subject to third-party package/model/dataset/TTS terms and the rights of any recorded speakers.
