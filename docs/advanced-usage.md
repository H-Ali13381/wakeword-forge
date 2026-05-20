# Advanced usage

Detailed commands and data paths for `wakeword-forge`.

Start with the short path in [README.md](../README.md). Use this file when you need the full command set, negative-data import options, synthesis backends, or output details.

## Command reference

Run commands from the repository root. Make targets install into `.venv` and call `.venv/bin/wakeword-forge` for you.

| Task | Command |
| --- | --- |
| Open the dashboard | `make start DIR=~/wakeword-forge-demo` |
| Run the terminal wizard | `make cli-run DIR=~/wakeword-forge-demo` |
| Show project status | `make info DIR=~/wakeword-forge-demo` |
| Record positives | `make record DIR=~/wakeword-forge-demo PHRASE='Hey Nova' N=20` |
| Generate TTS positives | `make synth DIR=~/wakeword-forge-demo PHRASE='Hey Nova' N=300` |
| Import background negatives | `make import-negatives DIR=~/wakeword-forge-demo NEG_SOURCE_DIR=~/clips NEG_LIMIT=150` |
| Review samples | `make review DIR=~/wakeword-forge-demo` |
| Audit generated clips | `make audit DIR=~/wakeword-forge-demo` |
| Train and export ONNX | `make train DIR=~/wakeword-forge-demo` |
| Run guided live check | `make quality-check DIR=~/wakeword-forge-demo` |
| Accept checked model | `make accept-model DIR=~/wakeword-forge-demo` |
| Test the accepted model on mic input | `make mic-test DIR=~/wakeword-forge-demo` |

Run `make help` for the full target list.

## Direct CLI usage

If you need the Typer CLI directly, install once and call the repo-local binary:

```bash
make install
.venv/bin/wakeword-forge --help
.venv/bin/wakeword-forge train --dir ~/wakeword-forge-demo
```

## Optional tools

Optional workflows may need extra tools:

- a working microphone for recording samples or live checks
- Docker for the QwenTTS voice-clone runner
- an NVIDIA/CUDA setup for GPU-heavy QwenTTS workflows
- compatible audio files if you plan to import existing positives or negatives

## Negative data

A wake-word detector needs more than positive examples. Training fails early if there is not enough negative coverage.

Current guardrails require:

- at least 10 positive clips
- at least 5 negative clips
- at least 150 general/background negatives in `samples/negatives/`
- for multi-word phrases with TTS enabled, at least 100 partial-phrase hard negatives in `samples/partials/`

For external data, prefer manifest imports so provenance survives review:

```bash
make import-negatives DIR=~/wakeword-forge-demo \
  NEG_MANIFEST=~/external_negatives.jsonl \
  NEG_LIMIT=150 \
  NEG_LIMIT_PER_SOURCE=25 \
  NEG_STRATA='speech=50,noise=50,silence=50'
```

Manifest rows can include `path`, `label`, `source_dataset`, `category`, `license`, `start_sec`, `end_sec`, and transcript fields. Imported clips get sidecar metadata and an aggregate `negative_import_manifest.jsonl`.

The Make variables above map to the CLI's `--limit-per-source` and `--strata` options.

## Synthesis options

QwenTTS is the recommended generator for synthetic data. It produces more speaker/style variety for wakeword positives and hard negatives than the lighter CPU fallbacks.

```bash
make synth DIR=~/wakeword-forge-demo PHRASE='Hey Nova' N=300 ENGINE=qwentts
```

QwenTTS baseline synthesis does not clone a speaker. It uses built-in voice designs and does not require external reference audio.

`make synth` defaults to `ENGINE=qwentts` and also supports:

- `ENGINE=qwentts`: recommended Qwen3-TTS CustomVoice baseline synthesis
- `ENGINE=kokoro`: lightweight local CPU fallback
- `ENGINE=piper`: Piper, if a voice is installed

## Optional QwenTTS voice-clone staging

There is also an optional Dockerized QwenTTS voice-clone flow. The responsible-use rule is simple: use it only with consent, compatible licenses, or a defensible fair use basis.

It generates one candidate at a time and stages it for human review:

```bash
make qwentts-build
make qwentts-voice-clone-one DIR=~/wakeword-forge-demo SOURCE_MANIFEST=~/wakeword-forge-demo/voice_clone_sources.jsonl
make review-cloned-samples DIR=~/wakeword-forge-demo
```

## Training and output

The default backend trains a WavLM teacher, distills it into a compact RepCNN student, and exports only the runtime graph.

It uses training-time acoustic augmentation by default. Reviewed clips can be varied during training without writing extra generated files.

After training, the project directory contains:

- `output/wakeword.onnx`: ONNX detector with input `waveform` and output `score`
- `output/config.json`: threshold, sample rate, mel settings, backend, and metadata
- `samples/`: local recordings, imports, and generated clips used for training

The WavLM teacher is training-only. It is not included in the exported ONNX graph.
