# wakeword-forge

Private custom wake words for local agents.

wakeword-forge helps you collect a small wake-phrase dataset, review the clips, train a compact detector, and export an ONNX model for local voice apps. Samples stay local unless you choose to share them.

This project is meant to run from a source checkout. It is not published on PyPI, so do not use `pip install wakeword-forge`.

## quickstart

```bash
git clone git@github.com:H-Ali13381/wakeword-forge.git
cd wakeword-forge
make start DIR=~/wakeword_forge_project
```

`make start` creates `.venv`, installs the local checkout, and opens the Streamlit dashboard. The dashboard is the recommended path for a first model because it walks through recording, review, training, live quality checks, and final model acceptance.

For development checks:

```bash
make check
```

## normal workflow

1. Choose one or more wake phrases, such as `Hey Nova`.
2. Record or import positive wake-phrase clips.
3. Add negatives: background speech, silence, noise, partial phrases, and near misses.
4. Optionally generate TTS positives or hard negatives.
5. Review recorded and generated samples.
6. Train the WavLM teacher -> RepCNN student model.
7. Run a live quality check.
8. Accept the model before using it as final.

## common commands

Use Make targets from the repository root. They install into the repo-local `.venv` and call `.venv/bin/wakeword-forge` for you.

| task | command |
| --- | --- |
| Open the dashboard | `make start DIR=~/wakeword_forge_project` |
| Run the terminal wizard | `make cli-run DIR=~/wakeword_forge_project` |
| Show project status | `make info DIR=~/wakeword_forge_project` |
| Record positives | `make record DIR=~/wakeword_forge_project PHRASE='Hey Nova' N=20` |
| Generate TTS positives | `make synth DIR=~/wakeword_forge_project PHRASE='Hey Nova' N=300 ENGINE=kokoro` |
| Import background negatives | `make import-negatives DIR=~/wakeword_forge_project NEG_SOURCE_DIR=~/clips NEG_LIMIT=150` |
| Review samples | `make review DIR=~/wakeword_forge_project` |
| Audit generated clips | `make audit DIR=~/wakeword_forge_project` |
| Train and export ONNX | `make train DIR=~/wakeword_forge_project` |
| Run guided live check | `make quality-check DIR=~/wakeword_forge_project` |
| Accept checked model | `make accept-model DIR=~/wakeword_forge_project` |
| Test the accepted model on mic input | `make mic-test DIR=~/wakeword_forge_project` |

Run `make help` for the full target list.

If you need the Typer CLI directly, install once and call the repo-local binary:

```bash
make install
.venv/bin/wakeword-forge --help
.venv/bin/wakeword-forge train --dir ~/wakeword_forge_project
```

## negative data

Training fails early if there is not enough negative coverage. The current guardrails require:

- at least 10 positive clips
- at least 5 negative clips
- at least 150 general/background negatives in `samples/negatives/`
- for multi-word phrases with TTS enabled, at least 100 partial-phrase hard negatives in `samples/partials/`

For external data, prefer manifest imports so provenance survives review:

```bash
make import-negatives DIR=~/wakeword_forge_project \
  NEG_MANIFEST=~/external_negatives.jsonl \
  NEG_LIMIT=150 \
  NEG_LIMIT_PER_SOURCE=25 \
  NEG_STRATA='speech=50,noise=50,silence=50'
```

Manifest rows can include `path`, `label`, `source_dataset`, `category`, `license`, `start_sec`, `end_sec`, and transcript fields. Imported clips get sidecar metadata and an aggregate `negative_import_manifest.jsonl`.

## synthesis options

`make synth` supports:

- `ENGINE=kokoro`: default lightweight local TTS
- `ENGINE=piper`: Piper, if a voice is installed
- `ENGINE=qwentts`: Qwen3-TTS CustomVoice baseline synthesis

QwenTTS baseline synthesis does not clone a speaker. It uses built-in voice designs and does not require external reference audio.

```bash
make synth DIR=~/wakeword_forge_project PHRASE='Hey Nova' N=300 ENGINE=qwentts
```

There is also an optional Dockerized QwenTTS voice-clone flow. Use it only with consent, compatible licenses, or a defensible fair-use basis. It generates one candidate at a time and stages it for human review:

```bash
make qwentts-build
make qwentts-voice-clone-one DIR=~/wakeword_forge_project SOURCE_MANIFEST=~/wakeword_forge_project/voice_clone_sources.jsonl
make review-cloned-samples DIR=~/wakeword_forge_project
```

## training and output

The default backend trains a WavLM teacher, distills it into a compact RepCNN student, and exports only the runtime graph.

After training, the project directory contains:

- `output/wakeword.onnx`: ONNX detector with input `waveform` and output `score`
- `output/config.json`: threshold, sample rate, mel settings, backend, and metadata
- `samples/`: local recordings, imports, and generated clips used for training

The WavLM teacher is training-only. It is not included in the exported ONNX graph.

## privacy, licensing, and limits

Voice samples stay on your machine unless you package or share them. Before publishing audio, generated samples, trained models, or datasets, read:

- `DATA_PROVENANCE.md`
- `THIRD_PARTY_NOTICES.md`
- `SECURITY.md`

Do not publish another person's voice without consent and compatible licensing.

Known limits:

- A model trained mostly on one speaker may work best for that speaker, mic, room, and accent.
- Synthetic data helps coverage but does not replace real-world testing.
- Optional datasets and TTS voices have their own licenses and output terms.
- Benchmark claims are preliminary until a reproducible sweep is published.

## project docs

- `docs/architecture.md`: dashboard/CLI flow, review gates, training, export, and release hygiene
- `CONTRIBUTING.md`: contribution notes
- `COMMERCIAL.md`: commercial-use notes
- `SUPPORT.md`: support options

Code is Apache-2.0. See `LICENSE`, `NOTICE`, and `CITATION.cff`.
