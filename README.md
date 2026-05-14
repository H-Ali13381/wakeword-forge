# wakeword-forge

Private custom wake words for local agents.

wakeword-forge trains a personal wake-word detector from a small number of voice samples, keeps recordings local by default, and exports an ONNX model that local agents or voice apps can embed.

## Quickstart: dashboard-first

```bash
pip install "wakeword-forge[ui,tts]"
wakeword-forge dashboard --dir ~/wakeword_forge_project
```

Development checkout:

```bash
make install-dev
make start DIR=~/wakeword_forge_project
```

`make start` opens the Streamlit dashboard. The dashboard guides the full flow:

1. choose one or more wake phrases
2. choose microphone recording or an existing folder for positive wake-phrase clips
3. collect positive and negative examples
4. generate optional TTS augmentation and hard negatives
5. review recorded/generated samples before training
6. train/export the detector
7. run a guided live quality check
8. explicitly accept the model before treating it as final
9. copy the pure CLI fallback commands when a terminal workflow is better

## Pure CLI fallback

The dashboard is the preferred interface, but every core action remains available from Make and Typer:

```bash
make cli-run DIR=~/wakeword_forge_project
make info DIR=~/wakeword_forge_project
make record DIR=~/wakeword_forge_project PHRASE='Hey Nova' N=20
make synth DIR=~/wakeword_forge_project PHRASE='Hey Nova' N=300 ENGINE=kokoro
make review DIR=~/wakeword_forge_project
make audit DIR=~/wakeword_forge_project
make train DIR=~/wakeword_forge_project
make quality-check DIR=~/wakeword_forge_project
make accept-model DIR=~/wakeword_forge_project
make mic-test DIR=~/wakeword_forge_project
```

Direct CLI equivalents:

```bash
wakeword-forge run --dir ~/wakeword_forge_project
wakeword-forge info --dir ~/wakeword_forge_project
wakeword-forge review-samples --dir ~/wakeword_forge_project
wakeword-forge audit-generated --dir ~/wakeword_forge_project
wakeword-forge train --dir ~/wakeword_forge_project
wakeword-forge quality-check --dir ~/wakeword_forge_project
wakeword-forge accept-model --dir ~/wakeword_forge_project
wakeword-forge test ~/wakeword_forge_project/output/wakeword.onnx
```

## Pipeline

```text
1. Choose a wake phrase           ("Hey Nova", "Okay Atlas", anything)
2. Record positive examples       (guided microphone prompts)
3. Record counter-examples        (similar words, ambient speech)
4. Add augmentation               (TTS variants, hard negatives, noise)
5. Review samples                 (play/delete/re-record bad takes)
6. Audit generated clips          (spot-check TTS positives and hard negatives)
7. Train                          (compact DS-CNN backend)
8. Quality check                  (wake hits, near misses, silence/background)
9. Accept                         (mark wakeword.onnx as final for runtime use)
```

## Output

After training you get:

- `output/wakeword.onnx`: detector model with stable ONNX input `waveform` and output `score`.
- `output/config.json`: sample rate, threshold, mel parameters, backend, and model metadata.
- `samples/`: local recordings and generated examples for the training run.

## Architecture

The public v0.1 backend is a compact DS-CNN-style keyword-spotting model:

- Input: raw mono waveform at 16 kHz.
- Frontend: normalized log-mel frames.
- Model: depthwise-separable 1D convolutions over mel frames.
- Runtime artifact: a single ONNX detector graph.

Heavy research backends are intentionally out of scope for the first public
release. The goal is a small, inspectable training path that is easy to run,
test, and integrate.

## Privacy and data provenance

Voice samples stay on your machine unless you explicitly package or share them.

Before publishing audio, generated samples, or trained models, read:

- `DATA_PROVENANCE.md`
- `THIRD_PARTY_NOTICES.md`
- `SECURITY.md`

Do not publish another person's voice without consent and compatible licensing.

## Limitations

- A model trained mostly on one speaker may work best for that speaker, microphone, room, and accent.
- Synthetic augmentation helps coverage but does not replace real-world validation.
- Optional datasets and TTS voices have their own licenses and output terms.
- Benchmark claims should be treated as preliminary until a full reproducible sweep is published.

## Tests

```bash
make check
```

Current checks run the unit tests under the project virtual environment.

## License, citation, and attribution

Code is licensed under Apache-2.0. See `LICENSE` and `NOTICE`.

Apache-2.0 requires preservation of legal notices in redistributed copies. Public links, citations, and model-card attribution are appreciated when you use wakeword-forge in demos, integrations, papers, or releases.

Citation metadata is in `CITATION.cff`.

## Commercial use and support

Commercial use is allowed under Apache-2.0 when license and NOTICE terms are followed. Optional paid support or integration help may be available; see `COMMERCIAL.md` and `SUPPORT.md`.
