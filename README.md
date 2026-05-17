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
make synth DIR=~/wakeword_forge_project PHRASE='Hey Nova' N=300 ENGINE=qwentts
make import-negatives DIR=~/wakeword_forge_project NEG_MANIFEST=~/external_negatives.jsonl NEG_LIMIT=150 NEG_LIMIT_PER_SOURCE=25 NEG_STRATA='speech=50,noise=50,silence=50'
make review DIR=~/wakeword_forge_project
make audit DIR=~/wakeword_forge_project
make train DIR=~/wakeword_forge_project AUGMENTATION_PRESET=standard REGULAR_NEGATIVE_PRESET=light
make quality-check DIR=~/wakeword_forge_project
make accept-model DIR=~/wakeword_forge_project
make qwentts-build
make qwentts-voice-clone-one DIR=~/wakeword_forge_project SOURCE_MANIFEST=~/wakeword_forge_project/voice_clone_sources.jsonl
make review-cloned-samples DIR=~/wakeword_forge_project
make mic-test DIR=~/wakeword_forge_project
```

Direct CLI equivalents:

```bash
wakeword-forge run --dir ~/wakeword_forge_project
wakeword-forge info --dir ~/wakeword_forge_project
wakeword-forge synth 'Hey Nova' --out ~/wakeword_forge_project/samples/synthetic --n 300 --engine qwentts
wakeword-forge import-negatives --dir ~/wakeword_forge_project --manifest ~/external_negatives.jsonl --limit 150 --limit-per-source 25
wakeword-forge review-samples --dir ~/wakeword_forge_project
wakeword-forge audit-generated --dir ~/wakeword_forge_project
wakeword-forge train --dir ~/wakeword_forge_project --augmentation-preset standard --regular-negative-preset light
wakeword-forge quality-check --dir ~/wakeword_forge_project
wakeword-forge accept-model --dir ~/wakeword_forge_project
wakeword-forge voice-clone-one --dir ~/wakeword_forge_project --source-manifest ~/wakeword_forge_project/voice_clone_sources.jsonl
wakeword-forge review-cloned-samples --dir ~/wakeword_forge_project
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

## Negative coverage guardrails

wakeword-forge now treats sparse negative data as a training blocker, not just a recommendation. Before training, the project must have:

- at least 150 general/background negatives in `samples/negatives/`
- for multi-word phrases with TTS augmentation enabled, at least 100 partial-phrase hard negatives in `samples/partials/`
- the basic hard minimums of 10 positive clips and 5 negative clips

This prevents the systematic false-positive failure mode where hundreds of wake-phrase positives are trained against only a handful of counter-examples. The dashboard exposes the missing counts as "Background negatives" and "Hard negatives" fill actions; `wakeword-forge run` also fills the background-negative target automatically before training.

For external negative data, use a manifest-driven methodology: build or provide a JSONL manifest, cap imports per source so one corpus cannot dominate, chunk long files, normalize to 16 kHz mono PCM WAV, stratify by category when mixing speech/noise/silence sources, and keep provenance. `wakeword-forge import-negatives` accepts either a source folder or JSONL rows with fields such as `path`, `label`, `source_dataset`, `category`, `license`, optional `start_sec`/`end_sec`, and optional `transcript_exclusion_terms` plus transcript text. Imported clips write per-file `.wav.json` sidecars and an aggregate `negative_import_manifest.jsonl` in the target sample directory.

Use `--strata speech=50,noise=50,silence=50` to enforce per-category quotas independent of manifest order. The default stratum field is `category`; pass `--stratify-by <field>` if a manifest uses another key. Rows whose stratum is not listed in `--strata` are intentionally ignored.

Examples:

```bash
wakeword-forge import-negatives --dir ~/wakeword_forge_project --source-dir ~/speech_or_noise_clips --kind background --limit 150 --limit-per-source 25
wakeword-forge import-negatives --dir ~/wakeword_forge_project --manifest ~/external_negatives.jsonl --kind background --limit 150 --limit-per-source 25 --max-chunks-per-file 20 --strata speech=50,noise=50,silence=50
wakeword-forge import-negatives --dir ~/wakeword_forge_project --source-dir ~/partial_phrase_clips --kind partial --limit 100
```

## Training-time acoustic augmentation

wakeword-forge applies training-time acoustic augmentation after sample review and before DS-CNN optimization. This is separate from TTS sample generation: it changes the waveform or mel features seen by the trainer without writing extra reviewed files.

Defaults:

- positives and phrase-specific hard negatives use `standard` cascading waveform augmentation
- broad/background negatives use `light` augmentation
- SpecAugment-style mel masking is available but off by default
- optional local WAV folders can provide background noise, short transient noise, low-frequency rumble, and room impulse responses

The waveform cascade includes Gaussian noise/SNR, filtering, gain/gain transitions, time masking, time stretch, pitch shift, polarity inversion, clipping distortion, optional noise mixing, and optional room-response convolution. The spectrogram augmentor includes frequency masking, time masking, time warping, and mel-domain noise.

No extra package install is required for the current training-time augmentation path; it uses the core `torch` / `torchaudio` stack.

Examples:

```bash
wakeword-forge train --dir ~/wakeword_forge_project \
  --augmentation \
  --augmentation-preset standard \
  --regular-negative-preset light \
  --augmentation-noise-dir ~/noise_wavs \
  --augmentation-ir-dir ~/room_ir_wavs

wakeword-forge train --dir ~/wakeword_forge_project --no-augmentation
wakeword-forge train --dir ~/wakeword_forge_project --spectrogram-augmentation
```

## Synthetic TTS sources

`wakeword-forge synth` can generate ordinary synthetic wake-phrase samples from three TTS sources:

- `kokoro` — default lightweight CPU-capable source from `wakeword-forge[tts]`.
- `piper` — fast local Piper voice source, when a Piper voice is installed.
- `qwentts` — GPU Qwen3-TTS CustomVoice source using 13 mirrored CustomVoice voice designs and a 20-prompt style instruction grid.

QwenTTS baseline synthesis is not voice cloning and does not use external speaker reference audio. The mirrored voice design set includes Ryan, Aiden, Vivian, Serena, Uncle_Fu, Dylan, Eric, Ono_Anna, Sohee, plus cross-language Ryan/Aiden French and Vivian/Serena English accent-diversity entries. Install the optional extra and run it explicitly:

```bash
pip install -e ".[tts,ui,qwentts]"
make synth DIR=~/wakeword_forge_project PHRASE='Hey Nova' N=300 ENGINE=qwentts
wakeword-forge synth 'Hey Nova' --out ~/wakeword_forge_project/samples/synthetic --n 300 --engine qwentts
```

## QwenTTS voice-cloned sample sourcing

wakeword-forge can optionally generate speaker-diverse wake-phrase clips with a Dockerized QwenTTS runner. This flow is intentionally one sample at a time:

1. download or locate one source-audio file from an open dataset, local file, or explicitly opted-in YouTube row
2. transcribe the original audio with Whisper
3. select one clean single-speaker reference snippet
4. run one QwenTTS Base voice-clone job in Docker
5. transcribe the generated wakeword clip
6. validate audio energy, duration, and fuzzy wake-phrase transcript match
7. stage the clip under `samples/cloned_review/`
8. require a human to label it `positive`, `negative`, or `unusable`
9. move positives/negatives into the training pool and delete unusable clips
10. rerun `review-samples`, `audit-generated`, and `train` when ready

Responsible-use guardrails are part of the workflow. Only use voices and source audio when you have consent, compatible license rights, or a defensible fair use basis. YouTube source rows are disabled unless you pass `--allow-youtube`, and source metadata is preserved in sidecar JSON files for review and model-card provenance.

Example source manifest at `~/wakeword_forge_project/voice_clone_sources.jsonl`:

```jsonl
{"path":"/data/common_voice/en/clips/example.mp3","speaker_id":"cv-speaker-1","source_type":"open_dataset","dataset_id":"common_voice_en","license":"verify-version-terms","usage_policy":"research_or_license_compatible_use_only"}
{"youtube_url":"https://youtu.be/example","speaker_id":"creator-owned-or-cleared","source_type":"youtube","license":"source-rights-required","usage_policy":"personal_use_only_unless_rights_are_cleared"}
```

```bash
make qwentts-build
pip install -e ".[tts,ui,voice]"
wakeword-forge voice-clone-one --dir ~/wakeword_forge_project --source-manifest ~/wakeword_forge_project/voice_clone_sources.jsonl
wakeword-forge voice-clone-one --dir ~/wakeword_forge_project --source-manifest ~/wakeword_forge_project/voice_clone_sources.jsonl --allow-youtube
wakeword-forge review-cloned-samples --dir ~/wakeword_forge_project --sample 1 --decision positive
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
