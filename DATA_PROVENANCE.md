# Data provenance

wakeword-forge is designed for auditable local wake-word training. Voice recordings stay local by default, and published audio/model artifacts should include a provenance record.

## Default local data flow

`wakeword-forge run` can create:

- `samples/positives/`: user-recorded wake phrase examples.
- `samples/negatives/`: user-recorded counter-examples and local synthetic noise/silence.
- `samples/synthetic/`: optional TTS-generated positive examples.
- `samples/partials/`: optional partial-phrase hard negatives for multi-word phrases.
- `samples/confusables/`: optional confusable hard negatives.
- `samples/cloned_review/`: optional QwenTTS voice-cloned clips waiting for human positive/negative/unusable review.
- `output/wakeword.onnx`: exported detector.
- `output/config.json`: preprocessing, threshold, and runtime metadata.

These files are not uploaded by default.

## Consent rules

Do not publish or contribute audio unless:

- The speaker gave explicit consent.
- The license allows the intended use and redistribution.
- Generated samples are allowed by the TTS backend, model, and voice-asset terms.
- The artifact clearly states whether it contains real user speech, synthetic audio, public dataset audio, or a mixture.

## Voice cloning and source-audio policy

Voice cloning is more sensitive than ordinary TTS. Treat every source row as provenance data, not just an input file:

- Prefer open datasets whose license and consent model allow your intended use.
- Keep YouTube disabled by default. Only pass `--allow-youtube` for personal experiments, creator-owned material, rights-cleared clips, or a fair use scenario you have evaluated yourself.
- Store `speaker_id`, `source_type`, `dataset_id`, `license`, `usage_policy`, and source URLs in the source manifest and generated sidecar metadata.
- Never publish voice-cloned audio or models trained on cloned voices unless the consent, license, and redistribution basis are clear.
- Human review is mandatory: staged cloned clips must be labeled `positive`, `negative`, or `unusable`; unusable clips should be deleted.

## Recommended provenance manifest

For packaged models or generated datasets, include a JSON manifest like:

```json
{
  "artifact": "models/example/wakeword.onnx",
  "phrase": "Hey Nova",
  "created_by": "wakeword-forge",
  "sample_rate": 16000,
  "training_data": [
    {
      "type": "user_recording",
      "count": 20,
      "speaker_consent": true,
      "redistributable": false,
      "notes": "Local user recordings; not bundled."
    },
    {
      "type": "synthetic_tts",
      "count": 300,
      "backend": "kokoro",
      "voice_assets_license": "verify before release",
      "generated_output_terms": "verify before release",
      "redistributable": "unknown_until_verified"
    }
  ],
  "excluded_sources": [
    "No unauthorized voice cloning",
    "No scraped speaker audio"
  ]
}
```

## Optional public datasets

- Common Voice: verify exact version, split, language, and terms before redistributing samples or trained models.
- ESC-50: CC BY-NC 3.0. Treat as research/non-commercial and keep disabled by default for commercial-safe workflows.

## Model cards

Every published demo model should include:

- Wake phrase.
- Intended use.
- Training data summary.
- Speaker/data consent basis.
- Known limitations.
- Evaluation metrics and threshold.
- License and citation.
- Whether the model is expected to generalize beyond the training speaker/environment.
