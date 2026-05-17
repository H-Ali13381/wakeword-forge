# Security and privacy

wakeword-forge is local-first. Voice recordings are saved under the project directory and are not uploaded by default.

## Reporting security issues

Please report security or privacy issues privately to the maintainer before opening a public issue. If no public contact is listed yet, open a minimal GitHub issue asking for a private contact channel without including sensitive details.

## Voice privacy

- Do not upload another person's voice without consent.
- Do not publish user recordings unless the speaker explicitly opted in and the license allows redistribution.
- Treat wake-word datasets as personal data when they contain identifiable voices.
- Review generated audio licenses before sharing generated samples or trained models.

## Local files

By default, a trained project may contain:

- `samples/` with user recordings and generated examples.
- `output/wakeword.onnx` with the trained detector.
- `output/config.json` with threshold and preprocessing metadata.

These paths are intended to stay local unless the user deliberately packages and publishes them with provenance metadata.
