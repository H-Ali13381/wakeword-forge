# Contributing to wakeword-forge

Thanks for helping make private custom wake-word training easier to use.

## Ground rules

By contributing, you certify that:

- You wrote the contribution yourself, or you have the right to submit it under Apache-2.0.
- You are not submitting employer-owned, client-owned, private, or otherwise restricted code.
- You are not submitting audio, model weights, generated samples, or datasets unless you have consent and compatible redistribution rights.
- You will clearly document any third-party source, dataset, model, or generated artifact involved in your contribution.

## Development setup

```bash
make install-dev
make check
```

The expected validation command is:

```bash
make check
```

## Code style

- Keep the default path simple and local-first.
- Prefer small, testable modules over large orchestration scripts.
- Add tests for behavior changes.
- Avoid large binary artifacts in git. Audio samples, model weights, and benchmark outputs should stay out of the repository unless explicitly packaged with a model card and provenance file.

## Data and model contributions

For audio/model/data contributions, include:

- Source and consent basis.
- License and redistribution status.
- Whether generated outputs can be shared.
- Any non-commercial or research-only restrictions.
- A short model card or data provenance note when applicable.

Do not submit voice recordings of other people without their explicit consent and compatible licensing.
