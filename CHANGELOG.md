# Changelog

All notable changes to wakeword-forge are tracked here.

## [0.1.1] - 2026-05-25

Release URL: https://github.com/H-Ali13381/wakeword-forge/releases/tag/v0.1.1

### Changed

- Bumped package, CLI, citation, changelog, and releasing metadata for the v0.1.1 source release.
- Moved the default workspace examples and runtime defaults to the ignored repo-local `./projects/default` project directory.
- Added release hygiene coverage for package metadata, CLI `--version`, changelog entries, and tagged GitHub Release instructions.

## [0.1.0] - 2026-05-25

Release URL: https://github.com/H-Ali13381/wakeword-forge/releases/tag/v0.1.0

### Added

- Initial public release version for the local-first wake-word training workflow.
- Dashboard and CLI workflows for recording, reviewing, synthesizing, training, checking, and accepting a custom wake-word detector.
- WavLM teacher to RepCNN student training backend with ONNX export metadata.
- Public release hygiene tests for keeping runtime samples, model artifacts, caches, and private credentials out of the source tree.
- Release-version checks for package metadata, CLI `--version`, changelog coverage, and release process documentation.
