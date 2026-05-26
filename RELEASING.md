# Releasing wakeword-forge

Current package release version: `0.1.2`.

wakeword-forge source releases are tagged in git and published as GitHub Releases. Trained wake-word ONNX files, generated audio, review packs, private manifests, and benchmark artifacts are not bundled with source releases. Publish model artifacts separately with a model card and provenance record.

## Version sources

Keep these in sync for every source release:

- `pyproject.toml` `[project].version`
- `wakeword_forge/__init__.py` `__version__`
- `CHANGELOG.md` entry for the same version
- Git tag `v<version>`
- GitHub Release `v<version>`

`tests/test_release_version.py` checks the package metadata, CLI `--version`, changelog entry, and this releasing guide.

## Preflight

Run from the repository root:

```bash
make release-check
git status --short --branch
git tag --list "v0.1.2"
```

A releasable tree should have only intended source/doc/test changes staged or committed. Do not stage `projects/`, `.venv/`, generated `.wav`, `.onnx`, review packs, or private manifests.

## Create the v0.1.2 source release

After the intended release commit is on `main`:

```bash
git tag -a v0.1.2 -m "wakeword-forge v0.1.2"
git push origin main
git push origin v0.1.2
gh release create v0.1.2 \
  --title "wakeword-forge v0.1.2" \
  --notes-file CHANGELOG.md \
  --verify-tag
```

If the release should stay unpublished while you review notes, add `--draft` to the `gh release create v0.1.2` command.

## Bumping a future release

1. Update `pyproject.toml` and `wakeword_forge/__init__.py` to the new semantic version.
2. Add a new top entry to `CHANGELOG.md` with the matching `v<version>` release URL.
3. Update the version examples in this file.
4. Run `make release-check` and the project checks needed for the release scope.
5. Commit, tag, push, and create the GitHub Release.
