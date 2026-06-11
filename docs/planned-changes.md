# Planned changes

This file records implementation ideas that are not ready to land as code yet, but should remain visible inside the project.

## 2026-06-02 — Condition-labeled quality-check coverage

**Status:** planned

**Source:** Okay/Hey Hermes field testing: low gain, farther/off-axis mic position, and fast speech can miss activations even when the model passes the current basic live quality gate.

### Problem

The current `quality-check` command is a basic acceptance gate, not a robustness matrix.

Current behavior:

- `forge/cli.py::quality_check` runs:
  - 5 generic positive trials,
  - 3 generic near-miss trials,
  - 2 silence/background trials,
  - `--duration 2.0` seconds per guided trial.
- `forge/review.py` currently models observations as `ObservationKind = Literal["positive", "near_miss", "silence"]` plus a score.
- `QualityReport.passed` only requires:
  - zero positive misses,
  - zero false triggers on near-miss/silence trials.
- `forge/config.py` persists aggregate quality-check fields: positive hits/trials, false triggers, score min/max, model fingerprint, and accepted-model fingerprint.

That means a model can pass if the user speaks the normal wake phrase close to the mic, at normal speed, in clean conditions. It does not prove robustness for the deployment conditions that caused failures.

### Non-goals / constraints

- Do **not** treat slow speech as a current activation failure unless future labeled eval data shows it.
- Do **not** “fix” this by increasing the 2-second speech-sample cap. The cap is intentional: it leaves slack inside the larger 3-second model/scoring window for start/middle/end placement and for slowed user-sample augmentation.
- Do **not** lower the model threshold as the first response. Measure per-condition recall and false triggers first.
- Keep the simple acceptance flow usable; robust scenario coverage can be an explicit mode or preset.

### Planned behavior

Add condition-labeled quality scenarios while preserving the current basic gate.

Suggested scenario set:

| Scenario ID | Expected behavior | Notes |
|---|---|---|
| `normal_close_mic` | trigger | current positive trial, close mic, normal speed |
| `quiet_low_gain` | trigger | quiet voice / low gain; evaluate low-SNR behavior, not just raw amplitude |
| `far_mic` | trigger | farther from mic, room coloration likely present |
| `off_axis_mic` | trigger | user not facing the mic directly |
| `fast_phrase` | trigger | fast but intelligible wake phrase |
| `noisy_background_positive` | trigger | wake phrase while realistic background noise is present |
| `near_miss` | no trigger | similar phrase, not the wake phrase |
| `silence_background` | no trigger | silence or ordinary background |

Optional placement labels for controlled/eval-mode checks:

- `early`
- `centered`
- `late`

Placement labels should not remove the 2-second speech-sample slack; they should verify that the phrase can appear at different positions inside the larger model/scoring window.

### Implementation sketch

1. Extend quality-check data modeling in `forge/review.py`.
   - Add a scenario identifier separate from pass/fail expectation.
   - Preserve backward compatibility for existing aggregate fields.
   - Candidate shape:
     - `scenario_id: str`
     - `expected: Literal["trigger", "no_trigger"]`
     - `score: float`
     - optional `tags: tuple[str, ...]`

2. Add scenario presets to `forge/cli.py::quality_check`.
   - Keep existing behavior as a `basic` preset.
   - Add a `robust` preset with the scenario table above.
   - Keep `--duration 2.0` as the default guided trial duration.

3. Improve reporting.
   - Print a per-scenario table: score, threshold, pass/fail, margin.
   - Summarize aggregate positives/false triggers as today, but do not hide condition failures inside the aggregate.

4. Persist scenario results.
   - Keep existing config fields for compatibility.
   - Add a structured field or sidecar for scenario-level quality results.
   - Future model exports should include quality-check scenario summaries in `output/wakeword.json` metadata.

5. Add tests.
   - Unit-test scenario pass/fail summarization in `tests/test_human_review.py` or a new focused test file.
   - CLI tests should verify that `basic` keeps current prompts/counts and `robust` emits condition-labeled prompts.
   - Config/project tests should verify that old aggregate fields still update and scenario results are retained.

### Near-term quick wins

These are low-effort/high-impact changes to improve observability and robustness before changing the model architecture or blindly tuning thresholds.

1. Add an acoustic-augmentation status report before training.
   - Surface counts for configured acoustic asset folders:
     - `augmentation_noise_dir`
     - `augmentation_ir_dir`
     - `augmentation_short_noise_dir`
     - `augmentation_truck_noise_dir`
   - Also print whether `training_augmentation_enabled` and `use_spectrogram_augmentation` are active.
   - Warn when training augmentation is enabled but all far-field/noise asset folders are empty.

2. Record augmentation provenance in `output/wakeword.json`.
   - Include `training_augmentation_enabled`, `training_augmentation_preset`, `regular_negative_augmentation_preset`, `training_augmentation_max_chain`, `training_augmentation_probability`, and `use_spectrogram_augmentation`.
   - Include acoustic asset counts and, where practical, a simple config/source fingerprint so future model comparisons are not muddy.

3. Add a lightweight offline robustness-eval command before retraining.
   - Candidate CLI: `wakeword-forge robustness-eval --dir ...`.
   - Load the current `output/wakeword.onnx`, score existing positives at the current threshold, then apply controlled eval-only transforms.
   - Report per-condition recall/miss counts for low-SNR/low-gain, far-field IR, fast phrase, background-noise positive, and early/center/late placement.
   - Treat this as measurement, not a replacement for live `quality-check`.

4. Enable or explicitly warn about disabled spectrogram augmentation.
   - `SpectrogramAugmentor` already supports time warp, frequency masking, time masking, and mel-domain noise.
   - First step: warn when `use_spectrogram_augmentation` is false for WavLM→RepCNN training.
   - Optional later step: consider defaulting it on after targeted tests confirm no regression.

5. Make speed augmentation more fast-speech aware.
   - Keep the existing slow-side coverage, but add stronger fast-side coverage for intelligible fast phrases.
   - Consider adding TTS/sample-generation rate variants around `1.20` and `1.30`.
   - Consider raising the waveform `speed` transform weight in `CascadingAugmentor` and widening the fast side modestly, e.g. toward `1.35`, while reserving extreme rates for stress eval.

6. Replace or supplement wraparound time shift with zero-padded onset jitter.
   - Current `time_shift(wav)` style behavior should not wrap audio from the end back to the beginning for realistic placement robustness.
   - Add a zero-padded onset/placement jitter transform that shifts wake phrase timing while preserving total model-window length.
   - Keep the 2-second speech-sample cap intact; the point is to test placement inside the larger 3-second scoring window.

7. Add condition-labeled positive capture guidance.
   - Prompt for small batches of `normal_close_mic`, `quiet_low_gain`, `far_mic`, `off_axis_mic`, `fast_phrase`, and `noisy_background_positive` samples.
   - Save sidecar metadata with the condition label, phrase, duration, RMS/peak level, and current-model score if a model exists.
   - This creates real field data instead of relying only on synthetic transforms.

8. Add a low-SNR mic transform.
   - Model low gain as SNR loss, not just amplitude scaling that later peak-normalization can erase.
   - Candidate transform: attenuate speech, add mic/self/background noise, optionally apply mild band-limit or high-frequency rolloff, then normalize.
   - Use this for eval first; promote to training augmentation only after observing useful failure coverage.

9. Add dashboard nudges for empty acoustic assets.
   - In dashboard/config surfaces, warn when training augmentation is on but no external acoustic/noise/IR assets are configured.
   - Include the expected folders/fields and explain that far-mic/room robustness will be limited without real or representative acoustic assets.

10. Add quality-check margin reporting.
    - Print per-trial score, threshold, and margin (`score - threshold`).
    - Summarize weakest positive, strongest negative/near-miss, and condition-specific misses.
    - Use this to distinguish barely passing models from robust ones.

Avoid these as first responses:

- lowering the threshold before per-condition evaluation;
- increasing the 2-second speech-sample cap to mask placement/timing issues;
- adding heavy precomputed augmentation shards;
- changing the WavLM→RepCNN architecture before robustness measurements identify the dominant failure mode.

### Acceptance criteria

- `wakeword-forge quality-check --scenario-preset basic` behaves like the current command.
- `wakeword-forge quality-check --scenario-preset robust` guides the user through condition-labeled trials.
- The report clearly identifies which condition failed, e.g. `far_mic` or `fast_phrase`, instead of only saying “positive miss”.
- The 2-second guided trial duration remains the default unless explicitly overridden.
- Scenario-level results are persisted or exported well enough that future model comparisons are not muddy.
- Training and dashboard surfaces make empty/missing acoustic augmentation assets visible instead of silently appearing robust.
- Exported model metadata records enough augmentation provenance to compare robustness across runs.
- Offline robustness evaluation can report per-condition recall at the current threshold before retraining or threshold changes.
- Existing tests pass, plus new tests for scenario summarization, CLI preset behavior, augmentation-provenance export, and robustness-eval reporting.

### Verification commands

```bash
make check
# or targeted during development:
.venv/bin/python -m pytest tests/test_human_review.py tests/test_config.py tests/test_dashboard_ux.py -q
.venv/bin/python -m ruff check forge tests
git diff --check
```
