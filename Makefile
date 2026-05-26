.PHONY: help start dashboard run cli-run record synth import-negatives review audit train quality-check accept-model mic-test test info install install-dev install-voice install-qwentts qwentts-build qwentts-voice-clone-one review-cloned-samples check release-check clean

# Default project directory and action parameters.
# Keep generated samples, caches, and model outputs in an ignored repo-local workspace.
DIR    ?= projects/default
PHRASE ?= Hey Nova
N      ?= 20
ENGINE ?= qwentts
SOURCE_MANIFEST ?= $(DIR)/voice_clone_sources.jsonl
NEG_SOURCE_DIR ?=
NEG_MANIFEST ?=
NEG_KIND ?= background
NEG_LIMIT ?=
NEG_STRATA ?=
NEG_STRATIFY_BY ?= category
NEG_LIMIT_PER_SOURCE ?= 100
NEG_MAX_CHUNKS_PER_FILE ?= 20
NEG_CHUNK_DURATION ?= 3.0
AUGMENTATION ?= --augmentation
AUGMENTATION_PRESET ?= standard
REGULAR_NEGATIVE_PRESET ?= light
SPECTROGRAM_AUGMENTATION ?= --no-spectrogram-augmentation
AUGMENTATION_NOISE_DIR ?=
AUGMENTATION_IR_DIR ?=
AUGMENTATION_SHORT_NOISE_DIR ?=
AUGMENTATION_TRUCK_NOISE_DIR ?=
QWENTTS_IMAGE ?= wakeword-forge-qwentts:latest
QWENTTS_ARGS ?=

ifneq ($(filter qwentts qwen qwen-tts qwen3-tts,$(ENGINE)),)
SYNTH_INSTALL := install-qwentts
else
SYNTH_INSTALL := install
endif

# Python / venv
VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(PYTHON) -m pip
FORGE   := $(VENV)/bin/wakeword-forge

# ── Setup ─────────────────────────────────────────────────────────────────────

$(VENV):
	python -m venv $(VENV)

$(VENV)/.installed: $(VENV) pyproject.toml
	$(PIP) install -e ".[tts,ui]" -q
	touch $@

$(VENV)/.installed-dev: $(VENV) pyproject.toml
	$(PIP) install -e ".[tts,ui,dev]" -q
	touch $@

$(VENV)/.installed-voice: $(VENV) pyproject.toml
	$(PIP) install -e ".[tts,ui,dev,voice]" -q
	touch $@

$(VENV)/.installed-qwentts: $(VENV) pyproject.toml
	$(PIP) install -e ".[tts,ui,qwentts]" -q
	touch $@

install: $(VENV)/.installed

install-dev: $(VENV)/.installed-dev

install-voice: $(VENV)/.installed-voice

install-qwentts: $(VENV)/.installed-qwentts

# ── Main commands ─────────────────────────────────────────────────────────────

## Default: local Streamlit dashboard workflow.
start: dashboard

## Launch dashboard.
dashboard: install
	$(FORGE) dashboard --dir "$(DIR)"

## Alias for the default dashboard workflow.
run: dashboard

## Pure CLI fallback: full interactive pipeline.
cli-run: install
	$(FORGE) run --dir "$(DIR)"

## Recording only.
record: install
	$(FORGE) record "$(PHRASE)" --out "$(DIR)/samples/positives" --n $(N)

## TTS synthesis only.
synth: $(SYNTH_INSTALL)
	$(FORGE) synth "$(PHRASE)" --out "$(DIR)/samples/synthetic" --n $(N) --engine $(ENGINE)

## Import external negative audio from a folder or JSONL manifest.
import-negatives: install
	$(FORGE) import-negatives --dir "$(DIR)" --kind "$(NEG_KIND)" --limit-per-source $(NEG_LIMIT_PER_SOURCE) --max-chunks-per-file $(NEG_MAX_CHUNKS_PER_FILE) --chunk-duration $(NEG_CHUNK_DURATION) $(if $(NEG_SOURCE_DIR),--source-dir "$(NEG_SOURCE_DIR)") $(if $(NEG_MANIFEST),--manifest "$(NEG_MANIFEST)") $(if $(NEG_LIMIT),--limit $(NEG_LIMIT)) $(if $(NEG_STRATA),--strata "$(NEG_STRATA)" --stratify-by "$(NEG_STRATIFY_BY)")

## Review recorded samples before training.
review: install
	$(FORGE) review-samples --dir "$(DIR)"

## Audit generated TTS / hard-negative clips before training.
audit: install
	$(FORGE) audit-generated --dir "$(DIR)"

## Training only (uses existing samples in DIR).
train: install
	$(FORGE) train --dir "$(DIR)" --backend wavlm-repcnn $(AUGMENTATION) --augmentation-preset "$(AUGMENTATION_PRESET)" --regular-negative-preset "$(REGULAR_NEGATIVE_PRESET)" $(SPECTROGRAM_AUGMENTATION) $(if $(AUGMENTATION_NOISE_DIR),--augmentation-noise-dir "$(AUGMENTATION_NOISE_DIR)") $(if $(AUGMENTATION_IR_DIR),--augmentation-ir-dir "$(AUGMENTATION_IR_DIR)") $(if $(AUGMENTATION_SHORT_NOISE_DIR),--augmentation-short-noise-dir "$(AUGMENTATION_SHORT_NOISE_DIR)") $(if $(AUGMENTATION_TRUCK_NOISE_DIR),--augmentation-truck-noise-dir "$(AUGMENTATION_TRUCK_NOISE_DIR)")

## Guided live quality checkpoint for a trained model.
quality-check: install
	$(FORGE) quality-check --dir "$(DIR)"

## Accept the current model after a passing quality check.
accept-model: install
	$(FORGE) accept-model --dir "$(DIR)"

## Build the Docker image used for one-sample QwenTTS voice cloning.
qwentts-build:
	docker build -t "$(QWENTTS_IMAGE)" docker/qwentts

## Generate one QwenTTS voice-cloned sample and stage it for human review.
qwentts-voice-clone-one: install-voice
	$(FORGE) voice-clone-one --dir "$(DIR)" --source-manifest "$(SOURCE_MANIFEST)" --image "$(QWENTTS_IMAGE)" $(QWENTTS_ARGS)

## Review pending QwenTTS voice-cloned samples as positive, negative, or unusable.
review-cloned-samples: install
	$(FORGE) review-cloned-samples --dir "$(DIR)"

## Live mic test against a trained model.
mic-test: install
	$(FORGE) test "$(DIR)/output/wakeword.onnx"

## Backward-compatible alias for live mic testing.
test: mic-test

## Show project status.
info: install
	$(FORGE) info --dir "$(DIR)"

# ── Dev ───────────────────────────────────────────────────────────────────────

check: install-dev
	$(PYTHON) -m pytest tests/ -v

release-check: install-dev
	$(PYTHON) -m pytest tests/test_release_version.py tests/test_public_release_hygiene.py -q

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache *.egg-info

help:
	@printf "wakeword-forge targets:\n"
	@printf "  make start/dashboard DIR=...     Launch Streamlit dashboard\n"
	@printf "  make cli-run DIR=...             Run the pure CLI wizard\n"
	@printf "  make record PHRASE='Hey Nova'    Record guided positives\n"
	@printf "  make synth ENGINE=qwentts|kokoro|piper  Generate TTS positives\n"
	@printf "  make import-negatives NEG_MANIFEST=... Import capped external negatives\n"
	@printf "  make review DIR=...              Review/approve recorded samples\n"
	@printf "  make audit DIR=...               Audit generated clips\n"
	@printf "  make train DIR=... AUGMENTATION_PRESET=standard|light  Train/export ONNX\n"
	@printf "  make quality-check DIR=...       Guided live quality check\n"
	@printf "  make accept-model DIR=...        Accept checked model\n"
	@printf "  make qwentts-build               Build Dockerized QwenTTS runner\n"
	@printf "  make qwentts-voice-clone-one     Clone one sample into human review queue\n"
	@printf "  make review-cloned-samples       Label cloned samples positive/negative/unusable\n"
	@printf "  make mic-test DIR=...            Live microphone threshold test\n"
	@printf "  make info DIR=...                Print project status\n"
	@printf "  make release-check               Verify version, changelog, and release hygiene\n"
	@printf "  make check                       Run unit tests\n"
