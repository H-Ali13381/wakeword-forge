.PHONY: help start dashboard run cli-run record synth review audit train quality-check accept-model mic-test test info install install-dev install-voice qwentts-build qwentts-voice-clone-one review-cloned-samples check clean

# Default project directory and action parameters.
DIR    ?= $(HOME)/wakeword_forge_project
PHRASE ?= Hey Nova
N      ?= 20
ENGINE ?= kokoro
SOURCE_MANIFEST ?= $(DIR)/voice_clone_sources.jsonl
QWENTTS_IMAGE ?= wakeword-forge-qwentts:latest
QWENTTS_ARGS ?=

# Python / venv
VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
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

install: $(VENV)/.installed

install-dev: $(VENV)/.installed-dev

install-voice: $(VENV)/.installed-voice

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
synth: install
	$(FORGE) synth "$(PHRASE)" --out "$(DIR)/samples/synthetic" --n $(N) --engine $(ENGINE)

## Review recorded samples before training.
review: install
	$(FORGE) review-samples --dir "$(DIR)"

## Audit generated TTS / hard-negative clips before training.
audit: install
	$(FORGE) audit-generated --dir "$(DIR)"

## Training only (uses existing samples in DIR).
train: install
	$(FORGE) train --dir "$(DIR)" --backend dscnn

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

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache *.egg-info

help:
	@printf "wakeword-forge targets:\n"
	@printf "  make start/dashboard DIR=...     Launch Streamlit dashboard\n"
	@printf "  make cli-run DIR=...             Run the pure CLI wizard\n"
	@printf "  make record PHRASE='Hey Nova'    Record guided positives\n"
	@printf "  make synth PHRASE='Hey Nova'     Generate TTS positives\n"
	@printf "  make review DIR=...              Review/approve recorded samples\n"
	@printf "  make audit DIR=...               Audit generated clips\n"
	@printf "  make train DIR=...               Train/export ONNX\n"
	@printf "  make quality-check DIR=...       Guided live quality check\n"
	@printf "  make accept-model DIR=...        Accept checked model\n"
	@printf "  make qwentts-build               Build Dockerized QwenTTS runner\n"
	@printf "  make qwentts-voice-clone-one     Clone one sample into human review queue\n"
	@printf "  make review-cloned-samples       Label cloned samples positive/negative/unusable\n"
	@printf "  make mic-test DIR=...            Live microphone threshold test\n"
	@printf "  make info DIR=...                Print project status\n"
	@printf "  make check                       Run unit tests\n"
