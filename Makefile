.PHONY: help start dashboard run cli-run record synth train mic-test test info install install-dev check clean

# Default project directory and action parameters.
DIR    ?= $(HOME)/wakeword_forge_project
PHRASE ?= Hey Nova
N      ?= 20
ENGINE ?= kokoro

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

install: $(VENV)/.installed

install-dev: $(VENV)/.installed-dev

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

## Training only (uses existing samples in DIR).
train: install
	$(FORGE) train --dir "$(DIR)" --backend dscnn

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
	@printf "  make train DIR=...               Train/export ONNX\n"
	@printf "  make mic-test DIR=...            Live microphone threshold test\n"
	@printf "  make info DIR=...                Print project status\n"
	@printf "  make check                       Run unit tests\n"
