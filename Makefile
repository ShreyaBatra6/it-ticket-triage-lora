# IT Ticket Triage LoRA - pipeline entry points.
#   Colab / Linux:  make <target>
#   Windows:        mingw32-make <target>   (uses the local .venv)
#
# Override the interpreter with:  make smoke PYTHON=python3

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= python
endif

ADAPTER ?= outputs/lora-adapter

.PHONY: help install install-cpu check data data-push baseline train eval publish-dry eval-merged publish readme smoke

help:
	@echo "Targets: install install-cpu check data data-push baseline train eval publish-dry eval-merged publish readme smoke"

## Install pinned deps (Colab/Kaggle: torch is already installed with CUDA).
install:
	$(PYTHON) -m pip install -r requirements.txt

## Local machine without a GPU: CPU-only torch wheel + pinned deps.
install-cpu:
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt

## Resolve the config and render the shared prompt with the real model's tokenizer.
check:
	$(PYTHON) config.py
	$(PYTHON) prompts.py

## Generate ~4,000 synthetic tickets, dedupe, split -> data/processed/
data:
	$(PYTHON) data_prep.py

## Same, and push the dataset to the Hub (needs HF_TOKEN or `hf auth login`).
data-push:
	$(PYTHON) data_prep.py --push

## "Before" rows of the results table: TF-IDF, base model zero-shot and few-shot.
baseline:
	$(PYTHON) baseline_classic.py
	$(PYTHON) eval.py --mode zero-shot --display-name "Base zero-shot"
	$(PYTHON) eval.py --mode few-shot --display-name "Base few-shot (k=3)"

## QLoRA fine-tuning (CUDA GPU). Logs: tensorboard --logdir outputs/tensorboard
train:
	$(PYTHON) train_lora.py

## "After" row: base model + the trained LoRA adapter (override with ADAPTER=...).
eval:
	$(PYTHON) eval.py --adapter $(ADAPTER) --display-name "LoRA fine-tuned"

## Build model cards, merged model and Space locally (outputs/hub/), push nothing. Review before publishing.
publish-dry:
	$(PYTHON) merge_and_publish.py --dry-run --merge --space

## Optional: score the merged bf16 weights directly (after publish-dry); the merged card then shows them.
eval-merged:
	$(PYTHON) eval.py --model outputs/hub/merged --run-name lora_merged --display-name "Merged model (bf16 weights)"

## Push the adapter, the merged model (with generated model cards) and the demo Space.
## Needs HF_TOKEN or `hf auth login`. Drop --space to skip the demo.
publish:
	$(PYTHON) merge_and_publish.py --merge --space

## Refresh the results table in README.md from results/*.json.
readme:
	$(PYTHON) merge_and_publish.py --readme

## End-to-end CPU smoke test: tiny model, 50 examples, 10 steps, no Hub push.
smoke:
	$(PYTHON) config.py --smoke
	$(PYTHON) prompts.py --smoke
	$(PYTHON) data_prep.py --smoke --show 0
	$(PYTHON) baseline_classic.py --smoke
	$(PYTHON) eval.py --smoke --mode zero-shot --display-name "Base zero-shot"
	$(PYTHON) eval.py --smoke --mode few-shot --display-name "Base few-shot (k=3)"
	$(PYTHON) train_lora.py --smoke
	$(PYTHON) eval.py --smoke --adapter outputs/smoke/lora-adapter --display-name "LoRA fine-tuned"
	$(PYTHON) merge_and_publish.py --smoke --dry-run --merge --space --readme
	$(PYTHON) eval.py --smoke --model outputs/smoke/hub/merged --run-name lora_merged --display-name "Merged model (bf16 weights)"
	$(PYTHON) app.py --smoke
