---
title: IT Ticket Triage
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Classify IT tickets into category + priority (QLoRA)
models:
  - __BASE_MODEL__
  - __ADAPTER_REPO__
datasets:
  - __DATASET_REPO__
---

# IT Ticket Triage demo

Paste an IT helpdesk ticket and get back strict JSON: `{"category": ..., "priority": ...}`.

- Model: [`__BASE_MODEL__`](https://huggingface.co/__BASE_MODEL__) + the QLoRA adapter
  [`__ADAPTER_REPO__`](https://huggingface.co/__ADAPTER_REPO__) (merged in memory at start-up).
- Trained on the **synthetic** dataset [`__DATASET_REPO__`](https://huggingface.co/datasets/__DATASET_REPO__);
  this is a demo, not a production triage system.

**Speed:** on the free CPU hardware the model loads in a few minutes and each ticket takes tens of
seconds. Upgrade the Space hardware to a small GPU (e.g. T4) for sub-second answers.

**Configuration (Space variables, optional):**

| Variable | Effect |
|---|---|
| `MODEL_ID` | load this full model instead, e.g. the merged repo `__MERGED_REPO__` |
| `ADAPTER_ID` | use a different LoRA adapter on the base model |
