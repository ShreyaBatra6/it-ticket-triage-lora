---
{{ front_matter }}---

# {{ title }}

{% if smoke %}
> ⚠️ **Smoke-test artifact.** Produced by the CPU pipeline test with a tiny model, 50 examples and 10
> training steps. The numbers below are meaningless; do not use this model.

{% endif %}
## Summary

{% set method = "QLoRA" if (not train or train.training.load_in_4bit) else "LoRA" %}
{% if variant == "adapter" %}
A **LoRA adapter** (trained with {{ method }}) for [`{{ base_model_id }}`](https://huggingface.co/{{ base_model_id }})
{% else %}
[`{{ base_model_id }}`](https://huggingface.co/{{ base_model_id }}) with a {{ method }}-trained LoRA adapter
**merged into the weights** (no `peft` needed at inference)
{% endif %}
that triages IT helpdesk tickets. Given a ticket's subject and body it replies with strict JSON:

```json
{"category": "<one of {{ labels.category_names | length }} categories>", "priority": "<Low | Medium | High | Critical>"}
```

{% if headline %}
{{ headline }}
{% endif %}

{% if variant == "adapter" %}
The merged full model is at [`{{ merged_repo }}`](https://huggingface.co/{{ merged_repo }}).
{% else %}
The standalone adapter is at [`{{ adapter_repo }}`](https://huggingface.co/{{ adapter_repo }}).
{% endif %}
{% if demo_url %}
Try it in the demo: [{{ demo_url }}]({{ demo_url }}).
{% endif %}
{% if github_url %}
Code (data generation, training, evaluation, this card): [{{ github_url }}]({{ github_url }}).
{% endif %}

## Intended use

- **Intended:** a first-pass triage suggestion for IT service desks: routing a new ticket to a queue
  and proposing a priority, with a human agent confirming. Also a reference for small-model QLoRA
  fine-tuning with a fair before/after evaluation.
- **Out of scope:** making final priority or SLA decisions without review; security incident
  response decisions; tickets in languages other than English; any input that is not an IT
  support ticket (the model always answers with one of the fixed labels, even when none fits).

## Label definitions

| Category | Definition |
|---|---|
{% for name, desc in labels.categories.items() %}
| {{ name }} | {{ desc }} |
{% endfor %}

| Priority | Definition |
|---|---|
{% for name, desc in labels.priorities.items() %}
| {{ name }} | {{ desc }} |
{% endfor %}

## Training data

> **The training data is fully synthetic.** No real tickets, users or companies are included.

[`{{ dataset_repo }}`](https://huggingface.co/datasets/{{ dataset_repo }}): tickets generated from
hand-written templates with randomized slots, writing styles (formal, casual, terse, frustrated,
mobile, non-native), typos, cross-category vocabulary, optional unrelated side requests, and
realistic label imbalance. Priority is expressed through *who is affected*, *whether they can still
work* and *any deadline*; 3% of priorities were moved one level to mimic annotator disagreement.
{% if data %}

| Split | Tickets |
|---|---:|
| train | {{ data.split_sizes.train }} |
| validation | {{ data.split_sizes.val }} |
| test | {{ data.split_sizes.test }} |

Split method: {{ data.stratified_by }}.
{% if "template" in data.stratified_by %}
Validation and test tickets only use problem templates that **never appear in training**, so the
scores below measure generalisation to unseen phrasing, not memorisation.
{% endif %}
A MinHash check removed validation/test tickets that were near-duplicates of training tickets
(Jaccard ≥ {{ data.near_duplicates.threshold }}).
{% endif %}

## Training procedure

{% if train %}
| Setting | Value |
|---|---|
| Method | {{ train.method }} |
| Base model | `{{ base_model_id }}` |
| LoRA rank / alpha / dropout | {{ train.lora.r }} / {{ train.lora.alpha }} / {{ train.lora.dropout }} |
| Target modules | {{ train.lora.target_modules | join(", ") }} |
| Trainable parameters | {{ "{:,}".format(train.parameters.trainable) }} ({{ train.parameters.trainable_pct }}% of {{ "{:,}".format(train.parameters.total) }}) |
| Loss | assistant answer only (prompt tokens masked) |
| Epochs / optimizer steps | {{ train.results.epochs }} / {{ train.results.global_steps }} |
| Learning rate / schedule | {{ train.training.learning_rate }}, {{ train.training.lr_scheduler_type }}, {{ (train.training.warmup_ratio * 100) | round(0) | int }}% warmup |
| Batch size (per device × grad. accumulation) | {{ train.training.per_device_train_batch_size }} × {{ train.training.gradient_accumulation_steps }} = {{ train.training.effective_batch_size }} |
| Max sequence length | {{ train.training.max_length }} |
| Precision / optimizer | {{ train.training.precision }} / {{ train.training.optimizer }} |
| Gradient checkpointing | {{ train.training.gradient_checkpointing }} |
| Checkpoint selection | lowest validation loss (step {{ train.results.best_checkpoint_step }}, loss {{ "%.4f" | format(train.results.best_eval_loss) }}) |
| Seed | {{ train.training.seed }} |
| Hardware | {{ hardware }} |
| Training time | {{ (train.results.train_runtime_s / 60) | round(1) }} min |
| Libraries | {{ libraries }} |

![Training and validation loss](images/train_loss.png)
{% else %}
Training summary not found (run `train_lora.py` before publishing).
{% endif %}

## Results: before vs. after fine-tuning

Test split ({{ n_test }} tickets). Every LLM row uses the **same prompt** and greedy decoding. An answer
counts only if the *entire* output is valid JSON with known labels; nothing is repaired.

| Model | Cat Acc | Cat F1 | Prio Acc | Prio F1 | Exact Match | Valid JSON % |
|---|---:|---:|---:|---:|---:|---:|
{% for row in table %}
| {{ row.label }} | {{ row.cells | join(" | ") }} |
{% endfor %}

Acc = accuracy, F1 = macro-F1, Exact Match = both labels correct. Best value per column in bold.
TF-IDF is a classifier, so "valid JSON" does not apply to it.
{% if variant == "merged" %}
{% if merged_measured %}
The "{{ merged_label }}" row evaluates this repository's weights directly. They are stored in bf16,
so a few borderline predictions can differ from the base + adapter row.
{% else %}
These scores were measured with the adapter on top of the base model. This repository stores the
merged weights in bf16, which can change a few borderline predictions; evaluate it directly with
`python eval.py --model {{ merged_repo }} --run-name lora_merged`.
{% endif %}
{% endif %}
{% if lenient_note %}
{{ lenient_note }}
{% endif %}
{% if latency_rows %}

| Model | Avg. latency per ticket | Batch size | Device |
|---|---:|---:|---|
{% for row in latency_rows %}
| {{ row.label }} | {{ row.latency }} | {{ row.batch_size }} | {{ row.device }} |
{% endfor %}
{% endif %}
{% if images %}

### Confusion matrices (fine-tuned model)

{% for image in images %}
![{{ image.title }}]({{ image.path }})
{% endfor %}
{% endif %}
{% if examples %}

## Example inputs and outputs

{% for ex in examples %}
**{{ "✅" if ex.correct else "❌" }} Expected `{{ ex.gold }}`**

```text
Subject: {{ ex.subject }}

{{ ex.body }}
```

Model output: `{{ ex.output }}`

{% endfor %}
{% endif %}

## Limitations and bias

- **Synthetic data shift.** The model was trained and tested on templated synthetic tickets. Real
  tickets are longer, messier, multilingual, contain signatures, threads and screenshots, and follow
  each organisation's own category scheme. Expect lower accuracy on real data; re-evaluate (and
  ideally fine-tune) on your own labelled tickets before relying on it.
- **Priority is subjective.** Different teams would label the same ticket differently, and users
  often overstate urgency ("URGENT" on routine requests). The model only sees the text, cannot
  verify claims, and inherits the priority conventions encoded in the synthetic generator.
- **Fixed label set.** It always answers with one of the {{ labels.category_names | length }} categories,
  including for tickets that fit none of them.
- **Ticket text is untrusted input.** A ticket can contain instructions aimed at the model; keep a
  human in the loop for anything consequential.
- **English only**, office-IT context.

## How to use

The model expects **exactly** the prompt it was trained with: the system prompt below plus the
ticket as `Subject: ...` followed by a blank line and the body.

```python
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
{% if variant == "adapter" %}
from peft import PeftModel

base_id = "{{ base_model_id }}"
adapter_id = "{{ adapter_repo }}"
tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
{% else %}

model_id = "{{ merged_repo }}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
{% endif %}

SYSTEM_PROMPT = """{{ system_prompt }}"""

ticket = "Subject: VPN keeps disconnecting\n\nSince this morning the VPN drops every 5 minutes and I can't stay on my calls."
messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ticket}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
output = model.generate(**inputs, max_new_tokens=48, do_sample=False)
answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(json.loads(answer))  # e.g. {'category': 'Network', 'priority': 'High'}
```

## Citation

```bibtex
@misc{ {{- citation_key -}} ,
  title        = { {{- title -}} },
  author       = { {{- author -}} },
  year         = { {{- year -}} },
  howpublished = {\url{https://huggingface.co/{{ repo_id }}}},
  note         = {QLoRA fine-tune of {{ base_model_id }} for IT ticket triage}
}
```

Built on QLoRA ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) and LoRA
([Hu et al., 2021](https://arxiv.org/abs/2106.09685)).
