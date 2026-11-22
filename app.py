"""Gradio demo: paste an IT ticket, get {"category": ..., "priority": ...} back.

Runs locally or as a Hugging Face Space (see space/README.md). Model selection, in order:

* ``MODEL_ID`` env var: a full model (e.g. the merged repo), loaded directly;
* otherwise the base model from config.yaml + ``ADAPTER_ID`` (default: the published adapter repo),
  merged in memory at start-up so inference runs at full-model speed without peft overhead.

The prompt and the strict JSON parsing are the same code used for evaluation.

Usage::

    python app.py            # launch the UI (http://127.0.0.1:7860)
    python app.py --smoke    # tiny model + local smoke adapter: build the UI, run one ticket, exit
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import gradio as gr

from config import Config, load_config
from metrics import parse_output
from modeling import generate, load_model, load_tokenizer
from prompts import build_prompt
from utils import get_device

EXAMPLES = [
    ["VPN keeps disconnecting",
     "Since this morning the VPN drops every 5 minutes and I can't stay on my client calls. Home wifi is fine."],
    ["URGENT: invoices@ mailbox not receiving",
     "The invoices@ mailbox stopped receiving around 9am and customers send their payment confirmations there."],
    ["Request: second monitor",
     "Hi, could I get a second monitor for my desk on the 3rd floor? No rush, whenever you have one spare."],
    ["Clicked a link in a weird email",
     "I typed my password into a page from an email that looked like our login page, then realised the address "
     "was wrong. I have admin rights on the finance system."],
]


class Triage:
    """Loads the model once and answers tickets with the shared prompt."""

    def __init__(self, cfg: Config, model_id: str, adapter: str | None) -> None:
        self.cfg = cfg
        self.label = model_id + (f" + {adapter}" if adapter else "")
        self.tokenizer = load_tokenizer(model_id)
        model = load_model(model_id, adapter=adapter)
        # Fold the adapter into the weights once: same predictions, faster generation.
        self.model = model.merge_and_unload() if adapter else model


    # categorize

    def __call__(self, subject: str, body: str) -> tuple[dict[str, Any], str, str]:
        if not (subject or "").strip() and not (body or "").strip():
            return {"error": "Enter a subject and/or body."}, "", ""
        prompt = build_prompt(self.tokenizer, self.cfg, subject or "", body or "")
        start = time.perf_counter()
        (raw,), _ = generate(self.model, self.tokenizer, [prompt], self.cfg.prompt.max_new_tokens, batch_size=1)
        ms = 1000 * (time.perf_counter() - start)
        parsed = parse_output(raw)
        known = parsed.category in self.cfg.labels.categories and parsed.priority in self.cfg.labels.priorities
        if parsed.valid_json and known:
            return {"category": parsed.category, "priority": parsed.priority}, raw, f"Valid JSON · {ms:.0f} ms"
        reason = parsed.error or "label outside the allowed set"
        return {"error": f"Model output rejected: {reason}"}, raw, f"Invalid output · {ms:.0f} ms"

# model selection 
def model_source(cfg: Config) -> tuple[str, str | None]:
    """(model id, adapter id or None) from env vars, falling back to config.yaml."""
    if os.environ.get("MODEL_ID"):
        return os.environ["MODEL_ID"], None
    if cfg.smoke:  # local smoke adapter from train_lora.py --smoke, if it exists
        adapter = cfg.paths.adapter_dir
        return cfg.model.base_model_id, str(adapter) if (adapter / "adapter_config.json").exists() else None
    return cfg.model.base_model_id, os.environ.get("ADAPTER_ID", cfg.hub.adapter_repo)

# gradio web page
def build_ui(triage: Triage, cfg: Config) -> gr.Blocks:
    categories = ", ".join(cfg.labels.category_names)
    with gr.Blocks(title="IT Ticket Triage") as demo:
        gr.Markdown(
            "# IT ticket triage\n"
            f"Fine-tuned **{cfg.model.short_name}** (QLoRA) assigns a category ({categories}) and a priority "
            f"({', '.join(cfg.labels.priority_names)}). Trained on **synthetic** tickets: a demo, not a production "
            "triage system."
        )
        with gr.Row():
            with gr.Column():
                subject = gr.Textbox(label="Subject", placeholder="e.g. Outlook not opening")
                body = gr.Textbox(label="Ticket text", lines=8, placeholder="Describe the problem...")
                submit = gr.Button("Triage", variant="primary")
            with gr.Column():
                result = gr.JSON(label="Prediction")
                status = gr.Markdown()
                raw = gr.Textbox(label="Raw model output", lines=2, max_lines=4)
        gr.Examples(examples=EXAMPLES, inputs=[subject, body])
        gr.Markdown(f"<sub>Model: `{triage.label}` · device: {get_device()}</sub>")
        submit.click(triage, inputs=[subject, body], outputs=[result, raw, status], api_name="triage")
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Tiny model: build the UI, triage one example, exit.")
    args = parser.parse_args()
    cfg = load_config(smoke=args.smoke)

    model_id, adapter = model_source(cfg)
    print(f"Loading {model_id}" + (f" + adapter {adapter}" if adapter else "") + f" on {get_device()} ...")
    triage = Triage(cfg, model_id, adapter)
    demo = build_ui(triage, cfg)

    if args.smoke:
        prediction, raw, status = triage(*EXAMPLES[0])
        print(f"UI built OK. Example -> {prediction} | raw: {raw!r} | {status}")
        return
    demo.queue().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))


if __name__ == "__main__":
    main()
