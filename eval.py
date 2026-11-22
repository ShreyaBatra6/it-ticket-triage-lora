"""Evaluate ANY causal LM on the ticket test set: base, base + LoRA adapter, or a merged model.

Every model gets the same prompt (prompts.py), greedy decoding, and strict JSON parsing
(metrics.py). Few-shot mode adds k solved examples drawn from the TRAIN split only.

Usage::

    python eval.py --mode zero-shot                          # base model -> results/base_zero_shot.json
    python eval.py --mode few-shot                           # base model, k=3 -> results/base_few_shot.json
    python eval.py --adapter outputs/lora-adapter            # base + LoRA -> results/lora_finetuned.json
    python eval.py --model outputs/merged-model --run-name lora_merged
    python eval.py --smoke --mode zero-shot                  # tiny model on 5 CPU examples
"""

from __future__ import annotations

import argparse

from config import load_config
from metrics import finalize_run, print_summary
from modeling import generate, load_model, load_tokenizer
from prompts import build_prompt, select_few_shot_examples
from utils import get_compute_dtype, get_device, load_split, set_seed


def default_run_name(adapter: str | None, mode: str) -> str:
    if adapter:
        return "lora_finetuned" if mode == "zero-shot" else "lora_few_shot"
    return f"base_{mode.replace('-', '_')}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="Tiny model + smoke dataset, CPU friendly.")
    p.add_argument("--model", default=None, help="Model id or local path (default: base model from config).")
    p.add_argument("--adapter", default=None, help="LoRA adapter path or Hub id to load on top of --model.")
    p.add_argument("--mode", choices=("zero-shot", "few-shot"), default="zero-shot")
    p.add_argument("--k", type=int, default=None, help="Few-shot examples (default: prompt.few_shot_k).")
    p.add_argument("--split", choices=("test", "val"), default="test")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    p.add_argument("--batch-size", type=int, default=None, help="Default: eval.batch_size from config.")
    p.add_argument("--load-in-4bit", action="store_true", help="Load the model in 4-bit (CUDA only).")
    p.add_argument("--run-name", default=None, help="Results file name (default derived from model/mode).")
    p.add_argument("--display-name", default=None, help="Label used in plot titles.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(smoke=args.smoke)
    set_seed(cfg.seed)

    model_id = args.model or cfg.model.base_model_id
    run_name = args.run_name or default_run_name(args.adapter, args.mode)
    batch_size = args.batch_size or cfg.eval.batch_size
    k = (args.k if args.k is not None else cfg.prompt.few_shot_k) if args.mode == "few-shot" else 0

    rows = load_split(cfg, args.split)[: args.limit]
    few_shot = select_few_shot_examples(load_split(cfg, "train"), k, cfg.seed) if k else []

    print(f"Loading {model_id}" + (f" + adapter {args.adapter}" if args.adapter else "") + f" on {get_device()}")
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, adapter=args.adapter, load_in_4bit=args.load_in_4bit)

    prompts = [build_prompt(tokenizer, cfg, r["subject"], r["body"], few_shot) for r in rows]
    print(f"Generating for {len(prompts)} {args.split} tickets ({args.mode}, batch size {batch_size})...")
    outputs, seconds = generate(model, tokenizer, prompts, cfg.prompt.max_new_tokens, batch_size)

    meta = {
        "display_name": args.display_name or run_name.replace("_", " "),
        "split": args.split,
        "model": {"model_id": model_id, "adapter": args.adapter, "load_in_4bit": args.load_in_4bit,
                  "dtype": str(get_compute_dtype()).replace("torch.", "")},
        "prompting": {"mode": args.mode, "k": k, "few_shot_ids": [r["id"] for r in few_shot],
                      "decoding": "greedy", "max_new_tokens": cfg.prompt.max_new_tokens,
                      "example_prompt": prompts[0] if prompts else None},
        "timing": {"avg_latency_ms": 1000 * seconds / max(len(rows), 1), "total_generation_s": seconds,
                   "batch_size": batch_size, "device": get_device()},
    }
    results = finalize_run(cfg, run_name, rows, outputs, meta, include_invalid=True)
    print_summary(results)
    print(f"Saved {cfg.paths.results_dir / (run_name + '.json')}")


if __name__ == "__main__":
    main()
