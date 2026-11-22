"""Build the model card(s) and publish to the Hugging Face Hub.

1. Render ``model_card_template.md`` from ``results/*.json`` (before/after table, confusion
   matrices, examples) and ``results/train_summary.json`` (hyperparameters, hardware).
2. Stage the adapter + card in ``outputs/hub/adapter`` and push it to
   ``<user>/<model-short>-it-ticket-triage-lora``.
3. ``--merge``: merge the adapter into the base weights (``merge_and_unload``, bf16, on CPU), save
   it to ``outputs/hub/merged`` with its own card, and push it to ``<user>/<model-short>-it-ticket-triage``.

4. ``--space``: stage app.py + its modules + space/README.md + space/requirements.txt in
   ``outputs/hub/space`` and push them as a Gradio Space (after the model repos exist).
5. ``--readme``: refresh the results block of the GitHub README.md from ``results/*.json``.

``--dry-run`` does everything locally (cards, staging folders, merged weights) and pushes nothing.

Usage::

    python merge_and_publish.py --dry-run --merge --space   # inspect outputs/hub/*/README.md first
    python merge_and_publish.py --merge --space             # push adapter, merged model and demo Space
    python merge_and_publish.py --readme                    # only refresh README.md results
    python merge_and_publish.py --smoke --dry-run --merge --space --readme
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import REPO_ROOT, Config, load_config
from prompts import build_system_prompt, format_target
from utils import get_hf_token, load_json, load_jsonl, split_path

TEMPLATE_PATH = REPO_ROOT / "model_card_template.md"
LORA_RUN = "lora_finetuned"
MERGED_RUN = "lora_merged"  # optional: eval.py --model outputs/hub/merged --run-name lora_merged
MERGED_LABEL = "Merged model (bf16 weights)"
SKIP_ADAPTER_FILES = {"README.md", "training_args.bin"}  # PEFT's stub card; a pickle we don't need to publish
COLUMNS = (  # (metric key, kind) in table order: Cat Acc, Cat F1, Prio Acc, Prio F1, Exact Match, Valid JSON %
    ("category_accuracy", "pct"), ("category_macro_f1", "f1"), ("priority_accuracy", "pct"),
    ("priority_macro_f1", "f1"), ("exact_match", "pct"), ("valid_json_rate", "pct"),
)


# =========================================================================== gather inputs
def load_results(cfg: Config) -> dict[str, dict[str, Any] | None]:
    """results/<run_name>.json for every row of the report table (None when that run is missing)."""
    out = {}
    for row in cfg.eval.report_rows:
        path = cfg.paths.results_dir / f"{row.run_name}.json"
        out[row.run_name] = load_json(path) if path.exists() else None
        if out[row.run_name] is None:
            print(f"[card] missing {path.name}: its table row will show '-'")
    return out


def _optional_json(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.exists() else None


def is_llm_run(result: dict[str, Any]) -> bool:
    return "prompting" in result  # the classic baseline has no prompt


def pick_examples(cfg: Config, n_correct: int = 3, n_wrong: int = 1) -> list[dict[str, Any]]:
    """A few real test tickets with the fine-tuned model's raw output: mostly correct, one mistake."""
    preds_path = cfg.paths.results_dir / f"{LORA_RUN}_predictions.jsonl"
    if not preds_path.exists() or not split_path(cfg, "test").exists():
        return []
    tickets = {r["id"]: r for r in load_jsonl(split_path(cfg, "test"))}
    correct, wrong, seen_categories = [], [], set()
    for p in load_jsonl(preds_path):
        ticket = tickets.get(p["id"])
        if ticket is None:
            continue
        ok = p["category"] == p["gold_category"] and p["priority"] == p["gold_priority"]
        example = {
            "subject": ticket["subject"], "body": ticket["body"][:400].rstrip(),
            "gold": format_target(p["gold_category"], p["gold_priority"]),
            "output": p["raw_output"].strip(), "correct": ok,
        }
        if ok and len(correct) < n_correct and ticket["category"] not in seen_categories:
            correct.append(example)
            seen_categories.add(ticket["category"])
        elif not ok and len(wrong) < n_wrong:
            wrong.append(example)
    return correct + wrong


# =========================================================================== card pieces
def _fmt(value: float, kind: str) -> str:
    return f"{100 * value:.1f}%" if kind == "pct" else f"{value:.3f}"


def results_table(entries: list[tuple[str, dict[str, Any] | None]]) -> list[dict[str, Any]]:
    """(label, results) pairs -> rows of formatted cells; best value per column in bold.
    Valid JSON is n/a for the TF-IDF classifier."""
    best: dict[str, float] = {}
    for key, _ in COLUMNS:
        values = [r["metrics"][key] for _, r in entries
                  if r and not (key == "valid_json_rate" and not is_llm_run(r))]
        best[key] = max(values) if values else float("nan")

    rows = []
    for label, r in entries:
        cells = []
        for key, kind in COLUMNS:
            if r is None:
                cells.append("-")
            elif key == "valid_json_rate" and not is_llm_run(r):
                cells.append("n/a")
            else:
                text = _fmt(r["metrics"][key], kind)
                cells.append(f"**{text}**" if r["metrics"][key] == best[key] else text)
        rows.append({"label": label, "cells": cells})
    return rows


def latency_rows(entries: list[tuple[str, dict[str, Any] | None]]) -> list[dict[str, str]]:
    rows = []
    for label, r in entries:
        if not r or "avg_latency_ms" not in r["metrics"]:
            continue
        gpu = (r.get("environment") or {}).get("gpu")
        device = gpu["name"] if gpu and r["metrics"].get("device") == "cuda" else r["metrics"].get("device", "?")
        rows.append({"label": label, "latency": f"{r['metrics']['avg_latency_ms']:.1f} ms",
                     "batch_size": str(r["metrics"].get("batch_size", "?")), "device": device})
    return rows


def headline(results: dict[str, dict[str, Any] | None]) -> str:
    lora, base, tfidf = results.get(LORA_RUN), results.get("base_zero_shot"), results.get("tfidf_lr")
    if not (lora and base):
        return ""
    lm, bm = lora["metrics"], base["metrics"]
    text = (f"**Result:** fine-tuning raises exact match (both labels right) from {100 * bm['exact_match']:.1f}% "
            f"(base model, zero-shot) to **{100 * lm['exact_match']:.1f}%**, and valid-JSON answers from "
            f"{100 * bm['valid_json_rate']:.1f}% to {100 * lm['valid_json_rate']:.1f}%")
    if tfidf:
        text += f"; a TF-IDF + logistic regression baseline reaches {100 * tfidf['metrics']['exact_match']:.1f}%"
    return text + "."


def lenient_note(results: dict[str, dict[str, Any] | None]) -> str:
    """Show how much of the base model's failure is formatting, using the separate lenient score."""
    parts = []
    for run, label in (("base_zero_shot", "zero-shot"), ("base_few_shot", "few-shot")):
        r = results.get(run)
        if r and r["metrics"]["lenient"]["exact_match"] - r["metrics"]["exact_match"] > 0.01:
            parts.append(f"{label} {100 * r['metrics']['lenient']['exact_match']:.1f}%")
    if not parts:
        return ""
    return ("If the JSON were instead extracted from anywhere in the output (lenient parsing, not used above), "
            f"the base model's exact match would be {' and '.join(parts)}: part of its failure is formatting.")


def front_matter(cfg: Config, variant: str, repo_id: str, lora: dict[str, Any] | None) -> str:
    """YAML metadata (built in Python so it is always valid YAML)."""
    tags = ["lora", "peft", "qlora", "text-classification", "it-support", "ticket-triage", "synthetic-data"]
    if variant == "merged":
        tags.append("merged")
    if cfg.smoke:
        tags.append("smoke-test")
    meta: dict[str, Any] = {
        "base_model": cfg.model.base_model_id,
        "base_model_relation": "adapter" if variant == "adapter" else "finetune",
        "library_name": "peft" if variant == "adapter" else "transformers",
        "license": cfg.model.license,
        "datasets": [cfg.hub.dataset_repo],
        "language": ["en"],
        "pipeline_tag": "text-generation",  # it generates the JSON answer; "text-classification" is a tag
        "tags": tags,
    }
    if lora:
        m = lora["metrics"]
        metric_specs = [("accuracy", "category_accuracy", "Category accuracy"),
                        ("f1", "category_macro_f1", "Category macro-F1"),
                        ("accuracy", "priority_accuracy", "Priority accuracy"),
                        ("f1", "priority_macro_f1", "Priority macro-F1"),
                        ("exact_match", "exact_match", "Exact match (both labels)"),
                        ("valid_json", "valid_json_rate", "Valid JSON rate")]
        meta["model-index"] = [{
            "name": repo_id.split("/")[-1],
            "results": [{
                "task": {"type": "text-classification", "name": "IT ticket triage"},
                "dataset": {"type": cfg.hub.dataset_repo, "name": "IT Ticket Triage (synthetic)", "split": "test"},
                "metrics": [{"type": t, "value": round(m[k], 4), "name": n} for t, k, n in metric_specs],
            }],
        }]
    return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)


def render_card(cfg: Config, variant: str, demo_url: str | None) -> str:
    """Fill model_card_template.md for the "adapter" or "merged" repo."""
    import jinja2

    results = load_results(cfg)
    train = _optional_json(cfg.paths.results_dir / "train_summary.json")
    data = _optional_json(cfg.paths.data_dir / "stats.json")
    repo_id = cfg.hub.adapter_repo if variant == "adapter" else cfg.hub.merged_repo

    entries = [(spec.label, results[spec.run_name]) for spec in cfg.eval.report_rows]
    # The merged repo stores bf16 weights; if they were evaluated directly, show that row and use it
    # for the card metadata, otherwise fall back to the base + adapter scores.
    merged_result = _optional_json(cfg.paths.results_dir / f"{MERGED_RUN}.json") if variant == "merged" else None
    if merged_result:
        entries.append((MERGED_LABEL, merged_result))
    index_result = merged_result or results.get(LORA_RUN)

    hardware, libraries = "-", "-"
    if train:
        env = train["environment"]
        gpu = env.get("gpu")
        hardware = f"1x {gpu['name']} ({gpu['memory_gb']} GB)" if gpu else f"CPU ({env.get('processor')})"
        libs = env["libraries"]
        libraries = ", ".join(f"{k} {libs[k]}" for k in ("transformers", "peft", "trl", "bitsandbytes", "torch")
                              if libs.get(k))
    n_test = next((r["metrics"]["n"] for r in results.values() if r), "?")

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(REPO_ROOT)), trim_blocks=True, lstrip_blocks=True,
                             undefined=jinja2.StrictUndefined, keep_trailing_newline=True, autoescape=False)
    title = f"{cfg.model.short_name} IT ticket triage ({'LoRA adapter' if variant == 'adapter' else 'merged'})"
    return env.get_template(TEMPLATE_PATH.name).render(
        front_matter=front_matter(cfg, variant, repo_id, index_result),
        merged_measured=merged_result is not None, merged_label=MERGED_LABEL,
        title=title, smoke=cfg.smoke, variant=variant, repo_id=repo_id,
        adapter_repo=cfg.hub.adapter_repo, merged_repo=cfg.hub.merged_repo,
        base_model_id=cfg.model.base_model_id, dataset_repo=cfg.hub.dataset_repo,
        demo_url=demo_url, github_url=cfg.github_url, labels=cfg.labels,
        system_prompt=build_system_prompt(cfg.labels),
        headline=headline(results), table=results_table(entries), lenient_note=lenient_note(results),
        latency_rows=latency_rows(entries), n_test=n_test,
        images=card_images(cfg), examples=pick_examples(cfg),
        train=train, data=data, hardware=hardware, libraries=libraries,
        author=cfg.hf_username, year=datetime.now(timezone.utc).year,
        citation_key=repo_id.split("/")[-1].replace("-", "_").replace(".", "_"),
    )


def card_images(cfg: Config) -> list[dict[str, str]]:
    """Images referenced by the card (paths inside the repo); only those that exist."""
    wanted = [(f"{LORA_RUN}_cm_category.png", "Category confusion matrix"),
              (f"{LORA_RUN}_cm_priority.png", "Priority confusion matrix")]
    return [{"title": title, "path": f"images/{name}"} for name, title in wanted
            if (cfg.paths.results_dir / name).exists()]


def write_card(cfg: Config, folder: Path, card: str) -> None:
    """README.md + images/ (confusion matrices, loss curve) into a staging folder."""
    from huggingface_hub import ModelCard

    ModelCard(card)  # parses the YAML front matter; raises if it is malformed
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text(card, encoding="utf-8")
    for name in (f"{LORA_RUN}_cm_category.png", f"{LORA_RUN}_cm_priority.png", "train_loss.png"):
        src = cfg.paths.results_dir / name
        if src.exists():
            shutil.copy2(src, folder / "images" / name)


# =========================================================================== staging + merge + push
def stage_adapter(cfg: Config, card: str) -> Path:
    adapter_dir, stage = cfg.paths.adapter_dir, cfg.paths.hub_adapter_dir
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No adapter in {adapter_dir}. Run train_lora.py{' --smoke' if cfg.smoke else ''} first.")
    shutil.rmtree(stage, ignore_errors=True)  # generated folder: rebuild from scratch
    stage.mkdir(parents=True)
    for f in adapter_dir.iterdir():
        if f.is_file() and f.name not in SKIP_ADAPTER_FILES:
            shutil.copy2(f, stage / f.name)
    write_card(cfg, stage, card)
    return stage


def merge_adapter(cfg: Config) -> Path:
    """Fold the LoRA weights into the base model and save a standalone checkpoint (bf16, on CPU)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from modeling import load_tokenizer

    out = cfg.paths.merged_dir
    shutil.rmtree(out, ignore_errors=True)
    # Merge on CPU in bf16 (the base model's native dtype): no GPU needed and no fp16 conversion.
    base = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(cfg.paths.adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out))
    load_tokenizer(cfg.model.base_model_id).save_pretrained(str(out))
    return out


def push_folder(cfg: Config, folder: Path, repo_id: str, token: str | None, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=cfg.hub.private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message=message)
    print(f"Pushed https://huggingface.co/{repo_id}")


SPACE_FILES = ("app.py", "config.py", "config.yaml", "prompts.py", "modeling.py", "utils.py", "metrics.py")


def stage_space(cfg: Config) -> Path:
    """Copy the demo app and the modules it imports, plus the Space README/requirements."""
    stage = cfg.paths.output_dir / "hub" / "space"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for name in SPACE_FILES:
        shutil.copy2(REPO_ROOT / name, stage / name)
    shutil.copy2(REPO_ROOT / "space" / "requirements.txt", stage / "requirements.txt")
    readme = (REPO_ROOT / "space" / "README.md").read_text(encoding="utf-8")
    for token, value in {"__BASE_MODEL__": cfg.model.base_model_id, "__ADAPTER_REPO__": cfg.hub.adapter_repo,
                         "__MERGED_REPO__": cfg.hub.merged_repo, "__DATASET_REPO__": cfg.hub.dataset_repo}.items():
        readme = readme.replace(token, value)
    (stage / "README.md").write_text(readme, encoding="utf-8")
    return stage


def push_space(cfg: Config, folder: Path, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(cfg.hub.space_repo, repo_type="space", space_sdk="gradio", private=cfg.hub.private, exist_ok=True)
    api.upload_folder(repo_id=cfg.hub.space_repo, repo_type="space", folder_path=str(folder),
                      commit_message="Deploy Gradio demo")
    print(f"Pushed https://huggingface.co/spaces/{cfg.hub.space_repo} (it builds for a few minutes)")


README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _picture(path: str, alt: str, width: int) -> str:
    """Light/dark image pair for GitHub (the *_dark.png twin is picked in dark mode)."""
    dark = path.replace(".png", "_dark.png")
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{dark}">\n'
            f'  <img alt="{alt}" src="{path}" width="{width}">\n</picture>')


def update_readme(cfg: Config) -> Path:
    """Rewrite the block between the RESULTS markers in README.md from results/*.json.

    In smoke mode a preview is written to outputs/smoke/README.md instead of touching the real README.
    """
    results = load_results(cfg)
    rows = results_table([(spec.label, results[spec.run_name]) for spec in cfg.eval.report_rows])
    lines = ["| Model | Cat Acc | Cat F1 | Prio Acc | Prio F1 | Exact Match | Valid JSON % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['label']} | {' | '.join(row['cells'])} |" for row in rows]
    n_test = next((r["metrics"]["n"] for r in results.values() if r), "?")
    block = [f"Test split: {n_test} tickets whose problem templates never appear in training. "
             "Strict JSON parsing; best value per column in bold. `-` = not run yet.", "", *lines]
    note = lenient_note(results)
    if note:
        block += ["", note]
    images = [(f"results/{LORA_RUN}_cm_category.png", "Category confusion matrix (fine-tuned)", 440),
              (f"results/{LORA_RUN}_cm_priority.png", "Priority confusion matrix (fine-tuned)", 360),
              ("results/train_loss.png", "Training and validation loss", 560)]
    pictures = [_picture(p, alt, w) for p, alt, w in images if (cfg.paths.results_dir / Path(p).name).exists()]
    if pictures and not cfg.smoke:
        block += ["", *pictures]
    block += ["", "<sub>Generated by `make readme` from `results/*.json`.</sub>"]

    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(f"README.md needs {README_START} and {README_END} markers around the results table.")
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    new_text = f"{head}{README_START}\n" + "\n".join(block) + f"\n{README_END}{tail}"
    out = cfg.paths.output_dir / "README.md" if cfg.smoke else readme_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(new_text, encoding="utf-8")
    return out


def demo_space_url(cfg: Config, token: str | None) -> str | None:
    """Link the Gradio Space from the card only if it actually exists."""
    from huggingface_hub import HfApi

    try:
        if HfApi(token=token).repo_exists(cfg.hub.space_repo, repo_type="space"):
            return f"https://huggingface.co/spaces/{cfg.hub.space_repo}"
    except Exception:  # offline / auth problems: just leave the link out
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Use smoke-test outputs (tiny model).")
    parser.add_argument("--dry-run", action="store_true", help="Build everything locally; push nothing.")
    parser.add_argument("--merge", action="store_true", help="Also merge the adapter and publish the full model.")
    parser.add_argument("--space", action="store_true", help="Also deploy the Gradio demo as a Space.")
    parser.add_argument("--readme", action="store_true",
                        help="Refresh the results block of README.md (alone: no model work, no Hub access).")
    args = parser.parse_args()
    cfg = load_config(smoke=args.smoke)

    if args.readme:
        print(f"Results block written to {update_readme(cfg)}")
        if not (args.merge or args.space or args.dry_run):
            return

    token = None if args.dry_run else get_hf_token()
    space_url = f"https://huggingface.co/spaces/{cfg.hub.space_repo}"
    # Link the demo from the cards if we are deploying it now, or if it already exists.
    demo_url = space_url if args.space else (None if args.dry_run else demo_space_url(cfg, token))

    adapter_stage = stage_adapter(cfg, render_card(cfg, "adapter", demo_url))
    print(f"Adapter repo staged in {adapter_stage}")
    merged_dir = None
    if args.merge:
        merged_dir = merge_adapter(cfg)
        write_card(cfg, merged_dir, render_card(cfg, "merged", demo_url))
        print(f"Merged model staged in {merged_dir}")
    space_stage = stage_space(cfg) if args.space else None
    if space_stage:
        print(f"Space staged in {space_stage}")

    if args.dry_run:
        print("Dry run: nothing pushed. Review the README.md files above, then run without --dry-run.")
        return
    push_folder(cfg, adapter_stage, cfg.hub.adapter_repo, token, "Upload LoRA adapter and model card")
    if merged_dir:
        push_folder(cfg, merged_dir, cfg.hub.merged_repo, token, "Upload merged model and model card")
    if space_stage:  # last, so the adapter the Space loads already exists
        push_space(cfg, space_stage, token)


if __name__ == "__main__":
    main()
