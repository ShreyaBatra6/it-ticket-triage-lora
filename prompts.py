"""The ONE prompt builder used everywhere: baseline eval, few-shot eval, training and the demo.

Using the same function in every place guarantees that the base model and the fine-tuned
model see exactly the same prompt. The only difference is that few-shot mode adds
k solved examples as earlier user/assistant turns.

Prompt layout (rendered with the tokenizer's own chat template)::

    system:    task instructions + label definitions (from config.yaml)
    [user:     example ticket 1]        <- few-shot only
    [assistant:{"category": ..., "priority": ...}]
    ...
    user:      Subject: <subject>\n\n<body>
    assistant: {"category": "...", "priority": "..."}   <- generated / trained on
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from config import Config, Labels


class Message(TypedDict):
    role: str
    content: str


def build_system_prompt(labels: Labels) -> str:
    """Instructions + label definitions. Generated from config.yaml so labels live in one place."""
    categories = "\n".join(f"- {name}: {desc}" for name, desc in labels.categories.items())
    priorities = "\n".join(f"- {name}: {desc}" for name, desc in labels.priorities.items())
    return (
        "You triage IT helpdesk tickets. Assign exactly one category and one priority.\n\n"
        f"Categories:\n{categories}\n\n"
        f"Priorities:\n{priorities}\n\n"
        'Reply with JSON only: {"category": "<category>", "priority": "<priority>"}'
    )


def format_ticket(subject: str, body: str, max_body_chars: int) -> str:
    """User-turn text. Long bodies are cut so prompt + answer always fit in the training window."""
    body = body.strip()
    if len(body) > max_body_chars:
        body = body[:max_body_chars].rstrip() + " ..."
    return f"Subject: {subject.strip()}\n\n{body}"


def format_target(category: str, priority: str) -> str:
    """The exact assistant answer the model is trained to produce."""
    return json.dumps({"category": category, "priority": priority})


def build_messages(
    cfg: Config,
    subject: str,
    body: str,
    few_shot: Sequence[dict[str, Any]] = (),
) -> list[Message]:
    """Chat messages for one ticket, ending with the user turn (no assistant answer)."""
    max_chars = cfg.prompt.max_body_chars
    messages: list[Message] = [{"role": "system", "content": build_system_prompt(cfg.labels)}]
    for example in few_shot:
        messages.append({"role": "user", "content": format_ticket(example["subject"], example["body"], max_chars)})
        messages.append({"role": "assistant", "content": format_target(example["category"], example["priority"])})
    messages.append({"role": "user", "content": format_ticket(subject, body, max_chars)})
    return messages


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
    subject: str,
    body: str,
    few_shot: Sequence[dict[str, Any]] = (),
) -> str:
    """Render the messages with the model's chat template, ready for generation."""
    return tokenizer.apply_chat_template(
        build_messages(cfg, subject, body, few_shot),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_training_example(cfg: Config, row: dict[str, Any]) -> dict[str, list[Message]]:
    """TRL "conversational prompt-completion" row. TRL computes the loss on the completion only."""
    return {
        "prompt": build_messages(cfg, row["subject"], row["body"]),
        "completion": [{"role": "assistant", "content": format_target(row["category"], row["priority"])}],
    }


def select_few_shot_examples(train_rows: Sequence[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """Pick k fixed demonstrations from the TRAIN split (never val/test).

    Prefers short tickets and covers as many different categories and priorities as possible,
    so the few-shot prompt stays compact and is not biased toward one label.
    """
    rng = random.Random(seed)
    pool = list(train_rows)
    rng.shuffle(pool)
    short = [r for r in pool if len(r["body"]) <= 400]
    pool = short if len(short) >= k else pool

    chosen: list[dict[str, Any]] = []
    seen_cats: set[str] = set()
    seen_prios: set[str] = set()
    for row in pool:  # first pass: new category AND new priority
        if row["category"] not in seen_cats and row["priority"] not in seen_prios:
            chosen.append(row)
            seen_cats.add(row["category"])
            seen_prios.add(row["priority"])
        if len(chosen) == k:
            return chosen
    for row in pool:  # fallback: fill up with anything not yet chosen
        if row not in chosen:
            chosen.append(row)
        if len(chosen) == k:
            break
    return chosen


def encode_prompts(tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str]) -> Any:
    """Tokenize rendered prompts for generation.

    ``add_special_tokens=False`` because the chat template already contains every special
    token; adding a BOS here would make inference token ids differ from training token ids.
    """
    return tokenizer(list(prompts), add_special_tokens=False, padding=True, return_tensors="pt")


def check_template_consistency(tokenizer: PreTrainedTokenizerBase, cfg: Config) -> None:
    """Fail loudly if the model would be trained on one prompt and evaluated on another.

    TRL masks the prompt by tokenizing ``prompt`` (with ``add_generation_prompt=True``) and
    ``prompt + completion`` and training only on the tokens after their common prefix. We check:
      1. the prompt token ids are an exact prefix of the prompt+completion token ids, and
      2. the ids used at inference (``build_prompt`` + ``encode_prompts``) equal those prompt ids.
    """
    example = {"subject": "Test", "body": "My laptop will not turn on.", "category": "Hardware", "priority": "High"}
    row = build_training_example(cfg, example)
    train_prompt_ids = tokenizer.apply_chat_template(
        row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]
    train_full_ids = tokenizer.apply_chat_template(
        row["prompt"] + row["completion"], tokenize=True, return_dict=True
    )["input_ids"]
    if train_full_ids[: len(train_prompt_ids)] != train_prompt_ids:
        raise ValueError(
            "Tokenized prompt is not a prefix of tokenized prompt+completion; TRL would train on a "
            "different context than the one used at inference. Check the chat template."
        )

    inference_text = build_prompt(tokenizer, cfg, example["subject"], example["body"])
    inference_ids = encode_prompts(tokenizer, [inference_text])["input_ids"][0].tolist()
    if inference_ids != train_prompt_ids:
        raise ValueError("Inference prompt token ids differ from training prompt token ids.")


if __name__ == "__main__":
    import argparse

    from transformers import AutoTokenizer

    from config import load_config

    parser = argparse.ArgumentParser(description="Render the shared prompt and sanity-check the chat template.")
    parser.add_argument("--smoke", action="store_true", help="Use the tiny smoke-test model.")
    parser.add_argument("--model", default=None, help="Override the tokenizer/model id.")
    args = parser.parse_args()

    cfg = load_config(smoke=args.smoke)
    model_id = args.model or cfg.model.base_model_id
    tok = AutoTokenizer.from_pretrained(model_id)
    check_template_consistency(tok, cfg)

    demo = {
        "subject": "VPN keeps disconnecting",
        "body": "Since this morning the VPN drops every 5 minutes, I can't stay on my calls. Using home wifi.",
        "category": "Network",
        "priority": "High",
    }
    text = build_prompt(tok, cfg, demo["subject"], demo["body"])
    target = format_target(demo["category"], demo["priority"])
    # Worst case: realistic English text at the maximum body length kept after truncation.
    max_body = ("Outlook crashes when I open the shared calendar, tried a restart already. " * 40)[
        : cfg.prompt.max_body_chars
    ]
    worst = build_prompt(tok, cfg, "Subject line of a typical length here", max_body) + target
    print(text + target)
    print("-" * 60)
    print(f"model:                   {model_id}")
    print(f"prompt tokens:           {len(tok(text, add_special_tokens=False).input_ids)}")
    print(f"target tokens:           {len(tok(target, add_special_tokens=False).input_ids)}")
    print(f"worst-case total tokens: {len(tok(worst, add_special_tokens=False).input_ids)} "
          f"(training max_length = {cfg.training.max_length})")
    print("template consistency:    OK (train prompt ids == inference prompt ids, prefix of train ids)")
