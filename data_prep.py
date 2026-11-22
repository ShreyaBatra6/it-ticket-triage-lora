"""Build the IT ticket triage dataset.

Pipeline::

    generate (templates [+ optional LLM])  ->  exact dedupe  ->  stratified 80/10/10 split
      ->  drop val/test near-duplicates of train (MinHash)  ->  JSONL + stats.json + dataset card
      ->  (optional) push to the Hugging Face Hub

Usage::

    python data_prep.py               # ~4,000 synthetic tickets -> data/processed/
    python data_prep.py --smoke       # 50 tickets -> data/smoke/
    python data_prep.py --push        # ...and push to <user>/it-ticket-triage
    python data_prep.py --llm 400     # add 400 Claude-written tickets (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

import ticket_templates as T
from config import Config, load_config
from utils import SPLITS, get_hf_token, load_env_file, save_json, save_jsonl, set_seed, split_path

Row = dict[str, Any]
HUB_SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}  # Hub convention


# =========================================================================== template generator
class _Slots(dict):
    """Fills ``{placeholders}`` lazily: one random value per slot, reused within a ticket."""

    def __init__(self, rng: random.Random) -> None:
        super().__init__()
        self.rng = rng

    def __missing__(self, key: str) -> str:
        value = self.rng.choice(T.SLOTS[key])
        self[key] = value
        return value


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def sample_labels(rng: random.Random, cfg: Config) -> tuple[str, str]:
    """Draw (category, priority) with the realistic class balance from ticket_templates.py."""
    category = _weighted_choice(rng, {c: T.CATEGORY_WEIGHTS[c] for c in cfg.labels.category_names})
    priority = _weighted_choice(rng, T.PRIORITY_GIVEN_CATEGORY[category])
    return category, priority


def add_typos(text: str, rng: random.Random, rate: float) -> str:
    """Swap, drop or double one letter in roughly ``rate`` of the (purely alphabetic) words."""
    words = text.split(" ")
    for i, word in enumerate(words):
        if len(word) > 3 and word.isalpha() and rng.random() < rate:
            j = rng.randrange(len(word) - 1)
            op = rng.choice(("swap", "drop", "double"))
            if op == "swap":
                word = word[:j] + word[j + 1] + word[j] + word[j + 2:]
            elif op == "drop":
                word = word[:j] + word[j + 1:]
            else:
                word = word[:j] + word[j] + word[j:]
            words[i] = word
    return " ".join(words)


class TicketGenerator:
    """Templated + randomized ticket generator. See ticket_templates.py for how a ticket is built."""

    def __init__(self, cfg: Config, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self._validate_templates()

    def _validate_templates(self) -> None:
        """Fail early if config.yaml labels and the template bank disagree."""
        if set(self.cfg.labels.priority_names) != set(T.ALL_PRIORITIES):
            raise ValueError(f"Priorities in config.yaml must be {T.ALL_PRIORITIES} for the template generator.")
        for cat in self.cfg.labels.category_names:
            if cat not in T.CATEGORY_ISSUES or cat not in T.CATEGORY_WEIGHTS:
                raise ValueError(f"No templates for category {cat!r}: add them to ticket_templates.py or use --llm.")
            for prio, weight in T.PRIORITY_GIVEN_CATEGORY[cat].items():
                if weight > 0 and not any(prio in issue.priorities for issue in T.CATEGORY_ISSUES[cat]):
                    raise ValueError(f"No {cat!r} template allows priority {prio!r}.")

    def generate(self, n: int) -> list[Row]:
        return [self.generate_one() for _ in range(n)]

    def generate_one(self) -> Row:
        rng = self.rng
        category, priority = sample_labels(rng, self.cfg)
        issues = T.CATEGORY_ISSUES[category]
        issue_idx = rng.choice([i for i, issue in enumerate(issues) if priority in issue.priorities])
        issue = issues[issue_idx]
        problem_idx = rng.randrange(len(issue.problems))
        style = _weighted_choice(rng, T.STYLE_WEIGHTS)
        length = "short" if style == "terse" else _weighted_choice(rng, {"short": 0.2, "normal": 0.55, "long": 0.25})
        slots = _Slots(rng)
        return {
            "subject": self._subject(issue, priority, style, slots),
            "body": self._body(issue, issue.problems[problem_idx], category, priority, style, length, slots),
            "category": category,
            "priority": self._label_noise(priority),
            "source": "template",
            # Which problem sentence was used; lets us hold out whole templates and analyse errors.
            "template_id": f"{category}/{issue_idx}/{problem_idx}",
        }

    # ------------------------------------------------------------------ pieces
    def _subject(self, issue: T.Issue, priority: str, style: str, slots: _Slots) -> str:
        rng = self.rng
        if rng.random() < 0.07:
            subject = rng.choice(T.VAGUE_SUBJECTS)  # forces the model to read the body
        else:
            subject = _capitalize(rng.choice(issue.subjects).format_map(slots))
        # "URGENT" shows up on real urgent tickets *and* on some routine ones, so it is a weak cue.
        urgent_rate = 0.25 if priority in ("High", "Critical") else 0.06
        if rng.random() < urgent_rate:
            subject = rng.choice(T.URGENCY_PREFIXES) + subject
        if rng.random() < 0.05:
            subject = rng.choice(("RE: ", "Fwd: ", "FW: ")) + subject
        if style in ("terse", "mobile") and rng.random() < 0.5:
            subject = subject.lower()
        return subject

    def _impact_sentences(self, issue: T.Issue, priority: str) -> list[str]:
        """Render the latent priority factors (scope / blocking / deadline), some left implicit."""
        rng = self.rng
        if rng.random() < 0.08:
            return []  # no explicit impact at all: priority must be inferred from the problem itself
        factors = T.PRIORITY_FACTORS[priority]
        values = {
            "scope": _weighted_choice(rng, T.SCOPE_BY_SCALE[issue.scale][priority]),
            "blocking": _weighted_choice(rng, factors["blocking"]),
            "deadline": _weighted_choice(rng, factors["deadline"]),
        }
        sentences = []
        for kind, value in values.items():
            if rng.random() >= T.OMIT_PROBABILITY.get(value, T.OMIT_PROBABILITY["default"]):
                sentences.append(rng.choice(T.FACTOR_PHRASES[kind][value]))
        rng.shuffle(sentences)
        return sentences

    def _body(self, issue: T.Issue, problem: str, category: str, priority: str, style: str, length: str,
              slots: _Slots) -> str:
        rng = self.rng
        sentences = [problem]
        n_details = {"short": 0, "normal": rng.randint(0, 1), "long": rng.randint(2, 3)}[length]
        sentences += rng.sample(issue.details, k=min(n_details, len(issue.details)))
        if length != "short" and rng.random() < (0.6 if length == "long" else 0.25):
            sentences.append(rng.choice(T.CONTEXT_SENTENCES))
        if style == "frustrated":
            sentences.insert(rng.randint(1, len(sentences)), rng.choice(T.STYLES["frustrated"]["extras"]))
        impact = self._impact_sentences(issue, priority)
        sentences += impact[:1] if length == "short" else impact
        if length != "short" and rng.random() < 0.12:
            sentences.append(self._side_request(category))

        rendered = [_capitalize(self._vary_wording(s.format_map(slots))) for s in sentences]
        if length == "long" and len(rendered) > 3:  # two paragraphs
            half = len(rendered) // 2
            text = " ".join(rendered[:half]) + "\n\n" + " ".join(rendered[half:])
        else:
            text = " ".join(rendered)

        if style in ("casual", "mobile", "terse"):
            if rng.random() < 0.35:
                text = text.lower()
            if rng.random() < 0.3:
                text = text.replace("'", "")
        typo_rate = {"mobile": 0.06, "casual": 0.04, "terse": 0.04}.get(style, 0.015)
        if rng.random() < (0.9 if style == "mobile" else 0.5):
            text = add_typos(text, rng, typo_rate)

        greeting = rng.choice(T.STYLES[style]["greetings"])
        closing = rng.choice(T.STYLES[style]["closings"]).format_map(slots)
        return "\n\n".join(part for part in (greeting, text, closing) if part)

    def _side_request(self, main_category: str) -> str:
        """An unrelated minor ask from another category; the label stays the main issue."""
        other = self.rng.choice([c for c in T.SIDE_REQUESTS if c != main_category])
        return self.rng.choice(T.SIDE_REQUEST_PREFIXES) + self.rng.choice(T.SIDE_REQUESTS[other])

    def _vary_wording(self, sentence: str) -> str:
        """Swap a few common words for synonyms so identical templates don't render identically."""
        for word, alternatives in T.SYNONYMS:
            if word in sentence and self.rng.random() < 0.3:
                sentence = sentence.replace(word, self.rng.choice(alternatives))
        return sentence

    def _label_noise(self, priority: str) -> str:
        """Move a small share of priorities one level up/down: annotators disagree on priority."""
        if self.rng.random() >= self.cfg.data.priority_label_noise:
            return priority
        names = self.cfg.labels.priority_names
        i = names.index(priority)
        j = i + self.rng.choice((-1, 1))
        if not 0 <= j < len(names):
            j = i - (j - i)
        return names[j]


# =========================================================================== optional LLM hook
LLM_SYSTEM_PROMPT = (
    "You write realistic synthetic IT helpdesk tickets. They are used to train and test a classifier "
    "that assigns each ticket a category and a priority, so each ticket must clearly fit the requested labels "
    "while reading like something a real employee would send."
)
TICKETS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tickets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["subject", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tickets"],
    "additionalProperties": False,
}


def _llm_batch(client: Any, cfg: Config, category: str, priority: str, k: int, rng: random.Random) -> list[Row]:
    """One API call -> up to k tickets for a single (category, priority) pair."""
    import anthropic

    tones = ", ".join(rng.sample(sorted(T.STYLES), 3))
    prompt = (
        f"Write {k} different IT helpdesk tickets that a service desk would label:\n"
        f"- category: {category} ({cfg.labels.categories[category]})\n"
        f"- priority: {priority} ({cfg.labels.priorities[priority]})\n\n"
        f"Vary the writer (role, department, seniority), the tone (for example {tones}), the length "
        "(from one line to a short paragraph) and the details (product names, error messages, what they "
        "already tried). A few may contain typos. The priority should follow from who is affected, whether "
        "they can still work and any deadline. Never write the category or priority label itself, and use "
        "invented names only."
    )
    try:
        response = client.beta.messages.create(
            model=cfg.data.llm_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the request is declined, the API retries it on a fallback model
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": TICKETS_SCHEMA}},
        )
    except anthropic.AuthenticationError as e:
        raise SystemExit("Anthropic authentication failed: set ANTHROPIC_API_KEY (see .env.example).") from e
    except anthropic.BadRequestError as e:
        raise SystemExit(f"Anthropic API rejected the request (check llm_model in config.yaml): {e}") from e
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        # The SDK has already retried these; skip this batch rather than abort the whole run.
        print(f"[llm] skipped {category}/{priority}: {type(e).__name__}: {e}")
        return []

    if response.stop_reason in ("refusal", "max_tokens"):
        print(f"[llm] skipped {category}/{priority}: stop_reason={response.stop_reason}")
        return []
    text = next((block.text for block in response.content if block.type == "text"), "")
    try:
        tickets = json.loads(text)["tickets"]
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"[llm] skipped {category}/{priority}: response was not the expected JSON")
        return []
    return [
        {"subject": t["subject"].strip(), "body": t["body"].strip(),
         "category": category, "priority": priority, "source": "llm"}
        for t in tickets
        if t.get("subject", "").strip() and t.get("body", "").strip()
    ]


def generate_llm_tickets(cfg: Config, n: int, seed: int) -> list[Row]:
    """Optional: ask Claude for ``n`` extra tickets, which are more varied than the templates.

    Needs ``pip install anthropic==2.12.0`` and ANTHROPIC_API_KEY (env var or .env).
    Labels are drawn with the same class balance as the templates.
    """
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit("`--llm` needs the anthropic package: pip install anthropic==2.12.0") from e

    load_env_file()
    client = anthropic.Anthropic()
    rng = random.Random(seed + 1)
    pairs = Counter(sample_labels(rng, cfg) for _ in range(n))
    rows: list[Row] = []
    for (category, priority), count in sorted(pairs.items()):
        for start in range(0, count, cfg.data.llm_batch_size):
            k = min(cfg.data.llm_batch_size, count - start)
            rows += _llm_batch(client, cfg, category, priority, k, rng)
    print(f"[llm] received {len(rows)} of {n} requested tickets")
    return rows


# =========================================================================== cleaning + splitting
def ticket_text(row: Row) -> str:
    return f"{row['subject']}\n{row['body']}"


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace (used for hashing and shingles)."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def dedupe_exact(rows: list[Row]) -> tuple[list[Row], int]:
    """Drop tickets whose normalized subject+body hash was already seen (keeps the first)."""
    seen: set[str] = set()
    kept = []
    for row in rows:
        digest = hashlib.sha1(normalize_text(ticket_text(row)).encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            kept.append(row)
    return kept, len(rows) - len(kept)


def template_holdout_split(rows: list[Row], val_fraction: float, test_fraction: float, seed: int
                           ) -> tuple[dict[str, list[Row]], str]:
    """~80/10/10 split where val and test only contain problem templates never seen in train.

    With templated data a random split puts every test ticket's problem sentence in train too,
    so a bag-of-words model can memorise it. Grouping by ``template_id`` measures generalisation
    to unseen phrasing instead. Groups are assigned with StratifiedGroupKFold (stratified on
    category+priority). LLM-written rows have no template and form their own group each.
    """
    import warnings

    from sklearn.model_selection import StratifiedGroupKFold

    def groups(rs: list[Row]) -> list[str]:
        return [r.get("template_id") or r["id"] for r in rs]

    def labels(rs: list[Row]) -> list[str]:
        return [f"{r['category']}|{r['priority']}" for r in rs]

    with warnings.catch_warnings():  # "least populated class has only N members" is expected here
        warnings.simplefilter("ignore", UserWarning)
        outer = StratifiedGroupKFold(n_splits=round(1 / test_fraction), shuffle=True, random_state=seed)
        rest_idx, test_idx = next(outer.split(rows, labels(rows), groups(rows)))
        rest = [rows[i] for i in rest_idx]
        inner = StratifiedGroupKFold(
            n_splits=round((1 - test_fraction) / val_fraction), shuffle=True, random_state=seed
        )
        train_idx, val_idx = next(inner.split(rest, labels(rest), groups(rest)))
    splits = {"train": [rest[i] for i in train_idx], "val": [rest[i] for i in val_idx],
              "test": [rows[i] for i in test_idx]}
    return splits, "template held out (StratifiedGroupKFold on category+priority)"


def stratified_split(rows: list[Row], val_fraction: float, test_fraction: float, seed: int
                     ) -> tuple[dict[str, list[Row]], str]:
    """80/10/10 split, stratified on category+priority (falls back to category, then random, for tiny data)."""
    from sklearn.model_selection import train_test_split

    strategies: list[tuple[str, Callable[[Row], str] | None]] = [
        ("category+priority", lambda r: f"{r['category']}|{r['priority']}"),
        ("category", lambda r: r["category"]),
        ("random (too few examples to stratify)", None),
    ]
    holdout = val_fraction + test_fraction
    for name, key in strategies:
        try:
            train, rest = train_test_split(
                rows, test_size=holdout, random_state=seed, stratify=[key(r) for r in rows] if key else None
            )
            val, test = train_test_split(
                rest, test_size=test_fraction / holdout, random_state=seed,
                stratify=[key(r) for r in rest] if key else None,
            )
            return {"train": train, "val": val, "test": test}, name
        except ValueError:  # a class has too few members for this stratification
            continue
    raise RuntimeError("unreachable: the random split cannot fail")


def _shingles(text: str, size: int) -> set[str]:
    words = normalize_text(text).split()
    if len(words) <= size:
        return {" ".join(words)}
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def minhash_signature(text: str, cfg: Config) -> np.ndarray:
    from datasketch import MinHash

    m = MinHash(num_perm=cfg.data.minhash_num_perm, seed=cfg.seed)
    for shingle in _shingles(text, cfg.data.shingle_size):
        m.update(shingle.encode("utf-8"))
    return m.hashvalues


def drop_near_duplicates(cfg: Config, splits: dict[str, list[Row]]) -> dict[str, Any]:
    """Remove val/test tickets that are near-duplicates of any train ticket.

    Estimated Jaccard similarity of word-shingle sets = share of equal MinHash values. At this
    data size a brute-force comparison against all train signatures is fast and exact (no LSH needed).
    Returns counts of dropped rows and nearest-train similarity stats for the remaining test set.
    """
    threshold = cfg.data.near_dup_threshold
    train_sigs = np.stack([minhash_signature(ticket_text(r), cfg) for r in splits["train"]])
    report: dict[str, Any] = {"threshold": threshold, "dropped": {}}
    for split in ("val", "test"):
        kept, nearest = [], []
        for row in splits[split]:
            sim = float((train_sigs == minhash_signature(ticket_text(row), cfg)).mean(axis=1).max())
            if sim < threshold:
                kept.append(row)
                nearest.append(sim)
        report["dropped"][split] = len(splits[split]) - len(kept)
        splits[split] = kept
        if split == "test" and nearest:
            report["test_nearest_train_similarity"] = {
                "median": round(float(np.median(nearest)), 3),
                "p90": round(float(np.percentile(nearest, 90)), 3),
                "max": round(float(np.max(nearest)), 3),
            }
    return report


# =========================================================================== stats + dataset card
def label_distribution(rows: list[Row], cfg: Config) -> dict[str, dict[str, int]]:
    cats, prios = Counter(r["category"] for r in rows), Counter(r["priority"] for r in rows)
    return {
        "category": {c: cats.get(c, 0) for c in cfg.labels.category_names},
        "priority": {p: prios.get(p, 0) for p in cfg.labels.priority_names},
    }


def _size_category(n: int) -> str:
    return "n<1K" if n < 1_000 else "1K<n<10K" if n < 10_000 else "10K<n<100K"


def build_dataset_card(cfg: Config, stats: dict[str, Any]) -> str:
    """Markdown dataset card with YAML front matter (also declares the split files for the Hub)."""
    sizes = stats["split_sizes"]
    dist = stats["label_distribution"]
    total = sum(sizes.values())
    near = stats["near_duplicates"]
    sim = near.get("test_nearest_train_similarity", {})
    n_llm = stats["n_llm"]
    source_line = "hand-written templates" + (f" plus {n_llm} tickets written by `{cfg.data.llm_model}`" if n_llm else "")

    def label_table(kind: str, definitions: dict[str, str]) -> str:
        lines = [f"| {kind} | definition | train | val | test |", "|---|---|---:|---:|---:|"]
        for name, desc in definitions.items():
            counts = " | ".join(str(dist[s][kind.lower()][name]) for s in SPLITS)
            lines.append(f"| {name} | {desc} | {counts} |")
        return "\n".join(lines)

    data_files = "\n".join(
        f"  - split: {HUB_SPLIT_NAMES[s]}\n    path: data/{HUB_SPLIT_NAMES[s]}-*" for s in SPLITS
    )
    return f"""---
license: apache-2.0
language:
- en
task_categories:
- text-classification
tags:
- synthetic
- it-support
- helpdesk
- ticket-triage
size_categories:
- {_size_category(total)}
pretty_name: IT Ticket Triage (synthetic){" - smoke test" if cfg.smoke else ""}
configs:
- config_name: default
  data_files:
{data_files}
---

# IT Ticket Triage (synthetic)

> **Every ticket in this dataset is synthetic.** It was generated from {source_line}.
> No real user, company or ticket data is included.

Each row is an IT helpdesk ticket (subject + body) labelled with one **category** and one **priority**.
It was built to fine-tune and evaluate small LLMs on ticket triage; see the companion model
[`{cfg.hub.adapter_repo}`](https://huggingface.co/{cfg.hub.adapter_repo}).

## Fields

| field | type | description |
|---|---|---|
| `id` | string | stable ticket id |
| `subject` | string | ticket subject line |
| `body` | string | ticket text (greeting, problem, details, impact, sign-off) |
| `category` | string | one of {", ".join(cfg.labels.category_names)} |
| `priority` | string | one of {", ".join(cfg.labels.priority_names)} |
| `source` | string | `template` or `llm` |
| `template_id` | string | `category/issue/problem` template used (template rows only); for analysis |

## Splits

| split | rows |
|---|---:|
| train | {sizes["train"]} |
| validation | {sizes["val"]} |
| test | {sizes["test"]} |

## Labels

{label_table("Category", cfg.labels.categories)}

{label_table("Priority", cfg.labels.priorities)}

## How it was built

1. **Generation.** {stats["n_generated"]} tickets. Each ticket picks a category, a priority (with realistic
   imbalance: Critical is rare, Security skews high, Other skews low) and a problem template. Priority is
   expressed through three latent factors, each with varied wording or left implicit: *who is affected*,
   *whether they can still work* and *any deadline*. Tickets vary in style (formal, casual, terse,
   frustrated, mobile, non-native phrasing), length and typos. Some mention other categories' vocabulary
   ("Outlook" in a Wi-Fi ticket), and "URGENT" appears on some routine tickets too.
2. **Label noise.** {stats["priority_label_noise"]:.0%} of priorities were moved one level up or down to mimic
   annotator disagreement.
3. **Exact dedupe.** {stats["exact_duplicates_removed"]} duplicates removed (normalized subject+body hash).
4. **Split.** About 80/10/10, seed {stats["seed"]}, split method: {stats["stratified_by"]}.{
   " Validation and test only contain problem templates that never appear in train, so scores measure"
   " generalisation to unseen phrasing rather than memorisation." if "template" in stats["stratified_by"] else ""}
5. **Leakage check.** MinHash ({cfg.data.minhash_num_perm} permutations, word {cfg.data.shingle_size}-grams).
   Validation/test tickets with estimated Jaccard similarity >= {near["threshold"]} to any training ticket were
   dropped (validation: {near["dropped"]["val"]}, test: {near["dropped"]["test"]}).
   Remaining test tickets vs. their most similar training ticket: median {sim.get("median", "n/a")},
   90th percentile {sim.get("p90", "n/a")}, max {sim.get("max", "n/a")}.

## Limitations

- Templated text is more regular than real tickets, so models score higher here than they would on real
  helpdesk data. Treat scores as a comparison between methods on the same data, not as real-world accuracy.
- Priority is subjective; the deliberate label noise caps achievable priority accuracy at roughly
  {1 - stats["priority_label_noise"]:.0%}.
- English only, office-IT context.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{cfg.hub.dataset_repo}")
print(ds["train"][0])
```
"""


def push_to_hub(cfg: Config, splits: dict[str, list[Row]], card: str) -> None:
    """Upload the splits as parquet + the dataset card. Requires HF_TOKEN or `huggingface-cli login`."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import DatasetCard

    token = get_hf_token()
    dataset = DatasetDict({HUB_SPLIT_NAMES[s]: Dataset.from_list(rows) for s, rows in splits.items()})
    dataset.push_to_hub(cfg.hub.dataset_repo, private=cfg.hub.private, token=token)
    DatasetCard(card).push_to_hub(cfg.hub.dataset_repo, token=token)
    print(f"Pushed dataset: https://huggingface.co/datasets/{cfg.hub.dataset_repo}")


# =========================================================================== main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="50 tickets into data/smoke/.")
    parser.add_argument("--n", type=int, default=None, help="Override the number of template tickets.")
    parser.add_argument("--llm", type=int, default=0, metavar="N", help="Also generate N tickets with Claude.")
    parser.add_argument("--split-mode", choices=("random", "template_holdout"), default=None,
                        help="Override data.split_mode from config.yaml.")
    parser.add_argument("--push", action="store_true", help="Push the dataset to the Hugging Face Hub.")
    parser.add_argument("--show", type=int, default=3, help="Print this many example tickets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(smoke=args.smoke)
    if args.split_mode:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, split_mode=args.split_mode))
    set_seed(cfg.seed)

    n_template = args.n or cfg.data.n_examples
    rows = TicketGenerator(cfg, seed=cfg.seed).generate(n_template)
    if args.llm:
        rows += generate_llm_tickets(cfg, args.llm, seed=cfg.seed)
    n_generated = len(rows)
    random.Random(cfg.seed).shuffle(rows)

    rows, n_dupes = dedupe_exact(rows)
    rows = [{"id": f"tkt-{i:05d}", **row} for i, row in enumerate(rows)]
    split_args = (rows, cfg.data.val_fraction, cfg.data.test_fraction, cfg.seed)
    if cfg.data.split_mode == "template_holdout":
        try:
            splits, strategy = template_holdout_split(*split_args)
        except ValueError:  # tiny data (smoke): every label class has fewer members than folds
            splits, strategy = stratified_split(*split_args)
    elif cfg.data.split_mode == "random":
        splits, strategy = stratified_split(*split_args)
    else:
        raise ValueError(f"data.split_mode must be 'random' or 'template_holdout', got {cfg.data.split_mode!r}")
    near_dups = drop_near_duplicates(cfg, splits)

    for split, split_rows in splits.items():
        split_rows.sort(key=lambda r: r["id"])
        save_jsonl(split_rows, split_path(cfg, split))

    stats = {
        "seed": cfg.seed,
        "n_generated": n_generated,
        "n_llm": sum(r["source"] == "llm" for r in rows),
        "priority_label_noise": cfg.data.priority_label_noise,
        "exact_duplicates_removed": n_dupes,
        "stratified_by": strategy,
        "near_duplicates": near_dups,
        "split_sizes": {s: len(r) for s, r in splits.items()},
        "label_distribution": {s: label_distribution(r, cfg) for s, r in splits.items()},
    }
    save_json(stats, cfg.paths.data_dir / "stats.json")
    card = build_dataset_card(cfg, stats)
    (cfg.paths.data_dir / "README.md").write_text(card, encoding="utf-8")

    print(f"Generated {n_generated} tickets, removed {n_dupes} exact duplicates.")
    print(f"Split ({strategy}): {stats['split_sizes']}")
    print(f"Near-duplicates of train dropped (Jaccard >= {near_dups['threshold']}): {near_dups['dropped']}")
    if "test_nearest_train_similarity" in near_dups:
        print(f"Test vs nearest train ticket similarity: {near_dups['test_nearest_train_similarity']}")
    print("Train label distribution:", json.dumps(stats["label_distribution"]["train"]))
    print(f"Saved to {cfg.paths.data_dir}")
    for row in splits["train"][: args.show]:
        print("-" * 70)
        print(f"[{row['category']} | {row['priority']}]  Subject: {row['subject']}\n{row['body']}")

    if args.push:
        push_to_hub(cfg, splits, card)


if __name__ == "__main__":
    main()
