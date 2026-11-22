"""Classic baseline: TF-IDF (word + character n-grams) + logistic regression, one model per field.

Scored with exactly the same metrics as the LLM runs (see metrics.py). The regularisation
strength C is chosen on the validation split by macro-F1; the test split is used once.

Usage::

    python baseline_classic.py            # -> results/tfidf_lr.json (+ confusion matrices)
    python baseline_classic.py --smoke
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from config import Config, load_config
from metrics import finalize_run, print_summary
from prompts import format_target, format_ticket
from utils import load_split, set_seed

RUN_NAME = "tfidf_lr"
C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)


def ticket_texts(cfg: Config, rows: list[dict[str, Any]]) -> list[str]:
    """Same text the LLM sees (subject + body, truncated identically)."""
    return [format_ticket(r["subject"], r["body"], cfg.prompt.max_body_chars) for r in rows]


def build_pipeline(c: float, seed: int) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        # Character n-grams make the baseline robust to typos ("pirnter", "outlok").
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed)
    return Pipeline([("tfidf", features), ("clf", clf)])


def fit_field(cfg: Config, train: list[dict[str, Any]], val: list[dict[str, Any]], field: str
              ) -> tuple[Pipeline, float, dict[float, float]]:
    """Grid-search C on validation macro-F1, return the model refit on train with the best C."""
    x_train, x_val = ticket_texts(cfg, train), ticket_texts(cfg, val)
    y_train, y_val = [r[field] for r in train], [r[field] for r in val]
    scores = {}
    for c in C_GRID:
        model = build_pipeline(c, cfg.seed).fit(x_train, y_train)
        scores[c] = float(f1_score(y_val, model.predict(x_val), average="macro", zero_division=0))
    best_c = max(scores, key=scores.get)
    return build_pipeline(best_c, cfg.seed).fit(x_train, y_train), best_c, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Use the 50-example smoke dataset.")
    args = parser.parse_args()

    cfg = load_config(smoke=args.smoke)
    set_seed(cfg.seed)
    train, val, test = (load_split(cfg, s) for s in ("train", "val", "test"))

    cat_model, cat_c, cat_scores = fit_field(cfg, train, val, "category")
    prio_model, prio_c, prio_scores = fit_field(cfg, train, val, "priority")

    x_test = ticket_texts(cfg, test)
    start = time.perf_counter()
    cats, prios = cat_model.predict(x_test), prio_model.predict(x_test)
    elapsed = time.perf_counter() - start
    raw_outputs = [format_target(c, p) for c, p in zip(cats, prios)]  # same format as the LLM answer

    meta = {
        "display_name": "TF-IDF + LogReg",
        "split": "test",
        "model": {"type": "tfidf(word 1-2 + char_wb 3-5) + logistic regression",
                  "C": {"category": cat_c, "priority": prio_c},
                  "val_macro_f1_by_C": {"category": cat_scores, "priority": prio_scores}},
        "timing": {"avg_latency_ms": 1000 * elapsed / len(test), "batch_size": len(test), "device": "cpu"},
    }
    results = finalize_run(cfg, RUN_NAME, test, raw_outputs, meta, include_invalid=False)
    print_summary(results)
    print(f"Saved {cfg.paths.results_dir / (RUN_NAME + '.json')}")


if __name__ == "__main__":
    main()
