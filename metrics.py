"""Output parsing, metrics, confusion matrices and the results file format.

Shared by ``eval.py`` (LLMs) and ``baseline_classic.py`` (TF-IDF) so every row of the
before/after table is computed the same way.

Parsing rule (strict): an output is valid only if the WHOLE generated text (whitespace
stripped) is a JSON object with string fields ``category`` and ``priority``. Markdown fences,
explanations or truncated JSON make the output invalid and the prediction counts as wrong.
A separate, clearly labelled ``lenient`` score (first ``{...}`` block in the text) is reported
only to show how many failures are formatting rather than understanding.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils import environment_info, save_json, save_jsonl

if TYPE_CHECKING:
    from config import Config, Labels

INVALID = "<invalid>"


# =========================================================================== parsing
@dataclass(frozen=True)
class ParsedOutput:
    category: str | None
    priority: str | None
    valid_json: bool
    error: str | None = None


def _from_obj(obj: Any) -> ParsedOutput:
    if not isinstance(obj, dict):
        return ParsedOutput(None, None, False, "JSON is not an object")
    cat, prio = obj.get("category"), obj.get("priority")
    if not isinstance(cat, str) or not isinstance(prio, str):
        return ParsedOutput(None, None, False, "missing or non-string 'category'/'priority'")
    return ParsedOutput(cat, prio, True)


def parse_output(text: str) -> ParsedOutput:
    """Strict: the entire output must be the JSON object. Nothing is repaired."""
    try:
        return _from_obj(json.loads(text.strip()))
    except json.JSONDecodeError as e:
        return ParsedOutput(None, None, False, f"invalid JSON: {e.msg}")


_FIRST_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_output_lenient(text: str) -> ParsedOutput:
    """Diagnostic only: take the first ``{...}`` anywhere in the text (e.g. inside ```json fences)."""
    match = _FIRST_OBJECT.search(text)
    if not match:
        return ParsedOutput(None, None, False, "no JSON object found")
    try:
        return _from_obj(json.loads(match.group(0)))
    except json.JSONDecodeError as e:
        return ParsedOutput(None, None, False, f"invalid JSON: {e.msg}")


# =========================================================================== metrics
def _normalise(pred: str | None, allowed: list[str]) -> str:
    """Anything that is not exactly one of the allowed labels counts as INVALID."""
    return pred if pred in allowed else INVALID


def compute_metrics(rows: list[dict[str, Any]], parsed: list[ParsedOutput], labels: Labels) -> dict[str, Any]:
    """Accuracy + macro-F1 per field, exact match, valid-JSON and valid-label rates."""
    from sklearn.metrics import accuracy_score, f1_score

    cats, prios = labels.category_names, labels.priority_names
    gold_c = [r["category"] for r in rows]
    gold_p = [r["priority"] for r in rows]
    pred_c = [_normalise(p.category, cats) for p in parsed]
    pred_p = [_normalise(p.priority, prios) for p in parsed]
    n = len(rows)

    def f1(gold: list[str], pred: list[str], names: list[str], average: str | None) -> Any:
        # Macro over the real labels only; INVALID predictions are simply wrong for the gold class.
        return f1_score(gold, pred, labels=names, average=average, zero_division=0)

    return {
        "n": n,
        "category_accuracy": float(accuracy_score(gold_c, pred_c)),
        "category_macro_f1": float(f1(gold_c, pred_c, cats, "macro")),
        "priority_accuracy": float(accuracy_score(gold_p, pred_p)),
        "priority_macro_f1": float(f1(gold_p, pred_p, prios, "macro")),
        "exact_match": sum(gc == pc and gp == pp for gc, pc, gp, pp in zip(gold_c, pred_c, gold_p, pred_p)) / n,
        "valid_json_rate": sum(p.valid_json for p in parsed) / n,
        "valid_label_rate": sum(pc != INVALID and pp != INVALID for pc, pp in zip(pred_c, pred_p)) / n,
        "per_class_f1": {
            "category": dict(zip(cats, map(float, f1(gold_c, pred_c, cats, None)))),
            "priority": dict(zip(prios, map(float, f1(gold_p, pred_p, prios, None)))),
        },
    }


# =========================================================================== confusion matrices
# Sequential single-hue ramp (blue, steps 100 -> 700) and chart chrome. Light and dark are
# separate themes: on dark, magnitude = contrast with the surface, so the ramp runs 600 -> 100.
_BLUE = {100: "#cde2fb", 200: "#9ec5f4", 300: "#6da7ec", 400: "#3987e5", 500: "#256abf",
         550: "#1c5cab", 600: "#184f95", 650: "#104281", 700: "#0d366b"}
THEMES: dict[str, dict[str, Any]] = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "divider": "#c3c2b7", "ramp": [_BLUE[s] for s in (100, 200, 300, 400, 500, 550, 650, 700)]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "divider": "#383835", "ramp": [_BLUE[s] for s in (600, 550, 500, 400, 300, 200, 100)]},
}


def _cell_text_color(hex_color: str) -> str:
    """Dark text on light fills, white on dark fills (relative luminance threshold)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.5 else "#ffffff"


def plot_confusion_matrix(gold: list[str], pred: list[str], names: list[str], title: str, path: Path,
                          include_invalid: bool, theme: str = "light") -> Path:
    """Row-normalised confusion matrix (colour = share of the true class), counts in non-zero cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    t = THEMES[theme]
    cols = names + ([INVALID] if include_invalid else [])
    counts = np.zeros((len(names), len(cols)), dtype=int)
    for g, p in zip(gold, pred):  # pred is already a label name or INVALID
        if g in names and p in cols:
            counts[names.index(g), cols.index(p)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)

    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    size = 0.62 * len(cols) + 2.6
    fig, ax = plt.subplots(figsize=(size, 0.62 * len(names) + 1.9), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    for i in range(len(names)):
        for j in range(len(cols)):
            if counts[i, j] == 0:
                continue  # empty cells stay as surface: zero is "nothing", not a colour
            fill = matplotlib.colors.to_hex(cmap(share[i, j]))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fill,
                                       edgecolor=t["surface"], linewidth=2))
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=8.5,
                    color=_cell_text_color(fill))
    if include_invalid:  # separate the "invalid output" column from the real labels
        ax.axvline(len(names) - 0.5, color=t["divider"], linewidth=1.2)

    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)), [c if c != INVALID else "invalid" for c in cols],
                  rotation=35, ha="right", fontsize=8.5, color=t["ink2"])
    ax.set_yticks(range(len(names)), names, fontsize=8.5, color=t["ink2"])
    ax.set_xlabel("Predicted", color=t["muted"], fontsize=9)
    ax.set_ylabel("True", color=t["muted"], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=t["ink"], fontsize=10.5, loc="left", pad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("% of true class", color=t["muted"], fontsize=8.5)
    cbar.ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


SERIES = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}  # categorical slots 1-2 (validated)
GRID = {"light": "#e1e0d9", "dark": "#2c2c2a"}


def plot_loss_curves(log_history: list[dict[str, Any]], path: Path, best_step: int | None,
                     theme: str = "light") -> Path:
    """Train loss (line) and validation loss (line + markers) against step, on one shared axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    train_c, eval_c = SERIES[theme]
    train = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    series = [(p, c, lab, m) for p, c, lab, m in ((train, train_c, "train loss", None),
                                                  (evals, eval_c, "validation loss", "o")) if p]
    for points, color, label, marker in series:
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6 if marker else 0,
                markeredgecolor=t["surface"], markeredgewidth=1.5, label=label)
    ax.margins(x=0.14)

    # Direct labels at the line ends (text ink, not series colour); nudge apart when the ends are close.
    ends = [(pts[-1][1], label) for pts, _, label, _ in series]
    y_lo, y_hi = ax.get_ylim()
    close = len(ends) == 2 and abs(ends[0][0] - ends[1][0]) < 0.06 * (y_hi - y_lo)
    for rank, (y_end, label) in enumerate(sorted(ends, reverse=True)):
        x_end = next(pts[-1][0] for pts, _, lab, _ in series if lab == label)
        dy = (6 if rank == 0 else -6) if close else 0
        ax.annotate(label, (x_end, y_end), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=8.5, color=t["ink2"])

    if best_step is not None:
        ax.axvline(best_step, color=t["divider"], linewidth=1, linestyle=(0, (3, 3)))
        x_lo, x_hi = ax.get_xlim()
        near_right = best_step > x_lo + 0.7 * (x_hi - x_lo)
        ax.annotate("best checkpoint", (best_step, y_hi), xytext=(-4 if near_right else 4, -10),
                    textcoords="offset points", ha="right" if near_right else "left", fontsize=8, color=t["muted"])

    ax.grid(axis="y", color=GRID[theme], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["divider"])
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    ax.set_xlabel("training step", color=t["muted"], fontsize=9)
    ax.set_ylabel("loss (answer tokens only)", color=t["muted"], fontsize=9)
    ax.set_title("QLoRA training", color=t["ink"], fontsize=10.5, loc="left", pad=22)
    # Legend in its own row under the title, so it never collides with the data or annotations.
    legend = ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                       borderaxespad=0.2, handlelength=1.8)
    for text in legend.get_texts():
        text.set_color(t["ink2"])

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


# =========================================================================== results files
def select_sample_errors(rows: list[dict[str, Any]], raw_outputs: list[str], parsed: list[ParsedOutput],
                         labels: Labels, n: int, seed: int) -> list[dict[str, Any]]:
    """A reproducible random sample of wrong or invalid predictions, for the error analysis."""
    errors = []
    for row, raw, p in zip(rows, raw_outputs, parsed):
        pc = _normalise(p.category, labels.category_names)
        pp = _normalise(p.priority, labels.priority_names)
        if pc != row["category"] or pp != row["priority"]:
            errors.append({
                "id": row["id"], "subject": row["subject"], "body": row["body"][:500],
                "gold": {"category": row["category"], "priority": row["priority"]},
                "predicted": {"category": pc, "priority": pp},
                "raw_output": raw, "parse_error": p.error,
            })
    return random.Random(seed).sample(errors, k=min(n, len(errors)))


def finalize_run(cfg: Config, run_name: str, rows: list[dict[str, Any]], raw_outputs: list[str],
                 meta: dict[str, Any], include_invalid: bool) -> dict[str, Any]:
    """Compute metrics, write PNGs, predictions and ``results/<run_name>.json``. Returns the results dict."""
    results_dir = cfg.paths.results_dir
    parsed = [parse_output(o) for o in raw_outputs]
    metrics = compute_metrics(rows, parsed, cfg.labels)
    lenient = compute_metrics(rows, [parse_output_lenient(o) for o in raw_outputs], cfg.labels)
    metrics["lenient"] = {k: lenient[k] for k in ("category_accuracy", "priority_accuracy", "exact_match",
                                                  "valid_json_rate")}
    metrics.update(meta.pop("timing", {}))

    figures: dict[str, str] = {}
    for field, names in (("category", cfg.labels.category_names), ("priority", cfg.labels.priority_names)):
        gold = [r[field] for r in rows]
        pred = [_normalise(getattr(p, field), names) for p in parsed]
        for theme in ("light", "dark"):
            suffix = "" if theme == "light" else "_dark"
            path = results_dir / f"{run_name}_cm_{field}{suffix}.png"
            plot_confusion_matrix(gold, pred, names, f"{meta.get('display_name', run_name)} - {field}", path,
                                  include_invalid, theme)
        figures[field] = f"{run_name}_cm_{field}.png"

    predictions = [
        {"id": r["id"], "gold_category": r["category"], "gold_priority": r["priority"], "raw_output": o, **asdict(p)}
        for r, o, p in zip(rows, raw_outputs, parsed)
    ]
    save_jsonl(predictions, results_dir / f"{run_name}_predictions.jsonl")

    results = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": cfg.smoke,
        **meta,
        "metrics": metrics,
        "confusion_matrices": figures,
        "sample_errors": select_sample_errors(rows, raw_outputs, parsed, cfg.labels,
                                              cfg.eval.n_sample_errors, cfg.seed),
        "environment": environment_info(),
    }
    save_json(results, results_dir / f"{run_name}.json")
    return results


def print_summary(results: dict[str, Any]) -> None:
    m = results["metrics"]
    print(f"\n=== {results['run_name']} (n={m['n']}) ===")
    print(f"category  acc {m['category_accuracy']:.3f}  macro-F1 {m['category_macro_f1']:.3f}")
    print(f"priority  acc {m['priority_accuracy']:.3f}  macro-F1 {m['priority_macro_f1']:.3f}")
    print(f"exact match {m['exact_match']:.3f} | valid JSON {m['valid_json_rate']:.3f} | "
          f"lenient exact match {m['lenient']['exact_match']:.3f}")
    if "avg_latency_ms" in m:
        print(f"avg latency {m['avg_latency_ms']:.1f} ms/example (batch size {m.get('batch_size')})")
