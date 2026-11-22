"""Typed access to ``config.yaml``.

Every script does::

    from config import load_config
    cfg = load_config(smoke=args.smoke)

``smoke=True`` deep-merges the ``smoke:`` block of the YAML over the defaults, so the
same code path runs with a tiny model, 50 examples and 10 training steps.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Labels:
    """Label sets with human-readable definitions (insertion order is preserved)."""

    categories: dict[str, str]
    priorities: dict[str, str]

    @property
    def category_names(self) -> list[str]:
        return list(self.categories)

    @property
    def priority_names(self) -> list[str]:
        return list(self.priorities)


@dataclass(frozen=True)
class ModelSettings:
    base_model_id: str
    license: str

    @property
    def short_name(self) -> str:
        """``ibm-granite/granite-3.3-2b-instruct`` -> ``granite-3.3-2b-instruct``."""
        return self.base_model_id.split("/")[-1].lower()


@dataclass(frozen=True)
class DataSettings:
    n_examples: int
    val_fraction: float
    test_fraction: float
    split_mode: str
    priority_label_noise: float
    near_dup_threshold: float
    minhash_num_perm: int
    shingle_size: int
    llm_model: str
    llm_batch_size: int


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    output_dir: Path
    results_dir: Path

    @property
    def adapter_dir(self) -> Path:
        """Where train_lora.py saves the best adapter."""
        return self.output_dir / "lora-adapter"

    @property
    def hub_adapter_dir(self) -> Path:
        """Staging folder uploaded to the adapter repo: adapter files + model card + images."""
        return self.output_dir / "hub" / "adapter"

    @property
    def merged_dir(self) -> Path:
        """Merged full model + its model card; uploaded to the merged repo (and usable with eval.py --model)."""
        return self.output_dir / "hub" / "merged"


@dataclass(frozen=True)
class HubSettings:
    dataset_repo: str
    adapter_repo: str
    merged_repo: str
    space_repo: str
    private: bool


@dataclass(frozen=True)
class PromptSettings:
    few_shot_k: int
    max_new_tokens: int
    max_body_chars: int


@dataclass(frozen=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass(frozen=True)
class TrainingSettings:
    load_in_4bit: bool
    num_train_epochs: float
    max_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    max_length: int
    gradient_checkpointing: bool
    logging_steps: int
    eval_steps: int
    save_total_limit: int


@dataclass(frozen=True)
class ReportRow:
    run_name: str
    label: str


@dataclass(frozen=True)
class EvalSettings:
    batch_size: int
    n_sample_errors: int
    report_rows: list[ReportRow]


@dataclass(frozen=True)
class Config:
    seed: int
    hf_username: str
    github_url: str
    smoke: bool
    labels: Labels
    model: ModelSettings
    data: DataSettings
    paths: Paths
    hub: HubSettings
    prompt: PromptSettings
    lora: LoraSettings
    training: TrainingSettings
    eval: EvalSettings
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``override`` merged in (nested dicts are merged, not replaced)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(smoke: bool = False, path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load ``config.yaml`` into a typed :class:`Config`.

    Args:
        smoke: apply the ``smoke:`` overrides (tiny model, tiny data, CPU friendly).
        path: path to the YAML file.
    """
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    smoke_overrides = raw.pop("smoke", {}) or {}
    if smoke:
        raw = _deep_merge(raw, smoke_overrides)

    model = ModelSettings(**raw["model"])
    fmt = {"user": raw["hf_username"], "model_short": model.short_name}
    hub_raw = dict(raw["hub"])
    if smoke:
        # Never let a smoke run overwrite the real repos, even by accident.
        for key in ("dataset_repo", "adapter_repo", "merged_repo", "space_repo"):
            hub_raw[key] = hub_raw[key] + "-smoke"
    hub = HubSettings(
        **{k: (v.format(**fmt) if isinstance(v, str) else v) for k, v in hub_raw.items()}
    )

    paths = Paths(**{k: REPO_ROOT / v for k, v in raw["paths"].items()})
    eval_raw = dict(raw["eval"])
    eval_raw["report_rows"] = [ReportRow(**row) for row in eval_raw["report_rows"]]

    return Config(
        seed=int(raw["seed"]),
        hf_username=raw["hf_username"],
        github_url=raw.get("github_url") or "",
        smoke=smoke,
        labels=Labels(**raw["labels"]),
        model=model,
        data=DataSettings(**raw["data"]),
        paths=paths,
        hub=hub,
        prompt=PromptSettings(**raw["prompt"]),
        lora=LoraSettings(**raw["lora"]),
        training=TrainingSettings(**raw["training"]),
        eval=EvalSettings(**eval_raw),
        raw=raw,
    )


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Print the resolved configuration.")
    parser.add_argument("--smoke", action="store_true", help="Apply smoke-test overrides.")
    pprint.pprint(load_config(smoke=parser.parse_args().smoke), sort_dicts=False)
