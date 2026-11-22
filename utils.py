"""Small helpers shared by all scripts: seeding, device/precision, env info, JSON I/O, data loading.

``torch`` is imported lazily so that the data and TF-IDF scripts stay fast and light.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from config import Config

SPLITS = ("train", "val", "test")
TRACKED_PACKAGES = (
    "torch", "transformers", "peft", "trl", "accelerate", "datasets",
    "bitsandbytes", "huggingface_hub", "scikit-learn", "numpy",
)


# --------------------------------------------------------------------------- reproducibility
def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch, for CPU and all GPUs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- hardware
def has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def get_device() -> str:
    return "cuda" if has_cuda() else "cpu"


def get_compute_dtype() -> torch.dtype:
    """bf16 on Ampere+ GPUs, fp16 on older GPUs such as the T4, fp32 on CPU."""
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def environment_info() -> dict[str, Any]:
    """Library versions + hardware, recorded in every results/training JSON."""
    versions: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "libraries": versions,
        "gpu": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "count": torch.cuda.device_count(),
                "memory_gb": round(props.total_memory / 1024**3, 1),
                "cuda": torch.version.cuda,
            }
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------- secrets
def load_env_file() -> None:
    """Load a local .env file into os.environ if python-dotenv is installed (it never overrides)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def get_hf_token() -> str | None:
    """Read HF_TOKEN from the environment or .env. Never hardcode it.

    Returns None when unset; huggingface_hub then falls back to a cached `hf auth login` /
    `notebook_login()` token.
    """
    load_env_file()
    return os.environ.get("HF_TOKEN") or None


# --------------------------------------------------------------------------- file I/O
def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_path(cfg: Config, split: str) -> Path:
    return cfg.paths.data_dir / f"{split}.jsonl"


def load_split(cfg: Config, split: str) -> list[dict[str, Any]]:
    """Load ``train`` / ``val`` / ``test`` rows written by data_prep.py."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    path = split_path(cfg, split)
    if not path.exists():
        smoke_flag = " --smoke" if cfg.smoke else ""
        raise FileNotFoundError(f"{path} not found. Run `python data_prep.py{smoke_flag}` first.")
    return load_jsonl(path)
