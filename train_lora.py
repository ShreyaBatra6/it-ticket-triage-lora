"""QLoRA fine-tuning with transformers + peft + trl's SFTTrainer (+ bitsandbytes on CUDA).

* 4-bit NF4 base weights (double quant), LoRA r=16 / alpha=32 / dropout=0.05 on all linear
  projections (q, k, v, o, gate, up, down) - all values come from config.yaml.
* Loss on the assistant answer only: rows use TRL's conversational "prompt-completion" format,
  so the prompt tokens are masked out (``completion_only_loss=True``).
* Validation loss every ``eval_steps``; the best checkpoint (lowest val loss) is restored at the
  end and saved as the adapter. Logs go to TensorBoard (``--wandb`` adds Weights & Biases).
* Fixed seeds; library versions + hardware are written to results/train_summary.json.

Usage::

    python train_lora.py              # full run on a CUDA GPU (Colab/Kaggle T4)
    python train_lora.py --smoke      # SmolLM2-135M, 10 steps, CPU, no 4-bit
    python train_lora.py --wandb      # also report to Weights & Biases
    tensorboard --logdir outputs/tensorboard
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from config import Config, load_config
from metrics import plot_loss_curves
from modeling import load_model, load_tokenizer
from prompts import build_training_example, check_template_consistency
from utils import environment_info, get_compute_dtype, has_cuda, load_split, save_json, set_seed

GC_KWARGS = {"use_reentrant": False}  # non-reentrant checkpointing is the recommended mode with PEFT


# =========================================================================== data
def build_dataset(cfg: Config, tokenizer: Any, split: str) -> tuple[Dataset, dict[str, Any]]:
    """Prompt-completion rows from the shared prompt builder, minus any that would be truncated."""
    rows = [build_training_example(cfg, r) for r in load_split(cfg, split)]
    lengths = [
        len(tokenizer.apply_chat_template(r["prompt"] + r["completion"], tokenize=True, return_dict=True)["input_ids"])
        for r in rows
    ]
    # Truncation would cut off the answer we train on, so drop over-long rows instead (should be none:
    # prompts.max_body_chars is sized so prompt + answer fits in max_length).
    keep = [r for r, n in zip(rows, lengths) if n <= cfg.training.max_length]
    stats = {"n": len(keep), "dropped_too_long": len(rows) - len(keep),
             "tokens_max": max(lengths), "tokens_mean": round(statistics.mean(lengths), 1)}
    return Dataset.from_list(keep), stats


# =========================================================================== config objects
def lora_config(cfg: Config) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def training_args(cfg: Config, report_to: list[str]) -> SFTConfig:
    t = cfg.training
    dtype = get_compute_dtype()
    return SFTConfig(
        output_dir=str(cfg.paths.output_dir / "checkpoints"),
        # schedule
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=t.warmup_ratio,  # transformers 5: a float < 1 is a ratio of total steps
        weight_decay=t.weight_decay,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        # memory (transformers 5.23 renamed gradient_checkpointing* -> activation_checkpointing*)
        activation_checkpointing=t.gradient_checkpointing,
        activation_checkpointing_kwargs=GC_KWARGS,
        dataloader_pin_memory=has_cuda(),
        # precision: set BOTH explicitly. SFTConfig turns bf16 on by default, which a T4 cannot run.
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        # SFT data handling
        max_length=t.max_length,
        packing=False,
        completion_only_loss=True,  # loss on the assistant answer only
        # evaluation + checkpoint selection
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.eval_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # logging + reproducibility
        logging_steps=t.logging_steps,
        logging_first_step=True,
        report_to=report_to,
        run_name=f"{cfg.model.short_name}-it-ticket-triage{'-smoke' if cfg.smoke else ''}",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )


def keep_adapter_weights_fp32(model: torch.nn.Module) -> int:
    """Undo TRL's bf16 cast of LoRA weights on GPUs without bf16 (e.g. T4).

    For 4-bit models TRL casts the trainable LoRA weights to bf16 (QLoRA paper recipe). A T4 has
    no native bf16, so we train with fp16 mixed precision instead, which requires fp32 master
    weights for the gradient scaler. Returns the number of tensors converted.
    """
    if get_compute_dtype() == torch.bfloat16:
        return 0
    converted = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
            converted += 1
    return converted


# =========================================================================== main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="Tiny model, 10 steps, CPU friendly, no 4-bit.")
    p.add_argument("--wandb", action="store_true", help="Also log to Weights & Biases (needs `pip install wandb`).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override training.max_steps, e.g. 20 for a quick GPU sanity run before the real one.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(smoke=args.smoke)
    if args.max_steps is not None:
        eval_steps = min(cfg.training.eval_steps, args.max_steps)
        cfg = dataclasses.replace(cfg, training=dataclasses.replace(
            cfg.training, max_steps=args.max_steps, eval_steps=eval_steps))
    set_seed(cfg.seed)
    model_id = cfg.model.base_model_id
    use_4bit = cfg.training.load_in_4bit and has_cuda()

    report_to = ["tensorboard"]
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.paths.output_dir / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    if args.wandb:
        try:
            import wandb  # noqa: F401
        except ImportError as e:
            raise SystemExit("--wandb needs the wandb package: pip install wandb==0.31.2") from e
        os.environ.setdefault("WANDB_PROJECT", "it-ticket-triage")
        report_to.append("wandb")

    # TRL pads on the right during training; generation uses left padding (modeling.load_tokenizer default).
    tokenizer = load_tokenizer(model_id, padding_side="right")
    check_template_consistency(tokenizer, cfg)  # training prompt == inference prompt, token for token
    train_ds, train_stats = build_dataset(cfg, tokenizer, "train")
    val_ds, val_stats = build_dataset(cfg, tokenizer, "val")
    print(f"train: {train_stats}\nval:   {val_stats}")

    print(f"Loading {model_id} ({'4-bit NF4' if use_4bit else get_compute_dtype()}) ...")
    model = load_model(model_id, load_in_4bit=use_4bit)
    model.config.use_cache = False  # incompatible with gradient checkpointing during training
    if use_4bit:
        # Standard QLoRA prep: layer norms / embeddings to fp32 (stability in fp16) + input grads for checkpointing.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.training.gradient_checkpointing,
            activation_checkpointing_kwargs=GC_KWARGS,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args(cfg, report_to),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
    )
    n_converted = keep_adapter_weights_fp32(trainer.model)
    if n_converted:
        print(f"Kept {n_converted} LoRA tensors in fp32 for fp16 mixed precision (no native bf16 on this GPU).")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    start = time.perf_counter()
    train_result = trainer.train()
    runtime = time.perf_counter() - start
    final_eval = trainer.evaluate()  # the best checkpoint has been restored at this point

    adapter_dir = cfg.paths.adapter_dir
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    state = trainer.state
    best_step = int(state.best_model_checkpoint.rsplit("-", 1)[-1]) if state.best_model_checkpoint else None
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": cfg.smoke,
        "base_model_id": model_id,
        "adapter_dir": str(adapter_dir),
        "method": "QLoRA (4-bit NF4 + LoRA)" if use_4bit else "LoRA (full-precision base; no CUDA for 4-bit)",
        "lora": {"r": cfg.lora.r, "alpha": cfg.lora.alpha, "dropout": cfg.lora.dropout,
                 "target_modules": cfg.lora.target_modules},
        "training": {
            **{k: v for k, v in vars(cfg.training).items()},
            "load_in_4bit": use_4bit,
            "effective_batch_size": cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps,
            "optimizer": trainer.args.optim,
            "precision": "bf16" if trainer.args.bf16 else "fp16" if trainer.args.fp16 else "fp32",
            "completion_only_loss": True,
            "seed": cfg.seed,
        },
        "data": {"train": train_stats, "val": val_stats},
        "parameters": {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / total, 3)},
        "results": {
            "train_runtime_s": round(runtime, 1),
            "global_steps": state.global_step,
            "epochs": round(state.epoch or 0, 2),
            "train_loss": round(train_result.training_loss, 4),
            "best_checkpoint_step": best_step,
            "best_eval_loss": state.best_metric,
            "final_eval": {k: round(v, 4) for k, v in final_eval.items() if isinstance(v, float)},
        },
        "log_history": state.log_history,
        "environment": environment_info(),
    }
    save_json(summary, cfg.paths.results_dir / "train_summary.json")
    save_json(summary, adapter_dir / "training_info.json")
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "_dark"
        plot_loss_curves(state.log_history, cfg.paths.results_dir / f"train_loss{suffix}.png", best_step, theme)

    print(f"\nBest val loss {state.best_metric} at step {best_step}; adapter saved to {adapter_dir}")
    print(f"Summary: {cfg.paths.results_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
