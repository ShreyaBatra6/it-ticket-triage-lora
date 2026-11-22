"""Model/tokenizer loading and batched greedy generation (shared by eval, training, app, publish)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from utils import get_compute_dtype, has_cuda

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(model_id: str, padding_side: str = "left") -> PreTrainedTokenizerBase:
    """Tokenizer with a pad token. Left padding is required for batched generation."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def bnb_4bit_config() -> Any:
    """QLoRA quantization: 4-bit NF4 weights, double quantization, compute in fp16/bf16."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=get_compute_dtype(),
    )


def load_model(model_id: str, adapter: str | None = None, load_in_4bit: bool = False) -> PreTrainedModel:
    """Load a base or merged model, optionally with a LoRA adapter on top.

    4-bit loading needs CUDA + bitsandbytes; on CPU it is skipped with a message so the
    same command works in the smoke test.
    """
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": get_compute_dtype()}
    if has_cuda():
        kwargs["device_map"] = {"": 0}
    if load_in_4bit:
        if has_cuda():
            kwargs["quantization_config"] = bnb_4bit_config()
        else:
            print("[modeling] CUDA not available: loading in full precision instead of 4-bit.")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str],
             max_new_tokens: int, batch_size: int) -> tuple[list[str], float]:
    """Greedy-decode every prompt. Returns (completions in input order, total seconds spent generating).

    Prompts are processed in length-sorted batches to minimise padding; results are put back
    in the original order. Only newly generated tokens are decoded.
    """
    import torch
    from transformers import GenerationConfig

    from prompts import encode_prompts

    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    gen_config = GenerationConfig(  # a fresh config so model defaults (temperature, top_p...) cannot leak in
        do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos
    )
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: list[str] = [""] * len(prompts)
    total_seconds = 0.0
    device = next(model.parameters()).device

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = encode_prompts(tokenizer, [prompts[i] for i in idx]).to(device)
        if has_cuda():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**batch, generation_config=gen_config)
        if has_cuda():
            torch.cuda.synchronize()
        total_seconds += time.perf_counter() - t0
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        for i, text in zip(idx, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
            outputs[i] = text
    return outputs, total_seconds
