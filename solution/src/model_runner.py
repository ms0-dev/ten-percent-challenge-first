"""Uniform local/Modal inference for the base and compressed models."""

from __future__ import annotations

import re
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_SPECIAL_RE = re.compile(r"<\|[^>]+\|>")


def load_model(
    model_path: str,
    device_map: str | dict = "auto",
    dtype: str = "auto",
    attn_implementation: str = "eager",
    python_experts: bool = False,
    batched_experts: bool = False,
    eager_experts: bool = False,
):
    if python_experts:
        from python_experts import load_compressed_python

        return load_compressed_python(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if device_map == "auto" and torch.cuda.is_available():
        # The base fits one 80 GB GPU and the compressed model fits one 24 GB
        # GPU. Force a single device instead of letting accelerate offload.
        device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if batched_experts:
        model.config._experts_implementation = "batched_mm"
    if eager_experts:
        model.config._experts_implementation = "eager"
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, messages: list[dict], force_final: bool = True) -> str:
    kwargs = {}
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low",
            **kwargs,
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if force_final and "<|channel|>" in tokenizer.get_vocab():
        # GPT-OSS / harmony: force the final channel so the model does not
        # spend tokens on an analysis channel. Other models do not have this
        # special token, so leave their prompt untouched.
        text = text + "<|channel|>final<|message|>"
    return text


def clean_generated(text: str) -> str:
    """Strip harmony/special tokens and stop at the end-of-turn marker."""
    for marker in ("<|return|>", "<|end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return _SPECIAL_RE.sub("", text).strip()


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 32,
    batch_size: int = 1,
    max_length: int = 4096,
) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        generated = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
        outputs.extend(clean_generated(x) for x in decoded)
    return outputs
