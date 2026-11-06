"""Load a reduced-expert GPT-OSS checkpoint through a pure-Python MoE path.

The fused MXFP4 Triton kernels only work with the original 128-expert layout;
any reduced expert count produces garbage. This loader:

1. reads the reduced MXFP4 checkpoint,
2. dequantizes its expert blocks/scales to BF16,
3. builds a standard GptOssForCausalLM with the reduced expert count and the
   standard GptOssExperts Python forward,
4. loads the non-expert weights, router, and dequantized experts.

The checkpoint on disk stays small; the runtime model is BF16. This is a
research/deployment loader, not a training step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding


def load_compressed_python(model_path: str, device: str = "cuda"):
    model_dir = Path(model_path)
    torch.set_default_dtype(torch.bfloat16)

    config = GptOssConfig.from_pretrained(model_dir)
    config.quantization_config = None
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    out: dict[str, torch.Tensor] = {}
    for key, tensor in list(state.items()):
        if key.endswith(".gate_up_proj_blocks") or key.endswith(".down_proj_blocks"):
            scale_key = key.replace("_blocks", "_scales")
            scales = state[scale_key]
            deq = convert_moe_packed_tensors(tensor, scales)
            if key.endswith(".down_proj_blocks") and os.environ.get("PYTHON_EXPERTS_DOWN_TRANSPOSE") == "1":
                deq = deq.transpose(-1, -2).contiguous()
            out[key.replace("_blocks", "")] = deq
        elif key.endswith("_scales"):
            continue
        else:
            out[key] = tensor

    print(f"Dequantized {len(out)} tensors; building {config.num_local_experts}-expert model", flush=True)
    with torch.device("meta"):
        model = GptOssForCausalLM(config)
    model = model.to_empty(device=device)

    # Non-persistent RoPE buffers are not in the checkpoint; rebuild them.
    model.model.rotary_emb = GptOssRotaryEmbedding(config).to(device)

    missing, unexpected = model.load_state_dict(out, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("missing sample:", missing[:8], flush=True)
    if unexpected:
        print("unexpected sample:", unexpected[:8], flush=True)

    del state, out
    model.eval()
    return model, tokenizer
