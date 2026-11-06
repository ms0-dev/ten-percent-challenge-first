"""Create a smaller GPT-OSS checkpoint by keeping the top-K experts per layer.

This is pure checkpoint surgery: every kept expert tensor is an exact slice of
the corresponding base tensor. No gradients, no fine-tuning, no weight updates.

The router is sliced to the same K experts and keeps top-4 routing. The
non-expert tensors (attention, embeddings, lm_head, norms) are copied as-is.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


EXPERT_SUFFIXES = (
    "mlp.experts.gate_up_proj_blocks",
    "mlp.experts.gate_up_proj_scales",
    "mlp.experts.gate_up_proj_bias",
    "mlp.experts.down_proj_blocks",
    "mlp.experts.down_proj_scales",
    "mlp.experts.down_proj_bias",
)
ROUTER_SUFFIXES = (
    "mlp.router.weight",
    "mlp.router.bias",
)


def parse_layer(key: str, suffix: str) -> int | None:
    marker = f".layers."
    if marker not in key or not key.endswith(suffix):
        return None
    layer = key.split(marker, 1)[1].split(".", 1)[0]
    return int(layer) if layer.isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--router-usage", type=Path, required=True)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and not args.force:
        raise SystemExit(f"{args.out_dir} already exists; pass --force to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    usage = json.loads(args.router_usage.read_text(encoding="utf-8"))
    counts = usage["counts"]  # [num_layers][num_experts]
    num_layers = len(counts)
    num_experts = len(counts[0])
    if args.k > num_experts:
        raise SystemExit(f"k={args.k} > num_experts={num_experts}")

    keep = [
        sorted(range(num_experts), key=lambda e: counts[layer][e], reverse=True)[: args.k]
        for layer in range(num_layers)
    ]
    print(f"Keeping top {args.k} of {num_experts} experts per layer")
    for layer in range(min(3, num_layers)):
        print(f"  layer {layer}: {keep[layer]}")

    index = json.loads((args.base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    print(f"Base shards: {len(shard_names)}")

    output: dict[str, torch.Tensor] = {}
    for shard_i, shard_name in enumerate(shard_names):
        shard_path = args.base_dir / shard_name
        tensors = load_file(str(shard_path), device="cpu")
        for key, tensor in tensors.items():
            layer = None
            for suffix in EXPERT_SUFFIXES + ROUTER_SUFFIXES:
                layer = parse_layer(key, suffix)
                if layer is not None:
                    break
            if layer is not None:
                if tensor.shape[0] != num_experts:
                    raise SystemExit(f"Unexpected expert dim for {key}: {tensor.shape}")
                tensor = tensor[keep[layer]].contiguous()
            output[key] = tensor
        print(f"  processed shard {shard_i + 1}/{len(shard_names)}: {shard_name} ({len(tensors)} tensors)")

    out_weights = args.out_dir / "model.safetensors"
    save_file(output, str(out_weights))
    print(f"Wrote {out_weights}")

    # Copy tokenizer / generation / chat-template files.
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        src = args.base_dir / name
        if src.exists():
            shutil.copy2(src, args.out_dir / name)

    config = json.loads((args.base_dir / "config.json").read_text(encoding="utf-8"))
    config["num_local_experts"] = args.k
    config["num_experts_per_tok"] = min(4, args.k)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Report logical parameter counts and bytes. The MXFP4 expert weights are
    # stored as U8 blocks with two 4-bit values per byte, so `numel()` of a
    # `*_blocks` tensor is half the logical parameter count. `*_scales` are
    # quantization metadata, not model parameters.
    def count_params(tensors: dict[str, torch.Tensor]) -> int:
        total = 0
        for key, tensor in tensors.items():
            if key.endswith(".gate_up_proj_blocks") or key.endswith(".down_proj_blocks"):
                total += 2 * tensor.numel()
            elif key.endswith("_scales"):
                continue
            else:
                total += tensor.numel()
        return total

    out_params = count_params(output)
    base_params = 0
    for shard_name in shard_names:
        base_params += count_params(load_file(str(args.base_dir / shard_name), device="cpu"))

    base_bytes = sum((args.base_dir / n).stat().st_size for n in shard_names)
    out_bytes = out_weights.stat().st_size
    report = {
        "k": args.k,
        "base_params": base_params,
        "compressed_params": out_params,
        "param_ratio": out_params / base_params,
        "base_weight_bytes": base_bytes,
        "compressed_weight_bytes": out_bytes,
        "byte_ratio": out_bytes / base_bytes,
        "keep": keep,
    }
    (args.out_dir / "compression_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
