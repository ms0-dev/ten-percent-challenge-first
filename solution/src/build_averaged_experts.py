"""Build a BF16 GPT-OSS model with K averaged/clustered experts per layer.

For each layer:
1. dequantize the 128 MXFP4 experts to BF16,
2. cluster the 128 router rows into K groups (k-means),
3. average the experts in each group (weighted by calibration usage),
4. merge the router rows of each group (weighted mean).

The output is a standard BF16 GptOssForCausalLM checkpoint with K experts.
It is a build artifact for measuring the merged-expert quality; the final
proof will also produce a small MXFP4 checkpoint from it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

EXPERT_PREFIX = "model.layers."
EXPERT_SUFFIXES = (
    "mlp.experts.gate_up_proj_blocks",
    "mlp.experts.gate_up_proj_scales",
    "mlp.experts.gate_up_proj_bias",
    "mlp.experts.down_proj_blocks",
    "mlp.experts.down_proj_scales",
    "mlp.experts.down_proj_bias",
)
ROUTER_SUFFIXES = ("mlp.router.weight", "mlp.router.bias")


def layer_of(key: str) -> int | None:
    if not key.startswith(EXPERT_PREFIX):
        return None
    rest = key[len(EXPERT_PREFIX) :]
    head = rest.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def kmeans_rows(x: torch.Tensor, k: int, seed: int, iters: int = 30) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    idx = torch.randperm(n, generator=g)[:k]
    centers = x[idx].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        dist = torch.cdist(x, centers)
        new_assign = dist.argmin(dim=1)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            members = x[assign == j]
            if members.numel():
                centers[j] = members.mean(dim=0)
    return assign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--router-usage", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = args.base_dir
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_local_experts"]
    usage = json.loads(args.router_usage.read_text(encoding="utf-8"))["counts"]
    counts = torch.tensor(usage, dtype=torch.float32)

    # Read the router rows for clustering.
    router_w: dict[int, torch.Tensor] = {}
    router_b: dict[int, torch.Tensor] = {}
    for layer in range(num_layers):
        for suffix, store in (("mlp.router.weight", router_w), ("mlp.router.bias", router_b)):
            key = f"model.layers.{layer}.{suffix}"
            shard = base / weight_map[key]
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                store[layer] = f.get_tensor(key).float().clone()

    groups: dict[int, torch.Tensor] = {}
    merged_router_w: dict[int, torch.Tensor] = {}
    merged_router_b: dict[int, torch.Tensor] = {}
    group_weights: dict[int, torch.Tensor] = {}
    for layer in range(num_layers):
        w = router_w[layer]
        x = w / w.norm(dim=1, keepdim=True).clamp_min(1e-6)
        assign = kmeans_rows(x, args.k, args.seed + layer)
        groups[layer] = assign
        c = counts[layer]
        mw = torch.zeros(args.k, w.shape[1])
        mb = torch.zeros(args.k)
        gw = torch.zeros(args.k)
        for j in range(args.k):
            members = (assign == j).nonzero(as_tuple=False).flatten()
            weights = c[members]
            if weights.sum() <= 0:
                weights = torch.ones_like(weights)
            weights = weights / weights.sum()
            mw[j] = (w[members] * weights[:, None]).sum(dim=0)
            mb[j] = (router_b[layer][members] * weights).sum()
            gw[j] = weights.sum()
        merged_router_w[layer] = mw
        merged_router_b[layer] = mb
        group_weights[layer] = gw

    print(f"Clustered {num_layers} layers into {args.k} groups", flush=True)

    # Copy all non-expert tensors first.
    out: dict[str, torch.Tensor] = {}
    shard_names = sorted(set(weight_map.values()))
    skip_suffixes = EXPERT_SUFFIXES + ROUTER_SUFFIXES
    for shard_name in shard_names:
        with safe_open(str(base / shard_name), framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(key.endswith(suffix) for suffix in skip_suffixes):
                    continue
                out[key] = f.get_tensor(key).clone()
        print(f"  copied non-expert shard {shard_name}", flush=True)

    # Dequantize, average, and write each layer's experts.
    for layer in range(num_layers):
        keys = {
            suffix: f"model.layers.{layer}.{suffix}"
            for suffix in EXPERT_SUFFIXES
        }
        tensors = {}
        for suffix, key in keys.items():
            shard = base / weight_map[key]
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                tensors[suffix] = f.get_tensor(key).clone()

        gu = convert_moe_packed_tensors(
            tensors["mlp.experts.gate_up_proj_blocks"],
            tensors["mlp.experts.gate_up_proj_scales"],
        )  # [E, hidden, 2*intermediate]
        down = convert_moe_packed_tensors(
            tensors["mlp.experts.down_proj_blocks"],
            tensors["mlp.experts.down_proj_scales"],
        )  # [E, intermediate, hidden]
        gu_bias = tensors["mlp.experts.gate_up_proj_bias"].float()
        down_bias = tensors["mlp.experts.down_proj_bias"].float()

        assign = groups[layer]
        c = counts[layer]
        gu_cent = torch.zeros(args.k, *gu.shape[1:], dtype=torch.bfloat16)
        down_cent = torch.zeros(args.k, *down.shape[1:], dtype=torch.bfloat16)
        gu_bias_cent = torch.zeros(args.k, gu_bias.shape[1], dtype=torch.bfloat16)
        down_bias_cent = torch.zeros(args.k, down_bias.shape[1], dtype=torch.bfloat16)
        for j in range(args.k):
            members = (assign == j).nonzero(as_tuple=False).flatten()
            if members.numel() == 0:
                raise RuntimeError(f"empty cluster at layer {layer}, group {j}; change seed or K")
            weights = c[members]
            if weights.sum() <= 0:
                weights = torch.ones_like(weights)
            weights = weights / weights.sum()
            w = weights[:, None, None]
            gu_cent[j] = (gu[members].float() * w).sum(dim=0).to(torch.bfloat16)
            down_cent[j] = (down[members].float() * w).sum(dim=0).to(torch.bfloat16)
            gu_bias_cent[j] = (gu_bias[members] * weights[:, None]).sum(dim=0).to(torch.bfloat16)
            down_bias_cent[j] = (down_bias[members] * weights[:, None]).sum(dim=0).to(torch.bfloat16)

        out[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = gu_cent
        out[f"model.layers.{layer}.mlp.experts.gate_up_proj_bias"] = gu_bias_cent
        out[f"model.layers.{layer}.mlp.experts.down_proj"] = down_cent
        out[f"model.layers.{layer}.mlp.experts.down_proj_bias"] = down_bias_cent
        out[f"model.layers.{layer}.mlp.router.weight"] = merged_router_w[layer].to(torch.bfloat16)
        out[f"model.layers.{layer}.mlp.router.bias"] = merged_router_b[layer].to(torch.bfloat16)
        del gu, down, gu_bias, down_bias, tensors
        print(f"  layer {layer + 1}/{num_layers}: averaged {args.k} experts", flush=True)

    save_file(out, str(args.out_dir / "model.safetensors"))
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        src = base / name
        if src.exists():
            shutil.copy2(src, args.out_dir / name)
    config["num_local_experts"] = args.k
    config["num_experts_per_tok"] = min(4, args.k)
    config.pop("quantization_config", None)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    report = {
        "k": args.k,
        "num_layers": num_layers,
        "router_merge": "usage_weighted_mean",
        "expert_merge": "usage_weighted_mean",
        "groups": {str(k): v.tolist() for k, v in groups.items()},
        "group_weights": {str(k): v.tolist() for k, v in group_weights.items()},
    }
    (args.out_dir / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
