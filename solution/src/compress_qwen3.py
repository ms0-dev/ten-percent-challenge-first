"""Compress Qwen3-30B-A3B to K experts per layer by router usage.

This writes the model's native 3D expert format:

  model.layers.i.mlp.experts.gate_up_proj  [K, 2*intermediate, hidden]
  model.layers.i.mlp.experts.down_proj     [K, hidden, intermediate]

The base HF checkpoint stores per-expert gate/up/down keys. We select the
top-K experts per layer, concatenate gate+up, stack the experts, slice the
router rows, and copy every non-expert tensor. All kept weights are exact
copies of base weights.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--router-usage", type=Path, required=True)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.base_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((args.base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    counts = json.loads(args.router_usage.read_text(encoding="utf-8"))["counts"]
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_experts"]
    keep = [
        sorted(range(num_experts), key=lambda e: counts[layer][e], reverse=True)[: args.k]
        for layer in range(num_layers)
    ]
    print(f"Keeping top {args.k} of {num_experts} experts per layer", flush=True)
    for layer in range(min(3, num_layers)):
        print(f"  layer {layer}: {keep[layer]}", flush=True)

    # Per-layer selected expert tensors, keyed by new expert index.
    selected: dict[int, dict[int, dict[str, torch.Tensor]]] = {
        layer: {new: {} for new in range(args.k)} for layer in range(num_layers)
    }
    out: dict[str, torch.Tensor] = {}

    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        with safe_open(str(args.base_dir / shard_name), framework="pt", device="cpu") as f:
            for key in f.keys():
                m = EXPERT_RE.match(key)
                if m:
                    layer = int(m.group(1))
                    orig_expert = int(m.group(2))
                    proj = m.group(3)
                    if orig_expert not in keep[layer]:
                        continue
                    new_expert = keep[layer].index(orig_expert)
                    selected[layer][new_expert][proj] = f.get_tensor(key).clone()
                elif key.endswith(".mlp.gate.weight"):
                    layer = int(key.split(".")[2])
                    out[key] = f.get_tensor(key)[keep[layer]].clone()
                else:
                    out[key] = f.get_tensor(key).clone()
        print(f"  processed {shard_name}", flush=True)

    for layer in range(num_layers):
        gate_up = []
        down = []
        for new_expert in range(args.k):
            parts = selected[layer][new_expert]
            gate_up.append(torch.cat([parts["gate_proj"], parts["up_proj"]], dim=0))
            down.append(parts["down_proj"])
        out[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.stack(gate_up)
        out[f"model.layers.{layer}.mlp.experts.down_proj"] = torch.stack(down)
        selected[layer] = {}

    save_file(out, str(args.out_dir / "model.safetensors"))
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json"):
        src = args.base_dir / name
        if src.exists():
            shutil.copy2(src, args.out_dir / name)
    config["num_experts"] = args.k
    config["num_experts_per_tok"] = min(args.k, config.get("num_experts_per_tok", args.k))
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    out_params = sum(t.numel() for t in out.values())
    base_bytes = 61079824827
    compressed_bytes = (args.out_dir / "model.safetensors").stat().st_size
    report = {
        "k": args.k,
        "base_params_expected": 30532122624,
        "compressed_params": out_params,
        "param_ratio": out_params / 30532122624,
        "base_bytes": base_bytes,
        "compressed_bytes": compressed_bytes,
        "byte_ratio": compressed_bytes / base_bytes,
        "keep": keep,
        "expert_format": "3d_gate_up_down",
    }
    (args.out_dir / "compression_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
