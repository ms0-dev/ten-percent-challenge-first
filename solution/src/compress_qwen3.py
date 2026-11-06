"""Compress Qwen3-30B-A3B to K experts per layer by router usage.

The base checkpoint stores each expert as separate gate/up/down weights. We
keep the top-K experts per layer (by task router usage), rename them to
experts 0..K-1, slice the router rows, and copy every non-expert tensor.
Every kept weight is an exact copy of a base weight; this is pure surgery.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

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

    out: dict[str, object] = {}
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
                    new_key = f"model.layers.{layer}.mlp.experts.{new_expert}.{proj}.weight"
                    out[new_key] = f.get_tensor(key).clone()
                elif key.endswith(".mlp.gate.weight"):
                    layer = int(key.split(".")[2])
                    out[key] = f.get_tensor(key)[keep[layer]].clone()
                else:
                    out[key] = f.get_tensor(key).clone()
        print(f"  processed {shard_name}", flush=True)

    save_file(out, str(args.out_dir / "model.safetensors"))
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json"):
        src = args.base_dir / name
        if src.exists():
            shutil.copy2(src, args.out_dir / name)
    config["num_experts"] = args.k
    config["num_experts_per_tok"] = min(args.k, config.get("num_experts_per_tok", args.k))
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Logical parameter / byte accounting.
    out_params = sum(t.numel() for t in out.values())
    report = {
        "k": args.k,
        "base_params_expected": 30532122624,
        "compressed_params": out_params,
        "param_ratio": out_params / 30532122624,
        "base_bytes": 61079824827,
        "compressed_bytes": (args.out_dir / "model.safetensors").stat().st_size,
        "byte_ratio": (args.out_dir / "model.safetensors").stat().st_size / 61079824827,
        "keep": keep,
    }
    report["byte_ratio"] = (args.out_dir / "model.safetensors").stat().st_size / report["base_bytes"]
    (args.out_dir / "compression_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
