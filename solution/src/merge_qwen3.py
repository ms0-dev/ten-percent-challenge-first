"""Compress Qwen3-30B-A3B to K merged experts per layer (no weight training).

Instead of dropping 121 of 128 experts, cluster all 128 experts into K groups
and store one centroid per group. Every expert contributes to exactly one
centroid, so no routing mass is discarded. The router is rebuilt as the
usage-weighted mean of the member router rows.

All output tensors are deterministic summaries of the base weights:
  - expert centroid = usage-weighted mean of member expert tensors
  - router row      = usage-weighted mean of member router rows
  - everything else = exact copy of the base tensor
No gradients or optimizer steps are involved.
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
GLOBAL_COPY = (
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
)


def weighted_kmeans(
    x: torch.Tensor,
    k: int,
    weights: torch.Tensor,
    iterations: int = 12,
    seed: int = 0,
) -> torch.Tensor:
    """Weighted cosine k-means. Returns an int assignment per row."""
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    # k-means++ initialization using weighted distances.
    first = int(torch.argmax(weights * xn.norm(dim=1)))
    centers = [xn[first]]
    closest = 1.0 - xn @ xn[first]
    for _ in range(1, k):
        score = weights * closest.clamp_min(0)
        if float(score.sum()) <= 0:
            idx = int(torch.randint(0, n, (1,), generator=gen))
        else:
            idx = int(torch.multinomial(score / score.sum(), 1, generator=gen))
        centers.append(xn[idx])
        closest = torch.minimum(closest, 1.0 - xn @ xn[idx])
    c = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iterations):
        sim = xn @ c.T
        new_assign = sim.argmax(dim=1)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            mask = assign == j
            if not bool(mask.any()):
                # Re-seed the empty cluster with the worst-fit point.
                worst = int(torch.argmax((1.0 - sim.max(dim=1).values) * weights))
                assign[worst] = j
                mask = assign == j
            w = weights[mask].unsqueeze(0)
            c[j] = (xn[mask] * w.T).sum(dim=0) / w.sum().clamp_min(1e-12)
        c = c / c.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return assign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--router-usage", type=Path, required=True)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--variant", choices=["weight", "router"], default="weight")
    parser.add_argument("--iterations", type=int, default=12)
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

    handles: dict[str, object] = {}

    def shard(name: str):
        handle = handles.get(name)
        if handle is None:
            handle = safe_open(str(args.base_dir / name), framework="pt", device="cpu")
            handles[name] = handle
        return handle

    def read(key: str) -> torch.Tensor:
        return shard(weight_map[key]).get_tensor(key)

    out: dict[str, torch.Tensor] = {}
    assignments: list[list[int]] = []
    for layer in range(num_layers):
        usage = torch.tensor(counts[layer], dtype=torch.float32).clamp_min(1.0)
        usage = usage / usage.sum()
        gate_up = torch.empty((num_experts, 2 * config["moe_intermediate_size"], config["hidden_size"]), dtype=torch.bfloat16)
        down = torch.empty((num_experts, config["hidden_size"], config["moe_intermediate_size"]), dtype=torch.bfloat16)
        router = torch.empty((num_experts, config["hidden_size"]), dtype=torch.bfloat16)
        for expert in range(num_experts):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gate_up[expert, : config["moe_intermediate_size"]] = read(f"{prefix}.gate_proj.weight")
            gate_up[expert, config["moe_intermediate_size"] :] = read(f"{prefix}.up_proj.weight")
            down[expert] = read(f"{prefix}.down_proj.weight")
        router = read(f"model.layers.{layer}.mlp.gate.weight")

        if args.variant == "weight":
            features = gate_up.flatten(1).float()
        else:
            features = router.float()
        assign = weighted_kmeans(features, args.k, usage, iterations=args.iterations, seed=layer)
        assignments.append(assign.tolist())

        new_gate_up = torch.empty((args.k, gate_up.shape[1], gate_up.shape[2]), dtype=torch.bfloat16)
        new_down = torch.empty((args.k, down.shape[1], down.shape[2]), dtype=torch.bfloat16)
        new_router = torch.empty((args.k, router.shape[1]), dtype=torch.bfloat16)
        for cluster in range(args.k):
            mask = assign == cluster
            w = usage[mask]
            w = w / w.sum().clamp_min(1e-12)
            new_gate_up[cluster] = (gate_up[mask].float() * w[:, None, None]).sum(dim=0)
            new_down[cluster] = (down[mask].float() * w[:, None, None]).sum(dim=0)
            new_router[cluster] = (router[mask].float() * w[:, None]).sum(dim=0)
        out[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = new_gate_up
        out[f"model.layers.{layer}.mlp.experts.down_proj"] = new_down
        out[f"model.layers.{layer}.mlp.gate.weight"] = new_router
        if layer % 4 == 0:
            print(f"merged layer {layer} ({args.variant})", flush=True)
        del gate_up, down, router, features, new_gate_up, new_down, new_router

    for shard_name in sorted(set(weight_map.values())):
        for key in shard(shard_name).keys():
            if key in out:
                continue
            if EXPERT_RE.match(key) or key.endswith(".mlp.gate.weight"):
                continue
            out[key] = shard(shard_name).get_tensor(key).clone()
        print(f"copied non-expert tensors from {shard_name}", flush=True)

    save_file(out, str(args.out_dir / "model.safetensors"))
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
    ):
        src = args.base_dir / name
        if src.exists():
            shutil.copy2(src, args.out_dir / name)
    config["num_experts"] = args.k
    config["num_experts_per_tok"] = min(args.k, config.get("num_experts_per_tok", args.k))
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    out_params = sum(t.numel() for t in out.values())
    compressed_bytes = (args.out_dir / "model.safetensors").stat().st_size
    report = {
        "k": args.k,
        "variant": args.variant,
        "iterations": args.iterations,
        "base_params_expected": 30532122624,
        "compressed_params": out_params,
        "param_ratio": out_params / 30532122624,
        "base_bytes": 61079824827,
        "compressed_bytes": compressed_bytes,
        "byte_ratio": compressed_bytes / 61079824827,
        "assignments": assignments,
        "method": "usage-weighted k-means merge of all 128 experts per layer",
    }
    (args.out_dir / "compression_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "assignments"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
