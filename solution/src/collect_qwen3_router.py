"""Collect Qwen3-30B router usage on the grouped banking task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import read_jsonl, write_json
from model_runner import load_model, render_prompt
from prompts import build_banking_messages
from retrieval import BM25Index


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--top-k-retrieval", type=int, default=4)
    parser.add_argument("--top-k-router", type=int, default=8)
    args = parser.parse_args()

    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    memory = list(read_jsonl(args.data_dir / "old_memory.jsonl"))[: args.limit]
    labels = [
        {"label_name": n, "label_desc": label_description(n)}
        for n in meta["old_intents"]
    ]
    index = BM25Index([x["text"] for x in memory])

    model, tokenizer = load_model(args.model_path)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, hook_args):
            captured[layer_idx] = hook_args[0].detach()

        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_pre_hook(make_hook(i)))
    num_layers = len(model.model.layers)
    print(f"Registered {len(handles)} hooks over {num_layers} layers", flush=True)

    counts = None
    for i, row in enumerate(memory):
        hits = index.search(row["text"], k=args.top_k_retrieval)
        examples = [memory[h.index] for h in hits]
        messages = build_banking_messages(row["text"], examples, labels)
        prompt = render_prompt(tokenizer, messages, force_final=True)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        captured.clear()
        with torch.inference_mode():
            model(**enc)
        layer_counts = []
        with torch.no_grad():
            for layer_idx in range(num_layers):
                hidden = captured[layer_idx].reshape(-1, captured[layer_idx].shape[-1])
                gate = model.model.layers[layer_idx].mlp.gate.weight
                logits = F.linear(hidden, gate)
                idx = torch.topk(logits, args.top_k_router, dim=-1).indices.reshape(-1)
                layer_counts.append(torch.bincount(idx.cpu(), minlength=gate.shape[0]))
        batch = torch.stack(layer_counts)
        counts = batch if counts is None else counts + batch
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(memory)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, {
        "counts": counts.tolist(),
        "top_k": args.top_k_router,
        "n_examples": len(memory),
        "source": str(args.data_dir),
    })
    print(f"Wrote {args.out}; layer 0 top experts {torch.topk(counts[0], 12).indices.tolist()}", flush=True)


if __name__ == "__main__":
    main()
