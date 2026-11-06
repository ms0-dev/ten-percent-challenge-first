"""Collect MoE router usage on task calibration data (forward passes only).

The GPT-OSS MLP is replaced by a fused kernel at runtime, so hooks on the
router module never fire. Instead we capture the input hidden states of each
MLP with a forward pre-hook and recompute the router logits directly:

    logits = F.linear(mlp_input, router.weight, router.bias)

This is exactly the math in GptOssTopKRouter, uses the correct hidden states,
and leaves the model's fast fused forward path untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import read_jsonl, write_json
from model_runner import load_model, render_prompt
from prompts import build_banking_messages, build_qa_messages
from retrieval import BM25Index


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def register_mlp_input_hooks(model):
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args):
            captured[layer_idx] = args[0].detach()

        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_pre_hook(make_hook(i)))
    return captured, handles


def layer_router_counts(model, captured, num_layers: int, num_experts: int = 128) -> torch.Tensor:
    layer_counts = []
    with torch.no_grad():
        for layer_idx in range(num_layers):
            hidden = captured[layer_idx].reshape(-1, captured[layer_idx].shape[-1])
            router = model.model.layers[layer_idx].mlp.router
            logits = F.linear(hidden, router.weight, router.bias)
            idx = torch.topk(logits, 4, dim=-1).indices.reshape(-1)
            layer_counts.append(torch.bincount(idx.cpu(), minlength=num_experts))
    return torch.stack(layer_counts)


def collect_banking(model, tokenizer, data_dir: Path, limit: int, top_k: int, captured, num_layers: int) -> torch.Tensor:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    memory = list(read_jsonl(data_dir / "old_memory.jsonl"))[:limit]
    labels = [{"label_name": n, "label_desc": label_description(n)} for n in meta["old_intents"]]
    index = BM25Index([x["text"] for x in memory])
    counts = None
    for i, row in enumerate(memory):
        hits = index.search(row["text"], k=top_k)
        messages = build_banking_messages(row["text"], [memory[h.index] for h in hits], labels)
        prompt = render_prompt(tokenizer, messages, force_final=True)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        captured.clear()
        with torch.inference_mode():
            model(**enc)
        batch_counts = layer_router_counts(model, captured, num_layers)
        counts = batch_counts if counts is None else counts + batch_counts
        if (i + 1) % 20 == 0:
            print(f"  banking {i + 1}/{len(memory)}", flush=True)
    return counts


def collect_qa(model, tokenizer, data_dir: Path, limit: int, top_k: int, captured, num_layers: int) -> torch.Tensor:
    rows = list(read_jsonl(data_dir / "dev.jsonl"))[:limit]
    corpus = list(read_jsonl(data_dir / "corpus.jsonl"))
    index = BM25Index([x["text"] for x in corpus])
    counts = None
    for i, row in enumerate(rows):
        hits = index.search(row["question"], k=top_k)
        passages = [corpus[h.index]["text"] for h in hits]
        messages = build_qa_messages(row["question"], passages)
        prompt = render_prompt(tokenizer, messages, force_final=True)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        captured.clear()
        with torch.inference_mode():
            model(**enc)
        batch_counts = layer_router_counts(model, captured, num_layers)
        counts = batch_counts if counts is None else counts + batch_counts
        if (i + 1) % 20 == 0:
            print(f"  qa {i + 1}/{len(rows)}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--banking-dir", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--banking-limit", type=int, default=120)
    parser.add_argument("--qa-limit", type=int, default=120)
    parser.add_argument("--top-k-retrieval", type=int, default=4)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    captured, handles = register_mlp_input_hooks(model)
    num_layers = len(model.model.layers)
    print(f"Registered {len(handles)} MLP input hooks across {num_layers} layers", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    banking = collect_banking(
        model, tokenizer, args.banking_dir, args.banking_limit, args.top_k_retrieval,
        captured, num_layers,
    )
    qa = collect_qa(
        model, tokenizer, args.qa_dir, args.qa_limit, args.top_k_retrieval,
        captured, num_layers,
    )

    combined = banking + qa
    for name, counts in (("banking", banking), ("qa", qa), ("combined", combined)):
        obj = {
            "counts": counts.tolist(),
            "top_k": 4,
            "n_examples": args.banking_limit if name == "banking" else args.qa_limit,
            "source": name,
        }
        write_json(args.out_dir / f"{name}_counts.json", obj)
        print(f"{name}: layer 0 top experts {torch.topk(counts[0], 8).indices.tolist()}", flush=True)


if __name__ == "__main__":
    main()
