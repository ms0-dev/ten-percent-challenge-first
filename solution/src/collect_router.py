"""Collect MoE router usage on task calibration data (forward passes only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import read_jsonl, write_json
from model_runner import load_model, render_prompt
from prompts import build_banking_messages, build_qa_messages
from retrieval import BM25Index


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def collect_banking(model, tokenizer, data_dir: Path, limit: int, top_k: int) -> torch.Tensor:
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
        with torch.inference_mode():
            out = model(**enc, output_router_logits=True)
        layer_counts = []
        for logits in out.router_logits:
            idx = torch.topk(logits, 4, dim=-1).indices.reshape(-1)
            layer_counts.append(torch.bincount(idx.cpu(), minlength=128))
        batch_counts = torch.stack(layer_counts)
        counts = batch_counts if counts is None else counts + batch_counts
        if (i + 1) % 20 == 0:
            print(f"  banking {i + 1}/{len(memory)}")
    return counts


def collect_qa(model, tokenizer, data_dir: Path, limit: int, top_k: int) -> torch.Tensor:
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
        with torch.inference_mode():
            out = model(**enc, output_router_logits=True)
        layer_counts = []
        for logits in out.router_logits:
            idx = torch.topk(logits, 4, dim=-1).indices.reshape(-1)
            layer_counts.append(torch.bincount(idx.cpu(), minlength=128))
        batch_counts = torch.stack(layer_counts)
        counts = batch_counts if counts is None else counts + batch_counts
        if (i + 1) % 20 == 0:
            print(f"  qa {i + 1}/{len(rows)}")
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    banking = collect_banking(model, tokenizer, args.banking_dir, args.banking_limit, args.top_k_retrieval)
    qa = collect_qa(model, tokenizer, args.qa_dir, args.qa_limit, args.top_k_retrieval)

    combined = banking + qa
    for name, counts in (("banking", banking), ("qa", qa), ("combined", combined)):
        obj = {
            "counts": counts.tolist(),
            "top_k": 4,
            "n_examples": args.banking_limit if name == "banking" else args.qa_limit,
            "source": name,
        }
        write_json(args.out_dir / f"{name}_counts.json", obj)
        print(f"{name}: layer 0 top experts {torch.topk(counts[0], 8).indices.tolist()}")


if __name__ == "__main__":
    main()
