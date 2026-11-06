"""Evaluate a model on the Banking77 dev split or the locked exam.

Development must use --split dev. The locked exam is only read with
--split exam --final-eval, once, at the very end.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from model_runner import generate_batch, load_model, render_prompt
from prompts import build_banking_messages, parse_label
from retrieval import BM25Index


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=["dev", "exam"], default="dev")
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77"))
    parser.add_argument("--out-dir", type=Path, default=Path("solution/results/banking"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=0)
    parser.add_argument("--candidate-pool", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--python-experts", action="store_true")
    parser.add_argument("--batched-experts", action="store_true")
    parser.add_argument("--eager-experts", action="store_true")
    parser.add_argument("--final-eval", action="store_true")
    args = parser.parse_args()

    if args.split == "exam" and not args.final_eval:
        raise SystemExit("Refusing to read the locked exam during development. Pass --final-eval only at the end.")

    dev_path = args.data_dir / "old_dev.jsonl"
    exam_path = args.data_dir / "locked" / "old_test.jsonl"
    mem_path = args.data_dir / "old_memory.jsonl"
    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))

    rows = list(read_jsonl(exam_path if args.split == "exam" else dev_path))
    if args.limit:
        rows = rows[: args.limit]
    memory = list(read_jsonl(mem_path))
    labels = [
        {"label_name": name, "label_desc": label_description(name)}
        for name in meta["old_intents"]
    ]

    index = BM25Index([x["text"] for x in memory])

    model, tokenizer = load_model(
        args.model_path,
        python_experts=args.python_experts,
        batched_experts=args.batched_experts,
        eager_experts=args.eager_experts,
    )
    prompts: list[str] = []
    row_labels_list: list[list[dict]] = []
    for row in rows:
        search_k = max(args.top_k, args.candidate_pool) if args.candidate_k else args.top_k
        hits = index.search(row["text"], k=search_k)
        examples = [memory[h.index] for h in hits[: args.top_k]]
        if args.candidate_k:
            row_labels: list[dict] = []
            seen: set[str] = set()
            for hit in hits:
                lab = memory[hit.index]
                name = lab["label_name"]
                if name in seen:
                    continue
                seen.add(name)
                row_labels.append({
                    "label_name": name,
                    "label_desc": lab.get("label_desc") or label_description(name),
                })
                if len(row_labels) >= args.candidate_k:
                    break
            if len(row_labels) < 2:
                row_labels = labels
        else:
            row_labels = labels
        row_labels_list.append(row_labels)
        messages = build_banking_messages(row["text"], examples, row_labels)
        prompts.append(render_prompt(tokenizer, messages, force_final=True))

    outputs = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    predictions = []
    correct = 0
    parse_failures = 0
    candidate_recall_total = 0
    for row, raw, row_labels in zip(rows, outputs, row_labels_list):
        valid_labels = [x["label_name"] for x in row_labels]
        pred = parse_label(raw, valid_labels)
        ok = pred == row["label_name"]
        candidate_recall = row["label_name"] in valid_labels
        correct += int(ok)
        candidate_recall_total += int(candidate_recall)
        parse_failures += int(pred is None)
        predictions.append(
            {
                "id": row["id"],
                "text": row["text"],
                "gold": row["label_name"],
                "pred": pred,
                "correct": ok,
                "raw": raw,
                "candidates": valid_labels,
                "candidate_recall": candidate_recall,
            }
        )

    accuracy = correct / len(rows) if rows else 0.0
    summary = {
        "model": args.name,
        "model_path": args.model_path,
        "benchmark": "banking77",
        "split": args.split,
        "n": len(rows),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "candidate_pool": args.candidate_pool,
        "candidate_recall": candidate_recall_total / len(rows) if rows else 0.0,
        "accuracy": accuracy,
        "correct": correct,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_eval": args.final_eval,
    }

    out_dir = args.out_dir
    if args.split == "exam":
        out_dir = out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{args.split}"
    write_jsonl(out_dir / f"{stem}.jsonl", predictions)
    write_json(out_dir / f"{stem}.summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / (stem + '.jsonl')}")


if __name__ == "__main__":
    main()
