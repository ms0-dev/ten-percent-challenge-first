"""Generic retrieval-augmented classification evaluation for growth demos.

Used to measure:
* new-intent accuracy before feedback (old memory, old label space)
* new-intent accuracy after feedback (combined memory, full label space)
* old-intent accuracy before/after feedback (old label space)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from model_runner import generate_batch, load_model, render_prompt
from prompts import build_banking_messages, parse_label
from retrieval import BM25Index


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def build_labels(kind: str, meta: dict, memory: list[dict]) -> list[dict]:
    if kind == "old":
        names = meta["old_intents"]
        return [{"label_name": n, "label_desc": label_description(n)} for n in names]
    if kind == "all":
        names = meta["old_intents"] + meta["new_intents"]
        return [{"label_name": n, "label_desc": label_description(n)} for n in names]
    if kind == "from-memory":
        seen: dict[str, str] = {}
        for row in memory:
            seen.setdefault(row["label_name"], row.get("label_desc") or label_description(row["label_name"]))
        return [{"label_name": n, "label_desc": d} for n, d in sorted(seen.items())]
    raise SystemExit(f"Unknown labels kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--labels", choices=["old", "all", "from-memory"], required=True)
    parser.add_argument("--meta", type=Path, default=Path("solution/data/banking77/meta.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--eager-experts", action="store_true")
    parser.add_argument("--final-eval", action="store_true")
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    memory = list(read_jsonl(args.memory))
    rows = list(read_jsonl(args.rows))
    if args.limit:
        rows = rows[: args.limit]
    labels = build_labels(args.labels, meta, memory)
    valid_labels = [x["label_name"] for x in labels]
    index = BM25Index([x["text"] for x in memory])

    model, tokenizer = load_model(args.model_path, eager_experts=args.eager_experts)
    prompts: list[str] = []
    for row in rows:
        hits = index.search(row["text"], k=args.top_k)
        examples = [memory[h.index] for h in hits]
        messages = build_banking_messages(row["text"], examples, labels)
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
    for row, raw in zip(rows, outputs):
        pred = parse_label(raw, valid_labels)
        gold = row.get("label_name") or row.get("label")
        ok = pred == gold
        correct += int(ok)
        parse_failures += int(pred is None)
        predictions.append({
            "id": row.get("id"),
            "text": row["text"],
            "gold": gold,
            "pred": pred,
            "correct": ok,
            "raw": raw,
        })

    summary = {
        "model": args.name,
        "model_path": args.model_path,
        "memory": str(args.memory),
        "rows": str(args.rows),
        "labels": args.labels,
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_eval": args.final_eval,
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{args.rows.stem}_{args.labels}"
    write_jsonl(out_dir / f"{stem}.jsonl", predictions)
    write_json(out_dir / f"{stem}.summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
