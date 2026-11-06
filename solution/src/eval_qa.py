"""Evaluate a model on the open-book QA dev split or the locked exam."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from model_runner import generate_batch, load_model, render_prompt
from prompts import build_qa_messages, exact_match, f1_score, normalize_answer
from retrieval import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=["dev", "exam"], default="dev")
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/squad_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("solution/results/qa"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--final-eval", action="store_true")
    args = parser.parse_args()

    if args.split == "exam" and not args.final_eval:
        raise SystemExit("Refusing to read the locked exam during development. Pass --final-eval only at the end.")

    rows = list(read_jsonl(args.data_dir / ("locked/exam.jsonl" if args.split == "exam" else "dev.jsonl")))
    if args.limit:
        rows = rows[: args.limit]
    corpus = list(read_jsonl(args.data_dir / "corpus.jsonl"))
    index = BM25Index([x["text"] for x in corpus])

    model, tokenizer = load_model(args.model_path)
    prompts: list[str] = []
    retrieved_ids: list[list[str]] = []
    for row in rows:
        hits = index.search(row["question"], k=args.top_k)
        passages = [corpus[h.index]["text"] for h in hits]
        retrieved_ids.append([corpus[h.index]["id"] for h in hits])
        messages = build_qa_messages(row["question"], passages)
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
    em_total = 0.0
    f1_total = 0.0
    contains_total = 0.0
    retrieval_hits = 0
    for row, raw, ret_ids in zip(rows, outputs, retrieved_ids):
        golds = row["answers"]
        em = max(exact_match(raw, g) for g in golds)
        f1 = max(f1_score(raw, g) for g in golds)
        pred_norm = normalize_answer(raw)
        contains = float(any(normalize_answer(g) in pred_norm for g in golds))
        em_total += em
        f1_total += f1
        contains_total += contains
        retrieval_hits += int(row.get("context_id") in ret_ids)
        predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "gold": golds,
                "pred": raw,
                "em": em,
                "f1": f1,
                "contains_gold": contains,
                "retrieved_ids": ret_ids,
                "gold_context_id": row.get("context_id"),
            }
        )

    n = len(rows)
    summary = {
        "model": args.name,
        "model_path": args.model_path,
        "benchmark": "squad_v2_open_book",
        "split": args.split,
        "n": n,
        "top_k": args.top_k,
        "em": em_total / n if n else 0.0,
        "f1": f1_total / n if n else 0.0,
        "contains_gold": contains_total / n if n else 0.0,
        "retrieval_hit_at_k": retrieval_hits / n if n else 0.0,
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
