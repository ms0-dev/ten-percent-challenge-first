"""Open-book QA as four-way answer selection.

Given a question, retrieved context passages, and four answer options (one
gold + three distractors from the training/dev answer pool), the model returns
only the option letter. This avoids span-formatting noise and gives a clean
accuracy metric.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from model_runner import generate_batch, load_model, render_prompt
from prompts import build_qa_mc_messages, normalize_answer, parse_option
from retrieval import BM25Index


def choose_options(row: dict, pool: list[str]) -> tuple[list[str], int]:
    gold = row["answers"][0]
    gold_norm = normalize_answer(gold)
    rng = random.Random(str(row["id"]))

    def usable(candidate: str) -> bool:
        cand_norm = normalize_answer(candidate)
        if not cand_norm or cand_norm == gold_norm:
            return False
        if cand_norm in gold_norm or gold_norm in cand_norm:
            return False
        return True

    # Prefer distractors of similar length, then fall back to the full pool.
    gold_len = max(1, len(gold.split()))
    similar = [
        x for x in pool
        if usable(x) and 0.5 <= len(x.split()) / gold_len <= 2.0
    ]
    candidates = similar if len(similar) >= 3 else [x for x in pool if usable(x)]
    rng.shuffle(candidates)
    distractors: list[str] = []
    seen = {gold_norm}
    for candidate in candidates:
        norm = normalize_answer(candidate)
        if norm in seen:
            continue
        distractors.append(candidate)
        seen.add(norm)
        if len(distractors) == 3:
            break
    if len(distractors) < 3:
        raise RuntimeError(f"Not enough distractors for {row['id']}")

    options = [gold] + distractors
    rng.shuffle(options)
    gold_index = options.index(gold)
    return options, gold_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=["dev", "exam"], default="dev")
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/squad_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("solution/results/qa_mc"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--python-experts", action="store_true")
    parser.add_argument("--batched-experts", action="store_true")
    parser.add_argument("--eager-experts", action="store_true")
    parser.add_argument("--final-eval", action="store_true")
    args = parser.parse_args()

    if args.split == "exam" and not args.final_eval:
        raise SystemExit("Refusing to read the locked exam during development. Pass --final-eval only at the end.")

    rows = list(read_jsonl(args.data_dir / ("locked/exam.jsonl" if args.split == "exam" else "dev.jsonl")))
    if args.limit:
        rows = rows[: args.limit]
    corpus = list(read_jsonl(args.data_dir / "corpus.jsonl"))
    pool = [a for row in read_jsonl(args.data_dir / "dev.jsonl") for a in row["answers"]]
    index = BM25Index([x["text"] for x in corpus])

    prepared = []
    for row in rows:
        hits = index.search(row["question"], k=args.top_k)
        passages = [corpus[h.index]["text"] for h in hits]
        options, gold_index = choose_options(row, pool)
        prepared.append((row, passages, options, gold_index))

    model, tokenizer = load_model(
        args.model_path,
        python_experts=args.python_experts,
        batched_experts=args.batched_experts,
        eager_experts=args.eager_experts,
    )
    prompts = [
        render_prompt(tokenizer, build_qa_mc_messages(row["question"], passages, options), force_final=True)
        for row, passages, options, _ in prepared
    ]
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
    for (row, passages, options, gold_index), raw in zip(prepared, outputs):
        pred_index = parse_option(raw, options)
        ok = pred_index == gold_index
        correct += int(ok)
        parse_failures += int(pred_index is None)
        predictions.append({
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["answers"][0],
            "options": options,
            "gold_index": gold_index,
            "pred_index": pred_index,
            "pred_letter": chr(65 + pred_index) if pred_index is not None else None,
            "correct": ok,
            "raw": raw,
            "retrieved_ids": [corpus[h.index]["id"] for h in index.search(row["question"], k=args.top_k)],
            "gold_context_id": row.get("context_id"),
        })

    n = len(rows)
    summary = {
        "model": args.name,
        "model_path": args.model_path,
        "benchmark": "squad_v2_open_book_mc",
        "split": args.split,
        "n": n,
        "top_k": args.top_k,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / n if n else 0.0,
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


if __name__ == "__main__":
    main()
