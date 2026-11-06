"""Evaluate a llama.cpp server on the Banking77 dev split or the locked exam.

The prompt construction, retrieval, label parsing, and output schema match
`eval_banking.py` exactly. Inference is a local llama.cpp server so the
compressed GGUF artifact is evaluated directly.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from prompts import build_banking_messages, parse_label
from retrieval import BM25Index


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
    except Exception:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def complete(port: int, prompt: str, max_new_tokens: int, timeout: float = 600.0) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "temperature": 0.0,
            "n_predict": max_new_tokens,
            "stream": False,
            "cache_prompt": True,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result.get("content", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", choices=["dev", "exam"], default="dev")
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77"))
    parser.add_argument("--out-dir", type=Path, default=Path("solution/results/banking"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--final-eval", action="store_true")
    args = parser.parse_args()

    if args.split == "exam" and not args.final_eval:
        raise SystemExit("Refusing to read the locked exam during development. Pass --final-eval only at the end.")

    from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    predictions = []
    correct = 0
    parse_failures = 0
    started = time.time()
    for n, row in enumerate(rows, 1):
        hits = index.search(row["text"], k=args.top_k)
        examples = [memory[h.index] for h in hits[: args.top_k]]
        messages = build_banking_messages(row["text"], examples, labels)
        prompt = render_prompt(tokenizer, messages)
        raw = complete(args.port, prompt, args.max_new_tokens)
        pred = parse_label(raw, [x["label_name"] for x in labels])
        ok = pred == row["label_name"]
        correct += int(ok)
        parse_failures += int(pred is None)
        predictions.append(
            {
                "id": row["id"],
                "text": row["text"],
                "gold": row["label_name"],
                "pred": pred,
                "correct": ok,
                "raw": raw,
                "candidates": [x["label_name"] for x in labels],
                "candidate_recall": True,
            }
        )
        if n % 25 == 0:
            print(f"{n}/{len(rows)} correct={correct} failures={parse_failures} elapsed={time.time() - started:.1f}s", flush=True)

    accuracy = correct / len(rows) if rows else 0.0
    summary = {
        "model": args.name,
        "model_path": args.model_path,
        "benchmark": "banking77",
        "split": args.split,
        "n": len(rows),
        "top_k": args.top_k,
        "candidate_k": 0,
        "candidate_pool": 30,
        "candidate_recall": 1.0,
        "accuracy": accuracy,
        "correct": correct,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_eval": args.final_eval,
        "elapsed_seconds": time.time() - started,
    }

    out_dir = args.out_dir
    if args.split == "exam":
        out_dir = out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{args.split}"
    write_jsonl(out_dir / f"{stem}.jsonl", predictions)
    write_json(out_dir / f"{stem}.summary.json", summary)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out_dir / (stem + '.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
