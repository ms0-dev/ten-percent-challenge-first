"""Write task-style calibration text for llama.cpp imatrix quantization.

The text is the exact prompt format used by the Banking77 evaluation plus the
gold label. Only development and memory rows are used; the locked exam is not
touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl
from model_runner import render_prompt
from prompts import build_banking_messages
from retrieval import BM25Index


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77_10class"))
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    memory = list(read_jsonl(args.data_dir / "old_memory.jsonl"))
    rows = list(read_jsonl(args.data_dir / "old_dev.jsonl"))[: args.limit]
    labels = [
        {"label_name": name, "label_desc": label_description(name)}
        for name in meta["old_intents"]
    ]
    index = BM25Index([x["text"] for x in memory])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    chunks: list[str] = []
    for row in rows:
        hits = index.search(row["text"], k=args.top_k)
        examples = [memory[h.index] for h in hits[: args.top_k]]
        messages = build_banking_messages(row["text"], examples, labels)
        prompt = render_prompt(tokenizer, messages)
        chunks.append(f"{prompt}{row['label_name']}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes, {len(chunks)} prompts)")


if __name__ == "__main__":
    main()
