"""Prepare a small open-book QA pilot from SQuAD v2.

The corpus is the set of unique paragraphs from the train and validation
splits. Dev questions come from the train split; the locked exam comes from
the validation split. The locked exam file is hashed and never printed.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from common import sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("solution/data/squad_v2"))
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--exam-size", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out
    locked = out / "locked"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite")

    ds = load_dataset("rajpurkar/squad_v2")

    def answerable(rows):
        for row in rows:
            texts = row["answers"]["text"]
            if texts:
                yield row

    train_rows = list(answerable(ds["train"]))
    val_rows = list(answerable(ds["validation"]))

    # Build a de-duplicated paragraph corpus from both splits. The corpus is
    # the knowledge base; it is not the exam label file.
    contexts: dict[str, int] = {}
    corpus: list[dict] = []
    for split_rows in (train_rows, val_rows):
        for row in split_rows:
            ctx = row["context"]
            if ctx not in contexts:
                idx = len(corpus)
                contexts[ctx] = idx
                corpus.append({"id": f"squad-ctx-{idx}", "text": ctx})

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    dev_rows = train_rows[: args.dev_size]
    exam_rows = val_rows[: args.exam_size]

    def to_qa_row(row: dict, qid: str) -> dict:
        return {
            "id": qid,
            "question": row["question"],
            "answers": list(row["answers"]["text"]),
            "context_id": f"squad-ctx-{contexts[row['context']]}",
            "title": row["title"],
        }

    dev = [to_qa_row(row, f"squad-dev-{i}") for i, row in enumerate(dev_rows)]
    exam = [to_qa_row(row, f"squad-exam-{i}") for i, row in enumerate(exam_rows)]

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "dev.jsonl", dev)

    locked.mkdir(parents=True, exist_ok=True)
    write_jsonl(locked / "exam.jsonl", exam)
    digest = sha256_file(locked / "exam.jsonl")

    meta = {
        "dataset": "rajpurkar/squad_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "counts": {
            "corpus": len(corpus),
            "dev": len(dev),
            "exam": len(exam),
        },
        "locked_sha256": {"exam.jsonl": digest},
    }
    write_json(out / "meta.json", meta)
    write_json(locked / "manifest.json", {
        "dataset": "rajpurkar/squad_v2",
        "created_at": meta["created_at"],
        "sha256": {"exam.jsonl": digest},
        "note": "Locked before model development; do not read until final evaluation.",
    })

    print(f"Wrote {out}")
    print(f"Corpus paragraphs: {len(corpus)}")
    print(f"Dev questions: {len(dev)} | Exam questions (locked): {len(exam)}")
    print(f"Locked SHA-256 exam.jsonl: {digest}")


if __name__ == "__main__":
    main()
