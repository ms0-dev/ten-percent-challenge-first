"""Create deterministic Banking77 splits for the 10% challenge.

Design
------
* 10 intents are held out as "new" intents for the user-feedback growth demo.
* The main task is the remaining 67 intents.
* The 67-intent train split is split 80/20 into retrieval memory and dev.
* The locked exam is the full official test split of the 67 intents.
* The new-intent test split is locked separately and is only used for the
  before/after feedback demo.

The locked files are written and hashed. This script never prints their
contents. Do not open them until the final evaluation.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from common import sha256_file, write_json, write_jsonl


def label_description(name: str) -> str:
    desc = name.replace("_", " ").replace("?", "").strip()
    return desc[:1].upper() + desc[1:]


def split_train_dev(rows: list[dict], dev_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified per-label split, deterministic for a given seed."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label_name"]].append(row)

    rng = random.Random(seed)
    dev: list[dict] = []
    memory: list[dict] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        n_dev = max(1, round(len(items) * dev_fraction))
        dev.extend(items[:n_dev])
        memory.extend(items[n_dev:])

    rng.shuffle(dev)
    rng.shuffle(memory)
    return dev, memory


def sample_feedback(rows: list[dict], per_intent: int, seed: int) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label_name"]].append(row)

    rng = random.Random(seed)
    out: list[dict] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        out.extend(items[:per_intent])
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("solution/data/banking77"))
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--new-intents", type=int, default=10)
    parser.add_argument("--feedback-per-intent", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out
    locked = out / "locked"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite")

    ds = load_dataset("PolyAI/banking77")
    label_names: list[str] = list(ds["train"].features["label"].names)

    rng = random.Random(args.seed)
    new_intents = sorted(rng.sample(sorted(label_names), args.new_intents))
    old_intents = [x for x in label_names if x not in set(new_intents)]

    def to_row(split: str, index: int, row: dict) -> dict:
        label_name = label_names[row["label"]]
        return {
            "id": f"banking77-{split}-{index}",
            "text": row["text"],
            "label": label_name,
            "label_name": label_name,
            "label_desc": label_description(label_name),
        }

    old_train: list[dict] = []
    new_train: list[dict] = []
    old_test: list[dict] = []
    new_test: list[dict] = []
    new_set = set(new_intents)

    for i, row in enumerate(ds["train"]):
        target = new_train if label_names[row["label"]] in new_set else old_train
        target.append(to_row("train", i, row))
    for i, row in enumerate(ds["test"]):
        target = new_test if label_names[row["label"]] in new_set else old_test
        target.append(to_row("test", i, row))

    dev_old, memory_old = split_train_dev(old_train, args.dev_fraction, args.seed)
    feedback_new = sample_feedback(new_train, args.feedback_per_intent, args.seed)

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "old_memory.jsonl", memory_old)
    write_jsonl(out / "old_dev.jsonl", dev_old)
    write_jsonl(out / "new_feedback.jsonl", feedback_new)

    # Locked files. Write but never print contents.
    locked.mkdir(parents=True, exist_ok=True)
    write_jsonl(locked / "old_test.jsonl", old_test)
    write_jsonl(locked / "new_test.jsonl", new_test)

    hashes = {
        "old_test.jsonl": sha256_file(locked / "old_test.jsonl"),
        "new_test.jsonl": sha256_file(locked / "new_test.jsonl"),
    }

    meta = {
        "dataset": "PolyAI/banking77",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "feedback_per_intent": args.feedback_per_intent,
        "new_intents": new_intents,
        "old_intents": old_intents,
        "counts": {
            "old_train": len(old_train),
            "old_memory": len(memory_old),
            "old_dev": len(dev_old),
            "old_test": len(old_test),
            "new_train": len(new_train),
            "new_feedback": len(feedback_new),
            "new_test": len(new_test),
        },
        "locked_sha256": hashes,
    }
    write_json(out / "meta.json", meta)
    write_json(locked / "manifest.json", {
        "dataset": "PolyAI/banking77",
        "created_at": meta["created_at"],
        "sha256": hashes,
        "note": "Locked before model development; do not read until final evaluation.",
    })

    print(f"Wrote {out}")
    print(f"Old intents: {len(old_intents)} | New intents: {len(new_intents)}")
    print(f"Old memory: {len(memory_old)} | Old dev: {len(dev_old)} | Old test (locked): {len(old_test)}")
    print(f"New feedback: {len(feedback_new)} | New test (locked): {len(new_test)}")
    print("Locked SHA-256:")
    for name, digest in hashes.items():
        print(f"  {name}: {digest}")
    print("New intents (this is task design, not exam data):")
    for name in new_intents:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
