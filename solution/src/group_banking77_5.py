"""Collapse the 10-family benchmark into 5 coarse commercial support queues."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import read_jsonl, sha256_file, write_json, write_jsonl

QUEUES: dict[str, list[str]] = {
    "cards": ["card_management", "card_payments"],
    "refunds_disputes": ["refunds_disputes"],
    "top_up": ["top_up"],
    "transfers": ["transfers"],
    "account_and_cash": ["cash_atm", "currency", "identity", "account", "direct_debit"],
}
FAMILY_TO_QUEUE = {f: q for q, families in QUEUES.items() for f in families}
QUEUE_DESC = {
    "cards": "Card ordering, activation, troubleshooting, payments, fees, lost/stolen, or PIN",
    "refunds_disputes": "Refunds, disputes, duplicate charges, or reverted payments",
    "top_up": "Top-up methods, charges, limits, or failed/pending/reverted top-ups",
    "transfers": "Transfers, recipients, timing, fees, balance updates, or receiving money",
    "account_and_cash": "ATM/cash, currency, identity verification, account details, or direct debit",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77_10class"))
    parser.add_argument("--out", type=Path, default=Path("solution/data/banking77_5class"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force")
    meta10 = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    missing = [f for f in meta10["old_intents"] if f not in FAMILY_TO_QUEUE]
    if missing:
        raise SystemExit(f"Unmapped families: {missing}")

    def map_row(row: dict) -> dict:
        queue = FAMILY_TO_QUEUE[row["label_name"]]
        return {
            **row,
            "orig_family": row["label_name"],
            "label_name": queue,
            "label_desc": QUEUE_DESC[queue],
        }

    memory = [map_row(r) for r in read_jsonl(args.data_dir / "old_memory.jsonl")]
    dev = [map_row(r) for r in read_jsonl(args.data_dir / "old_dev.jsonl")]
    exam = [map_row(r) for r in read_jsonl(args.data_dir / "locked/old_test.jsonl")]

    args.out.mkdir(parents=True, exist_ok=True)
    locked = args.out / "locked"
    locked.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "old_memory.jsonl", memory)
    write_jsonl(args.out / "old_dev.jsonl", dev)
    write_jsonl(locked / "old_test.jsonl", exam)
    for name in ("new_feedback.jsonl",):
        src = args.data_dir / name
        if src.exists():
            write_jsonl(args.out / name, read_jsonl(src))
    src_new_test = args.data_dir / "locked/new_test.jsonl"
    if src_new_test.exists():
        write_jsonl(locked / "new_test.jsonl", read_jsonl(src_new_test))

    digest = sha256_file(locked / "old_test.jsonl")
    new_digest = sha256_file(locked / "new_test.jsonl")
    meta = {
        "dataset": "PolyAI/banking77",
        "variant": "5_queue_grouped",
        "source": str(args.data_dir),
        "old_intents": sorted(QUEUES),
        "new_intents": meta10["new_intents"],
        "queue_mapping": QUEUES,
        "queue_descriptions": QUEUE_DESC,
        "counts": {
            "old_memory": len(memory),
            "old_dev": len(dev),
            "old_test": len(exam),
            "old_intents": len(QUEUES),
            "new_intents": len(meta10["new_intents"]),
        },
        "label_distribution": dict(Counter(r["label_name"] for r in dev)),
        "locked_sha256": {
            "old_test.jsonl": digest,
            "new_test.jsonl": new_digest,
        },
    }
    write_json(args.out / "meta.json", meta)
    write_json(locked / "manifest.json", {
        "dataset": "PolyAI/banking77",
        "variant": "5_queue_grouped",
        "sha256": meta["locked_sha256"],
        "note": "5-queue exam locked before model development; mapping is label-name based.",
    })
    print(f"Wrote {args.out}")
    print(f"Old memory {len(memory)} | Old dev {len(dev)} | Old test (locked) {len(exam)}")
    print(f"5-queue locked test SHA-256: {digest}")
    print(f"New test SHA-256: {new_digest}")


if __name__ == "__main__":
    main()
