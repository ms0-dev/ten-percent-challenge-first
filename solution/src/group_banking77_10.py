"""Collapse the 20-queue grouped benchmark into 10 commercial support families."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import read_jsonl, sha256_file, write_json, write_jsonl

FAMILIES: dict[str, list[str]] = {
    "card_management": [
        "card_ordering", "card_activation", "virtual_cards",
        "card_troubleshooting", "lost_stolen", "pin_access",
    ],
    "card_payments": ["card_payments", "card_fees"],
    "refunds_disputes": ["refunds_disputes"],
    "top_up": ["top_up_methods", "top_up_issues", "top_up_limits"],
    "transfers": ["transfers", "transfer_fees", "balance_updates"],
    "cash_atm": ["cash_atm"],
    "currency": ["currency"],
    "identity": ["identity"],
    "account": ["account"],
    "direct_debit": ["direct_debit"],
}
GROUP_TO_FAMILY = {g: f for f, groups in FAMILIES.items() for g in groups}
FAMILY_DESC = {
    "card_management": "Card ordering, activation, virtual cards, troubleshooting, lost/stolen, or PIN access",
    "card_payments": "Card payments, fees, or declined/pending card transactions",
    "refunds_disputes": "Refunds, disputes, duplicate charges, or reverted payments",
    "top_up": "Top-up methods, charges, limits, or failed/pending/reverted top-ups",
    "transfers": "Transfers, recipients, timing, fees, balance updates, or receiving money",
    "cash_atm": "ATM support, cash withdrawals, charges, or wrong amounts",
    "currency": "Exchange rates, currency support, or exchange charges",
    "identity": "Identity verification or age limits",
    "account": "Personal details, account closure, or country support",
    "direct_debit": "Direct debit payment not recognised",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77_grouped"))
    parser.add_argument("--out", type=Path, default=Path("solution/data/banking77_10class"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force")
    meta20 = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    missing = [g for g in meta20["old_intents"] if g not in GROUP_TO_FAMILY]
    if missing:
        raise SystemExit(f"Unmapped groups: {missing}")

    def map_row(row: dict) -> dict:
        family = GROUP_TO_FAMILY[row["label_name"]]
        return {
            **row,
            "orig_group": row["label_name"],
            "label_name": family,
            "label_desc": FAMILY_DESC[family],
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
        "variant": "10_family_grouped",
        "source": str(args.data_dir),
        "old_intents": sorted(FAMILIES),
        "new_intents": meta20["new_intents"],
        "family_mapping": FAMILIES,
        "family_descriptions": FAMILY_DESC,
        "counts": {
            "old_memory": len(memory),
            "old_dev": len(dev),
            "old_test": len(exam),
            "old_intents": len(FAMILIES),
            "new_intents": len(meta20["new_intents"]),
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
        "variant": "10_family_grouped",
        "sha256": meta["locked_sha256"],
        "note": "10-family exam locked before model development; mapping is label-name based.",
    })
    print(f"Wrote {args.out}")
    print(f"Old memory {len(memory)} | Old dev {len(dev)} | Old test (locked) {len(exam)}")
    print(f"10-family locked test SHA-256: {digest}")
    print(f"New test SHA-256: {new_digest}")


if __name__ == "__main__":
    main()
