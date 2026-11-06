"""Create a 20-queue banking support benchmark from Banking77.

The 67 fine-grained old intents are grouped into 20 commercially meaningful
support queues. The 10 held-out new intents stay fine-grained for the growth
demo. The locked grouped exam is a deterministic relabeling of the already
locked old test texts; its hash is committed before any grouped evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import read_jsonl, sha256_file, write_json, write_jsonl

GROUPS: dict[str, list[str]] = {
    "card_ordering": [
        "card_arrival", "card_delivery_estimate", "order_physical_card",
        "get_physical_card", "getting_spare_card", "card_about_to_expire",
    ],
    "card_activation": [
        "card_linking", "card_acceptance", "supported_cards_and_currencies",
        "visa_or_mastercard", "activate_my_card", "apple_pay_or_google_pay",
    ],
    "virtual_cards": [
        "get_disposable_virtual_card", "getting_virtual_card",
        "virtual_card_not_working", "disposable_card_limits",
    ],
    "card_troubleshooting": [
        "card_not_working", "card_swallowed", "contactless_not_working",
    ],
    "lost_stolen": [
        "lost_or_stolen_card", "lost_or_stolen_phone", "compromised_card",
    ],
    "pin_access": ["change_pin", "passcode_forgotten", "pin_blocked"],
    "card_payments": [
        "card_payment_not_recognised", "declined_card_payment",
        "pending_card_payment", "card_payment_wrong_exchange_rate",
    ],
    "refunds_disputes": [
        "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
        "transaction_charged_twice",
    ],
    "card_fees": ["card_payment_fee_charged", "extra_charge_on_statement"],
    "top_up_methods": [
        "automatic_top_up", "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge", "top_up_by_cash_or_cheque", "topping_up_by_card",
    ],
    "top_up_issues": ["pending_top_up", "top_up_failed", "top_up_reverted"],
    "top_up_limits": ["top_up_limits"],
    "transfers": [
        "cancel_transfer", "declined_transfer", "failed_transfer",
        "pending_transfer", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing",
        "beneficiary_not_allowed", "receiving_money",
    ],
    "transfer_fees": ["transfer_fee_charged"],
    "balance_updates": [
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
    ],
    "cash_atm": [
        "atm_support", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal", "pending_cash_withdrawal",
        "wrong_amount_of_cash_received",
    ],
    "currency": [
        "exchange_charge", "exchange_rate", "exchange_via_app",
        "fiat_currency_support", "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "identity": [
        "age_limit", "unable_to_verify_identity", "verify_my_identity",
        "verify_source_of_funds", "verify_top_up", "why_verify_identity",
    ],
    "account": ["edit_personal_details", "terminate_account", "country_support"],
    "direct_debit": ["direct_debit_payment_not_recognised"],
}

LABEL_TO_GROUP = {label: group for group, labels in GROUPS.items() for label in labels}
GROUP_DESC = {
    "card_ordering": "Ordering, delivery, or replacement of a physical card",
    "card_activation": "Activating a card, linking it, or supported card networks",
    "virtual_cards": "Virtual or disposable card issuance and limits",
    "card_troubleshooting": "Card not working, swallowed, or contactless issues",
    "lost_stolen": "Lost, stolen, or compromised card/phone",
    "pin_access": "PIN changes, forgotten passcodes, or blocked PINs",
    "card_payments": "Card payment declined, pending, unrecognised, or wrong exchange rate",
    "refunds_disputes": "Refunds, disputed transactions, or duplicate charges",
    "card_fees": "Card payment fees or extra statement charges",
    "top_up_methods": "How to top up and top-up charges",
    "top_up_issues": "Pending, failed, or reverted top-ups",
    "top_up_limits": "Top-up limits",
    "transfers": "Transfers, recipients, timing, or receiving money",
    "transfer_fees": "Transfer fees",
    "balance_updates": "Balance not updated after a transfer, cheque, or cash deposit",
    "cash_atm": "ATM support, cash withdrawals, charges, or wrong amounts",
    "currency": "Exchange rates, currency support, or exchange charges",
    "identity": "Identity verification or age limits",
    "account": "Personal details, account closure, or country support",
    "direct_debit": "Direct debit payment not recognised",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77"))
    parser.add_argument("--out", type=Path, default=Path("solution/data/banking77_grouped"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force")
    old_meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    missing = [x for x in old_meta["old_intents"] if x not in LABEL_TO_GROUP]
    if missing:
        raise SystemExit(f"Unmapped old intents: {missing}")

    def map_row(row: dict) -> dict:
        group = LABEL_TO_GROUP[row["label_name"]]
        return {
            **row,
            "orig_label": row["label_name"],
            "label_name": group,
            "label_desc": GROUP_DESC[group],
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
    # Copy the fine-grained new-intent growth files.
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
        "variant": "20_queue_grouped",
        "source": str(args.data_dir),
        "old_intents": sorted(GROUPS),
        "new_intents": old_meta["new_intents"],
        "group_mapping": GROUPS,
        "group_descriptions": GROUP_DESC,
        "counts": {
            "old_memory": len(memory),
            "old_dev": len(dev),
            "old_test": len(exam),
            "old_intents": len(GROUPS),
            "new_intents": len(old_meta["new_intents"]),
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
        "variant": "20_queue_grouped",
        "sha256": meta["locked_sha256"],
        "note": "Grouped exam locked before grouped model development; mapping is label-name based.",
    })
    print(f"Wrote {args.out}")
    print(f"Old memory {len(memory)} | Old dev {len(dev)} | Old grouped test (locked) {len(exam)}")
    print(f"Grouped old test SHA-256: {digest}")
    print(f"New test SHA-256: {new_digest}")


if __name__ == "__main__":
    main()
