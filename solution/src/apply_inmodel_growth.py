"""Write feedback prototypes into the model's existing output rows.

The model's lm_head is a [vocab, hidden] matrix. We overwrite ten existing
rows, one per new intent, with the mean final hidden state of that intent's
five feedback examples (normalized). No row is added, no external head or
memory file is created, and the checkpoint size does not change.

This is a closed-form edit. There are no gradients and no optimizer steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import read_jsonl, write_jsonl

# Chosen by hand from the new label names, excluding every token used by the
# ten old label names. Each token is unique among the ten new labels.
CLASS_TOKEN_TEXT = {
    "Refund_not_showing_up": "_show",
    "activate_my_card": "activate",
    "apple_pay_or_google_pay": "apple",
    "balance_not_updated_after_bank_transfer": "balance",
    "card_acceptance": "_accept",
    "card_payment_not_recognised": "_payment",
    "disposable_card_limits": "_limits",
    "extra_charge_on_statement": "extra",
    "getting_virtual_card": "getting",
    "receiving_money": "_money",
}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def build_prototypes(x: np.ndarray, y: list[str], classes: list[str]) -> np.ndarray:
    rows = []
    for c in classes:
        mean = x[np.array(y) == c].mean(axis=0)
        rows.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
    return np.stack(rows).astype(np.float32)


def accuracy(pred: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def resolve_tokens(tokenizer, new_classes: list[str], old_classes: list[str]) -> dict[str, int]:
    old_ids: set[int] = set()
    for label in old_classes:
        old_ids.update(tokenizer.encode(label, add_special_tokens=False))
    tokens: dict[str, int] = {}
    for label in new_classes:
        text = CLASS_TOKEN_TEXT[label]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"class token {text!r} is not a single token: {ids}")
        token = ids[0]
        if token in old_ids:
            raise SystemExit(f"class token {text!r} collides with an old label token")
        if token in tokens.values():
            raise SystemExit(f"class token {text!r} is not unique")
        if token in tokenizer.all_special_ids:
            raise SystemExit(f"class token {text!r} is a special token")
        tokens[label] = token
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--features-meta", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77_10class"))
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    data_meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    old_classes = list(data_meta["old_intents"])
    feedback_meta = json.loads(Path(args.features_meta).read_text(encoding="utf-8"))
    feedback_labels = feedback_meta["feedback_labels"]
    new_test_labels = feedback_meta["new_test_labels"]
    new_classes = sorted(set(feedback_labels))

    features = np.load(args.features)
    feedback = features["feedback"].astype(np.float32)
    new_test = features["new_test"].astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    class_tokens = resolve_tokens(tokenizer, new_classes, old_classes)

    prototypes = build_prototypes(feedback, feedback_labels, new_classes)
    pred_after = [new_classes[int(i)] for i in (new_test @ prototypes.T).argmax(axis=1)]
    after_acc = accuracy(pred_after, new_test_labels)

    loo_pred = []
    for i in range(len(feedback)):
        idx = [j for j in range(len(feedback)) if j != i]
        head = build_prototypes(feedback[idx], [feedback_labels[j] for j in idx], new_classes)
        loo_pred.append(new_classes[int((feedback[i] @ head.T).argmax())])
    loo_acc = accuracy(loo_pred, feedback_labels)

    # Reference only: centered prototypes cannot be represented exactly by a
    # bias-free lm_head, so the edit uses raw prototypes.
    center = np.concatenate([feedback, features["old_dev"].astype(np.float32)], axis=0).mean(axis=0)
    centered = build_prototypes(normalize(feedback - center), feedback_labels, new_classes)
    pred_centered = [new_classes[int(i)] for i in (normalize(new_test - center) @ centered.T).argmax(axis=1)]
    centered_acc = accuracy(pred_centered, new_test_labels)

    # Before: the model's own original rows for the same ten tokens.
    with safe_open(str(Path(args.model_path) / "model.safetensors"), framework="pt", device="cpu") as handle:
        lm_head = handle.get_tensor("lm_head.weight").float().numpy()
    token_ids = [class_tokens[c] for c in new_classes]
    before_rows = lm_head[token_ids].copy()
    pred_before = [new_classes[int(i)] for i in (new_test @ before_rows.T).argmax(axis=1)]
    before_acc = accuracy(pred_before, new_test_labels)

    new_test_rows = list(read_jsonl(args.data_dir / "locked" / "new_test.jsonl"))
    out_dir = Path(args.report).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "new_before.jsonl", [
        {"id": row.get("id"), "text": row.get("text"), "gold": row["label_name"], "pred": pred, "correct": pred == row["label_name"]}
        for row, pred in zip(new_test_rows, pred_before)
    ])
    write_jsonl(out_dir / "new_after.jsonl", [
        {"id": row.get("id"), "text": row.get("text"), "gold": row["label_name"], "pred": pred, "correct": pred == row["label_name"]}
        for row, pred in zip(new_test_rows, pred_after)
    ])

    # Edit the model's existing rows in place.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if model.config.tie_word_embeddings:
        raise SystemExit("lm_head is tied to embeddings; refusing to edit")
    with torch.no_grad():
        for class_name, token in class_tokens.items():
            proto = prototypes[new_classes.index(class_name)]
            model.lm_head.weight[token] = torch.tensor(
                args.alpha * proto, dtype=model.lm_head.weight.dtype
            )
    after_rows = model.lm_head.weight[token_ids].detach().float().numpy()

    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path, safe_serialization=True, max_shard_size="50GB")
    tokenizer.save_pretrained(out_path)
    print(f"saved {out_path}", flush=True)

    report = {
        "model_path": args.model_path,
        "out_path": args.out_path,
        "alpha": args.alpha,
        "class_tokens": {
            c: {"id": class_tokens[c], "text": tokenizer.decode([class_tokens[c]])}
            for c in new_classes
        },
        "old_classes": old_classes,
        "new_classes": new_classes,
        "feedback_examples": len(feedback),
        "before_accuracy": before_acc,
        "after_accuracy": after_acc,
        "centered_reference_accuracy": centered_acc,
        "feedback_leave_one_out_accuracy": loo_acc,
        "rows_before_sha256": sha256_array(before_rows),
        "rows_after_sha256": sha256_array(after_rows),
        "method": "overwrite ten existing lm_head rows with raw feedback class-mean prototypes; no added parameters",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
