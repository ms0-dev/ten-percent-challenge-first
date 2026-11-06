"""Extract final-norm hidden states for in-model feedback editing.

The edit target is the model's own output rows: existing vocabulary rows are
overwritten with intent prototypes. This script only collects features from
feedback, the new-intent test, and old dev. It never reads the locked exam.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import read_jsonl


def prompt(text: str) -> str:
    return f"Classify the banking customer message into an intent.\nMessage: {text}\nIntent:"


def extract(model, tokenizer, texts: list[str]) -> np.ndarray:
    rows = []
    for i, text in enumerate(texts, 1):
        enc = tokenizer(prompt(text), return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.model(**enc)
            # last_hidden_state is the exact input to lm_head (final norm applied)
            hidden = out.last_hidden_state[:, -1, :]
        rows.append(hidden.float().cpu().numpy()[0])
        if i % 50 == 0:
            print(f"extracted {i}/{len(texts)}", flush=True)
    return np.stack(rows)


def choose_class_tokens(tokenizer, labels: list[str]) -> dict[str, int]:
    tokenized = {label: tokenizer.encode(label, add_special_tokens=False) for label in labels}
    counts = Counter(tok for ids in tokenized.values() for tok in ids)
    chosen: dict[str, int] = {}
    used: set[int] = set()
    for label in labels:
        unique = [tok for tok in tokenized[label] if counts[tok] == 1 and tok not in used]
        if unique:
            tok = unique[0]
        else:
            tok = tokenized[label][-1]
        chosen[label] = tok
        used.add(tok)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77_10class"))
    parser.add_argument("--out-dir", type=Path, default=Path("solution/results/growth_inmodel"))
    parser.add_argument("--old-dev-limit", type=int, default=300)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": 0},
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    feedback = list(read_jsonl(args.data_dir / "new_feedback.jsonl"))
    new_test = list(read_jsonl(args.data_dir / "locked" / "new_test.jsonl"))
    old_dev = list(read_jsonl(args.data_dir / "old_dev.jsonl"))[: args.old_dev_limit]

    feedback_x = extract(model, tokenizer, [r["text"] for r in feedback])
    new_test_x = extract(model, tokenizer, [r["text"] for r in new_test])
    old_dev_x = extract(model, tokenizer, [r["text"] for r in old_dev])

    new_classes = sorted({r["label_name"] for r in feedback})
    class_tokens = choose_class_tokens(tokenizer, new_classes)
    np.savez_compressed(
        args.out_dir / "features.npz",
        feedback=feedback_x,
        new_test=new_test_x,
        old_dev=old_dev_x,
    )
    meta = {
        "model_path": args.model_path,
        "feedback_labels": [r["label_name"] for r in feedback],
        "new_test_labels": [r["label_name"] for r in new_test],
        "old_dev_labels": [r["label_name"] for r in old_dev],
        "new_classes": new_classes,
        "class_tokens": class_tokens,
        "class_token_text": {k: tokenizer.decode([v]) for k, v in class_tokens.items()},
        "prompt": "Classify the banking customer message into an intent.\nMessage: {text}\nIntent:",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "feedback": list(feedback_x.shape),
        "new_test": list(new_test_x.shape),
        "old_dev": list(old_dev_x.shape),
        "class_tokens": meta["class_token_text"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
