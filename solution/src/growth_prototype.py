"""Grow a new-intent head from user feedback without weight training or RAG.

The compressed model is frozen. We take its last-token hidden state for each
text, center the features by the mean of the old-memory + feedback features,
and store one L2-normalized prototype per new intent (mean of its feedback
examples). Old behaviour is untouched: the base model keeps its original
generative router path, and only the new-intent sidecar head is added.

No gradients, no optimizer, and no feedback text is placed in the context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from common import read_jsonl, write_json, write_jsonl


def embed_batch(port: int, texts: list[str], timeout: float = 600.0) -> np.ndarray:
    payload = json.dumps({"input": texts, "model": "local"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    ordered = sorted(result["data"], key=lambda x: x.get("index", 0))
    vectors = np.array([x["embedding"] for x in ordered], dtype=np.float32)
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def embed_all(port: int, texts: list[str], batch_size: int) -> np.ndarray:
    chunks = []
    started = time.time()
    for start in range(0, len(texts), batch_size):
        batch = [
            f"Classify the banking customer message into an intent.\nMessage: {t}\nIntent:"
            for t in texts[start : start + batch_size]
        ]
        chunks.append(embed_batch(port, batch))
        done = start + len(batch)
        if done % (batch_size * 20) == 0 or done == len(texts):
            print(f"embedded {done}/{len(texts)} in {time.time() - started:.1f}s", flush=True)
    return np.concatenate(chunks, axis=0)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def build_prototypes(vectors: np.ndarray, labels: list[str], classes: list[str]) -> np.ndarray:
    rows = []
    for name in classes:
        mask = np.array([x == name for x in labels])
        if not mask.any():
            raise SystemExit(f"no examples for class {name}")
        mean = vectors[mask].mean(axis=0)
        rows.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
    return np.stack(rows).astype(np.float32)


def predict(vectors: np.ndarray, head: np.ndarray, classes: list[str]) -> list[str]:
    scores = vectors @ head.T
    return [classes[int(i)] for i in scores.argmax(axis=1)]


def accuracy(pred: list[str], gold: list[str]) -> float:
    return sum(int(p == g) for p, g in zip(pred, gold)) / len(gold) if gold else 0.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_predictions(path: Path, rows: list[dict], pred: list[str]) -> None:
    write_jsonl(path, [
        {
            "id": row.get("id"),
            "text": row.get("text"),
            "gold": row["label_name"],
            "pred": p,
            "correct": p == row["label_name"],
        }
        for row, p in zip(rows, pred)
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", type=Path, default=Path("solution/data/banking77_10class"))
    parser.add_argument("--out-dir", type=Path, default=Path("solution/results/growth/prototype"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    old_memory = list(read_jsonl(args.data_dir / "old_memory.jsonl"))
    feedback = list(read_jsonl(args.data_dir / "new_feedback.jsonl"))
    new_test = list(read_jsonl(args.data_dir / "locked" / "new_test.jsonl"))
    new_classes = sorted({r["label_name"] for r in feedback})

    print(f"new classes {len(new_classes)}, feedback {len(feedback)}, new test {len(new_test)}", flush=True)

    old_mem_vecs = embed_all(args.port, [r["text"] for r in old_memory], args.batch_size)
    feedback_vecs = embed_all(args.port, [r["text"] for r in feedback], args.batch_size)
    new_test_vecs = embed_all(args.port, [r["text"] for r in new_test], args.batch_size)

    # Center on the deployment feature distribution, then normalize.
    center = np.concatenate([old_mem_vecs, feedback_vecs], axis=0).mean(axis=0)
    feedback_centered = normalize(feedback_vecs - center)
    test_centered = normalize(new_test_vecs - center)

    head_before = np.zeros((0, feedback_centered.shape[1]), dtype=np.float32)
    head_after = build_prototypes(feedback_centered, [r["label_name"] for r in feedback], new_classes)

    new_gold = [r["label_name"] for r in new_test]
    pred_before = [None] * len(new_test)
    pred_after = predict(test_centered, head_after, new_classes)
    before_acc = 0.0
    after_acc = accuracy(pred_after, new_gold)

    # Leave-one-out on the feedback examples, using the same centering scheme.
    loo_pred = []
    for i in range(len(feedback_centered)):
        train_idx = [j for j in range(len(feedback_centered)) if j != i]
        head = build_prototypes(
            feedback_centered[train_idx],
            [feedback[r]["label_name"] for r in train_idx],
            new_classes,
        )
        loo_pred.append(predict(feedback_centered[i : i + 1], head, new_classes)[0])
    loo_acc = accuracy(loo_pred, [r["label_name"] for r in feedback])

    write_predictions(out_dir / "new_before.jsonl", new_test, pred_before)
    write_predictions(out_dir / "new_after.jsonl", new_test, pred_after)
    np.save(out_dir / "head_before.npy", head_before)
    np.save(out_dir / "head_after.npy", head_after)

    model_sha = sha256_file(Path(args.model_path))
    summary = {
        "model_path": args.model_path,
        "model_sha256_before": model_sha,
        "model_sha256_after": model_sha,
        "new_classes": new_classes,
        "feedback_examples": len(feedback),
        "head_before": {
            "classes": 0,
            "shape": list(head_before.shape),
            "bytes": int(head_before.nbytes),
            "sha256": sha256_bytes(head_before.tobytes()),
        },
        "head_after": {
            "classes": len(new_classes),
            "shape": list(head_after.shape),
            "bytes": int(head_after.nbytes),
            "sha256": sha256_bytes(head_after.tobytes()),
        },
        "new_intent_test": {
            "n": len(new_test),
            "before_correct": 0,
            "before_accuracy": before_acc,
            "after_correct": int(round(after_acc * len(new_test))),
            "after_accuracy": after_acc,
        },
        "feedback_leave_one_out_accuracy": loo_acc,
        "method": "frozen hidden-state features, centered, one prototype per new intent; no gradients; no context edits",
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
