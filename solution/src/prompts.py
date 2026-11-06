"""Prompt construction and output parsing for the pilot benchmarks."""

from __future__ import annotations

import difflib
import re
import string
from typing import Iterable, Sequence

BANKING_SYSTEM = (
    "You are a precise banking customer-support triage assistant. "
    "Reasoning: low. Choose exactly one intent label from the provided list. "
    "Do not explain; return only the label."
)

QA_SYSTEM = (
    "You are a precise customer-support question-answering assistant. "
    "Reasoning: low. Answer the question using only the provided context. "
    "Return a short extractive answer copied from the context. Do not explain."
)

QA_MC_SYSTEM = (
    "You are a precise customer-support question-answering assistant. "
    "Reasoning: low. You are given a question, supporting context passages, "
    "and four answer options. Choose the single option that correctly answers "
    "the question using the context. Return only the option letter "
    "(A, B, C, or D). Do not explain."
)


def build_banking_messages(
    query: str,
    examples: Sequence[dict],
    labels: Sequence[dict],
) -> list[dict]:
    label_lines = [f"- {x['label_name']}: {x['label_desc']}" for x in labels]
    example_lines = []
    for ex in examples:
        example_lines.append(f"Message: {ex['text']}\nIntent: {ex['label_name']}")
    user = (
        "Valid intent labels:\n"
        + "\n".join(label_lines)
        + "\n\nSimilar labeled examples:\n"
        + "\n\n".join(example_lines)
        + f"\n\nCustomer message: {query}\n\n"
        "Return exactly one intent label from the valid list. Output only the label."
    )
    return [
        {"role": "system", "content": BANKING_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_qa_messages(question: str, passages: Sequence[str]) -> list[dict]:
    context = "\n\n".join(f"[Passage {i + 1}]\n{p}" for i, p in enumerate(passages))
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context. Give a short extractive answer."
    )
    return [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": user},
    ]


_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)
_ANALYSIS_RE = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)
_SPECIAL_RE = re.compile(r"<\|[^>]+\|>")


def parse_final(text: str) -> str:
    """Extract the final-channel content from a GPT-OSS/harmony output."""
    match = _FINAL_RE.search(text)
    if match:
        return _SPECIAL_RE.sub("", match.group(1)).strip()
    # Some models / templates may not emit channel tags. Fall back to the
    # last non-empty chunk after stripping special tokens.
    cleaned = _SPECIAL_RE.sub("", text).strip()
    return cleaned


def _norm_label(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("?", "").replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def parse_label(text: str, valid_labels: Sequence[str]) -> str | None:
    """Map a generated answer to one of the valid Banking77 labels."""
    raw = parse_final(text)
    normalized = _norm_label(raw)
    by_norm = {_norm_label(x): x for x in valid_labels}

    if normalized in by_norm:
        return by_norm[normalized]

    # Prefer the longest label that appears as a substring.
    candidates = sorted(by_norm, key=len, reverse=True)
    for candidate in candidates:
        if candidate and candidate in normalized:
            return by_norm[candidate]

    # Last resort: fuzzy match on the first line / first 80 chars.
    first = normalized.splitlines()[0][:80] if normalized else ""
    close = difflib.get_close_matches(first, list(by_norm), n=1, cutoff=0.72)
    if close:
        return by_norm[close[0]]
    return None


def build_qa_mc_messages(question: str, passages: Sequence[str], options: Sequence[str]) -> list[dict]:
    context = "\n\n".join(f"[Passage {i + 1}]\n{p}" for i, p in enumerate(passages))
    option_lines = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Return only the letter of the correct option (A, B, C, or D)."
    )
    return [
        {"role": "system", "content": QA_MC_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_option(text: str, options: Sequence[str]) -> int | None:
    raw = parse_final(text).strip()
    if not raw:
        return None
    upper = raw.upper()
    # Direct letter at the start: "A", "A.", "(A)", "A)".
    match = re.match(r"^\s*\(?([A-D])\)?\s*(?:[.):]|$)", upper)
    if match:
        return ord(match.group(1)) - ord("A")
    # "Option A" / "Answer: B".
    match = re.search(r"\b(?:OPTION|ANSWER(?:\s+IS)?)\s*[:\-]?\s*\(?([A-D])\)?\b", upper)
    if match:
        return ord(match.group(1)) - ord("A")
    # A standalone letter anywhere.
    match = re.search(r"\b([A-D])\b", upper)
    if match:
        return ord(match.group(1)) - ord("A")
    # Fallback: match the option text.
    normalized_raw = normalize_answer(raw)
    for i, opt in enumerate(options):
        if normalized_raw == normalize_answer(opt):
            return i
        if normalize_answer(opt) and normalize_answer(opt) in normalized_raw:
            return i
    return None


def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
