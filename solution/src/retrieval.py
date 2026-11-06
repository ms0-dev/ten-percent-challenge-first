"""Lexical retrieval used by both the base and the compressed model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    index: int
    score: float


class BM25Index:
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)
        self.tokens = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int = 8) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return [Hit(index=i, score=float(scores[i])) for i in order]


def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> list[str]:
    """Split a long document into overlapping word windows."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: list[str] = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks
