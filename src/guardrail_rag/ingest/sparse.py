"""Hashed-token term-frequency sparse encoder (BM25-like keyword leg, no model download)."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "did",
        "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "she",
        "so", "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "to", "was", "we", "were", "what", "when", "where", "which", "who", "will",
        "with", "you", "your",
    }
)  # fmt: skip

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, without stopwords or single characters."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


def token_index(token: str) -> int:
    """Stable non-negative index for a token (31-bit blake2b hash)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def sparse_encode(text: str) -> tuple[list[int], list[float]]:
    """Return (indices, values): sorted unique indices with weight 1 + log(tf).

    Hash collisions between different tokens are merged by summing their weights.
    """
    tf: dict[str, int] = defaultdict(int)
    for token in tokenize(text):
        tf[token] += 1
    weights: dict[int, float] = defaultdict(float)
    for token, count in tf.items():
        weights[token_index(token)] += 1.0 + math.log(count)
    indices = sorted(weights)
    return indices, [weights[i] for i in indices]
