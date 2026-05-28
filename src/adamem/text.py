from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def tokenize(text: str) -> list[str]:
    return [token for token in (match.lower() for match in TOKEN_RE.findall(text)) if token not in STOPWORDS]


def hashed_bow(text: str) -> dict[str, float]:
    counts = Counter(tokenize(text))
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if not norm:
        return {}
    return {key: value / norm for key, value in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def memory_key(content: str, metadata: dict[str, object] | None = None) -> str:
    if metadata and metadata.get("memory_key"):
        return str(metadata["memory_key"]).strip().lower()
    if metadata and metadata.get("subject") and metadata.get("predicate"):
        return f"{metadata['subject']}::{metadata['predicate']}".lower()
    head = content.splitlines()[0].strip().lower()
    if ":" in head:
        return head.split(":", 1)[0][:96]
    tokens = tokenize(head)
    return " ".join(tokens[:8])
