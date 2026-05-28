from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryItem:
    content: str
    kind: str = "observation"
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    last_seen_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    importance: float = 0.5
    confidence: float = 1.0
    feedback: float = 0.0
    access_count: int = 0
    links: list[str] = field(default_factory=list)
    cause_ids: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    staleness: float = 0.0
    stale_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, float] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.superseded_by is None


@dataclass(slots=True)
class MemoryResult:
    item: MemoryItem
    score: float
    contributions: dict[str, float]
    relation: str = "direct"


@dataclass(slots=True)
class MemoryPatch:
    content: str
    kind: str = "observation"
    importance: float = 0.5
    confidence: float = 1.0
    valid_from: str | None = None
    valid_to: str | None = None
    cause_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
