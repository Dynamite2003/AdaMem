from __future__ import annotations

from collections.abc import Callable
import math
from dataclasses import fields
from datetime import datetime, timezone
from typing import Iterable

from adamem.config import AdaMemConfig
from adamem.schema import MemoryItem, MemoryResult, utc_now
from adamem.store import InMemoryStore, MemoryStore
from adamem.text import cosine, hashed_bow, memory_key

Embedder = Callable[[str], dict[str, float]]


class AdaMem:
    """Adaptive delta memory without owning the surrounding agent loop."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        config: AdaMemConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.store = store or InMemoryStore()
        self.config = config or AdaMemConfig()
        self.embedder = embedder or hashed_bow

    def observe(
        self,
        content: str,
        *,
        kind: str = "observation",
        importance: float = 0.5,
        confidence: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
        cause_ids: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> MemoryItem:
        metadata = dict(metadata or {})
        cause_ids = list(cause_ids or [])
        embedding = self.embedder(_embedding_text(content, metadata))
        item = MemoryItem(
            content=content,
            kind=kind,
            importance=_clamp(importance),
            confidence=_clamp(confidence),
            valid_from=valid_from,
            valid_to=valid_to,
            cause_ids=cause_ids,
            metadata=metadata,
            embedding=embedding,
        )

        duplicate = self._nearest_active(embedding)
        if duplicate and cosine(embedding, duplicate.embedding) >= self.config.novelty_threshold:
            duplicate.last_seen_at = utc_now()
            duplicate.importance = max(duplicate.importance, item.importance)
            duplicate.confidence = max(duplicate.confidence, item.confidence)
            duplicate.access_count += 1
            self.store.upsert(duplicate)
            return duplicate

        if self.config.use_supersession:
            self._supersede_conflicting(item)

        item.links = self._link_candidates(item)
        for cause_id in cause_ids:
            if cause_id not in item.links:
                item.links.append(cause_id)
        self.store.upsert(item)
        return item

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        now: str | None = None,
        include_inactive: bool = False,
    ) -> list[MemoryResult]:
        query_embedding = self.embedder(query)
        direct = [
            self._score(item, query_embedding, now=now, relation="direct")
            for item in self.store.all()
            if include_inactive or item.active
        ]
        direct = [result for result in direct if result.score > 0]
        candidates = {result.item.id: result for result in direct}

        if self.config.use_graph:
            for seed in sorted(direct, key=lambda result: result.score, reverse=True)[: max(top_k, 3)]:
                for neighbor, distance in self._graph_neighbors(seed.item):
                    if not include_inactive and not neighbor.active:
                        continue
                    graph_boost = self.config.graph_boost / distance
                    if neighbor.id in candidates:
                        result = candidates[neighbor.id]
                        result.contributions["graph"] = max(result.contributions.get("graph", 0.0), graph_boost)
                        result.score = sum(result.contributions.values())
                        result.relation = f"{result.relation}+graph"
                    else:
                        result = self._score(neighbor, query_embedding, now=now, relation="graph")
                        result.contributions["graph"] = graph_boost
                        result.score += graph_boost
                        candidates[neighbor.id] = result

        ranked = sorted(candidates.values(), key=lambda result: result.score, reverse=True)
        if self.config.use_mmr:
            ranked = self._mmr(query_embedding, ranked, top_k)
        else:
            ranked = ranked[:top_k]

        for result in ranked:
            result.item.access_count += 1
            self.store.upsert(result.item)
        return ranked

    def context(self, query: str, *, top_k: int = 6, max_chars: int = 1800) -> str:
        blocks: list[str] = []
        used = 0
        for index, result in enumerate(self.retrieve(query, top_k=top_k), start=1):
            stamp = result.item.valid_from or result.item.created_at
            block = f"[M{index} score={result.score:.3f} kind={result.item.kind} at={stamp}]\n{result.item.content}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def feedback(self, item_id: str, value: float) -> None:
        item = self.store.get(item_id)
        if not item:
            return
        item.feedback = _clamp(item.feedback + value, -1.0, 1.0)
        self.store.upsert(item)

    def ablation(self, **overrides: bool) -> "AdaMem":
        values = {field.name: getattr(self.config, field.name) for field in fields(self.config)}
        config = AdaMemConfig(**{**values, **overrides})
        clone = AdaMem(store=self.store, config=config, embedder=self.embedder)
        return clone

    def _nearest_active(self, embedding: dict[str, float]) -> MemoryItem | None:
        best: tuple[float, MemoryItem] | None = None
        for item in self.store.all():
            if not item.active:
                continue
            score = cosine(embedding, item.embedding)
            if best is None or score > best[0]:
                best = (score, item)
        return best[1] if best else None

    def _supersede_conflicting(self, item: MemoryItem) -> None:
        key = memory_key(item.content, item.metadata)
        item.metadata.setdefault("memory_key", key)
        for previous in self.store.all():
            if not previous.active:
                continue
            previous_key = memory_key(previous.content, previous.metadata)
            if previous_key == key and previous.content.strip() != item.content.strip():
                previous.superseded_by = item.id
                item.supersedes.append(previous.id)
                self.store.upsert(previous)

    def _link_candidates(self, item: MemoryItem) -> list[str]:
        if not self.config.use_graph or not self.config.use_auto_links:
            return []
        links: list[str] = []
        for previous in self.store.all():
            if previous.id == item.id or not previous.active:
                continue
            sim = cosine(item.embedding, previous.embedding)
            if sim >= self.config.link_threshold:
                links.append(previous.id)
        return links[:12]

    def _graph_neighbors(self, item: MemoryItem) -> Iterable[tuple[MemoryItem, int]]:
        seen = {item.id}
        frontier = [(item, 0)]
        max_hops = max(0, self.config.max_graph_hops)
        while frontier:
            current, distance = frontier.pop(0)
            if distance >= max_hops:
                continue
            for neighbor_id in self._neighbor_ids(current):
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                neighbor = self.store.get(neighbor_id)
                if not neighbor:
                    continue
                next_distance = distance + 1
                yield neighbor, next_distance
                frontier.append((neighbor, next_distance))

    def _neighbor_ids(self, item: MemoryItem) -> set[str]:
        ids = set(item.links) | set(item.cause_ids) | set(item.supersedes)
        for other in self.store.all():
            if item.id in other.links or item.id in other.cause_ids or item.id in other.supersedes:
                ids.add(other.id)
        return ids

    def _score(
        self,
        item: MemoryItem,
        query_embedding: dict[str, float],
        *,
        now: str | None,
        relation: str,
    ) -> MemoryResult:
        cfg = self.config
        contributions: dict[str, float] = {}
        semantic = cosine(query_embedding, item.embedding) if cfg.use_semantic else 0.0
        if cfg.use_semantic:
            contributions["semantic"] = cfg.semantic_weight * semantic
        if relation == "direct" and semantic <= cfg.min_direct_relevance:
            return MemoryResult(
                item=item,
                score=sum(contributions.values()),
                contributions=contributions,
                relation=relation,
            )
        if cfg.use_temporal:
            validity = _temporal_validity(item, now)
            contributions["temporal"] = cfg.temporal_weight * validity
            if validity == 0.0:
                contributions["temporal_mismatch"] = -cfg.temporal_mismatch_penalty
        if cfg.use_importance:
            contributions["importance"] = cfg.importance_weight * item.importance
        if cfg.use_recency:
            contributions["recency"] = cfg.recency_weight * _recency(item, cfg.recency_half_life_seconds, now)
        if cfg.use_confidence:
            contributions["confidence"] = cfg.confidence_weight * item.confidence
        if cfg.use_feedback:
            contributions["feedback"] = cfg.feedback_weight * item.feedback
        return MemoryResult(
            item=item,
            score=sum(contributions.values()),
            contributions=contributions,
            relation=relation,
        )

    def _mmr(
        self,
        query_embedding: dict[str, float],
        ranked: list[MemoryResult],
        top_k: int,
    ) -> list[MemoryResult]:
        selected: list[MemoryResult] = []
        pool = ranked[:]
        while pool and len(selected) < top_k:
            best: tuple[float, MemoryResult] | None = None
            for result in pool:
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(cosine(result.item.embedding, chosen.item.embedding) for chosen in selected)
                blended = self.config.mmr_lambda * result.score - (1 - self.config.mmr_lambda) * diversity_penalty
                if best is None or blended > best[0]:
                    best = (blended, result)
            assert best is not None
            selected.append(best[1])
            pool.remove(best[1])
        return selected


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _temporal_validity(item: MemoryItem, now: str | None) -> float:
    current = _parse_time(now) or datetime.now(timezone.utc)
    start = _parse_time(item.valid_from)
    end = _parse_time(item.valid_to)
    if start and current < start:
        return 0.0
    if end and current > end:
        return 0.0
    return 1.0


def _recency(item: MemoryItem, half_life_seconds: float, now: str | None) -> float:
    current = _parse_time(now) or datetime.now(timezone.utc)
    anchor = _parse_time(item.last_seen_at) or _parse_time(item.created_at) or current
    age = max(0.0, (current - anchor).total_seconds())
    if half_life_seconds <= 0:
        return 0.0
    return math.exp(-math.log(2) * age / half_life_seconds)


def _embedding_text(content: str, metadata: dict[str, object]) -> str:
    parts = [content]
    for key in ("memory_key", "subject", "predicate", "keywords", "tags"):
        value = metadata.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return " ".join(parts)
