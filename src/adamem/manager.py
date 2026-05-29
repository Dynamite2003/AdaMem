from __future__ import annotations

from collections.abc import Callable
import math
import re
from dataclasses import fields
from datetime import datetime, timezone
from typing import Iterable

from adamem.config import AdaMemConfig
from adamem.schema import MemoryItem, MemoryResult, utc_now
from adamem.state import (
    StateExtractor,
    extract_state_patches,
    query_relevant_state_slots,
    state_slot_depends_on,
    state_slot_matches_query,
)
from adamem.store import InMemoryStore, MemoryStore
from adamem.text import cosine, hashed_bow, memory_key, tokenize

Embedder = Callable[[str], dict[str, float]]


class AdaMem:
    """Adaptive delta memory without owning the surrounding agent loop."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        config: AdaMemConfig | None = None,
        embedder: Embedder | None = None,
        state_extractor: StateExtractor | None = None,
    ) -> None:
        self.store = store or InMemoryStore()
        self.config = config or AdaMemConfig()
        self.embedder = embedder or hashed_bow
        self.state_extractor = state_extractor or extract_state_patches

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
        if self.config.use_memory_evolution:
            self._seed_memory_note_metadata(content, metadata)
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

        duplicate = self._nearest_active(embedding, metadata)
        if duplicate and cosine(embedding, duplicate.embedding) >= self.config.novelty_threshold:
            duplicate.last_seen_at = utc_now()
            duplicate.importance = max(duplicate.importance, item.importance)
            duplicate.confidence = max(duplicate.confidence, item.confidence)
            duplicate.access_count += 1
            self.store.upsert(duplicate)
            return duplicate

        if self.config.use_supersession:
            self._supersede_conflicting(item)
        if self.config.use_soft_staleness:
            self._soft_stale_conflicting(item)

        item.links = self._link_candidates(item)
        for cause_id in cause_ids:
            if cause_id not in item.links:
                item.links.append(cause_id)
        self.store.upsert(item)
        if self.config.use_memory_evolution:
            self._evolve_related_memories(item)
            self.store.upsert(item)
        if self.config.use_temporal_kg_memory:
            self._observe_temporal_kg_edges(item)
        if self.config.use_salient_memory:
            self._observe_salient_memories(item)
        if self.config.use_state_memory:
            self._observe_state_patches(item)
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
            if (include_inactive or item.active) and self._eligible_for_direct_retrieval(item)
        ]
        direct = [result for result in direct if result.score > 0]
        candidates = {result.item.id: result for result in direct}

        if self.config.use_state_readout:
            for state_result in self._state_readout(query, query_embedding, now=now):
                candidates[state_result.item.id] = state_result

        if self.config.use_temporal_kg_readout:
            for kg_result in self._temporal_kg_readout(query, query_embedding, now=now):
                candidates[kg_result.item.id] = kg_result

        if self.config.use_salient_memory_readout:
            for salient_result in self._salient_memory_readout(query, query_embedding, now=now):
                candidates[salient_result.item.id] = salient_result

        if self.config.use_trajectory_step_readout:
            for step_result in self._trajectory_step_readout(query, query_embedding, now=now):
                candidates[step_result.item.id] = step_result

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

        if self.config.use_state_source_adjudication:
            candidates = self._filter_state_adjudicated_sources(candidates, query)

        ranked = sorted(candidates.values(), key=lambda result: result.score, reverse=True)
        if self.config.use_mmr:
            ranked = self._mmr(query_embedding, ranked, top_k)
        else:
            ranked = ranked[:top_k]

        if self.config.use_adjudication_filter:
            ranked = self._adjudication_filter(ranked)

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
        clone = AdaMem(
            store=self.store,
            config=config,
            embedder=self.embedder,
            state_extractor=self.state_extractor,
        )
        return clone

    def _nearest_active(self, embedding: dict[str, float], metadata: dict[str, object]) -> MemoryItem | None:
        best: tuple[float, MemoryItem] | None = None
        new_key = metadata.get("memory_key")
        for item in self.store.all():
            if not item.active:
                continue
            existing_key = item.metadata.get("memory_key")
            if new_key is not None and existing_key is not None and new_key != existing_key:
                continue
            score = cosine(embedding, item.embedding)
            if best is None or score > best[0]:
                best = (score, item)
        return best[1] if best else None

    def _seed_memory_note_metadata(self, content: str, metadata: dict[str, object]) -> None:
        keywords = _evolution_keywords(content, metadata, limit=self.config.memory_evolution_keyword_limit)
        if keywords:
            _extend_metadata_values(metadata, "keywords", keywords)
            metadata.setdefault("evolution_keywords", keywords)
        metadata.setdefault("memory_note", True)

    def _evolve_related_memories(self, item: MemoryItem) -> None:
        """A-MEM-style deterministic approximation.

        A-MEM uses LLM-generated note attributes, dynamic linking, and memory
        evolution. This API-free approximation only uses observed text and
        metadata: it links semantically related active raw memories and folds
        the new memory's extracted keywords into the older memories'
        retrieval representation.
        """

        if item.kind == "state" or item.metadata.get("derived") is True:
            return
        threshold = self.config.memory_evolution_threshold
        new_keywords = _metadata_strings(item.metadata.get("evolution_keywords"))
        if not new_keywords:
            return
        evolved_ids: list[str] = []
        candidate_limit = max(0, self.config.memory_evolution_candidate_limit)
        considered = 0
        seed_tags = _cooccurrence_tags(item)
        for previous in reversed(self.store.all()):
            if previous.id == item.id:
                continue
            if not previous.active:
                continue
            if previous.kind == "state" or previous.metadata.get("derived") is True:
                continue
            if candidate_limit and considered >= candidate_limit:
                break
            considered += 1
            similarity = cosine(item.embedding, previous.embedding)
            if similarity < threshold and not _co_occurs(item, previous, seed_tags):
                continue
            if previous.id not in item.links:
                item.links.append(previous.id)
            if item.id not in previous.links:
                previous.links.append(item.id)
            _extend_metadata_values(previous.metadata, "evolved_keywords", new_keywords)
            _extend_metadata_values(previous.metadata, "keywords", new_keywords)
            _append_metadata_value(previous.metadata, "evolved_by", item.id)
            previous.embedding = self.embedder(_embedding_text(previous.content, previous.metadata))
            self.store.upsert(previous)
            evolved_ids.append(previous.id)
        if evolved_ids:
            item.metadata["evolved_memory_ids"] = evolved_ids
            item.embedding = self.embedder(_embedding_text(item.content, item.metadata))

    def _observe_state_patches(self, source: MemoryItem) -> None:
        for patch in self.state_extractor(source.content, source.metadata):
            metadata = {
                "derived": True,
                "source_id": source.id,
                "memory_key": patch.key,
                "state_slot": patch.slot,
                "state_value": patch.value,
                "subject": patch.subject,
                "keywords": [patch.slot, patch.value],
            }
            state_item = MemoryItem(
                content=patch.content,
                kind="state",
                importance=1.0,
                confidence=source.confidence,
                valid_from=source.valid_from,
                metadata=metadata,
                embedding=self.embedder(_embedding_text(patch.content, metadata)),
            )
            for previous in self.store.all():
                if not previous.active:
                    continue
                if previous.metadata.get("memory_key") != patch.key:
                    continue
                if previous.metadata.get("derived") is not True:
                    continue
                if previous.content.strip() == state_item.content.strip():
                    continue
                previous.superseded_by = state_item.id
                previous.staleness = max(previous.staleness, 1.0)
                if source.id not in previous.stale_sources:
                    previous.stale_sources.append(source.id)
                state_item.supersedes.append(previous.id)
                self.store.upsert(previous)
                if self.config.use_state_source_adjudication:
                    self._mark_source_evidence_stale(previous, state_item, source)
            if self.config.use_state_dependency_propagation and state_item.supersedes:
                self._propagate_state_dependency(state_item, source)
            state_item.links.append(source.id)
            self.store.upsert(state_item)

    def _observe_temporal_kg_edges(self, source: MemoryItem) -> None:
        if source.kind in {"state", "kg_fact"} or source.metadata.get("derived") is True:
            return
        for patch in self.state_extractor(source.content, source.metadata):
            key = f"kg.{patch.subject}.{patch.slot}"
            metadata = {
                "derived": True,
                "source_id": source.id,
                "memory_key": key,
                "kg_subject": patch.subject,
                "kg_relation": patch.slot,
                "kg_object": patch.value,
                "state_slot": patch.slot,
                "keywords": [patch.subject, patch.slot, patch.value, "temporal", "kg"],
            }
            kg_item = MemoryItem(
                content=_temporal_kg_content(patch.subject, patch.slot, patch.value, patch.evidence),
                kind="kg_fact",
                importance=1.0,
                confidence=source.confidence,
                valid_from=source.valid_from,
                metadata=metadata,
                embedding=self.embedder(_embedding_text(
                    _temporal_kg_content(patch.subject, patch.slot, patch.value, patch.evidence),
                    metadata,
                )),
            )
            for previous in self.store.all():
                if not previous.active:
                    continue
                if previous.kind != "kg_fact":
                    continue
                if previous.metadata.get("memory_key") != key:
                    continue
                if previous.metadata.get("kg_object") == patch.value:
                    continue
                previous.superseded_by = kg_item.id
                previous.valid_to = source.valid_from or utc_now()
                previous.staleness = max(previous.staleness, 1.0)
                if source.id not in previous.stale_sources:
                    previous.stale_sources.append(source.id)
                kg_item.supersedes.append(previous.id)
                self.store.upsert(previous)
            kg_item.links.append(source.id)
            if kg_item.id not in source.links:
                source.links.append(kg_item.id)
                self.store.upsert(source)
            self.store.upsert(kg_item)

    def _observe_salient_memories(self, source: MemoryItem) -> None:
        if source.kind in {"state", "kg_fact", "salient_fact"} or source.metadata.get("derived") is True:
            return
        for patch in self.state_extractor(source.content, source.metadata):
            key = f"salient.{patch.subject}.{patch.slot}"
            metadata = {
                "extracted": True,
                "source_id": source.id,
                "memory_key": key,
                "salient_subject": patch.subject,
                "salient_slot": patch.slot,
                "salient_value": patch.value,
                "state_slot": patch.slot,
                "keywords": [patch.subject, patch.slot, patch.value, "memory", "fact"],
            }
            salient_item = MemoryItem(
                content=_salient_memory_content(patch.subject, patch.slot, patch.value, patch.evidence),
                kind="salient_fact",
                importance=1.0,
                confidence=source.confidence,
                valid_from=source.valid_from,
                metadata=metadata,
                embedding=self.embedder(_embedding_text(
                    _salient_memory_content(patch.subject, patch.slot, patch.value, patch.evidence),
                    metadata,
                )),
            )
            for previous in self.store.all():
                if not previous.active:
                    continue
                if previous.kind != "salient_fact":
                    continue
                if previous.metadata.get("memory_key") != key:
                    continue
                if previous.metadata.get("salient_value") == patch.value:
                    continue
                previous.superseded_by = salient_item.id
                previous.staleness = max(previous.staleness, 1.0)
                if source.id not in previous.stale_sources:
                    previous.stale_sources.append(source.id)
                salient_item.supersedes.append(previous.id)
                self.store.upsert(previous)
            salient_item.links.append(source.id)
            self.store.upsert(salient_item)

    def _mark_source_evidence_stale(
        self,
        previous_state: MemoryItem,
        replacement_state: MemoryItem,
        replacement_source: MemoryItem,
    ) -> None:
        source_id = previous_state.metadata.get("source_id")
        if not isinstance(source_id, str) or source_id == replacement_source.id:
            return
        evidence = self.store.get(source_id)
        if evidence is None or not evidence.active:
            return
        slot = str(previous_state.metadata.get("state_slot") or "")
        if not slot:
            return
        evidence.staleness = max(evidence.staleness, 1.0)
        if replacement_source.id not in evidence.stale_sources:
            evidence.stale_sources.append(replacement_source.id)
        _append_metadata_value(evidence.metadata, "stale_state_slots", slot)
        _append_metadata_value(evidence.metadata, "state_adjudicated_by", replacement_state.id)
        self.store.upsert(evidence)

    def _propagate_state_dependency(self, state_item: MemoryItem, source: MemoryItem) -> None:
        changed_slot = str(state_item.metadata.get("state_slot") or "")
        if not changed_slot:
            return
        for candidate in self.store.all():
            if not candidate.active:
                continue
            if candidate.kind != "state" or candidate.metadata.get("derived") is not True:
                continue
            if candidate.id == state_item.id:
                continue
            candidate_slot = str(candidate.metadata.get("state_slot") or "")
            if not state_slot_depends_on(candidate_slot, changed_slot):
                continue
            candidate.superseded_by = state_item.id
            candidate.staleness = max(candidate.staleness, 1.0)
            if source.id not in candidate.stale_sources:
                candidate.stale_sources.append(source.id)
            if candidate.id not in state_item.supersedes:
                state_item.supersedes.append(candidate.id)
            self.store.upsert(candidate)
            source_id = candidate.metadata.get("source_id")
            if isinstance(source_id, str):
                evidence = self.store.get(source_id)
                if evidence and evidence.active:
                    evidence.staleness = max(evidence.staleness, 1.0)
                    if source.id not in evidence.stale_sources:
                        evidence.stale_sources.append(source.id)
                    self.store.upsert(evidence)

    def _state_readout(
        self,
        query: str,
        query_embedding: dict[str, float],
        *,
        now: str | None,
    ) -> list[MemoryResult]:
        relevant_slots = set(query_relevant_state_slots(query))
        if not relevant_slots:
            return []
        results: list[MemoryResult] = []
        for item in self.store.all():
            if not item.active:
                continue
            if item.kind != "state":
                continue
            slot = str(item.metadata.get("state_slot") or "")
            if not state_slot_matches_query(slot, relevant_slots):
                continue
            result = self._score(item, query_embedding, now=now, relation="state")
            result.contributions["state_readout"] = self.config.state_readout_boost
            result.score = sum(result.contributions.values())
            results.append(result)
        return results

    def _temporal_kg_readout(
        self,
        query: str,
        query_embedding: dict[str, float],
        *,
        now: str | None,
    ) -> list[MemoryResult]:
        relevant_slots = set(query_relevant_state_slots(query))
        if not relevant_slots:
            return []
        results: list[MemoryResult] = []
        for item in self.store.all():
            if not item.active:
                continue
            if item.kind != "kg_fact":
                continue
            slot = str(item.metadata.get("state_slot") or item.metadata.get("kg_relation") or "")
            if not state_slot_matches_query(slot, relevant_slots):
                continue
            result = self._score(item, query_embedding, now=now, relation="temporal_kg")
            result.contributions["temporal_kg_readout"] = self.config.temporal_kg_readout_boost
            result.score = sum(result.contributions.values())
            results.append(result)
        return results

    def _salient_memory_readout(
        self,
        query: str,
        query_embedding: dict[str, float],
        *,
        now: str | None,
    ) -> list[MemoryResult]:
        relevant_slots = set(query_relevant_state_slots(query))
        if not relevant_slots:
            return []
        results: list[MemoryResult] = []
        for item in self.store.all():
            if not item.active:
                continue
            if item.kind != "salient_fact":
                continue
            slot = str(item.metadata.get("state_slot") or item.metadata.get("salient_slot") or "")
            if not state_slot_matches_query(slot, relevant_slots):
                continue
            result = self._score(item, query_embedding, now=now, relation="salient")
            result.contributions["salient_memory_readout"] = self.config.salient_memory_readout_boost
            result.score = sum(result.contributions.values())
            results.append(result)
        return results

    def _trajectory_step_readout(
        self,
        query: str,
        query_embedding: dict[str, float],
        *,
        now: str | None,
    ) -> list[MemoryResult]:
        step_indices = _query_step_indices(query)
        if not step_indices:
            return []
        results: list[MemoryResult] = []
        for item in self.store.all():
            if not item.active:
                continue
            if item.metadata.get("benchmark") != "ama":
                continue
            step = item.metadata.get("trajectory_step")
            if step is None:
                continue
            try:
                step_index = int(step)
            except (TypeError, ValueError):
                continue
            if step_index not in step_indices:
                continue
            result = self._score(item, query_embedding, now=now, relation="trajectory_step")
            result.contributions["trajectory_step_readout"] = self.config.trajectory_step_readout_boost
            if item.kind == "action":
                result.contributions["trajectory_action"] = 0.2
            result.score = sum(result.contributions.values())
            results.append(result)
        return results

    def _eligible_for_direct_retrieval(self, item: MemoryItem) -> bool:
        if self.config.use_salient_memory_only and item.kind != "salient_fact":
            return False
        if not self.config.use_state_readout_authorization:
            return True
        if item.kind == "state" or item.metadata.get("derived") is True:
            return False
        return True

    def _filter_state_adjudicated_sources(
        self,
        candidates: dict[str, MemoryResult],
        query: str,
    ) -> dict[str, MemoryResult]:
        relevant_slots = set(query_relevant_state_slots(query))
        if not relevant_slots:
            return candidates
        filtered: dict[str, MemoryResult] = {}
        for item_id, result in candidates.items():
            if self._state_source_adjudicates(result.item, relevant_slots):
                result.relation = f"{result.relation}+state_adjudicated"
                result.contributions["state_adjudicated"] = -1.0
                continue
            filtered[item_id] = result
        return filtered

    def _state_source_adjudicates(self, item: MemoryItem, relevant_slots: set[str]) -> bool:
        slots = _metadata_strings(item.metadata.get("stale_state_slots"))
        for slot in slots:
            if state_slot_matches_query(slot, relevant_slots) and self._has_active_state_for_slot(slot):
                return True
        return False

    def _has_active_state_for_slot(self, slot: str) -> bool:
        for item in self.store.all():
            if not item.active:
                continue
            if item.kind == "state" and item.metadata.get("state_slot") == slot:
                return True
        return False

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

    def _soft_stale_conflicting(self, item: MemoryItem) -> None:
        """Mechanism A: Soft Active-Stale Scoring.

        Without requiring an explicit `memory_key` collision, accumulate a
        graded `staleness` on prior memories that this new item likely
        contradicts. Two prior items count as conflict candidates when their
        embeddings are highly similar to `item` AND their normalized contents
        differ. The accumulation is bounded to keep the field interpretable
        as a probability-like score in [0, soft_stale_max].
        """
        cfg = self.config
        new_text = item.content.strip().lower()
        if not new_text:
            return
        directly_marked: list[MemoryItem] = []
        for previous in self.store.all():
            if previous.id == item.id:
                continue
            if previous.kind == "state" or previous.metadata.get("derived") is True:
                continue
            if previous.superseded_by is not None:
                continue
            if previous.content.strip().lower() == new_text:
                continue
            sim = cosine(item.embedding, previous.embedding)
            if sim < cfg.soft_stale_threshold:
                continue
            increment = cfg.soft_stale_increment * sim
            previous.staleness = min(cfg.soft_stale_max, previous.staleness + increment)
            if item.id not in previous.stale_sources:
                previous.stale_sources.append(item.id)
            self.store.upsert(previous)
            directly_marked.append(previous)
        if cfg.use_stale_propagation and directly_marked:
            self._propagate_stale(item, directly_marked)

    def _propagate_stale(self, item: MemoryItem, seeds: list[MemoryItem]) -> None:
        """Mechanism B: Propagation via Co-occurrence.

        When `seeds` get marked stale because of `item`, also raise the
        staleness of items that co-occur with each seed. Two memories
        co-occur when they share at least one `session_*` / `memory_key`
        tag in metadata, OR when they are explicitly linked / cause-related.
        Propagation strength is `seed.staleness * decay`, gated by a
        threshold so isolated singletons don't pull others down.
        """
        cfg = self.config
        decay = cfg.stale_propagation_decay
        if decay <= 0:
            return
        all_items = self.store.all()
        for seed in seeds:
            propagated_amount = seed.staleness * decay
            if propagated_amount < cfg.stale_propagation_threshold:
                continue
            seed_tags = _cooccurrence_tags(seed)
            for other in all_items:
                if other.kind == "state" or other.metadata.get("derived") is True:
                    continue
                if other.id in {item.id, seed.id}:
                    continue
                if other.superseded_by is not None:
                    continue
                if not _co_occurs(seed, other, seed_tags):
                    continue
                other.staleness = min(cfg.soft_stale_max, other.staleness + propagated_amount)
                if item.id not in other.stale_sources:
                    other.stale_sources.append(item.id)
                self.store.upsert(other)

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
        if cfg.use_soft_staleness and item.staleness > 0.0:
            contributions["staleness"] = -cfg.staleness_penalty * item.staleness
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

    def _adjudication_filter(self, ranked: list[MemoryResult]) -> list[MemoryResult]:
        """Mechanism C: Adjudication-Aware Retrieval.

        Drop candidates whose `staleness` already passes
        `adjudication_drop_threshold` AND that have at least one stale_source
        still alive in the store. Items dropped here are recorded by setting
        `result.relation` so callers can report Stale Leak Rate.
        """
        threshold = self.config.adjudication_drop_threshold
        kept: list[MemoryResult] = []
        for result in ranked:
            item = result.item
            if item.staleness >= threshold and item.stale_sources:
                live_sources = [sid for sid in item.stale_sources if self.store.get(sid) is not None]
                if live_sources:
                    # Record but do not surface; expose via `result.relation`
                    # for upstream metrics if anything inspects the dropped list.
                    result.relation = f"{result.relation}+adjudicated"
                    result.contributions["adjudicated"] = -1.0
                    continue
            kept.append(result)
        return kept


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
    for key in ("memory_key", "subject", "predicate", "keywords", "evolved_keywords", "tags"):
        value = metadata.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for entry in value:
                text = str(entry)
                # Skip structural tags (session_*, relevant) so they do not
                # dilute the semantic similarity used for soft staleness.
                if text.startswith("session_") or text == "relevant":
                    continue
                parts.append(text)
    return " ".join(parts)


def _temporal_kg_content(subject: str, relation: str, value: str, evidence: str) -> str:
    return (
        f"Temporal KG fact: {subject} {relation} = {value}.\n"
        f"Evidence: {evidence}"
    )


def _salient_memory_content(subject: str, slot: str, value: str, evidence: str) -> str:
    return (
        f"Extracted memory fact: {subject} {slot} is {value}.\n"
        f"Source evidence: {evidence}"
    )


def _cooccurrence_tags(item: MemoryItem) -> set[str]:
    """Tags used to define co-occurrence between two memories.

    We treat any `session_*` or `subject` / `memory_key` tag as a binding
    that means two items belong to the same context window.
    """
    tags: set[str] = set()
    raw_tags = item.metadata.get("tags") if isinstance(item.metadata, dict) else None
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            text = str(tag)
            if text.startswith("session_") or text in {"relevant"}:
                tags.add(text)
    subject = item.metadata.get("subject") if isinstance(item.metadata, dict) else None
    if isinstance(subject, str) and subject:
        tags.add(f"subject:{subject}")
    return tags


def _co_occurs(seed: MemoryItem, other: MemoryItem, seed_tags: set[str]) -> bool:
    if seed.id in other.links or other.id in seed.links:
        return True
    if seed.id in other.cause_ids or other.id in seed.cause_ids:
        return True
    if not seed_tags:
        return False
    other_tags = _cooccurrence_tags(other)
    return bool(seed_tags & other_tags)


EVOLUTION_STOPWORDS = {
    "assistant",
    "content",
    "dialogue",
    "memory",
    "observation",
    "session",
    "system",
    "thanks",
    "user",
}


def _evolution_keywords(content: str, metadata: dict[str, object], *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    candidates = tokenize(content)
    raw_tags = metadata.get("tags")
    if isinstance(raw_tags, list):
        candidates.extend(str(tag).lower() for tag in raw_tags)
    subject = metadata.get("subject")
    if isinstance(subject, str):
        candidates.append(subject.lower())
    keywords: list[str] = []
    for token in candidates:
        token = token.strip().lower()
        if len(token) < 3 or token.isdigit():
            continue
        if token.startswith("session_") or token in EVOLUTION_STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _metadata_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [entry for entry in (str(entry) for entry in value) if entry]
    return []


def _query_step_indices(query: str) -> set[int]:
    steps: set[int] = set()
    for match in re.finditer(
        r"\b(?:from|between)?\s*steps?\s+(\d+)\s*(?:-|to|through|and)\s*(?:step\s+)?(\d+)\b",
        query,
        flags=re.IGNORECASE,
    ):
        start = int(match.group(1))
        end = int(match.group(2))
        if abs(end - start) <= 20:
            low, high = sorted((start, end))
            steps.update(range(low, high + 1))
        else:
            steps.update({start, end})
    for match in re.finditer(r"\bstep\s+(\d+)\b", query, flags=re.IGNORECASE):
        steps.add(int(match.group(1)))
    return steps


def _append_metadata_value(metadata: dict[str, object], key: str, value: str) -> None:
    current = metadata.get(key)
    if isinstance(current, list):
        values = current
    elif current is None:
        values = []
    else:
        values = [str(current)]
    if value not in values:
        values.append(value)
    metadata[key] = values


def _extend_metadata_values(metadata: dict[str, object], key: str, values: list[str]) -> None:
    for value in values:
        _append_metadata_value(metadata, key, value)
