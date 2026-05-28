from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AdaMemConfig:
    """Switches are intentionally coarse so ablations stay readable."""

    novelty_threshold: float = 0.92
    link_threshold: float = 0.5
    recency_half_life_seconds: float = 14 * 24 * 60 * 60
    max_graph_hops: int = 1
    graph_boost: float = 0.28
    temporal_mismatch_penalty: float = 0.35
    min_direct_relevance: float = 1e-9
    mmr_lambda: float = 0.72

    use_semantic: bool = True
    use_temporal: bool = True
    use_importance: bool = True
    use_recency: bool = True
    use_confidence: bool = True
    use_graph: bool = True
    use_auto_links: bool = False
    use_feedback: bool = True
    use_mmr: bool = True
    use_supersession: bool = True
    use_soft_staleness: bool = True
    use_stale_propagation: bool = True
    use_adjudication_filter: bool = True

    semantic_weight: float = 0.58
    temporal_weight: float = 0.08
    importance_weight: float = 0.38
    recency_weight: float = 0.08
    confidence_weight: float = 0.06
    feedback_weight: float = 0.08
    staleness_penalty: float = 0.55
    soft_stale_threshold: float = 0.65
    soft_stale_increment: float = 0.5
    soft_stale_max: float = 1.0
    stale_propagation_decay: float = 0.5
    stale_propagation_threshold: float = 0.15
    adjudication_drop_threshold: float = 0.6
