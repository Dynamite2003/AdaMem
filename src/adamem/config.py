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
    use_memory_evolution: bool = False
    use_supersession: bool = True
    use_soft_staleness: bool = True
    use_stale_propagation: bool = True
    use_adjudication_filter: bool = True
    use_state_memory: bool = False
    use_state_readout: bool = False
    use_state_dependency_propagation: bool = False
    use_state_source_adjudication: bool = False
    use_state_readout_authorization: bool = True
    use_temporal_kg_memory: bool = False
    use_temporal_kg_readout: bool = False
    use_salient_memory: bool = False
    use_salient_memory_only: bool = False
    use_salient_memory_readout: bool = False
    use_trajectory_step_readout: bool = False

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
    state_readout_boost: float = 2.0
    temporal_kg_readout_boost: float = 1.6
    salient_memory_readout_boost: float = 1.8
    trajectory_step_readout_boost: float = 2.2
    memory_evolution_threshold: float = 0.28
    memory_evolution_keyword_limit: int = 8
    memory_evolution_candidate_limit: int = 48
