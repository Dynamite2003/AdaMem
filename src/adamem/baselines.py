from __future__ import annotations

from dataclasses import asdict, dataclass

from adamem.config import AdaMemConfig


@dataclass(slots=True, frozen=True)
class BaselineSpec:
    name: str
    category: str
    description: str
    config: AdaMemConfig

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def baseline_registry() -> dict[str, BaselineSpec]:
    """Stable baseline registry for paper experiments.

    Names in this registry are the canonical experiment identifiers. Keep them
    stable once a result is reported; add a new name instead of silently
    changing the behavior behind an old one.
    """

    semantic_only = dict(
        use_graph=False,
        use_temporal=False,
        use_importance=False,
        use_recency=False,
        use_confidence=False,
        use_feedback=False,
        use_mmr=False,
        use_supersession=False,
        use_auto_links=False,
        use_soft_staleness=False,
        use_stale_propagation=False,
        use_adjudication_filter=False,
    )
    specs = [
        BaselineSpec(
            name="semantic_only",
            category="raw_turn_retrieval",
            description="Hashed bag-of-words similarity over active raw memories only.",
            config=AdaMemConfig(**semantic_only),
        ),
        BaselineSpec(
            name="semantic_importance",
            category="raw_turn_retrieval",
            description="Semantic retrieval plus static importance weighting.",
            config=AdaMemConfig(**{**semantic_only, "use_importance": True}),
        ),
        BaselineSpec(
            name="semantic_temporal",
            category="raw_turn_retrieval",
            description="Semantic retrieval plus temporal validity scoring.",
            config=AdaMemConfig(**{**semantic_only, "use_temporal": True}),
        ),
        BaselineSpec(
            name="semantic_graph",
            category="raw_turn_retrieval",
            description="Semantic retrieval plus explicit graph expansion.",
            config=AdaMemConfig(**{**semantic_only, "use_graph": True}),
        ),
        BaselineSpec(
            name="a_mem_evolution",
            category="mainstream_approximation",
            description=(
                "API-free approximation of A-MEM-style memory notes, dynamic linking, "
                "and memory evolution over raw episodes."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_graph": True,
                "use_auto_links": True,
                "use_memory_evolution": True,
            }),
        ),
        BaselineSpec(
            name="zep_temporal_kg",
            category="mainstream_approximation",
            description=(
                "API-free approximation of Zep/Graphiti-style temporal KG facts "
                "with invalidated old edges and graph readout."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_graph": True,
                "use_temporal": True,
                "use_temporal_kg_memory": True,
                "use_temporal_kg_readout": True,
            }),
        ),
        BaselineSpec(
            name="mem0_extraction",
            category="mainstream_approximation",
            description=(
                "API-free approximation of Mem0-style compact memory extraction and "
                "slot-level update over extracted facts."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_salient_memory": True,
                "use_salient_memory_only": True,
                "use_salient_memory_readout": True,
            }),
        ),
        BaselineSpec(
            name="delta_graph",
            category="adamem_ablation",
            description="Graph retrieval with hard memory-key supersession.",
            config=AdaMemConfig(**{**semantic_only, "use_graph": True, "use_supersession": True}),
        ),
        BaselineSpec(
            name="delta_soft",
            category="adamem_ablation",
            description="Hard supersession plus soft stale scoring, without graph propagation.",
            config=AdaMemConfig(**{
                **semantic_only,
                "use_supersession": True,
                "use_soft_staleness": True,
            }),
        ),
        BaselineSpec(
            name="delta_propagation",
            category="adamem_ablation",
            description="Soft stale scoring plus propagation over co-occurrence links.",
            config=AdaMemConfig(**{
                **semantic_only,
                "use_supersession": True,
                "use_soft_staleness": True,
                "use_stale_propagation": True,
            }),
        ),
        BaselineSpec(
            name="delta_full",
            category="adamem_ablation",
            description="Delta mechanisms plus adjudication filter, without full retrieval scoring.",
            config=AdaMemConfig(**{
                **semantic_only,
                "use_supersession": True,
                "use_soft_staleness": True,
                "use_stale_propagation": True,
                "use_adjudication_filter": True,
            }),
        ),
        BaselineSpec(
            name="full",
            category="adamem_full",
            description="Default AdaMem scoring with temporal, importance, recency, graph, stale, and MMR signals.",
            config=AdaMemConfig(),
        ),
        BaselineSpec(
            name="state_memory",
            category="state_aware_ablation",
            description="Default AdaMem plus deterministic state extraction, without authorized state readout.",
            config=AdaMemConfig(use_state_memory=True),
        ),
        BaselineSpec(
            name="semantic_state_readout",
            category="state_aware_ablation",
            description="Semantic-only retrieval plus authorized deterministic current-state readout.",
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
            }),
        ),
        BaselineSpec(
            name="semantic_state_propagation",
            category="state_aware_ablation",
            description="Semantic-only retrieval plus state readout and typed state dependency propagation.",
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
                "use_state_dependency_propagation": True,
            }),
        ),
        BaselineSpec(
            name="semantic_state_adjudication",
            category="state_aware_ablation",
            description=(
                "Semantic-only retrieval plus authorized state readout and query-scoped filtering "
                "of raw evidence superseded by the same state slot."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
                "use_state_source_adjudication": True,
            }),
        ),
        BaselineSpec(
            name="semantic_state_propagation_adjudication",
            category="state_aware_ablation",
            description=(
                "Semantic state adjudication plus typed dependency propagation for indirectly "
                "invalidated state slots."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
                "use_state_dependency_propagation": True,
                "use_state_source_adjudication": True,
            }),
        ),
        BaselineSpec(
            name="state_readout",
            category="state_aware",
            description="State-aware AdaMem with deterministic state extraction and authorized current-state readout.",
            config=AdaMemConfig(use_state_memory=True, use_state_readout=True),
        ),
        BaselineSpec(
            name="state_propagation",
            category="state_aware",
            description="State readout plus typed dependency propagation from changed slots to dependent state slots.",
            config=AdaMemConfig(
                use_state_memory=True,
                use_state_readout=True,
                use_state_dependency_propagation=True,
            ),
        ),
    ]
    return {spec.name: spec for spec in specs}


def default_ablation_configs() -> dict[str, AdaMemConfig]:
    return {name: spec.config for name, spec in baseline_registry().items()}


def select_baselines(
    names: list[str] | tuple[str, ...] | None,
    specs: dict[str, BaselineSpec] | None = None,
) -> dict[str, BaselineSpec]:
    """Return a stable subset of baseline specs.

    The returned dictionary preserves the user-requested order so pilot runs can
    keep small, focused result tables without changing canonical baseline names.
    """

    specs = specs or baseline_registry()
    if not names:
        return specs
    selected: dict[str, BaselineSpec] = {}
    unknown: list[str] = []
    for name in names:
        if name not in specs:
            unknown.append(name)
            continue
        selected[name] = specs[name]
    if unknown:
        available = ", ".join(specs)
        missing = ", ".join(unknown)
        raise ValueError(f"unknown baseline(s): {missing}; available: {available}")
    return selected


def baseline_report(specs: dict[str, BaselineSpec] | None = None) -> str:
    specs = specs or baseline_registry()
    lines = ["# AdaMem Baseline Registry", ""]
    lines.append("| name | category | description |")
    lines.append("| --- | --- | --- |")
    for spec in specs.values():
        lines.append(f"| {spec.name} | {spec.category} | {spec.description} |")
    return "\n".join(lines) + "\n"
