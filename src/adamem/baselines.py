from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adamem.config import AdaMemConfig


@dataclass(slots=True, frozen=True)
class BaselineSpec:
    name: str
    category: str
    description: str
    config: AdaMemConfig
    source_name: str = "AdaMem"
    source_url: str = ""
    implementation_status: str = "adamem_native"
    reproduction_note: str = "Project-native method or local control."
    reproduction_target_name: str = ""
    reproduction_target_url: str = ""
    reproduction_target_note: str = ""

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def provenance_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "implementation_status": self.implementation_status,
            "reproduction_note": self.reproduction_note,
            "reproduction_target_name": self.reproduction_target_name,
            "reproduction_target_url": self.reproduction_target_url,
            "reproduction_target_note": self.reproduction_target_note,
        }


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
            source_name="A-MEM",
            source_url="https://arxiv.org/abs/2502.12110",
            implementation_status="api_free_approximation",
            reproduction_note=(
                "Approximates memory evolution locally; replace with or validate against "
                "the official implementation before SOTA-style claims."
            ),
            reproduction_target_name="A-MEM reproduction code",
            reproduction_target_url="https://github.com/WujiangXu/A-mem",
            reproduction_target_note=(
                "Use the paper reproduction repository for official/faithful LoCoMo-style runs."
            ),
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
            source_name="Zep/Graphiti",
            source_url="https://arxiv.org/abs/2501.13956",
            implementation_status="api_free_approximation",
            reproduction_note=(
                "Approximates temporal KG validity and readout locally; replace with or "
                "validate against Graphiti/Zep before SOTA-style claims."
            ),
            reproduction_target_name="Graphiti",
            reproduction_target_url="https://github.com/getzep/graphiti",
            reproduction_target_note=(
                "Use the open-source temporal context graph engine for faithful temporal-KG baselines."
            ),
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
            source_name="Mem0",
            source_url="https://arxiv.org/abs/2504.19413",
            implementation_status="api_free_approximation",
            reproduction_note=(
                "Approximates salient extraction and compact readout locally; replace "
                "with or validate against official Mem0 before SOTA-style claims."
            ),
            reproduction_target_name="Mem0",
            reproduction_target_url="https://github.com/mem0ai/mem0",
            reproduction_target_note=(
                "Use the official memory-layer implementation or evaluation framework for faithful runs."
            ),
        ),
        BaselineSpec(
            name="trajectory_step_readout",
            category="trajectory_memory_ablation",
            description=(
                "Semantic retrieval plus authorized trajectory-step readout for queries "
                "that explicitly mention Step N or short step ranges."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_trajectory_step_readout": True,
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
            name="semantic_state_adjudication_trace",
            category="state_aware_ablation",
            description=(
                "Semantic state adjudication plus an explicit authorized trace explaining "
                "why stale raw evidence was suppressed."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
                "use_state_source_adjudication": True,
                "use_state_adjudication_trace": True,
            }),
        ),
        BaselineSpec(
            name="semantic_state_premise_correction",
            category="state_aware_ablation",
            description=(
                "Semantic state adjudication plus explicit ephemeral correction when a query "
                "mentions a stale value for an authorized current-state slot."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
                "use_state_source_adjudication": True,
                "use_state_premise_correction": True,
            }),
        ),
        BaselineSpec(
            name="semantic_llm_state_adjudication",
            category="state_extractor_ablation",
            description=(
                "Semantic state adjudication using an injected LLM JSON state extractor "
                "instead of the deterministic rule extractor."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
                "use_state_source_adjudication": True,
                "state_extractor_name": "llm_json",
            }),
        ),
        BaselineSpec(
            name="semantic_llm_state_premise_correction",
            category="state_extractor_ablation",
            description=(
                "LLM JSON state extraction plus semantic state adjudication and "
                "explicit stale-premise correction."
            ),
            config=AdaMemConfig(**{
                **semantic_only,
                "use_state_memory": True,
                "use_state_readout": True,
                "use_state_source_adjudication": True,
                "use_state_premise_correction": True,
                "state_extractor_name": "llm_json",
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
    return {
        name: spec.config
        for name, spec in baseline_registry().items()
        if spec.config.state_extractor_name != "llm_json"
    }


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
        return {
            name: spec
            for name, spec in specs.items()
            if spec.config.state_extractor_name != "llm_json"
        }
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
    lines.append("| name | category | implementation | source | reproduction target | description |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for spec in specs.values():
        source = spec.source_name
        if spec.source_url:
            source = f"[{source}]({spec.source_url})"
        target = spec.reproduction_target_name or "-"
        if spec.reproduction_target_url:
            target = f"[{target}]({spec.reproduction_target_url})"
        lines.append(
            f"| {spec.name} | {spec.category} | {spec.implementation_status} | "
            f"{source} | {target} | {spec.description} |"
        )
    return "\n".join(lines) + "\n"


def baseline_reproduction_plan(
    specs: dict[str, BaselineSpec] | None = None,
) -> dict[str, Any]:
    """Return the paper-track plan for replacing approximation baselines.

    The current project-native mainstream controls are useful for API-free
    ablations, but SOTA-style claims need official or faithful reproductions on
    the same benchmark split. This artifact makes that requirement explicit and
    machine-checkable before expensive answer/judge runs.
    """

    specs = specs or baseline_registry()
    targets: list[dict[str, Any]] = []
    for spec in specs.values():
        if spec.implementation_status != "api_free_approximation":
            continue
        targets.append({
            "baseline": spec.name,
            "source_name": spec.source_name,
            "source_url": spec.source_url,
            "current_status": spec.implementation_status,
            "current_role": "api_free_local_control_not_sota_baseline",
            "reproduction_target_name": spec.reproduction_target_name,
            "reproduction_target_url": spec.reproduction_target_url,
            "reproduction_target_note": spec.reproduction_target_note,
            "required_status_after_run": [
                "official_reproduction",
                "faithful_reimplementation",
            ],
            "required_evidence": [
                "external_repo_url",
                "external_repo_commit",
                "adapter_or_command",
                "dataset_split_and_question_ids",
                "model_provider_model_and_sampling_settings",
                "prompt_or_memory_policy_if_applicable",
                "raw_case_records_path",
                "metric_mapping_to_adamem_outputs",
                "license_and_dependency_notes",
            ],
            "claim_boundary": (
                "Treat the current API-free approximation as a mechanism control only. "
                "Do not use it as strong-baseline or SOTA evidence until a matching "
                "official or faithful run is recorded in experiment baseline_provenance."
            ),
        })
    return {
        "schema_version": "adamem.baseline_reproduction_plan.v1",
        "target_count": len(targets),
        "targets": targets,
        "ready_for_sota_claims": False,
        "ready_reason": (
            "This is a reproduction plan template. SOTA readiness is established by "
            "experiment artifacts whose baseline_provenance records official_reproduction "
            "or faithful_reimplementation for at least one mainstream baseline."
        ),
    }


def baseline_reproduction_plan_markdown(plan: dict[str, Any]) -> str:
    lines = ["# AdaMem Baseline Reproduction Plan", ""]
    lines.append(f"Schema: `{plan.get('schema_version')}`")
    lines.append(f"Targets: `{int(plan.get('target_count') or 0)}`")
    lines.append(f"Ready for SOTA claims: `{bool(plan.get('ready_for_sota_claims'))}`")
    lines.append("")
    lines.append(str(plan.get("ready_reason") or ""))
    lines.append("")
    lines.append("| baseline | source | current status | reproduction target | required next status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for target in plan.get("targets") or []:
        source = str(target.get("source_name") or "-")
        source_url = str(target.get("source_url") or "")
        if source_url:
            source = f"[{source}]({source_url})"
        reproduction = str(target.get("reproduction_target_name") or "-")
        reproduction_url = str(target.get("reproduction_target_url") or "")
        if reproduction_url:
            reproduction = f"[{reproduction}]({reproduction_url})"
        status = ", ".join(f"`{item}`" for item in target.get("required_status_after_run") or [])
        lines.append(
            f"| `{target.get('baseline')}` | {source} | "
            f"`{target.get('current_status')}` | {reproduction} | {status} |"
        )
    lines.append("")
    lines.append("## Required Evidence")
    required = []
    for target in plan.get("targets") or []:
        required.extend(str(item) for item in target.get("required_evidence") or [])
    for item in sorted(set(required)):
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Claim Boundary")
    for target in plan.get("targets") or []:
        lines.append(f"- `{target.get('baseline')}`: {target.get('claim_boundary')}")
    return "\n".join(lines) + "\n"


def write_baseline_reproduction_plan(output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = baseline_reproduction_plan()
    json_path = output / "baseline_reproduction_plan.json"
    markdown_path = output / "baseline_reproduction_plan.md"
    json_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(baseline_reproduction_plan_markdown(plan), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AdaMem baseline registry utilities.")
    parser.add_argument("--output-dir", type=Path, help="Write a baseline reproduction plan artifact.")
    parser.add_argument("--reproduction-plan", action="store_true", help="Print the reproduction plan instead of the registry.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    if args.output_dir:
        result = write_baseline_reproduction_plan(args.output_dir)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"wrote baseline reproduction plan to {args.output_dir}")
            print(f"report: {result['markdown_path']}")
        return

    if args.reproduction_plan:
        plan = baseline_reproduction_plan()
        if args.json:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(baseline_reproduction_plan_markdown(plan), end="")
        return

    if args.json:
        print(json.dumps({
            name: spec.provenance_dict()
            for name, spec in baseline_registry().items()
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(baseline_report(), end="")


if __name__ == "__main__":
    main(sys.argv[1:])
