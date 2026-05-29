from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from adamem.reporting import method_coverage_summary


DEFAULT_STALE_BASELINES = [
    "semantic_only",
    "semantic_temporal",
    "a_mem_evolution",
    "zep_temporal_kg",
    "mem0_extraction",
    "full",
    "semantic_state_readout",
    "semantic_state_adjudication",
    "semantic_state_premise_correction",
    "semantic_state_propagation_adjudication",
    "state_readout",
    "state_propagation",
]
DEFAULT_LLM_EXTRACTOR_BASELINES = [
    "semantic_state_adjudication",
    "semantic_llm_state_adjudication",
    "semantic_llm_state_premise_correction",
]
DEFAULT_TRANSFER_BASELINES = [
    "semantic_only",
    "a_mem_evolution",
    "zep_temporal_kg",
    "mem0_extraction",
    "semantic_state_readout",
    "semantic_state_adjudication",
    "semantic_state_premise_correction",
    "semantic_state_propagation_adjudication",
    "state_readout",
    "state_propagation",
]
DEFAULT_AMA_BASELINES = [
    "semantic_only",
    "full",
    "trajectory_step_readout",
]
DEFAULT_ANSWER_MODELS = [
    "<answer_provider_a>:<answer_model_a>",
    "<answer_provider_b>:<answer_model_b>",
]
DEFAULT_JUDGE_MODELS = [
    "<judge_provider_a>:<judge_model_a>",
    "<judge_provider_b>:<judge_model_b>",
]


@dataclass(slots=True, frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(slots=True)
class PlannedCommand:
    name: str
    stage: str
    purpose: str
    claim_boundary: str
    command: list[str]
    outputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def shell(self) -> str:
        return shlex.join(self.command)


def build_paper_study_plan(
    *,
    output_dir: str | Path,
    stale_dataset: str | Path = "benchmarks/stale.adamem.jsonl",
    transfer_dataset: str | Path = "benchmarks/longmemeval_s.adamem.jsonl",
    ama_output_source: str | Path | None = "results/ama_public_20_full/ama_public_20.raw.jsonl",
    stale_types: Iterable[str] = ("T1", "T2"),
    limit_per_stale_type: int = 50,
    transfer_max_cases: int = 60,
    ama_limit: int = 20,
    answer_models: Iterable[str] = DEFAULT_ANSWER_MODELS,
    judge_models: Iterable[str] = DEFAULT_JUDGE_MODELS,
    state_extractor_model: str = "<state_extractor_provider>:<state_extractor_model>",
    top_k: int = 8,
    max_context_chars: int = 4000,
) -> dict[str, Any]:
    output = Path(output_dir)
    stale_types = tuple(str(item) for item in stale_types)
    answers = [parse_model_spec(item) for item in answer_models]
    judges = [parse_model_spec(item) for item in judge_models]
    extractor = parse_model_spec(state_extractor_model)
    if not answers:
        raise ValueError("at least one answer model is required")
    if not judges:
        raise ValueError("at least one judge model is required")

    commands: list[PlannedCommand] = []
    commands.append(_stale_diagnostic_command(
        output,
        stale_dataset=stale_dataset,
        stale_types=stale_types,
        limit_per_stale_type=limit_per_stale_type,
        top_k=top_k,
    ))
    for answer in answers:
        for judge in judges:
            commands.append(_stale_answer_command(
                output,
                stale_dataset=stale_dataset,
                answer=answer,
                judge=judge,
                stale_types=stale_types,
                limit_per_stale_type=limit_per_stale_type,
                top_k=top_k,
                max_context_chars=max_context_chars,
            ))
    commands.append(_llm_extractor_ablation_command(
        output,
        stale_dataset=stale_dataset,
        answer=answers[0],
        judge=judges[0],
        extractor=extractor,
        stale_types=stale_types,
        limit_per_stale_type=limit_per_stale_type,
        top_k=top_k,
        max_context_chars=max_context_chars,
    ))
    commands.append(_transfer_retrieval_command(
        output,
        transfer_dataset=transfer_dataset,
        transfer_max_cases=transfer_max_cases,
        top_k=top_k,
    ))
    if ama_output_source is not None:
        commands.append(_ama_retrieval_command(
            output,
            source=ama_output_source,
            ama_limit=ama_limit,
            top_k=top_k,
        ))
    commands.append(_reporting_command(output))

    all_baselines = sorted({
        baseline
        for command in commands
        for baseline in _baselines_from_command(command.command)
    })
    method_coverage = method_coverage_summary([{
        "experiment": "planned_paper_study",
        "baselines": all_baselines,
    }])
    plan = {
        "schema_version": "adamem.paper_study_plan.v1",
        "output_dir": str(output),
        "objective": "paper_track_stale_memory_and_transfer_evaluation",
        "claim_boundary": (
            "This is an execution plan, not evidence. Claims become available only after "
            "the listed commands produce experiment records and adamem.reporting audits them."
        ),
        "datasets": {
            "primary_stale": str(stale_dataset),
            "transfer_long_memory": str(transfer_dataset),
            "transfer_ama_source": str(ama_output_source) if ama_output_source is not None else None,
        },
        "split": {
            "stale_types": list(stale_types),
            "limit_per_stale_type": limit_per_stale_type,
            "transfer_max_cases": transfer_max_cases,
            "ama_limit": ama_limit,
        },
        "model_requirements": {
            "answer_models": [item.label for item in answers],
            "judge_models": [item.label for item in judges],
            "state_extractor_model": extractor.label,
            "minimum_answer_models": 2,
            "minimum_judge_models": 2,
        },
        "baseline_sets": {
            "stale_main": list(DEFAULT_STALE_BASELINES),
            "llm_extractor_ablation": list(DEFAULT_LLM_EXTRACTOR_BASELINES),
            "transfer": list(DEFAULT_TRANSFER_BASELINES),
            "ama": list(DEFAULT_AMA_BASELINES),
        },
        "method_coverage_preview": method_coverage,
        "commands": [_command_dict(command) for command in commands],
        "post_run_gates": [
            "adamem.reporting must produce claim_matrix, method_coverage, benchmark_coverage, study_model_coverage, and paper_readiness artifacts.",
            "No answer-accuracy or SOTA claim is valid until raw outputs are cached and non-mock answer/judge models pass robustness gates.",
            "STALE remains the primary benchmark; transfer results are used for generality and no-regression evidence.",
        ],
    }
    return plan


def write_paper_study_plan(plan: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "paper_study_plan.json"
    md_path = output / "paper_study_plan.md"
    sh_path = output / "paper_study_commands.sh"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(paper_study_plan_markdown(plan), encoding="utf-8")
    sh_path.write_text(paper_study_plan_shell(plan), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "shell": str(sh_path),
    }


def paper_study_plan_markdown(plan: dict[str, Any]) -> str:
    lines = ["# AdaMem Paper Study Plan", ""]
    lines.append(f"Objective: `{plan.get('objective')}`")
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append(str(plan.get("claim_boundary") or ""))
    lines.append("")
    lines.append("## Datasets")
    for name, path in (plan.get("datasets") or {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    lines.append("## Model Requirements")
    requirements = plan.get("model_requirements") or {}
    lines.append("- Answer models: " + ", ".join(f"`{item}`" for item in requirements.get("answer_models") or []))
    lines.append("- Judge models: " + ", ".join(f"`{item}`" for item in requirements.get("judge_models") or []))
    lines.append(f"- State extractor: `{requirements.get('state_extractor_model')}`")
    coverage = plan.get("method_coverage_preview") or {}
    lines.append("")
    lines.append("## Method Coverage Preview")
    lines.append(f"- Complete required groups: `{bool(coverage.get('complete'))}`")
    missing = coverage.get("missing_requirements") or []
    lines.append("- Missing requirements: " + (", ".join(f"`{item}`" for item in missing) if missing else "`none`"))
    mechanisms = coverage.get("mechanism_flags") or {}
    for name, present in mechanisms.items():
        lines.append(f"- `{name}`: `{bool(present)}`")
    lines.append("")
    lines.append("## Commands")
    lines.append("| name | stage | purpose | output |")
    lines.append("| --- | --- | --- | --- |")
    for command in plan.get("commands") or []:
        output = ", ".join(f"`{key}`" for key in sorted((command.get("outputs") or {}).keys())) or "-"
        lines.append(
            f"| `{command['name']}` | `{command['stage']}` | "
            f"{command['purpose']} | {output} |"
        )
    lines.append("")
    lines.append("## Shell")
    lines.append("```bash")
    for command in plan.get("commands") or []:
        lines.append(command["shell"])
    lines.append("```")
    return "\n".join(lines) + "\n"


def paper_study_plan_shell(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated plan. Replace placeholder provider/model names before running API stages.",
        "",
    ]
    for command in plan.get("commands") or []:
        lines.append(f"# {command['name']}: {command['purpose']}")
        lines.append(command["shell"])
        lines.append("")
    return "\n".join(lines)


def parse_model_spec(value: str) -> ModelSpec:
    if ":" not in value:
        raise ValueError(f"model spec must be provider:model, got {value!r}")
    provider, model = value.split(":", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ValueError(f"model spec must be provider:model, got {value!r}")
    return ModelSpec(provider=provider, model=model)


def _stale_diagnostic_command(
    output: Path,
    *,
    stale_dataset: str | Path,
    stale_types: tuple[str, ...],
    limit_per_stale_type: int,
    top_k: int,
) -> PlannedCommand:
    experiment = output / "stale_diagnostics.experiment.json"
    cases = output / "stale_diagnostics.records.jsonl"
    report = output / "stale_diagnostics.report.md"
    command = [
        "python",
        "-m",
        "adamem.eval",
        "--stale-diagnostics",
        str(stale_dataset),
        "--baselines",
        *DEFAULT_STALE_BASELINES,
        "--stale-types",
        *stale_types,
        "--limit-per-stale-type",
        str(limit_per_stale_type),
        "--top-k",
        str(top_k),
        "--diagnostic-cases-output",
        str(cases),
        "--diagnostic-report-output",
        str(report),
        "--experiment-output",
        str(experiment),
    ]
    return PlannedCommand(
        name="stale_retrieval_diagnostics",
        stage="diagnostic",
        purpose="API-free STALE retrieval and stale-premise diagnostics for the full method matrix.",
        claim_boundary="diagnostics only; not answer accuracy",
        command=command,
        outputs={"experiment": str(experiment), "records": str(cases), "report": str(report)},
    )


def _stale_answer_command(
    output: Path,
    *,
    stale_dataset: str | Path,
    answer: ModelSpec,
    judge: ModelSpec,
    stale_types: tuple[str, ...],
    limit_per_stale_type: int,
    top_k: int,
    max_context_chars: int,
) -> PlannedCommand:
    label = _safe_label(f"{answer.label}__{judge.label}")
    experiment = output / f"stale_answer_{label}.experiment.json"
    command = [
        "python",
        "-m",
        "adamem.eval",
        "--stale",
        str(stale_dataset),
        "--baselines",
        *DEFAULT_STALE_BASELINES,
        "--answer-provider",
        answer.provider,
        "--answer-model",
        answer.model,
        "--judge-provider",
        judge.provider,
        "--judge-model",
        judge.model,
        "--stale-types",
        *stale_types,
        "--limit-per-stale-type",
        str(limit_per_stale_type),
        "--top-k",
        str(top_k),
        "--max-context-chars",
        str(max_context_chars),
        "--experiment-output",
        str(experiment),
    ]
    return PlannedCommand(
        name=f"stale_answer_{label}",
        stage="answer_judge",
        purpose="End-to-end STALE answer generation and judge scoring for one answer/judge model pair.",
        claim_boundary="candidate answer accuracy only after non-mock model and reproducibility audits pass",
        command=command,
        outputs={"experiment": str(experiment)},
    )


def _llm_extractor_ablation_command(
    output: Path,
    *,
    stale_dataset: str | Path,
    answer: ModelSpec,
    judge: ModelSpec,
    extractor: ModelSpec,
    stale_types: tuple[str, ...],
    limit_per_stale_type: int,
    top_k: int,
    max_context_chars: int,
) -> PlannedCommand:
    label = _safe_label(f"{extractor.label}__{answer.label}__{judge.label}")
    experiment = output / f"stale_llm_extractor_{label}.experiment.json"
    command = [
        "python",
        "-m",
        "adamem.eval",
        "--stale",
        str(stale_dataset),
        "--baselines",
        *DEFAULT_LLM_EXTRACTOR_BASELINES,
        "--answer-provider",
        answer.provider,
        "--answer-model",
        answer.model,
        "--judge-provider",
        judge.provider,
        "--judge-model",
        judge.model,
        "--state-extractor-provider",
        extractor.provider,
        "--state-extractor-model",
        extractor.model,
        "--stale-types",
        *stale_types,
        "--limit-per-stale-type",
        str(limit_per_stale_type),
        "--top-k",
        str(top_k),
        "--max-context-chars",
        str(max_context_chars),
        "--experiment-output",
        str(experiment),
    ]
    return PlannedCommand(
        name=f"stale_llm_extractor_{label}",
        stage="mechanism_ablation",
        purpose="LLM state-extractor ablation using the same state authority and readout layer.",
        claim_boundary="mechanism ablation; compare against deterministic state extraction before claiming lift",
        command=command,
        outputs={"experiment": str(experiment)},
    )


def _transfer_retrieval_command(
    output: Path,
    *,
    transfer_dataset: str | Path,
    transfer_max_cases: int,
    top_k: int,
) -> PlannedCommand:
    experiment = output / "longmemeval_transfer.experiment.json"
    records = output / "longmemeval_transfer.records.jsonl"
    report = output / "longmemeval_transfer.report.md"
    command = [
        "python",
        "-m",
        "adamem.eval",
        "--dataset",
        str(transfer_dataset),
        "--baselines",
        *DEFAULT_TRANSFER_BASELINES,
        "--max-cases",
        str(transfer_max_cases),
        "--top-k",
        str(top_k),
        "--benchmark-cases-output",
        str(records),
        "--benchmark-report-output",
        str(report),
        "--experiment-output",
        str(experiment),
    ]
    return PlannedCommand(
        name="longmemeval_transfer_retrieval",
        stage="transfer",
        purpose="Public long-memory transfer retrieval/no-regression diagnostic.",
        claim_boundary="transfer diagnostic; not generated answer accuracy",
        command=command,
        outputs={"experiment": str(experiment), "records": str(records), "report": str(report)},
    )


def _ama_retrieval_command(
    output: Path,
    *,
    source: str | Path,
    ama_limit: int,
    top_k: int,
) -> PlannedCommand:
    ama_dir = output / "ama_public"
    command = [
        "python",
        "-m",
        "adamem.pilot",
        "ama-public",
        "--limit",
        str(ama_limit),
        "--source",
        str(source),
        "--output-dir",
        str(ama_dir),
        "--baselines",
        *DEFAULT_AMA_BASELINES,
        "--top-k",
        str(top_k),
        "--answer-only",
        "--json",
    ]
    return PlannedCommand(
        name="ama_public_retrieval",
        stage="transfer",
        purpose="Agent-trajectory transfer diagnostic with trajectory-step readout baseline.",
        claim_boundary="answerability and evidence-support diagnostic; not final answer accuracy unless generation is added",
        command=command,
        outputs={"output_dir": str(ama_dir)},
    )


def _reporting_command(output: Path) -> PlannedCommand:
    report_dir = output / "report_bundle"
    command = [
        "python",
        "-m",
        "adamem.reporting",
        str(output),
        "--output-dir",
        str(report_dir),
        "--group-fields",
        "dim",
        "stale_type",
        "question_type",
        "selection_group",
        "--json",
    ]
    return PlannedCommand(
        name="paper_report_bundle",
        stage="reporting",
        purpose="Generate claim, method, benchmark, model, reproducibility, and readiness audits.",
        claim_boundary="post-run audit only",
        command=command,
        outputs={"output_dir": str(report_dir)},
        notes=["Run after experiment commands have produced JSON records."],
    )


def _command_dict(command: PlannedCommand) -> dict[str, Any]:
    data = asdict(command)
    data["shell"] = command.shell()
    return data


def _baselines_from_command(command: list[str]) -> list[str]:
    if "--baselines" not in command:
        return []
    start = command.index("--baselines") + 1
    names: list[str] = []
    for token in command[start:]:
        if token.startswith("--"):
            break
        names.append(token)
    return names


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return label or "model"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a paper-track AdaMem study plan without running API calls."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stale-dataset", default="benchmarks/stale.adamem.jsonl")
    parser.add_argument("--transfer-dataset", default="benchmarks/longmemeval_s.adamem.jsonl")
    parser.add_argument("--ama-output-source", default="results/ama_public_20_full/ama_public_20.raw.jsonl")
    parser.add_argument("--no-ama", action="store_true", help="Omit the AMA transfer command.")
    parser.add_argument("--stale-types", nargs="+", default=["T1", "T2"])
    parser.add_argument("--limit-per-stale-type", type=int, default=50)
    parser.add_argument("--transfer-max-cases", type=int, default=60)
    parser.add_argument("--ama-limit", type=int, default=20)
    parser.add_argument("--answer-model", action="append", dest="answer_models")
    parser.add_argument("--judge-model", action="append", dest="judge_models")
    parser.add_argument(
        "--state-extractor-model",
        default="<state_extractor_provider>:<state_extractor_model>",
        help="provider:model for LLM extractor ablation.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_paper_study_plan(
            output_dir=args.output_dir,
            stale_dataset=args.stale_dataset,
            transfer_dataset=args.transfer_dataset,
            ama_output_source=None if args.no_ama else args.ama_output_source,
            stale_types=args.stale_types,
            limit_per_stale_type=args.limit_per_stale_type,
            transfer_max_cases=args.transfer_max_cases,
            ama_limit=args.ama_limit,
            answer_models=args.answer_models or DEFAULT_ANSWER_MODELS,
            judge_models=args.judge_models or DEFAULT_JUDGE_MODELS,
            state_extractor_model=args.state_extractor_model,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
        )
    except ValueError as exc:
        parser.error(str(exc))
    artifacts = write_paper_study_plan(plan, args.output_dir)
    result = {"artifacts": artifacts, "plan": plan}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"wrote paper study plan to {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
