from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_STALE_SOURCE = "data/T1_T2_400_FULL.json"
DEFAULT_TRANSFER_SOURCE = "data/longmemeval_s_cleaned.json"


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
    stale_source: str | Path | None = DEFAULT_STALE_SOURCE,
    transfer_source: str | Path | None = DEFAULT_TRANSFER_SOURCE,
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
    if stale_source is not None:
        commands.append(_stale_conversion_command(
            stale_source,
            stale_dataset=stale_dataset,
            stale_types=stale_types,
            top_k=top_k,
        ))
    if transfer_source is not None:
        commands.append(_transfer_conversion_command(
            transfer_source,
            transfer_dataset=transfer_dataset,
            top_k=top_k,
        ))
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
        "data_sources": {
            "primary_stale": str(stale_source) if stale_source is not None else None,
            "transfer_long_memory": str(transfer_source) if transfer_source is not None else None,
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
    validation_path = output / "paper_study_validation.json"
    validation_md_path = output / "paper_study_validation.md"
    validation = validate_paper_study_plan(plan)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(paper_study_plan_markdown(plan), encoding="utf-8")
    sh_path.write_text(paper_study_plan_shell(plan), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_md_path.write_text(paper_study_validation_markdown(validation), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "shell": str(sh_path),
        "validation_json": str(validation_path),
        "validation_markdown": str(validation_md_path),
    }


def validate_paper_study_plan(
    plan: dict[str, Any],
    *,
    root: str | Path | None = None,
    check_env: bool = False,
) -> dict[str, Any]:
    root_path = Path(root) if root is not None else Path.cwd()
    datasets = plan.get("datasets") or {}
    data_sources = plan.get("data_sources") or {}
    prep_sources = _prep_sources_by_dataset(plan)
    dataset_checks: dict[str, dict[str, Any]] = {}
    missing_datasets: list[str] = []
    for name, value in datasets.items():
        if value in {None, ""}:
            dataset_checks[name] = {
                "path": value,
                "required": False,
                "exists": None,
            }
            continue
        path = Path(str(value))
        resolved = path if path.is_absolute() else root_path / path
        exists = resolved.exists()
        prep_source = prep_sources.get(str(value))
        source_exists = None
        if prep_source:
            source_path = Path(prep_source)
            resolved_source = source_path if source_path.is_absolute() else root_path / source_path
            source_exists = resolved_source.exists()
        dataset_checks[name] = {
            "path": str(value),
            "required": True,
            "exists": exists,
            "prepared_by_plan": bool(prep_source),
            "prep_source": prep_source,
            "prep_source_exists": source_exists,
        }
        if not exists and not (prep_source and source_exists):
            missing_datasets.append(name)

    source_checks: dict[str, dict[str, Any]] = {}
    missing_sources: list[str] = []
    for name, value in data_sources.items():
        if value in {None, ""}:
            source_checks[name] = {
                "path": value,
                "required": False,
                "exists": None,
            }
            continue
        path = Path(str(value))
        resolved = path if path.is_absolute() else root_path / path
        exists = resolved.exists()
        source_checks[name] = {
            "path": str(value),
            "required": True,
            "exists": exists,
        }
        if not exists and name in missing_datasets:
            missing_sources.append(name)

    requirements = plan.get("model_requirements") or {}
    answer_models = list(requirements.get("answer_models") or [])
    judge_models = list(requirements.get("judge_models") or [])
    extractor_model = requirements.get("state_extractor_model")
    minimum_answer_models = int(requirements.get("minimum_answer_models") or 2)
    minimum_judge_models = int(requirements.get("minimum_judge_models") or 2)
    model_labels = [*answer_models, *judge_models]
    if extractor_model:
        model_labels.append(str(extractor_model))
    placeholders = [label for label in model_labels if _contains_placeholder(label)]

    providers = sorted({
        spec.provider
        for label in model_labels
        if not _contains_placeholder(label)
        for spec in [_safe_parse_model_spec(label)]
        if spec is not None and spec.provider != "mock"
    })
    required_env_vars = _required_env_vars(providers)
    missing_env_vars = [
        name for name in required_env_vars if not os.environ.get(name)
    ] if check_env else []

    method_coverage = plan.get("method_coverage_preview") or {}
    commands = list(plan.get("commands") or [])
    command_stages = _count_values(command.get("stage") for command in commands)
    reporting_command_present = any(command.get("stage") == "reporting" for command in commands)

    missing_requirements: list[str] = []
    if missing_datasets:
        missing_requirements.append("dataset_paths_exist")
    if placeholders:
        missing_requirements.append("replace_model_placeholders")
    if len(set(answer_models)) < minimum_answer_models:
        missing_requirements.append("multiple_answer_models")
    if len(set(judge_models)) < minimum_judge_models:
        missing_requirements.append("multiple_judge_models")
    if not bool(method_coverage.get("complete")):
        missing_requirements.append("method_coverage_complete")
    if not reporting_command_present:
        missing_requirements.append("reporting_command_present")
    if missing_env_vars:
        missing_requirements.append("provider_credentials_available")

    return {
        "schema_version": "adamem.paper_study_validation.v1",
        "execution_ready": not missing_requirements,
        "missing_requirements": missing_requirements,
        "dataset_checks": dataset_checks,
        "missing_datasets": missing_datasets,
        "source_checks": source_checks,
        "missing_sources": missing_sources,
        "placeholder_models": placeholders,
        "answer_model_count": len(set(answer_models)),
        "judge_model_count": len(set(judge_models)),
        "minimum_answer_models": minimum_answer_models,
        "minimum_judge_models": minimum_judge_models,
        "provider_names": providers,
        "required_env_vars": required_env_vars,
        "env_checked": check_env,
        "missing_env_vars": missing_env_vars,
        "method_coverage_complete": bool(method_coverage.get("complete")),
        "method_missing_requirements": list(method_coverage.get("missing_requirements") or []),
        "method_missing_named_mechanism_ablations": list(
            method_coverage.get("missing_named_mechanism_ablations") or []
        ),
        "command_count": len(commands),
        "command_stage_counts": command_stages,
        "reporting_command_present": reporting_command_present,
    }


def paper_study_validation_markdown(validation: dict[str, Any]) -> str:
    lines = ["# AdaMem Paper Study Validation", ""]
    lines.append(f"Execution ready: `{bool(validation.get('execution_ready'))}`")
    missing = validation.get("missing_requirements") or []
    lines.append(
        "Missing requirements: "
        + (", ".join(f"`{item}`" for item in missing) if missing else "`none`")
    )
    lines.append("")
    lines.append("## Datasets")
    for name, check in (validation.get("dataset_checks") or {}).items():
        exists = check.get("exists")
        state = "optional" if exists is None else str(bool(exists))
        prep = ""
        if check.get("prepared_by_plan"):
            prep = (
                f", prepared by `{check.get('prep_source')}` "
                f"source exists `{check.get('prep_source_exists')}`"
            )
        lines.append(f"- `{name}`: `{check.get('path')}` exists `{state}`{prep}")
    source_checks = validation.get("source_checks") or {}
    if source_checks:
        lines.append("")
        lines.append("## Data Sources")
        for name, check in source_checks.items():
            exists = check.get("exists")
            state = "optional" if exists is None else str(bool(exists))
            lines.append(f"- `{name}`: `{check.get('path')}` exists `{state}`")
    placeholders = validation.get("placeholder_models") or []
    if placeholders:
        lines.append("")
        lines.append("## Placeholder Models")
        for label in placeholders:
            lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Models")
    lines.append(f"- Answer models: `{int(validation.get('answer_model_count') or 0)}`")
    lines.append(f"- Judge models: `{int(validation.get('judge_model_count') or 0)}`")
    env_vars = validation.get("required_env_vars") or []
    lines.append(
        "- Required env vars: "
        + (", ".join(f"`{item}`" for item in env_vars) if env_vars else "`none`")
    )
    if validation.get("env_checked"):
        missing_env = validation.get("missing_env_vars") or []
        lines.append(
            "- Missing env vars: "
            + (", ".join(f"`{item}`" for item in missing_env) if missing_env else "`none`")
        )
    lines.append("")
    lines.append("## Method Coverage")
    lines.append(f"- Complete: `{bool(validation.get('method_coverage_complete'))}`")
    method_gaps = validation.get("method_missing_requirements") or []
    lines.append(
        "- Missing groups: "
        + (", ".join(f"`{item}`" for item in method_gaps) if method_gaps else "`none`")
    )
    mechanism_gaps = validation.get("method_missing_named_mechanism_ablations") or []
    lines.append(
        "- Missing named mechanism ablations: "
        + (", ".join(f"`{item}`" for item in mechanism_gaps) if mechanism_gaps else "`none`")
    )
    lines.append("")
    lines.append("## Commands")
    lines.append(f"- Total: `{int(validation.get('command_count') or 0)}`")
    for stage, count in (validation.get("command_stage_counts") or {}).items():
        lines.append(f"- `{stage}`: `{count}`")
    lines.append(f"- Reporting command present: `{bool(validation.get('reporting_command_present'))}`")
    return "\n".join(lines) + "\n"


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


def _stale_conversion_command(
    source: str | Path,
    *,
    stale_dataset: str | Path,
    stale_types: tuple[str, ...],
    top_k: int,
) -> PlannedCommand:
    command = [
        "python",
        "-m",
        "adamem.convert",
        "stale",
        str(source),
        str(stale_dataset),
        "--top-k",
        str(top_k),
        "--types",
        *stale_types,
    ]
    return PlannedCommand(
        name="prepare_primary_stale_dataset",
        stage="data_prep",
        purpose="Convert the raw STALE file into AdaMem JSONL for primary stale-memory experiments.",
        claim_boundary="data preparation only; no method evidence",
        command=command,
        outputs={"source": str(source), "dataset": str(stale_dataset)},
    )


def _transfer_conversion_command(
    source: str | Path,
    *,
    transfer_dataset: str | Path,
    top_k: int,
) -> PlannedCommand:
    command = [
        "python",
        "-m",
        "adamem.convert",
        "longmemeval",
        str(source),
        str(transfer_dataset),
        "--expected",
        "evidence",
        "--top-k",
        str(top_k),
    ]
    return PlannedCommand(
        name="prepare_longmemeval_transfer_dataset",
        stage="data_prep",
        purpose="Convert LongMemEval-S source data into AdaMem JSONL for transfer diagnostics.",
        claim_boundary="data preparation only; transfer evidence comes from later eval records",
        command=command,
        outputs={"source": str(source), "dataset": str(transfer_dataset)},
    )


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


def _prep_sources_by_dataset(plan: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for command in plan.get("commands") or []:
        if command.get("stage") != "data_prep":
            continue
        outputs = command.get("outputs") or {}
        dataset = outputs.get("dataset")
        source = outputs.get("source")
        if dataset and source:
            sources[str(dataset)] = str(source)
    return sources


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return label or "model"


def _contains_placeholder(value: str) -> bool:
    return "<" in value or ">" in value


def _safe_parse_model_spec(value: str) -> ModelSpec | None:
    try:
        return parse_model_spec(value)
    except ValueError:
        return None


def _required_env_vars(providers: Iterable[str]) -> list[str]:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "modelhub": "MODELHUB_API_KEY",
    }
    return sorted({
        mapping[provider]
        for provider in providers
        if provider in mapping
    })


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "<missing>")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a paper-track AdaMem study plan without running API calls."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stale-dataset", default="benchmarks/stale.adamem.jsonl")
    parser.add_argument("--transfer-dataset", default="benchmarks/longmemeval_s.adamem.jsonl")
    parser.add_argument("--stale-source", default=DEFAULT_STALE_SOURCE)
    parser.add_argument("--transfer-source", default=DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--no-data-prep", action="store_true", help="Omit dataset conversion commands.")
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
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Also check whether required provider credential environment variables are set.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_paper_study_plan(
            output_dir=args.output_dir,
            stale_dataset=args.stale_dataset,
            transfer_dataset=args.transfer_dataset,
            stale_source=None if args.no_data_prep else args.stale_source,
            transfer_source=None if args.no_data_prep else args.transfer_source,
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
    if args.check_env:
        validation = validate_paper_study_plan(plan, check_env=True)
        Path(artifacts["validation_json"]).write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        Path(artifacts["validation_markdown"]).write_text(
            paper_study_validation_markdown(validation),
            encoding="utf-8",
        )
    else:
        validation = json.loads(Path(artifacts["validation_json"]).read_text(encoding="utf-8"))
    result = {"artifacts": artifacts, "plan": plan, "validation": validation}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"wrote paper study plan to {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
