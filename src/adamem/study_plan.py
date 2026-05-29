from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
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
SMOKE_ANSWER_MODELS = ["mock:answer-a", "mock:answer-b"]
SMOKE_JUDGE_MODELS = ["mock:judge-a", "mock:judge-b"]
SMOKE_STATE_EXTRACTOR_MODEL = "mock:state-extractor"
DEFAULT_STALE_SOURCE = "data/T1_T2_400_FULL.json"
DEFAULT_TRANSFER_SOURCE = "data/longmemeval_s_cleaned.json"
STUDY_SETTINGS_SCHEMA_VERSION = "adamem.study_settings.v1"


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
    stale_dataset: str | Path | None = None,
    transfer_dataset: str | Path | None = None,
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
    if stale_dataset is None:
        stale_dataset = output / "data" / "stale.adamem.jsonl"
    if transfer_dataset is None:
        transfer_dataset = output / "data" / "longmemeval_s.adamem.jsonl"
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
        "artifact_policy": {
            "generated_datasets_default": "OUTPUT_DIR/data",
            "reason": (
                "Full benchmark conversions can be large. Default generated datasets stay "
                "inside the study output directory instead of tracked benchmark fixtures."
            ),
        },
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
    plan["plan_fingerprint"] = plan_fingerprint(plan)
    return plan


def build_smoke_study_plan(
    *,
    output_dir: str | Path,
    stale_dataset: str | Path | None = "benchmarks/stale_mini.jsonl",
    transfer_dataset: str | Path | None = "benchmarks/dynamic_state_transfer.jsonl",
    stale_types: Iterable[str] = ("T1", "T2"),
    limit_per_stale_type: int = 1,
    transfer_max_cases: int = 2,
    answer_models: Iterable[str] = SMOKE_ANSWER_MODELS,
    judge_models: Iterable[str] = SMOKE_JUDGE_MODELS,
    state_extractor_model: str = SMOKE_STATE_EXTRACTOR_MODEL,
    top_k: int = 4,
    max_context_chars: int = 2000,
) -> dict[str, Any]:
    plan = build_paper_study_plan(
        output_dir=output_dir,
        stale_dataset=stale_dataset,
        transfer_dataset=transfer_dataset,
        stale_source=None,
        transfer_source=None,
        ama_output_source=None,
        stale_types=stale_types,
        limit_per_stale_type=limit_per_stale_type,
        transfer_max_cases=transfer_max_cases,
        ama_limit=0,
        answer_models=answer_models,
        judge_models=judge_models,
        state_extractor_model=state_extractor_model,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )
    plan["profile"] = "smoke"
    plan["objective"] = "api_free_smoke_study_plan"
    plan["claim_boundary"] = (
        "This smoke plan validates local plumbing only. It uses mini/local fixtures "
        "and mock LLM providers, so it cannot support paper benchmark, answer-accuracy, "
        "or SOTA claims."
    )
    plan["artifact_policy"] = {
        "generated_datasets_default": "tracked_smoke_fixtures",
        "reason": (
            "Smoke runs use existing small fixtures and write only run outputs under "
            "the requested study output directory."
        ),
    }
    plan["post_run_gates"] = [
        "Smoke output must be treated as plumbing evidence only.",
        "Paper claims still require full STALE data, non-mock answer/judge models, transfer benchmarks, and reporting audits.",
    ]
    plan["plan_fingerprint"] = plan_fingerprint(plan)
    return plan


def build_api_pilot_settings_template(
    *,
    output_dir: str | Path = "results/api_pilot_study",
) -> dict[str, Any]:
    return {
        "schema_version": STUDY_SETTINGS_SCHEMA_VERSION,
        "profile": "paper",
        "output_dir": str(output_dir),
        "claim_boundary": (
            "API pilot settings only. Edit provider:model labels and set provider "
            "keys in the shell environment; do not store credentials in this JSON."
        ),
        "include_data_prep": True,
        "stale_source": DEFAULT_STALE_SOURCE,
        "transfer_source": DEFAULT_TRANSFER_SOURCE,
        "stale_dataset": None,
        "transfer_dataset": None,
        "include_ama": False,
        "ama_output_source": None,
        "stale_types": ["T1", "T2"],
        "limit_per_stale_type": 5,
        "transfer_max_cases": 20,
        "ama_limit": 0,
        "answer_models": [
            "openai:gpt-4o-mini",
            "gemini:gemini-1.5-flash",
        ],
        "judge_models": [
            "openai:gpt-4o-mini",
            "gemini:gemini-1.5-flash",
        ],
        "state_extractor_model": "openai:gpt-4o-mini",
        "top_k": 8,
        "max_context_chars": 4000,
        "required_env_vars": [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
        ],
        "run_hint": (
            "PYTHONPATH=src python -m adamem.study_plan --settings "
            "path/to/api_pilot_settings.json --check-env --json"
        ),
    }


def load_study_settings(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"study settings JSON must contain an object: {settings_path}")
    schema = settings.get("schema_version")
    if schema not in {None, STUDY_SETTINGS_SCHEMA_VERSION}:
        raise ValueError(
            f"unsupported study settings schema {schema!r}; expected {STUDY_SETTINGS_SCHEMA_VERSION}"
        )
    credential_paths = _credential_like_setting_paths(settings)
    if credential_paths:
        joined = ", ".join(credential_paths)
        raise ValueError(
            "study settings must not contain credential-like fields; "
            f"use environment variables instead: {joined}"
        )
    return settings


def write_study_settings_template(
    path: str | Path,
    *,
    output_dir: str | Path = "results/api_pilot_study",
) -> dict[str, str]:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings = build_api_pilot_settings_template(output_dir=output_dir)
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"settings": str(settings_path)}


def build_study_plan_from_settings(
    settings: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    output_overridden = output_dir is not None
    output = output_dir or settings.get("output_dir")
    if output in {None, ""}:
        raise ValueError("study settings must include output_dir, or pass --output-dir")
    profile = str(settings.get("profile") or "paper")
    stale_types = list(settings.get("stale_types") or ["T1", "T2"])
    if profile == "smoke":
        plan = build_smoke_study_plan(
            output_dir=output,
            stale_dataset=settings.get("stale_dataset") or "benchmarks/stale_mini.jsonl",
            transfer_dataset=settings.get("transfer_dataset") or "benchmarks/dynamic_state_transfer.jsonl",
            stale_types=stale_types,
            limit_per_stale_type=int(settings.get("limit_per_stale_type") or 1),
            transfer_max_cases=int(settings.get("transfer_max_cases") or 2),
            answer_models=settings.get("answer_models") or SMOKE_ANSWER_MODELS,
            judge_models=settings.get("judge_models") or SMOKE_JUDGE_MODELS,
            state_extractor_model=settings.get("state_extractor_model") or SMOKE_STATE_EXTRACTOR_MODEL,
            top_k=int(settings.get("top_k") or 4),
            max_context_chars=int(settings.get("max_context_chars") or 2000),
        )
    else:
        if profile != "paper":
            raise ValueError(f"unsupported study settings profile: {profile}")
        include_data_prep = bool(settings.get("include_data_prep", True))
        include_ama = bool(settings.get("include_ama", False))
        plan = build_paper_study_plan(
            output_dir=output,
            stale_dataset=settings.get("stale_dataset"),
            transfer_dataset=settings.get("transfer_dataset"),
            stale_source=settings.get("stale_source", DEFAULT_STALE_SOURCE) if include_data_prep else None,
            transfer_source=settings.get("transfer_source", DEFAULT_TRANSFER_SOURCE) if include_data_prep else None,
            ama_output_source=settings.get("ama_output_source") if include_ama else None,
            stale_types=stale_types,
            limit_per_stale_type=int(settings.get("limit_per_stale_type") or 50),
            transfer_max_cases=int(settings.get("transfer_max_cases") or 60),
            ama_limit=int(settings.get("ama_limit") or 0),
            answer_models=settings.get("answer_models") or DEFAULT_ANSWER_MODELS,
            judge_models=settings.get("judge_models") or DEFAULT_JUDGE_MODELS,
            state_extractor_model=settings.get("state_extractor_model") or "<state_extractor_provider>:<state_extractor_model>",
            top_k=int(settings.get("top_k") or 8),
            max_context_chars=int(settings.get("max_context_chars") or 4000),
        )
    attach_settings_provenance(
        plan,
        settings,
        settings_path=settings_path,
        output_dir_overridden=output_overridden,
    )
    plan["plan_fingerprint"] = plan_fingerprint(plan)
    return plan


def settings_fingerprint(settings: dict[str, Any]) -> str:
    encoded = json.dumps(
        settings,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def attach_settings_provenance(
    plan: dict[str, Any],
    settings: dict[str, Any],
    *,
    settings_path: str | Path | None = None,
    output_dir_overridden: bool = False,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "schema_version": settings.get("schema_version") or STUDY_SETTINGS_SCHEMA_VERSION,
        "settings_fingerprint": settings_fingerprint(settings),
        "output_dir_overridden": bool(output_dir_overridden),
    }
    if settings_path is not None:
        provenance["settings_path"] = str(settings_path)
    plan["settings_provenance"] = provenance
    return plan


def write_paper_study_plan(plan: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "paper_study_plan.json"
    md_path = output / "paper_study_plan.md"
    sh_path = output / "paper_study_commands.sh"
    validation_path = output / "paper_study_validation.json"
    validation_md_path = output / "paper_study_validation.md"
    plan["plan_fingerprint"] = plan_fingerprint(plan)
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


def load_paper_study_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict):
        raise ValueError(f"study plan JSON must contain an object: {plan_path}")
    return plan


def refresh_plan_fingerprint(plan: dict[str, Any]) -> dict[str, Any]:
    plan["plan_fingerprint"] = plan_fingerprint(plan)
    return plan


def refresh_saved_plan_fingerprint(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    plan = load_paper_study_plan(plan_path)
    refresh_plan_fingerprint(plan)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def write_loaded_plan_artifacts(
    plan: dict[str, Any],
    *,
    plan_path: str | Path,
    output_dir: str | Path,
    check_env: bool = False,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    md_path = output / "paper_study_plan.md"
    sh_path = output / "paper_study_commands.sh"
    validation_path = output / "paper_study_validation.json"
    validation_md_path = output / "paper_study_validation.md"
    validation = validate_paper_study_plan(plan, check_env=check_env)
    md_path.write_text(paper_study_plan_markdown(plan), encoding="utf-8")
    sh_path.write_text(paper_study_plan_shell(plan), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_md_path.write_text(paper_study_validation_markdown(validation), encoding="utf-8")
    return {
        "json": str(plan_path),
        "markdown": str(md_path),
        "shell": str(sh_path),
        "validation_json": str(validation_path),
        "validation_markdown": str(validation_md_path),
    }


def plan_fingerprint(plan: dict[str, Any]) -> str:
    payload = _fingerprint_payload(plan)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_study_plan(
    plan: dict[str, Any],
    *,
    stages: Iterable[str] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    require_ready: bool = True,
    check_env: bool = False,
    root: str | Path | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    root_path = Path(root) if root is not None else Path.cwd()
    validation = validate_paper_study_plan(plan, root=root_path, check_env=check_env)
    if require_ready and not validation["execution_ready"]:
        missing = ", ".join(validation["missing_requirements"])
        raise ValueError(f"study plan is not execution-ready: {missing}")
    allowed_stages = set(str(stage) for stage in stages or [])
    selected = [
        command for command in plan.get("commands") or []
        if not allowed_stages or str(command.get("stage")) in allowed_stages
    ]
    output_dir = Path(str(plan.get("output_dir") or "."))
    log = Path(log_path) if log_path is not None else output_dir / "paper_study_run.records.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    prior_log_record_count = _count_jsonl_records(log) if log.exists() else 0
    current_fingerprint = plan_fingerprint(plan)
    run_metadata = {
        "plan_fingerprint": current_fingerprint,
        "recorded_plan_fingerprint": plan.get("plan_fingerprint"),
        "settings_provenance": dict(plan.get("settings_provenance") or {}),
    }
    completed_keys = (
        _completed_resume_keys(log, plan_fingerprint=current_fingerprint)
        if resume and log.exists()
        else set()
    )
    records: list[dict[str, Any]] = []
    status = "dry_run" if dry_run else "complete"
    log_mode = "a" if resume else "w"
    with log.open(log_mode, encoding="utf-8") as handle:
        for index, command in enumerate(selected, start=1):
            if not dry_run and _command_resume_key(command) in completed_keys:
                record = _skipped_completed_record(
                    command,
                    index=index,
                    run_metadata=run_metadata,
                )
            else:
                record = _run_command_record(
                    command,
                    index=index,
                    dry_run=dry_run,
                    root=root_path,
                    run_metadata=run_metadata,
                )
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if record["status"] == "failed":
                status = "failed"
                break
    return {
        "schema_version": "adamem.paper_study_run.v1",
        **run_metadata,
        "status": status,
        "dry_run": dry_run,
        "resume": resume,
        "log_path": str(log),
        "selected_stage_filter": sorted(allowed_stages),
        "selected_command_count": len(selected),
        "prior_log_record_count": prior_log_record_count if resume else 0,
        "appended_record_count": len(records),
        "final_log_record_count": (
            prior_log_record_count + len(records)
            if resume
            else len(records)
        ),
        "completed_command_count": sum(1 for record in records if record["status"] == "completed"),
        "skipped_completed_count": sum(1 for record in records if record["status"] == "skipped_completed"),
        "failed_command_count": sum(1 for record in records if record["status"] == "failed"),
        "missing_output_count": sum(
            len(record.get("missing_outputs") or [])
            for record in records
        ),
        "skipped_by_failure_count": max(0, len(selected) - len(records)),
        "validation": validation,
        "records": records,
    }


def study_run_summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# AdaMem Study Run Summary", ""]
    lines.append(f"Status: `{summary.get('status')}`")
    lines.append(f"Dry run: `{bool(summary.get('dry_run'))}`")
    lines.append(f"Resume: `{bool(summary.get('resume'))}`")
    lines.append(f"Selected commands: `{int(summary.get('selected_command_count') or 0)}`")
    lines.append(f"Prior log records: `{int(summary.get('prior_log_record_count') or 0)}`")
    lines.append(f"Appended records: `{int(summary.get('appended_record_count') or 0)}`")
    lines.append(f"Final log records: `{int(summary.get('final_log_record_count') or 0)}`")
    lines.append(f"Completed commands: `{int(summary.get('completed_command_count') or 0)}`")
    lines.append(f"Skipped completed commands: `{int(summary.get('skipped_completed_count') or 0)}`")
    lines.append(f"Failed commands: `{int(summary.get('failed_command_count') or 0)}`")
    lines.append(f"Missing outputs: `{int(summary.get('missing_output_count') or 0)}`")
    lines.append(f"Log: `{summary.get('log_path')}`")
    provenance = summary.get("settings_provenance") or {}
    if provenance:
        lines.append("")
        lines.append("## Settings Provenance")
        lines.append(f"- Settings fingerprint: `{provenance.get('settings_fingerprint')}`")
        if provenance.get("settings_path"):
            lines.append(f"- Settings path: `{provenance.get('settings_path')}`")
        lines.append(f"- Output dir overridden: `{bool(provenance.get('output_dir_overridden'))}`")
    lines.append("")
    lines.append("| # | status | stage | name | seconds | missing outputs |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: |")
    for record in summary.get("records") or []:
        lines.append(
            f"| {int(record.get('index') or 0)} | "
            f"`{record.get('status')}` | "
            f"`{record.get('stage')}` | "
            f"`{record.get('name')}` | "
            f"{float(record.get('elapsed_seconds') or 0.0):.3f} | "
            f"{len(record.get('missing_outputs') or [])} |"
        )
    return "\n".join(lines) + "\n"


def validate_paper_study_plan(
    plan: dict[str, Any],
    *,
    root: str | Path | None = None,
    check_env: bool = False,
) -> dict[str, Any]:
    root_path = Path(root) if root is not None else Path.cwd()
    current_fingerprint = plan_fingerprint(plan)
    recorded_fingerprint = plan.get("plan_fingerprint")
    fingerprint_matches_recorded = (
        recorded_fingerprint is None or recorded_fingerprint == current_fingerprint
    )
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
    if not fingerprint_matches_recorded:
        missing_requirements.append("plan_fingerprint_matches_recorded")

    return {
        "schema_version": "adamem.paper_study_validation.v1",
        "plan_fingerprint": current_fingerprint,
        "recorded_plan_fingerprint": recorded_fingerprint,
        "plan_fingerprint_recorded": bool(recorded_fingerprint),
        "plan_fingerprint_matches_recorded": fingerprint_matches_recorded,
        "settings_provenance": dict(plan.get("settings_provenance") or {}),
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
    lines.append(f"Plan fingerprint: `{validation.get('plan_fingerprint') or '<missing>'}`")
    if validation.get("recorded_plan_fingerprint"):
        lines.append(f"Recorded fingerprint: `{validation.get('recorded_plan_fingerprint')}`")
        lines.append(
            "Fingerprint matches recorded: "
            f"`{bool(validation.get('plan_fingerprint_matches_recorded'))}`"
        )
    provenance = validation.get("settings_provenance") or {}
    if provenance:
        lines.append("")
        lines.append("## Settings Provenance")
        lines.append(f"- Settings fingerprint: `{provenance.get('settings_fingerprint')}`")
        if provenance.get("settings_path"):
            lines.append(f"- Settings path: `{provenance.get('settings_path')}`")
        lines.append(f"- Output dir overridden: `{bool(provenance.get('output_dir_overridden'))}`")
    lines.append("")
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
    artifact_policy = plan.get("artifact_policy") or {}
    if artifact_policy:
        lines.append("")
        lines.append("## Artifact Policy")
        lines.append(f"- Generated datasets default: `{artifact_policy.get('generated_datasets_default')}`")
        lines.append(f"- Reason: {artifact_policy.get('reason')}")
    provenance = plan.get("settings_provenance") or {}
    if provenance:
        lines.append("")
        lines.append("## Settings Provenance")
        lines.append(f"- Settings fingerprint: `{provenance.get('settings_fingerprint')}`")
        if provenance.get("settings_path"):
            lines.append(f"- Settings path: `{provenance.get('settings_path')}`")
        lines.append(f"- Output dir overridden: `{bool(provenance.get('output_dir_overridden'))}`")
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


def _fingerprint_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _fingerprint_payload(item)
            for key, item in value.items()
            if key != "plan_fingerprint"
        }
    if isinstance(value, list):
        return [_fingerprint_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_fingerprint_payload(item) for item in value]
    return value


def _completed_resume_keys(
    log: Path,
    *,
    plan_fingerprint: str,
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not log.exists():
        return keys
    with log.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("plan_fingerprint") != plan_fingerprint:
                continue
            if record.get("status") != "completed":
                continue
            if record.get("missing_outputs"):
                continue
            key = _record_resume_key(record)
            if key is not None:
                keys.add(key)
    return keys


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _command_resume_key(command: dict[str, Any]) -> tuple[str, str, str]:
    argv = [str(item) for item in command.get("command") or []]
    return (
        str(command.get("name") or ""),
        str(command.get("stage") or ""),
        shlex.join(argv),
    )


def _record_resume_key(record: dict[str, Any]) -> tuple[str, str, str] | None:
    shell = record.get("shell")
    if not shell:
        argv = record.get("command")
        if not isinstance(argv, list):
            return None
        shell = shlex.join(str(item) for item in argv)
    return (
        str(record.get("name") or ""),
        str(record.get("stage") or ""),
        str(shell),
    )


def _credential_like_setting_paths(value: Any, *, prefix: str = "") -> list[str]:
    flagged = {"api_key", "apikey", "token", "secret", "password", "credential", "credentials"}
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            normalized = key_text.lower().replace("-", "_")
            if normalized in flagged or normalized.endswith("_api_key"):
                paths.append(path)
            paths.extend(_credential_like_setting_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_credential_like_setting_paths(item, prefix=path))
    return paths


def _skipped_completed_record(
    command: dict[str, Any],
    *,
    index: int,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    argv = [str(item) for item in command.get("command") or []]
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "index": index,
        "name": command.get("name"),
        "stage": command.get("stage"),
        **run_metadata,
        "purpose": command.get("purpose"),
        "claim_boundary": command.get("claim_boundary"),
        "command": argv,
        "shell": shlex.join(argv),
        "declared_outputs": dict(command.get("outputs") or {}),
        "started_at": now,
        "finished_at": now,
        "status": "skipped_completed",
        "returncode": None,
        "elapsed_seconds": 0.0,
        "output_checks": {},
        "missing_outputs": [],
        "resume_reason": "matching_prior_completed_record",
    }


def _run_command_record(
    command: dict[str, Any],
    *,
    index: int,
    dry_run: bool,
    root: Path,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    argv = [str(item) for item in command.get("command") or []]
    started = dt.datetime.now(dt.timezone.utc)
    record: dict[str, Any] = {
        "index": index,
        "name": command.get("name"),
        "stage": command.get("stage"),
        **run_metadata,
        "purpose": command.get("purpose"),
        "claim_boundary": command.get("claim_boundary"),
        "command": argv,
        "shell": shlex.join(argv),
        "declared_outputs": dict(command.get("outputs") or {}),
        "started_at": started.isoformat(),
    }
    if dry_run:
        finished = dt.datetime.now(dt.timezone.utc)
        output_checks = _output_checks(command, root=root)
        record.update({
            "status": "dry_run",
            "returncode": None,
            "elapsed_seconds": (finished - started).total_seconds(),
            "finished_at": finished.isoformat(),
            "output_checks": output_checks,
            "missing_outputs": _missing_outputs(output_checks),
        })
        return record
    env = _subprocess_env(root)
    result = subprocess.run(
        argv,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    finished = dt.datetime.now(dt.timezone.utc)
    output_checks = _output_checks(command, root=root)
    record.update({
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_seconds": (finished - started).total_seconds(),
        "finished_at": finished.isoformat(),
        "stdout_tail": _tail_text(result.stdout),
        "stderr_tail": _tail_text(result.stderr),
        "output_checks": output_checks,
        "missing_outputs": _missing_outputs(output_checks),
    })
    return record


def _output_checks(command: dict[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for name, value in (command.get("outputs") or {}).items():
        if value in {None, ""}:
            checks[str(name)] = {
                "path": value,
                "exists": None,
                "kind": "none",
            }
            continue
        path = Path(str(value))
        resolved = path if path.is_absolute() else root / path
        exists = resolved.exists()
        if resolved.is_dir():
            kind = "directory"
        elif resolved.is_file():
            kind = "file"
        else:
            kind = "missing"
        checks[str(name)] = {
            "path": str(value),
            "exists": exists,
            "kind": kind,
        }
    return checks


def _missing_outputs(output_checks: dict[str, dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for name, check in output_checks.items():
        if check.get("path") and not bool(check.get("exists")):
            missing.append(name)
    return missing


def _subprocess_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = str(root / "src")
    existing = env.get("PYTHONPATH")
    if existing:
        paths = existing.split(os.pathsep)
        if src not in paths:
            env["PYTHONPATH"] = os.pathsep.join([src, existing])
    else:
        env["PYTHONPATH"] = src
    return env


def _tail_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


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
        description="Generate, validate, or run AdaMem paper-track study plans."
    )
    parser.add_argument("--plan", type=Path, help="Load an existing paper_study_plan.json instead of generating a new one.")
    parser.add_argument("--settings", type=Path, help="Generate a study plan from an editable API-pilot settings JSON.")
    parser.add_argument(
        "--write-settings-template",
        type=Path,
        help="Write an editable API-pilot settings JSON template and exit.",
    )
    parser.add_argument(
        "--refresh-fingerprint",
        action="store_true",
        help="With --plan, rewrite the saved JSON's recorded fingerprint after intentional manual edits.",
    )
    parser.add_argument("--output-dir", type=Path, help="Artifact directory. Required when generating a new plan.")
    parser.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    parser.add_argument("--stale-dataset", help="Defaults to OUTPUT_DIR/data/stale.adamem.jsonl")
    parser.add_argument("--transfer-dataset", help="Defaults to OUTPUT_DIR/data/longmemeval_s.adamem.jsonl")
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
        help="provider:model for LLM extractor ablation.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Also check whether required provider credential environment variables are set.",
    )
    parser.add_argument("--run", action="store_true", help="Execute the generated plan after writing artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="With --run, log commands without executing them.")
    parser.add_argument(
        "--resume-run",
        action="store_true",
        help="With --run, append to the run log and skip matching prior completed commands with all outputs present.",
    )
    parser.add_argument("--stage", action="append", dest="run_stages", help="With --run, execute only this stage. Repeatable.")
    parser.add_argument("--allow-not-ready", action="store_true", help="With --run, allow execution even if validation has gaps.")
    parser.add_argument("--run-log", type=Path, help="With --run, write JSONL execution records to this path.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.plan and args.settings:
            parser.error("--settings cannot be used with --plan")
        if args.write_settings_template:
            output_dir = args.output_dir or "results/api_pilot_study"
            artifacts = write_study_settings_template(
                args.write_settings_template,
                output_dir=output_dir,
            )
            result = {"artifacts": artifacts, "settings": load_study_settings(args.write_settings_template)}
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"wrote API pilot settings template to {args.write_settings_template}")
            return
        if args.plan:
            if args.refresh_fingerprint:
                plan = refresh_saved_plan_fingerprint(args.plan)
            else:
                plan = load_paper_study_plan(args.plan)
            output_dir = args.output_dir or Path(str(plan.get("output_dir") or args.plan.parent))
            artifacts = write_loaded_plan_artifacts(
                plan,
                plan_path=args.plan,
                output_dir=output_dir,
                check_env=args.check_env,
            )
        elif args.settings:
            if args.refresh_fingerprint:
                parser.error("--refresh-fingerprint requires --plan")
            settings = load_study_settings(args.settings)
            output_dir = args.output_dir or settings.get("output_dir")
            plan = build_study_plan_from_settings(
                settings,
                output_dir=args.output_dir,
                settings_path=args.settings,
            )
            artifacts = write_paper_study_plan(plan, output_dir)
        else:
            if args.refresh_fingerprint:
                parser.error("--refresh-fingerprint requires --plan")
            if args.output_dir is None:
                parser.error("--output-dir is required when generating a new plan")
            output_dir = args.output_dir
            if args.profile == "smoke":
                plan = build_smoke_study_plan(
                    output_dir=output_dir,
                    stale_dataset=args.stale_dataset or "benchmarks/stale_mini.jsonl",
                    transfer_dataset=args.transfer_dataset or "benchmarks/dynamic_state_transfer.jsonl",
                    stale_types=args.stale_types,
                    limit_per_stale_type=args.limit_per_stale_type,
                    transfer_max_cases=args.transfer_max_cases,
                    answer_models=args.answer_models or SMOKE_ANSWER_MODELS,
                    judge_models=args.judge_models or SMOKE_JUDGE_MODELS,
                    state_extractor_model=args.state_extractor_model or SMOKE_STATE_EXTRACTOR_MODEL,
                    top_k=args.top_k,
                    max_context_chars=args.max_context_chars,
                )
            else:
                plan = build_paper_study_plan(
                    output_dir=output_dir,
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
                    state_extractor_model=args.state_extractor_model or "<state_extractor_provider>:<state_extractor_model>",
                    top_k=args.top_k,
                    max_context_chars=args.max_context_chars,
                )
            artifacts = write_paper_study_plan(plan, output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if args.check_env and not args.plan:
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
    run_summary = None
    if args.run:
        run_summary = run_study_plan(
            plan,
            stages=args.run_stages,
            dry_run=args.dry_run,
            resume=args.resume_run,
            require_ready=not args.allow_not_ready,
            check_env=args.check_env,
            log_path=args.run_log or Path(output_dir) / "paper_study_run.records.jsonl",
        )
        run_summary_path = Path(output_dir) / "paper_study_run.summary.json"
        run_summary_md_path = Path(output_dir) / "paper_study_run.summary.md"
        run_summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run_summary_md_path.write_text(study_run_summary_markdown(run_summary), encoding="utf-8")
        artifacts["run_summary_json"] = str(run_summary_path)
        artifacts["run_summary_markdown"] = str(run_summary_md_path)
    result = {"artifacts": artifacts, "plan": plan, "validation": validation, "run_summary": run_summary}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"wrote paper study plan to {output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
