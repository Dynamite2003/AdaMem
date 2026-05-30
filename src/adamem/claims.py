from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from adamem.baselines import baseline_registry
from adamem.compare import paired_comparison_summary
from adamem.error_taxonomy import attribution_counts
from adamem.tables import load_benchmark_records


BASELINE_COVERAGE_GROUPS: dict[str, set[str]] = {
    "raw_retrieval_reference": {"raw_turn_retrieval"},
    "mainstream_memory_approximation": {"mainstream_approximation"},
    "adamem_or_state_ablation": {
        "adamem_ablation",
        "state_aware",
        "state_aware_ablation",
        "state_extractor_ablation",
    },
}
MIN_ANSWER_MODELS_FOR_ROBUSTNESS = 2
MIN_JUDGE_MODELS_FOR_ROBUSTNESS = 2
ANSWER_RUN_TYPES = {"jsonl_answer_generation_benchmark", "ama_public_answer_generation_pilot"}
RETRIEVAL_RUN_TYPES = {
    "jsonl_retrieval_benchmark",
    "ama_public_answerability_pilot",
    "ama_public_evidence_pilot",
    "longmemeval_v2_prepared_answer_support_pilot",
}
STRONG_BASELINE_REPRODUCTION_STATUSES = {
    "official_reproduction",
    "faithful_reimplementation",
}


def audit_experiment(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise ValueError(f"{experiment_path} must contain a JSON experiment object")

    run_type = str(experiment.get("run_type") or "")
    notes = experiment.get("notes") or {}
    if not isinstance(notes, dict):
        notes = {}
    providers = _providers(notes)
    baselines = [str(name) for name in experiment.get("baseline_names") or []]
    baseline_provenance = _baseline_provenance(experiment, baselines)
    raw_outputs = experiment.get("raw_outputs") or []
    results = experiment.get("results")
    raw_output_count = _evidence_record_count(experiment_path, raw_outputs, results, notes)

    supported: list[str] = []
    blocked: dict[str, list[str]] = {
        "answer_accuracy": [],
        "stale_answer_accuracy": [],
        "retrieval_mechanism": [],
        "sota": [],
    }
    warnings: list[str] = []
    claim_evidence: dict[str, Any] = {}
    claim_records = _load_claim_records(experiment_path, raw_outputs, notes, warnings=warnings)
    dataset_scope = _dataset_scope(experiment)
    if dataset_scope["claim_limited"]:
        reasons = ", ".join(dataset_scope["reasons"]) or dataset_scope["scope"]
        warnings.append(f"dataset scope is claim-limited: {reasons}")

    runtime_use = notes.get("ground_truth_runtime_use")
    if runtime_use != "forbidden":
        warnings.append("ground_truth_runtime_use is not explicitly forbidden")

    if run_type in RETRIEVAL_RUN_TYPES:
        supported.append("retrieval_diagnostics")
        if run_type == "longmemeval_v2_prepared_answer_support_pilot":
            supported.append("longmemeval_v2_prepared_split_readiness")
            supported.append("retrieval_answer_string_support_diagnostics")
            if notes.get("metric_boundary") != "retrieval answer-string support, not final generated answer accuracy":
                warnings.append("LongMemEval-V2 prepared pilot metric_boundary is missing or unexpected")
            state_evidence = _load_state_evidence_summary(experiment_path, notes, warnings=warnings)
            if state_evidence:
                supported.append("prepared_state_evidence_audit")
                claim_evidence["prepared_state_evidence"] = state_evidence
            else:
                warnings.append("LongMemEval-V2 prepared pilot state_evidence_summary_path is missing or unreadable")
        else:
            supported.append("answerability_diagnostics")
        retrieval_evidence = _retrieval_claim_evidence(experiment, claim_records)
        if retrieval_evidence:
            claim_evidence["retrieval_transfer"] = retrieval_evidence
            supported.extend(retrieval_evidence["supported_claims"])
        blocked["answer_accuracy"].append("run_type is retrieval/answerability, not answer generation")
        blocked["sota"].append("no final answer model and judge model evaluation")
    elif run_type in ANSWER_RUN_TYPES:
        if _uses_mock_provider(providers):
            supported.append("harness_plumbing")
            blocked["answer_accuracy"].append("mock answer or judge provider")
        elif notes.get("scorer") == "substring":
            supported.append("exact_string_answer_smoke")
            blocked["answer_accuracy"].append("substring scorer is not a semantic judge")
        else:
            supported.append("answer_accuracy_candidate")
    elif run_type == "stale_llm_judge":
        if _uses_mock_provider(providers):
            supported.append("stale_judge_plumbing")
            blocked["stale_answer_accuracy"].append("mock answer or judge provider")
        else:
            supported.append("stale_answer_accuracy_candidate")
        if raw_output_count == 0:
            blocked["stale_answer_accuracy"].append("no raw per-query judge outputs")
    elif run_type == "stale_retrieval_diagnostics":
        supported.append("stale_retrieval_diagnostics")
        supported.append("mechanism_error_analysis")
        blocked["stale_answer_accuracy"].append("no answer generation or LLM judge")
        blocked["sota"].append("retrieval diagnostics cannot establish SOTA")
    else:
        supported.append("unclassified_experiment")
        blocked["answer_accuracy"].append(f"unrecognized run_type: {run_type or '<missing>'}")
        blocked["sota"].append("unclassified experiment cannot support SOTA")

    if len(baselines) < 2:
        blocked["sota"].append("fewer than two baselines in experiment record")
    if not experiment.get("commit"):
        warnings.append("experiment commit is missing")
    if not experiment.get("dataset"):
        warnings.append("experiment dataset is missing")
    baseline_coverage = _baseline_coverage_evidence(baselines)
    if baseline_coverage:
        claim_evidence["baseline_coverage"] = baseline_coverage
        if baseline_coverage["complete"]:
            supported.append("baseline_coverage_audit")
    model_coverage = _model_coverage_evidence(run_type, providers, claim_records)
    if model_coverage:
        claim_evidence["model_coverage"] = model_coverage
        if model_coverage["complete"]:
            supported.append("model_robustness_audit")
    baseline_reproduction = _baseline_reproduction_evidence(baselines, baseline_provenance)
    if baseline_reproduction:
        claim_evidence["baseline_reproduction"] = baseline_reproduction
        if baseline_reproduction["complete"]:
            supported.append("baseline_reproduction_audit")
    _add_sota_evidence_blockers(
        blocked,
        run_type=run_type,
        model_coverage=model_coverage,
        baseline_reproduction=baseline_reproduction,
    )
    reproducibility = _reproducibility_evidence(
        experiment,
        run_type=run_type,
        raw_output_count=raw_output_count,
    )
    if reproducibility:
        claim_evidence["reproducibility"] = reproducibility
        if reproducibility["complete"]:
            supported.append("reproducibility_audit")
    attribution_evidence = _failure_attribution_claim_evidence(claim_records)
    if attribution_evidence:
        claim_evidence["failure_attributions"] = attribution_evidence
        supported.append("failure_attribution_error_analysis")
    supported = list(dict.fromkeys(supported))

    return {
        "experiment": str(experiment_path),
        "run_name": experiment.get("run_name"),
        "run_type": run_type,
        "dataset": experiment.get("dataset"),
        "split_or_case_limit": experiment.get("split_or_case_limit"),
        "dataset_scope": dataset_scope,
        "baselines": baselines,
        "baseline_provenance": baseline_provenance,
        "providers": providers,
        "supported_claims": supported,
        "blocked_claims": {key: value for key, value in blocked.items() if value},
        "warnings": warnings,
        "raw_output_count": raw_output_count,
        "claim_evidence": claim_evidence,
        "notes": {
            "ground_truth_runtime_use": runtime_use,
            "ground_truth_evaluation_use": notes.get("ground_truth_evaluation_use"),
            "answer_model_required": notes.get("answer_model_required"),
            "judge_model_required": notes.get("judge_model_required"),
        },
    }


def claim_audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# AdaMem Claim Audit", ""]
    lines.append(f"Experiment: `{audit['experiment']}`")
    lines.append(f"Run type: `{audit['run_type'] or '<missing>'}`")
    lines.append(f"Dataset: `{audit.get('dataset') or '<missing>'}`")
    scope = audit.get("dataset_scope") or {}
    lines.append(f"Dataset scope: `{scope.get('scope') or '<missing>'}`")
    lines.append(f"Raw outputs: `{audit['raw_output_count']}`")
    lines.append("")

    lines.append("## Supported Claims")
    for claim in audit["supported_claims"]:
        lines.append(f"- `{claim}`")
    lines.append("")

    lines.append("## Blocked Claims")
    blocked = audit["blocked_claims"]
    if not blocked:
        lines.append("- None.")
    for claim, reasons in blocked.items():
        lines.append(f"- `{claim}`: {'; '.join(reasons)}")
    lines.append("")

    lines.append("## Warnings")
    warnings = audit["warnings"]
    if not warnings:
        lines.append("- None.")
    for warning in warnings:
        lines.append(f"- {warning}")
    claim_evidence = audit.get("claim_evidence") or {}
    retrieval = claim_evidence.get("retrieval_transfer")
    state_evidence = claim_evidence.get("prepared_state_evidence")
    baseline_coverage = claim_evidence.get("baseline_coverage")
    model_coverage = claim_evidence.get("model_coverage")
    reproducibility = claim_evidence.get("reproducibility")
    if baseline_coverage:
        lines.append("")
        lines.append("## Baseline Coverage")
        lines.append(f"- Baselines: `{baseline_coverage['baseline_count']}`")
        lines.append(f"- Categories: `{baseline_coverage['category_count']}`")
        lines.append(f"- Complete for paper audit: `{baseline_coverage['complete']}`")
        if baseline_coverage.get("unknown_baselines"):
            lines.append(
                "- Unknown baselines: "
                + ", ".join(f"`{name}`" for name in baseline_coverage["unknown_baselines"])
            )
        if baseline_coverage.get("missing_groups"):
            lines.append(
                "- Missing groups: "
                + ", ".join(f"`{name}`" for name in baseline_coverage["missing_groups"])
            )
        for category, names in baseline_coverage.get("categories", {}).items():
            lines.append(f"- Category `{category}`: {', '.join(f'`{name}`' for name in names)}")
    baseline_reproduction = claim_evidence.get("baseline_reproduction")
    if baseline_reproduction:
        lines.append("")
        lines.append("## Baseline Reproduction")
        lines.append(f"- Complete for SOTA audit: `{baseline_reproduction['complete']}`")
        if baseline_reproduction.get("official_or_faithful_mainstream_reproductions"):
            lines.append(
                "- Official/faithful mainstream reproductions: "
                + ", ".join(
                    f"`{name}`"
                    for name in baseline_reproduction["official_or_faithful_mainstream_reproductions"]
                )
            )
        if baseline_reproduction.get("api_free_mainstream_approximations"):
            lines.append(
                "- API-free mainstream approximations: "
                + ", ".join(
                    f"`{name}`"
                    for name in baseline_reproduction["api_free_mainstream_approximations"]
                )
            )
        if baseline_reproduction.get("missing_requirements"):
            lines.append(
                "- Missing requirements: "
                + ", ".join(f"`{name}`" for name in baseline_reproduction["missing_requirements"])
            )
    if model_coverage:
        lines.append("")
        lines.append("## Model Coverage")
        lines.append(f"- Answer models: `{model_coverage['answer_model_count']}`")
        lines.append(f"- Judge models: `{model_coverage['judge_model_count']}`")
        lines.append(f"- Complete for robustness audit: `{model_coverage['complete']}`")
        if model_coverage.get("answer_models"):
            lines.append(
                "- Answer model ids: "
                + ", ".join(f"`{name}`" for name in model_coverage["answer_models"])
            )
        if model_coverage.get("judge_models"):
            lines.append(
                "- Judge model ids: "
                + ", ".join(f"`{name}`" for name in model_coverage["judge_models"])
            )
        if model_coverage.get("missing_requirements"):
            lines.append(
                "- Missing requirements: "
                + ", ".join(f"`{name}`" for name in model_coverage["missing_requirements"])
            )
    if reproducibility:
        lines.append("")
        lines.append("## Reproducibility")
        lines.append(f"- Complete for paper audit: `{reproducibility['complete']}`")
        lines.append(f"- Present fields: `{len(reproducibility['present'])}`")
        if reproducibility.get("missing"):
            lines.append(
                "- Missing fields: "
                + ", ".join(f"`{name}`" for name in reproducibility["missing"])
            )
    if state_evidence:
        lines.append("")
        lines.append("## Prepared State Evidence")
        lines.append(f"- Questions: `{state_evidence['total_questions']}`")
        lines.append(f"- Expected-state questions: `{state_evidence['with_expected_state_slots']}`")
        lines.append(f"- With matching evidence: `{state_evidence['with_matching_state_evidence']}`")
        lines.append(f"- Without matching evidence: `{state_evidence['without_matching_state_evidence']}`")
        lines.append(f"- State-available rate: `{state_evidence['state_available_rate']:.2%}`")
        if state_evidence.get("by_state_family"):
            lines.append("- State families:")
            for family, aggregate in sorted(state_evidence["by_state_family"].items()):
                lines.append(
                    f"  - `{family}`: questions `{aggregate['questions']}`, "
                    f"with evidence `{aggregate['with_matching_state_evidence']}`, "
                    f"without evidence `{aggregate['without_matching_state_evidence']}`"
                )
    if retrieval:
        lines.append("")
        lines.append("## Claim Evidence")
        lines.append(f"- Paired metric: `{retrieval.get('paired_metric') or '<missing>'}`")
        for item in retrieval.get("paired_no_regression", []):
            lines.append(
                "- No-regression pair: "
                f"`{item['candidate']}` vs `{item['reference']}` "
                f"common `{item['common_total']}`, gained `{item['gained']}`, "
                f"lost `{item['lost']}`, net `{item['net_delta']}`"
            )
        for baseline, summary in retrieval.get("premise_correction", {}).items():
            lines.append(
                "- Premise correction: "
                f"`{baseline}` records `{summary['records']}`, "
                f"correction records `{summary['correction_records']}`, "
                f"items `{summary['correction_items']}`, "
                f"unresolved forbidden `{summary['unresolved_forbidden_records']}`"
            )
        for baseline, summary in retrieval.get("unknown_current", {}).items():
            lines.append(
                "- Unknown-current: "
                f"`{baseline}` records `{summary['records']}`, "
                f"state records `{summary['unknown_current_records']}`, "
                f"correction records `{summary['unknown_current_correction_records']}`, "
                f"resolved `{summary['corrected_forbidden_records']}`, "
                f"unresolved `{summary['unresolved_forbidden_records']}`"
            )
        for baseline, summary in retrieval.get("dependency_propagation", {}).items():
            lines.append(
                "- Dependency propagation: "
                f"`{baseline}` records `{summary['records']}`, "
                f"state records `{summary['dependency_unknown_current_records']}`, "
                f"correction records `{summary['dependency_unknown_current_correction_records']}`, "
                f"resolved `{summary['corrected_forbidden_records']}`, "
                f"unresolved `{summary['unresolved_forbidden_records']}`, "
                f"parent slots `{', '.join(summary['dependency_parent_slots']) or '<none>'}`"
            )
    attribution = claim_evidence.get("failure_attributions")
    if attribution:
        lines.append("")
        lines.append("## Failure Attribution Evidence")
        lines.append(f"- Records: `{attribution['records']}`")
        lines.append(
            "- Top attribution: "
            f"`{attribution['top_failure_attribution']}` "
            f"(`{attribution['top_failure_attribution_count']}`)"
        )
        for label, count in attribution.get("failure_attributions", {}).items():
            lines.append(f"- Attribution: `{label}` count `{count}`")
        for label, examples in attribution.get("examples_by_failure_attribution", {}).items():
            for example in examples:
                lines.append(
                    "- Example: "
                    f"`{label}` baseline `{example.get('baseline') or '?'}`, "
                    f"case `{example.get('case_id') or '?'}`, "
                    f"query `{example.get('query_id') or '?'}`"
                )
    return "\n".join(lines).rstrip() + "\n"


def _providers(notes: dict[str, Any]) -> dict[str, str | None]:
    return {
        "answer_provider": _optional_str(notes.get("answer_provider")),
        "answer_model": _optional_str(notes.get("answer_model")),
        "judge_provider": _optional_str(notes.get("judge_provider")),
        "judge_model": _optional_str(notes.get("judge_model")),
        "scorer": _optional_str(notes.get("scorer") or notes.get("answer_scorer")),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _uses_mock_provider(providers: dict[str, str | None]) -> bool:
    return any(value == "mock" for key, value in providers.items() if key.endswith("_provider"))


def _dataset_scope(experiment: dict[str, Any]) -> dict[str, Any]:
    dataset = str(experiment.get("dataset") or "")
    if not dataset:
        return {
            "scope": "unknown",
            "claim_limited": True,
            "reasons": ["dataset_missing"],
        }

    text = " ".join(
        str(experiment.get(key) or "")
        for key in ("run_name", "run_type", "dataset", "split_or_case_limit")
    ).lower()
    reasons: list[str] = []
    path = dataset.lower()
    if path.startswith("/tmp") or "/tmp/" in path:
        reasons.append("tmp_path")
    if any(marker in text for marker in ("tiny", "mini", "smoke", "debug")):
        reasons.append("mini_smoke_or_debug_name")
    if any(
        marker in text
        for marker in (
            "dynamic_state_transfer",
            "employer_dependency_transfer",
            "employer_state_transfer",
            "location_dependency_transfer",
            "unknown_current_state_transfer",
        )
    ):
        reasons.append("local_synthetic_fixture")

    if reasons:
        scope = "mini_or_smoke_fixture"
    elif "longmemeval_v2_prepared" in text or "longmemeval-v2" in text:
        scope = "public_transfer_prepared"
    elif any(marker in text for marker in ("longmemeval", "ama", "stale", "locomo", "state-bench")):
        scope = "benchmark_like"
    else:
        scope = "unspecified_benchmark"

    return {
        "scope": scope,
        "claim_limited": scope in {"unknown", "mini_or_smoke_fixture"},
        "reasons": reasons,
    }


def _load_claim_records(
    experiment_path: Path,
    raw_outputs: Any,
    notes: dict[str, Any],
    *,
    warnings: list[str] | None,
) -> list[dict[str, Any]]:
    if isinstance(raw_outputs, list) and raw_outputs:
        return [dict(record) for record in raw_outputs if isinstance(record, dict)]
    records_path = notes.get("records_path")
    if not records_path:
        return []
    try:
        return load_benchmark_records(experiment_path)
    except Exception as exc:  # pragma: no cover - defensive CLI path
        if warnings is not None:
            warnings.append(f"could not load claim records from notes.records_path: {exc}")
        return []


def _load_state_evidence_summary(
    experiment_path: Path,
    notes: dict[str, Any],
    *,
    warnings: list[str] | None,
) -> dict[str, Any]:
    summary_path = notes.get("state_evidence_summary_path")
    if not summary_path:
        return {}
    path = _resolve_records_path(experiment_path, str(summary_path))
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if warnings is not None:
            warnings.append(f"could not load state evidence summary: {exc}")
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "total_questions": int(payload.get("total_questions") or 0),
        "with_expected_state_slots": int(payload.get("with_expected_state_slots") or 0),
        "with_matching_state_evidence": int(payload.get("with_matching_state_evidence") or 0),
        "without_matching_state_evidence": int(payload.get("without_matching_state_evidence") or 0),
        "state_available_rate": float(payload.get("state_available_rate") or 0.0),
        "matching_state_evidence_candidate_total": int(
            payload.get("matching_state_evidence_candidate_total") or 0
        ),
        "questions_with_missing_trajectories": int(payload.get("questions_with_missing_trajectories") or 0),
        "missing_trajectory_total": int(payload.get("missing_trajectory_total") or 0),
        "by_state_family": _normalize_state_family_summary(payload.get("by_state_family") or {}),
    }


def _normalize_state_family_summary(raw: Any) -> dict[str, dict[str, int]]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, int]] = {}
    for family, aggregate in raw.items():
        if not isinstance(aggregate, dict):
            continue
        normalized[str(family)] = {
            "questions": int(aggregate.get("questions") or 0),
            "with_expected_state_slots": int(aggregate.get("with_expected_state_slots") or 0),
            "with_matching_state_evidence": int(aggregate.get("with_matching_state_evidence") or 0),
            "without_matching_state_evidence": int(aggregate.get("without_matching_state_evidence") or 0),
            "matching_state_evidence_candidate_total": int(
                aggregate.get("matching_state_evidence_candidate_total") or 0
            ),
        }
    return normalized


def _retrieval_claim_evidence(
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records or not all(isinstance(record, dict) and "baseline" in record for record in records):
        return {}

    evidence: dict[str, Any] = {
        "supported_claims": [],
        "paired_metric": None,
        "paired_no_regression": [],
        "premise_correction": {},
        "unknown_current": {},
        "dependency_propagation": {},
    }

    try:
        paired = paired_comparison_summary(records)
    except (KeyError, ValueError):
        paired = {}
    evidence["paired_metric"] = paired.get("metric")
    for candidate, comparison in (paired.get("comparisons") or {}).items():
        common_total = int(comparison.get("common_total") or 0)
        lost = int(comparison.get("lost") or 0)
        if common_total >= 10 and lost == 0:
            evidence["paired_no_regression"].append(
                {
                    "reference": comparison.get("reference"),
                    "candidate": candidate,
                    "common_total": common_total,
                    "gained": int(comparison.get("gained") or 0),
                    "lost": lost,
                    "net_delta": int(comparison.get("net_delta") or 0),
                }
            )
    if evidence["paired_no_regression"]:
        evidence["supported_claims"].append("paired_retrieval_no_regression")

    correction_baselines = _premise_correction_baselines(experiment, records)
    stale_dataset = _looks_like_stale_dataset(experiment)
    for baseline in correction_baselines:
        baseline_records = [record for record in records if str(record.get("baseline")) == baseline]
        if not baseline_records:
            continue
        summary = _premise_correction_summary(baseline_records)
        evidence["premise_correction"][baseline] = summary
        if summary["records"] >= 10 and summary["correction_records"] == 0 and not stale_dataset:
            evidence["supported_claims"].append("premise_correction_no_trigger_on_transfer")
        elif (
            summary["correction_records"] > 0
            and summary["corrected_forbidden_records"] > 0
            and summary["unresolved_forbidden_records"] == 0
        ):
            evidence["supported_claims"].append("premise_correction_trace_resolution")

    for baseline in _unknown_current_baselines(records):
        baseline_records = [record for record in records if str(record.get("baseline")) == baseline]
        if not baseline_records:
            continue
        summary = _unknown_current_summary(baseline_records)
        evidence["unknown_current"][baseline] = summary
        if (
            summary["unknown_current_records"] + summary["unknown_current_correction_records"] > 0
            and summary["corrected_forbidden_records"] > 0
            and summary["unresolved_forbidden_records"] == 0
        ):
            evidence["supported_claims"].append("unknown_current_trace_resolution")

    for baseline in _dependency_propagation_baselines(records):
        baseline_records = [record for record in records if str(record.get("baseline")) == baseline]
        if not baseline_records:
            continue
        summary = _dependency_propagation_summary(baseline_records)
        evidence["dependency_propagation"][baseline] = summary
        if (
            summary["dependency_unknown_current_records"]
            + summary["dependency_unknown_current_correction_records"] > 0
            and summary["corrected_forbidden_records"] > 0
            and summary["unresolved_forbidden_records"] == 0
        ):
            evidence["supported_claims"].append("dependency_propagation_trace_resolution")

    evidence["supported_claims"] = list(dict.fromkeys(evidence["supported_claims"]))
    if not evidence["supported_claims"]:
        return {}
    return evidence


def _baseline_coverage_evidence(baselines: list[str]) -> dict[str, Any]:
    registry = baseline_registry()
    categories: dict[str, list[str]] = {}
    unknown: list[str] = []
    for baseline in baselines:
        spec = registry.get(baseline)
        if spec is None:
            unknown.append(baseline)
            continue
        categories.setdefault(spec.category, []).append(baseline)
    missing_groups = [
        group
        for group, accepted_categories in BASELINE_COVERAGE_GROUPS.items()
        if not accepted_categories.intersection(categories)
    ]
    return {
        "baseline_count": len(baselines),
        "known_baseline_count": len(baselines) - len(unknown),
        "unknown_baselines": unknown,
        "categories": dict(sorted(categories.items())),
        "category_count": len(categories),
        "required_groups": sorted(BASELINE_COVERAGE_GROUPS),
        "missing_groups": missing_groups,
        "complete": not unknown and not missing_groups,
    }


def _model_coverage_evidence(
    run_type: str,
    providers: dict[str, str | None],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if run_type not in ANSWER_RUN_TYPES | {"stale_llm_judge"}:
        return {}

    answer_models = _model_ids(
        providers,
        records,
        provider_key="answer_provider",
        model_key="answer_model",
    )
    judge_models = _model_ids(
        providers,
        records,
        provider_key="judge_provider",
        model_key="judge_model",
    )
    missing: list[str] = []
    if len(answer_models) < MIN_ANSWER_MODELS_FOR_ROBUSTNESS:
        missing.append("multiple_answer_models")
    if run_type == "stale_llm_judge" and len(judge_models) < MIN_JUDGE_MODELS_FOR_ROBUSTNESS:
        missing.append("multiple_judge_models")
    if run_type in ANSWER_RUN_TYPES:
        scorer = providers.get("scorer")
        if scorer in {"llm", "llm_judge"} and len(judge_models) < MIN_JUDGE_MODELS_FOR_ROBUSTNESS:
            missing.append("multiple_judge_models")
        elif scorer not in {"llm", "llm_judge"}:
            missing.append("semantic_llm_judge")
    return {
        "answer_models": sorted(answer_models),
        "answer_model_count": len(answer_models),
        "judge_models": sorted(judge_models),
        "judge_model_count": len(judge_models),
        "required_answer_models": MIN_ANSWER_MODELS_FOR_ROBUSTNESS,
        "required_judge_models": MIN_JUDGE_MODELS_FOR_ROBUSTNESS,
        "missing_requirements": missing,
        "complete": not missing,
    }


def _baseline_reproduction_evidence(
    baselines: list[str],
    provenance: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if not baselines:
        return {}
    mainstream = [
        name for name in baselines
        if provenance.get(name, {}).get("category") == "mainstream_approximation"
    ]
    official_or_faithful = [
        name for name in mainstream
        if provenance.get(name, {}).get("implementation_status")
        in STRONG_BASELINE_REPRODUCTION_STATUSES
    ]
    api_free = [
        name for name in mainstream
        if provenance.get(name, {}).get("implementation_status") == "api_free_approximation"
    ]
    missing_provenance = [name for name in baselines if name not in provenance]
    missing_requirements: list[str] = []
    if missing_provenance:
        missing_requirements.append("baseline_provenance")
    if not mainstream:
        missing_requirements.append("mainstream_memory_baseline")
    if not official_or_faithful:
        missing_requirements.append("official_or_faithful_mainstream_reproduction")
    return {
        "baseline_count": len(baselines),
        "mainstream_baselines": mainstream,
        "api_free_mainstream_approximations": api_free,
        "official_or_faithful_mainstream_reproductions": official_or_faithful,
        "missing_provenance": missing_provenance,
        "missing_requirements": missing_requirements,
        "complete": not missing_requirements,
    }


def _add_sota_evidence_blockers(
    blocked: dict[str, list[str]],
    *,
    run_type: str,
    model_coverage: dict[str, Any],
    baseline_reproduction: dict[str, Any],
) -> None:
    if run_type not in ANSWER_RUN_TYPES | {"stale_llm_judge"}:
        return
    if not baseline_reproduction.get("complete"):
        blocked["sota"].append("no official strong-baseline reproduction evidence")
    if model_coverage and not model_coverage.get("complete"):
        blocked["sota"].append("model or judge robustness incomplete")


def _model_ids(
    providers: dict[str, str | None],
    records: list[dict[str, Any]],
    *,
    provider_key: str,
    model_key: str,
) -> set[str]:
    ids: set[str] = set()
    _add_model_id(ids, providers.get(provider_key), providers.get(model_key))
    for record in records:
        _add_model_id(ids, record.get(provider_key), record.get(model_key))
    return ids


def _add_model_id(ids: set[str], provider: Any, model: Any) -> None:
    if provider is None:
        return
    provider_text = str(provider)
    if not provider_text or provider_text == "mock":
        return
    model_text = str(model) if model is not None else "<unspecified>"
    ids.add(f"{provider_text}:{model_text}")


def _reproducibility_evidence(
    experiment: dict[str, Any],
    *,
    run_type: str,
    raw_output_count: int,
) -> dict[str, Any]:
    expected: list[str] = [
        "schema_version",
        "commit",
        "command",
        "dataset",
        "baseline_names",
        "baseline_configs",
        "baseline_provenance",
        "ground_truth_runtime_use",
        "case_level_records",
    ]
    if run_type in RETRIEVAL_RUN_TYPES | {"stale_retrieval_diagnostics"}:
        expected.append("metric_boundary_or_diagnostic_kind")
    if run_type in ANSWER_RUN_TYPES | {"stale_llm_judge"}:
        expected.extend([
            "answer_provider",
            "answer_model",
            "top_k",
            "max_context_chars",
            "answer_prompt",
        ])
    if run_type == "stale_llm_judge":
        expected.extend(["judge_provider", "judge_model", "judge_prompt"])
    if run_type in ANSWER_RUN_TYPES:
        expected.append("answer_scorer")
        notes = experiment.get("notes") if isinstance(experiment.get("notes"), dict) else {}
        scorer = _providers(notes).get("scorer")
        if scorer in {"llm", "llm_judge"}:
            expected.extend(["judge_provider", "judge_model", "judge_prompt"])

    present: list[str] = []
    for item in expected:
        if _has_reproducibility_item(experiment, item, raw_output_count=raw_output_count):
            present.append(item)
    missing = [item for item in expected if item not in present]
    return {
        "expected": expected,
        "present": present,
        "missing": missing,
        "complete": not missing,
    }


def _has_reproducibility_item(
    experiment: dict[str, Any],
    item: str,
    *,
    raw_output_count: int,
) -> bool:
    notes = experiment.get("notes") if isinstance(experiment.get("notes"), dict) else {}
    prompts = experiment.get("prompts") if isinstance(experiment.get("prompts"), dict) else {}
    if item == "schema_version":
        return bool(experiment.get("schema_version"))
    if item == "commit":
        return bool(experiment.get("commit"))
    if item == "command":
        return bool(experiment.get("command"))
    if item == "dataset":
        return bool(experiment.get("dataset"))
    if item == "baseline_names":
        return bool(experiment.get("baseline_names"))
    if item == "baseline_configs":
        return bool(experiment.get("baseline_configs"))
    if item == "baseline_provenance":
        return _has_baseline_provenance(experiment)
    if item == "ground_truth_runtime_use":
        return notes.get("ground_truth_runtime_use") == "forbidden"
    if item == "case_level_records":
        return raw_output_count > 0 or bool(notes.get("records_path"))
    if item == "metric_boundary_or_diagnostic_kind":
        return bool(notes.get("metric_boundary") or notes.get("benchmark_kind") or experiment.get("diagnostics"))
    if item == "answer_provider":
        return bool(notes.get("answer_provider"))
    if item == "answer_model":
        return bool(notes.get("answer_model"))
    if item == "judge_provider":
        return bool(notes.get("judge_provider"))
    if item == "judge_model":
        return bool(notes.get("judge_model"))
    if item == "top_k":
        return notes.get("top_k") is not None
    if item == "max_context_chars":
        return notes.get("max_context_chars") is not None
    if item == "answer_scorer":
        return bool(notes.get("scorer") or notes.get("answer_scorer"))
    if item == "answer_prompt":
        return bool(prompts.get("answer_system") and prompts.get("answer_template"))
    if item == "judge_prompt":
        return bool(prompts.get("judge_system") and prompts.get("judge_template"))
    return False


def _has_baseline_provenance(experiment: dict[str, Any]) -> bool:
    names = [str(name) for name in experiment.get("baseline_names") or []]
    return bool(_baseline_provenance(experiment, names))


def _baseline_provenance(
    experiment: dict[str, Any],
    baseline_names: list[str],
) -> dict[str, dict[str, str]]:
    provenance = experiment.get("baseline_provenance")
    if not baseline_names or not isinstance(provenance, dict):
        return {}
    required = {"category", "source_name", "implementation_status", "reproduction_note"}
    normalized: dict[str, dict[str, str]] = {}
    for name in baseline_names:
        item = provenance.get(name)
        if not isinstance(item, dict):
            return {}
        if any(not item.get(field) for field in required):
            return {}
        normalized[name] = {
            "category": str(item.get("category") or ""),
            "source_name": str(item.get("source_name") or ""),
            "source_url": str(item.get("source_url") or ""),
            "implementation_status": str(item.get("implementation_status") or ""),
            "reproduction_note": str(item.get("reproduction_note") or ""),
            "reproduction_target_name": str(item.get("reproduction_target_name") or ""),
            "reproduction_target_url": str(item.get("reproduction_target_url") or ""),
            "reproduction_target_note": str(item.get("reproduction_target_note") or ""),
        }
    return normalized


def _failure_attribution_claim_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    records_with_attributions = [
        record for record in records if record.get("failure_attributions")
    ]
    if not records_with_attributions:
        return {}
    counts = attribution_counts(records_with_attributions)
    top_label = ""
    top_count = 0
    if counts:
        top_label, top_count = next(iter(counts.items()))
    return {
        "records": len(records_with_attributions),
        "failure_attributions": counts,
        "top_failure_attribution": top_label,
        "top_failure_attribution_count": top_count,
        "examples_by_failure_attribution": _examples_by_failure_attribution(
            records_with_attributions
        ),
    }


def _examples_by_failure_attribution(
    records: list[dict[str, Any]],
    *,
    max_examples: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for attribution in record.get("failure_attributions") or []:
            bucket = examples.setdefault(str(attribution), [])
            if len(bucket) >= max_examples:
                continue
            bucket.append({
                "baseline": record.get("baseline"),
                "case_id": record.get("case_id"),
                "query_id": record.get("query_id"),
                "failure_modes": list(record.get("failure_modes") or []),
                "top_retrieved": _top_retrieved(record),
            })
    return examples


def _top_retrieved(record: dict[str, Any]) -> str:
    retrieved = record.get("retrieved") or record.get("retrieved_texts") or []
    if not isinstance(retrieved, list) or not retrieved:
        return ""
    first = retrieved[0]
    if isinstance(first, dict):
        text = first.get("text") or first.get("content") or first.get("memory") or ""
    else:
        text = str(first)
    return str(text)[:240]


def _premise_correction_baselines(
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[str]:
    configs = experiment.get("baseline_configs") or {}
    baselines = []
    for name in experiment.get("baseline_names") or []:
        baseline = str(name)
        config = configs.get(baseline) if isinstance(configs, dict) else None
        if (
            "premise_correction" in baseline
            or (isinstance(config, dict) and config.get("use_state_premise_correction") is True)
        ):
            baselines.append(baseline)
    if baselines:
        return baselines
    return list(
        dict.fromkeys(
            str(record.get("baseline"))
            for record in records
            if "premise_correction_count" in record
            and "premise_correction" in str(record.get("baseline"))
        )
    )


def _premise_correction_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    correction_records = 0
    correction_items = 0
    corrected_forbidden_records = 0
    unresolved_forbidden_records = 0
    for record in records:
        count = int(record.get("premise_correction_count") or 0)
        correction_items += count
        if count > 0:
            correction_records += 1
        if record.get("corrected_forbidden"):
            corrected_forbidden_records += 1
        if record.get("present_forbidden"):
            unresolved_forbidden_records += 1
    return {
        "records": len(records),
        "correction_records": correction_records,
        "correction_items": correction_items,
        "corrected_forbidden_records": corrected_forbidden_records,
        "unresolved_forbidden_records": unresolved_forbidden_records,
    }


def _unknown_current_baselines(records: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(record.get("baseline"))
            for record in records
            if _record_has_unknown_current(record)
        )
    )


def _unknown_current_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    unknown_current_records = 0
    unknown_current_correction_records = 0
    corrected_forbidden_records = 0
    unresolved_forbidden_records = 0
    for record in records:
        has_state = _record_has_unknown_current_state(record)
        has_correction = _record_has_unknown_current_correction(record)
        if has_state:
            unknown_current_records += 1
        if has_correction:
            unknown_current_correction_records += 1
        if has_state or has_correction:
            if record.get("corrected_forbidden"):
                corrected_forbidden_records += 1
            if record.get("present_forbidden"):
                unresolved_forbidden_records += 1
    return {
        "records": len(records),
        "unknown_current_records": unknown_current_records,
        "unknown_current_correction_records": unknown_current_correction_records,
        "corrected_forbidden_records": corrected_forbidden_records,
        "unresolved_forbidden_records": unresolved_forbidden_records,
    }


def _dependency_propagation_baselines(records: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(record.get("baseline"))
            for record in records
            if _record_has_dependency_propagation(record)
        )
    )


def _dependency_propagation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    state_records = 0
    correction_records = 0
    corrected_forbidden_records = 0
    unresolved_forbidden_records = 0
    parent_slots: set[str] = set()
    for record in records:
        has_state = _record_has_dependency_propagation_state(record)
        has_correction = _record_has_dependency_propagation_correction(record)
        if has_state:
            state_records += 1
        if has_correction:
            correction_records += 1
        if has_state or has_correction:
            if record.get("corrected_forbidden"):
                corrected_forbidden_records += 1
            if record.get("present_forbidden"):
                unresolved_forbidden_records += 1
        parent_slots.update(_dependency_parent_slots(record))
    return {
        "records": len(records),
        "dependency_unknown_current_records": state_records,
        "dependency_unknown_current_correction_records": correction_records,
        "corrected_forbidden_records": corrected_forbidden_records,
        "unresolved_forbidden_records": unresolved_forbidden_records,
        "dependency_parent_slots": sorted(parent_slots),
    }


def _record_has_unknown_current(record: dict[str, Any]) -> bool:
    return (
        _record_has_unknown_current_state(record)
        or _record_has_unknown_current_correction(record)
    )


def _record_has_unknown_current_state(record: dict[str, Any]) -> bool:
    for item in record.get("trace") or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and metadata.get("state_status") == "unknown_current":
            return True
    return False


def _record_has_unknown_current_correction(record: dict[str, Any]) -> bool:
    for item in record.get("trace") or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and metadata.get("current_value") == "unknown-current":
            return True
    return False


def _record_has_dependency_propagation(record: dict[str, Any]) -> bool:
    return (
        _record_has_dependency_propagation_state(record)
        or _record_has_dependency_propagation_correction(record)
    )


def _record_has_dependency_propagation_state(record: dict[str, Any]) -> bool:
    for item in record.get("trace") or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("state_status") == "unknown_current"
            and metadata.get("dependency_invalidated_by_slot")
        ):
            return True
    return False


def _record_has_dependency_propagation_correction(record: dict[str, Any]) -> bool:
    for item in record.get("trace") or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("current_value") == "unknown-current"
            and metadata.get("dependency_invalidated_by_slot")
        ):
            return True
    return False


def _dependency_parent_slots(record: dict[str, Any]) -> list[str]:
    slots: list[str] = []
    for item in record.get("trace") or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        slot = metadata.get("dependency_invalidated_by_slot")
        if slot:
            slots.append(str(slot))
    return slots


def _looks_like_stale_dataset(experiment: dict[str, Any]) -> bool:
    text = " ".join(
        str(experiment.get(key) or "")
        for key in ("run_name", "run_type", "dataset", "split_or_case_limit")
    ).lower()
    return "stale" in text


def _evidence_record_count(
    experiment_path: Path,
    raw_outputs: Any,
    results: Any,
    notes: dict[str, Any],
) -> int:
    if isinstance(raw_outputs, list) and raw_outputs:
        return len(raw_outputs)
    records_path = notes.get("records_path")
    if records_path:
        path = _resolve_records_path(experiment_path, str(records_path))
        if path is not None:
            return _count_jsonl_records(path)
    if isinstance(results, list):
        total = 0
        for result in results:
            if isinstance(result, dict):
                if "queries" in result and isinstance(result["queries"], list):
                    total += len(result["queries"])
                elif "n_total" in result:
                    total += int(result["n_total"] or 0)
        return total
    if isinstance(results, dict):
        total = 0
        for value in results.values():
            if isinstance(value, dict):
                total += int(value.get("total") or value.get("n_total") or 0)
        return total
    return 0


def _resolve_records_path(experiment_path: Path, records_path: str) -> Path | None:
    candidate = Path(records_path)
    if candidate.exists():
        return candidate
    relative = experiment_path.parent / candidate
    if relative.exists():
        return relative
    return None


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit what paper claims an AdaMem experiment can support."
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    audit = audit_experiment(args.experiment)
    text = (
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
        if args.json
        else claim_audit_markdown(audit)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
