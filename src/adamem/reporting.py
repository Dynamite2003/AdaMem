from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from adamem.baselines import baseline_registry
from adamem.claims import (
    ANSWER_RUN_TYPES,
    MIN_ANSWER_MODELS_FOR_ROBUSTNESS,
    MIN_JUDGE_MODELS_FOR_ROBUSTNESS,
    audit_experiment,
    claim_audit_markdown,
)
from adamem.compare import paired_comparison_markdown, paired_comparison_summary
from adamem.error_taxonomy import attribution_counts, attribution_counts_by_baseline
from adamem.tables import load_benchmark_records, paper_table_markdown, paper_table_summary

SOTA_BASELINE_REPRODUCTION_STATUSES = {
    "official_reproduction",
    "faithful_reimplementation",
}


def write_experiment_bundle(
    experiment_path: str | Path,
    output_dir: str | Path,
    *,
    group_fields: Iterable[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    experiment = Path(experiment_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = experiment.stem
    group_fields = tuple(group_fields or ())

    audit = audit_experiment(experiment)
    experiment_payload = _load_json(experiment)
    opportunity_evidence = _opportunity_evidence(experiment_payload)
    audit_md = output / f"{stem}.claim_audit.md"
    audit_json = output / f"{stem}.claim_audit.json"
    audit_md.write_text(claim_audit_markdown(audit), encoding="utf-8")
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "experiment": str(experiment),
        "run_type": audit["run_type"],
        "dataset": audit["dataset"],
        "split_or_case_limit": audit.get("split_or_case_limit"),
        "dataset_scope": audit["dataset_scope"],
        "baselines": audit["baselines"],
        "baseline_provenance": audit.get("baseline_provenance") or {},
        "raw_output_count": audit["raw_output_count"],
        "supported_claims": audit["supported_claims"],
        "blocked_claims": audit["blocked_claims"],
        "claim_evidence": audit.get("claim_evidence") or {},
        "diagnostic_evidence": {},
        "opportunity_evidence": opportunity_evidence,
        "warnings": audit.get("warnings") or [],
        "artifacts": {
            "claim_audit_markdown": str(audit_md),
            "claim_audit_json": str(audit_json),
        },
    }

    try:
        records = load_benchmark_records(experiment)
        manifest["diagnostic_evidence"] = _diagnostic_evidence(records)
        if manifest["diagnostic_evidence"].get("examples_by_failure_attribution"):
            case_studies_json = output / f"{stem}.failure_case_studies.json"
            case_studies_md = output / f"{stem}.failure_case_studies.md"
            case_studies = manifest["diagnostic_evidence"]["examples_by_failure_attribution"]
            case_studies_json.write_text(
                json.dumps(case_studies, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            case_studies_md.write_text(
                failure_case_studies_markdown(case_studies, title=f"{stem} Failure Case Studies"),
                encoding="utf-8",
            )
            manifest["artifacts"]["failure_case_studies_json"] = str(case_studies_json)
            manifest["artifacts"]["failure_case_studies_markdown"] = str(case_studies_md)
        table_group_fields = group_fields or None
        if table_group_fields:
            table_summary = paper_table_summary(records, group_fields=table_group_fields)
            table_text = paper_table_markdown(
                records,
                group_fields=table_group_fields,
                title=title or f"{stem} Paper Tables",
            )
        else:
            table_summary = paper_table_summary(records)
            table_text = paper_table_markdown(records, title=title or f"{stem} Paper Tables")
        table_md = output / f"{stem}.paper_tables.md"
        table_json = output / f"{stem}.paper_tables.json"
        table_md.write_text(table_text, encoding="utf-8")
        table_json.write_text(json.dumps(table_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record_kind = str(table_summary.get("kind") or "")
        manifest["record_kind"] = record_kind
        manifest["artifacts"]["paper_tables_markdown"] = str(table_md)
        manifest["artifacts"]["paper_tables_json"] = str(table_json)
        if record_kind == "stale_retrieval_diagnostics":
            manifest["paired_comparison_skipped"] = (
                "stale_retrieval_diagnostics records are aggregate diagnostics; "
                "use diagnostic tables or case-level records for paired analysis"
            )
        else:
            comparison_summary = paired_comparison_summary(
                records,
                group_fields=table_group_fields,
            )
            comparison_md = output / f"{stem}.paired_comparison.md"
            comparison_json = output / f"{stem}.paired_comparison.json"
            comparison_md.write_text(
                paired_comparison_markdown(
                    comparison_summary,
                    title=f"{stem} Paired Comparison",
                ),
                encoding="utf-8",
            )
            comparison_json.write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest["artifacts"]["paired_comparison_markdown"] = str(comparison_md)
            manifest["artifacts"]["paired_comparison_json"] = str(comparison_json)
    except Exception as exc:  # pragma: no cover - exercised by CLI workflows.
        manifest["table_error"] = f"{type(exc).__name__}: {exc}"

    method_coverage = method_coverage_summary([manifest])
    method_json = output / f"{stem}.method_coverage.json"
    method_md = output / f"{stem}.method_coverage.md"
    method_json.write_text(json.dumps(method_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    method_md.write_text(method_coverage_markdown(method_coverage), encoding="utf-8")
    manifest["method_coverage"] = method_coverage
    manifest["artifacts"]["method_coverage_json"] = str(method_json)
    manifest["artifacts"]["method_coverage_markdown"] = str(method_md)

    run_claim_rows = claim_matrix_rows([manifest])
    run_study_rows = study_model_coverage_rows([manifest])
    run_benchmark_coverage = benchmark_coverage_summary([manifest])
    run_readiness = paper_readiness_summary(
        run_claim_rows,
        run_study_rows,
        benchmark_coverage=run_benchmark_coverage,
        method_coverage=method_coverage,
    )
    next_steps_md = output / f"{stem}.paper_next_steps.md"
    readiness_json = output / f"{stem}.paper_readiness.json"
    readiness_md = output / f"{stem}.paper_readiness.md"
    next_steps_md.write_text(paper_next_steps_markdown(run_claim_rows), encoding="utf-8")
    readiness_json.write_text(json.dumps(run_readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readiness_md.write_text(paper_readiness_markdown(run_readiness), encoding="utf-8")
    manifest["paper_readiness"] = run_readiness
    manifest["artifacts"]["paper_next_steps_markdown"] = str(next_steps_md)
    manifest["artifacts"]["paper_readiness_json"] = str(readiness_json)
    manifest["artifacts"]["paper_readiness_markdown"] = str(readiness_md)

    manifest_path = output / f"{stem}.manifest.json"
    manifest["artifacts"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_experiment_bundle_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*experiment.json",
    group_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    source = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiments = sorted(path for path in source.glob(pattern) if path.is_file())
    manifests: list[dict[str, Any]] = []
    for experiment in experiments:
        bundle_dir = output / experiment.stem
        manifests.append(
            write_experiment_bundle(
                experiment,
                bundle_dir,
                group_fields=group_fields,
            )
        )
    batch_manifest = {
        "input_dir": str(source),
        "output_dir": str(output),
        "pattern": pattern,
        "experiment_count": len(experiments),
        "experiments": [manifest["experiment"] for manifest in manifests],
        "bundles": manifests,
    }
    claim_matrix = claim_matrix_rows(manifests)
    study_model_coverage = study_model_coverage_rows(manifests)
    benchmark_coverage = benchmark_coverage_summary(manifests)
    method_coverage = method_coverage_summary(manifests)
    paper_readiness = paper_readiness_summary(
        claim_matrix,
        study_model_coverage,
        benchmark_coverage=benchmark_coverage,
        method_coverage=method_coverage,
    )
    claim_matrix_json = output / "claim_matrix.json"
    claim_matrix_md = output / "claim_matrix.md"
    next_steps_md = output / "paper_next_steps.md"
    study_model_json = output / "study_model_coverage.json"
    study_model_md = output / "study_model_coverage.md"
    benchmark_json = output / "benchmark_coverage.json"
    benchmark_md = output / "benchmark_coverage.md"
    method_json = output / "method_coverage.json"
    method_md = output / "method_coverage.md"
    readiness_json = output / "paper_readiness.json"
    readiness_md = output / "paper_readiness.md"
    claim_matrix_json.write_text(json.dumps(claim_matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    claim_matrix_md.write_text(claim_matrix_markdown(claim_matrix), encoding="utf-8")
    next_steps_md.write_text(paper_next_steps_markdown(claim_matrix), encoding="utf-8")
    study_model_json.write_text(json.dumps(study_model_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    study_model_md.write_text(study_model_coverage_markdown(study_model_coverage), encoding="utf-8")
    benchmark_json.write_text(json.dumps(benchmark_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    benchmark_md.write_text(benchmark_coverage_markdown(benchmark_coverage), encoding="utf-8")
    method_json.write_text(json.dumps(method_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    method_md.write_text(method_coverage_markdown(method_coverage), encoding="utf-8")
    readiness_json.write_text(json.dumps(paper_readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readiness_md.write_text(paper_readiness_markdown(paper_readiness), encoding="utf-8")
    batch_manifest["paper_readiness"] = paper_readiness
    batch_manifest["paper_claim_ready"] = bool(paper_readiness.get("paper_claim_ready"))
    batch_manifest["paper_claim_blockers"] = list(paper_readiness.get("paper_claim_blockers") or [])
    batch_manifest["artifacts"] = {
        "claim_matrix_json": str(claim_matrix_json),
        "claim_matrix_markdown": str(claim_matrix_md),
        "paper_next_steps_markdown": str(next_steps_md),
        "study_model_coverage_json": str(study_model_json),
        "study_model_coverage_markdown": str(study_model_md),
        "benchmark_coverage_json": str(benchmark_json),
        "benchmark_coverage_markdown": str(benchmark_md),
        "method_coverage_json": str(method_json),
        "method_coverage_markdown": str(method_md),
        "paper_readiness_json": str(readiness_json),
        "paper_readiness_markdown": str(readiness_md),
    }
    manifest_path = output / "batch_manifest.json"
    batch_manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(batch_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return batch_manifest


def claim_matrix_rows(manifests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        evidence = manifest.get("claim_evidence") or {}
        state_evidence = evidence.get("prepared_state_evidence") or {}
        retrieval = evidence.get("retrieval_transfer") or {}
        baseline_coverage = evidence.get("baseline_coverage") or {}
        baseline_reproduction = evidence.get("baseline_reproduction") or {}
        model_coverage = evidence.get("model_coverage") or {}
        reproducibility = evidence.get("reproducibility") or {}
        dependency = retrieval.get("dependency_propagation") or {}
        dataset_scope = manifest.get("dataset_scope") or {}
        diagnostic = manifest.get("diagnostic_evidence") or {}
        opportunity = manifest.get("opportunity_evidence") or {}
        top_attribution, top_attribution_count = _top_count(diagnostic.get("failure_attributions") or {})
        dependency_state_records, dependency_correction_records = _dependency_propagation_counts(dependency)
        row = {
            "experiment": manifest.get("experiment"),
            "run_type": manifest.get("run_type"),
            "dataset": manifest.get("dataset"),
            "split_or_case_limit": manifest.get("split_or_case_limit"),
            "dataset_scope": dataset_scope.get("scope") or "unknown",
            "dataset_claim_limited": bool(dataset_scope.get("claim_limited")),
            "dataset_scope_reasons": list(dataset_scope.get("reasons") or []),
            "record_kind": manifest.get("record_kind"),
            "raw_output_count": int(manifest.get("raw_output_count") or 0),
            "supported_claims": list(manifest.get("supported_claims") or []),
            "blocked_claims": sorted((manifest.get("blocked_claims") or {}).keys()),
            "warning_count": len(manifest.get("warnings") or []),
            "warnings": list(manifest.get("warnings") or []),
            "state_expected_questions": int(state_evidence.get("with_expected_state_slots") or 0),
            "state_matching_questions": int(state_evidence.get("with_matching_state_evidence") or 0),
            "state_available_rate": float(state_evidence.get("state_available_rate") or 0.0),
            "state_family_evidence": dict(state_evidence.get("by_state_family") or {}),
            "paired_no_regression_count": len(retrieval.get("paired_no_regression") or []),
            "dependency_propagation_baseline_count": len(dependency),
            "dependency_state_records": dependency_state_records,
            "dependency_correction_records": dependency_correction_records,
            "dependency_parent_slots": _dependency_parent_slots(dependency),
            "stale_opportunity_queries": int(opportunity.get("queries") or 0),
            "stale_state_opportunity_queries": int(opportunity.get("state_labeled_queries") or 0),
            "stale_dependency_opportunity_queries": int(opportunity.get("dependency_labeled_queries") or 0),
            "stale_opportunity_state_slots": dict(opportunity.get("state_slots") or {}),
            "stale_opportunity_dependency_families": dict(opportunity.get("dependency_families") or {}),
            "stale_opportunity_observation_violations": int(
                opportunity.get("observation_metadata_violations") or 0
            ),
            "baseline_coverage_complete": bool(baseline_coverage.get("complete")),
            "baseline_category_count": int(baseline_coverage.get("category_count") or 0),
            "missing_baseline_groups": list(baseline_coverage.get("missing_groups") or []),
            "baseline_reproduction_complete": bool(baseline_reproduction.get("complete")),
            "official_or_faithful_baseline_count": len(
                baseline_reproduction.get("official_or_faithful_mainstream_reproductions") or []
            ),
            "baseline_reproduction_gaps": list(baseline_reproduction.get("missing_requirements") or []),
            "model_coverage_complete": bool(model_coverage.get("complete")),
            "answer_model_count": int(model_coverage.get("answer_model_count") or 0),
            "judge_model_count": int(model_coverage.get("judge_model_count") or 0),
            "missing_model_requirements": list(model_coverage.get("missing_requirements") or []),
            "reproducibility_complete": bool(reproducibility.get("complete")),
            "missing_reproducibility_items": list(reproducibility.get("missing") or []),
            "failure_attribution_count": len(diagnostic.get("failure_attributions") or {}),
            "top_failure_attribution": top_attribution,
            "top_failure_attribution_count": top_attribution_count,
            "supported_claim_count": len(manifest.get("supported_claims") or []),
            "blocked_claim_count": len(manifest.get("blocked_claims") or {}),
        }
        gate, reasons = _claim_readiness_gate(row)
        row["readiness_gate"] = gate
        row["readiness_reasons"] = reasons
        row["next_actions"] = _paper_next_actions(row)
        row["next_action"] = row["next_actions"][0]
        rows.append(row)
    return rows


def study_model_coverage_rows(manifests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    for manifest in manifests:
        evidence = manifest.get("claim_evidence") or {}
        coverage = evidence.get("model_coverage") or {}
        if not coverage:
            continue
        key = _study_model_key(manifest)
        group = groups.setdefault(
            key,
            {
                "run_type": key[0],
                "dataset": key[1],
                "split_or_case_limit": key[2],
                "baselines": list(key[3]),
                "experiments": [],
                "answer_models": set(),
                "judge_models": set(),
            },
        )
        group["experiments"].append(manifest.get("experiment"))
        group["answer_models"].update(str(item) for item in coverage.get("answer_models") or [])
        group["judge_models"].update(str(item) for item in coverage.get("judge_models") or [])

    rows: list[dict[str, Any]] = []
    for group in groups.values():
        answer_models = sorted(group["answer_models"])
        judge_models = sorted(group["judge_models"])
        missing = _study_model_missing_requirements(str(group["run_type"]), answer_models, judge_models)
        rows.append({
            "run_type": group["run_type"],
            "dataset": group["dataset"],
            "split_or_case_limit": group["split_or_case_limit"],
            "baselines": group["baselines"],
            "experiment_count": len(group["experiments"]),
            "experiments": sorted(str(item) for item in group["experiments"] if item is not None),
            "answer_models": answer_models,
            "answer_model_count": len(answer_models),
            "judge_models": judge_models,
            "judge_model_count": len(judge_models),
            "missing_requirements": missing,
            "complete": not missing,
        })
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("run_type") or ""),
            str(row.get("dataset") or ""),
            str(row.get("split_or_case_limit") or ""),
            ",".join(row.get("baselines") or []),
        ),
    )


def study_model_coverage_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AdaMem Study Model Coverage",
        "",
        "| run type | dataset | split | experiments | answer models | judge models | missing |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if not rows:
        lines.append("| <none> | <none> | - | 0 | 0 | 0 | no_model_coverage_records |")
        return "\n".join(lines) + "\n"
    for row in rows:
        missing = row.get("missing_requirements") or []
        lines.append(
            f"| {row.get('run_type') or '<missing>'} | "
            f"{row.get('dataset') or '<missing>'} | "
            f"{row.get('split_or_case_limit') or '-'} | "
            f"{int(row.get('experiment_count') or 0)} | "
            f"{int(row.get('answer_model_count') or 0)} | "
            f"{int(row.get('judge_model_count') or 0)} | "
            f"{', '.join(str(item) for item in missing) if missing else '-'} |"
        )
    return "\n".join(lines) + "\n"


def benchmark_coverage_summary(manifests: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    public_or_full = 0
    primary_stale = 0
    transfer = 0
    for manifest in manifests:
        family = _benchmark_family(manifest)
        family_counts[family] = family_counts.get(family, 0) + 1
        scope = manifest.get("dataset_scope") or {}
        claim_limited = bool(scope.get("claim_limited"))
        if not claim_limited:
            public_or_full += 1
        if family == "stale":
            primary_stale += 1
        elif family in {"longmemeval", "ama", "locomo", "state_bench"}:
            transfer += 1
        rows.append({
            "experiment": manifest.get("experiment"),
            "run_type": manifest.get("run_type"),
            "dataset": manifest.get("dataset"),
            "dataset_scope": scope.get("scope") or "unknown",
            "dataset_claim_limited": claim_limited,
            "benchmark_family": family,
        })

    missing: list[str] = []
    if primary_stale == 0:
        missing.append("primary_stale_benchmark")
    if transfer == 0:
        missing.append("transfer_benchmark")
    if public_or_full == 0:
        missing.append("public_or_full_benchmark_scope")
    return {
        "experiment_count": len(rows),
        "benchmark_families": dict(sorted(family_counts.items())),
        "primary_stale_experiment_count": primary_stale,
        "transfer_experiment_count": transfer,
        "public_or_full_experiment_count": public_or_full,
        "missing_requirements": missing,
        "complete": not missing,
        "experiments": rows,
    }


def benchmark_coverage_markdown(summary: dict[str, Any]) -> str:
    lines = ["# AdaMem Benchmark Coverage", ""]
    lines.append(f"Complete: `{bool(summary.get('complete'))}`")
    lines.append(f"Experiments: `{int(summary.get('experiment_count') or 0)}`")
    lines.append(f"Primary STALE experiments: `{int(summary.get('primary_stale_experiment_count') or 0)}`")
    lines.append(f"Transfer benchmark experiments: `{int(summary.get('transfer_experiment_count') or 0)}`")
    lines.append(f"Public/full-scope experiments: `{int(summary.get('public_or_full_experiment_count') or 0)}`")
    missing = summary.get("missing_requirements") or []
    lines.append(
        "Missing requirements: "
        + (", ".join(f"`{item}`" for item in missing) if missing else "`none`")
    )
    lines.append("")
    lines.append("## Families")
    for family, count in (summary.get("benchmark_families") or {}).items():
        lines.append(f"- `{family}`: `{count}`")
    return "\n".join(lines) + "\n"


def method_coverage_summary(manifests: Iterable[dict[str, Any]]) -> dict[str, Any]:
    manifest_list = list(manifests)
    specs = baseline_registry()
    categories: dict[str, set[str]] = {}
    unknown: set[str] = set()
    baseline_provenance: dict[str, dict[str, str]] = {}
    for manifest in manifest_list:
        artifact_provenance = _manifest_baseline_provenance(manifest)
        for baseline in manifest.get("baselines") or []:
            name = str(baseline)
            provenance = artifact_provenance.get(name)
            spec = specs.get(name)
            if provenance is None and spec is not None:
                provenance = spec.provenance_dict()
            category = str((provenance or {}).get("category") or "")
            if not category and spec is not None:
                category = spec.category
            if not category:
                unknown.add(name)
                continue
            categories.setdefault(category, set()).add(name)
            if provenance is None:
                provenance = {
                    "category": category,
                    "source_name": "unknown",
                    "source_url": "",
                    "implementation_status": "unknown",
                    "reproduction_note": "No baseline provenance was available in the experiment artifact.",
                }
            else:
                provenance = {**provenance, "category": category}
            baseline_provenance[name] = provenance

    baseline_names = sorted(
        name
        for names in categories.values()
        for name in names
    )
    category_lists = {
        category: sorted(names)
        for category, names in sorted(categories.items())
    }
    reproduction_status_counts = _count_values(
        item["implementation_status"] for item in baseline_provenance.values()
    )
    mainstream_names = sorted(category_lists.get("mainstream_approximation") or [])
    mainstream_approximations = [
        name for name in mainstream_names
        if baseline_provenance[name]["implementation_status"] == "api_free_approximation"
    ]
    official_or_faithful_mainstream = [
        name for name in mainstream_names
        if baseline_provenance[name]["implementation_status"] in SOTA_BASELINE_REPRODUCTION_STATUSES
    ]
    baseline_reproduction_gaps: list[str] = []
    if mainstream_names and not official_or_faithful_mainstream:
        baseline_reproduction_gaps.append("official_or_faithful_mainstream_reproduction")
    reproduction_plan = _baseline_reproduction_plan(
        baseline_provenance,
        mainstream_names,
    )
    required_groups = {
        "raw_retrieval_reference": _category_present(categories, {"raw_turn_retrieval"}),
        "mainstream_memory_approximation": _category_present(categories, {"mainstream_approximation"}),
        "proposed_state_aware_method": _category_present(categories, {"state_aware"}),
        "mechanism_ablation": _category_present(
            categories,
            {
                "adamem_ablation",
                "state_aware_ablation",
                "state_extractor_ablation",
                "trajectory_memory_ablation",
            },
        ),
    }
    mechanism_flags = {
        "state_readout": _baseline_present(category_lists, {"state_readout", "semantic_state_readout"}),
        "state_dependency_propagation": _baseline_present(
            category_lists,
            {"state_propagation", "semantic_state_propagation", "semantic_state_propagation_adjudication"},
        ),
        "state_source_adjudication": _baseline_present(
            category_lists,
            {"semantic_state_adjudication", "semantic_state_propagation_adjudication"},
        ),
        "premise_correction": _baseline_present(
            category_lists,
            {"semantic_state_premise_correction", "semantic_llm_state_premise_correction"},
        ),
        "llm_state_extractor": _baseline_present(
            category_lists,
            {"semantic_llm_state_adjudication", "semantic_llm_state_premise_correction"},
        ),
        "trajectory_step_readout": _baseline_present(category_lists, {"trajectory_step_readout"}),
    }
    missing_requirements = [
        group for group, present in required_groups.items() if not present
    ]
    missing_named_mechanism_ablations = [
        name for name, present in mechanism_flags.items() if not present
    ]
    if unknown:
        missing_requirements.append("known_baseline_names_only")
    return {
        "experiment_count": len(manifest_list),
        "baseline_count": len(baseline_names) + len(unknown),
        "known_baseline_count": len(baseline_names),
        "unknown_baselines": sorted(unknown),
        "categories": category_lists,
        "category_counts": {
            category: len(names)
            for category, names in category_lists.items()
        },
        "baseline_provenance": baseline_provenance,
        "reproduction_status_counts": reproduction_status_counts,
        "mainstream_approximation_names": mainstream_names,
        "mainstream_api_free_approximations": mainstream_approximations,
        "official_or_faithful_mainstream_reproductions": official_or_faithful_mainstream,
        "sota_baseline_reproduction_ready": bool(official_or_faithful_mainstream),
        "baseline_reproduction_gaps": baseline_reproduction_gaps,
        "baseline_reproduction_plan": reproduction_plan,
        "reproduction_target_count": sum(
            1 for item in reproduction_plan if item.get("reproduction_target_url")
        ),
        "required_groups": required_groups,
        "mechanism_flags": mechanism_flags,
        "missing_requirements": missing_requirements,
        "missing_named_mechanism_ablations": missing_named_mechanism_ablations,
        "complete": not missing_requirements,
    }


def method_coverage_markdown(summary: dict[str, Any]) -> str:
    lines = ["# AdaMem Method Coverage", ""]
    lines.append(f"Complete: `{bool(summary.get('complete'))}`")
    lines.append(f"Known baselines: `{int(summary.get('known_baseline_count') or 0)}`")
    unknown = summary.get("unknown_baselines") or []
    lines.append(
        "Unknown baselines: "
        + (", ".join(f"`{item}`" for item in unknown) if unknown else "`none`")
    )
    missing = summary.get("missing_requirements") or []
    lines.append(
        "Missing requirements: "
        + (", ".join(f"`{item}`" for item in missing) if missing else "`none`")
    )
    reproduction_gaps = summary.get("baseline_reproduction_gaps") or []
    lines.append(
        "SOTA baseline reproduction ready: "
        f"`{bool(summary.get('sota_baseline_reproduction_ready'))}`"
    )
    lines.append(
        "Baseline reproduction gaps: "
        + (", ".join(f"`{item}`" for item in reproduction_gaps) if reproduction_gaps else "`none`")
    )
    approximations = summary.get("mainstream_api_free_approximations") or []
    if approximations:
        lines.append(
            "API-free mainstream approximations: "
            + ", ".join(f"`{item}`" for item in approximations)
        )
    reproduction_plan = summary.get("baseline_reproduction_plan") or []
    if reproduction_plan:
        lines.append("")
        lines.append("## Baseline Reproduction Plan")
        lines.append("| baseline | status | target | next action |")
        lines.append("| --- | --- | --- | --- |")
        for item in reproduction_plan:
            target = str(item.get("reproduction_target_name") or "-")
            url = str(item.get("reproduction_target_url") or "")
            if url:
                target = f"[{target}]({url})"
            lines.append(
                f"| `{item.get('baseline')}` | `{item.get('status')}` | "
                f"{target} | {item.get('next_action') or ''} |"
            )
    lines.append("")
    lines.append("## Required Groups")
    for group, present in (summary.get("required_groups") or {}).items():
        lines.append(f"- `{group}`: `{bool(present)}`")
    lines.append("")
    lines.append("## Named Mechanisms")
    for mechanism, present in (summary.get("mechanism_flags") or {}).items():
        lines.append(f"- `{mechanism}`: `{bool(present)}`")
    missing_mechanisms = summary.get("missing_named_mechanism_ablations") or []
    if missing_mechanisms:
        lines.append("")
        lines.append("## Missing Named Mechanism Ablations")
        for item in missing_mechanisms:
            lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Categories")
    categories = summary.get("categories") or {}
    if not categories:
        lines.append("- `<none>`")
    for category, names in categories.items():
        lines.append(f"- `{category}`: {', '.join(f'`{name}`' for name in names)}")
    provenance = summary.get("baseline_provenance") or {}
    if provenance:
        lines.append("")
        lines.append("## Baseline Provenance")
        lines.append("| baseline | status | source | note |")
        lines.append("| --- | --- | --- | --- |")
        for name, info in provenance.items():
            source = str(info.get("source_name") or "-")
            url = str(info.get("source_url") or "")
            if url:
                source = f"[{source}]({url})"
            lines.append(
                f"| `{name}` | `{info.get('implementation_status') or '<missing>'}` | "
                f"{source} | {info.get('reproduction_note') or ''} |"
            )
    return "\n".join(lines) + "\n"


def _baseline_reproduction_plan(
    baseline_provenance: dict[str, dict[str, str]],
    mainstream_names: Iterable[str],
) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    for name in sorted(mainstream_names):
        info = baseline_provenance.get(name) or {}
        status = str(info.get("implementation_status") or "unknown")
        target_url = str(info.get("reproduction_target_url") or "")
        if status in SOTA_BASELINE_REPRODUCTION_STATUSES:
            plan_status = "ready"
            next_action = "Use this artifact as the official/faithful baseline evidence."
        elif target_url:
            plan_status = "needs_official_or_faithful_run"
            next_action = "Run or wrap the target implementation on the same split and record provenance."
        else:
            plan_status = "needs_reproduction_target"
            next_action = "Identify an official implementation or define a faithful reimplementation protocol."
        plan.append({
            "baseline": name,
            "source_name": str(info.get("source_name") or ""),
            "source_url": str(info.get("source_url") or ""),
            "implementation_status": status,
            "status": plan_status,
            "reproduction_target_name": str(info.get("reproduction_target_name") or ""),
            "reproduction_target_url": target_url,
            "reproduction_target_note": str(info.get("reproduction_target_note") or ""),
            "next_action": next_action,
        })
    return plan


def _manifest_baseline_provenance(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    provenance = manifest.get("baseline_provenance")
    if not isinstance(provenance, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for name, item in provenance.items():
        if not isinstance(item, dict):
            continue
        normalized[str(name)] = {
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


def paper_readiness_summary(
    claim_rows: list[dict[str, Any]],
    study_model_rows: list[dict[str, Any]],
    *,
    benchmark_coverage: dict[str, Any] | None = None,
    method_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_coverage = benchmark_coverage or {}
    method_coverage = method_coverage or {}
    gate_counts = _count_values(row.get("readiness_gate") for row in claim_rows)
    action_counts = _next_action_counts(claim_rows)
    if benchmark_coverage.get("missing_requirements"):
        _increment_action(action_counts, "add_missing_benchmark_coverage")
    if method_coverage.get("missing_requirements"):
        _increment_action(action_counts, "add_missing_method_coverage")
    if method_coverage.get("missing_named_mechanism_ablations"):
        _increment_action(action_counts, "add_named_mechanism_ablations")
    if method_coverage.get("baseline_reproduction_gaps"):
        _increment_action(action_counts, "add_official_or_faithful_baseline_reproduction")
    action_counts = _sorted_action_counts(action_counts)
    complete_studies = [row for row in study_model_rows if row.get("complete")]
    incomplete_studies = [row for row in study_model_rows if not row.get("complete")]
    status = _paper_readiness_status(claim_rows, complete_studies)
    paper_claim_blockers = _paper_claim_blockers(
        status,
        complete_studies,
        benchmark_coverage=benchmark_coverage,
        method_coverage=method_coverage,
    )
    return {
        "status": status,
        "paper_claim_ready": not paper_claim_blockers,
        "paper_claim_blockers": paper_claim_blockers,
        "experiment_count": len(claim_rows),
        "gate_counts": gate_counts,
        "action_counts": action_counts,
        "top_next_actions": _top_actions(action_counts),
        "study_model_group_count": len(study_model_rows),
        "complete_study_model_group_count": len(complete_studies),
        "incomplete_study_model_group_count": len(incomplete_studies),
        "complete_study_model_groups": [
            _compact_study_group(row) for row in complete_studies
        ],
        "incomplete_study_model_groups": [
            _compact_study_group(row) for row in incomplete_studies
        ],
        "benchmark_coverage_complete": bool(benchmark_coverage.get("complete")),
        "benchmark_missing_requirements": list(benchmark_coverage.get("missing_requirements") or []),
        "benchmark_families": dict(benchmark_coverage.get("benchmark_families") or {}),
        "method_coverage_complete": bool(method_coverage.get("complete")),
        "method_missing_requirements": list(method_coverage.get("missing_requirements") or []),
        "method_missing_named_mechanism_ablations": list(
            method_coverage.get("missing_named_mechanism_ablations") or []
        ),
        "method_categories": dict(method_coverage.get("category_counts") or {}),
        "sota_baseline_reproduction_ready": bool(
            method_coverage.get("sota_baseline_reproduction_ready")
        ),
        "baseline_reproduction_gaps": list(method_coverage.get("baseline_reproduction_gaps") or []),
        "baseline_reproduction_status_counts": dict(
            method_coverage.get("reproduction_status_counts") or {}
        ),
        "baseline_reproduction_plan": list(method_coverage.get("baseline_reproduction_plan") or []),
        "mainstream_api_free_approximations": list(
            method_coverage.get("mainstream_api_free_approximations") or []
        ),
    }


def paper_readiness_markdown(summary: dict[str, Any]) -> str:
    lines = ["# AdaMem Paper Readiness", ""]
    lines.append(f"Status: `{summary.get('status') or '<missing>'}`")
    lines.append(f"Paper claim ready: `{bool(summary.get('paper_claim_ready'))}`")
    lines.append(f"Experiments: `{int(summary.get('experiment_count') or 0)}`")
    lines.append(
        "Study model groups: "
        f"`{int(summary.get('complete_study_model_group_count') or 0)}` complete / "
        f"`{int(summary.get('study_model_group_count') or 0)}` total"
    )
    lines.append(f"Benchmark coverage complete: `{bool(summary.get('benchmark_coverage_complete'))}`")
    lines.append(f"Method coverage complete: `{bool(summary.get('method_coverage_complete'))}`")
    lines.append(
        "SOTA baseline reproduction ready: "
        f"`{bool(summary.get('sota_baseline_reproduction_ready'))}`"
    )
    lines.append("")
    lines.append("## Gates")
    for gate, count in (summary.get("gate_counts") or {}).items():
        lines.append(f"- `{gate}`: `{count}`")
    lines.append("")
    lines.append("## Top Next Actions")
    actions = summary.get("top_next_actions") or []
    if not actions:
        lines.append("- None.")
    for item in actions:
        lines.append(f"- `{item['action']}`: `{item['count']}`")
    blockers = summary.get("paper_claim_blockers") or []
    if blockers:
        lines.append("")
        lines.append("## Paper Claim Blockers")
        for item in blockers:
            lines.append(f"- `{item}`")
    incomplete = summary.get("incomplete_study_model_groups") or []
    if incomplete:
        lines.append("")
        lines.append("## Incomplete Study Model Groups")
        for item in incomplete:
            lines.append(
                "- "
                f"`{item['run_type']}` dataset `{item['dataset']}`, "
                f"split `{item['split_or_case_limit'] or '-'}`, "
                f"missing `{', '.join(item['missing_requirements'])}`"
            )
    missing_benchmarks = summary.get("benchmark_missing_requirements") or []
    if missing_benchmarks:
        lines.append("")
        lines.append("## Benchmark Coverage Gaps")
        for item in missing_benchmarks:
            lines.append(f"- `{item}`")
    method_gaps = summary.get("method_missing_requirements") or []
    if method_gaps:
        lines.append("")
        lines.append("## Method Coverage Gaps")
        for item in method_gaps:
            lines.append(f"- `{item}`")
    missing_mechanisms = summary.get("method_missing_named_mechanism_ablations") or []
    if missing_mechanisms:
        lines.append("")
        lines.append("## Missing Named Mechanism Ablations")
        for item in missing_mechanisms:
            lines.append(f"- `{item}`")
    reproduction_gaps = summary.get("baseline_reproduction_gaps") or []
    if reproduction_gaps:
        lines.append("")
        lines.append("## Baseline Reproduction Gaps")
        for item in reproduction_gaps:
            lines.append(f"- `{item}`")
    reproduction_plan = summary.get("baseline_reproduction_plan") or []
    if reproduction_plan:
        lines.append("")
        lines.append("## Baseline Reproduction Plan")
        for item in reproduction_plan:
            target = str(item.get("reproduction_target_name") or "-")
            url = str(item.get("reproduction_target_url") or "")
            if url:
                target = f"[{target}]({url})"
            lines.append(
                f"- `{item.get('baseline')}`: `{item.get('status')}` via {target}"
            )
    approximations = summary.get("mainstream_api_free_approximations") or []
    if approximations:
        lines.append("")
        lines.append("## API-Free Mainstream Approximations")
        for item in approximations:
            lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def claim_matrix_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AdaMem Claim Matrix",
        "",
        "| experiment | gate | next action | scope | run type | supported | blocked | warnings | state evidence | state rate | state families | dependency evidence | stale opportunities | baseline gaps | baseline repro | model gaps | repro gaps | no-reg pairs | top attribution |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    if not rows:
        lines.append(
            "| <none> | needs_attention | add_experiment_records | unknown | <none> | "
            "0 | 0 | 0 | 0/0 | 0.00% | - | - | - | - | - | - | - | 0 | - |"
        )
        return "\n".join(lines) + "\n"
    for row in rows:
        experiment = Path(str(row.get("experiment") or "<missing>")).name
        expected = int(row.get("state_expected_questions") or 0)
        matching = int(row.get("state_matching_questions") or 0)
        lines.append(
            f"| {experiment} | {row.get('readiness_gate') or '<missing>'} | "
            f"{row.get('next_action') or '<missing>'} | "
            f"{row.get('dataset_scope') or 'unknown'} | "
            f"{row.get('run_type') or '<missing>'} | "
            f"{row['supported_claim_count']} | {row['blocked_claim_count']} | "
            f"{row['warning_count']} | {matching}/{expected} | "
            f"{float(row.get('state_available_rate') or 0.0):.2%} | "
            f"{_format_state_family_evidence(row)} | "
            f"{_format_dependency_evidence(row)} | "
            f"{_format_stale_opportunity_evidence(row)} | "
            f"{_format_missing_baseline_groups(row)} | "
            f"{_format_baseline_reproduction(row)} | "
            f"{_format_missing_model_requirements(row)} | "
            f"{_format_missing_reproducibility_items(row)} | "
            f"{row['paired_no_regression_count']} | "
            f"{_format_top_attribution(row)} |"
        )
    return "\n".join(lines) + "\n"


def paper_next_steps_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# AdaMem Paper Next Steps", ""]
    if not rows:
        lines.append("- `add_experiment_records`: no experiment records were found.")
        return "\n".join(lines) + "\n"

    action_counts = _next_action_counts(rows)

    lines.append("## Action Summary")
    for action, count in sorted(action_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{action}`: `{count}`")
    lines.append("")
    lines.append("## Experiment Checklist")
    lines.append("| experiment | gate | actions | reasons |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        experiment = Path(str(row.get("experiment") or "<missing>")).name
        actions = ", ".join(f"`{action}`" for action in row.get("next_actions") or [])
        reasons = ", ".join(f"`{reason}`" for reason in row.get("readiness_reasons") or [])
        lines.append(
            f"| {experiment} | {row.get('readiness_gate') or '<missing>'} | "
            f"{actions or '`manual_review`'} | {reasons or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _claim_readiness_gate(row: dict[str, Any]) -> tuple[str, list[str]]:
    supported = set(str(claim) for claim in row.get("supported_claims") or [])
    blocked = set(str(claim) for claim in row.get("blocked_claims") or [])
    reasons: list[str] = []
    if int(row.get("warning_count") or 0) > 0:
        reasons.append("claim_audit_warnings_present")
    if bool(row.get("dataset_claim_limited")):
        reasons.append("dataset_scope_claim_limited")
    if int(row.get("raw_output_count") or 0) == 0:
        reasons.append("no_case_level_or_raw_records")
    if int(row.get("stale_opportunity_observation_violations") or 0) > 0:
        reasons.append("stale_opportunity_metadata_leakage")
    if "unclassified_experiment" in supported:
        reasons.append("unclassified_experiment")
    if reasons:
        return "needs_attention", reasons
    if "sota" not in blocked and (
        "answer_accuracy_candidate" in supported
        or "stale_answer_accuracy_candidate" in supported
    ):
        return "sota_candidate", ["no_sota_blocker_recorded"]
    if "answer_accuracy_candidate" in supported or "stale_answer_accuracy_candidate" in supported:
        return "answer_candidate", ["answer_accuracy_candidate_but_sota_blocked"]
    if _has_diagnostic_claim(supported):
        diagnostic_reasons = ["diagnostic_or_mechanism_claim_only"]
        if "answer_accuracy" in blocked or "stale_answer_accuracy" in blocked:
            diagnostic_reasons.append("answer_accuracy_blocked")
        if "sota" in blocked:
            diagnostic_reasons.append("sota_blocked")
        return "diagnostic_ready", diagnostic_reasons
    return "needs_attention", ["no_paper_relevant_supported_claim"]


def _has_diagnostic_claim(supported: set[str]) -> bool:
    return any(
        claim.endswith("_diagnostics")
        or claim.endswith("_readiness")
        or claim.endswith("_audit")
        or claim.endswith("_analysis")
        or claim.endswith("_resolution")
        or claim.endswith("_no_regression")
        or claim.endswith("_transfer")
        for claim in supported
    )


def _study_model_key(manifest: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        str(manifest.get("run_type") or ""),
        str(manifest.get("dataset") or ""),
        str(manifest.get("split_or_case_limit") or ""),
        tuple(str(item) for item in manifest.get("baselines") or []),
    )


def _study_model_missing_requirements(
    run_type: str,
    answer_models: list[str],
    judge_models: list[str],
) -> list[str]:
    missing: list[str] = []
    if len(answer_models) < MIN_ANSWER_MODELS_FOR_ROBUSTNESS:
        missing.append("multiple_answer_models")
    if run_type == "stale_llm_judge" and len(judge_models) < MIN_JUDGE_MODELS_FOR_ROBUSTNESS:
        missing.append("multiple_judge_models")
    if run_type in ANSWER_RUN_TYPES:
        if len(judge_models) == 0:
            missing.append("semantic_llm_judge")
        elif len(judge_models) < MIN_JUDGE_MODELS_FOR_ROBUSTNESS:
            missing.append("multiple_judge_models")
    return missing


def _benchmark_family(manifest: dict[str, Any]) -> str:
    text = " ".join(
        str(manifest.get(key) or "")
        for key in ("run_type", "dataset", "experiment", "split_or_case_limit")
    ).lower()
    if "stale" in text:
        return "stale"
    if "longmemeval" in text or "lme_v2" in text or "lme-v2" in text:
        return "longmemeval"
    if "ama" in text:
        return "ama"
    if "locomo" in text:
        return "locomo"
    if "state-bench" in text or "state_bench" in text:
        return "state_bench"
    return "other"


def _category_present(categories: dict[str, set[str]], required: set[str]) -> bool:
    return any(category in categories and bool(categories[category]) for category in required)


def _baseline_present(categories: dict[str, list[str]], names: set[str]) -> bool:
    return any(name in names for baselines in categories.values() for name in baselines)


def _paper_readiness_status(
    claim_rows: list[dict[str, Any]],
    complete_studies: list[dict[str, Any]],
) -> str:
    if not claim_rows:
        return "no_experiments"
    gates = set(str(row.get("readiness_gate") or "") for row in claim_rows)
    if "needs_attention" in gates:
        return "needs_attention"
    if "sota_candidate" in gates and complete_studies:
        return "sota_candidate_with_model_coverage"
    if "answer_candidate" in gates and complete_studies:
        return "answer_candidate_with_model_coverage"
    if "sota_candidate" in gates or "answer_candidate" in gates:
        return "answer_candidate_needs_model_coverage"
    if "diagnostic_ready" in gates:
        return "diagnostic_ready"
    return "needs_attention"


def _paper_claim_blockers(
    status: str,
    complete_studies: list[dict[str, Any]],
    *,
    benchmark_coverage: dict[str, Any],
    method_coverage: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if status in {"no_experiments", "needs_attention"}:
        blockers.append("attention_ready_experiment_records")
    if status == "diagnostic_ready":
        blockers.append("end_to_end_answer_or_stale_answer_evaluation")
    if status == "answer_candidate_needs_model_coverage":
        blockers.append("study_level_model_robustness")
    if status == "answer_candidate_with_model_coverage":
        blockers.append("sota_candidate_without_sota_gate")
    if not complete_studies:
        blockers.append("complete_answer_and_judge_model_group")
    if not bool(benchmark_coverage.get("complete")):
        blockers.append("benchmark_coverage_complete")
    if not bool(method_coverage.get("complete")):
        blockers.append("method_coverage_complete")
    if method_coverage.get("missing_named_mechanism_ablations"):
        blockers.append("named_mechanism_ablation_coverage")
    if method_coverage.get("baseline_reproduction_gaps"):
        blockers.append("official_or_faithful_baseline_reproduction")
    return list(dict.fromkeys(blockers))


def _next_action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for action in row.get("next_actions") or []:
            key = str(action)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _top_actions(counts: dict[str, int], *, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"action": action, "count": count}
        for action, count in list(counts.items())[:limit]
    ]


def _increment_action(counts: dict[str, int], action: str) -> None:
    counts[action] = counts.get(action, 0) + 1


def _sorted_action_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "<missing>")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _compact_study_group(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_type": row.get("run_type"),
        "dataset": row.get("dataset"),
        "split_or_case_limit": row.get("split_or_case_limit"),
        "baselines": list(row.get("baselines") or []),
        "experiment_count": int(row.get("experiment_count") or 0),
        "answer_model_count": int(row.get("answer_model_count") or 0),
        "judge_model_count": int(row.get("judge_model_count") or 0),
        "missing_requirements": list(row.get("missing_requirements") or []),
    }


def _paper_next_actions(row: dict[str, Any]) -> list[str]:
    supported = set(str(claim) for claim in row.get("supported_claims") or [])
    blocked = set(str(claim) for claim in row.get("blocked_claims") or [])
    gate = str(row.get("readiness_gate") or "")
    actions: list[str] = []

    if int(row.get("warning_count") or 0) > 0:
        actions.append("fix_claim_audit_warnings")
    if bool(row.get("dataset_claim_limited")):
        actions.append("rerun_on_public_or_full_benchmark")
    if int(row.get("raw_output_count") or 0) == 0:
        actions.append("export_case_level_or_raw_records")
    if int(row.get("stale_opportunity_observation_violations") or 0) > 0:
        actions.append("fix_stale_opportunity_metadata_leakage")
    if "unclassified_experiment" in supported:
        actions.append("classify_experiment_run_type")

    if int(row.get("state_expected_questions") or 0) > int(row.get("state_matching_questions") or 0):
        actions.append("audit_missing_state_evidence")
    if int(row.get("failure_attribution_count") or 0) > 0:
        actions.append("inspect_representative_failure_attributions")
    if row.get("missing_baseline_groups"):
        actions.append("add_missing_baseline_categories")
    if row.get("baseline_reproduction_gaps"):
        actions.append("add_official_or_faithful_baseline_reproduction")
    if row.get("missing_model_requirements"):
        actions.append("add_model_or_judge_robustness_runs")
    if row.get("missing_reproducibility_items"):
        actions.append("complete_reproducibility_packet")

    if _has_diagnostic_claim(supported) and (
        "answer_accuracy" in blocked or "stale_answer_accuracy" in blocked
    ):
        actions.append("run_end_to_end_answer_and_judge_eval")
    if gate == "answer_candidate":
        actions.append("add_strong_baselines_and_judge_robustness")
    elif gate == "sota_candidate":
        actions.append("prepare_sota_reproduction_packet")
    elif gate == "diagnostic_ready" and "sota" in blocked:
        actions.append("defer_sota_until_answer_eval_and_strong_baselines")

    if not actions:
        actions.append("manual_review")
    return list(dict.fromkeys(actions))


def _diagnostic_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    records_with_attributions = [
        record for record in records if record.get("failure_attributions")
    ]
    if not records_with_attributions:
        return {}
    return {
        "failure_attributions": attribution_counts(records_with_attributions),
        "failure_attributions_by_baseline": attribution_counts_by_baseline(records_with_attributions),
        "examples_by_failure_attribution": _examples_by_failure_attribution(records_with_attributions),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _opportunity_evidence(experiment: dict[str, Any]) -> dict[str, Any]:
    notes = experiment.get("notes")
    if not isinstance(notes, dict):
        return {}
    summary = notes.get("stale_opportunity_summary")
    if not isinstance(summary, dict):
        return {}
    return {
        "queries": int(summary.get("queries") or 0),
        "state_labeled_queries": int(summary.get("state_labeled_queries") or 0),
        "dependency_labeled_queries": int(summary.get("dependency_labeled_queries") or 0),
        "state_slots": {
            str(key): int(value or 0)
            for key, value in (summary.get("state_slots") or {}).items()
        },
        "dependency_families": {
            str(key): int(value or 0)
            for key, value in (summary.get("dependency_families") or {}).items()
        },
        "observation_metadata_violations": int(summary.get("observation_metadata_violations") or 0),
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
            if len(bucket) < max_examples:
                bucket.append({
                    "baseline": record.get("baseline"),
                    "case_id": record.get("case_id"),
                    "query_id": record.get("query_id"),
                    "failure_modes": list(record.get("failure_modes") or []),
                    "top_retrieved": _top_retrieved(record),
                    "top_trace": _compact_trace_item(record),
                    "trace_source_labels": _trace_source_labels(record),
                })
    return examples


def failure_case_studies_markdown(examples: dict[str, list[dict[str, Any]]], *, title: str) -> str:
    lines = [f"# {title}", ""]
    if not examples:
        lines.append("- No failure attribution examples.")
        return "\n".join(lines).rstrip() + "\n"
    for attribution, items in examples.items():
        lines.append(f"## {attribution}")
        for item in items:
            labels = item.get("trace_source_labels") or []
            label_text = ", ".join(str(label) for label in labels) if labels else "-"
            trace = item.get("top_trace") or {}
            trace_meta = trace.get("metadata") if isinstance(trace, dict) else {}
            slot = trace_meta.get("state_slot") if isinstance(trace_meta, dict) else None
            slot_text = f", state_slot={slot}" if slot else ""
            lines.append(
                f"- `{item.get('baseline')}` `{item.get('case_id')}/{item.get('query_id')}` "
                f"modes={item.get('failure_modes') or []}, sources={label_text}{slot_text}"
            )
            if item.get("top_retrieved"):
                lines.append(f"  top: {item['top_retrieved']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _top_retrieved(record: dict[str, Any]) -> str | None:
    retrieved = record.get("retrieved") or []
    if retrieved:
        first = retrieved[0]
        if isinstance(first, dict):
            return str(first.get("content") or "")[:180]
        return str(first)[:180]
    trace = record.get("trace") or []
    if trace and isinstance(trace[0], dict):
        return str(trace[0].get("content") or "")[:180]
    return None


def _compact_trace_item(record: dict[str, Any]) -> dict[str, Any] | None:
    trace = record.get("trace") or []
    if not trace or not isinstance(trace[0], dict):
        return None
    item = trace[0]
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "kind": item.get("kind"),
        "relation": item.get("relation"),
        "content": str(item.get("content") or "")[:180],
        "metadata": {
            key: metadata[key]
            for key in (
                "state_slot",
                "state_status",
                "source_observation_label",
                "stale_source_observation_label",
                "source_id",
                "source_state_id",
                "stale_state_id",
            )
            if key in metadata
        },
    }


def _trace_source_labels(record: dict[str, Any]) -> list[str]:
    labels: set[str] = set()
    for item in record.get("trace") or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key in ("source_observation_label", "stale_source_observation_label"):
            label = metadata.get(key)
            if label:
                labels.add(str(label))
    return sorted(labels)


def _top_count(counts: dict[str, Any]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    key, value = sorted(
        ((str(key), int(value or 0)) for key, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )[0]
    return key, value


def _format_top_attribution(row: dict[str, Any]) -> str:
    attribution = row.get("top_failure_attribution")
    if not attribution:
        return "-"
    return f"{attribution} ({int(row.get('top_failure_attribution_count') or 0)})"


def _dependency_propagation_counts(dependency: dict[str, Any]) -> tuple[int, int]:
    state_records = 0
    correction_records = 0
    for summary in dependency.values():
        if not isinstance(summary, dict):
            continue
        state_records += int(summary.get("dependency_unknown_current_records") or 0)
        correction_records += int(summary.get("dependency_unknown_current_correction_records") or 0)
    return state_records, correction_records


def _dependency_parent_slots(dependency: dict[str, Any]) -> list[str]:
    slots: list[str] = []
    for summary in dependency.values():
        if not isinstance(summary, dict):
            continue
        for slot in summary.get("dependency_parent_slots") or []:
            text = str(slot)
            if text and text not in slots:
                slots.append(text)
    return slots


def _format_dependency_evidence(row: dict[str, Any]) -> str:
    baselines = int(row.get("dependency_propagation_baseline_count") or 0)
    states = int(row.get("dependency_state_records") or 0)
    corrections = int(row.get("dependency_correction_records") or 0)
    if baselines == 0 and states == 0 and corrections == 0:
        return "-"
    parents = ", ".join(str(slot) for slot in row.get("dependency_parent_slots") or []) or "parents?"
    return f"{baselines} baselines; state {states}; correction {corrections}; {parents}"


def _format_state_family_evidence(row: dict[str, Any]) -> str:
    families = row.get("state_family_evidence") or {}
    if not families:
        return "-"
    parts: list[str] = []
    for family, aggregate in sorted(families.items(), key=lambda item: str(item[0])):
        if not isinstance(aggregate, dict):
            continue
        questions = int(aggregate.get("questions") or 0)
        matched = int(aggregate.get("with_matching_state_evidence") or 0)
        parts.append(f"{family}:{matched}/{questions}")
    return ", ".join(parts) if parts else "-"


def _format_stale_opportunity_evidence(row: dict[str, Any]) -> str:
    queries = int(row.get("stale_opportunity_queries") or 0)
    state_queries = int(row.get("stale_state_opportunity_queries") or 0)
    dependency_queries = int(row.get("stale_dependency_opportunity_queries") or 0)
    violations = int(row.get("stale_opportunity_observation_violations") or 0)
    if queries == 0 and state_queries == 0 and dependency_queries == 0 and violations == 0:
        return "-"
    slots = row.get("stale_opportunity_state_slots") or {}
    families = row.get("stale_opportunity_dependency_families") or {}
    slot_text = _format_count_map(slots)
    family_text = _format_count_map(families)
    return (
        f"q {queries}; state {state_queries}; dep {dependency_queries}; "
        f"slots {slot_text}; families {family_text}; obs-viol {violations}"
    )


def _format_count_map(counts: dict[str, Any]) -> str:
    if not counts:
        return "-"
    parts = [
        f"{key}:{int(value or 0)}"
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    ]
    return ", ".join(parts)


def _format_missing_baseline_groups(row: dict[str, Any]) -> str:
    missing = [str(item) for item in row.get("missing_baseline_groups") or []]
    if not missing:
        return "-"
    return ", ".join(missing)


def _format_baseline_reproduction(row: dict[str, Any]) -> str:
    missing = [str(item) for item in row.get("baseline_reproduction_gaps") or []]
    if missing:
        return ", ".join(missing)
    official = int(row.get("official_or_faithful_baseline_count") or 0)
    if official:
        return f"official/faithful {official}"
    return "-"


def _format_missing_model_requirements(row: dict[str, Any]) -> str:
    missing = [str(item) for item in row.get("missing_model_requirements") or []]
    if not missing:
        return "-"
    return ", ".join(missing)


def _format_missing_reproducibility_items(row: dict[str, Any]) -> str:
    missing = [str(item) for item in row.get("missing_reproducibility_items") or []]
    if not missing:
        return "-"
    if len(missing) <= 3:
        return ", ".join(missing)
    return ", ".join(missing[:3]) + f", +{len(missing) - 3}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build reproducible paper-report bundles for AdaMem experiments."
    )
    parser.add_argument("input", type=Path, help="Experiment JSON or directory containing *experiment.json files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*experiment.json", help="Directory mode glob pattern")
    parser.add_argument("--group-fields", nargs="+")
    parser.add_argument("--title")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.input.is_dir():
        manifest = write_experiment_bundle_batch(
            args.input,
            args.output_dir,
            pattern=args.pattern,
            group_fields=args.group_fields,
        )
    else:
        manifest = write_experiment_bundle(
            args.input,
            args.output_dir,
            group_fields=args.group_fields,
            title=args.title,
        )
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        if args.input.is_dir():
            print(f"wrote {manifest['experiment_count']} report bundles to {args.output_dir}")
            print(f"batch_manifest: {manifest['manifest']}")
        else:
            print(f"wrote report bundle to {args.output_dir}")
            for name, path in manifest["artifacts"].items():
                print(f"{name}: {path}")
            if "table_error" in manifest:
                print(f"table_error: {manifest['table_error']}")


if __name__ == "__main__":
    main(sys.argv[1:])
