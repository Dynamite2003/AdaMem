from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

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
        "raw_output_count": audit["raw_output_count"],
        "supported_claims": audit["supported_claims"],
        "blocked_claims": audit["blocked_claims"],
        "claim_evidence": audit.get("claim_evidence") or {},
        "diagnostic_evidence": {},
        "warnings": audit.get("warnings") or [],
        "artifacts": {
            "claim_audit_markdown": str(audit_md),
            "claim_audit_json": str(audit_json),
        },
    }

    try:
        records = load_benchmark_records(experiment)
        manifest["diagnostic_evidence"] = _diagnostic_evidence(records)
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
    claim_matrix_json = output / "claim_matrix.json"
    claim_matrix_md = output / "claim_matrix.md"
    next_steps_md = output / "paper_next_steps.md"
    study_model_json = output / "study_model_coverage.json"
    study_model_md = output / "study_model_coverage.md"
    claim_matrix_json.write_text(json.dumps(claim_matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    claim_matrix_md.write_text(claim_matrix_markdown(claim_matrix), encoding="utf-8")
    next_steps_md.write_text(paper_next_steps_markdown(claim_matrix), encoding="utf-8")
    study_model_json.write_text(json.dumps(study_model_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    study_model_md.write_text(study_model_coverage_markdown(study_model_coverage), encoding="utf-8")
    batch_manifest["artifacts"] = {
        "claim_matrix_json": str(claim_matrix_json),
        "claim_matrix_markdown": str(claim_matrix_md),
        "paper_next_steps_markdown": str(next_steps_md),
        "study_model_coverage_json": str(study_model_json),
        "study_model_coverage_markdown": str(study_model_md),
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
        model_coverage = evidence.get("model_coverage") or {}
        reproducibility = evidence.get("reproducibility") or {}
        dataset_scope = manifest.get("dataset_scope") or {}
        diagnostic = manifest.get("diagnostic_evidence") or {}
        top_attribution, top_attribution_count = _top_count(diagnostic.get("failure_attributions") or {})
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
            "paired_no_regression_count": len(retrieval.get("paired_no_regression") or []),
            "baseline_coverage_complete": bool(baseline_coverage.get("complete")),
            "baseline_category_count": int(baseline_coverage.get("category_count") or 0),
            "missing_baseline_groups": list(baseline_coverage.get("missing_groups") or []),
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


def claim_matrix_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AdaMem Claim Matrix",
        "",
        "| experiment | gate | next action | scope | run type | supported | blocked | warnings | state evidence | state rate | baseline gaps | model gaps | repro gaps | no-reg pairs | top attribution |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    if not rows:
        lines.append(
            "| <none> | needs_attention | add_experiment_records | unknown | <none> | "
            "0 | 0 | 0 | 0/0 | 0.00% | - | - | - | 0 | - |"
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
            f"{_format_missing_baseline_groups(row)} | "
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

    action_counts: dict[str, int] = {}
    for row in rows:
        for action in row.get("next_actions") or []:
            key = str(action)
            action_counts[key] = action_counts.get(key, 0) + 1

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
    if "unclassified_experiment" in supported:
        actions.append("classify_experiment_run_type")

    if int(row.get("state_expected_questions") or 0) > int(row.get("state_matching_questions") or 0):
        actions.append("audit_missing_state_evidence")
    if int(row.get("failure_attribution_count") or 0) > 0:
        actions.append("inspect_representative_failure_attributions")
    if row.get("missing_baseline_groups"):
        actions.append("add_missing_baseline_categories")
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
                })
    return examples


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


def _format_missing_baseline_groups(row: dict[str, Any]) -> str:
    missing = [str(item) for item in row.get("missing_baseline_groups") or []]
    if not missing:
        return "-"
    return ", ".join(missing)


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
