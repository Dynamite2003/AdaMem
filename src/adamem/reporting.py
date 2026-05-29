from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from adamem.claims import audit_experiment, claim_audit_markdown
from adamem.compare import paired_comparison_markdown, paired_comparison_summary
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
        "dataset_scope": audit["dataset_scope"],
        "baselines": audit["baselines"],
        "raw_output_count": audit["raw_output_count"],
        "supported_claims": audit["supported_claims"],
        "blocked_claims": audit["blocked_claims"],
        "claim_evidence": audit.get("claim_evidence") or {},
        "warnings": audit.get("warnings") or [],
        "artifacts": {
            "claim_audit_markdown": str(audit_md),
            "claim_audit_json": str(audit_json),
        },
    }

    try:
        records = load_benchmark_records(experiment)
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
    claim_matrix_json = output / "claim_matrix.json"
    claim_matrix_md = output / "claim_matrix.md"
    claim_matrix_json.write_text(json.dumps(claim_matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    claim_matrix_md.write_text(claim_matrix_markdown(claim_matrix), encoding="utf-8")
    batch_manifest["artifacts"] = {
        "claim_matrix_json": str(claim_matrix_json),
        "claim_matrix_markdown": str(claim_matrix_md),
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
        dataset_scope = manifest.get("dataset_scope") or {}
        row = {
            "experiment": manifest.get("experiment"),
            "run_type": manifest.get("run_type"),
            "dataset": manifest.get("dataset"),
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
            "supported_claim_count": len(manifest.get("supported_claims") or []),
            "blocked_claim_count": len(manifest.get("blocked_claims") or {}),
        }
        gate, reasons = _claim_readiness_gate(row)
        row["readiness_gate"] = gate
        row["readiness_reasons"] = reasons
        rows.append(row)
    return rows


def claim_matrix_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AdaMem Claim Matrix",
        "",
        "| experiment | gate | scope | run type | supported | blocked | warnings | state evidence | state rate | no-reg pairs |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| <none> | needs_attention | unknown | <none> | 0 | 0 | 0 | 0/0 | 0.00% | 0 |")
        return "\n".join(lines) + "\n"
    for row in rows:
        experiment = Path(str(row.get("experiment") or "<missing>")).name
        expected = int(row.get("state_expected_questions") or 0)
        matching = int(row.get("state_matching_questions") or 0)
        lines.append(
            f"| {experiment} | {row.get('readiness_gate') or '<missing>'} | "
            f"{row.get('dataset_scope') or 'unknown'} | "
            f"{row.get('run_type') or '<missing>'} | "
            f"{row['supported_claim_count']} | {row['blocked_claim_count']} | "
            f"{row['warning_count']} | {matching}/{expected} | "
            f"{float(row.get('state_available_rate') or 0.0):.2%} | "
            f"{row['paired_no_regression_count']} |"
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
        or claim.endswith("_resolution")
        or claim.endswith("_no_regression")
        or claim.endswith("_transfer")
        for claim in supported
    )


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
