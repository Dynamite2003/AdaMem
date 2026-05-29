from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from adamem.baselines import BaselineSpec, select_baselines
from adamem.bench import MemoryQACase, load_jsonl_cases
from adamem.convert import convert_stale_file
from adamem.diagnostics import (
    diagnostic_case_records,
    diagnostic_failure_report,
    diagnostic_failure_summary,
    diagnostics_report,
    run_stale_retrieval_diagnostics,
)
from adamem.experiments import experiment_record, write_experiment_record
from adamem.reporting import write_experiment_bundle
from adamem.tables import write_paper_table


DEFAULT_BASELINES = (
    "semantic_only",
    "semantic_state_adjudication",
    "semantic_state_premise_correction",
)


def run_stale_diagnostic_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    run_name: str | None = None,
    baselines: Iterable[str] = DEFAULT_BASELINES,
    top_k: int = 8,
    convert_limit: int | None = None,
    max_cases: int | None = None,
    stale_types: Iterable[str] | None = None,
    limit_per_stale_type: int | None = None,
    input_format: str = "auto",
) -> dict[str, Any]:
    """Convert raw STALE JSON and run the API-free diagnostic/report workflow."""

    source = Path(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"{source.stem}.stale_diagnostics"
    selected_specs = select_baselines(tuple(baselines))
    type_list = list(stale_types or [])

    converted_path = output / f"{run_name}.adamem.jsonl"
    resolved_format = _resolve_input_format(source, input_format)
    if resolved_format == "adamem-jsonl":
        if source.resolve() != converted_path.resolve():
            shutil.copyfile(source, converted_path)
        converted_count = len(load_jsonl_cases(converted_path))
    else:
        converted_count = convert_stale_file(
            source,
            converted_path,
            top_k=top_k,
            limit=convert_limit,
            types=type_list or None,
        )

    cases = _select_cases(
        load_jsonl_cases(converted_path),
        max_cases=max_cases,
        stale_types=type_list,
        limit_per_type=limit_per_stale_type,
    )
    diagnostic_results = run_stale_retrieval_diagnostics(
        cases,
        {name: spec.config for name, spec in selected_specs.items()},
    )
    case_records = diagnostic_case_records(diagnostic_results)
    failure_summary = diagnostic_failure_summary(case_records)

    case_records_path = output / f"{run_name}.diagnostic_cases.jsonl"
    diagnostic_report_path = output / f"{run_name}.diagnostic_report.md"
    experiment_path = output / f"{run_name}.experiment.json"
    tables_md_path = output / f"{run_name}.paper_tables.md"
    tables_json_path = output / f"{run_name}.paper_tables.json"
    bundle_dir = output / f"{run_name}.bundle"

    _write_jsonl(case_records_path, case_records)
    diagnostic_report_path.write_text(diagnostic_failure_report(case_records), encoding="utf-8")

    record = experiment_record(
        run_name=run_name,
        run_type="stale_retrieval_diagnostics",
        dataset=converted_path,
        split_or_case_limit=_split_note(max_cases, type_list, limit_per_stale_type),
        baselines=selected_specs,
        diagnostics=[asdict(result) for result in diagnostic_results],
        results={"failure_summary": failure_summary},
        raw_outputs=case_records,
        notes={
            "answer_model_required": False,
            "judge_model_required": False,
            "ground_truth_runtime_use": "forbidden",
            "raw_stale_input": str(source),
            "input_format": resolved_format,
            "converted_cases": converted_count,
            "diagnostic_cases": len(cases),
            "diagnostic_case_records": len(case_records),
            "top_k": top_k,
        },
        command=list(sys.argv),
    )
    write_experiment_record(experiment_path, record)

    write_paper_table(
        experiment_path,
        tables_md_path,
        title=f"{run_name} STALE Retrieval Diagnostics",
    )
    write_paper_table(
        experiment_path,
        tables_json_path,
        output_format="json",
    )
    bundle_manifest = write_experiment_bundle(
        experiment_path,
        bundle_dir,
        title=f"{run_name} STALE Retrieval Diagnostics",
    )

    manifest = {
        "run_name": run_name,
        "raw_input": str(source),
        "output_dir": str(output),
        "converted_cases": converted_count,
        "diagnostic_cases": len(cases),
        "baselines": list(selected_specs),
        "top_k": top_k,
        "input_format": resolved_format,
        "stale_types": type_list,
        "limit_per_stale_type": limit_per_stale_type,
        "artifacts": {
            "converted_dataset": str(converted_path),
            "experiment": str(experiment_path),
            "diagnostic_cases": str(case_records_path),
            "diagnostic_report": str(diagnostic_report_path),
            "paper_tables_markdown": str(tables_md_path),
            "paper_tables_json": str(tables_json_path),
            "report_bundle": str(bundle_dir),
            "report_bundle_manifest": bundle_manifest["artifacts"]["manifest"],
        },
    }
    manifest_path = output / f"{run_name}.manifest.json"
    manifest["artifacts"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _resolve_input_format(source: Path, input_format: str) -> str:
    allowed = {"auto", "raw", "adamem-jsonl"}
    if input_format not in allowed:
        raise ValueError(f"input_format must be one of {sorted(allowed)}")
    if input_format != "auto":
        return input_format
    if source.suffix.lower() == ".jsonl":
        return "adamem-jsonl"
    return "raw"


def _select_cases(
    cases: list[MemoryQACase],
    *,
    max_cases: int | None,
    stale_types: list[str],
    limit_per_type: int | None,
) -> list[MemoryQACase]:
    allowed = set(stale_types or [])
    selected: list[MemoryQACase] = []
    type_counts: dict[str, int] = {}
    for case in cases:
        case_type = _case_stale_type(case)
        if allowed and case_type not in allowed:
            continue
        if limit_per_type is not None:
            count = type_counts.get(case_type, 0)
            if count >= limit_per_type:
                continue
            type_counts[case_type] = count + 1
        selected.append(case)
        if max_cases is not None and len(selected) >= max_cases:
            break
    return selected


def _case_stale_type(case: MemoryQACase) -> str:
    if not case.queries:
        return "?"
    return str(case.queries[0].metadata.get("stale_type") or "?")


def _split_note(
    max_cases: int | None,
    stale_types: list[str],
    limit_per_stale_type: int | None,
) -> str:
    parts: list[str] = []
    if max_cases is not None:
        parts.append(f"max_cases={max_cases}")
    if stale_types:
        parts.append(f"stale_types={','.join(stale_types)}")
    if limit_per_stale_type is not None:
        parts.append(f"limit_per_stale_type={limit_per_stale_type}")
    return ";".join(parts) if parts else "all"


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the reproducible API-free STALE conversion and diagnostic workflow."
    )
    parser.add_argument("input", type=Path, help="Raw STALE JSON, e.g. T1_T2_400_FULL.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--input-format",
        choices=["auto", "raw", "adamem-jsonl"],
        default="auto",
        help="Use raw for STALE JSON arrays, adamem-jsonl for already converted files, or auto by extension.",
    )
    parser.add_argument("--convert-limit", type=int, help="Limit raw STALE samples during conversion")
    parser.add_argument("--max-cases", type=int, help="Limit converted cases during diagnostics")
    parser.add_argument("--stale-types", nargs="+", choices=["T1", "T2"])
    parser.add_argument("--limit-per-stale-type", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    manifest = run_stale_diagnostic_pipeline(
        args.input,
        args.output_dir,
        run_name=args.run_name,
        baselines=args.baselines,
        top_k=args.top_k,
        convert_limit=args.convert_limit,
        max_cases=args.max_cases,
        stale_types=args.stale_types,
        limit_per_stale_type=args.limit_per_stale_type,
        input_format=args.input_format,
    )
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(f"wrote STALE diagnostic pipeline outputs to {args.output_dir}")
        for name, path in manifest["artifacts"].items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
