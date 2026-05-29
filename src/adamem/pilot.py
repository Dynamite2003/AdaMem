from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from adamem.baselines import select_baselines
from adamem.bench import (
    benchmark_case_records,
    benchmark_failure_report,
    benchmark_failure_summary,
    load_jsonl_cases,
    run_benchmark,
)
from adamem.convert import convert_ama_file
from adamem.experiments import experiment_record, write_experiment_record


AMA_PUBLIC_TEST_URL = (
    "https://huggingface.co/datasets/AMA-bench/AMA-bench/resolve/main/test/open_end_qa_set.jsonl"
)


def run_ama_public_pilot(
    *,
    output_dir: str | Path,
    limit: int = 20,
    source: str | Path = AMA_PUBLIC_TEST_URL,
    baselines: list[str] | None = None,
    top_k: int = 8,
    include_evidence_mode: bool = True,
    include_raw_outputs: bool = False,
) -> dict[str, Any]:
    """Run a reproducible API-free AMA-Bench trajectory pilot.

    The pilot creates a raw JSONL subset, answer-mode and evidence-mode AdaMem
    JSONL conversions, Markdown reports, and experiment records. It never uses
    gold answers or evidence labels inside runtime memory; they are only used
    by the benchmark adapter after retrieval.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    raw_path = output / f"ama_public_{limit}.raw.jsonl"
    source_label = str(source)
    started = time.perf_counter()
    if _is_url(source_label):
        source_count = download_jsonl_prefix(source_label, raw_path, limit=limit)
    else:
        source_count = copy_jsonl_prefix(Path(source), raw_path, limit=limit)
    timings["source_seconds"] = time.perf_counter() - started

    answer_path = output / f"ama_public_{limit}.answer.adamem.jsonl"
    started = time.perf_counter()
    answer_count = convert_ama_file(raw_path, answer_path, expected="answer", top_k=top_k)
    timings["answer_convert_seconds"] = time.perf_counter() - started

    specs = select_baselines(baselines)
    started = time.perf_counter()
    answer_outputs = _run_jsonl_pilot(
        dataset=answer_path,
        output_prefix=output / f"ama_public_{limit}.answer",
        run_name=f"ama_public_{limit}_answer",
        run_type="ama_public_answerability_pilot",
        specs=specs,
        source=source_label,
        limit=limit,
        top_k=top_k,
        include_raw_outputs=include_raw_outputs,
    )
    timings["answer_eval_seconds"] = time.perf_counter() - started
    evidence_path = output / f"ama_public_{limit}.evidence.adamem.jsonl"
    evidence_count = 0
    evidence_outputs: dict[str, Any] | None = None
    if include_evidence_mode:
        started = time.perf_counter()
        evidence_count = convert_ama_file(raw_path, evidence_path, expected="evidence", top_k=top_k)
        timings["evidence_convert_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        evidence_outputs = _run_jsonl_pilot(
            dataset=evidence_path,
            output_prefix=output / f"ama_public_{limit}.evidence",
            run_name=f"ama_public_{limit}_evidence",
            run_type="ama_public_evidence_pilot",
            specs=specs,
            source=source_label,
            limit=limit,
            top_k=top_k,
            include_raw_outputs=include_raw_outputs,
        )
        timings["evidence_eval_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = sum(timings.values())

    return {
        "source": source_label,
        "limit": limit,
        "top_k": top_k,
        "baselines": list(specs),
        "source_records": source_count,
        "answer_cases": answer_count,
        "evidence_cases": evidence_count,
        "raw_path": str(raw_path),
        "answer_dataset": str(answer_path),
        "evidence_dataset": str(evidence_path) if include_evidence_mode else None,
        "timings": {key: round(value, 4) for key, value in timings.items()},
        "answer": answer_outputs,
        "evidence": evidence_outputs,
    }


def download_jsonl_prefix(url: str, output_path: str | Path, *, limit: int) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = output.with_suffix(output.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("w", encoding="utf-8") as handle:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            _validate_jsonl_object(line, count + 1)
            handle.write(line)
            handle.write("\n")
            count += 1
            if count >= limit:
                break
    tmp.replace(output)
    return count


def copy_jsonl_prefix(input_path: str | Path, output_path: str | Path, *, limit: int) -> int:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = output.with_suffix(output.suffix + ".tmp")
    with source.open("r", encoding="utf-8") as reader, tmp.open("w", encoding="utf-8") as writer:
        for line_number, raw_line in enumerate(reader, start=1):
            line = raw_line.strip()
            if not line:
                continue
            _validate_jsonl_object(line, line_number)
            writer.write(line)
            writer.write("\n")
            count += 1
            if count >= limit:
                break
    tmp.replace(output)
    return count


def _run_jsonl_pilot(
    *,
    dataset: Path,
    output_prefix: Path,
    run_name: str,
    run_type: str,
    specs,
    source: str,
    limit: int,
    top_k: int,
    include_raw_outputs: bool,
) -> dict[str, Any]:
    cases = load_jsonl_cases(dataset)
    configs = {name: spec.config for name, spec in specs.items()}
    started = time.perf_counter()
    results = run_benchmark(cases, configs)
    benchmark_seconds = time.perf_counter() - started
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)

    records_path = output_prefix.with_suffix(".records.jsonl")
    report_path = output_prefix.with_suffix(".report.md")
    experiment_path = output_prefix.with_suffix(".experiment.json")
    _write_jsonl(records_path, records)
    _write_text(report_path, benchmark_failure_report(records))

    record = experiment_record(
        run_name=run_name,
        run_type=run_type,
        dataset=dataset,
        split_or_case_limit=f"limit={limit}",
        baselines=specs,
        results={
            "by_baseline": summary["by_baseline"],
            "paper_metrics": summary["paper_metrics"],
            "answerability": summary["answerability"],
            "evidence_support": summary["evidence_support"],
        },
        diagnostics={"failure_summary": summary},
        raw_outputs=records if include_raw_outputs else [],
        notes={
            "source": source,
            "top_k": top_k,
            "records_path": str(records_path),
            "raw_outputs_embedded": include_raw_outputs,
            "benchmark_seconds": round(benchmark_seconds, 4),
            "answer_model_required": False,
            "judge_model_required": False,
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "query_metadata_only",
        },
    )
    write_experiment_record(experiment_path, record)
    return {
        "records_path": str(records_path),
        "report_path": str(report_path),
        "experiment_path": str(experiment_path),
        "summary": summary,
    }


def _validate_jsonl_object(line: str, line_number: int) -> None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSONL line {line_number} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSONL line {line_number} must be an object")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp.replace(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run reproducible AdaMem research pilots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ama = subparsers.add_parser("ama-public", help="Run an API-free public AMA-Bench pilot")
    ama.add_argument("--output-dir", type=Path, required=True)
    ama.add_argument("--limit", type=int, default=20)
    ama.add_argument("--source", default=AMA_PUBLIC_TEST_URL)
    ama.add_argument("--baselines", nargs="+", default=["semantic_only", "full", "trajectory_step_readout"])
    ama.add_argument("--top-k", type=int, default=8)
    ama.add_argument("--answer-only", action="store_true", help="Skip separate evidence-mode conversion/eval")
    ama.add_argument("--include-raw-outputs", action="store_true", help="Embed records in experiment JSON")
    ama.add_argument("--json", action="store_true", help="Emit JSON summary instead of text")

    args = parser.parse_args(argv)
    if args.command == "ama-public":
        summary = run_ama_public_pilot(
            output_dir=args.output_dir,
            limit=args.limit,
            source=args.source,
            baselines=args.baselines,
            top_k=args.top_k,
            include_evidence_mode=not args.answer_only,
            include_raw_outputs=args.include_raw_outputs,
        )
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"wrote AMA public pilot outputs to {args.output_dir}")
            print(f"answer report: {summary['answer']['report_path']}")
            if summary["evidence"]:
                print(f"evidence report: {summary['evidence']['report_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
