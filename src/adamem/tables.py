from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from adamem.answer_eval import answer_failure_summary
from adamem.bench import benchmark_failure_summary


DEFAULT_GROUP_FIELDS = ("question_type", "dimension", "state_slot", "abstention")
DEFAULT_STALE_GROUP_FIELDS = ("dim", "stale_type")


def load_benchmark_records(path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark records from JSONL, a JSON array, or an experiment JSON."""

    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if source.suffix.lower() == ".jsonl":
        return _load_jsonl_records(source)

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [dict(record) for record in parsed]
    if not isinstance(parsed, dict):
        raise ValueError(f"{source} must contain JSON records or an experiment object")

    raw_outputs = parsed.get("raw_outputs")
    if raw_outputs:
        return [dict(record) for record in raw_outputs]

    notes = parsed.get("notes") or {}
    records_path = notes.get("records_path")
    if not records_path:
        raise ValueError(
            f"{source} does not embed raw_outputs and notes.records_path is missing"
        )
    resolved = _resolve_records_path(source, str(records_path))
    return _load_jsonl_records(resolved)


def paper_table_summary(
    records: list[dict[str, Any]],
    *,
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
) -> dict[str, Any]:
    """Return compact paper-table rows from benchmark case records.

    The returned object intentionally keeps counts and rates together so tables
    can be regenerated without reparsing Markdown reports.
    """

    group_fields = tuple(group_fields)
    kind = _record_kind(records)
    if kind == "stale_judge":
        fields = tuple(group_fields)
        if fields == DEFAULT_GROUP_FIELDS:
            fields = DEFAULT_STALE_GROUP_FIELDS
        summary = _stale_judge_summary(records, group_fields=fields)
        return {
            "kind": kind,
            "total_records": summary["total_records"],
            "overall": _stale_overall_rows(summary),
            "by_group": _stale_group_rows(summary),
        }
    if kind == "answer_generation":
        summary = answer_failure_summary(records, group_fields=group_fields)
        return {
            "kind": kind,
            "total_records": summary["total_records"],
            "overall": _answer_overall_rows(summary),
            "by_group": _answer_group_rows(summary),
        }

    summary = benchmark_failure_summary(records, group_fields=group_fields)
    return {
        "kind": kind,
        "total_records": summary["total_records"],
        "overall": _overall_rows(summary),
        "by_group": _group_rows(summary),
    }


def paper_table_markdown(
    records: list[dict[str, Any]],
    *,
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
    title: str = "AdaMem Paper Tables",
) -> str:
    tables = paper_table_summary(records, group_fields=group_fields)
    lines = [f"# {title}", ""]
    lines.append(f"Total records: {tables['total_records']}")
    lines.append("")

    lines.append("## Overall")
    if tables["kind"] in {"answer_generation", "stale_judge"}:
        extra = " | stale leak |" if tables["kind"] == "stale_judge" else " |"
        lines.append(f"| baseline | correct | accuracy{extra}")
        lines.append("| --- | ---: | ---: | ---: |" if tables["kind"] == "stale_judge" else "| --- | ---: | ---: |")
        for row in tables["overall"]:
            line = f"| {row['baseline']} | {row['correct']} | {_format_rate(row['accuracy'])}"
            if tables["kind"] == "stale_judge":
                line += f" | {_format_rate(row['stale_leak_rate'])}"
            lines.append(f"{line} |")
    else:
        lines.append(
            "| baseline | support | support acc | evidence support | answer recall | "
            "basis recall | basis matched |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in tables["overall"]:
            lines.append(
                f"| {row['baseline']} | {row['support']} | "
                f"{_format_rate(row['support_accuracy'])} | "
                f"{row['evidence_support']} | "
                f"{_format_optional_rate(row['answer_keyword_recall_avg'])} | "
                f"{_format_optional_rate(row['basis_answer_keyword_recall_avg'])} | "
                f"{row['basis_matched']} |"
            )
    lines.append("")

    for field_name, rows in tables["by_group"].items():
        if not rows:
            continue
        lines.append(f"## By {field_name}")
        if tables["kind"] == "answer_generation":
            lines.append("| value | baseline | correct | accuracy |")
            lines.append("| --- | --- | ---: | ---: |")
            for row in rows:
                lines.append(
                    f"| {row['value']} | {row['baseline']} | {row['correct']} | "
                    f"{_format_rate(row['accuracy'])} |"
                )
        elif tables["kind"] == "stale_judge":
            lines.append("| value | baseline | correct | accuracy | stale leak |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for row in rows:
                lines.append(
                    f"| {row['value']} | {row['baseline']} | {row['correct']} | "
                    f"{_format_rate(row['accuracy'])} | "
                    f"{_format_rate(row['stale_leak_rate'])} |"
                )
        else:
            lines.append(
                "| value | baseline | support | support acc | evidence support | "
                "answer recall | basis recall | basis matched |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in rows:
                lines.append(
                    f"| {row['value']} | {row['baseline']} | {row['support']} | "
                    f"{_format_rate(row['support_accuracy'])} | "
                    f"{row['evidence_support']} | "
                    f"{_format_optional_rate(row['answer_keyword_recall_avg'])} | "
                    f"{_format_optional_rate(row['basis_answer_keyword_recall_avg'])} | "
                    f"{row['basis_matched']} |"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_paper_table(
    input_path: str | Path,
    output_path: str | Path,
    *,
    output_format: str = "markdown",
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
    title: str = "AdaMem Paper Tables",
) -> Path:
    records = load_benchmark_records(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        payload = paper_table_summary(records, group_fields=group_fields)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    elif output_format == "markdown":
        text = paper_table_markdown(records, group_fields=group_fields, title=title)
    else:
        raise ValueError("output_format must be 'markdown' or 'json'")
    output.write_text(text, encoding="utf-8")
    return output


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def _resolve_records_path(experiment_path: Path, records_path: str) -> Path:
    candidate = Path(records_path)
    if candidate.exists():
        return candidate
    relative = experiment_path.parent / candidate
    if relative.exists():
        return relative
    raise FileNotFoundError(f"records_path does not exist: {records_path}")


def _record_kind(records: list[dict[str, Any]]) -> str:
    if records and all("judge_correct" in record for record in records):
        return "stale_judge"
    if records and all("correct" in record for record in records):
        return "answer_generation"
    return "retrieval"


def _stale_judge_summary(
    records: list[dict[str, Any]],
    *,
    group_fields: Iterable[str],
) -> dict[str, Any]:
    baseline_order = list(dict.fromkeys(str(record["baseline"]) for record in records))
    summary: dict[str, Any] = {
        "total_records": len(records),
        "by_baseline": {},
        "by_metadata": {},
    }
    for baseline in baseline_order:
        subset = [record for record in records if record["baseline"] == baseline]
        summary["by_baseline"][baseline] = _aggregate_stale_records(subset)
    for field_name in group_fields:
        values = sorted({_stale_group_value(record, field_name) for record in records})
        if values == ["<missing>"]:
            continue
        field_summary: dict[str, Any] = {}
        for value in values:
            value_subset = [
                record
                for record in records
                if _stale_group_value(record, field_name) == value
            ]
            field_summary[value] = {
                baseline: _aggregate_stale_records([
                    record for record in value_subset if record["baseline"] == baseline
                ])
                for baseline in baseline_order
                if any(record["baseline"] == baseline for record in value_subset)
            }
        summary["by_metadata"][field_name] = field_summary
    return summary


def _aggregate_stale_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record["judge_correct"])
    stale_leaks = sum(1 for record in records if record.get("stale_leak"))
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "stale_leaks": stale_leaks,
        "stale_leak_rate": stale_leaks / total if total else 0.0,
    }


def _stale_group_value(record: dict[str, Any], field_name: str) -> str:
    if field_name == "dimension":
        field_name = "dim"
    value = record.get(field_name)
    if value is None or value == "":
        metadata = record.get("metadata") or {}
        value = metadata.get(field_name) if isinstance(metadata, dict) else None
    if value is None or value == "":
        return "<missing>"
    return str(value)


def _overall_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline, support in summary["by_baseline"].items():
        evidence = summary["evidence_support"][baseline]
        answerability = summary["answerability"][baseline]
        rows.append({
            "baseline": baseline,
            "support": _fraction(support["passed"], support["total"]),
            "support_accuracy": support["accuracy"],
            "evidence_support": _fraction(
                evidence["evidence_matched_records"],
                evidence["evidence_query_total"],
            ),
            "evidence_support_rate": _ratio(
                evidence["evidence_matched_records"],
                evidence["evidence_query_total"],
            ),
            "answer_keyword_matched": _fraction(
                answerability["answer_keyword_matched_records"],
                answerability["answer_query_total"],
            ),
            "answer_keyword_recall_avg": answerability["answer_keyword_recall_avg"],
            "basis_matched": _fraction(
                answerability["basis_answer_keyword_matched_records"],
                answerability["answer_query_total"],
            ),
            "basis_answer_keyword_recall_avg": (
                answerability["basis_answer_keyword_recall_avg"]
            ),
            "answer_basis_records": answerability["answer_basis_records"],
        })
    return rows


def _answer_overall_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline, aggregate in summary["by_baseline"].items():
        rows.append({
            "baseline": baseline,
            "correct": _fraction(aggregate["correct"], aggregate["total"]),
            "accuracy": aggregate["accuracy"],
        })
    return rows


def _stale_overall_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline, aggregate in summary["by_baseline"].items():
        rows.append({
            "baseline": baseline,
            "correct": _fraction(aggregate["correct"], aggregate["total"]),
            "accuracy": aggregate["accuracy"],
            "stale_leak": _fraction(aggregate["stale_leaks"], aggregate["total"]),
            "stale_leak_rate": aggregate["stale_leak_rate"],
        })
    return rows


def _group_rows(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field_name, by_value in summary["diagnostics_by_metadata"].items():
        rows: list[dict[str, Any]] = []
        if list(by_value) == ["<missing>"]:
            grouped[field_name] = rows
            continue
        for value, by_baseline in by_value.items():
            support_by_baseline = summary["by_metadata"][field_name].get(value, {})
            for baseline, metrics in by_baseline.items():
                support = support_by_baseline.get(baseline, {})
                rows.append({
                    "value": value,
                    "baseline": baseline,
                    "support": _fraction(
                        int(support.get("passed", 0)),
                        int(support.get("total", metrics["total"])),
                    ),
                    "support_accuracy": support.get("accuracy", 0.0),
                    "evidence_support": _fraction(
                        metrics["evidence_matched_records"],
                        metrics["evidence_query_total"],
                    ),
                    "evidence_support_rate": _ratio(
                        metrics["evidence_matched_records"],
                        metrics["evidence_query_total"],
                    ),
                    "answer_keyword_matched": _fraction(
                        metrics["answer_keyword_matched_records"],
                        metrics["answer_query_total"],
                    ),
                    "answer_keyword_recall_avg": metrics["answer_keyword_recall_avg"],
                    "basis_matched": _fraction(
                        metrics["basis_answer_keyword_matched_records"],
                        metrics["answer_query_total"],
                    ),
                    "basis_answer_keyword_recall_avg": (
                        metrics["basis_answer_keyword_recall_avg"]
                    ),
                    "answer_basis_records": metrics["answer_basis_records"],
                })
        grouped[field_name] = rows
    return grouped


def _stale_group_rows(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field_name, by_value in summary["by_metadata"].items():
        rows: list[dict[str, Any]] = []
        for value, by_baseline in by_value.items():
            for baseline, aggregate in by_baseline.items():
                rows.append({
                    "value": value,
                    "baseline": baseline,
                    "correct": _fraction(aggregate["correct"], aggregate["total"]),
                    "accuracy": aggregate["accuracy"],
                    "stale_leak": _fraction(aggregate["stale_leaks"], aggregate["total"]),
                    "stale_leak_rate": aggregate["stale_leak_rate"],
                })
        grouped[field_name] = rows
    return grouped


def _answer_group_rows(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field_name, by_value in summary["by_metadata"].items():
        rows: list[dict[str, Any]] = []
        for value, by_baseline in by_value.items():
            for baseline, aggregate in by_baseline.items():
                rows.append({
                    "value": value,
                    "baseline": baseline,
                    "correct": _fraction(aggregate["correct"], aggregate["total"]),
                    "accuracy": aggregate["accuracy"],
                })
        grouped[field_name] = rows
    return grouped


def _fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator}"


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _format_optional_rate(value: float | None) -> str:
    return _format_rate(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize AdaMem benchmark records as paper-style tables."
    )
    parser.add_argument("input", type=Path, help="Records JSONL, JSON array, or experiment JSON")
    parser.add_argument("--output", type=Path, help="Write the table to a file")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--title", default="AdaMem Paper Tables")
    parser.add_argument("--group-fields", nargs="+", default=list(DEFAULT_GROUP_FIELDS))
    args = parser.parse_args(argv)

    records = load_benchmark_records(args.input)
    if args.format == "json":
        text = json.dumps(
            paper_table_summary(records, group_fields=args.group_fields),
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    else:
        text = paper_table_markdown(
            records,
            group_fields=args.group_fields,
            title=args.title,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
