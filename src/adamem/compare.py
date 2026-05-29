from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

from adamem.tables import DEFAULT_GROUP_FIELDS, DEFAULT_STALE_GROUP_FIELDS, load_benchmark_records


def paired_comparison_summary(
    records: list[dict[str, Any]],
    *,
    reference: str | None = None,
    group_fields: Iterable[str] | None = None,
    metric: str = "auto",
) -> dict[str, Any]:
    if not records:
        return {"record_kind": "empty", "reference": reference, "comparisons": {}}
    kind = _record_kind(records)
    baseline_order = list(dict.fromkeys(str(record["baseline"]) for record in records))
    reference = reference or baseline_order[0]
    if reference not in baseline_order:
        raise ValueError(f"reference baseline not found: {reference}")
    fields = tuple(group_fields or (_default_group_fields(kind)))
    outcome_field = _outcome_field(kind, records=records, metric=metric)
    comparisons = {
        candidate: _paired_comparison(
            records,
            reference=reference,
            candidate=candidate,
            outcome_field=outcome_field,
            group_fields=fields,
            stale_groups=(kind == "stale_judge"),
        )
        for candidate in baseline_order
        if candidate != reference
    }
    return {
        "record_kind": kind,
        "metric": outcome_field,
        "reference": reference,
        "baselines": baseline_order,
        "comparisons": comparisons,
    }


def paired_comparison_markdown(summary: dict[str, Any], *, title: str = "AdaMem Paired Comparison") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Record kind: `{summary['record_kind']}`")
    lines.append(f"Metric: `{summary.get('metric') or '<missing>'}`")
    lines.append(f"Reference: `{summary.get('reference') or '<none>'}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("| candidate | common | gained | lost | net | both correct | both wrong | sign p |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    comparisons = summary.get("comparisons", {})
    if not comparisons:
        lines.append("| <none> | 0 | 0 | 0 | 0 | 0 | 0 | n/a |")
    for candidate, comparison in comparisons.items():
        lines.append(
            f"| {candidate} | {comparison['common_total']} | {comparison['gained']} | "
            f"{comparison['lost']} | {comparison['net_delta']} | {comparison['both_correct']} | "
            f"{comparison['both_wrong']} | {_format_p(comparison['sign_test_p'])} |"
        )
    lines.append("")
    for candidate, comparison in comparisons.items():
        for field_name, by_value in comparison.get("by_group", {}).items():
            lines.append(f"## {candidate} By {field_name}")
            lines.append("| value | common | gained | lost | net | sign p |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for value, counts in by_value.items():
                lines.append(
                    f"| {value} | {counts['common_total']} | {counts['gained']} | "
                    f"{counts['lost']} | {counts['net_delta']} | {_format_p(counts['sign_test_p'])} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_paired_comparison(
    input_path: str | Path,
    output_path: str | Path,
    *,
    output_format: str = "markdown",
    reference: str | None = None,
    group_fields: Iterable[str] | None = None,
    metric: str = "auto",
    title: str = "AdaMem Paired Comparison",
) -> Path:
    records = load_benchmark_records(input_path)
    summary = paired_comparison_summary(
        records,
        reference=reference,
        group_fields=group_fields,
        metric=metric,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    elif output_format == "markdown":
        text = paired_comparison_markdown(summary, title=title)
    else:
        raise ValueError("output_format must be 'markdown' or 'json'")
    output.write_text(text, encoding="utf-8")
    return output


def _paired_comparison(
    records: list[dict[str, Any]],
    *,
    reference: str,
    candidate: str,
    outcome_field: str,
    group_fields: tuple[str, ...],
    stale_groups: bool,
) -> dict[str, Any]:
    reference_records = {
        _record_key(record): record
        for record in records
        if str(record["baseline"]) == reference
    }
    candidate_records = {
        _record_key(record): record
        for record in records
        if str(record["baseline"]) == candidate
    }
    common_keys = sorted(set(reference_records) & set(candidate_records))
    base = _paired_counts(reference_records, candidate_records, common_keys, outcome_field)
    base["reference"] = reference
    base["candidate"] = candidate
    base["by_group"] = {}
    for field_name in group_fields:
        values = sorted({
            _group_value(reference_records[key], field_name, stale_groups=stale_groups)
            for key in common_keys
        })
        if values == ["<missing>"]:
            continue
        base["by_group"][field_name] = {
            value: _paired_counts(
                reference_records,
                candidate_records,
                [
                    key for key in common_keys
                    if _group_value(reference_records[key], field_name, stale_groups=stale_groups) == value
                ],
                outcome_field,
            )
            for value in values
        }
    return base


def _paired_counts(
    reference_records: dict[tuple[str, str], dict[str, Any]],
    candidate_records: dict[tuple[str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
    outcome_field: str,
) -> dict[str, Any]:
    both_correct = 0
    both_wrong = 0
    gained = 0
    lost = 0
    for key in keys:
        reference_ok = bool(reference_records[key][outcome_field])
        candidate_ok = bool(candidate_records[key][outcome_field])
        if reference_ok and candidate_ok:
            both_correct += 1
        elif not reference_ok and not candidate_ok:
            both_wrong += 1
        elif candidate_ok:
            gained += 1
        else:
            lost += 1
    return {
        "common_total": len(keys),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "gained": gained,
        "lost": lost,
        "net_delta": gained - lost,
        "sign_test_p": _two_sided_sign_test_p(gained, lost),
    }


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record["case_id"]), str(record["query_id"]))


def _record_kind(records: list[dict[str, Any]]) -> str:
    if all("judge_correct" in record for record in records):
        return "stale_judge"
    if all("correct" in record for record in records):
        return "answer_generation"
    return "retrieval"


def _outcome_field(kind: str, *, records: list[dict[str, Any]], metric: str) -> str:
    if kind == "stale_judge":
        _validate_metric(metric, {"auto", "judge_correct"})
        return "judge_correct"
    if kind == "answer_generation":
        _validate_metric(metric, {"auto", "correct"})
        return "correct"
    retrieval_metrics = {
        "auto": "evidence_support_matched" if any(record.get("expected_evidence") for record in records) else "passed",
        "passed": "passed",
        "evidence_support": "evidence_support_matched",
        "answer_keyword_support": "answer_keyword_support_matched",
        "basis_answer_keyword_support": "basis_answer_keyword_support_matched",
        "state_slot_match": "state_slot_matched",
    }
    if metric not in retrieval_metrics:
        raise ValueError(f"unsupported retrieval comparison metric: {metric}")
    return retrieval_metrics[metric]


def _validate_metric(metric: str, allowed: set[str]) -> None:
    if metric not in allowed:
        raise ValueError(f"unsupported comparison metric {metric!r}; allowed: {sorted(allowed)}")


def _default_group_fields(kind: str) -> tuple[str, ...]:
    if kind == "stale_judge":
        return DEFAULT_STALE_GROUP_FIELDS
    return DEFAULT_GROUP_FIELDS


def _group_value(record: dict[str, Any], field_name: str, *, stale_groups: bool) -> str:
    if stale_groups:
        if field_name == "dimension":
            field_name = "dim"
        value = record.get(field_name)
    else:
        metadata = record.get("metadata") or {}
        value = metadata.get(field_name) if isinstance(metadata, dict) else None
    if value is None or value == "":
        return "<missing>"
    return str(value)


def _two_sided_sign_test_p(gained: int, lost: int) -> float | None:
    discordant = gained + lost
    if discordant == 0:
        return None
    smaller = min(gained, lost)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def _format_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run paired baseline comparisons for AdaMem benchmark records."
    )
    parser.add_argument("input", type=Path, help="Records JSONL, JSON array, or experiment JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--reference")
    parser.add_argument("--group-fields", nargs="+")
    parser.add_argument(
        "--metric",
        default="auto",
        help=(
            "Comparison metric. Retrieval supports passed, evidence_support, "
            "answer_keyword_support, basis_answer_keyword_support, state_slot_match, or auto."
        ),
    )
    parser.add_argument("--title", default="AdaMem Paired Comparison")
    args = parser.parse_args(argv)

    records = load_benchmark_records(args.input)
    summary = paired_comparison_summary(
        records,
        reference=args.reference,
        group_fields=args.group_fields,
        metric=args.metric,
    )
    text = (
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else paired_comparison_markdown(summary, title=args.title)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
