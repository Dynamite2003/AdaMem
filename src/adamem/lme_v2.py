from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from adamem.state import query_relevant_state_slots

LONGMEMEVAL_V2_QUESTIONS_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/raw/main/questions.jsonl"
)
LONGMEMEVAL_V2_SMALL_HAYSTACK_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/raw/main/haystacks/lme_v2_small.json"
)

STATE_TRANSFER_TYPE_TERMS = (
    "dynamic",
    "procedure",
    "premise",
    "gotcha",
    "workflow",
    "awareness",
)


def write_longmemeval_v2_question_audit(
    questions_source: str | Path,
    output_dir: str | Path,
    *,
    haystack_source: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Write an API-free LongMemEval-V2 question-side transfer audit.

    This audit intentionally uses only question text/type metadata and optional
    haystack sizes. Reference answers are excluded so the artifact can be used
    to select public-transfer subsets without leaking evaluation labels into
    runtime memory experiments.
    """

    questions = _load_jsonl_objects(questions_source, limit=limit)
    haystacks = _load_haystack_json(haystack_source) if haystack_source else None
    records = list(longmemeval_v2_question_audit_records(questions, haystacks=haystacks))
    summary = summarize_longmemeval_v2_question_audit(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "longmemeval_v2_question_audit.records.jsonl"
    summary_path = output / "longmemeval_v2_question_audit.summary.json"
    report_path = output / "longmemeval_v2_question_audit.report.md"
    _write_jsonl(records_path, records)
    _write_json(summary_path, summary)
    report_path.write_text(longmemeval_v2_question_audit_report(summary), encoding="utf-8")
    return {
        "questions_source": str(questions_source),
        "haystack_source": str(haystack_source) if haystack_source else None,
        "limit": limit,
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def longmemeval_v2_question_audit_records(
    questions: Iterable[Mapping[str, Any]],
    *,
    haystacks: Mapping[str, list[str]] | None = None,
) -> Iterable[dict[str, Any]]:
    for question in questions:
        question_id = _question_id(question)
        question_text = str(question.get("question") or "")
        question_type = str(question.get("question_type") or "")
        inferred_slots = query_relevant_state_slots(question_text)
        candidate_reasons = _candidate_reasons(question_type, inferred_slots)
        type_candidate = _question_type_is_transfer_candidate(question_type)
        query_slot_candidate = bool(inferred_slots)
        haystack_ids = haystacks.get(question_id) if haystacks else None
        record: dict[str, Any] = {
            "id": question_id,
            "domain": question.get("domain"),
            "environment": question.get("environment"),
            "question_type": question_type,
            "abstention": question_type.endswith("-abs"),
            "image_required": bool(question.get("image")),
            "eval_function_family": _eval_function_family(question.get("eval_function")),
            "inferred_state_slots": inferred_slots,
            "type_transfer_candidate": type_candidate,
            "query_state_slot_candidate": query_slot_candidate,
            "state_transfer_candidate": bool(candidate_reasons),
            "candidate_reasons": candidate_reasons,
            "haystack_size": len(haystack_ids) if haystack_ids is not None else None,
        }
        if inferred_slots:
            record["state_slot"] = inferred_slots if len(inferred_slots) > 1 else inferred_slots[0]
            record["state_slot_source"] = "query_text_router"
        yield record


def summarize_longmemeval_v2_question_audit(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    record_list = list(records)
    total = len(record_list)
    inferred_state = [record for record in record_list if record.get("inferred_state_slots")]
    candidates = [record for record in record_list if record.get("state_transfer_candidate")]
    haystack_sizes = [
        int(record["haystack_size"])
        for record in record_list
        if isinstance(record.get("haystack_size"), int)
    ]
    return {
        "total_questions": total,
        "state_transfer_candidate_questions": len(candidates),
        "state_transfer_candidate_rate": _rate(len(candidates), total),
        "type_transfer_candidate_questions": sum(1 for record in record_list if record.get("type_transfer_candidate")),
        "query_state_slot_candidate_questions": sum(
            1 for record in record_list if record.get("query_state_slot_candidate")
        ),
        "static_query_state_slot_signal_questions": sum(
            1
            for record in record_list
            if str(record.get("question_type") or "").startswith("static-environment")
            and record.get("query_state_slot_candidate")
        ),
        "inferred_state_slot_questions": len(inferred_state),
        "inferred_state_slot_rate": _rate(len(inferred_state), total),
        "abstention_questions": sum(1 for record in record_list if record.get("abstention")),
        "image_required_questions": sum(1 for record in record_list if record.get("image_required")),
        "with_haystack_questions": len(haystack_sizes),
        "missing_haystack_questions": total - len(haystack_sizes),
        "haystack_size_min": min(haystack_sizes) if haystack_sizes else None,
        "haystack_size_max": max(haystack_sizes) if haystack_sizes else None,
        "haystack_size_avg": round(sum(haystack_sizes) / len(haystack_sizes), 4) if haystack_sizes else None,
        "by_domain": _count_by(record_list, "domain"),
        "by_environment": _count_by(record_list, "environment"),
        "by_question_type": _question_type_summary(record_list),
        "by_state_slot": _state_slot_summary(record_list),
        "by_candidate_reason": _candidate_reason_summary(record_list),
    }


def longmemeval_v2_question_audit_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# LongMemEval-V2 Question Audit",
        "",
        f"Total questions: {summary['total_questions']}",
        f"State-transfer candidates: {summary['state_transfer_candidate_questions']} "
        f"({_format_rate(summary['state_transfer_candidate_rate'])})",
        f"Type-level transfer candidates: {summary['type_transfer_candidate_questions']}",
        f"Query state-slot signals: {summary['query_state_slot_candidate_questions']}",
        f"Static-question state-slot signals: {summary['static_query_state_slot_signal_questions']}",
        f"Inferred state-slot questions: {summary['inferred_state_slot_questions']} "
        f"({_format_rate(summary['inferred_state_slot_rate'])})",
        f"Abstention questions: {summary['abstention_questions']}",
        f"Image-required questions: {summary['image_required_questions']}",
        f"Haystack coverage: {summary['with_haystack_questions']} present, "
        f"{summary['missing_haystack_questions']} missing",
        "",
        "## Question Types",
        "",
        "| question_type | total | candidates | inferred state | abstention |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for question_type, aggregate in sorted(summary["by_question_type"].items()):
        lines.append(
            f"| {question_type} | {aggregate['total']} | {aggregate['state_transfer_candidates']} | "
            f"{aggregate['inferred_state_slot_questions']} | {aggregate['abstention_questions']} |"
        )
    lines.extend([
        "",
        "## State Slots",
        "",
        "| state_slot | questions |",
        "| --- | ---: |",
    ])
    for slot, count in sorted(summary["by_state_slot"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {slot} | {count} |")
    return "\n".join(lines) + "\n"


def _load_jsonl_objects(source: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_source_text(source).splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record {line_number} must be an object")
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def _load_haystack_json(source: str | Path) -> dict[str, list[str]]:
    raw = json.loads(_read_source_text(source))
    if not isinstance(raw, Mapping):
        raise ValueError("LongMemEval-V2 haystack must be a JSON object")
    return {str(key): [str(item) for item in _as_list(value)] for key, value in raw.items()}


def _read_source_text(source: str | Path) -> str:
    source_text = str(source)
    if source_text.startswith("http://") or source_text.startswith("https://"):
        with urllib.request.urlopen(source_text, timeout=60) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def _question_id(question: Mapping[str, Any]) -> str:
    return str(question.get("id") or question.get("question_id") or "")


def _candidate_reasons(question_type: str, inferred_slots: list[str]) -> list[str]:
    reasons: list[str] = []
    if _question_type_is_transfer_candidate(question_type):
        reasons.append("question_type")
    if inferred_slots and not question_type.lower().startswith("static-environment"):
        reasons.append("query_state_slot")
    return reasons


def _question_type_is_transfer_candidate(question_type: str) -> bool:
    type_text = question_type.lower()
    return any(term in type_text for term in STATE_TRANSFER_TYPE_TERMS)


def _eval_function_family(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).split("|", 1)[0]


def _question_type_summary(records: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        question_type = str(record.get("question_type") or "<missing>")
        aggregate = summary.setdefault(
            question_type,
            {
                "total": 0,
                "state_transfer_candidates": 0,
                "type_transfer_candidates": 0,
                "query_state_slot_candidates": 0,
                "inferred_state_slot_questions": 0,
                "abstention_questions": 0,
                "image_required_questions": 0,
            },
        )
        aggregate["total"] += 1
        if record.get("state_transfer_candidate"):
            aggregate["state_transfer_candidates"] += 1
        if record.get("type_transfer_candidate"):
            aggregate["type_transfer_candidates"] += 1
        if record.get("query_state_slot_candidate"):
            aggregate["query_state_slot_candidates"] += 1
        if record.get("inferred_state_slots"):
            aggregate["inferred_state_slot_questions"] += 1
        if record.get("abstention"):
            aggregate["abstention_questions"] += 1
        if record.get("image_required"):
            aggregate["image_required_questions"] += 1
    return summary


def _state_slot_summary(records: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for slot in record.get("inferred_state_slots") or []:
            counts[str(slot)] = counts.get(str(slot), 0) + 1
    return counts


def _candidate_reason_summary(records: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in record.get("candidate_reasons") or []:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def _count_by(records: list[Mapping[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field_name) or "<missing>")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2%}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-V2 API-free utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("question-audit", help="Audit LongMemEval-V2 question-side transfer candidates")
    audit.add_argument("--questions", default=LONGMEMEVAL_V2_QUESTIONS_URL)
    audit.add_argument("--haystack", default=LONGMEMEVAL_V2_SMALL_HAYSTACK_URL)
    audit.add_argument("--no-haystack", action="store_true", help="Skip haystack-size coverage checks")
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--limit", type=int)
    audit.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "question-audit":
        result = write_longmemeval_v2_question_audit(
            args.questions,
            args.output_dir,
            haystack_source=None if args.no_haystack else args.haystack,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"wrote LongMemEval-V2 question audit to {args.output_dir}")
            print(f"report: {result['report_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
