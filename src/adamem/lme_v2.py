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


def write_longmemeval_v2_transfer_split(
    audit_records_source: str | Path,
    output_dir: str | Path,
    *,
    transfer_per_type: int = 10,
    control_per_group: int = 10,
    include_image_required: bool = False,
    require_haystack: bool = True,
) -> dict[str, Any]:
    """Write a deterministic LongMemEval-V2 public-transfer split manifest."""

    audit_records = _load_jsonl_objects(audit_records_source)
    selected = select_longmemeval_v2_transfer_split(
        audit_records,
        transfer_per_type=transfer_per_type,
        control_per_group=control_per_group,
        include_image_required=include_image_required,
        require_haystack=require_haystack,
    )
    summary = summarize_longmemeval_v2_transfer_split(
        selected,
        audit_records=audit_records,
        transfer_per_type=transfer_per_type,
        control_per_group=control_per_group,
        include_image_required=include_image_required,
        require_haystack=require_haystack,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "longmemeval_v2_transfer_split.records.jsonl"
    manifest_path = output / "longmemeval_v2_transfer_split.manifest.json"
    report_path = output / "longmemeval_v2_transfer_split.report.md"
    _write_jsonl(records_path, selected)
    _write_json(manifest_path, summary)
    report_path.write_text(longmemeval_v2_transfer_split_report(summary), encoding="utf-8")
    return {
        "audit_records_source": str(audit_records_source),
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def write_longmemeval_v2_trajectory_manifest(
    split_records_source: str | Path,
    haystack_source: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the trajectory id manifest needed by a selected LME-V2 split."""

    split_records = _load_jsonl_objects(split_records_source)
    haystacks = _load_haystack_json(haystack_source)
    question_records = list(longmemeval_v2_split_trajectory_records(split_records, haystacks=haystacks))
    summary = summarize_longmemeval_v2_trajectory_manifest(question_records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    question_records_path = output / "longmemeval_v2_split_trajectories.records.jsonl"
    trajectory_ids_path = output / "longmemeval_v2_split_trajectory_ids.jsonl"
    manifest_path = output / "longmemeval_v2_split_trajectory_manifest.json"
    report_path = output / "longmemeval_v2_split_trajectory_manifest.md"
    _write_jsonl(question_records_path, question_records)
    _write_jsonl(trajectory_ids_path, [{"id": trajectory_id} for trajectory_id in summary["trajectory_ids"]])
    _write_json(manifest_path, summary)
    report_path.write_text(longmemeval_v2_trajectory_manifest_report(summary), encoding="utf-8")
    return {
        "split_records_source": str(split_records_source),
        "haystack_source": str(haystack_source),
        "question_records_path": str(question_records_path),
        "trajectory_ids_path": str(trajectory_ids_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def write_longmemeval_v2_extracted_trajectories(
    trajectory_ids_source: str | Path,
    trajectories_source: str | Path,
    output_dir: str | Path,
    *,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Extract selected LongMemEval-V2 trajectories from a JSONL trajectory source."""

    requested_ids = _load_id_set(trajectory_ids_source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "longmemeval_v2_selected_trajectories.jsonl"
    missing_path = output / "longmemeval_v2_missing_trajectory_ids.jsonl"
    manifest_path = output / "longmemeval_v2_extract_manifest.json"
    report_path = output / "longmemeval_v2_extract_report.md"

    seen_ids: set[str] = set()
    records_scanned = 0
    tmp = selected_path.with_suffix(selected_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in _iter_jsonl_objects(trajectories_source):
            records_scanned += 1
            trajectory_id = _record_id(record)
            if trajectory_id in requested_ids and trajectory_id not in seen_ids:
                handle.write(json.dumps(_sanitize_trajectory_record(record), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                seen_ids.add(trajectory_id)
                if len(seen_ids) == len(requested_ids):
                    break
            if max_records is not None and records_scanned >= max_records:
                break
    tmp.replace(selected_path)

    missing_ids = sorted(requested_ids - seen_ids)
    _write_jsonl(missing_path, [{"id": trajectory_id} for trajectory_id in missing_ids])
    summary = {
        "trajectory_ids_source": str(trajectory_ids_source),
        "trajectories_source": str(trajectories_source),
        "selected_trajectories_path": str(selected_path),
        "missing_trajectory_ids_path": str(missing_path),
        "requested_trajectories": len(requested_ids),
        "matched_trajectories": len(seen_ids),
        "missing_trajectories": len(missing_ids),
        "records_scanned": records_scanned,
        "max_records": max_records,
        "completed_all_requested": len(seen_ids) == len(requested_ids),
        "label_use": "trajectory runtime fields only; answer/eval/question fields stripped",
    }
    _write_json(manifest_path, summary)
    report_path.write_text(longmemeval_v2_extract_report(summary), encoding="utf-8")
    return {
        "selected_trajectories_path": str(selected_path),
        "missing_trajectory_ids_path": str(missing_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def longmemeval_v2_extract_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# LongMemEval-V2 Trajectory Extraction",
        "",
        f"Requested trajectories: {summary['requested_trajectories']}",
        f"Matched trajectories: {summary['matched_trajectories']}",
        f"Missing trajectories: {summary['missing_trajectories']}",
        f"Records scanned: {summary['records_scanned']}",
        f"Completed all requested: {summary['completed_all_requested']}",
        f"Selected output: `{summary['selected_trajectories_path']}`",
        f"Missing-id output: `{summary['missing_trajectory_ids_path']}`",
    ]) + "\n"


def longmemeval_v2_split_trajectory_records(
    split_records: Iterable[Mapping[str, Any]],
    *,
    haystacks: Mapping[str, list[str]],
) -> Iterable[dict[str, Any]]:
    for record in split_records:
        question_id = str(record.get("id") or record.get("question_id") or "")
        trajectory_ids = haystacks.get(question_id, [])
        yield {
            "id": question_id,
            "split": record.get("split"),
            "selection_group": record.get("selection_group"),
            "question_type": record.get("question_type"),
            "domain": record.get("domain"),
            "environment": record.get("environment"),
            "abstention": record.get("abstention"),
            "image_required": record.get("image_required"),
            "state_transfer_candidate": record.get("state_transfer_candidate"),
            "type_transfer_candidate": record.get("type_transfer_candidate"),
            "query_state_slot_candidate": record.get("query_state_slot_candidate"),
            "state_slot": record.get("state_slot"),
            "inferred_state_slots": record.get("inferred_state_slots") or [],
            "trajectory_ids": list(trajectory_ids),
            "trajectory_count": len(trajectory_ids),
            "haystack_missing": question_id not in haystacks,
        }


def summarize_longmemeval_v2_trajectory_manifest(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    record_list = list(records)
    all_trajectory_ids = sorted({
        str(trajectory_id)
        for record in record_list
        for trajectory_id in record.get("trajectory_ids") or []
    })
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(record.get("split") or "<missing>") for record in record_list}):
        subset = [record for record in record_list if str(record.get("split") or "<missing>") == split]
        trajectory_ids = sorted({
            str(trajectory_id)
            for record in subset
            for trajectory_id in record.get("trajectory_ids") or []
        })
        by_split[split] = {
            "questions": len(subset),
            "unique_trajectories": len(trajectory_ids),
            "trajectory_references": sum(int(record.get("trajectory_count") or 0) for record in subset),
            "missing_haystack_questions": sum(1 for record in subset if record.get("haystack_missing")),
        }
    return {
        "total_questions": len(record_list),
        "unique_trajectories": len(all_trajectory_ids),
        "trajectory_references": sum(int(record.get("trajectory_count") or 0) for record in record_list),
        "missing_haystack_questions": sum(1 for record in record_list if record.get("haystack_missing")),
        "trajectory_ids": all_trajectory_ids,
        "by_split": by_split,
        "by_question_type": _question_type_summary(record_list),
        "by_domain": _count_by(record_list, "domain"),
        "by_environment": _count_by(record_list, "environment"),
    }


def longmemeval_v2_trajectory_manifest_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# LongMemEval-V2 Trajectory Manifest",
        "",
        f"Total questions: {summary['total_questions']}",
        f"Unique trajectories: {summary['unique_trajectories']}",
        f"Trajectory references: {summary['trajectory_references']}",
        f"Missing-haystack questions: {summary['missing_haystack_questions']}",
        "",
        "## Splits",
        "",
        "| split | questions | unique trajectories | references | missing haystack |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, aggregate in sorted(summary["by_split"].items()):
        lines.append(
            f"| {split} | {aggregate['questions']} | {aggregate['unique_trajectories']} | "
            f"{aggregate['trajectory_references']} | {aggregate['missing_haystack_questions']} |"
        )
    lines.extend([
        "",
        "## Question Types",
        "",
        "| question_type | total | inferred state | abstention |",
        "| --- | ---: | ---: | ---: |",
    ])
    for question_type, aggregate in sorted(summary["by_question_type"].items()):
        lines.append(
            f"| {question_type} | {aggregate['total']} | "
            f"{aggregate['inferred_state_slot_questions']} | {aggregate['abstention_questions']} |"
        )
    return "\n".join(lines) + "\n"


def select_longmemeval_v2_transfer_split(
    audit_records: Iterable[Mapping[str, Any]],
    *,
    transfer_per_type: int = 10,
    control_per_group: int = 10,
    include_image_required: bool = False,
    require_haystack: bool = True,
) -> list[dict[str, Any]]:
    records = list(audit_records)
    selected: list[dict[str, Any]] = []
    for question_type in sorted({
        str(record.get("question_type") or "")
        for record in records
        if record.get("type_transfer_candidate")
    }):
        candidates = [
            record for record in records
            if str(record.get("question_type") or "") == question_type
            and record.get("type_transfer_candidate")
            and _eligible_for_split(record, include_image_required=include_image_required, require_haystack=require_haystack)
        ]
        selected.extend(
            _tag_selected_records(
                _balanced_prefix(candidates, transfer_per_type, field_name="domain"),
                split="transfer",
                selection_group=question_type,
            )
        )

    static_records = [
        record for record in records
        if str(record.get("question_type") or "").startswith("static-environment")
        and _eligible_for_split(record, include_image_required=include_image_required, require_haystack=require_haystack)
    ]
    selected.extend(_tag_selected_records(
        _balanced_prefix(
            [record for record in static_records if record.get("query_state_slot_candidate")],
            control_per_group,
            field_name="domain",
        ),
        split="router_warning_control",
        selection_group="static_query_state_slot_signal",
    ))
    selected.extend(_tag_selected_records(
        _balanced_prefix(
            [record for record in static_records if not record.get("query_state_slot_candidate")],
            control_per_group,
            field_name="domain",
        ),
        split="static_clean_control",
        selection_group="static_no_state_slot_signal",
    ))
    return selected


def summarize_longmemeval_v2_transfer_split(
    selected_records: Iterable[Mapping[str, Any]],
    *,
    audit_records: Iterable[Mapping[str, Any]],
    transfer_per_type: int,
    control_per_group: int,
    include_image_required: bool,
    require_haystack: bool,
) -> dict[str, Any]:
    selected = list(selected_records)
    audit = list(audit_records)
    return {
        "total_selected": len(selected),
        "selection_policy": {
            "transfer_per_type": transfer_per_type,
            "control_per_group": control_per_group,
            "include_image_required": include_image_required,
            "require_haystack": require_haystack,
            "ordering": "source_order_with_sorted_question_type_groups_and_domain_round_robin",
            "label_use": "question metadata only; reference answers excluded",
        },
        "source_audit_total": len(audit),
        "source_type_transfer_candidates": sum(1 for record in audit if record.get("type_transfer_candidate")),
        "source_static_query_state_slot_signals": sum(
            1
            for record in audit
            if str(record.get("question_type") or "").startswith("static-environment")
            and record.get("query_state_slot_candidate")
        ),
        "excluded_image_required": sum(
            1 for record in audit if record.get("image_required") and not include_image_required
        ),
        "excluded_missing_haystack": sum(
            1
            for record in audit
            if require_haystack and not isinstance(record.get("haystack_size"), int)
        ),
        "transfer_candidate_availability": _transfer_candidate_availability(
            audit,
            selected,
            include_image_required=include_image_required,
            require_haystack=require_haystack,
        ),
        "by_split": _count_by(selected, "split"),
        "by_question_type": _question_type_summary(selected),
        "by_state_slot": _state_slot_summary(selected),
        "question_ids": [str(record.get("id") or "") for record in selected],
    }


def longmemeval_v2_transfer_split_report(summary: Mapping[str, Any]) -> str:
    policy = summary["selection_policy"]
    lines = [
        "# LongMemEval-V2 Transfer Split",
        "",
        f"Total selected: {summary['total_selected']}",
        f"Transfer per type: {policy['transfer_per_type']}",
        f"Control per group: {policy['control_per_group']}",
        f"Include image-required: {policy['include_image_required']}",
        f"Require haystack: {policy['require_haystack']}",
        f"Excluded image-required source records: {summary['excluded_image_required']}",
        f"Excluded missing-haystack source records: {summary['excluded_missing_haystack']}",
        "",
        "## Transfer Availability",
        "",
        "| question_type | source candidates | eligible | selected |",
        "| --- | ---: | ---: | ---: |",
    ]
    for question_type, aggregate in sorted(summary["transfer_candidate_availability"].items()):
        lines.append(
            f"| {question_type} | {aggregate['source_candidates']} | "
            f"{aggregate['eligible_candidates']} | {aggregate['selected']} |"
        )
    lines.extend([
        "",
        "## Splits",
        "",
        "| split | questions |",
        "| --- | ---: |",
    ])
    for split, count in sorted(summary["by_split"].items()):
        lines.append(f"| {split} | {count} |")
    lines.extend([
        "",
        "## Question Types",
        "",
        "| question_type | total | candidates | inferred state | abstention |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for question_type, aggregate in sorted(summary["by_question_type"].items()):
        lines.append(
            f"| {question_type} | {aggregate['total']} | {aggregate['state_transfer_candidates']} | "
            f"{aggregate['inferred_state_slot_questions']} | {aggregate['abstention_questions']} |"
        )
    return "\n".join(lines) + "\n"


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


def _eligible_for_split(
    record: Mapping[str, Any],
    *,
    include_image_required: bool,
    require_haystack: bool,
) -> bool:
    if not include_image_required and record.get("image_required"):
        return False
    if require_haystack and not isinstance(record.get("haystack_size"), int):
        return False
    return True


def _tag_selected_records(
    records: list[Mapping[str, Any]],
    *,
    split: str,
    selection_group: str,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        row = dict(record)
        row["split"] = split
        row["selection_group"] = selection_group
        row["selection_rank"] = index
        tagged.append(row)
    return tagged


def _balanced_prefix(
    records: list[Mapping[str, Any]],
    limit: int,
    *,
    field_name: str,
) -> list[Mapping[str, Any]]:
    if limit <= 0:
        return []
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = str(record.get(field_name) or "<missing>")
        groups.setdefault(key, []).append(record)
    selected: list[Mapping[str, Any]] = []
    group_names = sorted(groups)
    while len(selected) < limit:
        added = False
        for group_name in group_names:
            group = groups[group_name]
            if not group:
                continue
            selected.append(group.pop(0))
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def _transfer_candidate_availability(
    audit_records: list[Mapping[str, Any]],
    selected_records: list[Mapping[str, Any]],
    *,
    include_image_required: bool,
    require_haystack: bool,
) -> dict[str, dict[str, int]]:
    question_types = sorted({
        str(record.get("question_type") or "")
        for record in audit_records
        if record.get("type_transfer_candidate")
    })
    availability: dict[str, dict[str, int]] = {}
    for question_type in question_types:
        source = [
            record for record in audit_records
            if str(record.get("question_type") or "") == question_type
            and record.get("type_transfer_candidate")
        ]
        eligible = [
            record for record in source
            if _eligible_for_split(
                record,
                include_image_required=include_image_required,
                require_haystack=require_haystack,
            )
        ]
        selected = [
            record for record in selected_records
            if record.get("split") == "transfer"
            and str(record.get("question_type") or "") == question_type
        ]
        availability[question_type] = {
            "source_candidates": len(source),
            "eligible_candidates": len(eligible),
            "selected": len(selected),
        }
    return availability


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


def _iter_jsonl_objects(source: str | Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(_iter_source_lines(source), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record {line_number} must be an object")
        yield record


def _iter_source_lines(source: str | Path) -> Iterable[str]:
    source_text = str(source)
    if source_text.startswith("http://") or source_text.startswith("https://"):
        with urllib.request.urlopen(source_text, timeout=60) as response:
            for raw_line in response:
                yield raw_line.decode("utf-8")
        return
    with Path(source).open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line


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


def _load_id_set(source: str | Path) -> set[str]:
    ids: set[str] = set()
    for index, record in enumerate(_load_jsonl_objects(source), start=1):
        record_id = _record_id(record)
        if not record_id:
            raise ValueError(f"ID record {index} needs id or trajectory_id")
        ids.add(record_id)
    return ids


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("id") or record.get("trajectory_id") or "")


def _sanitize_trajectory_record(record: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: record[key]
        for key in ("id", "trajectory_id", "domain", "environment", "goal", "outcome", "start_url", "states")
        if key in record
    }
    if "id" not in sanitized and "trajectory_id" in sanitized:
        sanitized["id"] = sanitized["trajectory_id"]
    sanitized.pop("trajectory_id", None)
    return sanitized


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

    split = sub.add_parser("transfer-split", help="Select a deterministic LongMemEval-V2 transfer split")
    split.add_argument("--audit-records", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--transfer-per-type", type=int, default=10)
    split.add_argument("--control-per-group", type=int, default=10)
    split.add_argument("--include-image-required", action="store_true")
    split.add_argument("--allow-missing-haystack", action="store_true")
    split.add_argument("--json", action="store_true")

    trajectories = sub.add_parser(
        "trajectory-manifest",
        help="Map a selected LongMemEval-V2 split to required haystack trajectory ids",
    )
    trajectories.add_argument("--split-records", type=Path, required=True)
    trajectories.add_argument("--haystack", default=LONGMEMEVAL_V2_SMALL_HAYSTACK_URL)
    trajectories.add_argument("--output-dir", type=Path, required=True)
    trajectories.add_argument("--json", action="store_true")

    extract = sub.add_parser(
        "extract-trajectories",
        help="Stream selected LongMemEval-V2 trajectories from a full trajectories JSONL source",
    )
    extract.add_argument("--trajectory-ids", type=Path, required=True)
    extract.add_argument("--trajectories", required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--max-records", type=int)
    extract.add_argument("--json", action="store_true")

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
    elif args.command == "transfer-split":
        result = write_longmemeval_v2_transfer_split(
            args.audit_records,
            args.output_dir,
            transfer_per_type=args.transfer_per_type,
            control_per_group=args.control_per_group,
            include_image_required=args.include_image_required,
            require_haystack=not args.allow_missing_haystack,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"wrote LongMemEval-V2 transfer split to {args.output_dir}")
            print(f"report: {result['report_path']}")
    elif args.command == "trajectory-manifest":
        result = write_longmemeval_v2_trajectory_manifest(
            args.split_records,
            args.haystack,
            args.output_dir,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"wrote LongMemEval-V2 trajectory manifest to {args.output_dir}")
            print(f"report: {result['report_path']}")
    elif args.command == "extract-trajectories":
        result = write_longmemeval_v2_extracted_trajectories(
            args.trajectory_ids,
            args.trajectories,
            args.output_dir,
            max_records=args.max_records,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"wrote selected LongMemEval-V2 trajectories to {args.output_dir}")
            print(f"report: {result['report_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
