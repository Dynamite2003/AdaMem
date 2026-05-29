from __future__ import annotations

import json
from pathlib import Path

from adamem.lme_v2 import (
    longmemeval_v2_question_audit_records,
    select_longmemeval_v2_transfer_split,
    summarize_longmemeval_v2_question_audit,
    summarize_longmemeval_v2_transfer_split,
    write_longmemeval_v2_question_audit,
    write_longmemeval_v2_transfer_split,
)


def test_longmemeval_v2_question_audit_marks_state_transfer_candidates() -> None:
    questions = [
        {
            "id": "dynamic_q",
            "domain": "enterprise",
            "environment": "workarena",
            "question_type": "dynamic-environment",
            "question": "Is the staging build runner offline?",
            "answer": "No",
            "eval_function": "norm_phrase_set_match|lower=true",
        },
        {
            "id": "static_q",
            "domain": "web",
            "environment": "shopping",
            "question_type": "static-environment",
            "question": "What button is visible on the product page?",
            "answer": "Checkout",
            "eval_function": "norm_phrase_set_match",
        },
        {
            "id": "procedure_abs_q",
            "domain": "enterprise",
            "environment": "workarena",
            "question_type": "procedure-abs",
            "question": "In our company workflow, what module should I use?",
            "answer": "There is no such module.",
            "eval_function": "llm_abstention_checker|require_non_empty=true",
        },
    ]
    records = list(longmemeval_v2_question_audit_records(
        questions,
        haystacks={
            "dynamic_q": ["traj1", "traj2"],
            "static_q": ["traj3"],
        },
    ))
    by_id = {record["id"]: record for record in records}
    summary = summarize_longmemeval_v2_question_audit(records)

    assert by_id["dynamic_q"]["state_transfer_candidate"] is True
    assert by_id["dynamic_q"]["candidate_reasons"] == ["question_type", "query_state_slot"]
    assert by_id["dynamic_q"]["state_slot"] == "runtime.*.status"
    assert by_id["dynamic_q"]["haystack_size"] == 2
    assert by_id["procedure_abs_q"]["abstention"] is True
    assert by_id["procedure_abs_q"]["state_transfer_candidate"] is True
    assert by_id["procedure_abs_q"]["eval_function_family"] == "llm_abstention_checker"
    assert by_id["procedure_abs_q"]["haystack_size"] is None
    assert by_id["static_q"]["state_transfer_candidate"] is False
    assert all("answer" not in record for record in records)
    assert summary["total_questions"] == 3
    assert summary["state_transfer_candidate_questions"] == 2
    assert summary["inferred_state_slot_questions"] == 2
    assert summary["with_haystack_questions"] == 2
    assert summary["missing_haystack_questions"] == 1
    assert summary["by_question_type"]["dynamic-environment"]["state_transfer_candidates"] == 1
    assert summary["by_state_slot"]["runtime.*.status"] == 1
    assert summary["by_candidate_reason"]["question_type"] == 2


def test_write_longmemeval_v2_question_audit_outputs_records_summary_and_report(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join([
            json.dumps({
                "id": "q1",
                "domain": "enterprise",
                "environment": "workarena",
                "question_type": "dynamic-environment",
                "question": "What is the current workflow status?",
                "answer": "done",
                "eval_function": "norm_phrase_set_match",
            }),
            json.dumps({
                "id": "q2",
                "domain": "web",
                "environment": "shopping",
                "question_type": "static-environment",
                "question": "What color is the checkout button?",
                "answer": "blue",
                "eval_function": "norm_phrase_set_match",
            }),
        ]),
        encoding="utf-8",
    )
    haystack = tmp_path / "lme_v2_small.json"
    haystack.write_text(json.dumps({"q1": ["traj1"]}), encoding="utf-8")
    output_dir = tmp_path / "audit"

    result = write_longmemeval_v2_question_audit(
        questions,
        output_dir,
        haystack_source=haystack,
        limit=1,
    )
    records_path = Path(result["records_path"])
    summary_path = Path(result["summary_path"])
    report_path = Path(result["report_path"])
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert records_path.exists()
    assert summary_path.exists()
    assert report_path.exists()
    assert [record["id"] for record in records] == ["q1"]
    assert records[0]["haystack_size"] == 1
    assert "answer" not in records[0]
    assert summary["total_questions"] == 1
    assert "LongMemEval-V2 Question Audit" in report
    assert "dynamic-environment" in report


def test_longmemeval_v2_transfer_split_balances_candidates_and_controls() -> None:
    records = [
        _audit_record("dyn1", "dynamic-environment", state_slots=["runtime.*.status"]),
        _audit_record("dyn2", "dynamic-environment", state_slots=["resource.*.status"]),
        _audit_record("proc1", "procedure", state_slots=["workflow.*"]),
        _audit_record("proc_img", "procedure", image_required=True, state_slots=["workflow.*"]),
        _audit_record("gotcha_img", "errors-gotchas", image_required=True, state_slots=["workflow.*"]),
        _audit_record("static_warn1", "static-environment", type_candidate=False, state_slots=["location"]),
        _audit_record("static_warn2", "static-environment-abs", type_candidate=False, state_slots=["location"]),
        _audit_record("static_clean1", "static-environment", type_candidate=False),
        _audit_record("static_clean2", "static-environment-abs", type_candidate=False),
        _audit_record("missing_haystack", "dynamic-environment", haystack_size=None, state_slots=["runtime.*.status"]),
    ]

    selected = select_longmemeval_v2_transfer_split(
        records,
        transfer_per_type=1,
        control_per_group=1,
        include_image_required=False,
        require_haystack=True,
    )
    summary = summarize_longmemeval_v2_transfer_split(
        selected,
        audit_records=records,
        transfer_per_type=1,
        control_per_group=1,
        include_image_required=False,
        require_haystack=True,
    )

    assert [record["id"] for record in selected] == [
        "dyn1",
        "proc1",
        "static_warn1",
        "static_clean1",
    ]
    assert [record["split"] for record in selected] == [
        "transfer",
        "transfer",
        "router_warning_control",
        "static_clean_control",
    ]
    assert all("answer" not in record for record in selected)
    assert summary["total_selected"] == 4
    assert summary["by_split"] == {
        "transfer": 2,
        "router_warning_control": 1,
        "static_clean_control": 1,
    }
    assert summary["excluded_image_required"] == 2
    assert summary["excluded_missing_haystack"] == 1
    assert summary["transfer_candidate_availability"]["errors-gotchas"] == {
        "source_candidates": 1,
        "eligible_candidates": 0,
        "selected": 0,
    }
    assert summary["question_ids"] == ["dyn1", "proc1", "static_warn1", "static_clean1"]


def test_write_longmemeval_v2_transfer_split_outputs_manifest(tmp_path: Path) -> None:
    audit_records = tmp_path / "audit.records.jsonl"
    audit_records.write_text(
        "\n".join([
            json.dumps(_audit_record("dyn1", "dynamic-environment", state_slots=["runtime.*.status"])),
            json.dumps(_audit_record("proc1", "procedure", state_slots=["workflow.*"])),
            json.dumps(_audit_record("static_warn", "static-environment", type_candidate=False, state_slots=["location"])),
            json.dumps(_audit_record("static_clean", "static-environment", type_candidate=False)),
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "split"

    result = write_longmemeval_v2_transfer_split(
        audit_records,
        output_dir,
        transfer_per_type=2,
        control_per_group=1,
    )
    records_path = Path(result["records_path"])
    manifest_path = Path(result["manifest_path"])
    report_path = Path(result["report_path"])
    selected = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert records_path.exists()
    assert manifest_path.exists()
    assert report_path.exists()
    assert [record["id"] for record in selected] == ["dyn1", "proc1", "static_warn", "static_clean"]
    assert manifest["selection_policy"]["label_use"] == "question metadata only; reference answers excluded"
    assert manifest["transfer_candidate_availability"]["dynamic-environment"]["selected"] == 1
    assert manifest["by_split"]["transfer"] == 2
    assert "LongMemEval-V2 Transfer Split" in report


def _audit_record(
    question_id: str,
    question_type: str,
    *,
    type_candidate: bool = True,
    state_slots: list[str] | None = None,
    image_required: bool = False,
    haystack_size: int | None = 100,
) -> dict[str, object]:
    state_slots = list(state_slots or [])
    candidate_reasons = []
    if type_candidate:
        candidate_reasons.append("question_type")
    if state_slots and not question_type.startswith("static-environment"):
        candidate_reasons.append("query_state_slot")
    return {
        "id": question_id,
        "domain": "enterprise",
        "environment": "workarena",
        "question_type": question_type,
        "abstention": question_type.endswith("-abs"),
        "image_required": image_required,
        "eval_function_family": "norm_phrase_set_match",
        "inferred_state_slots": state_slots,
        "type_transfer_candidate": type_candidate,
        "query_state_slot_candidate": bool(state_slots),
        "state_transfer_candidate": bool(candidate_reasons),
        "candidate_reasons": candidate_reasons,
        "haystack_size": haystack_size,
    }
