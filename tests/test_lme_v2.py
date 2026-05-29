from __future__ import annotations

import json
from pathlib import Path

from adamem.lme_v2 import (
    longmemeval_v2_question_audit_records,
    summarize_longmemeval_v2_question_audit,
    write_longmemeval_v2_question_audit,
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
