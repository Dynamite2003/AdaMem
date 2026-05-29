from __future__ import annotations

import json
from pathlib import Path

from adamem.compare import (
    paired_comparison_markdown,
    paired_comparison_summary,
    write_paired_comparison,
)


def test_paired_comparison_supports_retrieval_records() -> None:
    records = [
        _retrieval_record("semantic_only", "q1", passed=False, question_type="A"),
        _retrieval_record("semantic_only", "q2", passed=True, question_type="B"),
        _retrieval_record("state_readout", "q1", passed=True, question_type="A"),
        _retrieval_record("state_readout", "q2", passed=False, question_type="B"),
    ]

    summary = paired_comparison_summary(records, reference="semantic_only", group_fields=["question_type"])
    comparison = summary["comparisons"]["state_readout"]

    assert summary["record_kind"] == "retrieval"
    assert comparison["gained"] == 1
    assert comparison["lost"] == 1
    assert comparison["net_delta"] == 0
    assert comparison["sign_test_p"] == 1.0
    assert comparison["by_group"]["question_type"]["A"]["gained"] == 1
    assert comparison["by_group"]["question_type"]["B"]["lost"] == 1


def test_paired_comparison_auto_uses_evidence_support_for_evidence_records() -> None:
    records = [
        _retrieval_record("semantic_only", "q1", passed=False, question_type="A", evidence=False),
        _retrieval_record("semantic_only", "q2", passed=False, question_type="A", evidence=False),
        _retrieval_record("trajectory_step_readout", "q1", passed=False, question_type="A", evidence=True),
        _retrieval_record("trajectory_step_readout", "q2", passed=False, question_type="A", evidence=True),
    ]

    summary = paired_comparison_summary(records, reference="semantic_only")
    comparison = summary["comparisons"]["trajectory_step_readout"]

    assert summary["metric"] == "evidence_support_matched"
    assert comparison["gained"] == 2
    assert comparison["lost"] == 0


def test_paired_comparison_supports_answer_generation_records() -> None:
    records = [
        _answer_record("semantic_only", "q1", correct=False),
        _answer_record("semantic_only", "q2", correct=False),
        _answer_record("trajectory_step_readout", "q1", correct=True),
        _answer_record("trajectory_step_readout", "q2", correct=True),
    ]

    summary = paired_comparison_summary(records)
    comparison = summary["comparisons"]["trajectory_step_readout"]

    assert summary["record_kind"] == "answer_generation"
    assert comparison["gained"] == 2
    assert comparison["lost"] == 0
    assert comparison["sign_test_p"] == 0.5


def test_paired_comparison_supports_stale_judge_records() -> None:
    records = [
        _stale_record("semantic_only", "q1", dim=1, stale_type="T1", correct=False),
        _stale_record("semantic_only", "q2", dim=2, stale_type="T1", correct=False),
        _stale_record("state_readout", "q1", dim=1, stale_type="T1", correct=True),
        _stale_record("state_readout", "q2", dim=2, stale_type="T1", correct=False),
    ]

    summary = paired_comparison_summary(records)
    comparison = summary["comparisons"]["state_readout"]
    markdown = paired_comparison_markdown(summary, title="STALE Paired")

    assert summary["record_kind"] == "stale_judge"
    assert comparison["gained"] == 1
    assert comparison["lost"] == 0
    assert comparison["by_group"]["dim"]["1"]["gained"] == 1
    assert comparison["by_group"]["stale_type"]["T1"]["net_delta"] == 1
    assert "# STALE Paired" in markdown
    assert "| state_readout | 2 | 1 | 0 | 1 | 0 | 1 | 1.0000 |" in markdown


def test_write_paired_comparison_from_experiment(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join([
            json.dumps(_answer_record("semantic_only", "q1", correct=False)),
            json.dumps(_answer_record("trajectory_step_readout", "q1", correct=True)),
        ]) + "\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps({"raw_outputs": [], "notes": {"records_path": "records.jsonl"}}),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"

    write_paired_comparison(experiment, output, output_format="json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["record_kind"] == "answer_generation"
    assert payload["comparisons"]["trajectory_step_readout"]["gained"] == 1


def _retrieval_record(
    baseline: str,
    query_id: str,
    *,
    passed: bool,
    question_type: str,
    evidence: bool | None = None,
) -> dict:
    record = {
        "baseline": baseline,
        "case_id": "case-1",
        "query_id": query_id,
        "passed": passed,
        "metadata": {"question_type": question_type},
    }
    if evidence is not None:
        record["expected_evidence"] = ["step001"]
        record["evidence_support_matched"] = evidence
    return record


def _answer_record(baseline: str, query_id: str, *, correct: bool) -> dict:
    return {
        "baseline": baseline,
        "case_id": "case-1",
        "query_id": query_id,
        "correct": correct,
        "metadata": {"question_type": "A"},
    }


def _stale_record(
    baseline: str,
    query_id: str,
    *,
    dim: int,
    stale_type: str,
    correct: bool,
) -> dict:
    return {
        "baseline": baseline,
        "case_id": "case-1",
        "query_id": query_id,
        "dim": dim,
        "stale_type": stale_type,
        "judge_correct": correct,
    }
