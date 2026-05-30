from __future__ import annotations

import json
from pathlib import Path

from adamem.tables import (
    load_benchmark_records,
    paper_table_markdown,
    paper_table_summary,
    write_paper_table,
)


def test_paper_table_summary_reports_overall_and_grouped_metrics() -> None:
    records = [
        _record(
            baseline="semantic_only",
            question_type="A",
            passed=False,
            evidence_matched=False,
            answer_recall=0.25,
            basis_recall=0.25,
        ),
        _record(
            baseline="trajectory_step_readout",
            question_type="A",
            passed=False,
            evidence_matched=True,
            answer_recall=0.25,
            basis_recall=0.75,
            basis_matched=True,
            answer_basis="Step 7 action: right",
        ),
        _record(
            baseline="semantic_only",
            question_type="B",
            passed=True,
            evidence_matched=True,
            answer_recall=1.0,
            basis_recall=1.0,
            answer_matched=True,
            basis_matched=True,
            answer_basis="direct support",
        ),
        _record(
            baseline="trajectory_step_readout",
            question_type="B",
            passed=True,
            evidence_matched=True,
            answer_recall=1.0,
            basis_recall=1.0,
            answer_matched=True,
            basis_matched=True,
            answer_basis="direct support",
        ),
    ]

    summary = paper_table_summary(records, group_fields=["question_type"])

    assert summary["total_records"] == 4
    by_baseline = {row["baseline"]: row for row in summary["overall"]}
    assert by_baseline["semantic_only"]["support"] == "1/2"
    assert by_baseline["semantic_only"]["evidence_support"] == "1/2"
    assert by_baseline["trajectory_step_readout"]["evidence_support"] == "2/2"
    assert by_baseline["trajectory_step_readout"]["basis_matched"] == "2/2"
    grouped = {
        (row["value"], row["baseline"]): row
        for row in summary["by_group"]["question_type"]
    }
    assert grouped[("A", "semantic_only")]["evidence_support"] == "0/1"
    assert grouped[("A", "trajectory_step_readout")]["basis_matched"] == "1/1"


def test_paper_table_markdown_is_compact_and_reproducible() -> None:
    markdown = paper_table_markdown(
        [_record(baseline="trajectory_step_readout", question_type="C")],
        group_fields=["question_type"],
        title="AMA Pilot Tables",
    )

    assert markdown.startswith("# AMA Pilot Tables\n")
    assert "| trajectory_step_readout | 0/1 | 0.00% | 1/1 | 50.00% | 50.00% | 0/1 |" in markdown
    assert "## By question_type" in markdown
    assert "| C | trajectory_step_readout | 0/1 | 0.00% | 1/1 | 50.00% | 50.00% | 0/1 |" in markdown


def test_load_benchmark_records_follows_experiment_records_path(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    record = _record(baseline="semantic_only")
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    experiment_path = tmp_path / "experiment.json"
    experiment_path.write_text(
        json.dumps({"raw_outputs": [], "notes": {"records_path": "records.jsonl"}}),
        encoding="utf-8",
    )

    assert load_benchmark_records(experiment_path) == [record]


def test_write_paper_table_json(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "table.json"
    records_path.write_text(
        json.dumps(_record(baseline="trajectory_step_readout")) + "\n",
        encoding="utf-8",
    )

    written = write_paper_table(records_path, output_path, output_format="json")
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written == output_path
    assert payload["overall"][0]["baseline"] == "trajectory_step_readout"
    assert payload["overall"][0]["evidence_support"] == "1/1"


def test_paper_table_summary_supports_answer_generation_records() -> None:
    records = [
        _answer_record(baseline="semantic_only", question_type="A", correct=False),
        _answer_record(baseline="semantic_only", question_type="B", correct=True),
        _answer_record(baseline="trajectory_step_readout", question_type="A", correct=True),
        _answer_record(baseline="trajectory_step_readout", question_type="B", correct=True),
    ]

    summary = paper_table_summary(records, group_fields=["question_type"])

    assert summary["kind"] == "answer_generation"
    by_baseline = {row["baseline"]: row for row in summary["overall"]}
    assert by_baseline["semantic_only"]["correct"] == "1/2"
    assert by_baseline["trajectory_step_readout"]["correct"] == "2/2"
    grouped = {
        (row["value"], row["baseline"]): row
        for row in summary["by_group"]["question_type"]
    }
    assert grouped[("A", "semantic_only")]["correct"] == "0/1"
    assert grouped[("A", "trajectory_step_readout")]["accuracy"] == 1.0


def test_paper_table_markdown_supports_answer_generation_records() -> None:
    markdown = paper_table_markdown(
        [
            _answer_record(baseline="trajectory_step_readout", question_type="A", correct=True),
            _answer_record(baseline="trajectory_step_readout", question_type="B", correct=False),
        ],
        group_fields=["question_type"],
        title="AMA Answer Tables",
    )

    assert markdown.startswith("# AMA Answer Tables\n")
    assert "| trajectory_step_readout | 1/2 | 50.00% |" in markdown
    assert "## By question_type" in markdown
    assert "| A | trajectory_step_readout | 1/1 | 100.00% |" in markdown


def test_write_paper_table_json_supports_answer_experiment(tmp_path: Path) -> None:
    records_path = tmp_path / "generation.records.jsonl"
    output_path = tmp_path / "generation.table.json"
    records_path.write_text(
        json.dumps(_answer_record(baseline="trajectory_step_readout", correct=True)) + "\n",
        encoding="utf-8",
    )
    experiment_path = tmp_path / "generation.experiment.json"
    experiment_path.write_text(
        json.dumps({"raw_outputs": [], "notes": {"records_path": "generation.records.jsonl"}}),
        encoding="utf-8",
    )

    write_paper_table(experiment_path, output_path, output_format="json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["kind"] == "answer_generation"
    assert payload["overall"][0]["correct"] == "1/1"


def test_paper_table_summary_supports_stale_judge_records() -> None:
    records = [
        _stale_record(baseline="semantic_only", dim=1, stale_type="T1", correct=False, stale_leak=True),
        _stale_record(baseline="semantic_only", dim=2, stale_type="T2", correct=True),
        _stale_record(baseline="state_readout", dim=1, stale_type="T1", correct=True),
        _stale_record(baseline="state_readout", dim=2, stale_type="T2", correct=True),
    ]

    summary = paper_table_summary(records)

    assert summary["kind"] == "stale_judge"
    by_baseline = {row["baseline"]: row for row in summary["overall"]}
    assert by_baseline["semantic_only"]["correct"] == "1/2"
    assert by_baseline["semantic_only"]["stale_leak"] == "1/2"
    assert by_baseline["state_readout"]["correct"] == "2/2"
    dim_rows = {
        (row["value"], row["baseline"]): row
        for row in summary["by_group"]["dim"]
    }
    assert dim_rows[("1", "semantic_only")]["correct"] == "0/1"
    type_rows = {
        (row["value"], row["baseline"]): row
        for row in summary["by_group"]["stale_type"]
    }
    assert type_rows[("T2", "state_readout")]["accuracy"] == 1.0


def test_paper_table_markdown_supports_stale_judge_records() -> None:
    markdown = paper_table_markdown(
        [
            _stale_record(baseline="state_readout", dim=1, stale_type="T1", correct=True),
            _stale_record(baseline="state_readout", dim=2, stale_type="T2", correct=False, stale_leak=True),
        ],
        title="STALE Judge Tables",
    )

    assert markdown.startswith("# STALE Judge Tables\n")
    assert "| state_readout | 1/2 | 50.00% | 50.00% |" in markdown
    assert "## By dim" in markdown
    assert "| 1 | state_readout | 1/1 | 100.00% | 0.00% |" in markdown
    assert "## By stale_type" in markdown


def test_write_paper_table_json_supports_stale_experiment_raw_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "stale.table.json"
    experiment_path = tmp_path / "stale.experiment.json"
    experiment_path.write_text(
        json.dumps({
            "run_type": "stale_llm_judge",
            "raw_outputs": [
                _stale_record(baseline="state_readout", dim=1, stale_type="T1", correct=True)
            ],
        }),
        encoding="utf-8",
    )

    write_paper_table(experiment_path, output_path, output_format="json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["kind"] == "stale_judge"
    assert payload["overall"][0]["correct"] == "1/1"


def test_paper_table_summary_supports_stale_retrieval_diagnostics() -> None:
    records = [
        _stale_diagnostic_record(
            name="semantic_state_adjudication",
            correction_hits=[False, False],
        ),
        _stale_diagnostic_record(
            name="semantic_state_premise_correction",
            correction_hits=[True, False],
        ),
    ]

    summary = paper_table_summary(records)

    assert summary["kind"] == "stale_retrieval_diagnostics"
    by_baseline = {row["baseline"]: row for row in summary["overall"]}
    assert by_baseline["semantic_state_adjudication"]["queries"] == 2
    assert by_baseline["semantic_state_adjudication"]["premise_correction_hit_rate"] == 0.0
    assert by_baseline["semantic_state_premise_correction"]["premise_correction_hit_rate"] == 0.5
    dim_rows = {
        (row["value"], row["baseline"]): row
        for row in summary["by_group"]["dim"]
    }
    assert dim_rows[("1", "semantic_state_premise_correction")]["premise_correction_hit_rate"] == 1.0
    state_rows = {
        (row["value"], row["baseline"]): row
        for row in summary["by_group"]["expected_state_slot"]
    }
    assert state_rows[("location", "semantic_state_premise_correction")]["queries"] == 2
    dependency_rows = {
        (row["value"], row["baseline"]): row
        for row in summary["by_group"]["dependency_target_family"]
    }
    assert dependency_rows[("local_context", "semantic_state_premise_correction")]["queries"] == 1
    assert dependency_rows[("local_context", "semantic_state_premise_correction")][
        "premise_correction_hit_rate"
    ] == 0.0


def test_paper_table_markdown_supports_stale_retrieval_diagnostic_experiment(tmp_path: Path) -> None:
    experiment_path = tmp_path / "stale_diagnostics.experiment.json"
    experiment_path.write_text(
        json.dumps({
            "run_type": "stale_retrieval_diagnostics",
            "diagnostics": [
                _stale_diagnostic_record(
                    name="semantic_state_premise_correction",
                    correction_hits=[True, False],
                ),
            ],
            "raw_outputs": [
                {"baseline": "semantic_state_premise_correction", "failure_modes": []}
            ],
        }),
        encoding="utf-8",
    )

    markdown = paper_table_markdown(load_benchmark_records(experiment_path), title="STALE Retrieval Tables")

    assert markdown.startswith("# STALE Retrieval Tables\n")
    assert "premise correction hit" in markdown
    assert "| semantic_state_premise_correction | 2 | 100.00% | 0.00% |" in markdown
    assert "## By dim" in markdown
    assert "| 1 | semantic_state_premise_correction | 1 | 100.00% | 0.00% | 100.00% | 100.00% |" in markdown
    assert "## By expected_state_slot" in markdown
    assert "## By dependency_target_family" in markdown


def _record(
    *,
    baseline: str,
    question_type: str = "A",
    passed: bool = False,
    evidence_matched: bool = True,
    answer_recall: float = 0.5,
    basis_recall: float = 0.5,
    answer_matched: bool = False,
    basis_matched: bool = False,
    answer_basis: str = "",
) -> dict[str, object]:
    missing_evidence = [] if evidence_matched else ["step001"]
    return {
        "baseline": baseline,
        "case_id": "case-1",
        "query_id": f"query-{question_type}",
        "query": "What happened at Step 1?",
        "passed": passed,
        "retrieved": [],
        "expected_substrings": ["answer"],
        "forbidden_substrings": [],
        "missing_expected": [] if passed else ["answer"],
        "present_forbidden": [],
        "failure_modes": [] if passed else ["expected_support_missing"],
        "metadata": {"question_type": question_type, "benchmark": "ama"},
        "trace": [],
        "expected_evidence": ["step001"],
        "missing_evidence": missing_evidence,
        "evidence_support_matched": evidence_matched,
        "graph_evidence_hit_count": 0,
        "graph_evidence_hits": [],
        "graph_retrieval_count": 0,
        "answer_keywords": ["answer", "step"],
        "missing_answer_keywords": [] if answer_matched else ["answer"],
        "answer_keyword_support_matched": answer_matched,
        "answer_keyword_recall": answer_recall,
        "answer_basis": answer_basis,
        "basis_missing_answer_keywords": [] if basis_matched else ["answer"],
        "basis_answer_keyword_support_matched": basis_matched,
        "basis_answer_keyword_recall": basis_recall,
        "state_retrieval_count": 0,
        "retrieved_state_slots": [],
        "expected_state_slots": [],
        "unexpected_state_slots": [],
        "state_slot_matched": False,
        "state_sensitive": False,
        "state_available": False,
        "state_readout_expected": False,
    }


def _answer_record(
    *,
    baseline: str,
    question_type: str = "A",
    correct: bool = False,
) -> dict[str, object]:
    return {
        "baseline": baseline,
        "case_id": "case-1",
        "query_id": f"answer-{question_type}",
        "query": "What happened at Step 1?",
        "correct": correct,
        "answer": "The agent moved left.",
        "score_raw": "CORRECT" if correct else "INCORRECT",
        "retrieved": ["[step001.action] action: left"],
        "trace": [],
        "expected_substrings": ["left"],
        "forbidden_substrings": [],
        "metadata": {"question_type": question_type, "benchmark": "ama"},
    }


def _stale_record(
    *,
    baseline: str,
    dim: int,
    stale_type: str,
    correct: bool,
    stale_leak: bool = False,
) -> dict[str, object]:
    return {
        "baseline": baseline,
        "case_id": "stale-case",
        "query_id": f"dim{dim}",
        "dim": dim,
        "stale_type": stale_type,
        "query": "Do I still live in Seattle?",
        "answer_raw": "The user lives in Boston.",
        "judge_raw": "CORRECT" if correct else "INCORRECT",
        "judge_correct": correct,
        "stale_leak": stale_leak,
        "retrieved": [],
    }


def _stale_diagnostic_record(*, name: str, correction_hits: list[bool]) -> dict[str, object]:
    queries = [
        {
            "case_id": "stale-case",
            "query_id": f"dim{index + 1}",
            "dim": index + 1,
            "stale_type": "T1",
            "query_mentions_old": True,
            "query_mentions_new": False,
            "current_evidence_recalled": True,
            "stale_evidence_exposed": False,
            "conflict_pair_covered": False,
            "premise_correction_opportunity": True,
            "premise_correction_hit": hit,
            "premise_correction_best_rank": 1 if hit else None,
            "current_before_stale": None,
            "current_best_rank": 1,
            "stale_best_rank": None,
            "retrieved_count": 4,
            "adjudicated_old_supports": 1,
            "old_supports": 2,
            "max_old_support_staleness": 0.0,
            "trace": [],
            "expected_state_slot": "location",
            "dependency_source_slot": "location" if index > 0 else "",
            "dependency_target_family": "local_context" if index > 0 else "",
        }
        for index, hit in enumerate(correction_hits)
    ]
    return {
        "name": name,
        "total": len(queries),
        "current_recall_rate": 1.0,
        "stale_exposure_rate": 0.0,
        "conflict_pair_coverage_rate": 0.0,
        "current_before_stale_rate": 0.0,
        "premise_old_mention_rate": 1.0,
        "premise_correction_opportunity_rate": 1.0,
        "premise_correction_hit_rate": sum(correction_hits) / len(correction_hits),
        "old_support_adjudication_rate": 0.5,
        "queries": queries,
    }
