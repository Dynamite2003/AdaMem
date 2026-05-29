from __future__ import annotations

import json
from pathlib import Path

from adamem.claims import audit_experiment, claim_audit_markdown, main


def test_claim_audit_blocks_retrieval_as_answer_accuracy(tmp_path: Path) -> None:
    records = tmp_path / "retrieval.records.jsonl"
    records.write_text("{}\n{}\n", encoding="utf-8")
    experiment = _write_experiment(
        tmp_path,
        run_type="ama_public_answerability_pilot",
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "query_metadata_only",
            "records_path": "retrieval.records.jsonl",
        },
        results={"semantic_only": {"total": 10}},
    )

    audit = audit_experiment(experiment)

    assert "retrieval_diagnostics" in audit["supported_claims"]
    assert "answerability_diagnostics" in audit["supported_claims"]
    assert "answer_accuracy" in audit["blocked_claims"]
    assert "not answer generation" in audit["blocked_claims"]["answer_accuracy"][0]
    assert audit["raw_output_count"] == 2


def test_claim_audit_supports_paired_retrieval_no_regression(tmp_path: Path) -> None:
    records = tmp_path / "retrieval.records.jsonl"
    rows = []
    for index in range(12):
        passed = index % 3 != 0
        rows.append(
            _retrieval_record(
                "semantic_state_adjudication",
                f"q{index}",
                evidence=passed,
            )
        )
        rows.append(
            _retrieval_record(
                "semantic_state_premise_correction",
                f"q{index}",
                evidence=passed,
                premise_correction_count=0,
            )
        )
    records.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    experiment = _write_experiment(
        tmp_path,
        run_type="jsonl_retrieval_benchmark",
        baseline_names=[
            "semantic_state_adjudication",
            "semantic_state_premise_correction",
        ],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "query_metadata_only",
            "records_path": "retrieval.records.jsonl",
        },
        baseline_configs={
            "semantic_state_premise_correction": {
                "use_state_premise_correction": True,
            },
        },
        results={"semantic_state_premise_correction": {"total": 12}},
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)
    transfer = audit["claim_evidence"]["retrieval_transfer"]

    assert "paired_retrieval_no_regression" in audit["supported_claims"]
    assert "premise_correction_no_trigger_on_transfer" in audit["supported_claims"]
    assert transfer["paired_metric"] == "evidence_support_matched"
    assert transfer["paired_no_regression"] == [
        {
            "reference": "semantic_state_adjudication",
            "candidate": "semantic_state_premise_correction",
            "common_total": 12,
            "gained": 0,
            "lost": 0,
            "net_delta": 0,
        }
    ]
    assert transfer["premise_correction"]["semantic_state_premise_correction"] == {
        "records": 12,
        "correction_records": 0,
        "correction_items": 0,
        "corrected_forbidden_records": 0,
        "unresolved_forbidden_records": 0,
    }
    assert "No-regression pair" in markdown


def test_claim_audit_supports_unknown_current_trace_resolution(tmp_path: Path) -> None:
    records = tmp_path / "unknown.records.jsonl"
    rows = [
        _unknown_current_record(
            "semantic_state_adjudication",
            "q1",
            kind="state",
            corrected_forbidden=["Seattle"],
        ),
        _unknown_current_record(
            "semantic_state_premise_correction",
            "q1",
            kind="state_correction",
            corrected_forbidden=["Seattle"],
            premise_correction_count=1,
        ),
    ]
    records.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    experiment = _write_experiment(
        tmp_path,
        run_type="jsonl_retrieval_benchmark",
        baseline_names=[
            "semantic_state_adjudication",
            "semantic_state_premise_correction",
        ],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "expected_substrings_only",
            "records_path": "unknown.records.jsonl",
        },
        baseline_configs={
            "semantic_state_premise_correction": {
                "use_state_premise_correction": True,
            },
        },
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)
    transfer = audit["claim_evidence"]["retrieval_transfer"]

    assert "unknown_current_trace_resolution" in audit["supported_claims"]
    assert "premise_correction_trace_resolution" in audit["supported_claims"]
    assert transfer["unknown_current"]["semantic_state_adjudication"] == {
        "records": 1,
        "unknown_current_records": 1,
        "unknown_current_correction_records": 0,
        "corrected_forbidden_records": 1,
        "unresolved_forbidden_records": 0,
    }
    assert transfer["unknown_current"]["semantic_state_premise_correction"] == {
        "records": 1,
        "unknown_current_records": 0,
        "unknown_current_correction_records": 1,
        "corrected_forbidden_records": 1,
        "unresolved_forbidden_records": 0,
    }
    assert "Unknown-current" in markdown


def test_claim_audit_marks_mock_answer_generation_as_plumbing(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="jsonl_answer_generation_benchmark",
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "answer_scorer_only",
            "answer_provider": "mock",
            "scorer": "substring",
        },
        results={"semantic_only": {"total": 2}},
    )

    audit = audit_experiment(experiment)

    assert audit["supported_claims"] == ["harness_plumbing"]
    assert audit["blocked_claims"]["answer_accuracy"] == ["mock answer or judge provider"]
    assert audit["raw_output_count"] == 2


def test_claim_audit_marks_real_stale_judge_as_candidate_not_sota(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="stale_llm_judge",
        baseline_names=["semantic_only", "state_readout"],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_judge_use": "allowed",
            "answer_provider": "openai",
            "judge_provider": "gemini",
        },
        raw_outputs=[
            {
                "baseline": "state_readout",
                "case_id": "c1",
                "query_id": "q1",
                "judge_correct": True,
            }
        ],
    )

    audit = audit_experiment(experiment)

    assert audit["supported_claims"] == ["stale_answer_accuracy_candidate"]
    assert "stale_answer_accuracy" not in audit["blocked_claims"]
    assert "sota" in audit["blocked_claims"]
    assert "strong-baseline" in audit["blocked_claims"]["sota"][0]
    assert audit["raw_output_count"] == 1


def test_claim_audit_markdown_and_cli_json(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="stale_retrieval_diagnostics",
        notes={"ground_truth_runtime_use": "forbidden"},
    )
    output = tmp_path / "audit.json"

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)
    main([str(experiment), "--json", "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert markdown.startswith("# AdaMem Claim Audit\n")
    assert "`stale_retrieval_diagnostics`" in markdown
    assert payload["run_type"] == "stale_retrieval_diagnostics"


def _write_experiment(
    tmp_path: Path,
    *,
    run_type: str,
    baseline_names: list[str] | None = None,
    notes: dict | None = None,
    raw_outputs: list[dict] | None = None,
    results=None,
    baseline_configs: dict | None = None,
) -> Path:
    path = tmp_path / f"{run_type}.json"
    payload = {
        "run_name": run_type,
        "run_type": run_type,
        "dataset": "benchmarks/example.jsonl",
        "baseline_names": baseline_names or ["semantic_only"],
        "baseline_configs": baseline_configs or {},
        "results": results,
        "raw_outputs": raw_outputs or [],
        "notes": notes or {},
        "commit": "abc123",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _retrieval_record(
    baseline: str,
    query_id: str,
    *,
    evidence: bool,
    premise_correction_count: int | None = None,
) -> dict:
    record = {
        "baseline": baseline,
        "case_id": f"case-{query_id}",
        "query_id": query_id,
        "passed": evidence,
        "expected_evidence": ["step001"],
        "evidence_support_matched": evidence,
        "present_forbidden": [],
        "corrected_forbidden": [],
    }
    if premise_correction_count is not None:
        record["premise_correction_count"] = premise_correction_count
    return record


def _unknown_current_record(
    baseline: str,
    query_id: str,
    *,
    kind: str,
    corrected_forbidden: list[str],
    premise_correction_count: int = 0,
) -> dict:
    metadata = {
        "state_slot": "location",
        "state_value": "unknown-current",
        "state_status": "unknown_current",
        "invalidated_state_value": "Seattle",
    }
    if kind == "state_correction":
        metadata = {
            "state_slot": "location",
            "stale_value": "Seattle",
            "current_value": "unknown-current",
        }
    return {
        "baseline": baseline,
        "case_id": "unknown",
        "query_id": query_id,
        "passed": True,
        "expected_evidence": [],
        "evidence_support_matched": False,
        "present_forbidden": [],
        "corrected_forbidden": corrected_forbidden,
        "premise_correction_count": premise_correction_count,
        "trace": [
            {
                "kind": kind,
                "content": "Current user location: unknown-current.",
                "metadata": metadata,
            }
        ],
    }
