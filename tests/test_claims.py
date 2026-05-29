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
) -> Path:
    path = tmp_path / f"{run_type}.json"
    payload = {
        "run_name": run_type,
        "run_type": run_type,
        "dataset": "benchmarks/example.jsonl",
        "baseline_names": baseline_names or ["semantic_only"],
        "results": results,
        "raw_outputs": raw_outputs or [],
        "notes": notes or {},
        "commit": "abc123",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
