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


def test_claim_audit_summarizes_failure_attribution_evidence(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="jsonl_retrieval_benchmark",
        baseline_names=["semantic_only", "semantic_state_readout"],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "query_metadata_only",
        },
        raw_outputs=[
            {
                "baseline": "semantic_state_readout",
                "case_id": "case-1",
                "query_id": "q1",
                "passed": False,
                "failure_modes": ["expected_support_missing"],
                "failure_attributions": [
                    "state_authority_absent_or_extraction_failure"
                ],
                "retrieved": ["old location evidence"],
            },
            {
                "baseline": "semantic_only",
                "case_id": "case-2",
                "query_id": "q2",
                "passed": False,
                "failure_modes": ["no_retrieval"],
                "failure_attributions": ["retrieval_failure"],
                "retrieved": [],
            },
        ],
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)
    attribution = audit["claim_evidence"]["failure_attributions"]

    assert "failure_attribution_error_analysis" in audit["supported_claims"]
    assert attribution["records"] == 2
    assert attribution["top_failure_attribution"] == "retrieval_failure"
    assert attribution["top_failure_attribution_count"] == 1
    assert attribution["failure_attributions"] == {
        "retrieval_failure": 1,
        "state_authority_absent_or_extraction_failure": 1,
    }
    assert (
        attribution["examples_by_failure_attribution"][
            "state_authority_absent_or_extraction_failure"
        ][0]["top_retrieved"]
        == "old location evidence"
    )
    assert "Failure Attribution Evidence" in markdown


def test_claim_audit_records_baseline_coverage(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="jsonl_retrieval_benchmark",
        baseline_names=[
            "semantic_only",
            "a_mem_evolution",
            "semantic_state_adjudication",
        ],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "query_metadata_only",
        },
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)
    coverage = audit["claim_evidence"]["baseline_coverage"]

    assert "baseline_coverage_audit" in audit["supported_claims"]
    assert coverage["complete"] is True
    assert coverage["missing_groups"] == []
    assert coverage["categories"] == {
        "mainstream_approximation": ["a_mem_evolution"],
        "raw_turn_retrieval": ["semantic_only"],
        "state_aware_ablation": ["semantic_state_adjudication"],
    }
    assert "Baseline Coverage" in markdown


def test_claim_audit_recognizes_longmemeval_v2_prepared_pilot_boundary(tmp_path: Path) -> None:
    records = tmp_path / "lme_v2.records.jsonl"
    state_evidence = tmp_path / "state_evidence.summary.json"
    state_evidence.write_text(
        json.dumps({
            "total_questions": 10,
            "with_expected_state_slots": 4,
            "with_matching_state_evidence": 3,
            "without_matching_state_evidence": 1,
            "state_available_rate": 0.75,
            "matching_state_evidence_candidate_total": 7,
            "questions_with_missing_trajectories": 0,
            "missing_trajectory_total": 0,
        }),
        encoding="utf-8",
    )
    rows = []
    for index in range(10):
        rows.append(
            _retrieval_record(
                "semantic_only",
                f"q{index}",
                evidence=index % 2 == 0,
            )
        )
        rows.append(
            _retrieval_record(
                "semantic_state_readout",
                f"q{index}",
                evidence=index % 2 == 0,
            )
        )
    records.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    experiment = _write_experiment(
        tmp_path,
        run_type="longmemeval_v2_prepared_answer_support_pilot",
        baseline_names=["semantic_only", "semantic_state_readout"],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "query_metadata_only",
            "records_path": "lme_v2.records.jsonl",
            "metric_boundary": "retrieval answer-string support, not final generated answer accuracy",
            "answer_model_required": False,
            "judge_model_required": False,
            "validation_summary_path": "validation/longmemeval_v2_prepared_validation.summary.json",
            "state_evidence_summary_path": state_evidence.name,
        },
        baseline_configs={
            "semantic_state_readout": {
                "use_state_memory": True,
                "use_state_readout": True,
            },
        },
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)

    assert "retrieval_diagnostics" in audit["supported_claims"]
    assert "longmemeval_v2_prepared_split_readiness" in audit["supported_claims"]
    assert "retrieval_answer_string_support_diagnostics" in audit["supported_claims"]
    assert "prepared_state_evidence_audit" in audit["supported_claims"]
    assert "answerability_diagnostics" not in audit["supported_claims"]
    assert "paired_retrieval_no_regression" in audit["supported_claims"]
    assert audit["claim_evidence"]["prepared_state_evidence"]["with_matching_state_evidence"] == 3
    assert audit["claim_evidence"]["prepared_state_evidence"]["state_available_rate"] == 0.75
    assert audit["blocked_claims"]["answer_accuracy"] == [
        "run_type is retrieval/answerability, not answer generation"
    ]
    assert audit["blocked_claims"]["sota"] == [
        "no final answer model and judge model evaluation"
    ]
    assert audit["warnings"] == []
    assert audit["raw_output_count"] == 20
    assert "`longmemeval_v2_prepared_split_readiness`" in markdown
    assert "Prepared State Evidence" in markdown


def test_claim_audit_warns_when_longmemeval_v2_prepared_notes_are_missing(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="longmemeval_v2_prepared_answer_support_pilot",
        baseline_names=["semantic_only", "semantic_state_readout"],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_evaluation_use": "query_metadata_only",
        },
    )

    audit = audit_experiment(experiment)

    assert "longmemeval_v2_prepared_split_readiness" in audit["supported_claims"]
    assert audit["warnings"] == [
        "LongMemEval-V2 prepared pilot metric_boundary is missing or unexpected",
        "LongMemEval-V2 prepared pilot state_evidence_summary_path is missing or unreadable",
    ]


def test_claim_audit_marks_mini_fixture_scope_as_claim_limited(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="stale_retrieval_diagnostics",
        dataset="benchmarks/stale_mini.jsonl",
        notes={"ground_truth_runtime_use": "forbidden"},
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)

    assert audit["dataset_scope"] == {
        "scope": "mini_or_smoke_fixture",
        "claim_limited": True,
        "reasons": ["mini_smoke_or_debug_name"],
    }
    assert "dataset scope is claim-limited: mini_smoke_or_debug_name" in audit["warnings"]
    assert "Dataset scope: `mini_or_smoke_fixture`" in markdown


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


def test_claim_audit_records_model_robustness_coverage(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="stale_llm_judge",
        baseline_names=["semantic_only", "a_mem_evolution", "state_readout"],
        notes={
            "ground_truth_runtime_use": "forbidden",
            "ground_truth_judge_use": "allowed",
            "answer_provider": "openai",
            "answer_model": "gpt-4o-mini",
            "judge_provider": "gemini",
            "judge_model": "gemini-2.5-pro",
        },
        raw_outputs=[
            {
                "baseline": "state_readout",
                "case_id": "c1",
                "query_id": "q1",
                "judge_correct": True,
                "answer_provider": "openai",
                "answer_model": "gpt-5-mini",
                "judge_provider": "openai",
                "judge_model": "gpt-5",
            }
        ],
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)
    coverage = audit["claim_evidence"]["model_coverage"]

    assert "model_robustness_audit" in audit["supported_claims"]
    assert coverage["complete"] is True
    assert coverage["answer_models"] == ["openai:gpt-4o-mini", "openai:gpt-5-mini"]
    assert coverage["judge_models"] == ["gemini:gemini-2.5-pro", "openai:gpt-5"]
    assert coverage["missing_requirements"] == []
    assert "Model Coverage" in markdown


def test_claim_audit_records_reproducibility_coverage(tmp_path: Path) -> None:
    experiment = tmp_path / "complete_stale.experiment.json"
    experiment.write_text(
        json.dumps({
            "schema_version": "adamem.experiment.v1",
            "run_name": "complete_stale",
            "run_type": "stale_llm_judge",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "baseline_names": ["semantic_only", "a_mem_evolution", "state_readout"],
            "baseline_configs": {
                "semantic_only": {"use_state_memory": False},
                "a_mem_evolution": {"use_memory_evolution": True},
                "state_readout": {"use_state_memory": True, "use_state_readout": True},
            },
            "baseline_provenance": {
                "semantic_only": {
                    "category": "raw_turn_retrieval",
                    "source_name": "AdaMem",
                    "source_url": "",
                    "implementation_status": "adamem_native",
                    "reproduction_note": "Project-native method or local control.",
                },
                "a_mem_evolution": {
                    "category": "mainstream_approximation",
                    "source_name": "A-MEM",
                    "source_url": "https://arxiv.org/abs/2502.12110",
                    "implementation_status": "api_free_approximation",
                    "reproduction_note": "API-free approximation; not an official reproduction.",
                },
                "state_readout": {
                    "category": "state_aware",
                    "source_name": "AdaMem",
                    "source_url": "",
                    "implementation_status": "adamem_native",
                    "reproduction_note": "Project-native method or local control.",
                },
            },
            "raw_outputs": [
                {
                    "baseline": "state_readout",
                    "case_id": "c1",
                    "query_id": "q1",
                    "judge_correct": True,
                }
            ],
            "notes": {
                "ground_truth_runtime_use": "forbidden",
                "ground_truth_judge_use": "allowed",
                "answer_provider": "openai",
                "answer_model": "gpt-4o-mini",
                "judge_provider": "gemini",
                "judge_model": "gemini-2.5-pro",
                "top_k": 8,
                "max_context_chars": 4000,
            },
            "prompts": {
                "answer_system": "answer system",
                "answer_template": "answer {context}",
                "judge_system": "judge system",
                "judge_template": "judge {answer}",
            },
            "command": ["python", "-m", "adamem.eval", "--stale", "benchmarks/stale.adamem.jsonl"],
            "commit": "abc123",
        }),
        encoding="utf-8",
    )

    audit = audit_experiment(experiment)
    markdown = claim_audit_markdown(audit)
    reproducibility = audit["claim_evidence"]["reproducibility"]

    assert "reproducibility_audit" in audit["supported_claims"]
    assert audit["baseline_provenance"]["a_mem_evolution"]["category"] == "mainstream_approximation"
    assert reproducibility["complete"] is True
    assert reproducibility["missing"] == []
    assert "baseline_provenance" in reproducibility["present"]
    assert "Reproducibility" in markdown


def test_claim_audit_flags_missing_baseline_provenance(tmp_path: Path) -> None:
    experiment = _write_experiment(
        tmp_path,
        run_type="jsonl_retrieval_benchmark",
        baseline_names=["semantic_only"],
        baseline_configs={"semantic_only": {"use_graph": False}},
        notes={
            "ground_truth_runtime_use": "forbidden",
            "benchmark_kind": "retrieval_support",
        },
        raw_outputs=[{"baseline": "semantic_only", "query_id": "q1"}],
    )

    audit = audit_experiment(experiment)
    reproducibility = audit["claim_evidence"]["reproducibility"]

    assert reproducibility["complete"] is False
    assert "baseline_provenance" in reproducibility["missing"]
    assert "reproducibility_audit" not in audit["supported_claims"]


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
    dataset: str = "benchmarks/example.jsonl",
) -> Path:
    path = tmp_path / f"{run_type}.json"
    payload = {
        "run_name": run_type,
        "run_type": run_type,
        "dataset": dataset,
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
