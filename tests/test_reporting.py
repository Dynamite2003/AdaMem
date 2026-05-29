from __future__ import annotations

import json
from pathlib import Path

from adamem.reporting import claim_matrix_markdown, claim_matrix_rows, main, write_experiment_bundle, write_experiment_bundle_batch


def test_write_experiment_bundle_for_answer_generation(tmp_path: Path) -> None:
    experiment = _write_answer_generation_experiment(tmp_path)
    output_dir = tmp_path / "bundle"

    manifest = write_experiment_bundle(
        experiment,
        output_dir,
        group_fields=["question_type"],
        title="Answer Bundle",
    )

    assert manifest["record_kind"] == "answer_generation"
    assert "harness_plumbing" in manifest["supported_claims"]
    assert "answer_accuracy" in manifest["blocked_claims"]
    artifacts = manifest["artifacts"]
    assert Path(artifacts["paper_tables_markdown"]).exists()
    assert Path(artifacts["paper_tables_json"]).exists()
    assert Path(artifacts["paired_comparison_markdown"]).exists()
    assert Path(artifacts["paired_comparison_json"]).exists()
    assert Path(artifacts["claim_audit_markdown"]).exists()
    table = Path(artifacts["paper_tables_markdown"]).read_text(encoding="utf-8")
    audit = Path(artifacts["claim_audit_markdown"]).read_text(encoding="utf-8")
    assert "# Answer Bundle" in table
    assert "| A | trajectory_step_readout | 1/1 | 100.00% |" in table
    assert "`harness_plumbing`" in audit
    comparison = Path(artifacts["paired_comparison_markdown"]).read_text(encoding="utf-8")
    assert "## Overall" in comparison


def test_reporting_cli_writes_manifest_json(tmp_path: Path) -> None:
    experiment = _write_answer_generation_experiment(tmp_path)
    output_dir = tmp_path / "cli-bundle"

    main([
        str(experiment),
        "--output-dir",
        str(output_dir),
        "--group-fields",
        "question_type",
        "--json",
    ])

    manifest_path = output_dir / f"{experiment.stem}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_kind"] == "answer_generation"
    assert Path(manifest["artifacts"]["claim_audit_json"]).exists()
    assert Path(manifest["artifacts"]["paired_comparison_json"]).exists()


def test_write_experiment_bundle_supports_stale_retrieval_diagnostics(tmp_path: Path) -> None:
    experiment = _write_stale_retrieval_diagnostic_experiment(tmp_path)
    output_dir = tmp_path / "stale-bundle"

    manifest = write_experiment_bundle(experiment, output_dir)

    assert manifest["record_kind"] == "stale_retrieval_diagnostics"
    assert "table_error" not in manifest
    assert "paired_comparison_skipped" in manifest
    assert Path(manifest["artifacts"]["paper_tables_markdown"]).exists()
    assert "paired_comparison_markdown" not in manifest["artifacts"]
    table = Path(manifest["artifacts"]["paper_tables_markdown"]).read_text(encoding="utf-8")
    assert "premise correction hit" in table


def test_write_experiment_bundle_supports_longmemeval_v2_prepared_pilot(tmp_path: Path) -> None:
    experiment = _write_lme_v2_prepared_experiment(tmp_path)
    output_dir = tmp_path / "lme-v2-bundle"

    manifest = write_experiment_bundle(
        experiment,
        output_dir,
        group_fields=["question_type", "selection_group"],
    )

    assert manifest["record_kind"] == "retrieval"
    assert "longmemeval_v2_prepared_split_readiness" in manifest["supported_claims"]
    assert "answer_accuracy" in manifest["blocked_claims"]
    assert manifest["dataset_scope"]["scope"] == "public_transfer_prepared"
    assert manifest["claim_evidence"]["prepared_state_evidence"]["with_matching_state_evidence"] == 1
    assert manifest["warnings"] == []
    assert "table_error" not in manifest
    artifacts = manifest["artifacts"]
    assert Path(artifacts["claim_audit_markdown"]).exists()
    assert Path(artifacts["paper_tables_markdown"]).exists()
    assert Path(artifacts["paired_comparison_markdown"]).exists()
    audit = Path(artifacts["claim_audit_markdown"]).read_text(encoding="utf-8")
    assert "`retrieval_answer_string_support_diagnostics`" in audit


def test_write_experiment_bundle_batch(tmp_path: Path) -> None:
    experiment_a = _write_answer_generation_experiment(tmp_path, stem="generation_a")
    experiment_b = _write_answer_generation_experiment(tmp_path, stem="generation_b")
    experiment_c = _write_lme_v2_prepared_experiment(tmp_path)
    output_dir = tmp_path / "batch"

    manifest = write_experiment_bundle_batch(
        tmp_path,
        output_dir,
        group_fields=["question_type"],
    )

    assert manifest["experiment_count"] == 3
    assert manifest["experiments"] == [str(experiment_a), str(experiment_b), str(experiment_c)]
    assert Path(manifest["manifest"]).exists()
    assert Path(manifest["artifacts"]["claim_matrix_json"]).exists()
    assert Path(manifest["artifacts"]["claim_matrix_markdown"]).exists()
    assert len(manifest["bundles"]) == 3
    for bundle in manifest["bundles"]:
        assert "claim_evidence" in bundle
        assert "diagnostic_evidence" in bundle
        assert "warnings" in bundle
        assert "dataset_scope" in bundle
        assert Path(bundle["artifacts"]["paper_tables_markdown"]).exists()
    matrix = json.loads(Path(manifest["artifacts"]["claim_matrix_json"]).read_text(encoding="utf-8"))
    by_name = {Path(row["experiment"]).name: row for row in matrix}
    assert by_name["lme_v2_prepared.experiment.json"]["state_matching_questions"] == 1
    assert by_name["lme_v2_prepared.experiment.json"]["state_available_rate"] == 1.0
    assert by_name["lme_v2_prepared.experiment.json"]["readiness_gate"] == "diagnostic_ready"
    assert "answer_accuracy_blocked" in by_name["lme_v2_prepared.experiment.json"]["readiness_reasons"]


def test_reporting_cli_accepts_directory_input(tmp_path: Path) -> None:
    _write_answer_generation_experiment(tmp_path, stem="generation_cli")
    output_dir = tmp_path / "cli-batch"

    main([
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--group-fields",
        "question_type",
        "--json",
    ])

    manifest = json.loads((output_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_count"] == 1
    assert manifest["bundles"][0]["record_kind"] == "answer_generation"
    assert Path(manifest["artifacts"]["claim_matrix_markdown"]).exists()


def test_claim_matrix_helpers_flatten_manifest_evidence() -> None:
    rows = claim_matrix_rows([
        {
            "experiment": "a.experiment.json",
            "run_type": "longmemeval_v2_prepared_answer_support_pilot",
            "dataset": "dataset.jsonl",
            "record_kind": "retrieval",
            "raw_output_count": 20,
            "supported_claims": ["retrieval_diagnostics", "prepared_state_evidence_audit"],
            "blocked_claims": {"answer_accuracy": ["not generation"]},
            "warnings": [],
            "claim_evidence": {
                "prepared_state_evidence": {
                    "with_expected_state_slots": 4,
                    "with_matching_state_evidence": 3,
                    "state_available_rate": 0.75,
                },
                "retrieval_transfer": {
                    "paired_no_regression": [{"candidate": "semantic_state_readout"}],
                },
            },
        }
    ])
    markdown = claim_matrix_markdown(rows)

    assert rows == [{
        "experiment": "a.experiment.json",
        "run_type": "longmemeval_v2_prepared_answer_support_pilot",
        "dataset": "dataset.jsonl",
        "dataset_scope": "unknown",
        "dataset_claim_limited": False,
        "dataset_scope_reasons": [],
        "record_kind": "retrieval",
        "raw_output_count": 20,
        "supported_claims": ["retrieval_diagnostics", "prepared_state_evidence_audit"],
        "blocked_claims": ["answer_accuracy"],
        "warning_count": 0,
        "warnings": [],
        "state_expected_questions": 4,
        "state_matching_questions": 3,
        "state_available_rate": 0.75,
        "paired_no_regression_count": 1,
        "failure_attribution_count": 0,
        "top_failure_attribution": None,
        "top_failure_attribution_count": 0,
        "supported_claim_count": 2,
        "blocked_claim_count": 1,
        "readiness_gate": "diagnostic_ready",
        "readiness_reasons": [
            "diagnostic_or_mechanism_claim_only",
            "answer_accuracy_blocked",
        ],
    }]
    assert "diagnostic_ready" in markdown
    assert "3/4" in markdown
    assert "75.00%" in markdown
    assert "| experiment | gate | scope | run type | supported | blocked | warnings | state evidence | state rate | no-reg pairs | top attribution |" in markdown


def test_claim_matrix_marks_answer_candidate_and_attention_gates() -> None:
    rows = claim_matrix_rows([
        {
            "experiment": "answer.experiment.json",
            "run_type": "jsonl_answer_generation_benchmark",
            "dataset": "dataset.jsonl",
            "raw_output_count": 12,
            "supported_claims": ["answer_accuracy_candidate"],
            "blocked_claims": {"sota": ["no strong baselines"]},
            "warnings": [],
            "claim_evidence": {},
        },
        {
            "experiment": "bad.experiment.json",
            "run_type": "unknown",
            "dataset": "dataset.jsonl",
            "raw_output_count": 0,
            "supported_claims": ["unclassified_experiment"],
            "blocked_claims": {"answer_accuracy": ["unrecognized"]},
            "warnings": ["ground_truth_runtime_use is not explicitly forbidden"],
            "claim_evidence": {},
        },
    ])

    by_name = {Path(row["experiment"]).name: row for row in rows}
    assert by_name["answer.experiment.json"]["readiness_gate"] == "answer_candidate"
    assert by_name["answer.experiment.json"]["readiness_reasons"] == [
        "answer_accuracy_candidate_but_sota_blocked"
    ]
    assert by_name["bad.experiment.json"]["readiness_gate"] == "needs_attention"
    assert by_name["bad.experiment.json"]["readiness_reasons"] == [
        "claim_audit_warnings_present",
        "no_case_level_or_raw_records",
        "unclassified_experiment",
    ]


def test_claim_matrix_flattens_failure_attribution_evidence() -> None:
    rows = claim_matrix_rows([
        {
            "experiment": "errors.experiment.json",
            "run_type": "jsonl_retrieval_benchmark",
            "dataset": "benchmarks/example.jsonl",
            "raw_output_count": 12,
            "supported_claims": ["retrieval_diagnostics"],
            "blocked_claims": {"answer_accuracy": ["not generation"]},
            "warnings": [],
            "claim_evidence": {},
            "diagnostic_evidence": {
                "failure_attributions": {
                    "retrieval_failure": 2,
                    "state_readout_failure": 5,
                },
                "examples_by_failure_attribution": {
                    "state_readout_failure": [{"case_id": "c1"}],
                },
            },
        }
    ])
    markdown = claim_matrix_markdown(rows)

    assert rows[0]["failure_attribution_count"] == 2
    assert rows[0]["top_failure_attribution"] == "state_readout_failure"
    assert rows[0]["top_failure_attribution_count"] == 5
    assert "state_readout_failure (5)" in markdown


def test_claim_matrix_gates_claim_limited_dataset_scope() -> None:
    rows = claim_matrix_rows([
        {
            "experiment": "stale_mini.experiment.json",
            "run_type": "stale_retrieval_diagnostics",
            "dataset": "benchmarks/stale_mini.jsonl",
            "dataset_scope": {
                "scope": "mini_or_smoke_fixture",
                "claim_limited": True,
                "reasons": ["mini_smoke_or_debug_name"],
            },
            "raw_output_count": 12,
            "supported_claims": ["stale_retrieval_diagnostics"],
            "blocked_claims": {"stale_answer_accuracy": ["no answer generation"]},
            "warnings": [],
            "claim_evidence": {},
        }
    ])
    markdown = claim_matrix_markdown(rows)

    assert rows[0]["readiness_gate"] == "needs_attention"
    assert rows[0]["readiness_reasons"] == ["dataset_scope_claim_limited"]
    assert rows[0]["dataset_scope"] == "mini_or_smoke_fixture"
    assert rows[0]["dataset_claim_limited"] is True
    assert rows[0]["dataset_scope_reasons"] == ["mini_smoke_or_debug_name"]
    assert "mini_or_smoke_fixture" in markdown


def _write_answer_generation_experiment(tmp_path: Path, *, stem: str = "generation") -> Path:
    records_path = tmp_path / f"{stem}.records.jsonl"
    records_path.write_text(
        json.dumps({
            "baseline": "trajectory_step_readout",
            "case_id": "case-1",
            "query_id": "q1",
            "query": "What happened at Step 1?",
            "correct": True,
            "answer": "The agent moved left.",
            "score_raw": "CORRECT",
            "retrieved": [],
            "trace": [],
            "expected_substrings": ["left"],
            "forbidden_substrings": [],
            "metadata": {"question_type": "A"},
        }) + "\n",
        encoding="utf-8",
    )
    experiment = tmp_path / f"{stem}.experiment.json"
    experiment.write_text(
        json.dumps({
            "run_name": "generation",
            "run_type": "ama_public_answer_generation_pilot",
            "dataset": "benchmarks/ama.answer.jsonl",
            "baseline_names": ["trajectory_step_readout"],
            "results": {"trajectory_step_readout": {"correct": 1, "total": 1}},
            "raw_outputs": [],
            "notes": {
                "records_path": records_path.name,
                "ground_truth_runtime_use": "forbidden",
                "ground_truth_evaluation_use": "answer_scorer_only",
                "answer_provider": "mock",
                "scorer": "substring",
            },
            "commit": "abc123",
        }),
        encoding="utf-8",
    )
    return experiment


def _write_stale_retrieval_diagnostic_experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "stale_diagnostics.experiment.json"
    experiment.write_text(
        json.dumps({
            "run_name": "stale_diagnostics",
            "run_type": "stale_retrieval_diagnostics",
            "dataset": "benchmarks/stale_mini.jsonl",
            "baseline_names": ["semantic_state_premise_correction"],
            "diagnostics": [
                {
                    "name": "semantic_state_premise_correction",
                    "total": 1,
                    "current_recall_rate": 1.0,
                    "stale_exposure_rate": 0.0,
                    "conflict_pair_coverage_rate": 0.0,
                    "current_before_stale_rate": 0.0,
                    "premise_old_mention_rate": 1.0,
                    "premise_correction_opportunity_rate": 1.0,
                    "premise_correction_hit_rate": 1.0,
                    "old_support_adjudication_rate": 0.5,
                    "queries": [
                        {
                            "case_id": "stale-case",
                            "query_id": "dim2",
                            "dim": 2,
                            "stale_type": "T1",
                            "query_mentions_old": True,
                            "current_evidence_recalled": True,
                            "stale_evidence_exposed": False,
                            "conflict_pair_covered": False,
                            "premise_correction_opportunity": True,
                            "premise_correction_hit": True,
                            "current_before_stale": None,
                            "old_supports": 2,
                            "adjudicated_old_supports": 1,
                        }
                    ],
                }
            ],
            "raw_outputs": [],
            "notes": {
                "ground_truth_runtime_use": "forbidden",
                "answer_model_required": False,
                "judge_model_required": False,
            },
            "commit": "abc123",
        }),
        encoding="utf-8",
    )
    return experiment


def _write_lme_v2_prepared_experiment(tmp_path: Path) -> Path:
    records_path = tmp_path / "lme_v2_prepared.records.jsonl"
    state_evidence_path = tmp_path / "lme_v2_prepared.state_evidence.summary.json"
    state_evidence_path.write_text(
        json.dumps({
            "total_questions": 1,
            "with_expected_state_slots": 1,
            "with_matching_state_evidence": 1,
            "without_matching_state_evidence": 0,
            "state_available_rate": 1.0,
            "matching_state_evidence_candidate_total": 1,
            "questions_with_missing_trajectories": 0,
            "missing_trajectory_total": 0,
        }),
        encoding="utf-8",
    )
    records = []
    for baseline in ["semantic_only", "semantic_state_readout"]:
        records.append({
            "baseline": baseline,
            "case_id": "q-dynamic",
            "query_id": "q-dynamic",
            "query": "Is the staging build runner online?",
            "passed": True,
            "expected": ["online"],
            "matched": ["online"],
            "missing": [],
            "expected_evidence": [],
            "evidence_support_matched": False,
            "answer_keywords": ["online"],
            "missing_answer_keywords": [],
            "answer_keyword_recall": 1.0,
            "answer_keyword_matched": True,
            "answer_keyword_support_matched": True,
            "answer_basis": "",
            "basis_missing_answer_keywords": ["online"],
            "basis_answer_keyword_recall": 0.0,
            "basis_answer_keyword_support_matched": False,
            "retrieved": ["The staging build runner status is online."],
            "trace": [],
            "failure_modes": [],
            "state_retrieval_count": 0,
            "retrieved_state_slots": [],
            "expected_state_slots": [],
            "unexpected_state_slots": [],
            "state_slot_matched": False,
            "state_sensitive": False,
            "state_available": False,
            "state_readout_expected": False,
            "graph_retrieval_count": 0,
            "graph_evidence_hits": [],
            "graph_evidence_hit_count": 0,
            "metadata": {
                "question_type": "dynamic-environment",
                "selection_group": "dynamic-environment",
            },
        })
    records_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "lme_v2_prepared.experiment.json"
    experiment.write_text(
        json.dumps({
            "run_name": "longmemeval_v2_prepared_answer",
            "run_type": "longmemeval_v2_prepared_answer_support_pilot",
            "dataset": "results/lme/longmemeval_v2_prepared.answer.adamem.jsonl",
            "baseline_names": ["semantic_only", "semantic_state_readout"],
            "baseline_configs": {
                "semantic_state_readout": {
                    "use_state_memory": True,
                    "use_state_readout": True,
                }
            },
            "results": {},
            "raw_outputs": [],
            "notes": {
                "records_path": records_path.name,
                "ground_truth_runtime_use": "forbidden",
                "ground_truth_evaluation_use": "query_metadata_only",
                "metric_boundary": "retrieval answer-string support, not final generated answer accuracy",
                "validation_summary_path": "validation/longmemeval_v2_prepared_validation.summary.json",
                "state_evidence_summary_path": state_evidence_path.name,
            },
            "commit": "abc123",
        }),
        encoding="utf-8",
    )
    return experiment
