from __future__ import annotations

import json
from pathlib import Path

from adamem.reporting import main, write_experiment_bundle, write_experiment_bundle_batch


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
    output_dir = tmp_path / "batch"

    manifest = write_experiment_bundle_batch(
        tmp_path,
        output_dir,
        group_fields=["question_type"],
    )

    assert manifest["experiment_count"] == 2
    assert manifest["experiments"] == [str(experiment_a), str(experiment_b)]
    assert Path(manifest["manifest"]).exists()
    assert len(manifest["bundles"]) == 2
    for bundle in manifest["bundles"]:
        assert bundle["record_kind"] == "answer_generation"
        assert Path(bundle["artifacts"]["paper_tables_markdown"]).exists()


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
            },
            "commit": "abc123",
        }),
        encoding="utf-8",
    )
    return experiment
