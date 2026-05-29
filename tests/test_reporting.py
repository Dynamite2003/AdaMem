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
