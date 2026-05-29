from __future__ import annotations

import json
from pathlib import Path

from adamem.reporting import main, write_experiment_bundle


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
    assert Path(artifacts["claim_audit_markdown"]).exists()
    table = Path(artifacts["paper_tables_markdown"]).read_text(encoding="utf-8")
    audit = Path(artifacts["claim_audit_markdown"]).read_text(encoding="utf-8")
    assert "# Answer Bundle" in table
    assert "| A | trajectory_step_readout | 1/1 | 100.00% |" in table
    assert "`harness_plumbing`" in audit


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


def _write_answer_generation_experiment(tmp_path: Path) -> Path:
    records_path = tmp_path / "generation.records.jsonl"
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
    experiment = tmp_path / "generation.experiment.json"
    experiment.write_text(
        json.dumps({
            "run_name": "generation",
            "run_type": "ama_public_answer_generation_pilot",
            "dataset": "benchmarks/ama.answer.jsonl",
            "baseline_names": ["trajectory_step_readout"],
            "results": {"trajectory_step_readout": {"correct": 1, "total": 1}},
            "raw_outputs": [],
            "notes": {
                "records_path": "generation.records.jsonl",
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
