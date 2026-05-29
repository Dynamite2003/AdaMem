from __future__ import annotations

import json
from pathlib import Path

from adamem.reporting import (
    benchmark_coverage_markdown,
    benchmark_coverage_summary,
    claim_matrix_markdown,
    claim_matrix_rows,
    main,
    method_coverage_markdown,
    method_coverage_summary,
    paper_next_steps_markdown,
    paper_readiness_markdown,
    paper_readiness_summary,
    study_model_coverage_markdown,
    study_model_coverage_rows,
    write_experiment_bundle,
    write_experiment_bundle_batch,
)


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
    assert Path(artifacts["method_coverage_markdown"]).exists()
    assert Path(artifacts["method_coverage_json"]).exists()
    assert Path(artifacts["paper_next_steps_markdown"]).exists()
    assert Path(artifacts["paper_readiness_markdown"]).exists()
    assert Path(artifacts["paper_readiness_json"]).exists()
    method_coverage = json.loads(Path(artifacts["method_coverage_json"]).read_text(encoding="utf-8"))
    paper_readiness = json.loads(Path(artifacts["paper_readiness_json"]).read_text(encoding="utf-8"))
    assert method_coverage["experiment_count"] == 1
    assert method_coverage["mechanism_flags"]["trajectory_step_readout"] is True
    assert "raw_retrieval_reference" in method_coverage["missing_requirements"]
    assert paper_readiness["experiment_count"] == 1
    assert "raw_retrieval_reference" in paper_readiness["method_missing_requirements"]
    table = Path(artifacts["paper_tables_markdown"]).read_text(encoding="utf-8")
    audit = Path(artifacts["claim_audit_markdown"]).read_text(encoding="utf-8")
    method_md = Path(artifacts["method_coverage_markdown"]).read_text(encoding="utf-8")
    next_steps = Path(artifacts["paper_next_steps_markdown"]).read_text(encoding="utf-8")
    readiness_md = Path(artifacts["paper_readiness_markdown"]).read_text(encoding="utf-8")
    assert "# Answer Bundle" in table
    assert "| A | trajectory_step_readout | 1/1 | 100.00% |" in table
    assert "`harness_plumbing`" in audit
    assert "`trajectory_step_readout`: `True`" in method_md
    assert "`add_missing_baseline_categories`" in next_steps
    assert "## Method Coverage Gaps" in readiness_md
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
    assert Path(manifest["artifacts"]["method_coverage_json"]).exists()
    assert Path(manifest["artifacts"]["paper_readiness_json"]).exists()
    assert Path(manifest["artifacts"]["paper_next_steps_markdown"]).exists()
    assert manifest["method_coverage"]["mechanism_flags"]["trajectory_step_readout"] is True
    assert manifest["paper_readiness"]["experiment_count"] == 1


def test_write_experiment_bundle_writes_traceable_failure_case_studies(tmp_path: Path) -> None:
    experiment = tmp_path / "failure_examples.experiment.json"
    experiment.write_text(
        json.dumps({
            "run_name": "failure_examples",
            "run_type": "jsonl_retrieval_benchmark",
            "dataset": "benchmarks/example.jsonl",
            "baseline_names": ["semantic_state_adjudication"],
            "raw_outputs": [
                {
                    "baseline": "semantic_state_adjudication",
                    "case_id": "case-1",
                    "query_id": "q1",
                    "query": "Where do I live now?",
                    "passed": False,
                    "expected_substrings": ["Portland"],
                    "missing_expected": ["Portland"],
                    "expected_evidence": [],
                    "answer_keywords": [],
                    "missing_answer_keywords": [],
                    "answer_keyword_recall": 0.0,
                    "answer_keyword_support_matched": False,
                    "answer_basis": "",
                    "basis_missing_answer_keywords": [],
                    "basis_answer_keyword_recall": 0.0,
                    "basis_answer_keyword_support_matched": False,
                    "missing_evidence": [],
                    "evidence_support_matched": False,
                    "graph_retrieval_count": 0,
                    "graph_evidence_hits": [],
                    "graph_evidence_hit_count": 0,
                    "forbidden_substrings": [],
                    "present_forbidden": [],
                    "corrected_forbidden": [],
                    "premise_correction_count": 0,
                    "failure_modes": ["expected_support_missing"],
                    "failure_attributions": ["retrieval_failure"],
                    "metadata": {"dimension": "state_resolution", "state_slot": "location"},
                    "retrieved": ["Current user location: Boston."],
                    "trace": [
                        {
                            "content": "Current user location: Boston.",
                            "kind": "state",
                            "relation": "state",
                            "metadata": {
                                "state_slot": "location",
                                "source_observation_label": "new_location",
                            },
                        }
                    ],
                    "state_retrieval_count": 1,
                    "retrieved_state_slots": ["location"],
                    "expected_state_slots": ["location"],
                    "unexpected_state_slots": [],
                    "state_slot_matched": True,
                    "state_sensitive": True,
                    "state_available": True,
                    "state_readout_expected": True,
                    "state_memory_count": 1,
                    "active_state_count": 1,
                    "stale_state_count": 0,
                    "unknown_current_state_count": 0,
                    "state_slots": ["location"],
                    "active_state_slots": ["location"],
                    "stale_state_slots": [],
                    "unknown_current_state_slots": [],
                }
            ],
            "notes": {"ground_truth_runtime_use": "forbidden"},
            "commit": "abc123",
        }),
        encoding="utf-8",
    )

    manifest = write_experiment_bundle(experiment, tmp_path / "bundle")

    artifacts = manifest["artifacts"]
    case_json = json.loads(Path(artifacts["failure_case_studies_json"]).read_text(encoding="utf-8"))
    case_md = Path(artifacts["failure_case_studies_markdown"]).read_text(encoding="utf-8")
    example = case_json["retrieval_failure"][0]
    assert example["top_trace"]["metadata"]["source_observation_label"] == "new_location"
    assert example["trace_source_labels"] == ["new_location"]
    assert "source" in case_md


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
    assert Path(manifest["artifacts"]["paper_next_steps_markdown"]).exists()
    assert Path(manifest["artifacts"]["study_model_coverage_json"]).exists()
    assert Path(manifest["artifacts"]["study_model_coverage_markdown"]).exists()
    assert Path(manifest["artifacts"]["benchmark_coverage_json"]).exists()
    assert Path(manifest["artifacts"]["benchmark_coverage_markdown"]).exists()
    assert Path(manifest["artifacts"]["method_coverage_json"]).exists()
    assert Path(manifest["artifacts"]["method_coverage_markdown"]).exists()
    assert Path(manifest["artifacts"]["paper_readiness_json"]).exists()
    assert Path(manifest["artifacts"]["paper_readiness_markdown"]).exists()
    assert len(manifest["bundles"]) == 3
    for bundle in manifest["bundles"]:
        assert "claim_evidence" in bundle
        assert "diagnostic_evidence" in bundle
        assert "method_coverage" in bundle
        assert "paper_readiness" in bundle
        assert "warnings" in bundle
        assert "dataset_scope" in bundle
        assert Path(bundle["artifacts"]["paper_tables_markdown"]).exists()
        assert Path(bundle["artifacts"]["method_coverage_markdown"]).exists()
        assert Path(bundle["artifacts"]["paper_readiness_markdown"]).exists()
        assert Path(bundle["artifacts"]["paper_next_steps_markdown"]).exists()
    matrix = json.loads(Path(manifest["artifacts"]["claim_matrix_json"]).read_text(encoding="utf-8"))
    by_name = {Path(row["experiment"]).name: row for row in matrix}
    assert by_name["lme_v2_prepared.experiment.json"]["state_matching_questions"] == 1
    assert by_name["lme_v2_prepared.experiment.json"]["state_available_rate"] == 1.0
    assert by_name["lme_v2_prepared.experiment.json"]["readiness_gate"] == "diagnostic_ready"
    assert "answer_accuracy_blocked" in by_name["lme_v2_prepared.experiment.json"]["readiness_reasons"]
    next_steps = Path(manifest["artifacts"]["paper_next_steps_markdown"]).read_text(encoding="utf-8")
    assert "`run_end_to_end_answer_and_judge_eval`" in next_steps


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
    assert Path(manifest["artifacts"]["paper_next_steps_markdown"]).exists()
    assert Path(manifest["artifacts"]["study_model_coverage_markdown"]).exists()
    assert Path(manifest["artifacts"]["benchmark_coverage_markdown"]).exists()
    assert Path(manifest["artifacts"]["method_coverage_markdown"]).exists()
    assert Path(manifest["artifacts"]["paper_readiness_markdown"]).exists()


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
        "split_or_case_limit": None,
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
        "baseline_coverage_complete": False,
        "baseline_category_count": 0,
        "missing_baseline_groups": [],
        "baseline_reproduction_complete": False,
        "official_or_faithful_baseline_count": 0,
        "baseline_reproduction_gaps": [],
        "model_coverage_complete": False,
        "answer_model_count": 0,
        "judge_model_count": 0,
        "missing_model_requirements": [],
        "reproducibility_complete": False,
        "missing_reproducibility_items": [],
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
        "next_actions": [
            "audit_missing_state_evidence",
            "run_end_to_end_answer_and_judge_eval",
        ],
        "next_action": "audit_missing_state_evidence",
    }]
    assert "diagnostic_ready" in markdown
    assert "3/4" in markdown
    assert "75.00%" in markdown
    assert "| experiment | gate | next action | scope | run type | supported | blocked | warnings | state evidence | state rate | baseline gaps | baseline repro | model gaps | repro gaps | no-reg pairs | top attribution |" in markdown


def test_study_model_coverage_merges_comparable_experiments() -> None:
    manifests = [
        _model_manifest(
            "stale_gpt.experiment.json",
            answer_models=["openai:gpt-4o-mini"],
            judge_models=["gemini:gemini-2.5-pro"],
        ),
        _model_manifest(
            "stale_gpt5.experiment.json",
            answer_models=["openai:gpt-5-mini"],
            judge_models=["openai:gpt-5"],
        ),
        _model_manifest(
            "other_split.experiment.json",
            split_or_case_limit="max_cases=10",
            answer_models=["openai:gpt-5-mini"],
            judge_models=["openai:gpt-5"],
        ),
    ]

    rows = study_model_coverage_rows(manifests)
    markdown = study_model_coverage_markdown(rows)

    assert rows[0]["complete"] is True
    assert rows[0]["experiment_count"] == 2
    assert rows[0]["answer_models"] == ["openai:gpt-4o-mini", "openai:gpt-5-mini"]
    assert rows[0]["judge_models"] == ["gemini:gemini-2.5-pro", "openai:gpt-5"]
    assert rows[0]["missing_requirements"] == []
    assert rows[1]["complete"] is False
    assert rows[1]["missing_requirements"] == ["multiple_answer_models", "multiple_judge_models"]
    assert "stale_llm_judge" in markdown
    assert "| stale_llm_judge | benchmarks/stale.adamem.jsonl | - | 2 | 2 | 2 | - |" in markdown


def test_benchmark_coverage_summary_tracks_stale_and_transfer_scope() -> None:
    summary = benchmark_coverage_summary([
        {
            "experiment": "stale.experiment.json",
            "run_type": "stale_llm_judge",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "dataset_scope": {"scope": "benchmark_like", "claim_limited": False},
        },
        {
            "experiment": "ama.experiment.json",
            "run_type": "ama_public_answerability_pilot",
            "dataset": "benchmarks/ama_bench.adamem.jsonl",
            "dataset_scope": {"scope": "benchmark_like", "claim_limited": False},
        },
    ])
    markdown = benchmark_coverage_markdown(summary)

    assert summary["complete"] is True
    assert summary["benchmark_families"] == {"ama": 1, "stale": 1}
    assert summary["primary_stale_experiment_count"] == 1
    assert summary["transfer_experiment_count"] == 1
    assert summary["public_or_full_experiment_count"] == 2
    assert summary["missing_requirements"] == []
    assert "Complete: `True`" in markdown


def test_method_coverage_summary_tracks_paper_method_groups() -> None:
    summary = method_coverage_summary([
        {
            "experiment": "stale.experiment.json",
            "baselines": [
                "semantic_only",
                "a_mem_evolution",
                "state_readout",
                "semantic_state_premise_correction",
            ],
        },
        {
            "experiment": "unknown.experiment.json",
            "baselines": ["custom_memory"],
        },
    ])
    markdown = method_coverage_markdown(summary)

    assert summary["experiment_count"] == 2
    assert summary["known_baseline_count"] == 4
    assert summary["unknown_baselines"] == ["custom_memory"]
    assert summary["required_groups"] == {
        "raw_retrieval_reference": True,
        "mainstream_memory_approximation": True,
        "proposed_state_aware_method": True,
        "mechanism_ablation": True,
    }
    assert summary["missing_requirements"] == ["known_baseline_names_only"]
    assert summary["baseline_provenance"]["a_mem_evolution"]["source_name"] == "A-MEM"
    assert summary["baseline_provenance"]["a_mem_evolution"]["category"] == (
        "mainstream_approximation"
    )
    assert summary["baseline_provenance"]["a_mem_evolution"]["implementation_status"] == (
        "api_free_approximation"
    )
    assert summary["baseline_provenance"]["a_mem_evolution"]["reproduction_target_url"] == (
        "https://github.com/WujiangXu/A-mem"
    )
    assert summary["reproduction_status_counts"]["api_free_approximation"] == 1
    assert summary["mainstream_api_free_approximations"] == ["a_mem_evolution"]
    assert summary["official_or_faithful_mainstream_reproductions"] == []
    assert summary["sota_baseline_reproduction_ready"] is False
    assert summary["baseline_reproduction_gaps"] == [
        "official_or_faithful_mainstream_reproduction"
    ]
    assert summary["reproduction_target_count"] == 1
    assert summary["baseline_reproduction_plan"] == [
        {
            "baseline": "a_mem_evolution",
            "source_name": "A-MEM",
            "source_url": "https://arxiv.org/abs/2502.12110",
            "implementation_status": "api_free_approximation",
            "status": "needs_official_or_faithful_run",
            "reproduction_target_name": "A-MEM reproduction code",
            "reproduction_target_url": "https://github.com/WujiangXu/A-mem",
            "reproduction_target_note": (
                "Use the paper reproduction repository for official/faithful LoCoMo-style runs."
            ),
            "next_action": (
                "Run or wrap the target implementation on the same split and record provenance."
            ),
        }
    ]
    assert summary["mechanism_flags"]["state_readout"] is True
    assert summary["mechanism_flags"]["premise_correction"] is True
    assert summary["mechanism_flags"]["llm_state_extractor"] is False
    assert "custom_memory" in markdown
    assert "`proposed_state_aware_method`: `True`" in markdown
    assert "SOTA baseline reproduction ready: `False`" in markdown
    assert "`official_or_faithful_mainstream_reproduction`" in markdown
    assert "[A-MEM](https://arxiv.org/abs/2502.12110)" in markdown
    assert "[A-MEM reproduction code](https://github.com/WujiangXu/A-mem)" in markdown


def test_method_coverage_uses_artifact_baseline_provenance_over_registry() -> None:
    summary = method_coverage_summary([
        {
            "experiment": "official_amem.experiment.json",
            "baselines": ["semantic_only", "a_mem_evolution", "state_readout"],
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
                    "implementation_status": "official_reproduction",
                    "reproduction_note": "Official A-MEM code path recorded by this experiment.",
                    "reproduction_target_name": "A-MEM reproduction code",
                    "reproduction_target_url": "https://github.com/WujiangXu/A-mem",
                    "reproduction_target_note": "Official target used.",
                },
                "state_readout": {
                    "category": "state_aware",
                    "source_name": "AdaMem",
                    "source_url": "",
                    "implementation_status": "adamem_native",
                    "reproduction_note": "Project-native method or local control.",
                },
            },
        }
    ])

    assert summary["baseline_provenance"]["a_mem_evolution"]["implementation_status"] == (
        "official_reproduction"
    )
    assert summary["sota_baseline_reproduction_ready"] is True
    assert summary["official_or_faithful_mainstream_reproductions"] == ["a_mem_evolution"]
    assert summary["mainstream_api_free_approximations"] == []
    assert summary["baseline_reproduction_gaps"] == []
    assert summary["baseline_reproduction_plan"][0]["status"] == "ready"
    assert summary["baseline_reproduction_plan"][0]["reproduction_target_url"] == (
        "https://github.com/WujiangXu/A-mem"
    )


def test_paper_readiness_summary_marks_answer_candidate_with_study_coverage() -> None:
    claim_rows = claim_matrix_rows([
        {
            "experiment": "answer.experiment.json",
            "run_type": "stale_llm_judge",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "raw_output_count": 12,
            "supported_claims": ["stale_answer_accuracy_candidate"],
            "blocked_claims": {"sota": ["no strong baseline reproduction"]},
            "warnings": [],
            "claim_evidence": {},
        }
    ])
    study_rows = study_model_coverage_rows([
        _model_manifest(
            "stale_gpt.experiment.json",
            answer_models=["openai:gpt-4o-mini"],
            judge_models=["gemini:gemini-2.5-pro"],
        ),
        _model_manifest(
            "stale_gpt5.experiment.json",
            answer_models=["openai:gpt-5-mini"],
            judge_models=["openai:gpt-5"],
        ),
    ])

    benchmark_coverage = benchmark_coverage_summary([
        {
            "experiment": "stale.experiment.json",
            "run_type": "stale_llm_judge",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "dataset_scope": {"scope": "benchmark_like", "claim_limited": False},
        },
        {
            "experiment": "ama.experiment.json",
            "run_type": "ama_public_answerability_pilot",
            "dataset": "benchmarks/ama_bench.adamem.jsonl",
            "dataset_scope": {"scope": "benchmark_like", "claim_limited": False},
        },
    ])
    method_coverage = method_coverage_summary([
        {
            "experiment": "stale.experiment.json",
            "baselines": [
                "semantic_only",
                "a_mem_evolution",
                "state_readout",
                "semantic_state_premise_correction",
            ],
        }
    ])

    summary = paper_readiness_summary(
        claim_rows,
        study_rows,
        benchmark_coverage=benchmark_coverage,
        method_coverage=method_coverage,
    )
    markdown = paper_readiness_markdown(summary)

    assert summary["status"] == "answer_candidate_with_model_coverage"
    assert summary["paper_claim_ready"] is False
    assert summary["paper_claim_blockers"] == [
        "sota_candidate_without_sota_gate",
        "named_mechanism_ablation_coverage",
        "official_or_faithful_baseline_reproduction",
    ]
    assert summary["gate_counts"] == {"answer_candidate": 1}
    assert summary["complete_study_model_group_count"] == 1
    assert summary["benchmark_coverage_complete"] is True
    assert summary["method_coverage_complete"] is True
    assert summary["sota_baseline_reproduction_ready"] is False
    assert summary["baseline_reproduction_gaps"] == [
        "official_or_faithful_mainstream_reproduction"
    ]
    assert summary["mainstream_api_free_approximations"] == ["a_mem_evolution"]
    assert summary["method_categories"] == {
        "mainstream_approximation": 1,
        "raw_turn_retrieval": 1,
        "state_aware": 1,
        "state_aware_ablation": 1,
    }
    assert summary["action_counts"]["add_official_or_faithful_baseline_reproduction"] == 1
    assert summary["action_counts"]["add_named_mechanism_ablations"] == 1
    assert summary["action_counts"]["add_strong_baselines_and_judge_robustness"] == 1
    assert {
        "action": "add_official_or_faithful_baseline_reproduction",
        "count": 1,
    } in summary["top_next_actions"]
    assert "answer_candidate_with_model_coverage" in markdown
    assert "Paper claim ready: `False`" in markdown
    assert "## Paper Claim Blockers" in markdown
    assert "Method coverage complete: `True`" in markdown
    assert "SOTA baseline reproduction ready: `False`" in markdown
    assert "## Baseline Reproduction Gaps" in markdown


def test_paper_readiness_marks_paper_claim_ready_only_after_full_gates() -> None:
    claim_rows = claim_matrix_rows([
        {
            "experiment": "sota.experiment.json",
            "run_type": "stale_llm_judge",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "raw_output_count": 12,
            "supported_claims": ["stale_answer_accuracy_candidate"],
            "blocked_claims": {},
            "warnings": [],
            "claim_evidence": {},
        }
    ])
    study_rows = study_model_coverage_rows([
        _model_manifest(
            "stale_complete_a.experiment.json",
            answer_models=["openai:gpt-4o-mini"],
            judge_models=["gemini:gemini-2.5-pro"],
        ),
        _model_manifest(
            "stale_complete_b.experiment.json",
            answer_models=["openai:gpt-5-mini"],
            judge_models=["openai:gpt-5"],
        ),
    ])
    benchmark_coverage = benchmark_coverage_summary([
        {
            "experiment": "stale.experiment.json",
            "run_type": "stale_llm_judge",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "dataset_scope": {"scope": "benchmark_like", "claim_limited": False},
        },
        {
            "experiment": "ama.experiment.json",
            "run_type": "ama_public_answerability_pilot",
            "dataset": "benchmarks/ama_bench.adamem.jsonl",
            "dataset_scope": {"scope": "benchmark_like", "claim_limited": False},
        },
    ])
    method_coverage = method_coverage_summary([
        {
            "experiment": "sota.experiment.json",
            "baselines": [
                "semantic_only",
                "a_mem_evolution",
                "state_readout",
                "semantic_state_propagation_adjudication",
                "semantic_llm_state_premise_correction",
                "trajectory_step_readout",
            ],
            "baseline_provenance": {
                "a_mem_evolution": {
                    "category": "mainstream_approximation",
                    "source_name": "A-MEM",
                    "source_url": "https://arxiv.org/abs/2502.12110",
                    "implementation_status": "official_reproduction",
                    "reproduction_note": "Official A-MEM code path recorded by this experiment.",
                    "reproduction_target_name": "A-MEM reproduction code",
                    "reproduction_target_url": "https://github.com/WujiangXu/A-mem",
                    "reproduction_target_note": "Official target used.",
                },
            },
        }
    ])

    summary = paper_readiness_summary(
        claim_rows,
        study_rows,
        benchmark_coverage=benchmark_coverage,
        method_coverage=method_coverage,
    )

    assert summary["status"] == "sota_candidate_with_model_coverage"
    assert summary["paper_claim_ready"] is True
    assert summary["paper_claim_blockers"] == []


def test_paper_next_steps_markdown_groups_actions() -> None:
    rows = claim_matrix_rows([
        {
            "experiment": "stale_diag.experiment.json",
            "run_type": "stale_retrieval_diagnostics",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "raw_output_count": 12,
            "supported_claims": ["stale_retrieval_diagnostics"],
            "blocked_claims": {
                "stale_answer_accuracy": ["no answer generation"],
                "sota": ["retrieval diagnostics cannot establish SOTA"],
            },
            "warnings": [],
            "claim_evidence": {},
            "diagnostic_evidence": {
                "failure_attributions": {"ranking_failure": 3},
            },
        }
    ])
    markdown = paper_next_steps_markdown(rows)

    assert rows[0]["next_actions"] == [
        "inspect_representative_failure_attributions",
        "run_end_to_end_answer_and_judge_eval",
        "defer_sota_until_answer_eval_and_strong_baselines",
    ]
    assert "`inspect_representative_failure_attributions`: `1`" in markdown
    assert "stale_diag.experiment.json" in markdown


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
            "claim_evidence": {
                "model_coverage": {
                    "complete": False,
                    "answer_model_count": 1,
                    "judge_model_count": 1,
                    "missing_requirements": ["multiple_answer_models"],
                },
            },
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
        {
            "experiment": "failure_analysis.experiment.json",
            "run_type": "jsonl_retrieval_benchmark",
            "dataset": "dataset.jsonl",
            "raw_output_count": 5,
            "supported_claims": ["failure_attribution_error_analysis"],
            "blocked_claims": {"answer_accuracy": ["not generation"]},
            "warnings": [],
            "claim_evidence": {
                "baseline_coverage": {
                    "complete": False,
                    "category_count": 1,
                    "missing_groups": ["mainstream_memory_approximation"],
                },
            },
        },
        {
            "experiment": "repro_gap.experiment.json",
            "run_type": "stale_llm_judge",
            "dataset": "benchmarks/stale.adamem.jsonl",
            "raw_output_count": 3,
            "supported_claims": ["stale_answer_accuracy_candidate"],
            "blocked_claims": {"sota": ["no robustness"]},
            "warnings": [],
            "claim_evidence": {
                "reproducibility": {
                    "complete": False,
                    "missing": ["command", "answer_prompt"],
                },
            },
        },
    ])

    by_name = {Path(row["experiment"]).name: row for row in rows}
    assert by_name["answer.experiment.json"]["readiness_gate"] == "answer_candidate"
    assert by_name["answer.experiment.json"]["readiness_reasons"] == [
        "answer_accuracy_candidate_but_sota_blocked"
    ]
    assert by_name["answer.experiment.json"]["next_action"] == (
        "add_model_or_judge_robustness_runs"
    )
    assert "add_strong_baselines_and_judge_robustness" in (
        by_name["answer.experiment.json"]["next_actions"]
    )
    assert by_name["bad.experiment.json"]["readiness_gate"] == "needs_attention"
    assert by_name["bad.experiment.json"]["readiness_reasons"] == [
        "claim_audit_warnings_present",
        "no_case_level_or_raw_records",
        "unclassified_experiment",
    ]
    assert by_name["bad.experiment.json"]["next_actions"] == [
        "fix_claim_audit_warnings",
        "export_case_level_or_raw_records",
        "classify_experiment_run_type",
    ]
    assert by_name["failure_analysis.experiment.json"]["readiness_gate"] == "diagnostic_ready"
    assert "add_missing_baseline_categories" in (
        by_name["failure_analysis.experiment.json"]["next_actions"]
    )
    assert "complete_reproducibility_packet" in (
        by_name["repro_gap.experiment.json"]["next_actions"]
    )


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


def _model_manifest(
    experiment: str,
    *,
    answer_models: list[str],
    judge_models: list[str],
    split_or_case_limit: str | None = None,
) -> dict:
    return {
        "experiment": experiment,
        "run_type": "stale_llm_judge",
        "dataset": "benchmarks/stale.adamem.jsonl",
        "split_or_case_limit": split_or_case_limit,
        "baselines": ["semantic_only", "a_mem_evolution", "state_readout"],
        "claim_evidence": {
            "model_coverage": {
                "answer_models": answer_models,
                "judge_models": judge_models,
            }
        },
    }
