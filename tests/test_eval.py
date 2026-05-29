from __future__ import annotations

import json
from pathlib import Path

from adamem.baselines import baseline_registry
from adamem.bench import (
    benchmark_case_records,
    benchmark_failure_report,
    benchmark_failure_summary,
    load_jsonl_cases,
    MemoryQACase,
    ObservationSpec,
    QuerySpec,
    run_benchmark,
)
from adamem.config import AdaMemConfig
from adamem.convert import (
    convert_ama_file,
    convert_locomo_file,
    convert_longmemeval_file,
    convert_longmemeval_v2_file,
    load_question_ids,
    load_state_audit_labels,
    summarize_longmemeval_state_audit_records,
)
from adamem.eval import _state_extractor_runtime, run_synthetic_benchmark
from adamem.experiments import experiment_record, write_experiment_record


def test_synthetic_benchmark_shows_full_beats_semantic_only() -> None:
    results = {result.name: result for result in run_synthetic_benchmark()}

    assert results["full"].accuracy > results["semantic_only"].accuracy
    assert results["full"].passed == results["full"].total
    assert results["semantic_only"].passed < results["semantic_only"].total


def test_synthetic_benchmark_exposes_case_traces() -> None:
    result = run_synthetic_benchmark()[0]
    case = result.cases[0]

    assert case.trace
    assert "score" in case.trace[0]
    assert "contributions" in case.trace[0]


def test_jsonl_benchmark_adapter_runs_fixture() -> None:
    cases = load_jsonl_cases(Path("benchmarks/tiny_memory_qa.jsonl"))
    results = {result.name: result for result in run_benchmark(cases)}

    assert results["full"].passed == results["full"].total
    assert results["semantic_only"].passed < results["full"].passed


def test_dynamic_state_transfer_fixture_favors_state_readout() -> None:
    cases = load_jsonl_cases(Path("benchmarks/dynamic_state_transfer.jsonl"))
    results = {result.name: result for result in run_benchmark(cases)}

    assert results["semantic_only"].passed < results["state_readout"].passed
    assert results["state_readout"].passed == results["state_readout"].total
    assert results["state_propagation"].passed == results["state_propagation"].total


def test_locomo_converter_emits_adamem_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "locomo.jsonl"
    count = convert_locomo_file("benchmarks/locomo_mini.json", output)

    cases = load_jsonl_cases(output)
    results = {result.name: result for result in run_benchmark(cases)}

    assert count == 1
    assert len(cases[0].observations) == 5
    assert len(cases[0].queries) == 2
    assert cases[0].queries[0].expected_substrings == ["D1:1"]
    assert results["full"].passed == results["full"].total


def test_longmemeval_converter_emits_adamem_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval.json"
    source.write_text(
        """
[
  {
    "question_id": "lme_dynamic_1",
    "question_type": "knowledge-update",
    "question": "What is the migration status?",
    "answer": "resolved",
    "question_date": "2026-05-01T00:00:00+00:00",
    "haystack_session_ids": ["s1", "s2"],
    "haystack_dates": ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"],
    "haystack_sessions": [
      [{"role": "user", "content": "The checkout migration status is blocked by missing approval.", "has_answer": true}],
      [{"role": "user", "content": "Marked checkout migration as resolved.", "has_answer": true}]
    ],
    "answer_session_ids": ["s2"]
  }
]
""".strip()
    )
    output = tmp_path / "longmemeval.adamem.jsonl"

    count = convert_longmemeval_file(source, output, expected="evidence", top_k=3)
    cases = load_jsonl_cases(output)
    results = {result.name: result for result in run_benchmark(cases)}

    assert count == 1
    assert cases[0].id == "lme_dynamic_1"
    assert len(cases[0].observations) == 2
    assert cases[0].queries[0].expected_substrings == ["s2"]
    assert cases[0].queries[0].metadata["answer"] == "resolved"
    assert results["state_readout"].queries[0].metadata["question_type"] == "knowledge-update"
    assert "state_slot" not in cases[0].queries[0].metadata
    for observation in cases[0].observations:
        assert "answer_session_ids" not in observation.metadata
        assert "has_answer" not in observation.metadata
    assert results["state_readout"].passed == results["state_readout"].total


def test_longmemeval_converter_can_infer_state_slots_for_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval_state_slots.json"
    source.write_text(
        """
[
  {
    "question_id": "lme_dynamic_1",
    "question_type": "knowledge-update",
    "question": "What is the migration status?",
    "answer": "resolved",
    "question_date": "2026-05-01T00:00:00+00:00",
    "haystack_session_ids": ["s1", "s2"],
    "haystack_dates": ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"],
    "haystack_sessions": [
      [{"role": "user", "content": "The checkout migration status is blocked by missing approval.", "has_answer": true}],
      [{"role": "user", "content": "Marked checkout migration as resolved.", "has_answer": true}]
    ],
    "answer_session_ids": ["s2"]
  }
]
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "longmemeval.state.adamem.jsonl"

    count = convert_longmemeval_file(
        source,
        output,
        expected="evidence",
        top_k=3,
        infer_state_slots=True,
    )
    cases = load_jsonl_cases(output)
    results = run_benchmark(cases, {
        "semantic_only": baseline_registry()["semantic_only"].config,
        "state_readout": baseline_registry()["state_readout"].config,
    })
    summary = benchmark_failure_summary(benchmark_case_records(results))

    assert count == 1
    assert cases[0].queries[0].metadata["state_slot"] == "task.*.status"
    assert cases[0].queries[0].metadata["state_slot_source"] == "query_text_router"
    for observation in cases[0].observations:
        assert "state_slot" not in observation.metadata
        assert "has_answer" not in observation.metadata
    assert summary["paper_metrics"]["semantic_only"]["state_readout_missing_rate"] == 1.0
    assert summary["paper_metrics"]["state_readout"]["state_slot_match_rate"] == 1.0


def test_longmemeval_converter_exports_and_applies_manual_state_audit(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval_audit.json"
    samples = [
        {
            "question_id": "state_q",
            "question_type": "knowledge-update",
            "question": "What is the migration status?",
            "answer": "resolved",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2026-01-01", "2026-02-01"],
            "haystack_sessions": [
                [{"role": "user", "content": "The checkout migration status is blocked.", "has_answer": True}],
                [{"role": "user", "content": "Marked checkout migration as resolved.", "has_answer": True}],
            ],
            "answer_session_ids": ["s2"],
        },
        {
            "question_id": "ordinary_q",
            "question_type": "single-session-user",
            "question": "What play did I attend at the local community theater?",
            "answer": "Hamlet",
            "haystack_session_ids": ["s3"],
            "haystack_dates": ["2026-03-01"],
            "haystack_sessions": [[{"role": "user", "content": "I attended Hamlet."}]],
            "answer_session_ids": ["s3"],
        },
    ]
    source.write_text(json.dumps(samples), encoding="utf-8")
    output = tmp_path / "longmemeval.audit.adamem.jsonl"
    audit_candidates = tmp_path / "state_audit_candidates.jsonl"
    audit_summary = tmp_path / "state_audit_summary.json"

    count = convert_longmemeval_file(
        source,
        output,
        expected="evidence",
        state_audit_output=audit_candidates,
        state_audit_summary_output=audit_summary,
    )
    candidate_records = [
        json.loads(line)
        for line in audit_candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert count == 2
    assert [record["question_id"] for record in candidate_records] == ["state_q"]
    assert candidate_records[0]["is_state_sensitive"] is None
    assert candidate_records[0]["state_slot"] == "task.*.status"
    assert [
        (candidate["state_slot"], candidate["state_value"])
        for candidate in candidate_records[0]["state_evidence_candidates"]
    ] == [
        ("task.checkout_migration.status", "blocked"),
        ("task.checkout_migration.status", "resolved"),
    ]
    assert all("answer_session_ids" not in candidate for candidate in candidate_records[0]["state_evidence_candidates"])
    summary = json.loads(audit_summary.read_text(encoding="utf-8"))
    assert summary["total_candidates"] == 1
    assert summary["with_state_evidence"] == 1
    assert summary["by_state_slot"]["task.*.status"]["state_evidence_candidate_total"] == 2
    assert summary["by_question_type"]["knowledge-update"]["total_candidates"] == 1

    audit_labels = tmp_path / "state_audit_labels.jsonl"
    audit_labels.write_text(
        "\n".join([
            json.dumps({
                "question_id": "state_q",
                "is_state_sensitive": True,
                "state_slot": "task.*.status",
                "state_available": True,
                "state_audit_id": "audit-001",
            }),
            json.dumps({
                "question_id": "ordinary_q",
                "is_state_sensitive": False,
                "state_slot": "location",
            }),
            json.dumps({
                "question_id": "string_false_q",
                "is_state_sensitive": True,
                "state_slot": "location",
                "state_available": "false",
            }),
        ]),
        encoding="utf-8",
    )
    audited_output = tmp_path / "longmemeval.audited.adamem.jsonl"
    labels = load_state_audit_labels(audit_labels)

    convert_longmemeval_file(
        source,
        audited_output,
        expected="evidence",
        state_audit_input=audit_labels,
    )
    cases = load_jsonl_cases(audited_output)
    by_id = {case.id: case for case in cases}

    assert by_id["state_q"].queries[0].metadata["state_slot"] == "task.*.status"
    assert by_id["state_q"].queries[0].metadata["state_slot_source"] == "manual_state_audit"
    assert by_id["state_q"].queries[0].metadata["state_available"] is True
    assert by_id["state_q"].queries[0].metadata["state_audit_id"] == "audit-001"
    assert "state_slot" not in by_id["ordinary_q"].queries[0].metadata
    assert labels["string_false_q"]["state_available"] is False
    for case in cases:
        for observation in case.observations:
            assert "state_slot" not in observation.metadata
            assert "has_answer" not in observation.metadata


def test_longmemeval_state_audit_summary_counts_evidence_coverage() -> None:
    summary = summarize_longmemeval_state_audit_records([
        {
            "question_id": "q1",
            "question_type": "knowledge-update",
            "inferred_state_slots": ["task.*.status"],
            "state_evidence_candidates": [{"state_slot": "task.checkout.status"}],
        },
        {
            "question_id": "q2",
            "question_type": "single-session-preference",
            "inferred_state_slots": ["location"],
            "state_evidence_candidates": [],
        },
    ])

    assert summary["total_candidates"] == 2
    assert summary["with_state_evidence"] == 1
    assert summary["without_state_evidence"] == 1
    assert summary["state_evidence_candidate_total"] == 1
    assert summary["by_state_slot"]["task.*.status"]["with_state_evidence"] == 1
    assert summary["by_state_slot"]["location"]["without_state_evidence"] == 1
    assert summary["by_question_type"]["knowledge-update"]["total_candidates"] == 1


def test_longmemeval_converter_can_sample_by_question_type(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval_types.json"
    samples = []
    for index, question_type in enumerate([
        "knowledge-update",
        "knowledge-update",
        "multi-session",
        "multi-session",
        "temporal-reasoning",
    ]):
        samples.append({
            "question_id": f"q{index}",
            "question_type": question_type,
            "question": f"Question {index}?",
            "answer": f"answer-{index}",
            "haystack_session_ids": [f"s{index}"],
            "haystack_dates": ["2026-05-01"],
            "haystack_sessions": [[{"role": "user", "content": f"answer-{index}"}]],
            "answer_session_ids": [f"s{index}"],
        })
    source.write_text(json.dumps(samples), encoding="utf-8")
    output = tmp_path / "sampled.jsonl"

    count = convert_longmemeval_file(
        source,
        output,
        question_types=["knowledge-update", "multi-session"],
        limit_per_type=1,
    )
    cases = load_jsonl_cases(output)

    assert count == 2
    assert [case.queries[0].metadata["question_type"] for case in cases] == [
        "knowledge-update",
        "multi-session",
    ]


def test_longmemeval_v2_converter_emits_haystack_trajectory_records(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join([
            json.dumps({
                "id": "q_dynamic",
                "domain": "enterprise",
                "environment": "workarena",
                "question_type": "dynamic-environment",
                "question": "Is the staging build runner offline?",
                "image": None,
                "answer": "No, the staging build runner is online.",
                "eval_function": "norm_phrase_set_match",
            }),
            json.dumps({
                "id": "q_static",
                "domain": "web",
                "environment": "shopping",
                "question_type": "static-environment",
                "question": "Which button is visible?",
                "answer": "Checkout",
                "eval_function": "norm_phrase_set_match",
            }),
        ]),
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(
        "\n".join([
            json.dumps({
                "id": "traj_current",
                "domain": "enterprise",
                "environment": "workarena",
                "goal": "Inspect the CI runner status.",
                "outcome": "success",
                "states": [
                    {
                        "state_index": 0,
                        "step": 3,
                        "url": "https://service.local/runners",
                        "action": "open runner status panel",
                        "thought": "Check the latest environment state.",
                        "accessibility_tree": "Runner staging-build status: online",
                        "screenshot": "screenshots/traj_current/3.png",
                    }
                ],
            }),
            json.dumps({
                "id": "traj_old",
                "domain": "enterprise",
                "environment": "workarena",
                "goal": "Open stale incident page.",
                "outcome": "failure",
                "states": [
                    {
                        "state_index": 1,
                        "step": 8,
                        "url": "https://service.local/incidents",
                        "action": "open old incident",
                        "accessibility_tree": "Old incident says staging runner offline",
                    }
                ],
            }),
            json.dumps({
                "id": "traj_unused",
                "domain": "web",
                "environment": "shopping",
                "goal": "Distractor.",
                "states": [{"state_index": 0, "accessibility_tree": "Checkout"}],
            }),
        ]),
        encoding="utf-8",
    )
    haystack = tmp_path / "lme_v2_small.json"
    haystack.write_text(
        json.dumps({
            "q_dynamic": ["traj_current", "traj_old"],
            "q_static": ["traj_unused"],
        }),
        encoding="utf-8",
    )
    question_ids = tmp_path / "split.records.jsonl"
    question_ids.write_text(json.dumps({"id": "q_dynamic", "split": "transfer"}) + "\n", encoding="utf-8")
    output = tmp_path / "longmemeval_v2.adamem.jsonl"

    count = convert_longmemeval_v2_file(
        questions,
        trajectories,
        haystack,
        output,
        expected="answer",
        top_k=4,
        question_types=["dynamic-environment"],
        question_ids=load_question_ids(question_ids),
    )
    cases = load_jsonl_cases(output)
    case = cases[0]

    assert count == 1
    assert case.id == "q_dynamic"
    assert [observation.label for observation in case.observations] == [
        "traj_current.s0000",
        "traj_old.s0001",
    ]
    assert "Runner staging-build status: online" in case.observations[0].content
    assert case.observations[0].metadata["benchmark"] == "longmemeval_v2"
    assert case.observations[0].metadata["trajectory_id"] == "traj_current"
    assert "answer" not in case.observations[0].metadata
    assert "eval_function" not in case.observations[0].metadata
    assert case.queries[0].expected_substrings == ["No, the staging build runner is online."]
    assert case.queries[0].metadata["benchmark"] == "longmemeval_v2"
    assert case.queries[0].metadata["haystack_trajectory_ids"] == ["traj_current", "traj_old"]
    assert case.queries[0].metadata["state_slot"] == "runtime.*.status"
    assert case.queries[0].metadata["state_slot_source"] == "query_text_router"


def test_longmemeval_v2_converter_limits_haystack_and_marks_missing_trajectories(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "id": "q_proc_abs",
            "domain": "enterprise",
            "environment": "workarena",
            "question_type": "procedure-abs",
            "question": "In our company workflow, what module should I use?",
            "answer": "The workflow does not use a module for that task.",
            "eval_function": "llm_abstention_checker",
        }) + "\n",
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(
        json.dumps({
            "id": "traj_present",
            "domain": "enterprise",
            "environment": "workarena",
            "goal": "Review workflow.",
            "states": [{"state_index": 0, "accessibility_tree": "Workflow uses reports."}],
        }) + "\n",
        encoding="utf-8",
    )
    haystack = tmp_path / "lme_v2_small.json"
    haystack.write_text(json.dumps({"q_proc_abs": ["traj_present", "traj_missing"]}), encoding="utf-8")
    output = tmp_path / "limited.adamem.jsonl"

    count = convert_longmemeval_v2_file(
        questions,
        trajectories,
        haystack,
        output,
        expected="evidence",
        max_trajectories_per_question=2,
        infer_state_slots=False,
    )
    cases = load_jsonl_cases(output)
    case = cases[0]

    assert count == 1
    assert len(case.observations) == 1
    assert case.queries[0].expected_substrings == ["traj_present", "traj_missing"]
    assert case.queries[0].metadata["missing_trajectory_ids"] == ["traj_missing"]
    assert case.queries[0].metadata["abstention"] is True
    assert "state_slot" not in case.queries[0].metadata


def test_ama_converter_preserves_action_observation_causality(tmp_path: Path) -> None:
    source = tmp_path / "ama.jsonl"
    source.write_text(
        json.dumps({
            "episode_id": "ama_checkout_1",
            "domain": "web_agent",
            "task_type": "incident_debugging",
            "trajectory": [
                {
                    "action": "Applied rotation ticket PRD-8842 in production.",
                    "observation": "Checkout returned HTTP 503 because the service used the old credential.",
                },
                {"observation": "Checkout monitoring later mentioned HTTP 503 in a generic status dashboard."},
                {"observation": "The HTTP 503 troubleshooting guide describes generic checkout retries."},
                {"observation": "The status page listed checkout HTTP 503 as an incident headline."},
            ],
            "qa_pairs": [
                {
                    "question_id": "cause",
                    "question_type": "causal",
                    "question": "What credential change caused checkout to return HTTP 503?",
                    "answer": "PRD-8842",
                    "evidence_steps": [0],
                }
            ],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "ama.adamem.jsonl"

    count = convert_ama_file(source, output, expected="answer", top_k=2)
    cases = load_jsonl_cases(output)
    semantic_no_graph = AdaMemConfig(
        use_graph=False,
        use_temporal=False,
        use_importance=False,
        use_recency=False,
        use_confidence=False,
        use_feedback=False,
        use_mmr=False,
        use_supersession=False,
        use_soft_staleness=False,
        use_stale_propagation=False,
        use_adjudication_filter=False,
    )
    causal_graph = AdaMemConfig(
        use_graph=True,
        graph_boost=1.0,
        use_temporal=False,
        use_importance=False,
        use_recency=False,
        use_confidence=False,
        use_feedback=False,
        use_mmr=False,
        use_supersession=False,
        use_soft_staleness=False,
        use_stale_propagation=False,
        use_adjudication_filter=False,
    )
    results = run_benchmark(cases, {
        "semantic_no_graph": semantic_no_graph,
        "causal_graph": causal_graph,
    })
    by_name = {result.name: result for result in results}
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    by_record = {record["baseline"]: record for record in records}

    assert count == 1
    assert cases[0].id == "ama_checkout_1"
    assert [observation.label for observation in cases[0].observations[:2]] == [
        "step000.action",
        "step000.observation",
    ]
    assert cases[0].observations[1].cause_labels == ["step000.action"]
    assert "answer" not in cases[0].observations[0].metadata
    assert cases[0].queries[0].metadata["benchmark"] == "ama"
    assert cases[0].queries[0].metadata["evidence"] == ["step000"]
    assert by_name["semantic_no_graph"].passed == 0
    assert by_name["causal_graph"].passed == 1
    assert by_name["causal_graph"].queries[0].trace[0]["relation"] == "graph"
    assert by_record["semantic_no_graph"]["missing_evidence"] == []
    assert by_record["semantic_no_graph"]["graph_evidence_hit_count"] == 0
    assert by_record["causal_graph"]["expected_evidence"] == ["step000"]
    assert by_record["causal_graph"]["graph_evidence_hits"] == ["step000"]
    assert summary["evidence_support"]["semantic_no_graph"]["evidence_matched_records"] == 1
    assert summary["evidence_support"]["causal_graph"]["graph_evidence_hit_records"] == 1


def test_ama_converter_matches_public_hf_schema_fields(tmp_path: Path) -> None:
    source = tmp_path / "ama_hf.jsonl"
    source.write_text(
        json.dumps({
            "episode_id": 0,
            "task": "Baba Is You style rule puzzle.",
            "domain": "Game",
            "task_type": "babaisai",
            "source": "agentbench",
            "success": False,
            "num_turns": 3,
            "total_tokens": 1200,
            "trajectory": [
                {"turn_idx": 6, "action": "left", "observation": "The agent returned to the previous tile."},
                {"turn_idx": 7, "action": "down", "observation": "The agent moved away from the goal."},
                {"turn_idx": 8, "action": "up", "observation": "The observation matches Step 6 again."},
            ],
            "qa_pairs": [
                {
                    "question": "The observation after the `up` action at Step 8 is identical to the observation from Step 6. What causal relationship explains this?",
                    "answer": "The up action reversed the previous down movement.",
                    "question_uuid": "ama-q-uuid",
                    "type": "B",
                },
                {
                    "question": "Between steps 6 and 8, which actions were taken?",
                    "answer": "left, down, up",
                    "question_uuid": "ama-q-range",
                    "type": "A",
                },
            ],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "ama_hf.adamem.jsonl"

    count = convert_ama_file(source, output, expected="answer")
    cases = load_jsonl_cases(output)
    by_query = {query.id: query for query in cases[0].queries}

    assert count == 1
    assert cases[0].id == "0"
    assert cases[0].observations[0].label == "step006.action"
    assert cases[0].observations[1].label == "step006.observation"
    assert cases[0].observations[1].cause_labels == ["step006.action"]
    assert by_query["ama-q-uuid"].metadata["question_type"] == "B"
    assert by_query["ama-q-uuid"].metadata["question_type_name"] == "Causal Inference"
    assert by_query["ama-q-uuid"].metadata["evidence"] == ["step006", "step008"]
    assert by_query["ama-q-range"].metadata["evidence"] == ["step006", "step007", "step008"]


def test_trajectory_step_readout_recovers_step_evidence() -> None:
    case = MemoryQACase(
        id="ama_step_readout",
        observations=[
            ObservationSpec(
                label="step006.action",
                content="[step006.action] action: left",
                kind="action",
                metadata={"benchmark": "ama", "trajectory_step": 6, "memory_key": "step006.action"},
            ),
            ObservationSpec(
                label="step006.observation",
                content="[step006.observation] observation: The agent returned to the previous tile.",
                kind="observation",
                cause_labels=["step006.action"],
                metadata={"benchmark": "ama", "trajectory_step": 6, "memory_key": "step006.observation"},
            ),
            ObservationSpec(
                label="step008.action",
                content="[step008.action] action: restore-alpha-token",
                kind="action",
                metadata={"benchmark": "ama", "trajectory_step": 8, "memory_key": "step008.action"},
            ),
            ObservationSpec(
                label="step008.observation",
                content="[step008.observation] observation: The agent returned to the previous tile.",
                kind="observation",
                cause_labels=["step008.action"],
                metadata={"benchmark": "ama", "trajectory_step": 8, "memory_key": "step008.observation"},
            ),
        ],
        queries=[
            QuerySpec(
                id="step_compare",
                query="The observation at Step 8 is identical to the observation from Step 6. What action was taken at Step 8?",
                expected_substrings=["restore-alpha-token"],
                top_k=2,
                metadata={
                    "benchmark": "ama",
                    "evidence": ["step006", "step008"],
                    "answer": "restore-alpha-token",
                },
            ),
        ],
    )

    results = run_benchmark(cases=[case], configs={
        "semantic_only": baseline_registry()["semantic_only"].config,
        "trajectory_step_readout": baseline_registry()["trajectory_step_readout"].config,
    })
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    by_record = {record["baseline"]: record for record in records}

    assert by_record["semantic_only"]["passed"] is False
    assert by_record["trajectory_step_readout"]["passed"] is True
    assert by_record["trajectory_step_readout"]["missing_evidence"] == []
    assert by_record["trajectory_step_readout"]["answer_keyword_support_matched"] is True
    assert "Step 8 action: restore-alpha-token" in by_record["trajectory_step_readout"]["answer_basis"]
    assert summary["answerability"]["semantic_only"]["answer_keyword_recall_avg"] == 0.0
    assert summary["answerability"]["trajectory_step_readout"]["basis_answer_keyword_recall_avg"] == 1.0
    assert any(
        item["relation"] == "trajectory_step"
        for item in by_record["trajectory_step_readout"]["trace"]
    )


def test_trajectory_answer_basis_marks_inverse_repeated_steps() -> None:
    case = MemoryQACase(
        id="ama_step_basis",
        observations=[
            ObservationSpec(
                label="step007.action",
                content="[step007.action] action: down",
                kind="action",
                metadata={"benchmark": "ama", "trajectory_step": 7, "memory_key": "step007.action"},
            ),
            ObservationSpec(
                label="step007.observation",
                content="[step007.observation] observation: Baba is one tile lower.",
                kind="observation",
                cause_labels=["step007.action"],
                metadata={"benchmark": "ama", "trajectory_step": 7, "memory_key": "step007.observation"},
            ),
            ObservationSpec(
                label="step008.action",
                content="[step008.action] action: up",
                kind="action",
                metadata={"benchmark": "ama", "trajectory_step": 8, "memory_key": "step008.action"},
            ),
            ObservationSpec(
                label="step008.observation",
                content="[step008.observation] observation: Baba returned to the previous tile.",
                kind="observation",
                cause_labels=["step008.action"],
                metadata={"benchmark": "ama", "trajectory_step": 8, "memory_key": "step008.observation"},
            ),
            ObservationSpec(
                label="step006.observation",
                content="[step006.observation] observation: Baba returned to the previous tile.",
                kind="observation",
                metadata={"benchmark": "ama", "trajectory_step": 6, "memory_key": "step006.observation"},
            ),
        ],
        queries=[
            QuerySpec(
                id="inverse",
                query="The observation after Step 8 is identical to Step 6. What explains steps 7 and 8?",
                expected_substrings=["up"],
                top_k=5,
                metadata={
                    "benchmark": "ama",
                    "evidence": ["step006", "step007", "step008"],
                    "answer": "The down and up actions are inverse and cancel out with zero net progress.",
                },
            ),
        ],
    )

    results = run_benchmark(cases=[case], configs={
        "trajectory_step_readout": baseline_registry()["trajectory_step_readout"].config,
    })
    record = benchmark_case_records(results)[0]

    assert "Steps 7-8 actions are inverse" in record["answer_basis"]
    assert "Steps 6 and 8 have identical observations" in record["answer_basis"]
    assert record["basis_answer_keyword_recall"] > record["answer_keyword_recall"]


def test_trajectory_answer_basis_extracts_blocked_rule_state() -> None:
    observation = """Active rules:
wall is stop
baba is you

Objects on the map:
wall 1 step to the right
rule `wall` 2 step to the left
rule `stop` 1 step to the left
"""
    case = MemoryQACase(
        id="ama_blocked_basis",
        observations=[
            ObservationSpec(
                label="step017.action",
                content="[step017.action] action: right",
                kind="action",
                metadata={"benchmark": "ama", "trajectory_step": 17, "memory_key": "step017.action"},
            ),
            ObservationSpec(
                label="step017.observation",
                content=f"[step017.observation] observation: {observation}",
                kind="observation",
                cause_labels=["step017.action"],
                metadata={"benchmark": "ama", "trajectory_step": 17, "memory_key": "step017.observation"},
            ),
            ObservationSpec(
                label="step018.action",
                content="[step018.action] action: right",
                kind="action",
                metadata={"benchmark": "ama", "trajectory_step": 18, "memory_key": "step018.action"},
            ),
            ObservationSpec(
                label="step018.observation",
                content=f"[step018.observation] observation: {observation}",
                kind="observation",
                cause_labels=["step018.action"],
                metadata={"benchmark": "ama", "trajectory_step": 18, "memory_key": "step018.observation"},
            ),
        ],
        queries=[
            QuerySpec(
                id="blocked",
                query="In steps 17 through 18, the agent repeatedly moves right but nothing changes. Why?",
                expected_substrings=["wall"],
                top_k=4,
                metadata={
                    "benchmark": "ama",
                    "evidence": ["step017", "step018"],
                    "answer": "The right movement is blocked by a wall because wall is stop, so the action makes no progress.",
                },
            ),
        ],
    )

    results = run_benchmark(cases=[case], configs={
        "trajectory_step_readout": baseline_registry()["trajectory_step_readout"].config,
    })
    record = benchmark_case_records(results)[0]

    assert "rule wall is stop makes wall objects impassable" in record["answer_basis"]
    assert "action right is blocked by adjacent wall due to wall is stop" in record["answer_basis"]
    assert "repeat action right with unchanged observations" in record["answer_basis"]
    assert record["basis_answer_keyword_recall"] > record["answer_keyword_recall"]


def test_jsonl_benchmark_experiment_record_shape(tmp_path: Path) -> None:
    cases = load_jsonl_cases(Path("benchmarks/dynamic_state_transfer.jsonl"))[:1]
    specs = {
        name: spec
        for name, spec in baseline_registry().items()
        if spec.config.state_extractor_name != "llm_json"
    }
    results = run_benchmark(cases, {name: spec.config for name, spec in specs.items()})
    record = experiment_record(
        run_name="dynamic-state-smoke",
        run_type="jsonl_retrieval_benchmark",
        dataset="benchmarks/dynamic_state_transfer.jsonl",
        split_or_case_limit="1",
        baselines=specs,
        results=[result.name for result in results],
        raw_outputs=[
            {"baseline": result.name, "query_id": query.query_id, "trace": query.trace}
            for result in results
            for query in result.queries
        ],
        notes={"benchmark_kind": "retrieval_support"},
        command=["adamem-eval", "--dataset"],
    )

    output = write_experiment_record(tmp_path / "dynamic-state-smoke.json", record)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["run_type"] == "jsonl_retrieval_benchmark"
    assert data["baseline_names"] == list(specs)
    assert data["raw_outputs"]
    assert data["notes"]["benchmark_kind"] == "retrieval_support"


def test_jsonl_benchmark_can_run_llm_state_extractor_ablation() -> None:
    case = MemoryQACase(
        id="llm-extractor",
        observations=[
            ObservationSpec(
                content="Structured telemetry update id 7.",
                label="telemetry",
            )
        ],
        queries=[
            QuerySpec(
                id="runtime-status",
                query="Is the staging runner online?",
                expected_substrings=["online"],
                top_k=2,
                metadata={"state_slot": "runtime.staging_runner.status"},
            )
        ],
    )
    configs = {
        "semantic_llm_state_adjudication": (
            baseline_registry()["semantic_llm_state_adjudication"].config
        )
    }
    state_extractors, notes, prompts = _state_extractor_runtime(
        configs,
        provider="mock",
        model="unused",
        mock_response=(
            '{"patches":[{"slot":"runtime.staging_runner.status",'
            '"value":"online","status":"active"}]}'
        ),
        max_tokens=128,
        temperature=0.0,
    )

    results = run_benchmark([case], configs, state_extractors=state_extractors)

    assert results[0].passed == 1
    assert results[0].queries[0].passed is True
    assert notes["state_extractor_provider"] == "mock"
    assert notes["state_extractor_baselines"] == ["semantic_llm_state_adjudication"]
    assert "state_extractor_system" in prompts


def test_jsonl_query_metadata_is_available_for_breakdowns() -> None:
    cases = load_jsonl_cases(Path("benchmarks/dynamic_state_transfer.jsonl"))
    results = {result.name: result for result in run_benchmark(cases)}
    query = results["state_readout"].queries[0]

    assert query.metadata["dimension"] == "implicit_policy_adaptation"


def test_jsonl_benchmark_failure_summary_groups_by_metadata() -> None:
    cases = load_jsonl_cases(Path("benchmarks/dynamic_state_transfer.jsonl"))
    results = run_benchmark(cases, {
        name: baseline_registry()[name].config
        for name in ("semantic_only", "state_readout")
    })
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    report = benchmark_failure_report(records, max_examples=1)

    assert summary["by_baseline"]["semantic_only"]["passed"] == 0
    assert summary["by_baseline"]["state_readout"]["passed"] == 7
    assert summary["failure_modes"]["expected_support_missing"] == 7
    assert "implicit_policy_adaptation" in summary["by_metadata"]["dimension"]
    assert "state_resolution" in summary["by_metadata"]["dimension"]
    assert "forbidden_support_present" in records[0]["failure_modes"]
    assert "# JSONL Retrieval Benchmark Failure Report" in report
    assert "## Paper Metrics" in report
    assert "## State Readout Exposure" in report
    assert "## State Memory Inventory" in report
    assert "## Premise Correction" in report
    assert "## Evidence Support" in report
    assert summary["diagnostics_by_metadata"]["dimension"]["implicit_policy_adaptation"]["state_readout"][
        "total"
    ] > 0
    assert summary["paper_metrics"]["state_readout"]["support_accuracy"] == 1.0
    assert summary["paper_metrics"]["state_readout"]["state_slot_match_rate"] == 1.0
    assert summary["paper_metrics"]["state_readout"]["state_readout_missing_rate"] == 0.0
    assert summary["paper_metrics"]["state_readout"]["unmarked_state_exposure_rate"] is None
    assert summary["paper_metrics"]["state_readout"]["premise_correction_rate"] == 0.0
    assert summary["paper_metrics"]["semantic_only"]["state_readout_missing_rate"] == 1.0
    assert summary["state_memory_inventory"]["semantic_only"]["max_state_memory_count"] == 0
    assert summary["state_memory_inventory"]["state_readout"]["records_with_state_memory"] == 7
    assert "runtime.staging_build_runner.status" in summary["state_memory_inventory"]["state_readout"][
        "active_state_slots"
    ]
    assert records[-1]["active_state_count"] > 0
    assert summary["state_readout_exposure"]["state_readout"]["state_retrieval_records"] == 7
    assert summary["state_readout_exposure"]["state_readout"]["state_slot_match_records"] == 7
    assert summary["state_readout_exposure"]["state_readout"]["unmarked_state_retrieval_records"] == 0
    assert summary["state_readout_exposure"]["semantic_only"]["state_readout_missing_records"] == 7
    assert summary["failure_modes"]["state_readout_missing"] == 7


def test_jsonl_benchmark_metadata_diagnostics_include_evidence_and_answerability() -> None:
    case = MemoryQACase(
        id="ama_grouped",
        observations=[
            ObservationSpec(
                label="step001.action",
                content="[step001.action] action: right",
                kind="action",
                metadata={"benchmark": "ama", "trajectory_step": 1, "memory_key": "step001.action"},
            ),
        ],
        queries=[
            QuerySpec(
                id="q-a",
                query="What happened at Step 1?",
                expected_substrings=["right"],
                top_k=1,
                metadata={
                    "benchmark": "ama",
                    "question_type": "A",
                    "answer": "right",
                    "evidence": ["step001"],
                },
            )
        ],
    )
    results = run_benchmark(cases=[case], configs={
        "trajectory_step_readout": baseline_registry()["trajectory_step_readout"].config,
    })
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    report = benchmark_failure_report(records)
    diagnostics = summary["diagnostics_by_metadata"]["question_type"]["A"]["trajectory_step_readout"]

    assert diagnostics["evidence_matched_records"] == 1
    assert diagnostics["evidence_query_total"] == 1
    assert diagnostics["basis_answer_keyword_recall_avg"] == 1.0
    assert "## By question_type Diagnostics" in report
    assert "| A | trajectory_step_readout | 1/1 (100.00%) | 100.00% | 100.00% | 1/1 |" in report


def test_jsonl_benchmark_summary_compares_pairs_against_first_baseline() -> None:
    cases = load_jsonl_cases(Path("benchmarks/dynamic_state_transfer.jsonl"))
    results = run_benchmark(cases, {
        name: baseline_registry()[name].config
        for name in ("semantic_only", "semantic_state_readout")
    })
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    comparison = summary["pairwise_vs_first_baseline"]["semantic_state_readout"]
    report = benchmark_failure_report(records)

    assert comparison["reference"] == "semantic_only"
    assert comparison["candidate"] == "semantic_state_readout"
    assert comparison["common_total"] == 7
    assert comparison["gained_passes"] == 7
    assert comparison["lost_passes"] == 0
    assert comparison["net_delta"] == 7
    assert comparison["by_metadata"]["dimension"]["implicit_policy_adaptation"]["gained_passes"] == 2
    assert comparison["by_metadata"]["dimension"]["premise_resistance"]["gained_passes"] == 4
    assert "Pairwise Vs semantic_only" in report


def test_jsonl_benchmark_treats_correction_text_as_resolved_forbidden_support() -> None:
    cases = load_jsonl_cases(Path("benchmarks/dynamic_state_transfer.jsonl"))
    results = run_benchmark(cases, {
        "semantic_state_premise_correction": (
            baseline_registry()["semantic_state_premise_correction"].config
        ),
    })
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    report = benchmark_failure_report(records)
    by_query = {record["query_id"]: record for record in records}

    assert results[0].passed == results[0].total
    correction = summary["premise_correction"]["semantic_state_premise_correction"]
    assert correction["correction_records"] == 3
    assert correction["correction_items"] == 3
    assert correction["corrected_forbidden_records"] == 3
    assert correction["unresolved_forbidden_records"] == 0
    assert summary["paper_metrics"]["semantic_state_premise_correction"]["premise_correction_rate"] == 3 / 7
    assert "## Premise Correction" in report
    for query_id in ("current_passport_status", "current_workflow_rule", "current_runtime_status"):
        record = by_query[query_id]
        assert record["premise_correction_count"] == 1
        assert record["corrected_forbidden"]
        assert record["present_forbidden"] == []
        assert "forbidden_support_present" not in record["failure_modes"]


def test_jsonl_benchmark_supports_unknown_current_state_correction() -> None:
    case = MemoryQACase(
        id="unknown_current_state",
        observations=[
            ObservationSpec(content="[2026-01-01] user: I just moved into a place in Seattle."),
            ObservationSpec(content="[2026-02-01] user: I no longer live in Seattle."),
        ],
        queries=[
            QuerySpec(
                id="unknown_location",
                query="Since I'm in Seattle, recommend local events there.",
                expected_substrings=["unknown-current"],
                forbidden_substrings=["Seattle"],
                top_k=1,
                metadata={
                    "dimension": "premise_resistance",
                    "state_slot": "location",
                },
            )
        ],
    )

    results = run_benchmark(
        [case],
        {
            "semantic_state_adjudication": baseline_registry()["semantic_state_adjudication"].config,
            "semantic_state_premise_correction": (
                baseline_registry()["semantic_state_premise_correction"].config
            ),
        },
    )
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    report = benchmark_failure_report(records)
    by_baseline = {record["baseline"]: record for record in records}
    adjudication = by_baseline["semantic_state_adjudication"]
    correction = by_baseline["semantic_state_premise_correction"]

    assert all(result.passed == 1 for result in results)
    assert adjudication["premise_correction_count"] == 0
    assert adjudication["corrected_forbidden"] == ["Seattle"]
    assert adjudication["present_forbidden"] == []
    assert adjudication["trace"][0]["metadata"]["state_status"] == "unknown_current"
    assert correction["premise_correction_count"] == 1
    assert correction["corrected_forbidden"] == ["Seattle"]
    assert correction["present_forbidden"] == []
    assert correction["trace"][0]["metadata"]["current_value"] == "unknown-current"
    assert "current value is unknown" in correction["trace"][0]["content"]
    assert summary["unknown_current"]["semantic_state_adjudication"]["unknown_current_records"] == 1
    assert summary["unknown_current"]["semantic_state_premise_correction"]["unknown_current_records"] == 0
    assert (
        summary["unknown_current"]["semantic_state_premise_correction"][
            "unknown_current_correction_records"
        ]
        == 1
    )
    assert (
        summary["unknown_current"]["semantic_state_premise_correction"][
            "resolved_invalidated_value_records"
        ]
        == 1
    )
    assert "## Unknown-Current State" in report


def test_unknown_current_transfer_fixture_favors_state_authority() -> None:
    cases = load_jsonl_cases(Path("benchmarks/unknown_current_state_transfer.jsonl"))
    results = run_benchmark(
        cases,
        {
            name: baseline_registry()[name].config
            for name in (
                "semantic_only",
                "semantic_state_adjudication",
                "semantic_state_premise_correction",
            )
        },
    )
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    by_baseline = {result.name: result for result in results}

    assert by_baseline["semantic_only"].passed == 0
    assert by_baseline["semantic_state_adjudication"].passed == 5
    assert by_baseline["semantic_state_premise_correction"].passed == 5
    assert summary["unknown_current"]["semantic_state_adjudication"]["unknown_current_records"] == 5
    assert (
        summary["unknown_current"]["semantic_state_premise_correction"][
            "unknown_current_correction_records"
        ]
        == 4
    )
    assert (
        summary["unknown_current"]["semantic_state_premise_correction"][
            "resolved_invalidated_value_records"
        ]
        == 5
    )
    assert summary["pairwise_vs_first_baseline"]["semantic_state_adjudication"]["gained_passes"] == 5


def test_jsonl_records_expose_state_pollution_metrics() -> None:
    case = MemoryQACase(
        id="state_pollution_boundary",
        observations=[
            ObservationSpec(content="The staging build runner is online."),
            ObservationSpec(content="The status report has three sections."),
        ],
        queries=[
            QuerySpec(
                id="runtime_status",
                query="Is the staging build runner online?",
                expected_substrings=["online"],
                top_k=1,
                metadata={"state_slot": "runtime.*.status", "state_available": True},
            ),
            QuerySpec(
                id="generic_status_report",
                query="How many sections were in the status report?",
                expected_substrings=["three sections"],
                top_k=1,
            ),
            QuerySpec(
                id="wrong_slot_annotation",
                query="Is the staging build runner online?",
                expected_substrings=["online"],
                top_k=1,
                metadata={"state_slot": "resource.*.status", "state_available": True},
            ),
            QuerySpec(
                id="state_unavailable",
                query="Can you suggest something around me?",
                expected_substrings=["three sections"],
                top_k=1,
                metadata={"state_slot": "location", "state_available": False},
            ),
        ],
    )

    results = run_benchmark(cases=[case], configs={
        "state_readout": baseline_registry()["state_readout"].config,
    })
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    by_query = {record["query_id"]: record for record in records}

    assert by_query["runtime_status"]["state_retrieval_count"] == 1
    assert by_query["runtime_status"]["retrieved_state_slots"] == ["runtime.staging_build_runner.status"]
    assert by_query["runtime_status"]["expected_state_slots"] == ["runtime.*.status"]
    assert by_query["runtime_status"]["unexpected_state_slots"] == []
    assert by_query["runtime_status"]["state_slot_matched"] is True
    assert by_query["runtime_status"]["trace"][0]["kind"] == "state"
    assert by_query["generic_status_report"]["state_retrieval_count"] == 0
    assert by_query["wrong_slot_annotation"]["retrieved_state_slots"] == ["runtime.staging_build_runner.status"]
    assert by_query["wrong_slot_annotation"]["unexpected_state_slots"] == ["runtime.staging_build_runner.status"]
    assert "state_readout_slot_mismatch" in by_query["wrong_slot_annotation"]["failure_modes"]
    assert by_query["state_unavailable"]["state_sensitive"] is True
    assert by_query["state_unavailable"]["state_available"] is False
    assert by_query["state_unavailable"]["state_readout_expected"] is False
    assert "state_readout_missing" not in by_query["state_unavailable"]["failure_modes"]
    assert summary["state_readout_exposure"]["state_readout"]["unmarked_state_retrieval_records"] == 0
    assert summary["state_readout_exposure"]["state_readout"]["state_slot_mismatch_records"] == 1
    assert summary["state_readout_exposure"]["state_readout"]["state_unavailable_total"] == 1


def test_jsonl_paper_metrics_report_n_a_for_no_state_queries() -> None:
    case = MemoryQACase(
        id="ordinary_retrieval",
        observations=[
            ObservationSpec(content="The notebook has a blue cover."),
        ],
        queries=[
            QuerySpec(
                id="cover",
                query="What color is the notebook cover?",
                expected_substrings=["blue"],
                top_k=1,
            ),
        ],
    )

    results = run_benchmark(cases=[case], configs={
        "semantic_only": baseline_registry()["semantic_only"].config,
        "state_readout": baseline_registry()["state_readout"].config,
    })
    records = benchmark_case_records(results)
    summary = benchmark_failure_summary(records)
    report = benchmark_failure_report(records)

    assert summary["paper_metrics"]["state_readout"]["state_slot_match_rate"] is None
    assert summary["paper_metrics"]["state_readout"]["state_readout_missing_rate"] is None
    assert (
        "| state_readout | 1/1 | 100.00% | 0 | n/a | n/a | n/a | "
        "n/a | n/a | n/a | n/a | 0.00% |"
    ) in report
