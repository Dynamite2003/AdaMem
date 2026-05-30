from __future__ import annotations

import json
from pathlib import Path

from adamem.bench import default_ablation_configs, load_jsonl_cases
from adamem.convert import annotate_stale_jsonl_file, convert_stale_file, convert_stale_sample
from adamem.diagnostics import (
    diagnostic_case_records,
    diagnostic_failure_report,
    diagnostic_failure_summary,
    diagnostics_report,
    run_stale_retrieval_diagnostics,
    stale_text_signal,
    text_supports_signal,
)
from adamem.eval import run_stale_benchmark, stale_report
from adamem.llm import MockLLMClient, build_client


def _toy_stale_instance(uid: str = "stale-toy-1", sample_type: str = "T1") -> dict:
    return {
        "uid": uid,
        "type": sample_type,
        "M_old": "User lives in Seattle.",
        "M_new": "User lives in Boston.",
        "explanation": "User moved from Seattle to Boston.",
        "probing_queries": {
            "dim1_query": "Do I still live in Seattle?",
            "dim2_query": "Since I'm in Seattle, what's a good local park?",
            "dim3_query": "Recommend a coffee shop near me.",
        },
        "relevant_session_index": [0, 2],
        "timestamps": [
            "2026-01-01T10:00:00+00:00",
            "2026-02-01T10:00:00+00:00",
            "2026-03-01T10:00:00+00:00",
        ],
        "haystack_session": [
            [
                {"role": "user", "content": "I just moved into a place in Seattle."},
                {"role": "assistant", "content": "Welcome to Seattle!"},
            ],
            [
                {"role": "user", "content": "Weather is rainy as usual."},
                {"role": "assistant", "content": "Typical PNW weather."},
            ],
            [
                {"role": "user", "content": "I relocated to Boston for a new job."},
                {"role": "assistant", "content": "Congrats on the move to Boston!"},
            ],
        ],
    }


def test_convert_stale_sample_emits_three_dim_queries() -> None:
    row = convert_stale_sample(_toy_stale_instance(), top_k=4)

    assert row["id"] == "stale-toy-1"
    # 3 sessions x 2 turns = 6 observations
    assert len(row["observations"]) == 6
    # 3 probing queries (dim1/2/3)
    assert len(row["queries"]) == 3
    dims = [q["metadata"]["stale_dim"] for q in row["queries"]]
    assert dims == [1, 2, 3]
    # Each query carries the stale metadata needed for judge prompts.
    for q in row["queries"]:
        meta = q["metadata"]
        assert meta["M_old"] == "User lives in Seattle."
        assert meta["M_new"] == "User lives in Boston."
        assert meta["stale_type"] == "T1"
    # Relevant sessions get a higher importance hint.
    relevant = [obs for obs in row["observations"] if "relevant" in obs["metadata"]["tags"]]
    assert relevant and all(obs["importance"] >= 0.6 for obs in relevant)


def test_convert_stale_sample_adds_query_only_state_opportunity_metadata() -> None:
    row = convert_stale_sample(_toy_stale_instance(), top_k=4)
    queries = {query["id"].rsplit(".", 1)[-1]: query for query in row["queries"]}

    dim1_meta = queries["dim1"]["metadata"]
    assert dim1_meta["state_slot"] == "location"
    assert dim1_meta["state_slot_source"] == "stale_metadata_heuristic"
    assert "dependency_source_slot" not in dim1_meta

    dim2_meta = queries["dim2"]["metadata"]
    dim3_meta = queries["dim3"]["metadata"]
    for meta in (dim2_meta, dim3_meta):
        assert meta["state_slot"] == "location"
        assert meta["state_slot_source"] == "stale_metadata_heuristic"
        assert meta["dependency_source_slot"] == "location"
        assert meta["dependency_source_slot_source"] == "stale_metadata_heuristic"
        assert meta["dependency_target_family"] == "local_context"

    forbidden_observation_keys = {
        "M_old",
        "M_new",
        "dependency_source_slot",
        "dependency_target_family",
        "explanation",
        "relevant_session_index",
        "state_slot",
        "state_slot_source",
    }
    for observation in row["observations"]:
        assert forbidden_observation_keys.isdisjoint(observation["metadata"])


def test_convert_stale_sample_labels_multiple_dependency_families() -> None:
    employer_sample = _toy_stale_instance()
    employer_sample.update({
        "M_old": "User works at Acme.",
        "M_new": "User works at Globex.",
        "explanation": "The user's employer changed from Acme to Globex.",
        "probing_queries": {"dim3_query": "Which benefits portal should I use?"},
    })
    employer_row = convert_stale_sample(employer_sample, top_k=4)
    employer_meta = employer_row["queries"][0]["metadata"]
    assert employer_meta["state_slot"] == "organization.employer"
    assert employer_meta["dependency_source_slot"] == "organization.employer"
    assert employer_meta["dependency_target_family"] == "employment_context"

    dietary_sample = _toy_stale_instance()
    dietary_sample.update({
        "M_old": "User can eat peanuts.",
        "M_new": "User is now allergic to peanuts.",
        "explanation": "The user's peanut allergy status changed.",
        "probing_queries": {"dim3_query": "What restaurant meal is safe to order?"},
    })
    dietary_row = convert_stale_sample(dietary_sample, top_k=4)
    dietary_meta = dietary_row["queries"][0]["metadata"]
    assert dietary_meta["state_slot"] == "health.*.status"
    assert dietary_meta["dependency_source_slot"] == "health.*.status"
    assert dietary_meta["dependency_target_family"] == "food_safety_context"


def test_convert_stale_file_writes_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance("u1"), _toy_stale_instance("u2", "T2")]))
    out = tmp_path / "stale.jsonl"

    count = convert_stale_file(src, out, top_k=4)

    assert count == 2
    cases = load_jsonl_cases(out)
    assert [c.id for c in cases] == ["u1", "u2"]
    # Metadata threading from bench loader.
    assert cases[0].queries[0].metadata["stale_dim"] == 1
    assert cases[1].queries[0].metadata["stale_type"] == "T2"


def test_convert_stale_file_filters_types(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance("u1", "T1"), _toy_stale_instance("u2", "T2")]))
    out = tmp_path / "stale.jsonl"

    count = convert_stale_file(src, out, types=["T2"])

    assert count == 1
    cases = load_jsonl_cases(out)
    assert [c.id for c in cases] == ["u2"]


def test_annotate_stale_jsonl_file_backfills_query_only_metadata(tmp_path: Path) -> None:
    row = convert_stale_sample(_toy_stale_instance(), top_k=4)
    for query in row["queries"]:
        metadata = query["metadata"]
        for key in (
            "dependency_source_slot",
            "dependency_source_slot_source",
            "dependency_target_family",
            "state_slot",
            "state_slot_source",
        ):
            metadata.pop(key, None)
    before_observation_metadata = [dict(obs["metadata"]) for obs in row["observations"]]

    src = tmp_path / "stale_in.jsonl"
    src.write_text(json.dumps(row) + "\n")
    out = tmp_path / "stale_annotated.jsonl"

    count = annotate_stale_jsonl_file(src, out)

    assert count == 1
    annotated = json.loads(out.read_text())
    assert annotated["observations"]
    assert [obs["metadata"] for obs in annotated["observations"]] == before_observation_metadata
    queries = {query["id"].rsplit(".", 1)[-1]: query for query in annotated["queries"]}
    assert queries["dim1"]["metadata"]["state_slot"] == "location"
    assert "dependency_source_slot" not in queries["dim1"]["metadata"]
    assert queries["dim2"]["metadata"]["dependency_source_slot"] == "location"
    assert queries["dim3"]["metadata"]["dependency_target_family"] == "local_context"


def test_run_stale_benchmark_with_mock_clients(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)

    answer_client = MockLLMClient("The user lives in Boston.")
    judge_client = MockLLMClient("CORRECT")
    configs = {"semantic_only": default_ablation_configs()["semantic_only"]}
    raw_outputs: list[dict] = []

    results = run_stale_benchmark(
        out,
        answer_client=answer_client,
        judge_client=judge_client,
        configs=configs,
        top_k=4,
        raw_outputs=raw_outputs,
    )

    assert len(results) == 1
    result = results[0]
    assert result.name == "semantic_only"
    # 1 case x 3 dims = 3 queries.
    assert result.n_total == 3
    assert result.n_correct == 3
    assert result.accuracy == 1.0
    # Mock client received both answer and judge prompts.
    assert len(answer_client.calls) == 3
    assert len(judge_client.calls) == 3
    assert len(raw_outputs) == 3
    assert raw_outputs[0]["baseline"] == "semantic_only"
    assert "Conversation memory excerpts" in raw_outputs[0]["answer_prompt"]
    assert "OLD belief" in raw_outputs[0]["judge_prompt"]
    assert raw_outputs[0]["answer_raw"] == "The user lives in Boston."
    assert raw_outputs[0]["judge_raw"] == "CORRECT"
    assert raw_outputs[0]["retrieved"]
    # By-dim breakdown covers all three dimensions.
    assert set(result.by_dim) == {1, 2, 3}
    for dim_stats in result.by_dim.values():
        assert dim_stats["n_total"] == 1


def test_run_stale_benchmark_judge_incorrect(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)

    answer_client = MockLLMClient("Whatever.")
    judge_client = MockLLMClient(["INCORRECT", "CORRECT", "INCORRECT"])
    configs = {"semantic_only": default_ablation_configs()["semantic_only"]}

    results = run_stale_benchmark(
        out,
        answer_client=answer_client,
        judge_client=judge_client,
        configs=configs,
        top_k=4,
    )
    result = results[0]

    assert result.n_correct == 1
    assert result.n_total == 3
    assert result.accuracy == 1 / 3


def test_run_stale_benchmark_filters_and_balances_types(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([
        _toy_stale_instance("t1-a", "T1"),
        _toy_stale_instance("t1-b", "T1"),
        _toy_stale_instance("t2-a", "T2"),
        _toy_stale_instance("t2-b", "T2"),
    ]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)

    answer_client = MockLLMClient("The user lives in Boston.")
    judge_client = MockLLMClient("CORRECT")
    configs = {"semantic_only": default_ablation_configs()["semantic_only"]}
    results = run_stale_benchmark(
        out,
        answer_client=answer_client,
        judge_client=judge_client,
        configs=configs,
        top_k=4,
        stale_types=["T1", "T2"],
        limit_per_stale_type=1,
    )

    result = results[0]
    assert result.n_total == 6
    assert result.by_type["T1"]["n_total"] == 3
    assert result.by_type["T2"]["n_total"] == 3


def test_build_mock_client_ignores_model_for_cli_smoke() -> None:
    client = build_client("mock", model="ignored")

    assert client.complete("hello") == "CORRECT"


def test_stale_report_has_all_dim_columns(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)

    answer_client = MockLLMClient("ok")
    judge_client = MockLLMClient("CORRECT")
    configs = {"semantic_only": default_ablation_configs()["semantic_only"]}
    results = run_stale_benchmark(
        out,
        answer_client=answer_client,
        judge_client=judge_client,
        configs=configs,
        top_k=4,
    )

    report = stale_report(results)
    assert "dim1 SR" in report
    assert "dim2 PR" in report
    assert "dim3 IPA" in report
    assert "ADR" in report
    assert "SLR" in report


def test_stale_leak_rate_drops_when_filter_active(tmp_path: Path) -> None:
    """SLR should be lower when mechanism C is on, because the OLD belief
    text is filtered out of the retrieved context."""
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)

    answer_client = MockLLMClient("ok")
    judge_client = MockLLMClient("CORRECT")
    configs = {
        "semantic_only": default_ablation_configs()["semantic_only"],
        "delta_full": default_ablation_configs()["delta_full"],
        "state_readout": default_ablation_configs()["state_readout"],
    }
    results = {r.name: r for r in run_stale_benchmark(
        out,
        answer_client=answer_client,
        judge_client=judge_client,
        configs=configs,
        top_k=4,
    )}
    # delta_full's SLR should not be worse than semantic_only's; for this toy
    # case (Seattle vs Boston) the M_old keywords overlap means we mainly check
    # the metric is well-defined and bounded in [0, 1].
    for name, res in results.items():
        assert 0.0 <= res.stale_leak_rate <= 1.0
    assert results["delta_full"].stale_leak_rate <= results["semantic_only"].stale_leak_rate


def test_stale_text_signal_ignores_generic_old_belief_words() -> None:
    signal = stale_text_signal("I've been staying in Seattle for the past few years, so that's where I'm located.")

    assert "seattle" in signal.tokens
    assert "years" not in signal.tokens
    assert "user" not in signal.tokens
    assert text_supports_signal("I recently moved away from Seattle.", signal)
    assert not text_supports_signal("The user has been busy for the past few years.", signal)


def test_retrieval_diagnostics_separate_current_and_stale_evidence(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)
    cases = load_jsonl_cases(out)
    configs = {
        "semantic_only": default_ablation_configs()["semantic_only"],
        "delta_full": default_ablation_configs()["delta_full"],
        "state_readout": default_ablation_configs()["state_readout"],
    }

    results = {r.name: r for r in run_stale_retrieval_diagnostics(cases, configs)}

    assert set(results) == {"semantic_only", "delta_full", "state_readout"}
    for result in results.values():
        assert result.total == 3
        assert 0.0 <= result.current_recall_rate <= 1.0
        assert 0.0 <= result.stale_exposure_rate <= 1.0
        assert 0.0 <= result.old_support_adjudication_rate <= 1.0
        assert all(record.old_supports <= 3 for record in result.queries)

    assert results["delta_full"].old_support_adjudication_rate >= results["semantic_only"].old_support_adjudication_rate
    assert results["state_readout"].current_recall_rate > results["delta_full"].current_recall_rate
    assert results["semantic_only"].queries[0].state_memory_count == 0
    assert results["state_readout"].queries[0].active_state_count > 0
    assert "location" in results["state_readout"].queries[0].active_state_slots
    report = diagnostics_report(list(results.values()))
    assert "current recall" in report
    assert "stale exposure" in report


def test_retrieval_diagnostics_measure_premise_correction_hits(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)
    cases = load_jsonl_cases(out)
    configs = {
        "semantic_state_adjudication": default_ablation_configs()["semantic_state_adjudication"],
        "semantic_state_premise_correction": (
            default_ablation_configs()["semantic_state_premise_correction"]
        ),
    }

    results = {result.name: result for result in run_stale_retrieval_diagnostics(cases, configs)}
    correction = results["semantic_state_premise_correction"]
    adjudication = results["semantic_state_adjudication"]
    report = diagnostics_report(list(results.values()))

    assert adjudication.premise_correction_hit_rate == 0.0
    assert correction.premise_correction_opportunity_rate == 2 / 3
    assert correction.premise_correction_hit_rate == 0.5
    corrected_queries = [query for query in correction.queries if query.premise_correction_hit]
    assert len(corrected_queries) == 1
    assert all(query.premise_correction_best_rank == 1 for query in corrected_queries)
    assert all(query.trace[0]["is_premise_correction"] for query in corrected_queries)
    assert "premise correction hit" in report


def test_diagnostic_case_records_export_failures(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)
    cases = load_jsonl_cases(out)
    configs = {
        "semantic_only": default_ablation_configs()["semantic_only"],
        "state_readout": default_ablation_configs()["state_readout"],
    }

    results = run_stale_retrieval_diagnostics(cases, configs)
    records = diagnostic_case_records(results)

    assert records
    assert any(record["baseline"] == "semantic_only" for record in records)
    assert all(record["failure_modes"] for record in records)
    assert any("current_evidence_not_recalled" in record["failure_modes"] for record in records)
    assert any("trace" in record for record in records)
    assert any(record["active_state_count"] > 0 for record in records if record["baseline"] == "state_readout")
    assert any(
        "state_authority_absent_or_extraction_failure" in record["failure_attributions"]
        for record in records
        if record["baseline"] == "semantic_only"
    )


def test_diagnostic_failure_summary_groups_records(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
    out = tmp_path / "stale.jsonl"
    convert_stale_file(src, out, top_k=4)
    cases = load_jsonl_cases(out)
    configs = {
        "semantic_only": default_ablation_configs()["semantic_only"],
        "state_readout": default_ablation_configs()["state_readout"],
    }

    results = run_stale_retrieval_diagnostics(cases, configs)
    records = diagnostic_case_records(results)
    summary = diagnostic_failure_summary(records)
    report = diagnostic_failure_report(records)

    assert summary["total_records"] == len(records)
    assert summary["by_baseline"]["semantic_only"] >= 1
    assert summary["by_failure_mode"]["current_evidence_not_recalled"] >= 1
    assert summary["by_failure_attribution"]["state_authority_absent_or_extraction_failure"] >= 1
    assert "state_authority_absent_or_extraction_failure" in summary["by_baseline_failure_attribution"][
        "semantic_only"
    ]
    assert summary["examples_by_failure_attribution"]["state_authority_absent_or_extraction_failure"][0][
        "baseline"
    ] == "semantic_only"
    assert "Failure Attributions By Baseline" in report
    assert "Representative Failure Attributions" in report
    assert "Failure Modes By Baseline" in report
    assert "Representative Examples" in report
