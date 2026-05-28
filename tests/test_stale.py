from __future__ import annotations

import json
from pathlib import Path

from adamem.bench import default_ablation_configs, load_jsonl_cases
from adamem.convert import convert_stale_file, convert_stale_sample
from adamem.eval import run_stale_benchmark, stale_report
from adamem.llm import MockLLMClient


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


def test_run_stale_benchmark_with_mock_clients(tmp_path: Path) -> None:
    src = tmp_path / "stale_in.json"
    src.write_text(json.dumps([_toy_stale_instance()]))
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
