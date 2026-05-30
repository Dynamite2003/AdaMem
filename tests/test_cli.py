from __future__ import annotations

import json

from adamem.cli import main


def test_demo_json_compares_adjudication_trace_baselines(capsys) -> None:
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--query-id",
        "current_runtime_status",
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "adamem.demo.v1"
    assert "not paper evidence" in payload["claim_boundary"]
    assert [baseline["name"] for baseline in payload["baselines"]] == [
        "semantic_state_adjudication",
        "semantic_state_adjudication_trace",
    ]
    assert all(baseline["passed"] for baseline in payload["baselines"])

    trace_baseline = payload["baselines"][1]
    trace_item = trace_baseline["trace"][0]
    assert trace_item["kind"] == "state_adjudication"
    assert "authorized current state" in trace_item["content"]
    assert "online" in trace_item["content"]
    assert "offline" not in trace_item["content"]
    assert trace_item["metadata"]["source_observation_label"] == "new_runtime"
    assert trace_item["metadata"]["adjudicated_source_observation_label"] == "old_runtime"
    assert trace_item["metadata"]["adjudication_reason"] == "active_state_authority"


def test_demo_markdown_mentions_claim_boundary(capsys) -> None:
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--query-id",
        "current_runtime_status",
    ])

    output = capsys.readouterr().out
    assert "# AdaMem Stale-Memory Demo" in output
    assert "not paper evidence" in output
    assert "semantic_state_adjudication_trace" in output
    assert "kind=state_adjudication" in output
