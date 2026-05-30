from __future__ import annotations

import json
from pathlib import Path

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
    assert payload["evidence_boundary"]["artifact_type"] == "api_free_mechanism_demo"
    assert "answer_accuracy" in payload["evidence_boundary"]["blocked_claims"]
    assert "sota" in payload["evidence_boundary"]["blocked_claims"]
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
    assert "## Evidence Boundary" in output
    assert "Blocked claims:" in output
    assert "semantic_state_adjudication_trace" in output
    assert "kind=state_adjudication" in output


def test_demo_all_queries_summarizes_state_family_sweep(capsys) -> None:
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "all_queries"
    assert payload["query_count"] == 9
    summary = payload["summary"]["by_baseline"]
    assert summary["semantic_state_adjudication"]["passed"] == 9
    assert summary["semantic_state_adjudication"]["total"] == 9
    assert summary["semantic_state_adjudication_trace"]["passed"] == 9
    assert summary["semantic_state_adjudication_trace"]["total"] == 9
    assert summary["semantic_state_adjudication_trace"]["state_adjudication_traces"] == 8
    assert "runtime.staging_build_runner.status" in (
        summary["semantic_state_adjudication_trace"]["state_slots"]
    )
    assert "workflow.checkout_deploys.rollback" in (
        summary["semantic_state_adjudication_trace"]["state_slots"]
    )


def test_demo_paper_profile_includes_mainstream_baseline_provenance(capsys) -> None:
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_profile"] == "paper"
    assert payload["baseline_names"] == [
        "semantic_only",
        "a_mem_evolution",
        "zep_temporal_kg",
        "mem0_extraction",
        "semantic_state_adjudication_trace",
    ]
    first_query = payload["queries"][0]
    by_name = {baseline["name"]: baseline for baseline in first_query["baselines"]}
    assert by_name["a_mem_evolution"]["source_name"] == "A-MEM"
    assert by_name["a_mem_evolution"]["implementation_status"] == "api_free_approximation"
    assert by_name["zep_temporal_kg"]["source_name"] == "Zep/Graphiti"
    assert by_name["mem0_extraction"]["source_name"] == "Mem0"
    assert payload["summary"]["by_baseline"]["semantic_state_adjudication_trace"]["passed"] == 9


def test_demo_writes_interactive_html_artifact(tmp_path: Path, capsys) -> None:
    output = tmp_path / "adamem_state_demo.html"

    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--html-output",
        str(output),
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"]["html"] == str(output)
    html = output.read_text(encoding="utf-8")
    assert "AdaMem State Authority Demo" in html
    assert "Evidence Boundary" in html
    assert "demo-data" in html
    assert "not paper evidence" in html
    assert "No answer model is called" in html
    assert "semantic_state_adjudication_trace" in html
    assert "state_adjudication" in html


def test_demo_paper_profile_html_includes_baseline_sources(tmp_path: Path, capsys) -> None:
    output = tmp_path / "adamem_state_demo_paper.html"

    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--html-output",
        str(output),
    ])

    message = capsys.readouterr().out
    assert str(output) in message
    html = output.read_text(encoding="utf-8")
    assert "A-MEM" in html
    assert "Zep/Graphiti" in html
    assert "Mem0" in html
    assert "api_free_approximation" in html
