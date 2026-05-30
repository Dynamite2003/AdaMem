from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert payload["provenance"]["schema_version"] == "adamem.demo_provenance.v1"
    assert payload["provenance"]["command"] == [
        "adamem",
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--query-id",
        "current_runtime_status",
        "--json",
    ]
    assert len(payload["provenance"]["payload_sha256"]) == 64
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
    assert "Artifact Provenance" in html
    assert payload["provenance"]["payload_sha256"] in html
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


def test_demo_bundle_writes_manifest_payload_and_html(tmp_path: Path, capsys) -> None:
    output = tmp_path / "bundle"

    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--bundle-output",
        str(output),
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    manifest = payload["bundle_manifest"]
    assert manifest["schema_version"] == "adamem.demo_bundle.v1"
    assert manifest["payload_sha256"] == payload["provenance"]["payload_sha256"]
    assert manifest["blocked_claims"]["answer_accuracy"]
    assert manifest["baseline_profile"] == "paper"
    assert manifest["query_count"] == 9

    html_path = Path(manifest["artifacts"]["html"])
    payload_path = Path(manifest["artifacts"]["payload_json"])
    manifest_path = Path(manifest["artifacts"]["bundle_manifest"])
    assert html_path.exists()
    assert payload_path.exists()
    assert manifest_path.exists()
    assert json.loads(payload_path.read_text(encoding="utf-8"))["provenance"]["payload_sha256"] == (
        manifest["payload_sha256"]
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert "Artifact Provenance" in html_path.read_text(encoding="utf-8")


def test_verify_demo_bundle_accepts_valid_bundle(tmp_path: Path, capsys) -> None:
    output = tmp_path / "bundle"
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--bundle-output",
        str(output),
        "--json",
    ])
    capsys.readouterr()

    main(["verify-demo", str(output), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "adamem.demo_bundle_verification.v1"
    assert report["valid"] is True
    assert report["checks"]["manifest_schema"] is True
    assert report["checks"]["payload_hash_matches"] is True
    assert report["checks"]["html_embeds_demo_data"] is True
    assert report["checks"]["blocked_claims_present"] is True
    assert report["blocked_claims"]["answer_accuracy"]
    assert report["baseline_names"] == [
        "semantic_only",
        "a_mem_evolution",
        "zep_temporal_kg",
        "mem0_extraction",
        "semantic_state_adjudication_trace",
    ]


def test_verify_demo_bundle_detects_tampered_payload(tmp_path: Path, capsys) -> None:
    output = tmp_path / "bundle"
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--bundle-output",
        str(output),
        "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    payload_path = Path(payload["bundle_manifest"]["artifacts"]["payload_json"])
    payload_json = json.loads(payload_path.read_text(encoding="utf-8"))
    payload_json["case_id"] = "tampered_case"
    payload_path.write_text(
        json.dumps(payload_json, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(["verify-demo", str(output), "--json"])

    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert report["checks"]["payload_hash_matches"] is False
    assert "payload hash mismatch" in "\n".join(report["errors"])


def test_demo_readiness_marks_verified_bundle_walkthrough_ready_not_paper_ready(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "bundle"
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--bundle-output",
        str(output),
        "--json",
    ])
    capsys.readouterr()

    main(["demo-readiness", str(output), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "adamem.demo_paper_readiness.v1"
    assert report["walkthrough_ready"] is True
    assert report["paper_claim_ready"] is False
    assert report["demo_verification_valid"] is True
    assert all(report["checklist"].values())
    assert report["supported_claims"] == ["interactive_demo_walkthrough_ready"]
    assert "answer_accuracy" in report["blocked_paper_claims"]
    assert "sota" in report["blocked_paper_claims"]
    assert "generality" in report["blocked_paper_claims"]
    assert "end_to_end_answer_evaluation" in report["blocked_paper_claims"]
    assert "official_or_faithful_mainstream_baselines" in report["blocked_paper_claims"]
    assert report["mainstream_api_free_approximations"] == [
        "a_mem_evolution",
        "mem0_extraction",
        "zep_temporal_kg",
    ]


def test_demo_readiness_fails_when_bundle_verification_fails(tmp_path: Path, capsys) -> None:
    output = tmp_path / "bundle"
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--bundle-output",
        str(output),
        "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    payload_path = Path(payload["bundle_manifest"]["artifacts"]["payload_json"])
    payload_json = json.loads(payload_path.read_text(encoding="utf-8"))
    payload_json["case_id"] = "tampered_case"
    payload_path.write_text(
        json.dumps(payload_json, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(["demo-readiness", str(output), "--json"])

    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["walkthrough_ready"] is False
    assert report["paper_claim_ready"] is False
    assert report["demo_verification_valid"] is False
    assert report["checklist"]["demo_bundle_verified"] is False
    assert report["blocked_paper_claims"][0] == "demo_bundle_verification_failed"


def test_demo_readiness_uses_external_not_ready_evidence_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "bundle"
    evidence = tmp_path / "report.manifest.json"
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--bundle-output",
        str(output),
        "--json",
    ])
    capsys.readouterr()
    evidence.write_text(
        json.dumps(
            {
                "paper_readiness": {
                    "status": "answer_candidate_with_model_coverage",
                    "paper_claim_ready": False,
                    "paper_claim_blockers": [
                        "official_or_faithful_baseline_reproduction",
                    ],
                    "top_next_actions": [
                        {
                            "action": "add_official_or_faithful_baseline_reproduction",
                            "count": 1,
                        }
                    ],
                    "experiment_count": 2,
                }
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    main([
        "demo-readiness",
        str(output),
        "--evidence-manifest",
        str(evidence),
        "--json",
    ])

    report = json.loads(capsys.readouterr().out)
    assert report["walkthrough_ready"] is True
    assert report["paper_claim_ready"] is False
    assert report["external_evidence_ready"] is False
    assert report["external_evidence_manifest_count"] == 1
    assert report["external_evidence_manifests"][0]["status"] == (
        "answer_candidate_with_model_coverage"
    )
    assert report["blocked_paper_claims"] == [
        "external_evidence_not_paper_ready",
        "official_or_faithful_baseline_reproduction",
        "mainstream_approximations_not_sota_ready",
    ]
    assert report["next_actions"] == [
        "add_official_or_faithful_baseline_reproduction",
    ]


def test_demo_readiness_marks_paper_ready_with_ready_external_evidence(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "bundle"
    readiness = tmp_path / "paper_readiness.json"
    main([
        "demo",
        "--dataset",
        "benchmarks/dynamic_state_transfer.jsonl",
        "--all-queries",
        "--baseline-profile",
        "paper",
        "--bundle-output",
        str(output),
        "--json",
    ])
    capsys.readouterr()
    readiness.write_text(
        json.dumps(
            {
                "status": "sota_candidate_with_model_coverage",
                "paper_claim_ready": True,
                "paper_claim_blockers": [],
                "experiment_count": 4,
                "benchmark_coverage_complete": True,
                "method_coverage_complete": True,
                "complete_study_model_group_count": 2,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    main([
        "demo-readiness",
        str(output),
        "--evidence-manifest",
        str(readiness),
        "--json",
    ])

    report = json.loads(capsys.readouterr().out)
    assert report["walkthrough_ready"] is True
    assert report["paper_claim_ready"] is True
    assert report["external_evidence_ready"] is True
    assert report["blocked_paper_claims"] == []
    assert report["supported_claims"] == [
        "interactive_demo_walkthrough_ready",
        "interactive_demo_backed_by_paper_ready_evidence",
    ]
    assert report["next_actions"] == [
        "attach_paper_readiness_report_to_demo_walkthrough",
    ]
