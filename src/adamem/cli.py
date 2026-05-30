from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adamem.baselines import baseline_registry
from adamem.bench import MemoryQACase, QuerySpec, load_jsonl_cases
from adamem.demo_html import write_demo_html
from adamem.experiments import current_git_commit
from adamem.manager import AdaMem
from adamem.schema import MemoryItem, MemoryResult
from adamem.store import JsonMemoryStore


_DEMO_BASELINES = ("semantic_state_adjudication", "semantic_state_adjudication_trace")
_DEMO_BASELINE_PROFILES = {
    "focused": _DEMO_BASELINES,
    "paper": (
        "semantic_only",
        "a_mem_evolution",
        "zep_temporal_kg",
        "mem0_extraction",
        "semantic_state_adjudication_trace",
    ),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tiny AdaMem prototype CLI")
    parser.add_argument("--store", default=".adamem/memory.json", help="JSON store path")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Add a memory")
    add.add_argument("content")
    add.add_argument("--kind", default="observation")
    add.add_argument("--importance", type=float, default=0.5)
    add.add_argument("--key", default=None)

    ask = sub.add_parser("retrieve", help="Retrieve memory context")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=6)
    ask.add_argument("--max-chars", type=int, default=1800)

    demo = sub.add_parser("demo", help="Run an API-free stale-memory mechanism demo")
    demo.add_argument("--dataset", default="benchmarks/dynamic_state_transfer.jsonl")
    demo.add_argument("--case-id", default="dynamic_state_transfer")
    demo.add_argument("--query-id", default="current_runtime_status")
    demo.add_argument(
        "--all-queries",
        action="store_true",
        help="Run every query in the selected case instead of one query",
    )
    demo.add_argument("--top-k", type=int, default=None)
    demo.add_argument(
        "--baseline-profile",
        choices=sorted(_DEMO_BASELINE_PROFILES),
        default="focused",
        help="Choose a predefined demo baseline matrix",
    )
    demo.add_argument(
        "--baselines",
        nargs="+",
        help="Override the baseline profile with explicit baseline names",
    )
    demo.add_argument("--json", action="store_true", help="Emit a machine-readable demo artifact")
    demo.add_argument("--html-output", help="Write a self-contained interactive HTML demo")
    demo.add_argument("--bundle-output", help="Write a demo bundle directory with HTML, payload JSON, and manifest")

    verify_demo = sub.add_parser("verify-demo", help="Verify an AdaMem demo bundle")
    verify_demo.add_argument("bundle", help="Path to a demo bundle directory or demo_manifest.json")
    verify_demo.add_argument("--json", action="store_true", help="Emit a machine-readable verification report")

    demo_readiness = sub.add_parser("demo-readiness", help="Audit paper-readiness boundaries for a demo bundle")
    demo_readiness.add_argument("bundle", help="Path to a demo bundle directory or demo_manifest.json")
    demo_readiness.add_argument(
        "--evidence-manifest",
        action="append",
        default=[],
        help=(
            "Attach a reporting bundle manifest, batch_manifest.json, or "
            "paper_readiness.json as external paper evidence"
        ),
    )
    demo_readiness.add_argument("--json", action="store_true", help="Emit a machine-readable readiness report")
    demo_readiness.add_argument("--output", help="Write the readiness report to this path")

    args = parser.parse_args(argv)
    command = ["adamem", *(argv or sys.argv[1:])]

    if args.command == "add":
        mem = AdaMem(store=JsonMemoryStore(args.store))
        metadata = {"memory_key": args.key} if args.key else None
        item = mem.observe(args.content, kind=args.kind, importance=args.importance, metadata=metadata)
        print(item.id)
    elif args.command == "retrieve":
        mem = AdaMem(store=JsonMemoryStore(args.store))
        print(mem.context(args.query, top_k=args.top_k, max_chars=args.max_chars))
    elif args.command == "demo":
        try:
            payload = _run_demo(
                args.dataset,
                case_id=args.case_id,
                query_id=args.query_id,
                all_queries=args.all_queries,
                top_k=args.top_k,
                baseline_names=tuple(args.baselines or _DEMO_BASELINE_PROFILES[args.baseline_profile]),
                baseline_profile=args.baseline_profile if not args.baselines else "custom",
            )
        except ValueError as exc:
            parser.error(str(exc))
        _attach_demo_provenance(payload, command=command)
        if args.bundle_output:
            bundle = _write_demo_bundle(payload, args.bundle_output)
            payload.setdefault("artifacts", {}).update(bundle["artifacts"])
            payload["bundle_manifest"] = bundle
        if args.html_output:
            html_path = write_demo_html(payload, args.html_output)
            payload.setdefault("artifacts", {})["html"] = str(html_path)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.bundle_output:
            print(f"wrote demo bundle: {payload['artifacts']['bundle_manifest']}")
        elif args.html_output:
            print(f"wrote HTML demo: {payload['artifacts']['html']}")
        else:
            print(_format_demo(payload))
    elif args.command == "verify-demo":
        report = _verify_demo_bundle(args.bundle)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(_format_demo_bundle_verification(report))
        if not report["valid"]:
            raise SystemExit(1)
    elif args.command == "demo-readiness":
        report = _demo_bundle_paper_readiness(
            args.bundle,
            evidence_manifests=tuple(args.evidence_manifest),
        )
        rendered = (
            json.dumps(report, indent=2, sort_keys=True)
            if args.json
            else _format_demo_bundle_paper_readiness(report)
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
        if args.json:
            print(rendered)
        else:
            print(rendered)
        if not report["walkthrough_ready"]:
            raise SystemExit(1)


def _run_demo(
    dataset: str,
    *,
    case_id: str,
    query_id: str,
    all_queries: bool,
    top_k: int | None,
    baseline_names: tuple[str, ...],
    baseline_profile: str,
) -> dict[str, Any]:
    case = _select_case(load_jsonl_cases(dataset), case_id)
    queries = case.queries if all_queries else [_select_query(case, query_id)]
    _validate_demo_baselines(baseline_names)
    query_payloads = [
        _run_demo_query(case, query, top_k=top_k, baseline_names=baseline_names)
        for query in queries
    ]
    common = {
        "schema_version": "adamem.demo.v1",
        "claim_boundary": (
            "API-free mechanism demo only; not paper evidence, not SOTA evidence, "
            "and not end-to-end answer accuracy."
        ),
        "evidence_boundary": _demo_evidence_boundary(),
        "dataset": dataset,
        "case_id": case.id,
        "baseline_profile": baseline_profile,
        "baseline_names": list(baseline_names),
        "comparison_note": (
            "The trace baseline should surface a state_adjudication notice when "
            "query-scoped state authority suppresses stale raw evidence."
        ),
    }
    if all_queries:
        return {
            **common,
            "mode": "all_queries",
            "query_count": len(query_payloads),
            "summary": _demo_summary(query_payloads),
            "queries": query_payloads,
        }
    return {
        **common,
        "mode": "single_query",
        **query_payloads[0],
    }


def _run_demo_query(
    case: MemoryQACase,
    query: QuerySpec,
    *,
    top_k: int | None,
    baseline_names: tuple[str, ...],
) -> dict[str, Any]:
    query_top_k = top_k if top_k is not None else query.top_k
    specs = baseline_registry()
    baseline_payloads = []
    for name in baseline_names:
        spec = specs[name]
        mem = AdaMem(config=spec.config)
        source_labels = _observe_case(mem, case)
        results = mem.retrieve(query.query, top_k=query_top_k, now=query.now)
        retrieved = [result.item.content for result in results]
        trace = _demo_trace(results, mem=mem, source_labels=source_labels)
        baseline_payloads.append(
            {
                "name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "source_name": spec.source_name,
                "source_url": spec.source_url,
                "implementation_status": spec.implementation_status,
                "reproduction_note": spec.reproduction_note,
                "passed": _retrieval_support_passed(
                    retrieved,
                    expected=query.expected_substrings,
                    forbidden=query.forbidden_substrings,
                ),
                "retrieved": retrieved,
                "trace": trace,
            }
        )
    return {
        "query_id": query.id or query.query,
        "query": query.query,
        "top_k": query_top_k,
        "expected_substrings": query.expected_substrings,
        "forbidden_substrings": query.forbidden_substrings,
        "baselines": baseline_payloads,
    }


def _validate_demo_baselines(baseline_names: tuple[str, ...]) -> None:
    if not baseline_names:
        raise ValueError("at least one demo baseline is required")
    registry = baseline_registry()
    unknown = [name for name in baseline_names if name not in registry]
    if unknown:
        available = ", ".join(sorted(registry))
        raise ValueError(f"unknown demo baseline(s): {', '.join(unknown)}; available: {available}")


def _demo_summary(query_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_baseline: dict[str, dict[str, Any]] = {}
    for query_payload in query_payloads:
        for baseline in query_payload["baselines"]:
            name = baseline["name"]
            row = by_baseline.setdefault(
                name,
                {
                    "passed": 0,
                    "total": 0,
                    "state_adjudication_traces": 0,
                    "state_slots": [],
                    "failed_query_ids": [],
                },
            )
            row["total"] += 1
            if baseline["passed"]:
                row["passed"] += 1
            else:
                row["failed_query_ids"].append(query_payload["query_id"])
            for trace_item in baseline["trace"]:
                if trace_item["kind"] == "state_adjudication":
                    row["state_adjudication_traces"] += 1
                state_slot = trace_item["metadata"].get("state_slot")
                if state_slot and state_slot not in row["state_slots"]:
                    row["state_slots"].append(state_slot)
    for row in by_baseline.values():
        total = row["total"]
        row["accuracy"] = row["passed"] / total if total else 0.0
        row["state_slots"] = sorted(row["state_slots"])
    return {
        "baseline_count": len(by_baseline),
        "query_count": len(query_payloads),
        "by_baseline": by_baseline,
    }


def _demo_evidence_boundary() -> dict[str, Any]:
    return {
        "artifact_type": "api_free_mechanism_demo",
        "supported_uses": [
            "Inspect state-authority and stale-source adjudication traces.",
            "Run local regression checks across deterministic state-family fixtures.",
            "Prepare a qualitative walkthrough before API-backed evaluation.",
        ],
        "blocked_claims": {
            "answer_accuracy": [
                "No answer model is called.",
                "No judge model or semantic scorer is called.",
            ],
            "sota": [
                "The demo uses a local fixture, not a full public benchmark.",
                "Mainstream baselines are not official or faithful reproductions in this artifact.",
            ],
            "generality": [
                "Transfer to STALE, LongMemEval, AMA-Bench, or other public benchmarks is not shown by this artifact.",
            ],
        },
        "next_evidence": [
            "Run STALE answer/judge experiments with multiple answer and judge models.",
            "Run the same mechanism matrix on at least one public transfer benchmark.",
            "Attach claim-audit and paper-readiness artifacts before making paper claims.",
        ],
    }


def _attach_demo_provenance(payload: dict[str, Any], *, command: list[str]) -> None:
    payload["provenance"] = {
        "schema_version": "adamem.demo_provenance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": current_git_commit(),
        "command": command,
        "payload_hash_algorithm": "sha256",
        "payload_hash_scope": "full demo payload before provenance/artifacts attachment",
        "payload_sha256": _demo_payload_hash(payload),
    }


def _demo_payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _demo_hash_scope_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scoped = dict(payload)
    scoped.pop("provenance", None)
    scoped.pop("artifacts", None)
    scoped.pop("bundle_manifest", None)
    return scoped


def _write_demo_bundle(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "index.html"
    payload_path = output / "demo_payload.json"
    manifest_path = output / "demo_manifest.json"
    artifacts = {
        "html": str(html_path),
        "payload_json": str(payload_path),
        "bundle_manifest": str(manifest_path),
    }
    payload.setdefault("artifacts", {}).update(artifacts)
    write_demo_html(payload, html_path)
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _demo_bundle_manifest(payload, output_dir=output, artifacts=artifacts)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _demo_bundle_manifest(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    provenance = payload.get("provenance") or {}
    evidence_boundary = payload.get("evidence_boundary") or {}
    summary = payload.get("summary") or {}
    return {
        "schema_version": "adamem.demo_bundle.v1",
        "output_dir": str(output_dir),
        "created_at": provenance.get("created_at"),
        "commit": provenance.get("commit"),
        "command": provenance.get("command") or [],
        "payload_sha256": provenance.get("payload_sha256"),
        "payload_hash_algorithm": provenance.get("payload_hash_algorithm"),
        "payload_hash_scope": provenance.get("payload_hash_scope"),
        "dataset": payload.get("dataset"),
        "case_id": payload.get("case_id"),
        "mode": payload.get("mode"),
        "baseline_profile": payload.get("baseline_profile"),
        "baseline_names": payload.get("baseline_names") or [],
        "query_count": payload.get("query_count", 1),
        "summary": summary,
        "claim_boundary": payload.get("claim_boundary"),
        "blocked_claims": evidence_boundary.get("blocked_claims") or {},
        "artifacts": artifacts,
    }


def _verify_demo_bundle(bundle: str | Path) -> dict[str, Any]:
    bundle_path = Path(bundle)
    manifest_path = bundle_path / "demo_manifest.json" if bundle_path.is_dir() else bundle_path
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    artifacts: dict[str, str] = {}

    def check(name: str, condition: bool, error: str | None = None) -> None:
        checks[name] = bool(condition)
        if not condition and error:
            errors.append(error)

    check("manifest_exists", manifest_path.exists(), f"manifest not found: {manifest_path}")
    if manifest_path.exists():
        manifest = _read_json_object(manifest_path, errors=errors, label="manifest")
    check(
        "manifest_json_object",
        bool(manifest),
        f"manifest is missing or is not a JSON object: {manifest_path}",
    )
    check(
        "manifest_schema",
        manifest.get("schema_version") == "adamem.demo_bundle.v1",
        "manifest schema_version must be adamem.demo_bundle.v1",
    )

    artifacts_raw = manifest.get("artifacts") if manifest else {}
    if isinstance(artifacts_raw, dict):
        artifacts = {str(key): str(value) for key, value in artifacts_raw.items()}
    check("artifacts_object", bool(artifacts), "manifest artifacts must be a non-empty object")

    artifact_paths = {
        name: _resolve_demo_artifact_path(path, manifest_path.parent)
        for name, path in artifacts.items()
    }
    for name in ("html", "payload_json", "bundle_manifest"):
        artifact_path = artifact_paths.get(name)
        check(
            f"artifact_{name}_exists",
            artifact_path is not None and artifact_path.exists(),
            f"artifact {name!r} not found: {artifacts.get(name)}",
        )

    payload_path = artifact_paths.get("payload_json")
    if payload_path and payload_path.exists():
        payload = _read_json_object(payload_path, errors=errors, label="payload")
    check(
        "payload_json_object",
        bool(payload),
        f"payload is missing or is not a JSON object: {payload_path}",
    )
    check(
        "payload_schema",
        payload.get("schema_version") == "adamem.demo.v1",
        "payload schema_version must be adamem.demo.v1",
    )

    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    check(
        "provenance_schema",
        provenance.get("schema_version") == "adamem.demo_provenance.v1",
        "payload provenance schema_version must be adamem.demo_provenance.v1",
    )
    expected_hash = provenance.get("payload_sha256")
    manifest_hash = manifest.get("payload_sha256")
    recomputed_hash = _demo_payload_hash(_demo_hash_scope_payload(payload)) if payload else None
    check(
        "payload_hash_matches",
        bool(expected_hash)
        and expected_hash == manifest_hash
        and recomputed_hash == expected_hash,
        "payload hash mismatch against provenance, manifest, or recomputed payload",
    )

    evidence_boundary = (
        payload.get("evidence_boundary")
        if isinstance(payload.get("evidence_boundary"), dict)
        else {}
    )
    blocked_claims = (
        evidence_boundary.get("blocked_claims")
        if isinstance(evidence_boundary.get("blocked_claims"), dict)
        else {}
    )
    required_blocked_claims = {"answer_accuracy", "sota", "generality"}
    check(
        "blocked_claims_present",
        required_blocked_claims.issubset(blocked_claims),
        "payload evidence_boundary must block answer_accuracy, sota, and generality claims",
    )
    check(
        "manifest_blocked_claims_match_payload",
        (manifest.get("blocked_claims") or {}) == blocked_claims,
        "manifest blocked_claims must match payload evidence_boundary blocked_claims",
    )
    check(
        "baseline_names_match",
        (manifest.get("baseline_names") or []) == (payload.get("baseline_names") or []),
        "manifest baseline_names must match payload baseline_names",
    )
    check(
        "summary_matches",
        (manifest.get("summary") or {}) == (payload.get("summary") or {}),
        "manifest summary must match payload summary",
    )
    check(
        "query_count_matches",
        manifest.get("query_count") == payload.get("query_count", 1),
        "manifest query_count must match payload query_count",
    )

    html_path = artifact_paths.get("html")
    html_text = ""
    if html_path and html_path.exists():
        html_text = html_path.read_text(encoding="utf-8")
    check(
        "html_embeds_demo_data",
        "id=\"demo-data\"" in html_text and "adamem.demo.v1" in html_text,
        "HTML artifact must embed the demo-data JSON payload",
    )
    check(
        "html_contains_payload_hash",
        bool(expected_hash) and str(expected_hash) in html_text,
        "HTML artifact must contain the payload provenance hash",
    )

    valid = all(checks.values()) and not errors
    return {
        "schema_version": "adamem.demo_bundle_verification.v1",
        "bundle": str(bundle_path),
        "manifest": str(manifest_path),
        "valid": valid,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "payload_sha256": expected_hash,
        "manifest_payload_sha256": manifest_hash,
        "recomputed_payload_sha256": recomputed_hash,
        "claim_boundary": payload.get("claim_boundary"),
        "blocked_claims": blocked_claims,
        "baseline_names": payload.get("baseline_names") or [],
    }


def _demo_bundle_paper_readiness(
    bundle: str | Path,
    *,
    evidence_manifests: tuple[str, ...] = (),
) -> dict[str, Any]:
    verification = _verify_demo_bundle(bundle)
    payload = _load_verified_demo_payload(verification)
    evidence_boundary = (
        payload.get("evidence_boundary")
        if isinstance(payload.get("evidence_boundary"), dict)
        else {}
    )
    blocked_claims = (
        evidence_boundary.get("blocked_claims")
        if isinstance(evidence_boundary.get("blocked_claims"), dict)
        else {}
    )
    baselines = _demo_payload_baselines(payload)
    baseline_statuses = {
        baseline["name"]: baseline.get("implementation_status")
        for baseline in baselines
        if baseline.get("name")
    }
    mainstream_approximations = sorted(
        name
        for name, status in baseline_statuses.items()
        if status == "api_free_approximation"
    )
    checklist = {
        "demo_bundle_verified": bool(verification.get("valid")),
        "interactive_html_verified": bool(
            verification.get("checks", {}).get("html_embeds_demo_data")
            and verification.get("checks", {}).get("html_contains_payload_hash")
        ),
        "evidence_boundary_present": all(
            key in blocked_claims
            for key in ("answer_accuracy", "sota", "generality")
        ),
        "artifact_provenance_verified": bool(
            verification.get("checks", {}).get("provenance_schema")
            and verification.get("checks", {}).get("payload_hash_matches")
        ),
        "baseline_provenance_present": bool(baseline_statuses)
        and all(status for status in baseline_statuses.values()),
    }
    walkthrough_ready = all(checklist.values())
    evidence_reports = _demo_paper_evidence_reports(evidence_manifests)
    evidence_ready = bool(evidence_reports) and all(
        report.get("paper_claim_ready") for report in evidence_reports
    )
    paper_claim_ready = walkthrough_ready and evidence_ready
    paper_claim_blockers = _demo_paper_claim_blockers(
        blocked_claims=blocked_claims,
        evidence_reports=evidence_reports,
        evidence_ready=evidence_ready,
    )
    if mainstream_approximations and "mainstream_approximations_not_sota_ready" not in paper_claim_blockers:
        if not evidence_ready:
            paper_claim_blockers.append("mainstream_approximations_not_sota_ready")
    if not verification.get("valid"):
        paper_claim_blockers.insert(0, "demo_bundle_verification_failed")

    next_actions = _demo_readiness_next_actions(
        verification_valid=bool(verification.get("valid")),
        evidence_reports=evidence_reports,
        evidence_ready=evidence_ready,
    )
    if not verification.get("valid"):
        next_actions.insert(0, "fix_demo_bundle_verification")

    supported_claims = ["interactive_demo_walkthrough_ready"] if walkthrough_ready else []
    if paper_claim_ready:
        supported_claims.append("interactive_demo_backed_by_paper_ready_evidence")
    return {
        "schema_version": "adamem.demo_paper_readiness.v1",
        "bundle": str(bundle),
        "walkthrough_ready": walkthrough_ready,
        "paper_claim_ready": paper_claim_ready,
        "demo_verification_valid": bool(verification.get("valid")),
        "external_evidence_ready": evidence_ready,
        "external_evidence_manifest_count": len(evidence_reports),
        "external_evidence_manifests": evidence_reports,
        "checklist": checklist,
        "supported_claims": supported_claims,
        "blocked_paper_claims": paper_claim_blockers,
        "evidence_boundary": evidence_boundary,
        "baseline_profile": payload.get("baseline_profile"),
        "baseline_names": payload.get("baseline_names") or [],
        "baseline_implementation_statuses": baseline_statuses,
        "mainstream_api_free_approximations": mainstream_approximations,
        "next_actions": next_actions,
        "demo_verification": verification,
    }


def _demo_paper_claim_blockers(
    *,
    blocked_claims: dict[str, Any],
    evidence_reports: list[dict[str, Any]],
    evidence_ready: bool,
) -> list[str]:
    if evidence_ready:
        return []
    if evidence_reports:
        blockers = ["external_evidence_not_paper_ready"]
        for report in evidence_reports:
            blockers.extend(str(blocker) for blocker in report.get("paper_claim_blockers") or [])
        return list(dict.fromkeys(blockers))

    blockers = list(blocked_claims.keys())
    for blocker in (
        "end_to_end_answer_evaluation",
        "official_or_faithful_mainstream_baselines",
        "public_benchmark_generality",
        "multi_model_judge_robustness",
    ):
        if blocker not in blockers:
            blockers.append(blocker)
    return blockers


def _demo_readiness_next_actions(
    *,
    verification_valid: bool,
    evidence_reports: list[dict[str, Any]],
    evidence_ready: bool,
) -> list[str]:
    if evidence_ready:
        return ["attach_paper_readiness_report_to_demo_walkthrough"]
    if evidence_reports:
        actions = []
        for report in evidence_reports:
            actions.extend(str(action) for action in report.get("next_actions") or [])
        return list(dict.fromkeys(actions)) or [
            "resolve_external_paper_readiness_blockers",
        ]
    if verification_valid:
        return [
            "run_stale_answer_judge_evaluation",
            "replace_or_validate_mainstream_approximation_baselines",
            "run_public_transfer_benchmark",
            "add_multi_answer_and_judge_model_robustness",
        ]
    return [
        "run_stale_answer_judge_evaluation",
        "replace_or_validate_mainstream_approximation_baselines",
        "run_public_transfer_benchmark",
        "add_multi_answer_and_judge_model_robustness",
    ]


def _demo_paper_evidence_reports(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    return [_demo_paper_evidence_report(Path(path)) for path in paths]


def _demo_paper_evidence_report(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    payload = _read_json_object(path, errors=errors, label="evidence manifest")
    readiness = payload.get("paper_readiness") if isinstance(payload.get("paper_readiness"), dict) else payload
    if not isinstance(readiness, dict):
        readiness = {}
    blockers = list(
        readiness.get("paper_claim_blockers")
        or payload.get("paper_claim_blockers")
        or []
    )
    top_actions = readiness.get("top_next_actions") or []
    next_actions = [
        str(item.get("action"))
        for item in top_actions
        if isinstance(item, dict) and item.get("action")
    ]
    if not next_actions:
        action_counts = readiness.get("action_counts") or {}
        if isinstance(action_counts, dict):
            next_actions = [str(action) for action in action_counts]
    paper_claim_ready = bool(
        readiness.get("paper_claim_ready")
        if "paper_claim_ready" in readiness
        else payload.get("paper_claim_ready")
    )
    if errors:
        paper_claim_ready = False
        blockers = list(dict.fromkeys([*blockers, "evidence_manifest_unreadable"]))
        next_actions = list(dict.fromkeys([*next_actions, "fix_evidence_manifest"]))
    return {
        "path": str(path),
        "schema_version": payload.get("schema_version"),
        "status": readiness.get("status"),
        "paper_claim_ready": paper_claim_ready,
        "paper_claim_blockers": blockers,
        "experiment_count": readiness.get("experiment_count") or payload.get("experiment_count"),
        "benchmark_coverage_complete": readiness.get("benchmark_coverage_complete"),
        "method_coverage_complete": readiness.get("method_coverage_complete"),
        "complete_study_model_group_count": readiness.get("complete_study_model_group_count"),
        "next_actions": next_actions,
        "errors": errors,
    }


def _load_verified_demo_payload(verification: dict[str, Any]) -> dict[str, Any]:
    payload_path_text = (verification.get("artifacts") or {}).get("payload_json")
    if not payload_path_text:
        return {}
    payload_path = Path(str(payload_path_text))
    if not payload_path.exists():
        return {}
    return _read_json_object(payload_path, errors=[], label="payload")


def _demo_payload_baselines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("baselines"), list):
        return [baseline for baseline in payload["baselines"] if isinstance(baseline, dict)]
    baselines: list[dict[str, Any]] = []
    for query in payload.get("queries") or []:
        if not isinstance(query, dict):
            continue
        for baseline in query.get("baselines") or []:
            if isinstance(baseline, dict) and baseline.get("name"):
                name = str(baseline["name"])
                if not any(existing.get("name") == name for existing in baselines):
                    baselines.append(baseline)
    return baselines


def _read_json_object(path: Path, *, errors: list[str], label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        errors.append(f"could not read {label} JSON at {path}: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        errors.append(f"could not parse {label} JSON at {path}: {exc}")
        return {}
    if not isinstance(parsed, dict):
        errors.append(f"{label} JSON at {path} must be an object")
        return {}
    return parsed


def _resolve_demo_artifact_path(path: str, manifest_dir: Path) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute() or artifact_path.exists():
        return artifact_path
    return manifest_dir / artifact_path


def _select_case(cases: list[MemoryQACase], case_id: str) -> MemoryQACase:
    for case in cases:
        if case.id == case_id:
            return case
    available = ", ".join(case.id for case in cases) or "<none>"
    raise ValueError(f"case id {case_id!r} not found in dataset; available: {available}")


def _select_query(case: MemoryQACase, query_id: str) -> QuerySpec:
    for query in case.queries:
        if query.id == query_id:
            return query
    available = ", ".join(query.id or query.query for query in case.queries) or "<none>"
    raise ValueError(f"query id {query_id!r} not found in case {case.id!r}; available: {available}")


def _observe_case(mem: AdaMem, case: MemoryQACase) -> dict[str, str]:
    labels: dict[str, MemoryItem] = {}
    source_labels: dict[str, str] = {}
    for index, observation in enumerate(case.observations):
        cause_ids = [
            labels[label].id
            for label in observation.cause_labels
            if label in labels
        ]
        item = mem.observe(
            observation.content,
            kind=observation.kind,
            importance=observation.importance,
            confidence=observation.confidence,
            valid_from=observation.valid_from,
            valid_to=observation.valid_to,
            cause_ids=cause_ids,
            metadata=dict(observation.metadata),
        )
        label = observation.label or str(index)
        labels[label] = item
        source_labels[item.id] = label
    return source_labels


def _demo_trace(
    results: list[MemoryResult],
    *,
    mem: AdaMem,
    source_labels: dict[str, str],
) -> list[dict[str, Any]]:
    state_items_by_id = {item.id: item for item in mem.store.all() if item.kind == "state"}
    return [
        {
            "content": result.item.content,
            "kind": result.item.kind,
            "score": round(result.score, 4),
            "relation": result.relation,
            "contributions": {
                key: round(value, 4)
                for key, value in result.contributions.items()
            },
            "metadata": _demo_trace_metadata(
                result.item,
                source_labels=source_labels,
                state_items_by_id=state_items_by_id,
            ),
        }
        for result in results
    ]


def _demo_trace_metadata(
    item: MemoryItem,
    *,
    source_labels: dict[str, str],
    state_items_by_id: dict[str, MemoryItem],
) -> dict[str, Any]:
    keys = (
        "memory_key",
        "label",
        "benchmark",
        "trajectory_step",
        "subject",
        "state_slot",
        "state_value",
        "state_status",
        "invalidated_state_value",
        "dependency_invalidated_by_state_id",
        "dependency_invalidated_by_slot",
        "stale_value",
        "current_value",
        "kg_relation",
        "kg_object",
        "salient_slot",
        "salient_value",
        "source_id",
        "source_state_id",
        "stale_state_id",
        "adjudicated_source_id",
        "adjudication_reason",
        "derived",
    )
    metadata = {key: item.metadata[key] for key in keys if key in item.metadata}
    if source_id := metadata.get("source_id"):
        if source_label := source_labels.get(str(source_id)):
            metadata["source_observation_label"] = source_label
    if source_state_id := metadata.get("source_state_id"):
        if source_state := state_items_by_id.get(str(source_state_id)):
            if source_label := _state_source_label(source_state, source_labels):
                metadata["source_observation_label"] = source_label
            _copy_state_dependency_metadata(source_state, metadata)
    if stale_state_id := metadata.get("stale_state_id"):
        if stale_state := state_items_by_id.get(str(stale_state_id)):
            if stale_label := _state_source_label(stale_state, source_labels):
                metadata["stale_source_observation_label"] = stale_label
    if adjudicated_source_id := metadata.get("adjudicated_source_id"):
        if adjudicated_label := source_labels.get(str(adjudicated_source_id)):
            metadata["adjudicated_source_observation_label"] = adjudicated_label
    return metadata


def _copy_state_dependency_metadata(state: MemoryItem, metadata: dict[str, Any]) -> None:
    for key in ("dependency_invalidated_by_state_id", "dependency_invalidated_by_slot"):
        if key not in metadata and key in state.metadata:
            metadata[key] = state.metadata[key]


def _state_source_label(state: MemoryItem, source_labels: dict[str, str]) -> str | None:
    source_id = state.metadata.get("source_id")
    if source_id is None:
        return None
    return source_labels.get(str(source_id))


def _retrieval_support_passed(
    retrieved: list[str],
    *,
    expected: list[str],
    forbidden: list[str],
) -> bool:
    text = "\n".join(retrieved).lower()
    return all(item.lower() in text for item in expected) and not any(
        item.lower() in text
        for item in forbidden
    )


def _format_demo(payload: dict[str, Any]) -> str:
    if payload.get("mode") == "all_queries":
        return _format_all_query_demo(payload)
    lines = [
        "# AdaMem Stale-Memory Demo",
        "",
        f"Claim boundary: {payload['claim_boundary']}",
        f"Dataset: {payload['dataset']}",
        f"Case: {payload['case_id']}",
        f"Query: {payload['query_id']} - {payload['query']}",
        f"Expected substrings: {', '.join(payload['expected_substrings']) or '<none>'}",
        f"Forbidden substrings: {', '.join(payload['forbidden_substrings']) or '<none>'}",
        "",
    ]
    lines.extend(_format_evidence_boundary(payload))
    for baseline in payload["baselines"]:
        lines.extend([
            f"## {baseline['name']}",
            baseline["description"],
            f"Passed retrieval-support check: {baseline['passed']}",
            "Retrieved:",
        ])
        if baseline["retrieved"]:
            for index, content in enumerate(baseline["retrieved"], start=1):
                lines.append(f"{index}. {content}")
        else:
            lines.append("<none>")
        lines.append("Trace:")
        if baseline["trace"]:
            for item in baseline["trace"]:
                metadata = item["metadata"]
                slot = metadata.get("state_slot", "<none>")
                source = metadata.get("source_observation_label", "<none>")
                suppressed = metadata.get("adjudicated_source_observation_label", "<none>")
                lines.append(
                    "- "
                    f"kind={item['kind']} relation={item['relation']} score={item['score']} "
                    f"slot={slot} source={source} suppressed={suppressed}"
                )
        else:
            lines.append("<none>")
        lines.append("")
    lines.append(f"Comparison note: {payload['comparison_note']}")
    return "\n".join(lines).rstrip()


def _format_all_query_demo(payload: dict[str, Any]) -> str:
    lines = [
        "# AdaMem Stale-Memory Demo",
        "",
        f"Claim boundary: {payload['claim_boundary']}",
        f"Dataset: {payload['dataset']}",
        f"Case: {payload['case_id']}",
        f"Queries: {payload['query_count']}",
        "",
        "## Summary",
    ]
    for name, row in payload["summary"]["by_baseline"].items():
        failed = ", ".join(row["failed_query_ids"]) or "<none>"
        lines.append(
            "- "
            f"{name}: {row['passed']}/{row['total']} "
            f"({row['accuracy']:.2%}); "
            f"state_adjudication_traces={row['state_adjudication_traces']}; "
            f"failed={failed}"
        )
    lines.append("")
    lines.extend(_format_evidence_boundary(payload))
    lines.append("")
    lines.append("## Queries")
    for query_payload in payload["queries"]:
        status = ", ".join(
            f"{baseline['name']}={'PASS' if baseline['passed'] else 'FAIL'}"
            for baseline in query_payload["baselines"]
        )
        lines.append(f"- {query_payload['query_id']}: {status}")
        for baseline in query_payload["baselines"]:
            top_trace = baseline["trace"][0] if baseline["trace"] else None
            if top_trace is None:
                continue
            metadata = top_trace["metadata"]
            slot = metadata.get("state_slot", "<none>")
            source = metadata.get("source_observation_label", "<none>")
            suppressed = metadata.get("adjudicated_source_observation_label", "<none>")
            lines.append(
                "  "
                f"{baseline['name']} top_trace="
                f"{top_trace['kind']} slot={slot} source={source} suppressed={suppressed}"
            )
    lines.append("")
    lines.append(f"Comparison note: {payload['comparison_note']}")
    return "\n".join(lines).rstrip()


def _format_evidence_boundary(payload: dict[str, Any]) -> list[str]:
    boundary = payload.get("evidence_boundary") or {}
    if not boundary:
        return []
    lines = ["## Evidence Boundary"]
    supported = boundary.get("supported_uses") or []
    if supported:
        lines.append("Supported uses:")
        lines.extend(f"- {item}" for item in supported)
    blocked = boundary.get("blocked_claims") or {}
    if blocked:
        lines.append("Blocked claims:")
        for claim, reasons in blocked.items():
            reason_text = "; ".join(str(reason) for reason in reasons)
            lines.append(f"- {claim}: {reason_text}")
    next_evidence = boundary.get("next_evidence") or []
    if next_evidence:
        lines.append("Next evidence:")
        lines.extend(f"- {item}" for item in next_evidence)
    return lines


def _format_demo_bundle_verification(report: dict[str, Any]) -> str:
    lines = [
        "# AdaMem Demo Bundle Verification",
        "",
        f"Valid: {report['valid']}",
        f"Manifest: {report['manifest']}",
        f"Payload SHA-256: {report.get('payload_sha256') or '<missing>'}",
        "",
        "## Checks",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} {name}")
    if report["errors"]:
        lines.append("")
        lines.append("## Errors")
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines).rstrip()


def _format_demo_bundle_paper_readiness(report: dict[str, Any]) -> str:
    lines = [
        "# AdaMem Demo Paper Readiness",
        "",
        f"Walkthrough ready: `{bool(report.get('walkthrough_ready'))}`",
        f"Paper claim ready: `{bool(report.get('paper_claim_ready'))}`",
        f"Demo verification valid: `{bool(report.get('demo_verification_valid'))}`",
        "",
        "## Checklist",
    ]
    for name, passed in (report.get("checklist") or {}).items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} {name}")
    if report.get("supported_claims"):
        lines.append("")
        lines.append("## Supported Claims")
        lines.extend(f"- `{claim}`" for claim in report["supported_claims"])
    if report.get("blocked_paper_claims"):
        lines.append("")
        lines.append("## Blocked Paper Claims")
        lines.extend(f"- `{claim}`" for claim in report["blocked_paper_claims"])
    if report.get("external_evidence_manifests"):
        lines.append("")
        lines.append("## External Evidence")
        for evidence in report["external_evidence_manifests"]:
            lines.append(
                "- "
                f"`{evidence.get('path')}`: "
                f"ready=`{bool(evidence.get('paper_claim_ready'))}` "
                f"status=`{evidence.get('status') or '<missing>'}`"
            )
    if report.get("mainstream_api_free_approximations"):
        lines.append("")
        lines.append("## API-Free Mainstream Approximations")
        lines.extend(f"- `{name}`" for name in report["mainstream_api_free_approximations"])
    if report.get("next_actions"):
        lines.append("")
        lines.append("## Next Actions")
        lines.extend(f"- `{action}`" for action in report["next_actions"])
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    main()
