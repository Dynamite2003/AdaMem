from __future__ import annotations

import argparse
import json
from typing import Any

from adamem.baselines import baseline_registry
from adamem.bench import MemoryQACase, QuerySpec, load_jsonl_cases
from adamem.manager import AdaMem
from adamem.schema import MemoryItem, MemoryResult
from adamem.store import JsonMemoryStore


_DEMO_BASELINES = ("semantic_state_adjudication", "semantic_state_adjudication_trace")


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
    demo.add_argument("--json", action="store_true", help="Emit a machine-readable demo artifact")

    args = parser.parse_args(argv)

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
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_format_demo(payload))


def _run_demo(
    dataset: str,
    *,
    case_id: str,
    query_id: str,
    all_queries: bool,
    top_k: int | None,
) -> dict[str, Any]:
    case = _select_case(load_jsonl_cases(dataset), case_id)
    queries = case.queries if all_queries else [_select_query(case, query_id)]
    query_payloads = [_run_demo_query(case, query, top_k=top_k) for query in queries]
    common = {
        "schema_version": "adamem.demo.v1",
        "claim_boundary": (
            "API-free mechanism demo only; not paper evidence, not SOTA evidence, "
            "and not end-to-end answer accuracy."
        ),
        "dataset": dataset,
        "case_id": case.id,
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


def _run_demo_query(case: MemoryQACase, query: QuerySpec, *, top_k: int | None) -> dict[str, Any]:
    query_top_k = top_k if top_k is not None else query.top_k
    specs = baseline_registry()
    baseline_payloads = []
    for name in _DEMO_BASELINES:
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


if __name__ == "__main__":
    main()
