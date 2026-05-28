from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from adamem.config import AdaMemConfig
from adamem.manager import AdaMem
from adamem.schema import MemoryItem


@dataclass(slots=True)
class ObservationSpec:
    content: str
    label: str | None = None
    kind: str = "observation"
    importance: float = 0.5
    confidence: float = 1.0
    valid_from: str | None = None
    valid_to: str | None = None
    cause_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuerySpec:
    query: str
    expected_substrings: list[str]
    forbidden_substrings: list[str] = field(default_factory=list)
    id: str | None = None
    top_k: int = 4
    now: str | None = None


@dataclass(slots=True)
class MemoryQACase:
    id: str
    observations: list[ObservationSpec]
    queries: list[QuerySpec]


@dataclass(slots=True)
class QueryEvalResult:
    case_id: str
    query_id: str
    passed: bool
    retrieved: list[str]
    expected_substrings: list[str]
    forbidden_substrings: list[str]
    trace: list[dict[str, Any]]


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    accuracy: float
    passed: int
    total: int
    queries: list[QueryEvalResult]


def default_ablation_configs() -> dict[str, AdaMemConfig]:
    semantic_only = dict(
        use_graph=False,
        use_temporal=False,
        use_importance=False,
        use_recency=False,
        use_confidence=False,
        use_feedback=False,
        use_mmr=False,
        use_supersession=False,
        use_auto_links=False,
    )
    return {
        "semantic_only": AdaMemConfig(**semantic_only),
        "semantic_importance": AdaMemConfig(**{**semantic_only, "use_importance": True}),
        "semantic_temporal": AdaMemConfig(**{**semantic_only, "use_temporal": True}),
        "semantic_graph": AdaMemConfig(**{**semantic_only, "use_graph": True}),
        "delta_graph": AdaMemConfig(**{**semantic_only, "use_graph": True, "use_supersession": True}),
        "full": AdaMemConfig(),
    }


def load_jsonl_cases(path: str | Path) -> list[MemoryQACase]:
    cases: list[MemoryQACase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            cases.append(_case_from_mapping(raw, line_number=line_number))
    return cases


def run_benchmark(
    cases: Iterable[MemoryQACase],
    configs: dict[str, AdaMemConfig] | None = None,
) -> list[BenchmarkResult]:
    configs = configs or default_ablation_configs()
    case_list = list(cases)
    results: list[BenchmarkResult] = []
    for name, config in configs.items():
        query_results: list[QueryEvalResult] = []
        for case in case_list:
            query_results.extend(_run_case(case, config))
        passed = sum(1 for result in query_results if result.passed)
        total = len(query_results)
        results.append(
            BenchmarkResult(
                name=name,
                accuracy=passed / total if total else 0.0,
                passed=passed,
                total=total,
                queries=query_results,
            )
        )
    return results


def benchmark_report(results: list[BenchmarkResult]) -> str:
    lines = ["# AdaMem QA Ablation", ""]
    lines.append("| ablation | passed | accuracy |")
    lines.append("| --- | ---: | ---: |")
    for result in results:
        lines.append(f"| {result.name} | {result.passed}/{result.total} | {result.accuracy:.2%} |")
    lines.append("")
    for result in results:
        lines.append(f"## {result.name}")
        for query in result.queries:
            mark = "PASS" if query.passed else "FAIL"
            first = query.retrieved[0] if query.retrieved else "<none>"
            lines.append(f"- {mark} `{query.case_id}/{query.query_id}`: {first}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _run_case(case: MemoryQACase, config: AdaMemConfig) -> list[QueryEvalResult]:
    mem = AdaMem(config=config)
    labels: dict[str, MemoryItem] = {}
    for index, observation in enumerate(case.observations):
        cause_ids = [labels[label].id for label in observation.cause_labels]
        item = mem.observe(
            observation.content,
            kind=observation.kind,
            importance=observation.importance,
            confidence=observation.confidence,
            valid_from=observation.valid_from,
            valid_to=observation.valid_to,
            cause_ids=cause_ids,
            metadata=observation.metadata,
        )
        labels[observation.label or str(index)] = item
    return [_run_query(case.id, mem, query) for query in case.queries]


def _run_query(case_id: str, mem: AdaMem, query: QuerySpec) -> QueryEvalResult:
    results = mem.retrieve(query.query, top_k=query.top_k, now=query.now)
    retrieved = [result.item.content for result in results]
    text = "\n".join(retrieved).lower()
    has_expected = all(expected.lower() in text for expected in query.expected_substrings)
    has_forbidden = any(forbidden.lower() in text for forbidden in query.forbidden_substrings)
    trace = [
        {
            "content": result.item.content,
            "score": round(result.score, 4),
            "relation": result.relation,
            "contributions": {key: round(value, 4) for key, value in result.contributions.items()},
        }
        for result in results
    ]
    return QueryEvalResult(
        case_id=case_id,
        query_id=query.id or query.query,
        passed=has_expected and not has_forbidden,
        retrieved=retrieved,
        expected_substrings=query.expected_substrings,
        forbidden_substrings=query.forbidden_substrings,
        trace=trace,
    )


def _case_from_mapping(raw: dict[str, Any], *, line_number: int) -> MemoryQACase:
    case_id = str(raw.get("id") or f"line-{line_number}")
    observations = [
        ObservationSpec(
            label=entry.get("label"),
            content=entry["content"],
            kind=entry.get("kind", "observation"),
            importance=float(entry.get("importance", 0.5)),
            confidence=float(entry.get("confidence", 1.0)),
            valid_from=entry.get("valid_from"),
            valid_to=entry.get("valid_to"),
            cause_labels=list(entry.get("cause_labels", [])),
            metadata=dict(entry.get("metadata", {})),
        )
        for entry in raw.get("observations", [])
    ]
    queries = [
        QuerySpec(
            id=entry.get("id"),
            query=entry["query"],
            expected_substrings=list(entry.get("expected_substrings") or entry.get("answers") or []),
            forbidden_substrings=list(entry.get("forbidden_substrings", [])),
            top_k=int(entry.get("top_k", 4)),
            now=entry.get("now"),
        )
        for entry in raw.get("queries", raw.get("qas", []))
    ]
    return MemoryQACase(id=case_id, observations=observations, queries=queries)
