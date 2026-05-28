from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adamem.bench import (
    benchmark_report,
    default_ablation_configs,
    load_jsonl_cases,
    run_benchmark,
)
from adamem.config import AdaMemConfig
from adamem.manager import AdaMem
from adamem.schema import MemoryItem


@dataclass(slots=True)
class SyntheticObservation:
    label: str
    content: str
    kind: str = "observation"
    importance: float = 0.5
    confidence: float = 1.0
    valid_from: str | None = None
    valid_to: str | None = None
    cause_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SyntheticCase:
    name: str
    query: str
    observations: list[SyntheticObservation]
    expected_substrings: list[str]
    forbidden_substrings: list[str] = field(default_factory=list)
    top_k: int = 2
    now: str | None = None


@dataclass(slots=True)
class CaseResult:
    case: str
    passed: bool
    retrieved: list[str]
    expected_substrings: list[str]
    forbidden_substrings: list[str]
    trace: list[dict[str, Any]]


@dataclass(slots=True)
class AblationResult:
    name: str
    accuracy: float
    passed: int
    total: int
    cases: list[CaseResult]


def synthetic_cases() -> list[SyntheticCase]:
    now = "2026-05-28T00:00:00+00:00"
    return [
        SyntheticCase(
            name="stale_fact_supersession",
            query="What is the current deployment target?",
            top_k=1,
            observations=[
                SyntheticObservation(
                    label="old_target",
                    content="Deployment target: staging.",
                    metadata={"memory_key": "deploy.target"},
                ),
                SyntheticObservation(
                    label="distractor",
                    content="Staging database backups run nightly.",
                    metadata={"memory_key": "db.backup"},
                ),
                SyntheticObservation(
                    label="new_target",
                    content="Deployment target: production.",
                    metadata={"memory_key": "deploy.target"},
                ),
            ],
            expected_substrings=["production"],
            forbidden_substrings=["staging"],
        ),
        SyntheticCase(
            name="causal_bridge",
            query="Which credential fixed the resolved checkout incident?",
            top_k=3,
            observations=[
                SyntheticObservation(
                    label="cause",
                    content="C42 was absent from the vault.",
                    importance=0.9,
                    metadata={"memory_key": "vault.secret"},
                ),
                SyntheticObservation(
                    label="outcome",
                    content="Checkout incident resolved after credential remediation.",
                    importance=0.8,
                    cause_labels=["cause"],
                    metadata={"memory_key": "checkout.outcome"},
                ),
                SyntheticObservation(
                    label="noise",
                    content="Checkout analytics dashboard refreshes every hour.",
                    metadata={"memory_key": "checkout.dashboard"},
                ),
            ],
            expected_substrings=["c42"],
        ),
        SyntheticCase(
            name="temporal_validity",
            query="What is the office door code?",
            top_k=1,
            now=now,
            observations=[
                SyntheticObservation(
                    label="old_code",
                    content="Office door code is 1234.",
                    valid_to="2026-01-01T00:00:00+00:00",
                    metadata={"memory_key": "door.code.old"},
                ),
                SyntheticObservation(
                    label="new_code",
                    content="Office door code is 9876.",
                    valid_from="2026-01-02T00:00:00+00:00",
                    metadata={"memory_key": "door.code.new"},
                ),
            ],
            expected_substrings=["9876"],
            forbidden_substrings=["1234"],
        ),
        SyntheticCase(
            name="importance_over_frequency",
            query="What safety rule applies to schema changes?",
            top_k=1,
            observations=[
                SyntheticObservation(
                    label="critical_rule",
                    content="Create a rollback checkpoint before irreversible operations.",
                    importance=1.0,
                    metadata={
                        "memory_key": "migration.safety",
                        "tags": ["schema", "changes", "safety", "rule"],
                    },
                ),
                SyntheticObservation(
                    label="frequent_noise",
                    content="Schema changes are listed in the dashboard. Schema changes can be filtered by owner.",
                    importance=0.1,
                    metadata={"memory_key": "migration.dashboard"},
                ),
            ],
            expected_substrings=["rollback checkpoint"],
        ),
    ]


def ablation_configs() -> dict[str, AdaMemConfig]:
    return default_ablation_configs()


def run_synthetic_benchmark(
    configs: dict[str, AdaMemConfig] | None = None,
    cases: list[SyntheticCase] | None = None,
) -> list[AblationResult]:
    configs = configs or ablation_configs()
    cases = cases or synthetic_cases()
    results: list[AblationResult] = []
    for name, config in configs.items():
        case_results = [_run_case(config, case) for case in cases]
        passed = sum(1 for result in case_results if result.passed)
        total = len(case_results)
        results.append(
            AblationResult(
                name=name,
                accuracy=passed / total if total else 0.0,
                passed=passed,
                total=total,
                cases=case_results,
            )
        )
    return results


def _run_case(config: AdaMemConfig, case: SyntheticCase) -> CaseResult:
    mem = AdaMem(config=config)
    labels: dict[str, MemoryItem] = {}
    for observation in case.observations:
        cause_ids = [labels[label].id for label in observation.cause_labels]
        labels[observation.label] = mem.observe(
            observation.content,
            kind=observation.kind,
            importance=observation.importance,
            confidence=observation.confidence,
            valid_from=observation.valid_from,
            valid_to=observation.valid_to,
            cause_ids=cause_ids,
            metadata=observation.metadata,
        )
    results = mem.retrieve(case.query, top_k=case.top_k, now=case.now)
    retrieved = [result.item.content for result in results]
    text = "\n".join(retrieved).lower()
    has_expected = all(expected.lower() in text for expected in case.expected_substrings)
    has_forbidden = any(forbidden.lower() in text for forbidden in case.forbidden_substrings)
    trace = [
        {
            "content": result.item.content,
            "score": round(result.score, 4),
            "relation": result.relation,
            "contributions": {key: round(value, 4) for key, value in result.contributions.items()},
        }
        for result in results
    ]
    return CaseResult(
        case=case.name,
        passed=has_expected and not has_forbidden,
        retrieved=retrieved,
        expected_substrings=case.expected_substrings,
        forbidden_substrings=case.forbidden_substrings,
        trace=trace,
    )


def as_report(results: list[AblationResult]) -> str:
    lines = ["# AdaMem Synthetic Ablation", ""]
    lines.append("| ablation | passed | accuracy |")
    lines.append("| --- | ---: | ---: |")
    for result in results:
        lines.append(f"| {result.name} | {result.passed}/{result.total} | {result.accuracy:.2%} |")
    lines.append("")
    for result in results:
        lines.append(f"## {result.name}")
        for case in result.cases:
            mark = "PASS" if case.passed else "FAIL"
            first = case.retrieved[0] if case.retrieved else "<none>"
            lines.append(f"- {mark} `{case.case}`: {first}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaMem's deterministic synthetic ablations.")
    parser.add_argument("--dataset", type=Path, help="JSONL QA benchmark in AdaMem thin format")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = parser.parse_args()
    if args.dataset:
        benchmark_results = run_benchmark(load_jsonl_cases(args.dataset))
        if args.json:
            print(json.dumps([asdict(result) for result in benchmark_results], indent=2, ensure_ascii=False))
        else:
            print(benchmark_report(benchmark_results))
        return

    results = run_synthetic_benchmark()
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
    else:
        print(as_report(results))


if __name__ == "__main__":
    main()
