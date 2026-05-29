from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adamem.bench import (
    benchmark_case_records,
    benchmark_failure_report,
    benchmark_failure_summary,
    benchmark_report,
    default_ablation_configs,
    load_jsonl_cases,
    run_benchmark,
)
from adamem.baselines import baseline_report, select_baselines
from adamem.config import AdaMemConfig
from adamem.diagnostics import (
    diagnostic_case_records,
    diagnostic_failure_report,
    diagnostic_failure_summary,
    diagnostics_report,
    run_stale_retrieval_diagnostics,
)
from adamem.experiments import experiment_record, write_experiment_record
from adamem.llm import LLMClient, build_client
from adamem.manager import AdaMem
from adamem.schema import MemoryItem
from adamem.state import (
    LLMStateExtractor,
    STATE_EXTRACTOR_SYSTEM,
    STATE_EXTRACTOR_TEMPLATE,
    StateExtractor,
)
from adamem.state_diagnostics import state_memory_inventory


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


# ---------------------------------------------------------------------------
# STALE LLM-judge evaluation
# ---------------------------------------------------------------------------

STALE_JUDGE_SYSTEM = (
    "You are a careful evaluator. Your job is to score whether the assistant "
    "correctly handles an implicit memory conflict between an OLD belief and a "
    "NEW belief about the user. Output exactly one token: CORRECT or INCORRECT."
)

STALE_JUDGE_TEMPLATE = """\
Evaluation dimension: {dim_name}
Dimension intent: {dim_intent}

OLD belief (now stale): {m_old}
NEW belief (current truth): {m_new}
Why they conflict: {explanation}

User question: {query}

Assistant answer:
{answer}

Decide if the assistant's answer is consistent with the NEW belief and does NOT
rely on or affirm the OLD belief. Reply with exactly one token: CORRECT or INCORRECT.
"""

STALE_DIMS: dict[int, tuple[str, str]] = {
    1: (
        "State Resolution",
        "The user explicitly asks whether the OLD belief still holds. The assistant "
        "should recognize the OLD belief is invalidated by the NEW belief.",
    ),
    2: (
        "Premise Resistance",
        "The query embeds the OLD belief as a presupposition. The assistant should "
        "reject the false premise and ground its response in the NEW belief.",
    ),
    3: (
        "Implicit Policy Adaptation",
        "The query mentions neither belief but should be answered by applying the "
        "updated NEW belief in downstream behavior.",
    ),
}

STALE_ANSWER_SYSTEM = (
    "You are a helpful assistant grounded ONLY in the conversation memory excerpts "
    "given to you. Use ONLY those excerpts to answer; do not invent facts. If the "
    "excerpts conflict, prefer the more recent excerpt."
)

STALE_ANSWER_TEMPLATE = """\
Conversation memory excerpts (most relevant first):
{context}

User question: {query}

Answer concisely (2-4 sentences) using only the excerpts above.
"""


@dataclass(slots=True)
class StaleQueryResult:
    case_id: str
    query_id: str
    dim: int
    stale_type: str
    correct: bool
    judge_raw: str
    answer: str
    retrieved: list[str]
    adjudication: dict[str, Any] = field(default_factory=dict)
    stale_leak: bool = False


@dataclass(slots=True)
class StaleAblationResult:
    name: str
    accuracy: float
    n_correct: int
    n_total: int
    by_dim: dict[int, dict[str, float]]
    by_type: dict[str, dict[str, float]]
    adjudication_rate: float
    stale_leak_rate: float
    queries: list[StaleQueryResult]


def run_stale_benchmark(
    dataset_path: str | Path,
    *,
    answer_client: LLMClient,
    judge_client: LLMClient,
    configs: dict[str, AdaMemConfig] | None = None,
    top_k: int = 8,
    max_context_chars: int = 4000,
    max_cases: int | None = None,
    stale_types: list[str] | None = None,
    limit_per_stale_type: int | None = None,
    request_delay: float = 0.0,
    raw_outputs: list[dict[str, Any]] | None = None,
    state_extractors: dict[str, StateExtractor] | None = None,
) -> list[StaleAblationResult]:
    """Score AdaMem on the STALE benchmark using LLM-judge methodology.

    For each ablation in `configs`, ingest each case's haystack into a fresh
    AdaMem, then for every probing query: retrieve top-k context, ask
    `answer_client` to answer using only that context, ask `judge_client` to
    score CORRECT/INCORRECT, and aggregate by dim and type.
    """
    cases = load_jsonl_cases(dataset_path)
    cases = _select_stale_cases(
        cases,
        stale_types=stale_types,
        limit_per_type=limit_per_stale_type,
        max_cases=max_cases,
    )
    configs = configs or default_ablation_configs()
    state_extractors = state_extractors or {}

    results: list[StaleAblationResult] = []
    for name, config in configs.items():
        per_query: list[StaleQueryResult] = []
        state_extractor = state_extractors.get(name)
        for case in cases:
            mem = AdaMem(config=config, state_extractor=state_extractor)
            for observation in case.observations:
                mem.observe(
                    observation.content,
                    kind=observation.kind,
                    importance=observation.importance,
                    confidence=observation.confidence,
                    valid_from=observation.valid_from,
                    valid_to=observation.valid_to,
                    metadata=observation.metadata,
                )
            for query in case.queries:
                retrieved_results = mem.retrieve(query.query, top_k=query.top_k or top_k)
                retrieved_texts = [result.item.content for result in retrieved_results]
                context = _truncate("\n---\n".join(retrieved_texts), max_context_chars)
                answer_prompt = STALE_ANSWER_TEMPLATE.format(context=context, query=query.query)
                answer = answer_client.complete(
                    answer_prompt,
                    system=STALE_ANSWER_SYSTEM,
                    max_tokens=200,
                    temperature=0.0,
                )
                dim = int(query.metadata.get("stale_dim") or 0)
                dim_name, dim_intent = STALE_DIMS.get(dim, ("Unknown", ""))
                judge_prompt = STALE_JUDGE_TEMPLATE.format(
                    dim_name=dim_name,
                    dim_intent=dim_intent,
                    m_old=query.metadata.get("M_old", ""),
                    m_new=query.metadata.get("M_new", ""),
                    explanation=query.metadata.get("explanation", ""),
                    query=query.query,
                    answer=answer,
                )
                judge_raw = judge_client.complete(
                    judge_prompt,
                    system=STALE_JUDGE_SYSTEM,
                    max_tokens=8,
                    temperature=0.0,
                )
                if request_delay > 0:
                    time.sleep(request_delay)
                correct = _parse_judge(judge_raw)
                leak = _detect_stale_leak(retrieved_texts, query.metadata)
                state_inventory = state_memory_inventory(mem.store.all())
                if raw_outputs is not None:
                    raw_outputs.append(
                        {
                            "baseline": name,
                            "case_id": case.id,
                            "query_id": query.id or "",
                            "dim": dim,
                            "stale_type": str(query.metadata.get("stale_type") or "?"),
                            "query": query.query,
                            "answer_prompt": answer_prompt,
                            "answer_system": STALE_ANSWER_SYSTEM,
                            "answer_raw": answer,
                            "judge_prompt": judge_prompt,
                            "judge_system": STALE_JUDGE_SYSTEM,
                            "judge_raw": judge_raw,
                            "judge_correct": correct,
                            "stale_leak": leak,
                            **state_inventory,
                            "retrieved": [
                                {
                                    "rank": index,
                                    "content": result.item.content,
                                    "score": round(result.score, 4),
                                    "staleness": round(result.item.staleness, 4),
                                    "relation": result.relation,
                                    "contributions": {
                                        key: round(value, 4)
                                        for key, value in result.contributions.items()
                                    },
                                }
                                for index, result in enumerate(retrieved_results, start=1)
                            ],
                        }
                    )
                per_query.append(
                    StaleQueryResult(
                        case_id=case.id,
                        query_id=query.id or "",
                        dim=dim,
                        stale_type=str(query.metadata.get("stale_type") or "?"),
                        correct=correct,
                        judge_raw=judge_raw,
                        answer=answer,
                        retrieved=retrieved_texts,
                        adjudication=_adjudication_snapshot(mem, query.metadata),
                        stale_leak=leak,
                    )
                )

        n_correct = sum(1 for r in per_query if r.correct)
        n_total = len(per_query)
        by_dim = _group_accuracy(per_query, key=lambda r: r.dim)
        by_type = _group_accuracy(per_query, key=lambda r: r.stale_type)
        adr = _aggregate_adjudication_rate(per_query)
        slr = _aggregate_stale_leak_rate(per_query)
        results.append(
            StaleAblationResult(
                name=name,
                accuracy=n_correct / n_total if n_total else 0.0,
                n_correct=n_correct,
                n_total=n_total,
                by_dim=by_dim,
                by_type=by_type,
                adjudication_rate=adr,
                stale_leak_rate=slr,
                queries=per_query,
            )
        )
    return results


def stale_report(results: list[StaleAblationResult]) -> str:
    lines = ["# AdaMem on STALE", ""]
    lines.append("| ablation | overall | dim1 SR | dim2 PR | dim3 IPA | T1 | T2 | ADR | SLR |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        d1 = result.by_dim.get(1, {}).get("accuracy", 0.0)
        d2 = result.by_dim.get(2, {}).get("accuracy", 0.0)
        d3 = result.by_dim.get(3, {}).get("accuracy", 0.0)
        t1 = result.by_type.get("T1", {}).get("accuracy", 0.0)
        t2 = result.by_type.get("T2", {}).get("accuracy", 0.0)
        lines.append(
            f"| {result.name} | {result.accuracy:.2%} ({result.n_correct}/{result.n_total}) "
            f"| {d1:.2%} | {d2:.2%} | {d3:.2%} | {t1:.2%} | {t2:.2%} "
            f"| {result.adjudication_rate:.2%} | {result.stale_leak_rate:.2%} |"
        )
    return "\n".join(lines) + "\n"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]"


def _parse_judge(raw: str) -> bool:
    token = (raw or "").strip().upper()
    if token.startswith("CORRECT"):
        return True
    if token.startswith("INCORRECT"):
        return False
    # Fall back: search for the words.
    return "CORRECT" in token and "INCORRECT" not in token


def _group_accuracy(records: list[StaleQueryResult], *, key) -> dict[Any, dict[str, float]]:
    buckets: dict[Any, list[StaleQueryResult]] = {}
    for record in records:
        buckets.setdefault(key(record), []).append(record)
    out: dict[Any, dict[str, float]] = {}
    for k, items in buckets.items():
        n = len(items)
        c = sum(1 for r in items if r.correct)
        out[k] = {"accuracy": c / n if n else 0.0, "n_correct": c, "n_total": n}
    return out


def _adjudication_snapshot(mem: AdaMem, query_meta: dict[str, Any]) -> dict[str, Any]:
    """Capture process-level signals for the Adjudication Rate metric.

    A stored memory is "adjudicated as stale" if either:
      * `superseded_by` is set (hard supersession from `memory_key` collision), or
      * `staleness >= 0.5` (soft staleness from mechanism A).
    """
    m_old = (query_meta.get("M_old") or "").lower()
    if not m_old:
        return {"old_match_count": 0, "old_superseded_count": 0, "max_staleness": 0.0}
    keywords = [token for token in m_old.split() if len(token) > 3][:6]
    matched = 0
    superseded = 0
    max_stale = 0.0
    for item in mem.store.all():
        text = item.content.lower()
        if any(token in text for token in keywords):
            matched += 1
            adjudicated = item.superseded_by is not None or item.staleness >= 0.5
            if adjudicated:
                superseded += 1
            max_stale = max(max_stale, item.staleness)
    return {
        "old_match_count": matched,
        "old_superseded_count": superseded,
        "max_staleness": max_stale,
    }


def _aggregate_adjudication_rate(records: list[StaleQueryResult]) -> float:
    """Adjudication Rate (ADR): fraction of queries where the OLD belief had at
    least one supporting memory adjudicated as stale."""
    eligible = [r for r in records if r.adjudication.get("old_match_count", 0) > 0]
    if not eligible:
        return 0.0
    flagged = sum(1 for r in eligible if r.adjudication.get("old_superseded_count", 0) > 0)
    return flagged / len(eligible)


def _detect_stale_leak(retrieved_texts: list[str], query_meta: dict[str, Any]) -> bool:
    """Stale Leak: at least one retrieved excerpt still contains the OLD belief
    signal. We approximate by checking the longest informative substring of
    `M_old` against each retrieved text."""
    m_old = (query_meta.get("M_old") or "").lower()
    if not m_old:
        return False
    keywords = [token for token in m_old.split() if len(token) > 3]
    if not keywords:
        return False
    needle = " ".join(keywords[:6])
    blob = "\n".join(text.lower() for text in retrieved_texts)
    if needle and needle in blob:
        return True
    # Fall back to high-token overlap (>=4 distinctive tokens present).
    hits = sum(1 for token in keywords if token in blob)
    return hits >= max(4, len(keywords) // 2)


def _aggregate_stale_leak_rate(records: list[StaleQueryResult]) -> float:
    """Stale Leak Rate (SLR): fraction of queries whose final retrieved
    context still contains a recognizable trace of the OLD belief."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.stale_leak) / len(records)


def _select_stale_cases(
    cases: list[Any],
    *,
    stale_types: list[str] | None = None,
    limit_per_type: int | None = None,
    max_cases: int | None = None,
) -> list[Any]:
    allowed = set(stale_types or [])
    selected: list[Any] = []
    type_counts: dict[str, int] = {}
    for case in cases:
        case_type = _case_stale_type(case)
        if allowed and case_type not in allowed:
            continue
        if limit_per_type is not None:
            count = type_counts.get(case_type, 0)
            if count >= limit_per_type:
                continue
            type_counts[case_type] = count + 1
        selected.append(case)
        if max_cases is not None and len(selected) >= max_cases:
            break
    return selected


def _case_stale_type(case: Any) -> str:
    queries = getattr(case, "queries", [])
    for query in queries:
        metadata = getattr(query, "metadata", {})
        value = metadata.get("stale_type") if isinstance(metadata, dict) else None
        if value:
            return str(value)
    return "<missing>"


def _state_extractor_runtime(
    configs: dict[str, AdaMemConfig],
    *,
    provider: str,
    model: str,
    mock_response: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, StateExtractor], dict[str, Any], dict[str, str]]:
    llm_baselines = [
        name
        for name, config in configs.items()
        if config.state_extractor_name in {"llm_json", "llm"}
    ]
    if not llm_baselines:
        return {}, {}, {}
    if provider == "none":
        raise ValueError(
            "selected baselines require state_extractor_name=llm_json; "
            "set --state-extractor-provider mock/openai/gemini/modelhub"
        )
    kwargs: dict[str, Any]
    if provider == "mock":
        kwargs = {"responses": mock_response}
    else:
        kwargs = {"model": model}
    client = build_client(provider, **kwargs)
    extractor = LLMStateExtractor(
        client,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    notes = {
        "state_extractor_provider": provider,
        "state_extractor_model": None if provider == "mock" else model,
        "state_extractor_baselines": llm_baselines,
        "state_extractor_max_tokens": max_tokens,
        "state_extractor_temperature": temperature,
        "state_extractor_runtime_use": "observation_text_only",
    }
    prompts = {
        "state_extractor_system": STATE_EXTRACTOR_SYSTEM,
        "state_extractor_template": STATE_EXTRACTOR_TEMPLATE,
    }
    return {name: extractor for name in llm_baselines}, notes, prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaMem's deterministic synthetic ablations.")
    parser.add_argument("--dataset", type=Path, help="JSONL QA benchmark in AdaMem thin format")
    parser.add_argument("--stale", type=Path, help="Run STALE LLM-judge evaluation on this JSONL")
    parser.add_argument("--stale-diagnostics", type=Path, help="Run API-free STALE retrieval diagnostics on this JSONL")
    parser.add_argument("--benchmark-cases-output", type=Path, help="Write JSONL per-query records for --dataset runs")
    parser.add_argument("--benchmark-report-output", type=Path, help="Write Markdown grouped report for --dataset runs")
    parser.add_argument("--diagnostic-cases-output", type=Path, help="Write JSONL case-level STALE diagnostic records")
    parser.add_argument("--diagnostic-report-output", type=Path, help="Write Markdown STALE diagnostic failure report")
    parser.add_argument("--list-baselines", action="store_true", help="List stable baseline registry entries")
    parser.add_argument("--baselines", nargs="+", help="Run only these stable baseline names")
    parser.add_argument("--experiment-output", type=Path, help="Write a JSON experiment record for supported runs")
    parser.add_argument("--answer-provider", default="openai", help="LLM provider for answer agent")
    parser.add_argument("--answer-model", default="gpt-4o-mini")
    parser.add_argument("--judge-provider", default="gemini", help="LLM provider for judge")
    parser.add_argument("--judge-model", default="gemini-1.5-flash")
    parser.add_argument(
        "--state-extractor-provider",
        default="none",
        choices=["none", "openai", "gemini", "modelhub", "mock"],
        help="Provider for baselines whose state_extractor_name is llm_json",
    )
    parser.add_argument("--state-extractor-model", default="gpt-4o-mini")
    parser.add_argument(
        "--state-extractor-mock-response",
        default='{"patches":[]}',
        help="Fixed mock JSON response for --state-extractor-provider mock",
    )
    parser.add_argument("--state-extractor-max-tokens", type=int, default=512)
    parser.add_argument("--state-extractor-temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-cases", type=int, help="Limit number of STALE cases evaluated")
    parser.add_argument("--stale-types", nargs="+", choices=["T1", "T2"], help="Filter STALE cases by conflict type")
    parser.add_argument("--limit-per-stale-type", type=int, help="Keep at most this many STALE cases per type")
    parser.add_argument("--request-delay", type=float, default=0.0, help="Delay between STALE answer/judge calls")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = parser.parse_args()

    try:
        specs = select_baselines(args.baselines)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_baselines:
        if args.json:
            print(json.dumps({name: asdict(spec) for name, spec in specs.items()}, indent=2, ensure_ascii=False))
        else:
            print(baseline_report(specs))
        return

    configs = {name: spec.config for name, spec in specs.items()}
    try:
        state_extractors, state_extractor_notes, state_extractor_prompts = _state_extractor_runtime(
            configs,
            provider=args.state_extractor_provider,
            model=args.state_extractor_model,
            mock_response=args.state_extractor_mock_response,
            max_tokens=args.state_extractor_max_tokens,
            temperature=args.state_extractor_temperature,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.stale_diagnostics:
        cases = load_jsonl_cases(args.stale_diagnostics)
        cases = _select_stale_cases(
            cases,
            stale_types=args.stale_types,
            limit_per_type=args.limit_per_stale_type,
            max_cases=args.max_cases,
        )
        diagnostic_results = run_stale_retrieval_diagnostics(
            cases,
            configs,
            state_extractors=state_extractors,
        )
        case_records = diagnostic_case_records(diagnostic_results)
        if args.diagnostic_cases_output:
            _write_jsonl(args.diagnostic_cases_output, case_records)
        failure_summary = diagnostic_failure_summary(case_records)
        if args.diagnostic_report_output:
            _write_text(args.diagnostic_report_output, diagnostic_failure_report(case_records))
        if args.experiment_output:
            record = experiment_record(
                run_name=args.experiment_output.stem,
                run_type="stale_retrieval_diagnostics",
                dataset=args.stale_diagnostics,
                split_or_case_limit=_split_note(args.max_cases, args.stale_types, args.limit_per_stale_type),
                baselines=specs,
                diagnostics=[asdict(result) for result in diagnostic_results],
                results={"failure_summary": failure_summary},
                raw_outputs=case_records,
                notes={
                    "answer_model_required": False,
                    "judge_model_required": False,
                    "ground_truth_runtime_use": "forbidden",
                    "diagnostic_case_records": len(case_records),
                    **state_extractor_notes,
                },
                prompts=state_extractor_prompts,
            )
            write_experiment_record(args.experiment_output, record)
        if args.json:
            print(json.dumps([asdict(result) for result in diagnostic_results], indent=2, ensure_ascii=False))
        else:
            print(diagnostics_report(diagnostic_results))
        return

    if args.stale:
        answer_client = build_client(args.answer_provider, model=args.answer_model)
        judge_client = build_client(args.judge_provider, model=args.judge_model)
        raw_outputs: list[dict[str, Any]] = []
        stale_results = run_stale_benchmark(
            args.stale,
            answer_client=answer_client,
            judge_client=judge_client,
            configs=configs,
            state_extractors=state_extractors,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            max_cases=args.max_cases,
            stale_types=args.stale_types,
            limit_per_stale_type=args.limit_per_stale_type,
            request_delay=args.request_delay,
            raw_outputs=raw_outputs,
        )
        if args.experiment_output:
            record = experiment_record(
                run_name=args.experiment_output.stem,
                run_type="stale_llm_judge",
                dataset=args.stale,
                split_or_case_limit=_split_note(args.max_cases, args.stale_types, args.limit_per_stale_type),
                baselines=specs,
                results=[asdict(result) for result in stale_results],
                prompts={
                    "answer_system": STALE_ANSWER_SYSTEM,
                    "answer_template": STALE_ANSWER_TEMPLATE,
                    "judge_system": STALE_JUDGE_SYSTEM,
                    "judge_template": STALE_JUDGE_TEMPLATE,
                    **state_extractor_prompts,
                },
                raw_outputs=raw_outputs,
                notes={
                    "answer_provider": args.answer_provider,
                    "answer_model": args.answer_model,
                    "judge_provider": args.judge_provider,
                    "judge_model": args.judge_model,
                    "top_k": args.top_k,
                    "max_context_chars": args.max_context_chars,
                    "stale_types": args.stale_types,
                    "limit_per_stale_type": args.limit_per_stale_type,
                    "request_delay": args.request_delay,
                    "ground_truth_runtime_use": "forbidden",
                    "ground_truth_judge_use": "allowed",
                    **state_extractor_notes,
                },
            )
            write_experiment_record(args.experiment_output, record)
        if args.json:
            print(json.dumps([asdict(r) for r in stale_results], indent=2, ensure_ascii=False))
        else:
            print(stale_report(stale_results))
        return

    if args.dataset:
        cases = load_jsonl_cases(args.dataset)
        if args.max_cases is not None:
            cases = cases[:args.max_cases]
        benchmark_results = run_benchmark(cases, configs, state_extractors=state_extractors)
        benchmark_records = benchmark_case_records(benchmark_results)
        benchmark_summary = benchmark_failure_summary(benchmark_records)
        if args.benchmark_cases_output:
            _write_jsonl(args.benchmark_cases_output, benchmark_records)
        if args.benchmark_report_output:
            _write_text(args.benchmark_report_output, benchmark_failure_report(benchmark_records))
        if args.experiment_output:
            record = experiment_record(
                run_name=args.experiment_output.stem,
                run_type="jsonl_retrieval_benchmark",
                dataset=args.dataset,
                split_or_case_limit=str(args.max_cases) if args.max_cases is not None else None,
                baselines=specs,
                results=[asdict(result) for result in benchmark_results],
                diagnostics={"failure_summary": benchmark_summary},
                raw_outputs=benchmark_records,
                notes={
                    "answer_model_required": False,
                    "judge_model_required": False,
                    "benchmark_kind": "retrieval_support",
                    "benchmark_case_records": len(benchmark_records),
                    "ground_truth_runtime_use": "forbidden",
                    "ground_truth_evaluation_use": "expected_substrings_only",
                    **state_extractor_notes,
                },
                prompts=state_extractor_prompts,
            )
            write_experiment_record(args.experiment_output, record)
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp.replace(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _split_note(
    max_cases: int | None,
    stale_types: list[str] | None,
    limit_per_stale_type: int | None,
) -> str | None:
    parts: list[str] = []
    if max_cases is not None:
        parts.append(f"max_cases={max_cases}")
    if stale_types:
        parts.append(f"stale_types={','.join(stale_types)}")
    if limit_per_stale_type is not None:
        parts.append(f"limit_per_stale_type={limit_per_stale_type}")
    return ";".join(parts) if parts else None


if __name__ == "__main__":
    main()
