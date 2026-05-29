from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from adamem.bench import MemoryQACase, QuerySpec
from adamem.config import AdaMemConfig
from adamem.manager import AdaMem
from adamem.schema import MemoryItem, MemoryResult
from adamem.text import tokenize


GENERIC_STALE_TERMS = {
    "about",
    "after",
    "again",
    "around",
    "based",
    "been",
    "before",
    "being",
    "belief",
    "current",
    "currently",
    "does",
    "done",
    "during",
    "exact",
    "finished",
    "good",
    "have",
    "here",
    "home",
    "info",
    "into",
    "just",
    "last",
    "live",
    "lived",
    "lives",
    "living",
    "local",
    "located",
    "long",
    "moved",
    "near",
    "needs",
    "new",
    "now",
    "over",
    "past",
    "place",
    "recommend",
    "recommendation",
    "should",
    "still",
    "staying",
    "that",
    "there",
    "trying",
    "updated",
    "updating",
    "user",
    "where",
    "with",
    "years",
}


@dataclass(slots=True, frozen=True)
class TextSignal:
    """Evaluation-only lexical signal derived from a benchmark belief string.

    This is deliberately stricter than the previous loose keyword overlap:
    generic STALE/query words are removed, and a support hit requires either a
    distinctive anchor token or multi-token overlap. The signal is suitable for
    diagnostics, not for the proposed runtime memory method.
    """

    raw: str
    anchors: tuple[str, ...]
    tokens: tuple[str, ...]


@dataclass(slots=True)
class StaleQueryDiagnostic:
    case_id: str
    query_id: str
    dim: int
    stale_type: str
    query_mentions_old: bool
    query_mentions_new: bool
    current_evidence_recalled: bool
    stale_evidence_exposed: bool
    conflict_pair_covered: bool
    premise_correction_opportunity: bool
    premise_correction_hit: bool
    premise_correction_best_rank: int | None
    current_before_stale: bool | None
    current_best_rank: int | None
    stale_best_rank: int | None
    retrieved_count: int
    adjudicated_old_supports: int
    old_supports: int
    max_old_support_staleness: float
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class StaleDiagnosticResult:
    name: str
    total: int
    current_recall_rate: float
    stale_exposure_rate: float
    conflict_pair_coverage_rate: float
    current_before_stale_rate: float
    premise_old_mention_rate: float
    premise_correction_opportunity_rate: float
    premise_correction_hit_rate: float
    old_support_adjudication_rate: float
    queries: list[StaleQueryDiagnostic]


def run_stale_retrieval_diagnostics(
    cases: Iterable[MemoryQACase],
    configs: dict[str, AdaMemConfig],
) -> list[StaleDiagnosticResult]:
    """Run API-free retrieval diagnostics for STALE-style JSONL cases."""

    case_list = list(cases)
    results: list[StaleDiagnosticResult] = []
    for name, config in configs.items():
        query_records: list[StaleQueryDiagnostic] = []
        for case in case_list:
            mem = AdaMem(config=config)
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
                retrieved = mem.retrieve(query.query, top_k=query.top_k, now=query.now)
                query_records.append(diagnose_stale_query(case.id, query, retrieved, mem.store.all()))
        results.append(_aggregate_diagnostics(name, query_records))
    return results


def diagnose_stale_query(
    case_id: str,
    query: QuerySpec,
    retrieved: list[MemoryResult],
    stored_items: list[MemoryItem],
) -> StaleQueryDiagnostic:
    meta = query.metadata
    old_signal = stale_text_signal(str(meta.get("M_old") or ""))
    new_signal = stale_text_signal(str(meta.get("M_new") or ""))

    current_ranks = [
        index
        for index, result in enumerate(retrieved, start=1)
        if text_supports_signal(result.item.content, new_signal)
    ]
    stale_ranks = [
        index
        for index, result in enumerate(retrieved, start=1)
        if text_supports_signal(result.item.content, old_signal)
        and not _is_premise_correction_result(result)
    ]
    correction_ranks = [
        index
        for index, result in enumerate(retrieved, start=1)
        if _is_premise_correction_result(result)
    ]

    current_best = min(current_ranks) if current_ranks else None
    stale_best = min(stale_ranks) if stale_ranks else None
    if current_best is None or stale_best is None:
        current_before_stale: bool | None = None
    else:
        current_before_stale = current_best < stale_best

    old_supports = [
        item
        for item in stored_items
        if text_supports_signal(item.content, old_signal)
    ]
    current_supports = [
        item
        for item in stored_items
        if text_supports_signal(item.content, new_signal)
    ]
    adjudicated = [
        item
        for item in old_supports
        if item.superseded_by is not None or item.staleness >= 0.5
    ]
    max_staleness = max((item.staleness for item in old_supports), default=0.0)

    return StaleQueryDiagnostic(
        case_id=case_id,
        query_id=query.id or query.query,
        dim=int(meta.get("stale_dim") or 0),
        stale_type=str(meta.get("stale_type") or "?"),
        query_mentions_old=text_supports_signal(query.query, old_signal),
        query_mentions_new=text_supports_signal(query.query, new_signal),
        current_evidence_recalled=bool(current_ranks),
        stale_evidence_exposed=bool(stale_ranks),
        conflict_pair_covered=bool(current_ranks and stale_ranks),
        premise_correction_opportunity=(
            text_supports_signal(query.query, old_signal) and bool(current_supports)
        ),
        premise_correction_hit=bool(correction_ranks),
        premise_correction_best_rank=min(correction_ranks) if correction_ranks else None,
        current_before_stale=current_before_stale,
        current_best_rank=current_best,
        stale_best_rank=stale_best,
        retrieved_count=len(retrieved),
        adjudicated_old_supports=len(adjudicated),
        old_supports=len(old_supports),
        max_old_support_staleness=max_staleness,
        trace=[
            {
                "rank": index,
                "content": result.item.content,
                "score": round(result.score, 4),
                "staleness": round(result.item.staleness, 4),
                "relation": result.relation,
                "kind": result.item.kind,
                "metadata": _trace_metadata(result.item),
                "is_premise_correction": _is_premise_correction_result(result),
                "supports_old": text_supports_signal(result.item.content, old_signal),
                "supports_new": text_supports_signal(result.item.content, new_signal),
                "contributions": {key: round(value, 4) for key, value in result.contributions.items()},
            }
            for index, result in enumerate(retrieved, start=1)
        ],
    )


def stale_text_signal(text: str) -> TextSignal:
    raw_tokens = tokenize(text)
    filtered = tuple(
        token
        for token in raw_tokens
        if len(token) >= 4 and token not in GENERIC_STALE_TERMS and not token.isdigit()
    )
    anchors = tuple(dict.fromkeys(token for token in filtered if _looks_like_anchor(token)))
    return TextSignal(raw=text, anchors=anchors, tokens=tuple(dict.fromkeys(filtered)))


def text_supports_signal(text: str, signal: TextSignal) -> bool:
    if not signal.tokens:
        return False
    text_tokens = set(tokenize(text))
    anchor_hits = sum(1 for token in signal.anchors if token in text_tokens)
    token_hits = sum(1 for token in signal.tokens if token in text_tokens)
    if signal.anchors:
        return anchor_hits >= 1 and token_hits >= min(2, len(signal.tokens))
    return token_hits >= min(3, len(signal.tokens))


def diagnostics_report(results: list[StaleDiagnosticResult]) -> str:
    lines = ["# AdaMem STALE Retrieval Diagnostics", ""]
    lines.append(
        "| ablation | current recall | stale exposure | conflict coverage | current before stale | premise old mention | premise correction hit | old support adjudication |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            f"| {result.name} | {result.current_recall_rate:.2%} | "
            f"{result.stale_exposure_rate:.2%} | {result.conflict_pair_coverage_rate:.2%} | "
            f"{result.current_before_stale_rate:.2%} | {result.premise_old_mention_rate:.2%} | "
            f"{result.premise_correction_hit_rate:.2%} | "
            f"{result.old_support_adjudication_rate:.2%} |"
        )
    return "\n".join(lines) + "\n"


def diagnostic_case_records(
    results: list[StaleDiagnosticResult],
    *,
    only_failures: bool = True,
    include_trace: bool = True,
) -> list[dict[str, Any]]:
    """Flatten diagnostic results into auditable case-level records.

    These records are meant for paper-oriented error analysis. They deliberately
    include benchmark metadata and retrieved text, so they are evaluation
    artifacts only and must not be used by the runtime memory method.
    """

    records: list[dict[str, Any]] = []
    for result in results:
        for query in result.queries:
            failure_modes = _diagnostic_failure_modes(query)
            if only_failures and not failure_modes:
                continue
            record: dict[str, Any] = {
                "baseline": result.name,
                "case_id": query.case_id,
                "query_id": query.query_id,
                "dim": query.dim,
                "stale_type": query.stale_type,
                "failure_modes": failure_modes,
                "analysis_flags": _diagnostic_analysis_flags(query),
                "query_mentions_old": query.query_mentions_old,
                "query_mentions_new": query.query_mentions_new,
                "current_evidence_recalled": query.current_evidence_recalled,
                "stale_evidence_exposed": query.stale_evidence_exposed,
                "conflict_pair_covered": query.conflict_pair_covered,
                "premise_correction_opportunity": query.premise_correction_opportunity,
                "premise_correction_hit": query.premise_correction_hit,
                "premise_correction_best_rank": query.premise_correction_best_rank,
                "current_before_stale": query.current_before_stale,
                "current_best_rank": query.current_best_rank,
                "stale_best_rank": query.stale_best_rank,
                "retrieved_count": query.retrieved_count,
                "old_supports": query.old_supports,
                "adjudicated_old_supports": query.adjudicated_old_supports,
                "max_old_support_staleness": round(query.max_old_support_staleness, 4),
            }
            if include_trace:
                record["trace"] = query.trace
            records.append(record)
    return records


def diagnostic_failure_summary(records: list[dict[str, Any]], *, max_examples: int = 3) -> dict[str, Any]:
    """Aggregate case-level diagnostic records for paper error analysis."""

    by_baseline: dict[str, int] = {}
    by_dim: dict[str, int] = {}
    by_stale_type: dict[str, int] = {}
    by_failure_mode: dict[str, int] = {}
    by_analysis_flag: dict[str, int] = {}
    by_baseline_failure_mode: dict[str, dict[str, int]] = {}
    examples_by_failure_mode: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        baseline = str(record.get("baseline") or "?")
        dim = str(record.get("dim") or "?")
        stale_type = str(record.get("stale_type") or "?")
        by_baseline[baseline] = by_baseline.get(baseline, 0) + 1
        by_dim[dim] = by_dim.get(dim, 0) + 1
        by_stale_type[stale_type] = by_stale_type.get(stale_type, 0) + 1

        failure_modes = [str(mode) for mode in record.get("failure_modes", [])]
        for mode in failure_modes:
            by_failure_mode[mode] = by_failure_mode.get(mode, 0) + 1
            nested = by_baseline_failure_mode.setdefault(baseline, {})
            nested[mode] = nested.get(mode, 0) + 1
            examples = examples_by_failure_mode.setdefault(mode, [])
            if len(examples) < max_examples:
                examples.append(_compact_failure_example(record))

        analysis_flags = [str(flag) for flag in record.get("analysis_flags", [])]
        for flag in analysis_flags:
            by_analysis_flag[flag] = by_analysis_flag.get(flag, 0) + 1

    return {
        "total_records": len(records),
        "by_baseline": _sorted_counts(by_baseline),
        "by_dim": _sorted_counts(by_dim),
        "by_stale_type": _sorted_counts(by_stale_type),
        "by_failure_mode": _sorted_counts(by_failure_mode),
        "by_analysis_flag": _sorted_counts(by_analysis_flag),
        "by_baseline_failure_mode": {
            baseline: _sorted_counts(counts)
            for baseline, counts in sorted(by_baseline_failure_mode.items())
        },
        "examples_by_failure_mode": examples_by_failure_mode,
    }


def diagnostic_failure_report(records: list[dict[str, Any]], *, max_examples: int = 3) -> str:
    summary = diagnostic_failure_summary(records, max_examples=max_examples)
    lines = ["# AdaMem STALE Diagnostic Failure Report", ""]
    lines.append(f"Total diagnostic records: {summary['total_records']}")
    lines.append("")
    _append_count_table(lines, "Failure Modes", summary["by_failure_mode"])
    _append_count_table(lines, "Baselines", summary["by_baseline"])
    _append_count_table(lines, "STALE Dimensions", summary["by_dim"])
    _append_count_table(lines, "STALE Types", summary["by_stale_type"])
    if summary["by_analysis_flag"]:
        _append_count_table(lines, "Analysis Flags", summary["by_analysis_flag"])

    lines.append("## Failure Modes By Baseline")
    lines.append("")
    lines.append("| baseline | failure mode | count |")
    lines.append("| --- | --- | ---: |")
    for baseline, counts in summary["by_baseline_failure_mode"].items():
        for mode, count in counts.items():
            lines.append(f"| {baseline} | {mode} | {count} |")
    lines.append("")

    lines.append("## Representative Examples")
    lines.append("")
    for mode, examples in summary["examples_by_failure_mode"].items():
        lines.append(f"### {mode}")
        for example in examples:
            first = example.get("top_retrieved", "<none>")
            lines.append(
                f"- `{example['baseline']}` `{example['query_id']}` dim={example['dim']} "
                f"type={example['stale_type']} top={first}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _aggregate_diagnostics(name: str, queries: list[StaleQueryDiagnostic]) -> StaleDiagnosticResult:
    total = len(queries)
    ordered_pairs = [record for record in queries if record.current_before_stale is not None]
    old_support_total = sum(record.old_supports for record in queries)
    old_support_adjudicated = sum(record.adjudicated_old_supports for record in queries)
    return StaleDiagnosticResult(
        name=name,
        total=total,
        current_recall_rate=_rate(record.current_evidence_recalled for record in queries),
        stale_exposure_rate=_rate(record.stale_evidence_exposed for record in queries),
        conflict_pair_coverage_rate=_rate(record.conflict_pair_covered for record in queries),
        current_before_stale_rate=_rate(record.current_before_stale for record in ordered_pairs),
        premise_old_mention_rate=_rate(record.query_mentions_old for record in queries),
        premise_correction_opportunity_rate=_rate(
            record.premise_correction_opportunity for record in queries
        ),
        premise_correction_hit_rate=(
            _rate(record.premise_correction_hit for record in queries if record.premise_correction_opportunity)
        ),
        old_support_adjudication_rate=(
            old_support_adjudicated / old_support_total if old_support_total else 0.0
        ),
        queries=queries,
    )


def _rate(values: Iterable[bool | None]) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return sum(1 for value in items if value) / len(items)


def _diagnostic_failure_modes(query: StaleQueryDiagnostic) -> list[str]:
    modes: list[str] = []
    if not query.current_evidence_recalled:
        modes.append("current_evidence_not_recalled")
    if query.stale_evidence_exposed:
        modes.append("stale_evidence_exposed")
    if query.current_before_stale is False:
        modes.append("stale_ranked_before_current")
    if query.premise_correction_opportunity and not query.premise_correction_hit:
        modes.append("premise_correction_missing")
    if query.old_supports and query.adjudicated_old_supports < query.old_supports:
        modes.append("old_support_not_fully_adjudicated")
    return modes


def _diagnostic_analysis_flags(query: StaleQueryDiagnostic) -> list[str]:
    flags: list[str] = []
    if query.query_mentions_old and query.current_evidence_recalled:
        flags.append("stale_premise_correction_opportunity")
    if query.premise_correction_hit:
        flags.append("stale_premise_corrected")
    return flags


def _is_premise_correction_result(result: MemoryResult) -> bool:
    return (
        result.item.kind == "state_correction"
        or result.relation == "state_premise_correction"
    )


def _trace_metadata(item: MemoryItem) -> dict[str, Any]:
    keep = {
        "ephemeral",
        "source_state_id",
        "stale_state_id",
        "state_slot",
        "state_value",
        "stale_value",
        "current_value",
    }
    return {key: value for key, value in item.metadata.items() if key in keep}


def _compact_failure_example(record: dict[str, Any]) -> dict[str, Any]:
    trace = record.get("trace")
    top_retrieved = None
    if isinstance(trace, list) and trace:
        first = trace[0]
        if isinstance(first, dict):
            top_retrieved = str(first.get("content") or "")[:180]
    return {
        "baseline": record.get("baseline"),
        "case_id": record.get("case_id"),
        "query_id": record.get("query_id"),
        "dim": record.get("dim"),
        "stale_type": record.get("stale_type"),
        "top_retrieved": top_retrieved,
    }


def _append_count_table(lines: list[str], title: str, counts: dict[str, int]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| key | count |")
    lines.append("| --- | ---: |")
    for key, count in counts.items():
        lines.append(f"| {key} | {count} |")
    lines.append("")


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _looks_like_anchor(token: str) -> bool:
    if any(char.isdigit() for char in token):
        return True
    if "_" in token:
        return True
    return token not in GENERIC_STALE_TERMS
