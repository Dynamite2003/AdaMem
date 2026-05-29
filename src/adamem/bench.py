from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from adamem.baselines import default_ablation_configs
from adamem.config import AdaMemConfig
from adamem.manager import AdaMem
from adamem.schema import MemoryItem
from adamem.state import state_slot_matches_query


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryQACase:
    id: str
    observations: list[ObservationSpec]
    queries: list[QuerySpec]


@dataclass(slots=True)
class QueryEvalResult:
    case_id: str
    query_id: str
    query: str
    passed: bool
    retrieved: list[str]
    expected_substrings: list[str]
    forbidden_substrings: list[str]
    trace: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    accuracy: float
    passed: int
    total: int
    queries: list[QueryEvalResult]


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


def benchmark_case_records(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        for query in result.queries:
            records.append(_query_record(result.name, query))
    return records


def benchmark_failure_summary(
    records: list[dict[str, Any]],
    *,
    group_fields: Iterable[str] = ("question_type", "dimension", "state_slot", "abstention"),
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_records": len(records),
        "by_baseline": {},
        "failure_modes": {},
        "by_metadata": {},
        "state_readout_exposure": {},
        "premise_correction": {},
        "evidence_support": {},
        "answerability": {},
        "paper_metrics": {},
        "pairwise_vs_first_baseline": {},
        "diagnostics_by_metadata": {},
    }
    baseline_order = list(dict.fromkeys(str(record["baseline"]) for record in records))
    for baseline in sorted(set(baseline_order)):
        subset = [record for record in records if record["baseline"] == baseline]
        summary["by_baseline"][baseline] = _aggregate_records(subset)
        summary["state_readout_exposure"][baseline] = _state_exposure_aggregate(subset)
        summary["premise_correction"][baseline] = _premise_correction_aggregate(subset)
        summary["evidence_support"][baseline] = _evidence_support_aggregate(subset)
        summary["answerability"][baseline] = _answerability_aggregate(subset)
    if baseline_order:
        reference = baseline_order[0]
        for candidate in baseline_order[1:]:
            summary["pairwise_vs_first_baseline"][candidate] = _paired_comparison(
                records,
                reference=reference,
                candidate=candidate,
                group_fields=group_fields,
            )
    for baseline in sorted(set(baseline_order)):
        summary["paper_metrics"][baseline] = _paper_metrics_for_baseline(
            baseline,
            summary=summary,
            reference=baseline_order[0] if baseline_order else None,
        )

    for record in records:
        for mode in record["failure_modes"]:
            summary["failure_modes"][mode] = summary["failure_modes"].get(mode, 0) + 1

    for field_name in group_fields:
        field_summary: dict[str, Any] = {}
        values = sorted({_metadata_group_value(record["metadata"], field_name) for record in records})
        for value in values:
            value_subset = [
                record
                for record in records
                if _metadata_group_value(record["metadata"], field_name) == value
            ]
            by_baseline = {
                baseline: _aggregate_records([
                    record for record in value_subset if record["baseline"] == baseline
                ])
                for baseline in sorted({str(record["baseline"]) for record in value_subset})
            }
            field_summary[value] = by_baseline
        summary["by_metadata"][field_name] = field_summary
        summary["diagnostics_by_metadata"][field_name] = _metadata_diagnostic_summary(
            records,
            field_name=field_name,
            values=values,
        )
    return summary


def benchmark_failure_report(
    records: list[dict[str, Any]],
    *,
    group_fields: Iterable[str] = ("question_type", "dimension", "state_slot", "abstention"),
    max_examples: int = 2,
) -> str:
    summary = benchmark_failure_summary(records, group_fields=group_fields)
    lines = ["# JSONL Retrieval Benchmark Failure Report", ""]
    lines.append("## Baselines")
    lines.append("| baseline | passed | accuracy |")
    lines.append("| --- | ---: | ---: |")
    for baseline, aggregate in summary["by_baseline"].items():
        lines.append(
            f"| {baseline} | {aggregate['passed']}/{aggregate['total']} | {aggregate['accuracy']:.2%} |"
        )
    lines.append("")

    pairwise = summary.get("pairwise_vs_first_baseline", {})
    if pairwise:
        reference = next(iter(records), {}).get("baseline", "<none>") if records else "<none>"
        lines.append(f"## Pairwise Vs {reference}")
        lines.append("| candidate | common | gained | lost | net | both pass | both fail |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for candidate, comparison in pairwise.items():
            lines.append(
                f"| {candidate} | {comparison['common_total']} | {comparison['gained_passes']} | "
                f"{comparison['lost_passes']} | {comparison['net_delta']} | "
                f"{comparison['both_pass']} | {comparison['both_fail']} |"
            )
        lines.append("")

    paper_metrics = summary.get("paper_metrics", {})
    if paper_metrics:
        lines.append("## Paper Metrics")
        lines.append(
            "| baseline | support | accuracy | net vs reference | state slot match | "
            "state missing | slot mismatch | evidence support | graph evidence hit | "
            "answer keyword recall | basis keyword recall | unmarked state exposure |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for baseline, metrics in paper_metrics.items():
            lines.append(
                f"| {baseline} | {metrics['support_passed']}/{metrics['support_total']} | "
                f"{metrics['support_accuracy']:.2%} | {metrics['net_delta_vs_reference']} | "
                f"{_format_optional_rate(metrics['state_slot_match_rate'])} | "
                f"{_format_optional_rate(metrics['state_readout_missing_rate'])} | "
                f"{_format_optional_rate(metrics['state_slot_mismatch_rate'])} | "
                f"{_format_optional_rate(metrics['evidence_support_rate'])} | "
                f"{_format_optional_rate(metrics['graph_evidence_hit_rate'])} | "
                f"{_format_optional_rate(metrics['answer_keyword_recall_avg'])} | "
                f"{_format_optional_rate(metrics['basis_answer_keyword_recall_avg'])} | "
                f"{_format_optional_rate(metrics['unmarked_state_exposure_rate'])} |"
            )
        lines.append("")

    exposure = summary.get("state_readout_exposure", {})
    if exposure:
        lines.append("## State Readout Exposure")
        lines.append(
            "| baseline | state queries | state available | state unavailable | queries with state | "
            "matched | missing | mismatched | unmarked with state | unmarked exposure |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for baseline, aggregate in exposure.items():
            lines.append(
                f"| {baseline} | {aggregate['state_sensitive_total']} | "
                f"{aggregate['state_available_total']} | "
                f"{aggregate['state_unavailable_total']} | "
                f"{aggregate['state_retrieval_records']} | "
                f"{aggregate['state_slot_match_records']} | "
                f"{aggregate['state_readout_missing_records']} | "
                f"{aggregate['state_slot_mismatch_records']} | "
                f"{aggregate['unmarked_state_retrieval_records']} | "
                f"{aggregate['unmarked_state_exposure_rate']:.2%} |"
            )
        lines.append("")

    correction = summary.get("premise_correction", {})
    if correction:
        lines.append("## Premise Correction")
        lines.append(
            "| baseline | queries | correction records | correction items | "
            "corrected forbidden | unresolved forbidden |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for baseline, aggregate in correction.items():
            lines.append(
                f"| {baseline} | {aggregate['total']} | "
                f"{aggregate['correction_records']} | "
                f"{aggregate['correction_items']} | "
                f"{aggregate['corrected_forbidden_records']} | "
                f"{aggregate['unresolved_forbidden_records']} |"
            )
        lines.append("")

    evidence = summary.get("evidence_support", {})
    if evidence:
        lines.append("## Evidence Support")
        lines.append(
            "| baseline | evidence queries | evidence matched | evidence missing | "
            "graph evidence hits | graph retrievals |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for baseline, aggregate in evidence.items():
            lines.append(
                f"| {baseline} | {aggregate['evidence_query_total']} | "
                f"{aggregate['evidence_matched_records']} | "
                f"{aggregate['evidence_missing_records']} | "
                f"{aggregate['graph_evidence_hit_records']} | "
                f"{aggregate['graph_retrieval_records']} |"
            )
        lines.append("")

    answerability = summary.get("answerability", {})
    if answerability:
        lines.append("## Answerability Diagnostics")
        lines.append(
            "| baseline | answer queries | keyword matched | avg recall | "
            "basis keyword matched | avg basis recall | basis records |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for baseline, aggregate in answerability.items():
            lines.append(
                f"| {baseline} | {aggregate['answer_query_total']} | "
                f"{aggregate['answer_keyword_matched_records']} | "
                f"{_format_optional_rate(aggregate['answer_keyword_recall_avg'])} | "
                f"{aggregate['basis_answer_keyword_matched_records']} | "
                f"{_format_optional_rate(aggregate['basis_answer_keyword_recall_avg'])} | "
                f"{aggregate['answer_basis_records']} |"
            )
        lines.append("")

    for field_name, field_summary in summary["by_metadata"].items():
        if list(field_summary) == ["<missing>"]:
            continue
        lines.append(f"## By {field_name}")
        lines.append("| value | baseline | passed | accuracy |")
        lines.append("| --- | --- | ---: | ---: |")
        for value, by_baseline in field_summary.items():
            for baseline, aggregate in by_baseline.items():
                lines.append(
                    f"| {value} | {baseline} | {aggregate['passed']}/{aggregate['total']} | "
                    f"{aggregate['accuracy']:.2%} |"
                )
        lines.append("")

    for field_name, field_summary in summary.get("diagnostics_by_metadata", {}).items():
        if list(field_summary) == ["<missing>"]:
            continue
        if not _has_grouped_diagnostic_signal(field_summary):
            continue
        lines.append(f"## By {field_name} Diagnostics")
        lines.append(
            "| value | baseline | evidence support | answer recall | basis recall | "
            "basis matched |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for value, by_baseline in field_summary.items():
            for baseline, metrics in by_baseline.items():
                lines.append(
                    f"| {value} | {baseline} | "
                    f"{_format_fraction_rate(metrics['evidence_matched_records'], metrics['evidence_query_total'])} | "
                    f"{_format_optional_rate(metrics['answer_keyword_recall_avg'])} | "
                    f"{_format_optional_rate(metrics['basis_answer_keyword_recall_avg'])} | "
                    f"{metrics['basis_answer_keyword_matched_records']}/{metrics['answer_query_total']} |"
                )
        lines.append("")

    lines.append("## Failure Modes")
    lines.append("| mode | count |")
    lines.append("| --- | ---: |")
    for mode, count in sorted(summary["failure_modes"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {mode} | {count} |")
    lines.append("")

    failures = [record for record in records if not record["passed"]]
    lines.append("## Representative Failures")
    for record in failures[:max_examples]:
        metadata = ", ".join(
            f"{key}={value}" for key, value in record["metadata"].items()
            if key in set(group_fields) and value is not None
        )
        metadata = metadata or "metadata=<none>"
        first = record["retrieved"][0] if record["retrieved"] else "<none>"
        lines.append(
            f"- `{record['baseline']}` `{record['case_id']}/{record['query_id']}` "
            f"({metadata}) modes={record['failure_modes']}: {first}"
        )
    if not failures:
        lines.append("- No failures.")
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
    trace = [
        {
            "content": result.item.content,
            "kind": result.item.kind,
            "score": round(result.score, 4),
            "relation": result.relation,
            "contributions": {key: round(value, 4) for key, value in result.contributions.items()},
            "metadata": _trace_metadata(result.item),
        }
        for result in results
    ]
    text = "\n".join(retrieved).lower()
    non_correction_text = "\n".join(
        str(item.get("content") or "")
        for item in trace
        if not _is_premise_correction_trace(item)
    ).lower()
    has_expected = all(expected.lower() in text for expected in query.expected_substrings)
    has_forbidden = any(
        forbidden.lower() in non_correction_text
        for forbidden in query.forbidden_substrings
    )
    return QueryEvalResult(
        case_id=case_id,
        query_id=query.id or query.query,
        query=query.query,
        passed=has_expected and not has_forbidden,
        retrieved=retrieved,
        expected_substrings=query.expected_substrings,
        forbidden_substrings=query.forbidden_substrings,
        trace=trace,
        metadata=dict(query.metadata),
    )


def _query_record(baseline: str, query: QueryEvalResult) -> dict[str, Any]:
    text = "\n".join(query.retrieved).lower()
    correction_items = [item for item in query.trace if _is_premise_correction_trace(item)]
    correction_text = "\n".join(str(item.get("content") or "") for item in correction_items).lower()
    non_correction_text = "\n".join(
        str(item.get("content") or "")
        for item in query.trace
        if not _is_premise_correction_trace(item)
    ).lower()
    missing_expected = [
        expected for expected in query.expected_substrings if expected.lower() not in text
    ]
    answer_keywords = _answer_keywords(query.metadata)
    missing_answer_keywords = [
        keyword for keyword in answer_keywords
        if keyword not in text
    ]
    answer_keyword_recall = _keyword_recall(answer_keywords, text)
    answer_basis = (
        _trajectory_answer_basis(query.query, query.trace)
        if query.metadata.get("benchmark") == "ama" else ""
    )
    answer_basis_text = f"{text}\n{answer_basis.lower()}"
    basis_missing_answer_keywords = [
        keyword for keyword in answer_keywords
        if keyword not in answer_basis_text
    ]
    basis_answer_keyword_recall = _keyword_recall(answer_keywords, answer_basis_text)
    expected_evidence = _expected_evidence_labels(query.metadata)
    missing_evidence = [
        evidence for evidence in expected_evidence
        if not _evidence_label_hit(evidence, query.retrieved, query.trace)
    ]
    graph_retrieval_count = _graph_trace_count(query.trace)
    graph_items = _graph_trace_items(query.trace)
    graph_retrieved = [str(item.get("content") or "") for item in graph_items]
    graph_evidence_hits = [
        evidence for evidence in expected_evidence
        if _evidence_label_hit(evidence, graph_retrieved, graph_items)
    ]
    present_forbidden = [
        forbidden for forbidden in query.forbidden_substrings
        if forbidden.lower() in non_correction_text
    ]
    corrected_forbidden = [
        forbidden for forbidden in query.forbidden_substrings
        if forbidden.lower() in correction_text and forbidden.lower() not in non_correction_text
    ]
    state_retrieval_count = _state_trace_count(query.trace)
    retrieved_state_slots = _state_trace_slots(query.trace)
    expected_state_slots = _expected_state_slots(query.metadata)
    state_sensitive = bool(expected_state_slots)
    state_available = _state_available(query.metadata, state_sensitive=state_sensitive)
    state_readout_expected = state_sensitive and state_available
    unexpected_state_slots = [
        slot for slot in retrieved_state_slots
        if not state_slot_matches_query(slot, set(expected_state_slots))
    ] if state_sensitive else retrieved_state_slots
    state_slot_matched = (
        any(state_slot_matches_query(slot, set(expected_state_slots)) for slot in retrieved_state_slots)
        if state_sensitive else False
    )
    failure_modes: list[str] = []
    if missing_expected:
        failure_modes.append("expected_support_missing")
    if missing_evidence:
        failure_modes.append("evidence_support_missing")
    if present_forbidden:
        failure_modes.append("forbidden_support_present")
    if not query.retrieved:
        failure_modes.append("no_retrieval")
    if state_readout_expected and state_retrieval_count == 0:
        failure_modes.append("state_readout_missing")
    if state_readout_expected and unexpected_state_slots:
        failure_modes.append("state_readout_slot_mismatch")
    if not state_sensitive and state_retrieval_count > 0:
        failure_modes.append("state_readout_unmarked_exposure")
    return {
        "baseline": baseline,
        "case_id": query.case_id,
        "query_id": query.query_id,
        "query": query.query,
        "passed": query.passed,
        "expected_substrings": query.expected_substrings,
        "missing_expected": missing_expected,
        "expected_evidence": expected_evidence,
        "answer_keywords": answer_keywords,
        "missing_answer_keywords": missing_answer_keywords,
        "answer_keyword_recall": answer_keyword_recall,
        "answer_keyword_support_matched": (
            bool(answer_keywords) and answer_keyword_recall >= _ANSWER_KEYWORD_MATCH_THRESHOLD
        ),
        "answer_basis": answer_basis,
        "basis_missing_answer_keywords": basis_missing_answer_keywords,
        "basis_answer_keyword_recall": basis_answer_keyword_recall,
        "basis_answer_keyword_support_matched": (
            bool(answer_keywords) and basis_answer_keyword_recall >= _ANSWER_KEYWORD_MATCH_THRESHOLD
        ),
        "missing_evidence": missing_evidence,
        "evidence_support_matched": bool(expected_evidence) and not missing_evidence,
        "graph_retrieval_count": graph_retrieval_count,
        "graph_evidence_hits": graph_evidence_hits,
        "graph_evidence_hit_count": len(graph_evidence_hits),
        "forbidden_substrings": query.forbidden_substrings,
        "present_forbidden": present_forbidden,
        "corrected_forbidden": corrected_forbidden,
        "premise_correction_count": len(correction_items),
        "failure_modes": failure_modes,
        "metadata": dict(query.metadata),
        "retrieved": query.retrieved,
        "trace": query.trace,
        "state_retrieval_count": state_retrieval_count,
        "retrieved_state_slots": retrieved_state_slots,
        "expected_state_slots": expected_state_slots,
        "unexpected_state_slots": unexpected_state_slots,
        "state_slot_matched": state_slot_matched,
        "state_sensitive": state_sensitive,
        "state_available": state_available,
        "state_readout_expected": state_readout_expected,
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record["passed"])
    return {
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total else 0.0,
    }


def _state_exposure_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    state_sensitive_total = sum(1 for record in records if record["state_sensitive"])
    state_available_total = sum(1 for record in records if record["state_readout_expected"])
    state_unavailable_total = sum(
        1 for record in records
        if record["state_sensitive"] and not record["state_readout_expected"]
    )
    state_retrieval_records = sum(1 for record in records if record["state_retrieval_count"] > 0)
    state_slot_match_records = sum(
        1 for record in records
        if record["state_readout_expected"] and record["state_slot_matched"]
    )
    state_readout_missing_records = sum(
        1 for record in records
        if record["state_readout_expected"] and record["state_retrieval_count"] == 0
    )
    state_slot_mismatch_records = sum(
        1 for record in records
        if record["state_readout_expected"] and record["unexpected_state_slots"]
    )
    unmarked_records = [record for record in records if not record["state_sensitive"]]
    unmarked_state_retrieval_records = sum(
        1 for record in unmarked_records if record["state_retrieval_count"] > 0
    )
    return {
        "total": total,
        "state_sensitive_total": state_sensitive_total,
        "state_available_total": state_available_total,
        "state_unavailable_total": state_unavailable_total,
        "state_retrieval_records": state_retrieval_records,
        "state_exposure_rate": state_retrieval_records / total if total else 0.0,
        "state_slot_match_records": state_slot_match_records,
        "state_slot_match_rate": (
            state_slot_match_records / state_available_total if state_available_total else 0.0
        ),
        "state_readout_missing_records": state_readout_missing_records,
        "state_slot_mismatch_records": state_slot_mismatch_records,
        "unmarked_total": len(unmarked_records),
        "unmarked_state_retrieval_records": unmarked_state_retrieval_records,
        "unmarked_state_exposure_rate": (
            unmarked_state_retrieval_records / len(unmarked_records)
            if unmarked_records else 0.0
        ),
    }


def _premise_correction_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correction_records = sum(1 for record in records if int(record.get("premise_correction_count") or 0) > 0)
    correction_items = sum(int(record.get("premise_correction_count") or 0) for record in records)
    corrected_forbidden_records = sum(1 for record in records if record.get("corrected_forbidden"))
    corrected_forbidden_count = sum(len(record.get("corrected_forbidden") or []) for record in records)
    unresolved_forbidden_records = sum(1 for record in records if record.get("present_forbidden"))
    unresolved_forbidden_count = sum(len(record.get("present_forbidden") or []) for record in records)
    return {
        "total": total,
        "correction_records": correction_records,
        "correction_items": correction_items,
        "correction_rate": correction_records / total if total else 0.0,
        "corrected_forbidden_records": corrected_forbidden_records,
        "corrected_forbidden_count": corrected_forbidden_count,
        "corrected_forbidden_rate": corrected_forbidden_records / total if total else 0.0,
        "unresolved_forbidden_records": unresolved_forbidden_records,
        "unresolved_forbidden_count": unresolved_forbidden_count,
        "unresolved_forbidden_rate": unresolved_forbidden_records / total if total else 0.0,
    }


def _evidence_support_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_records = [record for record in records if record["expected_evidence"]]
    return {
        "total": len(records),
        "evidence_query_total": len(evidence_records),
        "evidence_matched_records": sum(1 for record in evidence_records if record["evidence_support_matched"]),
        "evidence_missing_records": sum(1 for record in evidence_records if record["missing_evidence"]),
        "graph_evidence_hit_records": sum(1 for record in evidence_records if record["graph_evidence_hit_count"] > 0),
        "graph_retrieval_records": sum(1 for record in records if record["graph_retrieval_count"] > 0),
    }


def _answerability_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    answer_records = [record for record in records if record["answer_keywords"]]
    basis_records = [record for record in answer_records if record["answer_basis"]]
    return {
        "total": len(records),
        "answer_query_total": len(answer_records),
        "answer_keyword_matched_records": sum(
            1 for record in answer_records if record["answer_keyword_support_matched"]
        ),
        "answer_keyword_recall_avg": _mean_or_none(
            record["answer_keyword_recall"] for record in answer_records
        ),
        "basis_answer_keyword_matched_records": sum(
            1 for record in answer_records if record["basis_answer_keyword_support_matched"]
        ),
        "basis_answer_keyword_recall_avg": _mean_or_none(
            record["basis_answer_keyword_recall"] for record in answer_records
        ),
        "answer_basis_records": len(basis_records),
    }


def _metadata_diagnostic_summary(
    records: list[dict[str, Any]],
    *,
    field_name: str,
    values: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    field_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for value in values:
        value_subset = [
            record
            for record in records
            if _metadata_group_value(record["metadata"], field_name) == value
        ]
        field_summary[value] = {}
        for baseline in sorted({str(record["baseline"]) for record in value_subset}):
            subset = [record for record in value_subset if record["baseline"] == baseline]
            evidence = _evidence_support_aggregate(subset)
            answerability = _answerability_aggregate(subset)
            field_summary[value][baseline] = {
                "total": len(subset),
                "evidence_query_total": evidence["evidence_query_total"],
                "evidence_matched_records": evidence["evidence_matched_records"],
                "evidence_missing_records": evidence["evidence_missing_records"],
                "answer_query_total": answerability["answer_query_total"],
                "answer_keyword_matched_records": answerability["answer_keyword_matched_records"],
                "answer_keyword_recall_avg": answerability["answer_keyword_recall_avg"],
                "basis_answer_keyword_matched_records": (
                    answerability["basis_answer_keyword_matched_records"]
                ),
                "basis_answer_keyword_recall_avg": (
                    answerability["basis_answer_keyword_recall_avg"]
                ),
                "answer_basis_records": answerability["answer_basis_records"],
            }
    return field_summary


def _paper_metrics_for_baseline(
    baseline: str,
    *,
    summary: dict[str, Any],
    reference: str | None,
) -> dict[str, Any]:
    support = summary["by_baseline"][baseline]
    exposure = summary["state_readout_exposure"][baseline]
    correction = summary["premise_correction"][baseline]
    evidence = summary["evidence_support"][baseline]
    answerability = summary["answerability"][baseline]
    pairwise = summary.get("pairwise_vs_first_baseline", {})
    comparison = pairwise.get(baseline)
    net_delta = 0 if baseline == reference else (comparison or {}).get("net_delta")
    state_total = exposure["state_available_total"]
    return {
        "support_passed": support["passed"],
        "support_total": support["total"],
        "support_accuracy": support["accuracy"],
        "net_delta_vs_reference": net_delta,
        "state_query_total": state_total,
        "state_sensitive_total": exposure["state_sensitive_total"],
        "state_unavailable_total": exposure["state_unavailable_total"],
        "state_slot_match_rate": _ratio_or_none(exposure["state_slot_match_records"], state_total),
        "state_readout_missing_rate": _ratio_or_none(exposure["state_readout_missing_records"], state_total),
        "state_slot_mismatch_rate": _ratio_or_none(exposure["state_slot_mismatch_records"], state_total),
        "evidence_query_total": evidence["evidence_query_total"],
        "evidence_support_rate": _ratio_or_none(
            evidence["evidence_matched_records"],
            evidence["evidence_query_total"],
        ),
        "graph_evidence_hit_rate": _ratio_or_none(
            evidence["graph_evidence_hit_records"],
            evidence["evidence_query_total"],
        ),
        "answer_keyword_recall_avg": answerability["answer_keyword_recall_avg"],
        "basis_answer_keyword_recall_avg": answerability["basis_answer_keyword_recall_avg"],
        "unmarked_state_exposure_rate": _ratio_or_none(
            exposure["unmarked_state_retrieval_records"],
            exposure["unmarked_total"],
        ),
        "premise_correction_rate": correction["correction_rate"],
        "corrected_forbidden_rate": correction["corrected_forbidden_rate"],
        "unresolved_forbidden_rate": correction["unresolved_forbidden_rate"],
    }


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean_or_none(values: Iterable[float]) -> float | None:
    numbers = list(values)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _format_optional_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _format_fraction_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({numerator / denominator:.2%})"


def _is_premise_correction_trace(item: dict[str, Any]) -> bool:
    return (
        item.get("kind") == "state_correction"
        or item.get("relation") == "state_premise_correction"
    )


def _has_grouped_diagnostic_signal(field_summary: dict[str, dict[str, dict[str, Any]]]) -> bool:
    for by_baseline in field_summary.values():
        for metrics in by_baseline.values():
            if metrics.get("evidence_query_total", 0) > 0:
                return True
            if metrics.get("answer_query_total", 0) > 0:
                return True
    return False


def _state_trace_count(trace: list[dict[str, Any]]) -> int:
    return sum(1 for item in trace if _trace_is_state(item))


def _state_trace_slots(trace: list[dict[str, Any]]) -> list[str]:
    slots: list[str] = []
    for item in trace:
        if not _trace_is_state(item):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        slot = metadata.get("state_slot") or metadata.get("kg_relation") or metadata.get("salient_slot")
        if slot is not None:
            slots.append(str(slot))
    return slots


def _trace_is_state(item: dict[str, Any]) -> bool:
    if item.get("kind") in {"state", "kg_fact", "salient_fact"}:
        return True
    contributions = item.get("contributions")
    return isinstance(contributions, dict) and "state_readout" in contributions


def _graph_trace_count(trace: list[dict[str, Any]]) -> int:
    return len(_graph_trace_items(trace))


def _graph_trace_items(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in trace:
        relation = str(item.get("relation") or "")
        contributions = item.get("contributions")
        if "graph" in relation or (isinstance(contributions, dict) and contributions.get("graph", 0) > 0):
            items.append(item)
    return items


def _expected_evidence_labels(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("evidence")
    if raw is None:
        raw = metadata.get("answer_session_ids")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def _evidence_label_hit(label: str, retrieved: list[str], trace: list[dict[str, Any]]) -> bool:
    normalized = label.lower()
    if any(normalized in text.lower() for text in retrieved):
        return True
    for item in trace:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key in ("memory_key", "label", "source_id"):
            value = metadata.get(key)
            if value is not None and str(value).lower().startswith(normalized):
                return True
        step = metadata.get("trajectory_step")
        if step is not None and _step_label_from_value(step).lower().startswith(normalized):
            return True
    return False


def _step_label_from_value(value: Any) -> str:
    try:
        return f"step{int(value):03d}"
    except (TypeError, ValueError):
        return str(value)


def _expected_state_slots(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("state_slot")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def _state_available(metadata: dict[str, Any], *, state_sensitive: bool) -> bool:
    if not state_sensitive:
        return False
    raw = metadata.get("state_available")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "unavailable"}
    return bool(raw)


def _trace_metadata(item: MemoryItem) -> dict[str, Any]:
    keys = (
        "memory_key",
        "label",
        "benchmark",
        "trajectory_step",
        "subject",
        "state_slot",
        "state_value",
        "kg_relation",
        "kg_object",
        "salient_slot",
        "salient_value",
        "source_id",
        "derived",
    )
    return {key: item.metadata[key] for key in keys if key in item.metadata}


_ANSWER_KEYWORD_MATCH_THRESHOLD = 0.35

_ANSWER_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "agent",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "directly",
    "does",
    "during",
    "each",
    "from",
    "given",
    "have",
    "into",
    "itself",
    "likely",
    "made",
    "more",
    "move",
    "moved",
    "moving",
    "must",
    "only",
    "other",
    "position",
    "question",
    "result",
    "same",
    "step",
    "steps",
    "that",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "toward",
    "towards",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def _answer_keywords(metadata: dict[str, Any]) -> list[str]:
    answer = metadata.get("answer")
    if answer is None:
        return []
    if isinstance(answer, list):
        text = " ".join(str(item) for item in answer)
    else:
        text = str(answer)
    tokens = [
        token
        for token in _keyword_tokens(text)
        if token not in _ANSWER_STOPWORDS
    ]
    return sorted(set(tokens))


def _keyword_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if len(token) >= 3
    ]


def _keyword_recall(keywords: list[str], text: str) -> float:
    if not keywords:
        return 0.0
    return sum(1 for keyword in keywords if keyword in text) / len(keywords)


def _trajectory_answer_basis(query: str, trace: list[dict[str, Any]]) -> str:
    allowed_steps = set(_query_step_indices(query))
    steps: dict[int, dict[str, list[str]]] = {}
    for item in trace:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        step = metadata.get("trajectory_step")
        if step is None:
            continue
        try:
            step_index = int(step)
        except (TypeError, ValueError):
            continue
        if allowed_steps and step_index not in allowed_steps:
            continue
        kind = str(item.get("kind") or "")
        content = str(item.get("content") or "")
        field = "action" if kind == "action" else "observation"
        steps.setdefault(step_index, {"action": [], "observation": []})[field].append(
            _strip_step_prefix(content)
        )
    if not steps:
        return ""

    lines: list[str] = []
    for step_index in sorted(steps):
        fields = steps[step_index]
        if fields["action"]:
            lines.append(f"Step {step_index} action: {'; '.join(fields['action'])}")
        if fields["observation"]:
            lines.append(f"Step {step_index} observation: {'; '.join(fields['observation'])}")
    lines.extend(_trajectory_basis_relations(steps))
    return "\n".join(lines)


def _query_step_indices(query: str) -> list[int]:
    steps: set[int] = set()
    for match in re.finditer(
        r"\b(?:from|between)?\s*steps?\s+(\d+)\s*(?:-|to|through|and)\s*(?:step\s+)?(\d+)\b",
        query,
        flags=re.IGNORECASE,
    ):
        start = int(match.group(1))
        end = int(match.group(2))
        if start <= end and end - start <= 20:
            steps.update(range(start, end + 1))
        elif end <= start and start - end <= 20:
            steps.update(range(end, start + 1))
        else:
            steps.update({start, end})
    for match in re.finditer(r"\bstep\s+(\d+)\b", query, flags=re.IGNORECASE):
        steps.add(int(match.group(1)))
    return sorted(steps)


def _strip_step_prefix(content: str) -> str:
    return re.sub(
        r"^\[step\d+\.(?:action|observation|state)\]\s*(?:action|observation|state):\s*",
        "",
        content.strip(),
        flags=re.IGNORECASE,
    )


def _trajectory_basis_relations(steps: dict[int, dict[str, list[str]]]) -> list[str]:
    lines: list[str] = []
    sorted_steps = sorted(steps)
    action_by_step = {
        step: _first_action_word(values["action"][0])
        for step, values in steps.items()
        if values["action"]
    }
    for previous, current in zip(sorted_steps, sorted_steps[1:]):
        previous_action = action_by_step.get(previous)
        current_action = action_by_step.get(current)
        if previous_action and current_action and _inverse_actions(previous_action, current_action):
            lines.append(
                f"Steps {previous}-{current} actions are inverse; they cancel out with zero net progress."
            )
    observation_texts = {
        step: _normalize_observation(values["observation"][0])
        for step, values in steps.items()
        if values["observation"]
    }
    for index, previous in enumerate(sorted_steps):
        for current in sorted_steps[index + 1:]:
            if observation_texts.get(previous) and observation_texts.get(previous) == observation_texts.get(current):
                lines.append(
                    f"Steps {previous} and {current} have identical observations, indicating state reversion."
                )
    lines.extend(_trajectory_state_relations(steps, action_by_step, observation_texts))
    return lines


def _trajectory_state_relations(
    steps: dict[int, dict[str, list[str]]],
    action_by_step: dict[int, str],
    observation_texts: dict[int, str],
) -> list[str]:
    lines: list[str] = []
    sorted_steps = sorted(steps)
    for previous, current in zip(sorted_steps, sorted_steps[1:]):
        previous_observation = observation_texts.get(previous)
        current_observation = observation_texts.get(current)
        current_action = action_by_step.get(current)
        if previous_observation and current_observation and previous_observation == current_observation:
            if current_action:
                lines.append(
                    f"Step {current} action {current_action} caused no observable state change; it was ineffective."
                )
            lines.append(f"Steps {previous}-{current} made no progress because the observation did not change.")

    repeated: list[int] = []
    for step in sorted_steps:
        action = action_by_step.get(step)
        if repeated and action_by_step.get(repeated[-1]) == action:
            repeated.append(step)
        else:
            if len(repeated) >= 2:
                lines.extend(_repeated_action_relations(repeated, action_by_step, observation_texts))
            repeated = [step] if action else []
    if len(repeated) >= 2:
        lines.extend(_repeated_action_relations(repeated, action_by_step, observation_texts))

    for step in sorted_steps:
        if not steps[step]["observation"]:
            continue
        observation = steps[step]["observation"][0]
        action = action_by_step.get(step)
        lines.extend(_rule_object_relations(step, action, observation))
    return _dedupe_preserve_order(lines)


def _repeated_action_relations(
    repeated_steps: list[int],
    action_by_step: dict[int, str],
    observation_texts: dict[int, str],
) -> list[str]:
    action = action_by_step.get(repeated_steps[0])
    if not action:
        return []
    normalized = [observation_texts.get(step) for step in repeated_steps]
    if normalized and all(item and item == normalized[0] for item in normalized):
        return [
            f"Steps {repeated_steps[0]}-{repeated_steps[-1]} repeat action {action} with unchanged observations; the action is blocked and makes no progress."
        ]
    return [f"Steps {repeated_steps[0]}-{repeated_steps[-1]} repeat action {action}."]


def _rule_object_relations(step: int, action: str | None, observation: str) -> list[str]:
    lines: list[str] = []
    active_rules = _active_rules(observation)
    for subject, predicate in active_rules:
        lines.append(f"Step {step} active rule: {subject} is {predicate}.")
        if predicate == "stop":
            lines.append(f"Step {step} rule {subject} is stop makes {subject} objects impassable.")
        if predicate == "win":
            lines.append(f"Step {step} rule {subject} is win marks {subject} as a win object.")
        if predicate == "you":
            lines.append(f"Step {step} rule {subject} is you means the agent controls {subject}.")

    if action:
        blocking_object = _adjacent_object_for_action(observation, action)
        if blocking_object:
            predicates = [predicate for subject, predicate in active_rules if subject == blocking_object]
            if "stop" in predicates:
                lines.append(
                    f"Step {step} action {action} is blocked by adjacent {blocking_object} due to {blocking_object} is stop."
                )
            else:
                lines.append(f"Step {step} action {action} faces adjacent {blocking_object}.")
    return lines


def _active_rules(observation: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    in_rules = False
    for raw_line in observation.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        if line.startswith("active rules"):
            in_rules = True
            continue
        if line.startswith("objects on the map"):
            break
        if not in_rules:
            continue
        match = re.fullmatch(r"`?([a-z][a-z0-9_-]*)`?\s+is\s+`?([a-z][a-z0-9_-]*)`?", line)
        if match:
            rules.append((match.group(1), match.group(2)))
    return rules


def _adjacent_object_for_action(observation: str, action: str) -> str | None:
    direction = {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
    }.get(action)
    if not direction:
        return None
    pattern = re.compile(
        rf"^([a-z][a-z0-9_-]*)\s+1\s+steps?\s+to\s+the\s+{direction}$"
        if direction in {"left", "right"}
        else rf"^([a-z][a-z0-9_-]*)\s+1\s+steps?\s+{direction}$",
        flags=re.IGNORECASE,
    )
    for raw_line in observation.splitlines():
        line = raw_line.strip().strip("`").lower()
        if line.startswith("rule "):
            continue
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _first_action_word(text: str) -> str | None:
    match = re.search(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
    return match.group(0) if match else None


def _inverse_actions(left: str, right: str) -> bool:
    return (left, right) in {
        ("up", "down"),
        ("down", "up"),
        ("left", "right"),
        ("right", "left"),
    }


def _normalize_observation(text: str) -> str:
    return " ".join(_keyword_tokens(text))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _metadata_group_value(metadata: dict[str, Any], field_name: str) -> str:
    value = metadata.get(field_name)
    if value is None:
        return "<missing>"
    return str(value)


def _paired_comparison(
    records: list[dict[str, Any]],
    *,
    reference: str,
    candidate: str,
    group_fields: Iterable[str],
) -> dict[str, Any]:
    reference_records = {
        _record_key(record): record
        for record in records
        if record["baseline"] == reference
    }
    candidate_records = {
        _record_key(record): record
        for record in records
        if record["baseline"] == candidate
    }
    common_keys = sorted(set(reference_records) & set(candidate_records))
    base = _paired_counts(reference_records, candidate_records, common_keys)
    base["reference"] = reference
    base["candidate"] = candidate
    base["by_metadata"] = {}
    for field_name in group_fields:
        values = sorted({
            _metadata_group_value(reference_records[key]["metadata"], field_name)
            for key in common_keys
        })
        base["by_metadata"][field_name] = {
            value: _paired_counts(
                reference_records,
                candidate_records,
                [
                    key for key in common_keys
                    if _metadata_group_value(reference_records[key]["metadata"], field_name) == value
                ],
            )
            for value in values
        }
    return base


def _paired_counts(
    reference_records: dict[tuple[str, str], dict[str, Any]],
    candidate_records: dict[tuple[str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
) -> dict[str, Any]:
    both_pass = 0
    both_fail = 0
    gained = 0
    lost = 0
    for key in keys:
        reference_passed = bool(reference_records[key]["passed"])
        candidate_passed = bool(candidate_records[key]["passed"])
        if reference_passed and candidate_passed:
            both_pass += 1
        elif not reference_passed and not candidate_passed:
            both_fail += 1
        elif candidate_passed:
            gained += 1
        else:
            lost += 1
    return {
        "common_total": len(keys),
        "both_pass": both_pass,
        "both_fail": both_fail,
        "gained_passes": gained,
        "lost_passes": lost,
        "net_delta": gained - lost,
    }


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record["case_id"]), str(record["query_id"]))


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
            metadata=dict(entry.get("metadata", {})),
        )
        for entry in raw.get("queries", raw.get("qas", []))
    ]
    return MemoryQACase(id=case_id, observations=observations, queries=queries)
