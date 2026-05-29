from __future__ import annotations

from typing import Any


def jsonl_failure_attributions(record: dict[str, Any]) -> list[str]:
    """Map JSONL retrieval diagnostics to conservative paper-facing causes."""

    if record.get("passed") is True:
        return []
    modes = set(str(mode) for mode in record.get("failure_modes") or [])
    attributions: list[str] = []
    expected_state = bool(record.get("expected_state_slots"))
    state_available = bool(record.get("state_readout_expected"))
    state_count = int(record.get("state_memory_count") or 0)

    if "no_retrieval" in modes:
        attributions.append("retrieval_failure")
    if "evidence_support_missing" in modes:
        attributions.append("retrieval_failure")
    if "state_readout_slot_mismatch" in modes:
        attributions.append("state_routing_failure")
    if "state_readout_unmarked_exposure" in modes:
        attributions.append("state_routing_failure")
    if "state_readout_missing" in modes:
        if state_count == 0:
            attributions.append("state_authority_absent_or_extraction_failure")
        else:
            attributions.append("state_readout_failure")
    if "expected_support_missing" in modes:
        if state_available and expected_state and state_count == 0:
            attributions.append("state_authority_absent_or_extraction_failure")
        elif state_available and expected_state and not record.get("state_slot_matched"):
            attributions.append("state_readout_failure")
        elif not attributions:
            attributions.append("retrieval_failure")
    if "forbidden_support_present" in modes:
        if state_count > 0:
            attributions.append("stale_adjudication_failure")
        else:
            attributions.append("stale_filtering_or_baseline_limitation")
    return _dedupe(attributions)


def stale_failure_attributions(record: dict[str, Any]) -> list[str]:
    """Map STALE retrieval diagnostics to conservative paper-facing causes."""

    modes = set(str(mode) for mode in record.get("failure_modes") or [])
    attributions: list[str] = []
    active_state_count = int(record.get("active_state_count") or 0)
    stale_state_count = int(record.get("stale_state_count") or 0)

    if "current_evidence_not_recalled" in modes:
        if active_state_count == 0:
            attributions.append("state_authority_absent_or_extraction_failure")
        else:
            attributions.append("retrieval_or_readout_failure")
    if "stale_ranked_before_current" in modes:
        attributions.append("ranking_failure")
    if "stale_evidence_exposed" in modes or "old_support_not_fully_adjudicated" in modes:
        if stale_state_count > 0 or int(record.get("adjudicated_old_supports") or 0) > 0:
            attributions.append("stale_adjudication_failure")
        else:
            attributions.append("stale_adjudication_missing")
    if "premise_correction_missing" in modes:
        attributions.append("premise_correction_failure")
    return _dedupe(attributions)


def attribution_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for attribution in record.get("failure_attributions") or []:
            key = str(attribution)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def attribution_counts_by_baseline(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for record in records:
        baseline = str(record.get("baseline") or "?")
        counts = nested.setdefault(baseline, {})
        for attribution in record.get("failure_attributions") or []:
            key = str(attribution)
            counts[key] = counts.get(key, 0) + 1
    return {
        baseline: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        for baseline, counts in sorted(nested.items())
    }


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
