from __future__ import annotations

from typing import Any, Iterable

from adamem.schema import MemoryItem


def state_memory_inventory(items: Iterable[MemoryItem]) -> dict[str, Any]:
    """Summarize derived state memories without reading benchmark labels."""

    state_items = [
        item
        for item in items
        if item.kind == "state" or item.metadata.get("state_slot") is not None
    ]
    active = [item for item in state_items if item.active]
    stale = [item for item in state_items if not item.active or item.staleness > 0]
    unknown_current = [
        item
        for item in state_items
        if item.metadata.get("state_status") == "unknown_current"
    ]
    return {
        "state_memory_count": len(state_items),
        "active_state_count": len(active),
        "stale_state_count": len(stale),
        "unknown_current_state_count": len(unknown_current),
        "state_slots": _state_slots(state_items),
        "active_state_slots": _state_slots(active),
        "stale_state_slots": _state_slots(stale),
        "unknown_current_state_slots": _state_slots(unknown_current),
    }


def state_inventory_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    state_counts = [int(record.get("state_memory_count") or 0) for record in records]
    active_counts = [int(record.get("active_state_count") or 0) for record in records]
    stale_counts = [int(record.get("stale_state_count") or 0) for record in records]
    unknown_counts = [int(record.get("unknown_current_state_count") or 0) for record in records]
    return {
        "records": total,
        "records_with_state_memory": sum(1 for count in state_counts if count > 0),
        "avg_state_memory_count": _average(state_counts),
        "max_state_memory_count": max(state_counts) if state_counts else 0,
        "avg_active_state_count": _average(active_counts),
        "max_active_state_count": max(active_counts) if active_counts else 0,
        "avg_stale_state_count": _average(stale_counts),
        "max_stale_state_count": max(stale_counts) if stale_counts else 0,
        "avg_unknown_current_state_count": _average(unknown_counts),
        "state_slots": _record_slots(records, "state_slots"),
        "active_state_slots": _record_slots(records, "active_state_slots"),
        "stale_state_slots": _record_slots(records, "stale_state_slots"),
        "unknown_current_state_slots": _record_slots(records, "unknown_current_state_slots"),
    }


def _state_slots(items: Iterable[MemoryItem]) -> list[str]:
    return sorted(
        {
            str(item.metadata.get("state_slot"))
            for item in items
            if item.metadata.get("state_slot")
        }
    )


def _record_slots(records: list[dict[str, Any]], field_name: str) -> list[str]:
    slots: set[str] = set()
    for record in records:
        for slot in record.get(field_name) or []:
            slots.add(str(slot))
    return sorted(slots)


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
