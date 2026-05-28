from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

SESSION_RE = re.compile(r"session_(\d+)$")


def convert_locomo_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected: str = "evidence",
    top_k: int = 8,
    limit: int | None = None,
) -> int:
    with Path(input_path).open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError("LoCoMo input must be a JSON array")
    if limit is not None:
        samples = samples[:limit]

    rows = [convert_locomo_sample(sample, expected=expected, top_k=top_k) for sample in samples]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def convert_locomo_sample(sample: dict[str, Any], *, expected: str = "evidence", top_k: int = 8) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or "locomo-sample")
    conversation = sample.get("conversation", {})
    observations = list(_locomo_observations(conversation))
    queries = [
        _locomo_query(qa, index=index, expected=expected, top_k=top_k)
        for index, qa in enumerate(sample.get("qa", []), start=1)
        if int(qa.get("category", 0) or 0) != 5
    ]
    return {
        "id": sample_id,
        "observations": observations,
        "queries": queries,
    }


def convert_stale_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int = 8,
    limit: int | None = None,
    types: list[str] | None = None,
) -> int:
    """Convert STALE T1_T2_400_FULL.json to AdaMem JSONL.

    STALE schema (per instance):
      uid, M_old, M_new, explanation, type ("T1"|"T2"),
      probing_queries: {dim1_query, dim2_query, dim3_query},
      relevant_session_index: [int, int],
      timestamps: [str x 50],
      haystack_session: [[ {role, content}, ...] x 50]

    We emit ONE AdaMem case per STALE instance. Each haystack turn becomes
    an observation with valid_from = session timestamp. Each instance yields
    THREE probing queries (one per dim), all carrying STALE metadata so an
    LLM judge can score them later (`expected_substrings` is unused for SR/PR/IPA;
    we leave it empty and rely on judge-mode eval).
    """
    with Path(input_path).open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError("STALE input must be a JSON array")
    if types:
        samples = [s for s in samples if s.get("type") in types]
    if limit is not None:
        samples = samples[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = convert_stale_sample(sample, top_k=top_k)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(samples)


def convert_stale_sample(sample: dict[str, Any], *, top_k: int = 8) -> dict[str, Any]:
    uid = str(sample.get("uid") or "stale-sample")
    sample_type = str(sample.get("type") or "T?")
    sessions = sample.get("haystack_session") or []
    timestamps = sample.get("timestamps") or []
    relevant = set(int(idx) for idx in sample.get("relevant_session_index") or [])

    observations: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        timestamp = timestamps[session_index] if session_index < len(timestamps) else None
        for turn_index, turn in enumerate(session or []):
            role = str(turn.get("role") or "user")
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            label = f"s{session_index:02d}t{turn_index:02d}"
            observations.append({
                "label": label,
                "content": f"[{timestamp}] {role}: {content}",
                "kind": "dialogue",
                "importance": 0.6 if session_index in relevant else 0.4,
                "valid_from": timestamp,
                "metadata": {
                    "memory_key": label,
                    "subject": role,
                    "tags": ["stale", sample_type, f"session_{session_index}"]
                            + (["relevant"] if session_index in relevant else []),
                },
            })

    probing = sample.get("probing_queries") or {}
    queries: list[dict[str, Any]] = []
    for dim_index, dim_key in enumerate(("dim1_query", "dim2_query", "dim3_query"), start=1):
        text = str(probing.get(dim_key) or "").strip()
        if not text:
            continue
        queries.append({
            "id": f"{uid}.dim{dim_index}",
            "query": text,
            "expected_substrings": [],
            "top_k": top_k,
            "metadata": {
                "stale_uid": uid,
                "stale_type": sample_type,
                "stale_dim": dim_index,
                "M_old": sample.get("M_old"),
                "M_new": sample.get("M_new"),
                "explanation": sample.get("explanation"),
                "relevant_session_index": list(sample.get("relevant_session_index") or []),
            },
        })

    return {
        "id": uid,
        "metadata": {
            "type": sample_type,
            "M_old": sample.get("M_old"),
            "M_new": sample.get("M_new"),
            "explanation": sample.get("explanation"),
        },
        "observations": observations,
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert public memory benchmarks to AdaMem JSONL.")
    sub = parser.add_subparsers(dest="command", required=True)

    locomo = sub.add_parser("locomo", help="Convert LoCoMo locomo10.json to AdaMem JSONL")
    locomo.add_argument("input", type=Path)
    locomo.add_argument("output", type=Path)
    locomo.add_argument("--expected", choices=["evidence", "answer", "both"], default="evidence")
    locomo.add_argument("--top-k", type=int, default=8)
    locomo.add_argument("--limit", type=int)

    stale = sub.add_parser("stale", help="Convert STALE T1_T2_400_FULL.json to AdaMem JSONL")
    stale.add_argument("input", type=Path)
    stale.add_argument("output", type=Path)
    stale.add_argument("--top-k", type=int, default=8)
    stale.add_argument("--limit", type=int)
    stale.add_argument("--types", nargs="+", choices=["T1", "T2"], help="Filter by conflict type")

    args = parser.parse_args()
    if args.command == "locomo":
        count = convert_locomo_file(
            args.input,
            args.output,
            expected=args.expected,
            top_k=args.top_k,
            limit=args.limit,
        )
        print(f"wrote {count} cases to {args.output}")
    elif args.command == "stale":
        count = convert_stale_file(
            args.input,
            args.output,
            top_k=args.top_k,
            limit=args.limit,
            types=args.types,
        )
        print(f"wrote {count} STALE cases to {args.output}")


def _locomo_observations(conversation: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in sorted(conversation, key=_session_sort_key):
        match = SESSION_RE.fullmatch(key)
        if not match:
            continue
        session_index = int(match.group(1))
        session_date = conversation.get(f"{key}_date_time")
        turns = conversation.get(key) or []
        for turn in turns:
            dia_id = str(turn.get("dia_id") or "")
            speaker = str(turn.get("speaker") or "speaker")
            text = str(turn.get("text") or "").strip()
            image_caption = str(turn.get("blip_caption") or "").strip()
            parts = [part for part in [dia_id, session_date, f"{speaker}: {text}", image_caption] if part]
            yield {
                "label": dia_id or None,
                "content": " | ".join(parts),
                "kind": "dialogue",
                "importance": 0.5,
                "valid_from": session_date,
                "metadata": {
                    "memory_key": dia_id or f"session.{session_index}",
                    "subject": speaker,
                    "tags": ["locomo", f"session_{session_index}"],
                },
            }


def _locomo_query(qa: dict[str, Any], *, index: int, expected: str, top_k: int) -> dict[str, Any]:
    evidence = [str(item) for item in qa.get("evidence", [])]
    answer = qa.get("answer", "")
    answers = answer if isinstance(answer, list) else [answer]
    answer_strings = [str(item) for item in answers if str(item)]
    if expected == "evidence":
        expected_substrings = evidence
    elif expected == "answer":
        expected_substrings = answer_strings
    else:
        expected_substrings = evidence + answer_strings
    return {
        "id": str(qa.get("question_id") or qa.get("id") or f"q{index}"),
        "query": str(qa.get("question") or ""),
        "expected_substrings": expected_substrings,
        "top_k": top_k,
        "metadata": {
            "category": qa.get("category"),
            "answer": answer,
            "evidence": evidence,
        },
    }


def _session_sort_key(key: str) -> tuple[int, str]:
    match = SESSION_RE.fullmatch(key)
    if not match:
        return (10**9, key)
    return (int(match.group(1)), key)


if __name__ == "__main__":
    main()
