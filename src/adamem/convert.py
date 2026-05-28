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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert public memory benchmarks to AdaMem JSONL.")
    sub = parser.add_subparsers(dest="command", required=True)

    locomo = sub.add_parser("locomo", help="Convert LoCoMo locomo10.json to AdaMem JSONL")
    locomo.add_argument("input", type=Path)
    locomo.add_argument("output", type=Path)
    locomo.add_argument("--expected", choices=["evidence", "answer", "both"], default="evidence")
    locomo.add_argument("--top-k", type=int, default=8)
    locomo.add_argument("--limit", type=int)

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
