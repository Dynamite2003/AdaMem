from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from adamem.state import extract_state_patches, query_relevant_state_slots, state_slot_matches_query

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

    Evaluation-only state/dependency opportunity labels derived from STALE
    metadata are written only to query metadata. They are for grouping and
    diagnostics, not runtime memory input.
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


def annotate_stale_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Backfill STALE query-only opportunity labels into AdaMem JSONL.

    This is for already-converted STALE JSONL when the original STALE JSON is
    unavailable. It updates only query metadata and leaves observations
    untouched, preserving the runtime memory leakage boundary.
    """

    count = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                _annotate_stale_row(row)
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _annotate_stale_row(row: dict[str, Any]) -> None:
    queries = row.get("queries")
    if not isinstance(queries, list):
        return
    for query in queries:
        if not isinstance(query, dict):
            continue
        metadata = query.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            query["metadata"] = metadata
        source = str(metadata.get("state_slot_source") or "")
        if source and source != "stale_metadata_heuristic":
            continue
        annotation = _stale_evaluation_state_metadata(
            metadata,
            str(query.get("query") or ""),
        )
        if annotation:
            metadata.update(annotation)


def convert_longmemeval_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected: str = "evidence",
    top_k: int = 8,
    limit: int | None = None,
    question_types: list[str] | None = None,
    limit_per_type: int | None = None,
    infer_state_slots: bool = False,
    state_audit_input: str | Path | None = None,
    state_audit_output: str | Path | None = None,
    state_audit_summary_output: str | Path | None = None,
) -> int:
    """Convert LongMemEval JSON to AdaMem JSONL.

    Official LongMemEval instances include `question_id`, `question_type`,
    `question`, `answer`, `question_date`, `haystack_session_ids`,
    `haystack_dates`, `haystack_sessions`, and `answer_session_ids`.

    Evidence labels and answers are stored on queries for evaluation only. They
    are not written into observation metadata so runtime retrieval cannot use
    ground-truth evidence labels.
    """

    with Path(input_path).open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError("LongMemEval input must be a JSON array")
    samples = _select_longmemeval_samples(
        samples,
        limit=limit,
        question_types=question_types,
        limit_per_type=limit_per_type,
    )
    state_audit_labels = load_state_audit_labels(state_audit_input) if state_audit_input else {}

    rows = [
        convert_longmemeval_sample(
            sample,
            expected=expected,
            top_k=top_k,
            infer_state_slots=infer_state_slots,
            state_audit_label=state_audit_labels.get(_longmemeval_question_id(sample)),
        )
        for sample in samples
    ]
    if state_audit_output:
        write_longmemeval_state_audit_file(samples, state_audit_output)
    if state_audit_summary_output:
        write_longmemeval_state_audit_summary_file(samples, state_audit_summary_output)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def convert_ama_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected: str = "answer",
    top_k: int = 8,
    limit: int | None = None,
) -> int:
    """Convert AMA-Bench-style agent trajectories to AdaMem JSONL.

    The converter accepts a permissive JSON/JSONL schema so exported public
    records or local reproductions can be adapted without coupling runtime
    memory code to one dataset package. Trajectory actions and observations are
    emitted as runtime observations with action->observation `cause_labels`.
    Answers and evidence labels stay query-only for evaluation.
    """

    samples = _load_json_records(input_path)
    if limit is not None:
        samples = samples[:limit]
    rows = [convert_ama_sample(sample, expected=expected, top_k=top_k) for sample in samples]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def convert_longmemeval_v2_file(
    questions_path: str | Path,
    trajectories_path: str | Path,
    haystack_path: str | Path,
    output_path: str | Path,
    *,
    expected: str = "answer",
    top_k: int = 8,
    limit: int | None = None,
    question_types: list[str] | None = None,
    question_ids: list[str] | None = None,
    limit_per_type: int | None = None,
    max_trajectories_per_question: int | None = None,
    infer_state_slots: bool = True,
) -> int:
    """Convert LongMemEval-V2 text trajectories to AdaMem JSONL.

    The public schema stores questions, trajectories, and haystack mappings in
    separate files. Reference answers and evaluator strings remain query-only;
    runtime observations are built only from trajectory fields.
    """

    questions = _load_json_records(questions_path)
    selected = _select_longmemeval_v2_questions(
        questions,
        limit=limit,
        question_types=question_types,
        question_ids=question_ids,
        limit_per_type=limit_per_type,
    )
    haystacks = _load_longmemeval_v2_haystacks(haystack_path)
    selected_haystacks = {
        _longmemeval_v2_question_id(question): _limit_list(
            haystacks.get(_longmemeval_v2_question_id(question), []),
            max_trajectories_per_question,
        )
        for question in selected
    }
    needed_trajectory_ids = {
        trajectory_id
        for trajectory_ids in selected_haystacks.values()
        for trajectory_id in trajectory_ids
    }
    trajectories = _load_selected_records_by_id(trajectories_path, needed_trajectory_ids)

    rows = [
        convert_longmemeval_v2_sample(
            question,
            trajectories=trajectories,
            haystack_trajectory_ids=selected_haystacks.get(_longmemeval_v2_question_id(question), []),
            expected=expected,
            top_k=top_k,
            infer_state_slots=infer_state_slots,
        )
        for question in selected
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def convert_ama_sample(sample: Mapping[str, Any], *, expected: str = "answer", top_k: int = 8) -> dict[str, Any]:
    sample_id = str(_first_present(sample, "episode_id", "task_id", "id", default="ama-sample"))
    observations = list(_ama_observations(sample))
    queries = [
        _ama_query(sample, qa, index=index, expected=expected, top_k=top_k)
        for index, qa in enumerate(_ama_qa_records(sample), start=1)
    ]
    return {
        "id": sample_id,
        "metadata": {
            "benchmark": "ama",
            "domain": sample.get("domain"),
            "task_type": sample.get("task_type") or sample.get("category"),
            "source": sample.get("source"),
            "success": sample.get("success"),
            "num_turns": sample.get("num_turns"),
            "total_tokens": sample.get("total_tokens"),
        },
        "observations": observations,
        "queries": queries,
    }


def _load_json_records(input_path: str | Path) -> list[dict[str, Any]]:
    text = Path(input_path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("Expected a JSON array")
        return [dict(record) for record in records]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record {line_number} must be an object")
        records.append(record)
    return records


def _load_selected_records_by_id(input_path: str | Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    path = Path(input_path)
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        first_char = ""
        while True:
            char = handle.read(1)
            if not char:
                break
            if char.strip():
                first_char = char
                break
        handle.seek(0)
        if not first_char:
            return {}
        if first_char == "[":
            records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError("Expected a JSON array")
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                record_id = str(record.get("id") or record.get("trajectory_id") or "")
                if record_id in ids:
                    selected[record_id] = dict(record)
            return selected

        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"JSONL record {line_number} must be an object")
            record_id = str(record.get("id") or record.get("trajectory_id") or "")
            if record_id in ids:
                selected[record_id] = dict(record)
            if len(selected) == len(ids):
                break
    return selected


def _load_longmemeval_v2_haystacks(input_path: str | Path) -> dict[str, list[str]]:
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("LongMemEval-V2 haystack input must be a JSON object")
    haystacks: dict[str, list[str]] = {}
    for question_id, trajectory_ids in raw.items():
        haystacks[str(question_id)] = [str(item) for item in _as_list(trajectory_ids) if str(item)]
    return haystacks


def _limit_list(items: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return list(items)
    return list(items[:limit])


def _ama_observations(sample: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    trajectory = sample.get("trajectory") or sample.get("steps") or sample.get("history") or []
    if isinstance(trajectory, Mapping):
        trajectory = trajectory.get("steps") or trajectory.get("trajectory") or []
    for index, step in enumerate(trajectory):
        step_index = _ama_step_index(step, fallback=index)
        if not isinstance(step, Mapping):
            content = _stringify(step)
            if not content:
                continue
            label = f"step{step_index:03d}.observation"
            yield _ama_observation_row(
                sample,
                label=label,
                content=f"[{label}] {content}",
                kind="observation",
                step_index=step_index,
                source_field="trajectory",
                cause_labels=[],
            )
            continue

        action = _first_text(step, "action", "command", "tool_call", "tool_input")
        action_label = f"step{step_index:03d}.action"
        cause_labels: list[str] = []
        if action:
            yield _ama_observation_row(
                sample,
                label=action_label,
                content=f"[{action_label}] action: {action}",
                kind="action",
                step_index=step_index,
                source_field="action",
                cause_labels=[],
            )
            cause_labels = [action_label]

        observation = _first_text(step, "observation", "result", "tool_output", "response")
        if observation:
            label = f"step{step_index:03d}.observation"
            yield _ama_observation_row(
                sample,
                label=label,
                content=f"[{label}] observation: {observation}",
                kind="observation",
                step_index=step_index,
                source_field="observation",
                cause_labels=cause_labels,
            )

        state = _first_text(step, "state", "world_state", "environment_state")
        if state:
            label = f"step{step_index:03d}.state"
            yield _ama_observation_row(
                sample,
                label=label,
                content=f"[{label}] state: {state}",
                kind="trajectory_state",
                step_index=step_index,
                source_field="state",
                cause_labels=cause_labels,
            )


def _ama_observation_row(
    sample: Mapping[str, Any],
    *,
    label: str,
    content: str,
    kind: str,
    step_index: int,
    source_field: str,
    cause_labels: list[str],
) -> dict[str, Any]:
    tags = ["ama", f"step_{step_index}", source_field]
    for key in ("domain", "task_type", "category"):
        value = sample.get(key)
        if value:
            tags.append(str(value))
    return {
        "label": label,
        "content": content,
        "kind": kind,
        "importance": 0.55 if kind == "action" else 0.65,
        "cause_labels": cause_labels,
        "metadata": {
            "memory_key": label,
            "subject": source_field,
            "tags": tags,
            "trajectory_step": step_index,
            "benchmark": "ama",
        },
    }


def _ama_qa_records(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = sample.get("qa_pairs") or sample.get("questions") or sample.get("qas") or sample.get("qa") or []
    if isinstance(raw, Mapping):
        return [raw]
    if raw:
        return [qa for qa in raw if isinstance(qa, Mapping)]
    if sample.get("question"):
        return [sample]
    return []


def _ama_query(
    sample: Mapping[str, Any],
    qa: Mapping[str, Any],
    *,
    index: int,
    expected: str,
    top_k: int,
) -> dict[str, Any]:
    answer = qa.get("answer", qa.get("expected_answer", ""))
    answers = answer if isinstance(answer, list) else [answer]
    answer_strings = [str(item) for item in answers if str(item)]
    evidence = [
        _ama_evidence_label(item)
        for key in ("evidence", "evidence_steps", "supporting_steps", "supporting_step_ids", "answer_step_ids")
        for item in _as_list(qa.get(key))
    ]
    if not evidence:
        evidence = _ama_step_evidence_from_question(str(qa.get("question") or qa.get("query") or ""))
    evidence = [item for item in evidence if item]
    question_type = qa.get("question_type") or qa.get("type")
    if expected == "evidence":
        expected_substrings = evidence
    elif expected == "both":
        expected_substrings = evidence + answer_strings
    else:
        expected_substrings = answer_strings
    return {
        "id": str(_first_present(qa, "question_id", "question_uuid", "id", default=f"q{index}")),
        "query": str(qa.get("question") or qa.get("query") or ""),
        "expected_substrings": expected_substrings,
        "top_k": top_k,
        "metadata": {
            "benchmark": "ama",
            "question_type": question_type,
            "question_type_name": _ama_question_type_name(question_type),
            "domain": sample.get("domain"),
            "task_type": sample.get("task_type") or sample.get("category"),
            "answer": answer,
            "evidence": evidence,
        },
}


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _ama_step_index(step: Any, *, fallback: int) -> int:
    if not isinstance(step, Mapping):
        return fallback
    raw = step.get("turn_idx", step.get("step", step.get("step_idx", fallback)))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _ama_evidence_label(value: Any) -> str:
    if isinstance(value, int):
        return f"step{value:03d}"
    text = str(value)
    if text.isdigit():
        return f"step{int(text):03d}"
    return text


def _ama_step_evidence_from_question(question: str) -> list[str]:
    steps: set[int] = set()
    for match in re.finditer(
        r"\b(?:from|between)?\s*steps?\s+(\d+)\s*(?:-|to|through|and)\s*(?:step\s+)?(\d+)\b",
        question,
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
    for match in re.finditer(r"\bstep\s+(\d+)\b", question, flags=re.IGNORECASE):
        steps.add(int(match.group(1)))
    return [_ama_evidence_label(step) for step in sorted(steps)]


def _ama_question_type_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return {
        "A": "Recall",
        "B": "Causal Inference",
        "C": "State Updating",
        "D": "State Abstraction",
    }.get(text, text)


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in mapping:
            continue
        text = _stringify(mapping.get(key))
        if text:
            return text
    return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _select_longmemeval_samples(
    samples: list[dict[str, Any]],
    *,
    limit: int | None = None,
    question_types: list[str] | None = None,
    limit_per_type: int | None = None,
) -> list[dict[str, Any]]:
    allowed = set(question_types or [])
    selected: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    for sample in samples:
        question_type = str(sample.get("question_type") or "")
        if allowed and question_type not in allowed:
            continue
        if limit_per_type is not None:
            count = type_counts.get(question_type, 0)
            if count >= limit_per_type:
                continue
            type_counts[question_type] = count + 1
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _select_longmemeval_v2_questions(
    questions: list[dict[str, Any]],
    *,
    limit: int | None = None,
    question_types: list[str] | None = None,
    question_ids: list[str] | None = None,
    limit_per_type: int | None = None,
) -> list[dict[str, Any]]:
    allowed = set(question_types or [])
    allowed_ids = set(question_ids or [])
    selected: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    for question in questions:
        question_id = _longmemeval_v2_question_id(question)
        if allowed_ids and question_id not in allowed_ids:
            continue
        question_type = str(question.get("question_type") or "")
        if allowed and question_type not in allowed:
            continue
        if limit_per_type is not None:
            count = type_counts.get(question_type, 0)
            if count >= limit_per_type:
                continue
            type_counts[question_type] = count + 1
        selected.append(question)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def load_question_ids(input_path: str | Path) -> list[str]:
    records = _load_json_records(input_path)
    ids: list[str] = []
    for index, record in enumerate(records, start=1):
        value = record.get("id") or record.get("question_id")
        if value is None:
            raise ValueError(f"Question-id record {index} needs id or question_id")
        ids.append(str(value))
    return ids


def convert_longmemeval_sample(
    sample: dict[str, Any],
    *,
    expected: str = "evidence",
    top_k: int = 8,
    infer_state_slots: bool = False,
    state_audit_label: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    question_id = _longmemeval_question_id(sample)
    observations: list[dict[str, Any]] = []
    for turn in _longmemeval_turns(sample):
        observations.append({
            "label": turn["label"],
            "content": turn["observation_content"],
            "kind": "dialogue",
            "importance": 0.5,
            "valid_from": turn["date"],
            "metadata": {
                "memory_key": turn["label"],
                "subject": turn["role"],
                "tags": ["longmemeval", f"session_{turn['session_index']}"],
            },
        })

    answer = sample.get("answer", "")
    answers = answer if isinstance(answer, list) else [answer]
    answer_strings = [str(item) for item in answers if str(item)]
    evidence = [str(item) for item in sample.get("answer_session_ids") or []]
    if expected == "evidence":
        expected_substrings = evidence
    elif expected == "answer":
        expected_substrings = answer_strings
    else:
        expected_substrings = evidence + answer_strings
    query_text = str(sample.get("question") or "")
    query_metadata = {
        "question_type": sample.get("question_type"),
        "answer": answer,
        "answer_session_ids": evidence,
        "question_date": sample.get("question_date"),
        "abstention": question_id.endswith("_abs"),
    }
    if state_audit_label:
        query_metadata["state_slot"] = state_audit_label["state_slot"]
        query_metadata["state_slot_source"] = state_audit_label.get("state_slot_source", "manual_state_audit")
        if "state_available" in state_audit_label:
            query_metadata["state_available"] = state_audit_label["state_available"]
        if state_audit_label.get("state_audit_id"):
            query_metadata["state_audit_id"] = state_audit_label["state_audit_id"]
    elif infer_state_slots:
        inferred_slots = query_relevant_state_slots(query_text)
        if inferred_slots:
            query_metadata["state_slot"] = inferred_slots if len(inferred_slots) > 1 else inferred_slots[0]
            query_metadata["state_slot_source"] = "query_text_router"

    queries = [{
        "id": question_id,
        "query": query_text,
        "expected_substrings": expected_substrings,
        "top_k": top_k,
        "now": sample.get("question_date"),
        "metadata": query_metadata,
    }]
    return {
        "id": question_id,
        "metadata": {
            "question_type": sample.get("question_type"),
            "question_date": sample.get("question_date"),
        },
        "observations": observations,
        "queries": queries,
    }


def convert_longmemeval_v2_sample(
    question: Mapping[str, Any],
    *,
    trajectories: Mapping[str, Mapping[str, Any]],
    haystack_trajectory_ids: list[str],
    expected: str = "answer",
    top_k: int = 8,
    infer_state_slots: bool = True,
) -> dict[str, Any]:
    question_id = _longmemeval_v2_question_id(question)
    observations: list[dict[str, Any]] = []
    missing_trajectory_ids: list[str] = []
    for trajectory_id in haystack_trajectory_ids:
        trajectory = trajectories.get(trajectory_id)
        if trajectory is None:
            missing_trajectory_ids.append(trajectory_id)
            continue
        observations.extend(_longmemeval_v2_observations(trajectory))

    answer = question.get("answer", "")
    answers = answer if isinstance(answer, list) else [answer]
    answer_strings = [str(item) for item in answers if str(item)]
    if expected == "evidence":
        expected_substrings = list(haystack_trajectory_ids)
    elif expected == "both":
        expected_substrings = list(haystack_trajectory_ids) + answer_strings
    else:
        expected_substrings = answer_strings

    query_text = str(question.get("question") or "")
    query_metadata = {
        "benchmark": "longmemeval_v2",
        "domain": question.get("domain"),
        "environment": question.get("environment"),
        "question_type": question.get("question_type"),
        "question_type_name": question.get("question_type"),
        "answer": answer,
        "eval_function": question.get("eval_function"),
        "image": question.get("image"),
        "haystack_trajectory_ids": list(haystack_trajectory_ids),
        "missing_trajectory_ids": missing_trajectory_ids,
        "abstention": str(question.get("question_type") or "").endswith("-abs"),
    }
    if infer_state_slots:
        inferred_slots = query_relevant_state_slots(query_text)
        if inferred_slots:
            query_metadata["state_slot"] = inferred_slots if len(inferred_slots) > 1 else inferred_slots[0]
            query_metadata["state_slot_source"] = "query_text_router"

    return {
        "id": question_id,
        "metadata": {
            "benchmark": "longmemeval_v2",
            "domain": question.get("domain"),
            "environment": question.get("environment"),
            "question_type": question.get("question_type"),
            "haystack_size": len(haystack_trajectory_ids),
            "missing_trajectory_count": len(missing_trajectory_ids),
        },
        "observations": observations,
        "queries": [{
            "id": question_id,
            "query": query_text,
            "expected_substrings": expected_substrings,
            "top_k": top_k,
            "metadata": query_metadata,
        }],
    }


def _longmemeval_v2_question_id(question: Mapping[str, Any]) -> str:
    return str(question.get("id") or question.get("question_id") or "longmemeval-v2-question")


def _longmemeval_v2_observations(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    trajectory_id = str(trajectory.get("id") or trajectory.get("trajectory_id") or "trajectory")
    states = trajectory.get("states") or []
    if isinstance(states, Mapping):
        states = states.get("states") or []
    observations: list[dict[str, Any]] = []
    for fallback_index, state in enumerate(states):
        if not isinstance(state, Mapping):
            continue
        state_index = _longmemeval_v2_state_index(state, fallback=fallback_index)
        step = state.get("step", state_index)
        label = f"{trajectory_id}.s{state_index:04d}"
        content = _longmemeval_v2_state_content(trajectory, state, label=label)
        if not content:
            continue
        observations.append({
            "label": label,
            "content": content,
            "kind": "trajectory_state",
            "importance": 0.55 if trajectory.get("outcome") == "success" else 0.5,
            "metadata": {
                "memory_key": label,
                "subject": "trajectory_state",
                "tags": [
                    "longmemeval_v2",
                    str(trajectory.get("domain") or "unknown_domain"),
                    str(trajectory.get("environment") or "unknown_environment"),
                    f"trajectory_{trajectory_id}",
                ],
                "benchmark": "longmemeval_v2",
                "trajectory_id": trajectory_id,
                "trajectory_step": step,
                "state_index": state_index,
                "domain": trajectory.get("domain"),
                "environment": trajectory.get("environment"),
                "outcome": trajectory.get("outcome"),
                "url": state.get("url"),
            },
        })
    return observations


def _longmemeval_v2_state_index(state: Mapping[str, Any], *, fallback: int) -> int:
    raw = state.get("state_index", fallback)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _longmemeval_v2_state_content(
    trajectory: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    label: str,
) -> str:
    parts = [f"[{label}]"]
    goal = _stringify(trajectory.get("goal"))
    if goal:
        parts.append(f"goal: {goal}")
    url = _stringify(state.get("url"))
    if url:
        parts.append(f"url: {url}")
    thought = _stringify(state.get("thought"))
    if thought:
        parts.append(f"thought: {thought}")
    action = _stringify(state.get("action"))
    if action:
        parts.append(f"action: {action}")
    accessibility_tree = _stringify(state.get("accessibility_tree"))
    if accessibility_tree:
        parts.append(f"observation: {accessibility_tree}")
    return " | ".join(parts).strip()


def write_longmemeval_state_audit_file(samples: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    records = list(longmemeval_state_audit_records(samples))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(records)


def write_longmemeval_state_audit_summary_file(samples: Iterable[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    summary = summarize_longmemeval_state_audit_records(longmemeval_state_audit_records(samples))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def summarize_longmemeval_state_audit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    with_evidence = 0
    evidence_candidate_total = 0
    by_slot: dict[str, dict[str, int]] = {}
    by_question_type: dict[str, dict[str, int]] = {}
    for record in records:
        total += 1
        candidates = record.get("state_evidence_candidates") or []
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        evidence_candidate_total += candidate_count
        has_evidence = candidate_count > 0
        if has_evidence:
            with_evidence += 1
        slots = record.get("inferred_state_slots") or [record.get("state_slot")]
        for slot in slots:
            _increment_audit_summary(by_slot, str(slot), has_evidence=has_evidence, candidate_count=candidate_count)
        question_type = str(record.get("question_type") or "<missing>")
        _increment_audit_summary(
            by_question_type,
            question_type,
            has_evidence=has_evidence,
            candidate_count=candidate_count,
        )
    return {
        "total_candidates": total,
        "with_state_evidence": with_evidence,
        "without_state_evidence": total - with_evidence,
        "state_evidence_candidate_total": evidence_candidate_total,
        "by_state_slot": by_slot,
        "by_question_type": by_question_type,
    }


def _increment_audit_summary(
    bucket: dict[str, dict[str, int]],
    key: str,
    *,
    has_evidence: bool,
    candidate_count: int,
) -> None:
    aggregate = bucket.setdefault(
        key,
        {
            "total_candidates": 0,
            "with_state_evidence": 0,
            "without_state_evidence": 0,
            "state_evidence_candidate_total": 0,
        },
    )
    aggregate["total_candidates"] += 1
    aggregate["state_evidence_candidate_total"] += candidate_count
    if has_evidence:
        aggregate["with_state_evidence"] += 1
    else:
        aggregate["without_state_evidence"] += 1


def longmemeval_state_audit_records(samples: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for sample in samples:
        question = str(sample.get("question") or "")
        inferred_slots = query_relevant_state_slots(question)
        if not inferred_slots:
            continue
        yield {
            "question_id": _longmemeval_question_id(sample),
            "question_type": sample.get("question_type"),
            "question_date": sample.get("question_date"),
            "question": question,
            "inferred_state_slots": inferred_slots,
            "is_state_sensitive": None,
            "state_available": None,
            "state_slot": inferred_slots if len(inferred_slots) > 1 else inferred_slots[0],
            "state_evidence_candidates": _longmemeval_state_evidence_candidates(sample, inferred_slots),
            "notes": "",
        }


def load_state_audit_labels(input_path: str | Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("is_state_sensitive"):
                continue
            question_id = str(record.get("question_id") or record.get("id") or "")
            state_slot = record.get("state_slot")
            if not question_id or not state_slot:
                raise ValueError(f"State audit record {line_number} needs question_id and state_slot")
            labels[question_id] = {
                "state_slot": state_slot,
                "state_available": _audit_bool(record.get("state_available", True)),
                "state_slot_source": record.get("state_slot_source") or "manual_state_audit",
                "state_audit_id": record.get("state_audit_id") or record.get("audit_id"),
            }
    return labels


def _audit_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "unavailable"}
    return bool(value)


def _longmemeval_question_id(sample: Mapping[str, Any]) -> str:
    return str(sample.get("question_id") or sample.get("id") or "longmemeval-sample")


def _longmemeval_turns(sample: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    session_ids = [str(item) for item in sample.get("haystack_session_ids") or []]
    dates = [str(item) if item is not None else None for item in sample.get("haystack_dates") or []]
    sessions = sample.get("haystack_sessions") or []
    for session_index, session in enumerate(sessions):
        session_id = session_ids[session_index] if session_index < len(session_ids) else f"s{session_index}"
        date = dates[session_index] if session_index < len(dates) else None
        for turn_index, turn in enumerate(session or []):
            role = str(turn.get("role") or "unknown")
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            label = f"{session_id}.t{turn_index}"
            yield {
                "label": label,
                "session_index": session_index,
                "session_id": session_id,
                "turn_index": turn_index,
                "date": date,
                "role": role,
                "content": content,
                "observation_content": f"[{session_id} {date}] {role}: {content}",
            }


def _longmemeval_state_evidence_candidates(
    sample: Mapping[str, Any],
    inferred_slots: list[str],
) -> list[dict[str, Any]]:
    relevant_slots = set(inferred_slots)
    candidates: list[dict[str, Any]] = []
    for turn in _longmemeval_turns(sample):
        patches = extract_state_patches(str(turn["observation_content"]))
        for patch in patches:
            if not state_slot_matches_query(patch.slot, relevant_slots):
                continue
            candidates.append({
                "label": turn["label"],
                "date": turn["date"],
                "role": turn["role"],
                "state_slot": patch.slot,
                "state_value": patch.value,
                "evidence": patch.evidence,
                "source": "deterministic_state_extractor",
            })
    return candidates


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
                **_stale_evaluation_state_metadata(sample, text),
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


_STALE_LOCATION_EVIDENCE_TERMS = (
    "based",
    "city",
    "home base",
    "live",
    "lives",
    "living",
    "located",
    "location",
    "moved",
    "relocated",
    "settled",
    "settling",
    "staying",
)

_STALE_LOCAL_CONTEXT_QUERY_TERMS = (
    "area",
    "around me",
    "coffee",
    "commute",
    "gym",
    "local",
    "near me",
    "nearby",
    "neighborhood",
    "park",
    "places",
    "recommend",
    "restaurant",
    "suggest",
    "timezone",
)

_STALE_EMPLOYER_EVIDENCE_TERMS = (
    "company",
    "employer",
    "job",
    "manager",
    "office",
    "work",
    "workplace",
)

_STALE_EMPLOYMENT_CONTEXT_QUERY_TERMS = (
    "benefit",
    "benefits",
    "hr",
    "manager",
    "office",
    "payroll",
    "portal",
    "work",
    "workplace",
)

_STALE_DIETARY_EVIDENCE_TERMS = (
    "allergic",
    "allergy",
    "dairy",
    "diet",
    "dietary",
    "gluten",
    "nut",
    "peanut",
    "shellfish",
)

_STALE_FOOD_CONTEXT_QUERY_TERMS = (
    "eat",
    "food",
    "meal",
    "menu",
    "order",
    "restaurant",
    "safe",
)


def _stale_evaluation_state_metadata(
    sample: Mapping[str, Any],
    query_text: str,
) -> dict[str, str]:
    """Infer query-only opportunity labels from STALE metadata.

    STALE ground truth is allowed in conversion and diagnostics, but these
    labels must not be copied to observations because observations are runtime
    memory input.
    """

    evidence_text = " ".join(
        str(sample.get(key) or "")
        for key in ("M_old", "M_new", "explanation")
    )
    evidence_lower = evidence_text.lower()
    query_lower = query_text.lower()

    if _contains_any_phrase(evidence_lower, _STALE_LOCATION_EVIDENCE_TERMS):
        metadata = {
            "state_slot": "location",
            "state_slot_source": "stale_metadata_heuristic",
        }
        if _contains_any_phrase(query_lower, _STALE_LOCAL_CONTEXT_QUERY_TERMS):
            metadata.update({
                "dependency_source_slot": "location",
                "dependency_source_slot_source": "stale_metadata_heuristic",
                "dependency_target_family": "local_context",
            })
        return metadata

    if _contains_any_phrase(evidence_lower, _STALE_EMPLOYER_EVIDENCE_TERMS):
        metadata = {
            "state_slot": "organization.employer",
            "state_slot_source": "stale_metadata_heuristic",
        }
        if _contains_any_phrase(query_lower, _STALE_EMPLOYMENT_CONTEXT_QUERY_TERMS):
            metadata.update({
                "dependency_source_slot": "organization.employer",
                "dependency_source_slot_source": "stale_metadata_heuristic",
                "dependency_target_family": "employment_context",
            })
        return metadata

    if _contains_any_phrase(evidence_lower, _STALE_DIETARY_EVIDENCE_TERMS):
        metadata = {
            "state_slot": "health.*.status",
            "state_slot_source": "stale_metadata_heuristic",
        }
        if _contains_any_phrase(query_lower, _STALE_FOOD_CONTEXT_QUERY_TERMS):
            metadata.update({
                "dependency_source_slot": "health.*.status",
                "dependency_source_slot_source": "stale_metadata_heuristic",
                "dependency_target_family": "food_safety_context",
            })
        return metadata

    return {}


def _contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
        for phrase in phrases
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert public memory benchmarks to AdaMem JSONL.")
    sub = parser.add_subparsers(dest="command", required=True)

    locomo = sub.add_parser("locomo", help="Convert LoCoMo locomo10.json to AdaMem JSONL")
    locomo.add_argument("input", type=Path)
    locomo.add_argument("output", type=Path)
    locomo.add_argument("--expected", choices=["evidence", "answer", "both"], default="evidence")
    locomo.add_argument("--top-k", type=int, default=8)
    locomo.add_argument("--limit", type=int)

    longmemeval = sub.add_parser("longmemeval", help="Convert LongMemEval JSON to AdaMem JSONL")
    longmemeval.add_argument("input", type=Path)
    longmemeval.add_argument("output", type=Path)
    longmemeval.add_argument("--expected", choices=["evidence", "answer", "both"], default="evidence")
    longmemeval.add_argument("--top-k", type=int, default=8)
    longmemeval.add_argument("--limit", type=int)
    longmemeval.add_argument("--question-types", nargs="+", help="Filter LongMemEval question_type values")
    longmemeval.add_argument("--limit-per-type", type=int, help="Keep at most this many LongMemEval items per question_type")
    longmemeval.add_argument(
        "--infer-state-slots",
        action="store_true",
        help="Annotate LongMemEval query metadata with state slots inferred from query text for diagnostics only",
    )
    longmemeval.add_argument(
        "--state-audit-output",
        type=Path,
        help="Write query-state candidate JSONL for manual precision audit",
    )
    longmemeval.add_argument(
        "--state-audit-summary-output",
        type=Path,
        help="Write JSON summary for query-state audit candidates",
    )
    longmemeval.add_argument(
        "--state-audit-input",
        type=Path,
        help="Apply manually reviewed query-state JSONL labels to query metadata",
    )

    longmemeval_v2 = sub.add_parser(
        "longmemeval-v2",
        help="Convert LongMemEval-V2 questions, trajectories, and haystacks to AdaMem JSONL",
    )
    longmemeval_v2.add_argument("questions", type=Path)
    longmemeval_v2.add_argument("trajectories", type=Path)
    longmemeval_v2.add_argument("haystack", type=Path)
    longmemeval_v2.add_argument("output", type=Path)
    longmemeval_v2.add_argument("--expected", choices=["answer", "evidence", "both"], default="answer")
    longmemeval_v2.add_argument("--top-k", type=int, default=8)
    longmemeval_v2.add_argument("--limit", type=int)
    longmemeval_v2.add_argument("--question-types", nargs="+", help="Filter LongMemEval-V2 question_type values")
    longmemeval_v2.add_argument(
        "--question-ids-file",
        type=Path,
        help="Filter LongMemEval-V2 questions by JSON/JSONL records containing id or question_id",
    )
    longmemeval_v2.add_argument(
        "--limit-per-type",
        type=int,
        help="Keep at most this many LongMemEval-V2 items per question_type",
    )
    longmemeval_v2.add_argument(
        "--max-trajectories-per-question",
        type=int,
        help="Cap haystack trajectories per question for local smoke conversion",
    )
    longmemeval_v2.add_argument(
        "--no-infer-state-slots",
        action="store_true",
        help="Disable query-text state-slot annotations used for diagnostics",
    )

    stale = sub.add_parser("stale", help="Convert STALE T1_T2_400_FULL.json to AdaMem JSONL")
    stale.add_argument("input", type=Path)
    stale.add_argument("output", type=Path)
    stale.add_argument("--top-k", type=int, default=8)
    stale.add_argument("--limit", type=int)
    stale.add_argument("--types", nargs="+", choices=["T1", "T2"], help="Filter by conflict type")

    stale_annotate = sub.add_parser(
        "stale-annotate",
        help="Backfill STALE query-only state/dependency labels into converted AdaMem JSONL",
    )
    stale_annotate.add_argument("input", type=Path)
    stale_annotate.add_argument("output", type=Path)

    ama = sub.add_parser("ama", help="Convert AMA-Bench-style agent trajectory JSON/JSONL to AdaMem JSONL")
    ama.add_argument("input", type=Path)
    ama.add_argument("output", type=Path)
    ama.add_argument("--expected", choices=["answer", "evidence", "both"], default="answer")
    ama.add_argument("--top-k", type=int, default=8)
    ama.add_argument("--limit", type=int)

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
    elif args.command == "longmemeval":
        count = convert_longmemeval_file(
            args.input,
            args.output,
            expected=args.expected,
            top_k=args.top_k,
            limit=args.limit,
            question_types=args.question_types,
            limit_per_type=args.limit_per_type,
            infer_state_slots=args.infer_state_slots,
            state_audit_input=args.state_audit_input,
            state_audit_output=args.state_audit_output,
            state_audit_summary_output=args.state_audit_summary_output,
        )
        print(f"wrote {count} LongMemEval cases to {args.output}")
    elif args.command == "longmemeval-v2":
        count = convert_longmemeval_v2_file(
            args.questions,
            args.trajectories,
            args.haystack,
            args.output,
            expected=args.expected,
            top_k=args.top_k,
            limit=args.limit,
            question_types=args.question_types,
            question_ids=load_question_ids(args.question_ids_file) if args.question_ids_file else None,
            limit_per_type=args.limit_per_type,
            max_trajectories_per_question=args.max_trajectories_per_question,
            infer_state_slots=not args.no_infer_state_slots,
        )
        print(f"wrote {count} LongMemEval-V2 cases to {args.output}")
    elif args.command == "stale":
        count = convert_stale_file(
            args.input,
            args.output,
            top_k=args.top_k,
            limit=args.limit,
            types=args.types,
        )
        print(f"wrote {count} STALE cases to {args.output}")
    elif args.command == "stale-annotate":
        count = annotate_stale_jsonl_file(args.input, args.output)
        print(f"wrote {count} annotated STALE cases to {args.output}")
    elif args.command == "ama":
        count = convert_ama_file(
            args.input,
            args.output,
            expected=args.expected,
            top_k=args.top_k,
            limit=args.limit,
        )
        print(f"wrote {count} AMA-style cases to {args.output}")


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
