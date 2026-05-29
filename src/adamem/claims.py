from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def audit_experiment(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise ValueError(f"{experiment_path} must contain a JSON experiment object")

    run_type = str(experiment.get("run_type") or "")
    notes = experiment.get("notes") or {}
    if not isinstance(notes, dict):
        notes = {}
    providers = _providers(notes)
    baselines = [str(name) for name in experiment.get("baseline_names") or []]
    raw_outputs = experiment.get("raw_outputs") or []
    results = experiment.get("results")
    raw_output_count = _evidence_record_count(experiment_path, raw_outputs, results, notes)

    supported: list[str] = []
    blocked: dict[str, list[str]] = {
        "answer_accuracy": [],
        "stale_answer_accuracy": [],
        "retrieval_mechanism": [],
        "sota": [],
    }
    warnings: list[str] = []

    runtime_use = notes.get("ground_truth_runtime_use")
    if runtime_use != "forbidden":
        warnings.append("ground_truth_runtime_use is not explicitly forbidden")

    if run_type in {
        "jsonl_retrieval_benchmark",
        "ama_public_answerability_pilot",
        "ama_public_evidence_pilot",
    }:
        supported.append("retrieval_diagnostics")
        supported.append("answerability_diagnostics")
        blocked["answer_accuracy"].append("run_type is retrieval/answerability, not answer generation")
        blocked["sota"].append("no final answer model and judge model evaluation")
    elif run_type in {"jsonl_answer_generation_benchmark", "ama_public_answer_generation_pilot"}:
        if _uses_mock_provider(providers):
            supported.append("harness_plumbing")
            blocked["answer_accuracy"].append("mock answer or judge provider")
        elif notes.get("scorer") == "substring":
            supported.append("exact_string_answer_smoke")
            blocked["answer_accuracy"].append("substring scorer is not a semantic judge")
        else:
            supported.append("answer_accuracy_candidate")
        blocked["sota"].append("no official strong-baseline reproduction evidence")
    elif run_type == "stale_llm_judge":
        if _uses_mock_provider(providers):
            supported.append("stale_judge_plumbing")
            blocked["stale_answer_accuracy"].append("mock answer or judge provider")
        else:
            supported.append("stale_answer_accuracy_candidate")
        if raw_output_count == 0:
            blocked["stale_answer_accuracy"].append("no raw per-query judge outputs")
        blocked["sota"].append("no multi-model judge robustness or strong-baseline reproduction evidence")
    elif run_type == "stale_retrieval_diagnostics":
        supported.append("stale_retrieval_diagnostics")
        supported.append("mechanism_error_analysis")
        blocked["stale_answer_accuracy"].append("no answer generation or LLM judge")
        blocked["sota"].append("retrieval diagnostics cannot establish SOTA")
    else:
        supported.append("unclassified_experiment")
        blocked["answer_accuracy"].append(f"unrecognized run_type: {run_type or '<missing>'}")
        blocked["sota"].append("unclassified experiment cannot support SOTA")

    if len(baselines) < 2:
        blocked["sota"].append("fewer than two baselines in experiment record")
    if not experiment.get("commit"):
        warnings.append("experiment commit is missing")
    if not experiment.get("dataset"):
        warnings.append("experiment dataset is missing")

    return {
        "experiment": str(experiment_path),
        "run_name": experiment.get("run_name"),
        "run_type": run_type,
        "dataset": experiment.get("dataset"),
        "baselines": baselines,
        "providers": providers,
        "supported_claims": supported,
        "blocked_claims": {key: value for key, value in blocked.items() if value},
        "warnings": warnings,
        "raw_output_count": raw_output_count,
        "notes": {
            "ground_truth_runtime_use": runtime_use,
            "ground_truth_evaluation_use": notes.get("ground_truth_evaluation_use"),
            "answer_model_required": notes.get("answer_model_required"),
            "judge_model_required": notes.get("judge_model_required"),
        },
    }


def claim_audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# AdaMem Claim Audit", ""]
    lines.append(f"Experiment: `{audit['experiment']}`")
    lines.append(f"Run type: `{audit['run_type'] or '<missing>'}`")
    lines.append(f"Dataset: `{audit.get('dataset') or '<missing>'}`")
    lines.append(f"Raw outputs: `{audit['raw_output_count']}`")
    lines.append("")

    lines.append("## Supported Claims")
    for claim in audit["supported_claims"]:
        lines.append(f"- `{claim}`")
    lines.append("")

    lines.append("## Blocked Claims")
    blocked = audit["blocked_claims"]
    if not blocked:
        lines.append("- None.")
    for claim, reasons in blocked.items():
        lines.append(f"- `{claim}`: {'; '.join(reasons)}")
    lines.append("")

    lines.append("## Warnings")
    warnings = audit["warnings"]
    if not warnings:
        lines.append("- None.")
    for warning in warnings:
        lines.append(f"- {warning}")
    return "\n".join(lines).rstrip() + "\n"


def _providers(notes: dict[str, Any]) -> dict[str, str | None]:
    return {
        "answer_provider": _optional_str(notes.get("answer_provider")),
        "answer_model": _optional_str(notes.get("answer_model")),
        "judge_provider": _optional_str(notes.get("judge_provider")),
        "judge_model": _optional_str(notes.get("judge_model")),
        "scorer": _optional_str(notes.get("scorer") or notes.get("answer_scorer")),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _uses_mock_provider(providers: dict[str, str | None]) -> bool:
    return any(value == "mock" for key, value in providers.items() if key.endswith("_provider"))


def _evidence_record_count(
    experiment_path: Path,
    raw_outputs: Any,
    results: Any,
    notes: dict[str, Any],
) -> int:
    if isinstance(raw_outputs, list) and raw_outputs:
        return len(raw_outputs)
    records_path = notes.get("records_path")
    if records_path:
        path = _resolve_records_path(experiment_path, str(records_path))
        if path is not None:
            return _count_jsonl_records(path)
    if isinstance(results, list):
        total = 0
        for result in results:
            if isinstance(result, dict):
                if "queries" in result and isinstance(result["queries"], list):
                    total += len(result["queries"])
                elif "n_total" in result:
                    total += int(result["n_total"] or 0)
        return total
    if isinstance(results, dict):
        total = 0
        for value in results.values():
            if isinstance(value, dict):
                total += int(value.get("total") or value.get("n_total") or 0)
        return total
    return 0


def _resolve_records_path(experiment_path: Path, records_path: str) -> Path | None:
    candidate = Path(records_path)
    if candidate.exists():
        return candidate
    relative = experiment_path.parent / candidate
    if relative.exists():
        return relative
    return None


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit what paper claims an AdaMem experiment can support."
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    audit = audit_experiment(args.experiment)
    text = (
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
        if args.json
        else claim_audit_markdown(audit)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
