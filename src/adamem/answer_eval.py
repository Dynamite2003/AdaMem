from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from adamem.baselines import select_baselines
from adamem.bench import MemoryQACase, load_jsonl_cases
from adamem.config import AdaMemConfig
from adamem.experiments import experiment_record, write_experiment_record
from adamem.llm import LLMClient, build_client
from adamem.manager import AdaMem
from adamem.schema import MemoryItem


DEFAULT_GROUP_FIELDS = ("question_type", "dimension", "state_slot", "abstention")

ANSWER_SYSTEM = (
    "You are a careful assistant grounded only in the provided memory excerpts. "
    "Use the excerpts to answer the user question. If the excerpts are "
    "insufficient, say that the memory does not provide enough information."
)

ANSWER_TEMPLATE = """\
Memory excerpts:
{context}

User question: {query}

Answer concisely using only the memory excerpts above.
"""

ANSWER_JUDGE_SYSTEM = (
    "You are a strict evaluator. Decide whether the candidate answer is "
    "supported by the reference answer/support strings and avoids forbidden "
    "claims. Output exactly one token: CORRECT or INCORRECT."
)

ANSWER_JUDGE_TEMPLATE = """\
Question:
{query}

Reference answer/support strings:
{expected}

Forbidden strings:
{forbidden}

Candidate answer:
{answer}

Reply with exactly one token: CORRECT or INCORRECT.
"""


@dataclass(slots=True)
class AnswerScore:
    correct: bool
    raw: str
    prompt: str = ""
    system: str = ""


class AnswerScorer(Protocol):
    name: str

    def score(
        self,
        *,
        query: str,
        answer: str,
        expected_substrings: list[str],
        forbidden_substrings: list[str],
        metadata: dict[str, Any],
    ) -> AnswerScore:
        ...


@dataclass(slots=True)
class SubstringAnswerScorer:
    """Deterministic scorer for API-free smoke tests.

    This is not a semantic judge. It is useful for verifying plumbing and for
    tasks where exact support strings are intentionally short.
    """

    name: str = "substring"

    def score(
        self,
        *,
        query: str,
        answer: str,
        expected_substrings: list[str],
        forbidden_substrings: list[str],
        metadata: dict[str, Any],
    ) -> AnswerScore:
        del query, metadata
        answer_text = answer.lower()
        has_expected = all(item.lower() in answer_text for item in expected_substrings)
        has_forbidden = any(item.lower() in answer_text for item in forbidden_substrings)
        correct = has_expected and not has_forbidden
        return AnswerScore(
            correct=correct,
            raw="CORRECT" if correct else "INCORRECT",
            system="deterministic substring scorer",
        )


@dataclass(slots=True)
class LLMAnswerScorer:
    client: LLMClient
    name: str = "llm_judge"
    system: str = ANSWER_JUDGE_SYSTEM
    template: str = ANSWER_JUDGE_TEMPLATE

    def score(
        self,
        *,
        query: str,
        answer: str,
        expected_substrings: list[str],
        forbidden_substrings: list[str],
        metadata: dict[str, Any],
    ) -> AnswerScore:
        del metadata
        prompt = self.template.format(
            query=query,
            expected=_bullet_list(expected_substrings),
            forbidden=_bullet_list(forbidden_substrings),
            answer=answer,
        )
        raw = self.client.complete(prompt, system=self.system, max_tokens=8, temperature=0.0)
        return AnswerScore(
            correct=_parse_correct(raw),
            raw=raw,
            prompt=prompt,
            system=self.system,
        )


@dataclass(slots=True)
class AnswerQueryResult:
    case_id: str
    query_id: str
    query: str
    correct: bool
    answer: str
    score_raw: str
    retrieved: list[str]
    trace: list[dict[str, Any]]
    expected_substrings: list[str]
    forbidden_substrings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerBenchmarkResult:
    name: str
    accuracy: float
    n_correct: int
    n_total: int
    queries: list[AnswerQueryResult]


def run_answer_benchmark(
    cases: list[MemoryQACase],
    *,
    answer_client: LLMClient,
    scorer: AnswerScorer,
    configs: dict[str, AdaMemConfig],
    top_k: int = 8,
    max_context_chars: int = 4000,
    raw_outputs: list[dict[str, Any]] | None = None,
) -> list[AnswerBenchmarkResult]:
    results: list[AnswerBenchmarkResult] = []
    for name, config in configs.items():
        query_results: list[AnswerQueryResult] = []
        for case in cases:
            query_results.extend(
                _run_case(
                    case,
                    baseline=name,
                    config=config,
                    answer_client=answer_client,
                    scorer=scorer,
                    top_k=top_k,
                    max_context_chars=max_context_chars,
                    raw_outputs=raw_outputs,
                )
            )
        n_correct = sum(1 for result in query_results if result.correct)
        n_total = len(query_results)
        results.append(
            AnswerBenchmarkResult(
                name=name,
                accuracy=n_correct / n_total if n_total else 0.0,
                n_correct=n_correct,
                n_total=n_total,
                queries=query_results,
            )
        )
    return results


def answer_case_records(results: list[AnswerBenchmarkResult]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        for query in result.queries:
            records.append({
                "baseline": result.name,
                "case_id": query.case_id,
                "query_id": query.query_id,
                "query": query.query,
                "correct": query.correct,
                "answer": query.answer,
                "score_raw": query.score_raw,
                "retrieved": query.retrieved,
                "trace": query.trace,
                "expected_substrings": query.expected_substrings,
                "forbidden_substrings": query.forbidden_substrings,
                "metadata": query.metadata,
            })
    return records


def answer_failure_summary(
    records: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS,
) -> dict[str, Any]:
    baseline_order = list(dict.fromkeys(str(record["baseline"]) for record in records))
    summary: dict[str, Any] = {
        "total_records": len(records),
        "by_baseline": {},
        "by_metadata": {},
    }
    for baseline in baseline_order:
        subset = [record for record in records if record["baseline"] == baseline]
        summary["by_baseline"][baseline] = _aggregate_answer_records(subset)
    for field_name in group_fields:
        values = sorted({_metadata_group_value(record["metadata"], field_name) for record in records})
        if values == ["<missing>"]:
            continue
        field_summary: dict[str, Any] = {}
        for value in values:
            value_subset = [
                record
                for record in records
                if _metadata_group_value(record["metadata"], field_name) == value
            ]
            field_summary[value] = {
                baseline: _aggregate_answer_records([
                    record for record in value_subset if record["baseline"] == baseline
                ])
                for baseline in baseline_order
                if any(record["baseline"] == baseline for record in value_subset)
            }
        summary["by_metadata"][field_name] = field_summary
    return summary


def answer_report(
    results_or_records: list[AnswerBenchmarkResult] | list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS,
) -> str:
    if results_or_records and isinstance(results_or_records[0], AnswerBenchmarkResult):
        records = answer_case_records(results_or_records)  # type: ignore[arg-type]
    else:
        records = list(results_or_records)  # type: ignore[assignment]
    summary = answer_failure_summary(records, group_fields=group_fields)
    lines = ["# AdaMem Answer Evaluation", ""]
    lines.append("| baseline | correct | accuracy |")
    lines.append("| --- | ---: | ---: |")
    for baseline, aggregate in summary["by_baseline"].items():
        lines.append(
            f"| {baseline} | {aggregate['correct']}/{aggregate['total']} | "
            f"{aggregate['accuracy']:.2%} |"
        )
    lines.append("")
    for field_name, field_summary in summary["by_metadata"].items():
        lines.append(f"## By {field_name}")
        lines.append("| value | baseline | correct | accuracy |")
        lines.append("| --- | --- | ---: | ---: |")
        for value, by_baseline in field_summary.items():
            for baseline, aggregate in by_baseline.items():
                lines.append(
                    f"| {value} | {baseline} | {aggregate['correct']}/{aggregate['total']} | "
                    f"{aggregate['accuracy']:.2%} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _aggregate_answer_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def _metadata_group_value(metadata: dict[str, Any], field_name: str) -> str:
    value = metadata.get(field_name)
    if value is None or value == "":
        return "<missing>"
    return str(value)


def _run_case(
    case: MemoryQACase,
    *,
    baseline: str,
    config: AdaMemConfig,
    answer_client: LLMClient,
    scorer: AnswerScorer,
    top_k: int,
    max_context_chars: int,
    raw_outputs: list[dict[str, Any]] | None,
) -> list[AnswerQueryResult]:
    mem = AdaMem(config=config)
    labels: dict[str, MemoryItem] = {}
    for observation in case.observations:
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
        if observation.label is not None:
            labels[observation.label] = item

    results: list[AnswerQueryResult] = []
    for query in case.queries:
        retrieved_results = mem.retrieve(query.query, top_k=query.top_k or top_k, now=query.now)
        retrieved = [result.item.content for result in retrieved_results]
        context = _truncate("\n---\n".join(retrieved), max_context_chars)
        answer_prompt = ANSWER_TEMPLATE.format(context=context, query=query.query)
        answer = answer_client.complete(
            answer_prompt,
            system=ANSWER_SYSTEM,
            max_tokens=256,
            temperature=0.0,
        )
        score = scorer.score(
            query=query.query,
            answer=answer,
            expected_substrings=query.expected_substrings,
            forbidden_substrings=query.forbidden_substrings,
            metadata=query.metadata,
        )
        trace = [
            {
                "rank": index,
                "content": result.item.content,
                "score": round(result.score, 4),
                "relation": result.relation,
                "contributions": {
                    key: round(value, 4)
                    for key, value in result.contributions.items()
                },
            }
            for index, result in enumerate(retrieved_results, start=1)
        ]
        record = {
            "baseline": baseline,
            "case_id": case.id,
            "query_id": query.id or "",
            "query": query.query,
            "answer_prompt": answer_prompt,
            "answer_system": ANSWER_SYSTEM,
            "answer_raw": answer,
            "scorer": scorer.name,
            "score_raw": score.raw,
            "score_correct": score.correct,
            "score_prompt": score.prompt,
            "score_system": score.system,
            "retrieved": trace,
            "metadata": query.metadata,
        }
        if raw_outputs is not None:
            raw_outputs.append(record)
        results.append(
            AnswerQueryResult(
                case_id=case.id,
                query_id=query.id or "",
                query=query.query,
                correct=score.correct,
                answer=answer,
                score_raw=score.raw,
                retrieved=retrieved,
                trace=trace,
                expected_substrings=query.expected_substrings,
                forbidden_substrings=query.forbidden_substrings,
                metadata=query.metadata,
            )
        )
    return results


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]"


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- <none>"
    return "\n".join(f"- {item}" for item in items)


def _parse_correct(raw: str) -> bool:
    token = (raw or "").strip().upper()
    if token.startswith("CORRECT"):
        return True
    if token.startswith("INCORRECT"):
        return False
    return "CORRECT" in token and "INCORRECT" not in token


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run answer-generation evaluation over AdaMem JSONL cases."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baselines", nargs="+", default=["semantic_only"])
    parser.add_argument("--answer-provider", default="openai")
    parser.add_argument("--answer-model", default="gpt-4o-mini")
    parser.add_argument("--mock-answer", default="The memory does not provide enough information.")
    parser.add_argument("--scorer", choices=["substring", "llm"], default="substring")
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument("--judge-model", default="gemini-1.5-flash")
    parser.add_argument("--mock-judge", default="INCORRECT")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--experiment-output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    specs = select_baselines(args.baselines)
    cases = load_jsonl_cases(args.dataset)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    answer_client = _build_cli_client(
        args.answer_provider,
        model=args.answer_model,
        mock_response=args.mock_answer,
    )
    if args.scorer == "substring":
        scorer: AnswerScorer = SubstringAnswerScorer()
    else:
        judge_client = _build_cli_client(
            args.judge_provider,
            model=args.judge_model,
            mock_response=args.mock_judge,
        )
        scorer = LLMAnswerScorer(judge_client)

    raw_outputs: list[dict[str, Any]] = []
    results = run_answer_benchmark(
        cases,
        answer_client=answer_client,
        scorer=scorer,
        configs={name: spec.config for name, spec in specs.items()},
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        raw_outputs=raw_outputs,
    )
    records = answer_case_records(results)
    summary = answer_failure_summary(records)
    if args.records_output:
        _write_jsonl(args.records_output, records)
    if args.experiment_output:
        record = experiment_record(
            run_name=args.experiment_output.stem,
            run_type="jsonl_answer_generation_benchmark",
            dataset=args.dataset,
            split_or_case_limit=str(args.max_cases) if args.max_cases is not None else None,
            baselines=specs,
            results=summary["by_baseline"],
            diagnostics={"failure_summary": summary},
            prompts={
                "answer_system": ANSWER_SYSTEM,
                "answer_template": ANSWER_TEMPLATE,
                "judge_system": ANSWER_JUDGE_SYSTEM if args.scorer == "llm" else "",
                "judge_template": ANSWER_JUDGE_TEMPLATE if args.scorer == "llm" else "",
            },
            raw_outputs=raw_outputs,
            notes={
                "answer_provider": args.answer_provider,
                "answer_model": args.answer_model,
                "scorer": scorer.name,
                "judge_provider": args.judge_provider if args.scorer == "llm" else None,
                "judge_model": args.judge_model if args.scorer == "llm" else None,
                "top_k": args.top_k,
                "max_context_chars": args.max_context_chars,
                "answer_model_required": args.answer_provider != "mock",
                "judge_model_required": args.scorer == "llm" and args.judge_provider != "mock",
                "ground_truth_runtime_use": "forbidden",
                "ground_truth_evaluation_use": "answer_scorer_only",
            },
        )
        write_experiment_record(args.experiment_output, record)

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print(answer_report(records))


def _build_cli_client(provider: str, *, model: str, mock_response: str) -> LLMClient:
    if provider == "mock":
        return build_client(provider, model=model, responses=mock_response)
    return build_client(provider, model=model)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp.replace(path)
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
