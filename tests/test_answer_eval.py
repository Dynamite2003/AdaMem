from __future__ import annotations

import json
from pathlib import Path

from adamem.answer_eval import (
    LLMAnswerScorer,
    SubstringAnswerScorer,
    answer_case_records,
    answer_failure_summary,
    answer_report,
    main,
    run_answer_benchmark,
)
from adamem.baselines import default_ablation_configs
from adamem.bench import load_jsonl_cases
from adamem.llm import MockLLMClient


def test_run_answer_benchmark_with_substring_scorer(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    cases = load_jsonl_cases(dataset)
    answer_client = MockLLMClient(["The current city is Boston.", "The current city is Boston."])
    raw_outputs: list[dict] = []

    results = run_answer_benchmark(
        cases,
        answer_client=answer_client,
        scorer=SubstringAnswerScorer(),
        configs={"semantic_only": default_ablation_configs()["semantic_only"]},
        top_k=2,
        raw_outputs=raw_outputs,
    )

    assert len(results) == 1
    result = results[0]
    assert result.n_total == 2
    assert result.n_correct == 1
    assert result.accuracy == 0.5
    assert len(answer_client.calls) == 2
    assert "Memory excerpts" in answer_client.calls[0]["prompt"]
    assert raw_outputs[0]["answer_raw"] == "The current city is Boston."
    assert raw_outputs[0]["score_correct"] is True
    records = answer_case_records(results)
    assert records[0]["correct"] is True
    assert records[1]["correct"] is False
    records[0]["metadata"] = {"question_type": "A"}
    records[1]["metadata"] = {"question_type": "B"}
    summary = answer_failure_summary(records, group_fields=("question_type",))
    assert summary["by_baseline"]["semantic_only"]["correct"] == 1
    assert summary["by_metadata"]["question_type"]["A"]["semantic_only"]["accuracy"] == 1.0
    assert summary["by_metadata"]["question_type"]["B"]["semantic_only"]["accuracy"] == 0.0
    assert "| semantic_only | 1/2 | 50.00% |" in answer_report(results)
    grouped_report = answer_report(records, group_fields=("question_type",))
    assert "## By question_type" in grouped_report
    assert "| A | semantic_only | 1/1 | 100.00% |" in grouped_report


def test_llm_answer_scorer_records_judge_prompt() -> None:
    judge = MockLLMClient("CORRECT")
    scorer = LLMAnswerScorer(judge)

    score = scorer.score(
        query="Where does the user live?",
        answer="The user lives in Boston.",
        expected_substrings=["Boston"],
        forbidden_substrings=["Seattle"],
        metadata={},
    )

    assert score.correct is True
    assert score.raw == "CORRECT"
    assert "Reference answer/support strings" in score.prompt
    assert "- Boston" in score.prompt
    assert "- Seattle" in score.prompt
    assert judge.calls[0]["system"].startswith("You are a strict evaluator")


def test_answer_eval_cli_writes_records_and_experiment(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    records_path = tmp_path / "answer.records.jsonl"
    experiment_path = tmp_path / "answer.experiment.json"

    main([
        "--dataset",
        str(dataset),
        "--baselines",
        "semantic_only",
        "--answer-provider",
        "mock",
        "--mock-answer",
        "The current city is Boston.",
        "--records-output",
        str(records_path),
        "--experiment-output",
        str(experiment_path),
    ])

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0]["baseline"] == "semantic_only"
    assert experiment["run_type"] == "jsonl_answer_generation_benchmark"
    assert experiment["notes"]["ground_truth_runtime_use"] == "forbidden"
    assert experiment["notes"]["ground_truth_evaluation_use"] == "answer_scorer_only"
    assert experiment["results"]["semantic_only"]["total"] == 2
    assert "failure_summary" in experiment["diagnostics"]
    assert experiment["raw_outputs"][0]["answer_prompt"]


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps({
            "id": "case-1",
            "observations": [
                {
                    "label": "old",
                    "content": "The user used to live in Seattle.",
                    "metadata": {"memory_key": "home.city"},
                },
                {
                    "label": "new",
                    "content": "The user now lives in Boston.",
                    "metadata": {"memory_key": "home.city"},
                },
            ],
            "queries": [
                {
                    "id": "q1",
                    "query": "Where does the user currently live?",
                    "expected_substrings": ["Boston"],
                    "forbidden_substrings": ["Seattle"],
                    "top_k": 2,
                },
                {
                    "id": "q2",
                    "query": "Where did the user move from?",
                    "expected_substrings": ["Seattle"],
                    "top_k": 2,
                },
            ],
        }),
        encoding="utf-8",
    )
    return dataset
