from __future__ import annotations

from pathlib import Path

from adamem.bench import load_jsonl_cases, run_benchmark
from adamem.convert import convert_locomo_file
from adamem.eval import run_synthetic_benchmark


def test_synthetic_benchmark_shows_full_beats_semantic_only() -> None:
    results = {result.name: result for result in run_synthetic_benchmark()}

    assert results["full"].accuracy > results["semantic_only"].accuracy
    assert results["full"].passed == results["full"].total
    assert results["semantic_only"].passed < results["semantic_only"].total


def test_synthetic_benchmark_exposes_case_traces() -> None:
    result = run_synthetic_benchmark()[0]
    case = result.cases[0]

    assert case.trace
    assert "score" in case.trace[0]
    assert "contributions" in case.trace[0]


def test_jsonl_benchmark_adapter_runs_fixture() -> None:
    cases = load_jsonl_cases(Path("benchmarks/tiny_memory_qa.jsonl"))
    results = {result.name: result for result in run_benchmark(cases)}

    assert results["full"].passed == results["full"].total
    assert results["semantic_only"].passed < results["full"].passed


def test_locomo_converter_emits_adamem_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "locomo.jsonl"
    count = convert_locomo_file("benchmarks/locomo_mini.json", output)

    cases = load_jsonl_cases(output)
    results = {result.name: result for result in run_benchmark(cases)}

    assert count == 1
    assert len(cases[0].observations) == 5
    assert len(cases[0].queries) == 2
    assert cases[0].queries[0].expected_substrings == ["D1:1"]
    assert results["full"].passed == results["full"].total
