from __future__ import annotations

import json
from pathlib import Path

from adamem.study_plan import (
    build_paper_study_plan,
    main,
    parse_model_spec,
    write_paper_study_plan,
)


def test_parse_model_spec_requires_provider_and_model() -> None:
    spec = parse_model_spec("openai:gpt-test")

    assert spec.provider == "openai"
    assert spec.model == "gpt-test"
    assert spec.label == "openai:gpt-test"


def test_build_paper_study_plan_requires_answer_and_judge_models(tmp_path: Path) -> None:
    try:
        build_paper_study_plan(
            output_dir=tmp_path / "study",
            answer_models=[],
            judge_models=["openai:gpt-j"],
        )
    except ValueError as exc:
        assert "answer model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected empty answer model list to fail")


def test_build_paper_study_plan_covers_method_and_model_matrix(tmp_path: Path) -> None:
    plan = build_paper_study_plan(
        output_dir=tmp_path / "study",
        answer_models=["openai:gpt-a", "gemini:gemini-a"],
        judge_models=["openai:gpt-j", "gemini:gemini-j"],
        state_extractor_model="openai:gpt-extractor",
        limit_per_stale_type=2,
        transfer_max_cases=3,
        ama_limit=4,
    )

    command_names = [command["name"] for command in plan["commands"]]
    answer_commands = [
        command for command in plan["commands"] if command["stage"] == "answer_judge"
    ]
    coverage = plan["method_coverage_preview"]

    assert plan["model_requirements"]["answer_models"] == [
        "openai:gpt-a",
        "gemini:gemini-a",
    ]
    assert len(answer_commands) == 4
    assert "stale_retrieval_diagnostics" in command_names
    assert "longmemeval_transfer_retrieval" in command_names
    assert "ama_public_retrieval" in command_names
    assert "paper_report_bundle" in command_names
    assert coverage["complete"] is True
    assert coverage["missing_requirements"] == []
    assert coverage["mechanism_flags"]["premise_correction"] is True
    assert coverage["mechanism_flags"]["llm_state_extractor"] is True
    assert coverage["mechanism_flags"]["trajectory_step_readout"] is True
    assert "--state-extractor-provider" in next(
        command["command"]
        for command in plan["commands"]
        if command["stage"] == "mechanism_ablation"
    )


def test_write_paper_study_plan_outputs_json_markdown_and_shell(tmp_path: Path) -> None:
    plan = build_paper_study_plan(
        output_dir=tmp_path / "study",
        answer_models=["openai:gpt-a", "gemini:gemini-a"],
        judge_models=["openai:gpt-j", "gemini:gemini-j"],
        state_extractor_model="openai:gpt-extractor",
        ama_output_source=None,
    )

    artifacts = write_paper_study_plan(plan, tmp_path / "study")

    data = json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    shell = Path(artifacts["shell"]).read_text(encoding="utf-8")
    assert data["schema_version"] == "adamem.paper_study_plan.v1"
    assert "AdaMem Paper Study Plan" in markdown
    assert "method_coverage" not in shell
    assert "python -m adamem.reporting" in shell


def test_study_plan_cli_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-study"

    main([
        "--output-dir",
        str(output_dir),
        "--answer-model",
        "openai:gpt-a",
        "--answer-model",
        "gemini:gemini-a",
        "--judge-model",
        "openai:gpt-j",
        "--judge-model",
        "gemini:gemini-j",
        "--state-extractor-model",
        "openai:gpt-extractor",
        "--no-ama",
        "--json",
    ])

    assert (output_dir / "paper_study_plan.json").exists()
    assert (output_dir / "paper_study_plan.md").exists()
    assert (output_dir / "paper_study_commands.sh").exists()
