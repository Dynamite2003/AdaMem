from __future__ import annotations

import json
from pathlib import Path

from adamem.study_plan import (
    build_paper_study_plan,
    main,
    parse_model_spec,
    validate_paper_study_plan,
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


def test_validate_paper_study_plan_reports_placeholders_and_missing_paths(tmp_path: Path) -> None:
    plan = build_paper_study_plan(output_dir=tmp_path / "study")

    validation = validate_paper_study_plan(plan, root=tmp_path)

    assert validation["execution_ready"] is False
    assert validation["missing_datasets"] == [
        "primary_stale",
        "transfer_long_memory",
        "transfer_ama_source",
    ]
    assert "replace_model_placeholders" in validation["missing_requirements"]
    assert "<answer_provider_a>:<answer_model_a>" in validation["placeholder_models"]
    assert validation["method_coverage_complete"] is True
    assert validation["command_count"] == 11
    assert validation["command_stage_counts"]["data_prep"] == 2
    assert validation["reporting_command_present"] is True


def test_validate_paper_study_plan_accepts_missing_target_when_prep_source_exists(tmp_path: Path) -> None:
    transfer_source = tmp_path / "longmemeval_s_cleaned.json"
    stale_source = tmp_path / "T1_T2_400_FULL.json"
    ama = tmp_path / "ama.raw.jsonl"
    for path in [transfer_source, stale_source, ama]:
        path.write_text("[]", encoding="utf-8")
    plan = build_paper_study_plan(
        output_dir=tmp_path / "study",
        stale_dataset=tmp_path / "prepared_stale.jsonl",
        transfer_dataset=tmp_path / "prepared_lme.jsonl",
        stale_source=stale_source,
        transfer_source=transfer_source,
        ama_output_source=ama,
        answer_models=["openai:gpt-a", "gemini:gemini-a"],
        judge_models=["openai:gpt-j", "gemini:gemini-j"],
        state_extractor_model="openai:gpt-extractor",
    )

    validation = validate_paper_study_plan(plan, root=tmp_path)

    assert validation["execution_ready"] is True
    assert validation["missing_datasets"] == []
    assert validation["dataset_checks"]["primary_stale"]["prepared_by_plan"] is True
    assert validation["dataset_checks"]["primary_stale"]["prep_source_exists"] is True
    assert validation["source_checks"]["transfer_long_memory"]["exists"] is True


def test_validate_paper_study_plan_marks_ready_when_paths_and_models_are_set(tmp_path: Path) -> None:
    stale = tmp_path / "stale.jsonl"
    transfer = tmp_path / "transfer.jsonl"
    ama = tmp_path / "ama.raw.jsonl"
    for path in [stale, transfer, ama]:
        path.write_text("", encoding="utf-8")
    plan = build_paper_study_plan(
        output_dir=tmp_path / "study",
        stale_dataset=stale,
        transfer_dataset=transfer,
        ama_output_source=ama,
        answer_models=["openai:gpt-a", "gemini:gemini-a"],
        judge_models=["openai:gpt-j", "gemini:gemini-j"],
        state_extractor_model="openai:gpt-extractor",
    )

    validation = validate_paper_study_plan(plan, root=tmp_path)

    assert validation["execution_ready"] is True
    assert validation["missing_requirements"] == []
    assert validation["required_env_vars"] == ["GEMINI_API_KEY", "OPENAI_API_KEY"]
    assert validation["env_checked"] is False
    assert validation["missing_env_vars"] == []


def test_validate_paper_study_plan_can_check_missing_env_vars(tmp_path: Path, monkeypatch) -> None:
    stale = tmp_path / "stale.jsonl"
    transfer = tmp_path / "transfer.jsonl"
    for path in [stale, transfer]:
        path.write_text("", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan = build_paper_study_plan(
        output_dir=tmp_path / "study",
        stale_dataset=stale,
        transfer_dataset=transfer,
        ama_output_source=None,
        answer_models=["openai:gpt-a", "mock:answer"],
        judge_models=["openai:gpt-j", "mock:judge"],
        state_extractor_model="openai:gpt-extractor",
    )

    validation = validate_paper_study_plan(plan, root=tmp_path, check_env=True)

    assert validation["execution_ready"] is False
    assert validation["required_env_vars"] == ["OPENAI_API_KEY"]
    assert validation["missing_env_vars"] == ["OPENAI_API_KEY"]
    assert "provider_credentials_available" in validation["missing_requirements"]


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
    validation = json.loads(Path(artifacts["validation_json"]).read_text(encoding="utf-8"))
    validation_md = Path(artifacts["validation_markdown"]).read_text(encoding="utf-8")
    assert data["schema_version"] == "adamem.paper_study_plan.v1"
    assert "AdaMem Paper Study Plan" in markdown
    assert "method_coverage" not in shell
    assert "python -m adamem.reporting" in shell
    assert validation["schema_version"] == "adamem.paper_study_validation.v1"
    assert "AdaMem Paper Study Validation" in validation_md


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
    assert (output_dir / "paper_study_validation.json").exists()
    assert (output_dir / "paper_study_validation.md").exists()
