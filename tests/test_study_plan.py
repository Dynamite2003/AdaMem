from __future__ import annotations

import json
from pathlib import Path

from adamem.study_plan import (
    STUDY_SETTINGS_SCHEMA_VERSION,
    build_paper_study_plan,
    build_study_plan_from_settings,
    build_smoke_study_plan,
    load_paper_study_plan,
    load_study_settings,
    main,
    parse_model_spec,
    plan_fingerprint,
    run_study_plan,
    settings_fingerprint,
    study_plan_command_listing,
    study_plan_command_listing_markdown,
    validate_paper_study_plan,
    write_paper_study_plan,
    write_study_settings_template,
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

    assert plan["datasets"]["primary_stale"].endswith("study/data/stale.adamem.jsonl")
    assert plan["datasets"]["transfer_long_memory"].endswith("study/data/longmemeval_s.adamem.jsonl")
    assert plan["artifact_policy"]["generated_datasets_default"] == "OUTPUT_DIR/data"
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


def test_build_smoke_study_plan_is_api_free_and_execution_ready() -> None:
    plan = build_smoke_study_plan(output_dir="results/smoke")

    validation = validate_paper_study_plan(plan)
    command_stages = [command["stage"] for command in plan["commands"]]

    assert plan["profile"] == "smoke"
    assert plan["datasets"]["primary_stale"] == "benchmarks/stale_mini.jsonl"
    assert plan["datasets"]["transfer_long_memory"] == "benchmarks/dynamic_state_transfer.jsonl"
    assert plan["data_sources"] == {
        "primary_stale": None,
        "transfer_long_memory": None,
    }
    assert plan["plan_fingerprint"] == plan_fingerprint(plan)
    assert "data_prep" not in command_stages
    assert validation["execution_ready"] is True
    assert validation["plan_fingerprint"] == plan["plan_fingerprint"]
    assert validation["plan_fingerprint_matches_recorded"] is True
    assert validation["required_env_vars"] == []
    assert validation["placeholder_models"] == []
    assert validation["command_count"] == 8


def test_study_plan_command_listing_exposes_command_names() -> None:
    plan = build_smoke_study_plan(output_dir="results/smoke")

    listing = study_plan_command_listing(plan)
    markdown = study_plan_command_listing_markdown(plan)

    assert len(listing) == 8
    assert listing[0]["name"] == plan["commands"][0]["name"]
    assert listing[0]["shell"] == plan["commands"][0]["shell"]
    assert "AdaMem Study Plan Commands" in markdown
    assert "--command NAME" in markdown


def test_write_study_settings_template_is_key_free(tmp_path: Path) -> None:
    template_path = tmp_path / "api_pilot_settings.json"

    artifacts = write_study_settings_template(
        template_path,
        output_dir=tmp_path / "api-study",
    )

    settings = load_study_settings(artifacts["settings"])
    raw = template_path.read_text(encoding="utf-8")
    assert settings["schema_version"] == STUDY_SETTINGS_SCHEMA_VERSION
    assert settings["output_dir"] == str(tmp_path / "api-study")
    assert settings["required_env_vars"] == ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    assert "\"api_key\"" not in raw.lower()
    assert "OPENAI_API_KEY" in raw


def test_load_study_settings_rejects_credential_fields(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings_with_key.json"
    settings_path.write_text(
        json.dumps({
            "schema_version": STUDY_SETTINGS_SCHEMA_VERSION,
            "output_dir": "results/api-study",
            "openai_api_key": "do-not-store",
        }),
        encoding="utf-8",
    )

    try:
        load_study_settings(settings_path)
    except ValueError as exc:
        assert "credential-like fields" in str(exc)
        assert "openai_api_key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected credential-like settings key to fail")


def test_build_study_plan_from_settings_uses_api_pilot_models(tmp_path: Path) -> None:
    settings = {
        "schema_version": STUDY_SETTINGS_SCHEMA_VERSION,
        "profile": "paper",
        "output_dir": str(tmp_path / "api-study"),
        "include_data_prep": False,
        "include_ama": False,
        "stale_dataset": "benchmarks/stale_mini.jsonl",
        "transfer_dataset": "benchmarks/dynamic_state_transfer.jsonl",
        "stale_types": ["T1"],
        "limit_per_stale_type": 2,
        "transfer_max_cases": 3,
        "answer_models": ["openai:gpt-a", "gemini:gemini-a"],
        "judge_models": ["openai:gpt-j", "gemini:gemini-j"],
        "state_extractor_model": "openai:gpt-extractor",
        "top_k": 4,
        "max_context_chars": 1200,
    }

    plan = build_study_plan_from_settings(settings)
    validation = validate_paper_study_plan(plan)
    answer_commands = [
        command for command in plan["commands"] if command["stage"] == "answer_judge"
    ]

    assert plan["output_dir"] == str(tmp_path / "api-study")
    assert plan["data_sources"] == {
        "primary_stale": None,
        "transfer_long_memory": None,
    }
    assert plan["split"]["limit_per_stale_type"] == 2
    assert plan["split"]["transfer_max_cases"] == 3
    assert plan["model_requirements"]["answer_models"] == [
        "openai:gpt-a",
        "gemini:gemini-a",
    ]
    assert len(answer_commands) == 4
    assert plan["settings_provenance"]["schema_version"] == STUDY_SETTINGS_SCHEMA_VERSION
    assert plan["settings_provenance"]["settings_fingerprint"] == settings_fingerprint(settings)
    assert plan["settings_provenance"]["output_dir_overridden"] is False
    assert validation["settings_provenance"] == plan["settings_provenance"]
    assert validation["execution_ready"] is True
    assert validation["required_env_vars"] == ["GEMINI_API_KEY", "OPENAI_API_KEY"]


def test_build_study_plan_from_settings_records_output_override(tmp_path: Path) -> None:
    settings = {
        "schema_version": STUDY_SETTINGS_SCHEMA_VERSION,
        "profile": "smoke",
        "output_dir": str(tmp_path / "original"),
    }

    plan = build_study_plan_from_settings(
        settings,
        output_dir=tmp_path / "override",
        settings_path=tmp_path / "settings.json",
    )

    assert plan["output_dir"] == str(tmp_path / "override")
    assert plan["settings_provenance"]["settings_path"] == str(tmp_path / "settings.json")
    assert plan["settings_provenance"]["output_dir_overridden"] is True
    assert plan["plan_fingerprint"] == plan_fingerprint(plan)


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
    assert "Artifact Policy" in markdown
    assert "method_coverage" not in shell
    assert "python -m adamem.reporting" in shell
    assert validation["schema_version"] == "adamem.paper_study_validation.v1"
    assert "AdaMem Paper Study Validation" in validation_md


def test_load_paper_study_plan_reads_saved_plan(tmp_path: Path) -> None:
    plan = build_smoke_study_plan(output_dir=tmp_path / "study")
    artifacts = write_paper_study_plan(plan, tmp_path / "study")

    loaded = load_paper_study_plan(artifacts["json"])

    assert loaded["profile"] == "smoke"
    assert loaded["commands"][0]["name"] == plan["commands"][0]["name"]
    assert loaded["plan_fingerprint"] == plan_fingerprint(loaded)


def test_plan_fingerprint_detects_saved_plan_edits(tmp_path: Path) -> None:
    plan = build_smoke_study_plan(output_dir=tmp_path / "study")
    original = plan_fingerprint(plan)
    plan["commands"][0]["name"] = "edited"

    validation = validate_paper_study_plan(plan)

    assert plan_fingerprint(plan) != original
    assert validation["recorded_plan_fingerprint"] == original
    assert validation["plan_fingerprint_matches_recorded"] is False
    assert validation["execution_ready"] is False
    assert "plan_fingerprint_matches_recorded" in validation["missing_requirements"]


def test_run_study_plan_blocks_saved_plan_fingerprint_mismatch(tmp_path: Path) -> None:
    plan = build_smoke_study_plan(output_dir=tmp_path / "study")
    plan["commands"][0]["name"] = "edited"

    try:
        run_study_plan(plan, dry_run=True, log_path=tmp_path / "run.jsonl")
    except ValueError as exc:
        assert "plan_fingerprint_matches_recorded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected stale recorded fingerprint to block execution")

    summary = run_study_plan(
        plan,
        dry_run=True,
        require_ready=False,
        log_path=tmp_path / "allowed.records.jsonl",
    )

    assert summary["status"] == "dry_run"
    assert summary["validation"]["plan_fingerprint_matches_recorded"] is False


def test_run_study_plan_supports_dry_run_and_stage_filter(tmp_path: Path) -> None:
    plan = build_smoke_study_plan(output_dir=tmp_path / "study")
    log_path = tmp_path / "study" / "run.records.jsonl"

    summary = run_study_plan(
        plan,
        stages=["diagnostic"],
        dry_run=True,
        log_path=log_path,
    )

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["status"] == "dry_run"
    assert summary["plan_fingerprint"] == plan["plan_fingerprint"]
    assert summary["recorded_plan_fingerprint"] == plan["plan_fingerprint"]
    assert summary["settings_provenance"] == {}
    assert summary["selected_command_count"] == 1
    assert summary["prior_log_record_count"] == 0
    assert summary["appended_record_count"] == 1
    assert summary["final_log_record_count"] == 1
    assert summary["completed_command_count"] == 0
    assert records[0]["status"] == "dry_run"
    assert records[0]["plan_fingerprint"] == plan["plan_fingerprint"]
    assert records[0]["recorded_plan_fingerprint"] == plan["plan_fingerprint"]
    assert records[0]["stage"] == "diagnostic"
    assert records[0]["missing_outputs"] == ["experiment", "records", "report"]


def test_run_study_plan_supports_command_name_filter(tmp_path: Path) -> None:
    plan = build_smoke_study_plan(output_dir=tmp_path / "study")
    target = next(command for command in plan["commands"] if command["stage"] == "transfer")
    log_path = tmp_path / "study" / "command.records.jsonl"

    summary = run_study_plan(
        plan,
        command_names=[target["name"]],
        dry_run=True,
        log_path=log_path,
    )

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["selected_command_filter"] == [target["name"]]
    assert summary["selected_command_count"] == 1
    assert records[0]["name"] == target["name"]


def test_run_study_plan_command_filter_rejects_unknown_name(tmp_path: Path) -> None:
    plan = build_smoke_study_plan(output_dir=tmp_path / "study")

    try:
        run_study_plan(
            plan,
            command_names=["missing-command"],
            dry_run=True,
            log_path=tmp_path / "run.jsonl",
        )
    except ValueError as exc:
        assert "missing-command" in str(exc)
        assert "available" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unknown command filter to fail")


def test_run_study_plan_executes_simple_command(tmp_path: Path) -> None:
    output_path = tmp_path / "ok.txt"
    plan = {
        "output_dir": str(tmp_path / "study"),
        "datasets": {},
        "data_sources": {},
        "model_requirements": {
            "answer_models": ["mock:a", "mock:b"],
            "judge_models": ["mock:j1", "mock:j2"],
            "minimum_answer_models": 2,
            "minimum_judge_models": 2,
        },
        "method_coverage_preview": {"complete": True},
        "commands": [
            {
                "name": "ok",
                "stage": "unit",
                "purpose": "test command",
                "claim_boundary": "none",
                "command": ["python", "-c", f"from pathlib import Path; Path({str(output_path)!r}).write_text('ok')"],
                "outputs": {"artifact": str(output_path)},
            }
        ],
    }

    summary = run_study_plan(
        plan,
        require_ready=False,
        log_path=tmp_path / "run.jsonl",
    )

    assert summary["status"] == "complete"
    assert summary["completed_command_count"] == 1
    assert summary["missing_output_count"] == 0
    assert summary["records"][0]["output_checks"]["artifact"]["exists"] is True
    assert summary["records"][0]["missing_outputs"] == []


def test_run_study_plan_resume_skips_prior_completed_command(tmp_path: Path) -> None:
    output_path = tmp_path / "counter.txt"
    command = (
        "from pathlib import Path; "
        f"path=Path({str(output_path)!r}); "
        "value=int(path.read_text() or '0') if path.exists() else 0; "
        "path.write_text(str(value + 1))"
    )
    plan = {
        "output_dir": str(tmp_path / "study"),
        "datasets": {},
        "data_sources": {},
        "model_requirements": {
            "answer_models": ["mock:a", "mock:b"],
            "judge_models": ["mock:j1", "mock:j2"],
            "minimum_answer_models": 2,
            "minimum_judge_models": 2,
        },
        "method_coverage_preview": {"complete": True},
        "commands": [
            {
                "name": "increment",
                "stage": "unit",
                "purpose": "test resume",
                "claim_boundary": "none",
                "command": ["python", "-c", command],
                "outputs": {"artifact": str(output_path)},
            }
        ],
    }
    log_path = tmp_path / "run.records.jsonl"

    first = run_study_plan(plan, require_ready=False, log_path=log_path)
    second = run_study_plan(plan, require_ready=False, resume=True, log_path=log_path)

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert first["completed_command_count"] == 1
    assert second["completed_command_count"] == 0
    assert second["skipped_completed_count"] == 1
    assert second["prior_log_record_count"] == 1
    assert second["appended_record_count"] == 1
    assert second["final_log_record_count"] == 2
    assert second["records"][0]["status"] == "skipped_completed"
    assert second["records"][0]["plan_fingerprint"] == first["plan_fingerprint"]
    assert records[-1]["status"] == "skipped_completed"
    assert output_path.read_text(encoding="utf-8") == "1"


def test_run_study_plan_resume_does_not_skip_missing_outputs(tmp_path: Path) -> None:
    marker_path = tmp_path / "marker.txt"
    missing_path = tmp_path / "missing.txt"
    command = f"from pathlib import Path; Path({str(marker_path)!r}).write_text('ran')"
    plan = {
        "output_dir": str(tmp_path / "study"),
        "datasets": {},
        "data_sources": {},
        "model_requirements": {
            "answer_models": ["mock:a", "mock:b"],
            "judge_models": ["mock:j1", "mock:j2"],
            "minimum_answer_models": 2,
            "minimum_judge_models": 2,
        },
        "method_coverage_preview": {"complete": True},
        "commands": [
            {
                "name": "missing-output",
                "stage": "unit",
                "purpose": "test resume missing output",
                "claim_boundary": "none",
                "command": ["python", "-c", command],
                "outputs": {"artifact": str(missing_path)},
            }
        ],
    }
    log_path = tmp_path / "run.records.jsonl"

    first = run_study_plan(plan, require_ready=False, log_path=log_path)
    second = run_study_plan(plan, require_ready=False, resume=True, log_path=log_path)

    assert first["records"][0]["status"] == "completed"
    assert first["records"][0]["missing_outputs"] == ["artifact"]
    assert second["records"][0]["status"] == "completed"
    assert second["skipped_completed_count"] == 0


def test_run_study_plan_resume_does_not_skip_different_plan_fingerprint(tmp_path: Path) -> None:
    output_path = tmp_path / "counter.txt"
    command = (
        "from pathlib import Path; "
        f"path=Path({str(output_path)!r}); "
        "value=int(path.read_text() or '0') if path.exists() else 0; "
        "path.write_text(str(value + 1))"
    )
    plan = {
        "output_dir": str(tmp_path / "study"),
        "datasets": {},
        "data_sources": {},
        "model_requirements": {
            "answer_models": ["mock:a", "mock:b"],
            "judge_models": ["mock:j1", "mock:j2"],
            "minimum_answer_models": 2,
            "minimum_judge_models": 2,
        },
        "method_coverage_preview": {"complete": True},
        "commands": [
            {
                "name": "increment",
                "stage": "unit",
                "purpose": "test resume fingerprint",
                "claim_boundary": "none",
                "command": ["python", "-c", command],
                "outputs": {"artifact": str(output_path)},
            }
        ],
    }
    log_path = tmp_path / "run.records.jsonl"

    first = run_study_plan(plan, require_ready=False, log_path=log_path)
    plan["settings_provenance"] = {"settings_fingerprint": "changed"}
    plan["plan_fingerprint"] = plan_fingerprint(plan)
    second = run_study_plan(plan, require_ready=False, resume=True, log_path=log_path)

    assert first["plan_fingerprint"] != second["plan_fingerprint"]
    assert second["records"][0]["status"] == "completed"
    assert second["skipped_completed_count"] == 0
    assert output_path.read_text(encoding="utf-8") == "2"


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


def test_study_plan_cli_writes_smoke_profile(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-smoke"

    main([
        "--profile",
        "smoke",
        "--output-dir",
        str(output_dir),
        "--json",
    ])

    data = json.loads((output_dir / "paper_study_plan.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "paper_study_validation.json").read_text(encoding="utf-8"))
    assert data["profile"] == "smoke"
    assert validation["execution_ready"] is True


def test_study_plan_cli_writes_settings_template(tmp_path: Path) -> None:
    settings_path = tmp_path / "api_pilot_settings.json"

    main([
        "--write-settings-template",
        str(settings_path),
        "--output-dir",
        str(tmp_path / "api-study"),
        "--json",
    ])

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["schema_version"] == STUDY_SETTINGS_SCHEMA_VERSION
    assert settings["output_dir"] == str(tmp_path / "api-study")
    assert settings["limit_per_stale_type"] == 5
    assert settings["include_ama"] is False


def test_study_plan_cli_generates_from_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings = {
        "schema_version": STUDY_SETTINGS_SCHEMA_VERSION,
        "profile": "paper",
        "output_dir": str(tmp_path / "settings-study"),
        "include_data_prep": False,
        "include_ama": False,
        "stale_dataset": "benchmarks/stale_mini.jsonl",
        "transfer_dataset": "benchmarks/dynamic_state_transfer.jsonl",
        "answer_models": ["openai:gpt-a", "gemini:gemini-a"],
        "judge_models": ["openai:gpt-j", "gemini:gemini-j"],
        "state_extractor_model": "openai:gpt-extractor",
        "limit_per_stale_type": 1,
        "transfer_max_cases": 2,
    }
    settings_path.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")

    main([
        "--settings",
        str(settings_path),
        "--json",
    ])

    plan = json.loads((tmp_path / "settings-study" / "paper_study_plan.json").read_text(encoding="utf-8"))
    validation = json.loads((tmp_path / "settings-study" / "paper_study_validation.json").read_text(encoding="utf-8"))
    validation_md = (tmp_path / "settings-study" / "paper_study_validation.md").read_text(encoding="utf-8")
    assert plan["model_requirements"]["answer_models"] == ["openai:gpt-a", "gemini:gemini-a"]
    assert plan["split"]["limit_per_stale_type"] == 1
    assert plan["settings_provenance"]["settings_path"] == str(settings_path)
    assert plan["settings_provenance"]["settings_fingerprint"] == settings_fingerprint(settings)
    assert validation["settings_provenance"] == plan["settings_provenance"]
    assert "Settings Provenance" in validation_md
    assert validation["execution_ready"] is True


def test_study_plan_cli_run_summary_records_settings_provenance(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings-run.json"
    settings = {
        "schema_version": STUDY_SETTINGS_SCHEMA_VERSION,
        "profile": "smoke",
        "output_dir": str(tmp_path / "settings-run-study"),
    }
    settings_path.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")

    main([
        "--settings",
        str(settings_path),
        "--run",
        "--dry-run",
        "--stage",
        "diagnostic",
        "--json",
    ])

    summary = json.loads((tmp_path / "settings-run-study" / "paper_study_run.summary.json").read_text(encoding="utf-8"))
    summary_md = (tmp_path / "settings-run-study" / "paper_study_run.summary.md").read_text(encoding="utf-8")
    assert summary["settings_provenance"]["settings_path"] == str(settings_path)
    assert summary["settings_provenance"]["settings_fingerprint"] == settings_fingerprint(settings)
    assert "Settings Provenance" in summary_md


def test_study_plan_cli_can_resume_saved_plan_run(tmp_path: Path) -> None:
    output_path = tmp_path / "cli-counter.txt"
    command = (
        "from pathlib import Path; "
        f"path=Path({str(output_path)!r}); "
        "value=int(path.read_text() or '0') if path.exists() else 0; "
        "path.write_text(str(value + 1))"
    )
    plan = {
        "output_dir": str(tmp_path / "resume-study"),
        "datasets": {},
        "data_sources": {},
        "model_requirements": {
            "answer_models": ["mock:a", "mock:b"],
            "judge_models": ["mock:j1", "mock:j2"],
            "minimum_answer_models": 2,
            "minimum_judge_models": 2,
        },
        "method_coverage_preview": {"complete": True},
        "commands": [
            {
                "name": "increment",
                "stage": "unit",
                "purpose": "test cli resume",
                "claim_boundary": "none",
                "command": ["python", "-c", command],
                "shell": f"python -c {command!r}",
                "outputs": {"artifact": str(output_path)},
            }
        ],
    }
    output_dir = tmp_path / "resume-study"
    artifacts = write_paper_study_plan(plan, output_dir)

    main(["--plan", artifacts["json"], "--run", "--allow-not-ready", "--json"])
    main(["--plan", artifacts["json"], "--run", "--resume-run", "--allow-not-ready", "--json"])

    summary = json.loads((output_dir / "paper_study_run.summary.json").read_text(encoding="utf-8"))
    summary_md = (output_dir / "paper_study_run.summary.md").read_text(encoding="utf-8")
    assert summary["resume"] is True
    assert summary["skipped_completed_count"] == 1
    assert summary["prior_log_record_count"] == 1
    assert summary["appended_record_count"] == 1
    assert summary["final_log_record_count"] == 2
    assert "Skipped completed commands" in summary_md
    assert "Prior log records" in summary_md
    assert output_path.read_text(encoding="utf-8") == "1"


def test_study_plan_cli_can_filter_by_command_name(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-command-filter"
    main([
        "--profile",
        "smoke",
        "--output-dir",
        str(output_dir),
        "--json",
    ])
    plan = json.loads((output_dir / "paper_study_plan.json").read_text(encoding="utf-8"))
    target = next(command for command in plan["commands"] if command["stage"] == "transfer")

    main([
        "--plan",
        str(output_dir / "paper_study_plan.json"),
        "--run",
        "--dry-run",
        "--command",
        target["name"],
        "--json",
    ])

    summary = json.loads((output_dir / "paper_study_run.summary.json").read_text(encoding="utf-8"))
    summary_md = (output_dir / "paper_study_run.summary.md").read_text(encoding="utf-8")
    assert summary["selected_command_filter"] == [target["name"]]
    assert summary["selected_command_count"] == 1
    assert "Command filter" in summary_md


def test_study_plan_cli_can_list_saved_plan_commands(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "cli-list-commands"
    main([
        "--profile",
        "smoke",
        "--output-dir",
        str(output_dir),
        "--json",
    ])
    capsys.readouterr()

    main([
        "--plan",
        str(output_dir / "paper_study_plan.json"),
        "--list-commands",
        "--json",
    ])

    printed = json.loads(capsys.readouterr().out)
    names = [command["name"] for command in printed["command_listing"]]
    assert "longmemeval_transfer_retrieval" in names
    assert printed["command_listing"][0]["stage"]
    assert printed["command_listing"][0]["claim_boundary"]


def test_study_plan_cli_can_dry_run_smoke_profile(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-smoke-run"

    main([
        "--profile",
        "smoke",
        "--output-dir",
        str(output_dir),
        "--run",
        "--dry-run",
        "--stage",
        "diagnostic",
        "--json",
    ])

    summary = json.loads((output_dir / "paper_study_run.summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert summary["selected_command_count"] == 1
    assert (output_dir / "paper_study_run.summary.md").exists()


def test_study_plan_cli_can_run_saved_plan_without_regenerating(tmp_path: Path) -> None:
    output_dir = tmp_path / "saved"
    main([
        "--profile",
        "smoke",
        "--output-dir",
        str(output_dir),
        "--json",
    ])
    plan_path = output_dir / "paper_study_plan.json"

    main([
        "--plan",
        str(plan_path),
        "--run",
        "--dry-run",
        "--stage",
        "diagnostic",
        "--json",
    ])

    summary = json.loads((output_dir / "paper_study_run.summary.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "paper_study_validation.json").read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert summary["selected_command_count"] == 1
    assert validation["execution_ready"] is True


def test_study_plan_cli_can_refresh_saved_plan_fingerprint(tmp_path: Path) -> None:
    output_dir = tmp_path / "saved-refresh"
    main([
        "--profile",
        "smoke",
        "--output-dir",
        str(output_dir),
        "--json",
    ])
    plan_path = output_dir / "paper_study_plan.json"
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    data["objective"] = "edited_after_review"
    plan_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    before = validate_paper_study_plan(load_paper_study_plan(plan_path))
    assert before["execution_ready"] is False
    assert before["plan_fingerprint_matches_recorded"] is False

    main([
        "--plan",
        str(plan_path),
        "--refresh-fingerprint",
        "--json",
    ])

    refreshed = load_paper_study_plan(plan_path)
    validation = json.loads((output_dir / "paper_study_validation.json").read_text(encoding="utf-8"))
    assert refreshed["plan_fingerprint"] == plan_fingerprint(refreshed)
    assert validation["plan_fingerprint_matches_recorded"] is True
    assert validation["execution_ready"] is True
