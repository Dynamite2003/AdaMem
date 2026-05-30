from __future__ import annotations

import json
from pathlib import Path

from adamem.lme_v2 import (
    longmemeval_v2_split_trajectory_records,
    longmemeval_v2_prepared_state_evidence_records,
    longmemeval_v2_question_audit_records,
    select_longmemeval_v2_transfer_split,
    state_slot_family,
    summarize_longmemeval_v2_prepared_state_evidence,
    summarize_longmemeval_v2_question_audit,
    summarize_longmemeval_v2_trajectory_manifest,
    summarize_longmemeval_v2_transfer_split,
    validate_longmemeval_v2_prepared_split,
    write_longmemeval_v2_extracted_trajectories,
    write_longmemeval_v2_prepared_split_validation,
    write_longmemeval_v2_prepared_state_evidence_audit,
    write_longmemeval_v2_question_audit,
    write_longmemeval_v2_trajectory_manifest,
    write_longmemeval_v2_transfer_split,
)


def test_longmemeval_v2_question_audit_marks_state_transfer_candidates() -> None:
    questions = [
        {
            "id": "dynamic_q",
            "domain": "enterprise",
            "environment": "workarena",
            "question_type": "dynamic-environment",
            "question": "Is the staging build runner offline?",
            "answer": "No",
            "eval_function": "norm_phrase_set_match|lower=true",
        },
        {
            "id": "static_q",
            "domain": "web",
            "environment": "shopping",
            "question_type": "static-environment",
            "question": "What button is visible on the product page?",
            "answer": "Checkout",
            "eval_function": "norm_phrase_set_match",
        },
        {
            "id": "procedure_abs_q",
            "domain": "enterprise",
            "environment": "workarena",
            "question_type": "procedure-abs",
            "question": "In our company workflow, what module should I use?",
            "answer": "There is no such module.",
            "eval_function": "llm_abstention_checker|require_non_empty=true",
        },
        {
            "id": "gotcha_q",
            "domain": "web",
            "environment": "shopping",
            "question_type": "errors-gotchas",
            "question": "What gotcha should I remember for the checkout page?",
            "answer": "Apply coupons after shipping.",
            "eval_function": "norm_phrase_set_match",
        },
    ]
    records = list(longmemeval_v2_question_audit_records(
        questions,
        haystacks={
            "dynamic_q": ["traj1", "traj2"],
            "static_q": ["traj3"],
            "gotcha_q": ["traj4"],
        },
    ))
    by_id = {record["id"]: record for record in records}
    summary = summarize_longmemeval_v2_question_audit(records)

    assert by_id["dynamic_q"]["state_transfer_candidate"] is True
    assert by_id["dynamic_q"]["candidate_reasons"] == ["question_type", "query_state_slot"]
    assert by_id["dynamic_q"]["state_slot"] == "runtime.*.status"
    assert by_id["dynamic_q"]["haystack_size"] == 2
    assert by_id["procedure_abs_q"]["abstention"] is True
    assert by_id["procedure_abs_q"]["state_transfer_candidate"] is True
    assert by_id["procedure_abs_q"]["eval_function_family"] == "llm_abstention_checker"
    assert by_id["procedure_abs_q"]["haystack_size"] is None
    assert by_id["gotcha_q"]["state_transfer_candidate"] is True
    assert by_id["gotcha_q"]["candidate_reasons"] == ["question_type", "query_state_slot"]
    assert by_id["gotcha_q"]["state_slot"] == "environment.*.gotcha"
    assert by_id["static_q"]["state_transfer_candidate"] is False
    assert all("answer" not in record for record in records)
    assert summary["total_questions"] == 4
    assert summary["state_transfer_candidate_questions"] == 3
    assert summary["inferred_state_slot_questions"] == 3
    assert summary["with_haystack_questions"] == 3
    assert summary["missing_haystack_questions"] == 1
    assert summary["by_question_type"]["dynamic-environment"]["state_transfer_candidates"] == 1
    assert summary["by_state_slot"]["runtime.*.status"] == 1
    assert summary["by_state_slot"]["environment.*.gotcha"] == 1
    assert summary["by_candidate_reason"]["question_type"] == 3


def test_write_longmemeval_v2_question_audit_outputs_records_summary_and_report(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join([
            json.dumps({
                "id": "q1",
                "domain": "enterprise",
                "environment": "workarena",
                "question_type": "dynamic-environment",
                "question": "What is the current workflow status?",
                "answer": "done",
                "eval_function": "norm_phrase_set_match",
            }),
            json.dumps({
                "id": "q2",
                "domain": "web",
                "environment": "shopping",
                "question_type": "static-environment",
                "question": "What color is the checkout button?",
                "answer": "blue",
                "eval_function": "norm_phrase_set_match",
            }),
        ]),
        encoding="utf-8",
    )
    haystack = tmp_path / "lme_v2_small.json"
    haystack.write_text(json.dumps({"q1": ["traj1"]}), encoding="utf-8")
    output_dir = tmp_path / "audit"

    result = write_longmemeval_v2_question_audit(
        questions,
        output_dir,
        haystack_source=haystack,
        limit=1,
    )
    records_path = Path(result["records_path"])
    summary_path = Path(result["summary_path"])
    report_path = Path(result["report_path"])
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert records_path.exists()
    assert summary_path.exists()
    assert report_path.exists()
    assert [record["id"] for record in records] == ["q1"]
    assert records[0]["haystack_size"] == 1
    assert "answer" not in records[0]
    assert summary["total_questions"] == 1
    assert "LongMemEval-V2 Question Audit" in report
    assert "dynamic-environment" in report


def test_longmemeval_v2_transfer_split_balances_candidates_and_controls() -> None:
    records = [
        _audit_record("dyn1", "dynamic-environment", state_slots=["runtime.*.status"]),
        _audit_record("dyn2", "dynamic-environment", state_slots=["resource.*.status"]),
        _audit_record("proc1", "procedure", state_slots=["workflow.*"]),
        _audit_record("proc_img", "procedure", image_required=True, state_slots=["workflow.*"]),
        _audit_record("gotcha_img", "errors-gotchas", image_required=True, state_slots=["workflow.*"]),
        _audit_record("static_warn1", "static-environment", type_candidate=False, state_slots=["location"]),
        _audit_record("static_warn2", "static-environment-abs", type_candidate=False, state_slots=["location"]),
        _audit_record("static_clean1", "static-environment", type_candidate=False),
        _audit_record("static_clean2", "static-environment-abs", type_candidate=False),
        _audit_record("missing_haystack", "dynamic-environment", haystack_size=None, state_slots=["runtime.*.status"]),
    ]

    selected = select_longmemeval_v2_transfer_split(
        records,
        transfer_per_type=1,
        control_per_group=1,
        include_image_required=False,
        require_haystack=True,
    )
    summary = summarize_longmemeval_v2_transfer_split(
        selected,
        audit_records=records,
        transfer_per_type=1,
        control_per_group=1,
        include_image_required=False,
        require_haystack=True,
    )

    assert [record["id"] for record in selected] == [
        "dyn1",
        "proc1",
        "static_warn1",
        "static_clean1",
    ]
    assert [record["split"] for record in selected] == [
        "transfer",
        "transfer",
        "router_warning_control",
        "static_clean_control",
    ]
    assert all("answer" not in record for record in selected)
    assert summary["total_selected"] == 4
    assert summary["by_split"] == {
        "transfer": 2,
        "router_warning_control": 1,
        "static_clean_control": 1,
    }
    assert summary["excluded_image_required"] == 2
    assert summary["excluded_missing_haystack"] == 1
    assert summary["transfer_candidate_availability"]["errors-gotchas"] == {
        "source_candidates": 1,
        "eligible_candidates": 0,
        "selected": 0,
    }
    assert summary["question_ids"] == ["dyn1", "proc1", "static_warn1", "static_clean1"]


def test_longmemeval_v2_transfer_split_balances_domains_within_type() -> None:
    records = [
        _audit_record("dyn_ent_1", "dynamic-environment", domain="enterprise"),
        _audit_record("dyn_ent_2", "dynamic-environment", domain="enterprise"),
        _audit_record("dyn_web_1", "dynamic-environment", domain="web"),
        _audit_record("dyn_web_2", "dynamic-environment", domain="web"),
        _audit_record("static_ent", "static-environment", type_candidate=False, state_slots=["location"], domain="enterprise"),
        _audit_record("static_web", "static-environment", type_candidate=False, state_slots=["location"], domain="web"),
    ]

    selected = select_longmemeval_v2_transfer_split(
        records,
        transfer_per_type=4,
        control_per_group=2,
    )

    assert [record["id"] for record in selected[:4]] == [
        "dyn_ent_1",
        "dyn_web_1",
        "dyn_ent_2",
        "dyn_web_2",
    ]
    assert [record["id"] for record in selected[4:]] == ["static_ent", "static_web"]


def test_write_longmemeval_v2_transfer_split_outputs_manifest(tmp_path: Path) -> None:
    audit_records = tmp_path / "audit.records.jsonl"
    audit_records.write_text(
        "\n".join([
            json.dumps(_audit_record("dyn1", "dynamic-environment", state_slots=["runtime.*.status"])),
            json.dumps(_audit_record("proc1", "procedure", state_slots=["workflow.*"])),
            json.dumps(_audit_record("static_warn", "static-environment", type_candidate=False, state_slots=["location"])),
            json.dumps(_audit_record("static_clean", "static-environment", type_candidate=False)),
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "split"

    result = write_longmemeval_v2_transfer_split(
        audit_records,
        output_dir,
        transfer_per_type=2,
        control_per_group=1,
    )
    records_path = Path(result["records_path"])
    manifest_path = Path(result["manifest_path"])
    report_path = Path(result["report_path"])
    selected = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert records_path.exists()
    assert manifest_path.exists()
    assert report_path.exists()
    assert [record["id"] for record in selected] == ["dyn1", "proc1", "static_warn", "static_clean"]
    assert manifest["selection_policy"]["label_use"] == "question metadata only; reference answers excluded"
    assert manifest["transfer_candidate_availability"]["dynamic-environment"]["selected"] == 1
    assert manifest["by_split"]["transfer"] == 2
    assert "LongMemEval-V2 Transfer Split" in report


def test_longmemeval_v2_trajectory_manifest_summarizes_required_ids() -> None:
    split_records = [
        {**_audit_record("dyn1", "dynamic-environment"), "split": "transfer", "selection_group": "dynamic-environment"},
        {**_audit_record("proc1", "procedure"), "split": "transfer", "selection_group": "procedure"},
        {
            **_audit_record("static_warn", "static-environment", type_candidate=False, state_slots=["location"]),
            "split": "router_warning_control",
            "selection_group": "static_query_state_slot_signal",
        },
        {
            **_audit_record("missing", "procedure"),
            "split": "transfer",
            "selection_group": "procedure",
        },
    ]
    records = list(longmemeval_v2_split_trajectory_records(
        split_records,
        haystacks={
            "dyn1": ["traj-a", "traj-b"],
            "proc1": ["traj-b", "traj-c"],
            "static_warn": ["traj-static"],
        },
    ))
    summary = summarize_longmemeval_v2_trajectory_manifest(records)
    by_id = {record["id"]: record for record in records}

    assert by_id["dyn1"]["trajectory_ids"] == ["traj-a", "traj-b"]
    assert by_id["missing"]["haystack_missing"] is True
    assert "answer" not in by_id["dyn1"]
    assert summary["total_questions"] == 4
    assert summary["unique_trajectories"] == 4
    assert summary["trajectory_references"] == 5
    assert summary["missing_haystack_questions"] == 1
    assert summary["trajectory_ids"] == ["traj-a", "traj-b", "traj-c", "traj-static"]
    assert summary["by_split"]["transfer"]["questions"] == 3
    assert summary["by_split"]["transfer"]["unique_trajectories"] == 3
    assert summary["by_split"]["router_warning_control"]["unique_trajectories"] == 1


def test_write_longmemeval_v2_trajectory_manifest_outputs_artifacts(tmp_path: Path) -> None:
    split_records = tmp_path / "split.records.jsonl"
    split_records.write_text(
        "\n".join([
            json.dumps({
                **_audit_record("dyn1", "dynamic-environment", state_slots=["runtime.*.status"]),
                "split": "transfer",
                "selection_group": "dynamic-environment",
            }),
            json.dumps({
                **_audit_record("static_clean", "static-environment", type_candidate=False),
                "split": "static_clean_control",
                "selection_group": "static_no_state_slot_signal",
            }),
        ]),
        encoding="utf-8",
    )
    haystack = tmp_path / "haystack.json"
    haystack.write_text(
        json.dumps({
            "dyn1": ["traj-a", "traj-b"],
            "static_clean": ["traj-b", "traj-c"],
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "manifest"

    result = write_longmemeval_v2_trajectory_manifest(split_records, haystack, output_dir)
    question_records_path = Path(result["question_records_path"])
    trajectory_ids_path = Path(result["trajectory_ids_path"])
    manifest_path = Path(result["manifest_path"])
    report_path = Path(result["report_path"])
    question_records = [
        json.loads(line)
        for line in question_records_path.read_text(encoding="utf-8").splitlines()
    ]
    trajectory_ids = [
        json.loads(line)["id"]
        for line in trajectory_ids_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert [record["id"] for record in question_records] == ["dyn1", "static_clean"]
    assert trajectory_ids == ["traj-a", "traj-b", "traj-c"]
    assert manifest["unique_trajectories"] == 3
    assert manifest["by_split"]["transfer"]["trajectory_references"] == 2
    assert "LongMemEval-V2 Trajectory Manifest" in report


def test_extract_longmemeval_v2_trajectories_streams_selected_records(tmp_path: Path) -> None:
    trajectory_ids = tmp_path / "trajectory_ids.jsonl"
    trajectory_ids.write_text(
        "\n".join([
            json.dumps({"id": "traj-c"}),
            json.dumps({"id": "traj-a"}),
            json.dumps({"id": "traj-missing"}),
        ]),
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(
        "\n".join([
            json.dumps({
                "id": "traj-a",
                "domain": "enterprise",
                "environment": "workarena",
                "goal": "Find state.",
                "answer": "must not leak",
                "eval_function": "must_not_leak",
                "states": [{"state_index": 0, "accessibility_tree": "A"}],
            }),
            json.dumps({
                "id": "traj-b",
                "domain": "web",
                "states": [{"state_index": 0, "accessibility_tree": "B"}],
            }),
            json.dumps({
                "id": "traj-c",
                "domain": "web",
                "question": "must not leak",
                "states": [{"state_index": 0, "accessibility_tree": "C"}],
            }),
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "extract"

    result = write_longmemeval_v2_extracted_trajectories(
        trajectory_ids,
        trajectories,
        output_dir,
    )
    selected = [
        json.loads(line)
        for line in Path(result["selected_trajectories_path"]).read_text(encoding="utf-8").splitlines()
    ]
    missing = [
        json.loads(line)["id"]
        for line in Path(result["missing_trajectory_ids_path"]).read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert [record["id"] for record in selected] == ["traj-a", "traj-c"]
    assert all("answer" not in record and "eval_function" not in record and "question" not in record for record in selected)
    assert missing == ["traj-missing"]
    assert manifest["requested_trajectories"] == 3
    assert manifest["matched_trajectories"] == 2
    assert manifest["missing_trajectories"] == 1
    assert manifest["records_scanned"] == 3
    assert manifest["completed_all_requested"] is False


def test_extract_longmemeval_v2_trajectories_stops_after_all_requested(tmp_path: Path) -> None:
    trajectory_ids = tmp_path / "trajectory_ids.jsonl"
    trajectory_ids.write_text(
        "\n".join([json.dumps({"id": "traj-a"}), json.dumps({"id": "traj-b"})]),
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(
        "\n".join([
            json.dumps({"id": "traj-a", "states": []}),
            json.dumps({"id": "traj-b", "states": []}),
            json.dumps({"id": "traj-c", "states": []}),
        ]),
        encoding="utf-8",
    )

    result = write_longmemeval_v2_extracted_trajectories(
        trajectory_ids,
        trajectories,
        tmp_path / "extract",
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["records_scanned"] == 2
    assert manifest["completed_all_requested"] is True
    assert manifest["missing_trajectories"] == 0


def test_validate_longmemeval_v2_prepared_split_accepts_complete_runtime_files() -> None:
    summary = validate_longmemeval_v2_prepared_split(
        [
            {"id": "q1", "split": "transfer"},
            {"id": "q2", "split": "static_clean_control"},
        ],
        questions_by_id={
            "q1": {"id": "q1", "answer": "allowed only in question source"},
            "q2": {"id": "q2"},
        },
        haystacks={
            "q1": ["traj-a", "traj-b"],
            "q2": ["traj-b", "traj-c"],
        },
        trajectory_records=[
            {"id": "traj-a", "domain": "enterprise", "states": []},
            {"id": "traj-b", "domain": "enterprise", "states": []},
            {"id": "traj-c", "domain": "web", "states": []},
        ],
    )

    assert summary["valid"] is True
    assert summary["blocking_issue_count"] == 0
    assert summary["required_trajectories"] == 3
    assert summary["selected_trajectories"] == 3
    assert summary["label_leak_records"] == []


def test_validate_longmemeval_v2_prepared_split_flags_missing_and_leaked_records() -> None:
    summary = validate_longmemeval_v2_prepared_split(
        [
            {"id": "q1", "split": "transfer"},
            {"id": "q_missing", "split": "transfer"},
            {"id": "q_no_haystack", "split": "transfer"},
        ],
        questions_by_id={
            "q1": {"id": "q1"},
            "q_no_haystack": {"id": "q_no_haystack"},
        },
        haystacks={
            "q1": ["traj-a", "traj-b"],
            "q_missing": ["traj-c"],
        },
        trajectory_records=[
            {"id": "traj-a", "states": [], "answer": "leak"},
            {"id": "traj-a", "states": []},
            {"id": "traj-extra", "states": [], "unexpected": True},
        ],
    )

    assert summary["valid"] is False
    assert summary["missing_question_ids"] == ["q_missing"]
    assert summary["missing_haystack_question_ids"] == ["q_no_haystack"]
    assert summary["missing_trajectory_ids"] == ["traj-b", "traj-c"]
    assert summary["duplicate_trajectory_ids"] == ["traj-a"]
    assert summary["label_leak_records"] == [{"id": "traj-a", "fields": ["answer"]}]
    assert summary["extra_trajectory_ids"] == ["traj-extra"]
    assert summary["extra_field_records"] == [{"id": "traj-extra", "fields": ["unexpected"]}]
    assert summary["blocking_issue_count"] == 6


def test_write_longmemeval_v2_prepared_split_validation_outputs_report(tmp_path: Path) -> None:
    split_records = tmp_path / "split.jsonl"
    split_records.write_text(json.dumps({"id": "q1", "split": "transfer"}) + "\n", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": "q1", "answer": "not copied"}) + "\n", encoding="utf-8")
    haystack = tmp_path / "haystack.json"
    haystack.write_text(json.dumps({"q1": ["traj-a"]}), encoding="utf-8")
    trajectories = tmp_path / "selected.jsonl"
    trajectories.write_text(json.dumps({"id": "traj-a", "states": []}) + "\n", encoding="utf-8")

    result = write_longmemeval_v2_prepared_split_validation(
        split_records,
        questions,
        haystack,
        trajectories,
        tmp_path / "validation",
    )
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    report = Path(result["report_path"]).read_text(encoding="utf-8")

    assert summary["valid"] is True
    assert summary["required_trajectories"] == 1
    assert "Prepared Split Validation" in report


def test_longmemeval_v2_prepared_state_evidence_audit_matches_query_slots() -> None:
    records = list(longmemeval_v2_prepared_state_evidence_records(
        [
            {
                **_audit_record("q_dynamic", "dynamic-environment", state_slots=["runtime.*.status"]),
                "split": "transfer",
                "selection_group": "dynamic-environment",
            },
            {
                **_audit_record("q_static", "static-environment", type_candidate=False),
                "split": "static_clean_control",
                "selection_group": "static_no_state_slot_signal",
            },
        ],
        haystacks={
            "q_dynamic": ["traj-dynamic", "traj-missing"],
            "q_static": ["traj-static"],
        },
        trajectory_records=[
            {
                "id": "traj-dynamic",
                "domain": "enterprise",
                "environment": "workarena",
                "goal": "Inspect the runner.",
                "states": [
                    {
                        "state_index": 0,
                        "accessibility_tree": "The staging build runner status is offline.",
                    }
                ],
            },
            {
                "id": "traj-static",
                "domain": "web",
                "environment": "webarena-cms",
                "states": [{"state_index": 0, "accessibility_tree": "Button label: Publish."}],
            },
        ],
    ))
    summary = summarize_longmemeval_v2_prepared_state_evidence(records)
    by_id = {record["id"]: record for record in records}

    assert by_id["q_dynamic"]["state_available"] is True
    assert by_id["q_dynamic"]["missing_trajectory_ids"] == ["traj-missing"]
    assert by_id["q_dynamic"]["matching_state_evidence_candidate_count"] == 1
    assert by_id["q_dynamic"]["state_evidence_candidates"][0]["state_slot"] == "runtime.staging_build_runner_status.status"
    assert by_id["q_dynamic"]["state_evidence_candidates"][0]["state_value"] == "offline"
    assert by_id["q_dynamic"]["expected_state_families"] == ["runtime"]
    assert by_id["q_dynamic"]["matching_state_evidence_families"] == ["runtime"]
    assert by_id["q_static"]["state_available"] is False
    assert by_id["q_static"]["expected_state_slots"] == []
    assert all("answer" not in candidate for record in records for candidate in record["state_evidence_candidates"])
    assert summary["total_questions"] == 2
    assert summary["with_expected_state_slots"] == 1
    assert summary["with_matching_state_evidence"] == 1
    assert summary["missing_trajectory_total"] == 1
    assert summary["by_split"]["transfer"]["with_matching_state_evidence"] == 1
    assert summary["by_state_slot"]["runtime.*.status"]["matching_state_evidence_candidate_total"] == 1
    assert summary["by_state_family"]["runtime"]["matching_state_evidence_candidate_total"] == 1


def test_write_longmemeval_v2_prepared_state_evidence_audit_outputs_artifacts(tmp_path: Path) -> None:
    split_records = tmp_path / "split.records.jsonl"
    split_records.write_text(
        json.dumps({
            **_audit_record("q_runtime", "dynamic-environment", state_slots=["runtime.*.status"]),
            "split": "transfer",
            "selection_group": "dynamic-environment",
        }) + "\n",
        encoding="utf-8",
    )
    haystack = tmp_path / "haystack.json"
    haystack.write_text(json.dumps({"q_runtime": ["traj-runtime"]}), encoding="utf-8")
    trajectories = tmp_path / "selected_trajectories.jsonl"
    trajectories.write_text(
        json.dumps({
            "id": "traj-runtime",
            "domain": "enterprise",
            "environment": "workarena",
            "states": [
                {
                    "state_index": 3,
                    "accessibility_tree": "Staging build runner status is online.",
                }
            ],
        }) + "\n",
        encoding="utf-8",
    )

    result = write_longmemeval_v2_prepared_state_evidence_audit(
        split_records,
        haystack,
        trajectories,
        tmp_path / "state-evidence",
    )
    records = [
        json.loads(line)
        for line in Path(result["records_path"]).read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    report = Path(result["report_path"]).read_text(encoding="utf-8")

    assert records[0]["state_available"] is True
    assert summary["with_matching_state_evidence"] == 1
    assert summary["by_state_family"]["runtime"]["with_matching_state_evidence"] == 1
    assert "Prepared State Evidence Audit" in report
    assert "State Families" in report
    assert "| runtime | 1 | 1 | 0 | 1 |" in report


def test_state_slot_family_groups_cross_benchmark_slots() -> None:
    assert state_slot_family("runtime.*.status") == "runtime"
    assert state_slot_family("runtime.staging_build_runner_status.status") == "runtime"
    assert state_slot_family("resource.database.status") == "resource"
    assert state_slot_family("workflow.checkout") == "workflow"
    assert state_slot_family("environment.checkout_page.gotcha") == "environment"
    assert state_slot_family("tool.search.last_output") == "tool_output"
    assert state_slot_family("location") == "location"
    assert state_slot_family("local.gym") == "location"
    assert state_slot_family("organization.employer") == "employment"


def _audit_record(
    question_id: str,
    question_type: str,
    *,
    domain: str = "enterprise",
    type_candidate: bool = True,
    state_slots: list[str] | None = None,
    image_required: bool = False,
    haystack_size: int | None = 100,
) -> dict[str, object]:
    state_slots = list(state_slots or [])
    candidate_reasons = []
    if type_candidate:
        candidate_reasons.append("question_type")
    if state_slots and not question_type.startswith("static-environment"):
        candidate_reasons.append("query_state_slot")
    return {
        "id": question_id,
        "domain": domain,
        "environment": "workarena",
        "question_type": question_type,
        "abstention": question_type.endswith("-abs"),
        "image_required": image_required,
        "eval_function_family": "norm_phrase_set_match",
        "inferred_state_slots": state_slots,
        "type_transfer_candidate": type_candidate,
        "query_state_slot_candidate": bool(state_slots),
        "state_transfer_candidate": bool(candidate_reasons),
        "candidate_reasons": candidate_reasons,
        "haystack_size": haystack_size,
    }
