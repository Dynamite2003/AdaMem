from __future__ import annotations

import json
from pathlib import Path

import pytest

from adamem.answer_eval import SubstringAnswerScorer
from adamem.llm import MockLLMClient
from adamem.pilot import (
    copy_jsonl_prefix,
    run_ama_public_pilot,
    run_longmemeval_v2_prepared_pilot,
)


def test_copy_jsonl_prefix_validates_and_limits_records(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join([
            json.dumps({"episode_id": 1}),
            json.dumps({"episode_id": 2}),
            json.dumps({"episode_id": 3}),
        ]),
        encoding="utf-8",
    )
    output = tmp_path / "subset.jsonl"

    count = copy_jsonl_prefix(source, output, limit=2)

    assert count == 2
    assert [json.loads(line)["episode_id"] for line in output.read_text(encoding="utf-8").splitlines()] == [1, 2]


def test_ama_public_pilot_runs_from_local_source(tmp_path: Path) -> None:
    source = tmp_path / "ama.jsonl"
    source.write_text(
        json.dumps({
            "episode_id": "pilot-1",
            "domain": "Game",
            "task_type": "toy",
            "trajectory": [
                {
                    "turn_idx": 17,
                    "action": "right",
                    "observation": "Active rules:\nwall is stop\nbaba is you\n\nObjects on the map:\nwall 1 step to the right",
                },
                {
                    "turn_idx": 18,
                    "action": "right",
                    "observation": "Active rules:\nwall is stop\nbaba is you\n\nObjects on the map:\nwall 1 step to the right",
                },
            ],
            "qa_pairs": [
                {
                    "question_uuid": "blocked",
                    "type": "C",
                    "question": "In steps 17 through 18, the agent repeatedly moves right but nothing changes. Why?",
                    "answer": "The right action is blocked by a wall because wall is stop.",
                }
            ],
        }),
        encoding="utf-8",
    )

    summary = run_ama_public_pilot(
        output_dir=tmp_path / "pilot",
        limit=1,
        source=source,
        baselines=["semantic_only", "trajectory_step_readout"],
        top_k=4,
    )

    assert summary["source_records"] == 1
    assert summary["answer_cases"] == 1
    assert summary["evidence_cases"] == 1
    assert summary["timings"]["total_seconds"] >= 0
    assert Path(summary["answer"]["report_path"]).exists()
    assert Path(summary["evidence"]["experiment_path"]).exists()
    assert Path(summary["answer"]["records_path"]).name == "ama_public_1.answer.records.jsonl"
    assert Path(summary["evidence"]["records_path"]).name == "ama_public_1.evidence.records.jsonl"
    answerability = summary["answer"]["summary"]["answerability"]
    assert answerability["trajectory_step_readout"]["basis_answer_keyword_matched_records"] == 1
    assert summary["evidence"]["summary"]["evidence_support"]["trajectory_step_readout"]["evidence_matched_records"] == 1


def test_ama_public_pilot_can_skip_evidence_mode(tmp_path: Path) -> None:
    source = tmp_path / "ama.jsonl"
    source.write_text(
        json.dumps({
            "episode_id": "pilot-answer-only",
            "trajectory": [{"turn_idx": 1, "action": "left", "observation": "The agent moved left."}],
            "qa_pairs": [{
                "question_uuid": "q1",
                "type": "A",
                "question": "What happened at Step 1?",
                "answer": "The agent moved left.",
            }],
        }),
        encoding="utf-8",
    )

    summary = run_ama_public_pilot(
        output_dir=tmp_path / "pilot-answer-only",
        limit=1,
        source=source,
        baselines=["trajectory_step_readout"],
        include_evidence_mode=False,
    )
    experiment = json.loads(Path(summary["answer"]["experiment_path"]).read_text(encoding="utf-8"))

    assert summary["evidence"] is None
    assert summary["evidence_dataset"] is None
    assert summary["evidence_cases"] == 0
    assert experiment["raw_outputs"] == []
    assert experiment["notes"]["raw_outputs_embedded"] is False
    assert Path(experiment["notes"]["records_path"]).exists()


def test_ama_public_pilot_can_run_answer_generation_stage(tmp_path: Path) -> None:
    source = tmp_path / "ama.jsonl"
    source.write_text(
        json.dumps({
            "episode_id": "pilot-generation",
            "trajectory": [{"turn_idx": 1, "action": "left", "observation": "The agent moved left."}],
            "qa_pairs": [{
                "question_uuid": "q1",
                "type": "A",
                "question": "What happened at Step 1?",
                "answer": "The agent moved left.",
            }],
        }),
        encoding="utf-8",
    )

    summary = run_ama_public_pilot(
        output_dir=tmp_path / "pilot-generation",
        limit=1,
        source=source,
        baselines=["trajectory_step_readout"],
        include_evidence_mode=False,
        include_answer_generation=True,
        answer_client=MockLLMClient("The agent moved left."),
        answer_scorer=SubstringAnswerScorer(),
        answer_generation_notes={"answer_provider": "mock"},
    )

    generation = summary["answer_generation"]
    assert generation is not None
    assert generation["summary"]["by_baseline"]["trajectory_step_readout"]["correct"] == 1
    assert generation["summary"]["by_baseline"]["trajectory_step_readout"]["total"] == 1
    assert generation["summary"]["by_metadata"]["question_type"]["A"]["trajectory_step_readout"]["correct"] == 1
    assert Path(summary["answer"]["records_path"]).name == "ama_public_1.answer.records.jsonl"
    assert Path(generation["records_path"]).name == "ama_public_1.generation.records.jsonl"
    assert Path(generation["records_path"]).exists()
    experiment = json.loads(Path(generation["experiment_path"]).read_text(encoding="utf-8"))
    assert experiment["run_type"] == "ama_public_answer_generation_pilot"
    assert experiment["notes"]["answer_provider"] == "mock"
    assert experiment["notes"]["ground_truth_runtime_use"] == "forbidden"


def test_longmemeval_v2_prepared_pilot_runs_validation_conversion_and_retrieval(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join([
            json.dumps({
                "id": "q_dynamic",
                "domain": "enterprise",
                "environment": "workarena",
                "question_type": "dynamic-environment",
                "question": "Is the staging build runner online?",
                "answer": "online",
                "eval_function": "norm_phrase_set_match",
            }),
            json.dumps({
                "id": "q_static",
                "domain": "web",
                "environment": "webarena-cms",
                "question_type": "static-environment",
                "question": "What is the button label?",
                "answer": "Publish",
                "eval_function": "norm_phrase_set_match",
            }),
        ]),
        encoding="utf-8",
    )
    haystack = tmp_path / "haystack.json"
    haystack.write_text(
        json.dumps({
            "q_dynamic": ["traj_dynamic"],
            "q_static": ["traj_static"],
        }),
        encoding="utf-8",
    )
    trajectories = tmp_path / "selected_trajectories.jsonl"
    trajectories.write_text(
        "\n".join([
            json.dumps({
                "id": "traj_dynamic",
                "domain": "enterprise",
                "environment": "workarena",
                "goal": "Inspect runner state.",
                "states": [
                    {
                        "state_index": 0,
                        "accessibility_tree": "The staging build runner status is online.",
                    }
                ],
            }),
            json.dumps({
                "id": "traj_static",
                "domain": "web",
                "environment": "webarena-cms",
                "goal": "Inspect CMS.",
                "states": [{"state_index": 0, "accessibility_tree": "The button label is Publish."}],
            }),
        ]),
        encoding="utf-8",
    )
    split_records = tmp_path / "split.records.jsonl"
    split_records.write_text(
        "\n".join([
            json.dumps({
                "id": "q_dynamic",
                "split": "transfer",
                "selection_group": "dynamic-environment",
                "question_type": "dynamic-environment",
                "inferred_state_slots": ["runtime.*.status"],
            }),
            json.dumps({
                "id": "q_static",
                "split": "static_clean_control",
                "selection_group": "static_no_state_slot_signal",
                "question_type": "static-environment",
            }),
        ]),
        encoding="utf-8",
    )

    summary = run_longmemeval_v2_prepared_pilot(
        output_dir=tmp_path / "pilot",
        questions=questions,
        trajectories=trajectories,
        haystack=haystack,
        split_records=split_records,
        baselines=["semantic_only", "semantic_state_readout"],
        top_k=2,
    )
    experiment = json.loads(Path(summary["answer"]["experiment_path"]).read_text(encoding="utf-8"))

    assert summary["cases"] == 2
    assert summary["validation"]["summary"]["valid"] is True
    assert summary["state_evidence"]["summary"]["with_expected_state_slots"] == 1
    assert summary["state_evidence"]["summary"]["with_matching_state_evidence"] == 1
    assert Path(summary["dataset"]).exists()
    assert Path(summary["answer"]["records_path"]).name == "longmemeval_v2_prepared.answer.records.jsonl"
    assert experiment["run_type"] == "longmemeval_v2_prepared_answer_support_pilot"
    assert experiment["notes"]["metric_boundary"] == "retrieval answer-string support, not final generated answer accuracy"
    assert experiment["notes"]["validation_summary_path"] == summary["validation"]["summary_path"]
    assert experiment["notes"]["state_evidence_summary_path"] == summary["state_evidence"]["summary_path"]
    assert summary["answer"]["summary"]["by_baseline"]["semantic_only"]["total"] == 2


def test_longmemeval_v2_prepared_pilot_stops_on_validation_failure(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "id": "q_leaky",
            "domain": "enterprise",
            "environment": "workarena",
            "question_type": "dynamic-environment",
            "question": "What is the staging build runner status?",
            "answer": "online",
        }),
        encoding="utf-8",
    )
    haystack = tmp_path / "haystack.json"
    haystack.write_text(json.dumps({"q_leaky": ["traj_leaky"]}), encoding="utf-8")
    trajectories = tmp_path / "selected_trajectories.jsonl"
    trajectories.write_text(
        json.dumps({
            "id": "traj_leaky",
            "domain": "enterprise",
            "environment": "workarena",
            "answer": "online",
            "states": [{"state_index": 0, "accessibility_tree": "Runner status is online."}],
        }),
        encoding="utf-8",
    )
    split_records = tmp_path / "split.records.jsonl"
    split_records.write_text(
        json.dumps({
            "id": "q_leaky",
            "split": "transfer",
            "selection_group": "dynamic-environment",
            "question_type": "dynamic-environment",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prepared split validation failed"):
        run_longmemeval_v2_prepared_pilot(
            output_dir=tmp_path / "pilot",
            questions=questions,
            trajectories=trajectories,
            haystack=haystack,
            split_records=split_records,
            baselines=["semantic_only"],
        )
