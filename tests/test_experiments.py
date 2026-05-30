from __future__ import annotations

import json

import pytest

from adamem.baselines import (
    baseline_registry,
    baseline_report,
    baseline_reproduction_packet_template,
    baseline_reproduction_plan,
    baseline_reproduction_plan_markdown,
    load_baseline_reproduction_packet,
    main as baseline_main,
    select_baselines,
    verify_baseline_reproduction_packet,
    write_baseline_reproduction_packet_template,
    write_baseline_reproduction_plan,
)
from adamem.bench import default_ablation_configs
from adamem.experiments import experiment_record, write_experiment_record


def test_baseline_registry_matches_default_ablation_configs() -> None:
    specs = baseline_registry()
    configs = default_ablation_configs()

    assert list(specs) == [
        "semantic_only",
        "semantic_importance",
        "semantic_temporal",
        "semantic_graph",
        "a_mem_evolution",
        "zep_temporal_kg",
        "mem0_extraction",
        "trajectory_step_readout",
        "delta_graph",
        "delta_soft",
        "delta_propagation",
        "delta_full",
        "full",
        "state_memory",
        "semantic_state_readout",
        "semantic_state_propagation",
        "semantic_state_adjudication",
        "semantic_state_adjudication_trace",
        "semantic_state_premise_correction",
        "semantic_llm_state_adjudication",
        "semantic_llm_state_premise_correction",
        "semantic_state_propagation_adjudication",
        "state_readout",
        "state_propagation",
    ]
    assert list(configs) == [
        name
        for name, spec in specs.items()
        if spec.config.state_extractor_name != "llm_json"
    ]
    assert configs["semantic_only"].use_graph is False
    assert configs["a_mem_evolution"].use_memory_evolution is True
    assert configs["zep_temporal_kg"].use_temporal_kg_memory is True
    assert configs["mem0_extraction"].use_salient_memory_only is True
    assert specs["a_mem_evolution"].source_name == "A-MEM"
    assert specs["a_mem_evolution"].implementation_status == "api_free_approximation"
    assert specs["a_mem_evolution"].reproduction_target_url == "https://github.com/WujiangXu/A-mem"
    assert specs["zep_temporal_kg"].source_url == "https://arxiv.org/abs/2501.13956"
    assert specs["zep_temporal_kg"].reproduction_target_url == "https://github.com/getzep/graphiti"
    assert specs["mem0_extraction"].provenance_dict()["category"] == "mainstream_approximation"
    assert specs["mem0_extraction"].provenance_dict()["source_name"] == "Mem0"
    assert specs["mem0_extraction"].provenance_dict()["reproduction_target_url"] == (
        "https://github.com/mem0ai/mem0"
    )
    assert configs["trajectory_step_readout"].use_trajectory_step_readout is True
    assert configs["full"].use_graph is True
    assert configs["semantic_state_readout"].use_graph is False
    assert configs["semantic_state_readout"].use_state_readout is True
    assert configs["semantic_state_adjudication"].use_state_source_adjudication is True
    assert configs["semantic_state_adjudication_trace"].use_state_adjudication_trace is True
    assert configs["semantic_state_premise_correction"].use_state_premise_correction is True
    assert specs["semantic_llm_state_adjudication"].config.state_extractor_name == "llm_json"
    assert specs["semantic_llm_state_premise_correction"].config.use_state_premise_correction is True
    assert configs["state_readout"].use_state_memory is True
    assert configs["state_readout"].use_state_readout is True
    assert configs["state_readout"].state_extractor_name == "deterministic"

    report = baseline_report(specs)
    assert "semantic_only" in report
    assert "adamem_full" in report
    assert "implementation" in report
    assert "[A-MEM](https://arxiv.org/abs/2502.12110)" in report


def test_select_baselines_preserves_requested_order() -> None:
    selected = select_baselines(["state_readout", "semantic_only"])

    assert list(selected) == ["state_readout", "semantic_only"]
    assert selected["state_readout"].config.use_state_readout is True


def test_select_baselines_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        select_baselines(["not_a_baseline"])


def test_baseline_reproduction_plan_records_official_baseline_requirements(tmp_path) -> None:
    plan = baseline_reproduction_plan()
    by_name = {target["baseline"]: target for target in plan["targets"]}
    markdown = baseline_reproduction_plan_markdown(plan)
    artifacts = write_baseline_reproduction_plan(tmp_path / "baseline-plan")
    artifact_plan = json.loads(
        (tmp_path / "baseline-plan" / "baseline_reproduction_plan.json").read_text(
            encoding="utf-8"
        )
    )

    assert plan["schema_version"] == "adamem.baseline_reproduction_plan.v1"
    assert plan["ready_for_sota_claims"] is False
    assert set(by_name) == {"a_mem_evolution", "zep_temporal_kg", "mem0_extraction"}
    assert by_name["a_mem_evolution"]["current_role"] == "api_free_local_control_not_sota_baseline"
    assert by_name["a_mem_evolution"]["required_status_after_run"] == [
        "official_reproduction",
        "faithful_reimplementation",
    ]
    assert "external_repo_commit" in by_name["zep_temporal_kg"]["required_evidence"]
    assert by_name["mem0_extraction"]["reproduction_target_url"] == "https://github.com/mem0ai/mem0"
    assert "Do not use it as strong-baseline or SOTA evidence" in by_name["a_mem_evolution"]["claim_boundary"]
    assert "[A-MEM](https://arxiv.org/abs/2502.12110)" in markdown
    assert "`official_reproduction`, `faithful_reimplementation`" in markdown
    assert artifacts == {
        "json_path": str(tmp_path / "baseline-plan" / "baseline_reproduction_plan.json"),
        "markdown_path": str(tmp_path / "baseline-plan" / "baseline_reproduction_plan.md"),
    }
    assert artifact_plan["target_count"] == 3


def test_baseline_reproduction_packet_template_and_validator_require_real_evidence(
    tmp_path,
) -> None:
    packet = baseline_reproduction_packet_template("a_mem_evolution")

    report = verify_baseline_reproduction_packet(packet, packet_path=tmp_path / "packet.json")

    assert packet["schema_version"] == "adamem.baseline_reproduction_packet.v1"
    assert packet["baseline"] == "a_mem_evolution"
    assert packet["evidence"]["external_repo_url"] == "https://github.com/WujiangXu/A-mem"
    assert report["ready_for_sota_baseline_claim"] is False
    assert report["missing_evidence"] == [
        "external_repo_commit",
        "adapter_or_command",
        "dataset_split_and_question_ids",
        "model_provider_model_and_sampling_settings",
        "prompt_or_memory_policy_if_applicable",
        "raw_case_records_path",
        "metric_mapping_to_adamem_outputs",
        "license_and_dependency_notes",
    ]
    assert "required_evidence_incomplete" in report["blockers"]
    assert "raw_case_records_missing" in report["blockers"]


def test_baseline_reproduction_packet_validator_accepts_complete_packet(tmp_path) -> None:
    records = tmp_path / "amem.records.jsonl"
    records.write_text("{\"case_id\":\"q1\",\"baseline\":\"a_mem_evolution\"}\n", encoding="utf-8")
    packet = baseline_reproduction_packet_template("a_mem_evolution")
    packet["implementation_status"] = "official_reproduction"
    packet["evidence"].update({
        "external_repo_commit": "abc1234",
        "adapter_or_command": "python run_amem.py --split stale_ids.txt",
        "dataset_split_and_question_ids": "results/stale_split/question_ids.jsonl",
        "model_provider_model_and_sampling_settings": "openai:gpt-4o-mini temperature=0 seed=0",
        "prompt_or_memory_policy_if_applicable": "official A-MEM memory update prompt from commit abc1234",
        "raw_case_records_path": str(records),
        "metric_mapping_to_adamem_outputs": "Maps answer text and retrieved context to AdaMem stale_llm_judge records.",
        "license_and_dependency_notes": "MIT-compatible local reproduction environment recorded.",
    })
    packet["baseline_provenance_update"].update({
        "implementation_status": "official_reproduction",
        "reproduction_note": "Official A-MEM code path run on the same STALE split.",
    })

    report = verify_baseline_reproduction_packet(packet, packet_path=tmp_path / "packet.json")

    assert report["ready_for_sota_baseline_claim"] is True
    assert report["blockers"] == []
    assert report["missing_evidence"] == []
    assert report["baseline_provenance_update"]["implementation_status"] == (
        "official_reproduction"
    )


def test_write_and_load_baseline_reproduction_packet_template(tmp_path) -> None:
    packet_path = tmp_path / "amem_packet.json"

    artifacts = write_baseline_reproduction_packet_template("a_mem_evolution", packet_path)
    packet = load_baseline_reproduction_packet(packet_path)

    assert artifacts == {"packet_path": str(packet_path)}
    assert packet["baseline"] == "a_mem_evolution"
    assert packet["required_evidence"]


def test_baseline_reproduction_packet_cli_template_and_verify(tmp_path, capsys) -> None:
    packet_path = tmp_path / "amem_packet.json"

    baseline_main([
        "--packet-template",
        "a_mem_evolution",
        "--packet-output",
        str(packet_path),
        "--json",
    ])

    template_result = json.loads(capsys.readouterr().out)
    assert template_result == {"packet_path": str(packet_path)}
    with pytest.raises(SystemExit) as exc:
        baseline_main(["--verify-packet", str(packet_path), "--json"])

    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready_for_sota_baseline_claim"] is False
    assert "required_evidence_incomplete" in report["blockers"]


def test_experiment_record_writes_reproducible_json(tmp_path) -> None:
    specs = {
        "semantic_only": baseline_registry()["semantic_only"],
        "a_mem_evolution": baseline_registry()["a_mem_evolution"],
    }
    record = experiment_record(
        run_name="diagnostic-smoke",
        run_type="stale_retrieval_diagnostics",
        dataset="benchmarks/stale_mini.jsonl",
        split_or_case_limit="1",
        baselines=specs,
        diagnostics=[{"current_recall_rate": 0.0}],
        notes={"answer_model_required": False},
        command=["adamem-eval", "--stale-diagnostics"],
    )

    output = write_experiment_record(tmp_path / "run.json", record)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["schema_version"] == "adamem.experiment.v1"
    assert data["run_name"] == "diagnostic-smoke"
    assert data["run_type"] == "stale_retrieval_diagnostics"
    assert data["dataset"] == "benchmarks/stale_mini.jsonl"
    assert data["baseline_names"] == ["semantic_only", "a_mem_evolution"]
    assert data["baseline_configs"]["semantic_only"]["use_graph"] is False
    assert data["baseline_provenance"]["semantic_only"]["source_name"] == "AdaMem"
    assert data["baseline_provenance"]["a_mem_evolution"] == {
        "category": "mainstream_approximation",
        "source_name": "A-MEM",
        "source_url": "https://arxiv.org/abs/2502.12110",
        "implementation_status": "api_free_approximation",
        "reproduction_note": (
            "Approximates memory evolution locally; replace with or validate against "
            "the official implementation before SOTA-style claims."
        ),
        "reproduction_target_name": "A-MEM reproduction code",
        "reproduction_target_url": "https://github.com/WujiangXu/A-mem",
        "reproduction_target_note": (
            "Use the paper reproduction repository for official/faithful LoCoMo-style runs."
        ),
    }
    assert data["diagnostics"] == [{"current_recall_rate": 0.0}]
    assert data["command"] == ["adamem-eval", "--stale-diagnostics"]
