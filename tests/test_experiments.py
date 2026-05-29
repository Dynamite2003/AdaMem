from __future__ import annotations

import json

import pytest

from adamem.baselines import baseline_registry, baseline_report, select_baselines
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
        "semantic_state_premise_correction",
        "semantic_state_propagation_adjudication",
        "state_readout",
        "state_propagation",
    ]
    assert list(configs) == list(specs)
    assert configs["semantic_only"].use_graph is False
    assert configs["a_mem_evolution"].use_memory_evolution is True
    assert configs["zep_temporal_kg"].use_temporal_kg_memory is True
    assert configs["mem0_extraction"].use_salient_memory_only is True
    assert configs["trajectory_step_readout"].use_trajectory_step_readout is True
    assert configs["full"].use_graph is True
    assert configs["semantic_state_readout"].use_graph is False
    assert configs["semantic_state_readout"].use_state_readout is True
    assert configs["semantic_state_adjudication"].use_state_source_adjudication is True
    assert configs["semantic_state_premise_correction"].use_state_premise_correction is True
    assert configs["state_readout"].use_state_memory is True
    assert configs["state_readout"].use_state_readout is True
    assert configs["state_readout"].state_extractor_name == "deterministic"

    report = baseline_report(specs)
    assert "semantic_only" in report
    assert "adamem_full" in report


def test_select_baselines_preserves_requested_order() -> None:
    selected = select_baselines(["state_readout", "semantic_only"])

    assert list(selected) == ["state_readout", "semantic_only"]
    assert selected["state_readout"].config.use_state_readout is True


def test_select_baselines_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        select_baselines(["not_a_baseline"])


def test_experiment_record_writes_reproducible_json(tmp_path) -> None:
    specs = {"semantic_only": baseline_registry()["semantic_only"]}
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
    assert data["baseline_names"] == ["semantic_only"]
    assert data["baseline_configs"]["semantic_only"]["use_graph"] is False
    assert data["diagnostics"] == [{"current_recall_rate": 0.0}]
    assert data["command"] == ["adamem-eval", "--stale-diagnostics"]
