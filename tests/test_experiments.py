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
