from __future__ import annotations

import json
from pathlib import Path

from adamem.stale_pipeline import main, run_stale_diagnostic_pipeline


def test_stale_diagnostic_pipeline_writes_reproducible_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "stale_raw.json"
    source.write_text(json.dumps([_toy_stale_instance()]), encoding="utf-8")

    manifest = run_stale_diagnostic_pipeline(
        source,
        tmp_path / "pipeline",
        run_name="toy_stale",
        baselines=["semantic_state_adjudication", "semantic_state_premise_correction"],
        top_k=4,
    )

    artifacts = manifest["artifacts"]
    assert manifest["converted_cases"] == 1
    assert manifest["diagnostic_cases"] == 1
    assert Path(artifacts["converted_dataset"]).exists()
    assert Path(artifacts["experiment"]).exists()
    assert Path(artifacts["diagnostic_cases"]).exists()
    assert Path(artifacts["diagnostic_report"]).exists()
    assert Path(artifacts["paper_tables_markdown"]).exists()
    assert Path(artifacts["paper_tables_json"]).exists()
    assert Path(artifacts["report_bundle_manifest"]).exists()
    experiment = json.loads(Path(artifacts["experiment"]).read_text(encoding="utf-8"))
    assert experiment["run_type"] == "stale_retrieval_diagnostics"
    assert experiment["notes"]["raw_stale_input"] == str(source)
    assert experiment["notes"]["ground_truth_runtime_use"] == "forbidden"
    tables = Path(artifacts["paper_tables_markdown"]).read_text(encoding="utf-8")
    assert "premise correction hit" in tables


def test_stale_pipeline_cli_writes_manifest_json(tmp_path: Path, capsys) -> None:
    source = tmp_path / "stale_raw.json"
    source.write_text(json.dumps([_toy_stale_instance()]), encoding="utf-8")
    output_dir = tmp_path / "cli"

    main([
        str(source),
        "--output-dir",
        str(output_dir),
        "--run-name",
        "cli_stale",
        "--baselines",
        "semantic_state_premise_correction",
        "--top-k",
        "4",
        "--json",
    ])

    printed = json.loads(capsys.readouterr().out)
    assert printed["run_name"] == "cli_stale"
    assert (output_dir / "cli_stale.manifest.json").exists()


def test_stale_pipeline_accepts_converted_jsonl_input(tmp_path: Path) -> None:
    converted = tmp_path / "stale.adamem.jsonl"
    converted.write_text(json.dumps(_toy_adamem_case()) + "\n", encoding="utf-8")

    manifest = run_stale_diagnostic_pipeline(
        converted,
        tmp_path / "converted-pipeline",
        run_name="converted_stale",
        baselines=["semantic_state_premise_correction"],
        input_format="adamem-jsonl",
    )

    assert manifest["input_format"] == "adamem-jsonl"
    assert manifest["converted_cases"] == 1
    assert Path(manifest["artifacts"]["converted_dataset"]).exists()


def _toy_stale_instance() -> dict:
    return {
        "uid": "stale-toy-1",
        "type": "T1",
        "M_old": "User lives in Seattle.",
        "M_new": "User lives in Boston.",
        "explanation": "User moved from Seattle to Boston.",
        "probing_queries": {
            "dim1_query": "Do I still live in Seattle?",
            "dim2_query": "Since I'm in Seattle, what's a good local park?",
            "dim3_query": "Recommend a coffee shop near me.",
        },
        "relevant_session_index": [0, 2],
        "timestamps": [
            "2026-01-01T10:00:00+00:00",
            "2026-02-01T10:00:00+00:00",
            "2026-03-01T10:00:00+00:00",
        ],
        "haystack_session": [
            [
                {"role": "user", "content": "I just moved into a place in Seattle."},
                {"role": "assistant", "content": "Welcome to Seattle!"},
            ],
            [
                {"role": "user", "content": "Weather is rainy as usual."},
                {"role": "assistant", "content": "Typical PNW weather."},
            ],
            [
                {"role": "user", "content": "I relocated to Boston for a new job."},
                {"role": "assistant", "content": "Congrats on the move to Boston!"},
            ],
        ],
    }


def _toy_adamem_case() -> dict:
    return {
        "id": "converted-stale-toy",
        "observations": [
            {
                "label": "old_location",
                "content": "[2026-01-01] user: I just moved into a place in Seattle.",
                "metadata": {"memory_key": "old_location"},
            },
            {
                "label": "new_location",
                "content": "[2026-03-01] user: I relocated to Boston for a new job.",
                "metadata": {"memory_key": "new_location"},
            },
        ],
        "queries": [
            {
                "id": "dim2",
                "query": "Since I'm in Seattle, what's a good local park?",
                "expected_substrings": [],
                "top_k": 4,
                "metadata": {
                    "stale_uid": "converted-stale-toy",
                    "stale_type": "T1",
                    "stale_dim": 2,
                    "M_old": "User lives in Seattle.",
                    "M_new": "User lives in Boston.",
                },
            }
        ],
    }
