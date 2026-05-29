from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adamem.baselines import BaselineSpec


SCHEMA_VERSION = "adamem.experiment.v1"


@dataclass(slots=True)
class ExperimentRecord:
    run_name: str
    run_type: str
    dataset: str | None = None
    split_or_case_limit: str | None = None
    baseline_names: list[str] = field(default_factory=list)
    baseline_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: Any = None
    diagnostics: Any = None
    prompts: dict[str, str] = field(default_factory=dict)
    raw_outputs: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=lambda: list(sys.argv))
    commit: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = field(default_factory=lambda: uuid4().hex)
    schema_version: str = SCHEMA_VERSION


def experiment_record(
    *,
    run_name: str,
    run_type: str,
    dataset: str | Path | None = None,
    split_or_case_limit: str | None = None,
    baselines: dict[str, BaselineSpec] | None = None,
    results: Any = None,
    diagnostics: Any = None,
    prompts: dict[str, str] | None = None,
    raw_outputs: list[dict[str, Any]] | None = None,
    notes: dict[str, Any] | None = None,
    command: list[str] | None = None,
) -> ExperimentRecord:
    baselines = baselines or {}
    return ExperimentRecord(
        run_name=run_name,
        run_type=run_type,
        dataset=str(dataset) if dataset is not None else None,
        split_or_case_limit=split_or_case_limit,
        baseline_names=list(baselines),
        baseline_configs={name: spec.config_dict() for name, spec in baselines.items()},
        results=results,
        diagnostics=diagnostics,
        prompts=prompts or {},
        raw_outputs=raw_outputs or [],
        notes=notes or {},
        command=command or list(sys.argv),
        commit=current_git_commit(),
    )


def write_experiment_record(path: str | Path, record: ExperimentRecord) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(record), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(output)
    return output


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
