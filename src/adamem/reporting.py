from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from adamem.claims import audit_experiment, claim_audit_markdown
from adamem.compare import paired_comparison_markdown, paired_comparison_summary
from adamem.tables import load_benchmark_records, paper_table_markdown, paper_table_summary


def write_experiment_bundle(
    experiment_path: str | Path,
    output_dir: str | Path,
    *,
    group_fields: Iterable[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    experiment = Path(experiment_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = experiment.stem
    group_fields = tuple(group_fields or ())

    audit = audit_experiment(experiment)
    audit_md = output / f"{stem}.claim_audit.md"
    audit_json = output / f"{stem}.claim_audit.json"
    audit_md.write_text(claim_audit_markdown(audit), encoding="utf-8")
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "experiment": str(experiment),
        "run_type": audit["run_type"],
        "dataset": audit["dataset"],
        "baselines": audit["baselines"],
        "raw_output_count": audit["raw_output_count"],
        "supported_claims": audit["supported_claims"],
        "blocked_claims": audit["blocked_claims"],
        "artifacts": {
            "claim_audit_markdown": str(audit_md),
            "claim_audit_json": str(audit_json),
        },
    }

    try:
        records = load_benchmark_records(experiment)
        table_group_fields = group_fields or None
        if table_group_fields:
            table_summary = paper_table_summary(records, group_fields=table_group_fields)
            table_text = paper_table_markdown(
                records,
                group_fields=table_group_fields,
                title=title or f"{stem} Paper Tables",
            )
        else:
            table_summary = paper_table_summary(records)
            table_text = paper_table_markdown(records, title=title or f"{stem} Paper Tables")
        table_md = output / f"{stem}.paper_tables.md"
        table_json = output / f"{stem}.paper_tables.json"
        table_md.write_text(table_text, encoding="utf-8")
        table_json.write_text(json.dumps(table_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["record_kind"] = table_summary.get("kind")
        manifest["artifacts"]["paper_tables_markdown"] = str(table_md)
        manifest["artifacts"]["paper_tables_json"] = str(table_json)
        comparison_summary = paired_comparison_summary(
            records,
            group_fields=table_group_fields,
        )
        comparison_md = output / f"{stem}.paired_comparison.md"
        comparison_json = output / f"{stem}.paired_comparison.json"
        comparison_md.write_text(
            paired_comparison_markdown(
                comparison_summary,
                title=f"{stem} Paired Comparison",
            ),
            encoding="utf-8",
        )
        comparison_json.write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["artifacts"]["paired_comparison_markdown"] = str(comparison_md)
        manifest["artifacts"]["paired_comparison_json"] = str(comparison_json)
    except Exception as exc:  # pragma: no cover - exercised by CLI workflows.
        manifest["table_error"] = f"{type(exc).__name__}: {exc}"

    manifest_path = output / f"{stem}.manifest.json"
    manifest["artifacts"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_experiment_bundle_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*experiment.json",
    group_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    source = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiments = sorted(path for path in source.glob(pattern) if path.is_file())
    manifests: list[dict[str, Any]] = []
    for experiment in experiments:
        bundle_dir = output / experiment.stem
        manifests.append(
            write_experiment_bundle(
                experiment,
                bundle_dir,
                group_fields=group_fields,
            )
        )
    batch_manifest = {
        "input_dir": str(source),
        "output_dir": str(output),
        "pattern": pattern,
        "experiment_count": len(experiments),
        "experiments": [manifest["experiment"] for manifest in manifests],
        "bundles": manifests,
    }
    manifest_path = output / "batch_manifest.json"
    batch_manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(batch_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return batch_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build reproducible paper-report bundles for AdaMem experiments."
    )
    parser.add_argument("input", type=Path, help="Experiment JSON or directory containing *experiment.json files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*experiment.json", help="Directory mode glob pattern")
    parser.add_argument("--group-fields", nargs="+")
    parser.add_argument("--title")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.input.is_dir():
        manifest = write_experiment_bundle_batch(
            args.input,
            args.output_dir,
            pattern=args.pattern,
            group_fields=args.group_fields,
        )
    else:
        manifest = write_experiment_bundle(
            args.input,
            args.output_dir,
            group_fields=args.group_fields,
            title=args.title,
        )
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        if args.input.is_dir():
            print(f"wrote {manifest['experiment_count']} report bundles to {args.output_dir}")
            print(f"batch_manifest: {manifest['manifest']}")
        else:
            print(f"wrote report bundle to {args.output_dir}")
            for name, path in manifest["artifacts"].items():
                print(f"{name}: {path}")
            if "table_error" in manifest:
                print(f"table_error: {manifest['table_error']}")


if __name__ == "__main__":
    main(sys.argv[1:])
