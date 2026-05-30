# AdaMem Benchmark Readiness

Last checked: 2026-05-30.

This document records when AdaMem can move from API-free workflow hardening to
real benchmark measurement. It is a launch checklist, not experiment evidence.

## Current Status

API-free diagnostics can run now. Real paper-track STALE answer/judge
evaluation is not ready on this machine yet because provider credentials and
the full STALE source file are absent.

Local assets currently present:

- `benchmarks/stale_mini.jsonl`
- `benchmarks/dynamic_state_transfer.jsonl`
- `benchmarks/employer_dependency_transfer.jsonl`
- `benchmarks/employer_state_transfer.jsonl`
- `benchmarks/location_dependency_transfer.jsonl`
- `benchmarks/unknown_current_state_transfer.jsonl`
- `benchmarks/tiny_memory_qa.jsonl`
- `benchmarks/locomo_mini.json`
- `data/longmemeval_s_cleaned.json`

Local assets currently missing for the main paper run:

- Full STALE source, expected by default as `data/T1_T2_400_FULL.json`, or a
  converted equivalent such as `benchmarks/stale.adamem.jsonl`.
- Provider credentials such as `OPENAI_API_KEY` and `GEMINI_API_KEY`.
- Verified official or faithful mainstream baseline reproduction packets for
  at least one major baseline before making SOTA-facing claims.

## When Benchmarking Can Start

Start API-free benchmark diagnostics immediately when the goal is mechanism
debugging or regression testing:

```bash
python -m pytest
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl
```

Start a real STALE API pilot as soon as these are available:

- `OPENAI_API_KEY` and/or `GEMINI_API_KEY`;
- a full STALE source or converted STALE JSONL;
- a small pilot budget decision, usually `--limit-per-stale-type 2` to `5`.

The first API-backed run should be a pilot, not the full paper claim run. Its
purpose is to verify prompts, output caching, judge stability, cost, and
readiness gates.

## Recommended Launch Order

1. Generate or verify the demo bundle:

```bash
PYTHONPATH=src python -m adamem.cli demo --all-queries --baseline-profile paper --bundle-output results/adamem_state_demo_bundle --json
PYTHONPATH=src python -m adamem.cli verify-demo results/adamem_state_demo_bundle --json
```

2. Prepare a baseline reproduction packet template:

```bash
PYTHONPATH=src python -m adamem.baselines --packet-template a_mem_evolution --packet-output results/baseline_reproduction_plan/a_mem_evolution.reproduction_packet.json --json
```

3. Convert STALE if the raw source is available:

```bash
PYTHONPATH=src python -m adamem.convert stale data/T1_T2_400_FULL.json benchmarks/stale.adamem.jsonl
```

4. Build the API pilot packet:

```bash
PYTHONPATH=src python -m adamem.study_plan \
  --output-dir results/stale_api_packet \
  --no-data-prep \
  --no-ama \
  --stale-dataset benchmarks/stale.adamem.jsonl \
  --transfer-dataset benchmarks/dynamic_state_transfer.jsonl \
  --answer-model openai:gpt-4o-mini \
  --answer-model gemini:gemini-1.5-flash \
  --judge-model openai:gpt-4o-mini \
  --judge-model gemini:gemini-1.5-flash \
  --state-extractor-model openai:gpt-4o-mini \
  --baseline-reproduction-packet results/baseline_reproduction_plan/a_mem_evolution.reproduction_packet.json \
  --demo-bundle results/adamem_state_demo_bundle \
  --json
```

5. Inspect `results/stale_api_packet/paper_study_validation.json`.

The plan should not be treated as ready until `execution_ready=true`. If a
baseline packet or demo bundle is supplied but incomplete, validation will now
block execution with `baseline_reproduction_packets_ready` or
`demo_bundle_verified`.

6. Run a small API command subset before the full run:

```bash
PYTHONPATH=src python -m adamem.study_plan --plan results/stale_api_packet/paper_study_plan.json --list-commands
PYTHONPATH=src python -m adamem.study_plan --plan results/stale_api_packet/paper_study_plan.json --run --command stale_answer_openai_gpt_4o_mini_judged_by_openai_gpt_4o_mini
```

Use the exact command name from `paper_study_command_index.json` if model names
change.

7. Generate reporting and demo readiness only after raw outputs exist:

```bash
PYTHONPATH=src python -m adamem.reporting results/stale_api_packet --output-dir results/stale_api_packet/report_bundle --baseline-reproduction-packet results/baseline_reproduction_plan/a_mem_evolution.reproduction_packet.json --json
PYTHONPATH=src python -m adamem.cli demo-readiness results/adamem_state_demo_bundle --evidence-manifest results/stale_api_packet/report_bundle/paper_readiness.json --json --output results/stale_api_packet/demo_readiness.json
```

## Claim Boundary

API-free diagnostics can justify engineering iteration. Paper claims require
API-backed answer/judge records, raw outputs, exact model settings, verified
baseline evidence, and at least one transfer check beyond STALE.
