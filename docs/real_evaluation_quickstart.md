# Real Evaluation Quickstart

This is the short path for starting real AdaMem benchmark runs. Use it when
API keys and the full benchmark file are available.

## Need

Required keys:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

Optional ModelHub keys:

```bash
export MODELHUB_API_KEY="..."
export MODELHUB_ENDPOINT="..."
```

Required STALE data, one of:

```text
data/T1_T2_400_FULL.json
benchmarks/stale.adamem.jsonl
```

Recommended minimum for useful paper-track pilots:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- full STALE source or converted STALE JSONL
- enough budget for a 2-5 case-per-type pilot before a full run

Do not commit keys, full benchmark dumps, or raw provider outputs unless the
repo policy explicitly allows them.

## Start

Check what is available:

```bash
env | rg '^(OPENAI_API_KEY|GEMINI_API_KEY|MODELHUB_API_KEY|MODELHUB_ENDPOINT)='
ls -l data/T1_T2_400_FULL.json benchmarks/stale.adamem.jsonl
```

If only the raw STALE file exists, convert it:

```bash
PYTHONPATH=src python -m adamem.convert stale data/T1_T2_400_FULL.json benchmarks/stale.adamem.jsonl
```

Generate or refresh the demo bundle:

```bash
PYTHONPATH=src python -m adamem.cli demo --all-queries --baseline-profile paper --bundle-output results/adamem_state_demo_bundle --json
PYTHONPATH=src python -m adamem.cli verify-demo results/adamem_state_demo_bundle --json
```

Build a small API pilot plan:

```bash
PYTHONPATH=src python -m adamem.study_plan \
  --output-dir results/stale_api_pilot \
  --no-data-prep \
  --no-ama \
  --stale-dataset benchmarks/stale.adamem.jsonl \
  --transfer-dataset benchmarks/dynamic_state_transfer.jsonl \
  --answer-model openai:gpt-4o-mini \
  --answer-model gemini:gemini-1.5-flash \
  --judge-model openai:gpt-4o-mini \
  --judge-model gemini:gemini-1.5-flash \
  --state-extractor-model openai:gpt-4o-mini \
  --limit-per-stale-type 2 \
  --demo-bundle results/adamem_state_demo_bundle \
  --json
```

List exact runnable commands:

```bash
PYTHONPATH=src python -m adamem.study_plan --plan results/stale_api_pilot/paper_study_plan.json --list-commands
```

Run the first answer/judge command from the command index:

```bash
PYTHONPATH=src python -m adamem.study_plan --plan results/stale_api_pilot/paper_study_plan.json --run --command <COMMAND_NAME>
```

Use the command name from:

```text
results/stale_api_pilot/paper_study_command_index.json
```

## Next

After the first pilot run:

1. Inspect the generated records and report.
2. Identify where AdaMem loses: stale evidence exposure, premise-resistance
   failure, missing state extraction, or wrong current-state authorization.
3. Patch the memory mechanism, not the benchmark labels.
4. Rerun the same pilot command.
5. Scale `--limit-per-stale-type` from `2` to `5`, then `10`, then full.
6. Add at least one transfer benchmark before writing strong claims.

Generate a report bundle after raw outputs exist:

```bash
PYTHONPATH=src python -m adamem.reporting results/stale_api_pilot --output-dir results/stale_api_pilot/report_bundle --json
```

Attach evidence to the demo:

```bash
PYTHONPATH=src python -m adamem.cli demo-readiness results/adamem_state_demo_bundle --evidence-manifest results/stale_api_pilot/report_bundle/paper_readiness.json --json --output results/stale_api_pilot/demo_readiness.json
```

## Target

The immediate goal is not a perfect full run. The first target is a small,
real, rerunnable STALE pilot that shows whether AdaMem's state-aware memory
mechanism beats the current baselines on the same split and model settings.

Once the pilot is positive, scale to full STALE, then transfer. If it is not
positive, iterate on the memory mechanism using the failure cases.
