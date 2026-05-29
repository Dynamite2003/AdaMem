# AdaMem Benchmark Data Status

This document records which benchmark inputs are currently available in the
workspace and which claims they can support. Keep it updated when adding,
removing, converting, or regenerating benchmark data.

Last checked: 2026-05-30.

## STALE

Paper/source status:

- Paper: https://arxiv.org/abs/2605.06527
- Hugging Face paper page: https://huggingface.co/papers/2605.06527
- The paper describes 400 expert-validated conflict scenarios and 1,200
  evaluation queries across State Resolution, Premise Resistance, and Implicit
  Policy Adaptation.
- The Hugging Face paper page currently lists no linked dataset for the paper.

Local status:

- Available: `benchmarks/stale_mini.jsonl`
- Size: 2 local smoke cases.
- Missing: full STALE converted benchmark, expected path
  `benchmarks/stale.adamem.jsonl`.

Claim boundary:

- `stale_mini.jsonl` supports workflow validation, metric debugging, and
  trace-level smoke tests only.
- It does not support paper-level accuracy, generalization, or SOTA claims.

Useful commands:

```bash
PYTHONPATH=src python -m adamem.eval \
  --stale-diagnostics benchmarks/stale_mini.jsonl \
  --baselines semantic_state_adjudication semantic_state_premise_correction \
  --max-cases 1 \
  --experiment-output results/stale_mini_premise_correction_diagnostics.json \
  --diagnostic-cases-output results/stale_mini_premise_correction_cases.jsonl \
  --diagnostic-report-output results/stale_mini_premise_correction_report.md

PYTHONPATH=src python -m adamem.tables \
  results/stale_mini_premise_correction_diagnostics.json \
  --title "STALE Mini Premise Correction Tables" \
  --output results/stale_mini_premise_correction_tables.md
```

## LongMemEval-S

Local status:

- Available: `data/longmemeval_s_cleaned.json`
- Size at last check: 500 records.
- Converted pilots are generated to `/tmp` or `results/` as needed; do not
  assume old `/tmp` files persist.

Current use:

- Transfer and no-regression checks for public long-memory retrieval.
- State-aware readout exposure audits.
- Not a direct substitute for STALE Premise Resistance because most queries do
  not explicitly presuppose an invalidated state.

Latest no-regression command:

```bash
PYTHONPATH=src python -m adamem.convert longmemeval \
  data/longmemeval_s_cleaned.json \
  /tmp/longmemeval_s_balanced_60_premise_correction.adamem.jsonl \
  --expected evidence \
  --top-k 8 \
  --limit-per-type 10

PYTHONPATH=src python -m adamem.eval \
  --dataset /tmp/longmemeval_s_balanced_60_premise_correction.adamem.jsonl \
  --baselines semantic_state_adjudication semantic_state_premise_correction \
  --max-cases 60 \
  --benchmark-cases-output results/longmemeval_s_balanced_60_premise_correction_records.jsonl \
  --benchmark-report-output results/longmemeval_s_balanced_60_premise_correction_report.md \
  --experiment-output results/longmemeval_s_balanced_60_premise_correction.json
```

Latest observed result:

- `semantic_state_adjudication`: 39/60 evidence support.
- `semantic_state_premise_correction`: 39/60 evidence support.
- Paired comparison versus `semantic_state_adjudication`: gained 0, lost 0,
  net 0 across all six question types.
- Premise-correction readouts triggered 0 times, which is expected for this
  broad retrieval subset and useful as a no-pollution check.

Claim boundary:

- This supports a narrow no-regression claim for the correction mechanism on
  this LongMemEval-S balanced subset.
- It does not show that premise correction improves LongMemEval-S, because the
  subset provides no correction opportunities.

## AMA-Bench Public Pilot

Local status:

- Available bounded public pilot artifacts under `results/ama_public_*`.
- The strongest current result is retrieval/evidence-support only:
  `trajectory_step_readout` improves labeled step evidence recall over generic
  semantic retrieval on the first 20 public AMA episodes.

Claim boundary:

- Current AMA artifacts support trajectory evidence-recall and answerability
  diagnostics.
- They do not support end-to-end answer accuracy or SOTA claims without real
  answer/judge runs.

## Immediate Data Tasks

- Acquire or generate the full STALE input and convert it to
  `benchmarks/stale.adamem.jsonl`.
- Run a T1/T2-balanced STALE diagnostic with:
  `semantic_state_adjudication`,
  `semantic_state_premise_correction`, and a mainstream approximation baseline.
- Preserve every full or paper-facing run as an experiment JSON plus Markdown
  and JSON tables.
- Keep mini fixtures out of claim language except as workflow smoke tests.
