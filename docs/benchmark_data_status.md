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
PYTHONPATH=src python -m adamem.stale_pipeline \
  benchmarks/stale_mini.jsonl \
  --input-format adamem-jsonl \
  --output-dir /tmp/adamem_stale_pipeline_smoke \
  --run-name stale_mini_pipeline \
  --baselines semantic_state_adjudication semantic_state_premise_correction \
  --max-cases 1 \
  --json
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

## LongMemEval-V2

Paper/source status:

- Hugging Face dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-v2
- Public schema files include `questions.jsonl`, `trajectories.jsonl`, and
  haystack mappings under `haystacks/`.
- Question types include state-sensitive transfer targets such as
  `dynamic-environment`, `procedure`, `procedure-abs`, and related web or
  enterprise task categories.

Local status:

- Full trajectory data is not checked into this workspace. The public
  `trajectories.jsonl` file is large, so keep it under `data/` only when
  intentionally running public-transfer pilots.
- Converter support is available through:

```bash
PYTHONPATH=src python -m adamem.convert longmemeval-v2 \
  data/longmemeval-v2/questions.jsonl \
  data/longmemeval-v2/trajectories.jsonl \
  data/longmemeval-v2/haystacks/lme_v2_small.json \
  /tmp/longmemeval_v2_small.adamem.jsonl \
  --expected answer \
  --top-k 8 \
  --limit-per-type 5 \
  --max-trajectories-per-question 20
```

- Question-side audit support is available without downloading the large
  trajectory file:

```bash
PYTHONPATH=src python -m adamem.lme_v2 question-audit \
  --output-dir results/longmemeval_v2_question_audit \
  --json

PYTHONPATH=src python -m adamem.lme_v2 transfer-split \
  --audit-records results/longmemeval_v2_question_audit/longmemeval_v2_question_audit.records.jsonl \
  --output-dir results/longmemeval_v2_transfer_split \
  --transfer-per-type 10 \
  --control-per-group 10 \
  --json

PYTHONPATH=src python -m adamem.lme_v2 trajectory-manifest \
  --split-records results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl \
  --output-dir results/longmemeval_v2_trajectory_manifest \
  --json

PYTHONPATH=src python -m adamem.lme_v2 extract-trajectories \
  --trajectory-ids results/longmemeval_v2_trajectory_manifest/longmemeval_v2_split_trajectory_ids.jsonl \
  --trajectories data/longmemeval-v2/trajectories.jsonl \
  --output-dir data/longmemeval-v2/text_transfer_60 \
  --json

PYTHONPATH=src python -m adamem.lme_v2 validate-prep \
  --split-records results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl \
  --questions data/longmemeval-v2/questions.jsonl \
  --haystack data/longmemeval-v2/haystacks/lme_v2_small.json \
  --trajectories data/longmemeval-v2/text_transfer_60/longmemeval_v2_selected_trajectories.jsonl \
  --output-dir results/longmemeval_v2_text_transfer_60_validation \
  --json
```

Latest question-side audit:

- Public questions: `451`.
- Small-haystack coverage: `451/451`, with `100` trajectory ids per question.
- Type-level state-transfer candidates: `262/451`:
  `dynamic-environment`, `dynamic-environment-abs`, `procedure`,
  `procedure-abs`, and `errors-gotchas`.
- Query text produced state-slot signals for `341/451` questions, but `152`
  of these signals came from `static-environment*` questions. Treat those as
  router-audit warnings, not automatic state-transfer candidates.
- Dominant inferred slots were `location`, `workflow.*`, `resource.*.status`,
  `task.*.status`, and `runtime.*.status`.

Latest text-only transfer split:

- Selected questions: `60`.
- Transfer questions: `40`, with `10` each from `dynamic-environment`,
  `dynamic-environment-abs`, `procedure`, and `procedure-abs`.
- Static controls: `20`, split into `10` router-warning controls and `10`
  clean static controls.
- Domain/environment coverage after domain round-robin selection:
  `35` enterprise/workarena questions and `25` web questions across
  WebArena Reddit, CMS, and OneStopShop.
- `errors-gotchas` had `29` source candidates but `0` eligible text-only
  candidates because all require images. Include them only in a separate
  multimodal setting.
- Trajectory manifest for the split:
  `6,000` trajectory references collapse to `200` unique trajectory ids, with
  `0` missing haystack questions.
- Selected trajectory extraction is implemented as a streaming JSONL pass over
  the full `trajectories.jsonl`. It writes sanitized trajectory runtime fields
  only and strips accidental `answer`, `eval_function`, or `question` fields
  before conversion.
- Prepared-split validation checks that selected questions exist, haystacks are
  present, required trajectories are covered, trajectory ids are unique, and no
  selected trajectory record contains `answer`, `eval_function`, or `question`
  label fields.
- The split records can drive exact conversion after the trajectory file is
  available:

```bash
PYTHONPATH=src python -m adamem.convert longmemeval-v2 \
  data/longmemeval-v2/questions.jsonl \
  data/longmemeval-v2/text_transfer_60/longmemeval_v2_selected_trajectories.jsonl \
  data/longmemeval-v2/haystacks/lme_v2_small.json \
  /tmp/longmemeval_v2_text_transfer_60.adamem.jsonl \
  --question-ids-file results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl \
  --expected answer \
  --top-k 8
```

Claim boundary:

- Current LongMemEval-V2 support is conversion and diagnostic plumbing only.
- It can support public transfer pilots once the raw dataset is downloaded, but
  it does not yet support accuracy, generalization, or SOTA claims.
- Answers and evaluator strings are query-only metadata; trajectory
  observations must remain free of answer labels.

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
- Once a raw STALE JSON array is available, run:
  `PYTHONPATH=src python -m adamem.stale_pipeline data/T1_T2_400_FULL.json --output-dir results/stale_full_premise_correction --run-name stale_full_premise_correction --baselines semantic_only semantic_state_adjudication semantic_state_premise_correction --stale-types T1 T2 --limit-per-stale-type 10 --json`
- Run a T1/T2-balanced STALE diagnostic with:
  `semantic_state_adjudication`,
  `semantic_state_premise_correction`, and a mainstream approximation baseline.
- Preserve every full or paper-facing run as an experiment JSON plus Markdown
  and JSON tables.
- Keep mini fixtures out of claim language except as workflow smoke tests.
