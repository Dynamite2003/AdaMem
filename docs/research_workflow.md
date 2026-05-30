# AdaMem Research Workflow

This workflow is designed for a CCF-A-level research target. It separates work
that can be done without API keys from work that requires answer/judge models.
The goal is to make API-enabled runs mostly execution, not improvisation.

## Guiding Principle

Do not optimize for one small fixture or one judge prompt. Every step should
make the final paper more defensible by improving reproducibility, causal
validity, generality, or error analysis.

## Phase 0: Repository Baseline

Purpose: keep the research codebase stable before changing mechanisms.

Inputs:

- Current source tree.
- Existing unit tests and mini benchmarks.

Actions:

- Run deterministic tests.
- Run synthetic and JSONL smoke benchmarks.
- Record current behavior before each design iteration.

Commands:

```bash
python -m pytest
PYTHONPATH=src python -m adamem.eval
PYTHONPATH=src python -m adamem.eval --list-baselines
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/tiny_memory_qa.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/unknown_current_state_transfer.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_premise_correction
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 1 --experiment-output results/dynamic_state_transfer_smoke.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json benchmarks/longmemeval_s.adamem.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/longmemeval_s.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_propagation full state_readout state_propagation --max-cases 20 --experiment-output results/longmemeval_transfer_pilot.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10
PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_state_adjudication_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_state_adjudication_report.md --experiment-output results/longmemeval_s_balanced_60_state_adjudication_pilot.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_inferred_state.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --infer-state-slots
PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60_inferred_state.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-report-output results/longmemeval_s_balanced_60_inferred_state_report.md --experiment-output results/longmemeval_s_balanced_60_inferred_state.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_audit_probe.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-output results/longmemeval_s_balanced_60_state_audit_candidates.jsonl --state-audit-summary-output results/longmemeval_s_balanced_60_state_audit_summary.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_full_audit_probe.adamem.jsonl --expected evidence --top-k 8 --state-audit-output results/longmemeval_s_full_state_audit_candidates.jsonl --state-audit-summary-output results/longmemeval_s_full_state_audit_summary.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_manual_audit.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-input results/longmemeval_s_balanced_60_state_audit_reviewed.jsonl
PYTHONPATH=src python -m adamem.lme_v2 question-audit --output-dir results/longmemeval_v2_question_audit --json
PYTHONPATH=src python -m adamem.lme_v2 transfer-split --audit-records results/longmemeval_v2_question_audit/longmemeval_v2_question_audit.records.jsonl --output-dir results/longmemeval_v2_transfer_split --transfer-per-type 10 --control-per-group 10 --json
PYTHONPATH=src python -m adamem.lme_v2 trajectory-manifest --split-records results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl --output-dir results/longmemeval_v2_trajectory_manifest --json
PYTHONPATH=src python -m adamem.lme_v2 extract-trajectories --trajectory-ids results/longmemeval_v2_trajectory_manifest/longmemeval_v2_split_trajectory_ids.jsonl --trajectories data/longmemeval-v2/trajectories.jsonl --output-dir data/longmemeval-v2/text_transfer_60 --json
PYTHONPATH=src python -m adamem.lme_v2 validate-prep --split-records results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl --questions data/longmemeval-v2/questions.jsonl --haystack data/longmemeval-v2/haystacks/lme_v2_small.json --trajectories data/longmemeval-v2/text_transfer_60/longmemeval_v2_selected_trajectories.jsonl --output-dir results/longmemeval_v2_text_transfer_60_validation --json
PYTHONPATH=src python -m adamem.lme_v2 state-evidence-audit --split-records results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl --haystack data/longmemeval-v2/haystacks/lme_v2_small.json --trajectories data/longmemeval-v2/text_transfer_60/longmemeval_v2_selected_trajectories.jsonl --output-dir results/longmemeval_v2_text_transfer_60_state_evidence --json
PYTHONPATH=src python -m adamem.pilot lme-v2-prepared --questions data/longmemeval-v2/questions.jsonl --trajectories data/longmemeval-v2/text_transfer_60/longmemeval_v2_selected_trajectories.jsonl --haystack data/longmemeval-v2/haystacks/lme_v2_small.json --split-records results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl --output-dir results/longmemeval_v2_text_transfer_60_pilot --baselines semantic_only semantic_state_readout semantic_state_premise_correction --top-k 8 --json
PYTHONPATH=src python -m adamem.reporting results/longmemeval_v2_text_transfer_60_pilot/longmemeval_v2_prepared.answer.experiment.json --output-dir results/longmemeval_v2_text_transfer_60_bundle --group-fields question_type selection_group --json
PYTHONPATH=src python -m adamem.convert longmemeval-v2 data/longmemeval-v2/questions.jsonl data/longmemeval-v2/trajectories.jsonl data/longmemeval-v2/haystacks/lme_v2_small.json /tmp/longmemeval_v2_small.adamem.jsonl --expected answer --top-k 8 --limit-per-type 5 --max-trajectories-per-question 20
PYTHONPATH=src python -m adamem.convert longmemeval-v2 data/longmemeval-v2/questions.jsonl data/longmemeval-v2/text_transfer_60/longmemeval_v2_selected_trajectories.jsonl data/longmemeval-v2/haystacks/lme_v2_small.json /tmp/longmemeval_v2_text_transfer_60.adamem.jsonl --question-ids-file results/longmemeval_v2_transfer_split/longmemeval_v2_transfer_split.records.jsonl --expected answer --top-k 8
PYTHONPATH=src python -m adamem.convert ama data/ama_bench.jsonl benchmarks/ama_bench.adamem.jsonl --expected answer --top-k 8
PYTHONPATH=src python -m adamem.pilot ama-public --limit 20 --output-dir results/ama_public_20_light --baselines semantic_only trajectory_step_readout --top-k 8 --answer-only
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 2 --experiment-output results/stale_mini_state_adjudication_diagnostics.json --diagnostic-cases-output results/stale_mini_state_adjudication_cases.jsonl --diagnostic-report-output results/stale_mini_state_adjudication_report.md
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication state_readout --stale-types T1 T2 --limit-per-stale-type 10 --experiment-output results/stale_balanced20_state_adjudication_diagnostics.json --diagnostic-cases-output results/stale_balanced20_state_adjudication_cases.jsonl --diagnostic-report-output results/stale_balanced20_state_adjudication_report.md
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --experiment-output results/stale_diagnostics_smoke.json
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output results/stale_diagnostic_cases.jsonl
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output results/stale_diagnostic_cases.jsonl --diagnostic-report-output results/stale_failure_report.md
PYTHONPATH=src python -m adamem.convert stale-annotate benchmarks/stale_mini.jsonl /tmp/stale_mini.annotated.jsonl
PYTHONPATH=src python -m adamem.stale_pipeline benchmarks/stale_mini.jsonl --input-format adamem-jsonl --output-dir /tmp/adamem_stale_pipeline_smoke --run-name stale_mini_pipeline --baselines semantic_state_adjudication semantic_state_premise_correction --max-cases 1 --json
PYTHONPATH=src python -m adamem.eval --stale benchmarks/stale_mini.jsonl --answer-provider mock --judge-provider mock --max-cases 1 --experiment-output results/stale_pilot_mock.json
PYTHONPATH=src python -m adamem.study_plan --profile smoke --output-dir /tmp/adamem_study_smoke --json
PYTHONPATH=src python -m adamem.study_plan --profile smoke --output-dir /tmp/adamem_study_smoke_run --run --dry-run --stage diagnostic --json
PYTHONPATH=src python -m adamem.study_plan --plan /tmp/adamem_study_smoke/paper_study_plan.json --run --dry-run --stage diagnostic --json
PYTHONPATH=src python -m adamem.study_plan --output-dir results/paper_study_plan --json
```

Outputs:

- Clean test result.
- Baseline synthetic/JSONL tables.
- Dynamic-state transfer smoke table.
- JSON experiment records for any meaningful JSONL retrieval benchmark run,
  including converted LongMemEval or local transfer fixtures.
- LongMemEval-V2 converted public-transfer pilots once the raw question,
  trajectory, and haystack files are present locally.
- LongMemEval-V2 question-side audit records that separate type-level transfer
  candidates from noisy query-state-slot signals before selecting a public
  transfer split.
- LongMemEval-V2 transfer split manifest and exact question-id file for later
  answer-model and retrieval runs.
- LongMemEval-V2 trajectory-id manifest for the selected split so data
  acquisition and conversion can be checked before loading the full trajectory
  file.
- Sanitized LongMemEval-V2 selected trajectory file extracted from the full
  trajectory JSONL, containing only runtime trajectory fields needed by the
  converter.
- Prepared-split validation artifact proving question, haystack, trajectory,
  duplicate-id, and label-leak checks passed before conversion or API runs.
- Prepared-split state-evidence audit proving how many selected
  LongMemEval-V2 questions have deterministic state evidence for their routed
  state slots before spending answer-model or judge-model budget.
- API-free LongMemEval-V2 prepared-split pilot outputs, including validation,
  state-evidence audit, exact-split conversion, retrieval answer-string
  support records, Markdown report, and experiment JSON. This is a retrieval
  support diagnostic, not generated answer accuracy.
- LongMemEval-V2 prepared-pilot report bundle with claim audit, grouped paper
  tables, paired comparison, and an explicit block on answer-accuracy/SOTA
  claims.
- Paper-study plan artifacts generated before API runs:
  `paper_study_plan.json`, `paper_study_plan.md`, and
  `paper_study_commands.sh`, plus `paper_study_command_index.json/md` and
  `paper_study_validation.json/md`. These fix the intended data-preparation
  commands, STALE answer/judge matrix, LLM extractor ablation, transfer
  diagnostics, and post-run report command without claiming that the planned
  experiments have already run. The command index is the stable source for
  exact `--command NAME` values used in small API pilots; it also lists the
  providers and environment variables required by each command. The
  validation artifacts report missing dataset paths, whether a missing target
  can be prepared from an available source, default model placeholders,
  model-count gaps, method-coverage gaps, and whether the post-run reporting
  command is present. By default, generated full benchmark JSONL files are
  written under `OUTPUT_DIR/data/` instead of `benchmarks/`, because full
  conversions can be large and should not become tracked fixtures by accident.
- API-free smoke-study plan artifacts can be generated with
  `--profile smoke`. This profile uses tracked mini/local fixtures and mock
  providers only. It validates conversion-free STALE diagnostics, mock
  answer/judge plumbing, LLM-extractor plumbing, transfer retrieval, and batch
  reporting, but it is explicitly not paper evidence.
- API pilot settings can be generated before keys are available:
  `PYTHONPATH=src python -m adamem.study_plan --write-settings-template results/api_pilot_settings.json --output-dir results/api_pilot_study --json`.
  Edit only provider/model labels, limits, dataset paths, and inclusion flags
  in that JSON. Keep credentials in environment variables such as
  `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `MODELHUB_API_KEY`, never inside the
  settings JSON. Then generate the concrete plan with
  `PYTHONPATH=src python -m adamem.study_plan --settings results/api_pilot_settings.json --check-env --json`.
  The generated `paper_study_plan.json` records the settings fingerprint and,
  for CLI-generated plans, the settings path. The same provenance is carried
  into validation artifacts and run summaries. Settings files that contain
  credential-like keys such as `api_key`, `token`, `secret`, or `password` are
  rejected at load time.
- Study plans can also be executed through the same CLI with `--run`.
  `--dry-run` writes command records without executing them, and repeatable
  `--stage` filters the run to stages such as `diagnostic`, `answer_judge`,
  `mechanism_ablation`, `transfer`, or `reporting`. Use repeatable
  `--list-commands` to inspect exact command names, then repeatable
  `--command NAME` to execute a single planned command, for example one
  answer/judge pair before spending API budget on the whole `answer_judge`
  stage. During filtered runs, `--check-env` checks only providers used by the
  selected commands while preserving the global plan validation artifact. If a
  run fails after completing expensive API commands, rerun with `--resume-run`
  to append to the same run log and skip only prior commands from the same plan
  fingerprint whose name, stage, shell command, `status=completed`, and
  declared outputs are all clean. Resume summaries report prior, appended, and
  final log record counts so partial-run audits can distinguish old records
  from the current invocation.
- Saved or manually edited plans can be loaded with
  `--plan path/to/paper_study_plan.json`, then validated or run without
  regenerating the command matrix from CLI defaults.
- Saved plans carry a recorded `plan_fingerprint`. If manual edits make the
  recorded fingerprint stale, validation marks the plan not execution-ready and
  `--run` blocks unless `--allow-not-ready` is explicitly used. After an
  intentional review/edit, refresh the saved JSON with
  `--plan path/to/paper_study_plan.json --refresh-fingerprint`.
- Notes on any regression before new work begins.

Done when:

- Tests pass.
- Any intentional behavior change is documented.

## Phase 1: Evaluation Harness Hardening

Purpose: make STALE evaluation trustworthy before spending API budget.

Can be done without API keys.

Actions:

- Audit STALE conversion for schema correctness.
- Ensure ground-truth fields are evaluation-only:
  `M_old`, `M_new`, `explanation`, `relevant_session_index`, answers, and judge
  labels must not enter proposed runtime memory.
- Add strict diagnostics that do not overcount common words.
- Separate retrieval diagnostics from final answer correctness.
- Add output schemas for cached raw answer and judge results.
- Add config snapshots to every experiment output.

Expected diagnostics:

- Current-evidence recall: whether retrieved context contains evidence for the
  current belief.
- Stale-evidence exposure: whether retrieved context exposes old belief
  evidence.
- Conflict-pair coverage: whether both old and new supporting observations are
  identified by retrieval.
- Authorized-state hit rate: whether the readout exposes the active state that
  should govern the answer.
- Stale premise correction opportunity and hit rate: whether a Premise
  Resistance query mentions an old state, the system has enough current-state
  evidence to correct it, and a read-time correction is actually surfaced.

Outputs:

- Hardened diagnostic code.
- Deterministic tests for metric edge cases.
- A small debug report on `benchmarks/stale_mini.jsonl`.
- JSON experiment records containing command, commit, dataset, case limit,
  baseline names, configs, and diagnostics.
- JSONL retrieval benchmark records containing command, commit, dataset, case
  limit, baseline names, configs, pass/fail support checks, retrieved text, and
  per-query metadata/traces.
- Markdown JSONL retrieval reports grouped by query metadata such as
  LongMemEval `question_type`, local `dimension`, state slot, and abstention.
- Evidence-support diagnostics for JSONL retrieval records, including expected
  evidence labels, missing evidence labels, graph retrieval counts, and graph
  evidence hits. These diagnostics are required for AMA-style trajectory runs
  so answer-string success can be separated from causal step recall.
- Answerability diagnostics for open-ended JSONL records, including
  answer-keyword recall over retrieved context and over a deterministic
  trajectory answer basis derived from retrieved step/action/observation
  traces. The basis can include deterministic active-rule, blocked-action,
  no-progress, state-reversion, and inverse-action facts. This is
  evaluation-only and may use answer text; the basis itself must not use answer
  labels.
- Per-metadata diagnostic breakdowns for evidence support and answerability,
  not only exact answer-string pass/fail. This is required for AMA A/B/C/D
  analysis because exact answer-string support is too strict for open-ended
  trajectory answers.
- Pairwise baseline comparisons against the first requested baseline, including
  gained passes, lost passes, net delta, both-pass, and both-fail counts.
- Case-level diagnostic JSONL records for representative failure analysis.
- Markdown failure reports grouped by failure mode, baseline, STALE dimension,
  and stale type.

Done when:

- Diagnostics are stable, interpretable, and do not rely on loose keyword
  overlap alone.
- The same run can be reproduced from command-line arguments.

## Phase 2: Baseline Matrix Design

Purpose: decide what must be compared before API runs begin.

Can be done without API keys.

Baseline categories:

- No memory / recent context only.
- Similarity retrieval over raw turns.
- Similarity plus recency.
- Similarity plus reranking or MMR.
- Raw AdaMem variants: semantic-only, temporal, graph, delta, soft stale,
  propagation, full.
- Mainstream approximation: `a_mem_evolution`, an API-free A-MEM-style note,
  dynamic linking, and memory evolution baseline over raw episodes.
- Mainstream approximation: `zep_temporal_kg`, an API-free Zep/Graphiti-style
  temporal KG baseline with active and invalidated fact edges.
- Mainstream approximation: `mem0_extraction`, an API-free Mem0-style compact
  extraction/update baseline over extracted facts.
- State-aware AdaMem variants.
- Mainstream memory systems or faithful local approximations when official code
  is unavailable.

Actions:

- Define exact configs for each baseline.
- Assign stable names to every config.
- Make sure each baseline can be invoked by CLI or function call.
- Use `--baselines` to keep pilot runs focused while preserving canonical
  baseline names in the experiment record.
- Store config dictionaries with experiment outputs.

Outputs:

- Baseline registry.
- Mapping from paper table names to runnable configs.

Done when:

- A single command can enumerate all planned baselines.
- A single command can run any named subset of baselines for quick pilots.
- Each baseline has a short explanation of what mechanism it tests.

## Phase 3: State-Aware Method Design

Purpose: move beyond similarity-only stale detection.

Can be designed and partly implemented without API keys.

Target idea:

- Convert raw observations into state candidates.
- Assign each state to a typed slot.
- Track active, stale, replaced, and unknown-current states.
- Propagate invalidation across related slots.
- At read time, produce an authorized current-state basis before raw episodes.
- Correct stale query premises when the active state conflicts with the query.

Candidate state slots:

- Location.
- Schedule and availability.
- Preferences.
- Health, dietary, and safety constraints.
- Relationships and roles.
- Task, incident, and project status.
- Resource ownership or credentials.
- Plans, commitments, and goals.

API-free implementation options:

- Rule-based extractor for controlled tests and development.
- Metadata-backed state records for benchmark converters.
- Pluggable extractor interface with deterministic mock implementation.
- LLM JSON extractor adapter with injected client. Use mock clients or
  metadata-backed patch payloads for CI; use real providers only in explicitly
  named extractor ablations with provider/model/prompt settings recorded.
- Oracle extractor only for upper-bound/debug experiments, clearly separated
  from the proposed method.

Required ablations:

- No state extraction.
- State extraction only.
- State extraction plus stale adjudication.
- State extraction plus authorized readout.
- State extraction plus propagation.
- Full state-aware AdaMem.

Current API-free state-aware baselines:

- `state_memory`: deterministic state extraction writes derived state memories,
  but no explicit readout boost is applied.
- `a_mem_evolution`: A-MEM-style deterministic memory notes, dynamic links,
  and write-time evolution of raw memory retrieval attributes. This is a
  mainstream memory-evolution comparator, not a state-authority method.
- `zep_temporal_kg`: Zep/Graphiti-style temporal fact edges derived from the
  same extractor. It invalidates old KG edges and reads out active facts, but
  does not adjudicate stale raw source evidence.
- `mem0_extraction`: Mem0-style compact extracted facts derived from the same
  extractor. It hides raw observations from retrieval, updates same-slot facts,
  and tests whether compact memory alone can replace raw evidence plus
  adjudication.
- `semantic_state_readout`: semantic-only raw retrieval plus deterministic
  state extraction and authorized state readout. This isolates the state
  mechanism from default full AdaMem scoring.
- `semantic_state_propagation`: semantic-only retrieval plus state readout and
  typed dependency propagation.
- `semantic_state_adjudication`: semantic-only retrieval plus authorized state
  readout and query-scoped filtering of raw evidence superseded by the same
  state slot. This isolates stale suppression from global historical evidence
  deletion.
- `semantic_state_premise_correction`: semantic state adjudication plus an
  ephemeral read-time correction when a query explicitly mentions a stale value
  for a routed current-state slot. This isolates Premise Resistance behavior
  from ordinary current-state readout.
- `semantic_llm_state_adjudication`: same state-authority/adjudication path as
  `semantic_state_adjudication`, but with `state_extractor_name=llm_json`.
  This baseline is not part of default API-free runs; explicitly select it and
  provide `--state-extractor-provider`.
- `semantic_llm_state_premise_correction`: LLM JSON extraction plus the
  premise-correction readout path. Use this to separate extractor quality from
  the read-time stale-premise correction mechanism.
- `use_state_unknown_current`: state extraction can produce an active
  `unknown_current` slot when new evidence invalidates an old value without
  providing a replacement. This prevents the memory layer from continuing to
  authorize stale values or hallucinating a new current value.
  Current deterministic coverage includes location, resource status,
  workflow/runbook rules, and runtime/tool status.
  JSONL diagnostics report unknown-current records, corrections, and resolved
  invalidated values separately from unresolved stale evidence.
- `semantic_state_propagation_adjudication`: semantic state adjudication plus
  typed dependency propagation for indirectly invalidated state slots. JSONL
  diagnostics expose dependency-derived unknown-current records separately, so
  this mechanism can be audited apart from direct unknown-current extraction.
- `state_readout`: deterministic state extraction plus authorized current-state
  readout for state-sensitive queries.
- `state_propagation`: state readout plus typed dependency propagation from a
  changed state slot to dependent state slots and their source evidence.

Current deterministic state slots:

- `location`
- `local.gym`
- `preference.beverage`
- `schedule.availability`
- `task.*.status`
- `health.*.status`
- `resource.*.status`
- `workflow.*`
- `runtime.*.status`
- `role.current`
- `relationship.manager`
- `organization.employer`
- `employment.benefits_portal`

The extractor is pluggable through `AdaMem(..., state_extractor=...)`. Use the
deterministic extractor for API-free mechanism tests, and introduce LLM or
domain-specific extractors only as separately named baselines. The default
`AdaMemConfig.state_extractor_name` is `deterministic`; `metadata_mock_llm`
exists only for deterministic CI fixtures, while real LLM extraction should
inject `LLMStateExtractor(client)` so the answer/judge providers remain
separate from the memory write path.

`adamem.eval` can run LLM-extractor baselines through:

```bash
PYTHONPATH=src python -m adamem.eval \
  --dataset benchmarks/dynamic_state_transfer.jsonl \
  --baselines semantic_llm_state_adjudication \
  --state-extractor-provider mock \
  --state-extractor-mock-response '{"patches":[]}' \
  --experiment-output /tmp/llm_extractor_smoke.experiment.json
```

For real runs, replace `mock` with `openai`, `gemini`, or `modelhub`, set
`--state-extractor-model`, and keep the resulting experiment JSON because it
records extractor provider, model, prompt template, max tokens, temperature,
and affected baselines. STALE diagnostic and STALE LLM-judge paths use the
same flags.

Derived `state` memories are hidden from ordinary direct retrieval by default
when `use_state_readout_authorization=True`. They can still enter context
through authorized readout for state-sensitive queries. Disable the flag only
for named ablations that test uncontrolled state-memory exposure.

Outputs:

- Method design note.
- Data structures and config flags.
- Unit tests using synthetic examples.
- Literature-to-design map in `docs/literature_to_design.md`.

Done when:

- The method has a clear hypothesis.
- Every named mechanism has an ablation switch.
- Tests cover State Resolution, Premise Resistance, and Implicit Policy
  Adaptation.

## Phase 4: API-Free Diagnostics And Error Analysis

Purpose: learn as much as possible before real LLM answer runs.

Can be done without API keys.

Actions:

- Run retrieval diagnostics over mini and converted public data.
- Inspect failure cases by STALE dimension and type.
- Categorize failure causes.
- Compare raw-turn retrieval versus state-aware readout.
- Build an error taxonomy that can later be used in the paper.

Suggested error taxonomy:

- Current evidence not retrieved.
- Old evidence retrieved above current evidence.
- Both retrieved but no adjudication.
- State slot not extracted.
- State slot extracted but wrong value remains active.
- Query premise conflicts with active state but is not corrected.
- Answer model ignores current state despite correct context.
- Judge ambiguity or prompt sensitivity.

Outputs:

- Retrieval diagnostic tables.
- Failure-case notes.
- Case-level JSONL records with failure modes and retrieved traces.
- Markdown failure report with representative examples per failure mode.
- Candidate figures for the paper.

Done when:

- The top failure modes are known before API spending.
- Each proposed mechanism maps to at least one observed failure mode.

## Phase 5: API-Enabled Pilot

Purpose: run a small, controlled end-to-end evaluation once keys are available.

Requires API keys.

Actions:

- Run 5 to 20 STALE cases across selected baselines.
- Cache raw answers, raw judge outputs, prompts, configs, and timestamps.
- Compare answer correctness with retrieval diagnostics.
- Manually inspect disagreements between judge result and expected behavior.

Minimum metadata per run:

- Dataset path and split.
- Baseline config.
- Answer provider/model.
- Judge provider/model.
- Prompt version.
- Top-k and context budget.
- Temperature and max tokens.
- Code commit hash.
- Start time and cost estimate if available.

Outputs:

- Pilot run directory.
- Pilot summary table.
- JSON experiment record containing answer/judge prompts, raw answer outputs,
  raw judge outputs, retrieved traces, model settings, baseline configs, and
  command.
- Short decision note: proceed, fix harness, or redesign method.

Done when:

- End-to-end results are credible enough to scale.
- At least one manual audit sample confirms the judge is behaving reasonably.

Mock smoke command before spending API budget:

```bash
PYTHONPATH=src python -m adamem.eval \
  --stale benchmarks/stale_mini.jsonl \
  --answer-provider mock \
  --judge-provider mock \
  --max-cases 1 \
  --experiment-output results/stale_pilot_mock.json
```

Real API pilot shape once keys are available:

```bash
PYTHONPATH=src python -m adamem.eval \
  --stale benchmarks/stale.adamem.jsonl \
  --answer-provider openai \
  --answer-model gpt-4o-mini \
  --judge-provider gemini \
  --judge-model gemini-1.5-flash \
  --max-cases 20 \
  --top-k 8 \
  --max-context-chars 4000 \
  --request-delay 0.5 \
  --experiment-output results/stale_pilot_real.json
```

In this mode, STALE `M_old`, `M_new`, and explanations appear only in judge
prompts and experiment records. They must not enter `AdaMem.observe`,
`AdaMem.retrieve`, state extraction, or runtime context except in explicitly
named oracle/debug experiments.

## Phase 6: Full Experiment Run

Purpose: produce paper-quality quantitative evidence.

Requires API keys and a stable harness.

Actions:

- Run all selected baselines on the agreed STALE split.
- Run at least one additional memory benchmark for transfer.
- Run model and judge robustness checks.
- Run all ablations.
- Freeze prompts and configs.

Outputs:

- Main results table.
- Per-dimension and per-type STALE breakdown.
- Ablation table.
- Diagnostic table.
- Transfer benchmark table.
- Error analysis samples.

Done when:

- Results are reproducible from saved configs.
- Claims are supported by ablations and diagnostics.
- Mechanism-only claims such as `unknown_current_trace_resolution` remain
  clearly separated from answer accuracy and SOTA until public benchmark and
  answer/judge evidence exists.
- Mechanism-only claims such as `dependency_propagation_trace_resolution`
  remain trace-level claims when they come from local dependency fixtures; they
  do not establish public-benchmark generality or answer accuracy.
- Limitations are clear.

## Phase 7: Paper Assembly

Purpose: convert experiments into a coherent CCF-A submission.

Actions:

- Write problem framing around stale memory, premise resistance, and current
  state authorization.
- Define the method formally enough for reproduction.
- Include algorithm boxes or diagrams only after the mechanism is stable.
- Report baselines, settings, metrics, and statistical or robustness checks.
- Include limitations and failure modes.
- Package reproducibility artifacts.

Paper claims should be phrased around demonstrated evidence:

- "Improves STALE accuracy under these models and settings" only after
  end-to-end runs.
- "Reduces stale evidence exposure" only after retrieval diagnostics are
  validated.
- "Generalizes" only after at least one non-STALE benchmark supports it.

## Recommended Immediate Work Without API Keys

Completed API-free foundations:

- STALE retrieval diagnostics with stricter text signals.
- STALE diagnostic failure-case JSONL export.
- STALE diagnostic failure report aggregation.
- STALE LLM-judge experiment records with prompts, raw outputs, retrieved
  traces, model settings, and command.
- JSONL retrieval benchmark experiment records for local transfer fixtures and
  converted public benchmark adapters.
- CLI baseline filtering via `--baselines`, so public benchmark pilots can run
  a focused subset such as `semantic_only full state_readout state_propagation`.
- Experiment record schema for configs, command, commit, diagnostics, prompts,
  raw outputs, and notes.
- Stable baseline registry with runnable configs.
- Literature-to-design map connecting current papers and SOTA systems to
  AdaMem hypotheses.
- Deterministic state-aware prototype with typed state slots and pluggable
  extractor hook.
- Dynamic state readout for wildcard slots such as `task.*.status`.
- State dependency propagation from changed slots to dependent slots.
- Dependency-propagation diagnostics for JSONL runs, including
  dependency-derived unknown-current readouts, dependency-derived corrections,
  resolved invalidated values, and parent slots.
- State readout authorization boundary, which keeps derived state records out
  of ordinary direct retrieval unless the query is routed to an authorized
  state slot.
- State premise correction, which injects an ephemeral current-state correction
  only when a query mentions an inactive value for the same authorized state
  slot. This targets STALE Premise Resistance and should be evaluated separately
  from broader answer-prompt instructions.
- Premise-correction diagnostics for STALE-style retrieval runs, including
  opportunity rate, hit rate, correction rank, and correction-aware stale
  exposure traces.
- Premise-correction trigger diagnostics for generic JSONL retrieval benchmarks,
  so transfer runs can report correction records, corrected stale premises, and
  unresolved forbidden old values.
- `adamem.stale_pipeline`, a one-command API-free STALE workflow that converts
  raw STALE JSON, runs retrieval diagnostics, writes case records, paper
  tables, claim-audit report bundles, and a manifest. It also accepts already
  converted AdaMem JSONL with `--input-format adamem-jsonl` for smoke runs.
  Converted JSONL inputs are automatically backfilled with STALE query-only
  opportunity labels in the pipeline output dataset, and the manifest records
  an `opportunity_summary` plus observation-metadata violation count.
- STALE conversion can add evaluation-only query metadata for inferred
  state/dependency opportunities such as `state_slot`,
  `dependency_source_slot`, and `dependency_target_family`. These labels are
  derived from STALE metadata for grouping and opportunity audits only; they
  must not be written into observation metadata or consumed by the proposed
  runtime memory method.
- Report bundles surface STALE opportunity coverage from experiment notes as
  `opportunity_evidence`, and claim matrices include a `stale opportunities`
  column plus leakage-sensitive readiness actions.
- STALE diagnostic failure reports group records by opportunity labels,
  including expected state slot, dependency source slot, dependency target
  family, and dependency target family by baseline. These are evaluation-only
  slices for paper error analysis.
- STALE retrieval paper tables include opportunity-label grouped metrics by
  default when the labels are present, so state/dependency subsets can report
  current recall, stale exposure, premise-correction opportunity, and
  correction-hit rates.
- Semantic-only state-aware ablations, so state mechanisms can be tested
  independently of default full AdaMem scoring.
- Synthetic tests for State Resolution, Premise Resistance, and Implicit Policy
  Adaptation.
- Mini-fixture retrieval diagnostics for `benchmarks/stale_mini.jsonl`.
- Non-STALE dynamic-state transfer smoke fixture covering schedule
  availability, task status, and beverage preference.
- LongMemEval converter skeleton for the official
  `longmemeval_*_cleaned.json` schema.
- AMA-Bench-style trajectory converter for JSON or JSONL agent traces. It
  preserves action-to-observation causality through `cause_labels` and keeps
  answer/evidence labels on query metadata only. The converter now follows the
  public Hugging Face schema fields `episode_id`, `turn_idx`,
  `question_uuid`, and `type`, and infers diagnostic evidence labels from
  `Step N` query references when AMA records do not provide explicit evidence.
- JSONL benchmark evidence-support diagnostics, including an `Evidence Support`
  Markdown table and case-level `graph_evidence_hits` for causal trajectory
  audits.
- State-memory inventory diagnostics in JSONL records, JSONL reports, STALE
  diagnostic case records, and STALE LLM-judge raw outputs. These expose
  derived state count, active/stale state count, unknown-current count, and
  active/stale slots without using benchmark labels, which separates extractor
  failures from readout/adjudication failures.
- Failure-attribution taxonomy for JSONL retrieval records and STALE
  diagnostic records. Current categories are conservative labels such as
  `state_authority_absent_or_extraction_failure`, `state_readout_failure`,
  `state_extraction_no_state`, `state_extraction_missing_expected_slot`,
  `state_extraction_missing_active_expected_slot`, `state_routing_failure`,
  `retrieval_failure`, `stale_adjudication_failure`,
  `premise_correction_failure`, and `ranking_failure`; use them as triage
  signals, not as final causal claims without representative case inspection.
- Representative examples by failure attribution in JSONL and STALE diagnostic
  reports, so paper error analysis can inspect concrete cases for each
  machine-labeled attribution category.
- `trajectory_step_readout`, a narrow step-aware trajectory retrieval
  ablation. It authorizes matching `Step N` or short step-range memories by
  trajectory metadata and is useful for separating step evidence recall from
  answer generation on AMA-style questions.
- Deterministic AMA trajectory answer-basis diagnostics. The basis summarizes
  retrieved step actions, observations, active rules, blocked actions,
  no-progress repetitions, inverse action pairs, and repeated observations
  without reading gold answers; answer labels are used only to compute
  keyword-recall diagnostics.
- `adamem.pilot ama-public`, a reproducible API-free public AMA pilot command
  that downloads or copies a bounded JSONL prefix, converts it, runs selected
  baselines, writes Markdown reports, records JSONL, and compact experiment
  JSON. Use `--answer-only` for larger smoke runs because answer-mode reports
  already include evidence support and answerability diagnostics.
  `--run-answer-generation` additionally runs the shared answer-eval path and
  writes separate `.generation.*` artifacts. Pilot stage outputs include the
  stage name in each filename, such as `.answer.records.jsonl`,
  `.evidence.records.jsonl`, and `.generation.records.jsonl`, to prevent
  retrieval and answer-generation results from overwriting each other.
- `adamem.tables`, a compact paper-table summarizer for benchmark records or
  experiment JSON files. It follows `notes.records_path` when raw outputs are
  not embedded, and emits Markdown or JSON tables with overall and grouped
  support, evidence, answerability, and structured-basis diagnostics. It also
  auto-detects answer-generation records and switches to end-to-end `correct`
  / `accuracy` tables for generation reports, and auto-detects STALE
  LLM-judge raw outputs for `dim` / `stale_type` accuracy and stale-leak
  tables.
- `adamem.answer_eval`, a mockable answer-generation evaluation path for
  AdaMem JSONL cases. It separates runtime answer generation from evaluation
  scoring, supports deterministic substring smoke tests and LLM-judge scoring,
  records prompts/raw outputs in experiment JSON, and reports grouped answer
  accuracy by metadata fields such as AMA `question_type`.
- `adamem.claims`, a claim/evidence audit command for experiment JSON files.
  It reports which paper claims an artifact can support, which claims are
  blocked, ground-truth runtime-use notes, provider settings, and embedded or
  sidecar record counts. It recognizes LongMemEval-V2 prepared pilots as
  prepared-split readiness and retrieval answer-string support diagnostics,
  not as answer accuracy. When a prepared-pilot experiment includes
  `state_evidence_summary_path`, the audit also records the selected split's
  state-available question count and state-evidence candidate count. When
  case records include `failure_attributions`, the audit also records
  attribution counts and representative examples as mechanism error-analysis
  evidence, not as answer-accuracy or SOTA evidence.
- `adamem.reporting`, a report-bundle command that combines `adamem.tables`
  and `adamem.claims` for one experiment JSON and writes paper tables, claim
  audit files, method coverage, paper-readiness artifacts, next-step
  checklists, and a manifest. For a single experiment, the bundle writes
  `<run>.method_coverage.json/md`, `<run>.paper_readiness.json/md`, and
  `<run>.paper_next_steps.md`, and embeds the same summaries in the manifest.
  This makes one expensive STALE/API pilot self-auditing before it is merged
  into a larger result directory. If the input is a directory, it batches every
  `*experiment.json` file into per-experiment sub-bundles and a top-level
  batch manifest. Bundle manifests include supported claims, blocked claims,
  claim evidence, diagnostic evidence, method coverage, paper readiness, and
  warnings so large experiment directories can be triaged without opening
  every audit JSON. Batch mode also writes `claim_matrix.json` and
  `claim_matrix.md`, which flatten
  per-experiment claim evidence such as state-evidence coverage, paired
  no-regression counts, and top failure-attribution counts for paper-track
  screening. Each row includes a `readiness_gate` such as `diagnostic_ready`,
  `answer_candidate`, `sota_candidate`, or `needs_attention`, plus explicit
  reasons so paper scripts can filter results without parsing prose. Mechanism
  error-analysis claims count as diagnostic-ready only when the usual scope,
  warning, and raw-record gates pass. Batch mode also writes
  `paper_next_steps.md`, a deterministic action checklist that maps each
  experiment to the next evidence step, such as rerunning on a public/full
  benchmark, exporting case-level records, inspecting representative failure
  attributions, running end-to-end answer/judge evaluation, or adding strong
  baselines and judge robustness. Claim audits also record baseline coverage
  across required paper groups: a raw retrieval reference, at least one
  mainstream memory approximation, and an AdaMem/state ablation. Missing
  groups appear in `claim_matrix` and trigger `add_missing_baseline_categories`
  in `paper_next_steps.md`. Answer-generation and STALE judge audits also
  record answer-model and judge-model coverage. Missing multiple-answer-model,
  multiple-judge-model, or semantic-LLM-judge requirements appear as model
  gaps and trigger `add_model_or_judge_robustness_runs`. Claim audits also
  compute single-experiment baseline-reproduction evidence from artifact-level
  provenance and keep SOTA blocked when a run has only API-free mainstream
  approximations rather than official or faithful reproductions. They also
  record reproducibility packets covering schema version, commit, command,
  dataset, baseline configs, baseline provenance, case-level records,
  provider/model settings, prompt templates, and retrieval context settings
  where applicable. Missing packet fields appear as reproducibility gaps and
  trigger `complete_reproducibility_packet`. Batch report bundles also write
  `study_model_coverage.json` and `study_model_coverage.md`, which merge
  comparable answer/judge experiments by run type, dataset, split, and baseline
  set so one-model-per-run API sweeps can still be audited as one study. Batch
  mode also writes `benchmark_coverage.json` and `benchmark_coverage.md`,
  which check whether a result directory covers the primary STALE benchmark,
  at least one transfer benchmark, and at least one public/full-scope
  experiment. Batch mode also writes `method_coverage.json` and
  `method_coverage.md`, which check whether the directory includes a raw
  retrieval reference, a mainstream memory approximation, the proposed
  state-aware method, and mechanism ablations. It also flags missing named
  mechanisms such as state readout, dependency propagation, source
  adjudication, premise correction, LLM state extraction, and trajectory-step
  readout. The same method audit records baseline provenance and separates
  API-free mainstream approximations from official or faithful reproductions,
  so SOTA-style claims remain blocked until strong baselines are actually
  reproduced. It also emits a baseline reproduction plan for mainstream
  approximations, including the exact reference implementation target where
  known. Current targets are the A-MEM paper reproduction repository,
  Graphiti/Zep's temporal context graph engine, and the official Mem0
  implementation. Report bundles prefer artifact-level `baseline_provenance`
  over the current registry when computing method coverage, which keeps old
  runs and future official baseline reproductions auditable after code
  changes.
  Top-level `paper_readiness.json` and `paper_readiness.md` summarize
  directory-level gate counts, top next actions, complete vs incomplete
  study-level model coverage, benchmark coverage gaps, method coverage gaps,
  and baseline-reproduction gaps. They also expose `paper_claim_ready` and
  `paper_claim_blockers`, a conservative machine-readable gate for whether the
  current artifact set can support the paper's main empirical claim. A run can
  be `answer_candidate_with_model_coverage` while still having paper-claim
  blockers such as missing official/faithful baseline reproduction, missing
  named mechanism ablations, or incomplete benchmark coverage.
- `adamem.compare`, a paired baseline comparison command for retrieval,
  answer-generation, and STALE judge records. Report bundles include its
  Markdown/JSON artifacts so paper tables can report gained/lost/net records
  against a reference baseline.
- `adamem.study_plan`, a pre-run paper-study planner that writes
  `paper_study_plan.json`, `paper_study_plan.md`, and
  `paper_study_commands.sh`, plus `paper_study_validation.json/md`. It expands
  answer/judge model combinations, prepends STALE and LongMemEval conversion
  commands when source paths are configured, adds the API-free STALE diagnostic
  command, includes an LLM state-extractor ablation, includes LongMemEval and
  AMA transfer commands, and appends the batch reporting command. Its
  validation report checks local dataset paths, whether missing targets can be
  prepared by included conversion commands, placeholder models, model
  robustness counts, method coverage, and reporting command presence. Its
  default artifact policy keeps generated full benchmark datasets under
  `OUTPUT_DIR/data/` rather than tracked fixtures. The reporting command
  declares the key batch outputs it must produce, including `batch_manifest`,
  `claim_matrix`, `method_coverage`, `benchmark_coverage`, and
  `paper_readiness` artifacts, so dry runs and resume checks can detect an
  incomplete reporting stage. Its method-coverage preview is a planning check
  only; final claims must use the generated experiment records and report
  bundle audits.
- `adamem.study_plan --profile smoke`, an API-free local rehearsal profile
  over `benchmarks/stale_mini.jsonl` and
  `benchmarks/dynamic_state_transfer.jsonl` with mock LLM providers. Use it to
  verify the generated runbook, experiment writers, and batch reporting before
  spending API budget. Treat all smoke outputs as plumbing evidence only.
- `adamem.study_plan --run`, a plan runner that validates the generated plan,
  executes selected stages, and writes JSONL command records plus Markdown/JSON
  run summaries. Each record checks declared output paths so a command that
  returns successfully but fails to create its experiment/report artifact can
  still be caught in the run log. Use `--dry-run` and `--stage ...` before real
  API execution.
- `adamem.study_plan --plan`, a saved-plan loader for executing or validating
  an edited `paper_study_plan.json`. This is the intended path after replacing
  placeholder provider/model labels with real API-backed settings.
- Study plans carry a stable SHA-256 fingerprint. Validation and run summaries
  report the current fingerprint and whether it matches the recorded
  fingerprint, so edited plan JSON files can be traced precisely.
- `--max-cases` and `--experiment-output` support for `--dataset` runs, so
  converted public benchmark pilots can be recorded without API keys.
- STALE selection flags `--stale-types` and `--limit-per-stale-type`, so
  diagnostics and LLM-judge pilots can use reproducible T1/T2-balanced subsets
  after full STALE data is converted.
- Per-query JSONL benchmark metadata in result traces, enabling breakdowns by
  fields such as LongMemEval `question_type` or local `dimension`.
- JSONL retrieval failure records and Markdown reports via
  `--benchmark-cases-output` and `--benchmark-report-output`.
- Pairwise comparison diagnostics for retrieval-support runs, useful for
  non-degradation checks such as `semantic_state_readout` vs `semantic_only`.
- Claim audits and batch claim matrices tag dataset scope. Mini, smoke, debug,
  tmp-path, and local synthetic fixtures are marked claim-limited and force the
  batch readiness gate to `needs_attention`, even when diagnostic metrics look
  good. Use them for debugging only; paper-level claims require public/full
  benchmark scope plus the relevant answer or diagnostic evidence.
- LongMemEval balanced conversion through `--limit-per-type`, enabling small
  public benchmark pilots that cover all question types before scaling.
- Optional LongMemEval query-state diagnostics through
  `--infer-state-slots`. This labels query metadata from query text only, is
  evaluation-only, and is useful for measuring whether the runtime memory
  produced an authorized state readout for naturally occurring public
  benchmark questions.
- Manual-audit path for public state-sensitive subsets:
  `--state-audit-output` writes query-state candidates for review, and
  `--state-audit-input` imports only records explicitly marked
  `is_state_sensitive: true`. These labels are written only to query metadata,
  never to observation metadata. Reviewed records can also set
  `state_available: false` when the query is state-sensitive but the haystack
  lacks a reliable current state; these cases are reported separately and do
  not count as state-readout-missing failures. Candidate records include
  `state_evidence_candidates` from the deterministic observation-side state
  extractor so reviewers can inspect whether the current prototype found any
  matching state evidence without using answer labels.
- LongMemEval audit summary output through `--state-audit-summary-output`.
  The summary counts candidate volume, state-evidence coverage, and
  breakdowns by inferred state slot and LongMemEval `question_type`. Use this
  before manual review to decide whether a public transfer dataset has enough
  state-available cases to justify API spending.
- Query-scoped state-source adjudication, enabling semantic-only state
  ablations that suppress superseded raw evidence only when the query is routed
  to the affected state slot.
- LongMemEval-V2 prepared state-evidence audit through
  `adamem.lme_v2 state-evidence-audit`. It scans selected trajectory runtime
  text with the deterministic state extractor, reports matching evidence by
  split, question type, and state slot, and keeps reference answers out of the
  audit path.

Next API-free work:

1. Prepare scripts so API-enabled pilot runs need only provider/model/key
   settings. Intentional saved-plan edits should be followed by
   `--refresh-fingerprint`; accidental fingerprint mismatch should be treated
   as a reproducibility blocker.
2. Run retrieval-only diagnostics on larger converted STALE data when available
   and extend the failure taxonomy with representative cases.
3. Scale the public AMA-Bench trajectory pilot beyond the first 20 episodes
   and add step-level answer synthesis or LLM judge evaluation. The first five
   public samples converted successfully, but `semantic_only` and `full`
   scored `0/60` answer support and `0/60` evidence support.
   `trajectory_step_readout` improves evidence support on those same samples
   to `60/60`, while answer-string support remains `0/60`. The simple
   trajectory basis improved answer-keyword recall only from `22.73%` to
   `24.81%`, but the structured rule/blocking/no-progress basis reaches
   `32.25%` and matched queries improve from `8/60` to `20/60`. The next
   API-free 20-episode light pilot confirms evidence support scales:
   `trajectory_step_readout` reaches `239/239` evidence support versus
   `34/239` for `semantic_only`, and basis keyword recall `24.34%` versus
   `15.68%`. After bounded candidate pools and bounded soft-stale propagation,
   the 20-episode pilot including `full` finishes in about 33 seconds locally;
   `full` remains at `0/239` evidence support, while `trajectory_step_readout`
   remains at `239/239`. The next method work should scale this beyond 20
   episodes and add API-backed judge robustness over correctly recalled
   trajectory steps.
   Per-type diagnostics on the same run show `trajectory_step_readout` reaches
   full evidence support for every AMA type: A `79/79`, B `60/60`, C `60/60`,
   and D `40/40`; structured basis recall is also higher than semantic-only in
   every type.
4. Build a reliable public state-sensitive transfer subset. The first
   LongMemEval-S inferred-state pilot exposed query-router false positives;
   after word-boundary matching and intent gates, the balanced 60-case sample
   marks only 1/60 query as state-sensitive and the full 500-case file yields
   14 candidates with 0 deterministic state-evidence candidates. Treat
   LongMemEval-S as a broad retrieval-transfer/no-regression check, not as the
   main state-available transfer benchmark for AdaMem.
5. Run larger STALE retrieval diagnostics with `semantic_state_adjudication`
   and `semantic_state_propagation_adjudication` to check whether the mini
   stale-exposure gains hold beyond two smoke cases.
6. Broaden typed state slots beyond location, beverage preference, schedule
   availability, and task status.
7. Add faithful local baselines for mainstream memory systems after reviewing
   their code and licenses.
8. Run the LLM extractor ablations with real provider keys on STALE and at
   least one transfer benchmark, then compare against deterministic extraction
   to separate extraction failures from memory-management failures.
9. Keep `docs/progress_log.md` updated after each meaningful design decision,
   experiment, implementation change, or scope change.

## Experiment Record Template

Use this template for every meaningful run.

```text
Run name:
Date:
Commit:
Dataset:
Split or case limit:
Baseline/config:
Answer model:
Judge model:
Prompt version:
Top-k:
Context budget:
Temperature:

Purpose:

Expected mechanism:

Main results:

Diagnostics:

Representative failures:

Decision:
```
