# AdaMem Progress Log

This document is the running project memory for continued development. Update
it whenever a meaningful design decision, experiment, implementation change, or
scope change happens.

## Current Target

Build AdaMem into a CCF-A-level research project on stale-memory handling in
LLM agent memory systems.

Primary benchmark:

- STALE-style stale and invalidated memory evaluation.

Primary method direction:

- Move beyond similarity-only memory retrieval toward state-aware memory
  adjudication and authorized current-state readout.

## Current Phase

Phase 1 and Phase 3 from `docs/research_workflow.md`: Evaluation Harness
Hardening plus the first state-aware method prototype.

Retrieval diagnostics, stable baseline configs, experiment records, and a
deterministic state-aware prototype are now available. The current best
API-free variant combines authorized state readout with query-scoped
state-source adjudication. The deterministic state extractor now covers
location, schedule availability, beverage preference, task status,
health/dietary constraints, resource status, workflow/runbook rules, and
runtime/tool status. The latest public-transfer diagnostic shows the next
method bottleneck clearly: state-sensitive query routing must be precise before
state-readout metrics are meaningful. The first LongMemEval-S inferred-state
run overcounted ordinary episodic questions as state queries; the router now
uses word-boundary matching and slot-specific intent gates. The full
LongMemEval-S audit now produces only 14 query-state candidates and 0
deterministic state-evidence candidates, so LongMemEval-S should be treated as
a broad retrieval no-regression check rather than the main state-available
transfer benchmark. The next high-value task is to build or select a reliable
public state-sensitive transfer subset, then improve observation-side state
extraction on those true state cases.

## Confirmed Project Constraints

- Research paper quality is prioritized over production packaging.
- The core contribution should be a memory mechanism, representation, or
  retrieval policy, not a complete agent harness.
- API-dependent evaluation can wait until provider keys are available.
- API-free work should prepare the harness, baselines, diagnostics, method
  design, and reproducibility path.
- STALE ground-truth fields must not be used inside proposed runtime memory.
- Breaking API changes are allowed when useful, but must be documented.

## Resume Checkpoint

### 2026-05-30 end-of-day

- Current committed code has a reproducible public AMA pilot, bounded full
  baseline runtime, trajectory-step readout, structured answerability
  diagnostics, and grouped per-`question_type` diagnostics.
- The strongest API-free result so far is retrieval-side: on the first 20
  public AMA episodes, `trajectory_step_readout` recovers labeled step evidence
  across A/B/C/D question types, while generic semantic and full scoring do not.
- Current evidence does not support an answer-accuracy or SOTA claim. Exact
  answer-string support is still zero in the deterministic pilot, so the next
  paper-facing work should separate evidence recall, answerability, and final
  answer scoring rather than treating retrieval diagnostics as task accuracy.
- Suggested first task when resuming: add a compact paper-table summary utility
  that reads experiment JSON or records JSONL and emits reproducible Markdown /
  JSON tables for overall and grouped diagnostics. This will make future public
  pilots and API-backed runs easier to compare without manual report parsing.
- Suggested second task: design the first mockable AMA answer-generation /
  judge path so API keys can be plugged in later without changing the benchmark
  record format.

### 2026-05-31

- Added `src/adamem/tables.py` and the `adamem-tables` console script:
  - Reads benchmark records from JSONL, JSON arrays, or compact experiment JSON
    files.
  - When experiment JSON does not embed raw outputs, follows
    `notes.records_path` to the sidecar records file.
  - Emits Markdown or JSON paper-table summaries with overall support,
    evidence support, answer-keyword recall, structured-basis recall, and
    grouped metadata breakdowns.
- Added deterministic tests for paper-table summaries, Markdown formatting,
  experiment-record path resolution, and JSON output writing.
- Generated API-free paper tables from the existing 20-episode public AMA
  pilot:
  - Markdown:
    `results/ama_public_20_full/ama_public_20.paper_tables.md`
  - JSON:
    `results/ama_public_20_full/ama_public_20.paper_tables.json`
  - Overall table:
    - `full`: support `0/240`, evidence support `0/239`, answer recall
      `19.07%`, basis recall `19.07%`, basis matched `19/240`.
    - `semantic_only`: support `0/240`, evidence support `34/239`, answer
      recall `15.36%`, basis recall `15.68%`, basis matched `14/240`.
    - `trajectory_step_readout`: support `0/240`, evidence support `239/239`,
      answer recall `20.54%`, basis recall `24.34%`, basis matched `32/240`.
  - Interpretation: this strengthens the reproducibility and reporting path,
    but still supports only a retrieval/answerability claim, not final
    answer-accuracy or SOTA claims.
- Added `src/adamem/answer_eval.py` and the `adamem-answer-eval` console
  script:
  - Runs answer generation over AdaMem JSONL cases using the same baseline
    registry as retrieval benchmarks.
  - Keeps answer generation grounded in retrieved memory excerpts and separates
    runtime answer generation from evaluation scoring.
  - Provides deterministic `SubstringAnswerScorer` for API-free harness smoke
    tests and `LLMAnswerScorer` for later answer/judge providers.
  - Writes case records and experiment JSON with prompts, raw answers, scorer
    outputs, model/provider notes, and explicit `ground_truth_runtime_use:
    forbidden`.
- Added tests for answer benchmark aggregation, LLM-judge prompt recording, and
  CLI records/experiment output.
- Ran an API-free answer-eval smoke command:
  `PYTHONPATH=src python -m adamem.answer_eval --dataset benchmarks/tiny_memory_qa.jsonl --baselines semantic_only --answer-provider mock --mock-answer "Office door code is 9876." --records-output /tmp/adamem_answer_eval_records.jsonl --experiment-output /tmp/adamem_answer_eval_experiment.json`
  - Result: command completed and wrote records/experiment artifacts.
  - Interpretation: this validates the answer-evaluation plumbing only; the
    mock answer is not a benchmark result.
- Integrated answer generation into `adamem.pilot ama-public` as an explicit
  optional stage:
  - `--run-answer-generation` runs the shared answer-eval path after answer
    conversion.
  - Retrieval diagnostics still run by default; answer generation is opt-in so
    API-free retrieval pilots remain unchanged.
  - Stage artifacts now include stage names, e.g.
    `.answer.records.jsonl`, `.evidence.records.jsonl`, and
    `.generation.records.jsonl`. This fixes a discovered artifact-collision
    bug where `Path.with_suffix()` caused different pilot stages to write the
    same `ama_public_N.records.jsonl` path.
- Added pilot tests for distinct stage artifact names and the optional
  answer-generation stage.
- Ran an AMA pilot answer-generation CLI smoke:
  `PYTHONPATH=src python -m adamem.pilot ama-public --limit 1 --source results/ama_public_20_light/ama_public_20.raw.jsonl --output-dir /tmp/adamem_ama_answer_generation_smoke --baselines trajectory_step_readout --top-k 4 --answer-only --run-answer-generation --answer-provider mock --mock-answer "The memory does not provide enough information." --json`
  - Result: wrote separate
    `/tmp/adamem_ama_answer_generation_smoke/ama_public_1.answer.records.jsonl`
    and
    `/tmp/adamem_ama_answer_generation_smoke/ama_public_1.generation.records.jsonl`.
  - Generation summary: `trajectory_step_readout` correct `0/12` with the mock
    insufficiency answer.
  - Interpretation: this is still a plumbing smoke test, not an answer-quality
    benchmark.
- Added grouped answer-generation diagnostics:
  - `answer_failure_summary` now aggregates end-to-end answer records by
    baseline and metadata fields such as `question_type`, `dimension`,
    `state_slot`, and `abstention`.
  - `answer_report` now includes `By question_type` / related grouped answer
    accuracy tables when metadata is available.
  - AMA pilot generation experiments now store the grouped summary in
    `diagnostics.failure_summary` and write grouped Markdown reports.
- Re-ran the AMA pilot answer-generation smoke:
  - Generation records path:
    `/tmp/adamem_ama_answer_generation_smoke/ama_public_1.generation.records.jsonl`
  - Grouped generation report included `A`, `B`, `C`, and `D` question-type
    rows.
  - Because the answer provider was a fixed mock insufficiency answer, grouped
    accuracy was `0/12`; this validates reporting only, not method quality.
- Extended `adamem.tables` to auto-detect answer-generation records:
  - Retrieval records still produce support/evidence/answerability tables.
  - Generation records now produce end-to-end `correct` / `accuracy` paper
    tables, including grouped metadata tables such as AMA `question_type`.
  - Experiment JSON inputs still work through `notes.records_path`.
- Ran a generation-table CLI smoke:
  `PYTHONPATH=src python -m adamem.tables /tmp/adamem_ama_answer_generation_smoke/ama_public_1.generation.records.jsonl --group-fields question_type --title "AMA Generation Answer Tables" --output /tmp/adamem_ama_answer_generation_smoke/ama_public_1.generation.paper_tables.md`
  - Result: Markdown table reported `trajectory_step_readout` correct `0/12`
    overall and A/B/C/D grouped rows.
  - Interpretation: when real answer/judge providers are available, the same
    table command can produce paper-ready end-to-end answer accuracy tables
    from `.generation.records.jsonl` or `.generation.experiment.json`.
- Extended `adamem.tables` to auto-detect STALE LLM-judge raw outputs:
  - Records with `judge_correct` now produce overall `correct` / `accuracy`
    and stale-leak tables.
  - Default grouped tables for STALE are `By dim` and `By stale_type`.
  - This makes STALE's primary paper columns reproducible from
    `--experiment-output` JSON without hand-parsing `stale_report`.
- Ran a mock STALE judge table smoke:
  `PYTHONPATH=src python -m adamem.eval --stale benchmarks/stale_mini.jsonl --baselines semantic_only --answer-provider mock --judge-provider mock --max-cases 1 --experiment-output /tmp/adamem_stale_mock_judge_experiment.json`
  then
  `PYTHONPATH=src python -m adamem.tables /tmp/adamem_stale_mock_judge_experiment.json --title "STALE Mock Judge Tables" --output /tmp/adamem_stale_mock_judge_tables.md`
  - Result: table reported overall `3/3`, `By dim` rows for `1`, `2`, `3`,
    and `By stale_type` row for `T1`.
  - Interpretation: mock providers validate the STALE table workflow only;
    real method claims still require real answer/judge models and robustness
    checks.
- Added `src/adamem/claims.py` and the `adamem-claims` console script:
  - Reads experiment JSON and reports supported claims, blocked claims,
    warnings, provider settings, ground-truth runtime-use notes, and evidence
    record counts.
  - Follows compact experiment `notes.records_path` sidecars when raw outputs
    are not embedded.
  - Blocks answer-accuracy claims for retrieval/answerability runs.
  - Blocks answer-accuracy claims for mock answer/judge providers and
    substring-only scoring.
  - Blocks SOTA claims unless stronger evidence is added later; current audit
    requires more than a single smoke run and flags missing strong-baseline /
    robustness evidence.
- Ran claim-audit smoke commands:
  - `PYTHONPATH=src python -m adamem.claims results/ama_public_20_full/ama_public_20.experiment.json`
    reported `retrieval_diagnostics` and `answerability_diagnostics` as
    supported, blocked `answer_accuracy` and `sota`, and counted `720`
    sidecar records.
  - `PYTHONPATH=src python -m adamem.claims /tmp/adamem_stale_mock_judge_experiment.json`
    reported `stale_judge_plumbing`, blocked `stale_answer_accuracy` because
    providers were mock, and blocked `sota`.
- Added `src/adamem/reporting.py` and the `adamem-report` console script:
  - Takes one experiment JSON and writes a report bundle containing paper
    tables, claim-audit Markdown/JSON, and a manifest.
  - Uses `adamem.tables` for table generation and `adamem.claims` for claim
    boundary checks.
  - Keeps table generation and claim audit in one reproducible post-run step.
- Ran a report-bundle smoke:
  `PYTHONPATH=src python -m adamem.reporting results/ama_public_20_full/ama_public_20.experiment.json --output-dir /tmp/adamem_report_bundle_smoke --group-fields question_type --title "AMA Public 20 Bundle" --json`
  - Result: generated `/tmp/adamem_report_bundle_smoke/ama_public_20.experiment.paper_tables.md`,
    `/tmp/adamem_report_bundle_smoke/ama_public_20.experiment.claim_audit.md`,
    JSON counterparts, and a manifest.
  - The bundle table reproduced the 20-episode AMA retrieval/answerability
    diagnostics, while the claim audit blocked answer-accuracy and SOTA claims.
- Extended `adamem.reporting` with directory batch mode:
  - If the input path is a directory, it scans `*experiment.json`, writes one
    sub-bundle per experiment, and emits a top-level `batch_manifest.json`.
  - This is useful for AMA public pilots because a single output directory can
    contain answerability, evidence, and generation experiment files.
- Ran a batch reporting smoke:
  `PYTHONPATH=src python -m adamem.reporting /tmp/adamem_ama_answer_generation_smoke --output-dir /tmp/adamem_report_batch_smoke --group-fields question_type --json`
  - Result: found `2` experiments and produced one retrieval bundle plus one
    answer-generation bundle.
  - The batch manifest recorded record kinds `retrieval` and
    `answer_generation`.
- Added `src/adamem/compare.py` and the `adamem-compare` console script:
  - Computes paired baseline comparisons on shared `(case_id, query_id)` keys.
  - Supports retrieval records, answer-generation records, and STALE judge raw
    outputs.
  - Reports gained, lost, net delta, both-correct, both-wrong, and a two-sided
    paired sign-test p value.
  - Retrieval comparison auto-selects `evidence_support_matched` when records
    contain expected evidence labels, otherwise it uses exact `passed` support.
  - Report bundles now include paired-comparison Markdown/JSON artifacts.
- Re-ran the AMA 20 report-bundle smoke with paired comparisons:
  - Comparison metric: `evidence_support_matched`.
  - Reference: `semantic_only`.
  - `trajectory_step_readout` vs `semantic_only`: common `240`, gained `205`,
    lost `0`, net `+205`.
  - By AMA question type, trajectory-step readout gained A `67`, B `52`, C
    `50`, D `36` evidence-supported records with no losses.
  - Interpretation: this is a strong paired retrieval/evidence-support signal,
    but still not an answer-accuracy or SOTA claim.

## Completed Work

### 2026-05-29

- Added `src/adamem/diagnostics.py` for API-free STALE retrieval diagnostics.
- Added `src/adamem/baselines.py` for stable baseline names, categories,
  descriptions, and runnable configs.
- Added `src/adamem/state.py` as the first deterministic state-aware memory
  prototype. It extracts user location updates without API calls or STALE
  labels, writes derived `state` memory items, supersedes older state values,
  and supports current-state readout for location-sensitive queries.
- Extended `src/adamem/state.py` beyond location:
  - Added typed `preference.beverage` extraction.
  - Added typed `schedule.availability` extraction.
  - Added dynamic `task.*.status` extraction for task or project status
    updates such as blocked, completed, resolved, pending, paused, and
    cancelled.
  - Added slot-aware readout routing so location and preference states are not
    mixed for unrelated queries.
  - Added wildcard state readout matching so dynamic slots such as
    `task.checkout_migration.status` can be retrieved by status-oriented
    queries without enumerating every task name.
  - Added a pluggable `state_extractor` hook to `AdaMem`, preserving the same
    memory/readout mechanism for future LLM or domain extractors.
- Added state-slot dependency propagation:
  - New `use_state_dependency_propagation` config flag.
  - New `state_propagation` baseline.
  - Initial topology invalidates `local.*`, `commute.*`,
    `schedule.local.*`, and `timezone.*` state slots when the active location
    changes.
  - Dependent state source evidence is also marked stale so raw episode
    retrieval cannot bypass the state authority layer.
- Added `src/adamem/experiments.py` for JSON experiment records with schema
  version, command, commit, dataset, baseline configs, diagnostics, prompts,
  raw outputs, and notes.
- Added `docs/literature_to_design.md` to connect current memory papers and
  SOTA-style systems to AdaMem's design hypotheses, contribution candidates,
  baseline requirements, and evaluation gates.
- Added `benchmarks/dynamic_state_transfer.jsonl` as a non-STALE dynamic-state
  transfer smoke fixture. It covers schedule availability, task status, and
  beverage preference without using STALE labels or judge metadata.
- Added LongMemEval converter support:
  - `PYTHONPATH=src python -m adamem.convert longmemeval INPUT OUTPUT`
  - Supports official fields `question_id`, `question_type`, `question`,
    `answer`, `question_date`, `haystack_session_ids`, `haystack_dates`,
    `haystack_sessions`, and `answer_session_ids`.
  - Stores answers and evidence session ids only on query metadata /
    expected-substring fields for evaluation.
  - Does not write `answer_session_ids` or turn-level `has_answer` labels into
    observation metadata, so runtime retrieval cannot use those ground-truth
    evidence labels.
- Added JSONL retrieval benchmark experiment recording:
  - `PYTHONPATH=src python -m adamem.eval --dataset INPUT --max-cases N --experiment-output OUTPUT`
  - Records baseline configs, aggregate retrieval-support results, per-query
    pass/fail checks, query metadata, retrieved text, trace data, command,
    commit, dataset path, case limit, and notes that no answer or judge model
    is required.
  - This gives local fixtures and converted public benchmark pilots the same
    audit trail style as STALE diagnostics and STALE LLM-judge runs.
- Added focused baseline selection:
  - `PYTHONPATH=src python -m adamem.eval --baselines semantic_only state_readout ...`
  - Applies to baseline listing, JSONL retrieval benchmarks, STALE
    diagnostics, and STALE LLM-judge runs.
  - Preserves canonical baseline names and requested order in reports and
    experiment records, making small public-benchmark pilots practical.
- Added per-query metadata propagation for JSONL benchmark results. This keeps
  fields such as LongMemEval `question_type` and local `dimension` available in
  retrieval traces for later breakdowns and error analysis.
- Added state readout authorization boundary:
  - New `use_state_readout_authorization` config flag, enabled by default.
  - Derived `state` memories are hidden from ordinary direct retrieval and can
    enter results through authorized state readout only.
  - The flag can be disabled for explicit ablations that test uncontrolled
    state-memory exposure.
  - Task-status routing now requires state/status intent terms such as
    `status`, `resolved`, `blocked`, or `pending`; generic project-count
    questions no longer trigger `task.*.status` readout.
- Added semantic-only state-aware ablations:
  - `semantic_state_readout`: semantic-only retrieval plus deterministic state
    extraction and authorized current-state readout.
  - `semantic_state_propagation`: `semantic_state_readout` plus typed state
    dependency propagation.
  - These isolate the state mechanism from default full AdaMem scoring.
- Added JSONL retrieval failure analysis:
  - `--benchmark-cases-output` writes per-query records with missing expected
    support, present forbidden support, failure modes, retrieved text, trace
    data, query metadata, and baseline name.
  - `--benchmark-report-output` writes a Markdown report grouped by metadata
    fields such as LongMemEval `question_type`, local `dimension`, state slot,
    and abstention.
  - JSONL experiment records now include a `failure_summary` diagnostic for
    retrieval-support runs.
- Added pairwise JSONL retrieval comparison diagnostics:
  - The first requested baseline is treated as the reference.
  - Each later baseline records common query count, gained passes, lost passes,
    net delta, both-pass count, both-fail count, and the same comparison broken
    down by metadata fields such as `question_type` and `dimension`.
  - Markdown reports include a `Pairwise Vs <baseline>` table.
- Added balanced LongMemEval conversion support:
  - `--question-types ...` filters official LongMemEval samples by
    `question_type`.
  - `--limit-per-type N` keeps at most `N` examples per question type, enabling
    small public benchmark pilots that are not biased toward the first
    sequential examples.
- Added CLI support:
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2`
  - `PYTHONPATH=src python -m adamem.eval --list-baselines`
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 1 --experiment-output /tmp/adamem-diagnostic-record.json`
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output /tmp/adamem_stale_failures.jsonl`
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output /tmp/adamem_cases.jsonl --diagnostic-report-output /tmp/adamem_failure_report.md --experiment-output /tmp/adamem_diag_record.json`
- Added tests for stricter STALE signal matching and retrieval diagnostics.
- Added diagnostic failure-case export for API-free STALE runs. Each record
  includes baseline, case/query ids, dimension, stale type, failure modes,
  current/stale rank signals, old-support adjudication counts, and optional
  retrieved traces.
- Added diagnostic failure report aggregation for paper-style error analysis:
  - Counts by failure mode, baseline, STALE dimension, and stale type.
  - Counts for analysis flags such as stale premise correction opportunities.
  - Failure-mode-by-baseline table.
  - Representative examples per failure mode.
  - Failure summary stored in experiment records when
    `--experiment-output` is used.
- Added STALE LLM-judge pilot recording:
  - `--stale ... --experiment-output ...` now writes a JSON experiment record.
  - Records include baseline configs, aggregate results, answer/judge prompt
    templates, per-query answer prompts, judge prompts, raw answer outputs, raw
    judge outputs, retrieved traces, model/provider settings, top-k,
    max-context chars, request delay, command, and commit.
  - Runtime ground-truth use remains forbidden; judge ground-truth use is
    explicitly recorded as allowed.
- Added tests for the baseline registry and experiment record writer.
- Added harder state-aware unit tests mapped to STALE dimensions:
  - State Resolution: stale city question should surface current location.
  - Premise Resistance: stale Seattle premise should be overridden by current
    Boston state.
  - Implicit Policy Adaptation: nearby-place query should receive current
    location even without naming old or new state.
  These tests deliberately disable importance, recency, temporal, graph, MMR,
  and confidence scoring to isolate the `state_readout` mechanism.
- Added non-location state tests:
  - Beverage preference readout surfaces current tea preference without
    surfacing current location.
  - Schedule availability readout surfaces meeting availability without
    surfacing beverage state.
  - Dynamic task status readout supersedes blocked with resolved for the same
    task slot.
  - Custom extractor can inject a domain-specific state slot through the
    pluggable extractor hook.
- Added dynamic-state transfer adapter test:
  - `semantic_only` fails the local fixture.
  - `state_readout` and `state_propagation` pass all dynamic state queries.
  - This is a local smoke check only, not a public benchmark or paper claim.
- Added LongMemEval converter test:
  - Converts a toy `knowledge-update` instance to AdaMem JSONL.
  - Checks answer/evidence labels remain evaluation-only and do not enter
    observation metadata.
  - Checks the converted case can run through the JSONL benchmark adapter.
- Added indirect invalidation test:
  - A custom extractor writes `location` and `local.gym` state.
  - Moving from Seattle to Boston supersedes the location and invalidates the
    Seattle gym state plus its source evidence only when dependency propagation
    is enabled.
- Ran deterministic local tests:
  - `python -m pytest`
  - Result: `37 passed`.
- Re-ran deterministic local tests after JSONL experiment-record support:
  - `python -m pytest`
  - Result: `39 passed`.
- Re-ran deterministic local tests after baseline filtering, query metadata
  traces, and official LongMemEval-S pilot workflow:
  - `python -m pytest`
  - Result: `42 passed`.
- Re-ran deterministic local tests after JSONL failure reports and balanced
  LongMemEval conversion:
  - `python -m pytest`
  - Result: `44 passed`.
- Re-ran deterministic local tests after state readout authorization boundary
  and semantic-state ablations:
  - `python -m pytest`
  - Result: `47 passed`.
- Re-ran deterministic local tests after pairwise JSONL benchmark diagnostics
  and scaled LongMemEval-S pilot:
  - `python -m pytest`
  - Result: `48 passed`.
- Ran converter smoke checks after adding LongMemEval sampling options:
  - `PYTHONPATH=src python -m adamem.convert locomo benchmarks/locomo_mini.json /tmp/locomo_mini_smoke.jsonl`
  - Result: `wrote 1 cases`.
  - `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_ku2.adamem.jsonl --question-types knowledge-update --limit-per-type 2`
  - Result: `wrote 2 LongMemEval cases`.
- Ran failure-case export smoke test:
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output /tmp/adamem_stale_failures.jsonl --experiment-output /tmp/adamem_stale_diag_with_cases.json`
  - Result: diagnostics report succeeded and wrote `54` case-level failure
    records for paper-style error analysis.
- Ran failure report smoke test:
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output /tmp/adamem_cases.jsonl --diagnostic-report-output /tmp/adamem_failure_report.md --experiment-output /tmp/adamem_diag_record.json`
  - Result: Markdown report grouped `54` diagnostic records. The largest
    failure modes on the mini fixture were `current_evidence_not_recalled`
    (`50` records), `old_support_not_fully_adjudicated` (`36` records),
    `stale_evidence_exposed` (`15` records), and
    `stale_ranked_before_current` (`2` records).
- Ran STALE LLM-judge mock pilot:
  - `PYTHONPATH=src python -m adamem.eval --stale benchmarks/stale_mini.jsonl --answer-provider mock --answer-model ignored --judge-provider mock --judge-model ignored --max-cases 1 --top-k 4 --experiment-output /tmp/adamem_stale_pilot_mock.json`
  - Result: command completed and wrote prompts, raw outputs, retrieved traces,
    baseline configs, model settings, command, and commit into the experiment
    record. Mock correctness is not meaningful; this is a harness smoke test
    only.
- Ran dynamic-state transfer smoke:
  - `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl`
  - Result: `state_readout` and `state_propagation` scored `3/3`; `full`
    scored `1/3`; `semantic_only` scored `0/3`. This supports continued
    testing of current-state readout beyond location, but it is not a
    generalization claim because the fixture is local and small.
- Ran dynamic-state transfer experiment-record smoke:
  - `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --max-cases 1 --experiment-output /tmp/adamem_dynamic_transfer_record.json`
  - Result: command completed and wrote a `jsonl_retrieval_benchmark` record
    with all stable baselines, configs, retrieval-support results, per-query
    retrieved text, and trace data. `--max-cases 1` limits cases, not queries;
    the current fixture has one case with three queries.
- Ran LongMemEval converter smoke:
  - Created a toy official-schema LongMemEval JSON under `/tmp`.
  - `PYTHONPATH=src python -m adamem.convert longmemeval /tmp/longmemeval_toy.json /tmp/longmemeval_toy.adamem.jsonl --expected evidence --top-k 3`
  - `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_toy.adamem.jsonl`
  - Result: conversion and evaluation completed. This validates adapter shape,
    not transfer performance on official LongMemEval.
- Ran official LongMemEval-S retrieval pilot:
  - Downloaded `longmemeval_s_cleaned.json` from the official Hugging Face
    dataset `xiaowu0162/longmemeval-cleaned` into ignored local `data/`.
  - File size: `265M`.
  - SHA256 / Hugging Face linked ETag:
    `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
  - Converted the first 5 cases:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_5.adamem.jsonl --expected evidence --top-k 8 --limit 5`
  - Ran focused retrieval-support pilot:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_5.adamem.jsonl --baselines semantic_only full state_readout state_propagation --max-cases 5 --experiment-output results/longmemeval_s_5_retrieval_pilot.json`
  - Result: `semantic_only` scored `3/5`; `full`, `state_readout`, and
    `state_propagation` each scored `1/5`.
  - Interpretation: the current state-aware prototype does not yet transfer to
    general LongMemEval-S retrieval; this is useful negative evidence showing
    that the narrow deterministic slots are insufficient for public benchmark
    generality. It is a retrieval diagnostic only, not an end-to-end answer
    accuracy result.
- Inspected official LongMemEval-S question type distribution:
  - Total: `500`.
  - `multi-session`: `133`.
  - `temporal-reasoning`: `133`.
  - `knowledge-update`: `78`.
  - `single-session-user`: `70`.
  - `single-session-assistant`: `56`.
  - `single-session-preference`: `30`.
- Ran balanced LongMemEval-S retrieval pilot:
  - Converted 2 examples per LongMemEval-S `question_type`, 12 cases total:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_12.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 2`
  - Ran focused baselines and wrote grouped reports:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_12.adamem.jsonl --baselines semantic_only full state_readout state_propagation --max-cases 12 --benchmark-cases-output results/longmemeval_s_balanced_12_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_12_report.md --experiment-output results/longmemeval_s_balanced_12_pilot.json`
  - Result: `semantic_only` scored `9/12`; `full`, `state_readout`, and
    `state_propagation` each scored `1/12`.
  - Breakdown: `semantic_only` scored `2/2` on `knowledge-update`,
    `single-session-preference`, and `temporal-reasoning`; `1/2` on
    `multi-session`, `single-session-assistant`, and `single-session-user`.
    State-aware baselines only scored `1/2` on `single-session-user` and `0/2`
    on all other question types.
  - Interpretation: this is stronger negative transfer evidence. Current state
    readout likely pollutes generic LongMemEval retrieval because it activates
    for broad state-like cues and injects unrelated derived state memories. The
    next method iteration should introduce a stricter state-readout
    authorization boundary and preserve raw semantic retrieval behavior for
    non-state-sensitive public benchmark questions.
- Implemented and evaluated the state readout authorization boundary:
  - Dynamic-state fixture command:
    `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_propagation state_readout --max-cases 1`
  - Dynamic-state result: `semantic_only` scored `0/3`; `semantic_state_readout`,
    `semantic_state_propagation`, and `state_readout` each scored `3/3`.
  - Balanced LongMemEval-S command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_12.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_propagation full state_readout state_propagation --max-cases 12 --benchmark-cases-output results/longmemeval_s_auth_boundary_records.jsonl --benchmark-report-output results/longmemeval_s_auth_boundary_report.md --experiment-output results/longmemeval_s_auth_boundary_pilot.json`
  - Balanced LongMemEval-S result: `semantic_only`, `semantic_state_readout`,
    and `semantic_state_propagation` each scored `9/12`; `full`,
    `state_readout`, and `state_propagation` each scored `1/12`.
  - Interpretation: the clean semantic-state ablations preserve local
    dynamic-state gains without degrading the balanced LongMemEval-S retrieval
    pilot relative to semantic-only. The result does not yet prove generality,
    but it resolves the earlier state-pollution failure mode and gives a more
    defensible ablation path for larger public runs.
- Ran scaled balanced LongMemEval-S retrieval pilot:
  - Converted 10 examples per LongMemEval-S `question_type`, 60 cases total:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10`
  - Ran semantic-only and semantic-state ablations:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_propagation --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_semantic_state_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_semantic_state_report.md --experiment-output results/longmemeval_s_balanced_60_semantic_state_pilot.json`
  - Result: `semantic_only`, `semantic_state_readout`, and
    `semantic_state_propagation` each scored `40/60`.
  - Pairwise result against `semantic_only`: both semantic-state baselines had
    `common_total=60`, `gained_passes=0`, `lost_passes=0`, `net_delta=0`,
    `both_pass=40`, and `both_fail=20`.
  - Per-type result for `semantic_state_readout`: `knowledge-update` `8/10`,
    `multi-session` `4/10`, `single-session-assistant` `4/10`,
    `single-session-preference` `9/10`, `single-session-user` `7/10`, and
    `temporal-reasoning` `8/10`.
  - Interpretation: on this 60-case retrieval-support pilot, the authorization
    boundary preserves semantic-only public retrieval exactly while retaining
    dynamic-state gains on the local fixture. This supports continued
    development of the mechanism, but remains retrieval-level evidence only.
- Ran STALE mini retrieval diagnostics for semantic-state ablations:
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_readout semantic_state_propagation state_readout state_propagation --max-cases 2 --experiment-output results/stale_mini_semantic_state_diagnostics.json`
  - Result: `semantic_state_readout` and `semantic_state_propagation` reached
    `100%` current recall versus `0%` for `semantic_only`, but kept `33.33%`
    stale exposure and `0%` old-support adjudication. `state_readout` and
    `state_propagation` reached `100%` current recall, `0%` stale exposure, and
    `100%` old-support adjudication on this mini fixture.
  - Interpretation: semantic-state readout solves current-state availability
    without public retrieval degradation, but stale exposure/adjudication still
    requires additional mechanisms.
- Implemented query-scoped state-source adjudication:
  - Added `use_state_source_adjudication` to `AdaMemConfig`.
  - When a state slot value is replaced, the old state record is superseded and
    its raw source evidence is marked with `stale_state_slots`.
  - Retrieval filters that stale raw evidence only when the query routes to the
    same state slot and an active replacement state exists. This avoids using
    the global adjudication filter, which would also suppress historical
    queries.
  - Added canonical baselines `semantic_state_adjudication` and
    `semantic_state_propagation_adjudication`.
  - Updated deterministic state extraction to handle curly apostrophes in
    contractions such as `I’ve`, which occurred in STALE mini user text.
- Evaluated state-source adjudication:
  - Dynamic-state command:
    `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 1`
  - Dynamic-state result: `semantic_only` scored `0/3`; all three
    semantic-state variants scored `3/3`.
  - STALE mini diagnostics command:
    `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 2 --experiment-output results/stale_mini_state_adjudication_diagnostics.json --diagnostic-cases-output results/stale_mini_state_adjudication_cases.jsonl --diagnostic-report-output results/stale_mini_state_adjudication_report.md`
  - STALE mini result: `semantic_state_adjudication` reached `100%` current
    recall, `0%` stale exposure, and `57.14%` old-support adjudication. This
    improves over `semantic_state_readout`, which had `100%` current recall but
    `33.33%` stale exposure.
  - LongMemEval-S balanced 60 command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_state_adjudication_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_state_adjudication_report.md --experiment-output results/longmemeval_s_balanced_60_state_adjudication_pilot.json`
  - LongMemEval-S result: all four baselines scored `40/60`. Pairwise against
    `semantic_only`, both adjudication variants had `gained_passes=0`,
    `lost_passes=0`, and `net_delta=0`.
  - Interpretation: the new mechanism fixes stale exposure on the mini fixture
    without degrading the balanced LongMemEval-S retrieval-support pilot. It is
    still retrieval-level evidence; next work should scale STALE diagnostics and
    run answer/judge evaluation when provider keys are available.
- Added reproducible STALE subset selection for larger pilots:
  - `src/adamem/eval.py` now supports `--stale-types` and
    `--limit-per-stale-type` for both `--stale-diagnostics` and `--stale`.
  - `run_stale_benchmark` accepts `stale_types` and
    `limit_per_stale_type`, enabling the same split for API-free retrieval
    diagnostics and later LLM-judge runs.
  - Experiment records now include split notes such as
    `stale_types=T1;limit_per_stale_type=1`.
  - Verified CLI diagnostics on the current mini fixture:
    `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_adjudication state_readout --stale-types T1 --limit-per-stale-type 1 --experiment-output results/stale_mini_t1_limit1_state_adjudication_diagnostics.json --diagnostic-cases-output results/stale_mini_t1_limit1_state_adjudication_cases.jsonl --diagnostic-report-output results/stale_mini_t1_limit1_state_adjudication_report.md`
  - Result: `semantic_state_adjudication` kept `100%` current recall and `0%`
    stale exposure on the selected T1 case; `semantic_only` had `0%` current
    recall and `33.33%` stale exposure.
  - Verified mock LLM-judge mode on the same split:
    `PYTHONPATH=src python -m adamem.eval --stale benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_adjudication --stale-types T1 --limit-per-stale-type 1 --answer-provider mock --judge-provider mock --experiment-output results/stale_mini_t1_limit1_mock_judge.json`
  - The mock run is not an accuracy claim, but it confirms the exact split can
    be reused in answer/judge mode once provider keys are available.
- Added A-MEM-style mainstream approximation baseline:
  - Literature basis: A-MEM proposes agentic memory notes with structured
    attributes, dynamic links, and memory evolution over historical memories.
  - New config flags: `use_memory_evolution`,
    `memory_evolution_threshold`, `memory_evolution_keyword_limit`, and
    `memory_evolution_candidate_limit`.
  - New baseline: `a_mem_evolution`.
  - Implementation is API-free and deterministic: observed text gets note
    keywords; new raw memories link to recent related raw memories; older notes
    absorb new evolved keywords and recompute their retrieval embedding. This
    is documented as an approximation, not an official A-MEM reproduction.
  - Added a candidate limit after the first 60-case LongMemEval attempt showed
    the naive O(n²) evolution path was too slow for practical pilots.
  - Added unit coverage for bidirectional linking and evolved keyword updates.
- Evaluated A-MEM-style approximation against AdaMem state authority:
  - STALE mini command:
    `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only a_mem_evolution semantic_state_adjudication state_readout --max-cases 2 --experiment-output results/stale_mini_amem_vs_state_diagnostics.json --diagnostic-cases-output results/stale_mini_amem_vs_state_cases.jsonl --diagnostic-report-output results/stale_mini_amem_vs_state_report.md`
  - STALE mini result: `semantic_only` had `0%` current recall and `33.33%`
    stale exposure; `a_mem_evolution` had `33.33%` current recall and `16.67%`
    stale exposure but `0%` old-support adjudication; `semantic_state_adjudication`
    had `100%` current recall, `0%` stale exposure, and `57.14%` old-support
    adjudication.
  - LongMemEval-S balanced 12 result:
    `semantic_only` `9/12`, `a_mem_evolution` `6/12`,
    `semantic_state_adjudication` `9/12`. Pairwise vs `semantic_only`,
    `a_mem_evolution` gained `1`, lost `4`, net `-3`.
  - LongMemEval-S balanced 60 command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only a_mem_evolution semantic_state_adjudication --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_amem_state_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_amem_state_report.md --experiment-output results/longmemeval_s_balanced_60_amem_state_pilot.json`
  - LongMemEval-S balanced 60 result:
    `semantic_only` `40/60`, `a_mem_evolution` `27/60`,
    `semantic_state_adjudication` `40/60`. Pairwise vs `semantic_only`,
    `a_mem_evolution` gained `4`, lost `17`, net `-13`;
    `semantic_state_adjudication` gained `0`, lost `0`, net `0`.
  - Interpretation: generic memory evolution/linking is a useful mainstream
    comparator and weakly helps STALE mini retrieval, but it adds retrieval
    noise on LongMemEval-S. This strengthens the argument that stale-memory
    handling needs explicit current-state authority, not only richer episodic
    memory evolution.
- Added Zep/Graphiti-style temporal KG approximation baseline:
  - Literature basis: Zep/Graphiti represents dynamic information as temporal
    KG facts/edges with old facts invalidated when new information supersedes
    them.
  - New config flags: `use_temporal_kg_memory`, `use_temporal_kg_readout`, and
    `temporal_kg_readout_boost`.
  - New baseline: `zep_temporal_kg`.
  - Implementation is API-free and deterministic: extracted state patches write
    `kg_fact` memories with subject/relation/object metadata; a new object for
    the same subject/relation invalidates the old KG fact by setting
    `superseded_by`, `valid_to`, and staleness; active KG facts are read out for
    state-sensitive queries. Raw source evidence is intentionally not marked
    stale, so the baseline remains distinct from AdaMem state-source
    adjudication.
  - Added unit coverage for KG edge invalidation, `valid_to`, current KG
    readout, and absence of raw-source adjudication.
- Evaluated temporal KG approximation:
  - STALE mini command:
    `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only zep_temporal_kg semantic_state_adjudication state_readout --max-cases 2 --experiment-output results/stale_mini_zepkg_vs_state_diagnostics.json --diagnostic-cases-output results/stale_mini_zepkg_vs_state_cases.jsonl --diagnostic-report-output results/stale_mini_zepkg_vs_state_report.md`
  - STALE mini result: `zep_temporal_kg` reached `100%` current recall but kept
    `33.33%` stale exposure and `28.57%` old-support adjudication.
    `semantic_state_adjudication` reached `100%` current recall, `0%` stale
    exposure, and `57.14%` old-support adjudication.
  - LongMemEval-S balanced 60 command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only zep_temporal_kg semantic_state_adjudication --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_zepkg_state_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_zepkg_state_report.md --experiment-output results/longmemeval_s_balanced_60_zepkg_state_pilot.json`
  - LongMemEval-S balanced 60 result: all three baselines scored `40/60`.
    Pairwise versus `semantic_only`, both `zep_temporal_kg` and
    `semantic_state_adjudication` had `gained_passes=0`, `lost_passes=0`, and
    `net_delta=0`.
  - Interpretation: temporal KG readout can expose current state without
    degrading this LongMemEval retrieval-support pilot, but it does not control
    stale raw evidence. This sharpens AdaMem's claimed mechanism boundary:
    current-state authority requires both active state/fact readout and
    query-scoped source adjudication.
- Added Mem0-style compact extraction approximation baseline:
  - Literature basis: Mem0-style production memory extracts compact facts from
    messages, updates or supersedes older memories, and retrieves compact
    memories rather than replaying all raw conversation.
  - New config flags: `use_salient_memory`, `use_salient_memory_only`,
    `use_salient_memory_readout`, and `salient_memory_readout_boost`.
  - New baseline: `mem0_extraction`.
  - Implementation is API-free and deterministic: extracted state patches write
    `salient_fact` memories; new facts for the same subject/slot supersede old
    compact facts; raw observations remain stored for audit but are hidden from
    retrieval when `use_salient_memory_only=True`.
  - Added unit coverage for compact-fact replacement, retrieval of the current
    fact, and hiding raw source observations from the returned context.
- Evaluated compact extraction approximation:
  - STALE mini command:
    `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only mem0_extraction zep_temporal_kg semantic_state_adjudication --max-cases 2 --experiment-output results/stale_mini_mem0_zep_state_diagnostics.json --diagnostic-cases-output results/stale_mini_mem0_zep_state_cases.jsonl --diagnostic-report-output results/stale_mini_mem0_zep_state_report.md`
  - STALE mini result: `mem0_extraction` reached `100%` current recall and
    `0%` stale exposure, matching the surface stale-exposure behavior of
    `semantic_state_adjudication`; however its old-support adjudication was
    `28.57%` versus `57.14%` for `semantic_state_adjudication`, because it
    hides stale raw evidence rather than marking the source as stale.
  - LongMemEval-S balanced 60 command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only mem0_extraction zep_temporal_kg semantic_state_adjudication --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_mem0_zep_state_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_mem0_zep_state_report.md --experiment-output results/longmemeval_s_balanced_60_mem0_zep_state_pilot.json`
  - LongMemEval-S balanced 60 result: `semantic_only` `40/60`,
    `mem0_extraction` `1/60`, `zep_temporal_kg` `40/60`, and
    `semantic_state_adjudication` `40/60`. Pairwise versus `semantic_only`,
    `mem0_extraction` had `gained_passes=0`, `lost_passes=39`, and
    `net_delta=-39`; `zep_temporal_kg` and `semantic_state_adjudication` both
    had `net_delta=0`.
  - Interpretation: compact extraction/update alone can look strong on
    state-like stale cases but fails broad episodic retrieval on LongMemEval-S
    with the current narrow extractor. AdaMem's more defensible direction is to
    preserve raw evidence for general memory while adding explicit
    query-scoped source adjudication for stale-sensitive state slots.
- Ran STALE retrieval diagnostics on `benchmarks/stale_mini.jsonl` with
  `--max-cases 2`.
  - `delta_full` and `full` reached `100.00%` old support adjudication.
  - `delta_full` and `full` still had `0.00%` current recall under the stricter
    diagnostic, indicating that current-state authorization remains unsolved.
  - `state_memory` and `state_readout` reached `100.00%` current recall and
    `0.00%` stale exposure on this mini fixture. This supports the state-aware
    direction but is not yet a paper claim because the extractor is narrow and
    the fixture is small.
  - `semantic_importance` had nonzero current recall but high stale exposure,
    which suggests retrieval alone can surface new evidence without reliably
    controlling stale evidence.
- Updated `AGENTS.md` and `docs/research_workflow.md` to include the new
  diagnostics module and command.
- Updated `README.md` to expose the state-aware prototype and current synthetic
  ablation expectations.
- Exported `StatePatch` from `adamem` so custom extractors can be written
  without importing private module paths.

### 2026-05-30

- Broadened deterministic state extraction beyond location, schedule, task, and
  beverage state:
  - Added `health.*.status` slots for dietary/health constraints such as
    peanut, gluten, dairy, nut, and shellfish restrictions.
  - Added `resource.*.status` slots for resources such as passports, tokens,
    API keys, licenses, credentials, and access badges.
  - Added query routing for health/dietary and resource-sensitive questions.
  - Added a small dependency topology from selected health constraints to
    `meal.*`, `restaurant.*`, and `food.*` dependent state slots.
  - Kept resource query routing tied to resource nouns instead of generic
    `status` terms to reduce false-positive state readout.
- Extended `benchmarks/dynamic_state_transfer.jsonl` from 3 to 5 queries:
  schedule availability, task status, beverage preference, peanut-allergy
  clearance, and passport renewal.
- Added unit coverage for health constraint replacement and resource status
  replacement in `tests/test_adamem.py`.
- Updated JSONL benchmark summary tests for the expanded dynamic-state smoke
  fixture.
- Ran dynamic-state transfer smoke:
  - `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 1 --experiment-output results/dynamic_state_transfer_health_resource.json`
  - Result: `semantic_only` scored `0/5`; all listed state-aware variants
    scored `5/5`.
- Re-ran STALE mini diagnostics after the query-routing change:
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_adjudication state_readout --max-cases 2`
  - Result: unchanged smoke behavior: `semantic_state_adjudication` and
    `state_readout` reached `100.00%` current recall and `0.00%` stale
    exposure; `semantic_state_adjudication` had `57.14%` old-support
    adjudication and `state_readout` had `100.00%`.
- Ran full deterministic test suite:
  - `python -m pytest`
  - Result: `56 passed`.
- Interpretation: the state-authority mechanism now has API-free evidence that
  the same write-side replacement and read-time authorization path works across
  personal preference, scheduling, task state, health constraints, and resource
  status. This is still a local smoke result, not a generality claim.
- Added more agentic state slots:
  - Added `workflow.*.*` slots for current runbook, procedure, policy, or
    workflow rules such as checkout deployment rollback policy.
  - Added `runtime.*.status` slots for current tool/runtime state such as
    build runners, services, endpoints, queues, clusters, and environments.
  - Kept runtime query routing tied to runtime nouns instead of generic
    `status` terms to reduce false-positive readout.
- Added unit coverage for workflow-rule replacement, runtime-status
  replacement, and a false-positive boundary where a generic status report
  query should not surface runtime state.
- Extended `benchmarks/dynamic_state_transfer.jsonl` from 5 to 7 queries:
  schedule availability, task status, beverage preference, peanut-allergy
  clearance, passport renewal, checkout rollback runbook update, and staging
  build runner restoration.
- Ran dynamic-state transfer smoke:
  - `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 1 --experiment-output results/dynamic_state_transfer_workflow_runtime.json`
  - Result: `semantic_only` scored `0/7`; all listed state-aware variants
    scored `7/7`.
- Re-ran LongMemEval-S balanced 60 retrieval transfer after adding
  workflow/runtime routing:
  - Converted with `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10`
  - Evaluated with `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --experiment-output results/longmemeval_s_balanced_60_workflow_runtime_state_pilot.json`
  - Result: `semantic_only`, `semantic_state_adjudication`, and
    `semantic_state_propagation_adjudication` each scored `40/60`. The added
    routes did not introduce aggregate retrieval-support regressions in this
    balanced pilot.
- Re-ran STALE mini diagnostics after the workflow/runtime change:
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_adjudication state_readout --max-cases 2`
  - Result: unchanged smoke behavior: `semantic_state_adjudication` and
    `state_readout` reached `100.00%` current recall and `0.00%` stale
    exposure.
- Ran full deterministic test suite:
  - `python -m pytest`
  - Result: `59 passed`.
- Interpretation: workflow/runbook and runtime/tool state make the local
  transfer fixture closer to agent-memory trajectories, but the evidence is
  still smoke-level. The important public-transfer check is that these routes
  did not harm the current LongMemEval-S balanced retrieval-support pilot.
- Added state-readout exposure diagnostics for JSONL retrieval benchmarks:
  - Per-result trace entries now include `kind` and selected state/KG/salient
    metadata such as `state_slot` and `state_value`.
  - Case records now include `state_retrieval_count`,
    `retrieved_state_slots`, and `state_sensitive`.
  - Failure summaries and Markdown reports now include a
    `State Readout Exposure` table with total state exposure and unmarked-query
    state exposure.
- Used the new exposure diagnostic to find and fix a real LongMemEval-S false
  positive:
  - Before the fix, the word `local` in `local animal shelter` triggered
    location state readout for one unmarked LongMemEval-S query.
  - Narrowed location routing so `local` requires a location-intent context
    such as recommendation, nearby, places, resources, or spots.
  - Added a regression test where a `local animal shelter` event-history query
    must retrieve the historical event rather than current location state.
- Re-ran dynamic-state exposure report:
  - `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication --max-cases 1 --benchmark-report-output results/dynamic_state_transfer_state_exposure_report.md --experiment-output results/dynamic_state_transfer_state_exposure.json`
  - Result: `semantic_only` scored `0/7`; `semantic_state_readout` and
    `semantic_state_adjudication` scored `7/7`. State-aware variants exposed
    state on all 7 marked state queries and `0` unmarked queries.
- Re-ran LongMemEval-S balanced 60 exposure report after the routing fix:
  - `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-report-output results/longmemeval_s_balanced_60_state_exposure_report.md --experiment-output results/longmemeval_s_balanced_60_state_exposure.json`
  - Result: all three baselines remained `40/60`; state exposure for
    `semantic_state_adjudication` and
    `semantic_state_propagation_adjudication` dropped from the observed
    pre-fix `1/60` to `0/60`.
- Ran full deterministic test suite:
  - `python -m pytest`
  - Result: `61 passed`.
- Interpretation: the project now has an API-free diagnostic that measures
  prompt pollution from state readout and already caught one public-transfer
  false positive. This strengthens the causal-validity story because state
  gains can be separated from indiscriminate state-summary insertion.
- Extended state-readout diagnostics from exposure counts to slot-level
  authorization checks:
  - Records now include `expected_state_slots`, `unexpected_state_slots`, and
    `state_slot_matched`.
  - New failure modes: `state_readout_missing`,
    `state_readout_slot_mismatch`, and
    `state_readout_unmarked_exposure`.
  - The Markdown `State Readout Exposure` table now reports matched, missing,
    mismatched, and unmarked state exposure counts.
- Added tests for the three authorization cases:
  - A marked runtime-state query retrieves the matching runtime slot.
  - A generic status-report query does not retrieve state.
  - A deliberately wrong `state_slot` annotation produces
    `state_readout_slot_mismatch`.
- Re-ran dynamic-state exposure report after adding slot-level diagnostics:
  - Result: `semantic_only` had `7` `state_readout_missing` records.
  - `semantic_state_readout` and `semantic_state_adjudication` each matched
    all `7/7` expected state slots with `0` missing, `0` mismatched, and `0`
    unmarked exposures.
- Re-ran LongMemEval-S balanced 60 exposure report:
  - Result: all three baselines remained `40/60`.
  - Failure modes were limited to `expected_support_missing`; state-aware
    variants had `0` state exposure, `0` unmarked exposure, and `0` slot
    mismatch on the 60 unmarked public-transfer queries.
- Ran full deterministic test suite:
  - `python -m pytest`
  - Result: `61 passed`.
- Interpretation: the evaluation harness can now distinguish three separate
  questions that matter for a paper claim: whether support was retrieved,
  whether state authority was invoked when expected, and whether state authority
  polluted unrelated public-transfer queries.
- Added a paper-facing metrics table to JSONL benchmark summaries and Markdown
  reports:
  - Metrics include support pass rate, net delta versus the first/reference
    baseline, state-slot match rate, state-readout missing rate, slot-mismatch
    rate, and unmarked state exposure rate.
  - Metrics are also stored under `diagnostics.failure_summary.paper_metrics`
    in `--experiment-output`, so paper tables can be reproduced from raw run
    records.
- Re-ran dynamic-state exposure report:
  - Result: `semantic_only` `0/7`, `semantic_state_readout` `7/7`, and
    `semantic_state_adjudication` `7/7`.
  - Paper metrics: state-aware variants had `100.00%` state-slot match,
    `0.00%` state missing, `0.00%` slot mismatch, and no unmarked-query state
    exposure because all 7 queries are state-marked.
- Re-ran LongMemEval-S balanced 60 exposure report:
  - Result: `semantic_only`, `semantic_state_adjudication`, and
    `semantic_state_propagation_adjudication` each scored `40/60`.
  - Paper metrics: state-slot metrics are `n/a` because this public-transfer
    sample has no evaluation-marked state queries; all three baselines had
    `0.00%` unmarked state exposure.
- Ran full deterministic test suite:
  - `python -m pytest`
  - Result: `62 passed`.
- Interpretation: this makes the API-free JSONL harness closer to a paper
  experiment table. It still measures retrieval support, not final answer
  correctness, but it now reports the state-authority mechanism's precision
  boundary alongside retrieval accuracy.
- Added optional LongMemEval query-state annotation:
  - Converter flag:
    `PYTHONPATH=src python -m adamem.convert longmemeval INPUT OUTPUT --infer-state-slots`
  - The converter uses the same deterministic query router as evaluation
    diagnostics to add `query.metadata.state_slot` and
    `state_slot_source=query_text_router`.
  - These annotations are evaluation-only and are not written into observation
    metadata or runtime memory.
- Ran LongMemEval-S balanced 60 with inferred state slots:
  - Conversion:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_inferred_state.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --infer-state-slots`
  - Evaluation:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60_inferred_state.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-report-output results/longmemeval_s_balanced_60_inferred_state_report.md --experiment-output results/longmemeval_s_balanced_60_inferred_state.json`
  - Result: all three baselines scored `40/60`.
  - Paper metrics: the query router marked `18/60` questions as
    state-sensitive, but every baseline had `0.00%` state-slot match and
    `100.00%` state-readout missing on those marked questions.
  - Interpretation: the readout/adjudication machinery is not enough on real
    public-transfer data unless the observation-side extractor can derive
    matching state records from natural dialogue. This is the next API-free
    method bottleneck.
- Ran full deterministic test suite after the LongMemEval state-slot
  annotation change:
  - `python -m pytest`
  - Result: `63 passed`.
- Tightened query-state routing after auditing the LongMemEval-S inferred-state
  false positives:
  - Replaced broad substring checks with word-boundary term matching.
  - Added slot-specific intent gates so ordinary event-history queries such as
    `coffee creamer`, `local community theater`, `met up`, `my city`,
    `Netflix access`, and `work from home job list` no longer become
    state-sensitive diagnostics.
  - Preserved the intended state-routing behavior for dynamic-state queries
    such as `What time can I meet`, `What is the migration status`, peanut
    allergy premise resistance, passport status, workflow rollback, runtime
    runner status, and local/nearby recommendation queries.
- Added regression tests for query-router precision using the observed
  LongMemEval-S false-positive patterns.
- Re-ran LongMemEval-S balanced 60 with inferred state slots after the router
  precision change:
  - Conversion:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_inferred_state_router_v3.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --infer-state-slots`
  - Evaluation:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60_inferred_state_router_v3.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-report-output results/longmemeval_s_balanced_60_inferred_state_router_v3_report.md --experiment-output results/longmemeval_s_balanced_60_inferred_state_router_v3.json`
  - Result: all three baselines remained `40/60`.
  - Paper metrics: inferred state-sensitive queries dropped from `18/60` to
    `1/60`; unmarked state exposure stayed `0.00%`.
  - Interpretation: the earlier `18/60` missing-state result was mostly a
    query-router precision artifact. The remaining marked query is a local
    `around me` recommendation where the memory does not appear to contain a
    current user location. This makes the public-transfer diagnostic more
    honest, but also shows that LongMemEval-S balanced 60 is not a rich enough
    state-sensitive transfer set by itself.
- Re-ran dynamic-state transfer after the router precision change:
  - `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --experiment-output results/dynamic_state_transfer_router_precision.json --benchmark-report-output results/dynamic_state_transfer_router_precision_report.md`
  - Result: `semantic_only` `0/7`; all listed state-aware variants `7/7`.
- Re-ran STALE mini diagnostics after the router precision change:
  - `PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 2 --experiment-output results/stale_mini_router_precision_diagnostics.json --diagnostic-report-output results/stale_mini_router_precision_report.md`
  - Result: `semantic_state_adjudication`,
    `semantic_state_propagation_adjudication`, and `state_readout` kept
    `100.00%` current recall and `0.00%` stale exposure on the mini diagnostic.
- Re-ran full deterministic test suite:
  - `python -m pytest`
  - Result: `64 passed`.
- Added a manual audit workflow for public state-sensitive transfer subsets:
  - `--state-audit-output` exports LongMemEval query-state candidates as JSONL
    records with `is_state_sensitive: null`, inferred slots, question id/type,
    question date, question text, and notes.
  - `--state-audit-input` imports reviewed JSONL records only when
    `is_state_sensitive` is `true` and a `state_slot` is present.
  - Imported labels are written only to query metadata with
    `state_slot_source=manual_state_audit`; observation metadata is unchanged.
  - This separates automatic candidate generation from paper-facing
    state-readout metrics.
- Added converter tests for the manual audit path:
  - Candidate export includes only routed state-query candidates.
  - Rejected or unreviewed records are ignored.
  - Accepted audit labels do not enter observation metadata or runtime memory
    inputs.
- Generated the current LongMemEval-S balanced 60 audit artifacts:
  - Candidate export command:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_audit_probe.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-output results/longmemeval_s_balanced_60_state_audit_candidates.jsonl`
  - Result: `1` candidate query:
    `Can you recommend some interesting cultural events happening around me this weekend?`
  - Added reviewed label file:
    `results/longmemeval_s_balanced_60_state_audit_reviewed.jsonl`
  - Manual-audit conversion command:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_manual_audit.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-input results/longmemeval_s_balanced_60_state_audit_reviewed.jsonl`
  - Verified converted query labels:
    one query had `state_slot=location`,
    `state_slot_source=manual_state_audit`, and
    `state_audit_id=longmemeval_s_balanced_60_router_v3_manual_001`.
  - Verified observation-level `state_slot` leakage: `0`.
- Ran LongMemEval-S balanced 60 with manual audit labels:
  - `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60_manual_audit.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-report-output results/longmemeval_s_balanced_60_manual_audit_report.md --experiment-output results/longmemeval_s_balanced_60_manual_audit.json`
  - Result: all three baselines remained `40/60`.
  - Paper metrics: `1` manually audited state-sensitive query,
    `100.00%` state-readout missing, and `0.00%` unmarked state exposure.
- Re-ran deterministic local tests after the manual audit workflow:
  - `python -m pytest`
  - Result: `65 passed`.
- Split manual-audit state sensitivity from state availability:
  - Added `state_available` to the LongMemEval state-audit schema.
  - `state_available=false` marks a query as state-sensitive but lacking a
    reliable current state in the haystack. Such queries are no longer counted
    in the state-readout missing denominator.
  - Benchmark records now expose `state_available` and
    `state_readout_expected`.
  - `State Readout Exposure` reports now separate `state queries`,
    `state available`, and `state unavailable`.
- Updated the reviewed LongMemEval-S balanced 60 audit label:
  - The `around me` cultural-events query remains `state_slot=location`.
  - Manual inspection found no reliable current user location, so it is now
    marked `state_available=false`.
- Re-ran LongMemEval-S balanced 60 with state-availability-aware manual audit:
  - `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60_manual_audit_state_available.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-report-output results/longmemeval_s_balanced_60_manual_audit_state_available_report.md --experiment-output results/longmemeval_s_balanced_60_manual_audit_state_available.json`
  - Result: all three baselines remained `40/60`.
  - Paper metrics: `state_sensitive_total=1`, `state_query_total=0`,
    `state_unavailable_total=1`, state readout metrics `n/a`, and unmarked
    state exposure `0.00%`.
  - Failure modes now include only `expected_support_missing`; the old
    `state_readout_missing` count was removed for this state-unavailable case.
- Re-ran deterministic local tests after the state availability metric split:
  - `python -m pytest`
  - Result: `65 passed`.
- Added state-evidence candidates to the LongMemEval manual-audit export:
  - `--state-audit-output` now includes `state_evidence_candidates`, generated
    by running the deterministic observation-side state extractor over
    haystack turns.
  - Candidate evidence includes label, date, role, extracted state slot/value,
    evidence text, and `source=deterministic_state_extractor`.
  - It does not include LongMemEval `answer`, `answer_session_ids`, turn-level
    `has_answer`, or any judge-only field.
  - The converter now uses a shared LongMemEval turn iterator for both emitted
    observations and audit evidence candidates, reducing schema drift between
    runtime inputs and audit support.
- Re-generated `results/longmemeval_s_balanced_60_state_audit_candidates.jsonl`
  with evidence candidates:
  - The only candidate remains the `around me` location query.
  - `state_evidence_candidates` is empty, matching the reviewed
    `state_available=false` label.
  - This makes the state-unavailable decision auditable from the same
    deterministic extractor used by the runtime prototype, without using answer
    labels.
- Re-ran deterministic local tests after evidence-candidate export:
  - `python -m pytest`
  - Result: `65 passed`.
- Added LongMemEval audit summary output:
  - New converter option: `--state-audit-summary-output`.
  - Summary JSON reports `total_candidates`, `with_state_evidence`,
    `without_state_evidence`, `state_evidence_candidate_total`, and breakdowns
    by inferred state slot and LongMemEval `question_type`.
  - Added tests for summary generation and evidence-coverage aggregation.
- Tightened query-state routing after full LongMemEval-S audit exposed more
  false positives:
  - Task-status routing now requires explicit `status`, or a status-state term
    plus a task-like subject such as task, ticket, issue, migration, incident,
    workflow, project, deployment, or request.
  - Direct `live`/`based`/`located`/`staying` location routing now requires a
    self-location subject such as `I`, `me`, `user`, or `my location`.
  - Added regression tests for third-party residence, completed-course counts,
    and `Facebook Live` event queries.
- Generated LongMemEval-S audit summaries after router tightening:
  - Full 500-case command:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_full_audit_summary_router_v4.adamem.jsonl --expected evidence --top-k 8 --state-audit-output results/longmemeval_s_full_state_audit_candidates.jsonl --state-audit-summary-output results/longmemeval_s_full_state_audit_summary.json`
  - Full 500-case result: `14` query-state candidates, `0` with deterministic
    state evidence, and `0` total state-evidence candidates.
  - Balanced 60-case command:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_audit_summary_router_v4.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-output results/longmemeval_s_balanced_60_state_audit_candidates.jsonl --state-audit-summary-output results/longmemeval_s_balanced_60_state_audit_summary.json`
  - Balanced 60-case result: `1` query-state candidate, `0` with deterministic
    state evidence, and `0` total state-evidence candidates.
  - Interpretation: LongMemEval-S remains useful for broad retrieval
    transfer/no-regression checks, but it is currently too sparse in
    state-available cases to support AdaMem's main state-transfer claim.
- Re-ran deterministic local tests after audit-summary and router changes:
  - `python -m pytest`
  - Result: `66 passed`.

### 2026-05-28

- Reviewed repository structure and current code paths.
- Confirmed the project is a Python package under `src/adamem`.
- Ran deterministic local tests:
  - `python -m pytest`
  - Result: `20 passed`.
- Ran built-in synthetic ablation:
  - `PYTHONPATH=src python -m adamem.eval`
  - Result: `full` scored `4/4`; `semantic_only` scored `1/4`.
- Ran JSONL smoke benchmark:
  - `PYTHONPATH=src python -m adamem.eval --dataset benchmarks/tiny_memory_qa.jsonl`
  - Result: `full` scored `4/4`.
- Audited the current STALE mini fixture with retrieval-side diagnostics.
  Observed that current metrics are too loose and that old-state leakage can
  remain high even when soft staleness marks memories.
- Added `AGENTS.md` with project objective, research bar, scope, design
  constraints, risks, promising direction, and paper-track milestones.
- Added `docs/research_workflow.md` with an API-free and API-enabled workflow
  for CCF-A-level development.

## Key Decisions

### CCF-A research target

The project should be advanced under top-tier paper standards. A mechanism is
not enough unless it has clean ablations, diagnostics, reproducibility, and
credible baselines.

### STALE is primary

STALE is the first benchmark to optimize and understand because it tests state
revision, premise resistance, and implicit policy adaptation.

### Runtime cannot use evaluation-only labels

`M_old`, `M_new`, `explanation`, `relevant_session_index`, answers, and judge
labels are allowed for evaluation, diagnostics, oracle upper bounds, and
conversion. They are not allowed inside the proposed runtime method.

### Similarity-only stale detection is probably insufficient

The current soft staleness mechanism mostly relies on lexical or embedding
similarity. Real STALE cases require semantic state inference and an explicit
current-state basis.

### State-aware memory is the leading method direction

The next method iteration should extract state candidates, assign typed slots,
adjudicate active versus stale values, and expose authorized current state
before raw retrieved dialogue.

### Diagnostics should distinguish current recall from stale filtering

The first API-free diagnostics show that filtering or adjudicating old evidence
does not imply current evidence is retrieved. Future method work should treat
authorized current-state readout as a separate mechanism from stale suppression.

### State-aware memory is promising but not yet validated

The first deterministic location-state prototype improves API-free STALE mini
diagnostics, but it is intentionally narrow. It must be tested on larger STALE
splits, harder synthetic cases, and at least one non-STALE benchmark before it
can support a paper claim.

### Literature framing should drive mechanisms and gates

`docs/literature_to_design.md` now records how STALE, A-MEM, Zep, MemGPT,
MemoryBank, Generative Agents, Mem0, LongMemEval, AMA-Bench, LongMemEval-V2,
and the 2026 memory survey map to AdaMem hypotheses. This should prevent the
project from drifting into arbitrary engineering and keep each mechanism tied
to a paper-facing claim and evaluation gate.

## Known Risks

- Current `hashed_bow` embedder is weak for paraphrase and implicit state
  conflicts.
- Current raw-turn storage may be too noisy for STALE.
- Current ADR/SLR-style metrics can overcount common words from `M_old`.
- Query similarity can over-rank stale evidence in Premise Resistance queries.
- API-enabled results are not yet available; no final benchmark claim is valid.
- `benchmarks/stale_mini.jsonl` is useful for debugging but should not become
  the optimization target.

## Next Tasks

1. Prepare API-enabled pilot scripts for later provider keys.
2. Run a 5-20 case real STALE pilot when provider keys are available and
   manually audit at least one answer/judge disagreement.
3. Add official/faithful external memory baselines when their code and licenses
   are reviewed.
4. Replace or complement `benchmarks/dynamic_state_transfer.jsonl` with at
   least one public non-STALE memory benchmark for transfer.
5. Build a reliable public state-sensitive transfer subset. Query annotations
   must remain evaluation-only and should be precision-audited before being
   used as state-readout metrics.
6. Run state-aware diagnostics on a larger converted STALE sample when data is
   available locally.
7. Add remaining non-location state slots such as relationships, user roles,
   environment gotchas, and tool-output facts.
8. Evaluate `state_propagation` on larger STALE T2-style indirect conflicts.
9. Add a documented LLM extractor baseline using the pluggable extractor hook.
10. Use the compact failure report on larger STALE samples to select the next
   mechanism iteration and paper error categories.

## Change Log

### 2026-05-28

- Added `AGENTS.md`.
- Added `docs/research_workflow.md`.
- Added `docs/progress_log.md`.

### 2026-05-29

- Added `src/adamem/diagnostics.py`.
- Added `src/adamem/baselines.py`.
- Added `src/adamem/state.py`.
- Added `src/adamem/experiments.py`.
- Added `docs/literature_to_design.md`.
- Added `benchmarks/dynamic_state_transfer.jsonl`.
- Updated `src/adamem/convert.py` with LongMemEval conversion support.
- Updated `src/adamem/config.py` with state-memory/readout flags.
- Updated `src/adamem/config.py` with state dependency propagation flag.
- Updated `src/adamem/config.py` with state-source adjudication flag.
- Updated `src/adamem/config.py` with deterministic memory-evolution flags.
- Updated `src/adamem/config.py` with temporal-KG flags.
- Updated `src/adamem/config.py` with compact salient-memory flags.
- Updated `src/adamem/manager.py` to write and retrieve derived state memories.
- Updated `src/adamem/manager.py` to propagate changed state slots to dependent
  state slots and their source evidence.
- Updated `src/adamem/manager.py` to mark and query-scope-filter raw evidence
  superseded by replaced state slots.
- Updated `src/adamem/manager.py` with A-MEM-style deterministic note keyword,
  dynamic linking, and memory evolution support.
- Updated `src/adamem/manager.py` with temporal KG fact writing, invalidation,
  and readout support.
- Updated `src/adamem/manager.py` with compact salient fact writing,
  replacement, retrieval-only mode, and readout support.
- Updated `src/adamem/state.py` with schedule availability and dynamic task
  status extraction.
- Updated `src/adamem/state.py` with wildcard state readout matching and the
  initial state dependency topology.
- Updated `src/adamem/state.py` to handle curly apostrophes in contractions.
- Updated `src/adamem/baselines.py` with `state_propagation`.
- Updated `src/adamem/baselines.py` with `a_mem_evolution`.
- Updated `src/adamem/baselines.py` with `zep_temporal_kg`.
- Updated `src/adamem/baselines.py` with `mem0_extraction`.
- Updated `src/adamem/baselines.py` with `semantic_state_adjudication` and
  `semantic_state_propagation_adjudication`.
- Updated `src/adamem/eval.py` with `--stale-diagnostics`.
- Updated `src/adamem/eval.py` with `--list-baselines` and
  `--experiment-output`.
- Updated `src/adamem/eval.py` with STALE T1/T2 subset selection through
  `--stale-types` and `--limit-per-stale-type`.
- Updated `src/adamem/eval.py` with `--diagnostic-cases-output`.
- Updated `src/adamem/eval.py` with `--diagnostic-report-output`.
- Updated `src/adamem/eval.py` so `--stale ... --experiment-output` records
  prompts, raw model outputs, retrieved traces, model settings, and command.
- Updated `src/adamem/diagnostics.py` with case-level diagnostic record export.
- Updated `src/adamem/diagnostics.py` with failure summary and Markdown report
  aggregation.
- Updated `src/adamem/llm.py` so the mock provider can be used from CLI smoke
  runs without a real model argument.
- Updated `tests/test_stale.py` with diagnostics tests.
- Updated `tests/test_adamem.py` with state-aware mechanism tests.
- Updated `tests/test_eval.py` with dynamic-state transfer smoke coverage.
- Updated `tests/test_eval.py` with LongMemEval converter coverage.
- Added `tests/test_experiments.py`.
- Updated `src/adamem/__init__.py` to export `StatePatch`.
- Updated `AGENTS.md`, `README.md`, `docs/research_workflow.md`, and this
  progress log.

### 2026-05-30

- Updated `src/adamem/state.py` with health/dietary constraint slots,
  resource status slots, health/resource query routing, and selected
  health-to-food dependency prefixes.
- Updated `benchmarks/dynamic_state_transfer.jsonl` with peanut-allergy
  clearance and passport-renewal premise-resistance cases.
- Updated `tests/test_adamem.py` with health/resource state replacement tests.
- Updated `tests/test_eval.py` for the expanded 5-query dynamic-state fixture.
- Added `results/dynamic_state_transfer_health_resource.json` from the
  API-free dynamic-state transfer smoke run.
- Updated `src/adamem/state.py` with workflow/runbook rule slots,
  runtime/tool status slots, workflow/runtime query routing, and runtime noun
  guards.
- Updated `benchmarks/dynamic_state_transfer.jsonl` with checkout rollback
  runbook and staging build runner stale-premise cases.
- Updated `tests/test_adamem.py` with workflow/runtime state replacement tests
  and a generic status-report false-positive test.
- Updated `tests/test_eval.py` for the expanded 7-query dynamic-state fixture.
- Added `results/dynamic_state_transfer_workflow_runtime.json` from the
  expanded dynamic-state transfer smoke run.
- Added `results/longmemeval_s_balanced_60_workflow_runtime_state_pilot.json`
  from the LongMemEval-S no-regression transfer check.
- Updated `src/adamem/bench.py` with trace-level item kind/metadata and
  state-readout exposure aggregates.
- Updated `src/adamem/state.py` to require a location-intent context before
  `local` can trigger location state readout.
- Updated `tests/test_adamem.py` with a `local animal shelter` false-positive
  regression test.
- Updated `tests/test_eval.py` with state exposure metric coverage.
- Added `results/dynamic_state_transfer_state_exposure.json` and
  `results/dynamic_state_transfer_state_exposure_report.md`.
- Added `results/longmemeval_s_balanced_60_state_exposure.json` and
  `results/longmemeval_s_balanced_60_state_exposure_report.md`.
- Updated `src/adamem/bench.py` with slot-level state authorization diagnostics
  and failure modes.
- Updated `tests/test_eval.py` with missing/matched/mismatched state-readout
  assertions.
- Updated `src/adamem/bench.py` with paper-facing JSONL benchmark metrics.
- Updated `tests/test_eval.py` with paper metrics report coverage.
- Updated `src/adamem/convert.py` with LongMemEval
  `--infer-state-slots`, which annotates query metadata from query text for
  diagnostics only.
- Updated `tests/test_eval.py` to verify inferred LongMemEval state slots stay
  out of observation metadata and runtime memory inputs.
- Added `results/longmemeval_s_balanced_60_inferred_state.json` and
  `results/longmemeval_s_balanced_60_inferred_state_report.md` from the
  query-annotated LongMemEval-S transfer diagnostic.
- Updated `src/adamem/convert.py` with LongMemEval
  `--state-audit-summary-output`.
- Updated `src/adamem/state.py` with stricter task-status and self-location
  query routing gates.
- Updated `tests/test_eval.py` with LongMemEval state-audit summary coverage.
- Updated `tests/test_adamem.py` with additional LongMemEval-S router
  false-positive regressions.
- Added `results/longmemeval_s_full_state_audit_candidates.jsonl` and
  `results/longmemeval_s_full_state_audit_summary.json` from the full-file
  LongMemEval-S audit.
- Added `results/longmemeval_s_balanced_60_state_audit_summary.json` from the
  balanced LongMemEval-S audit summary.
- Updated `src/adamem/convert.py` with an AMA-Bench-style trajectory
  converter:
  - New command: `PYTHONPATH=src python -m adamem.convert ama INPUT OUTPUT`.
  - Accepts JSON arrays or JSONL records with `trajectory`/`steps`-style
    action-observation histories.
  - Emits actions, observations, and environment states as runtime
    observations while preserving action-to-observation causality through
    `cause_labels`.
  - Keeps answers and evidence labels query-only, avoiding runtime leakage.
- Updated `tests/test_eval.py` with an AMA causality smoke test:
  - A semantic no-graph config misses the hidden action identifier.
  - A graph-enabled config recovers it through the action-result cause edge.
- Updated `README.md` and `docs/research_workflow.md` with the AMA converter
  command and the next API-free trajectory-transfer step.
- Updated `src/adamem/bench.py` with evidence-support diagnostics for JSONL
  retrieval benchmarks:
  - Case records now expose `expected_evidence`, `missing_evidence`,
    `evidence_support_matched`, `graph_retrieval_count`,
    `graph_evidence_hits`, and `graph_evidence_hit_count`.
  - Markdown reports now include an `Evidence Support` table.
  - Paper metrics now include evidence-support and graph-evidence-hit rates.
  - Trace metadata now includes `memory_key`, `benchmark`,
    `trajectory_step`, and `subject`, so trajectory evidence labels can be
    checked without relying only on text substrings.
- Extended the AMA causality smoke test to verify that the semantic no-graph
  run retrieves the outcome evidence but misses the answer, while the
  graph-enabled run records a graph evidence hit for `step000`.
- Updated `README.md`, `docs/research_workflow.md`, and
  `docs/literature_to_design.md` with the causal trajectory evidence
  diagnostic and its paper-facing interpretation.
- Checked the public AMA-Bench Hugging Face schema:
  - Dataset card fields include `episode_id`, `task`, `task_type`, `domain`,
    `success`, `num_turns`, `total_tokens`, `trajectory`, and `qa_pairs`.
  - Trajectory entries use `turn_idx`, `action`, and `observation`.
  - QA entries use `question`, `answer`, `question_uuid`, and `type`.
- Updated `src/adamem/convert.py` to better match the public AMA-Bench schema:
  - Preserves numeric `episode_id=0` instead of falling back to `ama-sample`.
  - Uses `turn_idx` for `stepNNN.action` / `stepNNN.observation` labels.
  - Uses `question_uuid` as the query id when present.
  - Maps AMA `type` labels `A/B/C/D` to recall, causal inference, state
    updating, and state abstraction names.
  - Infers diagnostic evidence labels from `Step N` references in question
    text when explicit evidence fields are absent.
- Added `tests/test_eval.py` coverage for the public AMA-Bench Hugging Face
  schema shape, including `turn_idx`, `question_uuid`, type-name mapping, and
  step-range evidence extraction.
- Ran an API-free smoke check on the first public AMA-Bench sample downloaded
  from Hugging Face to `/tmp`:
  - Conversion result: `1` case, `200` action/observation memories, `12` QA
    pairs.
  - Retrieval diagnostic command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/ama_first.adamem.jsonl --baselines semantic_only full --max-cases 1 --benchmark-report-output /tmp/ama_first_report.md --experiment-output /tmp/ama_first_eval.json`
  - Result: `semantic_only` `0/12`, `full` `0/12`, evidence support `0/12`
    for both.
  - Interpretation: generic similarity/default graph retrieval is not enough
    for real AMA-Bench trajectories; the next mechanism should add step-aware
    trajectory indexing or query routing.
- Added step-aware trajectory retrieval:
  - New config flag: `use_trajectory_step_readout`.
  - New score weight: `trajectory_step_readout_boost`.
  - New baseline: `trajectory_step_readout`.
  - Runtime behavior: when a query explicitly mentions `Step N` or a short
    step range, AdaMem authorizes matching AMA trajectory memories through
    `trajectory_step` metadata. It does not use answer labels or query
    evidence labels.
- Added `tests/test_eval.py` coverage for the step-aware retrieval failure mode:
  - Semantic retrieval returns repeated observation text and misses the
    action-only answer.
  - `trajectory_step_readout` retrieves the explicit step action and covers all
    expected step evidence.
- Re-ran the first public AMA-Bench sample smoke with
  `trajectory_step_readout`:
  - Command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/ama_first.adamem.jsonl --baselines semantic_only full trajectory_step_readout --max-cases 1 --benchmark-report-output /tmp/ama_first_step_readout_report.md --experiment-output /tmp/ama_first_step_readout_eval.json`
  - Result: answer support remains `0/12` for all three baselines.
  - Evidence support: `semantic_only` `0/12`, `full` `0/12`,
    `trajectory_step_readout` `12/12`.
  - Interpretation: the retrieval layer can now recover the correct trajectory
    steps on this real sample; the remaining AMA gap is answer synthesis or
    LLM-as-judge scoring over those retrieved steps.
- Scaled the API-free public AMA-Bench smoke to the first five Hugging Face
  samples:
  - Downloaded the first five JSONL rows to `/tmp/ama_first5.jsonl`.
  - Converted answer-mode records:
    `PYTHONPATH=src python -m adamem.convert ama /tmp/ama_first5.jsonl /tmp/ama_first5.adamem.jsonl --expected answer --top-k 8`
  - Conversion result: `5` cases, `782` action/observation memories, `60` QA
    pairs.
  - Answer-mode command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/ama_first5.adamem.jsonl --baselines semantic_only full trajectory_step_readout --max-cases 5 --benchmark-report-output /tmp/ama_first5_step_readout_report.md --experiment-output /tmp/ama_first5_step_readout_eval.json`
  - Answer-mode result: answer support remains `0/60` for all three baselines;
    evidence support is `0/60` for `semantic_only`, `0/60` for `full`, and
    `60/60` for `trajectory_step_readout`.
  - Converted evidence-mode records:
    `PYTHONPATH=src python -m adamem.convert ama /tmp/ama_first5.jsonl /tmp/ama_first5_evidence.adamem.jsonl --expected evidence --top-k 8`
  - Evidence-mode command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/ama_first5_evidence.adamem.jsonl --baselines semantic_only full trajectory_step_readout --max-cases 5 --benchmark-report-output /tmp/ama_first5_evidence_step_readout_report.md --experiment-output /tmp/ama_first5_evidence_step_readout_eval.json`
  - Evidence-mode result: `semantic_only` `0/60`, `full` `0/60`,
    `trajectory_step_readout` `60/60`.
  - Interpretation: the mechanism is not only fitting one case. On this small
    public smoke subset, explicit step readout fully recovers the diagnostic
    step evidence that generic retrieval misses. This still does not support an
    AMA answer-accuracy or SOTA claim because exact answer strings are
    inappropriate for open-ended AMA answers; API-backed answer synthesis and
    judge evaluation remain required.
- Re-ran deterministic local tests:
  - `python -m pytest`
  - Result: `69 passed`.

## 2026-05-30 continued

- Added API-free answerability diagnostics for open-ended JSONL benchmarks:
  - `benchmark_case_records` now records answer keywords, missing answer
    keywords, retrieved-context keyword recall, and a matched/not-matched
    threshold signal.
  - AMA records also get a deterministic `answer_basis` derived only from
    retrieved step/action/observation traces. It summarizes explicit step
    actions, observations, inverse action pairs, and repeated observations.
  - The basis is query-scoped: when the question mentions `Step N` or a short
    step range, unrelated retrieved steps are excluded from the basis.
  - Gold answer text is used only for evaluation keyword recall, not for the
    runtime memory method or basis generation.
- Added tests for the answerability path:
  - Step-readout retrieval now records a basis containing the recovered action.
  - Repeated observations and inverse action pairs are surfaced in the basis.
  - The report table handles no-answer/no-state cases with `n/a` rates.
- Re-ran the first five public AMA-Bench answer-mode records with
  answerability diagnostics:
  - Command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/ama_first5.adamem.jsonl --baselines semantic_only full trajectory_step_readout --max-cases 5 --benchmark-report-output /tmp/ama_first5_answerability_report.md --experiment-output /tmp/ama_first5_answerability_eval.json`
  - Exact answer-string support remains `0/60` for all baselines.
  - Evidence support remains `0/60` for `semantic_only`, `0/60` for `full`,
    and `60/60` for `trajectory_step_readout`.
  - Answerability diagnostics:
    - `semantic_only`: keyword matched `8/60`, average recall `22.73%`,
      basis average recall `22.73%`.
    - `trajectory_step_readout`: keyword matched `8/60`, average recall
      `22.73%`, basis keyword matched `11/60`, basis average recall `24.81%`.
    - `full`: keyword matched `0/60`, average recall `3.25%`.
  - Interpretation: step evidence recall transfers on this small public smoke
    subset, but the deterministic step basis is too weak to bridge open-ended
    causal answers. The next mechanism should add richer state/causal
    summarization or use API-backed answer synthesis and judge scoring.
- Re-ran deterministic validation:
  - `python -m pytest`
  - Result: `70 passed`.
  - `git diff --check`
  - Result: clean.

## 2026-05-30 trajectory basis iteration

- Fixed a trajectory-memory identity bug in `AdaMem.observe`:
  - Near-duplicate memories with distinct `metadata["memory_key"]` are no
    longer merged by novelty deduplication.
  - This matters for repeated agent trajectories, where `step017.action:
    right` and `step018.action: right` are semantically identical but represent
    different temporal states.
  - Added a unit test to preserve distinct step memory keys.
- Extended deterministic AMA answer-basis diagnostics:
  - Extracts active rules such as `wall is stop`, `key is win`, and `baba is
    you` from retrieved observations.
  - Adds derived facts for `stop`/`win`/`you` rules.
  - Detects adjacent blockers for the current action, e.g. right action blocked
    by adjacent wall due to `wall is stop`.
  - Detects repeated same action with unchanged observations as blocked/no
    progress.
  - Keeps the basis query-scoped and answer-label-free.
- Added tests for blocked-rule trajectory summaries:
  - A `wall is stop` observation plus repeated right actions yields an
    answer-basis explanation for blocked/no-progress behavior.
- Re-ran the first five public AMA-Bench answer-mode records:
  - Command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/ama_first5.adamem.jsonl --baselines semantic_only full trajectory_step_readout --max-cases 5 --benchmark-report-output /tmp/ama_first5_structured_basis_report.md --experiment-output /tmp/ama_first5_structured_basis_eval.json`
  - Exact answer-string support remains `0/60` for all baselines.
  - Evidence support: `semantic_only` `2/60`, `full` `0/60`,
    `trajectory_step_readout` `60/60`.
  - Answerability diagnostics:
    - `semantic_only`: keyword matched `6/60`, average recall `21.28%`,
      basis matched `8/60`, basis average recall `22.01%`.
    - `full`: keyword matched `10/60`, average recall `24.62%`, no
      query-scoped basis records.
    - `trajectory_step_readout`: keyword matched `10/60`, average recall
      `25.03%`, basis keyword matched `20/60`, basis average recall `32.25%`.
  - Interpretation: structured trajectory-state facts are meaningfully more
    useful than simple step concatenation on this small public smoke subset,
    but exact answer support remains zero. This supports the trajectory
    basis direction while keeping the next gate as larger public runs and
    API-backed answer/judge validation.
- Re-ran deterministic validation:
  - `python -m pytest`
  - Result: `72 passed`.
  - `git diff --check`
  - Result: clean.

## 2026-05-30 public AMA pilot workflow

- Added `src/adamem/pilot.py` and the `adamem-pilot` console entry:
  - `ama-public` downloads or copies a bounded AMA-style JSONL prefix.
  - It writes raw JSONL, converted answer/evidence AdaMem JSONL, Markdown
    reports, per-query records JSONL, and compact experiment JSON.
  - `--answer-only` skips the separate evidence-mode conversion/eval for larger
    smoke runs, because answer-mode reports already include evidence support
    and answerability diagnostics.
  - Experiment JSON now stores compact aggregate results and points to records
    JSONL instead of embedding all per-query records.
- Added `tests/test_pilot.py`:
  - JSONL prefix copying validates objects and obeys limits.
  - Local AMA pilot runs without network and writes reports/experiments.
  - `--answer-only` omits evidence-mode outputs and keeps experiment
    `raw_outputs` empty.
- Attempted larger public AMA pilots with `full` included:
  - `limit=20` full answer+evidence and `limit=20` answer-only exceeded the
    useful smoke-test runtime and were stopped.
  - `limit=10` answer-only with `full` was also too slow for an interactive
    smoke run.
  - Interpretation: graph-heavy `full` needs separate performance work or a
    smaller reported sample; it should not be part of the default large public
    AMA smoke command yet.
- Ran a reproducible public AMA 20-episode light pilot:
  - Command:
    `PYTHONPATH=src python -m adamem.pilot ama-public --limit 20 --output-dir results/ama_public_20_light --baselines semantic_only trajectory_step_readout --top-k 8 --answer-only --json`
  - Dataset: first 20 public AMA-Bench test episodes from Hugging Face,
    producing 240 QA records.
  - Exact answer-string support remains `0/240` for both baselines.
  - Evidence support:
    - `semantic_only`: `34/239`.
    - `trajectory_step_readout`: `239/239`.
  - Answerability diagnostics:
    - `semantic_only`: keyword matched `12/240`, average recall `15.36%`,
      basis matched `14/240`, basis average recall `15.68%`.
    - `trajectory_step_readout`: keyword matched `20/240`, average recall
      `20.54%`, basis matched `32/240`, basis average recall `24.34%`.
  - Interpretation: step-authorized retrieval and structured trajectory basis
    continue to beat semantic retrieval at 20 public episodes, but the result
    remains a retrieval/answerability claim. Final paper evidence still needs
    larger runs, answer synthesis, and LLM-judge scoring.
- Re-ran deterministic validation:
  - `python -m pytest`
  - Result: `75 passed`.
  - `git diff --check`
  - Result: clean.
  - `PYTHONPATH=src python -m adamem.pilot ama-public --help`
  - Result: CLI help renders successfully.

## 2026-05-30 full baseline scalability

- Investigated the slow public AMA pilot when `full` was included:
  - `full` on the first two cases took `1.827s`, while `semantic_only` and
    `trajectory_step_readout` each took `0.017s`.
  - Per-case profiling showed long trajectory episodes were the bottleneck:
    350 observations took `5.744s`, and a 756-observation episode took
    `83.252s` before optimization.
  - The bottleneck is write-side soft staleness / propagation over long raw
    trajectories, plus large retrieval candidate pools.
- Added bounded engineering controls:
  - `candidate_pool_limit` in `AdaMemConfig`, applied before MMR.
  - `soft_stale_candidate_limit`, scanning recent prior candidates first.
  - `stale_propagation_seed_limit`, bounding propagation fanout from directly
    marked stale items.
  - Added tests for candidate-pool limiting and soft-stale candidate limiting.
- Added pilot runtime timing:
  - `run_ama_public_pilot` now reports source, conversion, eval, and total
    seconds.
  - Experiment JSON records `benchmark_seconds` and keeps per-query records in
    sidecar JSONL.
- Re-profiled `full` on the first 20 public AMA episodes:
  - The 756-observation case dropped from `83.252s` to `6.141s`.
  - The 1050-observation case took `13.163s`.
  - Full 20-case `full` benchmark time: `30.47s`.
- Re-ran the 20-episode public AMA pilot with `full` included:
  - Command:
    `PYTHONPATH=src python -m adamem.pilot ama-public --limit 20 --output-dir results/ama_public_20_full --baselines semantic_only full trajectory_step_readout --top-k 8 --answer-only --json`
  - Timings: source `2.3332s`, answer conversion `0.0542s`, answer eval
    `30.2272s`, total `32.6146s`.
  - Evidence support:
    - `semantic_only`: `34/239`.
    - `full`: `0/239`.
    - `trajectory_step_readout`: `239/239`.
  - Answerability diagnostics:
    - `full`: keyword matched `19/240`, average recall `19.07%`.
    - `trajectory_step_readout`: keyword matched `20/240`, average recall
      `20.54%`, basis matched `32/240`, basis average recall `24.34%`.
  - Interpretation: the scalability fix makes the default full baseline
    runnable on the 20-episode public AMA pilot, and the result strengthens the
    method claim that explicit trajectory-step authorization is not recovered
    by generic full-memory scoring.
- Added per-metadata diagnostics to JSONL benchmark reports:
  - `diagnostics_by_metadata` now records evidence support and answerability
    metrics for every grouped metadata value, not only exact pass/accuracy.
  - Markdown reports include sections such as `By question_type Diagnostics`.
  - This matters for AMA because exact answer-string support is zero, while
    evidence and answerability expose useful A/B/C/D differences.
- Re-ran the 20-episode public AMA pilot report with grouped diagnostics:
  - `trajectory_step_readout` evidence support by AMA type:
    - A: `79/79`.
    - B: `60/60`.
    - C: `60/60`.
    - D: `40/40`.
  - `semantic_only` evidence support by AMA type:
    - A: `12/79`.
    - B: `8/60`.
    - C: `10/60`.
    - D: `4/40`.
  - `trajectory_step_readout` structured basis recall by AMA type:
    - A: `25.00%`.
    - B: `24.46%`.
    - C: `23.62%`.
    - D: `23.91%`.
  - `semantic_only` basis recall by AMA type:
    - A: `16.25%`.
    - B: `15.64%`.
    - C: `16.50%`.
    - D: `13.39%`.
  - Interpretation: the step-authorized evidence-recall gain is not isolated
    to a single AMA question type; it appears across recall, causal inference,
    state updating, and state abstraction categories.
- Kept grouped diagnostic report sections conditional:
  - Reports now omit all-`n/a` grouped diagnostics for datasets without
    evidence or answer metadata.
  - This keeps dynamic-state/state-slot reports readable while preserving AMA
    `question_type` diagnostics.
- Re-ran deterministic validation:
  - `python -m pytest`
  - Result: `78 passed`.
  - `git diff --check`
  - Result: clean.
  - Regenerated `results/ama_public_20_full/ama_public_20.report.md` from
    existing records to include `By question_type Diagnostics`.
- Re-ran deterministic validation:
  - `python -m pytest`
  - Result: `77 passed`.
  - `git diff --check`
  - Result: clean.
  - Pilot smoke:
    `PYTHONPATH=src python -m adamem.pilot ama-public --limit 1 --source results/ama_public_20_light/ama_public_20.raw.jsonl --output-dir /tmp/adamem_pilot_smoke --baselines semantic_only full trajectory_step_readout --top-k 8 --answer-only --json`
  - Result: completed locally with `total_seconds=0.3877`.

### 2026-05-30 state premise correction checkpoint

- Added the first explicit Premise Resistance mechanism:
  `use_state_premise_correction`.
  - When a routed query mentions an inactive state value and an active value
    exists for the same slot, AdaMem now emits an ephemeral `state_correction`
    readout with the stale value, current value, active-state source id, and
    stale-state id.
  - The correction result is intentionally not stored back into memory, so it
    acts as a read-time authorization/correction surface rather than a new
    durable belief.
  - This targets the STALE failure mode where a user query presupposes old
    state, e.g. "Since I am in Seattle..." after the active location has
    changed to Boston.
- Added canonical baseline `semantic_state_premise_correction`, which layers
  the correction mechanism on top of semantic-only state readout plus
  state-source adjudication.
- Added focused deterministic tests:
  - A stale-premise location query surfaces `state_correction` before ordinary
    state/readout evidence.
  - A state-sensitive query without an explicit stale value does not create a
    correction result.
- Updated `docs/research_workflow.md` and `docs/literature_to_design.md` to
  define the mechanism, ablation boundary, and required next evidence.
- Claim boundary: this is currently a mechanism and trace-level improvement
  only. It does not support an answer-accuracy or SOTA claim until STALE
  Premise Resistance cases are evaluated with real answer/judge models.
- Diagnostic caveat: generic substring support checks that penalize forbidden
  old values are not yet appropriate for this baseline, because the correction
  readout intentionally names the stale premise before rejecting it. Add
  correction-opportunity and correction-hit diagnostics before using this
  baseline in paper tables.

### 2026-05-30 premise-correction diagnostics

- Resolved the first diagnostic caveat for premise correction:
  - JSONL benchmark support checks now exclude `state_correction` text when
    deciding whether a forbidden old value was exposed as unresolved evidence.
    The same old value is still counted as a failure if it appears in ordinary
    retrieved evidence.
  - Case records now include `corrected_forbidden` and
    `premise_correction_count`, making corrected stale premises auditable
    without hiding raw stale exposure.
- Extended STALE retrieval diagnostics:
  - Added premise-correction opportunity rate, hit rate, and best correction
    rank to `StaleQueryDiagnostic` / `StaleDiagnosticResult`.
  - Trace records now mark `kind`, selected state metadata, and
    `is_premise_correction`.
  - Failure records now distinguish `premise_correction_missing` from generic
    stale exposure.
- Smoke results:
  - Dynamic-state JSONL smoke with `semantic_state_adjudication` and
    `semantic_state_premise_correction`: both scored `7/7`; the correction
    variant surfaced explicit corrections for passport, workflow, and runtime
    stale-premise queries.
  - STALE mini diagnostic with the same pair showed current recall `100%` for
    both. `semantic_state_premise_correction` reached `100%` premise-correction
    hit on the detected opportunities in that mini run, while adjudication-only
    stayed at `0%`.
- Added tests to lock both behaviors:
  - `test_jsonl_benchmark_treats_correction_text_as_resolved_forbidden_support`
  - `test_retrieval_diagnostics_measure_premise_correction_hits`

### 2026-05-30 STALE retrieval diagnostic tables

- Extended `adamem.tables` to support `stale_retrieval_diagnostics`
  experiment JSON directly:
  - The loader now prefers full `diagnostics` records for this run type rather
    than failure-only `raw_outputs`.
  - Markdown/JSON tables now include current recall, stale exposure, conflict
    coverage, current-before-stale, premise-old mention, premise-correction
    opportunity, premise-correction hit, and old-support adjudication.
  - Default STALE grouping switches to `dim` and `stale_type`, so diagnostic
    tables line up with STALE paper dimensions and conflict types.
- Generated a local mini artifact set for the new premise-correction diagnostic
  path:
  - `results/stale_mini_premise_correction_diagnostics.json`
  - `results/stale_mini_premise_correction_cases.jsonl`
  - `results/stale_mini_premise_correction_report.md`
  - `results/stale_mini_premise_correction_tables.md`
  - `results/stale_mini_premise_correction_tables.json`
- Updated report bundles for this record kind:
  - `adamem.reporting` now writes paper tables and claim audits for
    `stale_retrieval_diagnostics` without trying to run paired comparison on
    aggregate diagnostic records.
  - The manifest records `paired_comparison_skipped` and points users to
    diagnostic tables or case-level records for paired analysis.
- Mini result interpretation:
  - `semantic_state_adjudication` and
    `semantic_state_premise_correction` both reached `100%` current recall and
    `0%` stale exposure on the one-case mini run.
  - The correction baseline changed the key paper-facing diagnostic:
    premise-correction hit increased from `0%` to `100%` on detected
    opportunities.
  - This is still mini-fixture evidence only. Full STALE data is not currently
    present in the workspace, so this should be treated as workflow validation
    and mechanism debugging, not as a paper result.

### 2026-05-30 benchmark data status and LongMemEval transfer check

- Checked current STALE availability:
  - arXiv/Hugging Face describe STALE as 400 expert-validated conflict
    scenarios and 1,200 queries across State Resolution, Premise Resistance,
    and Implicit Policy Adaptation.
  - The Hugging Face paper page currently lists no linked dataset.
  - Local workspace has only `benchmarks/stale_mini.jsonl` with 2 smoke cases;
    `benchmarks/stale.adamem.jsonl` is not present.
- Added `docs/benchmark_data_status.md` to track benchmark availability,
  claim boundaries, and reproduction commands for STALE, LongMemEval-S, and
  AMA public pilots.
- Ran a LongMemEval-S balanced 60 no-regression transfer check for the new
  premise-correction baseline:
  - Conversion command:
    `PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_premise_correction.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10`
  - Evaluation command:
    `PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60_premise_correction.adamem.jsonl --baselines semantic_state_adjudication semantic_state_premise_correction --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_premise_correction_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_premise_correction_report.md --experiment-output results/longmemeval_s_balanced_60_premise_correction.json`
  - Result: both baselines scored `39/60` evidence support.
  - Paired comparison versus `semantic_state_adjudication`: gained `0`, lost
    `0`, net `0` across all six question types.
  - Premise-correction readouts triggered `0` times, which is expected for this
    broad LongMemEval-S retrieval subset and supports a no-pollution /
    no-regression interpretation only.
- Generated local ignored artifacts:
  - `results/longmemeval_s_balanced_60_premise_correction_records.jsonl`
  - `results/longmemeval_s_balanced_60_premise_correction_report.md`
  - `results/longmemeval_s_balanced_60_premise_correction.json`
  - `results/longmemeval_s_balanced_60_premise_correction_tables.md`
  - `results/longmemeval_s_balanced_60_premise_correction_compare.md`

### 2026-05-30 STALE pipeline entrypoint

- Added `src/adamem/stale_pipeline.py` and console script
  `adamem-stale-pipeline`.
- The pipeline turns raw STALE JSON into a reproducible API-free diagnostic
  run directory:
  - Converts raw STALE JSON to AdaMem JSONL.
  - Runs selected retrieval-diagnostic baselines.
  - Writes diagnostic case JSONL, failure report, experiment JSON, paper
    tables, report bundle, claim audit, and manifest.
  - Supports T1/T2 filters and `--limit-per-stale-type` for balanced pilots.
- The pipeline also supports already converted AdaMem JSONL through
  `--input-format adamem-jsonl`, which lets the current `benchmarks/stale_mini.jsonl`
  fixture exercise the same artifact path.
- CLI smoke:
  `PYTHONPATH=src python -m adamem.stale_pipeline benchmarks/stale_mini.jsonl --output-dir /tmp/adamem_stale_pipeline_smoke --run-name stale_mini_pipeline --input-format adamem-jsonl --baselines semantic_state_adjudication semantic_state_premise_correction --top-k 8 --max-cases 1 --json`
  - Result: wrote converted dataset, experiment JSON, diagnostic case records,
    diagnostic report, Markdown/JSON paper tables, report bundle, and manifest.
- Added tests:
  - `test_stale_diagnostic_pipeline_writes_reproducible_artifacts`
  - `test_stale_pipeline_cli_writes_manifest_json`
  - `test_stale_pipeline_accepts_converted_jsonl_input`

### 2026-05-30 premise-correction transfer reporting

- Added premise-correction aggregate metrics to JSONL retrieval benchmark
  summaries and Markdown reports:
  - `correction_records`
  - `correction_items`
  - `corrected_forbidden_records`
  - `unresolved_forbidden_records`
  - corresponding rates in machine-readable `paper_metrics`
- This makes non-STALE transfer/no-pollution checks reproducible from ordinary
  benchmark reports instead of ad hoc scripts.
- Re-ran the LongMemEval-S balanced 60 premise-correction transfer report:
  - `semantic_state_adjudication`: `39/60` evidence support.
  - `semantic_state_premise_correction`: `39/60` evidence support.
  - `Premise Correction` report section: both baselines had `0/60`
    correction records, `0` correction items, and `0` unresolved forbidden
    records.
  - Interpretation: this public-transfer subset does not test correction
    usefulness, but it now provides auditable evidence that enabling the
    correction mechanism did not trigger spurious correction readouts or degrade
    retrieval support on the balanced LongMemEval-S pilot.

### 2026-05-30 claim audit transfer evidence

- Extended `adamem.claims` to read JSONL sidecar records referenced by
  `notes.records_path` and audit two narrow retrieval-transfer claims:
  - `paired_retrieval_no_regression`: supported only when a candidate baseline
    has at least 10 paired retrieval cases against the reference and loses zero
    cases on the selected retrieval metric.
  - `premise_correction_no_trigger_on_transfer`: supported only when a
    premise-correction baseline has at least 10 records and emits zero
    correction readouts on a non-STALE transfer benchmark.
- Added machine-readable `claim_evidence.retrieval_transfer` and Markdown
  `Claim Evidence` output with paired counts and premise-correction counts.
- Re-ran claim audit on
  `results/longmemeval_s_balanced_60_premise_correction.json`:
  - Supported claims: `retrieval_diagnostics`,
    `answerability_diagnostics`, `paired_retrieval_no_regression`,
    `premise_correction_no_trigger_on_transfer`.
  - Paired metric: `evidence_support_matched`.
  - Pair: `semantic_state_premise_correction` vs
    `semantic_state_adjudication`, common `60`, gained `0`, lost `0`,
    net `0`.
  - Premise correction: `60` records, `0` correction records, `0` correction
    items, `0` unresolved forbidden records.
  - Answer accuracy and SOTA remained blocked, preserving claim boundaries.
- Added unit coverage:
  - `test_claim_audit_supports_paired_retrieval_no_regression`
- Validation:
  - `python -m pytest tests/test_claims.py -q`
  - `python -m pytest -q`
