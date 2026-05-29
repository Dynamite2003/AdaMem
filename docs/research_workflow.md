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
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/tiny_memory_qa.jsonl
```

Outputs:

- Clean test result.
- Baseline synthetic/JSONL tables.
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
- Stale premise correction opportunity: whether a Premise Resistance query
  mentions an old state and the system has enough current-state evidence to
  correct it.

Outputs:

- Hardened diagnostic code.
- Deterministic tests for metric edge cases.
- A small debug report on `benchmarks/stale_mini.jsonl`.

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
- State-aware AdaMem variants.
- Mainstream memory systems or faithful local approximations when official code
  is unavailable.

Actions:

- Define exact configs for each baseline.
- Assign stable names to every config.
- Make sure each baseline can be invoked by CLI or function call.
- Store config dictionaries with experiment outputs.

Outputs:

- Baseline registry.
- Mapping from paper table names to runnable configs.

Done when:

- A single command can enumerate all planned baselines.
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
- Oracle extractor only for upper-bound/debug experiments, clearly separated
  from the proposed method.

Required ablations:

- No state extraction.
- State extraction only.
- State extraction plus stale adjudication.
- State extraction plus authorized readout.
- State extraction plus propagation.
- Full state-aware AdaMem.

Outputs:

- Method design note.
- Data structures and config flags.
- Unit tests using synthetic examples.

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
- Short decision note: proceed, fix harness, or redesign method.

Done when:

- End-to-end results are credible enough to scale.
- At least one manual audit sample confirms the judge is behaving reasonably.

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

1. Tighten STALE diagnostics so ADR/SLR-style metrics do not overcount common
   words.
2. Add an experiment output format with cached prompts, configs, raw answers,
   and raw judge outputs.
3. Define a baseline registry with stable names and runnable configs.
4. Draft a state-aware method design and implement a deterministic mock
   extractor.
5. Add synthetic tests that specifically cover STALE dimensions 1, 2, and 3.
6. Run retrieval-only diagnostics on `benchmarks/stale_mini.jsonl` and write a
   failure taxonomy.
7. Prepare scripts so API-enabled pilot runs need only provider/model/key
   settings.

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

