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

Phase 1 from `docs/research_workflow.md`: Evaluation Harness Hardening.

The next high-value task is to make retrieval diagnostics and experiment
records trustworthy before API-enabled answer/judge runs.

## Confirmed Project Constraints

- Research paper quality is prioritized over production packaging.
- The core contribution should be a memory mechanism, representation, or
  retrieval policy, not a complete agent harness.
- API-dependent evaluation can wait until provider keys are available.
- API-free work should prepare the harness, baselines, diagnostics, method
  design, and reproducibility path.
- STALE ground-truth fields must not be used inside proposed runtime memory.
- Breaking API changes are allowed when useful, but must be documented.

## Completed Work

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

1. Implement stricter STALE retrieval diagnostics.
2. Add tests for metric edge cases and no-label-leakage behavior.
3. Add an experiment output schema for configs, prompts, raw answers, raw judge
   outputs, and metadata.
4. Create a baseline registry with stable names and runnable configs.
5. Draft and then implement the first state-aware memory prototype with a
   deterministic mock extractor.
6. Add synthetic tests aligned to STALE dimensions 1, 2, and 3.
7. Prepare API-enabled pilot scripts for later provider keys.

## Change Log

### 2026-05-28

- Added `AGENTS.md`.
- Added `docs/research_workflow.md`.
- Added `docs/progress_log.md`.

