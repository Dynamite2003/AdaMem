# AdaMem Agent Notes

## Project Objective

AdaMem is a research-first prototype for studying how agent memory systems can
handle stale or invalidated memories. The main goal is to identify and validate
improvements over mainstream agent memory systems, with STALE-style implicit
conflict handling as the primary research target.

The project should optimize for CCF-A-level paper quality before production
packaging. Strong benchmark results are desirable, including SOTA-level results
when feasible, but claims must stay proportional to completed evaluations.

## Research Bar

Treat every major design decision as if it may appear in a top-tier paper.
Prototype convenience is acceptable only when it does not weaken the scientific
claim. A change is valuable when it improves at least one of these paper-facing
properties:

- Novelty: it identifies a memory failure mode or mechanism not adequately
  handled by existing agent memory systems.
- Causal validity: it makes stale-memory behavior improve for the intended
  reason, not through benchmark leakage or prompt tricks.
- Generality: it helps beyond one handcrafted fixture and transfers across
  STALE types, dimensions, models, and related memory benchmarks.
- Measurability: it creates clean metrics, ablations, traces, or error
  categories that can support a defensible claim.
- Reproducibility: it makes runs, prompts, configs, data splits, and model
  settings easier to reproduce.

For paper work, prefer a smaller, well-validated mechanism over a larger system
whose gains cannot be explained.

## Primary Research Scope

Focus on memory behavior that helps agents decide what is currently valid:

- State revision when later evidence invalidates earlier beliefs.
- Premise resistance when a user query presupposes stale information.
- Implicit policy adaptation when a current state should guide downstream
  behavior even if the query does not explicitly mention the old or new belief.
- Propagation from one changed state to related state slots, especially for
  indirect stale conflicts.
- Transparent scoring, adjudication, and ablations so mechanisms can be tested
  independently.

The project is not intended to become a full agent harness. Evaluation adapters,
answer generation, and LLM judge utilities are allowed when they support memory
research, but the core contribution should remain the memory layer and its
state/staleness mechanisms.

## Current Capability Map

- `src/adamem/manager.py`: core `AdaMem` API, write/manage/read logic, graph
  expansion, soft staleness, propagation, adjudication filter, context packing.
- `src/adamem/config.py`: mechanism switches and scoring weights. New research
  mechanisms should usually be ablatable here.
- `src/adamem/schema.py`: memory/result dataclasses.
- `src/adamem/state.py`: deterministic API-free typed state extractor prototype,
  pluggable state patch type, state-readout query detector, wildcard state slot
  matcher, and initial state dependency topology.
- `src/adamem/store.py`: store protocol, in-memory store, JSON prototype store.
- `src/adamem/text.py`: tokenizer, hashed bag-of-words embedder, memory key
  helper.
- `src/adamem/baselines.py`: stable baseline registry for paper tables and
  runnable ablation configs.
- `src/adamem/bench.py`: thin JSONL retrieval benchmark runner and ablation
  definitions.
- `src/adamem/diagnostics.py`: API-free STALE retrieval diagnostics for
  current-evidence recall, stale exposure, conflict coverage, and adjudication
  signals, case-level JSONL records, and Markdown failure reports for error
  analysis.
- `src/adamem/eval.py`: synthetic benchmark and STALE LLM-judge evaluation.
- `src/adamem/experiments.py`: experiment record schema and JSON writer for
  reproducible runs.
- `src/adamem/convert.py`: LoCoMo and STALE converters.
- `src/adamem/llm.py`: provider-agnostic HTTP clients for answer/judge calls.
- `benchmarks/`: small fixtures and local smoke data.
- `tests/`: deterministic unit and adapter tests.

## Evaluation Priorities

1. STALE is the primary benchmark because it directly tests invalidated memory.
2. LoCoMo, LongMemEval, AMA-Bench, STATE-Bench, and related agent-memory
   benchmarks are secondary but should be used to check generality.
3. Synthetic tests are CI guards only. They prove mechanisms are separable; they
   do not support SOTA claims.
4. Retrieval-only diagnostics are useful, but final claims need end-to-end
   answer scoring with a documented answer model, judge model, prompt, top-k,
   dataset split, and random seed or sampling settings.

Paper-level evaluation should include:

- Strong baselines from mainstream memory systems or faithful reimplementations
  where official code is unavailable.
- Multiple answer models and at least one judge robustness check, because a
  single answer/judge pair is too fragile for a top-tier claim.
- Per-dimension and per-type breakdowns for STALE, especially State Resolution,
  Premise Resistance, and Implicit Policy Adaptation.
- Retrieval diagnostics separated from answer correctness, including current
  evidence recall, stale evidence exposure, and stale premise correction.
- Ablations for every named mechanism and interaction ablations for mechanisms
  that are expected to work together.
- Error analysis with representative failures, not only aggregate scores.

## Design Constraints

- Keep the core package small and modular. Dependencies are allowed when they
  materially improve the research path, but avoid coupling the core API to one
  provider or storage backend.
- `AdaMem.observe`, `AdaMem.retrieve`, `AdaMem.context`, `MemoryStore`, and
  `AdaMemConfig` can change, but breaking changes should be recorded in docs or
  release notes.
- Every major mechanism should be controlled by config flags so ablations can
  isolate its effect.
- Retrieval results should continue exposing score contributions or equivalent
  trace data.
- Benchmark ground-truth fields such as STALE `M_old`, `M_new`, explanations,
  and query metadata are evaluation-only. Do not use them inside the memory
  system except in explicitly named oracle/debug experiments.
- Prefer write-side state adjudication and authorized readout over relying only
  on query-time prompt instructions.
- Do not claim production readiness without durable store, concurrency,
  latency, privacy, and failure-mode work.
- Do not use STALE `M_old`, `M_new`, explanations, relevant session indices, or
  answer labels inside the proposed runtime method. Those fields may be used
  only for conversion, diagnostics, oracle upper bounds, or evaluation.
- Do not optimize only for the in-repo mini fixtures. Treat them as debugging
  aids, not development targets.
- Avoid prompt-only fixes unless they are explicitly framed as baselines. The
  main contribution should be a memory mechanism, representation, or retrieval
  policy.

## Known Design Risks

- The default `hashed_bow` embedder is useful for deterministic tests but weak
  for real STALE cases, especially paraphrased or implicit conflicts.
- Current soft staleness mostly detects lexical similarity. STALE requires
  semantic state inference, so old and new beliefs may not overlap enough to be
  linked.
- Current storage ingests raw dialogue turns. Real stale-memory handling likely
  needs extracted state candidates or compact state records in addition to raw
  episodes.
- Query similarity can amplify stale premises. Premise Resistance queries often
  mention the old state directly, so naive retrieval may rank stale evidence
  above current evidence.
- Current ADR/SLR-style diagnostics are approximate. Keyword matching from
  `M_old` can overcount common words and inflate adjudication or leakage
  metrics.
- The answer prompt currently says to prefer recent excerpts when conflicts
  appear, but this is not a substitute for an authorized current-state readout.
- `benchmarks/stale_mini.jsonl` is large for a fixture and should be treated as
  local smoke data unless the repo intentionally keeps benchmark samples.

## Promising Research Direction

The next design iteration should move from similarity-only stale detection to a
state-aware memory layer:

- Extract candidate state updates from new observations.
- Assign each state to a typed slot such as location, schedule, preference,
  health constraint, relationship, task status, resource status,
  workflow/runbook rule, or runtime/tool status.
- Mark each slot value as active, stale, replaced, or unknown-current.
- Propagate invalidation through a small dependency topology between state
  slots.
- At read time, prepend or prioritize an authorized current-state basis before
  raw retrieved episodes.
- Block or explicitly correct stale premises when the query conflicts with the
  authorized current state.

This direction is aligned with the STALE paper's finding that current systems
often retrieve updated evidence but fail to act on it:
https://arxiv.org/abs/2605.06527

## Paper-Track Milestones

1. Establish a trustworthy STALE evaluation harness with no ground-truth leakage
   into runtime memory, stable prompts, cached raw outputs, and rerunnable
   configs.
2. Reproduce meaningful baselines and record exact settings, costs, and failure
   modes.
3. Build the first state-aware AdaMem variant and test whether it improves
   STALE answer accuracy for the intended dimensions.
4. Add ablations and diagnostics that explain where each mechanism helps or
   hurts.
5. Validate transfer on at least one additional public memory benchmark.
6. Convert the best mechanism into a crisp paper contribution: problem framing,
   method, theory or intuition, experiments, limitations, and reproducibility
   artifacts.

## Implementation Discipline

- Read existing tests before changing behavior.
- Follow `docs/research_workflow.md` when planning evaluation, baseline, or
  paper-track work.
- Use `docs/literature_to_design.md` to connect new mechanisms to papers,
  baseline gaps, hypotheses, and evaluation gates.
- Keep `docs/progress_log.md` updated when a meaningful design decision,
  experiment, implementation change, or scope change happens.
- Keep new mechanisms deterministic where possible so local ablations stay
  reproducible.
- Add focused tests for each mechanism and at least one ablation-level test when
  changing scoring, filtering, or benchmark logic.
- Keep benchmark scripts honest: separate diagnostic metrics from final answer
  accuracy.
- When adding LLM-dependent functionality, provide a mockable interface and a
  deterministic unit test path.
- Avoid using private credentials, full benchmark answers, or judge-only
  metadata in runtime memory code.

## Useful Local Commands

```bash
python -m pytest
PYTHONPATH=src python -m adamem.eval
PYTHONPATH=src python -m adamem.eval --list-baselines
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/tiny_memory_qa.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 1 --experiment-output results/dynamic_state_transfer_smoke.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json benchmarks/longmemeval_s.adamem.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/longmemeval_s.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_propagation full state_readout state_propagation --max-cases 20 --experiment-output results/longmemeval_transfer_pilot.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10
PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_state_adjudication_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_state_adjudication_report.md --experiment-output results/longmemeval_s_balanced_60_state_adjudication_pilot.json
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication state_readout --stale-types T1 T2 --limit-per-stale-type 10 --experiment-output results/stale_balanced20_state_adjudication_diagnostics.json --diagnostic-cases-output results/stale_balanced20_state_adjudication_cases.jsonl --diagnostic-report-output results/stale_balanced20_state_adjudication_report.md
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --experiment-output results/stale_diagnostics_smoke.json
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output results/stale_diagnostic_cases.jsonl
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output results/stale_diagnostic_cases.jsonl --diagnostic-report-output results/stale_failure_report.md
PYTHONPATH=src python -m adamem.eval --stale benchmarks/stale_mini.jsonl --answer-provider mock --judge-provider mock --max-cases 1 --experiment-output results/stale_pilot_mock.json
PYTHONPATH=src python -m adamem.convert locomo data/locomo10.json benchmarks/locomo10.adamem.jsonl
PYTHONPATH=src python -m adamem.convert stale data/T1_T2_400_FULL.json benchmarks/stale.adamem.jsonl
```
