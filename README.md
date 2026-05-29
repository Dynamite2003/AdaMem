# AdaMem

AdaMem is a minimal, plug-and-play memory layer for LLM agents. It is designed as a small library rather than a full agent harness: wire `observe()` after an agent step, call `retrieve()` before the next prompt, and swap in your own extractor, embedder, or store when needed.

The first implementation focuses on three ideas:

- **Delta memory:** new observations can supersede older active facts instead of creating stale duplicates.
- **Causal/temporal retrieval:** retrieval is not only similarity search; it can expand through explicit links, causes, and temporal validity.
- **Built-in ablations:** every scoring signal is controlled by `AdaMemConfig`, and retrieval returns score contributions for experiment traces.

## Quick Start

```python
from adamem import AdaMem, AdaMemConfig

mem = AdaMem(config=AdaMemConfig())

issue = mem.observe(
    "The checkout failure was caused by a missing STRIPE_SECRET in production.",
    kind="observation",
    importance=0.8,
    metadata={"memory_key": "checkout.failure.root_cause"},
)

mem.observe(
    "Set STRIPE_SECRET in production and checkout succeeded.",
    kind="outcome",
    importance=0.9,
    cause_ids=[issue.id],
    metadata={"memory_key": "checkout.failure.status"},
)

context = mem.context("Why did checkout fail last time?", max_chars=1200)
print(context)
```

## Ablation Example

```python
from adamem import AdaMemConfig

semantic_only = AdaMemConfig(
    use_graph=False,
    use_temporal=False,
    use_importance=False,
    use_recency=False,
    use_mmr=False,
)
```

Suggested first ablations:

- semantic-only retrieval
- semantic + temporal validity
- semantic + graph expansion
- semantic + graph + delta supersession
- full AdaMem scoring with MMR context packing
- state-aware AdaMem with derived state memories and authorized state readout

## Synthetic Ablation

Run the deterministic smoke benchmark:

```bash
PYTHONPATH=src python -m adamem.eval
```

Current expected result:

```text
semantic_only       1/4
semantic_importance 1/4
semantic_temporal   2/4
semantic_graph      2/4
a_mem_evolution     2/4
zep_temporal_kg     3/4
mem0_extraction     0/4
delta_graph         3/4
delta_soft          3/4
delta_propagation   3/4
delta_full          3/4
full                4/4
state_memory        4/4
semantic_state_readout 1/4
semantic_state_propagation 1/4
semantic_state_adjudication 1/4
semantic_state_propagation_adjudication 1/4
state_readout       4/4
state_propagation   4/4
```

This is not a substitute for LoCoMo, LongMemEval, or AMA-Bench, but it proves the local mechanisms are independently ablatable before paying for larger evaluations.

## State-Aware Prototype

AdaMem includes an early API-free state-aware prototype for STALE-style
experiments. When enabled, it extracts narrow typed user state updates such as
current location, beverage preference, schedule availability, and task status
from observations, writes derived `state` memories, supersedes older state
values, and can surface current state before raw episodic evidence for
state-sensitive queries.

Derived `state` memories are hidden from ordinary direct retrieval by default
and enter results through an authorized readout path only. This keeps state
summaries from polluting generic public benchmark retrieval while preserving a
clean ablation switch for the boundary.

The `semantic_state_adjudication` baseline additionally marks raw evidence
behind replaced state values and filters that evidence only for queries routed
to the same state slot. This is intentionally narrower than the global
adjudication filter: historical queries can still retrieve old episodes, while
current-state queries avoid stale raw support.

This prototype is intentionally narrow and deterministic. It is meant to test
the research hypothesis before spending API budget, not to replace a robust LLM
extractor. The extractor is pluggable so API-enabled or domain-specific
extractors can be evaluated under the same memory/readout mechanism.

The `state_propagation` baseline additionally tests typed dependency
propagation: for example, a changed location can invalidate dependent local
state such as `local.*` records and their source evidence.

The `a_mem_evolution` baseline is an API-free approximation of A-MEM-style
agentic memory notes. It adds deterministic note keywords, dynamic links, and
write-time memory evolution over raw episodes. It is a mainstream-design
comparison, not AdaMem's proposed state-authority mechanism.

The `zep_temporal_kg` baseline is an API-free approximation of Zep/Graphiti-
style temporal KG memory. It writes temporal fact edges from extracted state,
invalidates old edges when a relation changes, and exposes active KG facts for
state-sensitive queries. It intentionally does not perform AdaMem's raw-source
adjudication, so it can test whether temporal KG readout alone is enough.

The `mem0_extraction` baseline is an API-free approximation of Mem0-style
compact memory extraction and update. It keeps raw observations only as audit
sources, retrieves extracted compact facts, and replaces older facts for the
same slot. It tests whether a compact extraction-only memory is sufficient
without raw-evidence retrieval plus query-scoped adjudication.

## JSONL Benchmark Adapter

Run a retrieval-support ablation over a thin JSONL format:

```bash
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/tiny_memory_qa.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/tiny_memory_qa.jsonl --baselines semantic_only full --max-cases 1 --experiment-output results/tiny_memory_qa_smoke.json
```

Each line is one memory episode:

```json
{"id":"case-1","observations":[{"label":"cause","content":"TX91 token was missing.","importance":0.9}],"queries":[{"id":"q1","query":"Which token was missing?","expected_substrings":["TX91"],"top_k":2}]}
```

Observations may include `metadata.tags`, `metadata.keywords`, `metadata.subject`, and `metadata.predicate`; AdaMem indexes those structured attributes with the content. This adapter is intentionally narrow: it checks whether retrieved context contains expected support evidence. Full answer generation and LLM-as-judge evaluation can sit one layer above it.

## LoCoMo Converter

Convert the official `locomo10.json` file into the same thin JSONL format:

```bash
PYTHONPATH=src python -m adamem.convert locomo data/locomo10.json benchmarks/locomo10.adamem.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/locomo10.adamem.jsonl
```

By default, LoCoMo evaluation checks whether retrieved context contains the annotated evidence ids such as `D1:3`. Use `--expected answer` or `--expected both` to switch the support criterion.

On the first official LoCoMo sample (`--limit 1`, `--top-k 8`), the current default full configuration retrieves 52/152 evidence supports versus 50/152 for semantic-only. This is a smoke test, not a SOTA claim.

See [docs/design.md](docs/design.md) for the research notes and experiment plan.

## Research Workflow

For paper-track development, use [docs/research_workflow.md](docs/research_workflow.md)
as the phase checklist and keep [docs/progress_log.md](docs/progress_log.md)
updated after meaningful design decisions, experiments, implementation changes,
or scope changes.

Use [docs/literature_to_design.md](docs/literature_to_design.md) to keep
mechanism ideas tied to real papers, baseline gaps, hypotheses, and evaluation
gates.

Useful API-free commands:

```bash
python -m pytest
PYTHONPATH=src python -m adamem.eval --list-baselines
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/dynamic_state_transfer.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 1 --experiment-output results/dynamic_state_transfer_smoke.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json benchmarks/longmemeval_s.adamem.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/longmemeval_s.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_propagation full state_readout state_propagation --max-cases 20 --experiment-output results/longmemeval_transfer_pilot.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10
PYTHONPATH=src python -m adamem.eval --dataset /tmp/longmemeval_s_balanced_60.adamem.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication --max-cases 60 --benchmark-cases-output results/longmemeval_s_balanced_60_state_adjudication_records.jsonl --benchmark-report-output results/longmemeval_s_balanced_60_state_adjudication_report.md --experiment-output results/longmemeval_s_balanced_60_state_adjudication_pilot.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_audit_probe.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-output results/longmemeval_s_balanced_60_state_audit_candidates.jsonl --state-audit-summary-output results/longmemeval_s_balanced_60_state_audit_summary.json
PYTHONPATH=src python -m adamem.convert ama data/ama_bench.jsonl benchmarks/ama_bench.adamem.jsonl --expected answer --top-k 8
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --baselines semantic_only semantic_state_readout semantic_state_adjudication semantic_state_propagation_adjudication state_readout --max-cases 2 --experiment-output results/stale_mini_state_adjudication_diagnostics.json --diagnostic-cases-output results/stale_mini_state_adjudication_cases.jsonl --diagnostic-report-output results/stale_mini_state_adjudication_report.md
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale.adamem.jsonl --baselines semantic_only semantic_state_adjudication semantic_state_propagation_adjudication state_readout --stale-types T1 T2 --limit-per-stale-type 10 --experiment-output results/stale_balanced20_state_adjudication_diagnostics.json --diagnostic-cases-output results/stale_balanced20_state_adjudication_cases.jsonl --diagnostic-report-output results/stale_balanced20_state_adjudication_report.md
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --experiment-output results/stale_diagnostics_smoke.json
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output results/stale_diagnostic_cases.jsonl
PYTHONPATH=src python -m adamem.eval --stale-diagnostics benchmarks/stale_mini.jsonl --max-cases 2 --diagnostic-cases-output results/stale_diagnostic_cases.jsonl --diagnostic-report-output results/stale_failure_report.md
PYTHONPATH=src python -m adamem.eval --stale benchmarks/stale_mini.jsonl --answer-provider mock --judge-provider mock --max-cases 1 --experiment-output results/stale_pilot_mock.json
```

When API keys are available, replace the mock providers with real answer and
judge providers. `--experiment-output` records prompts, raw model outputs,
retrieved traces, configs, model settings, command, and commit for audit.
Use `--stale-types` and `--limit-per-stale-type` on converted STALE JSONL to
run reproducible T1/T2-balanced pilots before scaling to all 400 cases.
For `--dataset` runs, `--experiment-output` records retrieval-support
pass/fail results, query metadata, retrieved text, trace data, configs,
command, and commit without requiring answer or judge models. Use `--baselines`
to run a focused subset of canonical baseline names during public benchmark
pilots. Use `--benchmark-cases-output` and `--benchmark-report-output` to write
per-query retrieval records and grouped failure reports for error analysis.
JSONL reports include a `State Readout Exposure` table so state-summary
insertion and unmarked-query exposure can be audited separately from support
accuracy. The same records include expected/retrieved state slots and failure
modes for missing state readout, slot mismatch, and unmarked state exposure.
They also include a `Paper Metrics` table with support accuracy, net delta,
state-slot match, missing readout, slot mismatch, and unmarked state exposure.
Reports also include an `Evidence Support` table that separates answer/support
string success from evidence-label recall and graph evidence hits. This is
especially important for AMA-style trajectory runs, where the key question is
whether causal action-result edges retrieved the right trajectory step rather
than merely retrieving a semantically similar observation.
For open-ended trajectory questions, reports also include `Answerability
Diagnostics`: answer-keyword recall from retrieved context, plus recall after
adding a deterministic trajectory answer basis derived only from retrieved
step/action/observation traces. The basis can also expose deterministic
trajectory-state facts such as active rules, blocked actions, repeated
unchanged observations, and inverse action pairs. This is an API-free debugging
signal, not a replacement for LLM answer/judge accuracy.

`benchmarks/dynamic_state_transfer.jsonl` is a local non-STALE smoke fixture for
schedule, task status, preference, health/dietary, resource, workflow/runbook,
and runtime/tool state. It is useful for development but does not establish
transfer to public benchmarks.

The `longmemeval` converter targets the official cleaned LongMemEval schema.
It keeps answer/evidence labels evaluation-only and does not write
`answer_session_ids` or `has_answer` into observation metadata.

For state-authority diagnostics on LongMemEval-style public transfer runs, add
`--infer-state-slots` during conversion. This annotates query metadata from the
query text only, so reports can measure state-readout match/missing rates
without leaking answer or evidence labels into runtime memory. Treat these
inferred labels as diagnostic candidates, not ground truth; precision-audit the
marked queries before using state-readout rates as paper evidence.

For paper-facing public-transfer subsets, use the manual audit path instead of
raw inferred labels:

```bash
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_audit_probe.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-output results/longmemeval_s_balanced_60_state_audit_candidates.jsonl --state-audit-summary-output results/longmemeval_s_balanced_60_state_audit_summary.json
PYTHONPATH=src python -m adamem.convert longmemeval data/longmemeval_s_cleaned.json /tmp/longmemeval_s_balanced_60_manual_audit.adamem.jsonl --expected evidence --top-k 8 --limit-per-type 10 --state-audit-input results/longmemeval_s_balanced_60_state_audit_reviewed.jsonl
```

The reviewed JSONL must mark accepted records with `is_state_sensitive: true`.
Rejected or unreviewed records are ignored. Add `state_available: false` when
the query needs current state but the haystack does not contain a reliable
current-state record; these cases are reported separately from missing readout
failures. Candidate records include `state_evidence_candidates` extracted from
the haystack without answer or evidence labels, which helps reviewers justify
`state_available` decisions.

Use `--state-audit-summary-output` before manual review to estimate whether the
converted benchmark has enough state-available cases. The current LongMemEval-S
full-file audit yields only 14 query-state candidates and 0 deterministic
state-evidence candidates, so it should be treated as a broad retrieval
no-regression check rather than the main public state-transfer benchmark.

The `ama` converter accepts AMA-Bench-style JSON or JSONL agent trajectories.
It emits actions, observations, and environment-state snapshots as runtime
observations, links action results through `cause_labels`, and keeps answers or
evidence labels query-only. This prepares API-free tests of whether causal
trajectory structure helps beyond raw similarity retrieval. JSONL benchmark
records expose `expected_evidence`, `missing_evidence`, `graph_retrieval_count`,
and `graph_evidence_hits` for these trajectory diagnostics. For the public
AMA-Bench schema, the converter preserves `turn_idx`, `question_uuid`, and
`type`, and derives diagnostic evidence labels from `Step N` references in the
question text when explicit evidence fields are absent.

For reproducible public AMA API-free pilots, use:

```bash
PYTHONPATH=src python -m adamem.pilot ama-public --limit 20 --output-dir results/ama_public_20_light --baselines semantic_only trajectory_step_readout --top-k 8 --answer-only
```

The pilot writes a raw JSONL subset, converted AdaMem JSONL, Markdown report,
case records, and a compact experiment JSON. Use `--answer-only` for larger
API-free smoke runs because answer-mode reports already include evidence
support and answerability diagnostics. Reports include grouped diagnostics for
metadata such as AMA `question_type`, so A/B/C/D evidence and answerability can
be inspected even when exact answer-string accuracy is zero.

To regenerate compact paper-table summaries from records or experiment JSON,
use:

```bash
PYTHONPATH=src python -m adamem.tables results/ama_public_20_full/ama_public_20.answer.records.jsonl --group-fields question_type --title "AMA Public 20 API-Free Tables" --output results/ama_public_20_full/ama_public_20.paper_tables.md
PYTHONPATH=src python -m adamem.tables results/ama_public_20_full/ama_public_20.answer.experiment.json --format json --group-fields question_type --output results/ama_public_20_full/ama_public_20.paper_tables.json
```

The table utility reads benchmark records directly, or follows
`notes.records_path` from compact experiment JSON files. It reports exact
retrieval support, evidence support, answer-keyword recall, structured-basis
recall, and grouped breakdowns without manually parsing Markdown reports.

For API-key-free answer-evaluation plumbing, use the mockable answer path:

```bash
PYTHONPATH=src python -m adamem.answer_eval --dataset benchmarks/tiny_memory_qa.jsonl --baselines semantic_only --answer-provider mock --mock-answer "Office door code is 9876." --records-output /tmp/adamem_answer_eval_records.jsonl --experiment-output /tmp/adamem_answer_eval_experiment.json
```

This command is a harness smoke test, not a benchmark result. It fixes the
answer prompt, scorer interface, raw-output record format, and experiment JSON
notes before real answer and judge providers are plugged in.
Answer reports and experiment diagnostics include grouped accuracy breakdowns
for metadata such as AMA `question_type`, so end-to-end answer scoring can be
reported by A/B/C/D once real answer and judge providers are available.

The public AMA pilot can also run this answer-generation stage directly:

```bash
PYTHONPATH=src python -m adamem.pilot ama-public --limit 1 --source results/ama_public_20_light/ama_public_20.raw.jsonl --output-dir /tmp/adamem_ama_answer_generation_smoke --baselines trajectory_step_readout --top-k 4 --answer-only --run-answer-generation --answer-provider mock --mock-answer "The memory does not provide enough information."
```

Stage outputs use explicit names such as `ama_public_1.answer.records.jsonl`,
`ama_public_1.evidence.records.jsonl`, and
`ama_public_1.generation.records.jsonl` so retrieval diagnostics and answer
scoring cannot overwrite one another. The generation report includes grouped
answer accuracy tables such as `By question_type`.

The same table utility can summarize generation records:

```bash
PYTHONPATH=src python -m adamem.tables /tmp/adamem_ama_answer_generation_smoke/ama_public_1.generation.records.jsonl --group-fields question_type --title "AMA Generation Answer Tables" --output /tmp/adamem_ama_answer_generation_smoke/ama_public_1.generation.paper_tables.md
```

For generation records, the table columns switch to `correct` and `accuracy`
instead of retrieval-support diagnostics.

The table utility also supports STALE LLM-judge experiment JSON with embedded
`raw_outputs`:

```bash
PYTHONPATH=src python -m adamem.eval --stale benchmarks/stale_mini.jsonl --baselines semantic_only --answer-provider mock --judge-provider mock --max-cases 1 --experiment-output /tmp/adamem_stale_mock_judge_experiment.json
PYTHONPATH=src python -m adamem.tables /tmp/adamem_stale_mock_judge_experiment.json --title "STALE Mock Judge Tables" --output /tmp/adamem_stale_mock_judge_tables.md
```

For STALE judge records, tables report overall accuracy plus `By dim` and
`By stale_type` breakdowns, with stale-leak rates. Mock providers validate the
workflow only; real claims require real answer and judge models.

Before turning any experiment into a paper claim, audit the experiment record:

```bash
PYTHONPATH=src python -m adamem.claims results/ama_public_20_full/ama_public_20.experiment.json
PYTHONPATH=src python -m adamem.claims /tmp/adamem_stale_mock_judge_experiment.json
```

The audit reports supported claims, blocked claims, warnings, provider settings,
ground-truth runtime-use notes, and the number of embedded or sidecar records.

To create a full report bundle from one experiment JSON:

```bash
PYTHONPATH=src python -m adamem.reporting results/ama_public_20_full/ama_public_20.experiment.json --output-dir /tmp/adamem_report_bundle_smoke --group-fields question_type --title "AMA Public 20 Bundle"
```

The bundle writes paper tables, claim-audit Markdown/JSON, and a manifest that
links all generated artifacts.

If a directory contains multiple `*experiment.json` files, the same command
runs in batch mode:

```bash
PYTHONPATH=src python -m adamem.reporting /tmp/adamem_ama_answer_generation_smoke --output-dir /tmp/adamem_report_batch_smoke --group-fields question_type
```

Batch mode writes one sub-bundle per experiment and a `batch_manifest.json`.
Bundles also include paired baseline comparisons. For retrieval records with
evidence labels, the comparison metric defaults to evidence support; otherwise
it compares exact `passed` support. Generation and STALE judge records compare
end-to-end correctness.

You can run paired comparison directly:

```bash
PYTHONPATH=src python -m adamem.compare results/ama_public_20_full/ama_public_20.experiment.json --group-fields question_type --output /tmp/ama_public_20.paired.md
```

The `trajectory_step_readout` baseline is a narrow trajectory-memory ablation:
when a query explicitly mentions `Step N` or a short step range, it authorizes
retrieval of the matching trajectory steps by metadata instead of relying only
on lexical similarity. On the first five public AMA-Bench samples, this
improves evidence support from `0/60` for `semantic_only` and `full` to
`60/60`, while answer-string support remains `0/60`; that result is retrieval
evidence only, not an answer-accuracy claim. The first answerability diagnostic
on the same 60 questions showed only a small keyword-recall increase after the
simple trajectory basis (`22.73%` to `24.81%`, matched queries `8/60` to
`11/60`). After adding deterministic rule/blocking/no-progress relations, the
structured basis reaches `32.25%` average keyword recall and `20/60` matched
queries. This is a useful API-free signal, but stronger summarization and
API-backed answer/judge scoring are still required.
On the first 20 public AMA-Bench episodes, the light pilot gives
`trajectory_step_readout` `239/239` evidence support versus `34/239` for
`semantic_only`, and basis keyword recall `24.34%` versus `15.68%`.
With bounded candidate pools and bounded soft-stale propagation, the same
20-episode pilot including `full` finishes in about 33 seconds locally:
`trajectory_step_readout` remains `239/239`, while `full` is `0/239` evidence
support and `19.07%` answer-keyword recall.
