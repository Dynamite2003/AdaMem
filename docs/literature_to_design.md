# Literature-to-Design Map

This document maps current agent-memory literature to AdaMem design hypotheses,
implementation tasks, and evaluation gates. It should be updated whenever a new
paper changes the baseline landscape or suggests a mechanism worth testing.

Last checked: 2026-05-30.

## Paper-Track Position

AdaMem should not be framed as another generic memory store. The stronger
paper angle is:

> Agent memory systems need an explicit current-state authority layer, because
> retrieval can surface updated evidence without making the downstream agent
> reject stale beliefs, stale premises, or stale downstream policies.

This framing is directly motivated by STALE and fits the current implementation
direction: typed state extraction, active/stale state adjudication, and
authorized readout before raw episodic memories.

## Evidence From Recent Literature

| Source | What it shows | AdaMem implication |
| --- | --- | --- |
| STALE, 2026: https://arxiv.org/abs/2605.06527 | Existing memory systems often retrieve updated evidence but still fail to act on it. STALE tests State Resolution, Premise Resistance, and Implicit Policy Adaptation. | AdaMem should measure current evidence recall separately from stale evidence exposure and answer correctness. The method should prioritize write-time state adjudication and read-time current-state authorization. |
| A-MEM, 2025: https://arxiv.org/abs/2502.12110 | Agentic memory can create structured notes, links, and evolving memory representations instead of fixed static records. | AdaMem can borrow the "memory evolution" idea, but should specialize it for state validity: evolution should update slot authority and invalidation status, not only enrich note text and links. |
| Zep / Graphiti, 2025: https://arxiv.org/abs/2501.13956 | Temporal knowledge graphs are useful for dynamic knowledge and cross-session synthesis. | AdaMem should treat relations and temporal validity as first-class signals, but stale-memory claims require explicit active-vs-replaced status for state slots. |
| MemGPT, 2023: https://arxiv.org/abs/2310.08560 | Hierarchical virtual context management can move information between memory tiers under context limits. | AdaMem can use context packing and authorized state summaries as a top memory tier, while raw episodes remain lower-tier evidence. |
| MemoryBank, 2023: https://arxiv.org/abs/2305.10250 | Long-term companion memory needs continuous updates, personality/user-state synthesis, and selective forgetting or reinforcement. | AdaMem should distinguish forgetting due to decay from invalidation due to new evidence. Stale-memory handling is not the same as low-importance decay. |
| Generative Agents, 2023: https://arxiv.org/abs/2304.03442 | Reflection can synthesize higher-level memories from raw observations and support planning behavior. | AdaMem state records are a constrained form of reflection. They should remain evidence-linked and ablatable so the causal path from observation to answer is inspectable. |
| Mem0, 2025: https://arxiv.org/abs/2504.19413 | Production memory emphasizes salient extraction, consolidation, graph memory, latency, and token cost. | AdaMem should keep a production-aware cost story, but paper claims should focus on stale validity and generality rather than deployment readiness. |
| LongMemEval, 2024/2025: https://arxiv.org/abs/2410.10813 | Long-term memory needs indexing, retrieval, and reading-stage design; knowledge updates and abstention are core abilities. | AdaMem should evaluate transfer beyond STALE on knowledge updates, temporal reasoning, and abstention-style queries. |
| AMA-Bench, 2026: https://arxiv.org/abs/2602.22769 | Agent memory in real trajectories needs causality and objective information, not only dialogue similarity retrieval. | AdaMem should test whether state slots plus dependency propagation transfer to agentic trajectories, especially task status and tool-output state. |
| LongMemEval-V2, 2026: https://arxiv.org/abs/2605.12493 | Web-agent memory needs static state recall, dynamic tracking, workflow knowledge, gotchas, and premise awareness. | AdaMem should add non-personal state slots: environment state, workflow constraints, recurring failure modes, and current runbook authority. |
| Survey, 2026: https://arxiv.org/abs/2603.07670 | Modern agent memory can be understood as write-manage-read with open challenges in continual consolidation, causal retrieval, contradiction handling, and trustworthy reflection. | AdaMem should present itself as a manage-layer contribution: state consolidation, contradiction/staleness adjudication, and auditable readout. |

## Candidate Contributions

### 1. Current-State Authority Layer

Hypothesis:

State-sensitive questions improve when the memory system exposes an explicit
active-state basis before raw retrieved episodes.

Current implementation:

- `use_state_memory`: writes derived typed state memories.
- `use_state_readout`: retrieves active state records for relevant queries.
- `StatePatch`: pluggable extractor output type.

Required next evidence:

- Larger STALE retrieval diagnostics.
- End-to-end STALE answer/judge pilot once API keys are available.
- Ablation: raw retrieval vs state extraction only vs state readout.

### 2. Evidence-Linked State Adjudication

Hypothesis:

State records should not be free-floating summaries. They need source evidence,
replacement history, and exposed traces to support causal validity.

Current implementation:

- Derived state memories store `source_id`, `state_slot`, `state_value`, and
  `memory_key`.
- Older state values for the same slot are superseded and marked stale.

Required next evidence:

- Trace export showing which observation created the active state.
- Tests for state replacement under paraphrases and unrelated updates.
- Error analysis separating extraction failure from adjudication failure.

### 3. Slot-Aware Query Routing

Hypothesis:

The system should only expose current-state records relevant to the query,
otherwise the state layer becomes prompt pollution.

Current implementation:

- Location and beverage-preference slots route to different query terms.
- Schedule availability routes to `schedule.availability`.
- Task status uses dynamic slots such as `task.checkout_migration.status` and
  wildcard readout matching through `task.*.status`.
- Health/dietary constraints use dynamic slots such as
  `health.peanut_allergy.status` and route through `health.*.status`.
- Resource status uses dynamic slots such as `resource.passport.status` and
  route through `resource.*.status`.
- Workflow/runbook rules use dynamic slots such as
  `workflow.checkout_deploys.rollback` and route through `workflow.*`.
- Runtime/tool status uses dynamic slots such as
  `runtime.staging_build_runner.status` and route through
  `runtime.*.status`.
- Tests verify beverage, schedule, task, health, resource, workflow, and
  runtime queries surface the intended active slot rather than unrelated state.
- JSONL benchmark summaries now report state-readout exposure, including
  unmarked-query state exposure, so prompt pollution can be measured rather
  than inferred.
- JSONL case records also report expected slots, retrieved state slots,
  unexpected slots, and slot-match status. This creates separate failure modes
  for missing state readout, slot mismatch, and unmarked state exposure.
- The first LongMemEval-S exposure run caught one false positive from the broad
  `local` location trigger. Routing was tightened so `local` requires a
  location-intent context. The follow-up balanced 60-case run kept aggregate
  retrieval-support accuracy unchanged at `40/60` and reduced state exposure
  for state-aware variants from the observed `1/60` to `0/60`.
  A subsequent slot-level run kept `0/60` state exposure and `0` slot mismatch
  on unmarked LongMemEval-S queries.
- A later full-file LongMemEval-S audit exposed additional false positives from
  broad status and location triggers, such as historical education completion,
  third-party residence, and `Facebook Live` event queries. Routing now
  requires task-status intent plus a task-like subject for non-`status`
  queries, and self-location wording for direct `live`/`based`/`located`
  triggers.

Required next evidence:

- Add relationship, user-role, environment-gotcha, and tool-output fact slots.
- Track state-readout exposure on every public transfer pilot and use failures
  to tune query routing or replace it with a documented intent classifier.
- Report state-readout missing, slot-mismatch, and unmarked-exposure rates
  alongside answer accuracy in any paper-facing table where state-aware
  mechanisms are compared.
- JSONL benchmark reports now include a `Paper Metrics` table with support
  accuracy, net delta versus the reference baseline, state-slot match, missing
  readout, slot mismatch, and unmarked state exposure. These metrics are also
  saved in experiment records for reproducibility.
- Add a no-answer/abstention diagnostic for irrelevant state.

### 4. Propagation-Aware Invalidity

Hypothesis:

Some state changes invalidate related slots, not just the exact same slot. For
example, a city move can invalidate local recommendations, commute constraints,
and timezone assumptions.

Current implementation:

- Soft stale propagation exists for raw memories, but state-slot dependency
  propagation now exists for derived state memories.
- `use_state_dependency_propagation` invalidates active dependent state slots
  when a changed slot supersedes a previous active state. Current topology:
  `location -> local.*`, `location -> commute.*`, `location -> schedule.local.*`,
  and `location -> timezone.*`.
- Propagation also marks the dependent state's source evidence stale so raw
  episode retrieval cannot bypass the state authority layer.

Required next evidence:

- Extend tests for indirect invalidation beyond the current local-state smoke
  case.
- Compare direct slot replacement against propagation on STALE T2-style cases.

### 5. Paper-Grade Failure Taxonomy

Hypothesis:

Aggregate accuracy is too weak for a CCF-A claim. The project needs case-level
evidence for why each mechanism helps or fails.

Current implementation:

- STALE retrieval diagnostics compute current recall, stale exposure, conflict
  coverage, current-before-stale, premise-old mention, and old-support
  adjudication.
- Case-level records can be exported as JSONL.
- Markdown failure reports aggregate records by failure mode, baseline, STALE
  dimension, stale type, and representative examples.

Required next evidence:

- Categorize failures as extraction, adjudication, routing, retrieval, readout,
  answer-model, or judge ambiguity.
- Include representative failures in experiment records.

## Baseline Requirements

Paper tables should eventually include at least:

- Raw similarity retrieval.
- Similarity plus recency and temporal filters.
- Graph or link-augmented retrieval.
- Memory evolution style baseline inspired by A-MEM.
- Temporal knowledge graph style baseline inspired by Zep.
- Production extraction/consolidation baseline inspired by Mem0.
- State-aware AdaMem ablations.
- Oracle/debug state extractor upper bound, clearly labeled as non-runtime.

Official implementations should be preferred when licensing and runtime cost are
acceptable. Faithful local approximations are allowed only when their behavior
and deviations are documented.

## Evaluation Gates

No claim should be made until the matching gate is satisfied.

| Claim | Minimum evidence |
| --- | --- |
| "Reduces stale evidence exposure" | Retrieval diagnostics on STALE with case-level traces and no runtime use of STALE labels. |
| "Improves current-state use" | End-to-end answer accuracy on STALE plus retrieval diagnostics showing current evidence or state readout availability. |
| "Handles premise resistance" | Per-dimension STALE results where Dimension 2 improves without simply suppressing all memory. |
| "Handles implicit policy adaptation" | Dimension 3 improvement plus evidence that relevant current state is routed even when the query does not mention old or new belief text. |
| "Generalizes" | At least one non-STALE benchmark or adapted task, preferably LongMemEval, AMA-Bench, or LongMemEval-V2-style dynamic state tracking. |
| "SOTA" | Reproduction of strong official baselines under documented answer model, judge model, prompts, split, top-k, and cost settings. |

`benchmarks/dynamic_state_transfer.jsonl` is only a local smoke fixture. It can
show that the code path transfers beyond STALE labels and location state, but
it cannot support a paper generalization claim by itself.

The fixture now covers seven state-sensitive cases: schedule availability, task
status, beverage preference, peanut-allergy clearance, passport renewal,
checkout rollback runbook updates, and staging build runner restoration. The
API-free result after adding workflow/runtime slots is `semantic_only` `0/7`
and state-aware readout/adjudication variants `7/7`. This is useful as a guard
against overfitting to location updates, but it remains synthetic local
evidence.

AdaMem now has a LongMemEval converter for the official cleaned schema, but the
converter is only an evaluation adapter. Official LongMemEval transfer requires
downloading the public data, documenting the split/file, running the converted
benchmark, and comparing against faithful baselines.

Converted LongMemEval or other JSONL retrieval pilots should be run with
`--experiment-output` so the record captures the command, commit, dataset path,
case limit, baseline configs, support pass/fail checks, retrieved text, and
per-query metadata/traces. Use `--baselines` for focused pilot runs before
scaling, and `--benchmark-report-output` for grouped error analysis by fields
such as `question_type`. These records are still retrieval diagnostics; they do
not replace answer-model and judge-model evaluation for paper claims.

The first balanced LongMemEval-S retrieval pilot was negative evidence for the
default `state_readout` baseline because it mixed two effects: default full
AdaMem scoring underperformed raw semantic retrieval on LongMemEval, and
derived state records could enter ordinary retrieval. The current mitigation is
a stricter authorization boundary plus semantic-only state-aware ablations:
`semantic_state_readout` and `semantic_state_propagation` keep derived state
records out of direct retrieval and add state readout only for routed
state-sensitive queries. In the balanced 60-case LongMemEval-S pilot, these
semantic-state variants matched raw semantic retrieval exactly in aggregate and
in paired query-level comparison: `40/60`, gained `0`, lost `0`, net `0`.
They also preserve the local dynamic-state gains. This is promising but still
only retrieval-level evidence; it needs larger public runs and end-to-end answer
scoring before any generality claim.

On STALE mini diagnostics, the semantic-state variants improved current recall
from `0%` to `100%`, but did not reduce stale exposure or adjudicate old
support. The full state-aware baselines did reduce stale exposure on that mini
fixture. The next method question is therefore not just readout authorization;
it is how to combine the clean semantic-state boundary with explicit stale
adjudication without damaging public benchmark retrieval.

The first query-scoped state-source adjudication variant answers that immediate
question at smoke scale. It marks raw evidence behind a replaced state value
and filters that evidence only when the query routes to the same state slot.
On the two-case STALE mini diagnostic run, `semantic_state_adjudication` kept
`100%` current recall and reduced stale exposure from `33.33%` to `0%`. On the
balanced 60-case LongMemEval-S retrieval-support pilot, it matched
`semantic_only` exactly (`40/60`, gained `0`, lost `0`, net `0`). This is
promising mechanism evidence, not an answer-accuracy claim; it needs larger
STALE diagnostics and API-backed answer/judge runs.

The public STALE paper states that the full benchmark contains 400 expert-
validated conflict scenarios and 1,200 queries across State Resolution,
Premise Resistance, and Implicit Policy Adaptation. AdaMem's CLI now supports
`--stale-types` and `--limit-per-stale-type` so the next non-mini run can be a
documented T1/T2-balanced retrieval diagnostic, followed by the same split in
LLM-judge mode once provider keys are available.

A first API-free A-MEM-style baseline has now been added as `a_mem_evolution`.
It creates deterministic note keywords, dynamic links, and write-time evolution
of raw episodic memories. This baseline helps separate AdaMem's proposed
current-state authority mechanism from a mainstream memory-evolution design.
The initial results are informative: on STALE mini, `a_mem_evolution` improved
current recall from `0%` to `33.33%` and reduced stale exposure from `33.33%`
to `16.67%`, but it did not adjudicate old support. On LongMemEval-S balanced
60, it scored `27/60`, while `semantic_only` and
`semantic_state_adjudication` both scored `40/60`. This suggests that generic
memory evolution/linking can help some stale retrieval cases but can also
increase retrieval noise on public long-memory transfer tasks; explicit state
authority remains the cleaner hypothesis.

A second mainstream approximation, `zep_temporal_kg`, now tests the
Zep/Graphiti-style hypothesis that temporal fact edges and invalidated old
relationships are enough. It writes deterministic temporal KG facts from the
same extractor, invalidates old edges, and reads out active KG facts for routed
state-sensitive queries. On STALE mini it reached `100%` current recall, but
stale exposure remained `33.33%`; `semantic_state_adjudication` reached `100%`
current recall and `0%` stale exposure. On LongMemEval-S balanced 60,
`zep_temporal_kg`, `semantic_only`, and `semantic_state_adjudication` all
scored `40/60` with net `0` pairwise change versus semantic-only. The useful
distinction is therefore not general retrieval support, but stale raw evidence
control: temporal KG readout exposes current state, while AdaMem's
state-source adjudication also blocks old raw support when the query is about
the same state slot.

A third mainstream approximation, `mem0_extraction`, now tests the production
memory hypothesis that compact extraction and update are enough. It keeps raw
observations only as audit sources, retrieves extracted compact facts, and
supersedes same-slot facts. On STALE mini this reached `100%` current recall
and `0%` stale exposure, but only `28.57%` old-support adjudication because the
old raw source is hidden rather than explicitly adjudicated. On LongMemEval-S
balanced 60 it scored only `1/60`, while `semantic_only`,
`zep_temporal_kg`, and `semantic_state_adjudication` scored `40/60`. This is
important negative evidence: compact extraction-only memory can look strong on
state-like stale cases while losing broad episodic evidence needed for public
long-memory transfer. AdaMem's current direction is therefore to retain raw
episodic evidence for generality, but govern stale-sensitive readout with
state authority and source adjudication.

The first query-annotated LongMemEval-S pilot exposed a measurement problem
before it exposed a method problem. With the initial `--infer-state-slots`
router, `18/60` balanced LongMemEval-S questions were marked state-sensitive,
but manual inspection showed many false positives from substring matching and
broad slot terms, such as `theater` matching `eat`, `accessories` matching
`access`, `service` matching runtime service, and historical `meet up` queries
matching schedule availability. After word-boundary matching and slot-specific
intent gates, the same balanced 60-case sample marks only `1/60` query as
state-sensitive, while aggregate support remains `40/60` and unmarked state
exposure remains `0.00%`. The lesson is paper-facing: state-readout metrics
need a precision-audited query set. The next method iteration should use a
reliable state-sensitive transfer subset before drawing conclusions about
observation-side semantic extraction coverage.

AdaMem now supports that audit path directly for LongMemEval conversions.
`--state-audit-output` writes query-state candidates proposed by the deterministic
router, while `--state-audit-input` imports only reviewed records explicitly
marked `is_state_sensitive: true`. The imported label is stored only on query
metadata with `state_slot_source=manual_state_audit`; observations remain free
of state labels. The audit schema now also separates state sensitivity from
state availability: a query can be marked `state_available=false` when it
requires current state but the haystack does not contain a reliable active
state. On the current balanced 60-case LongMemEval-S sample, the reviewed audit
file contains one accepted location-sensitive query, but it is marked
state-unavailable after manual inspection. The manual-audit conversion has zero
observation-level `state_slot` leakage, and the report no longer counts this
case as a state-readout-missing failure. This turns the transfer check into an
auditable subset workflow rather than an automatic-router claim.

The audit candidate file now also includes `state_evidence_candidates` produced
by the deterministic observation-side extractor. For the current accepted
LongMemEval-S location-sensitive query, that list is empty, which supports the
`state_available=false` decision without consulting answer labels. This is a
useful paper-discipline pattern: separate three questions that are often
collapsed in memory papers and demos: whether the query needs current state,
whether the memory contains a usable current-state record, and whether the
method retrieves or authorizes that record.

The audit summary path now scales that discipline from a hand-reviewed sample
to a whole converted file. `--state-audit-summary-output` reports candidate
counts and deterministic evidence coverage by state slot and LongMemEval
`question_type`. On the full 500-case LongMemEval-S cleaned file after the
router tightening, the audit produced only 14 query-state candidates and 0
state-evidence candidates; the balanced 60-case sample produced 1 candidate and
0 state-evidence candidates. This is useful negative evidence: LongMemEval-S is
currently a good broad retrieval no-regression check, but it is not a rich
state-available transfer benchmark for AdaMem's stale/current-state mechanism.
The next public transfer target should therefore be STALE full data or a
dynamic-state benchmark such as AMA-Bench, LongMemEval-V2, or STATE-Bench-style
tasks rather than spending answer-model budget on LongMemEval-S state-readout
metrics.

## Immediate Research Backlog

1. Convert the full STALE release when available locally and run a balanced
   diagnostic subset, for example `--stale-types T1 T2 --limit-per-stale-type
   10`, to validate whether query-scoped state-source adjudication scales
   beyond the mini fixture.
2. Build or select a precision-audited public state-sensitive transfer subset.
   LongMemEval-S should stay in the workflow as broad retrieval-transfer and
   no-regression evidence, but the full-file audit shows it is too sparse in
   state-available cases to carry AdaMem's main transfer claim.
3. Add relationship, user-role, environment-gotcha, and tool-output fact state
   slots.
4. Connect or reproduce official implementations for at least one mainstream
   memory system where licensing and dependencies permit; local approximations
   are useful but cannot substitute for final paper baselines.
5. Expand state-slot dependency propagation beyond the initial location
   topology and evaluate it on larger STALE T2-style cases.
6. Add API pilot output directories with raw prompts and raw model outputs.
7. Review official A-MEM, Zep/Graphiti, Mem0, and LongMemEval code/licenses.
8. Run larger STALE diagnostics when full converted data is available.
9. Draft method section once at least one larger diagnostic run supports the
   state-aware direction.
