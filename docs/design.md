# AdaMem Design Notes

## SOTA Signals

The recent agent-memory line points to a clear pattern:

- **Generative Agents (2023):** memory quality improves when observation, reflection, and planning are separate, ablatable components. https://arxiv.org/abs/2304.03442
- **Reflexion (2023):** agents can learn across attempts by storing verbal feedback in episodic memory, without weight updates. https://arxiv.org/abs/2303.11366
- **MemGPT (2023/2024):** context management should be explicit, with movement between fast context and long-term storage. https://arxiv.org/abs/2310.08560
- **LoCoMo (2024):** long-term dialogue stresses temporal and causal dynamics; long context and vanilla RAG still lag human performance. https://arxiv.org/abs/2402.17753
- **Zep / Graphiti (2025):** temporal knowledge graphs improve dynamic cross-session memory and latency versus static RAG. https://arxiv.org/abs/2501.13956
- **A-MEM (2025):** Zettelkasten-style note construction, dynamic linking, and memory evolution beat rigid memory workflows. https://arxiv.org/abs/2502.12110
- **Mem0 (2025):** production memory needs extraction, consolidation, retrieval, and cost/latency discipline. https://arxiv.org/abs/2504.19413
- **AMA-Bench / AMA-Agent (2026):** real agent memory is a state-action-observation-tool trajectory; systems fail when they miss causal/objective information and rely only on lossy similarity retrieval. https://arxiv.org/abs/2602.22769

## Hypothesis

AdaMem should beat a similarity-only memory baseline when tasks require:

- changed facts or preferences,
- root-cause/action/outcome recall,
- cross-session procedural recall,
- retrieving low-frequency but high-importance instructions.

The bet is not "bigger memory harness"; it is a small write-manage-read loop with explicit deltas, causal links, and transparent scoring.

## Mechanism

1. **Write:** `observe()` accepts any agent event. It embeds content, deduplicates near-identical entries, supersedes old facts with the same `memory_key`, and links related/causal entries.
2. **Manage:** old active facts are not deleted; they become inactive through `superseded_by`, preserving provenance for audits.
3. **Read:** `retrieve()` combines semantic similarity, temporal validity, importance, recency, confidence, feedback, and graph expansion. Each result carries score contributions. Graph expansion defaults to explicit cause/update edges; automatic similarity links are available via `use_auto_links` for ablation.
4. **Pack:** `context()` builds a compact prompt block using MMR to avoid near-duplicate context.

## Plug-In Surface

- Replace `MemoryStore` with SQLite/Postgres/vector DB/Graphiti.
- Replace `embedder` with OpenAI, local sentence-transformers, BM25, or domain embeddings.
- Add an LLM extractor upstream that turns raw turns/tool logs into structured `observe()` calls.
- Keep the surrounding agent loop untouched.

## Ablations

Run the same benchmark with:

1. Full-context baseline.
2. Similarity-only AdaMem: `use_graph=False`, `use_temporal=False`, `use_importance=False`, `use_recency=False`, `use_feedback=False`.
3. Similarity + importance.
4. Similarity + temporal validity.
5. Similarity + graph expansion.
6. Similarity + graph + supersession.
7. Full AdaMem.

Track answer accuracy/F1, retrieval recall@k, hallucinated stale fact rate, p95 latency, prompt tokens, and write amplification.

The in-repo synthetic runner is intentionally small and deterministic:

```bash
PYTHONPATH=src python -m adamem.eval
```

It isolates four failure modes:

- stale current facts need delta supersession,
- resolved incidents need causal graph expansion,
- changed codes need temporal validity,
- low-frequency safety rules need importance-aware retrieval.

On the initial suite, semantic-only retrieval scores 1/4 and full AdaMem scores 4/4. This is a mechanism sanity check, not a SOTA claim.

## Initial Benchmarks

- LoCoMo for long-term conversational QA.
- LongMemEval for cross-session assistant memory and abstention.
- AMA-Bench for state/action/tool trajectories and causality.
- The in-repo synthetic benchmark for deterministic CI is now present in `adamem.eval`.

## Public Benchmark Adapter

AdaMem now includes a thin JSONL adapter so public datasets can be converted without introducing a harness dependency:

```json
{
  "id": "episode-id",
  "observations": [
    {
      "label": "cause",
      "content": "TX91 token was missing from the build runner.",
      "importance": 0.9,
      "valid_from": null,
      "valid_to": null,
      "cause_labels": [],
      "metadata": {"memory_key": "build.cause"}
    }
  ],
  "queries": [
    {
      "id": "q1",
      "query": "Which token fixed the build?",
      "expected_substrings": ["TX91"],
      "forbidden_substrings": [],
      "top_k": 2,
      "now": "2026-05-28T00:00:00+00:00"
    }
  ]
}
```

Mapping guidance:

- **LoCoMo:** convert each dialogue/session event into an observation; use annotated evidence ids as retrieval-support `expected_substrings`; use timestamps as `valid_from` where available.
- **LongMemEval:** convert each past interaction/message into observations; map abstention questions with empty expected evidence plus forbidden stale evidence.
- **AMA-Bench:** convert state/action/observation/tool-output trajectory steps into observations; encode action-result links with `cause_labels`.

The adapter evaluates retrieval support, not final generated answers. This keeps AdaMem minimal while making retrieval-recall, stale-fact rate, and ablation curves measurable before plugging in an LLM judge.

Structured attributes (`metadata.tags`, `metadata.keywords`, `metadata.subject`, `metadata.predicate`, and `metadata.memory_key`) are indexed with the raw content. This mirrors A-MEM-style note attributes while keeping the API a single `observe()` call.

### LoCoMo

The built-in converter supports the official `snap-research/locomo` `data/locomo10.json` schema:

```bash
PYTHONPATH=src python -m adamem.convert locomo data/locomo10.json benchmarks/locomo10.adamem.jsonl
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/locomo10.adamem.jsonl
```

Each dialogue turn becomes one observation whose content includes `dia_id`, session date, speaker, and text. By default, each QA expects retrieval of the annotated evidence ids (`D1:3` style), which makes this a retrieval-support benchmark. `--expected answer` switches to answer-string matching, and `--expected both` requires both evidence and answer strings.

Smoke result on the first official LoCoMo sample (`--limit 1`, 152 non-adversarial QA after category-5 filtering, `--top-k 8`):

| ablation | retrieval support |
| --- | ---: |
| semantic_only | 50/152 |
| semantic_graph | 50/152 |
| delta_graph | 50/152 |
| full | 52/152 |

This is not a paper-level result, but it is a useful guardrail: the default full configuration must not regress vanilla retrieval on static long-dialogue evidence search while still improving controlled stale/temporal/causal tasks.
