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
delta_graph         3/4
full                4/4
```

This is not a substitute for LoCoMo, LongMemEval, or AMA-Bench, but it proves the local mechanisms are independently ablatable before paying for larger evaluations.

## JSONL Benchmark Adapter

Run a retrieval-support ablation over a thin JSONL format:

```bash
PYTHONPATH=src python -m adamem.eval --dataset benchmarks/tiny_memory_qa.jsonl
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
