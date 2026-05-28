from __future__ import annotations

from adamem import AdaMem, AdaMemConfig


def test_observe_deduplicates_near_identical_memory() -> None:
    mem = AdaMem(config=AdaMemConfig(novelty_threshold=0.9))

    first = mem.observe("User prefers concise technical answers.", importance=0.4)
    second = mem.observe("User prefers concise technical answers.", importance=0.8)

    assert first.id == second.id
    assert second.importance == 0.8
    assert len(mem.store.all()) == 1


def test_supersession_hides_stale_fact_by_default() -> None:
    mem = AdaMem()

    old = mem.observe(
        "Deployment target: staging",
        metadata={"memory_key": "deploy.target"},
    )
    new = mem.observe(
        "Deployment target: production",
        metadata={"memory_key": "deploy.target"},
    )

    results = mem.retrieve("Where should I deploy?", top_k=5)
    ids = [result.item.id for result in results]

    assert old.superseded_by == new.id
    assert new.id in ids
    assert old.id not in ids


def test_graph_expansion_recovers_causal_parent() -> None:
    mem = AdaMem(config=AdaMemConfig(link_threshold=1.0))

    cause = mem.observe(
        "Checkout failed because STRIPE_SECRET was missing in production.",
        metadata={"memory_key": "checkout.root_cause"},
        importance=0.9,
    )
    outcome = mem.observe(
        "After setting STRIPE_SECRET, checkout succeeded.",
        cause_ids=[cause.id],
        metadata={"memory_key": "checkout.fix"},
        importance=0.9,
    )

    results = mem.retrieve("What fixed checkout?", top_k=4)
    ids = [result.item.id for result in results]

    assert outcome.id in ids
    assert cause.id in ids


def test_retrieval_exposes_ablation_contributions() -> None:
    mem = AdaMem()
    mem.observe("The API timeout budget is 15 seconds.", importance=1.0)

    full = mem.retrieve("timeout budget", top_k=1)[0]
    semantic_only = mem.ablation(
        use_graph=False,
        use_temporal=False,
        use_importance=False,
        use_recency=False,
        use_feedback=False,
    ).retrieve("timeout budget", top_k=1)[0]

    assert "importance" in full.contributions
    assert "importance" not in semantic_only.contributions
    assert full.score > semantic_only.score


def test_context_respects_character_budget() -> None:
    mem = AdaMem()
    mem.observe("Alpha memory about retry policy.", importance=0.9)
    mem.observe("Beta memory about retry policy and fallback.", importance=0.9)

    context = mem.context("retry policy", max_chars=80)

    assert len(context) <= 80
