from __future__ import annotations

from adamem import AdaMem, AdaMemConfig
from adamem.bench import default_ablation_configs


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


def test_soft_staleness_marks_similar_prior_without_memory_key() -> None:
    """Mechanism A: similar but-different priors get a graded staleness even
    without an explicit `memory_key`."""
    cfg = default_ablation_configs()["delta_soft"]
    mem = AdaMem(config=cfg)

    old = mem.observe("Office door code is 1234.")
    unrelated = mem.observe("Backups run every night at 2am.")
    new = mem.observe("Office door code is 9876.")

    old_after = mem.store.get(old.id)
    unrelated_after = mem.store.get(unrelated.id)

    assert old_after is not None and new.id != old.id
    # The conflicting prior accumulates staleness; the unrelated prior does not.
    assert old_after.staleness > 0.0
    assert new.id in old_after.stale_sources
    assert unrelated_after is not None and unrelated_after.staleness == 0.0


def test_soft_staleness_demotes_stale_prior_in_retrieval() -> None:
    """`delta_soft` should rank the NEW belief above the OLD one even though
    no `memory_key` collision was declared by the user."""
    cfg = default_ablation_configs()["delta_soft"]
    mem = AdaMem(config=cfg)

    old = mem.observe("Office door code is 1234.")
    new = mem.observe("Office door code is 9876.")

    results = mem.retrieve("What is the office door code?", top_k=2)
    top_ids = [r.item.id for r in results]

    assert top_ids[0] == new.id
    # The old belief is allowed in but penalized: staleness contribution < 0.
    old_result = next((r for r in results if r.item.id == old.id), None)
    if old_result is not None:
        assert old_result.contributions.get("staleness", 0.0) < 0.0


def test_stale_propagation_pulls_in_co_occurring_facts() -> None:
    """Mechanism B: items in the same `session_*` as a directly-stale item
    inherit some staleness; unrelated items do not."""
    cfg = default_ablation_configs()["delta_propagation"]
    mem = AdaMem(config=cfg)

    old = mem.observe(
        "Office door code is 1234.",
        metadata={"tags": ["session_0"]},
    )
    cooccurring = mem.observe(
        "Building A guard rotation starts 6am.",
        metadata={"tags": ["session_0"]},
    )
    isolated = mem.observe(
        "Coffee machine refills happen on Fridays.",
        metadata={"tags": ["session_5"]},
    )
    new = mem.observe(
        "Office door code is 9876.",
        metadata={"tags": ["session_3"]},
    )

    old_after = mem.store.get(old.id)
    co_after = mem.store.get(cooccurring.id)
    iso_after = mem.store.get(isolated.id)

    assert old_after is not None and old_after.staleness > 0.0
    # Same-session item should be pulled along, but at a smaller magnitude.
    assert co_after is not None
    assert 0.0 < co_after.staleness <= old_after.staleness
    assert new.id in co_after.stale_sources
    # Items from a different session are untouched.
    assert iso_after is not None and iso_after.staleness == 0.0


def test_adjudication_filter_drops_high_staleness_candidate() -> None:
    """Mechanism C: when staleness clears the drop threshold, the item is
    excluded from the returned context entirely."""
    cfg = default_ablation_configs()["delta_full"]
    mem = AdaMem(config=cfg)

    old = mem.observe("Office door code is 1234.")
    # Two confirming updates from distinct angles push staleness past the
    # drop threshold (default 0.6) by accumulating from different new items.
    new1 = mem.observe("Office door code is 9876.")
    new2 = mem.observe("Office door code: changed from 1234 to 9876.")

    results = mem.retrieve("What is the office door code?", top_k=4)
    ids = [r.item.id for r in results]

    # A current belief is surfaced; the stale prior is dropped.
    assert new1.id in ids or new2.id in ids
    assert old.id not in ids
    # Sanity-check: same scenario without mechanism C still returns the stale.
    cfg_no_c = default_ablation_configs()["delta_propagation"]
    mem2 = AdaMem(config=cfg_no_c)
    o2 = mem2.observe("Office door code is 1234.")
    mem2.observe("Office door code is 9876.")
    mem2.observe("Office door code: changed from 1234 to 9876.")
    ids2 = [r.item.id for r in mem2.retrieve("What is the office door code?", top_k=4)]
    assert o2.id in ids2

