from __future__ import annotations

from adamem import AdaMem, AdaMemConfig, LLMStateExtractor, StatePatch
from adamem.llm import MockLLMClient
from adamem.bench import default_ablation_configs
from adamem.state import query_relevant_state_slots


def test_observe_deduplicates_near_identical_memory() -> None:
    mem = AdaMem(config=AdaMemConfig(novelty_threshold=0.9))

    first = mem.observe("User prefers concise technical answers.", importance=0.4)
    second = mem.observe("User prefers concise technical answers.", importance=0.8)

    assert first.id == second.id
    assert second.importance == 0.8
    assert len(mem.store.all()) == 1


def test_observe_preserves_distinct_memory_keys_for_repeated_steps() -> None:
    mem = AdaMem(config=AdaMemConfig(novelty_threshold=0.9))

    first = mem.observe(
        "[step017.action] action: right",
        metadata={"memory_key": "step017.action", "benchmark": "ama", "trajectory_step": 17},
    )
    second = mem.observe(
        "[step018.action] action: right",
        metadata={"memory_key": "step018.action", "benchmark": "ama", "trajectory_step": 18},
    )

    assert first.id != second.id
    assert {item.metadata["memory_key"] for item in mem.store.all()} == {
        "step017.action",
        "step018.action",
    }


def test_candidate_pool_limit_bounds_mmr_without_losing_top_result() -> None:
    mem = AdaMem(config=AdaMemConfig(candidate_pool_limit=3, use_mmr=True))
    for index in range(12):
        mem.observe(
            f"Alpha retrieval candidate {index}.",
            importance=index / 12,
            metadata={"memory_key": f"candidate.{index}"},
        )

    results = mem.retrieve("Alpha retrieval candidate", top_k=5)

    assert len(results) == 5
    assert any("candidate 11" in result.item.content for result in results)


def test_soft_stale_candidate_limit_bounds_conflict_scan() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_soft_staleness=True,
            use_stale_propagation=False,
            soft_stale_candidate_limit=2,
            soft_stale_threshold=0.1,
        )
    )
    old_a = mem.observe("Deployment target is staging.", metadata={"memory_key": "old.a"})
    old_b = mem.observe("Deployment target is staging again.", metadata={"memory_key": "old.b"})
    old_c = mem.observe("Deployment target is staging once more.", metadata={"memory_key": "old.c"})
    for item in mem.store.all():
        item.staleness = 0.0
        item.stale_sources = []
        mem.store.upsert(item)

    mem.observe("Deployment target is production now.", metadata={"memory_key": "new"})

    assert mem.store.get(old_a.id).staleness == 0.0
    assert mem.store.get(old_b.id).staleness > 0.0
    assert mem.store.get(old_c.id).staleness > 0.0


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


def test_memory_evolution_links_and_updates_related_raw_notes() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=True,
            use_auto_links=True,
            use_mmr=False,
            use_supersession=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_memory_evolution=True,
            memory_evolution_threshold=0.15,
        )
    )

    old = mem.observe("Checkout migration is blocked by legal approval.")
    new = mem.observe("Legal approval arrived for checkout migration.")
    old_after = mem.store.get(old.id)

    assert old_after is not None
    assert new.id in old_after.links
    assert old.id in new.links
    assert "arrived" in old_after.metadata["evolved_keywords"]
    assert new.id in old_after.metadata["evolved_by"]

    results = mem.retrieve("What arrived for checkout migration?", top_k=3)

    assert any(result.item.id == old.id and "graph" in result.contributions for result in results)


def test_temporal_kg_readout_invalidates_old_edge_without_raw_source_adjudication() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=True,
            use_mmr=False,
            use_supersession=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_temporal_kg_memory=True,
            use_temporal_kg_readout=True,
        )
    )

    old = mem.observe(
        "[2026-01-01] user: I've been living in Seattle.",
        valid_from="2026-01-01T00:00:00+00:00",
    )
    new = mem.observe(
        "[2026-03-01] user: I relocated to Boston for a new job.",
        valid_from="2026-03-01T00:00:00+00:00",
    )

    kg_items = [item for item in mem.store.all() if item.kind == "kg_fact"]
    old_kg = next(item for item in kg_items if item.metadata["kg_object"] == "Seattle")
    new_kg = next(item for item in kg_items if item.metadata["kg_object"] == "Boston")
    old_source = mem.store.get(old.id)

    assert not old_kg.active
    assert old_kg.superseded_by == new_kg.id
    assert old_kg.valid_to == new.valid_from
    assert old_source is not None
    assert old_source.staleness == 0.0
    assert old_source.metadata.get("stale_state_slots") is None

    results = mem.retrieve("Since I'm in Seattle, recommend local resources.", top_k=4)

    assert results[0].item.kind == "kg_fact"
    assert "Boston" in results[0].item.content
    assert any(result.item.id == old.id for result in results)


def test_salient_memory_only_retrieves_extracted_fact_and_hides_raw_source() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_supersession=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_salient_memory=True,
            use_salient_memory_only=True,
            use_salient_memory_readout=True,
        )
    )

    old = mem.observe("[2026-01-01] user: I've been living in Seattle.")
    mem.observe("[2026-03-01] user: I relocated to Boston for a new job.")

    salient_items = [item for item in mem.store.all() if item.kind == "salient_fact"]
    old_salient = next(item for item in salient_items if item.metadata["salient_value"] == "Seattle")
    new_salient = next(item for item in salient_items if item.metadata["salient_value"] == "Boston")

    assert not old_salient.active
    assert old_salient.superseded_by == new_salient.id

    results = mem.retrieve("Since I'm in Seattle, recommend local resources.", top_k=4)

    assert results
    assert results[0].item.kind == "salient_fact"
    assert "Boston" in results[0].item.content
    assert all(result.item.id != old.id for result in results)


def test_state_readout_surfaces_current_location_for_implicit_local_query() -> None:
    mem = AdaMem(config=AdaMemConfig(use_state_memory=True, use_state_readout=True))

    mem.observe("[2026-01-01] user: I just moved into a place in Seattle.")
    mem.observe("[2026-03-01] user: I relocated to Boston for a new job.")

    state_items = [item for item in mem.store.all() if item.kind == "state"]
    active_state_values = [item.metadata.get("state_value") for item in state_items if item.active]
    inactive_state_values = [item.metadata.get("state_value") for item in state_items if not item.active]

    assert "Boston" in active_state_values
    assert "Seattle" in inactive_state_values

    results = mem.retrieve("Recommend a coffee shop near me.", top_k=3)

    assert results[0].item.kind == "state"
    assert "Boston" in results[0].item.content
    assert results[0].contributions["state_readout"] > 0.0


def test_state_readout_is_necessary_for_stale_state_resolution_query() -> None:
    without_readout = _state_isolation_memory(use_state_readout=False)
    with_readout = _state_isolation_memory(use_state_readout=True)
    query = "Is Seattle still the right city for me?"

    without_results = without_readout.retrieve(query, top_k=2)
    with_results = with_readout.retrieve(query, top_k=2)

    assert without_results
    assert "Seattle" in without_results[0].item.content
    assert with_results[0].item.kind == "state"
    assert "Boston" in with_results[0].item.content


def test_state_readout_resists_stale_query_premise() -> None:
    without_readout = _state_isolation_memory(use_state_readout=False)
    with_readout = _state_isolation_memory(use_state_readout=True)
    query = "Since I'm in Seattle, recommend a coffee shop near me."

    without_results = without_readout.retrieve(query, top_k=2)
    with_results = with_readout.retrieve(query, top_k=2)

    assert without_results
    assert "Seattle" in without_results[0].item.content
    assert with_results[0].item.kind == "state"
    assert "Boston" in with_results[0].item.content


def test_state_premise_correction_explicitly_flags_stale_premise() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_premise_correction=True,
        )
    )
    mem.observe("[2026-01-01] user: I just moved into a place in Seattle.")
    mem.observe("[2026-03-01] user: I relocated to Boston for a new job.")

    results = mem.retrieve("Since I'm in Seattle, recommend a coffee shop near me.", top_k=2)

    assert results[0].item.kind == "state_correction"
    assert results[0].relation == "state_premise_correction"
    assert results[0].item.metadata["ephemeral"] is True
    assert results[0].item.metadata["state_slot"] == "location"
    assert results[0].item.metadata["stale_value"] == "Seattle"
    assert results[0].item.metadata["current_value"] == "Boston"
    assert "Premise correction" in results[0].item.content
    assert "stale location 'Seattle'" in results[0].item.content
    assert "current value is 'Boston'" in results[0].item.content
    assert all(item.kind != "state_correction" for item in mem.store.all())


def test_state_premise_correction_does_not_fire_without_stale_value() -> None:
    mem = _state_isolation_memory(use_state_readout=True)
    mem.config.use_state_premise_correction = True

    results = mem.retrieve("Any good weekend spots nearby?", top_k=2)

    assert results[0].item.kind == "state"
    assert all(result.item.kind != "state_correction" for result in results)


def test_state_unknown_current_invalidates_old_location_without_replacement() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )
    old_source = mem.observe("[2026-01-01] user: I just moved into a place in Seattle.")
    mem.observe("[2026-02-01] user: I no longer live in Seattle.")

    states = [item for item in mem.store.all() if item.kind == "state"]
    active_state = next(item for item in states if item.active)
    old_state = next(item for item in states if not item.active)

    assert old_state.metadata["state_value"] == "Seattle"
    assert active_state.metadata["state_value"] == "unknown-current"
    assert active_state.metadata["state_status"] == "unknown_current"
    assert active_state.metadata["invalidated_state_value"] == "Seattle"
    assert old_source.metadata["stale_state_slots"] == ["location"]

    results = mem.retrieve("Since I'm in Seattle, recommend a coffee shop near me.", top_k=2)

    assert results[0].item.kind == "state_correction"
    assert results[0].item.metadata["stale_value"] == "Seattle"
    assert results[0].item.metadata["current_value"] == "unknown-current"
    assert "current value is unknown" in results[0].item.content
    assert "Invalidated prior value: Seattle" in results[0].item.content


def test_state_unknown_current_can_be_disabled_for_ablation() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_unknown_current=False,
        )
    )
    mem.observe("[2026-01-01] user: I just moved into a place in Seattle.")
    mem.observe("[2026-02-01] user: I no longer live in Seattle.")

    states = [item for item in mem.store.all() if item.kind == "state"]

    assert len(states) == 1
    assert states[0].active
    assert states[0].metadata["state_value"] == "Seattle"
    assert all(item.metadata.get("state_status") != "unknown_current" for item in states)


def test_state_unknown_current_handles_resource_workflow_and_runtime_slots() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )
    mem.observe("My passport is expired.")
    mem.observe("My passport is no longer expired.")
    mem.observe("For checkout deploys, the rollback rule is manual approval.")
    mem.observe("For checkout deploys, the rollback rule is no longer manual approval.")
    mem.observe("The staging build runner is offline.")
    mem.observe("The staging build runner is no longer offline.")

    active_unknown = {
        item.metadata["state_slot"]: item
        for item in mem.store.all()
        if item.kind == "state"
        and item.active
        and item.metadata.get("state_status") == "unknown_current"
    }

    assert active_unknown["resource.passport.status"].metadata["invalidated_state_value"] == "expired"
    assert active_unknown["workflow.checkout_deploys.rollback"].metadata["invalidated_state_value"] == "manual approval"
    assert active_unknown["runtime.staging_build_runner.status"].metadata["invalidated_state_value"] == "offline"

    passport = mem.retrieve("Is my passport expired?", top_k=1)
    workflow = mem.retrieve(
        "Using the old manual approval runbook, what rollback procedure applies to checkout deploys?",
        top_k=1,
    )
    runtime = mem.retrieve("Is the staging build runner offline?", top_k=1)

    assert passport[0].item.kind == "state_correction"
    assert passport[0].item.metadata["current_value"] == "unknown-current"
    assert workflow[0].item.kind == "state_correction"
    assert workflow[0].item.metadata["stale_value"] == "manual approval"
    assert runtime[0].item.kind == "state_correction"
    assert runtime[0].item.metadata["stale_value"] == "offline"


def test_state_readout_handles_employer_premise_correction() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )
    old_source = mem.observe("[2026-01-01] user: I work at Acme Labs.")
    mem.observe("[2026-03-01] user: My employer is Nova Health.")

    slots = query_relevant_state_slots("Do I still work at Acme Labs?")
    states = [item for item in mem.store.all() if item.kind == "state"]
    active_state = next(item for item in states if item.active)
    old_state = next(item for item in states if not item.active)
    results = mem.retrieve("Do I still work at Acme Labs for my benefits?", top_k=2)

    assert "organization.employer" in slots
    assert "organization.employer" not in query_relevant_state_slots("What work should I do next?")
    assert old_state.metadata["state_slot"] == "organization.employer"
    assert old_state.metadata["state_value"] == "Acme Labs"
    assert active_state.metadata["state_value"] == "Nova Health"
    assert old_source.metadata["stale_state_slots"] == ["organization.employer"]
    assert results[0].item.kind == "state_correction"
    assert results[0].item.metadata["state_slot"] == "organization.employer"
    assert results[0].item.metadata["stale_value"] == "Acme Labs"
    assert results[0].item.metadata["current_value"] == "Nova Health"


def test_state_unknown_current_handles_employer_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )
    mem.observe("[2026-01-01] user: I work at Acme Labs.")
    mem.observe("[2026-03-01] user: I no longer work at Acme Labs.")

    active_unknown = next(
        item for item in mem.store.all()
        if item.kind == "state" and item.active
    )
    results = mem.retrieve("Should I list Acme Labs as my employer?", top_k=2)

    assert active_unknown.metadata["state_slot"] == "organization.employer"
    assert active_unknown.metadata["state_status"] == "unknown_current"
    assert active_unknown.metadata["invalidated_state_value"] == "Acme Labs"
    assert results[0].item.kind == "state_correction"
    assert results[0].item.metadata["stale_value"] == "Acme Labs"
    assert results[0].item.metadata["current_value"] == "unknown-current"


def test_state_dependency_propagation_creates_unknown_current_dependent_state() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_dependency_propagation=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )
    mem.observe("[2026-01-01] user: I work at Acme Labs.")
    old_benefits_source = mem.observe("[2026-01-02] user: My benefits portal is Acme Benefits.")
    mem.observe("[2026-03-01] user: My employer is Nova Health.")

    benefits_states = [
        item
        for item in mem.store.all()
        if item.kind == "state"
        and item.metadata.get("state_slot") == "employment.benefits_portal"
    ]
    stale_benefits = next(item for item in benefits_states if not item.active)
    active_unknown = next(item for item in benefits_states if item.active)
    results = mem.retrieve("Should I use Acme Benefits for benefits enrollment?", top_k=2)

    assert "employment.benefits_portal" in query_relevant_state_slots(
        "Should I use Acme Benefits for benefits enrollment?"
    )
    assert stale_benefits.metadata["state_value"] == "Acme Benefits"
    assert active_unknown.metadata["state_status"] == "unknown_current"
    assert active_unknown.metadata["state_value"] == "unknown-current"
    assert active_unknown.metadata["invalidated_state_value"] == "Acme Benefits"
    assert active_unknown.metadata["dependency_invalidated_by_slot"] == "organization.employer"
    assert old_benefits_source.metadata["stale_state_slots"] == ["employment.benefits_portal"]
    assert results[0].item.kind == "state_correction"
    assert results[0].item.metadata["state_slot"] == "employment.benefits_portal"
    assert results[0].item.metadata["stale_value"] == "Acme Benefits"
    assert results[0].item.metadata["current_value"] == "unknown-current"


def test_state_readout_supports_implicit_policy_adaptation_query() -> None:
    without_readout = _state_isolation_memory(use_state_readout=False)
    with_readout = _state_isolation_memory(use_state_readout=True)
    query = "Any good weekend spots nearby?"

    without_results = without_readout.retrieve(query, top_k=2)
    with_results = with_readout.retrieve(query, top_k=2)

    assert all(result.item.kind != "state" for result in without_results)
    assert with_results[0].item.kind == "state"
    assert "Boston" in with_results[0].item.content


def test_state_readout_handles_non_location_preference_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: My favorite drink is coffee.")
    mem.observe("[2026-02-01] user: I prefer tea now.")
    mem.observe("[2026-03-01] user: I relocated to Boston for a new job.")

    results = mem.retrieve("What drink should I order for the user?", top_k=3)

    assert results[0].item.kind == "state"
    assert results[0].item.metadata["state_slot"] == "preference.beverage"
    assert "tea" in results[0].item.content
    assert "Boston" not in results[0].item.content


def test_state_unknown_current_handles_preference_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )

    mem.observe("[2026-01-01] user: My favorite drink is coffee.")
    mem.observe("[2026-02-01] user: My usual drink is no longer coffee.")

    states = [
        item
        for item in mem.store.all()
        if item.kind == "state" and item.metadata.get("state_slot") == "preference.beverage"
    ]
    active = [item for item in states if item.active]
    inactive = [item for item in states if not item.active]

    assert [item.metadata["state_value"] for item in active] == ["unknown-current"]
    assert active[0].metadata["state_status"] == "unknown_current"
    assert active[0].metadata["invalidated_state_value"] == "coffee"
    assert [item.metadata["state_value"] for item in inactive] == ["coffee"]

    results = mem.retrieve("Should I order coffee for the user?", top_k=1)

    assert results[0].item.kind == "state_correction"
    assert results[0].item.metadata["state_slot"] == "preference.beverage"
    assert results[0].item.metadata["stale_value"] == "coffee"
    assert results[0].item.metadata["current_value"] == "unknown-current"


def test_state_readout_handles_schedule_availability_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: I'm free on Friday afternoons.")
    mem.observe("[2026-01-02] user: My favorite drink is coffee.")

    results = mem.retrieve("What time can I meet with the user?", top_k=3)

    assert results[0].item.kind == "state"
    assert results[0].item.metadata["state_slot"] == "schedule.availability"
    assert "friday afternoons" in results[0].item.content.lower()
    assert "coffee" not in results[0].item.content.lower()


def test_state_readout_handles_dynamic_task_status_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: The checkout migration is blocked by missing approval.")
    mem.observe("[2026-01-03] user: Checkout migration is now resolved.")

    active_states = [
        item
        for item in mem.store.all()
        if item.kind == "state" and item.metadata.get("state_slot") == "task.checkout_migration.status"
    ]
    active_values = [item.metadata.get("state_value") for item in active_states if item.active]
    inactive_values = [item.metadata.get("state_value") for item in active_states if not item.active]

    assert active_values == ["resolved"]
    assert inactive_values == ["blocked"]

    results = mem.retrieve("What is the migration status?", top_k=3)

    assert results[0].item.kind == "state"
    assert results[0].item.metadata["state_slot"] == "task.checkout_migration.status"
    assert "resolved" in results[0].item.content
    assert "blocked" not in results[0].item.content


def test_state_readout_handles_health_constraint_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: I'm allergic to peanuts.")
    mem.observe("[2026-02-01] user: I can eat peanuts now.")

    active_states = [
        item
        for item in mem.store.all()
        if item.kind == "state" and item.metadata.get("state_slot") == "health.peanut_allergy.status"
    ]
    active_values = [item.metadata.get("state_value") for item in active_states if item.active]
    inactive_values = [item.metadata.get("state_value") for item in active_states if not item.active]

    assert active_values == ["peanut allergy cleared"]
    assert inactive_values == ["peanut allergy active"]

    results = mem.retrieve("Given the user's peanut allergy, what meal constraint applies?", top_k=3)

    assert results[0].item.kind == "state"
    assert results[0].item.metadata["state_slot"] == "health.peanut_allergy.status"
    assert "peanut allergy cleared" in results[0].item.content
    assert "allergic to peanuts" not in results[0].item.content


def test_state_readout_handles_resource_status_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: My passport is expired.")
    mem.observe("[2026-02-01] user: My passport is now renewed.")

    active_states = [
        item
        for item in mem.store.all()
        if item.kind == "state" and item.metadata.get("state_slot") == "resource.passport.status"
    ]
    active_values = [item.metadata.get("state_value") for item in active_states if item.active]
    inactive_values = [item.metadata.get("state_value") for item in active_states if not item.active]

    assert active_values == ["renewed"]
    assert inactive_values == ["expired"]

    results = mem.retrieve("Is my passport expired?", top_k=3)

    assert results[0].item.kind == "state"
    assert results[0].item.metadata["state_slot"] == "resource.passport.status"
    assert "renewed" in results[0].item.content
    assert "expired" not in results[0].item.content


def test_state_readout_handles_workflow_rule_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: For checkout deploys, the rollback rule is manual approval.")
    mem.observe("[2026-02-01] user: For checkout deploys, the rollback rule is automatic canary rollback.")

    active_states = [
        item
        for item in mem.store.all()
        if item.kind == "state" and item.metadata.get("state_slot") == "workflow.checkout_deploys.rollback"
    ]
    active_values = [item.metadata.get("state_value") for item in active_states if item.active]
    inactive_values = [item.metadata.get("state_value") for item in active_states if not item.active]

    assert active_values == ["automatic canary rollback"]
    assert inactive_values == ["manual approval"]

    results = mem.retrieve("What rollback procedure applies to checkout deploys?", top_k=3)

    assert results[0].item.kind == "state"
    assert results[0].item.metadata["state_slot"] == "workflow.checkout_deploys.rollback"
    assert "automatic canary rollback" in results[0].item.content
    assert "manual approval" not in results[0].item.content


def test_state_readout_handles_runtime_status_slot() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: The staging build runner is offline.")
    mem.observe("[2026-02-01] user: The staging build runner is now online.")

    active_states = [
        item
        for item in mem.store.all()
        if item.kind == "state" and item.metadata.get("state_slot") == "runtime.staging_build_runner.status"
    ]
    active_values = [item.metadata.get("state_value") for item in active_states if item.active]
    inactive_values = [item.metadata.get("state_value") for item in active_states if not item.active]

    assert active_values == ["online"]
    assert inactive_values == ["offline"]

    results = mem.retrieve("Is the staging build runner offline?", top_k=3)

    assert results[0].item.kind == "state"
    assert results[0].item.metadata["state_slot"] == "runtime.staging_build_runner.status"
    assert "online" in results[0].item.content
    assert "offline" not in results[0].item.content


def test_state_readout_handles_role_and_manager_slots() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: My role is frontend lead.")
    mem.observe("[2026-02-01] user: My current role is platform lead.")
    mem.observe("[2026-01-01] user: My manager is Sam.")
    mem.observe("[2026-02-01] user: My new manager is Priya.")

    active = {
        item.metadata["state_slot"]: item.metadata["state_value"]
        for item in mem.store.all()
        if item.kind == "state" and item.active
    }

    assert active["role.current"] == "platform lead"
    assert active["relationship.manager"] == "Priya"

    role_results = mem.retrieve("What role should guide my planning responsibilities?", top_k=2)
    manager_results = mem.retrieve("Who is my manager for approvals?", top_k=2)

    assert role_results[0].item.kind == "state"
    assert role_results[0].item.metadata["state_slot"] == "role.current"
    assert "platform lead" in role_results[0].item.content
    assert "frontend lead" not in role_results[0].item.content
    assert manager_results[0].item.kind == "state"
    assert manager_results[0].item.metadata["state_slot"] == "relationship.manager"
    assert "Priya" in manager_results[0].item.content
    assert "Sam" not in manager_results[0].item.content


def test_state_premise_correction_handles_stale_role_and_manager() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )

    mem.observe("[2026-01-01] user: My role is frontend lead.")
    mem.observe("[2026-02-01] user: My current role is platform lead.")
    mem.observe("[2026-01-01] user: My manager is Sam.")
    mem.observe("[2026-02-01] user: My new manager is Priya.")

    role_results = mem.retrieve(
        "As the frontend lead, which planning responsibilities apply?",
        top_k=1,
    )
    manager_results = mem.retrieve("Should I send approval to my manager Sam?", top_k=1)

    assert role_results[0].item.kind == "state_correction"
    assert role_results[0].item.metadata["state_slot"] == "role.current"
    assert role_results[0].item.metadata["stale_value"] == "frontend lead"
    assert role_results[0].item.metadata["current_value"] == "platform lead"
    assert manager_results[0].item.kind == "state_correction"
    assert manager_results[0].item.metadata["state_slot"] == "relationship.manager"
    assert manager_results[0].item.metadata["stale_value"] == "Sam"
    assert manager_results[0].item.metadata["current_value"] == "Priya"


def test_state_unknown_current_handles_role_and_manager_slots() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
            use_state_premise_correction=True,
        )
    )

    mem.observe("[2026-01-01] user: My role is frontend lead.")
    mem.observe("[2026-02-01] user: My role is no longer frontend lead.")
    mem.observe("[2026-01-01] user: My manager is Sam.")
    mem.observe("[2026-02-01] user: My manager is no longer Sam.")

    active_unknown = {
        item.metadata["state_slot"]: item
        for item in mem.store.all()
        if item.kind == "state"
        and item.active
        and item.metadata.get("state_status") == "unknown_current"
    }

    assert active_unknown["role.current"].metadata["state_value"] == "unknown-current"
    assert active_unknown["role.current"].metadata["invalidated_state_value"] == "frontend lead"
    assert active_unknown["relationship.manager"].metadata["state_value"] == "unknown-current"
    assert active_unknown["relationship.manager"].metadata["invalidated_state_value"] == "Sam"

    role_results = mem.retrieve(
        "As the frontend lead, which planning responsibilities apply?",
        top_k=1,
    )
    manager_results = mem.retrieve("Should I send approval to my manager Sam?", top_k=1)

    assert role_results[0].item.kind == "state_correction"
    assert role_results[0].item.metadata["state_slot"] == "role.current"
    assert role_results[0].item.metadata["stale_value"] == "frontend lead"
    assert role_results[0].item.metadata["current_value"] == "unknown-current"
    assert manager_results[0].item.kind == "state_correction"
    assert manager_results[0].item.metadata["state_slot"] == "relationship.manager"
    assert manager_results[0].item.metadata["stale_value"] == "Sam"
    assert manager_results[0].item.metadata["current_value"] == "unknown-current"


def test_state_records_do_not_pollute_generic_retrieval_without_authorization() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: I relocated to Boston for a new job.")
    mem.observe("[2026-01-02] user: The Boston architecture article covered brick facades.")

    results = mem.retrieve("What did the Boston architecture article cover?", top_k=3)

    assert results
    assert results[0].item.kind != "state"
    assert "brick facades" in results[0].item.content


def test_task_state_readout_requires_status_intent_not_project_count() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: The checkout migration status is completed.")
    mem.observe("[2026-01-02] user: I led three migration projects this year.")

    count_results = mem.retrieve("How many migration projects have I led?", top_k=2)
    status_results = mem.retrieve("What is the migration status?", top_k=2)

    assert count_results
    assert count_results[0].item.kind != "state"
    assert "three migration projects" in count_results[0].item.content
    assert status_results[0].item.kind == "state"


def test_runtime_state_readout_requires_runtime_intent_not_generic_status_report() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: The staging build runner is online.")
    mem.observe("[2026-01-02] user: The status report has three sections.")

    results = mem.retrieve("How many sections were in the status report?", top_k=2)

    assert results
    assert results[0].item.kind != "state"
    assert "three sections" in results[0].item.content


def test_location_state_readout_does_not_trigger_on_local_event_history() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=True,
        )
    )

    mem.observe("[2026-01-01] user: I moved to Boston.")
    mem.observe("[2026-01-02] user: I volunteered at the local animal shelter dinner in February.")

    results = mem.retrieve("When did I volunteer at the local animal shelter dinner?", top_k=2)

    assert results
    assert results[0].item.kind != "state"
    assert "february" in results[0].item.content.lower()


def test_query_state_router_uses_word_boundaries_and_intent_gates() -> None:
    assert query_relevant_state_slots("What is the migration status?") == ["task.*.status"]
    assert query_relevant_state_slots("What time can I meet with the user?") == ["schedule.availability"]
    assert query_relevant_state_slots("Is the staging build runner offline?") == ["runtime.*.status"]
    assert query_relevant_state_slots("What role should guide my responsibilities?") == ["role.current"]
    assert query_relevant_state_slots("Who is my manager for approvals?") == ["relationship.manager"]
    assert query_relevant_state_slots("Given the user's peanut allergy, what meal constraint applies?") == [
        "health.*.status"
    ]

    assert query_relevant_state_slots("What play did I attend at the local community theater?") == []
    assert query_relevant_state_slots("Can you suggest accessories for my current photography setup?") == []
    assert query_relevant_state_slots("How many days ago did I attend the Maundy Thursday service?") == []
    assert query_relevant_state_slots("What time did I go to bed before the appointment?") == []
    assert query_relevant_state_slots("Which event happened the day I ordered a customized phone case?") == []
    assert query_relevant_state_slots("Where did I redeem a $5 coupon on coffee creamer?") == []
    assert query_relevant_state_slots("How many Korean restaurants have I tried in my city?") == []
    assert query_relevant_state_slots("What was the hostel near the Red Light District you recommended last time?") == []
    assert query_relevant_state_slots("Can I access all seasons of old shows on Netflix?") == []
    assert query_relevant_state_slots("What old show only had the last season available on Netflix?") == []
    assert query_relevant_state_slots("What was the 7th work from home job in the list?") == []
    assert query_relevant_state_slots("Where does my sister Emily live?") == []
    assert query_relevant_state_slots("What is the total number of online courses I've completed?") == []
    assert query_relevant_state_slots("What is the total number of comments on my Facebook Live session?") == []
    assert query_relevant_state_slots("Who managed the 2024 project retrospective?") == []


def test_state_authorization_can_be_disabled_for_ablation() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_graph=False,
            use_mmr=False,
            use_state_memory=True,
            use_state_readout=False,
            use_state_readout_authorization=False,
        )
    )

    mem.observe("[2026-01-01] user: I relocated to Boston for a new job.")

    results = mem.retrieve("Boston", top_k=2)

    assert any(result.item.kind == "state" for result in results)


def test_state_source_adjudication_filters_replaced_raw_evidence_for_state_queries() -> None:
    base_config = dict(
        use_temporal=False,
        use_importance=False,
        use_recency=False,
        use_confidence=False,
        use_feedback=False,
        use_graph=False,
        use_mmr=False,
        use_supersession=False,
        use_soft_staleness=False,
        use_stale_propagation=False,
        use_adjudication_filter=False,
        use_state_memory=True,
        use_state_readout=True,
    )
    without_adjudication = AdaMem(config=AdaMemConfig(**base_config))
    with_adjudication = AdaMem(
        config=AdaMemConfig(
            **{
                **base_config,
                "use_state_source_adjudication": True,
            }
        )
    )

    for mem in (without_adjudication, with_adjudication):
        mem.observe("[2026-01-01] user: I've been living in Seattle.")
        mem.observe("[2026-03-01] user: I relocated to Boston for a new job.")

    old_source = with_adjudication.store.all()[0]
    new_source = with_adjudication.store.all()[2]

    assert old_source.staleness == 1.0
    assert new_source.id in old_source.stale_sources
    assert old_source.metadata["stale_state_slots"] == ["location"]

    query = "Since I'm in Seattle, recommend local resources."
    without_ids = [result.item.id for result in without_adjudication.retrieve(query, top_k=5)]
    with_results = with_adjudication.retrieve(query, top_k=5)
    with_ids = [result.item.id for result in with_results]

    assert without_adjudication.store.all()[0].id in without_ids
    assert old_source.id not in with_ids
    assert with_results[0].item.kind == "state"
    assert "Boston" in with_results[0].item.content


def test_state_source_adjudication_keeps_historical_raw_evidence_outside_state_queries() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_supersession=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=True,
            use_state_source_adjudication=True,
        )
    )

    old = mem.observe("[2026-01-01] user: I've been living in Seattle.")
    mem.observe("[2026-03-01] user: I relocated to Boston for a new job.")

    results = mem.retrieve("What did I say about Seattle?", top_k=3)

    assert old.id in [result.item.id for result in results]


def test_custom_state_extractor_can_inject_domain_state() -> None:
    def extractor(content: str, metadata: dict[str, object] | None) -> list[StatePatch]:
        if "workspace moved to" not in content:
            return []
        value = content.rsplit("workspace moved to", 1)[1].strip(" .")
        return [StatePatch(slot="workspace", value=value, evidence=content)]

    mem = AdaMem(config=AdaMemConfig(use_state_memory=True), state_extractor=extractor)

    mem.observe("The workspace moved to /tmp/adamem.")

    state_items = [item for item in mem.store.all() if item.kind == "state"]
    assert len(state_items) == 1
    assert state_items[0].metadata["state_slot"] == "workspace"
    assert state_items[0].metadata["state_value"] == "/tmp/adamem"


def test_llm_state_extractor_parses_client_json_without_using_labels() -> None:
    client = MockLLMClient(
        """
        ```json
        {"patches":[{"slot":"runtime.staging_runner.status","value":"online","status":"active"}]}
        ```
        """
    )
    extractor = LLMStateExtractor(client)

    patches = extractor("The staging runner came back online.", metadata={})

    assert len(patches) == 1
    assert patches[0] == StatePatch(
        slot="runtime.staging_runner.status",
        value="online",
        evidence="The staging runner came back online.",
    )
    assert "durable current-state updates" in client.calls[0]["system"]
    assert "The staging runner came back online." in client.calls[0]["prompt"]


def test_metadata_mock_llm_state_extractor_is_configurable_for_ci() -> None:
    mem = AdaMem(
        config=AdaMemConfig(
            use_state_memory=True,
            use_state_readout=True,
            state_extractor_name="metadata_mock_llm",
        )
    )

    mem.observe(
        "Runtime update recorded in structured telemetry.",
        metadata={
            "mock_state_patches": [
                {"slot": "runtime.staging_runner.status", "value": "online"},
            ]
        },
    )

    state_items = [item for item in mem.store.all() if item.kind == "state"]
    results = mem.retrieve("Is the staging runner online?", top_k=2)

    assert len(state_items) == 1
    assert state_items[0].metadata["state_slot"] == "runtime.staging_runner.status"
    assert state_items[0].metadata["state_value"] == "online"
    assert any(result.item.kind == "state" and "online" in result.item.content for result in results)


def test_state_dependency_propagation_invalidates_local_state_on_location_change() -> None:
    def extractor(content: str, metadata: dict[str, object] | None) -> list[StatePatch]:
        patches: list[StatePatch] = []
        if "moved to Seattle" in content:
            patches.append(StatePatch(slot="location", value="Seattle", evidence=content))
        if "relocated to Boston" in content:
            patches.append(StatePatch(slot="location", value="Boston", evidence=content))
        if "local gym is Rain City Fitness" in content:
            patches.append(StatePatch(slot="local.gym", value="Rain City Fitness", evidence=content))
        return patches

    without_propagation = AdaMem(
        config=AdaMemConfig(use_state_memory=True, use_state_readout=True),
        state_extractor=extractor,
    )
    with_propagation = AdaMem(
        config=AdaMemConfig(
            use_state_memory=True,
            use_state_readout=True,
            use_state_dependency_propagation=True,
        ),
        state_extractor=extractor,
    )
    for mem in (without_propagation, with_propagation):
        mem.observe("I moved to Seattle.")
        mem.observe("My local gym is Rain City Fitness.")
        mem.observe("I relocated to Boston.")

    stale_local_states = [
        item
        for item in with_propagation.store.all()
        if item.metadata.get("state_slot") == "local.gym" and not item.active
    ]
    active_without = [
        item
        for item in without_propagation.store.all()
        if item.metadata.get("state_slot") == "local.gym" and item.active
    ]

    assert active_without
    assert stale_local_states
    assert stale_local_states[0].staleness == 1.0

    without_results = without_propagation.retrieve("Which local gym near me should I use?", top_k=3)
    with_results = with_propagation.retrieve("Which local gym near me should I use?", top_k=3)

    assert any("Rain City Fitness" in result.item.content for result in without_results)
    assert all(
        not (
            result.item.metadata.get("state_slot") == "local.gym"
            and result.item.metadata.get("state_value") == "Rain City Fitness"
        )
        for result in with_results
    )
    assert any(
        result.item.metadata.get("state_slot") == "local.gym"
        and result.item.metadata.get("state_status") == "unknown_current"
        for result in with_results
    )
    assert any("Boston" in result.item.content for result in with_results)


def _state_isolation_memory(*, use_state_readout: bool) -> AdaMem:
    mem = AdaMem(
        config=AdaMemConfig(
            use_temporal=False,
            use_importance=False,
            use_recency=False,
            use_confidence=False,
            use_feedback=False,
            use_graph=False,
            use_mmr=False,
            use_soft_staleness=False,
            use_stale_propagation=False,
            use_adjudication_filter=False,
            use_state_memory=True,
            use_state_readout=use_state_readout,
        )
    )
    mem.observe("[2026-01-01] user: I just moved into a place in Seattle.")
    mem.observe("[2026-02-01] user: My favorite coffee order is a cappuccino.")
    mem.observe("[2026-03-01] user: I relocated to Boston for a new job.")
    return mem
