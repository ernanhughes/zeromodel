from __future__ import annotations

from decimal import Decimal

from examples.fx_triangular_arbitrage import (
    compile_opportunity_surface,
    demo_payload,
    deterministic_snapshots,
    evaluate_cycle,
    mutate_forward_opportunity,
    replay,
)


def test_forward_formula_uses_bid_ask_prices_not_midpoints() -> None:
    snapshots = deterministic_snapshots()
    contract = compile_opportunity_surface(snapshots).contract

    opportunity = evaluate_cycle(snapshots[1], "forward", contract)

    assert opportunity.opportunity_id == "fx-snapshot-002|forward"
    assert opportunity.decision == "EXECUTE_FORWARD"
    assert Decimal("1.82") < opportunity.net_edge_bps < Decimal("1.84")
    assert opportunity.arithmetic == "ending_eur = eurusd_bid / gbpusd_ask / eurgbp_ask"


def test_vpm_places_strongest_eligible_opportunity_first() -> None:
    result = compile_opportunity_surface(deterministic_snapshots())

    top_left = result.artifact.cell(0, 0)

    assert top_left.row_id == "fx-snapshot-002|forward"
    assert top_left.metric_id == "gross_edge_bps"
    assert result.selected.decision == "EXECUTE_FORWARD"
    assert result.selected.expected_profit > Decimal("18")
    assert result.selected.cycle == "EUR -> USD -> GBP -> EUR"


def test_stale_profitable_candidate_is_rejected_not_executed() -> None:
    result = compile_opportunity_surface(deterministic_snapshots())
    stale = next(
        item
        for item in result.opportunities
        if item.opportunity_id == "fx-snapshot-005|forward"
    )

    assert stale.net_edge_bps > 0
    assert stale.quote_skew_ms == 850
    assert stale.decision == "REJECT_STALE_QUOTES"
    assert not stale.execution_eligible


def test_one_quote_mutation_changes_identity_and_decision_then_restores() -> None:
    coherent = deterministic_snapshots()[0]
    before = compile_opportunity_surface((coherent,))
    mutated = compile_opportunity_surface(
        (mutate_forward_opportunity(coherent, Decimal("3.0")),)
    )
    restored = compile_opportunity_surface((coherent,))

    assert before.selected.decision == "SKIP"
    assert mutated.selected.decision == "EXECUTE_FORWARD"
    assert before.artifact.artifact_id != mutated.artifact.artifact_id
    assert before.artifact.artifact_id == restored.artifact.artifact_id


def test_replay_recomputes_same_decision_and_artifact_identity() -> None:
    snapshots = deterministic_snapshots()
    result = compile_opportunity_surface(snapshots)

    receipt = replay(result, snapshots)

    assert receipt["original_decision"] == "EXECUTE_FORWARD"
    assert receipt["replayed_decision"] == "EXECUTE_FORWARD"
    assert receipt["replay_match"] is True


def test_demo_payload_contains_money_shot_and_rejection_trace() -> None:
    payload = demo_payload()

    assert payload["selected"]["decision"] == "EXECUTE_FORWARD"
    assert payload["mutation"]["after_decision"] == "EXECUTE_FORWARD"
    assert payload["stale_rejection"]["decision"] == "REJECT_STALE_QUOTES"
