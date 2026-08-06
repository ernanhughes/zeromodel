"""Compile deterministic FX triangular-arbitrage opportunities into a VPM sign.

This example is fully offline. It demonstrates bid/ask triangular-arbitrage
calculation, declared cost/freshness/liquidity checks, deterministic VPM
ordering, selected-sign recovery, mutation sensitivity, and replay.

It does not use live prices, prove profitable market arbitrage, execute orders,
or implement statistical arbitrage.

Run:

    python examples/fx_triangular_arbitrage.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, getcontext
import json
from pathlib import Path
from typing import Iterable, Literal

from zeromodel.core.artifact import LayoutRecipe, ScoreTable, VPMArtifact, build_vpm
from zeromodel.core.bundle import to_bundle

getcontext().prec = 28

D = Decimal
Direction = Literal["forward", "reverse"]


@dataclass(frozen=True, slots=True)
class FxQuote:
    pair: str
    bid: Decimal
    ask: Decimal
    timestamp_ms: int
    available_notional: Decimal

    def __post_init__(self) -> None:
        if not self.pair:
            raise ValueError("pair must be non-empty")
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("bid and ask must be positive")
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")
        if self.available_notional <= 0:
            raise ValueError("available_notional must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "bid": str(self.bid),
            "ask": str(self.ask),
            "timestamp_ms": self.timestamp_ms,
            "available_notional": str(self.available_notional),
        }


@dataclass(frozen=True, slots=True)
class FxSnapshot:
    snapshot_id: str
    quotes: tuple[FxQuote, ...]
    observed_at_ms: int

    def quote(self, pair: str) -> FxQuote:
        matches = [quote for quote in self.quotes if quote.pair == pair]
        if len(matches) != 1:
            raise ValueError(
                f"snapshot {self.snapshot_id!r} needs exactly one {pair} quote"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "observed_at_ms": self.observed_at_ms,
            "quotes": [quote.to_dict() for quote in self.quotes],
        }


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    starting_notional_eur: Decimal = D("100000")
    commission_bps: Decimal = D("0.60")
    slippage_reserve_bps: Decimal = D("0.40")
    max_quote_skew_ms: int = 50
    max_oldest_quote_age_ms: int = 100
    minimum_notional_eur: Decimal = D("100000")

    @property
    def cost_fraction(self) -> Decimal:
        return (self.commission_bps + self.slippage_reserve_bps) / D("10000")

    def to_dict(self) -> dict[str, object]:
        return {
            "starting_notional_eur": str(self.starting_notional_eur),
            "commission_bps": str(self.commission_bps),
            "slippage_reserve_bps": str(self.slippage_reserve_bps),
            "max_quote_skew_ms": self.max_quote_skew_ms,
            "max_oldest_quote_age_ms": self.max_oldest_quote_age_ms,
            "minimum_notional_eur": str(self.minimum_notional_eur),
        }


@dataclass(frozen=True, slots=True)
class Opportunity:
    opportunity_id: str
    snapshot_id: str
    direction: Direction
    cycle: str
    ending_eur: Decimal
    gross_edge_bps: Decimal
    commission_bps: Decimal
    slippage_reserve_bps: Decimal
    net_edge_bps: Decimal
    quote_skew_ms: int
    oldest_quote_age_ms: int
    available_notional: Decimal
    expected_profit: Decimal
    freshness_score: Decimal
    execution_eligible: bool
    decision: str
    source_quotes: tuple[FxQuote, ...]
    arithmetic: str

    def metric_row(self) -> tuple[float, ...]:
        return (
            float(self.gross_edge_bps),
            float(self.commission_bps),
            float(self.slippage_reserve_bps),
            float(self.net_edge_bps),
            float(self.quote_skew_ms),
            float(self.oldest_quote_age_ms),
            float(self.available_notional),
            float(self.expected_profit),
            float(self.freshness_score),
            1.0 if self.execution_eligible else 0.0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "opportunity_id": self.opportunity_id,
            "snapshot_id": self.snapshot_id,
            "direction": self.direction,
            "cycle": self.cycle,
            "ending_eur": str(self.ending_eur),
            "gross_edge_bps": str(self.gross_edge_bps),
            "commission_bps": str(self.commission_bps),
            "slippage_reserve_bps": str(self.slippage_reserve_bps),
            "net_edge_bps": str(self.net_edge_bps),
            "quote_skew_ms": self.quote_skew_ms,
            "oldest_quote_age_ms": self.oldest_quote_age_ms,
            "available_notional": str(self.available_notional),
            "expected_profit": str(self.expected_profit),
            "freshness_score": str(self.freshness_score),
            "execution_eligible": self.execution_eligible,
            "decision": self.decision,
            "source_quotes": [quote.to_dict() for quote in self.source_quotes],
            "arithmetic": self.arithmetic,
        }


@dataclass(frozen=True, slots=True)
class FxArbitrageResult:
    artifact: VPMArtifact
    opportunities: tuple[Opportunity, ...]
    selected: Opportunity
    contract: ExecutionContract

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact.artifact_id,
            "selected_opportunity_id": self.selected.opportunity_id,
            "decision": self.selected.decision,
            "cycle": self.selected.cycle,
            "expected_profit": str(self.selected.expected_profit),
            "net_edge_bps": str(self.selected.net_edge_bps),
            "opportunities": [
                opportunity.to_dict() for opportunity in self.opportunities
            ],
            "contract": self.contract.to_dict(),
        }


METRIC_IDS = (
    "gross_edge_bps",
    "commission_bps",
    "slippage_reserve_bps",
    "net_edge_bps",
    "quote_skew_ms",
    "oldest_quote_age_ms",
    "available_notional",
    "expected_profit",
    "freshness_score",
    "execution_eligible",
)


def quote(
    pair: str,
    bid: str,
    ask: str,
    timestamp_ms: int,
    available_notional: str = "250000",
) -> FxQuote:
    return FxQuote(pair, D(bid), D(ask), timestamp_ms, D(available_notional))


def deterministic_snapshots() -> tuple[FxSnapshot, ...]:
    return (
        FxSnapshot(
            "fx-snapshot-001",
            (
                quote("EUR/USD", "1.1000", "1.1002", 1000),
                quote("GBP/USD", "1.2500", "1.2503", 1004),
                quote("EUR/GBP", "0.87956", "0.87986", 1007),
            ),
            1010,
        ),
        FxSnapshot(
            "fx-snapshot-002",
            (
                quote("EUR/USD", "1.1000", "1.1002", 2000),
                quote("GBP/USD", "1.2500", "1.2503", 2004),
                quote("EUR/GBP", "0.87937", "0.87954", 2007),
            ),
            2010,
        ),
        FxSnapshot(
            "fx-snapshot-003",
            (
                quote("EUR/USD", "1.1000", "1.1002", 3000),
                quote("GBP/USD", "1.2500", "1.2503", 3003),
                quote("EUR/GBP", "0.8804", "0.8807", 3005),
            ),
            3008,
        ),
        FxSnapshot(
            "fx-snapshot-004",
            (
                quote("EUR/USD", "1.1000", "1.1002", 4000),
                quote("GBP/USD", "1.2500", "1.2503", 4005),
                quote("EUR/GBP", "0.87975", "0.87990", 4007),
            ),
            4010,
        ),
        FxSnapshot(
            "fx-snapshot-005",
            (
                quote("EUR/USD", "1.1000", "1.1002", 5000),
                quote("GBP/USD", "1.2500", "1.2503", 5850),
                quote("EUR/GBP", "0.87950", "0.87970", 5007),
            ),
            5860,
        ),
    )


def _freshness_score(
    skew_ms: int, oldest_age_ms: int, contract: ExecutionContract
) -> Decimal:
    skew_component = max(D("0"), D("1") - D(skew_ms) / D(contract.max_quote_skew_ms))
    age_component = max(
        D("0"),
        D("1") - D(oldest_age_ms) / D(contract.max_oldest_quote_age_ms),
    )
    return (skew_component + age_component) / D("2")


def evaluate_cycle(
    snapshot: FxSnapshot,
    direction: Direction,
    contract: ExecutionContract,
) -> Opportunity:
    eurusd = snapshot.quote("EUR/USD")
    gbpusd = snapshot.quote("GBP/USD")
    eurgbp = snapshot.quote("EUR/GBP")
    quotes = (eurusd, gbpusd, eurgbp)
    timestamps = [item.timestamp_ms for item in quotes]
    quote_skew_ms = max(timestamps) - min(timestamps)
    oldest_quote_age_ms = snapshot.observed_at_ms - min(timestamps)
    available_notional = min(item.available_notional for item in quotes)

    if direction == "forward":
        ending_eur = eurusd.bid / gbpusd.ask / eurgbp.ask
        cycle = "EUR -> USD -> GBP -> EUR"
        arithmetic = "ending_eur = eurusd_bid / gbpusd_ask / eurgbp_ask"
        execute_decision = "EXECUTE_FORWARD"
    else:
        ending_eur = eurgbp.bid * gbpusd.bid / eurusd.ask
        cycle = "EUR -> GBP -> USD -> EUR"
        arithmetic = "ending_eur = eurgbp_bid * gbpusd_bid / eurusd_ask"
        execute_decision = "EXECUTE_REVERSE"

    gross_edge = ending_eur - D("1")
    gross_edge_bps = gross_edge * D("10000")
    net_edge = gross_edge - contract.cost_fraction
    net_edge_bps = net_edge * D("10000")
    expected_profit = contract.starting_notional_eur * net_edge
    freshness = _freshness_score(quote_skew_ms, oldest_quote_age_ms, contract)

    stale = (
        quote_skew_ms > contract.max_quote_skew_ms
        or oldest_quote_age_ms > contract.max_oldest_quote_age_ms
    )
    insufficient_liquidity = available_notional < contract.minimum_notional_eur
    execution_eligible = net_edge > 0 and not stale and not insufficient_liquidity
    if stale:
        decision = "REJECT_STALE_QUOTES"
    elif insufficient_liquidity:
        decision = "REJECT_INSUFFICIENT_LIQUIDITY"
    elif execution_eligible:
        decision = execute_decision
    else:
        decision = "SKIP"

    return Opportunity(
        opportunity_id=f"{snapshot.snapshot_id}|{direction}",
        snapshot_id=snapshot.snapshot_id,
        direction=direction,
        cycle=cycle,
        ending_eur=ending_eur,
        gross_edge_bps=gross_edge_bps,
        commission_bps=contract.commission_bps,
        slippage_reserve_bps=contract.slippage_reserve_bps,
        net_edge_bps=net_edge_bps,
        quote_skew_ms=quote_skew_ms,
        oldest_quote_age_ms=oldest_quote_age_ms,
        available_notional=available_notional,
        expected_profit=expected_profit,
        freshness_score=freshness,
        execution_eligible=execution_eligible,
        decision=decision,
        source_quotes=quotes,
        arithmetic=arithmetic,
    )


def evaluate_snapshots(
    snapshots: Iterable[FxSnapshot],
    contract: ExecutionContract,
) -> tuple[Opportunity, ...]:
    opportunities: list[Opportunity] = []
    for snapshot in snapshots:
        opportunities.append(evaluate_cycle(snapshot, "forward", contract))
        opportunities.append(evaluate_cycle(snapshot, "reverse", contract))
    return tuple(opportunities)


def fx_layout_recipe() -> LayoutRecipe:
    return LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "fx-triangular-arbitrage-opportunity-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [
                    {"metric_id": "execution_eligible", "direction": "desc"},
                    {"metric_id": "expected_profit", "direction": "desc"},
                    {"metric_id": "net_edge_bps", "direction": "desc"},
                    {"metric_id": "freshness_score", "direction": "desc"},
                ],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


def compile_opportunity_surface(
    snapshots: Iterable[FxSnapshot] | None = None,
    contract: ExecutionContract | None = None,
) -> FxArbitrageResult:
    contract = contract or ExecutionContract()
    snapshots_tuple = tuple(snapshots or deterministic_snapshots())
    opportunities = evaluate_snapshots(snapshots_tuple, contract)
    table = ScoreTable(
        values=[opportunity.metric_row() for opportunity in opportunities],
        row_ids=[opportunity.opportunity_id for opportunity in opportunities],
        metric_ids=METRIC_IDS,
        metadata={
            "kind": "fx_triangular_arbitrage_opportunity_surface",
            "contract": contract.to_dict(),
            "snapshots": [snapshot.to_dict() for snapshot in snapshots_tuple],
            "opportunities": [opportunity.to_dict() for opportunity in opportunities],
        },
    )
    artifact = build_vpm(
        table,
        fx_layout_recipe(),
        provenance={
            "kind": "fx_triangular_arbitrage",
            "parents": [],
            "ordering_contract": (
                "execution_eligible desc; expected_profit desc; "
                "net_edge_bps desc; freshness_score desc; row_id asc"
            ),
        },
    )
    selected_row_id = artifact.cell(0, 0).row_id
    selected = next(
        item for item in opportunities if item.opportunity_id == selected_row_id
    )
    return FxArbitrageResult(
        artifact=artifact,
        opportunities=opportunities,
        selected=selected,
        contract=contract,
    )


def mutate_forward_opportunity(
    snapshot: FxSnapshot, ask_reduction_bps: Decimal
) -> FxSnapshot:
    factor = D("1") - ask_reduction_bps / D("10000")
    quotes = []
    for item in snapshot.quotes:
        if item.pair == "EUR/GBP":
            quotes.append(
                FxQuote(
                    item.pair,
                    bid=item.bid,
                    ask=(item.ask * factor).quantize(D("0.00001")),
                    timestamp_ms=item.timestamp_ms,
                    available_notional=item.available_notional,
                )
            )
        else:
            quotes.append(item)
    return FxSnapshot(
        snapshot.snapshot_id + "-mutated", tuple(quotes), snapshot.observed_at_ms
    )


def replay(
    result: FxArbitrageResult, snapshots: Iterable[FxSnapshot]
) -> dict[str, object]:
    replayed = compile_opportunity_surface(
        snapshots=snapshots, contract=result.contract
    )
    return {
        "original_decision": result.selected.decision,
        "replayed_decision": replayed.selected.decision,
        "original_artifact_id": result.artifact.artifact_id,
        "replayed_artifact_id": replayed.artifact.artifact_id,
        "replay_match": result.artifact.artifact_id == replayed.artifact.artifact_id
        and result.selected.decision == replayed.selected.decision,
    }


def demo_payload() -> dict[str, object]:
    snapshots = deterministic_snapshots()
    result = compile_opportunity_surface(snapshots)
    coherent = snapshots[0]
    before = compile_opportunity_surface((coherent,))
    mutated_snapshot = mutate_forward_opportunity(coherent, D("3.0"))
    after = compile_opportunity_surface((mutated_snapshot,))
    restored = compile_opportunity_surface((coherent,))
    stale = next(
        opportunity
        for opportunity in result.opportunities
        if opportunity.snapshot_id == "fx-snapshot-005"
        and opportunity.direction == "forward"
    )
    return {
        "bounded_claim": (
            "ZeroModel can compile identified FX quotes and a declared "
            "triangular-arbitrage calculation into a deterministic VPM where "
            "the strongest executable opportunity occupies a predictable "
            "location and can be recovered with its complete calculation and "
            "source trace."
        ),
        "selected": result.selected.to_dict(),
        "artifact_id": result.artifact.artifact_id,
        "row_order": [
            result.artifact.cell(index, 0).row_id
            for index in range(result.artifact.shape[0])
        ],
        "metrics": list(result.artifact.source.metric_ids),
        "mutation": {
            "before_decision": before.selected.decision,
            "after_decision": after.selected.decision,
            "restored_decision": restored.selected.decision,
            "before_artifact_id": before.artifact.artifact_id,
            "after_artifact_id": after.artifact.artifact_id,
            "restored_artifact_id": restored.artifact.artifact_id,
            "changed": before.artifact.artifact_id != after.artifact.artifact_id,
            "restored": before.artifact.artifact_id == restored.artifact.artifact_id,
        },
        "stale_rejection": stale.to_dict(),
        "replay": replay(result, snapshots),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    snapshots = deterministic_snapshots()
    result = compile_opportunity_surface(snapshots)
    payload = demo_payload()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        to_bundle(result.artifact, args.output_dir / "fx-triangular-arbitrage.vpm")
        (args.output_dir / "fx-triangular-arbitrage.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        payload["output_dir"] = str(args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
