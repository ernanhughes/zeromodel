"""The `TraversalRule` seam and a simple, deterministic reference rule.

This is explicitly not search: `FixedKeySelectorRule` routes on an exact
match against a declared request attribute, never a similarity score. A
later Search package implements this same protocol with similarity-driven
rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

from zeromodel.navigation.dto import (
    NavigationTileDTO,
    TilePointerDTO,
    TraversalRequestDTO,
    TraversalRuleDescriptorDTO,
)


@dataclass(frozen=True)
class ChildSelection:
    """The result of one rule invocation - enough to build a `TraversalStepDTO`."""

    selected: Optional[TilePointerDTO]
    eligible: Tuple[TilePointerDTO, ...]
    tie_candidates: Tuple[str, ...]
    tie_resolution: str


@runtime_checkable
class TraversalRule(Protocol):
    """A stable, declared routing rule - never a live, unstable closure."""

    def descriptor(self) -> TraversalRuleDescriptorDTO: ...

    def select_child(
        self,
        request: TraversalRequestDTO,
        tile: NavigationTileDTO,
        children: Tuple[TilePointerDTO, ...],
    ) -> ChildSelection: ...


@dataclass(frozen=True)
class FixedKeySelectorRule:
    """Select the child whose `order_key` equals a declared request attribute.

    Deterministic non-search routing: `routing_key` names the attribute on
    `TraversalRequestDTO.attributes` whose value must exactly equal a
    child's `order_key`. Multiple matches (which should not occur given
    `NavigationTileDTO`'s duplicate-order_key rejection) tie-break on the
    lowest `order_key` string; no match is a declared failure, not a guess.
    """

    routing_key: str

    def descriptor(self) -> TraversalRuleDescriptorDTO:
        return TraversalRuleDescriptorDTO(
            rule_kind="fixed_selector_by_request_key",
            parameters=(("routing_key", self.routing_key),),
        )

    def select_child(
        self,
        request: TraversalRequestDTO,
        tile: NavigationTileDTO,
        children: Tuple[TilePointerDTO, ...],
    ) -> ChildSelection:
        target_value = request.attributes_map.get(self.routing_key)
        matches = tuple(child for child in children if child.order_key == target_value)
        if len(matches) == 1:
            return ChildSelection(
                selected=matches[0],
                eligible=children,
                tie_candidates=(),
                tie_resolution="exact_match",
            )
        if len(matches) > 1:
            winner = min(matches, key=lambda child: child.order_key)
            return ChildSelection(
                selected=winner,
                eligible=children,
                tie_candidates=tuple(child.order_key for child in matches),
                tie_resolution="lowest_order_key",
            )
        return ChildSelection(
            selected=None,
            eligible=children,
            tie_candidates=(),
            tie_resolution="no_match",
        )


@dataclass(frozen=True)
class DeclaredPriorityRule:
    """Select the child at a declared priority position (index) among
    the tile's ordered children - a fixed target-range routing rule.

    `priority_attribute` names the request attribute holding the desired
    zero-based position. Out-of-range or non-numeric values are a declared
    failure, never a fallback guess.
    """

    priority_attribute: str

    def descriptor(self) -> TraversalRuleDescriptorDTO:
        return TraversalRuleDescriptorDTO(
            rule_kind="declared_priority_index",
            parameters=(("priority_attribute", self.priority_attribute),),
        )

    def select_child(
        self,
        request: TraversalRequestDTO,
        tile: NavigationTileDTO,
        children: Tuple[TilePointerDTO, ...],
    ) -> ChildSelection:
        raw_value = request.attributes_map.get(self.priority_attribute)
        if raw_value is None:
            return ChildSelection(
                selected=None,
                eligible=children,
                tie_candidates=(),
                tie_resolution="not_an_index",
            )
        try:
            index = int(raw_value)
        except ValueError:
            return ChildSelection(
                selected=None,
                eligible=children,
                tie_candidates=(),
                tie_resolution="not_an_index",
            )
        if index < 0 or index >= len(children):
            return ChildSelection(
                selected=None,
                eligible=children,
                tie_candidates=(),
                tie_resolution="out_of_range",
            )
        return ChildSelection(
            selected=children[index],
            eligible=children,
            tie_candidates=(),
            tie_resolution="declared_priority_index",
        )
