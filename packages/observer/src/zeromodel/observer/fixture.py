"""Deterministic Observer fixture contracts for Stage O3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id

OBSERVER_FIXTURE_RULE_SET_VERSION: Final = "observer-fixture-rule-set/1"
OBSERVER_FIXTURE_STATE_VERSION: Final = "observer-fixture-state/1"
OBSERVER_FIXTURE_ACTION_VERSION: Final = "observer-fixture-action/1"
OBSERVER_EXECUTED_FIXTURE_STEP_VERSION: Final = "observer-executed-fixture-step/1"

FIXTURE_ACTIONS: Final = frozenset({"move_left", "move_right", "wait"})
COOLDOWN_EFFECTS: Final = frozenset({"block", "reverse", "ignore"})


class ObserverFixtureError(ValueError):
    """Raised when deterministic Observer fixture contracts are invalid."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverFixtureError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverFixtureError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverFixtureRuleSetDTO:
    """Canonical finite fixture rule set."""

    fixture_rule_set_id: str
    fixture_id: str
    rule_version: str
    minimum_position: int
    maximum_position: int
    cooldown_period: int
    cooldown_effect: str
    allowed_actions: tuple[str, ...]
    observation_schema_id: str
    version: str = OBSERVER_FIXTURE_RULE_SET_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FIXTURE_RULE_SET_VERSION:
            raise ObserverFixtureError("unsupported fixture rule-set version")
        for field_name in ("fixture_id", "rule_version", "observation_schema_id"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.minimum_position > self.maximum_position:
            raise ObserverFixtureError(
                "minimum_position cannot exceed maximum_position"
            )
        if self.cooldown_period < 0:
            raise ObserverFixtureError("cooldown_period must be non-negative")
        if self.cooldown_effect not in COOLDOWN_EFFECTS:
            raise ObserverFixtureError("unsupported cooldown_effect")
        _ensure_sorted_unique(self.allowed_actions, "allowed_actions")
        if set(self.allowed_actions) - FIXTURE_ACTIONS:
            raise ObserverFixtureError("allowed_actions contains unsupported action")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.fixture_rule_set_id != expected_id:
            raise ObserverFixtureError(
                "fixture_rule_set_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "allowed_actions": list(self.allowed_actions),
            "cooldown_effect": self.cooldown_effect,
            "cooldown_period": self.cooldown_period,
            "fixture_id": self.fixture_id,
            "maximum_position": self.maximum_position,
            "minimum_position": self.minimum_position,
            "observation_schema_id": self.observation_schema_id,
            "rule_version": self.rule_version,
            "version": self.version,
        }
        if include_id:
            payload["fixture_rule_set_id"] = self.fixture_rule_set_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        fixture_id: str,
        rule_version: str,
        minimum_position: int,
        maximum_position: int,
        cooldown_period: int,
        cooldown_effect: str,
        observation_schema_id: str,
        allowed_actions: tuple[str, ...] = ("move_left", "move_right", "wait"),
    ) -> "ObserverFixtureRuleSetDTO":
        allowed_actions = tuple(sorted(allowed_actions))
        payload = {
            "allowed_actions": list(allowed_actions),
            "cooldown_effect": cooldown_effect,
            "cooldown_period": cooldown_period,
            "fixture_id": fixture_id,
            "maximum_position": maximum_position,
            "minimum_position": minimum_position,
            "observation_schema_id": observation_schema_id,
            "rule_version": rule_version,
            "version": OBSERVER_FIXTURE_RULE_SET_VERSION,
        }
        return cls(
            fixture_rule_set_id=canonical_id(payload),
            fixture_id=fixture_id,
            rule_version=rule_version,
            minimum_position=minimum_position,
            maximum_position=maximum_position,
            cooldown_period=cooldown_period,
            cooldown_effect=cooldown_effect,
            allowed_actions=allowed_actions,
            observation_schema_id=observation_schema_id,
        )


@dataclass(frozen=True)
class ObserverFixtureActionDTO:
    """Canonical fixture action."""

    fixture_action_id: str
    action_name: str
    version: str = OBSERVER_FIXTURE_ACTION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FIXTURE_ACTION_VERSION:
            raise ObserverFixtureError("unsupported fixture action version")
        if self.action_name not in FIXTURE_ACTIONS:
            raise ObserverFixtureError("unsupported fixture action")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.fixture_action_id != expected_id:
            raise ObserverFixtureError(
                "fixture_action_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = {"action_name": self.action_name, "version": self.version}
        if include_id:
            payload["fixture_action_id"] = self.fixture_action_id
        return payload

    @classmethod
    def create(cls, *, action_name: str) -> "ObserverFixtureActionDTO":
        payload = {
            "action_name": action_name,
            "version": OBSERVER_FIXTURE_ACTION_VERSION,
        }
        return cls(fixture_action_id=canonical_id(payload), action_name=action_name)


@dataclass(frozen=True)
class ObserverFixtureStateDTO:
    """Canonical fixture state."""

    fixture_state_id: str
    fixture_id: str
    rule_set_id: str
    episode_id: str
    step_index: int
    agent_x: int
    target_x: int
    previous_action: str | None
    cooldown_remaining: int
    terminal: bool
    version: str = OBSERVER_FIXTURE_STATE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FIXTURE_STATE_VERSION:
            raise ObserverFixtureError("unsupported fixture state version")
        for field_name in ("fixture_id", "rule_set_id", "episode_id"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.step_index < 0:
            raise ObserverFixtureError("step_index must be non-negative")
        if self.cooldown_remaining < 0:
            raise ObserverFixtureError("cooldown_remaining must be non-negative")
        if (
            self.previous_action is not None
            and self.previous_action not in FIXTURE_ACTIONS
        ):
            raise ObserverFixtureError("previous_action is unsupported")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.fixture_state_id != expected_id:
            raise ObserverFixtureError(
                "fixture_state_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "agent_x": self.agent_x,
            "cooldown_remaining": self.cooldown_remaining,
            "episode_id": self.episode_id,
            "fixture_id": self.fixture_id,
            "previous_action": self.previous_action,
            "rule_set_id": self.rule_set_id,
            "step_index": self.step_index,
            "target_x": self.target_x,
            "terminal": self.terminal,
            "version": self.version,
        }
        if include_id:
            payload["fixture_state_id"] = self.fixture_state_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        fixture_id: str,
        rule_set_id: str,
        episode_id: str,
        step_index: int,
        agent_x: int,
        target_x: int,
        previous_action: str | None = None,
        cooldown_remaining: int = 0,
        terminal: bool = False,
    ) -> "ObserverFixtureStateDTO":
        payload = {
            "agent_x": agent_x,
            "cooldown_remaining": cooldown_remaining,
            "episode_id": episode_id,
            "fixture_id": fixture_id,
            "previous_action": previous_action,
            "rule_set_id": rule_set_id,
            "step_index": step_index,
            "target_x": target_x,
            "terminal": terminal,
            "version": OBSERVER_FIXTURE_STATE_VERSION,
        }
        return cls(
            fixture_state_id=canonical_id(payload),
            fixture_id=fixture_id,
            rule_set_id=rule_set_id,
            episode_id=episode_id,
            step_index=step_index,
            agent_x=agent_x,
            target_x=target_x,
            previous_action=previous_action,
            cooldown_remaining=cooldown_remaining,
            terminal=terminal,
        )


@dataclass(frozen=True)
class ObserverExecutedFixtureStepDTO:
    """Environment truth for one deterministic fixture step."""

    executed_step_id: str
    source_state_id: str
    action_id: str
    environment_rule_set_id: str
    actual_state: ObserverFixtureStateDTO
    actual_observation_id: str
    action_effect: str
    version: str = OBSERVER_EXECUTED_FIXTURE_STEP_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_EXECUTED_FIXTURE_STEP_VERSION:
            raise ObserverFixtureError("unsupported executed fixture step version")
        for field_name in (
            "source_state_id",
            "action_id",
            "environment_rule_set_id",
            "actual_observation_id",
            "action_effect",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.executed_step_id != expected_id:
            raise ObserverFixtureError(
                "executed_step_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action_effect": self.action_effect,
            "action_id": self.action_id,
            "actual_observation_id": self.actual_observation_id,
            "actual_state": self.actual_state.canonical_payload(),
            "environment_rule_set_id": self.environment_rule_set_id,
            "source_state_id": self.source_state_id,
            "version": self.version,
        }
        if include_id:
            payload["executed_step_id"] = self.executed_step_id
        return payload
