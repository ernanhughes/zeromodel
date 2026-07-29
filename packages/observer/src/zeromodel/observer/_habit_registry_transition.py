"""Shared semantic transition helpers for Observer habit registries."""

from __future__ import annotations

from collections.abc import Mapping

from zeromodel.observer.habit_registry import (
    ObserverHabitRegistryEventDTO,
    ObserverHabitRegistrySnapshotDTO,
    _apply_registry_event,
)


def apply_observer_habit_registry_event(
    *,
    source: ObserverHabitRegistrySnapshotDTO,
    event: ObserverHabitRegistryEventDTO,
    known_snapshots: Mapping[str, ObserverHabitRegistrySnapshotDTO],
) -> tuple[ObserverHabitRegistrySnapshotDTO, tuple[str, ...]]:
    """Apply one canonical registry event without persistence concerns."""

    result, failures = _apply_registry_event(
        source=source,
        event=event,
        known_snapshots=known_snapshots,
    )
    return result, tuple(sorted(failures))
