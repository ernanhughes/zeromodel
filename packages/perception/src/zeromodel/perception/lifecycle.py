"""Immutable promoted-model ledger and rollback-safe lifecycle state for Stage P12.

P12 separates immutable promoted model artifacts from mutable operational selection.
Models and lifecycle transitions are append-only. The active model is represented by a
revisioned pointer whose updates require an expected previous revision, preventing stale
writers from silently replacing newer operational state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping, Protocol

from .promoted_inference import PromotedTestEvaluationReportDTO
from .promotion import PromotedPerceptionModelDTO

MODEL_LEDGER_ENTRY_VERSION: Final = "perception-model-ledger-entry/1"
MODEL_TRANSITION_VERSION: Final = "perception-model-transition/1"
ACTIVE_MODEL_POINTER_VERSION: Final = "perception-active-model-pointer/1"
MODEL_LIFECYCLE_SNAPSHOT_VERSION: Final = "perception-model-lifecycle-snapshot/1"
MODEL_LEDGER_SEMANTICS: Final = "append_only_promoted_model_registration"
MODEL_TRANSITION_SEMANTICS: Final = "append_only_model_activation_supersession_and_rollback"
ACTIVE_POINTER_SEMANTICS: Final = "revisioned_active_model_pointer_with_optimistic_concurrency"
MODEL_TRANSITION_KINDS: Final = {"activate", "supersede", "rollback", "deactivate"}


class PerceptionModelLifecycleError(ValueError):
    """Raised when promoted-model lifecycle contracts are violated."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class PromotedModelLedgerEntryDTO:
    ledger_entry_id: str
    promoted_model: PromotedPerceptionModelDTO
    test_evaluation_report_id: str | None
    registered_by: str
    registration_reason: str
    semantics: str = MODEL_LEDGER_SEMANTICS
    version: str = MODEL_LEDGER_ENTRY_VERSION

    def __post_init__(self) -> None:
        if not all((self.ledger_entry_id, self.registered_by, self.registration_reason)):
            raise PerceptionModelLifecycleError("ledger entry identities and rationale must be non-empty")
        if self.semantics != MODEL_LEDGER_SEMANTICS:
            raise PerceptionModelLifecycleError("unsupported model ledger semantics")


@dataclass(frozen=True)
class ModelLifecycleTransitionDTO:
    transition_id: str
    sequence_number: int
    transition_kind: str
    previous_promoted_model_id: str | None
    next_promoted_model_id: str | None
    actor: str
    reason: str
    related_transition_id: str | None = None
    semantics: str = MODEL_TRANSITION_SEMANTICS
    version: str = MODEL_TRANSITION_VERSION

    def __post_init__(self) -> None:
        if self.transition_kind not in MODEL_TRANSITION_KINDS:
            raise PerceptionModelLifecycleError("unsupported model transition kind")
        if self.sequence_number <= 0:
            raise PerceptionModelLifecycleError("transition sequence_number must be positive")
        if not all((self.transition_id, self.actor, self.reason)):
            raise PerceptionModelLifecycleError("transition identity, actor, and reason must be non-empty")
        if self.transition_kind == "activate":
            if self.previous_promoted_model_id is not None or self.next_promoted_model_id is None:
                raise PerceptionModelLifecycleError("activation requires no previous model and one next model")
        elif self.transition_kind in {"supersede", "rollback"}:
            if not self.previous_promoted_model_id or not self.next_promoted_model_id:
                raise PerceptionModelLifecycleError(
                    "supersession and rollback require previous and next model identities"
                )
            if self.previous_promoted_model_id == self.next_promoted_model_id:
                raise PerceptionModelLifecycleError("transition cannot replace a model with itself")
        elif self.transition_kind == "deactivate":
            if self.previous_promoted_model_id is None or self.next_promoted_model_id is not None:
                raise PerceptionModelLifecycleError(
                    "deactivation requires one previous model and no next model"
                )
        if self.semantics != MODEL_TRANSITION_SEMANTICS:
            raise PerceptionModelLifecycleError("unsupported model transition semantics")


@dataclass(frozen=True)
class ActiveModelPointerDTO:
    pointer_id: str
    revision: int
    active_promoted_model_id: str | None
    last_transition_id: str | None
    semantics: str = ACTIVE_POINTER_SEMANTICS
    version: str = ACTIVE_MODEL_POINTER_VERSION

    def __post_init__(self) -> None:
        if not self.pointer_id:
            raise PerceptionModelLifecycleError("active pointer identity must be non-empty")
        if self.revision < 0:
            raise PerceptionModelLifecycleError("active pointer revision cannot be negative")
        if self.revision == 0 and (
            self.active_promoted_model_id is not None or self.last_transition_id is not None
        ):
            raise PerceptionModelLifecycleError("revision zero pointer must be empty")
        if self.revision > 0 and not self.last_transition_id:
            raise PerceptionModelLifecycleError("non-zero pointer revision requires transition identity")
        if self.semantics != ACTIVE_POINTER_SEMANTICS:
            raise PerceptionModelLifecycleError("unsupported active pointer semantics")


@dataclass(frozen=True)
class ModelLifecycleSnapshotDTO:
    snapshot_id: str
    active_pointer: ActiveModelPointerDTO
    ledger_entries: tuple[PromotedModelLedgerEntryDTO, ...]
    transitions: tuple[ModelLifecycleTransitionDTO, ...]
    version: str = MODEL_LIFECYCLE_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise PerceptionModelLifecycleError("lifecycle snapshot identity must be non-empty")
        if self.ledger_entries != tuple(
            sorted(self.ledger_entries, key=lambda item: item.promoted_model.promoted_model_id)
        ):
            raise PerceptionModelLifecycleError("ledger entries must be sorted by promoted model identity")
        if self.transitions != tuple(sorted(self.transitions, key=lambda item: item.sequence_number)):
            raise PerceptionModelLifecycleError("transitions must be sorted by sequence number")
        sequences = tuple(item.sequence_number for item in self.transitions)
        if sequences != tuple(range(1, len(sequences) + 1)):
            raise PerceptionModelLifecycleError("transition sequence must be contiguous from one")
        if self.active_pointer.revision != len(self.transitions):
            raise PerceptionModelLifecycleError("pointer revision must equal transition count")


class PerceptionModelLifecycleStore(Protocol):
    """DTO-only persistence boundary for promoted model lifecycle state."""

    def put_ledger_entry(self, entry: PromotedModelLedgerEntryDTO) -> None: ...

    def get_ledger_entry(self, promoted_model_id: str) -> PromotedModelLedgerEntryDTO: ...

    def list_ledger_entries(self) -> tuple[PromotedModelLedgerEntryDTO, ...]: ...

    def append_transition(self, transition: ModelLifecycleTransitionDTO) -> None: ...

    def list_transitions(self) -> tuple[ModelLifecycleTransitionDTO, ...]: ...

    def get_active_pointer(self) -> ActiveModelPointerDTO: ...

    def replace_active_pointer(
        self,
        pointer: ActiveModelPointerDTO,
        *,
        expected_revision: int,
    ) -> None: ...


class InMemoryPerceptionModelLifecycleStore:
    """Deterministic in-memory implementation of the P12 lifecycle store protocol."""

    def __init__(self) -> None:
        self._entries: dict[str, PromotedModelLedgerEntryDTO] = {}
        self._transitions: list[ModelLifecycleTransitionDTO] = []
        self._pointer = _empty_pointer()

    def put_ledger_entry(self, entry: PromotedModelLedgerEntryDTO) -> None:
        key = entry.promoted_model.promoted_model_id
        existing = self._entries.get(key)
        if existing is not None and existing != entry:
            raise PerceptionModelLifecycleError("promoted model identity already has different ledger entry")
        self._entries[key] = entry

    def get_ledger_entry(self, promoted_model_id: str) -> PromotedModelLedgerEntryDTO:
        try:
            return self._entries[promoted_model_id]
        except KeyError as exc:
            raise PerceptionModelLifecycleError(
                f"unknown promoted model identity: {promoted_model_id}"
            ) from exc

    def list_ledger_entries(self) -> tuple[PromotedModelLedgerEntryDTO, ...]:
        return tuple(sorted(self._entries.values(), key=lambda item: item.promoted_model.promoted_model_id))

    def append_transition(self, transition: ModelLifecycleTransitionDTO) -> None:
        expected_sequence = len(self._transitions) + 1
        if transition.sequence_number != expected_sequence:
            raise PerceptionModelLifecycleError("transition sequence does not follow ledger history")
        if any(item.transition_id == transition.transition_id for item in self._transitions):
            raise PerceptionModelLifecycleError("transition identity already exists")
        self._transitions.append(transition)

    def list_transitions(self) -> tuple[ModelLifecycleTransitionDTO, ...]:
        return tuple(self._transitions)

    def get_active_pointer(self) -> ActiveModelPointerDTO:
        return self._pointer

    def replace_active_pointer(
        self,
        pointer: ActiveModelPointerDTO,
        *,
        expected_revision: int,
    ) -> None:
        if self._pointer.revision != expected_revision:
            raise PerceptionModelLifecycleError(
                "active model pointer revision changed during lifecycle update"
            )
        if pointer.revision != expected_revision + 1:
            raise PerceptionModelLifecycleError("replacement pointer must advance revision by one")
        self._pointer = pointer


def _empty_pointer() -> ActiveModelPointerDTO:
    payload: Mapping[str, object] = {
        "active_promoted_model_id": None,
        "last_transition_id": None,
        "revision": 0,
        "semantics": ACTIVE_POINTER_SEMANTICS,
        "version": ACTIVE_MODEL_POINTER_VERSION,
    }
    return ActiveModelPointerDTO(
        pointer_id=_digest(_canonical_json(payload)),
        revision=0,
        active_promoted_model_id=None,
        last_transition_id=None,
    )


def register_promoted_model(
    store: PerceptionModelLifecycleStore,
    promoted_model: PromotedPerceptionModelDTO,
    *,
    registered_by: str,
    registration_reason: str,
    test_evaluation: PromotedTestEvaluationReportDTO | None = None,
) -> PromotedModelLedgerEntryDTO:
    """Register one immutable promoted model and optional P11 test report identity."""

    if not registered_by or not registration_reason:
        raise PerceptionModelLifecycleError("registration actor and reason must be non-empty")
    if test_evaluation is not None:
        if test_evaluation.promoted_model_id != promoted_model.promoted_model_id:
            raise PerceptionModelLifecycleError("test evaluation does not belong to promoted model")
        test_report_id = test_evaluation.report_id
    else:
        test_report_id = None
    payload: Mapping[str, object] = {
        "promoted_model_id": promoted_model.promoted_model_id,
        "registered_by": registered_by,
        "registration_reason": registration_reason,
        "semantics": MODEL_LEDGER_SEMANTICS,
        "test_evaluation_report_id": test_report_id,
        "version": MODEL_LEDGER_ENTRY_VERSION,
    }
    entry = PromotedModelLedgerEntryDTO(
        ledger_entry_id=_digest(_canonical_json(payload)),
        promoted_model=promoted_model,
        test_evaluation_report_id=test_report_id,
        registered_by=registered_by,
        registration_reason=registration_reason,
    )
    store.put_ledger_entry(entry)
    return entry


def _transition(
    store: PerceptionModelLifecycleStore,
    *,
    transition_kind: str,
    next_promoted_model_id: str | None,
    actor: str,
    reason: str,
    related_transition_id: str | None = None,
) -> tuple[ModelLifecycleTransitionDTO, ActiveModelPointerDTO]:
    if not actor or not reason:
        raise PerceptionModelLifecycleError("transition actor and reason must be non-empty")
    pointer = store.get_active_pointer()
    previous = pointer.active_promoted_model_id
    if next_promoted_model_id is not None:
        store.get_ledger_entry(next_promoted_model_id)
    sequence_number = pointer.revision + 1
    payload: Mapping[str, object] = {
        "actor": actor,
        "next_promoted_model_id": next_promoted_model_id,
        "previous_promoted_model_id": previous,
        "reason": reason,
        "related_transition_id": related_transition_id,
        "semantics": MODEL_TRANSITION_SEMANTICS,
        "sequence_number": sequence_number,
        "transition_kind": transition_kind,
        "version": MODEL_TRANSITION_VERSION,
    }
    transition = ModelLifecycleTransitionDTO(
        transition_id=_digest(_canonical_json(payload)),
        sequence_number=sequence_number,
        transition_kind=transition_kind,
        previous_promoted_model_id=previous,
        next_promoted_model_id=next_promoted_model_id,
        actor=actor,
        reason=reason,
        related_transition_id=related_transition_id,
    )
    pointer_payload: Mapping[str, object] = {
        "active_promoted_model_id": next_promoted_model_id,
        "last_transition_id": transition.transition_id,
        "revision": sequence_number,
        "semantics": ACTIVE_POINTER_SEMANTICS,
        "version": ACTIVE_MODEL_POINTER_VERSION,
    }
    replacement = ActiveModelPointerDTO(
        pointer_id=_digest(_canonical_json(pointer_payload)),
        revision=sequence_number,
        active_promoted_model_id=next_promoted_model_id,
        last_transition_id=transition.transition_id,
    )
    store.append_transition(transition)
    try:
        store.replace_active_pointer(replacement, expected_revision=pointer.revision)
    except Exception:
        # Store implementations should make transition+pointer atomic. The in-memory store
        # cannot remove through the protocol, so callers should discard a failed store.
        raise
    return transition, replacement


def activate_promoted_model(
    store: PerceptionModelLifecycleStore,
    promoted_model_id: str,
    *,
    actor: str,
    reason: str,
) -> tuple[ModelLifecycleTransitionDTO, ActiveModelPointerDTO]:
    """Activate the first registered model when no active model exists."""

    if store.get_active_pointer().active_promoted_model_id is not None:
        raise PerceptionModelLifecycleError("activation requires no active promoted model")
    return _transition(
        store,
        transition_kind="activate",
        next_promoted_model_id=promoted_model_id,
        actor=actor,
        reason=reason,
    )


def supersede_active_model(
    store: PerceptionModelLifecycleStore,
    promoted_model_id: str,
    *,
    actor: str,
    reason: str,
) -> tuple[ModelLifecycleTransitionDTO, ActiveModelPointerDTO]:
    """Replace the active model with another registered promoted model."""

    current = store.get_active_pointer().active_promoted_model_id
    if current is None:
        raise PerceptionModelLifecycleError("supersession requires an active promoted model")
    if current == promoted_model_id:
        raise PerceptionModelLifecycleError("cannot supersede active model with itself")
    return _transition(
        store,
        transition_kind="supersede",
        next_promoted_model_id=promoted_model_id,
        actor=actor,
        reason=reason,
    )


def rollback_active_model(
    store: PerceptionModelLifecycleStore,
    target_promoted_model_id: str,
    *,
    actor: str,
    reason: str,
) -> tuple[ModelLifecycleTransitionDTO, ActiveModelPointerDTO]:
    """Reactivate an earlier registered model while retaining complete history."""

    current = store.get_active_pointer().active_promoted_model_id
    if current is None:
        raise PerceptionModelLifecycleError("rollback requires an active promoted model")
    if current == target_promoted_model_id:
        raise PerceptionModelLifecycleError("rollback target is already active")
    transitions = store.list_transitions()
    target_prior_transition = next(
        (
            item.transition_id
            for item in reversed(transitions)
            if item.next_promoted_model_id == target_promoted_model_id
        ),
        None,
    )
    if target_prior_transition is None:
        raise PerceptionModelLifecycleError("rollback target was never previously active")
    return _transition(
        store,
        transition_kind="rollback",
        next_promoted_model_id=target_promoted_model_id,
        actor=actor,
        reason=reason,
        related_transition_id=target_prior_transition,
    )


def deactivate_active_model(
    store: PerceptionModelLifecycleStore,
    *,
    actor: str,
    reason: str,
) -> tuple[ModelLifecycleTransitionDTO, ActiveModelPointerDTO]:
    """Clear operational selection without deleting any promoted model artifact."""

    if store.get_active_pointer().active_promoted_model_id is None:
        raise PerceptionModelLifecycleError("deactivation requires an active promoted model")
    return _transition(
        store,
        transition_kind="deactivate",
        next_promoted_model_id=None,
        actor=actor,
        reason=reason,
    )


def resolve_active_promoted_model(
    store: PerceptionModelLifecycleStore,
) -> PromotedPerceptionModelDTO:
    """Resolve the active immutable promoted model from the revisioned pointer."""

    active_id = store.get_active_pointer().active_promoted_model_id
    if active_id is None:
        raise PerceptionModelLifecycleError("no promoted model is active")
    return store.get_ledger_entry(active_id).promoted_model


def build_model_lifecycle_snapshot(
    store: PerceptionModelLifecycleStore,
) -> ModelLifecycleSnapshotDTO:
    """Materialize a deterministic complete lifecycle snapshot."""

    pointer = store.get_active_pointer()
    entries = store.list_ledger_entries()
    transitions = store.list_transitions()
    payload: Mapping[str, object] = {
        "active_pointer_id": pointer.pointer_id,
        "ledger_entry_ids": [item.ledger_entry_id for item in entries],
        "transition_ids": [item.transition_id for item in transitions],
        "version": MODEL_LIFECYCLE_SNAPSHOT_VERSION,
    }
    return ModelLifecycleSnapshotDTO(
        snapshot_id=_digest(_canonical_json(payload)),
        active_pointer=pointer,
        ledger_entries=entries,
        transitions=transitions,
    )
