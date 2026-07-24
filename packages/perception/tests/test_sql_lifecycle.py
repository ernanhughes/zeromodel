from __future__ import annotations

import sqlite3

import pytest

from zeromodel.perception import (
    PerceptionSqlLifecycleError,
    PromotedPerceptionModelDTO,
    SqlitePerceptionModelLifecycleStore,
    activate_promoted_model,
    build_model_lifecycle_snapshot,
    register_promoted_model,
    resolve_active_promoted_model,
    rollback_active_model,
    supersede_active_model,
)


def _model(name: str) -> PromotedPerceptionModelDTO:
    return PromotedPerceptionModelDTO(
        promoted_model_id=f"promoted:{name}",
        model_kind="single_frame",
        model_id=f"translator:{name}",
        rejection_threshold=0.25,
        calibration_id=f"calibration:{name}",
        promotion_decision_id=f"decision:{name}",
        validation_comparison_report_id=f"validation:{name}",
        training_split="train",
        evaluation_split="validation",
    )


def test_sql_store_survives_restart_and_resolves_active_model(tmp_path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    first = _model("first")
    with SqlitePerceptionModelLifecycleStore(database) as store:
        register_promoted_model(
            store,
            first,
            registered_by="test",
            registration_reason="initial candidate",
        )
        activate_promoted_model(
            store,
            first.promoted_model_id,
            actor="operator",
            reason="initial activation",
        )
        assert resolve_active_promoted_model(store) == first

    with SqlitePerceptionModelLifecycleStore(database) as reopened:
        assert resolve_active_promoted_model(reopened) == first
        assert reopened.get_active_pointer().revision == 1
        assert len(reopened.list_transitions()) == 1
        assert len(reopened.list_ledger_entries()) == 1


def test_sql_store_commits_supersession_and_rollback_atomically(tmp_path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    first = _model("first")
    second = _model("second")
    with SqlitePerceptionModelLifecycleStore(database) as store:
        for model in (first, second):
            register_promoted_model(
                store,
                model,
                registered_by="test",
                registration_reason="candidate",
            )
        activate_promoted_model(
            store,
            first.promoted_model_id,
            actor="operator",
            reason="activate first",
        )
        supersede_active_model(
            store,
            second.promoted_model_id,
            actor="operator",
            reason="promote second",
        )
        rollback_active_model(
            store,
            first.promoted_model_id,
            actor="operator",
            reason="restore first",
        )
        snapshot = build_model_lifecycle_snapshot(store)
        assert snapshot.active_pointer.revision == 3
        assert snapshot.active_pointer.active_promoted_model_id == first.promoted_model_id
        assert tuple(item.transition_kind for item in snapshot.transitions) == (
            "activate",
            "supersede",
            "rollback",
        )

    with SqlitePerceptionModelLifecycleStore(database) as reopened:
        assert resolve_active_promoted_model(reopened) == first
        assert reopened.get_active_pointer().revision == 3


def test_sql_store_rejects_conflicting_registration(tmp_path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    model = _model("same")
    with SqlitePerceptionModelLifecycleStore(database) as store:
        register_promoted_model(
            store,
            model,
            registered_by="test",
            registration_reason="first registration",
        )
        with pytest.raises(PerceptionSqlLifecycleError):
            register_promoted_model(
                store,
                model,
                registered_by="other",
                registration_reason="different registration content",
            )


def test_failed_pointer_update_rolls_back_transition(tmp_path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    model = _model("first")
    with SqlitePerceptionModelLifecycleStore(database) as store:
        register_promoted_model(
            store,
            model,
            registered_by="test",
            registration_reason="candidate",
        )
        pointer = store.get_active_pointer()
        transition, replacement = activate_promoted_model(
            store,
            model.promoted_model_id,
            actor="operator",
            reason="activate",
        )
        assert transition.sequence_number == 1
        assert replacement.revision == 1

        with pytest.raises(PerceptionSqlLifecycleError):
            store.replace_active_pointer(replacement, expected_revision=pointer.revision)

    with SqlitePerceptionModelLifecycleStore(database) as reopened:
        assert reopened.get_active_pointer().revision == 1
        assert len(reopened.list_transitions()) == 1


def test_schema_version_is_rejected_when_unknown(tmp_path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    with SqlitePerceptionModelLifecycleStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE perception_lifecycle_metadata SET value = 'future' WHERE key = 'schema_version'"
    )
    connection.commit()
    connection.close()

    with pytest.raises(PerceptionSqlLifecycleError):
        SqlitePerceptionModelLifecycleStore(database)
