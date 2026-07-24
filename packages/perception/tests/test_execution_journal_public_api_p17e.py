from __future__ import annotations

import zeromodel.perception as perception


def test_p17e_execution_governance_is_public() -> None:
    expected = {
        "GovernanceExecutionReceiptDTO",
        "GovernedExecutionAttemptDTO",
        "GovernedExecutionAttemptEventDTO",
        "PerceptionExecutionJournalError",
        "PerceptionGovernedExecutionError",
        "SqliteGovernedExecutionAttemptStore",
        "SqlitePerceptionGovernanceLedgerStore",
        "build_governed_execution_attempt",
        "execute_journaled_approved_rollback",
        "execute_or_reconcile_approved_rollback",
    }

    assert perception.PERCEPTION_STAGE == "P17E"
    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
