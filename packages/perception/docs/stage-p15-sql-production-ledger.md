# Stage P15 — Durable Production Ledger Persistence

P15 implements the P14 production-ledger store contract with SQLite.

Production inference and outcome DTOs remain immutable and append-only. Their global sequence numbers, pointer revisions, promoted-model identities, acceptance status, observed outcomes, and correctness survive process restart without changing the service API.

The schema uses separate inference and outcome tables, an explicit schema-version record, foreign-key enforcement, WAL journaling, uniqueness constraints, and indexes for promoted-model and inclusive sequence windows.

`SqlitePerceptionProductionLedgerStore.list_inferences_in_window()` provides an indexed retrieval path while `build_production_metrics_report()` continues to operate against the unchanged DTO-only store protocol.

P15 does not add timestamps, retention or deletion policy, drift detection, alert thresholds, automatic lifecycle transitions, or deployment orchestration.
