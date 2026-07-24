# Stage P13 — Durable SQL Lifecycle Persistence

P13 implements the P12 promoted-model lifecycle store boundary with restart-safe SQLite persistence.

The database contains an immutable promoted-model ledger, append-only ordered transitions, and one revisioned active-model pointer. Existing P12 lifecycle services operate unchanged against the SQL store.

Transition insertion and active-pointer replacement share one `BEGIN IMMEDIATE` transaction. The pointer update uses an expected revision predicate; a stale writer rolls back the transition instead of leaving history and operational state inconsistent.

The store uses canonical JSON for DTO reconstruction, validates a declared database schema version, enforces model and transition foreign keys, and preserves deterministic DTO identities across process restarts.

P13 does not add deployment orchestration, production observation capture, drift analysis, automatic rollback policy, multi-database support, or schema migration beyond rejecting an unknown schema version.
