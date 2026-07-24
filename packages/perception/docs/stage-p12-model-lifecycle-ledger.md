# Stage P12 — Promoted Model Lifecycle Ledger

P12 separates immutable promoted-model artifacts from operational selection.

Promoted models are registered in an append-only ledger. Activation, supersession, rollback, and deactivation are retained as ordered lifecycle transitions. A revisioned active-model pointer identifies the currently selected model without mutating or deleting any historical model artifact.

Rollback may only target a registered model that was previously active. Lifecycle snapshots bind the complete model ledger, transition history, and active pointer into one deterministic identity.

The in-memory store implements the DTO-only persistence protocol and optimistic revision checks. Durable SQL persistence, transactional database locking, production observation capture, drift monitoring, and automated rollback policy remain later stages.
