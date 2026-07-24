# Stage P14 — Production Observation and Inference Ledger

P14 records every runtime inference as immutable operational evidence tied to the exact active promoted-model pointer revision.

Observed outcomes are appended later as separate immutable records. They do not rewrite the original prediction, margin, rejection status, model identity, pointer identity, or input identity.

Windowed reports derive coverage, mean margin, raw labeled accuracy, accepted-only labeled accuracy, model revisions, and evidence identities over an inclusive inference sequence range.

Rejection remains separate from correctness: a rejected inference can later receive an outcome, and unlabeled records remain part of coverage and margin metrics without being treated as correct or incorrect.

P14 provides a DTO-only store protocol and deterministic in-memory implementation. Durable SQL production-ledger persistence, feature-distribution drift, alert policy, automated rollback, retention policy, and deployment orchestration remain future stages.
