# Video Action-Set Finalization Authority

This implementation uses a separate finalization authority (database model B).
It does not migrate or reinterpret a Stage 8 database.

## Database Boundary

The final CLI may initialize only a fresh empty SQLite database. Initialization
creates the current schema and one marker row with:

- authority ID `local-finalization-authority`;
- kind `separate-finalization-authority`;
- schema version `zeromodel-video-finalization-schema/v1`.

Every authoritative final-access operation verifies the marker, required tables,
event ordinal column, `observation.final_access_id`, and enabled SQLite foreign
keys. Stage 8 databases, unmarked databases, missing final tables, partially
upgraded schemas, missing observation ownership, and unknown versions fail before
authorization or reservation.

Historical Stage 8 evidence remains authoritative in its original database and
artifact directory. The final authorization's exact execution contract binds its
path, database SHA-256, evidence-manifest digest, and Stage 8 commit. Their
canonical aggregate digest is included in the final receipt. A future production
executor must explicitly copy the immutable benchmark identity, sealed final
plan, and final episode plans into the finalization database before observations
can be written there. Pre-final observations remain only in Stage 8; authorized
final observations live only in the finalization database and carry an access ID.

## Local Authority

Within the local trust scope, writable access to the finalization database plus
an approved authorization artifact, its approved protocol, and executor
registration constitutes execution authority. The executor is an internal API;
not registering one is an operational default, not a security boundary.

The service derives the evidence, evaluation, event-chain, artifact-manifest, and
receipt digests. It does not expose a completion method that accepts caller-chosen
measurements or digests. Canonical evidence must match the authorization's exact
sealed episode IDs, row/episode/frame/provider counts, and provider order; each
frame/provider identity pair must be unique.

## Publication Order

Artifacts are generated in an authorization-specific directory beside the
canonical output, so staging and promotion share a volume. The service rejects
pre-existing staging/output paths, symlinks, traversal identifiers, duplicate or
unexpected names, missing files, and manifest/count/provider/digest mismatches.

After staged validation, one same-volume directory rename promotes the artifact
set. The service then commits completion, reconstructs the completed ledger,
records receipt-publication boundaries, revalidates all receipt bindings, and
atomically publishes `final-execution-receipt.json` last. Canonical artifacts or
a completed ledger without that receipt are not successful finalization.

## Preflight

`--preflight-only` intentionally constructs the normal in-memory runtime. It
validates authorization/protocol structure, approval and digest bindings, request
paths, sealed-plan identity, and unattended policy without opening the SQLite
authority, creating its schema, reserving access, or touching final evidence.
