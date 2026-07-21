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
authority ID, absolute database path and SHA-256, absolute canonical
evidence-manifest path and digest, and Stage 8 commit. Verification rejects
missing/non-regular files and symlink or junction components, streams the complete
database through SHA-256, reconstructs the exact-key canonical manifest, and
requires the manifest's authority ID, database path/hash, and commit to agree in
both directions. The computed database hash, computed manifest digest, declared
authority ID, and their verified aggregate digest are included in the receipt.

This verification runs during read-only preflight, immediately before
reservation, and immediately before terminal completion. A mismatch before
reservation leaves authorization unconsumed. A mismatch after reservation marks
the access failed, preserves consumption, and publishes no receipt. A future
production executor must explicitly copy the immutable benchmark identity,
sealed final plan, and final episode plans into the finalization database before
observations can be written there. Pre-final observations remain only in Stage 8;
authorized final observations live only in the finalization database and carry
an access ID.

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
unexpected names, missing files, Windows device names (including extension and
case variants), and manifest/count/provider/digest mismatches.

After staged validation, one same-volume directory rename promotes the artifact
set. The service then commits completion, reconstructs the completed ledger,
records receipt-publication boundaries, revalidates all receipt bindings, and
atomically publishes `final-execution-receipt.json` last. Canonical artifacts or
a completed ledger without that receipt are not successful finalization.
Reconstruction exposes the latter as `completed_receipt_missing`; invalid
receipts are distinct, and only `completed_receipt_valid` is publishable success.

## Preflight

`--preflight-only` intentionally constructs the normal in-memory runtime. It
validates authorization/protocol structure, approval and digest bindings, request
paths, sealed-plan identity, historical database and manifest bytes, and
unattended policy without opening the SQLite authority, creating its schema,
reserving access, or touching final evidence.

## Decimal Evaluation

Decision arithmetic uses an isolated precision-128 Decimal context with
`ROUND_HALF_EVEN`. Evidence and thresholds are parsed exactly; sums that would
round at that precision are rejected, while mean division uses the declared
rounding mode. Minimum/maximum comparisons remain exact, threshold equality is
inclusive, and canonical values use plain decimal notation without redundant
trailing zeros. The evaluator implementation identity is included in the
digest-bearing evaluation measurements. Ambient process Decimal settings are
never mutated or consulted.
