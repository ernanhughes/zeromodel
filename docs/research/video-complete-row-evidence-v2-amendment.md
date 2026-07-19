# Video Complete Row Evidence v2 Amendment

- Evidence schema: `zeromodel-video-complete-row-evidence/v2`
- Scientific score-vector identity: `zeromodel-video-quantized-score-vector/v2`
- Raw diagnostic vector: `zeromodel-video-raw-score-diagnostic/v1`
- Canonical row ordering: `zeromodel-video-policy-row-order/v1`

## Scientific identity

Scientific identity binds:

- schema version
- policy artifact ID
- policy row-universe digest
- canonical ordered `(row_id, quantized_score)` pairs
- quantizer identity

Raw floating-point values are excluded from the scientific digest.

## Raw diagnostics

Raw diagnostics bind:

- raw diagnostic schema version
- canonical ordered `(row_id, raw_score)` pairs

Canonical float serialization uses IEEE-754 binary64 bytes after explicit `float64` conversion. NaN and infinity are rejected.

## Ranking identity

Ranking identity binds:

- ranking schema version
- canonical ranked row IDs
- quantized scores
- explicit tie groups

This amendment preserves ties with stable identity but does not yet repair semantic top-tie classification.

## Compatibility

V1 artifacts remain historical. V2 readers must reject or explicitly adapt v1. Compatibility is only valid when every required v2 identity field can be recomputed from trusted v1 content; otherwise the result is `insufficient_v1_identity`.
