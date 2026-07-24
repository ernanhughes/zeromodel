# Stage P16C — Deterministic fitting and partition provenance

P16C repairs two provenance foundations identified by the external review.

## Canonical temporal fitting

`fit_temporal_translator` now sorts examples by `temporal_source_id` before validation,
feature extraction, target construction, matrix multiplication, fitting, and identity
construction. Duplicate temporal source identities are rejected. The translator records
`canonical_temporal_source_id_order_before_numerical_fit`, and its artifact version is
advanced to `/2` because fitted identities change for previously non-canonical inputs.

This removes caller-order floating-point variation. It does not claim bitwise equality
across different NumPy, BLAS, CPU, or operating-system implementations; cross-platform
coefficient canonicalization remains a separate question.

## Manifest-owned partition provenance

`DatasetPartitionDTO` is derived from an immutable P16B manifest and contains the exact:

- dataset and action-schema identities;
- split name;
- interaction identities;
- sequence identities;
- source-pixel digests.

A string such as `"validation"` or `"test"` is therefore no longer the only available
statement of ownership. The next provenance repair will require calibration and final test
evaluation to consume these partition identities rather than trusting caller declarations.

## Scope boundary

P16C does not yet alter the P10/P11 function signatures, promoted model compatibility,
health statistics, production labeling semantics, or lifecycle recommendations.
