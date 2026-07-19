# INVALID PROSPECTIVE INSTRUMENT — DO NOT USE FOR SCIENTIFIC CONCLUSIONS

The files in this directory are preserved historical scaffold, profiling, and verification artifacts. Several contain invalid or vacuous measurements. Their presence does not establish benchmark validity.

- `historical_stack_integrated`
- `reference_instrument_invalid`
- `optimized_path_unverified`
- `prospective_materialization_prohibited`

## Artifact Classification

- `runtime-comparison.json`: `invalid_measurement`
  Reason: compares cold-cache reference execution against warm delegated execution through the same implementation.
- `runtime-profile-optimized.json`: `invalid_measurement`
  Reason: optimized path delegates to the reference path.
- `runtime-profile-optimized.md`: `invalid_measurement`
  Reason: prose summary derived from invalid optimized timing claims.
- `provider-runtime-equivalence.json`: `vacuous_verification`
  Reason: compares delegated optimized calls to the same reference implementation.
- `provider-runtime-equivalence.csv`: `vacuous_verification`
  Reason: row-level outputs are derived from the same delegated implementation.
- `phase-access-audits.json`: `assigned_status_not_measurement`
  Reason: zero counters are assigned constants rather than derived from an access-event log.

## Current Boundary

No development, calibration, or architecture-selection materialization may proceed until the reference instrument is corrected and measured verification fails against known-bad mutations.
