# Visual Sign Reader Contract Hardening Final Assessment

## 1. Executive conclusion

The Visual Sign Reader now exposes explicit acceptance profiles and separate raw, canonical and feature identities. Policy execution is no longer implied by visual-address evidence alone.

## 2. Starting main SHA

`68166bc3e8ace98e806bae98782ccaf5697123e6`

## 3. Prior hardening result

The previous pass established complete canonical arcade replay and showed that some corrupted observations collapse to valid feature codewords.

## 4. Simplest accurate capability description

ZeroModel can address an identified finite policy from a deterministic observation codebook under a declared acceptance profile.

## 5. Acceptance-profile model

`canonical_only` executes policy only on canonical input digest equality. `exact_codeword` executes policy on exact feature-codeword equality. `calibrated_nearest` preserves prior distance and margin semantics and remains the default. `evidence_only` returns address evidence without policy execution.

## 6. Decision-trace invariants

Executable decisions require a matched row, action, value, source/view coordinates and candidate policy scores. Rejected and evidence-only decisions carry no executable policy fields.

## 7. Raw, canonical and feature identity

The trace now records `raw_input_digest`, `canonical_input_digest` and `feature_digest`. The legacy `input_digest` remains as an alias for the canonical digest.

## 8. Golden feature-contract conformance

`packages/vision/tests/fixtures/visual_feature_contract_v1.json` pins v1 behavior with compact golden vectors.

## 9. Reference implementation parity

The test-only reference implementation uses explicit loops and matches production across golden vectors, seeded random frames and all 112 arcade canonical frames.

## 10. Restoration and compatibility

The compatibility policy is strict version rejection. Identity-relevant missing fields are not reconstructed silently.

## 11. Codeword alias analysis

`alias-analysis.json` records canonical and noncanonical variants with raw, canonical and feature digests, addressed rows, actions and profile results.

## 12. Production changes

Production changes are limited to `zeromodel.vision.visual`, the vision public exports and the visual-policy adapter metadata.

## 13. Performance impact

`performance-results.json` records median and p95 costs for extraction, digest generation, profile reads, serialization and visual-index metadata size on the local machine.

## 14. Supported deployment profiles

Canonical replay should use `canonical_only`. Closed finite codebook deployments can use `exact_codeword` or `calibrated_nearest` while monitoring canonical mismatches. Shadow evaluation can use `evidence_only`.

## 15. Unsupported deployment profiles

Open-world visual recognition remains unsupported by VisualSignReader alone.

## 16. Claims strengthened

Canonical observation verification, feature-codeword addressing and feature-contract conformance now have explicit runtime and test contracts.

## 17. Claims reduced or refuted

Exact feature equality is not canonical observation verification. Calibrated nearest addressing is not semantic visual understanding.

## 18. Remaining risks

Feature aliases remain possible by design when distinct observations canonicalize or quantize to the same codeword.

## 19. Recommended final disposition

Treat VisualSignReader as bounded deterministic visual addressing infrastructure, suitable for controlled replay and closed-codebook experiments with explicit acceptance profiles.

## 20. Readiness for transition-evidence work

Ready to proceed to visual transition evidence, provided transition work consumes the declared profile and preserves the raw/canonical/feature identity trace.
