# Visual Sign Reader Hardening Final Assessment

## 1. Executive conclusion

Disposition: experimental observation adapter / supporting infrastructure. The strongest supported claim is exact closed-world visual codebook addressing for the committed arcade fixture with identity-bound policy lookup and explicit rejection traces. The perturbation matrix refutes any natural-image or general robustness wording.

## 2. Starting repository state

Started from latest `main` at `a45a62dcf4412009bab8b1401330f7708111aa66` and created branch `analysis/visual-sign-reader-hardening`. Local editable packages resolved from `C:/Projects/zeromodel/packages/*`; versions reported `1.2.0`.

## 3. Simplest accurate description

The reader extracts deterministic pooled quantized integer features from a fixed-shape frame, finds the nearest row in a complete visual index, rejects by distance/margin, and only then delegates to `VPMPolicyLookup`.

## 4. Architecture and package ownership

`zeromodel.vision` owns feature extraction, visual-index construction, calibration, lookup and provider adapter. `zeromodel.video` owns arcade rendering. `zeromodel.core` owns VPM artifacts and policy lookup. The split is coherent.

## 5. Current implementation strengths

The visual index is separate from the policy artifact and bound to the exact policy identity through metadata and provenance. Complete row coverage and duplicate feature vectors are rejected. Rejected visual-policy decisions cannot carry executable policy evidence.

## 6. Reproduced original evidence

Canonical rows/actions: `112/112` and `112/112`. Exhaustive trajectories: `2401/2401` waves cleared and `31213` visual decisions matched symbolic policy actions.

## 7. Defects and weaknesses found

Structured corruptions can land on another valid feature codeword. Cooldown corruption accepted 112/112 frames with 14 action errors. Removed-target corruption accepted 112/112 frames with 91 action errors. This is not fixable by calibration alone because some corruptions are exact feature aliases under the current representation.

## 8. Production hardening changes

Added `VISUAL_FEATURE_IMPLEMENTATION` to the feature spec and digest; stored canonical input digests in visual-index metadata; validated input-digest coverage and shape on restoration; added `nearest_input_digest` and `canonical_input_match` to `VisualDecision`; added tests for noncanonical exact feature aliases and malformed input digest metadata.

## 9. Perturbation results

Small global brightness changes accept only 7/112 canonical rows. One-pixel, translation, crop, resize, block corruption, salt noise and tank corruption are rejected under the current global threshold. Pepper noise accepts 96/112 correctly. Some semantic corruptions are accepted as other states, which is the strongest negative result.

## 10. Calibration comparison

The existing global rule uses min separation 2.0, threshold 0.5 and required margin 1.5. Local/ration rules were measured but not adopted; they add complexity and do not resolve exact codeword aliasing without accepting additional false positives.

## 11. Conventional baseline comparison

Exact feature-byte dictionary, plain nearest-neighbour, and direct symbolic state all recover 112/112 canonical rows/actions. ZeroModel adds policy/index artifact identity, mismatch rejection, bundle restoration, calibration/provenance metadata, and JSON-safe accepted/rejected traces.

## 12. Performance and size measurements

Mean Python lookup time over 11,200 canonical reads: `0.000256351` seconds. Visual index bundle size: `11679` bytes.

## 13. Identity, provenance and trace analysis

Hardened visual index ID: `cce9cf330743a31ef2876ae05955e7e8124678f023f258a14fdc0b2fc524f9b0`. Policy ID: `eb7523f406b45ac30b478fe9528db8f89a548693b0add2fc8d3e51c4badd857e`. Wrong policy identity rejected: `True`. Duplicate feature rejected: `True`.

## 14. Utility assessment

Useful when a closed visual fixture must replace symbolic row addressing while preserving reviewable identity and traces. Not useful as a general visual recognizer.

## 15. Claims supported

Exact closed-world canonical addressing; policy/index identity binding; explicit rejection for malformed input; bundle reload parity; deterministic trace generation.

## 16. Claims refuted or unsupported

Natural-image robustness, open-world seeing, and broad perturbation tolerance are unsupported. Some structured corruptions are false accepts relative to intended source state.

## 17. Remaining risks

Feature aliasing remains possible by design under coarse pooling/quantization. The system has no semantic detector for a corrupted frame that is indistinguishable from another valid codeword.

## 18. Recommended disposition

Experimental observation adapter / supporting infrastructure.

## 19. Next capability to inspect

The provider-neutral visual address contract and learned/local visual-address benchmark path, because that is where non-canonical robustness claims would have to be earned.

## 20. Complete command and artifact index

See `commands.jsonl`, `manifest.json`, `test-summary.json`, `perturbation-matrix.json`, `baseline-comparison.json`, and `identity-and-lineage-results.json` in this directory.
