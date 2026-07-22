# Post-split remediation — Stage 1A: artifact authority and navigation integrity closure

**Baseline:** the merged commit `3eaff43cadb54e2d2fb908aab8b7c0eaef5623b7` on `main`, delivering the Artifacts storage kernel, Trust kernel, and Navigation kernel (Stages 1–3, see [post-split-stage-2-trust-validation.md](post-split-stage-2-trust-validation.md) and [post-split-stage-3-navigation-validation.md](post-split-stage-3-navigation-validation.md)).
**Trigger:** an external review of that commit identified nine findings (six blockers, two high, one medium). This document records the six findings that were concrete, verifiable code defects and their fixes; a seventh finding (an "adapter-first Artifacts" capability) referenced design context this repository had no record of and is tracked separately as Stage B.
**Final state at time of writing:** uncommitted — every change below is staged in the working tree, pending an explicit commit decision.

## Findings addressed

Every finding below was independently re-verified against the actual merged code (not just the review's claims) before being fixed.

### 1. Package architecture had two conflicting authorities

`docs/architecture/package-system-1.0.13.md` described six distributions; `package-boundaries.toml` declared nine. Fixed by adding [package-system-next.md](../architecture/package-system-next.md) (current nine-package authority) and [adr-artifacts-trust-navigation.md](../architecture/adr-artifacts-trust-navigation.md) (decision record), while marking the six-package document as historical baseline in its own status line — its content is preserved, not rewritten.

### 2. Manifests were mutable metadata, not part of artifact identity

`InMemoryArtifactStore.put()`'s collision check only compared `canonical_bytes`; re-`put()`-ing the same bytes under a different manifest silently replaced it. Fixed in [store.py](../../packages/artifacts/src/zeromodel/artifacts/store.py): a new `ArtifactManifestConflictError` is raised when an existing record's manifest differs from the one being written; identical payload + identical manifest is idempotent (returns the existing ref, no-op). Tests: `test_store_put_is_idempotent_for_identical_payload_and_manifest`, `test_store_put_rejects_a_different_manifest_under_the_same_identity`.

### 3. Navigation resolved authoritative structure from the unbound manifest

`load_tile`/`load_leaf_binding` reconstructed DTOs from the store's `manifest` (informational metadata) rather than verifying against canonical bytes, and never checked the reconstructed id equaled the requested id — a manifest substitution at the storage layer could return a different tile/leaf binding than the one requested. Fixed in [storage.py](../../packages/navigation/src/zeromodel/navigation/storage.py): both functions now resolve canonical bytes, recompute the digest, require it to equal the requested id, and decode+reconstruct from the canonical JSON payload directly — the manifest is never read back. Tests: `test_load_integrity.py` (5 tests, including a simulated manifest-substitution attack and a misbehaving-resolver digest mismatch).

### 4. Hierarchy closure did not resolve the actual source artifacts

`validate_hierarchy` checked that leaf-binding records resolved and declared the right kind, but never checked that the leaf binding's *referenced source artifact* resolved through the store — test fixtures exposed this by never actually storing the source artifacts they referenced by digest. Fixed in [compiler.py](../../packages/navigation/src/zeromodel/navigation/compiler.py): closure now additionally verifies `store.has(binding.artifact_ref)` for every leaf, root reference kind, per-tile `corpus_id` consistency, declared-vs-actual depth consistency between parent and child tiles, declared-vs-actual `leaf_count` reconciliation at every level, and a final source-artifact-set digest reconciliation against `manifest.source_artifact_digest` (order-independent, catches silent addition/removal/substitution). Duplicate source artifacts are now rejected at compile time. Test fixtures (`conftest.py::make_source_artifacts`) were updated to actually register source artifacts in the store, and five new tests (`test_closure_over_corpus.py`) directly exercise the new checks (unstored source artifact, leaf-count mismatch, source-digest mismatch, duplicate source rejection, valid-path regression).

### 5. Traversal receipt identity did not cover the full receipt

`receipt_identity_payload` omitted `final_artifact_kind`/`final_artifact_id`, each step's rule descriptor, eligible children, tie candidates, and full failure detail — two materially different traversals reaching the same leaf id but binding a different artifact could share a `receipt_id`. Fixed in [dto.py](../../packages/navigation/src/zeromodel/navigation/dto.py): the payload now includes every distinguishing field; a shared `_pairs_to_dict` helper rejects duplicate keys in every tuple-to-dict conversion across the module (previously silent via `dict(pairs)`) rather than only in the receipt path. Tests: `test_receipt_identity_distinguishes_same_leaf_different_bound_artifact`, `test_duplicate_attribute_key_is_rejected_not_silently_collapsed`.

### 6. Trust permitted false issuer attribution

Signature verification selected the public key from `signature_envelope.signer_id` without requiring it to equal `authorization.issuer_signer_id` — a trusted signer could produce a validly-signed authorization whose signed content falsely named a different signer as issuer. Fixed in [verify.py](../../packages/trust/src/zeromodel/trust/verify.py): a new `SIGNER_ISSUER_MISMATCH` failure code fails closed whenever the envelope signer and the authorization's declared issuer diverge. Test: `test_signer_issuer_mismatch_is_rejected`.

### 7. TrustPolicy identity was arbitrary

`TrustPolicyDTO.policy_id` was a freely-chosen non-empty string, not bound to the policy's actual content, and had no duplicate/unknown-signer validation. Fixed in [dto.py](../../packages/trust/src/zeromodel/trust/dto.py): `policy_id` is now a self-validating content digest (same pattern as `ArtifactAuthorizationDTO.authorization_id`) via `trust_policy_identity_payload`/`compute_trust_policy_id`; construction now rejects duplicate trusted `signer_id`s, two signers sharing one `public_key_hex`, duplicate `rule_id`s, and rules referencing an unknown `signer_id`. Tests: 5 new cases in `test_dto_and_crypto.py`.

### 8. Freshness validation needed stronger boundaries

`epoch_valid` only compared against caller-supplied `minimum_epoch`, never the active `trust_policy`'s own epoch; timestamp parsing accepted timezone-naive values and malformed `evaluation_time` raised an uncaught exception rather than failing closed with a declared code. Fixed: a shared `parse_iso8601_utc` (in `zeromodel.trust.dto`) requires an explicit UTC offset and is used both by `ArtifactAuthorizationDTO.__post_init__` (which now also rejects `valid_from > valid_until`) and by `verify.py`'s `_check_freshness`; a malformed `evaluation_time` now produces `TrustFailureCode.MALFORMED_EVALUATION_TIME` and `decision="rejected"` instead of raising; `epoch_valid` now requires `authorization.policy_epoch >= minimum_epoch AND authorization.policy_epoch >= trust_policy.policy_epoch`. Tests: 6 new cases across `test_dto_and_crypto.py` and `test_verify_artifact_for_scope.py`.

## Test counts

| package | tests before Stage 1A | tests after |
|---|---|---|
| artifacts | 12 | 14 |
| trust | 31 | 42 |
| navigation | 26 | 33 (+ new `test_closure_over_corpus.py` with 5, `test_load_integrity.py` with 5) → 89 total incl. all navigation files |

Combined: 64 → 145 tests across the three new packages, all passing.

## Two refactors required to stay under the code-quality gate

Adding this many new checks pushed three functions over the repository's 100-line hard limit (`verify_artifact_for_scope`, `compile_hierarchy`, `traverse` initially; then `validate_hierarchy` and `verify_artifact_for_scope` again after the corpus-closure and freshness work). Each was factored into single-concern private helpers with no behavior change (full test suite re-verified green after every refactor) rather than granted a legacy-size exception — none of this code is legacy debt.

## Validation run this session

```
python -m pytest -q packages/artifacts/tests packages/trust/tests packages/navigation/tests   # 145 passed
python -m mypy packages/artifacts/src packages/trust/src packages/navigation/src               # Success: no issues found
python scripts/check_quality.py                                                                 # Quality checks passed (all 9 packages)
python scripts/validate_release_candidate.py                                                    # Release candidate validation passed; 261 public symbols across 9 distributions
python scripts/run_fast_tests.py                                                                 # 892 passed, 1 skipped, 0 failed, 92.77s / 120s budget
```

## Explicitly not run without further authorization

- Large hierarchy builds, traversal benchmarks, or any `@pytest.mark.slow` scenario — none added.
- Building/publishing to TestPyPI or PyPI.
- No git commit, push, or tag was made for this stage's changes.

## Next stage

Stage B — adapter-first Artifacts report compilation (`ReportAdapter`, `AdaptedReportDTO`, `compile_report()`, `CompiledReportArtifactDTO`, `load_compiled_report_artifact()`) — per the design brief this document's trigger review prompted. Tracked separately once complete. `zeromodel.search` remains planned, not started, and should not begin before Stage B closes.
