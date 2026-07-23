# Response to the external review of commit 12746ff

**Status:** Complete.
**Reviewed commit:** `12746ff0b7cbca8c21d0bd502732fe32c43ef020` (round-2 fix-forward on the compiled-report aggregate).
**Scope:** All three findings from the third external review (1 blocker, 1 high, 1 medium), fixed in full.

## Verdict recap

The review kept the commit ("Keep the commit and fix forward"), confirming the fabricated-visual attack is correctly closed *inside the aggregate-validation path*. It found one remaining blocker (the public rendering shortcut `load_compiled_report_vpm` still bypassed aggregate validation entirely) and two narrower hardening gaps (reference-kind proof and receipt self-validation). This document records how each was closed.

## Finding 1 (Blocker) — the public rendering shortcut bypassed aggregate validation

`load_compiled_report_vpm(compiled, *, resolver)` was documented as "the operation used to resolve and render a compiled report's VPM," but its body was just `return load_vpm_artifact(compiled.vpm_artifact_ref, resolver=resolver)` - a direct call to the low-level, digest-only Core loader. The fabricated-pixel VPM from the previous round's blocker fix would be correctly rejected by `load_compiled_report_aggregate(...)`, but this documented "safe" shortcut would still happily return it, since `load_vpm_artifact` only proves the VPM's own digest is self-consistent - exactly the property the deterministic-reconstruction check proved is *not* sufficient.

**Fix:** the function moved from `report_loading.py` to `aggregate.py` and its signature changed from `(compiled: CompiledReportArtifactDTO, *, resolver)` to `(*, ref: ArtifactRef, resolver: ArtifactResolver)`, matching `load_compiled_report_aggregate`'s own shape. Its body is now exactly `return load_compiled_report_aggregate(ref=ref, resolver=resolver).vpm_artifact` - every public path to a compiled report's `VPMArtifact` now runs the complete aggregate closure, including the deterministic-reconstruction check, before returning anything. `report_loading.py` no longer imports `load_vpm_artifact` at all (it was only used by the removed function). The one caller in `test_report_compilation.py` was updated to the new signature.

`core_artifact_persistence.load_vpm_artifact` remains public and unchanged - it is a legitimate low-level primitive (used internally by `aggregate.py` itself, and by any caller who genuinely has no compiled-report context to close an aggregate against), but it is no longer reachable under the name that documented itself as the safe rendering path.

**Regression test:** [test_public_render_shortcut_also_rejects_fabricated_pixels](../../packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py) reproduces the exact fabricated-pixel VPM from the previous round's blocker test, but calls `load_compiled_report_vpm` directly (not `load_compiled_report_aggregate`) and asserts it now raises.

## Finding 2 (High) — reference proof validated digests but not artifact kinds

`_check_resolved_objects_match_declared_refs` recomputed each object's canonical digest and compared it to the declared ref's `artifact_id`, but never checked the declared ref's `artifact_kind`. `CompiledReportArtifactDTO._validate_basic_shape` similarly left the five nested refs' kinds unconstrained. A ref with a correct `artifact_id` but a wrong `artifact_kind` (e.g. `"wrong-kind"`) would pass every existing check - a reference is the pair `(artifact_kind, artifact_id)`, and proving only the digest proves content, not the complete declared reference.

**Fix:** two layers, as the review requested ("Validate exact nested kinds in `CompiledReportArtifactDTO` and again at aggregate closure"):

- [compiled_artifact.py](../../packages/artifacts/src/zeromodel/artifacts/compiled_artifact.py): a new `_validate_nested_ref_kinds` (called from `_validate_basic_shape`) requires each of the five nested refs' `artifact_kind` to equal its expected constant (`ADAPTED_REPORT_ARTIFACT_KIND`, `REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND`, `SCORE_TABLE_ARTIFACT_KIND`, `LAYOUT_RECIPE_ARTIFACT_KIND`, `VPM_ARTIFACT_ARTIFACT_KIND`). This runs on every `CompiledReportArtifactDTO` construction, including reconstruction from decoded store bytes - a wrong-kind ref can no longer even be represented as a valid DTO.
- [aggregate.py](../../packages/artifacts/src/zeromodel/artifacts/aggregate.py): a new `_check_declared_ref_kinds`, called first in `_check_resolved_objects_match_declared_refs`, re-asserts the same five kind checks against `compiled_report`'s refs - defense in depth for the aggregate-closure path specifically, not merely trusting that every `CompiledReportArtifactDTO` a caller supplies to `validate_compiled_report_aggregate` was actually constructed through the normal path (which, per the fix above, now always enforces this anyway).

**Regression test:** [test_score_table_ref_with_wrong_kind_but_correct_digest_is_rejected_at_construction](../../packages/artifacts/tests/test_reference_kind_and_receipt_invariants.py) builds a `score_table_ref` with `artifact_kind="wrong-kind"` and the *genuinely correct* `artifact_id`, recomputes the compiled-report identity over it (so the record is otherwise fully self-consistent), and asserts `CompiledReportArtifactDTO` construction itself rejects it.

## Finding 3 (Medium) — closure receipts were hash-valid but not semantically self-validating

`CompiledReportClosureReceiptDTO.__post_init__` verified only that `receipt_id` matched the canonical payload of its own fields - it never required `checks` to be the exact expected closure-check names (in order), all `True`, or `failure_codes` to be empty. A receipt with `checks = (("compiled_report_valid", True),)` and `failure_codes = ("vpm_not_checked",)` would construct cleanly as long as `receipt_id` was computed over that (malformed) content, contradicting the class's documented promise that it represents an aggregate that passed *every* closure check.

**Fix:** `__post_init__` now requires `self.checks == tuple((name, True) for name in _CLOSURE_CHECK_NAMES)` (exact names, exact order, all `True`) and `self.failure_codes == ()`, in addition to the five nested-ref-kind checks from finding 2 (also applicable here, since the receipt itself carries all five refs) and the pre-existing digest check. `build_compiled_report_closure_receipt` already produced exactly this shape, so no behavior changed for the one legitimate constructor - only direct, malformed construction is now rejected.

**Regression tests:** [test_closure_receipt_rejects_incomplete_checks_despite_matching_digest](../../packages/artifacts/tests/test_reference_kind_and_receipt_invariants.py) constructs a receipt with one check and a lying failure code, computes a genuinely correct `receipt_id` over that content, and asserts construction still fails. [test_closure_receipt_requires_expected_check_names_in_order](../../packages/artifacts/tests/test_reference_kind_and_receipt_invariants.py) proves a complete, all-`True`, but *reordered* `checks` tuple is also rejected - not merely "the right count of `True` values."

## Validation performed

```
python -m pytest -q packages/artifacts/tests                      # 97 passed
python -m pytest -q packages/trust/tests                          # 52 passed
python -m pytest -q tests/test_public_api_manifest.py -k "not slow"  # 15 passed
python -m ruff format --check packages/artifacts packages/trust   # passed
python -m ruff check packages/artifacts packages/trust            # passed
python -m mypy packages/artifacts/src packages/trust/src          # Success: no issues found in 23 source files
python scripts/check_package_boundaries.py                        # passed: 142 production modules
python scripts/check_architecture.py                              # passed: 142 production modules inspected
python scripts/check_quality.py                                   # Quality checks passed
python scripts/run_fast_tests.py                                  # 985 passed, 1 skipped, 0 failed, 67.77s / 120s budget
```

No new hard-limit violations were introduced this round.

## Files added

- `packages/artifacts/tests/test_reference_kind_and_receipt_invariants.py`
- This document.

## Files modified

- `packages/artifacts/src/zeromodel/artifacts/report_loading.py` — `load_compiled_report_vpm` removed (moved to `aggregate.py`); unused `load_vpm_artifact` import removed.
- `packages/artifacts/src/zeromodel/artifacts/aggregate.py` — new public `load_compiled_report_vpm(*, ref, resolver)`; new `_require_ref_kind`/`_check_declared_ref_kinds` (used by both the ref-proof check and the receipt DTO); `CompiledReportClosureReceiptDTO.__post_init__` now enforces exact checks/failure_codes/ref-kinds.
- `packages/artifacts/src/zeromodel/artifacts/compiled_artifact.py` — new `_require_ref_kind`/`_validate_nested_ref_kinds`, called from `_validate_basic_shape`.
- `packages/artifacts/src/zeromodel/artifacts/__init__.py` — `load_compiled_report_vpm` now imported from `aggregate` instead of `report_loading`.
- `packages/artifacts/tests/test_report_compilation.py` — updated call site for the new `load_compiled_report_vpm` signature.
- `packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py` — new bypass-closure regression test.
- `docs/architecture/package-public-api-1.0.13.csv` — `load_compiled_report_vpm`'s `source_module` updated to `zeromodel.artifacts.aggregate`.

## Known limitations (unchanged)

The claims boundary recorded in [post-c203e7a7-aggregate-closure.md](post-c203e7a7-aggregate-closure.md) still applies; this response and [post-0e56558-review-response.md](post-0e56558-review-response.md) close gaps in *how thoroughly* that stage's own claim is proven and enforced across every public entry point, not expand the claim itself. `zeromodel.search` remains not started.
