# ZeroModel 1.0.12 video policy reader: implementation boundary

**Status:** implementation slice, not a measured Stage 2 outcome  
**Target release:** `1.0.12`  
**Package version during this work:** remains `1.0.11`

## Inherited state

- Current main at the start of this work: `5860d5d7fb0d82c5896c93772f14efeaafd2c347`.
- The most recent recorded clean full-suite result is `209 passed, 1 skipped` at the July 18 visual programme status cut. The documentation-only commits after that cut do not establish a new test run.
- The deterministic policy core, exact visual codeword reader, provider-neutral visual-address seam, deployment binding, and evidence contracts are the stable downstream foundation.
- Frozen approximate systems already measured are Phase 1 B/C/D/G, System B v2, and registered local baseline R1.
- Registration improved raw exact-row ranking from `75%` to `87.5%` and repaired the declared held-out two-pixel translation family at raw top-1, but R1 still accepted `0 / 1,344` benign final observations at the selected zero-observed-distinguishable-false-accept point.
- Global DINOv2 CLS retrieval and the ridge probe are not promoted directions for this fixture.
- The fresh v3 local-evidence fixture and independent registered calibration are preparation-only until a verified evidence bundle is committed.

## Unresolved Stage 2 question

> Can translation-equivariant local evidence plus explicit temporal consistency produce greater governed accepted exact-row coverage than frame-local local evidence while preserving zero observed distinguishable false accepts and zero accepted conflicting-action errors on a fresh frozen final split?

This change does not answer that empirical question.

## Implemented boundary

This slice establishes the dependency-light exact temporal baseline required before approximate temporal inference:

1. immutable `VideoFrame` values with frame index, decode order, timestamp, source digest, frame digest, pixel digest, and owned pixels;
2. an identity-bearing `VideoClipManifest` over ordered lossless frame payloads;
3. `VideoFrameSource` plus `InMemoryVideoFrameSource` for deterministic frame streams;
4. a versioned `PolicyTransitionSpec` that distinguishes possible, impossible, and gap-unknown transitions;
5. `VideoPolicyReader`, which retains raw frame-local predictions, applies temporal gates before policy execution, forbids silent carry-forward, and delegates only an accepted exact row to `VPMPolicyLookup`;
6. complete JSON-safe frame and sequence traces;
7. an exact canonical arcade clip whose row and action sequences must reproduce the symbolic policy path;
8. tests for ownership, identity, ordering, timestamps, shape changes, legal and impossible transitions, explicit gap recovery, stale repeated pixels, and raw-evidence retention after rejection.

## Validation

The current implementation branch produced a clean local validation result of:

```text
233 passed, 1 skipped
```

Stage 1 required commands also passed locally:

```text
python -m pytest -q
python -m build
python -m twine check dist/*
python examples/arcade_visual_video_baseline.py
```

The canonical video baseline reproduced the full `22`-frame symbolic row and action path without rejection. The repository CI matrix for the inherited PR state had already passed on Python 3.10, 3.11, and 3.12 together with the Lua policy fixture, package build, Twine metadata check, claims-audit participation, and visual evidence-impact guard.

## Claim boundary

Valid wording after this slice passes CI:

> ZeroModel can consume an ordered lossless frame sequence, preserve frame and clip identity, apply a declared temporal transition contract, delegate only accepted exact rows to the existing deterministic policy, and reconstruct the complete canonical arcade sequence from explicit evidence.

This is not evidence that approximate video observations have useful governed coverage. It is not a fixed-camera result, real-camera validation, arbitrary motion invariance, semantic object understanding, or a completed Stage 2 benchmark.

## Next isolated slice

Implement and freeze the first deterministic local-correlation frame provider (V2), cache its local-region evidence, then compare it with the temporal path (V3) on the fresh video fixture. Do not tune on the final split and do not create a `1.0.12` version bump until the release definition of done is satisfied.
