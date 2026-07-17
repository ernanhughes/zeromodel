# ADR: Visual representation identity and policy separation

- **Status:** Accepted for the contract-first visual-address implementation
- **Date:** 2026-07-17
- **Scope:** Visual observations, dense representations, policy binding, and replay

## Context

ZeroModel already supports a deterministic `VisualSignReader` for a closed,
enumerable arcade world. Its visual index is a `VPMArtifact` because the current
integer feature matrix was introduced before the distinction between a policy
map and a general representation tensor was made explicit.

The next research stage may use frozen visual embeddings, multiple prototypes
per policy row, patch banks, projections, or conventional classifiers. Those
values are not human-meaningful policy metrics:

- an embedding dimension is not an action or evidence column;
- per-column VPM normalization has no semantic meaning for a learned vector;
- multiple prototypes must be able to address one policy row without inventing
  fake policy rows;
- perception and policy have different change and calibration lifecycles.

## Decision

### 1. Keep the visual address and policy separate

A visual provider returns an accepted policy row or a first-class rejection.
Only `VPMPolicyLookup` selects the action.

```text
observation
    ↓
VisualAddressProvider
    ↓
VisualAddressDecision
    ↓ accepted row id
VPMPolicyLookup
    ↓
action + policy evidence
```

The provider declares an identity-bearing `VisualAddressContract` before any
runtime decision is trusted. `score_semantics` is explicit so distance and
similarity providers cannot silently reverse score polarity.

### 2. Store dense representations as `MatrixBlob`

`MatrixBlob` is a content-addressed tensor contract, not a VPM. It records:

- canonical dtype;
- canonical big-endian bytes for multi-byte values;
- shape;
- optional scale and zero point;
- JSON-safe metadata;
- a digest over all identity-bearing content.

A thin `VisualAddressManifest` maps each matrix row to a stable prototype ID and
an exact policy row. Several prototypes may address the same policy row.

The current deterministic visual index remains supported for compatibility. New
embedding or patch implementations must use `MatrixBlob` plus a manifest rather
than adding hundreds of opaque embedding columns to a `ScoreTable`.

### 3. Pin the deployed pair

`DeploymentBinding` records the exact address, calibration, policy, encoder
manifest when present, source scope, and validation status approved together.
It is an identity record, not a cryptographic signature.

A research binding is rejected by the validated runtime unless the caller
explicitly enables research execution.

### 4. Declare replay strength

A visual-address contract declares one replay level:

- `exact_bytes` — canonical representation bytes and decision are expected to
  replay exactly;
- `exact_decision` — representation bytes may differ, but the accepted row or
  rejection must be exact;
- `tolerance_equivalent` — equivalence is defined by a separately validated
  numerical tolerance.

The implemented deterministic adapter uses `exact_bytes`.

A future learned encoder must not claim exact-byte replay merely because its
floating-point output was hashed. Encoder graph, weights, preprocessing,
runtime, canonicalization, and cross-runtime tests determine the valid replay
contract.

### 5. Do not choose an encoder in this ADR

DINOv2, DINOv3, ONNX Runtime, and other encoders remain benchmark candidates.
No model dependency, download, licence assumption, quantization scheme, or
runtime is added by the contract-first PR.

## Consequences

### Positive

- policy identity remains stable while visual perception is recalibrated;
- dense tensor storage no longer abuses VPM metric semantics;
- deterministic, embedding, classifier, and future temporal providers share one
  governed seam;
- runtime pairing is auditable before inference;
- learned representation determinism is stated honestly rather than inferred
  from a digest.

### Costs

- deployment requires several linked records rather than one overloaded
  artifact;
- `MatrixBlob` needs a persistence or bundle integration decision in a later PR;
- a future embedding implementation must define an encoder manifest and
  independent calibration report.

## Explicit non-decisions

This ADR does not approve:

- a frozen embedding implementation;
- policy-aware projection or metric learning;
- patch-based critical-evidence claims;
- automatic mode selection;
- video or multisensor observations;
- a safety-critical physical controller;
- int8 as the universal learned-embedding identity format.
