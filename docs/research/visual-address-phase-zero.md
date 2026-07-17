# Governed visual addressing: Phase 0 contract

## Status

This phase establishes the implementation seam and evidence contracts for future
visual-address experiments. It does **not** implement a learned encoder.

The package version remains `1.0.11`.

## Goal

Permit different observation-address engines to identify the same independently
versioned VPM policy row while preserving:

- provider contract identity before inference;
- explicit score polarity;
- accepted or rejected address evidence;
- exact policy binding;
- separate perception and policy traces;
- research versus validated deployment status;
- held-out benchmark and governance evidence.

```text
ImageObservation
    ↓
VisualAddressProvider.contract()
    ↓
DeploymentBinding verification
    ↓
VisualAddressProvider.read()
    ↓
VisualAddressDecision
    ├── rejection
    └── matched policy row
            ↓
        VPMPolicyLookup
            ↓
        VisualPolicyDecision
```

## Implemented in Phase 0

### Provider-neutral contracts

- `ImageObservation` owns immutable `uint8` pixels and provides a raw digest.
- `VisualAddressContract` declares provider, observation, representation,
  address, calibration, policy, source-scope, score, and replay semantics.
- `VisualAddressDecision` contains candidate scores, ambiguity evidence,
  accepted checks, provider identities, and a canonical-JSON trace.
- `VisualAddressProvider` is the shared protocol.

### Existing deterministic reader as conformance proof

`DeterministicVisualAddressProvider` wraps `VisualSignReader` without changing
the existing visual-index or policy artifact identities. It maps the current
exact-codeword, distance-threshold, and absolute-gap evidence into the provider-
neutral decision contract.

`VisualPolicyReader` performs the policy lookup independently after a visual
address is accepted. A rejected address never produces an action.

### Dense representation storage contract

`MatrixBlob` gives dense vectors deterministic storage identity without treating
embedding dimensions as VPM policy metrics. `VisualAddressManifest` binds matrix
rows to prototype IDs and policy rows.

No embedding matrix is produced in Phase 0.

### Deployment pairing

`DeploymentBinding` pins one address/calibration/policy/source pairing. Research
bindings are blocked by default.

### Dataset and benchmark contracts

`VisualDatasetManifest` requires separate `prototype`, `calibration`, and `test`
splits. With family holdout enabled, one declared corruption family may not
appear in multiple splits. All three core splits must cover the same policy
rows. OOD records must not pretend to have a valid row or action.

The benchmark result contract records explicit denominators for false acceptance
and false rejection rather than publishing ambiguous percentages.

## Required Phase 1 systems

The benchmark schema reserves the following systems:

| ID | System | Question |
|---|---|---|
| A | Current deterministic reader | What does the committed baseline achieve? |
| B | Normalized template matching | Is any learned representation necessary? |
| C | Frozen embedding with medoid prototypes | Does the proposed address mechanism generalize? |
| D | Raw embedding k-NN | Do prototypes add value beyond stored examples? |
| G | Rejection-equipped linear probe | Does retrieval add value over classification? |
| H | Governance-parity wrapper | Does the full artifact chain add value over a disciplined model digest and log? |

Phase 0 supplies result slots and validation rules. It does not implement B, C,
D, G, or H.

## Dataset design for the first benchmark

The initial arcade experiment should contain three distinct evaluation classes.

### Benign held-out variation

Examples include:

- brightness and contrast;
- translation;
- unseen palette or sprite appearance;
- non-critical random or structured masks.

The desired result is action stability when sufficient policy evidence remains.

### Critical-evidence interventions

Examples remove or obscure the tank, target, cooldown indicator, or another
feature needed to distinguish conflicting actions.

The desired result is rejection. A strong average accuracy that accepts these
images is a failure.

### Out-of-domain observations

Examples include blank frames, impossible state combinations, severe crops, or
foreign game screens.

The desired result is rejection.

A random per-image split is insufficient. Entire variation families must be held
out so the test does not merely measure memorization of augmentation parameters.

## Required metrics

- exact-row accuracy;
- action accuracy;
- conflicting-action errors;
- acceptance and rejection counts;
- false-acceptance count and opportunity count;
- false-rejection count and opportunity count;
- results by held-out family;
- runtime, memory, and representation size in the implementation report;
- governance question fidelity and effort.

A future experiment should publish operating curves rather than selecting an
unsupported universal threshold in advance.

## Governance-parity audit

Accuracy alone cannot establish ZeroModel's distinctive value. System H should
be a conventional rejection-equipped model with:

- a model-file digest;
- preprocessing identity;
- an append-only JSON decision log.

Both the wrapper and the full ZeroModel chain should answer a predeclared audit
set:

1. Which exact perception and policy versions produced decision N?
2. What candidates, scores, thresholds, and rejection reason were recorded?
3. Can decision N be replayed under its declared replay contract?
4. What changed between deployment versions?
5. Which historical decisions would change under a replacement address or
   policy artifact?
6. Can accidental cross-pairing or tampering be detected?

If the lightweight wrapper reaches parity at materially lower complexity, the
visual architecture should be reduced to the shared seam, explicit policy
binding, and claims discipline.

## Explicit non-goals

Phase 0 does not add:

- PyTorch, ONNX Runtime, or an encoder download;
- frozen embeddings or prototype retrieval;
- automatic fitting or policy-aware projection;
- NetVLAD or patch verification;
- a claim that critical absence can be distinguished from hidden presence;
- video or multisensor observation types;
- `--mode auto`;
- bridge or other physical-control execution;
- a public claim of tolerant or learned visual recognition.

## Next PR gate

A frozen embedding PR should begin only after this contract is accepted. It must:

- use an optional dependency group;
- declare an encoder manifest and licence/source record;
- store vectors in `MatrixBlob`;
- keep prototype construction, calibration, and untouched testing separate;
- implement at least B, C, D, and G for comparison;
- leave policy-aware fitting, patch safety, video, and automatic mode selection
  out of scope;
- update the claims audit only with measured fixture-scoped results.
