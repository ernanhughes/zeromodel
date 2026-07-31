# Visual Transition Evidence Hardening - Architecture Inventory

Starting main SHA: `85d4fd50607cbef607ddbe4a5f73c1468ad76955`

## Production Core

- `zeromodel.perception.transition_evidence` owns exact before/after `SourceVPMDTO` fieldwise change evidence. Its transition identity includes ordered before and after source identities, pixel digests, field schema identity, threshold, field measurements, and PNG digest.
- `zeromodel.perception.transition_conformance` owns declared annotation/relation expectations and deterministic conformance reports. Existing statuses distinguish confirmed, missing expected change, unexpected change, excessive/insufficient change, wrong direction, unexplained change, and inconclusive.
- `zeromodel.vision.visual` owns Visual Sign Reader identity, including raw/canonical/feature digests, acceptance profile, and `policy_executed`.
- `zeromodel.video` remains domain/runtime code and does not own generic transition semantics.

## Hardened Composition

`zeromodel.perception.transition_analysis` adds the smallest reusable composition layer found necessary:

- `TransitionActionDeclarationDTO` gives declared actions canonical identity without claiming causality.
- `TransitionExpectationSetDTO` gives the exact expectation set a canonical identity and rejects duplicate/conflicting targets.
- `VisualTransitionReaderTraceDTO` preserves Visual Sign Reader evidence fields, including `evidence_only` and `policy_executed`.
- `VisualTransitionAnalysisDTO` embeds the exact ordered transition evidence object and binds it to action identity, expectation-set identity, and conformance report identity in one replayable artifact identity.

## Benchmark Ownership

Arcade and warehouse renderers, decoders, value contracts, baselines, privileged state, fault injectors, metrics, and reports remain under `examples/visual_transition_benchmark/`.
