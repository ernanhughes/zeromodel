# zeromodel-observer

`zeromodel-observer` is the bounded Observer demonstration package for
canonical perceptual artifacts, verified transitions, and auditable policy
replacement.

## Scope

This package does not claim a general Observer architecture, habit formation,
continual learning, universal state similarity, or open-world perception. Its
initial contract is narrower:

```text
policy artifact
  + predicted transition
  + observed transition
  + declared comparison recipe
  + contradiction artifact
  + replacement lineage
```

The supported claim is that a small perceptual action policy can record and
replay why a local transition mismatch produced a replacement artifact.

## Package Boundary

Observer is a consumer package. It depends on `core`, `observation`, and
`perception`; it must not widen the `zeromodel.core` artifact kernel.

## Public API

The top-level public surface is:

- `ObserverObservationArtifactDTO`
- `ObserverComparisonRecipeDTO`
- `ObserverComparisonResultDTO`
- `ObserverTransitionRecordDTO`
- `ObserverContradictionArtifactDTO`
- `ObserverReplacementPolicyArtifactDTO`
- `compare_observer_transition`
- `build_transition_record`
- `build_contradiction_artifact`
- `build_replacement_policy_artifact`

## Design Position

The package is for the first experiment described by the Observer design note:
detect one bounded transition contradiction, identify the affected policy
region, and activate a replacement policy with deterministic lineage. Broader
habit promotion, schema revision, and comparative System A-D evaluation remain
future work.
