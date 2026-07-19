# Video Action-Equivalence Protocol Audit v1

Version: `zeromodel-video-action-equivalence-audit/v1`

This contract freezes the protocol before any action-equivalence measurement.

## Scientific Questions

1. Does exact-row governance materially understate correct policy-action utility?
2. Can bounded row candidate sets produce an unambiguous correct action without establishing exact row identity?
3. Can a statistically defensible conformal row set be constructed from existing frozen evidence?
4. Does composition with a compiled reachability tile reduce ambiguity without injecting unsupported rows or carrying stale state?

## Required Distinctions

Exact row identity: which exact policy row is visually supported?

Policy-action utility: do all supported rows map through the exact bound policy artifact to the same action?

Invalid-input rejection: should this observation be considered a valid member of the policy universe at all?

Reachability composition: which currently supported rows remain possible after applying the previous belief set and the action actually executed?

## Core Row-Set Action Image

For visual row set `V`:

`ActionImage(V) = { policy_action(row) : row in V }`

Outcome definitions:

- `V empty` -> `no_sufficient_evidence`
- `|ActionImage(V)| = 1` -> `action_unanimous_candidate_set`
- `|ActionImage(V)| > 1` -> `action_ambiguous_candidate_set`

An action-unanimous set must retain unresolved row identities and must never be serialized as exact row acceptance.

## Invalid Negative Rule

A distinguishable negative observation has no valid policy row. Any nonempty action-unanimous set on a distinguishable negative is `invalid_input_action_support`. It is not automatically an exact-row false acceptance.

## Materiality

Material absolute coverage difference is frozen at `5 percentage points`.

A frame-local candidate-set method materially recovers action utility only when correct action-unanimous coverage gain is at least `5 percentage points` and wrong action-unanimous count is `0`.

Invalid-input support remains a separate binding dimension.

## Conformal Boundary

- Standard split-conformal row-set coverage is marginal.
- It does not automatically prove conditional wrong-action risk among accepted actions.
- Correlated frames may not be treated as independent calibration units.
- Unsupported alpha values must not be run.
- Invalid-input rejection is separate from valid-row-set coverage.
- Learn-then-Test is not implemented in this audit.

## Reachability Invariants

`B_t = retained row belief after frame t`

`a_t = action actually executed`

`V_(t+1) = next visual row set`

`R_(t+1) = union over row in B_t of T(row, a_t)`

`B_(t+1) = V_(t+1) intersection R_(t+1)`

Require `B_(t+1) subset of V_(t+1)`.

Reachability may remove rows. It may never inject a row absent from current visual evidence. Unknown action, excessive gap, or empty intersection must reset belief.

## Historical-Data Boundary

Use only committed score evidence, committed observation pixels, exactly reproducible historical providers, already-unblinded historical results, and ordered historical clips or episodes.

Do not generate new observations, capture camera frames, retrain a model, alter a provider, access the untouched Stage 3 v3 final split, or run PR #42 grids.

## Versions

- audit version: `zeromodel-video-action-equivalence-audit/v1`
- action-set decision: `zeromodel-video-action-set-decision/v1`
- evidence inventory: `zeromodel-video-retrospective-evidence-inventory/v1`
- conformal protocol: `zeromodel-video-row-set-conformal/v1`
- reachability tile: `zeromodel-video-policy-reachability-tile/v1`
- reachability replay: `zeromodel-video-reachability-replay/v1`
- composition trace: `zeromodel-video-tile-composition-trace/v1`

Output directories:

- `docs/results/video-policy-action-equivalence-audit-v1/`
- `docs/results/video-policy-reachability-tile-v1/`

The reachability-tile directory must not be created in this block.
