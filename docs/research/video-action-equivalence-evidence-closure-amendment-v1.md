# Video Action-Equivalence Evidence Closure Amendment v1

Version: `zeromodel-video-retrospective-evidence-closure/v1`

This amendment freezes the evidence-closure rules for retrospective action-equivalence analysis. It separates historical evidence capabilities that must not be inferred from one another.

## Aggregate Metric Evidence

Committed summary metrics such as `top-1 row accuracy = 0.75` or `top-1 action accuracy = 0.96875` permit historical metric verification only.

Aggregate metric evidence does not permit:

- per-observation rescoring
- candidate-set construction
- conformal calibration
- reachability belief replay

## Per-Observation Top-1 Evidence

Per-observation top-1 evidence requires each observation to retain:

- observation ID
- expected row
- predicted top-1 row
- provider identity
- split identity
- policy artifact identity

This permits exact top-1 row-to-action rescoring.

## Ordered Ranking Evidence

Ordered ranking evidence requires every observation to retain:

- row IDs
- scores or distances
- tie semantics
- score orientation
- provider identity

This permits fixed top-k construction.

## Complete Score-Vector Evidence

Complete score-vector evidence requires a score or distance for every policy row for every observation.

This permits:

- score-gap sets
- split-conformal row sets
- alternative frozen set constructions

## Reproducible Score Evidence

Reproducible score evidence may substitute for stored score evidence only when all of these hold:

- exact historical source commit known
- exact provider version known
- exact benchmark version known
- exact observation bytes or immutable observation generator known
- exact split identity known
- exact command recorded
- temporary regeneration performed
- generated per-observation output compared deterministically

A provider is not reproducible merely because related code still exists.

## Sequence Metadata

Sequence metadata requires:

- clip or episode ID
- frame sequence number
- gap semantics
- actual executed action

Sequence metadata is necessary but not sufficient for reachability replay.

## Frame-Level Visual Belief Evidence

Frame-level visual belief evidence requires at least one current-frame visual row set for every replayed frame:

- visual candidate rows
- candidate-set construction identity
- provider identity
- observation identity

A rejection with no retained candidates is not a visual belief set.

A case-level sequence summary is not a visual belief set.

## Reachability Replay Closure

Reachability replay closure requires both:

- sequence metadata
- frame-level visual belief evidence

An artifact may support a historical metric, a per-observation re-score, a candidate-set analysis, or a temporal replay. These are different evidence capabilities and must never be inferred from one another.
