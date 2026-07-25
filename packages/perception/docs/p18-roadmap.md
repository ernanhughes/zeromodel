# Perception P18 — Visual Transition Testing Roadmap

P18 returns the perception package to its visual research spine after the P17 governance sequence.

## P18A — Observed transition evidence — complete

Materialize a deterministic fieldwise VPM describing what changed between an exact source observation and its exact result observation. Optional region annotations bind known object identities to the measurements without changing them.

## P18B — Annotated transition conformance — complete

Declare expected state changes for marked objects, relations, and control regions, then compare those declarations with immutable P18A measurements. Findings preserve confirmed, missing, unexpected, excessive, insufficient, directionally wrong, inconclusive, and unexplained changes rather than collapsing them into one boolean.

## P18C — Recurrent unexplained transition discovery — complete

Aggregate P18A evidence and P18B findings across an explicit discovery cohort. Preserve complete recurrence statistics for individual unexplained fields and exact co-occurrence signatures, then emit thresholded `candidate_unvalidated` missing-component hypotheses.

Observations without unexplained findings remain in the denominator. Repeated deterministic transition artifacts from distinct interactions remain valid evidence. Discovery never doubles as validation.

## P18D — Held-out candidate validation — complete

Derive explicit, content-addressed expectations from P18C candidates and their source statistics, then evaluate every candidate against a separately identified validation cohort. Discovery and validation cohort, interaction, transition, and observation identities remain disjoint.

Preserve validated, rejected, inconclusive, and insufficient-evidence outcomes. Failed candidates remain visible and cannot be silently discarded.

## P18E — Governed candidate promotion — complete

Convert only P18D-validated candidates into reviewable, content-addressed proposals. Bind complete discovery and validation lineage, require an explicit reviewer decision, and require reviewer-supplied semantic identity for approval.

Approved proposals remain `not_materialized`. P18E records authorization but cannot create annotations, relations, production expectations, or runtime changes.

## P18F — Reversible promotion materialization — current

Convert approved proposals from a fully reviewed P18E ledger into staged annotations or relations plus transition expectations. Require an explicit ontology directive per approval, bind the exact baseline identity, reject additive collisions, and preserve globally ordered forward operations with exact reverse-order inverse operations.

P18F change sets remain `staged_inactive` and `not_admitted`. A complete review without approvals produces an explicit `no_approved_changes` result rather than an empty implicit success.

## P18G — Materialization admission and atomic activation — next

Verify that the active model still matches the P18F baseline, audit every staged payload and operation pair, and apply all forward operations atomically or none. Persist the inverse operation plan and require a separate governed decision for rollback.

## Boundary

P18 measures, tests, validates, governs, and stages visual transition hypotheses. It does not infer semantic labels from pixels, claim that correlation is causation, or activate unadmitted changes in runtime behavior.
