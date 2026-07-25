# Stage P18E — Governed Candidate Promotion

## Objective

Turn P18D-validated transition candidates into immutable, reviewable promotion proposals without mutating production annotations, relations, or expectations.

```text
P18C discovery report
    +
P18D validation report
    ↓ validated results only
CandidatePromotionProposalSetDTO
    ↓ explicit reviewer decisions
CandidatePromotionReviewDTO
```

P18E is an authorization layer. It is not a deployment layer.

## Promotion proposals

`propose_validated_candidate_promotions(...)` creates proposals only for P18D results whose status is `validated`. Rejected, inconclusive, and insufficient-evidence results remain in the validation report but do not become proposals.

Each proposal binds:

- discovery report, candidate, statistic, cohort, and supporting evidence identities;
- validation report, result, expectation, cohort, observation, interaction, and transition identities;
- exact field identities and candidate kind;
- the held-out expectation thresholds;
- discovery recurrence and direction measurements;
- held-out confirmation counts and fraction.

Every proposal begins as `pending_review` and `not_materialized`.

## Decisions

A `CandidatePromotionDecisionDTO` records one reviewer decision for one proposal:

- `approved`;
- `rejected`;
- `deferred`;
- `needs_semantic_annotation`.

Approval requires an explicit semantic name and semantic type. Those values are supplied by the reviewer; P18E never invents them from spatial recurrence.

Non-approved decisions cannot carry semantic materialization fields. This prevents deferred or rejected hypotheses from quietly acquiring production meaning.

All decisions remain `not_materialized`, including approved decisions.

## Review ledger

`review_candidate_promotion_proposals(...)` produces an immutable review ledger with one of three coverage states:

- `pending_review` — no proposals have decisions;
- `partially_reviewed` — some proposals have decisions;
- `review_complete` — every proposal has exactly one decision.

The ledger partitions every proposal into:

- pending;
- approved;
- rejected;
- deferred;
- semantic annotation required.

It rejects duplicate decisions, multiple decisions for one proposal, and decisions belonging to another proposal set.

## Integrity boundary

Proposals, proposal sets, decisions, and reviews are content-addressed. P18E rejects:

- validation reports that do not reference the supplied discovery report;
- discovery or validation cohort/schema mismatches;
- validation-result expectations that disagree with discovery candidates;
- candidates that disagree with their source statistics;
- proposal lineage/count inconsistencies;
- missing semantic identity on approval;
- semantic fields on non-approved decisions;
- unknown, duplicate, or conflicting decisions;
- identity tampering.

## Non-claims

An approved promotion decision does not mean that:

- a production annotation was created;
- a relation was added;
- a production transition expectation was changed;
- the candidate has a universally correct semantic interpretation;
- the candidate caused the observed transition.

It means only that an identified reviewer approved the exact validated proposal and supplied the semantic description required for a later materialization stage.

## Next stage

P18F should materialize only approved, fully reviewed proposals into versioned annotation and expectation change sets. Materialization must remain reversible, preserve the P18C/P18D/P18E lineage, and produce no runtime activation until a separate admission step succeeds.
