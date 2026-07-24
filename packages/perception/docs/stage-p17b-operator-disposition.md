# Stage P17B — Operator Recommendation Disposition

## Objective

Turn a non-mutating P17 recommendation into an explicit, immutable operator decision while preserving a hard boundary between advice and lifecycle mutation.

## Contract

`OperationalRecommendationDispositionDTO` records:

- the exact recommendation reviewed;
- the active pointer identity and revision seen by the reviewer;
- the active and selected promoted-model identities;
- the selected compatibility assessment;
- approval or rejection;
- reviewer identity;
- operator reason.

Creating a disposition never changes lifecycle state.

## Governed execution

`execute_approved_rollback` accepts only an approved `rollback_candidate` disposition and revalidates:

1. disposition and recommendation identity;
2. active pointer identity;
3. active pointer revision;
4. active promoted model;
5. current compatibility contract;
6. selected target contract;
7. selected compatible assessment.

Only after these checks does it delegate to the existing P16F `rollback_compatible_model` operation.

## Stale-decision rule

An approval is not a standing authorization. Any pointer revision or model change after review makes the approval stale and execution fails without mutation.

## Deliberate boundary

P17B does not add automatic execution, scheduled execution, multi-party approval, or persistence for dispositions. Those require separate stages and explicit threat-model review.
