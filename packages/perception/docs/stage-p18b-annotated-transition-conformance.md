# Stage P18B — Annotated Transition Conformance

## Objective

Compare declared state-change expectations for marked objects, relations, and control regions with one immutable P18A transition artifact.

```text
P18A TransitionEvidenceVPMDTO
    +
PerceptionRegionAnnotationDTO / RelationAnnotationDTO
    +
TransitionExpectationDTO
    ↓
TransitionConformanceReportDTO
```

P18B does not recalculate or rewrite the P18A measurements. It resolves each declaration to exact field identities and evaluates the weighted field evidence already present in the transition artifact.

## Declarations

A `TransitionExpectationDTO` targets one or more annotation or relation identities and declares one of four behaviours:

- `stable` — change must remain within maximum tolerances;
- `change` — non-directional change must satisfy minimum and maximum tolerances;
- `increase` — change must satisfy the magnitude thresholds and have positive signed direction;
- `decrease` — change must satisfy the magnitude thresholds and have negative signed direction.

Thresholds cover:

- mean absolute change;
- changed-value fraction;
- signed change magnitude for directional expectations.

## Findings

P18B preserves distinct outcomes rather than reducing the test to one boolean:

- `confirmed`;
- `missing_expected_change`;
- `unexpected_change`;
- `excessive_change`;
- `insufficient_change`;
- `wrong_change_direction`;
- `inconclusive`;
- `unexplained_change`.

Report status is:

- `conformant` when all declarations are confirmed and no unexplained fields qualify;
- `attention_required` for inconclusive or unexplained evidence without a declared failure;
- `nonconformant` when a declared expectation fails.

## Relation fields

A relation expectation uses `derived_field_ids` when they are declared. Otherwise it resolves to the union of its member annotation fields. This keeps relation testing explicit and addressable without inventing a second spatial representation.

## Integrity boundary

P18B verifies that:

- supplied annotation identities exactly match the P18A artifact;
- each annotation's field bindings exactly match the bindings persisted by P18A;
- expectations and annotations use the same field schema as the transition;
- relation members and derived fields exist;
- identities, thresholds, findings, report status, and unexplained-evidence thresholds are content-addressed.

## Non-claims

A confirmed expectation means only that the declared visual transition matched the measured transition under the chosen thresholds. It does not prove that the marked object caused the action or result.

## Next stage

P18C aggregates P18A and P18B evidence across interactions to identify recurrent, unmarked transition fields and propose candidate missing components for testing. It must preserve recurrence and association as evidence without converting them into causal conclusions.
