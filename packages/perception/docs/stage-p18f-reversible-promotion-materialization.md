# Stage P18F — Reversible Promotion Materialization

## Objective

Convert approved proposals from a fully reviewed P18E ledger into a versioned, reversible, inactive annotation/relation and transition-expectation change set.

```text
P18E proposal set
    + complete review ledger
    + explicit materialization directives
    + exact field schema
    + baseline identity snapshot
    ↓
PromotionMaterializationChangeSetDTO
    ├── staged annotations or relations
    ├── staged transition expectations
    ├── ordered forward operations
    └── reverse-ordered inverse operations
```

P18F materializes data contracts. It does not admit, persist into an active model, or execute them.

## Explicit ontology directives

A P18C co-occurrence signature is not automatically a relation, and a single recurrent field is not automatically an object. Each approved proposal therefore requires one `PromotionMaterializationDirectiveDTO` declaring either:

- `region_annotation`; or
- `relation_annotation`.

A region directive may provide additional annotation properties. Reserved provenance and semantic-property keys cannot be overridden.

A relation directive must name at least two existing baseline annotation identities. Its recurrent candidate fields become explicit `derived_field_ids`; they are not interpreted as relation members.

The directive set must exactly cover approved proposals—no missing directives and no directives for rejected, deferred, or semantic-annotation-required proposals.

## Baseline snapshot

`PromotionMaterializationBaselineDTO` records:

- baseline model-version identity;
- field-schema identity;
- existing annotation identities;
- existing relation identities;
- existing transition-expectation identities.

The baseline is content-addressed. P18F rejects generated identities that already exist in the baseline or collide with another generated change. This matters for reversibility: a rollback may remove only objects introduced by its own change set.

Relation directives may reference only annotation identities present in this baseline.

## Staged objects

For a region target, P18F creates:

- `PerceptionRegionAnnotationDTO` using the reviewer-supplied semantic name and role;
- a `semantic_type` property from the reviewer decision;
- proposal and decision provenance properties;
- `provenance_ref` bound to the approval decision;
- `TransitionExpectationDTO` targeting the new annotation.

For a relation target, P18F creates:

- `RelationAnnotationDTO` using the reviewer-supplied semantic type;
- explicit baseline member annotations;
- candidate fields as derived fields;
- reviewer-supplied semantic name as the relation value;
- `TransitionExpectationDTO` targeting the new relation.

Expectation direction and minimum magnitude thresholds are copied exactly from the P18E proposal, which already preserves the held-out P18D expectation.

## Forward and inverse operations

Every materialized approval produces two forward operations:

1. add the annotation or relation;
2. add its transition expectation.

It also produces two exact inverse operations:

1. remove the transition expectation;
2. remove the annotation or relation.

Each forward/inverse pair shares a content-addressed `pair_id` over:

- object kind;
- object identity;
- exact payload digest;
- proposal identity;
- decision identity.

Across multiple changes, forward operations are ordered from first target to final expectation. Inverse operations reverse the entire sequence, removing dependent expectations before their targets.

## Change-set states

A complete review with approvals produces:

- `status = staged_inactive`;
- `activation_status = not_admitted`.

A complete review with no approvals produces a content-addressed `no_approved_changes` change set with no operations.

Partial reviews are rejected. An approval cannot be materialized twice through an already-materialized proposal or decision contract.

## Integrity boundary

P18F rejects:

- reviews for another proposal set;
- incomplete review ledgers;
- schema or baseline mismatches;
- inexact directive coverage;
- implicit relation members;
- relation members outside the baseline;
- candidate fields outside the schema;
- baseline or within-change-set identity collisions;
- semantic payloads that disagree with the approved decision;
- annotation or relation identity tampering;
- transition expectations targeting the wrong materialized object;
- incomplete, duplicated, reordered, or non-paired forward/inverse operations;
- baseline, directive, operation, change, or change-set identity tampering.

## Non-claims

A staged P18F change set does not mean that:

- the annotation or relation is active;
- the expectation participates in runtime decisions;
- the current baseline still matches when execution is attempted;
- the semantic interpretation is universally correct;
- the candidate caused the observed transition.

It means only that a fully reviewed approval has been converted into an exact, reversible and inactive change proposal.

## Next stage

P18G should implement materialization admission and atomic activation. Admission must verify that the active baseline still equals `baseline_id` and `baseline_version_id`, audit every payload and operation pair, then either apply all forward operations atomically or apply none. It should also persist the inverse operation plan for governed rollback.
