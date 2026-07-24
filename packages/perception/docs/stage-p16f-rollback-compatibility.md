# Stage P16F — Rollback Compatibility Contracts

## Objective

Prevent lifecycle history alone from making a promoted model rollback-eligible.

## Contract

`ModelCompatibilityContractDTO` binds a promoted model to the exact runtime contract it can consume:

- model kind;
- action schema identity;
- source encoder identity;
- field schema identity;
- temporal window identity when applicable;
- inference semantics version;
- deployment slot.

`assess_rollback_compatibility` compares the active and target contracts field by field and emits an immutable assessment containing the exact mismatches.

`rollback_compatible_model` requires both:

1. the existing P12 historical rule that the target was previously active; and
2. an exact compatible runtime signature.

An incompatible target leaves the active pointer unchanged.

## Deliberate boundary

This stage does not automatically select a rollback candidate and does not mutate lifecycle state without an explicit operator call. P17 recommendations remain paused until the full end-to-end integration stage is complete.
