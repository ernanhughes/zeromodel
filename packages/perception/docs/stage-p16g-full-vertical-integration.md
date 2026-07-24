# Stage P16G — Full Vertical Integration and Restart Validation

## Objective

Prove that the repaired perception architecture operates as one connected system rather than as isolated DTO tests.

## Exercised path

The P16G integration test executes this real artifact chain:

```text
encoded source VPMs
    -> recorded interactions
    -> sequence-owned dataset manifest
    -> train / validation / test partitions
    -> fitted single-frame translator
    -> fitted temporal translator
    -> validation comparison
    -> partition-owned promotion
    -> partition-owned final test evaluation
    -> SQLite lifecycle registration and activation
    -> promoted runtime inference
    -> SQLite production inference and outcome ledger
    -> SQLite restart
    -> operational reference and gated health diagnosis
    -> compatibility assessment
    -> compatible historical rollback
```

No calibration, promotion, test, production, health, or rollback identity is inserted independently of its upstream artifact.

## Required invariants

The test proves that:

- one dataset identity is preserved across all three partitions;
- validation and test evidence are manifest-owned;
- promoted model identity is derived from the validation comparison;
- the final test report belongs to the promoted model;
- lifecycle and production ledgers survive SQLite restart;
- production records preserve the active pointer revision;
- health reports preserve exact inference and outcome identities;
- rollback requires an exact compatibility assessment;
- temporal window identity survives fitting, promotion, testing, and runtime inference.

## Boundary

P16G is an integration proof, not a performance claim. It uses a deterministic synthetic action domain to exercise lineage and governance. It does not establish external benchmark quality or statistical generalization.

P17 recommendations remain paused until this test is green in the package CI workflow.