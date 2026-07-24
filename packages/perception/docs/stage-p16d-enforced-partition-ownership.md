# Stage P16D — Enforced partition ownership

P16D closes the caller-declared validation and test ownership gap identified by the external adversarial review.

The low-level comparison and inference functions remain reusable numerical primitives. Governed calibration, promotion, and final test evaluation now use content-addressed `DatasetPartitionDTO` evidence through `partitioned_governance.py`.

The governed chain is:

```text
P16B dataset manifest
    ↓
P16C DatasetPartitionDTO
    ↓
PartitionOwnedComparisonReportDTO
    ↓
validation-only calibration and promotion
    ↓
PartitionOwnedTestEvaluationReportDTO
```

The boundary rejects:

- comparison examples not owned by the supplied partition;
- split declarations that disagree with partition ownership;
- calibration or promotion from a test partition;
- final evaluation without a test partition;
- test action-schema mismatches;
- test interactions outside the supplied partition.

This stage does not yet repair P16 statistical evidence sufficiency, production label-estimand mismatch, rollback compatibility, or automatic recommendations. P17 remains paused.
