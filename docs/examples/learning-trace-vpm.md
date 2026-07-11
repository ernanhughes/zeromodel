# Learning trace VPM

ZeroModel can show more than tracking when the input data contains the right evidence.

Tracking means a value moved over time. Learning means a behavior changed after feedback and the change survives checks beyond the exact corrected example.

## Minimum evidence

A learning trace should contain at least three kinds of observations:

| Split | Question answered |
|---|---|
| `train` | Did the corrected or experienced unit improve? |
| `heldout` | Did related unseen or future work improve too? |
| `regression` | Did unrelated or previously good work avoid degradation? |

ZeroModel marks an assessment as `learned=True` only when train improvement clears the training threshold, held-out improvement clears the transfer threshold, and regression degradation stays below the safety threshold.

## Example

```python
from zeromodel import LearningObservation, build_learning_vpm

assessment = build_learning_vpm([
    LearningObservation("claim-support", before=0.42, after=0.72, split="train"),
    LearningObservation("related-claim", before=0.50, after=0.63, split="heldout"),
    LearningObservation("summary-quality", before=0.82, after=0.81, split="regression"),
])

print(assessment.learned)
artifact = assessment.artifact
```

The resulting artifact is an ordinary VPM. It can be inspected, rendered, bundled, gated, and compared like any other artifact.

## Metrics

`build_learning_vpm()` creates these metrics:

| Metric | Meaning |
|---|---|
| `learning_score` | Split-aware score used to put strongest evidence first. |
| `delta_positive` | Positive before/after improvement. |
| `feedback_alignment` | How well the update corresponds to the feedback. |
| `generalization` | Held-out improvement, zero for non-held-out rows. |
| `regression_safety` | One minus observed degradation. |
| `after_score` | Final observed score. |

## Correct claim

Use this wording:

> ZeroModel can make learning visible as a deterministic before/after/held-out/regression artifact trace.

Avoid this wording:

> ZeroModel watches AI think.

The first statement is testable. The second is metaphorical and should not be used as a literal technical claim.
