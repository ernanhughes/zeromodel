# Training progress VPM

ZeroModel can turn model-training telemetry into deterministic progress artifacts.

A normal tracker shows movement over time: loss down, accuracy up, throughput, GPU memory. A training progress VPM adds an evidence layer: did the checkpoint improve the training objective, transfer to held-out evaluation, avoid regression, remain stable, and keep acceptable efficiency?

## Minimum telemetry

A useful training progress trace should include at least two checkpoints and metrics for:

| Evidence | Typical metric |
|---|---|
| Train objective | `train_loss`, `reward`, `train_score` |
| Held-out transfer | `heldout_score`, `eval_accuracy`, `validation_reward` |
| Regression safety | `regression_safety`, `safety_score`, `backward_compatibility` |
| Stability | `stability`, `grad_health`, `nan_free_rate` |
| Efficiency | `tokens_per_second`, `samples_per_second`, `cost_efficiency` |

## Example

```python
from zeromodel import TrainingCheckpoint, build_training_progress_vpm

assessment = build_training_progress_vpm(
    [
        TrainingCheckpoint(step=1000, metrics={
            "train_loss": 1.00,
            "heldout_score": 0.50,
            "regression_safety": 0.99,
            "stability": 0.96,
            "tokens_per_second": 1000,
        }),
        TrainingCheckpoint(step=2000, metrics={
            "train_loss": 0.82,
            "heldout_score": 0.57,
            "regression_safety": 0.98,
            "stability": 0.94,
            "tokens_per_second": 1120,
        }),
    ],
    stability_metric="stability",
    efficiency_metric="tokens_per_second",
)

print(assessment.best_checkpoint_id)
print(assessment.learned)
artifact = assessment.artifact
```

The artifact is a normal VPM. You can inspect cells, render PNG/SVG, save a `.vpm` bundle, compare artifacts across runs, or use a gate to decide whether a checkpoint should be promoted.

## Metrics

`build_training_progress_vpm()` creates these metrics:

| Metric | Meaning |
|---|---|
| `progress_score` | Weighted checkpoint progress score. |
| `train_progress` | Relative train-objective improvement from the first checkpoint. |
| `heldout_progress` | Relative held-out improvement from the first checkpoint. |
| `regression_safety` | Safety/regression score, where higher is safer. |
| `stability` | Stability score, where higher is healthier. |
| `efficiency` | Relative efficiency progress or default neutral score. |

## Correct claim

Use this wording:

> ZeroModel can turn model-training telemetry into deterministic progress artifacts that show train improvement, held-out transfer, regression risk, stability, efficiency, and checkpoint selection evidence.

Avoid this wording:

> ZeroModel understands why the model learned.

The artifact summarizes observable evidence. It does not inspect or prove internal mechanisms.
