# ZeroModel

**ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts and small consumers that can operate without a model at decision time.**

A VPM is a deterministic spatial view over a table of scored items. It carries values, stable row and metric identifiers, a layout recipe, view ordering, source mapping, provenance, and deterministic identity.

The package is now the clean new ZeroModel surface. There is no public `zeromodel.v2` namespace: import directly from `zeromodel`.

Public claims are tracked in [`docs/claims-audit.md`](docs/claims-audit.md). Treat that file as the source of truth for what is validated, what is implemented with thin evidence, and what remains a roadmap claim.

## Install from GitHub

```bash
pip install git+https://github.com/ernanhughes/zeromodel.git@main
```

For development:

```bash
pip install -e .[dev]
pytest
```

## Core artifact

```python
from zeromodel import LayoutRecipe, ScoreTable, build_vpm

score_table = ScoreTable(
    values=[[0.9, 0.2], [0.4, 0.8]],
    row_ids=["candidate-a", "candidate-b"],
    metric_ids=["quality", "uncertainty"],
)

recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "quality-first",
    "row_order": {
        "kind": "lexicographic",
        "keys": [{"metric_id": "quality", "direction": "desc"}],
        "tie_break": "row_id",
    },
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})

artifact = build_vpm(score_table, recipe)
cell = artifact.cell(view_row=0, view_column=0)
region = artifact.region(rows=slice(0, 1), columns=slice(0, 2))
```

## Capability surface

| Capability | Module |
|---|---|
| Immutable artifact kernel | `zeromodel.artifact` |
| Metric alias packing and score-table building | `zeromodel.metrics` |
| PHOS sort-pack and guarded top-left concentration | `zeromodel.phos` |
| Visual AND/OR/NOT/XOR/add/subtract | `zeromodel.compose` |
| Baseline-vs-target differential comparison | `zeromodel.compare` |
| Lossless `.vpm` bundle serialization | `zeromodel.bundle` |
| Dependency-light PNG/SVG rendering | `zeromodel.render` |
| Hierarchical pyramids | `zeromodel.hierarchy` |
| Edge top-left gates | `zeromodel.edge` |
| Trend-aware EDIT/RESAMPLE/ESCALATE/STOP/SPINOFF control | `zeromodel.controller` |
| Before/after/held-out/regression learning traces | `zeromodel.learning` |
| Model-training progress artifacts | `zeromodel.training` |
| Tracker-export adapters | `zeromodel.adapters` |

## PHOS and edge usage

```python
from zeromodel import TopLeftGate, guarded_pack_artifact, write_png

packed = guarded_pack_artifact(artifact)
write_png(packed.packed, "artifact_phos.png")

result = TopLeftGate(threshold=0.75).evaluate(packed.packed)
print(result.accepted, result.score)
```

## Learning trace usage

Tracking means a score moved. Learning means a feedback-driven change improves corrected work, transfers to held-out work, and avoids unacceptable regression.

```python
from zeromodel import LearningObservation, build_learning_vpm

assessment = build_learning_vpm([
    LearningObservation("claim-support", before=0.42, after=0.72, split="train"),
    LearningObservation("related-claim", before=0.50, after=0.63, split="heldout"),
    LearningObservation("summary-quality", before=0.82, after=0.81, split="regression"),
])

print(assessment.learned)
learning_artifact = assessment.artifact
```

See [`docs/examples/learning-trace-vpm.md`](docs/examples/learning-trace-vpm.md).

## Training progress usage

Training telemetry can become a checkpoint-level VPM that shows train improvement, held-out transfer, regression safety, stability, efficiency, and best-checkpoint evidence.

```python
from zeromodel import TrainingCheckpoint, build_training_progress_vpm

progress = build_training_progress_vpm(
    [
        TrainingCheckpoint(step=1000, metrics={
            "train_loss": 1.00,
            "heldout_score": 0.50,
            "regression_safety": 0.99,
        }),
        TrainingCheckpoint(step=2000, metrics={
            "train_loss": 0.82,
            "heldout_score": 0.57,
            "regression_safety": 0.98,
        }),
    ]
)

print(progress.best_checkpoint_id, progress.learned, progress.warnings)
training_artifact = progress.artifact
```

See [`docs/examples/training-progress-vpm.md`](docs/examples/training-progress-vpm.md).

## Tracker adapter usage

Adapters parse exported tracker files into `TrainingCheckpoint` objects without requiring TensorBoard, W&B, or Trackio SDKs at runtime.

```python
from zeromodel import build_training_progress_vpm
from zeromodel.adapters import checkpoints_from_tensorboard_scalars

checkpoints = checkpoints_from_tensorboard_scalars("runs/scalars.csv")
progress = build_training_progress_vpm(checkpoints)
print(progress.best_checkpoint_id, progress.learned, progress.warnings)
```

Supported inputs are JSON, JSONL/NDJSON, and CSV exports. TensorBoard scalar CSV rows shaped like `wall_time,step,tag,value` are grouped into one checkpoint per step.

See [`docs/examples/training-tracker-adapters.md`](docs/examples/training-tracker-adapters.md).

## Bundle usage

```python
from zeromodel import from_bundle, to_bundle

to_bundle(artifact, "artifact.vpm")
loaded = from_bundle("artifact.vpm")
assert loaded.artifact_id == artifact.artifact_id
```

## Design rule

The artifact remains a representation. Routing, gates, visual logic, hierarchy, rendering, PHOS packing, learning traces, training progress, tracker adapters, and controllers are consumers around the artifact. This keeps the core auditable while allowing the full ZeroModel system to grow.
