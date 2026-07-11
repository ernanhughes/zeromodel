# Training tracker adapters

ZeroModel training adapters convert exported tracker logs into `TrainingCheckpoint` objects. The adapters intentionally read files rather than live tracker APIs so progress artifacts can be reproduced from committed fixtures.

## Supported export shapes

| Source | Function | Input shape |
|---|---|---|
| Generic JSONL | `checkpoints_from_jsonl()` | one checkpoint per line, either flat metrics or nested `metrics` |
| Generic JSON | `checkpoints_from_json()` | list/object with `records`, `history`, `checkpoints`, `rows`, or `data` |
| Generic CSV | `checkpoints_from_csv()` | flat metric columns |
| TensorBoard | `checkpoints_from_tensorboard_scalars()` | scalar rows such as `wall_time,step,tag,value` |
| W&B | `checkpoints_from_wandb_export()` | history JSONL/JSON/CSV exports, usually flat rows with `_step` |
| Trackio | `checkpoints_from_trackio_export()` | JSON/JSONL/CSV exports with flat metrics or nested `metrics` |

## TensorBoard scalar CSV

```csv
wall_time,step,tag,value
1,1000,train/loss,1.0
1,1000,eval/accuracy,0.50
1,1000,regression_safety,0.99
2,2000,train/loss,0.82
2,2000,eval/accuracy,0.57
2,2000,regression_safety,0.98
```

```python
from zeromodel import build_training_progress_vpm
from zeromodel.adapters import checkpoints_from_tensorboard_scalars

checkpoints = checkpoints_from_tensorboard_scalars("tb_scalars.csv")
progress = build_training_progress_vpm(checkpoints)
```

TensorBoard event protobuf parsing is not included. Export scalar data to CSV/JSON first to keep the ZeroModel dependency surface small and reproducible.

## W&B history JSONL

```jsonl
{"_step":1000,"train/loss":1.0,"val/accuracy":0.50,"regression/safety":0.99}
{"_step":2000,"train/loss":0.82,"val/accuracy":0.57,"regression/safety":0.98}
```

```python
from zeromodel.adapters import checkpoints_from_wandb_export

checkpoints = checkpoints_from_wandb_export("wandb-history.jsonl")
```

## Trackio or generic nested JSON

```json
{
  "checkpoints": [
    {"step": 1000, "metrics": {"train_loss": 1.0, "heldout_score": 0.50, "regression_safety": 0.99}},
    {"step": 2000, "metrics": {"train_loss": 0.76, "heldout_score": 0.62, "regression_safety": 0.97}}
  ]
}
```

```python
from zeromodel.adapters import checkpoints_from_trackio_export

checkpoints = checkpoints_from_trackio_export("trackio.json")
```

## Metric aliases

Adapters include common aliases such as:

| Export metric | ZeroModel metric |
|---|---|
| `train/loss` | `train_loss` |
| `eval/accuracy` | `heldout_score` |
| `val/accuracy` | `heldout_score` |
| `regression/safety` | `regression_safety` |
| `tokens/sec` | `tokens_per_second` |

You can pass `metric_aliases={...}` to override or extend these mappings.

## Correct claim

Use this wording:

> ZeroModel can ingest dependency-light tracker exports and turn them into deterministic training progress artifacts.

Avoid this wording:

> ZeroModel connects to every tracker automatically.

The adapters currently parse exported files. Live tracker APIs should remain optional integration layers outside the core package.
