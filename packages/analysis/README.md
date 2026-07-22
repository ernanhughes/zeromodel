# zeromodel-analysis

`zeromodel-analysis` contains deterministic analysis utilities for ZeroModel
Visual Policy Map artifacts. It is the second package in the ZeroModel 1.0.13
split and exposes its API through `zeromodel.analysis`.

The package depends on the validated core distribution, `zeromodel==1.0.13`.
Core data objects such as `ScoreTable`, `LayoutRecipe`, and `VPMArtifact` should
be imported from `zeromodel.core`, not re-exported from this package.

## Install

```powershell
python -m pip install zeromodel==1.0.13 zeromodel-analysis==1.0.13
```

## Includes

- Field comparison and fuzzy field composition.
- Hierarchical matrix reduction and pyramids.
- PHOS packing and top-left concentration analysis.
- Spatial optimization and optimized view generation.
- Matrix-pattern detection and discovered views.
- Decision manifolds and inflection-point detection.
- Policy diagnostics and finite policy-property checking.
- Critic, learning, and training-progress artifact builders.
- Lightweight tracker-export adapters for training checkpoint records.

## Excludes

`zeromodel-analysis` deliberately excludes observation capture, vision encoders,
video runtime/instrumentation, SQLAlchemy persistence, research orchestration,
examples, Torch, TorchVision, Transformers, Pillow, and database runtimes.

Analysis functions operate over numeric artifacts and bounded finite policies.
They do not claim semantic reasoning, causal detection, continuous formal
verification, or benchmark adjudication beyond the explicit deterministic
contracts documented by each API.

## Minimal Comparison

```python
import numpy as np

from zeromodel.analysis import compare_fields, vpm_and

baseline = np.array([[0.2, 0.8], [0.4, 0.1]])
candidate = np.array([[0.5, 0.7], [0.4, 0.3]])

comparison = compare_fields(baseline, candidate)
intersection = vpm_and(baseline, candidate)
```

## Spatial Analysis

```python
from zeromodel.analysis import SpatialOptimizer, build_optimized_view
from zeromodel.core import ScoreTable

table = ScoreTable(
    values=[[0.1, 1.0], [0.9, 0.2], [0.8, 0.3]],
    row_ids=["background", "target-a", "target-b"],
    metric_ids=["target", "noise"],
)

optimizer = SpatialOptimizer(Kc=1, Kr=2, max_evals=20)
view = build_optimized_view(table, name="target-first", optimizer=optimizer)
```

## Finite Policy Verification

```python
from zeromodel.analysis import PolicyPropertyChecker, PolicyPropertySpec
from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm

table = ScoreTable(
    values=[[0.0, 1.0], [1.0, 0.0]],
    row_ids=["enabled=true", "enabled=false"],
    metric_ids=["WAIT", "ACT"],
)
recipe = LayoutRecipe.from_dict(
    {
        "version": "vpm-layout/0",
        "name": "policy",
        "row_order": {"kind": "source", "tie_break": "row_id"},
        "column_order": {"kind": "source"},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    }
)
artifact = build_vpm(table, recipe)

spec = PolicyPropertySpec.from_dict(
    {
        "id": "act_requires_enabled",
        "version": "1",
        "assert": {
            "implies": [
                {"eq": [{"var": "winner"}, "ACT"]},
                {"eq": [{"var": "state.enabled"}, True]},
            ]
        },
    }
)
report = PolicyPropertyChecker(artifact, action_metric_ids=("WAIT", "ACT")).check(
    [spec]
)
```
