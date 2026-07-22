# zeromodel-vision

`zeromodel-vision` provides the deterministic bounded visual-address runtime for
ZeroModel. It implements the provider-neutral contracts from
`zeromodel-observation` and uses the core artifact and policy lookup primitives
from `zeromodel`.

The supported production path is closed-world and codebook-based: a bounded
`ImageObservation` is converted into deterministic integer features, matched
against a calibrated visual index, and optionally passed through core bounded
policy lookup.

## Install

```powershell
python -m pip install zeromodel==1.0.13 zeromodel-observation==1.0.13 zeromodel-vision==1.0.13
```

## Includes

- `VisualFeatureSpec` for the exact image shape, pooling, grayscale conversion,
  and quantization contract.
- `extract_visual_features`, `visual_input_digest`, and
  `visual_feature_digest`.
- `build_visual_index` and `VisualIndexCalibration` for deterministic codebook
  construction and rejection thresholds.
- `VisualSignReader` for exact visual address recovery or rejection evidence.
- `DeterministicVisualAddressProvider` for the observation-owned
  `VisualAddressProvider` protocol.
- `VisualPolicyReader` and `VisualPolicyDecision` for bridging an accepted
  visual address to core `VPMPolicyLookup`.

## Minimal Example

```python
import numpy as np

from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation import ImageObservation
from zeromodel.vision import (
    DeterministicVisualAddressProvider,
    VisualFeatureSpec,
    VisualPolicyReader,
    VisualSignReader,
    build_visual_index,
)

table = ScoreTable(
    values=[[1.0, 0.0], [0.0, 1.0]],
    row_ids=["left", "right"],
    metric_ids=["A", "B"],
)
recipe = LayoutRecipe.from_dict(
    {
        "version": "vpm-layout/0",
        "name": "visual-policy",
        "row_order": {"kind": "source", "tie_break": "row_id"},
        "column_order": {"kind": "source"},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    }
)
policy = build_vpm(table, recipe)

spec = VisualFeatureSpec(1, 1, 1, 1, quantization_levels=256)
index = build_visual_index(
    policy,
    {
        "left": np.array([[0]], dtype=np.uint8),
        "right": np.array([[4]], dtype=np.uint8),
    },
    spec,
    threshold_fraction=0.25,
    margin_fraction=0.75,
)
reader = VisualSignReader(index.artifact, policy, action_metric_ids=("A", "B"))
provider = DeterministicVisualAddressProvider(reader, source_scope="fixture")
policy_reader = VisualPolicyReader(
    provider,
    VPMPolicyLookup(policy, action_metric_ids=("A", "B")),
)

decision = policy_reader.read(ImageObservation(np.array([[4]], dtype=np.uint8)))
assert decision.action == "B"
```

## Limitations

This package does not claim general computer vision, object recognition, semantic
perception, learned embedding quality, or benchmark superiority. It validates a
closed-world deterministic codebook and bounded rejection rules.

Learned and approximate provider families, DINOv2/Hugging Face encoders, linear
probes, registration experiments, corruption sweeps, local-correlation
providers, discriminative evidence providers, benchmark calibration, and
architecture-selection systems remain research or later package concerns.

Temporal video behavior belongs to `zeromodel-video`. SQL persistence belongs to
`zeromodel-sqlalchemy`. This package does not import SQLAlchemy, Torch,
TorchVision, Transformers, Pillow, model weights, datasets, examples, or
research results.
