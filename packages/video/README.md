# zeromodel-video

`zeromodel-video` provides ZeroModel's deterministic video runtime contracts,
provider-neutral temporal policy reader, and video action-set domain DTO/Store
architecture.

It depends on:

- `zeromodel==1.0.13`
- `zeromodel-observation==1.0.13`
- `numpy>=1.23`

It does not depend on `zeromodel-analysis`, `zeromodel-vision`,
`zeromodel-sqlalchemy`, SQLAlchemy, Torch, TorchVision, Transformers, Pillow,
research modules, examples, or the repository root.

## Install

```powershell
python -m pip install zeromodel==1.0.13 zeromodel-observation==1.0.13 zeromodel-video==1.0.13
```

## Includes

- `VideoFrame`, `VideoFrameSource`, `InMemoryVideoFrameSource`, and
  `VideoClipManifest` for immutable frame and clip identity.
- `VideoPolicyReader`, `TemporalEvidence`, `VideoPolicyDecision`, and
  `VideoPolicyTrace` for temporal policy over observation-owned visual address
  contracts.
- Video action-set DTOs for benchmark identity, episode plans, sealed final
  split plans, observation ledgers, provider descriptors, and operation chains.
- `VideoActionSetStore` and `InMemoryVideoActionSetStore` for DTO-only Store
  semantics.
- `build_runtime()` for default in-memory runtime composition.

## Temporal Policy Example

```python
import numpy as np

from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.core.policy_transitions import PolicyTransitionSpec
from zeromodel.video import InMemoryVideoFrameSource, VideoPolicyReader

table = ScoreTable(
    values=[[1.0, 0.0], [0.0, 1.0]],
    row_ids=["left", "right"],
    metric_ids=["A", "B"],
)
recipe = LayoutRecipe.from_dict(
    {
        "version": "vpm-layout/0",
        "name": "video-policy",
        "row_order": {"kind": "source", "tie_break": "row_id"},
        "column_order": {"kind": "source"},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    }
)
policy = build_vpm(table, recipe)

source = InMemoryVideoFrameSource.from_arrays(
    [np.zeros((1, 1), dtype=np.uint8)],
    clip_id="clip-a",
    nominal_fps=1.0,
)

# Supply any object implementing zeromodel.observation.VisualAddressProvider.
provider = ...
reader = VideoPolicyReader(
    provider,
    VPMPolicyLookup(policy, action_metric_ids=("A", "B")),
    PolicyTransitionSpec({"left": ("left", "right"), "right": ("right",)}),
)
trace = reader.read(source)
```

## In-Memory Action-Set Example

```python
from zeromodel.video import BenchmarkIdentityDTO, build_runtime

runtime = build_runtime()
identity = BenchmarkIdentityDTO(
    contract_commit="contract",
    seed_material="seed",
    seed_digest="sha256:" + __import__("hashlib").sha256(b"seed").hexdigest(),
    policy_artifact_id="policy",
    parent_audit_sha="audit",
    parent_v3_sha="v3",
)
runtime.video_action_set.save_identity(identity)
assert runtime.video_action_set.get_identity(identity.seed_digest) == identity
```

## Exclusions

Concrete visual providers are injected by callers. This package consumes
provider-neutral observation contracts and does not import `zeromodel.vision`.

SQL persistence belongs to `zeromodel-sqlalchemy`. This package provides Store
protocols and in-memory implementations only.

Research benchmark orchestration, provider comparison, evidence generation,
large sweeps, committed-result reconstruction, and final cross-package
integration remain outside this package.

