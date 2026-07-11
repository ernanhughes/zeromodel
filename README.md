# ZeroModel

**ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts and small consumers that can operate without a model at decision time.**

A VPM is a deterministic spatial view over a table of scored items. It carries values, stable row and metric identifiers, layout recipe, view ordering, source mapping, provenance, and deterministic identity.

The project is now v2-first. The original experimental implementation is being removed from `main`; reusable ideas from the original work and the Stephanie integration have been reimplemented as clean modules around the artifact kernel.

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
from zeromodel.v2 import LayoutRecipe, ScoreTable, build_vpm

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

## Proven capabilities now implemented

The blog/Stephanie capabilities are implemented as explicit v2 modules:

| Capability | Module |
|---|---|
| Immutable artifact kernel | `zeromodel.v2.artifact` |
| Metric alias packing and score-table building | `zeromodel.v2.metrics` |
| PHOS sort-pack and guarded top-left concentration | `zeromodel.v2.phos` |
| Visual AND/OR/NOT/XOR/add/subtract | `zeromodel.v2.compose` |
| Baseline-vs-target differential comparison | `zeromodel.v2.compare` |
| Lossless `.vpm` bundle serialization | `zeromodel.v2.bundle` |
| Dependency-light PNG/SVG rendering | `zeromodel.v2.render` |
| Hierarchical pyramids | `zeromodel.v2.hierarchy` |
| Edge top-left gates | `zeromodel.v2.edge` |
| Trend-aware EDIT/RESAMPLE/ESCALATE/STOP/SPINOFF control | `zeromodel.v2.controller` |

## PHOS and edge usage

```python
from zeromodel.v2 import TopLeftGate, guarded_pack_artifact, write_png

packed = guarded_pack_artifact(artifact)
write_png(packed.packed, "artifact_phos.png")

result = TopLeftGate(threshold=0.75).evaluate(packed.packed)
print(result.accepted, result.score)
```

## Bundle usage

```python
from zeromodel.v2 import from_bundle, to_bundle

to_bundle(artifact, "artifact.vpm")
loaded = from_bundle("artifact.vpm")
assert loaded.artifact_id == artifact.artifact_id
```

## Design rule

The artifact remains a representation. Routing, gates, visual logic, hierarchy, rendering, PHOS packing, and controllers are consumers around the artifact. This keeps the core auditable while allowing the full ZeroModel system to grow.

## Website

The zero-dependency static website lives in `site/`:

```bash
python -m http.server 4173 -d site
```

Then open `http://localhost:4173`.
