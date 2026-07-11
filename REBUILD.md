# ZeroModel rebuild

ZeroModel has been rebuilt from first principles around one core abstraction:

> A Visual Policy Map is a deterministic spatial view over a table of scored items.

The clean package now exposes that abstraction directly from `zeromodel`, with consumers for metric packing, PHOS packing, visual composition, differential comparison, bundle serialization, rendering, hierarchy, edge gates, and trend-aware control.

## Current package shape

```text
zeromodel/
  artifact.py
  metrics.py
  phos.py
  compose.py
  compare.py
  bundle.py
  render.py
  hierarchy.py
  edge.py
  controller.py
```

The artifact kernel remains a representation. Consumers own routing, thresholds, rendering, visual logic, hierarchy, and controller decisions.

## Core pipeline

```text
scores + identifiers + layout recipe
                 |
                 v
          VPM artifact
          /     |      \
         v      v       v
    renderer  inspector  consumer policy
      PNG      browser    router/ranker/gate
```

## Design rules

1. Score producers own what metrics mean.
2. Layout recipes declare normalization, ordering, direction, and tie-break behavior.
3. VPM artifacts preserve the view and exact source mapping.
4. Consumers own any threshold, route, ranking, controller signal, or action.
5. Provenance and deterministic IDs travel with artifacts.
6. Claims should remain backed by code, tests, and reproducible examples.
