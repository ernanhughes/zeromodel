# Representation comparison

All compared runs share provider_configuration_id=sha256:8f878520d89c432ff73073849150e1423b281cebbe4389de2ddd631738b8a4e0, model_digest=sha256:b8571dca341cc2f9ee4365b30ccaecae43617284b5e44630b8985a12cc1576a7, prompt_digest=sha256:356a657aecb1b3c7e3b5e13eb830ae24ceeb76acc2156dfd771ea8921375efda, policy_artifact_id=sha256:c27d05243c6f960ba30b84fcfe0ddf67c03ee5de053821e29091694caf9d5d4a, fixture_identity=arcade-png-representation-benchmark:smoke, case_mode=arcade-smoke-v1. Only representation_mode and recipe_id varied across the compared runs. Baseline variant: unlabelled-v1.

| Variant | Recipe | Exact | Action-equiv | Action-changing | Rejected | Action-correct | Latency median (us) | Latency p95 (us) | Classification |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cooldown-dual-v1 | `sha256:69f74cd87...` | 3 | 4 | 1 | 0 | 7 | 1946052 | 2173042 | no_material_change |
| cooldown-redundant-v1 | `sha256:2d9de1ead...` | 3 | 3 | 2 | 0 | 6 | 2008380 | 2053426 | regression |
| cooldown-shape-v1 | `sha256:3b7989211...` | 4 | 3 | 1 | 0 | 7 | 1957883 | 3164967 | no_material_change |
| labelled-v1 | `sha256:393c0e10a...` | 8 | 0 | 0 | 0 | 8 | 2080155 | 6454331 | advance |
| lane-enhanced-v1 | `sha256:87420173e...` | 2 | 3 | 2 | 1 | 5 | 1997680 | 2039776 | regression |
| unlabelled-v1 | `sha256:6785eea03...` | 3 | 4 | 1 | 0 | 7 | 2047138 | 2181562 | n/a |

## Classification reasoning

- **cooldown-dual-v1**: no_material_change - compatible run with no regression and no declared target metric improvement
- **cooldown-redundant-v1**: regression - action_changing_count increased (1 -> 2)
- **cooldown-shape-v1**: no_material_change - compatible run with no regression and no declared target metric improvement
- **labelled-v1**: advance - declared target metric(s) improved: exact_count
- **lane-enhanced-v1**: regression - action_changing_count increased (1 -> 2)
