# Cross-Domain Visual Contract Replication -- Summary

## Executive result

- **Replicated in both domains**: ['visible_component_attribution_micro_f1', 'relation_correctness_rate']
- **Domain-specific (one domain only)**: ['unexpected_change_detection_rate', 'missing_change_detection_rate']
- **Not replicated in either domain**: ['direction_correctness_rate', 'value_correctness_rate']
- **Not measurable in both domains** (e.g. identity is arcade-unavailable): none

## Exact environment

- **git_commit**: 6b75b15a6178f018a99a8e1394fc7dde99e4c034
- **python_version**: 3.11.4
- **numpy_version**: 2.2.3
- **command**: C:\Projects\zeromodel\examples\visual_transition_benchmark\cross_domain_run.py --arcade-dev-episodes 40 --arcade-eval-episodes 100 --warehouse-dev-episodes 20 --warehouse-eval-episodes 100 --output-dir C:/Projects/zeromodel/artifacts/cross_domain_visual_contracts
- **arcade_dev_episodes**: 40
- **arcade_eval_episodes**: 100
- **arcade_eval_transitions**: 1800
- **warehouse_dev_episodes**: 20
- **warehouse_eval_episodes**: 100
- **warehouse_eval_transitions**: 2300
- **duration_seconds**: 397.066

## Capability table (Arcade vs. Warehouse)

| Capability | Arcade | Warehouse | Threshold | Status |
|---|---:|---:|---:|---|
| visible_component_attribution_micro_f1 | 1.000 | 1.000 | 0.950 | replicated |
| unexpected_change_detection_rate | 0.500 | 1.000 | 0.900 | domain_specific |
| missing_change_detection_rate | 0.692 | 1.000 | 0.900 | domain_specific |
| direction_correctness_rate | 0.833 | 0.870 | 0.900 | not_replicated |
| magnitude_correctness_rate | 0.833 | 0.826 | n/a | not_applicable |
| value_correctness_rate | 0.889 | 0.826 | 0.900 | not_replicated |
| relation_correctness_rate | 1.000 | 0.957 | 0.900 | replicated |
| identity_correctness_rate | n/a | 0.250 | n/a | not_applicable |

## Hidden value-fault headline (per domain)

- **arcade**: 175 of 1000 faulty transitions were component-label-clean yet value-wrong.
- **warehouse**: 900 of 1500 faulty transitions were component-label-clean yet value-wrong.

## What replicated, what did not

See `domain-results/arcade.json` and `domain-results/warehouse.json` for full per-domain metrics, and `transition-level-results.jsonl` for the per-transition record.
