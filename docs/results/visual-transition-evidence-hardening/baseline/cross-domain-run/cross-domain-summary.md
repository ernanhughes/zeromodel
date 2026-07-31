# Cross-Domain Visual Contract Replication -- Summary

## Executive result

- **Replicated in both domains**: ['visible_component_attribution_micro_f1', 'relation_correctness_rate']
- **Domain-specific (one domain only)**: ['unexpected_change_detection_rate', 'missing_change_detection_rate']
- **Not replicated in either domain**: ['direction_correctness_rate', 'value_correctness_rate']
- **Not measurable in both domains** (e.g. identity is arcade-unavailable): none

## Exact environment

- **git_commit**: 85d4fd50607cbef607ddbe4a5f73c1468ad76955
- **python_version**: 3.11.4
- **numpy_version**: 2.4.6
- **command**: C:\Projects\zeromodel\examples\visual_transition_benchmark\cross_domain_run.py --arcade-dev-episodes 1 --arcade-eval-episodes 2 --warehouse-dev-episodes 1 --warehouse-eval-episodes 2 --skip-render --output-dir docs/results/visual-transition-evidence-hardening/baseline/cross-domain-run
- **arcade_dev_episodes**: 1
- **arcade_eval_episodes**: 2
- **arcade_eval_transitions**: 36
- **warehouse_dev_episodes**: 1
- **warehouse_eval_episodes**: 2
- **warehouse_eval_transitions**: 46
- **duration_seconds**: 6.805

## Capability table (Arcade vs. Warehouse)

| Capability | Arcade | Warehouse | Threshold | Status |
|---|---:|---:|---:|---|
| visible_component_attribution_micro_f1 | 1.000 | 1.000 | 0.950 | replicated |
| unexpected_change_detection_rate | 0.500 | 1.000 | 0.900 | domain_specific |
| missing_change_detection_rate | 0.714 | 1.000 | 0.900 | domain_specific |
| direction_correctness_rate | 0.833 | 0.870 | 0.900 | not_replicated |
| magnitude_correctness_rate | 0.833 | 0.826 | n/a | not_applicable |
| value_correctness_rate | 0.889 | 0.826 | 0.900 | not_replicated |
| relation_correctness_rate | 1.000 | 0.957 | 0.900 | replicated |
| identity_correctness_rate | n/a | 0.250 | n/a | not_applicable |

## Hidden value-fault headline (per domain)

- **arcade**: 3 of 20 faulty transitions were component-label-clean yet value-wrong.
- **warehouse**: 18 of 30 faulty transitions were component-label-clean yet value-wrong.

## What replicated, what did not

See `domain-results/arcade.json` and `domain-results/warehouse.json` for full per-domain metrics, and `transition-level-results.jsonl` for the per-transition record.
