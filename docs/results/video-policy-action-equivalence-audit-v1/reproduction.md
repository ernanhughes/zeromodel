# Reproduction

```powershell
python examples/arcade_visual_action_equivalence_audit.py --audit-evidence-closure
python examples/arcade_visual_action_equivalence_audit.py --rescore-supported-top1
python examples/arcade_visual_action_equivalence_audit.py --build-reachability-tile
python examples/arcade_visual_action_equivalence_audit.py --replay-reachability
python examples/arcade_visual_action_equivalence_audit.py --verify-bounded-measurements
python examples/arcade_visual_action_equivalence_audit.py --verify-audit
```

These commands regenerate only the bounded retrospective outputs. They do not create new visual observations, run PR #42 grids, or execute a production temporal reader.
