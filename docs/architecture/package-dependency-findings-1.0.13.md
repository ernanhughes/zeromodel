# ZeroModel 1.0.13 Package Dependency Findings

Baseline commit: `52be5f838e15d4b32a4fdf5a393762101afd2656`

## Blocker

### B1. Forbidden proposed package edge `zeromodel.content_identity` -> `zeromodel.visual_address`

- Paths/import edge: `zeromodel.content_identity` imports `zeromodel.visual_address` at line 13 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B2. Forbidden proposed package edge `zeromodel.domains.video_action_set.arcade_observation` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.arcade_observation` imports `zeromodel.arcade_policy` at line 7 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B3. Forbidden proposed package edge `zeromodel.domains.video_action_set.build_orchestration` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.build_orchestration` imports `zeromodel.arcade_policy` at line 7 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B4. Forbidden proposed package edge `zeromodel.domains.video_action_set.build_orchestration` -> `zeromodel.db.runtime`

- Paths/import edge: `zeromodel.domains.video_action_set.build_orchestration` imports `zeromodel.db.runtime` at line 92 (deferred).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B5. Forbidden proposed package edge `zeromodel.domains.video_action_set.control_histories` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.control_histories` imports `zeromodel.arcade_policy` at line 6 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B6. Forbidden proposed package edge `zeromodel.domains.video_action_set.episode_materialization` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.episode_materialization` imports `zeromodel.arcade_policy` at line 8 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B7. Forbidden proposed package edge `zeromodel.domains.video_action_set.family_intervention_planning` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.family_intervention_planning` imports `zeromodel.arcade_policy` at line 6 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B8. Forbidden proposed package edge `zeromodel.domains.video_action_set.frame_family_kernels` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.frame_family_kernels` imports `zeromodel.arcade_policy` at line 8 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B9. Forbidden proposed package edge `zeromodel.domains.video_action_set.materialization_reachability` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.materialization_reachability` imports `zeromodel.arcade_policy` at line 6 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B10. Forbidden proposed package edge `zeromodel.domains.video_action_set.observation_universe` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.observation_universe` imports `zeromodel.arcade_policy` at line 8 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B11. Forbidden proposed package edge `zeromodel.domains.video_action_set.verification_orchestration` -> `zeromodel.arcade_policy`

- Paths/import edge: `zeromodel.domains.video_action_set.verification_orchestration` imports `zeromodel.arcade_policy` at line 12 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B12. Forbidden proposed package edge `zeromodel.render` -> `zeromodel.compose`

- Paths/import edge: `zeromodel.render` imports `zeromodel.compose` at line 16 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B13. Forbidden proposed package edge `zeromodel.video_action_set_final_admin_cli` -> `zeromodel.db.runtime`

- Paths/import edge: `zeromodel.video_action_set_final_admin_cli` imports `zeromodel.db.runtime` at line 9 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B14. Forbidden proposed package edge `zeromodel.video_action_set_final_admin_cli` -> `zeromodel.db.session`

- Paths/import edge: `zeromodel.video_action_set_final_admin_cli` imports `zeromodel.db.session` at line 10 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B15. Forbidden proposed package edge `zeromodel.video_action_set_final_cli` -> `zeromodel.db.runtime`

- Paths/import edge: `zeromodel.video_action_set_final_cli` imports `zeromodel.db.runtime` at line 10 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B16. Forbidden proposed package edge `zeromodel.video_action_set_final_cli` -> `zeromodel.db.session`

- Paths/import edge: `zeromodel.video_action_set_final_cli` imports `zeromodel.db.session` at line 11 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B17. Forbidden proposed package edge `zeromodel.video_policy` -> `zeromodel.policy_transitions`

- Paths/import edge: `zeromodel.video_policy` imports `zeromodel.policy_transitions` at line 12 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

### B18. Forbidden proposed package edge `zeromodel.visual_policy` -> `zeromodel.visual`

- Paths/import edge: `zeromodel.visual_policy` imports `zeromodel.visual` at line 10 (runtime).
- Conflict: proposed classification graph contains an edge not permitted by the six-package target graph.
- Extraction stage affected: Stage 1.0.13A boundary manifest and all later package moves.
- Smallest remedy: split or invert the import so the downstream package imports through a public DTO/protocol owned by the allowed dependency.
- Blocks Stage 1.0.13A: yes.

## High

- Root `zeromodel/__init__.py` imports heavyweight and research-facing modules at import time, contradicting the lightweight core import requirement. Remedy: remove root compatibility exports during Stage 1.0.13A after package-local APIs are declared.

## Medium

- Optional dependencies are declared globally in one distribution, while modules requiring them are intermingled in the `zeromodel` namespace. Remedy: move dependency-owning implementations to vision, research, or sqlalchemy packages before publishing wheels.

## Low

- CI and release scripts assume one distribution and one `dist/` directory. Remedy: replace with workspace-aware build matrix in Stage 1.0.13H.
