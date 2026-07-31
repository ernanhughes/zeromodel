# Visual Sign Reader Genuine Alias Corpus - Final Assessment

## 1. Executive conclusion
The frozen confirmation corpus found 194 accepted wrong-row profile cases, representing 72 unique transformed wrong-row visual aliases.

## 2. Starting `main` SHA
38aee1edba9bc3e5a291f186407bebb349f9dd4a

## 3. Research question
Can deterministic, bounded transformations of a true visual observation cause the Visual Sign Reader to accept and execute a policy row other than the true source row?

## 4. Production contracts reused
Reused VisualSignReader, VisualDecision, VisualFeatureSpec, compiled arcade policy, visual index, and calibration without production-package edits.

## 5. Transform-registry design
Registry `sha256:25186b831961e93eed63fbd2124917ac83dc253862113f5631e3806eff020890` is target-agnostic and source-only. It covers representation, geometric, photometric, blur/compression, occlusion, noise, local-corruption, and destructive negative-control families.

## 6. Target-row leakage controls
Transform functions accept only source observation, transform spec, and optional fixed seed; no target-row, target image, or target feature input exists.

## 7. Transition-leakage controls
Alias membership is computed only from static reader results. No after frame, transition evidence, conformance report, or next-state consequence is an input.

## 8. Source-state coverage
Frozen confirmation uses the predeclared confirmation split over arcade finite source rows.

## 9. Discovery/confirmation split
Discovery and confirmation are split by source-row identity hash before inspecting reader outcomes.

## 10. Generated case counts
`{"acceptance_rate": 0.483718487394958, "accepted_case_count": 2763, "accepted_wrong_row_count": 194, "accepted_wrong_row_per_1000_transformations": 33.963585434173666, "accepted_wrong_row_profile_cases_per_1000_transformations": 33.963585434173666, "action_changing_alias_rate": 0.06441947565543071, "action_equivalent_alias_rate": 0.08089887640449438, "duplicate_count": 848, "evidence_only_accepted_count": 1428, "exact_row_accuracy_among_accepted": 0.854681647940075, "failed_transformation_count": 0, "generated_case_count": 5712, "policy_action_accuracy_among_accepted": 0.9355805243445693, "policy_executed_accepted_count": 1335, "reader_accepted_case_count": 2763, "rejection_rate": 0.5162815126050421, "runtime_seconds": 2.690308, "transformation_case_count": 5712, "unique_matched_rows_receiving_aliases": 36, "unique_source_matched_row_pairs": 72, "unique_source_rows_producing_aliases": 28, "unique_transformed_observation_count": 4864, "unique_wrong_row_observation_count": 72, "wrong_row_accepted_rate": 0.1453183520599251, "wrong_row_different_action_count": 86, "wrong_row_profile_case_count": 194, "wrong_row_rate_among_all_accepts": 0.07021353601158162, "wrong_row_rate_among_policy_executes": 0.1453183520599251, "wrong_row_same_action_count": 108, "wrong_row_wilson_interval": {"count": 194, "denominator": 1335, "high": 0.1652413779440623, "low": 0.1274307350228488}}`

## 11. Deduplication
Wrong-row profile cases are separated from profile-independent visual aliases. The handoff collapses duplicate profile outcomes onto one transformed observation with all accepting profiles preserved.

## 12. Acceptance-profile results
Wrong-row profile cases were produced by canonical_only, exact_codeword, and calibrated_nearest. Evidence-only cases are reported separately and never counted as policy-executed aliases.

## 13. Canonical collisions
The canonical source-row collision audit found zero collision groups before transformation.

## 14. Feature-codeword collisions
The canonical source-row feature-codeword collision audit found zero collision groups before transformation.

## 15. Calibrated-nearest results
Calibrated-nearest cases preserve nearest distance, second distance, margin, acceptance threshold, required margin, and calibration digest; rule mismatches are reported in nearest-margin-results.json.

## 16. Accepted correct perturbations
Representation, ordinary geometric, photometric, and blur/compression transforms mostly preserve the correct row or reject without producing wrong-row profile cases in this registry.

## 17. Accepted wrong-row aliases
The core positive result is genuine: source-derived transformed observations can be accepted as a different finite policy row.

## 18. Action-equivalent aliases
108 wrong-row profile cases preserved the policy action while changing the addressed row.

## 19. Action-changing aliases
86 wrong-row profile cases changed the selected policy action.

## 20. Transform-family analysis
Wrong-row profile cases concentrate in local_corruption, destructive negative_control, occlusion, and noise. Representation, geometric, photometric, and blur/compression families did not produce wrong-row cases in the frozen confirmation summary.

## 21. Severity analysis
The strongest finding is semantic erasure: destructive controls can collapse a source observation into another accepted finite visual state rather than merely causing rejection.

## 22. Source-state analysis
Source-action and state-family breakdowns are preserved in source-action-results.json and state-family-results.json.

## 23. Row-pair analysis
Row-pair-results.json records source-to-matched directionality; mappings are not assumed symmetric.

## 24. Negative controls
Negative controls did not all reject. Accepted wrong-row negative-control outcomes are reported as negative-control failures, not hidden.

## 25. Adversarial controls
Adversarial-controls.json records only generated checks executed by the evidence generator and lists prior declarative controls that were removed rather than reported as passes.

## 26. Replay validation
Replay artifacts preserve transformed observations for wrong-row aliases and selected controls; replay-results.json verifies raw/canonical/feature digests and VisualDecision outputs.

## 27. Failure atlas
The atlas groups accepted wrong-row aliases when present, otherwise closest rejected/low-margin cases.

## 28. Production changes
No production API changes were required.

## 29. Claims strengthened
A bounded positive claim is now supported: committed target-agnostic transformations can produce genuine accepted wrong-row arcade Visual Sign Reader decisions.

## 30. Claims reduced or refuted
The result does not establish natural-image robustness, adversarial optimality, or transition-based correction.

## 31. Recommended disposition
Use the deduplicated frozen handoff, not raw profile-case counts, as input to the next transition-adjudication stage.

## 32. Handoff corpus for transition adjudication
frozen-alias-handoff.json contains one replay-verified entry per unique transformed visual alias with accepting profiles preserved.

## 33. Next research question
Can one-step transition evidence adjudicate the deduplicated replay-verified wrong-row handoff aliases without changing corpus membership?

## 34. Complete command and artifact index
See `docs\results\visual-sign-reader-genuine-alias-corpus-v1\discovery` for manifest, commands, registry, cases, audits, replay artifacts, atlas, and handoff.
