# Visual Transition Evidence Hardening - Final Assessment

## 1. Executive conclusion

ZeroModel has a bounded production transition capability: exact before/after `SourceVPMDTO` evidence can be measured, evaluated against declared annotation/relation expectations, and now bound to a declared action, exact expectation set, Visual Sign Reader trace, and conformance report identity. This supports localized contract-violation findings, not causal root-cause diagnosis.

## 2. Starting `main` SHA

`85d4fd50607cbef607ddbe4a5f73c1468ad76955`

## 3. Existing transition architecture

P18A fieldwise evidence and P18B conformance were already present in `zeromodel.perception`. The benchmark-owned arcade and warehouse code remains under `examples/visual_transition_benchmark/`.

## 4. Simplest accurate capability description

Given exact before/after observations, a field schema, annotations or relations, a declared action, and explicit expectations, ZeroModel can produce deterministic transition evidence and localize selected deviations to declared fields, components, or relations.

## 5. Baseline reproduction

Focused P18A/P18B tests passed. Focused Visual Sign Reader and video policy reader tests passed. Arcade component/value and cross-domain smoke runs completed. The compiler smoke run completed in 213.815 seconds with 11 compiled cases and 1 insufficient-observability case.

## 6. Production versus benchmark ownership

Production owns generic evidence, expectations, conformance, action identity, expectation-set identity, and composite analysis identity. Benchmarks own renderers, component names, decoders, fault injection, privileged state, metrics, and reports.

## 7. Transition identity chain

The hardened chain is:

`before_source_vpm_id + after_source_vpm_id + transition_evidence_id + action_id + expectation_set_id + conformance_report_id -> analysis_id`.

## 8. Visual Sign Reader integration

`VisualTransitionReaderTraceDTO` preserves raw/canonical/feature digests, visual index and policy IDs, acceptance profile, `policy_executed`, nearest/matched rows, and exact/canonical match flags. `evidence_only` cannot be represented as policy execution.

## 9. Action and expectation identity

`TransitionActionDeclarationDTO` canonicalizes action payloads. `TransitionExpectationSetDTO` canonicalizes expectation order and rejects duplicate/conflicting targets.

## 10. Fieldwise evidence hardening

Identity checks show swapped before/after changes transition evidence identity and reverses signed change for the changed field.

## 11. Component-level conformance

Existing P18B conformance continues to distinguish confirmed, missing expected change, unexpected change, wrong direction, excessive/insufficient change, unexplained change, and inconclusive.

## 12. Value-level conformance

Value-level benchmark results are reproduced at smoke scale only through `value_run`; value decoders remain benchmark-owned.

## 13. Representation compiler assessment

The compiler smoke run completed and reproduced the historical outcome shape at small sample counts. Compiler abstractions were still intentionally not promoted because the runtime is high for the fast tier and the compiler remains benchmark/research machinery.

## 14. Observability and representation outcomes

The production change preserves explicit reader evidence boundaries. Broader insufficient-representation versus insufficient-observability compiler claims remain historical unless separately reproduced.

## 15. Static-address versus transition evidence

The current hardening preserves the fields needed to compare static Visual Sign Reader aliases against transition evidence, but a dedicated alias comparison generator was not completed in this pass.

## 16. Adversarial results

Focused adversarial checks cover changed action payloads, duplicate expectations, report/evidence mismatches, evidence-only policy execution, and swapped before/after identity.

## 17. Baseline comparison

Raw pixel, privileged symbolic, component ZeroModel, and value-aware ZeroModel smoke baselines ran through existing benchmark scripts. No general superiority claim is made.

## 18. Cross-domain replication

Cross-domain arcade and warehouse smoke runs completed. The results remain per-domain in `cross-domain-results.json`.

## 19. Performance and evidence size

No detailed latency or size benchmark was completed. Existing command durations are recorded in generated result files, and the full fast suite completed successfully in 92.69 seconds.

## 20. Production changes

Added `packages/perception/src/zeromodel/perception/transition_analysis.py` and exported its public DTOs.

## 21. Promoted abstractions

Promoted action declaration identity, expectation-set identity, Visual reader trace preservation, and composite transition analysis identity.

## 22. Rejected promotions

Compiler/search abstractions, region geometry, value analysis results, domain transitions, metrics, and renderers remain in examples.

## 23. Claims strengthened

The transition identity claim is strengthened: the analysis object now prevents mixing ordered evidence, actions, expectation sets, and reports silently.

## 24. Claims reduced or refuted

No compiler production abstraction was promoted from this run; the compiler remains benchmark-owned despite smoke reproduction.

## 25. Remaining risks

The full adversarial matrix, static-address alias comparison, and detailed performance measurements remain incomplete. Full fast-suite validation passed.

## 26. Recommended disposition

Keep the production hardening. Do not promote compiler or value-decoder abstractions until separate evidence completes.

## 27. Practical applications

Strongest immediate use: visual regression and simulator/game debugging where operations have declared visual consequences and stable regions. Agent action verification is plausible when the action is declared and frames are exact. Fixed-camera monitoring remains future work requiring calibration, drift detection, false-alarm studies, and fallback procedures.

## 28. Next capability stage

Complete the static-address versus transition alias comparison and the full adversarial matrix, then run any larger compiler evidence job under an explicit longer budget.

## 29. Complete command and artifact index

See `commands.jsonl`, `manifest.json`, `identity-and-pairing-results.json`, and the `baseline/` directory.
