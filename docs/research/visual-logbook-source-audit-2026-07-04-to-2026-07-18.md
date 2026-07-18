# Visual Logbook Source Audit

**Period:** July 4–18, 2026
**Main SHA reviewed:** `bc99e04c775beb818cff8531fd6ab1d9a04c2c76`
**Commit count reviewed:** 385
**First-parent commits:** 65
**All branch commits considered:** 385

Implementation during this period was substantially AI-assisted. Git commits,
tests, reports and evidence artifacts are therefore treated as the authoritative
record of what entered the repository; commit prose alone is not treated as a
reliable account of research meaning.

The working tree was not clean at the start of reconstruction because
pre-existing untracked files `git-log-last-2weeks.txt` and
`git-log-last-week.txt` were present in the repository root. They were
preserved unchanged and treated as unrelated to the documentation update.

## Research episodes

### Deterministic policy infrastructure rebuilt before vision

**Date range:** July 6–17, 2026
**Mainline merge or delivery commit:** `e113bd4` — `Merge pull request #21 from ernanhughes/test/exhaustive-arcade-validation`
**Supporting commits:**
- `a9793e0` — `Merge pull request #2 from ernanhughes/codex/zeromodel-first-principles`
- `98775ab` — `Merge pull request #3 from ernanhughes/codex/vpm-artifact-kernel`
- `d909e9d` — `Merge pull request #4 from ernanhughes/codex/clean-v2-package`
- `69ca00e` — `Merge pull request #6 from ernanhughes/codex/claims-audit`
- `22f2699` — `Merge pull request #17 from ernanhughes/agent/pre-1-claims-hardening`
- `554737b` — `Merge pull request #19 from ernanhughes/agent/reproducible-signs-demo`
- `cb95a1c` — `Merge pull request #20 from ernanhughes/agent/v1.0.11-criticality-verification`
**Files:**
- `zeromodel/artifact.py`
- `zeromodel/policy_lookup.py`
- `zeromodel/policy_properties.py`
- `docs/claims-audit.md`
- `docs/results/arcade-validation.md`
**Reports:**
- `docs/results/arcade-validation.md`
**Tests:**
- `tests/test_artifact_kernel.py`
- `tests/test_arcade_shooter_example.py`
- `tests/test_arcade_shooter_exhaustive.py`
- `tests/test_policy_lookup.py`
- `tests/test_policy_properties.py`
**Classification:** C. Operationally important
**Logbook entry:** Yes
**Notes:** This episode did not measure approximate vision, but it fixed the deterministic reference object that later visual work addressed.

### Observation-addressed visual policy lookup opens the programme

**Date range:** July 17, 2026
**Mainline merge or delivery commit:** `92ed26b` — `Merge observation-addressed visual policy lookup`
**Supporting commits:**
- `183f570` — `Merge pull request #28 from ernanhughes/fix/visual-reader-calibration-contract`
**Files:**
- `zeromodel/visual.py`
- `examples/arcade_visual_sign_reader.py`
- `docs/research/visual-sign-reader.md`
**Reports:**
- `docs/research/visual-sign-reader.md`
**Tests:**
- `tests/test_visual_sign_reader.py`
**Classification:** A. Major research transition
**Logbook entry:** Yes
**Notes:** Measured exact canonical recovery and exhaustive wave equivalence on the declared fixture; bounded positive result.

### Governed visual address contracts separate made images from found images

**Date range:** July 17, 2026
**Mainline merge or delivery commit:** `58b7a34` — `Merge pull request #29 from ernanhughes/feat/governed-visual-address-contract`
**Supporting commits:**
- `76e827a` — `docs: record visual representation identity decision`
- `9d34197` — `docs: define phase zero visual address benchmark`
**Files:**
- `zeromodel/visual_address.py`
- `zeromodel/visual_policy.py`
- `zeromodel/matrix_blob.py`
- `docs/adr/visual-representation-identity.md`
- `docs/research/visual-address-phase-zero.md`
**Reports:**
- `docs/research/visual-address-phase-zero.md`
- `docs/adr/visual-representation-identity.md`
**Tests:**
- `tests/test_visual_address.py`
- `tests/test_visual_benchmark.py`
- `tests/test_matrix_blob.py`
- `tests/test_deployment_binding.py`
**Classification:** A. Major research transition
**Logbook entry:** Yes
**Notes:** Design correction rather than experimental result. This is the architectural point where dense observation evidence stopped being treated as another VPM coordinate view.

### Phase 1 held-out benchmark adds global baselines and closes first negative result

**Date range:** July 17, 2026
**Mainline merge or delivery commit:** `bed5fe8` — `Merge pull request #31 from ernanhughes/agent/visual-evidence-closure`
**Supporting commits:**
- `08eb31f` — `Merge pull request #30 from ernanhughes/feat/phase-one-visual-benchmark`
- `d3ba2ef` — `docs: adjudicate visual address external review`
- `40ab32f` — `feat: promote visual benchmark fidelity metrics`
- `197d1e2` — `feat: separate visual ranking from calibration`
**Files:**
- `examples/arcade_visual_address_benchmark.py`
- `zeromodel/visual_encoder.py`
- `zeromodel/visual_retrieval.py`
- `zeromodel/visual_analysis.py`
- `docs/results/visual-address-phase-one-v1/`
**Reports:**
- `docs/research/visual-address-phase-one.md`
- `docs/research/visual-address-research-status.md`
- `docs/research/visual-address-review-adjudication.md`
- `docs/results/visual-address-phase-one-v1/README.md`
**Tests:**
- `tests/test_arcade_visual_address_benchmark.py`
- `tests/test_visual_encoder.py`
- `tests/test_visual_precomputed.py`
- `tests/test_visual_retrieval.py`
- `tests/test_visual_analysis.py`
**Classification:** A. Major research transition
**Logbook entry:** Yes
**Notes:** This episode implemented the benchmark, measured global baselines, recovered the historical Phase 1 raw result, and formalized the negative DINOv2 interpretation.

### System B semantics and evidence closure are repaired

**Date range:** July 17, 2026
**Mainline merge or delivery commit:** `e7d2670` — `Merge pull request #32 from ernanhughes/research/visual-b-operating-curves`
**Supporting commits:**
- `35d5153` — `fix(visual): bind adjudication semantics and calibration boundaries`
- `b1b7185` — `data(results): regenerate system-b evidence with run-manifest binding`
**Files:**
- `zeromodel/visual_benchmark.py`
- `zeromodel/visual_system_b.py`
- `examples/arcade_visual_system_b_adjudication.py`
- `docs/results/visual-address-system-b-v2/`
- `scripts/check_visual_evidence_impact.py`
**Reports:**
- `docs/research/visual-address-system-b-v2-adjudication.md`
- `docs/results/visual-address-system-b-v2/README.md`
- `docs/results/visual-address-system-b-v2/final-summary.json`
- `docs/results/visual-address-system-b-v2/run-manifest.json`
**Tests:**
- `tests/test_visual_system_b.py`
- `tests/test_visual_result_records.py`
- `tests/test_visual_evidence_impact.py`
- `tests/test_visual_analysis.py`
**Classification:** A. Major research transition
**Logbook entry:** Yes
**Notes:** The key correction was epistemic, not merely mechanical: raw ranking and governed acceptance became separate reported quantities, and evidence identity was bound to a run manifest.

### Registered local baseline showdown confirms translation-locality mechanism

**Date range:** July 18, 2026
**Mainline merge or delivery commit:** `7c88afd` — `Merge pull request #34 from ernanhughes/research/visual-local-baseline-showdown`
**Supporting commits:**
- `20f2d90` — `Add registered local visual baseline showdown`
- `58ddd49` — `Record registered local baseline showdown evidence`
- `592cee8` — `docs(research): summarize visual AI status after registration`
- `3bf83f0` — `fix analysis`
**Files:**
- `zeromodel/visual_registration.py`
- `zeromodel/visual_local_baselines.py`
- `examples/arcade_visual_local_baseline_showdown.py`
- `examples/arcade_visual_local_baseline_postanalysis.py`
- `docs/results/visual-local-baseline-showdown-v1/`
- `docs/results/visual-local-baseline-showdown-v1-postanalysis/`
**Reports:**
- `docs/research/visual-local-baseline-showdown.md`
- `docs/research/visual-ai-research-status-after-registration.md`
- `docs/results/visual-local-baseline-showdown-v1/README.md`
- `docs/results/visual-local-baseline-showdown-v1-postanalysis/README.md`
**Tests:**
- `tests/test_visual_registration.py`
- `tests/test_visual_local_baselines.py`
- `tests/test_visual_local_baseline_result_records.py`
- `tests/test_arcade_visual_local_baseline_showdown.py`
- `tests/test_arcade_visual_local_baseline_postanalysis.py`
**Classification:** A. Major research transition
**Logbook entry:** Yes
**Notes:** Registration improved raw top-1 exact-row and action performance, repaired the held-out two-pixel translation family at raw top-1, but still yielded zero final benign accepted coverage.

### Fresh local-evidence and independent registered-calibration work lands as preparation only

**Date range:** July 18, 2026
**Mainline merge or delivery commit:** `336f7c4` — `stage 3`
**Supporting commits:**
- none treated as separate research episodes
**Files:**
- `examples/arcade_visual_local_evidence_benchmark.py`
- `examples/arcade_visual_registered_calibration_v2.py`
- `tests/test_visual_local_evidence_benchmark.py`
- `tests/test_arcade_visual_registered_calibration_v2.py`
- `tests/test_visual_registered_calibration_v2.py`
**Reports:**
- none
**Tests:**
- `tests/test_visual_local_evidence_benchmark.py`
- `tests/test_arcade_visual_registered_calibration_v2.py`
- `tests/test_visual_registered_calibration_v2.py`
**Classification:** C. Operationally important
**Logbook entry:** Yes
**Notes:** Implemented and tested only. No committed final evidence directory or adjudication exists on `main`, so this remained a preparation episode rather than a measured result.

### First visual chapter logbook and status cut recorded

**Date range:** July 18, 2026
**Mainline merge or delivery commit:** `bc99e04` — `docs(research): record visual programme logbook and status cut`
**Supporting commits:**
- none
**Files:**
- `docs/research/visual-programme-status-cut-2026-07-18.md`
- `docs/research/visual-research-logbook.md`
**Reports:**
- `docs/research/visual-programme-status-cut-2026-07-18.md`
- `docs/research/visual-research-logbook.md`
**Tests:**
- none
**Classification:** B. Supporting implementation or evidence
**Logbook entry:** Revised by this task
**Notes:** Documentation-only closeout; no frozen evidence, source, or test content changed in that commit.

## Commits omitted from narrative

| SHA | Subject | Reason omitted |
|---|---|---|
| `1f00191` | `version bump` | Packaging and release metadata only. |
| `9884db7` | `fix version and add agents markdown` | Version metadata and agent instructions only. |
| `17467e1` | `ruff` | Small source cleanups without research effect. |
| `5c39c3d` | `main` | Added sign-demo assets; supportive but not a separate research transition. |
| `4df3913` | `add blog post` | Draft prose only. |
| `1de70a1` | `update blog` | Draft prose only. |
| `8a5e3f2` | `Merge branch 'main' of https://github.com/ernanhughes/zeromodel` | Delivery-only merge with no distinct research meaning. |
| `c259ce3` | `format` | Formatting-only draft edit. |
| `b237ed0` | `Merge branch 'main' of https://github.com/ernanhughes/zeromodel` | Delivery-only merge with no distinct research meaning. |
| `b8c9a9d` | `review` | Added external-review brief and a comparator test, but the research meaning was absorbed into later Phase 1 adjudication. |
| `0d53f75` | `Merge optimized policy lookup and Lua edge fixture` | Important package work, but not part of the visual research change in understanding. |
| `367cde2` | `Merge Bertin pattern discovery MVP` | Separate pattern-discovery line, outside the visual-address narrative. |
| `38b8c36` | `fix: enforce pattern alpha and complete lineage materialization` | Follow-up to pattern discovery; outside this logbook’s scope. |
| `bae5dff` | `api: expose complete pattern discovery artifact set` | Follow-up to pattern discovery; outside this logbook’s scope. |
| `77d916e` | `test: enforce alpha identity and complete pattern lineage` | Follow-up to pattern discovery; outside this logbook’s scope. |
| `c51dd53` | `example: materialize complete pattern artifact lineage` | Follow-up to pattern discovery; outside this logbook’s scope. |
| `a8c4da5` | `docs: close pattern alpha and lineage contract gaps` | Follow-up to pattern discovery; outside this logbook’s scope. |
| `4c263bc` | `docs: synchronize claims audit with Lua and pattern capabilities` | Claims-audit maintenance tied to omitted pattern/Lua work. |
| `3f9ed6f` | `ci: enforce claims-audit participation on package PRs` | CI policy only. |
| `bb2985e` | `later` | Branch-local staging commit; meaning captured by merge `58b7a34`. |
| `fe75e5b` | `becnhmark script` | Branch-local implementation step absorbed into Phase 1 episode. |
| `0516ad9` | `results` | Branch-local evidence staging absorbed into Phase 1 closure episode. |
| `895c573` | `docs of current status` | Intermediate synthesis superseded by later adjudication documents. |
| `4e49499` | `add research` | Branch-local doc staging absorbed into Phase 1 closure merge. |
| `7d8af01` | `upgrade` | Branch-local preparation absorbed into System B repair episode. |
| `6791095` | `latest` | Branch-local preparation absorbed into System B repair episode. |
| `537ed82` | `add log` | Draft logbook work superseded by final documentation commit. |
| `ad4a329` | `exclude log` | Draft logbook work superseded by final documentation commit. |
| `9be2c3d` | `original` | Branch-local staging for showdown work; meaning captured by later commits. |
| `530d696` | `script` | Branch-local staging for showdown work; meaning captured by later commits. |
| `783a1dc` | `fix` | Branch-local staging for showdown work; meaning captured by later commits. |
| `832bca7` | `repair` | Branch-local staging for showdown work; meaning captured by later commits. |
| `4d5b3e2` | `visual` | Draft prose only. |
