# Video Action-Set Stage 8 Runbook

This runbook describes the live Stage 7C instrument. It does not authorize or execute Stage 8. Commands that call verification or mutation APIs print JSON to stdout because the current CLI does not persist those return-only reports.

## Preconditions

- Work from a clean `main` containing merged Stage 7C and Stage 7B commit `541716d5ad55f22466837e7f5af2321b8ba49377`.
- Use Python 3.10, 3.11, or 3.12 from the repository root.
- Install the package with development and persistence dependencies: `python -m pip install -e ".[dev,persistence]"`.
- Choose a new or intentionally discarded output directory. There is no resume or repair protocol.
- Ensure the output filesystem can hold SQLite, all frame metadata, and all provider evidence. The code exposes no reliable free-space estimate.
- The final split must remain plan-only and sealed. `build_split("final", ...)` is prohibited by the live code.

Set the execution context once in PowerShell:

```powershell
$Repo = (Resolve-Path .).Path
$Out = Join-Path $Repo "build/stage8-video-action-set"
$env:ZEROMODEL_REPO = $Repo
$env:ZEROMODEL_OUT = $Out
```

## Command Inventory

### Test Execution Tiers

Fast unit tests are the default developer check:

```powershell
python scripts/run_fast_tests.py
```

Integration tests require fresh, explicit human authorization:

```powershell
python -m pytest -q --run-integration -m integration
```

Slow tests require fresh, explicit human authorization:

```powershell
python -m pytest -q --run-slow -m slow
```

Tests marked with both tiers require both opt-in flags.

Scientific/manual checks require fresh, explicit human authorization and should
be scripted by coding agents rather than executed opportunistically:

```powershell
python -m pytest -q --run-integration tests/test_video_action_set_reference_verification.py -m "not slow"
python -m pytest -q --run-integration tests/test_installed_wheel_video_instrument.py
```

Coding agents must not execute integration tests, slow tests, scientific builds,
benchmarks, or mutation audits unless the user explicitly authorizes that exact
execution. Agents should implement the code and provide operator commands for
long-running validation. For multi-step long validation, agents should create or
update a script under `scripts/` and leave execution to the operator.

### Controlled Stage 8 Sequence

The controlled sequence freezes exactly once through the operational package CLI:

```powershell
python -m zeromodel.video_action_set_cli `
  --output-dir $Out `
  --freeze-benchmark
```

Build each non-final split directly so the frozen plans and final seal are reused:

```powershell
python -c 'import os; from pathlib import Path; from zeromodel.domains.video_action_set.build_orchestration import build_split; out=Path(os.environ["ZEROMODEL_OUT"]); repo=Path(os.environ["ZEROMODEL_REPO"]); build_split("development", out, repo)'
python -c 'import os; from pathlib import Path; from zeromodel.domains.video_action_set.build_orchestration import build_split; out=Path(os.environ["ZEROMODEL_OUT"]); repo=Path(os.environ["ZEROMODEL_REPO"]); build_split("calibration", out, repo)'
python -c 'import os; from pathlib import Path; from zeromodel.domains.video_action_set.build_orchestration import build_split; out=Path(os.environ["ZEROMODEL_OUT"]); repo=Path(os.environ["ZEROMODEL_REPO"]); build_split("selection", out, repo)'
```

Continue with the remaining operational CLI commands:

```powershell
python -m zeromodel.video_action_set_cli --output-dir $Out --profile-runtime --profile-provider all --profile-frame-count 8
python -m zeromodel.video_action_set_cli --output-dir $Out --verify-provider-runtime-equivalence
python -m zeromodel.video_action_set_cli --output-dir $Out --audit-canonical-providers
python -m zeromodel.video_action_set_cli --output-dir $Out --audit-evidence-completeness
python -m zeromodel.video_action_set_cli --output-dir $Out --verify-prospective-instrument
```

### Historical One-Shot Build Flags

The operational CLI retains the historical `--build-development`,
`--build-calibration`, and `--build-selection` convenience flags. Each flag calls
`freeze_benchmark()` before `build_split()`. They are one-shot workflows and must not
be used after the explicit freeze in the controlled Stage 8 sequence.

`zeromodel.video_action_set_benchmark` retains `main` as an import compatibility
export but is intentionally inert when executed with `python -m`. Invoke
`python -m zeromodel.video_action_set_cli` for all operational flags.

The current CLI has no flags for the detailed reference verifier, read-only verifier, mutation audits, standalone matrix, or closure payload. Their exact live API commands are:

```powershell
python -c "import json,os; from pathlib import Path; from zeromodel.domains.video_action_set.verification_orchestration import verify_reference_instrument; print(json.dumps(verify_reference_instrument(Path(os.environ['ZEROMODEL_OUT']),Path(os.environ['ZEROMODEL_REPO'])),indent=2,sort_keys=True))"
python -c "import json,os; from pathlib import Path; from zeromodel.domains.video_action_set.verification_orchestration import verify_reference_read_only; print(json.dumps(verify_reference_read_only(Path(os.environ['ZEROMODEL_OUT']),Path(os.environ['ZEROMODEL_REPO'])),indent=2,sort_keys=True))"
python -c "import json,os; from pathlib import Path; from zeromodel.domains.video_action_set.mutation_orchestration import run_reference_mutation_audit; print(json.dumps(run_reference_mutation_audit(Path(os.environ['ZEROMODEL_OUT']),Path(os.environ['ZEROMODEL_REPO'])),indent=2,sort_keys=True))"
python -c "import json,os; from pathlib import Path; from zeromodel.domains.video_action_set.mutation_orchestration import run_repeated_reference_mutation_audit; print(json.dumps(run_repeated_reference_mutation_audit(Path(os.environ['ZEROMODEL_OUT']),Path(os.environ['ZEROMODEL_REPO'])),indent=2,sort_keys=True))"
python -c "import json,os; from pathlib import Path; from zeromodel.domains.video_action_set.mutation_orchestration import run_reference_mutation_audit; from zeromodel.domains.video_action_set.mutation_matrix import build_mutation_matrix; audit=run_reference_mutation_audit(Path(os.environ['ZEROMODEL_OUT']),Path(os.environ['ZEROMODEL_REPO'])); print(json.dumps(build_mutation_matrix(audit),indent=2,sort_keys=True))"
python -c "import json,os; from pathlib import Path; from zeromodel.domains.video_action_set.mutation_orchestration import build_reference_closure_report; print(json.dumps(build_reference_closure_report(Path(os.environ['ZEROMODEL_OUT']),Path(os.environ['ZEROMODEL_REPO'])),indent=2,sort_keys=True))"
```

No live command unlocks or evaluates final. Do not invent one in Stage 8. Final access requires a separately reviewed and explicitly authorized protocol change.

## Execution Order

1. Freeze the benchmark once. This compiles policy identity, generates all sealed plans, persists episode plans, seals final, and writes root contracts.
2. Build development. Stop on any identity, overlap, persistence, materialization, or digest error.
3. Build calibration. Do not tune provider formulas or architecture from final data.
4. Build selection. Do not touch final and do not revise calibration after selection results are inspected.
5. Profile providers using bounded records. Profiling is operational evidence, not provider selection.
6. Verify reference and optimized provider equivalence.
7. Audit canonical providers and evidence completeness.
8. Run independent reference verification and then read-only verification.
9. Run the mutation audit only against the verified baseline. Run the repeated audit to establish deterministic mutation results.
10. Build the standalone mutation matrix only when independently requested. It is not part of closure.
11. Build closure only after every required verification, read-only, and repeated mutation condition passes.
12. Stop. The live instrument keeps final sealed and inaccessible.

This order prevents calibration or selection leakage, provider tuning on final, mutation testing against an invalid baseline, and closure over unavailable evidence.

## Stop Conditions

Stop immediately when any of the following occurs:

- benchmark, generator, policy, plan, seed, tile, provider, quantizer, or digest identity mismatch;
- duplicate or overlapping episode, frame, observation, or split identity;
- any verification gate is `failed` or `unavailable`;
- provider runtime equivalence reports a mismatch;
- evidence completeness reports missing, orphaned, malformed, or gap-scored rows;
- reference reconstruction differs from stored pixels, scores, rankings, semantics, or reachability;
- read-only verification changes any size, timestamp, or digest;
- a mutation is missed, invokes the wrong primary detector, violates isolation, or produces an application error;
- repeated mutation audits differ;
- closure status is `reference_instrument_correctness_unresolved`;
- any final, calibration-selection, architecture-selection, or tuning access counter is nonzero;
- an unexpected artifact appears or an expected artifact is missing.

Discard the affected output directory before retrying a phase unless the failure was an external command interruption before any write. Production code has no resume, migration, or repair behavior.

## Artifact Manifest

| Relative path | Producer | Primary consumer | Version or identity | Closure required |
| --- | --- | --- | --- | --- |
| `benchmark-contract-identity.json` | freeze | all phases | seed and contract digests | yes |
| `generator-identity.json` | freeze | structural verification | `GENERATOR_VERSION` and seed digest | yes |
| `benchmark-manifest.json` | freeze | structural verification | `BENCHMARK_VERSION`, policy artifact ID | yes |
| `policy-artifact.json` | freeze | builds and verification | policy artifact ID and row-action digest | yes |
| `reachability-tile-reference.json` | freeze | builds and verification | tile version and digest | yes |
| `episode-plan.json` | freeze | non-final builds and gates | `EPISODE_PLAN_VERSION` | yes |
| `final-split-sealed-plan.json` | freeze | access and plan gates | sealed plan digest; plan-only | yes |
| `final-split-sealed-digest.json` | freeze | plan gate | sealed plan digest | yes |
| `phase-access-audits.json` | freeze/build | access gate and closure | `PHASE_ACCESS_VERSION` | yes |
| `<split>/frame-metadata.jsonl` | split build | verification and evidence audit | frame and pixel digests | yes for three non-final splits |
| `<split>/provider-evidence.jsonl` | split build | semantic and reachability gates | score, ranking, semantic, and trace digests | yes for three non-final splits |
| `<split>-manifest.json` | split build | structural checks | frame and provider-evidence digests | yes |
| `<split>-family-closure-report.json` | split build | family checks | `FAMILY_CLOSURE_VERSION` | selection aggregate required |
| `observation-identity-manifest.json` | split build | overlap/completeness audit | observation identity counts | yes |
| `split-overlap-audit.json` | split build | overlap gate | split overlap counts | yes |
| `runtime-profile-*.json` and `.md` | profile | resource planning | runtime profile payload version | no |
| `provider-runtime-equivalence.json` and `.csv` | equivalence | provider verification | provider comparison digest fields | yes |
| `canonical-provider-results.csv` and summary JSON | canonical audit | evidence review | provider versions and score identities | yes |
| `evidence-completeness-summary.json` | completeness audit | closure preparation | evidence audit fields | yes |
| detailed verification, mutation, matrix, and closure payloads | API commands above | operator review | reference, mutation, matrix, and closure versions | closure payload is return-only in the current shell |

## Final Protocol

The current high-level protocol does not permit final evaluation:

- `freeze_benchmark` creates only a sealed final plan;
- the plan declares `materialization_prohibited=True`;
- `_materialize_records("final", ...)` raises;
- no CLI flag unlocks final.

Therefore Stage 8 must stop before final unless a separate reviewed authorization supplies an exact final-access protocol. Under that protocol, final is touched once; no provider tuning, architecture change, or calibration change may follow final access.

## Claim Language

- **Architecture complete**: allowed only for merged Stage 7C with architecture, quality, fast tests, and build passing. It is not a scientific result.
- **Instrument verified**: use only when closure emits `reference_instrument_correct`.
- **Mutation audit passed**: use only when audit status is `passed`, all 91 detection cases and 2 semantic invariants satisfy their contracts, isolation passes, and the repeated audit is deterministic.
- **Scientific claim unresolved**: use when closure emits `reference_instrument_correctness_unresolved` or any gate is failed/unavailable.
- Do not claim `materialization_ready`, `benchmark_utility_verified`, `provider_selected`, `calibration_complete`, or `final_evaluation_complete`; the live closure explicitly lists them as unsupported.
- The live instrument exposes no status that supports “benchmark utility demonstrated” or “scientific claim supported.”

## Resource Planning

- Measured repository fast-test runtime: record the Stage 7C CI result before Stage 8; it is not a benchmark runtime estimate.
- Derived split sizes: development has 112 episodes; calibration has 112 episodes and 448 frames; selection has 252 episodes and 1,008 frames; sealed final has 252 episodes and 1,008 expected frames.
- Derived provider count: three, ordered `P1`, `P2`, `P3`.
- Derived mutation count: 93 total, comprising 91 expected detections and 2 semantic invariants.
- Measured provider runtime: unknown until the runtime-profile command completes. Use its emitted measurements; do not substitute estimates.
- Disk, memory, and total Stage 8 duration: unknown from the repository.

## Recovery Rules

- Progress observer failure during `build_split()`: the observer is called from
  provider measurement after a frame or typed gap is processed, and observer
  exceptions intentionally propagate. At that point the split may already have
  identical SQLite episode plans for the split and SQLite observations,
  matrix blobs, and observation operation chains for materialized records.
  Provider scoring may also have produced provider descriptors and evidence
  rows in memory for the current process, but `frame-metadata.jsonl`,
  `provider-evidence.jsonl`, `<split>-manifest.json`,
  `<split>-family-closure-report.json`, the selection
  `family-closure-report.json`, `observation-identity-manifest.json`,
  `split-overlap-audit.json`, and refreshed `phase-access-audits.json` are
  written only after measurement returns.
- Same-directory retry after progress observer failure is supported only for
  the same split, unchanged code and inputs, and only when none of that split's
  filesystem completion artifacts exist. The durable stores return existing
  identical episode plans, observations, matrix blobs, and operation chains and
  raise on conflicts. If any current split JSONL, split manifest, or family
  closure artifact exists, use a fresh output directory.
- No partially completed split may be treated as valid. Progress output is an
  operator liveness signal, not a scientific completion marker. A split is
  complete only when both split JSONL files, `<split>-manifest.json`, and
  `<split>-family-closure-report.json` exist for that split; selection also
  publishes `family-closure-report.json`. Stage 8 completion still requires the
  later verification, audit, mutation, and closure gates listed in the execution
  order.
- Freeze failure: discard the output directory and restart freeze.
- Split build failure: discard the whole output directory, refreeze, and rebuild preceding splits in order.
- Profiling failure: profiling outputs may be discarded and rerun; scientific artifacts must remain unchanged.
- Verification or audit failure: preserve the failed directory for diagnosis, then create a fresh output directory for any rerun.
- Mutation case failure or nondeterminism: do not continue to closure. Mutation temporary directories are isolated and automatically cleaned.
- Closure unresolved: do not access final. Correct code or inputs through a reviewed change, then restart from a fresh freeze.

## Stage 8 Completion Report

```text
Base commit:
Python version and platform:
Output directory:
Freeze command/result:
Development build result and manifest digest:
Calibration build result and manifest digest:
Selection build result and manifest digest:
Final sealed-plan digest:
Runtime profile result:
Provider equivalence status:
Canonical-provider audit status:
Evidence-completeness status:
Reference verification status and primary failure:
Read-only verification status:
Mutation audit status and counts:
Repeated mutation audit deterministic:
Standalone matrix requested/result:
Closure supported status and digest:
All access counters:
Unexpected artifacts:
Stop conditions encountered:
Final accessed: yes/no
Final access authorization:
Final accessed exactly once:
Post-final tuning or changes: none/details
Permitted claim language used:
Unsupported claims avoided:
Commands executed:
Artifact manifest and digests:
Integration or slow tests executed under explicit authorization:
Unresolved scientific issues:
```
