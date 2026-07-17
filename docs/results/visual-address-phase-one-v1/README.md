# Phase 1 full visual-address run: v1 evidence record

**Status:** measured historical research result; partially recovered provenance package  
**Original run date:** July 17, 2026  
**Recovery decision date:** July 17, 2026  
**Historical dataset digest:** `91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e`  
**Historical report digest:** `d7d0b4db13c9f96b2ac4583aae25fa2159fb62471fb90110103548317f084035`

## Preserved historical files

The following original local-run files were recovered from:

```text
build/visual-phase-one-local/
```

and attached unchanged under:

```text
docs/results/visual-address-phase-one-v1/recovered-originals/
```

Recovered files:

- `arcade-visual-phase-one.json`
- `command.txt`
- `dino-full-run.log`
- `runtime.txt`

The recovered raw JSON has:

- embedded benchmark-report digest matching the historical report digest exactly;
- embedded `dataset_manifest_digest` matching the historical dataset digest exactly.

## Recovery caveat

The current repository code no longer recomputes the same dataset-manifest digest
from the recovered raw manifest bytes, because dataset semantics and hashing
contracts changed in later corrective work. The recovered raw file should
therefore be treated as the authoritative historical artifact, and the embedded
historical digest should be used for verification rather than re-hashing the
manifest through newer code paths.

## Still-missing original files

The following original local-run files were not recovered:

- `argv.json`
- `environment.json`
- `run-manifest.json`

This means the repository now contains the original raw report and partial
runtime provenance, but not the complete original environment bundle.

## Command provenance defect

The recovered `command.txt` and `runtime.txt` confirm the previously disclosed
PowerShell runner defect:

- `command.txt` recorded `--device` with an empty value;
- `runtime.txt` recorded `device=` with an empty value;
- the executed run nevertheless completed successfully.

This command/device text is original evidence and must remain unchanged. It is
not a faithful device provenance record.

## Permitted and prohibited uses

Permitted:

- preserve the historical run that triggered the Phase 1 continuation decision;
- verify the quoted benchmark-report digest against the recovered raw JSON;
- inspect historical per-observation traces and family counts from the recovered
  raw file;
- compare later corrected reruns against the historical v1 result.

Prohibited:

- present Phase 1 v1 as a fully reproducible environment-complete benchmark;
- claim that the original run’s device and command provenance were captured
  faithfully;
- overwrite or silently replace this result with a corrected rerun.

## Historical integrity

Do not overwrite this v1 result after code or calibration fixes.

A corrected rerun must remain separately identified, for example:

```text
docs/results/visual-address-system-b-v2/
```

The historical v1 result remains important because it is the run that fired the
predeclared Phase 1 kill conditions.
