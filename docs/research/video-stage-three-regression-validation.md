# Video Stage Three Regression Validation

Date: July 18, 2026

Status: local regression classification for the Stage 2 timeout observed during Stage 3 development

## Environment

- repository: `C:\Projects\zeromodel`
- branch during classification: `research/video-discriminative-local-evidence-v1`
- local date: July 18, 2026
- Python: local `python` interpreter used by the existing test suite

## Commands and observed behavior

Collected tests:

```powershell
python -m pytest tests/test_video_local_correlation.py --collect-only -q
```

Result:

- collected `7` tests in about `0.18s`

Whole file under the local execution window:

```powershell
python -m pytest tests/test_video_local_correlation.py -vv --durations=0
```

Result:

- local command timed out after about `124s`

Per-test classification:

```powershell
python -m pytest tests/test_video_local_correlation.py::test_local_correlation_provider_is_deterministic -q
```

- passed in about `0.37s`

```powershell
python -m pytest tests/test_video_local_correlation.py::test_local_correlation_provider_separates_identical_pixels_across_clips_in_cache_identity -q
```

- passed in about `0.49s`

```powershell
python -m pytest tests/test_video_local_correlation.py::test_local_correlation_provider_rejects_geometry_mismatch -q
```

- passed in about `0.20s`

```powershell
python -m pytest tests/test_video_local_correlation.py::test_local_correlation_provider_retains_conflicting_action_candidate -q
```

- passed in about `0.31s`

```powershell
python -m pytest tests/test_video_local_correlation.py::test_video_benchmark_generation_is_deterministic -q
```

- passed in about `0.22s`

```powershell
python -m pytest tests/test_video_local_correlation.py::test_video_benchmark_reordered_case_has_distinct_identity -q
```

- passed in about `0.26s`

```powershell
python -m pytest tests/test_video_local_correlation.py::test_stage2_harness_freezes_and_verifies_negative_result -vv
```

- local command timed out after about `124s`

Calibration phase isolation:

```powershell
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from research.video_action_set.benchmarks.arcade_visual_video_local_correlation_benchmark import run_calibrate
with TemporaryDirectory() as d:
    run_calibrate(output_dir=Path(d))
PY
```

- local command timed out after about `124s`

Package-root import check:

```powershell
python - <<'PY'
import time
start = time.perf_counter()
import zeromodel
print(time.perf_counter() - start)
PY
```

- local `import zeromodel` completed in about `0.20s`

## Root-cause classification

Current local evidence supports this classification:

- the timeout is not an import-time Stage 3 regression
- the timeout is not a Stage 2 provider semantic failure in the fast unit tests
- the timeout is localized to the frozen Stage 2 benchmark calibration path
- the timeout is therefore currently classified as a local runtime-window limitation on an existing expensive calibration/harness path

This note does not claim that the frozen Stage 2 harness is fast. It only records that the observed local timeout was not reproduced as a broad Stage 2 semantic regression in the provider-level tests that completed quickly.

## Stage 2 semantics

No Stage 2 semantic change was intentionally introduced.

Additional explicit regression coverage was added in:

- [tests/test_visual_registration.py](/C:/Projects/zeromodel/tests/test_visual_registration.py)

That regression test preserves the diagnosed tiny-region overlap pathology in the frozen Stage 2 registration layer.

## Corrective action

No Stage 2 assertions were weakened.

No skip markers were added.

No sleeps or timeout inflation were added as a substitute for diagnosis.

The Stage 3 work instead:

- measured package-root import behavior
- isolated the slow test to the Stage 2 benchmark harness
- isolated the slow benchmark phase further to `run_calibrate(...)`
- kept Stage 2 semantics unchanged while implementing the Stage 3 V4 provider in a separate module layer

## Residual limitation

This local note does not claim a completed full-file pass for:

```powershell
python -m pytest tests/test_video_local_correlation.py -q
```

under the same approximately `124s` local execution window.

At the end of this block, the timeout remains reproducible on the frozen Stage 2 calibration path in this local environment, so that command must not be reported as passed without a longer-running environment or separate CI evidence.
