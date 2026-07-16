# Release process

ZeroModel 1.0.0 should reach PyPI only after the package install path and the
claims-audit posture are both proven by CI and a clean-environment smoke test.

The validated public claim for 1.0.0 is:

> ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts and small consumers that can operate without a model at decision time.

The new 1.0.0 policy-lookup example supports a narrower headline:

> A bounded policy can be compiled into an addressable VPM artifact. Runtime state finds the row, the row says what to do, and the decision can cite the exact artifact cell that produced it.

Do not use stronger claims such as planet-scale traversal, automatic semantic
view learning, task-level decision accuracy improvement for open-world systems,
real-world hallucination detection, or tiny-hardware latency unless the
repository contains the matching benchmark or fixture evidence.

## Version policy

- `1.0.0` is the first stable public API release.
- Keep future breaking artifact-contract changes for `2.x`.
- Use pre-release suffixes such as `1.0.1rc1` or `1.1.0rc1` for release candidates when testing PyPI/TestPyPI paths.

## Local build check

From a clean checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
pytest -q
python -m build
python -m twine check dist/*
python examples/arcade_shooter_policy.py
```

The GitHub `Python package` workflow runs the test matrix and distribution
metadata checks on pull requests.

## TestPyPI trusted publishing setup

Before running `.github/workflows/publish-testpypi.yml`, configure a trusted
publisher on TestPyPI with these values:

- Owner: `ernanhughes`
- Repository: `zeromodel`
- Workflow name: `publish-testpypi.yml`
- Environment name: `testpypi`

The workflow uses GitHub OIDC (`id-token: write`) and does not require a
long-lived API token.

## Publish to TestPyPI

1. Merge the release candidate PR.
2. Open GitHub Actions.
3. Run **Publish TestPyPI release candidate** manually from the target branch or tag.
4. Confirm the workflow builds both `sdist` and wheel, checks metadata, and uploads to TestPyPI.

## Verify TestPyPI install

Use a clean virtual environment. Because TestPyPI may not host dependencies such
as NumPy, keep PyPI as an extra dependency index:

```bash
python -m venv .venv-testpypi
. .venv-testpypi/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  zeromodel==1.0.0
python - <<'PY'
from zeromodel import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm

score_table = ScoreTable(
    values=[[1.0, 0.0], [0.0, 1.0]],
    row_ids=["state:left", "state:right"],
    metric_ids=["LEFT", "RIGHT"],
)
recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "policy-source-order",
    "row_order": {"kind": "source", "tie_break": "row_id"},
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})
artifact = build_vpm(score_table, recipe)
assert VPMPolicyLookup(artifact).read("state:right").action == "RIGHT"
print("zeromodel 1.0.0 policy lookup smoke test passed")
PY
```

On Windows PowerShell, activate with:

```powershell
.venv-testpypi\Scripts\Activate.ps1
```

## Promote to production PyPI

Only cut the production PyPI release after:

- the test matrix is green,
- source and wheel builds pass,
- `twine check` passes,
- TestPyPI install/import works in a clean environment,
- README install instructions are updated for the production release,
- the tiny arcade-shooter example runs, and
- `docs/claims-audit.md` still matches the public wording.
