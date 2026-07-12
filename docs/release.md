# Release process

ZeroModel should reach PyPI in two steps:

1. Publish the current alpha package to TestPyPI.
2. Verify install/import behavior from TestPyPI in a clean environment before promoting the same release posture to production PyPI.

The release is intentionally alpha. The validated public claim is:

> ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts.

Do not use stronger claims such as planet-scale traversal, automatic semantic view learning, task-level decision accuracy improvement, real-world hallucination detection, or real training-run validation unless the repository contains the matching benchmark or fixture evidence.

## Version policy

Use pre-release versions for release candidates until the package install path is proven:

- `0.1.1a1` for the first TestPyPI alpha.
- `0.1.0a2`, `0.1.0a3`, etc. for packaging fixes.
- `0.1.0` only after TestPyPI install/import checks pass.

Do not publish the current package as `2.0.0` unless there is a deliberate historical compatibility reason.

## Local build check

From a clean checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
pytest -q
python -m build
python -m twine check dist/*
```

The GitHub `Python package` workflow runs the test matrix and distribution metadata checks on pull requests.

## TestPyPI trusted publishing setup

Before running `.github/workflows/publish-testpypi.yml`, configure a trusted publisher on TestPyPI with these values:

- Owner: `ernanhughes`
- Repository: `zeromodel`
- Workflow name: `publish-testpypi.yml`
- Environment name: `testpypi`

The workflow uses GitHub OIDC (`id-token: write`) and does not require a long-lived API token.

## Publish to TestPyPI

1. Merge the release candidate PR.
2. Open GitHub Actions.
3. Run **Publish TestPyPI release candidate** manually from the target branch or tag.
4. Confirm the workflow builds both `sdist` and wheel, checks metadata, and uploads to TestPyPI.

## Verify TestPyPI install

Use a clean virtual environment. Because TestPyPI may not host dependencies such as NumPy, keep PyPI as an extra dependency index:

```bash
python -m venv .venv-testpypi
. .venv-testpypi/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  zeromodel==0.1.1a1
python - <<'PY'
from zeromodel import LayoutRecipe, ScoreTable, build_vpm

score_table = ScoreTable(
    values=[[0.9, 0.2], [0.4, 0.8]],
    row_ids=["candidate-a", "candidate-b"],
    metric_ids=["quality", "uncertainty"],
)
recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "quality-first",
    "row_order": {
        "kind": "lexicographic",
        "keys": [{"metric_id": "quality", "direction": "desc"}],
        "tie_break": "row_id",
    },
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})
artifact = build_vpm(score_table, recipe)
assert artifact.cell(0, 0).row_id == "candidate-a"
print("zeromodel import/build smoke test passed")
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
- README install instructions are updated for the production release, and
- `docs/claims-audit.md` still matches the public wording.

The first production release should probably be `0.1.0`, not `2.0.0`.
