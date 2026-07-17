# Release process

ZeroModel 1.0.11 should reach PyPI only after the package install path, the
criticality/verification fixtures, and the claims-audit posture are all proven
by CI and a clean-environment smoke test.

The validated public headline remains:

> ZeroModel turns scored data into deterministic, inspectable Visual Policy Map artifacts and small consumers that can operate without a model at decision time.

ZeroModel 1.0.11 adds a narrower research-backed claim:

> A Q-bearing finite policy can preserve criticality and decision-margin evidence without allowing those evidence columns to participate in action selection. Named row-level properties can be checked exhaustively and recorded as verification artifacts linked to the exact policy artifact checked.

Do not describe best-minus-worst score spread as VIPER-style criticality unless
the source columns carry Q-values or an equivalent consequence-bearing teacher
signal. Do not describe the finite row checker as general formal verification of
continuous dynamics, temporal safety, or universal policy correctness.

Do not use stronger claims such as planet-scale traversal, automatic semantic
view learning, task-level decision accuracy improvement for open-world systems,
real-world hallucination detection, or tiny-hardware latency unless the
repository contains the matching benchmark or fixture evidence.

## Version policy

- `1.0.11` is an additive release over the stable 1.0 artifact contract.
- Keep future breaking artifact-contract changes for `2.x`.
- Use pre-release suffixes such as `1.0.12rc1` or `1.1.0rc1` for release candidates when testing PyPI/TestPyPI paths.

## Local build check

From a clean checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
pytest -q
python -m build
python -m twine check dist/*
python examples/arcade_shooter_policy.py
python examples/criticality_verification.py \
  --output-dir docs/assets/criticality-verification
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
  zeromodel==1.0.11
python - <<'PY'
from zeromodel import (
    LayoutRecipe,
    PolicyPropertyChecker,
    PolicyPropertySpec,
    ScoreTable,
    VPMPolicyLookup,
    build_vpm,
    with_q_diagnostics,
)

ACTIONS = ("LEFT", "RIGHT")
source = with_q_diagnostics(
    ScoreTable(
        values=[[2.0, 0.0], [0.0, 2.0]],
        row_ids=["side=left", "side=right"],
        metric_ids=ACTIONS,
    ),
    action_metric_ids=ACTIONS,
)
recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "policy-source-order",
    "row_order": {"kind": "source", "tie_break": "row_id"},
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})
artifact = build_vpm(source, recipe)
reader = VPMPolicyLookup(
    artifact,
    action_metric_ids=ACTIONS,
    evidence_metric_ids=("criticality", "decision_margin"),
)
assert reader.read("side=right").action == "RIGHT"

property_spec = PolicyPropertySpec.from_dict({
    "id": "right_state_moves_right",
    "version": "1",
    "assert": {
        "implies": [
            {"eq": [{"var": "state.side"}, "right"]},
            {"eq": [{"var": "winner"}, "RIGHT"]},
        ]
    },
})
report = PolicyPropertyChecker(
    artifact,
    action_metric_ids=ACTIONS,
    evidence_metric_ids=("criticality", "decision_margin"),
).check([property_spec])
assert report.passed is True
assert report.to_vpm().provenance["parents"][0]["artifact_id"] == artifact.artifact_id
print("zeromodel 1.0.11 criticality and verification smoke test passed")
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
- the tiny arcade-shooter example runs,
- the criticality/verification example produces one exact counterexample and a passing repaired artifact,
- verification artifacts preserve identity across `.vpm` round-trip, and
- `docs/claims-audit.md` still matches the public wording.
