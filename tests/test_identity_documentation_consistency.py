from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _enforced_core_golden_digest() -> str:
    source = (REPO_ROOT / "packages" / "core" / "tests" / "test_artifact_kernel.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r'GOLDEN_SAMPLE_ARTIFACT_ID = \(\s*"([0-9a-f]{64})"', source)
    assert match is not None, "could not find GOLDEN_SAMPLE_ARTIFACT_ID in test_artifact_kernel.py"
    return match.group(1)


def test_core_validation_doc_matches_the_enforced_golden_digest() -> None:
    digest = _enforced_core_golden_digest()
    doc = (
        REPO_ROOT / "docs" / "architecture" / "package-core-validation-1.0.13.md"
    ).read_text(encoding="utf-8")
    assert digest in doc, (
        f"package-core-validation-1.0.13.md does not contain the enforced golden digest {digest}"
    )
    # The historically-wrong value is allowed to remain ONLY inside the
    # explanatory correction note, never presented as "the" digest again.
    assert "**Expected digest:** `32f8013789e4ff463569e2ccbbdc8c3802bc42c6edeb8ceb361afca9a6025db1`" not in doc
    assert f"**Expected digest:** `{digest}`" in doc


def test_integration_validation_doc_matches_the_enforced_golden_digest() -> None:
    digest = _enforced_core_golden_digest()
    doc = (
        REPO_ROOT / "docs" / "architecture" / "package-integration-validation-1.0.13.md"
    ).read_text(encoding="utf-8")
    assert digest in doc


def test_core_and_integration_docs_agree_on_the_core_golden_digest() -> None:
    core_doc = (
        REPO_ROOT / "docs" / "architecture" / "package-core-validation-1.0.13.md"
    ).read_text(encoding="utf-8")
    integration_doc = (
        REPO_ROOT / "docs" / "architecture" / "package-integration-validation-1.0.13.md"
    ).read_text(encoding="utf-8")
    digest = _enforced_core_golden_digest()
    assert digest in core_doc and digest in integration_doc
