import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from zeromodel.core.artifact import VPMValidationError
from research.video_action_set import mutation_filesystem


def test_directory_snapshot_is_ordered_and_content_addressed(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "same.txt").write_text("beta\n", encoding="utf-8")
    (tmp_path / "a" / "same.txt").write_text("alpha\n", encoding="utf-8")

    snapshot = mutation_filesystem._directory_snapshot(tmp_path)

    assert list(snapshot) == ["a/same.txt", "b/same.txt"]
    expected = f"alpha{os.linesep}".encode()
    assert snapshot["a/same.txt"]["size"] == len(expected)
    assert (
        snapshot["a/same.txt"]["digest"]
        == "sha256:" + hashlib.sha256(expected).hexdigest()
    )


def test_mutation_application_changes_only_copied_case(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    case = tmp_path / "case"
    baseline.mkdir()
    payload = {
        "policy_artifact_id": "original",
        "row_ids": ["r0"],
        "row_action": {"r0": "LEFT"},
        "row_action_digest": "old",
    }
    (baseline / "policy-artifact.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    shutil.copytree(baseline, case)

    mutation_filesystem._apply_reference_mutation(
        case, "policy_alter_artifact_identity"
    )

    assert json.loads((baseline / "policy-artifact.json").read_text()) == payload
    assert (
        json.loads((case / "policy-artifact.json").read_text())["policy_artifact_id"]
        == "sha256:foreign-policy"
    )
    with pytest.raises(VPMValidationError, match="unsupported mutation case"):
        mutation_filesystem._apply_reference_mutation(case, "unknown_case")
