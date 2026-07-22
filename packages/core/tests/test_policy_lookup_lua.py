from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from zeromodel.core import (
    LayoutRecipe,
    POLICY_PLAN_VERSION,
    ScoreTable,
    SignReader,
    VPMPolicyLookup,
    VPMValidationError,
    build_vpm,
    compiled_plan_id,
    lua_policy_source,
    write_lua_policy,
)


def _policy_artifact():
    table = ScoreTable(
        values=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0]],
        row_ids=["state:left", "state:right", "state:aligned"],
        metric_ids=["LEFT", "RIGHT", "STAY", "FIRE"],
        metadata={"kind": "toy_policy"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "policy-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe)


def test_policy_lookup_reads_best_action_and_cell_proof() -> None:
    artifact = _policy_artifact()
    reader = VPMPolicyLookup(artifact)
    decision = reader.read("state:aligned")

    assert decision.action == "FIRE"
    assert decision.metric_id == "FIRE"
    assert decision.value == pytest.approx(1.0)
    assert decision.source_row_index == 2
    assert decision.source_metric_index == 3
    assert decision.view_row == 2
    assert decision.view_column == 3
    assert decision.candidates == {"LEFT": 0.0, "RIGHT": 0.0, "STAY": 0.5, "FIRE": 1.0}
    assert SignReader is VPMPolicyLookup


def test_policy_lookup_limits_actions_and_rejects_invalid_inputs() -> None:
    artifact = _policy_artifact()
    assert (
        VPMPolicyLookup(artifact, action_metric_ids=["LEFT", "RIGHT", "STAY"])
        .read("state:aligned")
        .action
        == "STAY"
    )

    with pytest.raises(VPMValidationError, match="Unknown policy row_id"):
        VPMPolicyLookup(artifact).read("missing")
    with pytest.raises(VPMValidationError, match="Unknown action metric ids"):
        VPMPolicyLookup(artifact, action_metric_ids=["JUMP"])


def test_compiled_plan_identity_and_diagnostics_metadata() -> None:
    artifact = _policy_artifact()
    reader = VPMPolicyLookup(
        artifact, action_metric_ids=["LEFT", "RIGHT"], evidence_metric_ids=["STAY"]
    )
    plan = reader.to_compiled_plan()

    assert plan["format"] == POLICY_PLAN_VERSION
    assert plan["artifact_id"] == artifact.artifact_id
    assert plan["evidence_metric_ids"] == ["STAY"]
    assert len(compiled_plan_id(plan)) == 64
    assert set(reader.read("state:aligned").evidence) == {"STAY"}


def test_lua_policy_source_is_deterministic_and_identity_linked(tmp_path: Path) -> None:
    reader = VPMPolicyLookup(_policy_artifact())
    first = lua_policy_source(reader)
    target = write_lua_policy(reader, tmp_path / "policy.lua")

    assert first == lua_policy_source(reader) == target.read_text(encoding="utf-8")
    assert reader.artifact.artifact_id in first
    assert compiled_plan_id(reader.to_compiled_plan()) in first


def test_lua_policy_executes_when_lua_is_available(tmp_path: Path) -> None:
    executable = shutil.which("lua5.4") or shutil.which("lua")
    if executable is None:
        pytest.skip("Lua interpreter is not installed")
    policy_path = write_lua_policy(
        VPMPolicyLookup(_policy_artifact()), tmp_path / "policy.lua"
    )
    runner = tmp_path / "runner.lua"
    runner.write_text(
        'local policy = dofile(arg[1])\nassert(policy.choose("state:left") == "LEFT")\nprint(policy.artifact_id)\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [executable, str(runner), str(policy_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert _policy_artifact().artifact_id in completed.stdout
