from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    build_vpm,
)
from zeromodel.core.lua import (
    compiled_plan_id,
    lua_policy_source,
    write_lua_policy,
)
from zeromodel.core.policy_lookup import VPMPolicyLookup


def _reader() -> VPMPolicyLookup:
    artifact = build_vpm(
        ScoreTable(
            values=[[1.0, 0.0], [0.0, 1.0]],
            row_ids=["state:left", "state:right"],
            metric_ids=["LEFT", "RIGHT"],
        ),
        LayoutRecipe.from_dict(
            {
                "version": "vpm-layout/0",
                "name": "lua-policy-test",
                "row_order": {"kind": "source", "tie_break": "row_id"},
                "column_order": {"kind": "source"},
                "normalization": {
                    "kind": "per_metric_minmax",
                    "clip": True,
                },
            }
        ),
    )
    return VPMPolicyLookup(artifact)


def test_lua_policy_source_is_deterministic_and_identity_linked(tmp_path: Path) -> None:
    reader = _reader()

    first = lua_policy_source(reader)
    second = lua_policy_source(reader)
    target = write_lua_policy(reader, tmp_path / "policy.lua")

    assert first == second == target.read_text(encoding="utf-8")
    assert reader.artifact.artifact_id in first
    assert compiled_plan_id(reader.to_compiled_plan()) in first
    assert "function policy.choose(row_id)" in first
    assert "function policy.read(row_id)" in first


def test_lua_policy_executes_when_lua_is_available(tmp_path: Path) -> None:
    executable = shutil.which("lua5.4") or shutil.which("lua")
    if executable is None:
        pytest.skip("Lua interpreter is not installed")

    policy_path = write_lua_policy(_reader(), tmp_path / "policy.lua")
    runner = tmp_path / "runner.lua"
    runner.write_text(
        """
local policy = dofile(arg[1])
assert(policy.choose("state:left") == "LEFT")
assert(policy.choose("state:right") == "RIGHT")
local decision = policy.read("state:right")
assert(decision.action == "RIGHT")
assert(decision.value == 1.0)
assert(decision.source_row_index == 1)
assert(decision.source_metric_index == 1)
assert(decision.view_row == 1)
assert(decision.view_column == 1)
assert(decision.candidates.LEFT == 0.0)
assert(decision.candidates.RIGHT == 1.0)
print(policy.artifact_id .. " " .. policy.plan_id)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [executable, str(runner), str(policy_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert _reader().artifact.artifact_id in completed.stdout
