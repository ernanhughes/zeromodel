"""Export the arcade VPM policy as a dependency-free Lua module.

Run from the repository root:

    python examples/lua_edge_policy.py --output build/lua/arcade_policy.lua
    lua5.4 examples/lua/run_arcade_policy.lua build/lua/arcade_policy.lua
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcade_shooter_policy import ACTIONS, compile_policy_artifact
from zeromodel import VPMPolicyLookup, compiled_plan_id, write_lua_policy


def export_lua_policy(output: str | Path) -> dict[str, object]:
    artifact = compile_policy_artifact()
    reader = VPMPolicyLookup(artifact, action_metric_ids=ACTIONS)
    target = write_lua_policy(reader, output)
    return {
        "artifact_id": artifact.artifact_id,
        "plan_id": compiled_plan_id(reader.to_compiled_plan()),
        "row_count": len(artifact.source.row_ids),
        "action_count": len(reader.action_metric_ids),
        "output": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the ZeroModel arcade policy for a tiny Lua runtime."
    )
    parser.add_argument(
        "--output",
        default="build/lua/arcade_policy.lua",
        help="Path for the generated Lua module.",
    )
    args = parser.parse_args()
    print(json.dumps(export_lua_policy(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
