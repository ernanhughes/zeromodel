"""Compare dictionary and compiled VPM policy lookup paths.

The benchmark separates action-only lookup from trace-rich lookup so the outputs
being compared are explicit.

Run:

    python examples/policy_lookup_benchmark.py --lookups 200000 --repeat 5
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import Callable, Sequence

from arcade_shooter_policy import ACTIONS, compile_policy_artifact
from zeromodel.core.policy_lookup import VPMPolicyLookup


def _time(
    function: Callable[[str], object],
    row_ids: Sequence[str],
    *,
    repeat: int,
) -> float:
    best = float("inf")
    for _ in range(repeat):
        started = perf_counter()
        for row_id in row_ids:
            function(row_id)
        best = min(best, perf_counter() - started)
    return best


def run_benchmark(*, lookups: int = 200_000, repeat: int = 5) -> dict[str, object]:
    if lookups <= 0:
        raise ValueError("lookups must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    artifact = compile_policy_artifact()
    reader = VPMPolicyLookup(artifact, action_metric_ids=ACTIONS)
    row_ids = tuple(
        artifact.source.row_ids[index % len(artifact.source.row_ids)]
        for index in range(lookups)
    )

    value_map = {
        row_id: tuple(float(value) for value in artifact.source.values[row_index])
        for row_index, row_id in enumerate(artifact.source.row_ids)
    }
    action_map = {
        row_id: reader.choose(row_id)
        for row_id in artifact.source.row_ids
    }

    def dict_values_argmax(row_id: str) -> str:
        values = value_map[row_id]
        winner = max(
            range(len(ACTIONS)),
            key=lambda index: (values[index], -index),
        )
        return ACTIONS[winner]

    timings = {
        "dict_precompiled_action": _time(action_map.__getitem__, row_ids, repeat=repeat),
        "dict_values_argmax": _time(dict_values_argmax, row_ids, repeat=repeat),
        "vpm_choose_action_only": _time(reader.choose, row_ids, repeat=repeat),
        "vpm_read_full_trace": _time(reader.read, row_ids, repeat=repeat),
    }
    per_lookup_ns = {
        name: seconds * 1_000_000_000.0 / lookups
        for name, seconds in timings.items()
    }
    return {
        "artifact_id": artifact.artifact_id,
        "rows": len(artifact.source.row_ids),
        "actions": len(ACTIONS),
        "lookups": lookups,
        "repeat": repeat,
        "best_seconds": timings,
        "nanoseconds_per_lookup": per_lookup_ns,
        "vpm_choose_vs_dict_values_argmax": (
            timings["vpm_choose_action_only"] / timings["dict_values_argmax"]
        ),
        "vpm_choose_vs_precompiled_dict": (
            timings["vpm_choose_action_only"] / timings["dict_precompiled_action"]
        ),
        "note": (
            "Action-only and full-trace paths are reported separately; no fixed "
            "performance claim should be made without the recorded environment."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookups", type=int, default=200_000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(lookups=args.lookups, repeat=args.repeat),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
