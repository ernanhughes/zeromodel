"""Generate VisualSignReader acceptance-profile evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.vision import (
    VisualAcceptanceProfile,
    VisualFeatureSpec,
    VisualSignReader,
    build_visual_index,
)


ACTIONS = ("A", "B")


def _policy():
    table = ScoreTable(
        [[1.0, 0.0], [0.0, 1.0], [0.4, 0.8]],
        ["left", "right", "stay"],
        ACTIONS,
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "visual-profile-audit",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    policy = _policy()
    spec = VisualFeatureSpec(1, 1, 1, 1, quantization_levels=16)
    index = build_visual_index(
        policy,
        {
            "left": np.array([[0]], dtype=np.uint8),
            "right": np.array([[128]], dtype=np.uint8),
            "stay": np.array([[255]], dtype=np.uint8),
        },
        spec,
    )
    reader = VisualSignReader(index.artifact, policy, action_metric_ids=ACTIONS)
    cases = {
        "canonical": np.array([[0]], dtype=np.uint8),
        "exact_alias": np.array([[1]], dtype=np.uint8),
        "approximate": np.array([[17]], dtype=np.uint8),
        "distant": np.array([[90]], dtype=np.uint8),
    }
    results = {}
    for case_name, frame in cases.items():
        results[case_name] = {}
        for profile in sorted(VisualAcceptanceProfile._ALLOWED):
            results[case_name][profile] = reader.read(
                frame,
                acceptance_profile=profile,
            ).to_dict()

    Path(args.output).write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
