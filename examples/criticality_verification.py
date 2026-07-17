"""Criticality-aware policy and linked verification artifact demo.

This example extends the bounded arcade-shooter fixture with Q-bearing teacher
values, derived criticality and decision-margin evidence, exhaustive finite
property checks, counterexample localization, repair, and re-verification.

Run:

    python examples/criticality_verification.py \
        --output-dir docs/assets/criticality-verification
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

from zeromodel import (
    LayoutRecipe,
    PolicyPropertyChecker,
    PolicyPropertySpec,
    ScoreTable,
    VPMPolicyLookup,
    build_vpm,
    to_bundle,
    with_q_diagnostics,
    write_png,
)

ACTIONS: Tuple[str, ...] = ("LEFT", "RIGHT", "STAY", "FIRE")
EVIDENCE_METRICS: Tuple[str, ...] = ("criticality", "decision_margin")


def state_row_id(tank_x: int, target_x: Optional[int], cooldown: int) -> str:
    target = "none" if target_x is None else str(int(target_x))
    return "tank=%s|target=%s|cooldown=%s" % (
        int(tank_x),
        target,
        int(cooldown),
    )


def teacher_q_values(
    tank_x: int,
    target_x: Optional[int],
    cooldown: int,
) -> tuple[float, ...]:
    """Return consequence-bearing teacher values in ACTIONS order."""

    if target_x is None:
        return (0.8, 0.8, 1.0, -1.0)

    if cooldown == 0 and tank_x == target_x:
        return (-1.0, -1.0, 0.0, 5.0)

    if tank_x > target_x:
        return (3.0, -1.0, 0.5, -2.0)

    if tank_x < target_x:
        return (-1.0, 3.0, 0.5, -2.0)

    return (0.5, 0.5, 2.0, -2.0)


def _source_table(
    *,
    width: int = 7,
    unsafe_fire_row: str | None = None,
) -> ScoreTable:
    row_ids: list[str] = []
    values: list[tuple[float, ...]] = []
    targets: tuple[Optional[int], ...] = (None,) + tuple(range(width))

    for tank_x in range(width):
        for target_x in targets:
            for cooldown in (0, 1):
                row_id = state_row_id(tank_x, target_x, cooldown)
                q_values = list(
                    teacher_q_values(tank_x, target_x, cooldown)
                )
                if row_id == unsafe_fire_row:
                    q_values[ACTIONS.index("FIRE")] = max(q_values) + 1.0
                row_ids.append(row_id)
                values.append(tuple(q_values))

    return with_q_diagnostics(
        ScoreTable(
            values=values,
            row_ids=row_ids,
            metric_ids=ACTIONS,
            metadata={
                "kind": "arcade_shooter_q_policy",
                "world": "tiny_arcade_shooter",
                "addressing": "tank_x,target_x,cooldown",
                "teacher_values": "handcrafted_q_bearing_fixture",
            },
        ),
        action_metric_ids=ACTIONS,
    )


def compile_q_policy_artifact(
    *,
    width: int = 7,
    unsafe_fire_row: str | None = None,
    parent_artifact_id: str | None = None,
    criticality_first: bool = False,
):
    """Compile one Q-bearing policy surface with diagnostic evidence columns."""

    table = _source_table(
        width=width,
        unsafe_fire_row=unsafe_fire_row,
    )
    row_order = (
        {
            "kind": "weighted_score",
            "metrics": [
                {
                    "metric_id": "criticality",
                    "direction": "desc",
                    "weight": 1.0,
                }
            ],
            "tie_break": "row_id",
        }
        if criticality_first
        else {"kind": "source", "tie_break": "row_id"}
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": (
                "arcade-q-policy-criticality-first"
                if criticality_first
                else "arcade-q-policy-source-order"
            ),
            "row_order": row_order,
            "column_order": {"kind": "source"},
            "normalization": {
                "kind": "per_metric_minmax",
                "clip": True,
            },
        }
    )
    parents = (
        []
        if parent_artifact_id is None
        else [
            {
                "artifact_id": parent_artifact_id,
                "relation": "derived_from",
            }
        ]
    )
    return build_vpm(
        table,
        recipe,
        provenance={
            "kind": "criticality_aware_compiled_policy",
            "consumer": "VPMPolicyLookup",
            "parents": parents,
        },
    )


FIRE_REQUIRES_ALIGNMENT = PolicyPropertySpec.from_dict(
    {
        "id": "fire_requires_alignment_and_ready",
        "version": "1",
        "description": "FIRE may win only when tank and target align and cooldown is zero.",
        "assert": {
            "implies": [
                {"eq": [{"var": "winner"}, "FIRE"]},
                {
                    "all": [
                        {
                            "eq": [
                                {"var": "state.tank"},
                                {"var": "state.target"},
                            ]
                        },
                        {"eq": [{"var": "state.cooldown"}, 0]},
                    ]
                },
            ]
        },
    }
)

NO_TARGET_REQUIRES_STAY = PolicyPropertySpec.from_dict(
    {
        "id": "no_target_requires_stay",
        "version": "1",
        "description": "When no target exists, STAY must win.",
        "assert": {
            "implies": [
                {"eq": [{"var": "state.target"}, None]},
                {"eq": [{"var": "winner"}, "STAY"]},
            ]
        },
    }
)

PROPERTIES = (FIRE_REQUIRES_ALIGNMENT, NO_TARGET_REQUIRES_STAY)


def verify_policy(artifact):
    checker = PolicyPropertyChecker(
        artifact,
        action_metric_ids=ACTIONS,
        evidence_metric_ids=EVIDENCE_METRICS,
    )
    return checker.check(PROPERTIES)


def run_demo(output_dir: str | Path) -> dict[str, object]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    original = compile_q_policy_artifact()
    criticality_view = compile_q_policy_artifact(
        parent_artifact_id=original.artifact_id,
        criticality_first=True,
    )
    original_report = verify_policy(original)
    original_verification = original_report.to_vpm()

    unsafe_row = state_row_id(0, 1, 0)
    unsafe = compile_q_policy_artifact(
        unsafe_fire_row=unsafe_row,
        parent_artifact_id=original.artifact_id,
    )
    failed_report = verify_policy(unsafe)
    failed_verification = failed_report.to_vpm()

    repaired = compile_q_policy_artifact(
        parent_artifact_id=unsafe.artifact_id,
    )
    repaired_report = verify_policy(repaired)
    repaired_verification = repaired_report.to_vpm()

    artifacts = {
        "policy_original": original,
        "policy_criticality_view": criticality_view,
        "verification_original": original_verification,
        "policy_unsafe": unsafe,
        "verification_failed": failed_verification,
        "policy_repaired": repaired,
        "verification_repaired": repaired_verification,
    }
    for name, artifact in artifacts.items():
        to_bundle(artifact, target / ("%s.vpm" % name))

    write_png(original, target / "policy_original.png")
    write_png(criticality_view, target / "policy_criticality_view.png")
    write_png(failed_verification, target / "verification_failed.png")
    write_png(repaired_verification, target / "verification_repaired.png")

    critical_decision = VPMPolicyLookup(
        original,
        action_metric_ids=ACTIONS,
        evidence_metric_ids=EVIDENCE_METRICS,
    ).read(state_row_id(3, 3, 0))

    violations = [
        violation.to_dict()
        for result in failed_report.results
        for violation in result.violations
    ]
    result = {
        "version": "1.0.11",
        "original_policy_id": original.artifact_id,
        "criticality_view_id": criticality_view.artifact_id,
        "unsafe_policy_id": unsafe.artifact_id,
        "repaired_policy_id": repaired.artifact_id,
        "original_verification_id": original_verification.artifact_id,
        "failed_verification_id": failed_verification.artifact_id,
        "repaired_verification_id": repaired_verification.artifact_id,
        "original_passed": original_report.passed,
        "unsafe_passed": failed_report.passed,
        "repaired_passed": repaired_report.passed,
        "counterexamples": violations,
        "critical_decision": critical_decision.to_dict(),
    }
    (target / "criticality_verification_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="docs/assets/criticality-verification",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_demo(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
