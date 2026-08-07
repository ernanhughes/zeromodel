from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from examples.fixed_camera_state_claims.evidence_compiler import (
    PanelObservationEvidenceCompiler,
)
from examples.fixed_camera_state_claims.panel_renderer import ROOT, render_dataset
from zeromodel.perception import (
    INVALID_OBSERVATION_DECISION,
    REJECT_AMBIGUOUS_DECISION,
    StateSpecification,
    build_policy_compatibility_report,
    build_state_claim_set,
)


def _states() -> tuple[StateSpecification, ...]:
    rows = json.loads((ROOT / "states.json").read_text(encoding="utf-8"))
    return tuple(
        StateSpecification(
            state_id=row["state_id"],
            fields={
                "power": row["power"],
                "mode": row["mode"],
                "temperature": row["temperature"],
                "door": row["door"],
                "alarm": row["alarm"],
            },
            action_id=row["action"],
        )
        for row in rows
    )


def _compile(path: Path):
    compiler = PanelObservationEvidenceCompiler.load_default()
    observation, report = compiler.compile_image(
        path, observation_id=f"obs-{path.stem}"
    )
    claim_set = build_state_claim_set(
        observation.observation_id,
        _states(),
        report.evidence,
    )
    policy = build_policy_compatibility_report(claim_set, _states())
    return report, claim_set, policy


def _occlude(
    source: Path, target: Path, boxes: tuple[tuple[int, int, int, int], ...]
) -> Path:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill=(236, 240, 242))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", optimize=False)
    return target


def test_compiler_resolves_exact_canonical_panel_fixture() -> None:
    render_dataset(sessions=("development",))
    image = ROOT / "captures/development/state-001/canonical-panel-fixture-01.png"

    report, claim_set, policy = _compile(image)

    assert report.validity_report.valid is True
    assert report.registration_result.valid is True
    assert {item.field_id: item.supported_values for item in report.evidence} == {
        "power": ("green",),
        "mode": ("auto",),
        "temperature": ("normal",),
        "door": ("closed",),
        "alarm": ("inactive",),
    }
    assert claim_set.compatible_state_ids == ("state-001",)
    assert policy.decision == "CONTINUE"
    assert all(item.raw_measurement for item in report.evidence)


def test_door_occlusion_keeps_action_equivalent_states_and_executes(
    tmp_path: Path,
) -> None:
    render_dataset(sessions=("development",))
    source = ROOT / "captures/development/state-001/canonical-panel-fixture-01.png"
    occluded = _occlude(
        source,
        tmp_path / "door-occluded-test.png",
        ((260, 295, 555, 375),),
    )

    report, claim_set, policy = _compile(occluded)

    door = next(item for item in report.evidence if item.field_id == "door")
    assert door.status == "unresolved"
    assert claim_set.compatible_state_ids == ("state-001", "state-002")
    assert claim_set.unresolved_fields == ("door",)
    assert policy.decision == "CONTINUE"


def test_action_changing_occlusion_rejects_with_unresolved_fields(
    tmp_path: Path,
) -> None:
    render_dataset(sessions=("development",))
    source = ROOT / "captures/development/state-011/canonical-panel-fixture-01.png"
    occluded = _occlude(
        source,
        tmp_path / "mode-door-occluded-test.png",
        ((250, 126, 632, 196), (260, 295, 555, 375)),
    )

    _, claim_set, policy = _compile(occluded)

    assert claim_set.compatible_state_ids == ("state-011", "state-012")
    assert claim_set.unresolved_fields == ("door", "mode")
    assert policy.decision == REJECT_AMBIGUOUS_DECISION


def test_missing_anchors_short_circuits_invalid_observation(tmp_path: Path) -> None:
    render_dataset(sessions=("development",))
    source = ROOT / "captures/development/state-001/canonical-panel-fixture-01.png"
    invalid = _occlude(
        source,
        tmp_path / "missing-anchors-test.png",
        ((0, 0, 90, 90), (630, 0, 720, 90), (0, 390, 90, 480), (630, 390, 720, 480)),
    )

    report, claim_set, policy = _compile(invalid)

    assert report.registration_result.valid is False
    assert claim_set.invalid_observation is True
    assert policy.decision == INVALID_OBSERVATION_DECISION
