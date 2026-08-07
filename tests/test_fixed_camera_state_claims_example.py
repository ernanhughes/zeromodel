from __future__ import annotations

import hashlib
import json

from examples.fixed_camera_state_claims.panel_renderer import ROOT, render_dataset


def test_fixed_camera_state_claims_renderer_writes_manifests() -> None:
    manifests = render_dataset(sessions=("development",))

    records = manifests["development"]
    assert len(records) == 12
    first = records[0]
    assert first["image"] == ("captures/development/state-001/canonical-render-01.png")
    assert first["ground_truth"] == {
        "power": "green",
        "mode": "auto",
        "temperature": "normal",
        "door": "closed",
        "alarm": "inactive",
    }
    assert first["expected_action"] == "CONTINUE"

    image_path = ROOT / str(first["image"])
    assert image_path.exists()
    assert first["image_digest"] == (
        f"sha256:{hashlib.sha256(image_path.read_bytes()).hexdigest()}"
    )

    manifest_path = ROOT / "manifests" / "development.jsonl"
    manifest_records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_records == records
