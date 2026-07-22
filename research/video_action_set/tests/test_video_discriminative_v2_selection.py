from __future__ import annotations

from pathlib import Path

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench

import pytest

pytestmark = pytest.mark.research


@pytest.mark.slow
def test_v2_selection_reports_invalid_measurement_when_exact_sanity_fails(tmp_path: Path) -> None:
    bench.run_freeze_benchmark_v2(tmp_path)
    selection = bench.run_select_architecture_v2(tmp_path)
    assert selection["selection_status"] == "invalid_architecture_measurement"
    assert selection["selected_architecture"] is None


@pytest.mark.slow
def test_v2_calibration_reports_invalid_measurement_without_selected_architecture(tmp_path: Path) -> None:
    bench.run_freeze_benchmark_v2(tmp_path)
    bench.run_select_architecture_v2(tmp_path)
    calibration = bench.run_calibrate_v2(tmp_path)
    assert calibration["selection_status"] == "invalid_calibration_measurement"


@pytest.mark.slow
def test_v2_pre_final_verification_rebuilds_invalid_state_consistently(tmp_path: Path) -> None:
    bench.run_freeze_benchmark_v2(tmp_path)
    bench.run_select_architecture_v2(tmp_path)
    bench.run_calibrate_v2(tmp_path)
    verification = bench.run_verify_pre_final_v2(tmp_path)
    assert verification["verified"] is True
    assert verification["selection_status"] == "invalid_architecture_measurement"
    assert verification["calibration_status"] == "invalid_calibration_measurement"
