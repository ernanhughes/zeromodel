from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench


def _copy_artifacts(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in bench.AUDIT_ARTIFACT_NAMES:
        shutil.copy2(src / name, dst / name)


def test_artifact_comparison_detects_changed_sample_size(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _copy_artifacts(Path("docs/results/video-discriminative-local-evidence-v1"), expected)
    _copy_artifacts(Path("docs/results/video-discriminative-local-evidence-v1"), actual)
    manifest = json.loads((actual / "benchmark-manifest.json").read_text(encoding="utf-8"))
    manifest["metadata"]["nonprototype_row_sample_size"] = 99
    (actual / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison = bench._artifact_comparison(expected, actual)
    assert comparison["semantic_match"] is False
    assert next(item for item in comparison["artifacts"] if item["artifact"] == "benchmark-manifest.json")["status"] == "semantic_mismatch"


def test_artifact_comparison_detects_changed_family_definition_and_sampling(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _copy_artifacts(Path("docs/results/video-discriminative-local-evidence-v1"), expected)
    _copy_artifacts(Path("docs/results/video-discriminative-local-evidence-v1"), actual)
    split_manifest = json.loads((actual / "split-manifest.json").read_text(encoding="utf-8"))
    split_manifest["observation_membership"]["architecture_selection_benign"] = split_manifest["observation_membership"]["architecture_selection_benign"][:-1]
    split_manifest["clip_membership"]["architecture_selection_benign"] = split_manifest["clip_membership"]["architecture_selection_benign"][:-1]
    (actual / "split-manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison = bench._artifact_comparison(expected, actual)
    assert comparison["semantic_match"] is False
    assert next(item for item in comparison["artifacts"] if item["artifact"] == "split-manifest.json")["status"] == "semantic_mismatch"


def test_artifact_comparison_detects_changed_mask_digest_and_stale_files(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _copy_artifacts(Path("docs/results/video-discriminative-local-evidence-v1"), expected)
    _copy_artifacts(Path("docs/results/video-discriminative-local-evidence-v1"), actual)
    mask_manifest = json.loads((actual / "mask-manifest.json").read_text(encoding="utf-8"))
    mask_manifest["mask_spec_digest"] = "sha256:deadbeef"
    (actual / "mask-manifest.json").write_text(json.dumps(mask_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (actual / "selected-operating-point.json").unlink()
    (actual / "unexpected.json").write_text("{}", encoding="utf-8")
    comparison = bench._artifact_comparison(expected, actual)
    by_name = {item["artifact"]: item["status"] for item in comparison["artifacts"]}
    assert by_name["mask-manifest.json"] == "semantic_mismatch"
    assert by_name["selected-operating-point.json"] == "missing_artifact"
    assert by_name["unexpected.json"] == "unexpected_artifact"


def test_prototype_closure_reports_missing_rows_and_zero_stability() -> None:
    benchmark = bench._build_stage3_benchmark(materialize_final=False)
    freeze = bench._freeze_regions_and_masks(benchmark, output_dir=Path("docs/results/video-discriminative-local-evidence-v1/measurement-audit/test-scratch"))
    closure = bench.validate_prototype_and_development_closure(benchmark=benchmark, masks=freeze["masks"])
    assert len(closure["provider_prototype_rows"]) == 112
    assert len(closure["prototype_split_rows"]) == 4
    assert "missing_prototype_manifest_rows" in closure["closure_failures"]
    assert "missing_development_rows" in closure["closure_failures"]
    assert "tank=0|target=0|cooldown=0" in closure["rows_with_zero_stability"]
    shutil.rmtree(Path("docs/results/video-discriminative-local-evidence-v1/measurement-audit/test-scratch"), ignore_errors=True)


def test_exact_frame_audit_shows_current_code_has_nonzero_evidence(tmp_path: Path) -> None:
    summary = bench._audit_exact_frames(output_dir=tmp_path)
    assert summary["exact_observation_count"] == 4
    for architecture_id in ("A", "B", "C"):
        assert summary["architectures"][architecture_id]["expected_rows_with_nonzero_evidence"] == 4


@pytest.mark.slow
def test_audit_pre_final_v1_marks_stale_artifacts_invalid() -> None:
    with pytest.raises(SystemExit, match="invalid_multiple_failures"):
        bench._run_audit_pre_final_v1(Path("docs/results/video-discriminative-local-evidence-v1"))
    ruling = bench._load_v1_ruling(Path("docs/results/video-discriminative-local-evidence-v1"))
    assert ruling is not None
    assert ruling["ruling"] == "invalid_multiple_failures"


def test_evaluate_is_blocked_when_v1_is_invalid() -> None:
    with pytest.raises(SystemExit, match="failed the integrity audit"):
        bench._run_evaluate(Path("docs/results/video-discriminative-local-evidence-v1"))
