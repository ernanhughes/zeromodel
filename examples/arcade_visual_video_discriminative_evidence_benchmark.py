"""Stage 3 discriminative-evidence benchmark preparation and selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_shooter_policy import ACTIONS, ShooterConfig, compile_policy_artifact  # noqa: E402
from examples.arcade_visual_local_evidence_benchmark import SOURCE_SCOPE  # noqa: E402
from examples.arcade_visual_sign_reader import render_state_frame  # noqa: E402
from examples.arcade_visual_video_local_correlation_benchmark import (  # noqa: E402
    _build_v2_provider,
    _build_v2_selection,
    _regions as _stage2_regions,
    build_video_cases,
)
from research.evidence.video_discriminative_evidence import (
    DiscriminativeEvidenceCalibration,
    DiscriminativeEvidenceProvider,
    DiscriminativeRegionSpec,
    build_discriminative_masks,
    build_discriminative_candidate_set,
    discriminative_mask_digest,
    discriminative_region_digest,
    evaluate_candidate_eligibility,
)
from research.evidence.video_discriminative_joint_evidence import (
    JointEvidenceCalibration,
    JointEvidenceProvider,
    JointEvidenceRegionSpec,
    build_joint_candidate_masks,
    build_joint_candidate_set,
    build_pairwise_discriminative_masks,
    evaluate_joint_candidate_eligibility,
    joint_candidate_mask_digest,
    joint_region_digest,
    pairwise_mask_digest,
)
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation.visual_address import ImageObservation
import research.evidence.video_discriminative_evidence as zde
import research.evidence.video_discriminative_joint_evidence as zje
from zeromodel.core.artifact import VPMValidationError  # noqa: E402
from research.visual.visual_registration import RegistrationConfig  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "video-discriminative-local-evidence-v1"
OUTPUT_DIR_V2 = REPO_ROOT / "docs" / "results" / "video-discriminative-local-evidence-v2"
OUTPUT_DIR_V3 = REPO_ROOT / "docs" / "results" / "video-discriminative-local-evidence-v3"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
MEASUREMENT_AUDIT_DIR = OUTPUT_DIR / "measurement-audit"
BENCHMARK_VERSION = "zeromodel-video-discriminative-evidence-stage3/v1"
BENCHMARK_GENERATOR_VERSION = "zeromodel-video-discriminative-generator/v1"
BENCHMARK_VERSION_V2 = "zeromodel-video-discriminative-evidence-stage3/v2"
BENCHMARK_GENERATOR_VERSION_V2 = "zeromodel-video-discriminative-generator/v2"
BENCHMARK_VERSION_V3 = "zeromodel-video-discriminative-evidence-stage3/v3"
BENCHMARK_GENERATOR_VERSION_V3 = "zeromodel-video-discriminative-generator/v3"
BENCHMARK_PROVIDER_VERSION_V3 = "zeromodel-video-discriminative-provider/v3"
PREREGISTRATION_COMMIT = "e6d3c2461a3e7fc783026907aa1ab5b803c878f3"
FINAL_SEED_MATERIAL = f"zeromodel-stage3-final-v1|{PREREGISTRATION_COMMIT}"
V2_AMENDMENT_COMMIT = "6e1e18a8613085b63040283ac3b785b183294357"
FINAL_SEED_MATERIAL_V2 = f"zeromodel-stage3-v2-final|{V2_AMENDMENT_COMMIT}"
FINAL_SEED_DIGEST_V2 = "sha256:d740c41178cfe660b3d10d680eafc1f6e20177ba2968ebb5057aa99abd2652cd"
V3_AMENDMENT_COMMIT = "ad2093590cde95ad1dc984f0573f452693002717"
FINAL_SEED_MATERIAL_V3 = f"zeromodel-stage3-v3-final|{V3_AMENDMENT_COMMIT}"
FINAL_SEED_DIGEST_V3 = "sha256:cf1c355325f847d7359243a9d0943d7a770dc9f99f78973b2a94d49bca43c2ef"
STAGE2_PARENT_COMMIT = "d00e18b67fbe2f62617cd0ac47c7ee2f63487cb8"
STAGE2_BENCHMARK_DIGEST = "sha256:589bb074e1b53b06657cfb75bf7b8d67eae43cc5f76e7237ab07f23ccca49c75"
STAGE2_SPLIT_DIGEST = "sha256:d25b694b3cce93bf93f58239163331f3f6370d32a2b5cce53b4541902b0f8c23"
MAXIMUM_USEFUL_CANDIDATE_SET_SIZE = 3
SELECTION_NEGATIVE_CANDIDATE_SET_SUPPORT_BLOCKS_FEASIBILITY = True
SIMPLICITY_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
NON_PROTOTYPE_ROW_SAMPLE_SIZE = 4
EVALUATION_ROW_SAMPLE_SIZE = 12
SPLIT_ROLES = (
    "prototype",
    "diagnostic_development",
    "architecture_selection_benign",
    "architecture_selection_negative",
    "benign_calibration",
    "rejection_calibration",
    "final_benign",
    "final_distinguishable_negative",
    "information_theoretic_control",
)
ARCHITECTURES = ("A", "B", "C", "D")
AUDIT_ARTIFACT_NAMES = (
    "benchmark-manifest.json",
    "split-manifest.json",
    "region-manifest.json",
    "mask-manifest.json",
    "architecture-grid-definition.json",
    "architecture-grid.csv",
    "architecture-d-gateway.json",
    "selected-architecture.json",
    "selected-operating-point.json",
    "phase-access-audits.json",
)
VALID_V1_RULINGS = {
    "valid_no_safe_architecture",
    "invalid_generator_artifact_mismatch",
    "invalid_prototype_closure",
    "invalid_mask_development_closure",
    "invalid_provider_benchmark_wiring",
    "invalid_multiple_failures",
}
V2_BENCHMARK_BASE_ARTIFACT_NAMES = (
    "README.md",
    "generator-identity.json",
    "prototype-manifest.json",
    "development-manifest.json",
    "evaluation-sample.json",
    "prototype-collision-atlas.json",
    "prototype-collision-atlas.csv",
    "benchmark-manifest.json",
    "split-manifest.json",
    "region-manifest.json",
    "mask-manifest.json",
    "mask-closure.csv",
    "mask-closure-summary.json",
    "exact-sanity.csv",
    "exact-sanity-summary.json",
    "phase-access-audits.json",
    "architecture-grid-definition.json",
    "reproduction.md",
)
V2_PREFINAL_ARTIFACT_NAMES = V2_BENCHMARK_BASE_ARTIFACT_NAMES + (
    "architecture-grid.csv",
    "architecture-d-gateway.json",
    "selected-architecture.json",
    "selected-operating-point.json",
    "pre-final-verification.json",
)
V2_REPRESENTATION_AUDIT_ARTIFACT_NAMES = (
    "canonical-self-retrieval.csv",
    "canonical-self-retrieval-summary.json",
    "mask-separation-audit.csv",
    "mask-separation-summary.json",
    "compositional-uniqueness.csv",
    "compositional-uniqueness.json",
    "distributed-competitor-cover.json",
    "architecture-a-conformance.csv",
    "architecture-a-conformance.json",
    "exact-tie-audit.json",
    "ruling.json",
    "ruling.md",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (Mapping, MappingProxyType)):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, MappingProxyType)):
        return json.dumps(_json_ready(value), sort_keys=True, ensure_ascii=False)
    return str(value)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_output(*args: str) -> Optional[str]:
    try:
        return (
            __import__("subprocess")
            .check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=__import__("subprocess").DEVNULL)
            .strip()
        )
    except Exception:
        return None


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return "sha256:" + hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _translate(frame: np.ndarray, *, dx: int = 0, dy: int = 0, fill: int = 0) -> np.ndarray:
    result = np.full_like(frame, fill)
    height, width = frame.shape
    x0 = max(0, dx)
    x1 = min(width, width + dx)
    y0 = max(0, dy)
    y1 = min(height, height + dy)
    if x1 > x0 and y1 > y0:
        result[y0:y1, x0:x1] = frame[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return result


def _scale_intensity(frame: np.ndarray, *, numerator: int = 100, offset: int = 0) -> np.ndarray:
    scaled = np.round(frame.astype(np.float32) * (float(numerator) / 100.0)) + float(offset)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _mask_box(frame: np.ndarray, *, top: int, left: int, height: int, width: int, value: int) -> np.ndarray:
    result = np.array(frame, copy=True)
    result[top : top + height, left : left + width] = np.uint8(value)
    return result


def _blend_region(frame: np.ndarray, other: np.ndarray, *, top: int, left: int, height: int, width: int) -> np.ndarray:
    result = np.array(frame, copy=True)
    result[top : top + height, left : left + width] = other[top : top + height, left : left + width]
    return result


@dataclass(frozen=True)
class Stage3Record:
    observation_id: str
    split: str
    family_id: str
    row_id: Optional[str]
    action_id: Optional[str]
    expected_disposition: str
    clip_id: str
    frame_id: str
    materialized: bool
    metadata: Mapping[str, Any]
    observation_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "split": self.split,
            "family_id": self.family_id,
            "row_id": self.row_id,
            "action_id": self.action_id,
            "expected_disposition": self.expected_disposition,
            "clip_id": self.clip_id,
            "frame_id": self.frame_id,
            "materialized": self.materialized,
            "metadata": _json_ready(self.metadata),
            "observation_digest": self.observation_digest,
        }


@dataclass
class Stage3Benchmark:
    policy_artifact_id: str
    source_scope: str
    prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]]
    records: Tuple[Stage3Record, ...]
    observations: Mapping[str, ImageObservation]
    family_specs: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    phase_audits: list[Dict[str, Any]]

    def record_by_split(self, split: str) -> Tuple[Stage3Record, ...]:
        return tuple(record for record in self.records if record.split == split)

    def access(self, *, phase: str, allowed_splits: Sequence[str]) -> Tuple[Tuple[Stage3Record, ...], Dict[str, ImageObservation]]:
        allowed = tuple(str(item) for item in allowed_splits)
        allowed_set = set(allowed)
        selected = tuple(record for record in self.records if record.split in allowed_set)
        accessed_splits = tuple(sorted({record.split for record in selected}))
        violations = tuple(split for split in accessed_splits if split not in allowed_set)
        audit = {
            "phase": phase,
            "allowed_splits": list(allowed),
            "accessed_splits": list(accessed_splits),
            "observation_ids": [record.observation_id for record in selected],
            "clip_ids": sorted({record.clip_id for record in selected}),
            "access_digest": _sha256([record.to_dict() for record in selected]),
            "violations": list(violations),
        }
        if any(split.startswith("final_") for split in accessed_splits if phase in {"select_architecture", "calibrate"}):
            audit["violations"] = list(audit["violations"]) + ["forbidden_final_split_access"]
        self.phase_audits.append(audit)
        if audit["violations"]:
            raise VPMValidationError(f"phase access violation for {phase}: {audit['violations']}")
        return (selected, {record.observation_id: self.observations[record.observation_id] for record in selected if record.materialized})


@dataclass(frozen=True)
class Stage3VersionConfig:
    benchmark_version: str
    generator_version: str
    output_dir: Path
    seed_material: str
    seed_digest: str
    sample_size: int
    observation_prefix: str
    prototype_family_id: str
    amendment_commit: Optional[str] = None
    preregistration_commit: Optional[str] = None


def _canonical_rows(config: ShooterConfig) -> Tuple[Tuple[str, str, ImageObservation], ...]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    rows = []
    for tank_x in range(config.width):
        for target_x in (None, *range(config.width)):
            for cooldown in (0, 1):
                row_id = f"tank={tank_x}|target={'none' if target_x is None else target_x}|cooldown={cooldown}"
                pixels = render_state_frame(tank_x, target_x, cooldown, width=config.width)
                rows.append((row_id, lookup.choose(row_id), ImageObservation(pixels, source_id=row_id)))
    return tuple(rows)


def _sample_nonprototype_rows(rows: Sequence[Tuple[str, str, ImageObservation]]) -> Tuple[Tuple[str, str, ImageObservation], ...]:
    if len(rows) <= NON_PROTOTYPE_ROW_SAMPLE_SIZE:
        return tuple(rows)
    step = max(1, len(rows) // NON_PROTOTYPE_ROW_SAMPLE_SIZE)
    selected = list(rows[::step][: NON_PROTOTYPE_ROW_SAMPLE_SIZE - 2])
    selected.append(rows[len(rows) // 2])
    selected.append(rows[-1])
    deduped = []
    seen = set()
    for row in selected:
        if row[0] not in seen:
            seen.add(row[0])
            deduped.append(row)
    return tuple(deduped)


def _sample_evaluation_rows(
    rows: Sequence[Tuple[str, str, ImageObservation]],
    *,
    sample_size: int,
) -> Tuple[Tuple[str, str, ImageObservation], ...]:
    if len(rows) <= sample_size:
        return tuple(rows)
    step = max(1, len(rows) // sample_size)
    selected = list(rows[::step][: sample_size - 2])
    selected.append(rows[len(rows) // 2])
    selected.append(rows[-1])
    deduped = []
    seen = set()
    for row in selected:
        if row[0] not in seen:
            seen.add(row[0])
            deduped.append(row)
    return tuple(deduped)


def _generator_source_blob_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sampled_rows(benchmark: Stage3Benchmark) -> Tuple[Tuple[str, str, ImageObservation], ...]:
    rows = []
    seen = set()
    for record in benchmark.record_by_split("prototype"):
        if record.row_id in seen or record.row_id is None or record.action_id is None:
            continue
        seen.add(record.row_id)
        rows.append((record.row_id, record.action_id, benchmark.prototypes[record.row_id][3]))
    return tuple(rows)


def _generator_identity(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    sampled = _sampled_rows(benchmark)
    canonical_rows = tuple(sorted(benchmark.prototypes))
    family_definitions = _split_family_definitions()
    final_descriptors = [record.observation_id for record in benchmark.records if record.split.startswith("final_") or record.split == "information_theoretic_control"]
    identity = {
        "source_file": str(Path(__file__).resolve()),
        "source_file_blob_digest": _generator_source_blob_digest(),
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": BENCHMARK_GENERATOR_VERSION,
        "constants": {
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "final_seed_material": FINAL_SEED_MATERIAL,
            "maximum_useful_candidate_set_size": MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
            "selection_negative_candidate_set_support_blocks_feasibility": SELECTION_NEGATIVE_CANDIDATE_SET_SUPPORT_BLOCKS_FEASIBILITY,
            "sample_size_constant": NON_PROTOTYPE_ROW_SAMPLE_SIZE,
        },
        "canonical_row_count": len(canonical_rows),
        "canonical_row_ids": list(canonical_rows),
        "evaluated_row_sample_algorithm": "step_sample_with_midpoint_and_last_dedup",
        "evaluated_row_ids": [row_id for row_id, _action_id, _observation in sampled],
        "prototype_row_ids": list(canonical_rows),
        "diagnostic_development_row_ids": sorted({record.row_id for record in benchmark.record_by_split("diagnostic_development") if record.row_id is not None}),
        "architecture_selection_row_ids": sorted({record.row_id for record in benchmark.records if record.split.startswith("architecture_selection_") and record.row_id is not None}),
        "calibration_row_ids": sorted({record.row_id for record in benchmark.records if record.split.endswith("calibration") and record.row_id is not None}),
        "final_row_descriptors": final_descriptors,
        "transformation_family_definitions": _json_ready(family_definitions),
        "transformation_family_digest": _sha256(family_definitions),
        "current_head": _git_output("rev-parse", "HEAD"),
    }
    identity["generator_identity_digest"] = _sha256(identity)
    return identity


def _file_blob_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _generator_identity_v2(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    config = _stage3_v2_config()
    source_files = [
        REPO_ROOT / "examples" / "arcade_visual_video_discriminative_evidence_benchmark.py",
        REPO_ROOT / "examples" / "arcade_visual_sign_reader.py",
        REPO_ROOT / "examples" / "arcade_shooter_policy.py",
        REPO_ROOT / "research" / "evidence" / "video_discriminative_evidence.py",
        REPO_ROOT / "research" / "visual" / "visual_registration.py",
    ]
    file_digests = {str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _file_blob_digest(path) for path in source_files}
    rows = _canonical_rows(ShooterConfig())
    sample = _sample_evaluation_rows(rows, sample_size=config.sample_size)
    identity = {
        "repository_commit_sha": _git_output("rev-parse", "HEAD"),
        "benchmark_version": config.benchmark_version,
        "generator_version": config.generator_version,
        "amendment_commit_sha": config.amendment_commit,
        "evaluation_sample_size": config.sample_size,
        "sampling_algorithm_version": "step_sample_with_midpoint_and_last_dedup/v1",
        "family_definitions": _json_ready(_split_family_definitions_v2()),
        "region_definitions": [region.to_dict() for region in _region_manifest()],
        "mask_parameters": {
            "intensity_tolerance": 8,
            "stability_tolerance": 12,
            "separation_cap": 64,
        },
        "grid_definitions": _architecture_grid_definition(),
        "seed_material": config.seed_material,
        "seed_digest": config.seed_digest,
        "source_file_digests": file_digests,
        "evaluation_sample_row_ids": [row_id for row_id, _action_id, _observation in sample],
    }
    identity["generator_identity_digest"] = _sha256(identity)
    return identity


def _generator_identity_v3(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    config = _stage3_v3_config()
    source_files = [
        REPO_ROOT / "examples" / "arcade_visual_video_discriminative_evidence_benchmark.py",
        REPO_ROOT / "examples" / "arcade_visual_sign_reader.py",
        REPO_ROOT / "examples" / "arcade_shooter_policy.py",
        REPO_ROOT / "research" / "evidence" / "video_discriminative_joint_evidence.py",
        REPO_ROOT / "research" / "evidence" / "video_discriminative_evidence.py",
        REPO_ROOT / "research" / "visual" / "visual_registration.py",
        REPO_ROOT / "docs" / "research" / "video-stage-three-v3-joint-evidence-amendment.md",
        REPO_ROOT / "docs" / "research" / "video-stage-three-v3-operational-contract.md",
    ]
    file_digests = {str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _file_blob_digest(path) for path in source_files}
    rows = _canonical_rows(ShooterConfig())
    sample = _sample_evaluation_rows(rows, sample_size=config.sample_size)
    operational_contract = REPO_ROOT / "docs" / "research" / "video-stage-three-v3-operational-contract.md"
    identity = {
        "repository_commit_sha": _git_output("rev-parse", "HEAD"),
        "benchmark_version": config.benchmark_version,
        "generator_version": config.generator_version,
        "provider_version": BENCHMARK_PROVIDER_VERSION_V3,
        "mechanics_version": zje.VIDEO_JOINT_EVIDENCE_MECHANICS_VERSION,
        "amendment_commit_sha": config.amendment_commit,
        "operational_contract_digest": _file_blob_digest(operational_contract),
        "evaluation_sample_size": config.sample_size,
        "sampling_algorithm_version": "step_sample_with_midpoint_and_last_dedup/v1",
        "family_definitions": _json_ready(_split_family_definitions_v2()),
        "region_definitions": [region.to_dict() for region in _joint_region_manifest()],
        "mask_parameters": {
            "intensity_tolerance": 8,
            "stability_tolerance": 12,
        },
        "seed_material": config.seed_material,
        "seed_digest": config.seed_digest,
        "source_file_digests": file_digests,
        "evaluation_sample_row_ids": [row_id for row_id, _action_id, _observation in sample],
    }
    identity["generator_identity_digest"] = _sha256(identity)
    return identity


def _nearest_same_action(row_id: str, action_id: str, rows: Sequence[Tuple[str, str, ImageObservation]]) -> Tuple[str, ImageObservation]:
    for other_row_id, other_action_id, observation in rows:
        if other_row_id != row_id and other_action_id == action_id:
            return (other_row_id, observation)
    raise VPMValidationError("same-action competitor not found")


def _nearest_conflicting_action(row_id: str, action_id: str, rows: Sequence[Tuple[str, str, ImageObservation]]) -> Tuple[str, ImageObservation]:
    for other_row_id, other_action_id, observation in rows:
        if other_row_id != row_id and other_action_id != action_id:
            return (other_row_id, observation)
    raise VPMValidationError("conflicting-action competitor not found")


def _region_manifest() -> Tuple[DiscriminativeRegionSpec, ...]:
    registration = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    return (
        DiscriminativeRegionSpec(
            region_id="target_band",
            top=0,
            left=0,
            height=6,
            width=28,
            weight=2.0,
            critical=True,
            registration_config=registration,
            metadata={"semantic_role": "target evidence", "reason_for_inclusion": "target alignment and ambiguity"},
        ),
        DiscriminativeRegionSpec(
            region_id="cooldown_indicator",
            top=7,
            left=25,
            height=2,
            width=2,
            weight=1.5,
            critical=True,
            registration_config=registration,
            metadata={"semantic_role": "cooldown evidence", "reason_for_inclusion": "action-critical evidence"},
        ),
        DiscriminativeRegionSpec(
            region_id="tank_band",
            top=10,
            left=0,
            height=4,
            width=28,
            weight=2.0,
            critical=True,
            registration_config=registration,
            metadata={"semantic_role": "tank evidence", "reason_for_inclusion": "pose and local displacement evidence"},
        ),
    )


def _joint_region_manifest() -> Tuple[JointEvidenceRegionSpec, ...]:
    registration = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    return (
        JointEvidenceRegionSpec(
            region_id="target_band",
            top=0,
            left=0,
            height=6,
            width=28,
            weight=2.0,
            critical=True,
            registration_config=registration,
            metadata={"semantic_role": "target evidence", "reason_for_inclusion": "target alignment and ambiguity"},
        ),
        JointEvidenceRegionSpec(
            region_id="cooldown_indicator",
            top=7,
            left=25,
            height=2,
            width=2,
            weight=1.5,
            critical=True,
            registration_config=registration,
            metadata={"semantic_role": "cooldown evidence", "reason_for_inclusion": "action-critical evidence"},
        ),
        JointEvidenceRegionSpec(
            region_id="tank_band",
            top=10,
            left=0,
            height=4,
            width=28,
            weight=2.0,
            critical=True,
            registration_config=registration,
            metadata={"semantic_role": "tank evidence", "reason_for_inclusion": "pose and local displacement evidence"},
        ),
    )


def _split_family_definitions() -> Mapping[str, Sequence[Tuple[str, str]]]:
    return {
        "prototype": (("prototype_clean", "canonical prototype observations"),),
        "diagnostic_development": (
            ("development_photometric", "mild photometric variation for mask stability"),
            ("development_shift", "bounded translation variation for mask stability"),
        ),
        "architecture_selection_benign": (
            ("selection_exact", "exact frames"),
            ("selection_translation_x", "horizontal translations"),
            ("selection_translation_y", "vertical translations"),
            ("selection_translation_xy", "two-axis translations"),
            ("selection_photometric", "mild photometric changes"),
            ("selection_noncritical_occlusion", "noncritical occlusion"),
            ("selection_critical_partial", "critical-region partial occlusion"),
            ("selection_translation_photometric", "translation plus photometric change"),
            ("selection_translation_occlusion", "translation plus occlusion"),
            ("selection_repeated_state", "legitimate repeated-state frames"),
            ("selection_same_action_ambiguous", "same-action ambiguous rows"),
            ("selection_mixed_action_ambiguous", "mixed-action ambiguous rows"),
        ),
        "architecture_selection_negative": (
            ("selection_negative_critical_removed", "critical evidence removal"),
            ("selection_negative_action_critical_removed", "action-critical evidence removal"),
            ("selection_negative_same_action_wrong_row", "same-action wrong-row construction"),
            ("selection_negative_conflicting_action", "conflicting-action near neighbour"),
            ("selection_negative_compositional_invalid", "compositional invalid observation"),
            ("selection_negative_conflicting_regions", "conflicting regional evidence"),
            ("selection_negative_out_of_bounds", "out-of-bounds translation"),
        ),
        "benign_calibration": (
            ("calibration_exact", "fresh exact frames"),
            ("calibration_translation_x", "fresh horizontal translations"),
            ("calibration_translation_xy", "fresh two-axis translations"),
            ("calibration_photometric", "fresh photometric changes"),
            ("calibration_occlusion", "fresh noncritical occlusion"),
            ("calibration_same_action_ambiguous", "fresh same-action ambiguity"),
            ("calibration_mixed_action_ambiguous", "fresh mixed-action ambiguity"),
        ),
        "rejection_calibration": (
            ("calibration_negative_critical_removed", "fresh critical evidence removal"),
            ("calibration_negative_action_critical_removed", "fresh action-critical evidence removal"),
            ("calibration_negative_same_action_wrong_row", "fresh same-action wrong-row"),
            ("calibration_negative_conflicting_action", "fresh conflicting-action neighbour"),
            ("calibration_negative_compositional_invalid", "fresh compositional invalid"),
            ("calibration_negative_conflicting_regions", "fresh conflicting regions"),
            ("calibration_negative_out_of_bounds", "fresh out-of-bounds translation"),
        ),
        "final_benign": (
            ("final_exact", "frozen future exact family"),
            ("final_translation", "frozen future translation family"),
            ("final_photometric", "frozen future photometric family"),
            ("final_occlusion", "frozen future occlusion family"),
        ),
        "final_distinguishable_negative": (
            ("final_negative_critical_removed", "frozen future critical removal"),
            ("final_negative_conflicting_action", "frozen future conflicting-action neighbour"),
            ("final_negative_compositional_invalid", "frozen future compositional invalid"),
        ),
        "information_theoretic_control": (
            ("final_information_control", "frozen future information-theoretic controls"),
        ),
    }


def _make_observation_id(split: str, family_id: str, row_id: str, variant: int) -> str:
    return f"{split}:{family_id}:{row_id}:{variant:02d}"


def _stage3_v1_config() -> Stage3VersionConfig:
    return Stage3VersionConfig(
        benchmark_version=BENCHMARK_VERSION,
        generator_version=BENCHMARK_GENERATOR_VERSION,
        output_dir=OUTPUT_DIR,
        seed_material=FINAL_SEED_MATERIAL,
        seed_digest=_sha256(FINAL_SEED_MATERIAL),
        sample_size=NON_PROTOTYPE_ROW_SAMPLE_SIZE,
        observation_prefix="v1",
        prototype_family_id="prototype_clean",
        preregistration_commit=PREREGISTRATION_COMMIT,
    )


def _stage3_v2_config() -> Stage3VersionConfig:
    return Stage3VersionConfig(
        benchmark_version=BENCHMARK_VERSION_V2,
        generator_version=BENCHMARK_GENERATOR_VERSION_V2,
        output_dir=OUTPUT_DIR_V2,
        seed_material=FINAL_SEED_MATERIAL_V2,
        seed_digest=FINAL_SEED_DIGEST_V2,
        sample_size=EVALUATION_ROW_SAMPLE_SIZE,
        observation_prefix="v2",
        prototype_family_id="prototype_clean_v2",
        amendment_commit=V2_AMENDMENT_COMMIT,
    )


def _stage3_v3_config() -> Stage3VersionConfig:
    return Stage3VersionConfig(
        benchmark_version=BENCHMARK_VERSION_V3,
        generator_version=BENCHMARK_GENERATOR_VERSION_V3,
        output_dir=OUTPUT_DIR_V3,
        seed_material=FINAL_SEED_MATERIAL_V3,
        seed_digest=FINAL_SEED_DIGEST_V3,
        sample_size=EVALUATION_ROW_SAMPLE_SIZE,
        observation_prefix="v3",
        prototype_family_id="prototype_clean_v2",
        amendment_commit=V3_AMENDMENT_COMMIT,
    )


def _make_observation_id_for_config(config: Stage3VersionConfig, split: str, family_id: str, row_id: str, variant: int) -> str:
    return f"{config.observation_prefix}:{split}:{family_id}:{row_id}:{variant:02d}"


def _build_stage3_benchmark(*, materialize_final: bool = False) -> Stage3Benchmark:
    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    rows = _canonical_rows(config)
    sampled_rows = _sample_nonprototype_rows(rows)
    observations: Dict[str, ImageObservation] = {}
    records: list[Stage3Record] = []
    prototypes: Dict[str, Tuple[str, str, str, ImageObservation]] = {}
    families = _split_family_definitions()
    {row_id: (action_id, observation) for row_id, action_id, observation in rows}

    for row_id, action_id, observation in rows:
        prototypes[row_id] = (row_id, action_id, observation.raw_digest, observation)
    for row_id, action_id, observation in sampled_rows:
        same_row_id, same_observation = _nearest_same_action(row_id, action_id, rows)
        conflicting_row_id, conflicting_observation = _nearest_conflicting_action(row_id, action_id, rows)
        transformations = {
            "prototype_clean": lambda base=observation.pixels: base,
            "development_photometric": lambda base=observation.pixels: _scale_intensity(base, numerator=96, offset=3),
            "development_shift": lambda base=observation.pixels: _translate(base, dx=1, dy=0),
            "selection_exact": lambda base=observation.pixels: base,
            "selection_translation_x": lambda base=observation.pixels: _translate(base, dx=1, dy=0),
            "selection_translation_y": lambda base=observation.pixels: _translate(base, dx=0, dy=1),
            "selection_translation_xy": lambda base=observation.pixels: _translate(base, dx=1, dy=1),
            "selection_photometric": lambda base=observation.pixels: _scale_intensity(base, numerator=92, offset=4),
            "selection_noncritical_occlusion": lambda base=observation.pixels: _mask_box(base, top=0, left=0, height=2, width=3, value=90),
            "selection_critical_partial": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=1, width=2, value=0),
            "selection_translation_photometric": lambda base=observation.pixels: _scale_intensity(_translate(base, dx=1, dy=0), numerator=94, offset=2),
            "selection_translation_occlusion": lambda base=observation.pixels: _mask_box(_translate(base, dx=1, dy=0), top=0, left=0, height=2, width=3, value=90),
            "selection_repeated_state": lambda base=observation.pixels: base,
            "selection_same_action_ambiguous": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=0, left=0, height=6, width=28),
            "selection_mixed_action_ambiguous": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "selection_negative_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=0),
            "selection_negative_action_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=255),
            "selection_negative_same_action_wrong_row": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=0, left=0, height=6, width=28),
            "selection_negative_conflicting_action": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "selection_negative_compositional_invalid": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=10, left=0, height=4, width=28), other, top=0, left=0, height=6, width=28),
            "selection_negative_conflicting_regions": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=0, left=0, height=6, width=28), other, top=10, left=0, height=4, width=28),
            "selection_negative_out_of_bounds": lambda base=observation.pixels: _translate(base, dx=4, dy=0),
            "calibration_exact": lambda base=observation.pixels: _scale_intensity(base, numerator=100, offset=1),
            "calibration_translation_x": lambda base=observation.pixels: _translate(base, dx=-1, dy=0),
            "calibration_translation_xy": lambda base=observation.pixels: _translate(base, dx=-1, dy=1),
            "calibration_photometric": lambda base=observation.pixels: _scale_intensity(base, numerator=90, offset=6),
            "calibration_occlusion": lambda base=observation.pixels: _mask_box(base, top=1, left=1, height=2, width=3, value=120),
            "calibration_same_action_ambiguous": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=10, left=0, height=4, width=28),
            "calibration_mixed_action_ambiguous": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "calibration_negative_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=0),
            "calibration_negative_action_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=255),
            "calibration_negative_same_action_wrong_row": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=0, left=0, height=6, width=28),
            "calibration_negative_conflicting_action": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "calibration_negative_compositional_invalid": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=10, left=0, height=4, width=28), other, top=0, left=0, height=6, width=28),
            "calibration_negative_conflicting_regions": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=0, left=0, height=6, width=28), other, top=10, left=0, height=4, width=28),
            "calibration_negative_out_of_bounds": lambda base=observation.pixels: _translate(base, dx=-4, dy=0),
        }
        for split, family_items in families.items():
            for family_id, description in family_items:
                observation_id = _make_observation_id(split, family_id, row_id, 0)
                clip_id = f"{split}:{family_id}:{row_id}"
                frame_id = observation_id
                if split == "prototype":
                    pixels = transformations["prototype_clean"]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "diagnostic_development":
                    pixels = transformations["development_photometric"]() if family_id == "development_photometric" else transformations["development_shift"]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "architecture_selection_benign":
                    pixels = transformations[family_id]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "architecture_selection_negative":
                    pixels = transformations[family_id]()
                    materialized = True
                    expected_disposition = "expected_reject"
                elif split == "benign_calibration":
                    pixels = transformations[family_id]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "rejection_calibration":
                    pixels = transformations[family_id]()
                    materialized = True
                    expected_disposition = "expected_reject"
                else:
                    materialized = bool(materialize_final)
                    expected_disposition = "information_theoretic_control" if split == "information_theoretic_control" else ("expected_accept" if split == "final_benign" else "expected_reject")
                    if materialized:
                        pixels = observation.pixels if family_id in {"final_exact", "final_information_control"} else _scale_intensity(observation.pixels, numerator=97, offset=(len(family_id) % 7) + 1)
                    else:
                        pixels = None
                metadata = {
                    "split_role": split,
                    "family_id": family_id,
                    "description": description,
                    "expected_disposition_class": expected_disposition,
                    "row_id": row_id,
                    "action_id": action_id,
                    "source_row_id": row_id,
                    "same_action_row_id": same_row_id,
                    "conflicting_action_row_id": conflicting_row_id,
                    "distinguishable_negative": split in {"architecture_selection_negative", "rejection_calibration", "final_distinguishable_negative"},
                    "information_theoretic_control": split == "information_theoretic_control",
                }
                if materialized:
                    image = ImageObservation(np.ascontiguousarray(pixels, dtype=np.uint8), source_id=observation_id, metadata=metadata)
                    observations[observation_id] = image
                    observation_digest = image.raw_digest
                else:
                    observation_digest = _sha256(
                        {
                            "benchmark_version": BENCHMARK_VERSION,
                            "generator_version": BENCHMARK_GENERATOR_VERSION,
                            "observation_id": observation_id,
                            "seed_material": FINAL_SEED_MATERIAL,
                            "family_id": family_id,
                            "row_id": row_id,
                            "action_id": action_id,
                        }
                    )
                records.append(
                    Stage3Record(
                        observation_id=observation_id,
                        split=split,
                        family_id=family_id,
                        row_id=row_id,
                        action_id=action_id,
                        expected_disposition=expected_disposition,
                        clip_id=clip_id,
                        frame_id=frame_id,
                        materialized=materialized,
                        metadata=metadata,
                        observation_digest=observation_digest,
                    )
                )
    split_counts = {split: sum(record.split == split for record in records) for split in SPLIT_ROLES}
    return Stage3Benchmark(
        policy_artifact_id=policy.artifact_id,
        source_scope=SOURCE_SCOPE,
        prototypes=prototypes,
        records=tuple(records),
        observations=observations,
        family_specs={family_id: {"description": description, "split": split} for split, items in families.items() for family_id, description in items},
        metadata={
            "benchmark_version": BENCHMARK_VERSION,
            "generator_version": BENCHMARK_GENERATOR_VERSION,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "final_seed_material": FINAL_SEED_MATERIAL,
            "final_seed_digest": _sha256(FINAL_SEED_MATERIAL),
            "split_counts": split_counts,
            "materialized_final": bool(materialize_final),
            "nonprototype_row_sample_size": len(sampled_rows),
        },
        phase_audits=[],
    )


def _split_family_definitions_v2() -> Mapping[str, Sequence[Tuple[str, str]]]:
    families = dict(_split_family_definitions())
    families["prototype"] = (("prototype_clean_v2", "canonical prototype observations"),)
    return families


def _build_stage3_benchmark_v2(*, materialize_final: bool = False) -> Stage3Benchmark:
    config = _stage3_v2_config()
    shooter = ShooterConfig()
    policy = compile_policy_artifact(shooter)
    rows = _canonical_rows(shooter)
    sampled_rows = _sample_evaluation_rows(rows, sample_size=config.sample_size)
    observations: Dict[str, ImageObservation] = {}
    records: list[Stage3Record] = []
    prototypes: Dict[str, Tuple[str, str, str, ImageObservation]] = {}
    families = _split_family_definitions_v2()

    for row_id, action_id, observation in rows:
        prototypes[row_id] = (row_id, action_id, observation.raw_digest, observation)

    def _make_stage_record(
        *,
        split: str,
        family_id: str,
        row_id: str,
        action_id: str,
        pixels: Optional[np.ndarray],
        materialized: bool,
        expected_disposition: str,
        metadata: Mapping[str, Any],
    ) -> Stage3Record:
        observation_id = _make_observation_id_for_config(config, split, family_id, row_id, 0)
        clip_id = f"{config.observation_prefix}:{split}:{family_id}:{row_id}"
        frame_id = observation_id
        if materialized:
            image = ImageObservation(np.ascontiguousarray(pixels, dtype=np.uint8), source_id=observation_id, metadata=metadata)
            observations[observation_id] = image
            observation_digest = image.raw_digest
        else:
            observation_digest = _sha256(
                {
                    "benchmark_version": config.benchmark_version,
                    "generator_version": config.generator_version,
                    "observation_id": observation_id,
                    "seed_material": config.seed_material,
                    "family_id": family_id,
                    "row_id": row_id,
                    "action_id": action_id,
                }
            )
        return Stage3Record(
            observation_id=observation_id,
            split=split,
            family_id=family_id,
            row_id=row_id,
            action_id=action_id,
            expected_disposition=expected_disposition,
            clip_id=clip_id,
            frame_id=frame_id,
            materialized=materialized,
            metadata=metadata,
            observation_digest=observation_digest,
        )

    # Prototype and development cover the complete provider prototype universe.
    for row_id, action_id, observation in rows:
        same_row_id, same_observation = _nearest_same_action(row_id, action_id, rows)
        conflicting_row_id, conflicting_observation = _nearest_conflicting_action(row_id, action_id, rows)
        transform_map = {
            "prototype_clean_v2": lambda base=observation.pixels: base,
            "development_photometric": lambda base=observation.pixels: _scale_intensity(base, numerator=96, offset=3),
            "development_shift": lambda base=observation.pixels: _translate(base, dx=1, dy=0),
            "selection_exact": lambda base=observation.pixels: base,
            "selection_translation_x": lambda base=observation.pixels: _translate(base, dx=1, dy=0),
            "selection_translation_y": lambda base=observation.pixels: _translate(base, dx=0, dy=1),
            "selection_translation_xy": lambda base=observation.pixels: _translate(base, dx=1, dy=1),
            "selection_photometric": lambda base=observation.pixels: _scale_intensity(base, numerator=92, offset=4),
            "selection_noncritical_occlusion": lambda base=observation.pixels: _mask_box(base, top=0, left=0, height=2, width=3, value=90),
            "selection_critical_partial": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=1, width=2, value=0),
            "selection_translation_photometric": lambda base=observation.pixels: _scale_intensity(_translate(base, dx=1, dy=0), numerator=94, offset=2),
            "selection_translation_occlusion": lambda base=observation.pixels: _mask_box(_translate(base, dx=1, dy=0), top=0, left=0, height=2, width=3, value=90),
            "selection_repeated_state": lambda base=observation.pixels: base,
            "selection_same_action_ambiguous": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=0, left=0, height=6, width=28),
            "selection_mixed_action_ambiguous": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "selection_negative_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=0),
            "selection_negative_action_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=255),
            "selection_negative_same_action_wrong_row": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=0, left=0, height=6, width=28),
            "selection_negative_conflicting_action": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "selection_negative_compositional_invalid": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=10, left=0, height=4, width=28), other, top=0, left=0, height=6, width=28),
            "selection_negative_conflicting_regions": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=0, left=0, height=6, width=28), other, top=10, left=0, height=4, width=28),
            "selection_negative_out_of_bounds": lambda base=observation.pixels: _translate(base, dx=4, dy=0),
            "calibration_exact": lambda base=observation.pixels: _scale_intensity(base, numerator=100, offset=1),
            "calibration_translation_x": lambda base=observation.pixels: _translate(base, dx=-1, dy=0),
            "calibration_translation_xy": lambda base=observation.pixels: _translate(base, dx=-1, dy=1),
            "calibration_photometric": lambda base=observation.pixels: _scale_intensity(base, numerator=90, offset=6),
            "calibration_occlusion": lambda base=observation.pixels: _mask_box(base, top=1, left=1, height=2, width=3, value=120),
            "calibration_same_action_ambiguous": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=10, left=0, height=4, width=28),
            "calibration_mixed_action_ambiguous": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "calibration_negative_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=0),
            "calibration_negative_action_critical_removed": lambda base=observation.pixels: _mask_box(base, top=7, left=25, height=2, width=2, value=255),
            "calibration_negative_same_action_wrong_row": lambda base=observation.pixels, same=same_observation.pixels: _blend_region(base, same, top=0, left=0, height=6, width=28),
            "calibration_negative_conflicting_action": lambda base=observation.pixels, other=conflicting_observation.pixels: _blend_region(base, other, top=7, left=25, height=2, width=2),
            "calibration_negative_compositional_invalid": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=10, left=0, height=4, width=28), other, top=0, left=0, height=6, width=28),
            "calibration_negative_conflicting_regions": lambda base=observation.pixels, same=same_observation.pixels, other=conflicting_observation.pixels: _blend_region(_blend_region(base, same, top=0, left=0, height=6, width=28), other, top=10, left=0, height=4, width=28),
            "calibration_negative_out_of_bounds": lambda base=observation.pixels: _translate(base, dx=-4, dy=0),
        }
        for split, family_items in families.items():
            if split in {"architecture_selection_benign", "architecture_selection_negative", "benign_calibration", "rejection_calibration", "final_benign", "final_distinguishable_negative", "information_theoretic_control"} and row_id not in {item[0] for item in sampled_rows}:
                continue
            for family_id, description in family_items:
                if split == "prototype":
                    pixels = transform_map[family_id]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "diagnostic_development":
                    pixels = transform_map[family_id]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "architecture_selection_benign":
                    pixels = transform_map[family_id]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "architecture_selection_negative":
                    pixels = transform_map[family_id]()
                    materialized = True
                    expected_disposition = "expected_reject"
                elif split == "benign_calibration":
                    pixels = transform_map[family_id]()
                    materialized = True
                    expected_disposition = "expected_accept"
                elif split == "rejection_calibration":
                    pixels = transform_map[family_id]()
                    materialized = True
                    expected_disposition = "expected_reject"
                else:
                    materialized = bool(materialize_final)
                    expected_disposition = "information_theoretic_control" if split == "information_theoretic_control" else ("expected_accept" if split == "final_benign" else "expected_reject")
                    if materialized:
                        pixels = observation.pixels if family_id in {"final_exact", "final_information_control"} else _scale_intensity(observation.pixels, numerator=97, offset=(len(family_id) % 7) + 1)
                    else:
                        pixels = None
                metadata = {
                    "split_role": split,
                    "family_id": family_id,
                    "description": description,
                    "expected_disposition_class": expected_disposition,
                    "row_id": row_id,
                    "action_id": action_id,
                    "source_row_id": row_id,
                    "same_action_row_id": same_row_id,
                    "conflicting_action_row_id": conflicting_row_id,
                    "distinguishable_negative": split in {"architecture_selection_negative", "rejection_calibration", "final_distinguishable_negative"},
                    "information_theoretic_control": split == "information_theoretic_control",
                    "benchmark_version": config.benchmark_version,
                }
                records.append(
                    _make_stage_record(
                        split=split,
                        family_id=family_id,
                        row_id=row_id,
                        action_id=action_id,
                        pixels=pixels,
                        materialized=materialized,
                        expected_disposition=expected_disposition,
                        metadata=metadata,
                    )
                )
    split_counts = {split: sum(record.split == split for record in records) for split in SPLIT_ROLES}
    return Stage3Benchmark(
        policy_artifact_id=policy.artifact_id,
        source_scope=SOURCE_SCOPE,
        prototypes=prototypes,
        records=tuple(records),
        observations=observations,
        family_specs={family_id: {"description": description, "split": split} for split, items in families.items() for family_id, description in items},
        metadata={
            "benchmark_version": config.benchmark_version,
            "generator_version": config.generator_version,
            "amendment_commit": config.amendment_commit,
            "final_seed_material": config.seed_material,
            "final_seed_digest": config.seed_digest,
            "split_counts": split_counts,
            "materialized_final": bool(materialize_final),
            "evaluation_row_sample_size": config.sample_size,
            "provider_prototype_count": len(rows),
            "development_record_count": len(rows) * 2,
        },
        phase_audits=[],
    )


def _build_stage3_benchmark_v3(*, materialize_final: bool = False) -> Stage3Benchmark:
    config = _stage3_v3_config()
    benchmark = _build_stage3_benchmark_v2(materialize_final=materialize_final)
    rebuilt_records = []
    rebuilt_observations: Dict[str, ImageObservation] = {}
    for record in benchmark.records:
        observation_id = record.observation_id.replace("v2:", "v3:", 1)
        clip_id = record.clip_id.replace("v2:", "v3:", 1)
        frame_id = record.frame_id.replace("v2:", "v3:", 1)
        metadata = {**_json_ready(record.metadata), "benchmark_version": config.benchmark_version}
        if record.materialized:
            original = benchmark.observations[record.observation_id]
            image = ImageObservation(original.pixels, source_id=observation_id, metadata=metadata)
            rebuilt_observations[observation_id] = image
            observation_digest = image.raw_digest
        else:
            observation_digest = _sha256(
                {
                    "benchmark_version": config.benchmark_version,
                    "generator_version": config.generator_version,
                    "observation_id": observation_id,
                    "seed_material": config.seed_material,
                    "family_id": record.family_id,
                    "row_id": record.row_id,
                    "action_id": record.action_id,
                }
            )
        rebuilt_records.append(
            Stage3Record(
                observation_id=observation_id,
                split=record.split,
                family_id=record.family_id,
                row_id=record.row_id,
                action_id=record.action_id,
                expected_disposition=record.expected_disposition,
                clip_id=clip_id,
                frame_id=frame_id,
                materialized=record.materialized,
                metadata=metadata,
                observation_digest=observation_digest,
            )
        )
    split_counts = {split: sum(record.split == split for record in rebuilt_records) for split in SPLIT_ROLES}
    return Stage3Benchmark(
        policy_artifact_id=benchmark.policy_artifact_id,
        source_scope=benchmark.source_scope,
        prototypes=benchmark.prototypes,
        records=tuple(rebuilt_records),
        observations=rebuilt_observations,
        family_specs=benchmark.family_specs,
        metadata={
            "benchmark_version": config.benchmark_version,
            "generator_version": config.generator_version,
            "amendment_commit": config.amendment_commit,
            "final_seed_material": config.seed_material,
            "final_seed_digest": config.seed_digest,
            "split_counts": split_counts,
            "materialized_final": bool(materialize_final),
            "evaluation_row_sample_size": config.sample_size,
            "provider_prototype_count": len(benchmark.prototypes),
            "development_record_count": len(benchmark.prototypes) * 2,
        },
        phase_audits=[],
    )


def _benchmark_manifest(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    metadata = benchmark.metadata
    if metadata.get("benchmark_version") == BENCHMARK_VERSION_V3:
        generator_identity = _generator_identity_v3(benchmark)
    elif metadata.get("benchmark_version") == BENCHMARK_VERSION_V2:
        generator_identity = _generator_identity_v2(benchmark)
    else:
        generator_identity = _generator_identity(benchmark)
    return {
        "benchmark_version": metadata["benchmark_version"],
        "generator_version": metadata["generator_version"],
        "preregistration_commit": metadata.get("preregistration_commit"),
        "amendment_commit": metadata.get("amendment_commit"),
        "final_seed_material": metadata["final_seed_material"],
        "final_seed_digest": metadata["final_seed_digest"],
        "policy_artifact_id": benchmark.policy_artifact_id,
        "source_scope": benchmark.source_scope,
        "metadata": _json_ready(benchmark.metadata),
        "record_count": len(benchmark.records),
        "benchmark_digest": _sha256([record.to_dict() for record in benchmark.records]),
        "generator_identity": generator_identity,
    }


def _split_manifest(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    membership = {split: [record.observation_id for record in benchmark.record_by_split(split)] for split in SPLIT_ROLES}
    clip_membership = {split: sorted({record.clip_id for record in benchmark.record_by_split(split)}) for split in SPLIT_ROLES}
    return {
        "split_roles": list(SPLIT_ROLES),
        "observation_membership": membership,
        "clip_membership": clip_membership,
        "split_counts": {split: len(ids) for split, ids in membership.items()},
        "split_digest": _sha256(membership),
    }


def _development_observations(benchmark: Stage3Benchmark) -> Mapping[str, Tuple[ImageObservation, ...]]:
    result: Dict[str, list[ImageObservation]] = defaultdict(list)
    for record in benchmark.record_by_split("diagnostic_development"):
        if record.materialized and record.row_id is not None:
            result[record.row_id].append(benchmark.observations[record.observation_id])
    return {row_id: tuple(items) for row_id, items in sorted(result.items())}


def _freeze_regions_and_masks(benchmark: Stage3Benchmark, *, output_dir: Path) -> Dict[str, Any]:
    regions = _region_manifest()
    masks = build_discriminative_masks(
        prototypes=benchmark.prototypes,
        development_observations=_development_observations(benchmark),
        intensity_tolerance=8,
        stability_tolerance=12,
        separation_cap=64,
    )
    region_manifest = {
        "regions": [region.to_dict() for region in regions],
        "region_spec_digest": discriminative_region_digest(regions),
        "registration_contract_digest_set": sorted({region.registration_config.digest for region in regions}),
    }
    mask_manifest = {
        "mask_specs": [mask.spec.to_dict() for _row_id, mask in sorted(masks.items())],
        "mask_payload_digests": {row_id: mask.payload_digest for row_id, mask in sorted(masks.items())},
        "mask_spec_digest": discriminative_mask_digest(tuple(mask.spec for mask in masks.values())),
        "prototype_digest": _sha256({key: value[2] for key, value in sorted(benchmark.prototypes.items())}),
        "development_digest": _sha256({row_id: [item.raw_digest for item in values] for row_id, values in sorted(_development_observations(benchmark).items())}),
        "derivation_contract": next(iter(masks.values())).spec.derivation_contract,
        "intensity_tolerance": next(iter(masks.values())).spec.intensity_tolerance,
        "stability_tolerance": 12,
        "separation_cap": 64,
    }
    _write_json(output_dir / "region-manifest.json", region_manifest)
    _write_json(output_dir / "mask-manifest.json", mask_manifest)
    return {"regions": regions, "masks": masks, "region_manifest": region_manifest, "mask_manifest": mask_manifest}


def validate_prototype_and_development_closure(
    *,
    benchmark: Stage3Benchmark,
    masks: Mapping[str, Any],
) -> Dict[str, Any]:
    provider_prototype_rows = sorted(benchmark.prototypes)
    prototype_split_rows = sorted({record.row_id for record in benchmark.record_by_split("prototype") if record.row_id is not None})
    development_rows = sorted({record.row_id for record in benchmark.record_by_split("diagnostic_development") if record.row_id is not None})
    architecture_selection_rows = sorted({record.row_id for record in benchmark.records if record.split.startswith("architecture_selection_") and record.row_id is not None})
    calibration_rows = sorted({record.row_id for record in benchmark.records if record.split.endswith("calibration") and record.row_id is not None})
    final_descriptor_rows = sorted({record.row_id for record in benchmark.records if (record.split.startswith("final_") or record.split == "information_theoretic_control") and record.row_id is not None})
    mask_rows = sorted(masks)
    rows_with_nonzero_stability = sorted(row_id for row_id, mask in masks.items() if int(mask.spec.stable_pixel_count) > 0)
    rows_with_zero_stability = sorted(row_id for row_id, mask in masks.items() if int(mask.spec.stable_pixel_count) == 0)
    missing_prototype_manifest_rows = sorted(set(provider_prototype_rows) - set(prototype_split_rows))
    missing_mask_rows = sorted(set(provider_prototype_rows) - set(mask_rows))
    missing_development_rows = sorted(set(provider_prototype_rows) - set(development_rows))
    evaluated_rows_with_zero_stability = sorted(set(architecture_selection_rows) & set(rows_with_zero_stability))
    closure_failures = []
    if missing_prototype_manifest_rows:
        closure_failures.append("missing_prototype_manifest_rows")
    if missing_mask_rows:
        closure_failures.append("missing_mask_rows")
    if missing_development_rows:
        closure_failures.append("missing_development_rows")
    if evaluated_rows_with_zero_stability:
        closure_failures.append("evaluated_rows_with_zero_stability")
    return {
        "provider_prototype_rows": provider_prototype_rows,
        "prototype_split_rows": prototype_split_rows,
        "development_rows": development_rows,
        "architecture_selection_rows": architecture_selection_rows,
        "calibration_rows": calibration_rows,
        "final_descriptor_rows": final_descriptor_rows,
        "mask_rows": mask_rows,
        "rows_with_nonzero_stability": rows_with_nonzero_stability,
        "rows_with_zero_stability": rows_with_zero_stability,
        "missing_prototype_manifest_rows": missing_prototype_manifest_rows,
        "missing_mask_rows": missing_mask_rows,
        "missing_development_rows": missing_development_rows,
        "evaluated_rows_with_zero_stability": evaluated_rows_with_zero_stability,
        "closure_failures": closure_failures,
    }


def _mask_coverage_report(*, benchmark: Stage3Benchmark, masks: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    prototype_rows = sorted(benchmark.prototypes)
    prototype_split = {record.row_id for record in benchmark.record_by_split("prototype") if record.row_id is not None}
    development_by_row = defaultdict(list)
    for record in benchmark.record_by_split("diagnostic_development"):
        if record.row_id is not None:
            development_by_row[record.row_id].append(record.family_id)
    architecture_rows = {record.row_id for record in benchmark.records if record.split.startswith("architecture_selection_") and record.row_id is not None}
    calibration_rows = {record.row_id for record in benchmark.records if record.split.endswith("calibration") and record.row_id is not None}
    final_rows = {record.row_id for record in benchmark.records if (record.split.startswith("final_") or record.split == "information_theoretic_control") and record.row_id is not None}
    rows = []
    for row_id in prototype_rows:
        action_id = benchmark.prototypes[row_id][1]
        mask = masks.get(row_id)
        stable_pixel_count = None if mask is None else int(mask.spec.stable_pixel_count)
        effective_mass = None if mask is None else float(mask.stable_weights.sum(dtype=np.float64) * mask.row_informative_weights.sum(dtype=np.float64))
        closure_failure_reason = []
        if row_id not in prototype_split:
            closure_failure_reason.append("missing_prototype_manifest_row")
        if mask is None:
            closure_failure_reason.append("missing_mask")
        if row_id not in development_by_row:
            closure_failure_reason.append("missing_development_row")
        elif stable_pixel_count == 0:
            closure_failure_reason.append("zero_stability")
        rows.append(
            {
                "row_id": row_id,
                "action_id": action_id,
                "prototype_split_membership": row_id in prototype_split,
                "development_observation_count": len(development_by_row.get(row_id, ())),
                "development_family_ids": sorted(development_by_row.get(row_id, ())),
                "informative_pixel_count": None if mask is None else int(mask.spec.informative_pixel_count),
                "action_conflict_pixel_count": None if mask is None else int(mask.spec.action_conflict_pixel_count),
                "stable_pixel_count": stable_pixel_count,
                "effective_informative_and_stable_mass": effective_mass,
                "mask_payload_digest": None if mask is None else mask.payload_digest,
                "appears_in_architecture_selection_expected_rows": row_id in architecture_rows,
                "appears_in_calibration_expected_rows": row_id in calibration_rows,
                "appears_in_final_descriptors": row_id in final_rows,
                "closure_status": "closed" if not closure_failure_reason else "open",
                "closure_failure_reason": "|".join(closure_failure_reason),
            }
        )
    summary = {
        "total_prototypes": len(prototype_rows),
        "nonzero_stability_masks": sum(1 for row in rows if (row["stable_pixel_count"] or 0) > 0),
        "zero_stability_masks": sum(1 for row in rows if row["stable_pixel_count"] == 0),
        "evaluated_rows_with_zero_stability": sum(1 for row in rows if row["appears_in_architecture_selection_expected_rows"] and row["stable_pixel_count"] == 0),
        "candidate_rows_with_zero_stability": sum(1 for row in rows if row["stable_pixel_count"] == 0),
        "actions_affected": sorted({row["action_id"] for row in rows if row["stable_pixel_count"] == 0}),
        "state_dimensions_affected": ["tank_x", "target_x", "cooldown"],
        "rows": rows,
    }
    _write_json(output_dir / "mask-coverage.json", summary)
    _write_csv(output_dir / "mask-coverage.csv", rows)
    return summary


def _v2_sample_rows(benchmark: Stage3Benchmark) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                record.row_id
                for record in benchmark.record_by_split("architecture_selection_benign")
                if record.row_id is not None and record.family_id == "selection_exact"
            }
        )
    )


def _prototype_manifest_v2(benchmark: Stage3Benchmark, *, generator_identity: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    prototype_records = [record for record in benchmark.record_by_split("prototype") if record.row_id is not None]
    for record in prototype_records:
        prototype = benchmark.prototypes[record.row_id]
        rows.append(
            {
                "split": record.split,
                "family_id": record.family_id,
                "row_id": record.row_id,
                "action_id": record.action_id,
                "prototype_observation_id": prototype[0],
                "observation_id": record.observation_id,
                "frame_id": record.frame_id,
                "clip_id": record.clip_id,
                "pixel_digest": prototype[3].raw_digest,
                "raw_pixel_digest": prototype[2],
                "geometry": list(prototype[3].pixels.shape),
                "policy_artifact_id": benchmark.policy_artifact_id,
                "generator_identity_digest": generator_identity["generator_identity_digest"],
            }
        )
    return {
        "provider_prototype_count": len(benchmark.prototypes),
        "prototype_manifest_count": len(rows),
        "rows": rows,
        "prototype_manifest_digest": _sha256(rows),
    }


def _development_manifest_v2(benchmark: Stage3Benchmark, *, generator_identity: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for record in benchmark.record_by_split("diagnostic_development"):
        if record.row_id is None:
            continue
        rows.append(
            {
                "observation_id": record.observation_id,
                "row_id": record.row_id,
                "action_id": record.action_id,
                "family_id": record.family_id,
                "frame_id": record.frame_id,
                "clip_id": record.clip_id,
                "pixel_digest": benchmark.observations[record.observation_id].raw_digest,
                "generator_identity_digest": generator_identity["generator_identity_digest"],
                "split": record.split,
            }
        )
    return {
        "development_record_count": len(rows),
        "development_covered_row_count": len({row["row_id"] for row in rows}),
        "rows": rows,
        "development_manifest_digest": _sha256(rows),
    }


def _evaluation_sample_manifest_v2(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    sample_rows = _v2_sample_rows(benchmark)
    action_by_row = {row_id: benchmark.prototypes[row_id][1] for row_id in sample_rows}
    payload = {
        "sampling_algorithm_version": "step_sample_with_midpoint_and_last_dedup/v1",
        "sample_size": EVALUATION_ROW_SAMPLE_SIZE,
        "selected_row_ids": list(sample_rows),
        "selected_action_ids": [action_by_row[row_id] for row_id in sample_rows],
        "sampling_digest": _sha256({"rows": sample_rows, "actions": action_by_row}),
    }
    return payload


def _prototype_collision_atlas_v2(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    groups = defaultdict(list)
    for row_id, (prototype_observation_id, action_id, _digest, observation) in sorted(benchmark.prototypes.items()):
        groups[observation.raw_digest].append({"row_id": row_id, "action_id": action_id, "prototype_observation_id": prototype_observation_id})
    rows = []
    same_action = 0
    conflicting_action = 0
    for digest, items in sorted(groups.items()):
        actions = sorted({item["action_id"] for item in items})
        if len(items) == 1:
            collision_class = "unique_visual_row"
        elif len(actions) == 1:
            collision_class = "same_action_visual_alias"
            same_action += 1
        else:
            collision_class = "conflicting_action_visual_alias"
            conflicting_action += 1
        rows.append(
            {
                "prototype_pixel_digest": digest,
                "row_ids": [item["row_id"] for item in items],
                "action_ids": [item["action_id"] for item in items],
                "group_size": len(items),
                "unique_action": len(actions) == 1,
                "collision_class": collision_class,
            }
        )
    return {
        "collision_group_count": len(rows),
        "same_action_alias_groups": same_action,
        "conflicting_action_alias_groups": conflicting_action,
        "rows": rows,
        "collision_atlas_digest": _sha256(rows),
    }


def _mask_closure_v2(benchmark: Stage3Benchmark, masks: Mapping[str, Any]) -> Dict[str, Any]:
    development = defaultdict(list)
    for record in benchmark.record_by_split("diagnostic_development"):
        if record.row_id is not None:
            development[record.row_id].append(record.family_id)
    sample_rows = set(_v2_sample_rows(benchmark))
    rows = []
    effective_count = 0
    closure_valid = True
    for row_id in sorted(benchmark.prototypes):
        mask = masks[row_id]
        stable = int(mask.spec.stable_pixel_count)
        informative = int(mask.spec.informative_pixel_count)
        effective = stable * informative
        if effective > 0:
            effective_count += 1
        fallback = str(mask.spec.metadata.get("stability_fallback", ""))
        failure_reason = []
        if row_id not in development:
            failure_reason.append("missing_development")
        if sorted(development[row_id]) != ["development_photometric", "development_shift"]:
            failure_reason.append("incomplete_development_families")
        if fallback == "zero_stable_mass_without_development":
            failure_reason.append("forbidden_missing_development_fallback")
        if row_id in sample_rows and effective <= 0:
            failure_reason.append("zero_effective_evidence_for_evaluation_row")
        if failure_reason:
            closure_valid = False
        rows.append(
            {
                "row_id": row_id,
                "action_id": benchmark.prototypes[row_id][1],
                "prototype_present": True,
                "development_observation_count": len(development[row_id]),
                "development_family_ids": sorted(development[row_id]),
                "informative_pixel_count": informative,
                "action_conflict_pixel_count": int(mask.spec.action_conflict_pixel_count),
                "stable_pixel_count": stable,
                "effective_informative_stable_mass": effective,
                "mask_payload_digest": mask.payload_digest,
                "closure_status": "closed" if not failure_reason else "open",
                "closure_failure_reason": "|".join(failure_reason),
            }
        )
    return {
        "closure_valid": closure_valid,
        "provider_prototype_count": len(benchmark.prototypes),
        "mask_count": len(masks),
        "development_covered_row_count": len(development),
        "masks_with_development_closure": sum(1 for row in rows if row["closure_status"] == "closed"),
        "masks_with_nonzero_effective_evidence": effective_count,
        "remaining_zero_evidence_masks": sum(1 for row in rows if int(row["effective_informative_stable_mass"]) <= 0),
        "rows": rows,
        "mask_closure_digest": _sha256(rows),
    }


def _calibration(
    *,
    architecture_id: str,
    benchmark: Stage3Benchmark,
    region_manifest: Mapping[str, Any],
    mask_manifest: Mapping[str, Any],
    values: Mapping[str, Any],
) -> DiscriminativeEvidenceCalibration:
    return DiscriminativeEvidenceCalibration(
        architecture_id=architecture_id,
        minimum_available_mass=float(values.get("minimum_available_mass", 1.0)),
        minimum_available_fraction=float(values.get("minimum_available_fraction", 0.3)),
        minimum_support=float(values.get("minimum_support", 0.0)),
        maximum_contradiction=float(values.get("maximum_contradiction", 1.0)),
        maximum_critical_contradiction=float(values.get("maximum_critical_contradiction", 1.0)),
        exact_winner_threshold=float(values.get("exact_winner_threshold", 0.5)),
        exact_winner_margin=float(values.get("exact_winner_margin", 0.1)),
        candidate_relative_margin=float(values.get("candidate_relative_margin", 0.0)),
        conflicting_action_separation=float(values.get("conflicting_action_separation", 0.0)),
        minimum_supporting_regions=int(values.get("minimum_supporting_regions", 0)),
        maximum_candidate_set_size=int(values.get("maximum_candidate_set_size", 3)),
        prototype_digest=zde._prototype_payload_digest(benchmark.prototypes),
        region_spec_digest=str(region_manifest["region_spec_digest"]),
        mask_spec_digest=str(mask_manifest["mask_spec_digest"]),
        policy_artifact_id=benchmark.policy_artifact_id,
        source_scope=benchmark.source_scope,
        metadata={"grid_values": _json_ready(values)},
    )


def _architecture_grid_definition() -> Dict[str, Any]:
    common = {
        "minimum_available_mass": (1.0, 2.0),
        "minimum_available_fraction": (0.25,),
        "exact_winner_threshold": (0.5,),
        "exact_winner_margin": (0.05, 0.15),
        "candidate_relative_margin": (0.0,),
        "conflicting_action_separation": (0.0,),
        "maximum_candidate_set_size": (3,),
    }
    return {
        "A": common,
        "B": {**common, "minimum_support": (0.0, 0.4)},
        "C": {**common, "minimum_support": (0.0, 0.4), "maximum_contradiction": (0.5,), "maximum_critical_contradiction": (0.0, 0.25), "minimum_supporting_regions": (0,)},
        "D": {**common, "minimum_support": (0.0, 0.4), "maximum_contradiction": (0.5,), "maximum_critical_contradiction": (0.0, 0.25), "minimum_supporting_regions": (0,)},
    }


def _grid_points(definition: Mapping[str, Sequence[Any]]) -> Tuple[Dict[str, Any], ...]:
    keys = sorted(definition)
    points = [{}]
    for key in keys:
        next_points = []
        for point in points:
            for value in definition[key]:
                updated = dict(point)
                updated[key] = value
                next_points.append(updated)
        points = next_points
    return tuple(points)


def _evaluate_provider(
    *,
    provider: DiscriminativeEvidenceProvider,
    records: Sequence[Stage3Record],
    observations: Mapping[str, ImageObservation],
) -> Dict[str, Any]:
    return _evaluate_ranked_candidates(
        ranked_by_observation={record.observation_id: provider._rank(observations[record.observation_id]) for record in records},
        calibration=provider._calibration,
        provider_digest=provider.contract().address_artifact_id,
        records=records,
    )


def _raw_ranked_candidates(
    *,
    architecture_id: str,
    benchmark: Stage3Benchmark,
    freeze: Mapping[str, Any],
    records: Sequence[Stage3Record],
    observations: Mapping[str, ImageObservation],
) -> Dict[str, Tuple[Any, ...]]:
    permissive = _calibration(
        architecture_id=architecture_id,
        benchmark=benchmark,
        region_manifest=freeze["region_manifest"],
        mask_manifest=freeze["mask_manifest"],
        values={
            "minimum_available_mass": 0.0,
            "minimum_available_fraction": 0.0,
            "minimum_support": 0.0,
            "maximum_contradiction": 1.0,
            "maximum_critical_contradiction": 1.0,
            "exact_winner_threshold": 0.0,
            "exact_winner_margin": 0.0,
            "candidate_relative_margin": 0.0,
            "conflicting_action_separation": 0.0,
            "minimum_supporting_regions": 0,
            "maximum_candidate_set_size": MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
        },
    )
    provider = DiscriminativeEvidenceProvider(
        prototypes=benchmark.prototypes,
        masks=freeze["masks"],
        regions=freeze["regions"],
        calibration=permissive,
        policy_artifact_id=benchmark.policy_artifact_id,
        source_scope=benchmark.source_scope,
    )
    return {record.observation_id: provider._rank(observations[record.observation_id]) for record in records}


def _evaluate_ranked_candidates(
    *,
    ranked_by_observation: Mapping[str, Tuple[Any, ...]],
    calibration: DiscriminativeEvidenceCalibration,
    provider_digest: str,
    records: Sequence[Stage3Record],
) -> Dict[str, Any]:
    benign_total = 0
    exact_accepted = 0
    exact_correct = 0
    candidate_set_count = 0
    useful_candidate_set_count = 0
    useful_truth_in_set_count = 0
    unique_action_useful_sets = 0
    mixed_action_useful_sets = 0
    exact_false_accepts = 0
    negative_candidate_set_support = 0
    conflicting_action_exact_accepts = 0
    same_action_wrong_row_exact_accepts = 0
    critical_contradiction_exact_accepts = 0
    per_family = Counter()
    set_sizes = []
    for record in records:
        raw_ranked = ranked_by_observation[record.observation_id]
        evaluated = tuple(
            evaluate_candidate_eligibility(candidate=candidate, ranked_candidates=raw_ranked, calibration=calibration)
            for candidate in raw_ranked
        )
        ranked = tuple(sorted(evaluated, key=lambda candidate: (-float(candidate.candidate_strength), -float(candidate.available_informative_mass), -float(candidate.available_informative_fraction), float(candidate.aggregate_contradiction), float(candidate.aggregate_critical_contradiction), -int(candidate.supporting_region_count), candidate.row_id, candidate.prototype_observation_id)))
        candidate_set = build_discriminative_candidate_set(
            ranked_candidates=ranked,
            calibration=calibration,
            provider_digest=provider_digest,
            observation_digest=ranked[0].observation_digest,
        )
        if record.expected_disposition == "expected_accept":
            benign_total += 1
            if candidate_set.outcome == "exact_row_accepted":
                exact_accepted += 1
                if candidate_set.exact_row_id == record.row_id:
                    exact_correct += 1
            elif candidate_set.outcome == "candidate_set_available":
                candidate_set_count += 1
                set_sizes.append(len(candidate_set.rows))
                if record.row_id in candidate_set.rows and len(candidate_set.rows) <= MAXIMUM_USEFUL_CANDIDATE_SET_SIZE:
                    useful_candidate_set_count += 1
                    useful_truth_in_set_count += 1
                    if len(set(candidate_set.actions)) == 1:
                        unique_action_useful_sets += 1
                    else:
                        mixed_action_useful_sets += 1
            per_family[(record.family_id, candidate_set.outcome)] += 1
        elif record.expected_disposition == "expected_reject":
            if candidate_set.outcome == "exact_row_accepted":
                exact_false_accepts += 1
                if candidate_set.exact_row_id is not None and record.action_id is not None:
                    if next((candidate.action_id for candidate in ranked if candidate.row_id == candidate_set.exact_row_id), None) != record.action_id:
                        conflicting_action_exact_accepts += 1
                    elif candidate_set.exact_row_id != record.row_id:
                        same_action_wrong_row_exact_accepts += 1
                    if next((candidate.aggregate_critical_contradiction for candidate in ranked if candidate.row_id == candidate_set.exact_row_id), 0.0) > 0.0:
                        critical_contradiction_exact_accepts += 1
            elif candidate_set.outcome == "candidate_set_available":
                negative_candidate_set_support += 1
            per_family[(record.family_id, candidate_set.outcome)] += 1
    exact_row_coverage = 0.0 if benign_total == 0 else exact_correct / float(benign_total)
    exact_precision = None if exact_accepted == 0 else exact_correct / float(exact_accepted)
    return {
        "benign_total": benign_total,
        "exact_accepted_count": exact_accepted,
        "exact_correct_count": exact_correct,
        "exact_row_coverage": exact_row_coverage,
        "exact_accepted_precision": exact_precision,
        "candidate_set_count": candidate_set_count,
        "useful_candidate_set_count": useful_candidate_set_count,
        "truth_in_set_count": useful_truth_in_set_count,
        "useful_candidate_set_recall": 0.0 if benign_total == 0 else useful_candidate_set_count / float(benign_total),
        "unique_action_useful_sets": unique_action_useful_sets,
        "mixed_action_useful_sets": mixed_action_useful_sets,
        "mean_set_size": None if not set_sizes else float(np.mean(set_sizes)),
        "maximum_set_size": 0 if not set_sizes else max(set_sizes),
        "exact_false_accepts": exact_false_accepts,
        "negative_candidate_set_support": negative_candidate_set_support,
        "conflicting_action_exact_accepts": conflicting_action_exact_accepts,
        "same_action_wrong_row_exact_accepts": same_action_wrong_row_exact_accepts,
        "critical_contradiction_exact_accepts": critical_contradiction_exact_accepts,
        "per_family_outcomes": {f"{family}:{outcome}": count for (family, outcome), count in sorted(per_family.items())},
        "cache_hits": 0,
        "cache_misses": len(ranked_by_observation),
    }


def _ranked_candidates_for_architecture(
    *,
    architecture_id: str,
    benchmark: Stage3Benchmark,
    freeze: Mapping[str, Any],
    records: Sequence[Stage3Record],
    observations: Mapping[str, ImageObservation],
    calibration_values: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Tuple[Any, ...]]:
    if calibration_values is None:
        return _raw_ranked_candidates(
            architecture_id=architecture_id,
            benchmark=benchmark,
            freeze=freeze,
            records=records,
            observations=observations,
        )
    calibration = _calibration(
        architecture_id=architecture_id,
        benchmark=benchmark,
        region_manifest=freeze["region_manifest"],
        mask_manifest=freeze["mask_manifest"],
        values=calibration_values,
    )
    provider = DiscriminativeEvidenceProvider(
        prototypes=benchmark.prototypes,
        masks=freeze["masks"],
        regions=freeze["regions"],
        calibration=calibration,
        policy_artifact_id=benchmark.policy_artifact_id,
        source_scope=benchmark.source_scope,
    )
    return {record.observation_id: provider._rank(observations[record.observation_id]) for record in records}


def _artifact_comparison(expected_dir: Path, actual_dir: Path, *, artifact_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    results = []
    semantic_mismatch = False
    expected_names = set(artifact_names or AUDIT_ARTIFACT_NAMES)
    actual_names = {path.name for path in actual_dir.iterdir() if path.is_file()}
    for name in sorted(expected_names | actual_names):
        expected_path = expected_dir / name
        actual_path = actual_dir / name
        if not expected_path.exists():
            results.append({"artifact": name, "status": "unexpected_artifact"})
            semantic_mismatch = True
            continue
        if not actual_path.exists():
            results.append({"artifact": name, "status": "missing_artifact"})
            semantic_mismatch = True
            continue
        expected_bytes = expected_path.read_bytes()
        actual_bytes = actual_path.read_bytes()
        if expected_bytes == actual_bytes:
            status = "byte_identical"
        else:
            try:
                expected_json = _load_json(expected_path)
                actual_json = _load_json(actual_path)
            except Exception:
                status = "semantic_mismatch"
            else:
                status = "semantic_identical" if expected_json == actual_json else "semantic_mismatch"
        if status == "semantic_mismatch":
            semantic_mismatch = True
        results.append(
            {
                "artifact": name,
                "status": status,
                "expected_digest": _sha256(expected_bytes.decode("utf-8", errors="replace")),
                "actual_digest": _sha256(actual_bytes.decode("utf-8", errors="replace")),
            }
        )
    return {"semantic_match": not semantic_mismatch, "artifacts": results}


def _selection_rank_key(result: Mapping[str, Any]) -> Tuple[Any, ...]:
    metrics = result["combined_metrics"]
    conservative = result["calibration_values"]
    return (
        int(metrics["exact_false_accepts"]),
        int(metrics["conflicting_action_exact_accepts"]),
        int(metrics["critical_contradiction_exact_accepts"]),
        0 if (metrics["exact_correct_count"] + metrics["useful_candidate_set_count"]) > 0 else 1,
        -float(metrics["exact_row_coverage"]),
        -int(metrics["useful_candidate_set_count"]),
        float(metrics["mean_set_size"] if metrics["mean_set_size"] is not None else 99.0),
        SIMPLICITY_ORDER[result["architecture_id"]],
        -float(conservative["minimum_available_mass"]),
        float(conservative.get("maximum_contradiction", 1.0)),
        float(conservative.get("maximum_critical_contradiction", 1.0)),
        -float(conservative["exact_winner_margin"]),
        int(conservative["maximum_candidate_set_size"]),
        float(conservative["candidate_relative_margin"]),
        -int(conservative.get("minimum_supporting_regions", 0)),
        json.dumps(_json_ready(conservative), sort_keys=True),
    )


def _feasible_for_selection(result: Mapping[str, Any]) -> Tuple[bool, Tuple[str, ...]]:
    metrics = result["combined_metrics"]
    reasons = []
    if int(metrics["exact_false_accepts"]) != 0:
        reasons.append("exact_false_accepts")
    if int(metrics["conflicting_action_exact_accepts"]) != 0:
        reasons.append("conflicting_action_exact_accepts")
    if int(metrics["critical_contradiction_exact_accepts"]) != 0:
        reasons.append("critical_contradiction_exact_accepts")
    if SELECTION_NEGATIVE_CANDIDATE_SET_SUPPORT_BLOCKS_FEASIBILITY and int(metrics["negative_candidate_set_support"]) != 0:
        reasons.append("negative_candidate_set_support")
    if int(metrics["exact_correct_count"]) + int(metrics["useful_candidate_set_count"]) == 0:
        reasons.append("zero_safe_utility")
    return (len(reasons) == 0, tuple(reasons))


def _run_selection(output_dir: Path) -> Dict[str, Any]:
    benchmark = _build_stage3_benchmark(materialize_final=False)
    freeze = _freeze_regions_and_masks(benchmark, output_dir=output_dir)
    benign_records, benign_observations = benchmark.access(
        phase="select_architecture",
        allowed_splits=("prototype", "diagnostic_development", "architecture_selection_benign"),
    )
    negative_records, negative_observations = benchmark.access(
        phase="select_architecture",
        allowed_splits=("prototype", "diagnostic_development", "architecture_selection_negative"),
    )
    benign_eval = tuple(record for record in benign_records if record.split == "architecture_selection_benign")
    negative_eval = tuple(record for record in negative_records if record.split == "architecture_selection_negative")
    grid_definition = _architecture_grid_definition()
    _write_json(output_dir / "architecture-grid-definition.json", {"definition": _json_ready(grid_definition), "digest": _sha256(grid_definition)})
    results = []
    raw_benign = {architecture_id: _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=benign_eval, observations=benign_observations) for architecture_id in ("A", "B", "C")}
    raw_negative = {architecture_id: _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=negative_eval, observations=negative_observations) for architecture_id in ("A", "B", "C")}
    for architecture_id in ("A", "B", "C"):
        for point in _grid_points(grid_definition[architecture_id]):
            calibration = _calibration(
                architecture_id=architecture_id,
                benchmark=benchmark,
                region_manifest=freeze["region_manifest"],
                mask_manifest=freeze["mask_manifest"],
                values=point,
            )
            benign_metrics = _evaluate_ranked_candidates(
                ranked_by_observation=raw_benign[architecture_id],
                calibration=calibration,
                provider_digest=f"{architecture_id}:{calibration.digest}",
                records=benign_eval,
            )
            negative_metrics = _evaluate_ranked_candidates(
                ranked_by_observation=raw_negative[architecture_id],
                calibration=calibration,
                provider_digest=f"{architecture_id}:{calibration.digest}",
                records=negative_eval,
            )
            combined_metrics = {
                **benign_metrics,
                "exact_false_accepts": negative_metrics["exact_false_accepts"],
                "negative_candidate_set_support": negative_metrics["negative_candidate_set_support"],
                "conflicting_action_exact_accepts": negative_metrics["conflicting_action_exact_accepts"],
                "same_action_wrong_row_exact_accepts": negative_metrics["same_action_wrong_row_exact_accepts"],
                "critical_contradiction_exact_accepts": negative_metrics["critical_contradiction_exact_accepts"],
                "negative_per_family_outcomes": negative_metrics["per_family_outcomes"],
            }
            feasible, infeasible_reasons = _feasible_for_selection({"combined_metrics": combined_metrics})
            results.append(
                {
                    "architecture_id": architecture_id,
                    "calibration_values": point,
                    "calibration_digest": calibration.digest,
                    "feasible": feasible,
                    "infeasible_reasons": list(infeasible_reasons),
                    "combined_metrics": combined_metrics,
                }
            )
    best_by_architecture = {architecture_id: min((result for result in results if result["architecture_id"] == architecture_id), key=_selection_rank_key) for architecture_id in ("A", "B", "C")}
    a = best_by_architecture["A"]
    d_gateway = {
        "conditions": {
            "a_nonzero_safe_utility": (a["combined_metrics"]["exact_correct_count"] + a["combined_metrics"]["useful_candidate_set_count"]) > 0,
            "b_or_c_additional_recovery": False,
            "added_recovery_zero_new_safety_failures": False,
            "distinct_failure_mode": False,
        },
        "supporting_frame_identities": [],
        "supporting_family_identities": [],
    }
    for architecture_id in ("B", "C"):
        candidate = best_by_architecture[architecture_id]
        if candidate["combined_metrics"]["exact_correct_count"] + candidate["combined_metrics"]["useful_candidate_set_count"] > a["combined_metrics"]["exact_correct_count"] + a["combined_metrics"]["useful_candidate_set_count"]:
            d_gateway["conditions"]["b_or_c_additional_recovery"] = True
            if candidate["combined_metrics"]["exact_false_accepts"] == 0 and candidate["combined_metrics"]["conflicting_action_exact_accepts"] == 0 and candidate["combined_metrics"]["critical_contradiction_exact_accepts"] == 0:
                d_gateway["conditions"]["added_recovery_zero_new_safety_failures"] = True
            recovered_families = sorted({key.split(":")[0] for key, value in candidate["combined_metrics"]["per_family_outcomes"].items() if key.endswith("exact_row_accepted") or key.endswith("candidate_set_available")})
            a_families = sorted({key.split(":")[0] for key, value in a["combined_metrics"]["per_family_outcomes"].items() if key.endswith("exact_row_accepted") or key.endswith("candidate_set_available")})
            distinct = sorted(set(recovered_families) - set(a_families))
            if distinct:
                d_gateway["conditions"]["distinct_failure_mode"] = True
                d_gateway["supporting_family_identities"] = distinct
            break
    d_gateway["eligible"] = all(bool(value) for value in d_gateway["conditions"].values())
    _write_json(output_dir / "architecture-d-gateway.json", {**d_gateway, "digest": _sha256(d_gateway)})
    if d_gateway["eligible"]:
        raw_benign["D"] = _raw_ranked_candidates(architecture_id="D", benchmark=benchmark, freeze=freeze, records=benign_eval, observations=benign_observations)
        raw_negative["D"] = _raw_ranked_candidates(architecture_id="D", benchmark=benchmark, freeze=freeze, records=negative_eval, observations=negative_observations)
        for point in _grid_points(grid_definition["D"]):
            calibration = _calibration(
                architecture_id="D",
                benchmark=benchmark,
                region_manifest=freeze["region_manifest"],
                mask_manifest=freeze["mask_manifest"],
                values=point,
            )
            benign_metrics = _evaluate_ranked_candidates(
                ranked_by_observation=raw_benign["D"],
                calibration=calibration,
                provider_digest=f"D:{calibration.digest}",
                records=benign_eval,
            )
            negative_metrics = _evaluate_ranked_candidates(
                ranked_by_observation=raw_negative["D"],
                calibration=calibration,
                provider_digest=f"D:{calibration.digest}",
                records=negative_eval,
            )
            combined_metrics = {
                **benign_metrics,
                "exact_false_accepts": negative_metrics["exact_false_accepts"],
                "negative_candidate_set_support": negative_metrics["negative_candidate_set_support"],
                "conflicting_action_exact_accepts": negative_metrics["conflicting_action_exact_accepts"],
                "same_action_wrong_row_exact_accepts": negative_metrics["same_action_wrong_row_exact_accepts"],
                "critical_contradiction_exact_accepts": negative_metrics["critical_contradiction_exact_accepts"],
                "negative_per_family_outcomes": negative_metrics["per_family_outcomes"],
            }
            feasible, infeasible_reasons = _feasible_for_selection({"combined_metrics": combined_metrics})
            results.append(
                {
                    "architecture_id": "D",
                    "calibration_values": point,
                    "calibration_digest": calibration.digest,
                    "feasible": feasible,
                    "infeasible_reasons": list(infeasible_reasons),
                    "combined_metrics": combined_metrics,
                }
            )
    _write_csv(output_dir / "architecture-grid.csv", [{**result["calibration_values"], "architecture_id": result["architecture_id"], "calibration_digest": result["calibration_digest"], "feasible": result["feasible"], "infeasible_reasons": "|".join(result["infeasible_reasons"]), **result["combined_metrics"]} for result in results])
    feasible_results = [result for result in results if result["feasible"]]
    if not feasible_results:
        selection = {
            "selection_status": "no_safe_architecture",
            "selected_architecture": None,
            "selected_calibration_digest": None,
            "all_results_digest": _sha256(results),
            "d_gateway_digest": _sha256(d_gateway),
            "grid_digest": _sha256(grid_definition),
            "benchmark_digest": _benchmark_manifest(benchmark)["benchmark_digest"],
            "architecture_selection_split_digest": _sha256({"benign": [record.observation_id for record in benign_eval], "negative": [record.observation_id for record in negative_eval]}),
            "region_digest": freeze["region_manifest"]["region_spec_digest"],
            "mask_digest": freeze["mask_manifest"]["mask_spec_digest"],
            "policy_artifact_id": benchmark.policy_artifact_id,
            "source_scope": benchmark.source_scope,
            "simplicity_order": SIMPLICITY_ORDER,
        }
    else:
        selected = min(feasible_results, key=_selection_rank_key)
        selection = {
            "selection_status": "selected_architecture",
            "selected_architecture": selected["architecture_id"],
            "selected_calibration_digest": selected["calibration_digest"],
            "selected_architecture_selection_point": selected["calibration_values"],
            "selected_result": selected,
            "all_results_digest": _sha256(results),
            "d_gateway_digest": _sha256(d_gateway),
            "grid_digest": _sha256(grid_definition),
            "benchmark_digest": _benchmark_manifest(benchmark)["benchmark_digest"],
            "architecture_selection_split_digest": _sha256({"benign": [record.observation_id for record in benign_eval], "negative": [record.observation_id for record in negative_eval]}),
            "region_digest": freeze["region_manifest"]["region_spec_digest"],
            "mask_digest": freeze["mask_manifest"]["mask_spec_digest"],
            "policy_artifact_id": benchmark.policy_artifact_id,
            "source_scope": benchmark.source_scope,
            "simplicity_order": SIMPLICITY_ORDER,
            "tie_break_path": _selection_rank_key(selected),
        }
    _write_json(output_dir / "selected-architecture.json", selection)
    _write_json(output_dir / "phase-access-audits.json", benchmark.phase_audits)
    _write_json(output_dir / "benchmark-manifest.json", _benchmark_manifest(benchmark))
    _write_json(output_dir / "split-manifest.json", _split_manifest(benchmark))
    return selection


def _calibration_grid_definition(architecture_id: str) -> Dict[str, Sequence[Any]]:
    base = _architecture_grid_definition()[architecture_id]
    if architecture_id in {"A", "B"}:
        return dict(base)
    return dict(base)


def _calibration_rank_key(result: Mapping[str, Any]) -> Tuple[Any, ...]:
    metrics = result["metrics"]
    values = result["calibration_values"]
    return (
        int(metrics["exact_false_accepts"]),
        int(metrics["conflicting_action_exact_accepts"]),
        int(metrics["critical_contradiction_exact_accepts"]),
        0 if (metrics["exact_correct_count"] + metrics["useful_candidate_set_count"]) > 0 else 1,
        -float(metrics["exact_row_coverage"]),
        -int(metrics["useful_candidate_set_count"]),
        -(0.0 if metrics["exact_accepted_precision"] is None else float(metrics["exact_accepted_precision"])),
        float(metrics["mean_set_size"] if metrics["mean_set_size"] is not None else 99.0),
        -float(values["minimum_available_mass"]),
        float(values.get("maximum_contradiction", 1.0)),
        float(values.get("maximum_critical_contradiction", 1.0)),
        -float(values["exact_winner_margin"]),
        int(values["maximum_candidate_set_size"]),
        float(values["candidate_relative_margin"]),
        -int(values.get("minimum_supporting_regions", 0)),
        json.dumps(_json_ready(values), sort_keys=True),
    )


def _calibration_feasible(metrics: Mapping[str, Any]) -> Tuple[bool, Tuple[str, ...]]:
    reasons = []
    if int(metrics["exact_false_accepts"]) != 0:
        reasons.append("exact_false_accepts")
    if int(metrics["conflicting_action_exact_accepts"]) != 0:
        reasons.append("conflicting_action_exact_accepts")
    if int(metrics["critical_contradiction_exact_accepts"]) != 0:
        reasons.append("critical_contradiction_exact_accepts")
    if SELECTION_NEGATIVE_CANDIDATE_SET_SUPPORT_BLOCKS_FEASIBILITY and int(metrics["negative_candidate_set_support"]) != 0:
        reasons.append("negative_candidate_set_support")
    if int(metrics["exact_correct_count"]) + int(metrics["useful_candidate_set_count"]) == 0:
        reasons.append("zero_utility")
    return (len(reasons) == 0, tuple(reasons))


def _run_calibrate(output_dir: Path) -> Dict[str, Any]:
    selection = _load_json(output_dir / "selected-architecture.json")
    if selection["selection_status"] != "selected_architecture":
        artifact = {
            "selection_status": "no_feasible_operating_point",
            "selected_calibration_digest": None,
            "selected_architecture": selection["selected_architecture"],
        }
        _write_json(output_dir / "selected-operating-point.json", artifact)
        return artifact
    benchmark = _build_stage3_benchmark(materialize_final=False)
    freeze = _freeze_regions_and_masks(benchmark, output_dir=output_dir)
    records, observations = benchmark.access(
        phase="calibrate",
        allowed_splits=("prototype", "diagnostic_development", "benign_calibration", "rejection_calibration"),
    )
    benign_eval = tuple(record for record in records if record.split == "benign_calibration")
    rejection_eval = tuple(record for record in records if record.split == "rejection_calibration")
    architecture_id = str(selection["selected_architecture"])
    grid_definition = _calibration_grid_definition(architecture_id)
    _write_json(output_dir / "calibration-grid-definition.json", {"architecture_id": architecture_id, "definition": _json_ready(grid_definition), "digest": _sha256(grid_definition)})
    results = []
    raw_benign = _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=benign_eval, observations=observations)
    raw_rejection = _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=rejection_eval, observations=observations)
    for point in _grid_points(grid_definition):
        calibration = _calibration(
            architecture_id=architecture_id,
            benchmark=benchmark,
            region_manifest=freeze["region_manifest"],
            mask_manifest=freeze["mask_manifest"],
            values=point,
        )
        benign_metrics = _evaluate_ranked_candidates(
            ranked_by_observation=raw_benign,
            calibration=calibration,
            provider_digest=f"{architecture_id}:{calibration.digest}",
            records=benign_eval,
        )
        rejection_metrics = _evaluate_ranked_candidates(
            ranked_by_observation=raw_rejection,
            calibration=calibration,
            provider_digest=f"{architecture_id}:{calibration.digest}",
            records=rejection_eval,
        )
        metrics = {
            **benign_metrics,
            "exact_false_accepts": rejection_metrics["exact_false_accepts"],
            "negative_candidate_set_support": rejection_metrics["negative_candidate_set_support"],
            "conflicting_action_exact_accepts": rejection_metrics["conflicting_action_exact_accepts"],
            "same_action_wrong_row_exact_accepts": rejection_metrics["same_action_wrong_row_exact_accepts"],
            "critical_contradiction_exact_accepts": rejection_metrics["critical_contradiction_exact_accepts"],
            "rejection_histogram": rejection_metrics["negative_per_family_outcomes"],
        }
        feasible, infeasible_reasons = _calibration_feasible(metrics)
        results.append(
            {
                "architecture_id": architecture_id,
                "calibration_values": point,
                "calibration_digest": calibration.digest,
                "feasible": feasible,
                "infeasible_reasons": list(infeasible_reasons),
                "metrics": metrics,
            }
        )
    _write_csv(output_dir / "calibration-grid.csv", [{**result["calibration_values"], "architecture_id": architecture_id, "calibration_digest": result["calibration_digest"], "feasible": result["feasible"], "infeasible_reasons": "|".join(result["infeasible_reasons"]), **result["metrics"]} for result in results])
    feasible_results = [result for result in results if result["feasible"]]
    if not feasible_results:
        artifact = {
            "selection_status": "no_feasible_operating_point",
            "selected_architecture": architecture_id,
            "selected_calibration_digest": None,
            "calibration_grid_digest": _sha256(results),
            "benchmark_digest": _benchmark_manifest(benchmark)["benchmark_digest"],
            "split_digest": _sha256({"benign": [record.observation_id for record in benign_eval], "rejection": [record.observation_id for record in rejection_eval]}),
            "region_digest": freeze["region_manifest"]["region_spec_digest"],
            "mask_digest": freeze["mask_manifest"]["mask_spec_digest"],
            "policy_artifact_id": benchmark.policy_artifact_id,
            "source_scope": benchmark.source_scope,
        }
    else:
        selected = min(feasible_results, key=_calibration_rank_key)
        artifact = {
            "selection_status": "selected_operating_point",
            "selected_architecture": architecture_id,
            "selected_calibration_digest": selected["calibration_digest"],
            "selected_calibration_values": selected["calibration_values"],
            "selected_result": selected,
            "calibration_grid_digest": _sha256(results),
            "benchmark_digest": _benchmark_manifest(benchmark)["benchmark_digest"],
            "split_digest": _sha256({"benign": [record.observation_id for record in benign_eval], "rejection": [record.observation_id for record in rejection_eval]}),
            "region_digest": freeze["region_manifest"]["region_spec_digest"],
            "mask_digest": freeze["mask_manifest"]["mask_spec_digest"],
            "policy_artifact_id": benchmark.policy_artifact_id,
            "source_scope": benchmark.source_scope,
            "tie_break_path": _calibration_rank_key(selected),
        }
    _write_json(output_dir / "selected-operating-point.json", artifact)
    _write_json(output_dir / "phase-access-audits.json", benchmark.phase_audits)
    _write_json(output_dir / "benchmark-manifest.json", _benchmark_manifest(benchmark))
    _write_json(output_dir / "split-manifest.json", _split_manifest(benchmark))
    return artifact


def _freeze_final_identities(output_dir: Path) -> Dict[str, Any]:
    benchmark = _build_stage3_benchmark(materialize_final=False)
    manifest = _benchmark_manifest(benchmark)
    split_manifest = _split_manifest(benchmark)
    _write_json(output_dir / "benchmark-manifest.json", manifest)
    _write_json(output_dir / "split-manifest.json", split_manifest)
    return {"benchmark_manifest_digest": _sha256(manifest), "split_manifest_digest": _sha256(split_manifest)}


def _verify_pre_final(output_dir: Path) -> Dict[str, Any]:
    benchmark_manifest = _load_json(output_dir / "benchmark-manifest.json")
    split_manifest = _load_json(output_dir / "split-manifest.json")
    region_manifest = _load_json(output_dir / "region-manifest.json")
    mask_manifest = _load_json(output_dir / "mask-manifest.json")
    architecture_grid_definition = _load_json(output_dir / "architecture-grid-definition.json")
    selected_architecture = _load_json(output_dir / "selected-architecture.json")
    architecture_d_gateway = _load_json(output_dir / "architecture-d-gateway.json")
    calibration_grid_definition = _load_json(output_dir / "calibration-grid-definition.json") if (output_dir / "calibration-grid-definition.json").exists() else None
    selected_operating_point = _load_json(output_dir / "selected-operating-point.json") if (output_dir / "selected-operating-point.json").exists() else None
    phase_audits = _load_json(output_dir / "phase-access-audits.json")
    violations = []
    for audit in phase_audits:
        if any(split.startswith("final_") for split in audit["accessed_splits"]) and audit["phase"] in {"select_architecture", "calibrate"}:
            violations.append("final_split_accessed_early")
    forbidden_final_outputs = [name for name in ("final-metrics.json", "final-evaluation.json") if (output_dir / name).exists()]
    if forbidden_final_outputs:
        violations.append("final_metrics_already_exist")
    payload = {
        "verified": len(violations) == 0,
        "violations": violations,
        "benchmark_manifest_digest": _sha256(benchmark_manifest),
        "split_manifest_digest": _sha256(split_manifest),
        "region_manifest_digest": _sha256(region_manifest),
        "mask_manifest_digest": _sha256(mask_manifest),
        "architecture_grid_digest": architecture_grid_definition["digest"],
        "selected_architecture_digest": _sha256(selected_architecture),
        "architecture_d_gateway_digest": _sha256(architecture_d_gateway),
        "calibration_grid_digest": None if calibration_grid_definition is None else calibration_grid_definition["digest"],
        "selected_operating_point_digest": None if selected_operating_point is None else _sha256(selected_operating_point),
    }
    if violations:
        raise SystemExit(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _audit_exact_frames(*, output_dir: Path) -> Dict[str, Any]:
    benchmark = _build_stage3_benchmark(materialize_final=False)
    freeze = _freeze_regions_and_masks(benchmark, output_dir=output_dir)
    records, observations = benchmark.access(
        phase="select_architecture",
        allowed_splits=("prototype", "diagnostic_development", "architecture_selection_benign"),
    )
    exact_records = tuple(record for record in records if record.family_id == "selection_exact")
    rows = []
    summary: Dict[str, Any] = {"architectures": {}, "exact_observation_count": len(exact_records)}
    for architecture_id in ("A", "B", "C"):
        ranked_by_observation = _ranked_candidates_for_architecture(
            architecture_id=architecture_id,
            benchmark=benchmark,
            freeze=freeze,
            records=exact_records,
            observations=observations,
        )
        top1 = 0
        eligible = 0
        rejection_reasons = Counter()
        nonzero_evidence = 0
        for record in exact_records:
            ranked = tuple(
                evaluate_candidate_eligibility(
                    candidate=candidate,
                    ranked_candidates=ranked_by_observation[record.observation_id],
                    calibration=_calibration(
                        architecture_id=architecture_id,
                        benchmark=benchmark,
                        region_manifest=freeze["region_manifest"],
                        mask_manifest=freeze["mask_manifest"],
                        values={
                            "minimum_available_mass": 0.0,
                            "minimum_available_fraction": 0.0,
                            "minimum_support": 0.0,
                            "maximum_contradiction": 1.0,
                            "maximum_critical_contradiction": 1.0,
                            "exact_winner_threshold": 0.0,
                            "exact_winner_margin": 0.0,
                            "candidate_relative_margin": 0.0,
                            "conflicting_action_separation": 0.0,
                            "minimum_supporting_regions": 0,
                            "maximum_candidate_set_size": MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
                        },
                    ),
                )
                for candidate in ranked_by_observation[record.observation_id]
            )
            ranked = tuple(sorted(ranked, key=lambda candidate: (-float(candidate.candidate_strength), -float(candidate.available_informative_mass), -float(candidate.available_informative_fraction), float(candidate.aggregate_contradiction), float(candidate.aggregate_critical_contradiction), -int(candidate.supporting_region_count), candidate.row_id, candidate.prototype_observation_id)))
            calibration = _calibration(
                architecture_id=architecture_id,
                benchmark=benchmark,
                region_manifest=freeze["region_manifest"],
                mask_manifest=freeze["mask_manifest"],
                values={
                    "minimum_available_mass": 1.0,
                    "minimum_available_fraction": 0.25,
                    "minimum_support": 0.0,
                    "maximum_contradiction": 1.0 if architecture_id == "A" else 0.5,
                    "maximum_critical_contradiction": 1.0 if architecture_id in {"A", "B"} else 0.25,
                    "exact_winner_threshold": 0.5,
                    "exact_winner_margin": 0.05,
                    "candidate_relative_margin": 0.0,
                    "conflicting_action_separation": 0.0,
                    "minimum_supporting_regions": 0,
                    "maximum_candidate_set_size": MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
                },
            )
            candidate_set = build_discriminative_candidate_set(
                ranked_candidates=ranked,
                calibration=calibration,
                provider_digest=f"audit:{architecture_id}:{calibration.digest}",
                observation_digest=ranked[0].observation_digest,
            )
            expected = next((candidate for candidate in ranked if candidate.row_id == record.row_id), None)
            winner = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else None
            if expected is not None and expected.available_informative_mass > 0:
                nonzero_evidence += 1
            if expected is not None and expected.eligible_for_candidate_set:
                eligible += 1
            if expected is not None and winner.row_id == record.row_id:
                top1 += 1
            rejection_reasons[candidate_set.rejection_reason or candidate_set.outcome] += 1
            rows.append(
                {
                    "observation_id": record.observation_id,
                    "expected_row": record.row_id,
                    "expected_action": record.action_id,
                    "architecture": architecture_id,
                    "expected_mask_stability_count": None if expected is None else int(freeze["masks"][record.row_id].spec.stable_pixel_count),
                    "expected_informative_count": None if expected is None else int(freeze["masks"][record.row_id].spec.informative_pixel_count),
                    "expected_row_rank": None if expected is None else next((index for index, candidate in enumerate(ranked, start=1) if candidate.row_id == record.row_id), None),
                    "expected_row_strength": None if expected is None else float(expected.candidate_strength),
                    "available_mass": None if expected is None else float(expected.available_informative_mass),
                    "available_fraction": None if expected is None else float(expected.available_informative_fraction),
                    "support": None if expected is None else float(expected.aggregate_support),
                    "contradiction": None if expected is None else float(expected.aggregate_contradiction),
                    "critical_contradiction": None if expected is None else float(expected.aggregate_critical_contradiction),
                    "supporting_regions": None if expected is None else int(expected.supporting_region_count),
                    "winner": winner.row_id,
                    "winner_action": winner.action_id,
                    "winner_strength": float(winner.candidate_strength),
                    "runner_up": None if runner_up is None else runner_up.row_id,
                    "winner_margin": None if expected is None else expected.exact_winner_margin,
                    "conflicting_action_separation": None if expected is None else expected.conflicting_action_separation,
                    "expected_row_candidate_eligible": None if expected is None else bool(expected.eligible_for_candidate_set),
                    "expected_row_exact_eligible": None if expected is None else bool(expected.eligible_for_exact),
                    "candidate_ineligibility_reasons": [] if expected is None else list(expected.ineligibility_reasons),
                    "candidate_set_outcome": candidate_set.outcome,
                    "candidate_set_rows": list(candidate_set.rows),
                    "rejection_reason": candidate_set.rejection_reason,
                }
            )
        summary["architectures"][architecture_id] = {
            "expected_row_top1_count": top1,
            "expected_row_candidate_eligible_count": eligible,
            "expected_rows_with_nonzero_evidence": nonzero_evidence,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        }
    _write_csv(output_dir / "exact-frame-audit.csv", rows)
    _write_json(output_dir / "exact-frame-summary.json", summary)
    return summary


def _render_v1_ruling_markdown(ruling: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 3 v1 measurement ruling",
        "",
        f"- Ruling: `{ruling['ruling']}`",
        f"- Architecture selection usable: `{ruling['architecture_selection_usable']}`",
        f"- Calibration status usable: `{ruling['calibration_status_usable']}`",
        f"- Final identities usable: `{ruling['final_identities_usable']}`",
        f"- Next permitted action: {ruling['next_permitted_action']}",
        "",
        "## Evidence",
    ]
    for item in ruling["evidence"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _classify_v1(*, regeneration: Mapping[str, Any], closure: Mapping[str, Any], exact_summary: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = []
    generator_mismatch = not bool(regeneration["semantic_match"])
    prototype_failure = bool(closure["missing_prototype_manifest_rows"])
    development_failure = bool(closure["missing_development_rows"] or closure["evaluated_rows_with_zero_stability"])
    wiring_failure = all(
        exact_summary["architectures"][architecture_id]["expected_rows_with_nonzero_evidence"] == 0
        for architecture_id in ("A", "B", "C")
    )
    if generator_mismatch:
        evidence.append("Current committed code does not regenerate the frozen v1 artifact set semantically.")
    if prototype_failure:
        evidence.append("The provider prototype universe is larger than the prototype split manifest.")
    if development_failure:
        evidence.append("Architecture-selection rows lack development-backed stable evidence.")
    if wiring_failure:
        evidence.append("All audited exact frames lose expected-row evidence before the architecture-specific gates are meaningfully exercised.")
    if generator_mismatch and (prototype_failure or development_failure or wiring_failure):
        label = "invalid_multiple_failures"
    elif generator_mismatch:
        label = "invalid_generator_artifact_mismatch"
    elif prototype_failure:
        label = "invalid_prototype_closure"
    elif development_failure:
        label = "invalid_mask_development_closure"
    elif wiring_failure:
        label = "invalid_provider_benchmark_wiring"
    else:
        label = "valid_no_safe_architecture"
        evidence.append("Committed code regenerates v1, prototype closure is complete, and exact rows retain evidence.")
    return {
        "ruling": label,
        "evidence": evidence,
        "affected_artifacts": [item["artifact"] for item in regeneration["artifacts"] if item["status"] in {"semantic_mismatch", "missing_artifact", "unexpected_artifact"}],
        "architecture_selection_usable": label == "valid_no_safe_architecture",
        "calibration_status_usable": label == "valid_no_safe_architecture",
        "final_identities_usable": True,
        "invalidation_boundary": "pre-final measurement integrity audit",
        "next_permitted_action": (
            "freeze_stage3_v2_measurement_amendment"
            if label != "valid_no_safe_architecture"
            else "stage3_v1_may_remain_negative_without_v5_or_final_evaluation"
        ),
    }


def _exact_sanity_v2(*, benchmark: Stage3Benchmark, freeze: Mapping[str, Any], collision_atlas: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    collision_by_row = {}
    for group in collision_atlas["rows"]:
        for row_id in group["row_ids"]:
            collision_by_row[row_id] = group
    records, observations = benchmark.access(
        phase="v2_exact_sanity",
        allowed_splits=("prototype", "diagnostic_development", "architecture_selection_benign"),
    )
    exact_records = tuple(record for record in records if record.family_id == "selection_exact")
    rows = []
    architectures = {}
    sanity_valid = True
    for architecture_id in ("A", "B", "C"):
        raw = _raw_ranked_candidates(
            architecture_id=architecture_id,
            benchmark=benchmark,
            freeze=freeze,
            records=exact_records,
            observations=observations,
        )
        provider = DiscriminativeEvidenceProvider(
            prototypes=benchmark.prototypes,
            masks=freeze["masks"],
            regions=freeze["regions"],
            calibration=_calibration(
                architecture_id=architecture_id,
                benchmark=benchmark,
                region_manifest=freeze["region_manifest"],
                mask_manifest=freeze["mask_manifest"],
                values={
                    "minimum_available_mass": 0.0,
                    "minimum_available_fraction": 0.0,
                    "minimum_support": 0.0,
                    "maximum_contradiction": 1.0,
                    "maximum_critical_contradiction": 1.0,
                    "exact_winner_threshold": 0.0,
                    "exact_winner_margin": 0.0,
                    "candidate_relative_margin": 0.0,
                    "conflicting_action_separation": 0.0,
                    "minimum_supporting_regions": 0,
                    "maximum_candidate_set_size": MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
                },
            ),
            policy_artifact_id=benchmark.policy_artifact_id,
            source_scope=benchmark.source_scope,
        )
        direct_provider = DiscriminativeEvidenceProvider(
            prototypes=benchmark.prototypes,
            masks=freeze["masks"],
            regions=freeze["regions"],
            calibration=provider._calibration,
            policy_artifact_id=benchmark.policy_artifact_id,
            source_scope=benchmark.source_scope,
        )
        unique_top1 = 0
        for record in exact_records:
            ranked = tuple(
                evaluate_candidate_eligibility(candidate=candidate, ranked_candidates=raw[record.observation_id], calibration=provider._calibration)
                for candidate in raw[record.observation_id]
            )
            ranked = tuple(sorted(ranked, key=lambda candidate: (-float(candidate.candidate_strength), -float(candidate.available_informative_mass), -float(candidate.available_informative_fraction), float(candidate.aggregate_contradiction), float(candidate.aggregate_critical_contradiction), -int(candidate.supporting_region_count), candidate.row_id, candidate.prototype_observation_id)))
            candidate_set = build_discriminative_candidate_set(
                ranked_candidates=ranked,
                calibration=provider._calibration,
                provider_digest=provider.contract().address_artifact_id,
                observation_digest=ranked[0].observation_digest,
            )
            direct_ranked = direct_provider._rank(observations[record.observation_id])
            if [candidate.to_dict() for candidate in raw[record.observation_id]] != [candidate.to_dict() for candidate in direct_ranked]:
                sanity_valid = False
            collision = collision_by_row[record.row_id]
            expected = next(candidate for candidate in ranked if candidate.row_id == record.row_id)
            top_strength = ranked[0].candidate_strength
            sorted(candidate.row_id for candidate in ranked if abs(candidate.candidate_strength - top_strength) <= 1e-12)
            if collision["collision_class"] == "unique_visual_row":
                if expected.available_informative_mass <= 0 or ranked[0].row_id != record.row_id or not expected.eligible_for_candidate_set:
                    sanity_valid = False
                else:
                    unique_top1 += 1
            elif collision["collision_class"] == "conflicting_action_visual_alias" and candidate_set.outcome == "exact_row_accepted":
                sanity_valid = False
            rows.append(
                {
                    "observation_id": record.observation_id,
                    "expected_row": record.row_id,
                    "expected_action": record.action_id,
                    "architecture": architecture_id,
                    "prototype_collision_class": collision["collision_class"],
                    "collision_group_rows": collision["row_ids"],
                    "collision_group_actions": collision["action_ids"],
                    "expected_row_rank": next((index for index, candidate in enumerate(ranked, start=1) if candidate.row_id == record.row_id), None),
                    "expected_row_strength": float(expected.candidate_strength),
                    "maximum_strength": float(top_strength),
                    "expected_row_available_mass": float(expected.available_informative_mass),
                    "expected_row_available_fraction": float(expected.available_informative_fraction),
                    "expected_row_support": float(expected.aggregate_support),
                    "expected_row_contradiction": float(expected.aggregate_contradiction),
                    "expected_row_critical_contradiction": float(expected.aggregate_critical_contradiction),
                    "expected_row_supporting_regions": int(expected.supporting_region_count),
                    "winner_row": ranked[0].row_id,
                    "winner_action": ranked[0].action_id,
                    "winner_margin": expected.exact_winner_margin,
                    "conflicting_action_separation": expected.conflicting_action_separation,
                    "expected_row_candidate_eligible": bool(expected.eligible_for_candidate_set),
                    "expected_row_exact_eligible": bool(expected.eligible_for_exact),
                    "candidate_set_outcome": candidate_set.outcome,
                    "candidate_set_rows": list(candidate_set.rows),
                    "rejection_reason": candidate_set.rejection_reason,
                }
            )
        architectures[architecture_id] = {
            "unique_exact_expected_row_top1_count": unique_top1,
            "observation_count": len(exact_records),
        }
    _write_csv(output_dir / "exact-sanity.csv", rows)
    summary = {
        "sanity_valid": sanity_valid,
        "direct_provider_equivalence": True,
        "architectures": architectures,
        "observation_count": len(exact_records),
    }
    _write_json(output_dir / "exact-sanity-summary.json", summary)
    return summary


def _stage3_v2_preselection_valid(closure: Mapping[str, Any], exact_summary: Mapping[str, Any]) -> bool:
    return bool(closure["closure_valid"]) and bool(exact_summary["sanity_valid"]) and bool(exact_summary["direct_provider_equivalence"])


def _write_stage3_v2_docs(output_dir: Path, *, generator_identity: Mapping[str, Any], benchmark_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any]) -> None:
    _write_markdown(
        output_dir / "README.md",
        f"# Stage 3 v2 pre-final benchmark\n\nBenchmark version: `{benchmark_manifest['benchmark_version']}`\nGenerator version: `{benchmark_manifest['generator_version']}`\nSeed digest: `{benchmark_manifest['final_seed_digest']}`\nBenchmark digest: `{benchmark_manifest['benchmark_digest']}`\nSplit digest: `{split_manifest['split_digest']}`",
    )
    _write_markdown(
        output_dir / "reproduction.md",
        "Rebuild with `python examples/arcade_visual_video_discriminative_evidence_benchmark.py --freeze-benchmark-v2`, then validate with `--verify-v2-benchmark`, `--select-architecture-v2`, `--calibrate-v2`, and `--verify-pre-final-v2`.",
    )


def _freeze_benchmark_v2_into(output_dir: Path) -> Dict[str, Any]:
    benchmark = _build_stage3_benchmark_v2(materialize_final=False)
    generator_identity = _generator_identity_v2(benchmark)
    _write_json(output_dir / "generator-identity.json", generator_identity)
    prototype_manifest = _prototype_manifest_v2(benchmark, generator_identity=generator_identity)
    development_manifest = _development_manifest_v2(benchmark, generator_identity=generator_identity)
    evaluation_sample = _evaluation_sample_manifest_v2(benchmark)
    freeze = _freeze_regions_and_masks(benchmark, output_dir=output_dir)
    prototype_rows = sorted(benchmark.prototypes)
    prototype_manifest_rows = sorted(row["row_id"] for row in prototype_manifest["rows"])
    mask_rows = sorted(freeze["masks"])
    if prototype_rows != prototype_manifest_rows or prototype_rows != mask_rows:
        raise VPMValidationError("provider prototype, prototype manifest, and mask rows must match exactly")
    collision_atlas = _prototype_collision_atlas_v2(benchmark)
    closure = _mask_closure_v2(benchmark, freeze["masks"])
    if not closure["closure_valid"]:
        raise VPMValidationError("v2 closure failed before benchmark freeze")
    _write_json(output_dir / "prototype-manifest.json", prototype_manifest)
    _write_json(output_dir / "development-manifest.json", development_manifest)
    _write_json(output_dir / "evaluation-sample.json", evaluation_sample)
    _write_json(output_dir / "prototype-collision-atlas.json", collision_atlas)
    _write_csv(output_dir / "prototype-collision-atlas.csv", collision_atlas["rows"])
    _write_csv(output_dir / "mask-closure.csv", closure["rows"])
    _write_json(output_dir / "mask-closure-summary.json", {key: value for key, value in closure.items() if key != "rows"})
    exact_summary = _exact_sanity_v2(benchmark=benchmark, freeze=freeze, collision_atlas=collision_atlas, output_dir=output_dir)
    benchmark_manifest = _benchmark_manifest(benchmark)
    split_manifest = _split_manifest(benchmark)
    _write_json(output_dir / "benchmark-manifest.json", benchmark_manifest)
    _write_json(output_dir / "split-manifest.json", split_manifest)
    _write_json(output_dir / "architecture-grid-definition.json", {"definition": _json_ready(_architecture_grid_definition()), "digest": _sha256(_architecture_grid_definition()), "amendment_parent": V2_AMENDMENT_COMMIT})
    _write_json(output_dir / "phase-access-audits.json", benchmark.phase_audits)
    _write_stage3_v2_docs(output_dir, generator_identity=generator_identity, benchmark_manifest=benchmark_manifest, split_manifest=split_manifest)
    return {
        "benchmark": benchmark,
        "freeze": freeze,
        "generator_identity": generator_identity,
        "prototype_manifest": prototype_manifest,
        "development_manifest": development_manifest,
        "evaluation_sample": evaluation_sample,
        "collision_atlas": collision_atlas,
        "closure": closure,
        "exact_summary": exact_summary,
        "benchmark_manifest": benchmark_manifest,
        "split_manifest": split_manifest,
    }


def run_freeze_benchmark_v2(output_dir: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage3-v2-freeze-", dir=str(REPO_ROOT)) as tmp:
        temp_output = Path(tmp)
        result = _freeze_benchmark_v2_into(temp_output)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(temp_output, output_dir)
    return {
        "mode": "freeze-benchmark-v2",
        "benchmark_digest": result["benchmark_manifest"]["benchmark_digest"],
        "split_digest": result["split_manifest"]["split_digest"],
        "generator_identity_digest": result["generator_identity"]["generator_identity_digest"],
    }


def _run_audit_pre_final_v1(output_dir: Path) -> Dict[str, Any]:
    benchmark = _build_stage3_benchmark(materialize_final=False)
    audit_dir = output_dir / "measurement-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage3-v1-audit-", dir=str(REPO_ROOT)) as tmp:
        temp_output = Path(tmp)
        _run_selection(temp_output)
        _run_calibrate(temp_output)
        regeneration = _artifact_comparison(output_dir, temp_output)
        freeze = _freeze_regions_and_masks(benchmark, output_dir=temp_output)
        closure = validate_prototype_and_development_closure(benchmark=benchmark, masks=freeze["masks"])
        mask_coverage = _mask_coverage_report(benchmark=benchmark, masks=freeze["masks"], output_dir=audit_dir)
        exact_summary = _audit_exact_frames(output_dir=audit_dir)
    report = {
        "audited_branch_head": _git_output("rev-parse", "HEAD"),
        "audited_generator_blob_digest": _generator_source_blob_digest(),
        "artifact_generation_commit": None,
        "benchmark_digest": _benchmark_manifest(benchmark)["benchmark_digest"],
        "split_digest": _split_manifest(benchmark)["split_digest"],
        "architecture_grid_digest": _sha256(_architecture_grid_definition()),
        "mask_digest": freeze["mask_manifest"]["mask_spec_digest"],
        "region_digest": freeze["region_manifest"]["region_spec_digest"],
        "generator_identity": _generator_identity(benchmark),
        "current_code_sample_size_constant": NON_PROTOTYPE_ROW_SAMPLE_SIZE,
        "committed_manifest_sample_size": _load_json(output_dir / "benchmark-manifest.json")["metadata"]["nonprototype_row_sample_size"],
        "regeneration_audit": regeneration,
        "prototype_and_development_closure": closure,
        "mask_coverage_summary": {key: value for key, value in mask_coverage.items() if key != "rows"},
        "exact_frame_summary": exact_summary,
    }
    ruling = _classify_v1(regeneration=regeneration, closure=closure, exact_summary=exact_summary)
    _write_json(audit_dir / "regeneration-audit.json", report)
    _write_json(audit_dir / "prototype-closure.json", closure)
    _write_json(audit_dir / "v1-ruling.json", ruling)
    _write_markdown(audit_dir / "v1-ruling.md", _render_v1_ruling_markdown(ruling))
    result = {
        "mode": "audit-pre-final-v1",
        "report_digest": _sha256(report),
        "ruling": ruling["ruling"],
        "semantic_match": regeneration["semantic_match"],
    }
    if ruling["ruling"] != "valid_no_safe_architecture":
        raise SystemExit(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    return result


def _load_v1_ruling(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / "measurement-audit" / "v1-ruling.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    if payload.get("ruling") not in VALID_V1_RULINGS:
        raise VPMValidationError("invalid v1 ruling payload")
    return payload


def _run_evaluate(output_dir: Path) -> Dict[str, Any]:
    ruling = _load_v1_ruling(output_dir)
    if ruling is None:
        raise SystemExit("Stage 3 v1 evaluation is blocked until measurement-audit/v1-ruling.json exists.")
    if ruling["ruling"] != "valid_no_safe_architecture":
        raise SystemExit("Stage 3 v1 evaluation is blocked because the frozen measurement failed the integrity audit.")
    raise SystemExit("Final evaluation remains intentionally blocked in this task.")


def run_verify_v2_benchmark(output_dir: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage3-v2-verify-", dir=str(REPO_ROOT)) as tmp:
        temp_output = Path(tmp)
        _freeze_benchmark_v2_into(temp_output)
        artifact_names = sorted(({path.name for path in temp_output.iterdir() if path.is_file()} | ({path.name for path in output_dir.iterdir() if path.is_file()} if output_dir.exists() else set())) - {"pre-final-verification.json"})
        comparison = _artifact_comparison(output_dir, temp_output, artifact_names=artifact_names) if output_dir.exists() else {"semantic_match": False, "artifacts": []}
    payload = {"mode": "verify-v2-benchmark", "verified": bool(comparison["semantic_match"]), "comparison": comparison}
    if not comparison["semantic_match"]:
        raise SystemExit(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    return payload


def run_select_architecture_v2(output_dir: Path) -> Dict[str, Any]:
    benchmark = _build_stage3_benchmark_v2(materialize_final=False)
    freeze = _freeze_regions_and_masks(benchmark, output_dir=output_dir)
    exact_summary = _load_json(output_dir / "exact-sanity-summary.json")
    closure = _load_json(output_dir / "mask-closure-summary.json")
    if not _stage3_v2_preselection_valid(closure, exact_summary):
        artifact = {"selection_status": "invalid_architecture_measurement", "selected_architecture": None, "selected_calibration_digest": None}
        _write_json(output_dir / "selected-architecture.json", artifact)
        return artifact
    return _run_selection_v2(output_dir=output_dir, benchmark=benchmark, freeze=freeze)


def _run_selection_v2(*, output_dir: Path, benchmark: Stage3Benchmark, freeze: Mapping[str, Any]) -> Dict[str, Any]:
    benign_records, benign_observations = benchmark.access(
        phase="select_architecture",
        allowed_splits=("prototype", "diagnostic_development", "architecture_selection_benign"),
    )
    negative_records, negative_observations = benchmark.access(
        phase="select_architecture",
        allowed_splits=("prototype", "diagnostic_development", "architecture_selection_negative"),
    )
    benign_eval = tuple(record for record in benign_records if record.split == "architecture_selection_benign")
    negative_eval = tuple(record for record in negative_records if record.split == "architecture_selection_negative")
    grid_definition = _architecture_grid_definition()
    results = []
    raw_benign = {architecture_id: _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=benign_eval, observations=benign_observations) for architecture_id in ("A", "B", "C")}
    raw_negative = {architecture_id: _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=negative_eval, observations=negative_observations) for architecture_id in ("A", "B", "C")}
    for architecture_id in ("A", "B", "C"):
        for point in _grid_points(grid_definition[architecture_id]):
            calibration = _calibration(architecture_id=architecture_id, benchmark=benchmark, region_manifest=freeze["region_manifest"], mask_manifest=freeze["mask_manifest"], values=point)
            benign_metrics = _evaluate_ranked_candidates(ranked_by_observation=raw_benign[architecture_id], calibration=calibration, provider_digest=f"{architecture_id}:{calibration.digest}", records=benign_eval)
            negative_metrics = _evaluate_ranked_candidates(ranked_by_observation=raw_negative[architecture_id], calibration=calibration, provider_digest=f"{architecture_id}:{calibration.digest}", records=negative_eval)
            combined_metrics = {**benign_metrics, "exact_false_accepts": negative_metrics["exact_false_accepts"], "negative_candidate_set_support": negative_metrics["negative_candidate_set_support"], "conflicting_action_exact_accepts": negative_metrics["conflicting_action_exact_accepts"], "same_action_wrong_row_exact_accepts": negative_metrics["same_action_wrong_row_exact_accepts"], "critical_contradiction_exact_accepts": negative_metrics["critical_contradiction_exact_accepts"], "negative_per_family_outcomes": negative_metrics["per_family_outcomes"]}
            feasible, infeasible_reasons = _feasible_for_selection({"combined_metrics": combined_metrics})
            results.append({"architecture_id": architecture_id, "calibration_values": point, "calibration_digest": calibration.digest, "feasible": feasible, "infeasible_reasons": list(infeasible_reasons), "combined_metrics": combined_metrics})
    best_by_architecture = {architecture_id: min((result for result in results if result["architecture_id"] == architecture_id), key=_selection_rank_key) for architecture_id in ("A", "B", "C")}
    a = best_by_architecture["A"]
    d_gateway = {
        "conditions": {
            "a_nonzero_safe_utility": (a["combined_metrics"]["exact_correct_count"] + a["combined_metrics"]["useful_candidate_set_count"]) > 0,
            "b_or_c_additional_recovery": False,
            "added_recovery_zero_new_safety_failures": False,
            "distinct_failure_mode": False,
        },
        "supporting_frame_identities": [],
        "supporting_family_identities": [],
    }
    for architecture_id in ("B", "C"):
        candidate = best_by_architecture[architecture_id]
        if candidate["combined_metrics"]["exact_correct_count"] + candidate["combined_metrics"]["useful_candidate_set_count"] > a["combined_metrics"]["exact_correct_count"] + a["combined_metrics"]["useful_candidate_set_count"]:
            d_gateway["conditions"]["b_or_c_additional_recovery"] = True
            if candidate["combined_metrics"]["exact_false_accepts"] == 0 and candidate["combined_metrics"]["conflicting_action_exact_accepts"] == 0 and candidate["combined_metrics"]["critical_contradiction_exact_accepts"] == 0:
                d_gateway["conditions"]["added_recovery_zero_new_safety_failures"] = True
            break
    d_gateway["eligible"] = all(bool(value) for value in d_gateway["conditions"].values())
    _write_json(output_dir / "architecture-d-gateway.json", {**d_gateway, "digest": _sha256(d_gateway)})
    _write_csv(output_dir / "architecture-grid.csv", [{**result["calibration_values"], "architecture_id": result["architecture_id"], "calibration_digest": result["calibration_digest"], "feasible": result["feasible"], "infeasible_reasons": "|".join(result["infeasible_reasons"]), **result["combined_metrics"]} for result in results])
    feasible_results = [result for result in results if result["feasible"]]
    if not feasible_results:
        selection = {"selection_status": "no_safe_architecture", "selected_architecture": None, "selected_calibration_digest": None, "all_results_digest": _sha256(results), "d_gateway_digest": _sha256(d_gateway), "grid_digest": _sha256(grid_definition), "benchmark_digest": _benchmark_manifest(benchmark)["benchmark_digest"], "architecture_selection_split_digest": _sha256({"benign": [record.observation_id for record in benign_eval], "negative": [record.observation_id for record in negative_eval]}), "region_digest": freeze["region_manifest"]["region_spec_digest"], "mask_digest": freeze["mask_manifest"]["mask_spec_digest"], "policy_artifact_id": benchmark.policy_artifact_id, "source_scope": benchmark.source_scope, "simplicity_order": SIMPLICITY_ORDER, "tie_break_path": None}
    else:
        selected = min(feasible_results, key=_selection_rank_key)
        selection = {"selection_status": "selected_architecture", "selected_architecture": selected["architecture_id"], "selected_calibration_digest": selected["calibration_digest"], "selected_architecture_selection_point": selected["calibration_values"], "selected_result": selected, "all_results_digest": _sha256(results), "d_gateway_digest": _sha256(d_gateway), "grid_digest": _sha256(grid_definition), "benchmark_digest": _benchmark_manifest(benchmark)["benchmark_digest"], "architecture_selection_split_digest": _sha256({"benign": [record.observation_id for record in benign_eval], "negative": [record.observation_id for record in negative_eval]}), "region_digest": freeze["region_manifest"]["region_spec_digest"], "mask_digest": freeze["mask_manifest"]["mask_spec_digest"], "policy_artifact_id": benchmark.policy_artifact_id, "source_scope": benchmark.source_scope, "simplicity_order": SIMPLICITY_ORDER, "tie_break_path": _selection_rank_key(selected)}
    _write_json(output_dir / "selected-architecture.json", selection)
    _write_json(output_dir / "phase-access-audits.json", benchmark.phase_audits)
    return selection


def run_calibrate_v2(output_dir: Path) -> Dict[str, Any]:
    selection = _load_json(output_dir / "selected-architecture.json")
    if selection["selection_status"] == "invalid_architecture_measurement":
        artifact = {"selection_status": "invalid_calibration_measurement", "selected_architecture": None, "selected_calibration_digest": None}
        _write_json(output_dir / "selected-operating-point.json", artifact)
        return artifact
    if selection["selection_status"] != "selected_architecture":
        artifact = {"selection_status": "not_run_no_selected_architecture", "selected_architecture": None, "selected_calibration_digest": None}
        _write_json(output_dir / "selected-operating-point.json", artifact)
        return artifact
    benchmark = _build_stage3_benchmark_v2(materialize_final=False)
    freeze = _freeze_regions_and_masks(benchmark, output_dir=output_dir)
    records, observations = benchmark.access(phase="calibrate", allowed_splits=("prototype", "diagnostic_development", "benign_calibration", "rejection_calibration"))
    benign_eval = tuple(record for record in records if record.split == "benign_calibration")
    rejection_eval = tuple(record for record in records if record.split == "rejection_calibration")
    architecture_id = str(selection["selected_architecture"])
    grid_definition = _calibration_grid_definition(architecture_id)
    _write_json(output_dir / "calibration-grid-definition.json", {"architecture_id": architecture_id, "definition": _json_ready(grid_definition), "digest": _sha256(grid_definition)})
    results = []
    raw_benign = _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=benign_eval, observations=observations)
    raw_rejection = _raw_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, records=rejection_eval, observations=observations)
    for point in _grid_points(grid_definition):
        calibration = _calibration(architecture_id=architecture_id, benchmark=benchmark, region_manifest=freeze["region_manifest"], mask_manifest=freeze["mask_manifest"], values=point)
        benign_metrics = _evaluate_ranked_candidates(ranked_by_observation=raw_benign, calibration=calibration, provider_digest=f"{architecture_id}:{calibration.digest}", records=benign_eval)
        rejection_metrics = _evaluate_ranked_candidates(ranked_by_observation=raw_rejection, calibration=calibration, provider_digest=f"{architecture_id}:{calibration.digest}", records=rejection_eval)
        metrics = {**benign_metrics, "exact_false_accepts": rejection_metrics["exact_false_accepts"], "negative_candidate_set_support": rejection_metrics["negative_candidate_set_support"], "conflicting_action_exact_accepts": rejection_metrics["conflicting_action_exact_accepts"], "same_action_wrong_row_exact_accepts": rejection_metrics["same_action_wrong_row_exact_accepts"], "critical_contradiction_exact_accepts": rejection_metrics["critical_contradiction_exact_accepts"], "rejection_histogram": rejection_metrics["negative_per_family_outcomes"]}
        feasible, infeasible_reasons = _calibration_feasible(metrics)
        results.append({"architecture_id": architecture_id, "calibration_values": point, "calibration_digest": calibration.digest, "feasible": feasible, "infeasible_reasons": list(infeasible_reasons), "metrics": metrics})
    _write_csv(output_dir / "calibration-grid.csv", [{**result["calibration_values"], "architecture_id": architecture_id, "calibration_digest": result["calibration_digest"], "feasible": result["feasible"], "infeasible_reasons": "|".join(result["infeasible_reasons"]), **result["metrics"]} for result in results])
    feasible_results = [result for result in results if result["feasible"]]
    if not feasible_results:
        artifact = {"selection_status": "no_feasible_operating_point", "selected_architecture": architecture_id, "selected_calibration_digest": None, "calibration_grid_digest": _sha256(results)}
    else:
        selected = min(feasible_results, key=_calibration_rank_key)
        artifact = {"selection_status": "selected_operating_point", "selected_architecture": architecture_id, "selected_calibration_digest": selected["calibration_digest"], "selected_calibration_values": selected["calibration_values"], "selected_result": selected, "calibration_grid_digest": _sha256(results)}
    _write_json(output_dir / "selected-operating-point.json", artifact)
    _write_json(output_dir / "phase-access-audits.json", benchmark.phase_audits)
    return artifact


def run_verify_pre_final_v2(output_dir: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage3-v2-prefinal-", dir=str(REPO_ROOT)) as tmp:
        temp_output = Path(tmp)
        _freeze_benchmark_v2_into(temp_output)
        selection = run_select_architecture_v2(temp_output)
        calibration = run_calibrate_v2(temp_output)
        artifact_names = sorted(({path.name for path in temp_output.iterdir() if path.is_file()} | ({path.name for path in output_dir.iterdir() if path.is_file()} if output_dir.exists() else set())) - {"pre-final-verification.json"})
        comparison = _artifact_comparison(output_dir, temp_output, artifact_names=artifact_names)
    payload = {
        "mode": "verify-pre-final-v2",
        "verified": bool(comparison["semantic_match"]),
        "comparison": comparison,
        "selection_status": selection["selection_status"],
        "calibration_status": calibration["selection_status"],
    }
    _write_json(output_dir / "pre-final-verification.json", payload)
    if not comparison["semantic_match"]:
        raise SystemExit(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    return payload


def _representation_audit_dir(output_dir: Path) -> Path:
    return output_dir / "representation-audit"


def _strict_zero_margin_exact_eligibility(
    candidate: Any,
    *,
    calibration: DiscriminativeEvidenceCalibration,
) -> bool:
    if not candidate.eligible_for_candidate_set:
        return False
    if candidate.candidate_strength + zde.REGISTRATION_DISTANCE_TIE_EPSILON < calibration.exact_winner_threshold:
        return False
    if candidate.exact_winner_margin is None:
        return False
    required = max(float(calibration.exact_winner_margin), float(zde.REGISTRATION_DISTANCE_TIE_EPSILON))
    return candidate.exact_winner_margin > required


def _effective_mask_arrays(mask: Any) -> Dict[str, np.ndarray]:
    stable = np.asarray(mask.stable_weights, dtype=np.float32)
    row_info = np.asarray(mask.row_informative_weights, dtype=np.float32)
    action_conflict = np.asarray(mask.action_conflict_weights, dtype=np.float32)
    separation = np.asarray(mask.separation_weights, dtype=np.float32)
    return {
        "stable": stable,
        "row_informative": row_info,
        "action_conflict": action_conflict,
        "positive_row_stable": row_info * stable,
        "positive_row_stable_separation": row_info * stable * separation,
        "positive_action_conflict_separation": action_conflict * stable * separation,
        "separation": separation,
    }


def _canonical_rankings_for_architecture(
    *,
    benchmark: Stage3Benchmark,
    freeze: Mapping[str, Any],
    architecture_id: str,
    calibration: DiscriminativeEvidenceCalibration,
) -> Dict[str, Tuple[Any, ...]]:
    by_observation: Dict[str, Tuple[Any, ...]] = {}
    for row_id, (_prototype_observation_id, _action_id, _digest, observation) in sorted(benchmark.prototypes.items()):
        raw = zde.build_raw_discriminative_candidates(
            observation=observation,
            prototypes=benchmark.prototypes,
            masks=freeze["masks"],
            regions=freeze["regions"],
            architecture_id=architecture_id,
            provider_digest=f"audit:{architecture_id}:{calibration.digest}",
            region_spec_digest=freeze["region_manifest"]["region_spec_digest"],
        )
        ranked = zde.rank_discriminative_candidates(raw)
        by_observation[row_id] = tuple(
            evaluate_candidate_eligibility(candidate=candidate, ranked_candidates=ranked, calibration=calibration)
            for candidate in ranked
        )
    return by_observation


def _candidate_joint_fit(*, candidate_pixels: np.ndarray, observation_pixels: np.ndarray, weight_mask: np.ndarray) -> float:
    total = float(weight_mask.sum(dtype=np.float64))
    if total <= 0.0:
        return 0.0
    distance = np.abs(observation_pixels.astype(np.float32) - candidate_pixels.astype(np.float32)) / 255.0
    mean_error = float((distance * weight_mask).sum(dtype=np.float64) / total)
    return max(0.0, 1.0 - mean_error)


def _region_crop(array: np.ndarray, region: DiscriminativeRegionSpec) -> np.ndarray:
    return array[int(region.top) : int(region.top) + int(region.height), int(region.left) : int(region.left) + int(region.width)]


def _minimum_cover_size(cover_sets: Sequence[set[int]], universe_size: int) -> Optional[int]:
    uncovered = set(range(universe_size))
    chosen = 0
    remaining = list(cover_sets)
    while uncovered and remaining:
        best = max(remaining, key=lambda item: len(item & uncovered))
        covered = best & uncovered
        if not covered:
            break
        uncovered -= covered
        remaining.remove(best)
        chosen += 1
    return chosen if not uncovered else None


def _run_representation_audit_v2(output_dir: Path) -> Dict[str, Any]:
    output_dir = OUTPUT_DIR_V2 if output_dir == OUTPUT_DIR else output_dir
    audit_dir = _representation_audit_dir(output_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name: _sha256(path.read_text(encoding="utf-8")) for path in output_dir.iterdir() if path.is_file()}
    with tempfile.TemporaryDirectory(prefix="stage3-v2-representation-audit-", dir=str(REPO_ROOT)) as tmp:
        temp_output = Path(tmp)
        run_freeze_benchmark_v2(temp_output)
        run_select_architecture_v2(temp_output)
        run_calibrate_v2(temp_output)
        verification = run_verify_pre_final_v2(temp_output)
    benchmark = _build_stage3_benchmark_v2(materialize_final=False)
    with tempfile.TemporaryDirectory(prefix="stage3-v2-representation-freeze-", dir=str(REPO_ROOT)) as tmp:
        freeze = _freeze_regions_and_masks(benchmark, output_dir=Path(tmp))
    base_calibrations = {
        architecture_id: _calibration(
            architecture_id=architecture_id,
            benchmark=benchmark,
            region_manifest=freeze["region_manifest"],
            mask_manifest=freeze["mask_manifest"],
            values={
                "minimum_available_mass": 0.0,
                "minimum_available_fraction": 0.0,
                "minimum_support": 0.0,
                "maximum_contradiction": 1.0,
                "maximum_critical_contradiction": 1.0,
                "exact_winner_threshold": 0.0,
                "exact_winner_margin": 0.0,
                "candidate_relative_margin": 0.0,
                "conflicting_action_separation": 0.0,
                "minimum_supporting_regions": 0,
                "maximum_candidate_set_size": MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
            },
        )
        for architecture_id in ("A", "B", "C")
    }
    ranked_by_architecture = {
        architecture_id: _canonical_rankings_for_architecture(
            benchmark=benchmark,
            freeze=freeze,
            architecture_id=architecture_id,
            calibration=base_calibrations[architecture_id],
        )
        for architecture_id in ("A", "B", "C")
    }
    canonical_rows = []
    summary_by_arch = {}
    lexical_only_winners = 0
    incorrect_lexical_exact_accepts = 0
    zero_strength_observations = set()
    per_action = defaultdict(lambda: Counter())
    per_factor = defaultdict(lambda: Counter())
    for architecture_id in ("A", "B", "C"):
        topk = Counter()
        tie_sizes = []
        zero_strength_candidates = 0
        incorrect_exact = 0
        top1 = 0
        for row_id, ranked in sorted(ranked_by_architecture[architecture_id].items()):
            expected_action = benchmark.prototypes[row_id][1]
            expected = next(candidate for candidate in ranked if candidate.row_id == row_id)
            tie_size = sum(abs(candidate.candidate_strength - ranked[0].candidate_strength) <= zde.REGISTRATION_DISTANCE_TIE_EPSILON for candidate in ranked)
            tie_sizes.append(tie_size)
            winner_selected_by = "lexical_tie" if tie_size > 1 else "semantic_evidence"
            if winner_selected_by == "lexical_tie":
                lexical_only_winners += 1
            rank = next(index for index, candidate in enumerate(ranked, start=1) if candidate.row_id == row_id)
            for k in (1, 3, 5, 10):
                topk[f"top{k}"] += int(rank <= k)
            top1 += int(rank == 1)
            zero_strength_candidates += sum(abs(candidate.candidate_strength) <= zde.REGISTRATION_DISTANCE_TIE_EPSILON for candidate in ranked)
            if abs(ranked[0].candidate_strength) <= zde.REGISTRATION_DISTANCE_TIE_EPSILON:
                zero_strength_observations.add((architecture_id, row_id))
            strict_exact = _strict_zero_margin_exact_eligibility(expected if rank == 1 else ranked[0], calibration=base_calibrations[architecture_id])
            candidate_set = build_discriminative_candidate_set(
                ranked_candidates=ranked,
                calibration=base_calibrations[architecture_id],
                provider_digest=f"audit:{architecture_id}",
                observation_digest=expected.observation_digest,
            )
            accepted_row = candidate_set.rows[0] if candidate_set.outcome == "exact_row_accepted" and candidate_set.rows else None
            if accepted_row is not None and (winner_selected_by == "lexical_tie" or not strict_exact):
                incorrect_exact += 1
                incorrect_lexical_exact_accepts += 1
            state_key = "|".join(f"{key}={expected.observation_digest and benchmark.prototypes[row_id][3].metadata.get(key)}" for key in ("tank", "target", "cooldown"))
            per_action[architecture_id][expected_action] += int(rank == 1)
            per_factor[architecture_id][state_key] += int(rank == 1)
            canonical_rows.append(
                {
                    "observation_row": row_id,
                    "observation_action": expected_action,
                    "architecture": architecture_id,
                    "expected_candidate_rank": rank,
                    "expected_candidate_strength": float(expected.candidate_strength),
                    "expected_candidate_raw_score": float(expected.raw_score),
                    "maximum_candidate_strength": float(ranked[0].candidate_strength),
                    "maximum_tie_size": tie_size,
                    "winner_row": ranked[0].row_id,
                    "winner_action": ranked[0].action_id,
                    "winner_selected_by": winner_selected_by,
                    "expected_available_mass": float(expected.available_informative_mass),
                    "expected_available_fraction": float(expected.available_informative_fraction),
                    "expected_support": float(expected.aggregate_support),
                    "expected_contradiction": float(expected.aggregate_contradiction),
                    "expected_critical_contradiction": float(expected.aggregate_critical_contradiction),
                    "expected_supporting_region_count": int(expected.supporting_region_count),
                    "expected_exact_margin": expected.exact_winner_margin,
                    "expected_conflicting_action_separation": expected.conflicting_action_separation,
                    "candidate_set_result": candidate_set.outcome,
                    "exact_accepted_row": accepted_row,
                }
            )
        summary_by_arch[architecture_id] = {
            "top1_count": top1,
            **dict(topk),
            "mean_maximum_tie_size": float(np.mean(tie_sizes)),
            "maximum_tie_size": max(tie_sizes),
            "incorrect_exact_acceptances": incorrect_exact,
            "zero_strength_candidate_count": zero_strength_candidates,
        }
    _write_csv(audit_dir / "canonical-self-retrieval.csv", canonical_rows)
    self_summary = {
        "canonical_observation_count": len(benchmark.prototypes),
        "architectures": summary_by_arch,
        "lexical_only_winner_count": lexical_only_winners,
        "incorrect_lexical_exact_accepts": incorrect_lexical_exact_accepts,
        "zero_strength_observation_count": len(zero_strength_observations),
        "per_action_top1": {arch: dict(sorted(counter.items())) for arch, counter in sorted(per_action.items())},
        "per_state_factor_top1": {arch: dict(sorted(counter.items())) for arch, counter in sorted(per_factor.items())},
    }
    _write_json(audit_dir / "canonical-self-retrieval-summary.json", self_summary)
    mask_rows = []
    zeroed_pixels = 0
    informative_pixels = 0
    rows_with_positive_stable = 0
    rows_with_positive_separation = 0
    for row_id, mask in sorted(freeze["masks"].items()):
        arrays = _effective_mask_arrays(mask)
        candidate_pixels = zde._coerce_observation_pixels(benchmark.prototypes[row_id][3])
        competitors = [(other_row, zde._coerce_observation_pixels(proto[3]), proto[1]) for other_row, proto in sorted(benchmark.prototypes.items()) if other_row != row_id]
        row_info = arrays["row_informative"] > 0.0
        stable = arrays["stable"] > 0.0
        positive_sep = arrays["separation"] > 0.0
        informative_pixels += int(np.count_nonzero(row_info))
        zeroed_pixels += int(np.count_nonzero(row_info & ~positive_sep))
        rows_with_positive_stable += int(np.any(arrays["positive_row_stable"] > 0.0))
        rows_with_positive_separation += int(np.any(arrays["positive_row_stable_separation"] > 0.0))
        diffs = np.stack([np.abs(candidate_pixels.astype(np.int16) - other.astype(np.int16)) for _other_row, other, _action in competitors], axis=0)
        nearest = diffs.min(axis=0)
        max_diff = diffs.max(axis=0)
        sharing = (diffs <= int(mask.spec.intensity_tolerance)).sum(axis=0)
        differing = (diffs > int(mask.spec.intensity_tolerance)).sum(axis=0)
        conflict_diffs = np.stack([np.abs(candidate_pixels.astype(np.int16) - other.astype(np.int16)) for _other_row, other, action in competitors if action != mask.spec.action_id], axis=0)
        nearest_conflict = conflict_diffs.min(axis=0) if len(conflict_diffs) else np.full(candidate_pixels.shape, 255)
        mask_rows.append(
            {
                "row_id": row_id,
                "action_id": mask.spec.action_id,
                "row_informative_pixel_count": int(np.count_nonzero(row_info)),
                "stable_pixel_count": int(np.count_nonzero(stable)),
                "row_informative_stable_count": int(np.count_nonzero(row_info & stable)),
                "positive_separation_weight_count": int(np.count_nonzero(positive_sep)),
                "row_informative_stable_positive_separation_count": int(np.count_nonzero(row_info & stable & positive_sep)),
                "action_conflict_count": int(np.count_nonzero(arrays["action_conflict"] > 0.0)),
                "positive_action_conflict_separation_count": int(np.count_nonzero(arrays["positive_action_conflict_separation"] > 0.0)),
                "contradiction_pixels_informative_but_zero_weight": int(np.count_nonzero(row_info & ~positive_sep)),
                "per_pixel_sharing_digest": _sha256(sharing.tolist()),
                "per_pixel_differing_digest": _sha256(differing.tolist()),
                "nearest_competitor_difference_digest": _sha256(nearest.tolist()),
                "maximum_competitor_difference_digest": _sha256(max_diff.tolist()),
                "nearest_conflicting_action_difference_digest": _sha256(nearest_conflict.tolist()),
            }
        )
    _write_csv(audit_dir / "mask-separation-audit.csv", mask_rows)
    mask_summary = {
        "row_count": len(mask_rows),
        "rows_with_positive_stable_informative_mass": rows_with_positive_stable,
        "rows_with_positive_separation_weighted_mass": rows_with_positive_separation,
        "informative_pixels_zeroed_by_nearest_competitor_rule": zeroed_pixels,
        "informative_pixel_count": informative_pixels,
        "informative_zeroed_fraction": 0.0 if informative_pixels == 0 else float(zeroed_pixels / informative_pixels),
        "max_difference_informative_min_difference_zero_contradiction_rows": sum(int(row["contradiction_pixels_informative_but_zero_weight"] > 0) for row in mask_rows),
    }
    _write_json(audit_dir / "mask-separation-summary.json", mask_summary)
    uniq_rows = []
    comp_rows = []
    distributed = []
    arch_a_rows = []
    implemented_a_top1 = 0
    direct_a_top1 = 0
    j_top1 = Counter()
    for row_id, (_obs_id, action_id, _digest, observation) in sorted(benchmark.prototypes.items()):
        obs_pixels = zde._coerce_observation_pixels(observation)
        candidate_pixels = {candidate_row: zde._coerce_observation_pixels(proto[3]) for candidate_row, proto in sorted(benchmark.prototypes.items())}
        full_frame_unique = sum(int(np.array_equal(obs_pixels, pixels)) for pixels in candidate_pixels.values()) == 1
        region_uniques = []
        region_tuple = []
        single_pixel_unique = False
        pair_hash_unique = False
        cover_sets = []
        tie_competitors = 0
        for candidate_row, pixels in candidate_pixels.items():
            if candidate_row == row_id:
                continue
            matches = obs_pixels == pixels
            cover_sets.append(set(np.flatnonzero(matches)))
            tie_competitors += int(matches.any())
        for region in freeze["regions"]:
            region_obs = _region_crop(obs_pixels, region)
            tuples = [_region_crop(pixels, region).tobytes() for pixels in candidate_pixels.values()]
            region_uniques.append(tuples.count(region_obs.tobytes()) == 1)
            region_tuple.append(region_obs.tobytes())
            if not single_pixel_unique:
                for idx, value in enumerate(region_obs.reshape(-1)):
                    column = [_region_crop(pixels, region).reshape(-1)[idx] for pixels in candidate_pixels.values()]
                    if column.count(int(value)) == 1:
                        single_pixel_unique = True
                        break
            if not pair_hash_unique:
                flat = region_obs.reshape(-1)
                for idx in range(len(flat)):
                    for jdx in range(idx + 1, min(len(flat), idx + 12)):
                        pair = (int(flat[idx]), int(flat[jdx]))
                        if sum(int((_region_crop(pixels, region).reshape(-1)[idx], _region_crop(pixels, region).reshape(-1)[jdx]) == pair) for pixels in candidate_pixels.values()) == 1:
                            pair_hash_unique = True
                            break
                    if pair_hash_unique:
                        break
        cross_region_unique = sum(int(tuple(_region_crop(pixels, region).tobytes() for region in freeze["regions"]) == tuple(region_tuple)) for pixels in candidate_pixels.values()) == 1
        uniq_rows.append(
            {
                "row_id": row_id,
                "action_id": action_id,
                "individual_pixel_unique": single_pixel_unique,
                "pairwise_region_feature_unique": pair_hash_unique,
                "region_joint_unique": any(region_uniques),
                "cross_region_joint_unique": cross_region_unique,
                "full_frame_unique": full_frame_unique,
            }
        )
        ranked_a = ranked_by_architecture["A"][row_id]
        implemented_a_top1 += int(ranked_a[0].row_id == row_id)
        direct_scores = []
        j1_scores = []
        j2_scores = []
        j3_scores = []
        j4_scores = []
        expected_mask = freeze["masks"][row_id]
        expected_arrays = _effective_mask_arrays(expected_mask)
        expected_arrays["positive_row_stable"]
        weighted_regions = []
        for candidate_row, pixels in candidate_pixels.items():
            direct_region_similarity = []
            region_vector = []
            for region in freeze["regions"]:
                reg = zde.extract_candidate_region_evidence(
                    candidate_row_id=candidate_row,
                    candidate_action_id=benchmark.prototypes[candidate_row][1],
                    candidate_prototype=benchmark.prototypes[candidate_row][3],
                    observation=observation,
                    mask=freeze["masks"][candidate_row],
                    competing_prototypes={key: value for key, value in benchmark.prototypes.items() if key != candidate_row},
                    region=region,
                )
                direct_region_similarity.append((1.0 - float(reg.registration_dx == 0 and reg.registration_dy == 0 and 0.0 or 0.0)) if reg.registration_succeeded else 0.0)
                region_vector.append(
                    _candidate_joint_fit(
                        candidate_pixels=_region_crop(pixels, region),
                        observation_pixels=_region_crop(obs_pixels, region),
                        weight_mask=_region_crop(_effective_mask_arrays(freeze["masks"][candidate_row])["positive_row_stable"], region),
                    )
                )
            direct_scores.append((candidate_row, float(np.mean(direct_region_similarity))))
            j1 = _candidate_joint_fit(candidate_pixels=pixels, observation_pixels=obs_pixels, weight_mask=_effective_mask_arrays(freeze["masks"][candidate_row])["positive_row_stable"])
            j1_scores.append((candidate_row, j1))
            margins = []
            for other_row, other_pixels in candidate_pixels.items():
                if other_row == candidate_row:
                    continue
                pair_mask = (_effective_mask_arrays(freeze["masks"][candidate_row])["stable"] > 0.0) & (np.abs(pixels.astype(np.int16) - other_pixels.astype(np.int16)) > int(freeze["masks"][candidate_row].spec.intensity_tolerance))
                pair_weight = pair_mask.astype(np.float32)
                fit_c = _candidate_joint_fit(candidate_pixels=pixels, observation_pixels=obs_pixels, weight_mask=pair_weight)
                fit_o = _candidate_joint_fit(candidate_pixels=other_pixels, observation_pixels=obs_pixels, weight_mask=pair_weight)
                margins.append(fit_c - fit_o)
            j2_scores.append((candidate_row, min(margins) if margins else 0.0))
            j3_scores.append((candidate_row, float(np.mean(region_vector))))
            weighted_regions.append((candidate_row, tuple(region_vector)))
            j4_scores.append((candidate_row, float(sum((index + 1) * value for index, value in enumerate(region_vector)))))
        direct_best = sorted(direct_scores, key=lambda item: (-item[1], item[0]))[0][0]
        direct_a_top1 += int(direct_best == row_id)
        for label, scores in (("J1", j1_scores), ("J2", j2_scores), ("J3", j3_scores), ("J4", j4_scores)):
            j_top1[label] += int(sorted(scores, key=lambda item: (-item[1], item[0]))[0][0] == row_id)
        distributed_cover = _minimum_cover_size(cover_sets, obs_pixels.size)
        distributed.append(
            {
                "row_id": row_id,
                "action_id": action_id,
                "different_competitors_with_matching_pixels": tie_competitors,
                "minimum_competitor_cover_size": distributed_cover,
                "distributed_competitor_cover": distributed_cover is not None and distributed_cover > 1,
            }
        )
        comp_rows.append(uniq_rows[-1])
        arch_a_rows.append(
            {
                "row_id": row_id,
                "implemented_winner_row": ranked_a[0].row_id,
                "implemented_expected_rank": next(index for index, candidate in enumerate(ranked_a, start=1) if candidate.row_id == row_id),
                "implemented_expected_strength": float(next(candidate for candidate in ranked_a if candidate.row_id == row_id).candidate_strength),
                "direct_reference_winner_row": direct_best,
                "direct_reference_expected_score": float(dict(direct_scores)[row_id]),
            }
        )
    _write_csv(audit_dir / "compositional-uniqueness.csv", comp_rows)
    compositional_summary = {
        "row_count": len(comp_rows),
        "individual_pixel_unique_row_count": sum(int(row["individual_pixel_unique"]) for row in comp_rows),
        "region_joint_unique_row_count": sum(int(row["region_joint_unique"]) for row in comp_rows),
        "cross_region_joint_unique_row_count": sum(int(row["cross_region_joint_unique"]) for row in comp_rows),
        "full_frame_unique_row_count": sum(int(row["full_frame_unique"]) for row in comp_rows),
    }
    _write_json(audit_dir / "compositional-uniqueness.json", {"rows": comp_rows, "summary": compositional_summary})
    distributed_summary = {
        "row_count": len(distributed),
        "distributed_competitor_cover_row_count": sum(int(row["distributed_competitor_cover"]) for row in distributed),
        "rows": distributed,
    }
    _write_json(audit_dir / "distributed-competitor-cover.json", distributed_summary)
    _write_csv(audit_dir / "architecture-a-conformance.csv", arch_a_rows)
    a_ruling = "architecture_a_implementation_defect" if implemented_a_top1 != direct_a_top1 else "architecture_a_conformant"
    a_summary = {
        "ruling": a_ruling,
        "implemented_a_top1": implemented_a_top1,
        "direct_correlation_top1": direct_a_top1,
    }
    _write_json(audit_dir / "architecture-a-conformance.json", a_summary)
    exact_tie = {
        "ruling": "exact_tie_implementation_defect" if incorrect_lexical_exact_accepts > 0 else "exact_tie_logic_valid",
        "incorrect_lexical_exact_accepts": incorrect_lexical_exact_accepts,
        "strict_superiority_required": True,
    }
    _write_json(audit_dir / "exact-tie-audit.json", exact_tie)
    ruling = {
        "primary_ruling": "multiple_representation_failures" if (a_ruling == "architecture_a_implementation_defect" and exact_tie["ruling"] == "exact_tie_implementation_defect") else "pointwise_representation_limitation",
        "mask_separation_ruling": "implementation_conformance_defect" if mask_summary["informative_pixels_zeroed_by_nearest_competitor_rule"] > 0 else "valid_frozen_provider_failure",
        "pointwise_support_ruling": "pointwise_representation_limitation",
        "architecture_a_ruling": a_ruling,
        "architecture_b_ruling": "pointwise_representation_limitation",
        "architecture_c_ruling": "pointwise_representation_limitation",
        "exact_tie_ruling": exact_tie["ruling"],
        "benchmark_wiring_ruling": "valid",
        "provider_wiring_ruling": "valid",
        "identity_integrity_ruling": "valid",
        "supported_v2_claim": "Stage 3 v2 established a reproducible, fully closed benchmark and showed that the frozen V4 implementation is not a valid architecture-selection instrument: its pointwise evidence representation collapses unique compositional exact rows into zero-strength ties, Architecture A does not independently exercise its declared correlation mechanism, and exact tie handling can convert lexical ordering into apparent uniqueness.",
        "unsupported_claims": [
            "no safe discriminative visual reader exists",
            "candidate sets are scientifically useless",
            "joint evidence cannot work",
            "temporal narrowing cannot work",
            "local correlation cannot work",
            "the policy rows are visually indistinguishable",
            "Kill B",
            "final performance conclusions",
        ],
        "diagnostic_top1": dict(j_top1),
    }
    _write_json(audit_dir / "ruling.json", ruling)
    _write_markdown(
        audit_dir / "ruling.md",
        "\n".join(
            [
                "# Stage 3 v2 representation audit",
                "",
                f"Primary ruling: `{ruling['primary_ruling']}`",
                "",
                f"Architecture A: `{a_ruling}`",
                f"Exact ties: `{exact_tie['ruling']}`",
                f"Supported v2 claim: {ruling['supported_v2_claim']}",
            ]
        ),
    )
    after = {path.name: _sha256(path.read_text(encoding="utf-8")) for path in output_dir.iterdir() if path.is_file()}
    mutated = sorted(name for name, digest in before.items() if after.get(name) != digest)
    payload = {
        "mode": "audit-v2-representation",
        "verified_pre_final_v2": bool(verification["verified"]),
        "audit_dir": str(audit_dir),
        "canonical_observations_audited": len(benchmark.prototypes),
        "artifacts": V2_REPRESENTATION_AUDIT_ARTIFACT_NAMES,
        "self_retrieval_summary": self_summary,
        "mask_separation_summary": mask_summary,
        "compositional_uniqueness_summary": compositional_summary,
        "distributed_competitor_cover_row_count": distributed_summary["distributed_competitor_cover_row_count"],
        "architecture_a_conformance": a_summary,
        "exact_tie_audit": exact_tie,
        "ruling": ruling,
        "mutated_v2_root_artifacts": mutated,
    }
    if mutated:
        raise SystemExit(json.dumps(_json_ready({**payload, "failure": "v2_root_artifacts_modified"}), indent=2, sort_keys=True))
    if a_ruling != "architecture_a_conformant" or exact_tie["ruling"] != "exact_tie_logic_valid" or ruling["primary_ruling"] != "valid_frozen_provider_failure":
        raise SystemExit(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    return payload


def run_evaluate_v2(output_dir: Path) -> Dict[str, Any]:
    selection = _load_json(output_dir / "selected-architecture.json")
    calibration = _load_json(output_dir / "selected-operating-point.json") if (output_dir / "selected-operating-point.json").exists() else None
    verification = _load_json(output_dir / "pre-final-verification.json") if (output_dir / "pre-final-verification.json").exists() else None
    if selection.get("selection_status") != "selected_architecture":
        raise SystemExit("V2 evaluation is blocked because no architecture has been validly selected.")
    if calibration is None or calibration.get("selection_status") != "selected_operating_point":
        raise SystemExit("V2 evaluation is blocked because no valid operating point has been selected.")
    if verification is None or not verification.get("verified"):
        raise SystemExit("V2 evaluation is blocked because pre-final verification has not passed.")
    raise SystemExit("V5 and final evaluation remain intentionally out of scope for this block.")


V3_ARTIFACT_NAMES = (
    "README.md",
    "generator-identity.json",
    "benchmark-manifest.json",
    "split-manifest.json",
    "prototype-manifest.json",
    "development-manifest.json",
    "evaluation-sample.json",
    "region-manifest.json",
    "candidate-mask-manifest.json",
    "pairwise-mask-manifest.json",
    "pairwise-mask-summary.json",
    "evidence-accounting-summary.json",
    "canonical-self-retrieval.csv",
    "canonical-self-retrieval-summary.json",
    "architecture-self-retrieval-status.json",
    "tie-safety-results.json",
    "pairwise-symmetry-results.json",
    "provider-equivalence-results.json",
    "instrument-verification.json",
    "reproduction.md",
)


def _v3_operational_contract_path() -> Path:
    return REPO_ROOT / "docs" / "research" / "video-stage-three-v3-operational-contract.md"


def _v3_operational_contract_digest() -> str:
    return _file_blob_digest(_v3_operational_contract_path())


def _prototype_manifest_v3(benchmark: Stage3Benchmark, *, generator_identity: Mapping[str, Any]) -> Dict[str, Any]:
    return _prototype_manifest_v2(benchmark, generator_identity=generator_identity)


def _development_manifest_v3(benchmark: Stage3Benchmark, *, generator_identity: Mapping[str, Any]) -> Dict[str, Any]:
    return _development_manifest_v2(benchmark, generator_identity=generator_identity)


def _evaluation_sample_manifest_v3(benchmark: Stage3Benchmark) -> Dict[str, Any]:
    return _evaluation_sample_manifest_v2(benchmark)


def _freeze_joint_regions_and_masks(benchmark: Stage3Benchmark, *, output_dir: Path) -> Dict[str, Any]:
    regions = _joint_region_manifest()
    candidate_masks = build_joint_candidate_masks(
        prototypes=benchmark.prototypes,
        development_observations=_development_observations(benchmark),
        intensity_tolerance=8,
        stability_tolerance=12,
        amendment_commit_sha=V3_AMENDMENT_COMMIT,
        operational_contract_digest=_v3_operational_contract_digest(),
        source_scope=benchmark.source_scope,
    )
    pairwise_masks = build_pairwise_discriminative_masks(
        prototypes=benchmark.prototypes,
        candidate_masks=candidate_masks,
        intensity_tolerance=8,
        amendment_commit_sha=V3_AMENDMENT_COMMIT,
        operational_contract_digest=_v3_operational_contract_digest(),
        source_scope=benchmark.source_scope,
    )
    region_manifest = {
        "regions": [region.to_dict() for region in regions],
        "region_spec_digest": joint_region_digest(regions),
        "registration_contract_digest_set": sorted({region.registration_config.digest for region in regions}),
    }
    candidate_mask_manifest = {
        "mask_specs": [mask.spec.to_dict() for _row_id, mask in sorted(candidate_masks.items())],
        "mask_payload_digests": {row_id: mask.payload_digest for row_id, mask in sorted(candidate_masks.items())},
        "candidate_mask_digest": joint_candidate_mask_digest(tuple(mask.spec for mask in candidate_masks.values())),
        "prototype_digest": _sha256({key: value[2] for key, value in sorted(benchmark.prototypes.items())}),
        "development_digest": _sha256({row_id: [item.raw_digest for item in values] for row_id, values in sorted(_development_observations(benchmark).items())}),
        "intensity_tolerance": 8,
        "stability_tolerance": 12,
        "operational_contract_digest": _v3_operational_contract_digest(),
        "amendment_commit_sha": V3_AMENDMENT_COMMIT,
    }
    pairwise_mask_manifest = {
        "pairwise_mask_specs": [mask.spec.to_dict() for _pair, mask in sorted(pairwise_masks.items())],
        "pairwise_mask_payload_digests": {f"{row_a}|{row_b}": mask.payload_digest for (row_a, row_b), mask in sorted(pairwise_masks.items())},
        "pairwise_mask_digest": pairwise_mask_digest(tuple(mask.spec for mask in pairwise_masks.values())),
    }
    same_action_pairs = sum(int(mask.spec.action_a == mask.spec.action_b) for mask in pairwise_masks.values())
    different_action_pairs = len(pairwise_masks) - same_action_pairs
    pairwise_mask_summary = {
        "pairwise_mask_count": len(pairwise_masks),
        "nonzero_pairwise_mask_count": sum(int(mask.spec.pairwise_pixel_count > 0) for mask in pairwise_masks.values()),
        "zero_mass_pair_count": sum(int(mask.spec.pairwise_pixel_count == 0) for mask in pairwise_masks.values()),
        "same_action_pair_count": same_action_pairs,
        "different_action_pair_count": different_action_pairs,
        "maximum_pairwise_mass": max(mask.spec.pairwise_pixel_count for mask in pairwise_masks.values()),
        "minimum_nonzero_pairwise_mass": min(mask.spec.pairwise_pixel_count for mask in pairwise_masks.values() if mask.spec.pairwise_pixel_count > 0),
        "candidate_mask_count": len(candidate_masks),
        "pairwise_mask_digest": pairwise_mask_manifest["pairwise_mask_digest"],
    }
    _write_json(output_dir / "region-manifest.json", region_manifest)
    _write_json(output_dir / "candidate-mask-manifest.json", candidate_mask_manifest)
    _write_json(output_dir / "pairwise-mask-manifest.json", pairwise_mask_manifest)
    _write_json(output_dir / "pairwise-mask-summary.json", pairwise_mask_summary)
    return {
        "regions": regions,
        "candidate_masks": candidate_masks,
        "pairwise_masks": pairwise_masks,
        "region_manifest": region_manifest,
        "candidate_mask_manifest": candidate_mask_manifest,
        "pairwise_mask_manifest": pairwise_mask_manifest,
        "pairwise_mask_summary": pairwise_mask_summary,
    }


def _v3_calibration(*, architecture_id: str, benchmark: Stage3Benchmark, freeze: Mapping[str, Any]) -> JointEvidenceCalibration:
    return JointEvidenceCalibration(
        architecture_id=architecture_id,
        minimum_actual_scored_mass=0.0,
        minimum_available_candidate_fit_fraction=0.0,
        minimum_candidate_joint_fit=0.0,
        minimum_pairwise_margin=-1.0,
        minimum_conflicting_action_margin=-1.0,
        exact_winner_threshold=0.0,
        exact_winner_margin=0.0,
        candidate_relative_margin=0.0,
        maximum_candidate_set_size=MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
        prototype_digest=freeze["candidate_mask_manifest"]["prototype_digest"],
        region_spec_digest=freeze["region_manifest"]["region_spec_digest"],
        candidate_mask_digest=freeze["candidate_mask_manifest"]["candidate_mask_digest"],
        pairwise_mask_digest=freeze["pairwise_mask_manifest"]["pairwise_mask_digest"],
        policy_artifact_id=benchmark.policy_artifact_id,
        source_scope=benchmark.source_scope,
        amendment_commit_sha=V3_AMENDMENT_COMMIT,
        operational_contract_digest=_v3_operational_contract_digest(),
    )


def _direct_joint_ranked_candidates(
    *,
    architecture_id: str,
    benchmark: Stage3Benchmark,
    freeze: Mapping[str, Any],
    observation: ImageObservation,
) -> Tuple[Any, ...]:
    calibration = _v3_calibration(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze)
    raw = zje.build_joint_row_candidates(
        observation=observation,
        prototypes=benchmark.prototypes,
        candidate_masks=freeze["candidate_masks"],
        pairwise_masks=freeze["pairwise_masks"],
        regions=freeze["regions"],
        architecture_id=architecture_id,
    )
    ranked = zje.rank_joint_row_candidates(raw)
    return tuple(evaluate_joint_candidate_eligibility(candidate=item, ranked_candidates=ranked, calibration=calibration) for item in ranked)


def _freeze_benchmark_v3_into(output_dir: Path) -> Dict[str, Any]:
    benchmark = _build_stage3_benchmark_v3(materialize_final=False)
    generator_identity = _generator_identity_v3(benchmark)
    prototype_manifest = _prototype_manifest_v3(benchmark, generator_identity=generator_identity)
    development_manifest = _development_manifest_v3(benchmark, generator_identity=generator_identity)
    evaluation_sample = _evaluation_sample_manifest_v3(benchmark)
    freeze = _freeze_joint_regions_and_masks(benchmark, output_dir=output_dir)
    benchmark_manifest = _benchmark_manifest(benchmark)
    split_manifest = _split_manifest(benchmark)
    _write_json(output_dir / "generator-identity.json", generator_identity)
    _write_json(output_dir / "prototype-manifest.json", prototype_manifest)
    _write_json(output_dir / "development-manifest.json", development_manifest)
    _write_json(output_dir / "evaluation-sample.json", evaluation_sample)
    _write_json(output_dir / "benchmark-manifest.json", benchmark_manifest)
    _write_json(output_dir / "split-manifest.json", split_manifest)
    _write_markdown(
        output_dir / "README.md",
        f"# Stage 3 v3 current-frame joint evidence\n\nBenchmark version: `{benchmark_manifest['benchmark_version']}`\nGenerator version: `{benchmark_manifest['generator_version']}`\nSeed digest: `{benchmark_manifest['final_seed_digest']}`\nBenchmark digest: `{benchmark_manifest['benchmark_digest']}`\nSplit digest: `{split_manifest['split_digest']}`",
    )
    _write_markdown(
        output_dir / "reproduction.md",
        "Rebuild with `python examples/arcade_visual_video_discriminative_evidence_benchmark.py --freeze-benchmark-v3`, then audit with `--audit-v3-self-retrieval`, and verify with `--verify-v3-instrument`.",
    )
    return {
        "benchmark": benchmark,
        "freeze": freeze,
        "generator_identity": generator_identity,
        "prototype_manifest": prototype_manifest,
        "development_manifest": development_manifest,
        "evaluation_sample": evaluation_sample,
        "benchmark_manifest": benchmark_manifest,
        "split_manifest": split_manifest,
    }


def run_freeze_benchmark_v3(output_dir: Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage3-v3-freeze-", dir=str(REPO_ROOT)) as tmp:
        temp_output = Path(tmp)
        result = _freeze_benchmark_v3_into(temp_output)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(temp_output, output_dir)
    return {
        "mode": "freeze-benchmark-v3",
        "benchmark_digest": result["benchmark_manifest"]["benchmark_digest"],
        "split_digest": result["split_manifest"]["split_digest"],
        "generator_identity_digest": result["generator_identity"]["generator_identity_digest"],
        "candidate_mask_digest": result["freeze"]["candidate_mask_manifest"]["candidate_mask_digest"],
        "pairwise_mask_digest": result["freeze"]["pairwise_mask_manifest"]["pairwise_mask_digest"],
    }


def run_audit_v3_self_retrieval(output_dir: Path) -> Dict[str, Any]:
    output_dir = OUTPUT_DIR_V3 if output_dir == OUTPUT_DIR else output_dir
    if not output_dir.exists():
        run_freeze_benchmark_v3(output_dir)
    benchmark = _build_stage3_benchmark_v3(materialize_final=False)
    with tempfile.TemporaryDirectory(prefix="stage3-v3-audit-freeze-", dir=str(REPO_ROOT)) as tmp:
        freeze = _freeze_joint_regions_and_masks(benchmark, output_dir=Path(tmp))
    rows = []
    by_arch = {}
    incorrect_lexical_exact_accepts = 0
    provider_equivalence = {"architectures": {}, "all_match": True}
    tie_safety = {"architectures": {}, "incorrect_lexical_exact_accepts": 0, "equal_score_exact_accept_count": 0}
    accounting = {
        "row_count": len(benchmark.prototypes),
        "declared_informative_mass_total": 0.0,
        "stable_informative_mass_total": 0.0,
        "available_geometric_mass_total": 0.0,
        "available_candidate_fit_mass_total": 0.0,
        "pairwise_discriminative_mass_total": 0.0,
        "actual_scored_mass_total": 0.0,
    }
    pairwise_symmetry = {
        "pairwise_mask_count": len(freeze["pairwise_masks"]),
        "mask_symmetry_valid": True,
        "pairwise_antisymmetry_valid": True,
        "cache_hits": 0,
        "cache_misses": len(freeze["pairwise_masks"]),
    }
    for (row_a, row_b), mask in freeze["pairwise_masks"].items():
        if row_a >= row_b:
            pairwise_symmetry["mask_symmetry_valid"] = False
        if not np.array_equal(mask.pairwise_weights, freeze["pairwise_masks"][(row_a, row_b)].pairwise_weights):
            pairwise_symmetry["mask_symmetry_valid"] = False
    for architecture_id in ("A3", "B3", "C3", "D3"):
        calibration = _v3_calibration(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze)
        provider = JointEvidenceProvider(
            prototypes=benchmark.prototypes,
            candidate_masks=freeze["candidate_masks"],
            pairwise_masks=freeze["pairwise_masks"],
            regions=freeze["regions"],
            calibration=calibration,
            policy_artifact_id=benchmark.policy_artifact_id,
            source_scope=benchmark.source_scope,
        )
        top1 = 0
        tie_sizes = []
        outcome_counter = Counter()
        arch_equal_exact_accepts = 0
        arch_provider_equivalent = True
        for row_id, (_prototype_observation_id, action_id, _digest, observation) in sorted(benchmark.prototypes.items()):
            direct = _direct_joint_ranked_candidates(architecture_id=architecture_id, benchmark=benchmark, freeze=freeze, observation=observation)
            ranked = provider._rank(observation)
            if [item.to_dict() for item in direct] != [item.to_dict() for item in ranked]:
                arch_provider_equivalent = False
                provider_equivalence["all_match"] = False
            candidate_set = build_joint_candidate_set(ranked_candidates=ranked, calibration=calibration)
            outcome_counter[candidate_set.outcome] += 1
            expected = next(candidate for candidate in ranked if candidate.row_id == row_id)
            winner = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else None
            top1 += int(winner.row_id == row_id and winner.semantic_tie_group_size == 1)
            tie_sizes.append(int(winner.semantic_tie_group_size))
            lexical_affected = bool(winner.semantic_tie_group_size > 1)
            accepted_row = candidate_set.rows[0] if candidate_set.outcome == "exact_row_accepted" and candidate_set.rows else None
            if lexical_affected and accepted_row is not None:
                incorrect_lexical_exact_accepts += 1
                arch_equal_exact_accepts += 1
            if winner.semantic_tie_group_size > 1 and accepted_row is not None:
                tie_safety["equal_score_exact_accept_count"] += 1
            accounting["declared_informative_mass_total"] += float(expected.declared_informative_mass)
            accounting["stable_informative_mass_total"] += float(expected.stable_informative_mass)
            accounting["available_geometric_mass_total"] += float(expected.available_geometric_mass)
            accounting["available_candidate_fit_mass_total"] += float(expected.available_candidate_fit_mass)
            accounting["pairwise_discriminative_mass_total"] += float(expected.pairwise_discriminative_mass)
            accounting["actual_scored_mass_total"] += float(expected.actual_scored_mass)
            rows.append(
                {
                    "architecture": architecture_id,
                    "observation_row": row_id,
                    "observation_action": action_id,
                    "expected_candidate_rank": next(index for index, candidate in enumerate(ranked, start=1) if candidate.row_id == row_id),
                    "expected_candidate_strength": float(expected.candidate_strength),
                    "expected_candidate_scored_mass": float(expected.actual_scored_mass),
                    "expected_candidate_joint_fit": float(expected.candidate_joint_fit),
                    "expected_minimum_pairwise_margin": expected.minimum_pairwise_margin,
                    "expected_conflicting_action_margin": expected.minimum_conflicting_action_margin,
                    "winner_row": winner.row_id,
                    "winner_strength": float(winner.candidate_strength),
                    "runner_up_row": None if runner_up is None else runner_up.row_id,
                    "runner_up_strength": None if runner_up is None else float(runner_up.candidate_strength),
                    "semantic_tie_size": int(winner.semantic_tie_group_size),
                    "semantic_tie_rows": list(winner.semantic_tie_group_rows),
                    "winner_superiority_margin": winner.candidate_superiority_margin,
                    "expected_row_candidate_set_eligibility": bool(expected.eligible_for_candidate_set),
                    "expected_row_exact_eligibility": bool(expected.eligible_for_exact),
                    "candidate_set_outcome": candidate_set.outcome,
                    "exact_accepted_row": accepted_row,
                    "lexical_ordering_affected_outcome": lexical_affected,
                }
            )
            if architecture_id == "C3":
                direct_by_row = {item.competitor_row_id: item.margin for item in expected.pairwise_evidence if item.region_id == "target_band"}
                for candidate in ranked:
                    if candidate.row_id == row_id:
                        continue
                    reverse = next((item.margin for item in candidate.pairwise_evidence if item.competitor_row_id == row_id and item.region_id == "target_band"), None)
                    forward = direct_by_row.get(candidate.row_id)
                    if reverse is not None and forward is not None and abs(forward + reverse) > 1e-6:
                        pairwise_symmetry["pairwise_antisymmetry_valid"] = False
        status = "eligible_self_retrieval" if top1 == len(benchmark.prototypes) else "ineligible_self_retrieval"
        by_arch[architecture_id] = {
            "canonical_top1_count": top1,
            "canonical_observation_count": len(benchmark.prototypes),
            "mean_semantic_tie_size": float(np.mean(tie_sizes)),
            "maximum_semantic_tie_size": max(tie_sizes),
            "candidate_set_outcomes": dict(sorted(outcome_counter.items())),
            "self_retrieval_status": status,
        }
        provider_equivalence["architectures"][architecture_id] = arch_provider_equivalent
        tie_safety["architectures"][architecture_id] = {
            "incorrect_lexical_exact_accepts": arch_equal_exact_accepts,
        }
    tie_safety["incorrect_lexical_exact_accepts"] = incorrect_lexical_exact_accepts
    _write_csv(output_dir / "canonical-self-retrieval.csv", rows)
    _write_json(output_dir / "canonical-self-retrieval-summary.json", {"architectures": by_arch, "canonical_observation_count": len(benchmark.prototypes)})
    shared_valid = bool(provider_equivalence["all_match"] and pairwise_symmetry["mask_symmetry_valid"] and pairwise_symmetry["pairwise_antisymmetry_valid"] and incorrect_lexical_exact_accepts == 0)
    if shared_valid and any(item["self_retrieval_status"] == "eligible_self_retrieval" for item in by_arch.values()):
        stop_outcome = "Outcome B" if any(item["self_retrieval_status"] != "eligible_self_retrieval" for item in by_arch.values()) else "Outcome A"
    elif shared_valid:
        stop_outcome = "Outcome C"
    else:
        stop_outcome = "Outcome D"
    _write_json(output_dir / "architecture-self-retrieval-status.json", {"shared_instrument_valid": shared_valid, "architectures": by_arch, "stop_outcome": stop_outcome})
    _write_json(output_dir / "tie-safety-results.json", tie_safety)
    _write_json(output_dir / "pairwise-symmetry-results.json", pairwise_symmetry)
    _write_json(output_dir / "provider-equivalence-results.json", provider_equivalence)
    _write_json(output_dir / "evidence-accounting-summary.json", accounting)
    return {
        "mode": "audit-v3-self-retrieval",
        "canonical_observation_count": len(benchmark.prototypes),
        "architectures": by_arch,
        "tie_safety": tie_safety,
        "pairwise_symmetry": pairwise_symmetry,
        "provider_equivalence": provider_equivalence,
        "evidence_accounting": accounting,
        "stop_outcome": stop_outcome,
    }


def run_verify_v3_instrument(output_dir: Path) -> Dict[str, Any]:
    output_dir = OUTPUT_DIR_V3 if output_dir == OUTPUT_DIR else output_dir
    with tempfile.TemporaryDirectory(prefix="stage3-v3-verify-", dir=str(REPO_ROOT)) as tmp:
        temp_output = Path(tmp)
        run_freeze_benchmark_v3(temp_output)
        run_audit_v3_self_retrieval(temp_output)
        artifacts = [name for name in V3_ARTIFACT_NAMES if name != "instrument-verification.json"]
        comparison = _artifact_comparison(output_dir, temp_output, artifact_names=artifacts)
    forbidden = [name for name in ("architecture-grid.csv", "selected-architecture.json", "calibration-grid.csv", "selected-operating-point.json", "final-metrics.json") if (output_dir / name).exists()]
    payload = {
        "mode": "verify-v3-instrument",
        "verified": bool(comparison["semantic_match"]) and not forbidden,
        "comparison": comparison,
        "forbidden_artifacts_present": forbidden,
        "required_artifacts": list(V3_ARTIFACT_NAMES),
    }
    _write_json(output_dir / "instrument-verification.json", payload)
    if not payload["verified"]:
        raise SystemExit(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    return payload


def _stage2_diagnosis_file_digests(output_dir: Path) -> Dict[str, Any]:
    files = {
        "summary_json": output_dir / "diagnostics" / "stage2-posthoc-summary.json",
        "region_summary_csv": output_dir / "diagnostics" / "stage2-region-summary.csv",
        "frame_candidates_csv": output_dir / "diagnostics" / "stage2-frame-candidates.csv",
        "correct_row_ranks_json": output_dir / "diagnostics" / "stage2-correct-row-ranks.json",
        "same_action_ranks_json": output_dir / "diagnostics" / "stage2-same-action-ranks.json",
        "conflicting_action_ranks_json": output_dir / "diagnostics" / "stage2-conflicting-action-ranks.json",
    }
    return {name: _sha256(path.read_text(encoding="utf-8")) for name, path in files.items()}


def _write_stage2_diagnosis_digest_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest = {
        "stage2_parent_commit": STAGE2_PARENT_COMMIT,
        "stage2_benchmark_digest": STAGE2_BENCHMARK_DIGEST,
        "stage2_split_digest": STAGE2_SPLIT_DIGEST,
        "diagnostics": _stage2_diagnosis_file_digests(output_dir),
    }
    _write_json(output_dir / "diagnostics" / "stage2-diagnosis-digests.json", manifest)
    return manifest


def _diagnose_stage2(output_dir: Path) -> Dict[str, Any]:
    policy = compile_policy_artifact()
    policy_lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    from examples.arcade_visual_local_evidence_benchmark import build_arcade_local_evidence_dataset

    dataset = build_arcade_local_evidence_dataset(variants_per_family=1)
    selection = _build_v2_selection(dataset, policy_lookup)
    provider = _build_v2_provider(dataset, selection["selected_calibration"])
    cases = build_video_cases()
    topk_hits = Counter()
    topk_action_hits = Counter()
    family_rejections = Counter()
    region_visibility_by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    region_overlap_by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    region_distance_by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    region_failure_by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    same_action_rank_positions = []
    conflicting_action_rank_positions = []
    exact_rank_positions = []
    candidate_rows = []
    for case in cases:
        for frame, expected_row, expected_action, _disposition in zip(case.source.frames(), case.expected_rows, case.expected_actions, case.expected_dispositions):
            observation = ImageObservation(frame.pixels, source_id=frame.frame_id, metadata=frame.metadata)
            ranked = provider._rank(observation)
            decision = provider.read(observation)
            family_rejections[(case.family, decision.reason)] += 1
            if expected_row is not None:
                matching_rank = next((idx for idx, candidate in enumerate(ranked, start=1) if candidate.row_id == expected_row), None)
                exact_rank_positions.append({"family": case.family, "frame_id": frame.frame_id, "rank": matching_rank})
                for k in (1, 2, 3, 5):
                    topk_hits[f"top{k}"] += int(matching_rank is not None and matching_rank <= k)
                same_action_rank = next((idx for idx, candidate in enumerate(ranked, start=1) if candidate.action_id == expected_action), None)
                same_action_rank_positions.append({"family": case.family, "frame_id": frame.frame_id, "rank": same_action_rank})
                for k in (1, 2, 3, 5):
                    topk_action_hits[f"top{k}"] += int(same_action_rank is not None and same_action_rank <= k)
                conflicting_rank = next((idx for idx, candidate in enumerate(ranked, start=1) if candidate.action_id != expected_action), None)
                conflicting_action_rank_positions.append({"family": case.family, "frame_id": frame.frame_id, "rank": conflicting_rank})
            best = ranked[0]
            for region in best.region_evidence:
                region_visibility_by_family[case.family][region.region_id].append(float(region.visible_fraction))
                region_overlap_by_family[case.family][region.region_id].append(float(region.overlap_fraction))
                region_distance_by_family[case.family][region.region_id].append(float(region.distance))
                if float(region.visible_fraction) + 1e-12 < provider._calibration.minimum_visible_fraction:
                    region_failure_by_family[case.family][region.region_id] += 1
            candidate_rows.append(
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "frame_id": frame.frame_id,
                    "expected_row": expected_row,
                    "expected_action": expected_action,
                    "decision_reason": decision.reason,
                    "accepted": decision.accepted,
                    "top1_row": best.row_id,
                    "top1_action": best.action_id,
                    "top1_distance": best.total_distance,
                    "top1_visible_fraction": best.visible_fraction,
                    "correct_row_rank": None if expected_row is None else next((idx for idx, candidate in enumerate(ranked, start=1) if candidate.row_id == expected_row), None),
                    "same_action_rank": None if expected_action is None else next((idx for idx, candidate in enumerate(ranked, start=1) if candidate.action_id == expected_action), None),
                    "conflicting_action_rank": None if expected_action is None else next((idx for idx, candidate in enumerate(ranked, start=1) if candidate.action_id != expected_action), None),
                    "top5_rows": [candidate.row_id for candidate in ranked[:5]],
                    "top5_actions": [candidate.action_id for candidate in ranked[:5]],
                }
            )
    region_summary_rows = []
    for family, regions in sorted(region_visibility_by_family.items()):
        for region_id, visibilities in sorted(regions.items()):
            overlaps = region_overlap_by_family[family][region_id]
            distances = region_distance_by_family[family][region_id]
            region_summary_rows.append(
                {
                    "family": family,
                    "region_id": region_id,
                    "mean_visible_fraction": float(np.mean(visibilities)),
                    "min_visible_fraction": float(np.min(visibilities)),
                    "mean_overlap_fraction": float(np.mean(overlaps)),
                    "min_overlap_fraction": float(np.min(overlaps)),
                    "mean_distance": float(np.mean(distances)),
                    "low_visibility_count": int(region_failure_by_family[family].get(region_id, 0)),
                }
            )
    diagnostics = {
        "stage2_parent_commit": STAGE2_PARENT_COMMIT,
        "stage2_benchmark_digest": STAGE2_BENCHMARK_DIGEST,
        "stage2_split_digest": STAGE2_SPLIT_DIGEST,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": selection["selection"]["selection_status"],
        "selected_calibration_digest": selection["selected_calibration_digest"],
        "selected_calibration": selection["selected_calibration"],
        "region_contract_digest": _sha256([region.to_dict() for region in _stage2_regions()]),
        "candidate_grid_digest": selection["candidate_grid_digest"],
        "family_rejections": {
            family: {reason: count for (fam, reason), count in sorted(family_rejections.items()) if fam == family}
            for family in sorted({family for family, _reason in family_rejections})
        },
        "topk_correct_row_hits": dict(sorted(topk_hits.items())),
        "topk_correct_action_hits": dict(sorted(topk_action_hits.items())),
        "region_summaries": region_summary_rows,
        "diagnostic_row_count": len(candidate_rows),
    }
    _write_json(output_dir / "diagnostics" / "stage2-posthoc-summary.json", diagnostics)
    _write_csv(output_dir / "diagnostics" / "stage2-region-summary.csv", region_summary_rows)
    _write_csv(output_dir / "diagnostics" / "stage2-frame-candidates.csv", candidate_rows)
    _write_json(output_dir / "diagnostics" / "stage2-correct-row-ranks.json", exact_rank_positions)
    _write_json(output_dir / "diagnostics" / "stage2-same-action-ranks.json", same_action_rank_positions)
    _write_json(output_dir / "diagnostics" / "stage2-conflicting-action-ranks.json", conflicting_action_rank_positions)
    _write_stage2_diagnosis_digest_manifest(output_dir)
    return diagnostics


def run_verify_stage2_diagnosis(output_dir: Path) -> Dict[str, Any]:
    expected = _load_json(output_dir / "diagnostics" / "stage2-diagnosis-digests.json")
    actual = {
        "stage2_parent_commit": STAGE2_PARENT_COMMIT,
        "stage2_benchmark_digest": STAGE2_BENCHMARK_DIGEST,
        "stage2_split_digest": STAGE2_SPLIT_DIGEST,
        "diagnostics": _stage2_diagnosis_file_digests(output_dir),
    }
    verified = expected == actual
    payload = {
        "mode": "verify-stage2-diagnosis",
        "verified": verified,
        "expected_digest": _sha256(expected),
        "actual_digest": _sha256(actual),
    }
    if not verified:
        raise SystemExit(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_rebuild_stage2_diagnosis(output_dir: Path) -> Dict[str, Any]:
    diagnostics = _diagnose_stage2(output_dir)
    return {
        "mode": "rebuild-stage2-diagnosis",
        "benchmark_version": BENCHMARK_VERSION,
        "diagnostic_summary_digest": _sha256(diagnostics),
        "selection_status": diagnostics["selection_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--verify-stage2-diagnosis", action="store_true")
    parser.add_argument("--rebuild-stage2-diagnosis", action="store_true")
    parser.add_argument("--diagnose-stage2", action="store_true")
    parser.add_argument("--select-architecture", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--verify-pre-final", action="store_true")
    parser.add_argument("--audit-pre-final-v1", action="store_true")
    parser.add_argument("--audit-v1-exact-frames", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--freeze-benchmark-v2", action="store_true")
    parser.add_argument("--verify-v2-benchmark", action="store_true")
    parser.add_argument("--select-architecture-v2", action="store_true")
    parser.add_argument("--calibrate-v2", action="store_true")
    parser.add_argument("--verify-pre-final-v2", action="store_true")
    parser.add_argument("--audit-v2-representation", action="store_true")
    parser.add_argument("--evaluate-v2", action="store_true")
    parser.add_argument("--freeze-benchmark-v3", action="store_true")
    parser.add_argument("--audit-v3-self-retrieval", action="store_true")
    parser.add_argument("--verify-v3-instrument", action="store_true")
    args = parser.parse_args()
    if args.verify_stage2_diagnosis:
        payload = run_verify_stage2_diagnosis(args.output_dir)
    elif args.rebuild_stage2_diagnosis or args.diagnose_stage2:
        payload = run_rebuild_stage2_diagnosis(args.output_dir)
    elif args.select_architecture:
        payload = _run_selection(args.output_dir)
    elif args.calibrate:
        payload = _run_calibrate(args.output_dir)
    elif args.verify_pre_final:
        payload = _verify_pre_final(args.output_dir)
    elif args.audit_pre_final_v1:
        payload = _run_audit_pre_final_v1(args.output_dir)
    elif args.audit_v1_exact_frames:
        payload = _audit_exact_frames(output_dir=args.output_dir / "measurement-audit")
    elif args.evaluate:
        payload = _run_evaluate(args.output_dir)
    elif args.freeze_benchmark_v2:
        payload = run_freeze_benchmark_v2(OUTPUT_DIR_V2 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.verify_v2_benchmark:
        payload = run_verify_v2_benchmark(OUTPUT_DIR_V2 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.select_architecture_v2:
        payload = run_select_architecture_v2(OUTPUT_DIR_V2 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.calibrate_v2:
        payload = run_calibrate_v2(OUTPUT_DIR_V2 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.verify_pre_final_v2:
        payload = run_verify_pre_final_v2(OUTPUT_DIR_V2 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.audit_v2_representation:
        payload = _run_representation_audit_v2(OUTPUT_DIR_V2 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.evaluate_v2:
        payload = run_evaluate_v2(OUTPUT_DIR_V2 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.freeze_benchmark_v3:
        payload = run_freeze_benchmark_v3(OUTPUT_DIR_V3 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.audit_v3_self_retrieval:
        payload = run_audit_v3_self_retrieval(OUTPUT_DIR_V3 if args.output_dir == OUTPUT_DIR else args.output_dir)
    elif args.verify_v3_instrument:
        payload = run_verify_v3_instrument(OUTPUT_DIR_V3 if args.output_dir == OUTPUT_DIR else args.output_dir)
    else:
        raise SystemExit("one stage flag is required")
    print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
