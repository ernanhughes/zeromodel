from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from zeromodel.core.bundle import from_bundle, to_bundle
from zeromodel.core.artifact import build_vpm
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.video.arcade_policy import (
    ACTIONS,
    CELL_PIXELS,
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
    FRAME_HEIGHT,
    TARGET_VALUE,
    ShooterConfig,
    TinyArcadeShooter,
    compile_policy_artifact,
    enumerate_visual_frames,
    parse_state_row_id,
    render_state_frame,
)
from zeromodel.vision import VisualFeatureSpec, VisualSignReader, build_visual_index
from zeromodel.vision.visual import extract_visual_features


RESULT_ROOT = Path("docs/results/visual-sign-reader-hardening")


def _json_default(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arcade_visual_feature_spec(config: ShooterConfig) -> VisualFeatureSpec:
    return VisualFeatureSpec(
        input_height=FRAME_HEIGHT,
        input_width=config.width * CELL_PIXELS,
        target_height=8,
        target_width=config.width * 2,
        quantization_levels=16,
    )


@dataclass(frozen=True)
class Runtime:
    config: ShooterConfig
    policy: Any
    frames: Mapping[str, np.ndarray]
    feature_spec: VisualFeatureSpec
    visual_build: Any
    reader: VisualSignReader
    policy_lookup: VPMPolicyLookup


def make_runtime() -> Runtime:
    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    frames = dict(enumerate_visual_frames(config))
    spec = arcade_visual_feature_spec(config)
    visual_build = build_visual_index(
        policy,
        frames,
        spec,
        threshold_fraction=0.25,
        margin_fraction=0.75,
        name="arcade-visual-index-source-order",
    )
    reader = VisualSignReader(
        visual_build.artifact,
        policy,
        action_metric_ids=ACTIONS,
        value_source="raw",
        tie_break="metric_order",
    )
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    return Runtime(config, policy, frames, spec, visual_build, reader, lookup)


def summarize_decisions(records: list[dict[str, Any]]) -> dict[str, Any]:
    distances = [float(item["nearest_distance"]) for item in records]
    margins = [float(item["distance_margin"]) for item in records]
    reasons: dict[str, int] = {}
    for item in records:
        reasons[str(item["reason"])] = reasons.get(str(item["reason"]), 0) + 1

    def stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "median": None, "max": None}
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
        }

    return {
        "state_count": len(records),
        "accepted_count": sum(int(item["accepted"]) for item in records),
        "rejected_count": sum(int(not item["accepted"]) for item in records),
        "correct_row_count": sum(int(item["correct_row"]) for item in records),
        "incorrect_row_count": sum(int(item["incorrect_row"]) for item in records),
        "correct_action_count": sum(int(item["correct_action"]) for item in records),
        "incorrect_action_count": sum(
            int(item["incorrect_action"]) for item in records
        ),
        "false_acceptance_count": sum(
            int(item["false_acceptance"]) for item in records
        ),
        "false_rejection_count": sum(int(item["false_rejection"]) for item in records),
        "nearest_distance": stats(distances),
        "distance_margin": stats(margins),
        "rejection_reasons": reasons,
    }


def evaluate_frames(
    runtime: Runtime,
    frames: Mapping[str, np.ndarray],
    *,
    expected_valid: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id, frame in frames.items():
        expected_action = runtime.policy_lookup.choose(row_id)
        try:
            decision = runtime.reader.read(frame)
            accepted = bool(decision.accepted)
            matched = decision.matched_row_id
            action = decision.action
            reason = decision.reason
            nearest = decision.nearest_distance
            second = decision.second_nearest_distance
            margin = decision.distance_margin
            exact = decision.exact_feature_match
            nearest_row_id = decision.nearest_row_id
        except Exception as exc:  # noqa: BLE001 - audit record
            accepted = False
            matched = None
            action = None
            reason = f"exception:{type(exc).__name__}:{exc}"
            nearest = math.nan
            second = math.nan
            margin = math.nan
            exact = False
            nearest_row_id = None
        records.append(
            {
                "row_id": row_id,
                "accepted": accepted,
                "reason": reason,
                "matched_row_id": matched,
                "nearest_row_id": nearest_row_id,
                "action": action,
                "expected_action": expected_action,
                "correct_row": accepted and matched == row_id,
                "incorrect_row": accepted and matched != row_id,
                "correct_action": accepted and action == expected_action,
                "incorrect_action": accepted and action != expected_action,
                "false_acceptance": accepted and not expected_valid,
                "false_rejection": (not accepted) and expected_valid,
                "nearest_distance": nearest,
                "second_nearest_distance": second,
                "distance_margin": margin,
                "exact_feature_match": exact,
            }
        )
    return records, summarize_decisions(records)


def canonical_state_results(runtime: Runtime) -> dict[str, Any]:
    records, summary = evaluate_frames(runtime, runtime.frames, expected_valid=True)
    feature_vectors = {
        row_id: extract_visual_features(frame, runtime.feature_spec).tolist()
        for row_id, frame in runtime.frames.items()
    }
    return {
        "summary": summary,
        "distinct_feature_vectors": len(
            {tuple(vector) for vector in feature_vectors.values()}
        ),
        "feature_count": runtime.feature_spec.feature_count,
        "records": records,
        "golden_feature_vectors": feature_vectors,
    }


def trajectory_results(runtime: Runtime, *, exhaustive: bool) -> dict[str, Any]:
    total_waves = 0
    cleared = 0
    decisions = 0
    failures: list[dict[str, Any]] = []
    waves: Iterable[tuple[int, ...]]
    if exhaustive:
        waves = product(range(runtime.config.width), repeat=len(runtime.config.wave))
    else:
        waves = [runtime.config.wave]
    for wave in waves:
        config = ShooterConfig(
            width=runtime.config.width,
            wave=tuple(int(value) for value in wave),
            max_steps=runtime.config.max_steps,
        )
        game = TinyArcadeShooter(config)
        wave_ok = True
        while not game.done:
            frame = render_state_frame(
                game.tank_x,
                game.target_x,
                game.cooldown,
                width=config.width,
            )
            row_id = game.row_id()
            visual = runtime.reader.read(frame)
            symbolic_action = runtime.policy_lookup.choose(row_id)
            if (
                not visual.accepted
                or visual.action != symbolic_action
                or visual.matched_row_id != row_id
            ):
                wave_ok = False
                failures.append(
                    {
                        "wave": list(wave),
                        "row_id": row_id,
                        "visual": visual.to_dict(),
                        "symbolic_action": symbolic_action,
                    }
                )
                break
            game.step(str(visual.action))
            decisions += 1
        total_waves += 1
        if wave_ok and game.cleared and game.score == len(wave):
            cleared += 1
    return {
        "mode": "exhaustive" if exhaustive else "default_wave",
        "waves_evaluated": total_waves,
        "waves_cleared": cleared,
        "visual_decisions_compared": decisions,
        "failures": failures[:10],
    }


def rejection_results(runtime: Runtime) -> dict[str, Any]:
    blank = np.zeros((FRAME_HEIGHT, runtime.config.width * CELL_PIXELS), dtype=np.uint8)
    maximum = np.full_like(blank, 255)
    random_frame = np.random.default_rng(0).integers(
        0, 256, size=blank.shape, dtype=np.uint8
    )
    corrupted = dict(runtime.frames)
    first_row = next(iter(corrupted))
    named_corruptions = {
        "blank": blank,
        "uniform_maximum": maximum,
        "random_noise": random_frame,
        "removed_target_element": remove_target(runtime.frames[first_row], first_row),
        "tank_corruption": corrupt_tank(runtime.frames[first_row], first_row),
    }
    records = []
    for name, frame in named_corruptions.items():
        try:
            decision = runtime.reader.read(frame)
            records.append({"name": name, **decision.to_dict()})
        except Exception as exc:  # noqa: BLE001 - invalid input evidence
            records.append(
                {
                    "name": name,
                    "accepted": False,
                    "reason": f"exception:{type(exc).__name__}:{exc}",
                }
            )
    return {
        "fixtures": records,
        "accepted_count": sum(int(item.get("accepted", False)) for item in records),
        "rejected_count": sum(int(not item.get("accepted", False)) for item in records),
    }


def identity_results(runtime: Runtime, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.vpm"
    index_path = output_dir / "visual-index.vpm"
    to_bundle(runtime.policy, policy_path)
    to_bundle(runtime.visual_build.artifact, index_path)
    restored_index = from_bundle(index_path)
    restored_reader = VisualSignReader(
        restored_index,
        runtime.policy,
        action_metric_ids=ACTIONS,
        value_source="raw",
        tie_break="metric_order",
    )
    restored_records, restored_summary = evaluate_frames(
        Runtime(
            runtime.config,
            runtime.policy,
            runtime.frames,
            runtime.feature_spec,
            runtime.visual_build,
            restored_reader,
            runtime.policy_lookup,
        ),
        runtime.frames,
        expected_valid=True,
    )
    mismatch_rejected = False
    mismatch_error = ""
    other_policy = build_vpm(
        runtime.policy.source,
        runtime.policy.recipe,
        provenance={
            "kind": "compiled_policy",
            "consumer": "VPMPolicyLookup",
            "compile_time_intelligence": "identity_mismatch_probe",
        },
    )
    try:
        VisualSignReader(restored_index, other_policy, action_metric_ids=ACTIONS)
    except Exception as exc:  # noqa: BLE001 - audit evidence
        mismatch_rejected = True
        mismatch_error = f"{type(exc).__name__}: {exc}"
    duplicate_rejected = False
    duplicate_error = ""
    frames = dict(runtime.frames)
    keys = list(frames)
    frames[keys[1]] = frames[keys[0]]
    try:
        build_visual_index(runtime.policy, frames, runtime.feature_spec)
    except Exception as exc:  # noqa: BLE001
        duplicate_rejected = True
        duplicate_error = f"{type(exc).__name__}: {exc}"
    return {
        "policy_artifact_id": runtime.policy.artifact_id,
        "visual_index_artifact_id": runtime.visual_build.artifact.artifact_id,
        "restored_visual_index_artifact_id": restored_index.artifact_id,
        "restored_canonical_summary": restored_summary,
        "restored_record_count": len(restored_records),
        "policy_bundle_sha256": sha256_file(policy_path),
        "visual_index_bundle_sha256": sha256_file(index_path),
        "wrong_policy_identity_rejected": mismatch_rejected,
        "wrong_policy_identity_error": mismatch_error,
        "duplicate_feature_rejected": duplicate_rejected,
        "duplicate_feature_error": duplicate_error,
        "metadata_addresses_policy_artifact_id": runtime.visual_build.artifact.source.metadata[
            "addresses_policy_artifact_id"
        ],
        "provenance_parents": runtime.visual_build.artifact.provenance["parents"],
    }


def _copy(frame: np.ndarray) -> np.ndarray:
    return np.array(frame, dtype=np.uint8, copy=True)


def _clip_int(frame: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def one_pixel(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    out[0, 0] = np.uint8(255 - int(out[0, 0]))
    return out


def small_block(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    out[0:2, 0:2] = 255
    return out


def remove_target(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    _tank, target, _cooldown = parse_state_row_id(row_id)
    if target is not None:
        centre = int(target) * CELL_PIXELS + CELL_PIXELS // 2
        out[2:5, centre - 1 : centre + 2] = 0
    return out


def extra_target(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    tank, target, _cooldown = parse_state_row_id(row_id)
    column = 0 if target not in {0, None} else runtime_width() - 1
    if column == tank:
        column = (column + 1) % runtime_width()
    centre = int(column) * CELL_PIXELS + CELL_PIXELS // 2
    out[2:4, centre - 1 : centre + 2] = TARGET_VALUE
    out[4, centre] = TARGET_VALUE
    return out


def runtime_width() -> int:
    return ShooterConfig().width


def corrupt_cooldown(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    _tank, _target, cooldown = parse_state_row_id(row_id)
    out[7:9, -3:-1] = COOLDOWN_READY_VALUE if cooldown else COOLDOWN_BLOCKED_VALUE
    return out


def corrupt_tank(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    tank, _target, _cooldown = parse_state_row_id(row_id)
    centre = int(tank) * CELL_PIXELS + CELL_PIXELS // 2
    out[11:14, max(0, centre - 2) : centre + 3] = 0
    return out


def translate_one(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = np.zeros_like(frame)
    out[:, 1:] = frame[:, :-1]
    return out


def crop_pad(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = np.zeros_like(frame)
    out[:-1, :-1] = frame[1:, 1:]
    return out


def resize_down_up(frame: np.ndarray, row_id: str) -> np.ndarray:
    down = frame[::2, ::2]
    return np.repeat(np.repeat(down, 2, axis=0), 2, axis=1)[
        : frame.shape[0], : frame.shape[1]
    ]


def sparse_salt(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    seed = int(hashlib.sha256(("salt:" + row_id).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    for _ in range(4):
        out[rng.randrange(out.shape[0]), rng.randrange(out.shape[1])] = 255
    return out


def sparse_pepper(frame: np.ndarray, row_id: str) -> np.ndarray:
    out = _copy(frame)
    seed = int(
        hashlib.sha256(("pepper:" + row_id).encode("utf-8")).hexdigest()[:8],
        16,
    )
    rng = random.Random(seed)
    for _ in range(4):
        out[rng.randrange(out.shape[0]), rng.randrange(out.shape[1])] = 0
    return out


def bounded_noise(frame: np.ndarray, row_id: str) -> np.ndarray:
    seed = int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    noise = rng.integers(-8, 9, size=frame.shape)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def midpoint_between_closest(runtime: Runtime) -> dict[str, np.ndarray]:
    left, right = runtime.visual_build.calibration.closest_pair_row_ids
    return {
        left: (
            (
                runtime.frames[left].astype(np.uint16)
                + runtime.frames[right].astype(np.uint16)
            )
            // 2
        ).astype(np.uint8),
        right: (
            (
                runtime.frames[left].astype(np.uint16)
                + runtime.frames[right].astype(np.uint16)
            )
            // 2
        ).astype(np.uint8),
    }


def partial_combination(runtime: Runtime) -> dict[str, np.ndarray]:
    keys = list(runtime.frames)
    frames = {}
    for index, row_id in enumerate(keys):
        other = keys[(index + 1) % len(keys)]
        out = _copy(runtime.frames[row_id])
        out[: FRAME_HEIGHT // 2] = runtime.frames[other][: FRAME_HEIGHT // 2]
        frames[row_id] = out
    return frames


def invalid_input_results(runtime: Runtime) -> dict[str, Any]:
    good = next(iter(runtime.frames.values()))
    cases: dict[str, Any] = {
        "blank": np.zeros_like(good),
        "uniform_maximum": np.full_like(good, 255),
        "random_noise": np.random.default_rng(1).integers(
            0, 256, size=good.shape, dtype=np.uint8
        ),
        "wrong_shape": np.zeros((good.shape[0] + 1, good.shape[1]), dtype=np.uint8),
        "wrong_dtype_float": good.astype(np.float32),
        "out_of_range_integer": good.astype(np.int16) + 300,
        "non_finite_float": np.full(good.shape, np.nan, dtype=np.float64),
    }
    rows = {}
    for name, frame in cases.items():
        try:
            rows[name] = runtime.reader.read(frame).to_dict()
        except Exception as exc:  # noqa: BLE001
            rows[name] = {
                "accepted": False,
                "reason": f"exception:{type(exc).__name__}:{exc}",
            }
    return rows


def perturbation_matrix(runtime: Runtime) -> dict[str, Any]:
    families: dict[str, Mapping[str, np.ndarray]] = {}
    transforms: dict[str, Callable[[np.ndarray, str], np.ndarray]] = {
        "small_global_darkening": lambda f, r: _clip_int(f, -8),
        "small_global_brightening": lambda f, r: _clip_int(f, 8),
        "larger_darkening": lambda f, r: _clip_int(f, -40),
        "larger_brightening": lambda f, r: _clip_int(f, 40),
        "one_pixel_change": one_pixel,
        "small_block_corruption": small_block,
        "removed_target_element": remove_target,
        "extra_target_like_element": extra_target,
        "cooldown_indicator_corruption": corrupt_cooldown,
        "tank_corruption": corrupt_tank,
        "one_pixel_translation": translate_one,
        "small_crop_with_padding": crop_pad,
        "nearest_resize_down_up": resize_down_up,
        "sparse_salt_noise": sparse_salt,
        "sparse_pepper_noise": sparse_pepper,
        "bounded_integer_noise": bounded_noise,
    }
    for name, transform in transforms.items():
        families[name] = {
            row_id: transform(frame, row_id) for row_id, frame in runtime.frames.items()
        }
    families["feature_or_image_midpoint_between_closest_states"] = (
        midpoint_between_closest(runtime)
    )
    families["partial_combination_of_two_valid_states"] = partial_combination(runtime)

    summaries = {}
    for name, frames in families.items():
        records, summary = evaluate_frames(runtime, frames, expected_valid=True)
        summaries[name] = {"summary": summary, "sample_records": records[:10]}
    summaries["clearly_invalid_inputs"] = {
        "cases": invalid_input_results(runtime),
    }
    return summaries


def calibration_comparison(runtime: Runtime) -> dict[str, Any]:
    canonical_vectors = np.asarray(
        [
            extract_visual_features(runtime.frames[row_id], runtime.feature_spec)
            for row_id in runtime.policy.source.row_ids
        ],
        dtype=np.float64,
    )
    row_ids = tuple(runtime.policy.source.row_ids)
    pairwise = np.sqrt(
        np.maximum(
            np.sum(canonical_vectors**2, axis=1)[:, None]
            + np.sum(canonical_vectors**2, axis=1)[None, :]
            - 2.0 * (canonical_vectors @ canonical_vectors.T),
            0.0,
        )
    )
    np.fill_diagonal(pairwise, np.inf)
    per_row = pairwise.min(axis=1)
    global_min = float(per_row.min())
    return {
        "existing_global_rule": {
            "threshold_fraction": runtime.visual_build.calibration.threshold_fraction,
            "margin_fraction": runtime.visual_build.calibration.margin_fraction,
            "min_between_distance": global_min,
            "acceptance_threshold": runtime.visual_build.calibration.acceptance_threshold,
            "required_margin": runtime.visual_build.calibration.required_margin,
            "canonical_acceptance": canonical_state_results(runtime)["summary"],
        },
        "candidate_local_rule": {
            "per_row_min_distance": {
                row_id: float(value) for row_id, value in zip(row_ids, per_row)
            },
            "radius_fraction": 0.25,
            "margin_fraction": 0.75,
            "interpretation": (
                "Measured only; not adopted. Local radii are larger for isolated "
                "states but add artifact and trace complexity."
            ),
        },
        "optional_ratio_rule": {
            "ratio": "nearest_distance / second_nearest_distance",
            "interpretation": "Measured only; exact states have ratio 0.0.",
        },
    }


def performance_results(runtime: Runtime) -> dict[str, Any]:
    rows = list(runtime.frames.items())
    start = time.perf_counter()
    for _ in range(100):
        for _row_id, frame in rows:
            runtime.reader.read(frame)
    lookup_seconds = time.perf_counter() - start
    index_path = RESULT_ROOT / "hardened" / "visual-index.vpm"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    to_bundle(runtime.visual_build.artifact, index_path)
    return {
        "iterations": 100 * len(rows),
        "total_lookup_seconds": lookup_seconds,
        "mean_lookup_seconds": lookup_seconds / (100 * len(rows)),
        "visual_index_bundle_bytes": index_path.stat().st_size,
        "visual_index_bundle_sha256": sha256_file(index_path),
    }


def environment() -> dict[str, Any]:
    import importlib.metadata
    import zeromodel.core
    import zeromodel.vision
    import zeromodel.video

    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "zeromodel_core_file": zeromodel.core.__file__,
        "zeromodel_vision_file": zeromodel.vision.__file__,
        "zeromodel_video_file": zeromodel.video.__file__,
        "versions": {
            "zeromodel": importlib.metadata.version("zeromodel"),
            "zeromodel-vision": importlib.metadata.version("zeromodel-vision"),
            "zeromodel-video": importlib.metadata.version("zeromodel-video"),
            "numpy": importlib.metadata.version("numpy"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--phase", choices=["baseline", "hardened"], default="baseline")
    args = parser.parse_args()

    root = args.output_root
    phase_dir = root / args.phase
    runtime = make_runtime()

    write_json(root / "environment.json", environment())
    write_json(
        phase_dir / "canonical-state-results.json", canonical_state_results(runtime)
    )
    write_json(
        phase_dir / "trajectory-results.json",
        trajectory_results(runtime, exhaustive=args.exhaustive),
    )
    write_json(phase_dir / "rejection-results.json", rejection_results(runtime))
    write_json(
        phase_dir / "identity-results.json", identity_results(runtime, phase_dir)
    )
    write_json(root / "perturbation-matrix.json", perturbation_matrix(runtime))
    write_json(root / "calibration-comparison.json", calibration_comparison(runtime))
    write_json(root / "performance-results.json", performance_results(runtime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
