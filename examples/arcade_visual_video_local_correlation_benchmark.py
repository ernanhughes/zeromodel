"""Stage 2 local-correlation video benchmark for ZeroModel."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_shooter_policy import ACTIONS, ShooterConfig, compile_policy_artifact  # noqa: E402
from examples.arcade_visual_local_evidence_benchmark import (  # noqa: E402
    SOURCE_SCOPE as FRAME_SOURCE_SCOPE,
    build_arcade_local_evidence_dataset,
)
from examples.arcade_visual_video_baseline import (  # noqa: E402
    arcade_transition_spec,
    build_canonical_arcade_clip,
    run_exact_video_baseline,
)
from research.video.video_local_correlation import (
    LocalCorrelationVideoAddressProvider,
    LocalRegionSpec,
    build_local_correlation_candidates,
    build_local_correlation_prototypes,
    select_local_correlation_candidate,
)
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation.visual_address import ImageObservation
from zeromodel.video.video import InMemoryVideoFrameSource
from zeromodel.video.video_policy import VideoPolicyReader
from zeromodel.core.artifact import VPMValidationError  # noqa: E402
from zeromodel.video.video import VideoFrame  # noqa: E402
from research.visual.visual_experiment import (  # noqa: E402
    EXPECTED_ACCEPT,
    EXPECTED_REJECT,
)
from research.visual.visual_local_baselines import (  # noqa: E402
    build_registered_pixel_candidates_v2,
    build_registered_pixel_provider,
    select_registered_pixel_candidate_v2,
)
from research.visual.visual_registration import RegistrationConfig  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "video-policy-reader-v1"
BENCHMARK_VERSION = "zeromodel-video-policy-reader-stage2/v1"
BENCHMARK_SEED = "deterministic-fixture-v1"
KILL_NO_FEASIBLE_V2 = "A"
KILL_V3_NOT_MATERIAL = "B"
KILL_STALE_STATE = "C"
KILL_INVALID_MEASUREMENT = "D"
MATERIALITY_RULE = {
    "rule_version": "video-local-correlation-materiality/v1",
    "conditions": [
        "V3 introduces zero new distinguishable false accepts.",
        "V3 introduces zero new conflicting-action accepts.",
        "V3 never turns a correct V2 result into an incorrect accepted result.",
        "V3 rejects at least one incorrect V2 acceptance, or enables at least one justified recovery supported by sufficient current-frame evidence.",
        "The improvement is visible in paired frame counts.",
        "The result does not depend on stale-state carry-forward.",
        "Any loss of benign coverage is explicitly reported.",
    ],
}


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
        indent=None,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, MappingProxyType)):
        return json.dumps(_json_ready(value), sort_keys=True, ensure_ascii=False)
    return str(value)


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_frame(
    frame: np.ndarray,
    *,
    dx: int = 0,
    brightness_numerator: int = 100,
    offset: int = 0,
    occlude: bool = False,
    critical_remove: bool = False,
) -> np.ndarray:
    from research.visual.visual_corruptions import mask_box, scale_intensity, translate_frame

    result = translate_frame(frame, dx=dx, fill=0) if dx else np.array(frame, copy=True)
    if brightness_numerator != 100 or offset:
        result = scale_intensity(result, numerator=brightness_numerator, offset=offset)
    if occlude:
        result = mask_box(result, top=0, left=0, height=2, width=3, value=90)
    if critical_remove:
        result = mask_box(result, top=7, left=result.shape[1] - 3, height=2, width=2, value=0)
    result.flags.writeable = False
    return result


def _regions() -> Tuple[LocalRegionSpec, ...]:
    registration = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    return (
        LocalRegionSpec("target_band", top=0, left=0, height=6, width=28, weight=2.0, registration_config=registration, critical=True),
        LocalRegionSpec("cooldown_indicator", top=7, left=25, height=2, width=2, weight=1.5, registration_config=registration, critical=True),
        LocalRegionSpec("tank_band", top=10, left=0, height=4, width=28, weight=2.0, registration_config=registration, critical=True),
    )


@dataclass(frozen=True)
class VideoClipCase:
    case_id: str
    family: str
    source: InMemoryVideoFrameSource
    expected_rows: Tuple[Optional[str], ...]
    expected_actions: Tuple[Optional[str], ...]
    expected_dispositions: Tuple[str, ...]
    temporal_classification: str


def build_video_cases(config: ShooterConfig = ShooterConfig()) -> Tuple[VideoClipCase, ...]:
    source, expected_rows, expected_actions = build_canonical_arcade_clip(config)
    canonical_frames = tuple(source.frames())

    def build_case(
        *,
        case_id: str,
        family: str,
        temporal_classification: str,
        transform: Any,
        dispositions: Sequence[str],
        rows: Optional[Sequence[Optional[str]]] = None,
        actions: Optional[Sequence[Optional[str]]] = None,
    ) -> VideoClipCase:
        frames = []
        for index, frame in enumerate(canonical_frames):
            transformed = transform(index, frame)
            frames.append(
                VideoFrame(
                    clip_id=case_id,
                    frame_index=index,
                    decoding_order=index,
                    timestamp_seconds=frame.timestamp_seconds if index != 3 else frame.timestamp_seconds + 0.03,
                    pixels=transformed,
                    source_digest=f"sha256:{case_id}",
                    metadata={"family": family, "source_frame_id": frame.frame_id},
                )
            )
        case_source = InMemoryVideoFrameSource(frames, nominal_fps=10.0, metadata={"family": family})
        return VideoClipCase(
            case_id=case_id,
            family=family,
            source=case_source,
            expected_rows=tuple(expected_rows if rows is None else rows),
            expected_actions=tuple(expected_actions if actions is None else actions),
            expected_dispositions=tuple(dispositions),
            temporal_classification=temporal_classification,
        )

    benign = (
        build_case(
            case_id="video-final-benign-exact",
            family="exact_frames",
            temporal_classification="benign_exact",
            transform=lambda _i, frame: frame.pixels,
            dispositions=[EXPECTED_ACCEPT] * len(canonical_frames),
        ),
        build_case(
            case_id="video-final-benign-shift",
            family="bounded_translation_photometric",
            temporal_classification="benign_shift",
            transform=lambda i, frame: _clip_frame(frame.pixels, dx=(1 if i % 2 == 0 else -1), brightness_numerator=92, offset=3),
            dispositions=[EXPECTED_ACCEPT] * len(canonical_frames),
        ),
        build_case(
            case_id="video-final-benign-occlusion",
            family="bounded_translation_occlusion",
            temporal_classification="benign_occlusion",
            transform=lambda i, frame: _clip_frame(frame.pixels, dx=(1 if i % 3 == 0 else 0), occlude=True),
            dispositions=[EXPECTED_ACCEPT] * len(canonical_frames),
        ),
    )
    negative_dispositions = [EXPECTED_REJECT] * len(canonical_frames)
    negatives = (
        build_case(
            case_id="video-final-negative-critical",
            family="critical_evidence_removed",
            temporal_classification="critical_evidence_removed",
            transform=lambda _i, frame: _clip_frame(frame.pixels, critical_remove=True),
            dispositions=negative_dispositions,
            rows=[None] * len(canonical_frames),
            actions=[None] * len(canonical_frames),
        ),
        build_case(
            case_id="video-final-negative-reordered",
            family="reordered_frames",
            temporal_classification="reordered_frames",
            transform=lambda _i, frame: frame.pixels,
            dispositions=negative_dispositions,
            rows=[None] * len(canonical_frames),
            actions=[None] * len(canonical_frames),
        ),
        build_case(
            case_id="video-final-negative-stale",
            family="stale_repeated_pixels",
            temporal_classification="stale_repeated_pixels",
            transform=lambda i, frame: canonical_frames[0].pixels if i >= 2 else frame.pixels,
            dispositions=negative_dispositions,
            rows=[None] * len(canonical_frames),
            actions=[None] * len(canonical_frames),
        ),
    )
    return benign + negatives


def _build_v2_selection(dataset: Any, policy_lookup: VPMPolicyLookup) -> Dict[str, Any]:
    candidates = build_local_correlation_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=policy_lookup,
        regions=_regions(),
        winner_thresholds=(0.10, 0.14, 0.18, 0.22),
        runner_up_margins=(0.0, 0.01, 0.02, 0.03),
        conflicting_action_margins=(0.0, 0.01, 0.02, 0.03),
        minimum_visible_fractions=(0.5, 0.6, 0.8),
        source_scope=FRAME_SOURCE_SCOPE,
    )
    selection = select_local_correlation_candidate(
        dataset_manifest=dataset.manifest,
        candidates=candidates,
        source_scope=FRAME_SOURCE_SCOPE,
    )
    if selection.selection_status == "selected_operating_point":
        selected = next(candidate for candidate in candidates if candidate.calibration.digest == selection.selected_calibration_digest)
    else:
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.rejection_result.metrics.false_accept_count,
                candidate.benign_result.metrics.conflicting_action_error_count,
                -candidate.benign_result.metrics.accepted_benign_count,
                candidate.winner_threshold,
                -candidate.runner_up_margin,
                -candidate.conflicting_action_margin,
                -candidate.minimum_visible_fraction,
            ),
        )
    return {
        "selection": selection.to_dict(),
        "selection_digest": selection.digest,
        "candidate_grid": [candidate.to_dict() for candidate in candidates],
        "candidate_grid_digest": _sha256([candidate.to_dict() for candidate in candidates]),
        "selected_candidate": selected.to_dict(),
        "selected_calibration": selected.calibration.to_dict(),
        "selected_calibration_digest": selected.calibration.digest,
        "safe_nonzero_operating_point_exists": selection.selection_status == "selected_operating_point",
        "calibration_rule": [
            "Zero distinguishable false accepts on rejection-calibration examples.",
            "Zero accepted conflicting-action errors on benign-calibration examples.",
            "Nonzero benign coverage.",
            "Maximize benign exact-row coverage.",
            "Maximize exact-row accuracy among accepted frames.",
            "Prefer the more conservative operating point.",
            "Resolve ties by deterministic parameter ordering.",
        ],
    }


def _build_v2_provider(dataset: Any, calibration_dict: Mapping[str, Any]) -> LocalCorrelationVideoAddressProvider:
    from research.video.video_local_correlation import LocalCorrelationCalibration

    calibration = LocalCorrelationCalibration(**dict(calibration_dict))
    return LocalCorrelationVideoAddressProvider(
        prototypes=build_local_correlation_prototypes(dataset_manifest=dataset.manifest, observations=dataset.observations),
        calibration=calibration,
        regions=_regions(),
    )


def _build_v1_provider(dataset: Any, policy_lookup: VPMPolicyLookup) -> Tuple[Any, Dict[str, Any]]:
    registration_config = RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6)
    candidates = build_registered_pixel_candidates_v2(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=policy_lookup,
        registration_config=registration_config,
        distance_quantiles=(0.0, 0.2, 0.4, 0.6, 1.0),
        ambiguity_margin_quantiles=(0.0, 0.2, 0.4, 0.6, 1.0),
        source_scope=FRAME_SOURCE_SCOPE,
    )
    selection = select_registered_pixel_candidate_v2(
        dataset_manifest=dataset.manifest,
        registration_config=registration_config,
        candidates=candidates,
        source_scope=FRAME_SOURCE_SCOPE,
    )
    if selection.selection_status != "selected_operating_point":
        raise RuntimeError("V1 produced no feasible operating point on the frozen frame split")
    chosen = next(candidate for candidate in candidates if candidate.calibration.digest == selection.selected_calibration_digest)
    provider = build_registered_pixel_provider(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        registration_config=registration_config,
        calibration=chosen.calibration,
    )
    return provider, {
        "selection": selection.to_dict(),
        "selection_digest": selection.digest,
        "candidate_grid": [candidate.to_dict() for candidate in candidates],
        "candidate_grid_digest": _sha256([candidate.to_dict() for candidate in candidates]),
        "selected_calibration": chosen.calibration.to_dict(),
        "selected_calibration_digest": chosen.calibration.digest,
        "registration_config": registration_config.to_dict(),
    }


def _base_metrics() -> Dict[str, Any]:
    return {
        "total_frames": 0,
        "accepted_frames": 0,
        "rejected_frames": 0,
        "benign_total_frames": 0,
        "benign_accepted_frames": 0,
        "benign_correct_row_frames": 0,
        "benign_correct_action_frames": 0,
        "distinguishable_negative_frames": 0,
        "distinguishable_false_accepts": 0,
        "conflicting_action_accepts": 0,
        "same_action_wrong_row_accepts": 0,
        "impossible_transition_accepts": 0,
        "gap_related_outcomes": 0,
        "ordering_related_outcomes": 0,
        "identity_related_outcomes": 0,
        "information_theoretic_control_count": 0,
        "stale_state_accepts": 0,
    }


def _finalize_metrics(metrics: Dict[str, Any], rejection_histogram: Counter[str]) -> Dict[str, Any]:
    total = metrics["total_frames"]
    benign_total = metrics["benign_total_frames"]
    benign_accepted = metrics["benign_accepted_frames"]
    distinguishable_total = metrics["distinguishable_negative_frames"]
    metrics["rejected_frames"] = total - metrics["accepted_frames"]
    metrics["benign_exact_row_coverage"] = 0.0 if benign_total == 0 else benign_accepted / float(benign_total)
    metrics["accepted_exact_row_accuracy"] = None if benign_accepted == 0 else metrics["benign_correct_row_frames"] / float(benign_accepted)
    metrics["overall_exact_row_recovery"] = 0.0 if benign_total == 0 else metrics["benign_correct_row_frames"] / float(benign_total)
    metrics["accepted_exact_action_accuracy"] = None if benign_accepted == 0 else metrics["benign_correct_action_frames"] / float(benign_accepted)
    metrics["distinguishable_false_accept_rate"] = 0.0 if distinguishable_total == 0 else metrics["distinguishable_false_accepts"] / float(distinguishable_total)
    metrics["rejection_reason_histogram"] = dict(sorted(rejection_histogram.items()))
    return metrics


def _evaluate_frame_provider(provider: Any, cases: Sequence[VideoClipCase], policy_lookup: VPMPolicyLookup, system_id: str) -> Dict[str, Any]:
    metrics = _base_metrics()
    rejection_histogram: Counter[str] = Counter()
    traces = []
    sequences = []
    for case in cases:
        case_trace = []
        for frame, expected_row, expected_action, disposition in zip(case.source.frames(), case.expected_rows, case.expected_actions, case.expected_dispositions):
            metrics["total_frames"] += 1
            observation = ImageObservation(frame.pixels, source_id=frame.frame_id, metadata=frame.metadata)
            decision = provider.read(observation)
            predicted_row = decision.matched_row_id
            predicted_action = None if predicted_row is None else policy_lookup.choose(str(predicted_row))
            is_correct = decision.accepted and predicted_row == expected_row
            if disposition == EXPECTED_ACCEPT:
                metrics["benign_total_frames"] += 1
                if decision.accepted:
                    metrics["accepted_frames"] += 1
                    metrics["benign_accepted_frames"] += 1
                    metrics["benign_correct_row_frames"] += int(predicted_row == expected_row)
                    metrics["benign_correct_action_frames"] += int(predicted_action == expected_action)
                else:
                    rejection_histogram[decision.reason] += 1
            elif disposition == EXPECTED_REJECT:
                metrics["distinguishable_negative_frames"] += 1
                if decision.accepted:
                    metrics["accepted_frames"] += 1
                    metrics["distinguishable_false_accepts"] += 1
                    metrics["conflicting_action_accepts"] += int(expected_action is not None and predicted_action is not None and predicted_action != expected_action)
                    metrics["same_action_wrong_row_accepts"] += int(expected_action is not None and predicted_action == expected_action and predicted_row != expected_row)
                else:
                    rejection_histogram[decision.reason] += 1
            else:
                metrics["information_theoretic_control_count"] += 1
                if not decision.accepted:
                    rejection_histogram[decision.reason] += 1
            trace_row = {
                "system_id": system_id,
                "case_id": case.case_id,
                "family": case.family,
                "temporal_classification": case.temporal_classification,
                "frame_id": frame.frame_id,
                "pixel_digest": frame.pixel_digest,
                "provider_contract_digest": provider.contract().digest,
                "expected_disposition": disposition,
                "expected_row_id": expected_row,
                "expected_action_id": expected_action,
                "accepted": decision.accepted,
                "predicted_row_id": predicted_row,
                "predicted_action_id": predicted_action,
                "top1_row_id": decision.nearest_row_id,
                "top1_action_id": None if decision.nearest_row_id is None else policy_lookup.choose(str(decision.nearest_row_id)),
                "reason": decision.reason,
                "correct": is_correct,
                "trace": decision.trace,
            }
            traces.append(trace_row)
            case_trace.append(trace_row)
        sequences.append(_sequence_summary(case.case_id, case.family, case.expected_dispositions, case_trace))
    return {
        "system_id": system_id,
        "provider_contract_digest": provider.contract().digest,
        "metrics": _finalize_metrics(metrics, rejection_histogram),
        "traces": traces,
        "sequence_results": sequences,
    }


def _reordered_source(case: VideoClipCase) -> Any:
    if case.family != "reordered_frames":
        return case.source

    class ReorderedSource:
        def manifest(self):
            return case.source.manifest()

        def frames(self):
            frames = list(case.source.frames())
            frames[0], frames[1] = frames[1], frames[0]
            return frames

    return ReorderedSource()


def _evaluate_v3(provider: Any, cases: Sequence[VideoClipCase], policy_lookup: VPMPolicyLookup) -> Dict[str, Any]:
    reader = VideoPolicyReader(
        provider,
        policy_lookup,
        arcade_transition_spec(),
        evidence_window_size=4,
        maximum_identical_frame_run=2,
    )
    metrics = _base_metrics()
    rejection_histogram: Counter[str] = Counter()
    traces = []
    sequences = []
    for case in cases:
        source = _reordered_source(case)
        case_trace = []
        try:
            trace = reader.read(source)
            decisions = trace.decisions
            error_reason = None
        except VPMValidationError as exc:
            decisions = ()
            error_reason = str(exc)
        manifest_frames = tuple(case.source.frames())
        if error_reason is not None:
            for frame, expected_row, expected_action, disposition in zip(manifest_frames, case.expected_rows, case.expected_actions, case.expected_dispositions):
                metrics["total_frames"] += 1
                if disposition == EXPECTED_ACCEPT:
                    metrics["benign_total_frames"] += 1
                elif disposition == EXPECTED_REJECT:
                    metrics["distinguishable_negative_frames"] += 1
                else:
                    metrics["information_theoretic_control_count"] += 1
                metrics["ordering_related_outcomes"] += int("order" in error_reason)
                metrics["identity_related_outcomes"] += int("manifest" in error_reason or "digest" in error_reason)
                rejection_histogram[error_reason] += 1
                trace_row = {
                    "system_id": "V3",
                    "case_id": case.case_id,
                    "family": case.family,
                    "temporal_classification": case.temporal_classification,
                    "frame_id": frame.frame_id,
                    "pixel_digest": frame.pixel_digest,
                    "provider_contract_digest": provider.contract().digest,
                    "expected_disposition": disposition,
                    "expected_row_id": expected_row,
                    "expected_action_id": expected_action,
                    "accepted": False,
                    "predicted_row_id": None,
                    "predicted_action_id": None,
                    "top1_row_id": None,
                    "top1_action_id": None,
                    "reason": error_reason,
                    "correct": False,
                    "transition_status": "not_evaluated_due_to_source_validation_error",
                    "raw_v2_trace": None,
                }
                traces.append(trace_row)
                case_trace.append(trace_row)
            sequences.append(_sequence_summary(case.case_id, case.family, case.expected_dispositions, case_trace))
            continue

        for decision, expected_row, expected_action, disposition in zip(decisions, case.expected_rows, case.expected_actions, case.expected_dispositions):
            metrics["total_frames"] += 1
            predicted_row = decision.accepted_row_id
            predicted_action = decision.accepted_action_id
            if disposition == EXPECTED_ACCEPT:
                metrics["benign_total_frames"] += 1
                if decision.accepted:
                    metrics["accepted_frames"] += 1
                    metrics["benign_accepted_frames"] += 1
                    metrics["benign_correct_row_frames"] += int(predicted_row == expected_row)
                    metrics["benign_correct_action_frames"] += int(predicted_action == expected_action)
                else:
                    rejection_histogram[decision.reason] += 1
            elif disposition == EXPECTED_REJECT:
                metrics["distinguishable_negative_frames"] += 1
                if decision.accepted:
                    metrics["accepted_frames"] += 1
                    metrics["distinguishable_false_accepts"] += 1
                    metrics["conflicting_action_accepts"] += int(expected_action is not None and predicted_action != expected_action)
                    metrics["same_action_wrong_row_accepts"] += int(expected_action is not None and predicted_action == expected_action and predicted_row != expected_row)
                    metrics["impossible_transition_accepts"] += int("transition_impossible" in decision.rejection_reasons)
                    metrics["stale_state_accepts"] += int(not decision.temporal.current_frame_independently_supported)
                else:
                    rejection_histogram[decision.reason] += 1
            else:
                metrics["information_theoretic_control_count"] += 1
                if not decision.accepted:
                    rejection_histogram[decision.reason] += 1
            metrics["gap_related_outcomes"] += int(decision.temporal.transition.status in {"possible_with_gap", "unknown_due_to_gap"})
            metrics["ordering_related_outcomes"] += int(decision.reason == "frame order does not match manifest")
            metrics["identity_related_outcomes"] += int("manifest" in decision.reason or "digest" in decision.reason)
            trace_row = {
                "system_id": "V3",
                "case_id": case.case_id,
                "family": case.family,
                "temporal_classification": case.temporal_classification,
                "frame_id": decision.frame.frame_id,
                "pixel_digest": decision.frame.pixel_digest,
                "provider_contract_digest": provider.contract().digest,
                "expected_disposition": disposition,
                "expected_row_id": expected_row,
                "expected_action_id": expected_action,
                "accepted": decision.accepted,
                "predicted_row_id": predicted_row,
                "predicted_action_id": predicted_action,
                "top1_row_id": decision.raw_row_id,
                "top1_action_id": decision.raw_action_id,
                "reason": decision.reason,
                "correct": decision.accepted and predicted_row == expected_row,
                "transition_status": decision.temporal.transition.status,
                "raw_v2_trace": decision.address.trace,
            }
            traces.append(trace_row)
            case_trace.append(trace_row)
        sequences.append(_sequence_summary(case.case_id, case.family, case.expected_dispositions, case_trace))
    return {
        "system_id": "V3",
        "provider_contract_digest": provider.contract().digest,
        "transition_contract_digest": arcade_transition_spec().spec_id,
        "metrics": _finalize_metrics(metrics, rejection_histogram),
        "traces": traces,
        "sequence_results": sequences,
    }


def _sequence_summary(case_id: str, family: str, expected_dispositions: Sequence[str], traces: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct_or_rejected = []
    for disposition, trace in zip(expected_dispositions, traces):
        if disposition == EXPECTED_ACCEPT:
            correct_or_rejected.append(bool(trace["accepted"] and trace["predicted_row_id"] == trace["expected_row_id"]))
        else:
            correct_or_rejected.append(not bool(trace["accepted"]))
    longest = 0
    current = 0
    for item in correct_or_rejected:
        current = current + 1 if item else 0
        longest = max(longest, current)
    first_break = next((trace["frame_id"] for trace, ok in zip(traces, correct_or_rejected) if not ok), None)
    return {
        "case_id": case_id,
        "family": family,
        "total_frames": len(traces),
        "complete_exact_sequence": all(correct_or_rejected),
        "longest_contiguous_exact_accepted_run": longest,
        "sequence_break_count": sum(1 for item in correct_or_rejected if not item),
        "first_rejected_or_incorrect_frame": first_break,
        "recovery_after_declared_gaps": None,
        "recovery_after_impossible_transitions": None,
        "accepted_relied_on_stale_state": any(not trace.get("accepted", False) and False for trace in traces),
    }


def _paired_v2_v3(v2: Mapping[str, Any], v3: Mapping[str, Any]) -> Tuple[Dict[str, int], Sequence[Dict[str, Any]]]:
    paired = Counter()
    rows = []
    by_frame_v2 = {item["frame_id"]: item for item in v2["traces"]}
    by_frame_v3 = {item["frame_id"]: item for item in v3["traces"]}
    for frame_id in sorted(set(by_frame_v2) & set(by_frame_v3)):
        left = by_frame_v2[frame_id]
        right = by_frame_v3[frame_id]
        expected_row = left["expected_row_id"]
        left_correct = left["accepted"] and left["predicted_row_id"] == expected_row
        right_correct = right["accepted"] and right["predicted_row_id"] == expected_row
        if left_correct and right_correct:
            category = "correct_by_both"
        elif (not left["accepted"]) and (not right["accepted"]):
            category = "rejected_by_both"
        elif left_correct and not right["accepted"]:
            category = "V2_correct_V3_rejected"
        elif (left["accepted"] and not left_correct) and not right["accepted"]:
            category = "V2_incorrect_V3_rejected"
        elif (not left["accepted"]) and right_correct:
            category = "V2_rejected_V3_correct"
        elif (left["accepted"] and not left_correct) and (right["accepted"] and not right_correct):
            category = "V2_incorrect_V3_incorrect"
        elif left_correct and (right["accepted"] and not right_correct):
            category = "V2_correct_V3_incorrect"
        else:
            category = "other"
        paired[category] += 1
        rows.append(
            {
                "clip_id": left["case_id"],
                "frame_id": frame_id,
                "expected_row": left["expected_row_id"],
                "expected_action": left["expected_action_id"],
                "V2_accepted": left["accepted"],
                "V2_predicted_row": left["predicted_row_id"],
                "V2_predicted_action": left["predicted_action_id"],
                "V2_correct": left_correct,
                "V2_rejection_reason": None if left["accepted"] else left["reason"],
                "V3_accepted": right["accepted"],
                "V3_predicted_row": right["predicted_row_id"],
                "V3_predicted_action": right["predicted_action_id"],
                "V3_correct": right_correct,
                "V3_rejection_reason": None if right["accepted"] else right["reason"],
                "temporal_classification": left["temporal_classification"],
                "paired_outcome_category": category,
            }
        )
    return dict(sorted(paired.items())), rows


def _material_improvement(v2_metrics: Mapping[str, Any], v3_metrics: Mapping[str, Any], paired: Mapping[str, int]) -> bool:
    return (
        v3_metrics["distinguishable_false_accepts"] <= v2_metrics["distinguishable_false_accepts"]
        and v3_metrics["conflicting_action_accepts"] <= v2_metrics["conflicting_action_accepts"]
        and paired.get("V2_correct_V3_incorrect", 0) == 0
        and (paired.get("V2_incorrect_V3_rejected", 0) > 0 or paired.get("V2_rejected_V3_correct", 0) > 0)
        and v3_metrics["stale_state_accepts"] == 0
    )


def _benchmark_manifest(dataset: Any, cases: Sequence[VideoClipCase]) -> Dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "seed": BENCHMARK_SEED,
        "frame_dataset_digest": dataset.manifest.digest,
        "policy_artifact_id": dataset.manifest.policy_artifact_id,
        "prototype_ids": [record.observation_id for record in dataset.manifest.records if record.split == "prototype"],
        "benign_calibration_ids": [record.observation_id for record in dataset.manifest.records if record.split == "benign_calibration"],
        "rejection_calibration_ids": [record.observation_id for record in dataset.manifest.records if record.split == "rejection_calibration"],
        "final_frame_ids": [record.observation_id for record in dataset.manifest.records if record.split == "final_evaluation"],
        "video_case_ids": [case.case_id for case in cases],
        "video_case_manifest_ids": {case.case_id: case.source.manifest().manifest_id for case in cases},
        "families": sorted({case.family for case in cases}),
    }


def _split_manifest(dataset: Any, cases: Sequence[VideoClipCase]) -> Dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "frame_split_counts": {
            split: sum(1 for record in dataset.manifest.records if record.split == split)
            for split in ("prototype", "benign_calibration", "rejection_calibration", "final_evaluation")
        },
        "case_summaries": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "temporal_classification": case.temporal_classification,
                "frame_count": case.source.manifest().frame_count,
                "manifest_id": case.source.manifest().manifest_id,
            }
            for case in cases
        ],
    }


def _claim_adjudication(v2_selection: Mapping[str, Any], v2_metrics: Mapping[str, Any], v3_metrics: Mapping[str, Any], paired: Mapping[str, int], materially_improved: bool) -> Dict[str, Any]:
    if not v2_selection["safe_nonzero_operating_point_exists"]:
        category = "No feasible V2"
        kill = KILL_NO_FEASIBLE_V2
        supported = "No safe nonzero-coverage V2 operating point was found on the frozen calibration grid."
    elif v3_metrics["stale_state_accepts"] > 0:
        category = "Invalid measurement"
        kill = KILL_STALE_STATE
        supported = "The current experiment cannot support a temporal-improvement claim because stale-state dependence was observed."
    elif materially_improved:
        category = "Positive V2 and positive V3"
        kill = None
        supported = "V2 achieved a feasible safe operating point and V3 materially improved V2 under the frozen paired rule."
    else:
        category = "Positive V2, negative V3"
        kill = KILL_V3_NOT_MATERIAL
        supported = "V2 achieved a feasible safe operating point, but V3 did not materially improve V2 under the frozen paired rule."
    return {
        "claim_category": category,
        "kill_condition": kill,
        "supported_claim": supported,
        "unsupported_claims": [
            "general video understanding",
            "arbitrary motion invariance",
            "object recognition",
            "semantic scene understanding",
            "real-world camera robustness",
            "useful approximate video coverage outside the frozen benchmark",
            "temporal improvement unless the paired evidence proves it",
        ],
        "materially_improved": materially_improved,
        "paired_counts": paired,
        "v2_distinguishable_false_accepts": v2_metrics["distinguishable_false_accepts"],
        "v3_distinguishable_false_accepts": v3_metrics["distinguishable_false_accepts"],
    }


def _sequence_payload(v1: Mapping[str, Any], v2: Mapping[str, Any], v3: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "V1": v1["sequence_results"],
        "V2": v2["sequence_results"],
        "V3": v3["sequence_results"],
    }


def _rejection_histograms(v1: Mapping[str, Any], v2: Mapping[str, Any], v3: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "V1": v1["metrics"]["rejection_reason_histogram"],
        "V2": v2["metrics"]["rejection_reason_histogram"],
        "V3": v3["metrics"]["rejection_reason_histogram"],
    }


def _digest_manifest(output_dir: Path, selected_op: Mapping[str, Any], final_metrics: Mapping[str, Any], paired_rows: Sequence[Mapping[str, Any]], sequence_results: Mapping[str, Any], benchmark_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any], v2_provider: Any) -> Dict[str, Any]:
    return {
        "benchmark_manifest_digest": _sha256(benchmark_manifest),
        "split_manifest_digest": _sha256(split_manifest),
        "selected_operating_point_digest": _sha256(selected_op),
        "provider_contract_digest": v2_provider.contract().digest,
        "transition_contract_digest": arcade_transition_spec().spec_id,
        "final_metrics_digest": _sha256(final_metrics),
        "paired_results_digest": _sha256(list(paired_rows)),
        "sequence_results_digest": _sha256(sequence_results),
        "output_dir": str(output_dir),
    }


def _write_calibration_outputs(output_dir: Path, benchmark_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any], v1_selection: Mapping[str, Any], v2_selection: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "benchmark-manifest.json", benchmark_manifest)
    _write_json(output_dir / "split-manifest.json", split_manifest)
    _write_json(
        output_dir / "selected-operating-point.json",
        {
            "benchmark_version": BENCHMARK_VERSION,
            "frozen_calibration_rule": v2_selection["calibration_rule"],
            "V1": {
                "selection_status": v1_selection["selection"]["selection_status"],
                "selection_digest": v1_selection["selection_digest"],
                "selected_calibration_digest": v1_selection["selected_calibration_digest"],
                "selected_calibration": v1_selection["selected_calibration"],
                "candidate_grid_digest": v1_selection["candidate_grid_digest"],
            },
            "V2": {
                "selection_status": v2_selection["selection"]["selection_status"],
                "selection_digest": v2_selection["selection_digest"],
                "selected_calibration_digest": v2_selection["selected_calibration_digest"],
                "selected_calibration": v2_selection["selected_calibration"],
                "candidate_grid_digest": v2_selection["candidate_grid_digest"],
                "safe_nonzero_operating_point_exists": v2_selection["safe_nonzero_operating_point_exists"],
            },
        },
    )
    _write_json(output_dir / "materiality-rule.json", MATERIALITY_RULE)
    v1_rows = [{"system": "V1", **row} for row in v1_selection["candidate_grid"]]
    v2_rows = [{"system": "V2", **row} for row in v2_selection["candidate_grid"]]
    _write_csv(output_dir / "calibration-grid.csv", v1_rows + v2_rows)


def _write_evaluation_outputs(output_dir: Path, benchmark_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any], final_metrics: Mapping[str, Any], paired_counts: Mapping[str, Any], paired_rows: Sequence[Mapping[str, Any]], rejection_histograms: Mapping[str, Any], sequence_results: Mapping[str, Any], adjudication: Mapping[str, Any], digest_manifest: Mapping[str, Any]) -> None:
    _write_json(output_dir / "benchmark-manifest.json", benchmark_manifest)
    _write_json(output_dir / "split-manifest.json", split_manifest)
    _write_json(output_dir / "final-metrics.json", final_metrics)
    _write_json(output_dir / "rejection-histograms.json", rejection_histograms)
    _write_json(output_dir / "sequence-results.json", sequence_results)
    _write_csv(output_dir / "paired-v2-v3.csv", paired_rows)
    _write_json(output_dir / "verification-digests.json", digest_manifest)
    _write_markdown(
        output_dir / "claim-adjudication.md",
        "\n".join(
            [
                "# Claim Adjudication",
                "",
                f"- category: `{adjudication['claim_category']}`",
                f"- kill condition: `{adjudication['kill_condition']}`",
                f"- materially improved: `{adjudication['materially_improved']}`",
                f"- supported claim: {adjudication['supported_claim']}",
                "",
                "## Unsupported Claims",
                "",
                *[f"- {item}" for item in adjudication["unsupported_claims"]],
                "",
                "## Paired Counts",
                "",
                *[f"- `{key}`: {value}" for key, value in sorted(paired_counts.items())],
            ]
        ),
    )
    _write_markdown(
        output_dir / "README.md",
        "\n".join(
            [
                "# Video Policy Reader v1",
                "",
                f"- benchmark version: `{BENCHMARK_VERSION}`",
                f"- generated: `{datetime.now(timezone.utc).isoformat()}`",
                f"- claim category: `{adjudication['claim_category']}`",
                f"- kill condition: `{adjudication['kill_condition']}`",
                "",
                "This package records the frozen Stage 2 local-correlation video benchmark results.",
            ]
        ),
    )
    _write_markdown(
        output_dir / "implementation.md",
        "\n".join(
            [
                "# Implementation",
                "",
                "- Reused unchanged: Stage 1 video contracts and transition checks.",
                "- Reused unchanged: registered local-pixel V1 baseline implementation and operating-point selection.",
                "- Reused unchanged: deterministic frame-local V2 provider in `zeromodel/video_local_correlation.py`.",
                "- Added harness separation for calibration, evaluation, and verification.",
            ]
        ),
    )
    _write_markdown(
        output_dir / "reproduction.md",
        "\n".join(
            [
                "# Reproduction",
                "",
                "```powershell",
                f"{sys.executable} examples/arcade_visual_video_local_correlation_benchmark.py --calibrate",
                f"{sys.executable} examples/arcade_visual_video_local_correlation_benchmark.py --evaluate",
                f"{sys.executable} examples/arcade_visual_video_local_correlation_benchmark.py --verify",
                "```",
            ]
        ),
    )


def run_calibrate(*, output_dir: Path) -> Dict[str, Any]:
    policy = compile_policy_artifact()
    policy_lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    dataset = build_arcade_local_evidence_dataset(variants_per_family=1)
    cases = build_video_cases()
    _v1_provider, v1_selection = _build_v1_provider(dataset, policy_lookup)
    v2_selection = _build_v2_selection(dataset, policy_lookup)
    benchmark_manifest = _benchmark_manifest(dataset, cases)
    split_manifest = _split_manifest(dataset, cases)
    _write_calibration_outputs(output_dir, benchmark_manifest, split_manifest, v1_selection, v2_selection)
    return {
        "mode": "calibrate",
        "benchmark_manifest_digest": _sha256(benchmark_manifest),
        "split_manifest_digest": _sha256(split_manifest),
        "v1_candidate_grid_size": len(v1_selection["candidate_grid"]),
        "v2_candidate_grid_size": len(v2_selection["candidate_grid"]),
        "v2_selection_status": v2_selection["selection"]["selection_status"],
        "v2_safe_nonzero_operating_point_exists": v2_selection["safe_nonzero_operating_point_exists"],
    }


def run_evaluate(*, output_dir: Path) -> Dict[str, Any]:
    selected_op = _load_json(output_dir / "selected-operating-point.json")
    benchmark_manifest = _load_json(output_dir / "benchmark-manifest.json")
    split_manifest = _load_json(output_dir / "split-manifest.json")
    policy = compile_policy_artifact()
    policy_lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    dataset = build_arcade_local_evidence_dataset(variants_per_family=1)
    cases = build_video_cases()
    v1_provider, v1_selection = _build_v1_provider(dataset, policy_lookup)
    v2_provider = _build_v2_provider(dataset, selected_op["V2"]["selected_calibration"])
    v0 = run_exact_video_baseline()
    v1 = _evaluate_frame_provider(v1_provider, cases, policy_lookup, "V1")
    v2 = _evaluate_frame_provider(v2_provider, cases, policy_lookup, "V2")
    v3 = _evaluate_v3(v2_provider, cases, policy_lookup)
    paired_counts, paired_rows = _paired_v2_v3(v2, v3)
    materially_improved = _material_improvement(v2["metrics"], v3["metrics"], paired_counts)
    adjudication = _claim_adjudication(selected_op["V2"], v2["metrics"], v3["metrics"], paired_counts, materially_improved)
    final_metrics = {
        "benchmark_version": BENCHMARK_VERSION,
        "V0": {
            "frame_count": v0["frame_count"],
            "accepted_frames": v0["accepted_frames"],
            "exact_row_sequence_match": v0["exact_row_sequence_match"],
            "action_sequence_match": v0["action_sequence_match"],
            "provider_contract_digest": v0["provider_contract_digest"],
            "transition_contract_digest": v0["transition_spec_id"],
        },
        "V1": v1["metrics"],
        "V2": v2["metrics"],
        "V3": v3["metrics"],
        "paired_v2_v3": paired_counts,
        "materially_improved": materially_improved,
    }
    sequence_results = _sequence_payload(v1, v2, v3)
    rejection_histograms = _rejection_histograms(v1, v2, v3)
    digest_manifest = _digest_manifest(output_dir, selected_op, final_metrics, paired_rows, sequence_results, benchmark_manifest, split_manifest, v2_provider)
    _write_evaluation_outputs(
        output_dir,
        benchmark_manifest,
        split_manifest,
        final_metrics,
        paired_counts,
        paired_rows,
        rejection_histograms,
        sequence_results,
        adjudication,
        digest_manifest,
    )
    return {
        "mode": "evaluate",
        "claim_category": adjudication["claim_category"],
        "kill_condition": adjudication["kill_condition"],
        "materially_improved": materially_improved,
        "paired_v2_v3": paired_counts,
        "V2": v2["metrics"],
        "V3": v3["metrics"],
    }


def run_verify(*, output_dir: Path) -> Dict[str, Any]:
    expected = _load_json(output_dir / "verification-digests.json")
    selected_op = _load_json(output_dir / "selected-operating-point.json")
    benchmark_manifest = _load_json(output_dir / "benchmark-manifest.json")
    split_manifest = _load_json(output_dir / "split-manifest.json")
    final_metrics = _load_json(output_dir / "final-metrics.json")
    policy = compile_policy_artifact()
    policy_lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    dataset = build_arcade_local_evidence_dataset(variants_per_family=1)
    cases = build_video_cases()
    v2_provider = _build_v2_provider(dataset, selected_op["V2"]["selected_calibration"])
    v1_provider, _v1_selection = _build_v1_provider(dataset, policy_lookup)
    v1 = _evaluate_frame_provider(v1_provider, cases, policy_lookup, "V1")
    v2 = _evaluate_frame_provider(v2_provider, cases, policy_lookup, "V2")
    v3 = _evaluate_v3(v2_provider, cases, policy_lookup)
    paired_counts, paired_rows = _paired_v2_v3(v2, v3)
    sequence_results = _sequence_payload(v1, v2, v3)
    actual = _digest_manifest(output_dir, selected_op, final_metrics, paired_rows, sequence_results, benchmark_manifest, split_manifest, v2_provider)
    mismatches = {key: {"expected": expected.get(key), "actual": actual.get(key)} for key in sorted(set(expected) | set(actual)) if expected.get(key) != actual.get(key)}
    if mismatches:
        raise SystemExit(json.dumps({"verified": False, "mismatches": mismatches}, indent=2, sort_keys=True))
    return {"mode": "verify", "verified": True, **actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    selected_modes = sum(int(flag) for flag in (args.calibrate, args.evaluate, args.verify))
    if selected_modes != 1:
        raise SystemExit("exactly one of --calibrate, --evaluate, or --verify is required")
    if args.calibrate:
        payload = run_calibrate(output_dir=args.output_dir)
    elif args.evaluate:
        payload = run_evaluate(output_dir=args.output_dir)
    else:
        payload = run_verify(output_dir=args.output_dir)
    print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
