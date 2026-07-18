"""Stage 3 discriminative-evidence benchmark driver for ZeroModel."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_shooter_policy import ACTIONS, compile_policy_artifact  # noqa: E402
from examples.arcade_visual_local_evidence_benchmark import (  # noqa: E402
    build_arcade_local_evidence_dataset,
)
from examples.arcade_visual_video_local_correlation_benchmark import (  # noqa: E402
    _build_v2_provider,
    _build_v2_selection,
    _regions,
    build_video_cases,
)
from zeromodel import ImageObservation, VPMPolicyLookup  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "video-discriminative-local-evidence-v1"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
BENCHMARK_VERSION = "zeromodel-video-discriminative-evidence-stage3/v1"
STAGE2_PARENT_COMMIT = "d00e18b67fbe2f62617cd0ac47c7ee2f63487cb8"
STAGE2_BENCHMARK_DIGEST = "sha256:589bb074e1b53b06657cfb75bf7b8d67eae43cc5f76e7237ab07f23ccca49c75"
STAGE2_SPLIT_DIGEST = "sha256:d25b694b3cce93bf93f58239163331f3f6370d32a2b5cce53b4541902b0f8c23"


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


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
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


def _diagnose_stage2(output_dir: Path) -> Dict[str, Any]:
    policy = compile_policy_artifact()
    policy_lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    dataset = build_arcade_local_evidence_dataset(variants_per_family=1)
    selection = _build_v2_selection(dataset, policy_lookup)
    provider = _build_v2_provider(dataset, selection["selected_calibration"])
    cases = build_video_cases()
    prototypes = provider._items
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
        for frame, expected_row, expected_action, disposition in zip(case.source.frames(), case.expected_rows, case.expected_actions, case.expected_dispositions):
            observation = ImageObservation(frame.pixels, source_id=frame.frame_id, metadata=frame.metadata)
            ranked = provider._rank(observation)
            decision = provider.read(observation)
            family_rejections[(case.family, decision.reason)] += 1
            if expected_row is not None:
                matching_rank = next((idx for idx, candidate in enumerate(ranked, start=1) if candidate.row_id == expected_row), None)
                exact_rank_positions.append({"family": case.family, "frame_id": frame.frame_id, "rank": matching_rank})
                for k in (1, 2, 3, 5):
                    topk_hits[f"top{k}"] += int(matching_rank is not None and matching_rank <= k)
                same_action_rank = next(
                    (idx for idx, candidate in enumerate(ranked, start=1) if candidate.action_id == expected_action),
                    None,
                )
                same_action_rank_positions.append({"family": case.family, "frame_id": frame.frame_id, "rank": same_action_rank})
                for k in (1, 2, 3, 5):
                    topk_action_hits[f"top{k}"] += int(same_action_rank is not None and same_action_rank <= k)
                conflicting_rank = next(
                    (idx for idx, candidate in enumerate(ranked, start=1) if candidate.action_id != expected_action),
                    None,
                )
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
        "region_contract_digest": _sha256([region.to_dict() for region in _regions()]),
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
    return diagnostics


def run_diagnose_stage2(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = _diagnose_stage2(output_dir)
    return {
        "mode": "diagnose-stage2",
        "benchmark_version": BENCHMARK_VERSION,
        "diagnostic_summary_digest": _sha256(diagnostics),
        "selection_status": diagnostics["selection_status"],
        "region_contract_digest": diagnostics["region_contract_digest"],
        "diagnostic_row_count": diagnostics["diagnostic_row_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--diagnose-stage2", action="store_true")
    parser.add_argument("--select-architecture", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.diagnose_stage2:
        payload = run_diagnose_stage2(args.output_dir)
    elif args.select_architecture or args.calibrate or args.evaluate or args.verify:
        raise SystemExit("Stage 3 implementation is not complete yet; only --diagnose-stage2 is currently implemented on this branch state.")
    else:
        raise SystemExit("one stage flag is required")
    print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
