from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visual_transition_benchmark.alias_discovery._json import file_digest
from visual_transition_benchmark.alias_discovery.corpus import ReaderContext, VisualAliasCase


def _save(path: Path, array: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.uint8)
    if arr.ndim == 3:
        Image.fromarray(arr, mode="RGB").save(path)
    else:
        Image.fromarray(arr, mode="L").save(path)
    return {"path": str(path), "file_digest": file_digest(path)}


def write_atlas(
    output_dir: Path,
    *,
    cases: list[VisualAliasCase],
    observations: dict[str, np.ndarray],
    context: ReaderContext,
) -> dict[str, object]:
    atlas_dir = output_dir / "failure-atlas"
    wrong = [
        case
        for case in cases
        if case.policy_executed and case.matched_row_id != case.source_row_id
    ]
    selected = wrong[:20]
    if not selected:
        selected = sorted(
            cases,
            key=lambda case: (case.distance_margin, -case.changed_pixel_fraction),
        )[:20]
    entries = []
    for case in selected:
        case_dir = atlas_dir / case.case_id.replace("sha256:", "")
        source = context.frames_by_row_id[case.source_row_id]
        transformed = observations[case.case_id]
        transformed_gray = transformed[:, :, 0] if transformed.ndim == 3 else transformed
        diff = np.abs(source.astype(np.int16) - transformed_gray.astype(np.int16)).astype(np.uint8)
        entries.append(
            {
                **case.to_dict(),
                "source_image": _save(case_dir / "source.png", source),
                "transformed_image": _save(case_dir / "transformed.png", transformed),
                "difference_image": _save(case_dir / "difference.png", diff),
            }
        )
    return {"selected_case_count": len(entries), "accepted_wrong_row_case_count": len(wrong), "entries": entries}
