"""ZM-HAND-03a: run the existing ZM-HAND-01/02 comparison on real hand crops.

Representation changes (documented, separate from the frozen originals):

1. Forearm-free templates: crops rarely contain a wrist/forearm, so the template family is
   used with its forearm component removed (new identities; originals untouched).  The
   refinement stage drops the forearm too (``drop_forearm``, default off in ZM-HAND-02).
2. Rotation: crops have arbitrary in-plane orientation, so the observation mask is tried at
   the four quadrant rotations (0/90/180/270 deg) on top of the +/-45 deg alignment search;
   the rotation with the best library IoU is kept.  No label information is used.
3. Border contact is a reported covariate, not an exclusion (crops are cut at the border).

Eligibility is decided from input quality only (segmentation status, minimum mask size),
never from the method's correctness or the label.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import dataclasses
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from examples.hand_anatomy._common import UnsupportedInput, digest
from examples.hand_anatomy.observations import observation_from_arrays
from examples.hand_anatomy.pose_library import TemplateLibrary, build_library
from examples.hand_anatomy_02.inspect02 import inspect02, residual_overlay

ROTATIONS = (0, 1, 2, 3)  # np.rot90 quarter turns
VARIANT = "forearm-free/1"

_LIB: Optional[TemplateLibrary] = None


def forearm_free_library() -> TemplateLibrary:
    base = build_library()
    out = []
    for t in base.templates:
        labels = np.where(t.labels == 2, 0, t.labels).astype(np.uint8)
        tid = digest({"derived_from": t.template_id, "variant": VARIANT})
        out.append(dataclasses.replace(t, labels=labels, template_id=tid, render_id=tid,
                                       name=t.name + "/no-forearm"))
    return TemplateLibrary(out, base.recipe)


def _library() -> TemplateLibrary:
    global _LIB
    if _LIB is None:
        _LIB = forearm_free_library()
    return _LIB


def assess_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """job: name, label, image path, mask path, seg record, overlay dir (optional)."""
    rec: Dict[str, Any] = {"name": job["name"], "label": job["label"], "video": job.get("video"),
                           "seg_status": job["seg"]["status"], "seg_reasons": job["seg"]["reasons"],
                           "input_size": job["seg"]["input_size"], "mask_area_fraction": job["seg"]["mask_area_fraction"],
                           "mean_probability_inside_mask": job["seg"]["mean_probability_inside_mask"]}
    if job["seg"]["status"] == "failed":
        rec.update(eligible=False, ineligible_reason="segmentation_failed:" + ";".join(job["seg"]["reasons"]))
        return rec
    mask = np.asarray(Image.open(job["mask"])) > 127
    t0 = time.perf_counter()
    best = None
    try:
        for k in ROTATIONS:
            obs = observation_from_arrays(np.rot90(mask, k).copy(), method={"mask_source": "u2netp", "rot90": k})
            res = inspect02(obs, _library(), drop_forearm=True)
            key = (res.scores["A_iou"], k)
            if best is None or key < best[0]:
                best = (key, k, res)
    except UnsupportedInput as exc:
        rec.update(eligible=False, ineligible_reason="mask_rejected:" + str(exc))
        return rec
    assert best is not None
    _, k, res = best
    top = max(res.rows, key=lambda r: r["metrics"]["silhouette_agreement"])
    rec.update(
        eligible=True, rotation_quarter_turns=k, scores={k2: round(v, 6) for k2, v in res.scores.items()},
        best_library_iou=round(1 - res.scores["A_iou"], 4), best_refined_iou=round(1 - res.scores["A_refined_iou"], 4),
        border_contact_fraction=round(res.observation.border_contact_fraction, 4),
        best_template=top["name"], vpm_artifact_id=res.vpm.artifact_id,
        top_residual=res.residual_diag[res.explainer_template_id]["top_residual"],
        seconds=round(time.perf_counter() - t0, 2),
    )
    if job.get("overlay_dir"):
        d = Path(job["overlay_dir"])
        d.mkdir(parents=True, exist_ok=True)
        residual_overlay(res, 1).save(d / (Path(job["name"]).stem + ".compare.png"))
    return rec


def run_pilot(selection: List[Dict[str, Any]], image_dir: Path, seg_dir: Path, out_dir: Path,
              workers: int = 4) -> Dict[str, Any]:
    seg = json.loads((seg_dir / "segmentation.json").read_text(encoding="utf-8"))
    jobs = []
    for s in selection:
        stem = Path(s["name"]).stem
        jobs.append({"name": s["name"], "label": s["label"], "video": s["video"], "image": str(image_dir / s["name"]),
                     "mask": str(seg_dir / (stem + ".mask.png")), "seg": seg["images"][s["name"]],
                     "overlay_dir": str(out_dir / "compare")})
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(assess_one, jobs))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"segmentation": seg["meta"], "variant": VARIANT, "rotations": list(ROTATIONS),
              "wall_seconds": round(time.perf_counter() - t0, 1), "cases": records}
    (out_dir / "pilot_cases.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return result
