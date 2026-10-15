"""ZM-HAND-03b: automatic vs reference masks through the unchanged ZM-HAND-01/02 pipeline.

Decision criteria are in docs/research/ZM-HAND-03b-preregistration.md (frozen before running).
Reference masks (V) are agent-drawn and model-reviewed; they are NOT human-verified.

    python -m examples.hand_anatomy_03.exp03b
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .real_pipeline import assess_one

WORK = Path("artifacts/zm_hand_03/mask_work")
OUT = Path("artifacts/zm_hand_03/exp03b")
VENV_PY = Path("artifacts/zm_hand_03/venv_seg/Scripts/python.exe")
MODEL = Path("artifacts/zm_hand_03/seg/u2netp.onnx")
G1_BLOB_MARGIN, G1_HULL_MARGIN = 0.05, 0.03
G2_MIN_AUC, G2_BASELINE_MARGIN = 0.70, 0.10
ATTRIBUTION_DELTA = 0.15
MIN_ACCEPTED = 20
BOOT, SEED = 2000, 7
METHODS = ("A_iou", "A_refined_iou", "C_residual")


def hull_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    pts = sorted(set(zip(xs.tolist(), ys.tolist())))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    poly = lower[:-1] + upper[:-1]
    img = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(img).polygon(poly, fill=255)
    return np.asarray(img) > 127


def solidity(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(hull_mask(mask).sum())


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def boot_ci(pos: Sequence[float], neg: Sequence[float]) -> List[float]:
    rng = np.random.default_rng(SEED)
    pos, neg = np.asarray(pos), np.asarray(neg)
    vals = []
    for _ in range(BOOT):
        p = rng.choice(pos, len(pos))
        n = rng.choice(neg, len(neg))
        vals.append(auc(p, n))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def _job(fid: str, label: int, image: Path, mask: Path, status: Dict[str, Any], overlay: Path) -> Dict[str, Any]:
    return {"name": fid + ".png", "label": label, "video": None, "image": str(image), "mask": str(mask),
            "seg": status, "overlay_dir": str(overlay)}


def run(workers: int = 4, exploratory_all: bool = False) -> Dict[str, Any]:
    out_dir = OUT / "exploratory_all_masks" if exploratory_all else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = json.loads(Path("examples/hand_anatomy_03/data/sample_03b_labels.json").read_text(encoding="utf-8"))
    review = json.loads((WORK / "review_03b.json").read_text(encoding="utf-8"))
    accepted = sorted(review) if exploratory_all else sorted(f for f, v in review.items() if v["verdict"] == "accept")
    n_accepted_prereg = sum(1 for v in review.values() if v["verdict"] == "accept")
    images = WORK / "images"
    seg_dir = out_dir / "seg_auto"
    subprocess.run([str(VENV_PY.resolve()), str(Path(__file__).with_name("segment_u2netp.py")), "--images", str(images),
                    "--output", str(seg_dir), "--model", str(MODEL.resolve())], check=True, stdout=subprocess.DEVNULL)
    seg = json.loads((seg_dir / "segmentation.json").read_text(encoding="utf-8"))["images"]
    hull_dir = out_dir / "hull_masks"
    hull_dir.mkdir(exist_ok=True)
    jobs_a, jobs_v, jobs_h = [], [], []
    solid: Dict[str, float] = {}
    for fid in accepted:
        v = np.asarray(Image.open(WORK / "masks" / (fid + ".mask.png"))) > 127
        solid[fid] = solidity(v)
        h = hull_mask(v)
        Image.fromarray((h * 255).astype(np.uint8), "L").save(hull_dir / (fid + ".mask.png"))
        ok = {"status": "ok", "reasons": [], "input_size": list(v.shape[::-1]), "mask_area_fraction": float(v.mean()),
              "mean_probability_inside_mask": 1.0}
        hok = dict(ok, mask_area_fraction=float(h.mean()))
        jobs_a.append(_job(fid, labels[fid], images / (fid + ".png"), seg_dir / (fid + ".mask.png"), seg[fid + ".png"],
                           out_dir / "compare_auto"))
        jobs_v.append(_job(fid, labels[fid], images / (fid + ".png"), WORK / "masks" / (fid + ".mask.png"), ok,
                           out_dir / "compare_ref"))
        jobs_h.append(_job(fid, labels[fid], images / (fid + ".png"), hull_dir / (fid + ".mask.png"), hok,
                           out_dir / "compare_hull"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rec_a, rec_v, rec_h = (list(ex.map(assess_one, jobs)) for jobs in (jobs_a, jobs_v, jobs_h))
    # auto-vs-reference mask agreement
    agree = {}
    for fid in accepted:
        a = np.asarray(Image.open(seg_dir / (fid + ".mask.png"))) > 127
        v = np.asarray(Image.open(WORK / "masks" / (fid + ".mask.png"))) > 127
        agree[fid] = float((a & v).sum()) / max(1.0, float((a | v).sum()))
    blobs = json.loads(Path("artifacts/zm_hand_03/control/control_cases.json").read_text(encoding="utf-8"))
    res: Dict[str, Any] = {"mode": "EXPLORATORY_ALL_MASKS_NOT_PREREGISTERED" if exploratory_all else "preregistered", "accepted": n_accepted_prereg, "analysed": len(accepted), "reviewed": len(review), "wall_seconds": round(time.perf_counter() - t0, 1),
                           "rejected": {f: v for f, v in review.items() if v["verdict"] != "accept"},
                           "auto_vs_reference_mask_iou": {"median": round(statistics.median(agree.values()), 3),
                                                          "values": {k: round(v, 3) for k, v in agree.items()}}}

    def table(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        el = [r for r in recs if r["eligible"]]
        pos = [r for r in el if r["label"] == 1]
        neg = [r for r in el if r["label"] == 0]
        out: Dict[str, Any] = {"eligible": len(el), "abnormal": len(pos), "normal": len(neg),
                               "median_library_iou": round(statistics.median(r["best_library_iou"] for r in el), 4) if el else None,
                               "median_refined_iou": round(statistics.median(r["best_refined_iou"] for r in el), 4) if el else None,
                               "auc": {}}
        if pos and neg:
            for m in METHODS:
                p, n = [r["scores"][m] for r in pos], [r["scores"][m] for r in neg]
                out["auc"][m] = {"auc": round(auc(p, n), 3), "ci95": boot_ci(p, n)}
        return out

    res["auto_masks"], res["reference_masks"], res["hull_control"] = table(rec_a), table(rec_v), table(rec_h)
    bl = [r for r in blobs if r.get("eligible")]
    res["blob_control"] = {"n": len(bl), "median_library_iou": round(statistics.median(r["best_library_iou"] for r in bl), 4),
                           "median_refined_iou": round(statistics.median(r["best_refined_iou"] for r in bl), 4)}
    el = [r for r in rec_v if r["eligible"]]
    fid_of = {r["name"][:-4]: r for r in el}
    pos = [f for f in fid_of if labels[f] == 1]
    neg = [f for f in fid_of if labels[f] == 0]
    base = {"size_short_side(smaller=abnormal)": [-min(Image.open(images / (f + ".png")).size) for f in fid_of],
            "solidity(lower=abnormal)": [-solid[f] for f in fid_of]}
    res["baselines_on_reference_masks"] = {}
    for k, vals in base.items():
        d = dict(zip(fid_of, vals))
        res["baselines_on_reference_masks"][k] = {"auc": round(auc([d[f] for f in pos], [d[f] for f in neg]), 3),
                                                  "ci95": boot_ci([d[f] for f in pos], [d[f] for f in neg])}
    ref = res["reference_masks"]
    best = max(ref["auc"], key=lambda m: ref["auc"][m]["auc"]) if ref["auc"] else None
    g1 = (ref["median_refined_iou"] - res["blob_control"]["median_refined_iou"] >= G1_BLOB_MARGIN and
          ref["median_refined_iou"] - res["hull_control"]["median_refined_iou"] >= G1_HULL_MARGIN)
    g2 = False
    if best:
        b = ref["auc"][best]
        g2 = (b["auc"] >= G2_MIN_AUC and b["ci95"][0] > 0.5 and
              all(b["auc"] - v["auc"] >= G2_BASELINE_MARGIN for v in res["baselines_on_reference_masks"].values()))
    auto_best = max((v["auc"] for v in res["auto_masks"]["auc"].values()), default=None)
    res["gates"] = {"best_method_on_reference_masks": best,
                    "G1_representation_specificity": bool(g1), "G2_discrimination": bool(g2),
                    "G1_detail": {"hand_minus_blob_refined_iou": round(ref["median_refined_iou"] - res["blob_control"]["median_refined_iou"], 4),
                                  "hand_minus_hull_refined_iou": round(ref["median_refined_iou"] - res["hull_control"]["median_refined_iou"], 4)},
                    "attribution_auc_reference_minus_auto": None if (auto_best is None or best is None) else round(ref["auc"][best]["auc"] - auto_best, 3)}
    if res["accepted"] < MIN_ACCEPTED:
        res["formal_conclusion"] = "DATASET_UNSUITABLE (fewer than %d accepted reference masks)" % MIN_ACCEPTED
    if exploratory_all:
        res["conclusion"] = "EXPLORATORY_ONLY (masks were not reviewer-accepted; formal conclusion above)"
    elif res["accepted"] < MIN_ACCEPTED:
        res["conclusion"] = "DATASET_UNSUITABLE (fewer than %d accepted reference masks)" % MIN_ACCEPTED
    elif g1 and g2:
        res["conclusion"] = "PROCEED_TO_PREREGISTERED_HELDOUT"
    else:
        res["conclusion"] = "STOP_SILHOUETTE_DEVELOPMENT_RECOMMEND_LANDMARK_PART_LEVEL"
    (out_dir / "results.json").write_text(json.dumps({"results": res, "auto": rec_a, "reference": rec_v, "hull": rec_h},
                                                 indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    import sys

    print(json.dumps(run(exploratory_all="--exploratory-all" in sys.argv), indent=1, sort_keys=True))
