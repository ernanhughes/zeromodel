"""ZM-HAND-03 data-feasibility inventory over Spotlight ``test/metadata.jsonl``.

Metadata only; no video is read.  Keyword matching below **nominates candidates for
manual review; it never verifies a defect** and never labels anything normal.

    python -m examples.hand_anatomy_03.spotlight_inventory \
        --metadata artifacts/zm_hand_03/raw/metadata.jsonl --output artifacts/zm_hand_03

Outputs: ``inventory.json`` (all counts used by the feasibility report) and
``candidates.csv`` (one row per annotation with a hand/arm/finger term, for human review).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Dict, List, Sequence

HAND_TERMS = r"hands?|fingers?|thumbs?|palms?|wrists?|knuckles?|digits?|fists?|handshake|pinky|pinkie"
ARM_TERMS = r"arms?|forearms?|elbows?"
STRUCTURE_TERMS = (r"extra|additional|six|seven|eight|four|three|fused|merg\w*|melt\w*|missing|fewer|"
                   r"too many|too few|deform\w*|distort\w*|mutat\w*|disfigur\w*|"
                   r"extra\s+(?:finger|digit|thumb)s?|disappear\w*|vanish\w*|appear\w*|morph\w*|"
                   r"warp\w*|bend\w*|twist\w*|unnatural\w*|rubber\w*")
HAND_RE = re.compile(r"\b(?:%s)\b" % HAND_TERMS, re.I)
ARM_RE = re.compile(r"\b(?:%s)\b" % ARM_TERMS, re.I)
STRUCT_RE = re.compile(r"\b(?:%s)\b" % STRUCTURE_TERMS, re.I)
DIGIT_PART_RE = re.compile(r"\b(?:fingers?|thumbs?|digits?|knuckles?|pinky|pinkie)\b", re.I)
TYPES = ("anatomy_body_pose", "app_disapp", "motion", "physical_viol", "adherence", "logical")

# Held-out quotas from the provisional acceptance requirements
MIN_INTERVALS = 60
MIN_HELDOUT_CLIPS = 15
SPLIT_FRACTIONS = (("development", 0.40), ("calibration", 0.20), ("heldout", 0.40))


def classify(ann: Dict[str, Any]) -> Dict[str, Any]:
    text = str(ann.get("reason", ""))
    hand = sorted({m.lower() for m in HAND_RE.findall(text)})
    arm = sorted({m.lower() for m in ARM_RE.findall(text)})
    struct = sorted({m.lower() for m in STRUCT_RE.findall(text)})
    digit_part = bool(DIGIT_PART_RE.search(text))
    t = ann.get("type")
    if not hand and not arm:
        cls = "non_hand"
    elif t == "anatomy_body_pose" and hand:
        cls = "candidate_hand_anatomy"
    elif t == "app_disapp" and hand:
        cls = "candidate_hand_appearance"
    elif t in ("motion", "physical_viol") and hand:
        cls = "candidate_hand_motion_or_contact"
    else:
        cls = "requires_manual_review"  # arm-only, or hand terms under adherence/logical, etc.
    return {"class": cls, "hand_terms": hand, "arm_terms": arm, "structure_terms": struct,
            "digit_part_term": digit_part}


def pid_split(pids: Sequence[str]) -> Dict[str, str]:
    """Deterministic pid-level partition by hash rank (never by video filename)."""
    ranked = sorted(pids, key=lambda p: hashlib.sha256(("zm-hand-03:" + p).encode()).hexdigest())
    out: Dict[str, str] = {}
    n = len(ranked)
    start = 0
    for name, frac in SPLIT_FRACTIONS:
        end = n if name == SPLIT_FRACTIONS[-1][0] else start + int(round(frac * n))
        for p in ranked[start:end]:
            out[p] = name
        start = end
    return out


def _stats(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"n": len(s), "min": round(s[0], 2), "p25": round(q(0.25), 2), "median": round(statistics.median(s), 2),
            "p75": round(q(0.75), 2), "max": round(s[-1], 2), "mean": round(statistics.fmean(s), 2)}


def wilson_halfwidth(n: int, p: float = 0.5, z: float = 1.96) -> float:
    d = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def build_inventory(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    split = pid_split(sorted({r["pid"] for r in rows}))
    anns: List[Dict[str, Any]] = []
    for r in rows:
        for i, a in enumerate(json.loads(r["annotation"])):
            seg = a.get("segment") or [0, 0]
            c = classify(a)
            anns.append({
                "video_id": r["id"], "pid": r["pid"], "generator": r["video_from"], "index": i,
                "start": float(seg[0]), "end": float(seg[1]), "duration": float(seg[1]) - float(seg[0]),
                "type": a.get("type"), "difficulty": a.get("difficulty"), "reason": a.get("reason", ""),
                "split": split[r["pid"]], **c,
            })
    by = lambda key, items: {k: sum(1 for x in items if x[key] == k) for k in sorted({x[key] for x in items})}  # noqa: E731
    hand_any = [a for a in anns if a["class"] != "non_hand"]
    cand = [a for a in anns if a["class"].startswith("candidate_hand")]
    strict = [a for a in anns if a["class"] == "candidate_hand_anatomy"]
    digit_struct = [a for a in strict if a["digit_part_term"] and a["structure_terms"]]
    prompts_with_hands = {r["pid"] for r in rows if HAND_RE.search(r["prompt"])}
    videos_with_cand = {a["video_id"] for a in cand}
    videos_any_hand_ann = {a["video_id"] for a in hand_any}

    def split_table(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {}
        for name, _ in SPLIT_FRACTIONS:
            sub = [a for a in items if a["split"] == name]
            out[name] = {"intervals": len(sub), "clips": len({a["video_id"] for a in sub}),
                         "pids": len({a["pid"] for a in sub})}
        return out

    def duration_groups(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"all": _stats([a["duration"] for a in items]),
                "lt_1s": sum(1 for a in items if a["duration"] < 1.0),
                "1_to_3s": sum(1 for a in items if 1.0 <= a["duration"] < 3.0),
                "ge_3s": sum(1 for a in items if a["duration"] >= 3.0)}

    inv: Dict[str, Any] = {
        "videos": len(rows), "unique_pids": len({r["pid"] for r in rows}),
        "videos_per_generator": by("video_from", rows),
        "annotations_total": len(anns), "annotations_by_type": by("type", anns),
        "annotations_by_generator": by("generator", anns), "annotations_by_difficulty": by("difficulty", anns),
        "annotation_duration_seconds": duration_groups(anns),
        "refvideo_exists": {"true": sum(1 for r in rows if r.get("refvideo_exists")),
                            "false": sum(1 for r in rows if not r.get("refvideo_exists"))},
        "hand_arm_term_annotations": {
            "any_hand_or_arm_term": len(hand_any),
            "hand_finger_wrist_term": sum(1 for a in anns if a["hand_terms"]),
            "arm_only_term": sum(1 for a in anns if a["arm_terms"] and not a["hand_terms"]),
            "by_class": by("class", anns),
            "by_type_with_hand_term": by("type", [a for a in anns if a["hand_terms"]]),
        },
        "anatomy_body_pose": {
            "total": sum(1 for a in anns if a["type"] == "anatomy_body_pose"),
            "with_hand_finger_wrist_term": len(strict),
            "with_digit_part_term": sum(1 for a in strict if a["digit_part_term"]),
            "with_digit_part_and_structure_term": len(digit_struct),
            "arm_only": sum(1 for a in anns if a["type"] == "anatomy_body_pose" and a["arm_terms"] and not a["hand_terms"]),
        },
        "candidates_unverified": {
            "candidate_hand_intervals": len(cand), "clips": len(videos_with_cand),
            "pids": len({a["pid"] for a in cand}),
            "by_class": by("class", cand), "by_generator": by("generator", cand),
            "by_difficulty": by("difficulty", cand), "duration": duration_groups(cand),
            "strict_anatomy_by_generator": by("generator", strict),
            "strict_anatomy_by_difficulty": by("difficulty", strict),
        },
        "requires_manual_review_intervals": sum(1 for a in anns if a["class"] == "requires_manual_review"),
        "clips_with_any_hand_annotation": len(videos_any_hand_ann),
        "negatives": {
            "prompts_mentioning_hand_terms": len(prompts_with_hands),
            "videos_whose_prompt_mentions_hands_and_have_no_hand_annotation": sum(
                1 for r in rows if r["pid"] in prompts_with_hands and r["id"] not in videos_any_hand_ann),
            "note": "An absence of annotation is NOT a verified-normal interval. These are only candidates to review.",
        },
        "pid_level_split_all_candidates": split_table(cand),
        "pid_level_split_strict_anatomy": split_table(strict),
        "pid_level_split_sizes": {n: sum(1 for v in split.values() if v == n) for n, _ in SPLIT_FRACTIONS},
    }
    held = inv["pid_level_split_all_candidates"]["heldout"]
    inv["power"] = {
        "requirement": {"min_defect_intervals_total": MIN_INTERVALS, "min_heldout_clips": MIN_HELDOUT_CLIPS},
        "candidate_intervals_total": len(cand),
        "candidate_intervals_heldout_unverified_upper_bound": held["intervals"],
        "candidate_clips_heldout_unverified_upper_bound": held["clips"],
        "ci95_halfwidth_recall_at_p50_by_n": {str(n): round(wilson_halfwidth(n), 3) for n in (10, 20, 30, 60, 100)},
    }
    inv["_annotations"] = anns
    return inv


def write_outputs(inv: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    anns = inv.pop("_annotations")
    (out_dir / "inventory.json").write_text(json.dumps(inv, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    cols = ["video_id", "pid", "generator", "split", "start", "end", "duration", "type", "difficulty", "class",
            "hand_terms", "arm_terms", "structure_terms", "reason", "manual_review_verdict"]
    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for a in anns:
            if a["class"] != "non_hand":
                row = {k: a[k] for k in cols if k in a}
                for k in ("hand_terms", "arm_terms", "structure_terms"):
                    row[k] = ";".join(a[k])
                row["manual_review_verdict"] = ""  # to be filled by a human: hand_defect / not_hand / ambiguous
                w.writerow(row)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.metadata).read_text(encoding="utf-8").splitlines() if line.strip()]
    inv = build_inventory(rows)
    write_outputs(inv, Path(args.output))
    print(json.dumps({k: inv[k] for k in ("videos", "annotations_total", "candidates_unverified", "power")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
