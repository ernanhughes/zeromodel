import io
import json
from pathlib import Path
import zipfile

import pytest

from examples.hand_anatomy_03 import spotlight_inventory as si
from examples.hand_anatomy_03 import zip_tail_inventory as zt
from examples.hand_anatomy_03.fetch_vbench_hand_test import PLAN, main as fetch_main

pytestmark = pytest.mark.research


def _ann(reason, type_="anatomy_body_pose"):
    return {"segment": [1.0, 2.5], "reason": reason, "type": type_, "difficulty": "easy"}


def test_classification_nominates_candidates_but_never_verifies():
    assert si.classify(_ann("Too many fingers on the left hand"))["class"] == "candidate_hand_anatomy"
    assert si.classify(_ann("hand disappears", "app_disapp"))["class"] == "candidate_hand_appearance"
    assert si.classify(_ann("hand passes through the cup", "motion"))["class"] == "candidate_hand_motion_or_contact"
    assert si.classify(_ann("his arm is too short"))["class"] == "requires_manual_review"
    assert si.classify(_ann("a person holds the hand of a friend", "adherence"))["class"] == "requires_manual_review"
    assert si.classify(_ann("The legs move oddly"))["class"] == "non_hand"
    assert si.classify(_ann("shorthand notes"))["class"] == "non_hand"  # word-boundary match only


def test_pid_split_is_deterministic_and_never_splits_a_prompt():
    pids = ["sid_%03d" % i for i in range(1, 201)]
    a, b = si.pid_split(pids), si.pid_split(list(reversed(pids)))
    assert a == b and set(a) == set(pids)
    sizes = {n: sum(1 for v in a.values() if v == n) for n in ("development", "calibration", "heldout")}
    assert sizes == {"development": 80, "calibration": 40, "heldout": 80}


def test_inventory_counts_on_a_tiny_metadata_set():
    rows = []
    for i, generator in enumerate(("veo3", "ltx2", "seedance")):
        rows.append({"id": "sid_001_%s" % generator, "pid": "sid_001", "video_from": generator,
                     "prompt": "A man waves his hand.", "refvideo_exists": False,
                     "annotation": json.dumps([_ann("six fingers on the hand"), _ann("legs", "motion")])})
    inv = si.build_inventory(rows)
    assert inv["videos"] == 3 and inv["unique_pids"] == 1 and inv["annotations_total"] == 6
    assert inv["candidates_unverified"]["candidate_hand_intervals"] == 3
    assert len({a["split"] for a in inv["_annotations"]}) == 1  # one prompt id -> one partition
    assert "NOT a verified-normal" in inv["negatives"]["note"]


def test_zip_tail_parser_lists_entries_without_extracting():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dataset/a.jsonl", "x" * 500)
        z.writestr("dataset/all_images/b.jpg", b"\xff\xd8" + b"y" * 100)
    info = zt.parse_tail(buf.getvalue(), len(buf.getvalue()))
    assert [e["name"] for e in info["entries"]] == ["dataset/a.jsonl", "dataset/all_images/b.jpg"]
    assert info["entries"][0]["uncompressed"] == 500
    with pytest.raises(ValueError):
        zt.parse_tail(b"not a zip", 9)


def test_fetch_plan_is_consistent_and_dry_run_downloads_nothing(capsys):
    plan = json.loads(Path(PLAN).read_text(encoding="utf-8"))["test_images"]
    assert len(plan) == 142 and {p["label"] for p in plan} == {0, 1}
    assert sum(p["label"] for p in plan) == 72
    assert fetch_main(["--output", "unused"]) == 0  # no --confirm: dry run
    assert "dry run" in capsys.readouterr().out
