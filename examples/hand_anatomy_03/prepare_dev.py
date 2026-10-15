"""Stage 1-2: split-integrity check, video-disjoint dev selection, bounded fetch of the dev crops.

Needs artifacts/zm_hand_03/raw/{hand_train.jsonl,hand_test.jsonl,zip_central_directory.json}
(produced by the range reads documented in docs/research/ZM-HAND-03a-real-image-pilot.md).

    python -m examples.hand_anatomy_03.prepare_dev
"""

from __future__ import annotations

import json
from pathlib import Path

from . import vbench_data as vd

RAW = Path("artifacts/zm_hand_03/raw")


def main() -> int:
    train, test = vd.read_labels(RAW / "hand_train.jsonl"), vd.read_labels(RAW / "hand_test.jsonl")
    print(json.dumps(vd.split_integrity(train, test), indent=1))
    seen = {r[0] for r in train}
    rows = train + [r for r in test if r[0] not in seen]
    part = vd.partition_videos([vd.video_id(r[0]) for r in rows])
    dev = vd.select_crops(rows, part, "development", 20)
    central = vd.load_central(RAW / "zip_central_directory.json")
    print("dev crops:", len(dev), "bytes:", sum(central[vd.IMAGE_PREFIX + d["name"]]["compressed"] for d in dev))
    print(vd.fetch_images([d["name"] for d in dev], central, Path("artifacts/zm_hand_03/vbench_dev")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
