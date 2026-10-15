"""Bounded access to the VBench-2.0 human-anomaly hand crops (ZM-HAND-03a).

* Archive members are read by HTTP range request from the split zip; the 10.9 GB
  archive is never downloaded.  A member that spans a segment boundary is stitched
  from consecutive segments.
* The split is VIDEO-disjoint: the published train/test files share 74 of the 77 test
  video ids, so the published test set is not an independent video-level test.
* Selection uses only labels, video ids and hashes, never image content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request
from typing import Any, Dict, List, Sequence
import zlib

BASE = "https://huggingface.co/datasets/Vchitect/VBench-2.0_human_anomaly/resolve/main/"
SEGMENTS = {0: ("opensource.z01", 4294967296), 1: ("opensource.z02", 4294967296), 2: ("opensource.zip", 2319717137)}
IMAGE_PREFIX = "opensource/dataset/all_images/"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")
MAX_IMAGE_BYTES = 5_000_000
SPLIT_SALT = "zm-hand-03a:"
# video-hash rank fractions: development pool, calibration pool, gap, held-out pool
DEV_POOL_FRACTION = 0.05
HELDOUT_POOL_START = 0.50


def video_id(filename: str) -> str:
    return filename.split("_")[0]


def video_rank_key(vid: str) -> str:
    return hashlib.sha256((SPLIT_SALT + vid).encode()).hexdigest()


def partition_videos(vids: Sequence[str]) -> Dict[str, str]:
    """Deterministic video-level partition: development / unused / held-out."""
    ranked = sorted(set(vids), key=video_rank_key)
    n = len(ranked)
    out: Dict[str, str] = {}
    for i, v in enumerate(ranked):
        f = i / n
        out[v] = "development" if f < DEV_POOL_FRACTION else ("heldout" if f >= HELDOUT_POOL_START else "unused")
    return out


def select_crops(rows: Sequence[Sequence[Any]], part: Dict[str, str], pool: str, per_label: int) -> List[Dict[str, Any]]:
    """Balanced selection, one crop per video, deterministic, label/id/hash only."""
    by_video: Dict[str, List[Sequence[Any]]] = {}
    for r in rows:
        v = video_id(r[0])
        if part.get(v) == pool:
            by_video.setdefault(v, []).append(r)
    chosen: List[Dict[str, Any]] = []
    count = {0: 0, 1: 0}
    for v in sorted(by_video, key=video_rank_key):
        crops = sorted(by_video[v], key=lambda r: hashlib.sha256(r[0].encode()).hexdigest())
        r = crops[0]
        if count[int(r[1])] < per_label:
            chosen.append({"name": r[0], "label": int(r[1]), "video": v})
            count[int(r[1])] += 1
        if all(c >= per_label for c in count.values()):
            break
    return chosen


def _range(seg: int, start: int, end: int) -> bytes:
    name, size = SEGMENTS[seg]
    end = min(end, size - 1)
    req = urllib.request.Request(BASE + name, headers={"Range": "bytes=%d-%d" % (start, end)})
    with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310 (fixed https host)
        return resp.read()


def read_member(entry: Dict[str, Any]) -> bytes:
    """Fetch, inflate and CRC-check one zip member by range request (stitching across segments)."""
    need = 30 + len(entry["name"].encode()) + 512 + entry["compressed"]
    seg, off = entry["disk_start"], entry["offset"]
    blob = _range(seg, off, off + need - 1)
    seg_size = SEGMENTS[seg][1]
    while len(blob) < need and off + len(blob) >= seg_size and seg + 1 in SEGMENTS:
        seg += 1
        blob += _range(seg, 0, need - len(blob) - 1)
        off = -len(blob)  # only used for the loop condition
        break
    if blob[:4] != b"PK\x03\x04":
        raise ValueError("bad local header for %s" % entry["name"])
    (_, _, _, method, _, _, _, _, _, nlen, elen) = struct.unpack("<IHHHHHIIIHH", blob[:30])
    data = blob[30 + nlen + elen:30 + nlen + elen + entry["compressed"]]
    raw = zlib.decompress(data, -15) if method == 8 else data
    if len(raw) > MAX_IMAGE_BYTES or len(raw) != entry["uncompressed"]:
        raise ValueError("unexpected member size for %s" % entry["name"])
    if zlib.crc32(raw) != entry["crc32"]:
        raise ValueError("CRC mismatch for %s" % entry["name"])
    return raw


def fetch_images(names: Sequence[str], central: Dict[str, Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for n in names:
        if not SAFE_NAME.match(n):
            raise ValueError("unsafe file name %r" % n)
        raw = read_member(central[IMAGE_PREFIX + n])
        (out_dir / n).write_bytes(raw)
        total += len(raw)
    return {"images": len(names), "bytes": total}


def load_central(path: Path) -> Dict[str, Dict[str, Any]]:
    return {e["name"]: e for e in json.loads(path.read_text(encoding="utf-8"))["entries"]}


def read_labels(path: Path) -> List[List[Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_integrity(train: Sequence[Sequence[Any]], test: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    tv, sv = {video_id(r[0]) for r in train}, {video_id(r[0]) for r in test}
    return {
        "train_rows": len(train), "test_rows": len(test), "train_videos": len(tv), "test_videos": len(sv),
        "test_videos_also_in_train": len(tv & sv),
        "test_images_whose_video_is_in_train": sum(1 for r in test if video_id(r[0]) in tv),
        "filenames_in_both": len({r[0] for r in train} & {r[0] for r in test}),
        "video_disjoint": not (tv & sv),
    }
