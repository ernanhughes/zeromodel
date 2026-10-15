"""Fetch ONLY the 142 VBench-2.0 hand test images via HTTP range requests.

NOT run by the feasibility pass.  Total transfer is about 3.2 MB (compressed) plus
small per-request overhead; the 10.9 GB archive is never downloaded.  The plan in
``data/vbench_hand_test_fetch_plan.json`` was derived from the zip central directory
(a metadata-only tail read) and ``hand_test.jsonl``.

Extracted bytes are untrusted image files: they are written under ``--output`` with
a sanitised basename, never executed, and the zip-local-header CRC is not relied on.

    python -m examples.hand_anatomy_03.fetch_vbench_hand_test --output artifacts/zm_hand_03/vbench_hand_test --confirm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct
import urllib.request
import zlib

BASE = "https://huggingface.co/datasets/Vchitect/VBench-2.0_human_anomaly/resolve/main/"
SEGMENTS = {0: "opensource.z01"}  # all 142 test images live in the first segment
PLAN = Path(__file__).with_name("data") / "vbench_hand_test_fetch_plan.json"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")
MAX_IMAGE_BYTES = 5_000_000


def _range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https host)
        return resp.read()


def extract_member(blob: bytes, expected_compressed: int) -> bytes:
    if blob[:4] != b"PK\x03\x04":
        raise ValueError("not a zip local header")
    (_, _, _, method, _, _, _, _, usize, nlen, elen) = struct.unpack("<IHHHHHIIIHH", blob[:30])
    data = blob[30 + nlen + elen:30 + nlen + elen + expected_compressed]
    raw = zlib.decompress(data, -15) if method == 8 else data
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image larger than the safety limit")
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--confirm", action="store_true", help="required: acknowledges a ~3.2 MB download")
    args = ap.parse_args(argv)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))["test_images"]
    total = sum(p["compressed"] for p in plan)
    print("%d images, %.2f MB compressed, from %s" % (len(plan), total / 1e6, BASE))
    if not args.confirm:
        print("dry run: pass --confirm to download")
        return 0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for p in plan:
        if not SAFE.match(p["name"]) or p["disk_start"] not in SEGMENTS:
            raise SystemExit("unexpected plan entry: %r" % p["name"])
        start = p["offset"]
        blob = _range(BASE + SEGMENTS[p["disk_start"]], start, start + 30 + len(p["name"]) + 300 + p["compressed"])
        (out / p["name"]).write_bytes(extract_member(blob, p["compressed"]))
    (out / "labels.json").write_text(json.dumps({p["name"]: p["label"] for p in plan}, indent=1), encoding="utf-8")
    print("wrote %d images to %s" % (len(plan), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
