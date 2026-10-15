"""List the entries of a (possibly split) zip from its last bytes only.

Parses End-of-Central-Directory (+ZIP64) and the central directory out of a tail
slice fetched with an HTTP range request.  No archive member is extracted.
Treats the input as untrusted data: bounded reads, no extraction, no eval.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List


def parse_tail(tail: bytes, total_size: int) -> Dict[str, Any]:
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("EOCD not found in tail")
    (_, disk, cd_disk, n_disk, n_total, cd_size, cd_off, _clen) = struct.unpack("<IHHHHIIH", tail[eocd:eocd + 22])
    info: Dict[str, Any] = {"disk": disk, "cd_disk": cd_disk, "entries_total": n_total, "cd_size": cd_size,
                            "cd_offset": cd_off}
    loc = tail.rfind(b"PK\x06\x07", 0, eocd)
    if n_total == 0xFFFF or cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        if loc < 0:
            raise ValueError("ZIP64 locator missing")
        z64 = tail.find(b"PK\x06\x06")
        (_, _, _, _, d, cdd, n_d, n_t, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", tail[z64:z64 + 56])
        info.update(disk=d, cd_disk=cdd, entries_total=n_t, cd_size=cd_size, cd_offset=cd_off, zip64=True)
    # Central directory lives on disk `cd_disk`; with a tail from the last segment its start must lie in the tail.
    start = eocd - info["cd_size"]
    if "zip64" in info:
        start = tail.find(b"PK\x06\x06") - info["cd_size"]
    if start < 0:
        raise ValueError("central directory larger than the fetched tail (%d bytes)" % info["cd_size"])
    entries: List[Dict[str, Any]] = []
    pos = start
    while tail[pos:pos + 4] == b"PK\x01\x02" and len(entries) < 200000:
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, elen, clen, dstart, _, _, off) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", tail[pos:pos + 46])
        name = tail[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        extra = tail[pos + 46 + nlen:pos + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, off) or dstart == 0xFFFF:
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[e:e + 4])
                if hid == 1:
                    vals = list(struct.unpack("<%dQ" % ((hsz - (4 if dstart == 0xFFFF else 0)) // 8),
                                              extra[e + 4:e + 4 + hsz - (4 if dstart == 0xFFFF else 0)]))
                    if usize == 0xFFFFFFFF and vals:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF and vals:
                        csize = vals.pop(0)
                    if off == 0xFFFFFFFF and vals:
                        off = vals.pop(0)
                e += 4 + hsz
        entries.append({"name": name, "method": method, "compressed": csize, "uncompressed": usize,
                        "disk_start": dstart, "offset": off, "crc32": crc})
        pos += 46 + nlen + elen + clen
    info["entries"] = entries
    return info


def main(argv: List[str]) -> int:
    tail = Path(argv[1]).read_bytes()
    total = int(argv[2])
    info = parse_tail(tail, total)
    Path(argv[3]).write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    ents = info["entries"]
    print("entries listed: %d of %d" % (len(ents), info["entries_total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
