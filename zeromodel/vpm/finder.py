#  zeromodel/vpm/finder.py
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from zeromodel.constants import _PNG_SIG


@dataclass
class FindStep:
    level: int
    tile_id: bytes
    path: str
    pointer_index: int  # which child we took
    span: int  # block span (docs)
    x_offset: int  # start offset
    doc_block_size: int


class VPMFinder:
    @staticmethod
    def find_target(
        start_path: str,
        *,
        resolver: Callable[[bytes], str],
        choose_child: Callable[[Dict[str, Any]], int],
        max_hops: int = 64,
    ) -> Tuple[str, bytes, List[FindStep]]:
        """
        Follow router pointers from start tile until choose_child() says 'stop' or max_hops is reached.

        - resolver(tile_id) -> path: maps a tile_id to a file to open next
        - choose_child(meta_dict) -> index: returns which child pointer to take; return -1 to stop

        Returns: (final_path, final_tile_id, audit_trail)
        """
        path = start_path
        audit: List[FindStep] = []
        for _ in range(max_hops):
            md = VPMFinder._read_metadata_fast(path)  # header-only, no IDAT inflate
            # md is a dict with: tile_id, level, pointers=[{tile_id, level, x_offset, span, doc_block_size, agg_id}, ...]
            choice = choose_child(md)
            if choice is None or choice < 0 or choice >= len(md.get("pointers", [])):
                return path, md["tile_id"], audit

            p = md["pointers"][choice]
            audit.append(
                FindStep(
                    level=md.get("level", 0),
                    tile_id=md["tile_id"],
                    path=path,
                    pointer_index=choice,
                    span=int(p.get("span", 0)),
                    x_offset=int(p.get("x_offset", 0)),
                    doc_block_size=int(p.get("doc_block_size", 1)),
                )
            )
            # hop to child
            path = resolver(p["tile_id"])
        # safety stop
        md = VPMFinder._read_metadata_fast(path)
        return path, md["tile_id"], audit

    @staticmethod
    def _read_metadata_fast(path: str) -> Dict[str, Any]:
        """
        Read VPM metadata without decoding image pixels.
        Looks for a custom ancillary chunk 'vpMm' first.
        Falls back to zero metadata if missing.
        """
        with open(path, "rb") as f:
            sig = f.read(8)
            if sig != _PNG_SIG:
                raise ValueError("Not a PNG")

            md_bytes = b""
            width = None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                (length,) = struct.unpack(">I", hdr[:4])
                ctype = hdr[4:8]
                data = f.read(length)
                _ = f.read(4)  # CRC, ignore

                if ctype == b"IHDR":
                    width, height = struct.unpack(">II", data[:8])

                # Our custom metadata chunk (must be written by VPMImageWriter)
                if ctype == b"vpMm":  # custom ancillary chunk name
                    md_bytes = data
                    # we can stop here; we have the full metadata
                    break

                # If we reached IDAT without vpMm, stop scanning chunks.
                if ctype == b"IDAT":
                    break

            if not md_bytes:
                # No custom chunk found: return minimal info (still usable for audit)
                return {
                    "tile_id": b"",
                    "level": 0,
                    "metric_count": 0,
                    "doc_count": width or 0,
                    "pointers": [],
                }

            # Decode your existing VPMMetadata binary format:
            # Assuming you already have VPMMetadata.from_bytes(...)
            from zeromodel.vpm.metadata import VPMMetadata

            meta = VPMMetadata.from_bytes(md_bytes)

            pointers = []
            for p in getattr(meta, "pointers", []) or []:
                pointers.append(
                    {
                        "tile_id": p.tile_id,
                        "level": p.level,
                        "x_offset": p.x_offset,
                        "span": p.span,
                        "doc_block_size": p.doc_block_size,
                        "agg_id": p.agg_id,
                    }
                )

            return {
                "tile_id": meta.tile_id,
                "level": getattr(meta, "level", 0),
                "metric_count": getattr(meta, "metric_count", 0),
                "doc_count": getattr(meta, "doc_count", 0),
                "doc_block_size": getattr(meta, "doc_block_size", 1),
                "agg_id": getattr(meta, "agg_id", 0),
                "pointers": pointers,
                "task_hash": getattr(meta, "task_hash", 0),
            }


def hottest_child(meta: Dict[str, Any]) -> int:
    """
    Example policy: choose child with largest span (or any other hint you pack in pointers).
    Return -1 to stop.
    """
    ptrs = meta.get("pointers", [])
    if not ptrs:
        return -1
    # pick by span; replace with your own heuristic (e.g., position hint)
    return max(range(len(ptrs)), key=lambda i: ptrs[i].get("span", 0))


def id_to_path(tile_id: bytes) -> str:
    # map tile_id -> file path (DictResolver / FilenameResolver / DB lookup)
    hexid = tile_id.hex()
    return f"/data/vpm/tiles/{hexid}.png"


final_path, final_id, steps = VPMFinder.find_target(
    start_path="/data/vpm/root.png",
    resolver=id_to_path,
    choose_child=hottest_child,
    max_hops=64,
)
