# zeromodel/vpm/metadata.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Iterable, Optional, Protocol, Tuple
import struct
import hashlib
import math

# ---------- enums ----------

class MapKind(IntEnum):
    VPM = 0          # canonical VPM tile
    ROUTER_FRAME = 1  # tiny step frame used to route / All right cinema
    SEARCH_VIEW = 2   # orthogonal view (composite / manifold lens)

class AggId(IntEnum):
    MAX = 0
    MEAN = 1
    RAW = 65535   # base (no aggregation)

# ---------- resolvers (pluggable) ----------

class TargetResolver(Protocol):
    """Pluggable resolver for tile IDs -> target path/handle."""
    def resolve(self, tile_id: bytes) -> Optional[str]: ...

@dataclass
class FilenameResolver:
    """Maps 16b tile_id to a conventional filename pattern."""
    pattern: str = "vpm_{hexid}_L{level}_B{block}.png"
    default_level: int = 0
    default_block: int = 1

    def resolve(self, tile_id: bytes) -> Optional[str]:
        hexid = tile_id.hex()
        return self.pattern.format(hexid=hexid, level=self.default_level, block=self.default_block)

@dataclass
class DictResolver:
    """In-memory map for testing."""
    mapping: Dict[bytes, str]

    def resolve(self, tile_id: bytes) -> Optional[str]:
        return self.mapping.get(tile_id)

# ---------- 36-byte router pointer block ----------
# Layout (big-endian):
#   0: kind (1 byte)
#   1: reserved/version nibble (1 byte, currently 0x01)
#   2-3: level (u16)
#   4-7: x_offset (u32)     -- start column (doc index) within child tile’s logical span
#   8-11: span (u32)        -- number of docs represented by that column in parent
#   12-15: doc_block_size (u32)
#   16-17: agg_id (u16)
#   18-33: tile_id digest (16 bytes)
#   34-35: reserved/padding (u16)
_ROUTER_PTR_FMT = ">BBHIIIH16sH"
_ROUTER_PTR_SIZE = struct.calcsize(_ROUTER_PTR_FMT)  # 36

@dataclass
class RouterPointer:
    kind: MapKind
    level: int
    x_offset: int
    span: int
    doc_block_size: int
    agg_id: int
    tile_id: bytes  # 16 bytes

    def to_bytes(self) -> bytes:
        assert len(self.tile_id) == 16
        return struct.pack(
            _ROUTER_PTR_FMT,
            int(self.kind) & 0xFF,
            0x01,                         # version/reserved
            self.level & 0xFFFF,
            self.x_offset & 0xFFFFFFFF,
            self.span & 0xFFFFFFFF,
            self.doc_block_size & 0xFFFFFFFF,
            self.agg_id & 0xFFFF,
            self.tile_id,
            0                              # padding
        )

    @staticmethod
    def from_bytes(b: bytes) -> "RouterPointer":
        if len(b) != _ROUTER_PTR_SIZE:
            raise ValueError(f"router pointer must be 36 bytes; got {len(b)}")
        k, ver, lvl, xoff, span, block, agg, tid, _pad = struct.unpack(_ROUTER_PTR_FMT, b)
        return RouterPointer(MapKind(k), lvl, xoff, span, block, agg, tid)

# ---------- helpers for metric weights nibble packing ----------

def _weights_to_nibbles(weights: Dict[str, float], metric_names: List[str]) -> bytes:
    """4-bit per metric (two per byte) in metric_names order."""
    out = bytearray()
    for i in range(0, len(metric_names), 2):
        b = 0
        # high nibble
        w0 = max(0.0, min(1.0, float(weights.get(metric_names[i], 0.0))))
        n0 = int(round(w0 * 15.0)) & 0x0F
        b |= (n0 << 4)
        # low nibble
        if i + 1 < len(metric_names):
            w1 = max(0.0, min(1.0, float(weights.get(metric_names[i+1], 0.0))))
            n1 = int(round(w1 * 15.0)) & 0x0F
            b |= n1
        out.append(b)
    return bytes(out)

def _nibbles_to_weights(nibbles: bytes, metric_names: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, name in enumerate(metric_names):
        byte = nibbles[i // 2] if (i // 2) < len(nibbles) else 0
        nib = (byte >> 4) & 0x0F if (i % 2 == 0) else (byte & 0x0F)
        out[name] = nib / 15.0
    return out

# ---------- VPM metadata (single source of truth) ----------

# Binary layout (big-endian) for the fixed header (“VMETA”):
#  0-4:   magic "VMETA" (5 bytes)
#  5:     version (u8)
#  6:     kind (u8)  -> MapKind
#  7:     reserved (u8)
#  8-9:   level (u16)
#  10-11: agg_id (u16)
#  12-13: metric_count (u16)
#  14-17: doc_count (u32)   (logical D in this image)
#  18-19: doc_block_size (u16)  (logical grouping)
#  20-23: task_hash (u32)
#  24-39: tile_id (16 bytes)
#  40-55: parent_id (16 bytes)  (zeros if root)
#  56-63: step_id (u64)         (router frame / trace)
#  64-71: parent_step_id (u64)
#  72-79: timestamp_ns (u64)
#  80-81: weights_len_bytes (u16)
#  82-83: ptr_count (u16)
#  84-..: weights_nibbles (weights_len_bytes bytes)
#  ..-..: router pointers (ptr_count * 36 bytes)

_META_MAGIC = b"VMETA"
_META_FIXED_FMT = ">5s BBB HHH I H I 16s16s Q Q Q H H"
# Note: spaces are for readability; struct ignores them.
_META_FIXED_SIZE = struct.calcsize(_META_FIXED_FMT)  # 84 bytes

@dataclass
class VPMMetadata:
    # fixed
    version: int = 1
    kind: MapKind = MapKind.VPM
    level: int = 0
    agg_id: int = int(AggId.RAW)
    metric_count: int = 0
    doc_count: int = 0
    doc_block_size: int = 1
    task_hash: int = 0
    tile_id: bytes = field(default_factory=lambda: b"\x00" * 16)
    parent_id: bytes = field(default_factory=lambda: b"\x00" * 16)
    step_id: int = 0
    parent_step_id: int = 0
    timestamp_ns: int = 0
    # variable
    weights_nibbles: bytes = b""
    pointers: List[RouterPointer] = field(default_factory=list)

    # ---------- convenience constructors ----------

    @staticmethod
    def make_tile_id(payload: bytes, algo: str = "blake2s") -> bytes:
        """Stable 16-byte digest from payload."""
        if algo == "blake2s":
            return hashlib.blake2s(payload, digest_size=16).digest()
        elif algo == "md5":
            return hashlib.md5(payload).digest()
        else:
            return hashlib.blake2s(payload, digest_size=16).digest()

    @staticmethod
    def for_tile(*, level: int, metric_count: int, doc_count: int,
                 doc_block_size: int, agg_id: int,
                 metric_weights: Dict[str, float] | None,
                 metric_names: List[str],
                 task_hash: int,
                 tile_id: bytes,
                 parent_id: bytes = b"\x00"*16) -> "VPMMetadata":
        nibbles = _weights_to_nibbles(metric_weights or {}, metric_names)
        return VPMMetadata(
            version=1, kind=MapKind.VPM, level=level, agg_id=agg_id,
            metric_count=metric_count, doc_count=doc_count,
            doc_block_size=doc_block_size, task_hash=task_hash,
            tile_id=tile_id, parent_id=parent_id, weights_nibbles=nibbles
        )

    @staticmethod
    def for_router_frame(*, step_id: int, parent_step_id: int,
                         lane_weights: Dict[str, float], metric_names: List[str],
                         tile_id: bytes, parent_id: bytes,
                         level: int, timestamp_ns: int) -> "VPMMetadata":
        nibbles = _weights_to_nibbles(lane_weights, metric_names)
        return VPMMetadata(
            version=1, kind=MapKind.ROUTER_FRAME, level=level, agg_id=int(AggId.RAW),
            metric_count=len(metric_names), doc_count=0, doc_block_size=1,
            task_hash=0, tile_id=tile_id, parent_id=parent_id,
            step_id=step_id, parent_step_id=parent_step_id,
            timestamp_ns=timestamp_ns, weights_nibbles=nibbles
        )

    # ---------- (de)serialization ----------

    def to_bytes(self) -> bytes:
        ptr_count = len(self.pointers)
        weights_len = len(self.weights_nibbles)
        head = struct.pack(
            _META_FIXED_FMT,
            _META_MAGIC,
            self.version & 0xFF,
            int(self.kind) & 0xFF,
            0,  # reserved
            self.level & 0xFFFF,
            self.agg_id & 0xFFFF,
            self.metric_count & 0xFFFF,
            self.doc_count & 0xFFFFFFFF,
            self.doc_block_size & 0xFFFF,
            self.task_hash & 0xFFFFFFFF,
            self.tile_id,
            self.parent_id,
            self.step_id & 0xFFFFFFFFFFFFFFFF,
            self.parent_step_id & 0xFFFFFFFFFFFFFFFF,
            self.timestamp_ns & 0xFFFFFFFFFFFFFFFF,
            weights_len & 0xFFFF,
            ptr_count & 0xFFFF,
        )
        buf = bytearray()
        buf += head
        buf += self.weights_nibbles
        for p in self.pointers:
            buf += p.to_bytes()
        return bytes(buf)

    @staticmethod
    def from_bytes(b: bytes) -> "VPMMetadata":
        if len(b) < _META_FIXED_SIZE:
            raise ValueError("metadata payload too small")
        tup = struct.unpack(_META_FIXED_FMT, b[:_META_FIXED_SIZE])
        magic = tup[0]
        if magic != _META_MAGIC:
            raise ValueError("bad metadata magic")
        (
            _magic,
            ver, kind, _rsv,
            level, agg_id,
            metric_count,
            doc_count,
            doc_block_size,
            task_hash,
            tile_id,
            parent_id,
            step_id, parent_step_id, timestamp_ns,
            weights_len, ptr_count,
        ) = tup
        cursor = _META_FIXED_SIZE
        weights_nibbles = b[cursor:cursor+weights_len] if weights_len else b""
        cursor += weights_len
        pointers: List[RouterPointer] = []
        for _ in range(ptr_count):
            block = b[cursor:cursor+_ROUTER_PTR_SIZE]
            if len(block) != _ROUTER_PTR_SIZE:
                raise ValueError("truncated router pointer block")
            pointers.append(RouterPointer.from_bytes(block))
            cursor += _ROUTER_PTR_SIZE
        return VPMMetadata(
            version=ver, kind=MapKind(kind), level=level, agg_id=agg_id,
            metric_count=metric_count, doc_count=doc_count,
            doc_block_size=doc_block_size, task_hash=task_hash,
            tile_id=tile_id, parent_id=parent_id,
            step_id=step_id, parent_step_id=parent_step_id,
            timestamp_ns=timestamp_ns, weights_nibbles=weights_nibbles,
            pointers=pointers
        )

    # ---------- integration helpers ----------

    def set_weights(self, weights: Dict[str, float], metric_names: List[str]) -> None:
        self.weights_nibbles = _weights_to_nibbles(weights, metric_names)
        self.metric_count = len(metric_names)

    def get_weights(self, metric_names: List[str], default: float = 0.5) -> Dict[str, float]:
        if not self.weights_nibbles:
            return {m: float(default) for m in metric_names}
        return _nibbles_to_weights(self.weights_nibbles, metric_names)

    def add_pointer(self, ptr: RouterPointer) -> None:
        self.pointers.append(ptr)

    # file naming via resolvers
    def resolve_child_paths(self, resolver: TargetResolver) -> List[Tuple[RouterPointer, Optional[str]]]:
        return [(ptr, resolver.resolve(ptr.tile_id)) for ptr in self.pointers]
