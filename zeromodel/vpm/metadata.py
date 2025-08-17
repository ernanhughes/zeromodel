"""
Visual Policy Map (VPM) Metadata System

Defines the binary format and data structures for VPM tile metadata, including:
- Tile identification and hierarchy tracking
- Aggregation methods and map types
- Weight encoding for metric prioritization
- Router pointers for navigation between tiles
- Timestamping and task tracking

The system enables:
- Lossless serialization/deserialization of metadata
- Efficient storage of weights in compact nibble format
- Hierarchical navigation through router pointers
- Consistent tile identification across systems
"""

import hashlib
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Protocol, Tuple

# ---------- Enum Definitions ----------


class MapKind(IntEnum):
    """
    Type of visual map represented by the tile.

    VPM: Standard Visual Policy Map tile containing document/metric data
    ROUTER_FRAME: Navigation frame containing pointers to child tiles
    SEARCH_VIEW: Orthogonal view for composite/manifold analysis
    """

    VPM = 0  # Canonical VPM tile
    ROUTER_FRAME = 1  # Navigation frame (All Right Cinema)
    SEARCH_VIEW = 2  # Composite/manifold view


class AggId(IntEnum):
    """Aggregation methods for hierarchical data"""

    MAX = 0  # Maximum value aggregation
    MEAN = 1  # Mean value aggregation
    RAW = 65535  # Base level (no aggregation)


# ---------- Resolver Protocols ----------


class TargetResolver(Protocol):
    """Protocol for resolving tile IDs to storage paths/handles"""

    def resolve(self, tile_id: bytes) -> Optional[str]:
        """
        Resolve a 16-byte tile ID to a storage path or handle.

        Args:
            tile_id: 16-byte tile identifier

        Returns:
            Storage path or None if not found
        """


@dataclass
class FilenameResolver:
    """
    Default filename-based resolver using pattern formatting.

    Generates filenames like: vpm_{hexid}_L{level}_B{block}.png
    """

    pattern: str = "vpm_{hexid}_L{level}_B{block}.png"
    default_level: int = 0
    default_block: int = 1

    def resolve(self, tile_id: bytes) -> Optional[str]:
        """Generate filename from tile ID using pattern"""
        hexid = tile_id.hex()
        return self.pattern.format(
            hexid=hexid, level=self.default_level, block=self.default_block
        )


@dataclass
class DictResolver:
    """In-memory resolver for testing purposes"""

    mapping: Dict[bytes, str]

    def resolve(self, tile_id: bytes) -> Optional[str]:
        """Lookup path in internal dictionary"""
        return self.mapping.get(tile_id)


# ---------- Router Pointer Implementation ----------
# 36-byte structure for child tile navigation

# Binary layout (big-endian):
#   0: kind (1 byte)
#   1: version/reserved (1 byte, currently 0x01)
#   2-3: level (u16)
#   4-7: x_offset (u32) - Start column in child's logical span
#   8-11: span (u32) - Docs represented by this column
#   12-15: doc_block_size (u32)
#   16-17: agg_id (u16)
#   18-33: tile_id digest (16 bytes)
#   34-35: reserved/padding (u16)

_ROUTER_PTR_FMT = ">BBHIIIH16sH"  # Big-endian format string
_ROUTER_PTR_SIZE = struct.calcsize(_ROUTER_PTR_FMT)  # 36 bytes


@dataclass
class RouterPointer:
    """
    Navigation pointer to a child tile.

    Attributes:
        kind: Type of child tile (MapKind)
        level: Hierarchy level of child
        x_offset: Start column in child's document space
        span: Number of documents represented
        doc_block_size: Document grouping size at child level
        agg_id: Aggregation method used (AggId)
        tile_id: 16-byte unique identifier for child tile
    """

    kind: MapKind
    level: int
    x_offset: int
    span: int
    doc_block_size: int
    agg_id: int
    tile_id: bytes  # 16-byte identifier

    def to_bytes(self) -> bytes:
        """Serialize to 36-byte binary format"""
        assert len(self.tile_id) == 16, "Tile ID must be 16 bytes"
        return struct.pack(
            _ROUTER_PTR_FMT,
            int(self.kind) & 0xFF,  # Ensure single byte
            0x01,  # Version/reserved byte
            self.level & 0xFFFF,  # Clamp to u16 range
            self.x_offset & 0xFFFFFFFF,
            self.span & 0xFFFFFFFF,
            self.doc_block_size & 0xFFFFFFFF,
            self.agg_id & 0xFFFF,
            self.tile_id,
            0,  # Padding
        )

    @staticmethod
    def from_bytes(b: bytes) -> "RouterPointer":
        """Deserialize from 36-byte binary data"""
        if len(b) != _ROUTER_PTR_SIZE:
            raise ValueError(
                f"Router pointer requires {_ROUTER_PTR_SIZE} bytes, got {len(b)}"
            )
        # Unpack fields according to format
        k, ver, lvl, xoff, span, block, agg, tid, _pad = struct.unpack(
            _ROUTER_PTR_FMT, b
        )
        return RouterPointer(MapKind(k), lvl, xoff, span, block, agg, tid)


# ---------- Weight Encoding Helpers ----------


def _weights_to_nibbles(weights: Dict[str, float], metric_names: List[str]) -> bytes:
    """
    Compress metric weights to 4-bit nibble format.

    Stores two weights per byte (4 bits per weight) in metric_names order.
    Weights are scaled to 0-15 range (4-bit precision).

    Args:
        weights: Dictionary of metric_name: weight
        metric_names: Ordering of metrics

    Returns:
        Compact byte string of nibble-encoded weights
    """
    out = bytearray()
    # Process metrics in pairs (two per byte)
    for i in range(0, len(metric_names), 2):
        byte_val = 0
        # High nibble (first metric)
        w0 = max(0.0, min(1.0, weights.get(metric_names[i], 0.0)))
        n0 = int(round(w0 * 15.0)) & 0x0F
        byte_val |= n0 << 4

        # Low nibble (second metric if exists)
        if i + 1 < len(metric_names):
            w1 = max(0.0, min(1.0, weights.get(metric_names[i + 1], 0.0)))
            n1 = int(round(w1 * 15.0)) & 0x0F
            byte_val |= n1

        out.append(byte_val)
    return bytes(out)


def _nibbles_to_weights(nibbles: bytes, metric_names: List[str]) -> Dict[str, float]:
    """
    Expand nibble-encoded weights to float dictionary.

    Args:
        nibbles: Compact byte string of weights
        metric_names: Ordered metric names

    Returns:
        Dictionary of metric_name: weight (0.0-1.0)
    """
    weights = {}
    for i, name in enumerate(metric_names):
        byte_idx = i // 2
        # Get containing byte if exists
        byte_val = nibbles[byte_idx] if byte_idx < len(nibbles) else 0

        # Extract correct nibble
        if i % 2 == 0:  # Even index: high nibble
            weight_val = (byte_val >> 4) & 0x0F
        else:  # Odd index: low nibble
            weight_val = byte_val & 0x0F

        # Convert to float in [0.0, 1.0]
        weights[name] = weight_val / 15.0
    return weights


# ---------- Core Metadata Class ----------

# Binary header layout (big-endian):
#  0-4:   magic "VMETA" (5B)
#  5:     version (u8)
#  6:     kind (u8) -> MapKind
#  7:     reserved (u8)
#  8-9:   level (u16)
#  10-11: agg_id (u16)
#  12-13: metric_count (u16)
#  14-17: doc_count (u32)
#  18-19: doc_block_size (u16)
#  20-23: task_hash (u32)
#  24-39: tile_id (16B)
#  40-55: parent_id (16B)
#  56-63: step_id (u64)
#  64-71: parent_step_id (u64)
#  72-79: timestamp_ns (u64)
#  80-81: weights_len_bytes (u16)
#  82-83: ptr_count (u16)
#  84-..: weights_nibbles (variable)
#  ..-..: router pointers (36B each)

_META_MAGIC = b"VMETA"
_META_FIXED_FMT = ">5s BBB HHH I H I 16s16s Q Q Q H H"
_META_FIXED_SIZE = struct.calcsize(_META_FIXED_FMT)  # 84 bytes


@dataclass
class VPMMetadata:
    """
    Comprehensive metadata container for VPM tiles.

    Represents all metadata associated with a VPM tile, including:
    - Identification and hierarchy information
    - Aggregation and map type
    - Metric weights
    - Navigation pointers
    - Temporal and task context

    Supports binary serialization/deserialization for efficient storage.
    """

    # --- Fixed Header Fields ---
    version: int = 1  # Format version
    kind: MapKind = MapKind.VPM  # Tile type (VPM, ROUTER_FRAME, etc.)
    level: int = 0  # Hierarchy level (0 = coarsest)
    agg_id: int = int(AggId.RAW)  # Aggregation method
    metric_count: int = 0  # Number of metrics
    doc_count: int = 0  # Number of documents
    doc_block_size: int = 1  # Document grouping factor
    task_hash: int = 0  # Contextual task identifier
    tile_id: bytes = field(default_factory=lambda: b"\x00" * 16)  # 16-byte unique ID
    parent_id: bytes = field(default_factory=lambda: b"\x00" * 16)  # Parent tile ID
    step_id: int = 0  # Current processing step
    parent_step_id: int = 0  # Parent's processing step
    timestamp_ns: int = 0  # Timestamp in nanoseconds

    # --- Variable Content ---
    weights_nibbles: bytes = b""  # Compact weight storage (nibbles)
    pointers: List[RouterPointer] = field(default_factory=list)  # Child pointers

    # ---------- Factory Methods ----------

    @staticmethod
    def make_tile_id(payload: bytes, algo: str = "blake2s") -> bytes:
        """
        Generate 16-byte tile ID from content payload.

        Args:
            payload: Tile content bytes
            algo: Hashing algorithm (blake2s or md5)

        Returns:
            16-byte tile identifier
        """
        if algo == "blake2s":
            return hashlib.blake2s(payload, digest_size=16).digest()
        elif algo == "md5":
            return hashlib.md5(payload).digest()
        else:
            # Default to BLAKE2s
            return hashlib.blake2s(payload, digest_size=16).digest()

    @staticmethod
    def for_tile(
        *,
        level: int,
        metric_count: int,
        doc_count: int,
        doc_block_size: int,
        agg_id: int,
        metric_weights: Dict[str, float] | None,
        metric_names: List[str],
        task_hash: int,
        tile_id: bytes,
        parent_id: bytes = b"\x00" * 16,
    ) -> "VPMMetadata":
        """
        Create metadata for a standard VPM tile.

        Args:
            level: Hierarchy level
            metric_count: Number of metrics
            doc_count: Number of documents
            doc_block_size: Document grouping size
            agg_id: Aggregation method ID
            metric_weights: Metric weights dictionary
            metric_names: Ordered metric names
            task_hash: Contextual task hash
            tile_id: 16-byte tile identifier
            parent_id: 16-byte parent tile identifier

        Returns:
            Configured VPMMetadata instance
        """
        nibbles = _weights_to_nibbles(metric_weights or {}, metric_names)
        return VPMMetadata(
            version=1,
            kind=MapKind.VPM,
            level=level,
            agg_id=agg_id,
            metric_count=metric_count,
            doc_count=doc_count,
            doc_block_size=doc_block_size,
            task_hash=task_hash,
            tile_id=tile_id,
            parent_id=parent_id,
            weights_nibbles=nibbles,
        )

    @staticmethod
    def for_router_frame(
        *,
        step_id: int,
        parent_step_id: int,
        lane_weights: Dict[str, float],
        metric_names: List[str],
        tile_id: bytes,
        parent_id: bytes,
        level: int,
        timestamp_ns: int,
    ) -> "VPMMetadata":
        """
        Create metadata for a router frame tile.

        Args:
            step_id: Current processing step
            parent_step_id: Parent processing step
            lane_weights: Weighting for navigation lanes
            metric_names: Ordered metric names
            tile_id: 16-byte tile identifier
            parent_id: 16-byte parent tile identifier
            level: Hierarchy level
            timestamp_ns: Creation timestamp (nanoseconds)

        Returns:
            Configured VPMMetadata instance
        """
        nibbles = _weights_to_nibbles(lane_weights, metric_names)
        return VPMMetadata(
            version=1,
            kind=MapKind.ROUTER_FRAME,
            level=level,
            agg_id=int(AggId.RAW),
            metric_count=len(metric_names),
            doc_count=0,  # Router frames have no documents
            doc_block_size=1,
            task_hash=0,  # Not used in router frames
            tile_id=tile_id,
            parent_id=parent_id,
            step_id=step_id,
            parent_step_id=parent_step_id,
            timestamp_ns=timestamp_ns,
            weights_nibbles=nibbles,
        )

    # ---------- Serialization Methods ----------

    def to_bytes(self) -> bytes:
        """Serialize metadata to binary format"""
        # Calculate variable section sizes
        ptr_count = len(self.pointers)
        weights_len = len(self.weights_nibbles)

        # Pack fixed header
        head = struct.pack(
            _META_FIXED_FMT,
            _META_MAGIC,
            self.version & 0xFF,  # Ensure single byte
            int(self.kind) & 0xFF,  # Convert enum to byte
            0,  # Reserved byte
            self.level & 0xFFFF,  # Clamp to u16
            self.agg_id & 0xFFFF,
            self.metric_count & 0xFFFF,
            self.doc_count & 0xFFFFFFFF,  # u32
            self.doc_block_size & 0xFFFF,
            self.task_hash & 0xFFFFFFFF,
            self.tile_id,
            self.parent_id,
            self.step_id & 0xFFFFFFFFFFFFFFFF,  # u64
            self.parent_step_id & 0xFFFFFFFFFFFFFFFF,
            self.timestamp_ns & 0xFFFFFFFFFFFFFFFF,
            weights_len & 0xFFFF,  # u16
            ptr_count & 0xFFFF,
        )

        # Build complete binary representation
        buf = bytearray()
        buf += head  # Fixed header
        buf += self.weights_nibbles  # Weight nibbles
        for p in self.pointers:  # Router pointers
            buf += p.to_bytes()
        return bytes(buf)

    @staticmethod
    def from_bytes(b: bytes) -> "VPMMetadata":
        """Deserialize metadata from binary format"""
        if len(b) < _META_FIXED_SIZE:
            raise ValueError(f"Metadata too small ({len(b)} < {_META_FIXED_SIZE})")

        # Unpack fixed header
        tup = struct.unpack(_META_FIXED_FMT, b[:_META_FIXED_SIZE])
        magic = tup[0]
        if magic != _META_MAGIC:
            raise ValueError(f"Invalid metadata magic: {magic!r}")

        # Extract fixed fields
        (
            _magic,
            ver,
            kind,
            _rsv,
            level,
            agg_id,
            metric_count,
            doc_count,
            doc_block_size,
            task_hash,
            tile_id,
            parent_id,
            step_id,
            parent_step_id,
            timestamp_ns,
            weights_len,
            ptr_count,
        ) = tup

        # Process variable sections
        cursor = _META_FIXED_SIZE

        # Weight nibbles
        weights_nibbles = b[cursor : cursor + weights_len] if weights_len else b""
        cursor += weights_len

        # Router pointers
        pointers = []
        for _ in range(ptr_count):
            if cursor + _ROUTER_PTR_SIZE > len(b):
                raise ValueError("Truncated router pointers")
            block = b[cursor : cursor + _ROUTER_PTR_SIZE]
            pointers.append(RouterPointer.from_bytes(block))
            cursor += _ROUTER_PTR_SIZE

        return VPMMetadata(
            version=ver,
            kind=MapKind(kind),
            level=level,
            agg_id=agg_id,
            metric_count=metric_count,
            doc_count=doc_count,
            doc_block_size=doc_block_size,
            task_hash=task_hash,
            tile_id=tile_id,
            parent_id=parent_id,
            step_id=step_id,
            parent_step_id=parent_step_id,
            timestamp_ns=timestamp_ns,
            weights_nibbles=weights_nibbles,
            pointers=pointers,
        )

    # ---------- Utility Methods ----------

    def set_weights(self, weights: Dict[str, float], metric_names: List[str]) -> None:
        """
        Set metric weights using nibble encoding.

        Args:
            weights: Metric weights dictionary
            metric_names: Ordered metric names
        """
        self.weights_nibbles = _weights_to_nibbles(weights, metric_names)
        self.metric_count = len(metric_names)

    def get_weights(
        self, metric_names: List[str], default: float = 0.5
    ) -> Dict[str, float]:
        """
        Retrieve metric weights from nibble encoding.

        Args:
            metric_names: Ordered metric names
            default: Default weight if not specified

        Returns:
            Dictionary of metric weights
        """
        if not self.weights_nibbles:
            return dict.fromkeys(metric_names, default)
        return _nibbles_to_weights(self.weights_nibbles, metric_names)

    def add_pointer(self, ptr: RouterPointer) -> None:
        """Add a router pointer to child tile"""
        self.pointers.append(ptr)

    def resolve_child_paths(
        self, resolver: TargetResolver
    ) -> List[Tuple[RouterPointer, Optional[str]]]:
        """
        Resolve all child pointers to storage paths.

        Args:
            resolver: TargetResolver implementation

        Returns:
            List of (pointer, resolved_path) tuples
        """
        return [(ptr, resolver.resolve(ptr.tile_id)) for ptr in self.pointers]

    def validate(self) -> None:
        """Validate metadata integrity"""
        # Validate ID lengths
        if len(self.tile_id) != 16:
            raise ValueError("tile_id must be 16 bytes")
        if len(self.parent_id) != 16:
            raise ValueError("parent_id must be 16 bytes")

        # Validate counts
        if self.metric_count < 0:
            raise ValueError("metric_count cannot be negative")
        if self.doc_count < 0:
            raise ValueError("doc_count cannot be negative")
            
        # Validate weight encoding length
        max_nibbles = (self.metric_count + 1) // 2
        if len(self.weights_nibbles) > max_nibbles:
            raise ValueError("weights_nibbles exceeds expected length")
            
        # Validate router pointers
        for p in self.pointers:
            if len(p.tile_id) != 16:
                raise ValueError("Child tile_id must be 16 bytes")