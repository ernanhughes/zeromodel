# zeromodel/vpm/image.py
"""
VPM-IMG v1 — Image-only Pixel-Parametric Memory Implementation

This module implements the VPM-IMG v1 specification for storing multi-metric
score matrices in standard PNG files with built-in metadata and virtual reordering
capabilities. The format enables efficient hierarchical aggregation and fast
access to critical regions without modifying the original image.

Key Features:
- All-in-image storage (no external metadata files)
- 16-bit RGB PNG format (portable, lossless)
- Built-in document hierarchy with aggregation
- Virtual reordering without data movement
- Efficient critical tile extraction
"""

import numpy as np
import png
from typing import Optional, Dict, Tuple, List

# --- Constants ---
MAGIC = [ord('V'), ord('P'), ord('M'), ord('1')]  # ASCII for 'VPM1'
VERSION = 1                                      # Format version
META_MIN_COLS = 12                               # Minimum columns for metadata
DEFAULT_H_META_BASE = 2                          # Base metadata rows (row0 + row1)

# Aggregation types
AGG_MAX = 0       # Maximum aggregation
AGG_MEAN = 1      # Mean aggregation
AGG_RAW = 65535   # Base level (no aggregation)

# --- Helper Functions ---

def _u16_clip(a: np.ndarray) -> np.ndarray:
    """Clip values to 16-bit unsigned integer range [0, 65535]"""
    return np.clip(a, 0, 65535).astype(np.uint16)

def _round_u16(a: np.ndarray) -> np.ndarray:
    """Round and convert to 16-bit unsigned integers"""
    return np.round(a).astype(np.uint16)

def _check_header_width(D: int):
    """Validate image width meets metadata requirements"""
    if D < META_MIN_COLS:
        raise ValueError(f"VPM-IMG requires width D≥{META_MIN_COLS} for header; got D={D}")


# --- Writer Class ---

class VPMImageWriter:
    """
    Writes multi-metric score matrices to VPM-IMG v1 format PNG files.
    
    Supports hierarchical storage with configurable aggregation methods:
    - Base level (AGG_RAW): Original document-level scores
    - Aggregated levels: Coarser representations for efficient visualization
    
    Attributes:
        score_matrix (np.ndarray): Input scores (M x D)
        metric_names (list): Optional metric identifiers
        store_minmax (bool): Store per-metric min/max for denormalization
        compression (int): PNG compression level (0-9)
        M (int): Number of metrics
        D (int): Number of documents
        level (int): Hierarchy level (0 = coarsest)
        doc_block_size (int): Documents aggregated per pixel at this level
        agg_id (int): Aggregation method (AGG_RAW, AGG_MAX, AGG_MEAN)
    """

    def __init__(
        self,
        score_matrix: np.ndarray,           # shape (M, D)
        metric_names: Optional[list[str]] = None,
        metadata_bytes: Optional[bytes] = None,
        store_minmax: bool = False,
        compression: int = 6,
        # Hierarchy parameters:
        level: int = 0,                     # 0 = coarsest level
        doc_block_size: int = 1,            # Documents per pixel
        agg_id: int = AGG_RAW,              # Aggregation method
    ):
        # Validate and store input parameters
        self.score_matrix = np.asarray(score_matrix, dtype=np.float64)
        self.metric_names = metric_names or []
        self.store_minmax = store_minmax
        self.compression = int(compression)
        self.M, self.D = self.score_matrix.shape
        self.metadata_bytes = metadata_bytes or b""
        
        # Validate dimensions
        if self.M <= 0 or self.D <= 0:
            raise ValueError("Score matrix must be non-empty (M,D > 0)")
        if self.M > 65535 or self.D > 0xFFFFFFFF:
            raise ValueError("Dimensions exceed format limits (M≤65535, D≤2^32-1)")
        _check_header_width(self.D)

        # Hierarchy configuration
        self.level = int(level)
        self.doc_block_size = int(doc_block_size)
        self.agg_id = int(agg_id)

    def _meta_capacity_per_row(self, start_col: int) -> int:
        # 4 bytes/column using 16-bit G and B channels (2 bytes each)
        usable_cols = max(0, self.D - start_col)
        return usable_cols * 4

    def _extra_meta_rows_needed(self, payload_len: int, start_col: int = 7) -> int:
        if payload_len <= 0:
            return 0
        first_row_cap = self._meta_capacity_per_row(start_col)
        if payload_len <= first_row_cap:
            return 0
        # spill rows use full width starting at col 0
        remaining = payload_len - first_row_cap
        row_cap = self._meta_capacity_per_row(0)
        return int(np.ceil(remaining / row_cap))

    def _embed_metadata_into_meta_rows(self, meta: np.ndarray) -> None:
        if not self.metadata_bytes:
            return
        payload = self.metadata_bytes
        L = len(payload)

        # Row 1 markers & length (R channel)
        meta[1, 1, 0] = ord('M')
        meta[1, 2, 0] = ord('E')
        meta[1, 3, 0] = ord('T')
        meta[1, 4, 0] = ord('A')
        meta[1, 5, 0] = (L >> 16) & 0xFFFF
        meta[1, 6, 0] = L & 0xFFFF

        # pack bytes into 16-bit words: word = (hi<<8)|lo
        def pack2(bi0, bi1):
            hi = payload[bi0] if bi0 < L else 0
            lo = payload[bi1] if bi1 < L else 0
            return (hi << 8) | lo

        # write Row 1 from col=7 using G (ch=1) and B (ch=2)
        row = 1
        col = 7
        idx = 0
        H = meta.shape[0]

        def write_pair(r, c, w_g, w_b):
            meta[r, c, 1] = w_g  # G channel
            meta[r, c, 2] = w_b  # B channel

        # first row (start at col 7)
        while idx < L and col < self.D:
            w_g = pack2(idx, idx+1); idx += 2
            w_b = pack2(idx, idx+1); idx += 2
            write_pair(row, col, w_g, w_b)
            col += 1

        # spill into extra meta rows, if any (start col 0)
        r = 2
        while idx < L and r < H:
            c = 0
            while idx < L and c < self.D:
                w_g = pack2(idx, idx+1); idx += 2
                w_b = pack2(idx, idx+1); idx += 2
                write_pair(r, c, w_g, w_b)
                c += 1
            r += 1


    def _normalize_scores(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Normalize scores to [0,1] range per metric.
        
        Returns:
            normalized: Scores scaled to [0,1]
            mins: Per-metric minimums (if store_minmax=True)
            maxs: Per-metric maximums (if store_minmax=True)
        """
        mins = self.score_matrix.min(axis=1, keepdims=True)
        maxs = self.score_matrix.max(axis=1, keepdims=True)
        spans = maxs - mins
        spans[spans == 0] = 1.0  # Avoid division by zero
        normalized = (self.score_matrix - mins) / spans
        
        if self.store_minmax:
            return normalized, mins.squeeze(1), maxs.squeeze(1)
        return normalized, None, None

    def _compute_percentiles(self, normalized: np.ndarray) -> np.ndarray:
        """
        Compute percentile ranks for each metric row.
        
        Uses double argsort technique:
        1. argsort(axis=1) gets rank positions
        2. Second argsort converts to rank order
        3. Scale to 16-bit range [0, 65535]
        
        Returns:
            uint16 array of percentile values
        """
        ranks = np.argsort(np.argsort(normalized, axis=1), axis=1)
        if self.D == 1:
            # Special case: single document
            percent = np.full_like(ranks, 65535 // 2, dtype=np.float64)
        else:
            percent = (ranks / (self.D - 1)).astype(np.float64) * 65535.0
        return _round_u16(percent)

    def _assemble_metadata(self, h_meta: int, 
                          mins: Optional[np.ndarray], 
                          maxs: Optional[np.ndarray]) -> np.ndarray:
        """
        Construct metadata section of the image.
        
        Args:
            h_meta: Total metadata rows
            mins: Per-metric minimums (if stored)
            maxs: Per-metric maximums (if stored)
            
        Returns:
            uint16 array of shape (h_meta, D, 3)
        """
        meta = np.zeros((h_meta, self.D, 3), dtype=np.uint16)

        # --- Row 0: Core metadata ---
        # Magic number (ASCII 'PPM1')
        for i, v in enumerate(MAGIC):
            meta[0, i, 0] = v
        
        # Version and dimensions
        meta[0, 4, 0] = VERSION
        meta[0, 5, 0] = np.uint16(self.M)  # Metric count
        meta[0, 6, 0] = np.uint16((self.D >> 16) & 0xFFFF)  # D_hi
        meta[0, 7, 0] = np.uint16(self.D & 0xFFFF)          # D_lo
        meta[0, 8, 0] = np.uint16(h_meta)  # Total metadata rows
        meta[0, 9, 0] = np.uint16(self.level)  # Hierarchy level
        meta[0, 10, 0] = np.uint16(min(self.doc_block_size, 0xFFFF))  # Docs per pixel
        meta[0, 11, 0] = np.uint16(self.agg_id)  # Aggregation method

        # --- Row 1: Normalization flag ---
        meta[1, 0, 0] = 1 if self.store_minmax else 0

        # --- Rows 2+: Min/Max values (Q16.16 fixed-point) ---
        if self.store_minmax and mins is not None and maxs is not None:
            # Convert to 32-bit fixed-point (16.16 format)
            mins_fixed = (np.asarray(mins) * 65536.0).astype(np.uint32)
            maxs_fixed = (np.asarray(maxs) * 65536.0).astype(np.uint32)
            
            for m in range(self.M):
                # MIN value storage
                min_col = m * 2
                min_row = 2 + (min_col // self.D)
                min_col_in_row = min_col % self.D
                meta[min_row, min_col_in_row, 0] = np.uint16(mins_fixed[m] >> 16)
                meta[min_row, min_col_in_row, 1] = np.uint16(mins_fixed[m] & 0xFFFF)

                # MAX value storage (next column)
                max_col = min_col + 1
                max_row = 2 + (max_col // self.D)
                max_col_in_row = max_col % self.D
                meta[max_row, max_col_in_row, 0] = np.uint16(maxs_fixed[m] >> 16)
                meta[max_row, max_col_in_row, 1] = np.uint16(maxs_fixed[m] & 0xFFFF)

        return meta

    def write(self, file_path: str) -> None:
        """
        Write score matrix to VPM-IMG v1 PNG file.
        
        Process:
        1. Normalize scores to [0,1]
        2. Compute percentile ranks
        3. Assemble metadata
        4. Combine with data section
        5. Write as 16-bit PNG
        
        Args:
            file_path: Output file path
        """
        # Step 1: Normalize and compute percentiles
        normalized, mins, maxs = self._normalize_scores()
        value_chan = _round_u16(normalized * 65535.0)
        percentile_chan = self._compute_percentiles(normalized)
        aux_chan = np.zeros_like(value_chan, dtype=np.uint16)  # B channel

        # before assembling meta, compute extra rows if needed
        minmax_rows = 0
        if self.store_minmax:
            num_words = self.M * 2
            minmax_rows = (num_words + self.D - 1) // self.D

        # compute extra rows for metadata payload
        extra_meta_rows = self._extra_meta_rows_needed(len(self.metadata_bytes), start_col=7)

        h_meta = DEFAULT_H_META_BASE + minmax_rows + extra_meta_rows
        meta = self._assemble_metadata(h_meta, mins, maxs)

        # finally embed metadata bytes into 'meta'
        self._embed_metadata_into_meta_rows(meta)

        data = np.stack([value_chan, percentile_chan, aux_chan], axis=-1)
        full = np.vstack([meta, data])  # (h_meta + M, D, 3)

        # Step 4: Prepare for PNG writing (flatten to 2D)
        rows = full.reshape(full.shape[0], -1)

        # Step 5: Write PNG file
        with open(file_path, "wb") as f:
            writer = png.Writer(
                width=self.D,
                height=full.shape[0],
                bitdepth=16,
                greyscale=False,
                compression=self.compression,
                planes=3,
            )
            writer.write(f, rows.tolist())


# --- Reader Class ---

class VPMImageReader:
    """
    Reads and interprets VPM-IMG v1 files.
    
    Provides:
    - Metadata extraction
    - Virtual reordering of documents
    - Critical tile extraction
    - Hierarchy navigation
    
    Attributes:
        image (np.ndarray): Image data (H, W, 3)
        M (int): Number of metrics
        D (int): Number of documents
        h_meta (int): Metadata rows
        version (int): Format version
        level (int): Hierarchy level
        doc_block_size (int): Documents per pixel
        agg_id (int): Aggregation method
        norm_flag (int): Normalization flag
        min_vals (np.ndarray): Per-metric minimums
        max_vals (np.ndarray): Per-metric maximums
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.image = None
        self.M = None
        self.D = None
        self.h_meta = None
        self.version = None
        self.level = 0
        self.doc_block_size = 1
        self.agg_id = AGG_RAW
        self.norm_flag = 0
        self.min_vals = None
        self.max_vals = None
        self._load_and_parse()

    def _load_and_parse(self):
        """Load PNG file and parse metadata"""
        # Load PNG data
        r = png.Reader(self.file_path)
        w, h, data, meta = r.read()
        
        # Validate format
        if meta.get("bitdepth") != 16 or meta.get("planes") != 3:
            raise ValueError("Only 16-bit RGB PNG supported.")
        
        # Convert to 3D numpy array (H, W, 3)
        arr = np.vstack(list(data)).astype(np.uint16)
        self.image = arr.reshape(h, w, 3)
        _check_header_width(w)

        # --- Parse Row 0 metadata (R channel) ---
        row0 = self.image[0, :, 0]
        
        # Magic number validation
        magic = bytes([row0[0], row0[1], row0[2], row0[3]]).decode("ascii")
        if magic != "VPM1":
            raise ValueError(f"Bad magic: {magic}")

        # Core metadata
        self.version = int(row0[4])
        if self.version != VERSION:
            raise ValueError(f"Unsupported version: {self.version}")

        self.M = int(row0[5])
        D_hi = int(row0[6])
        D_lo = int(row0[7])
        self.D = (D_hi << 16) | D_lo
        self.h_meta = int(row0[8])
        self.level = int(row0[9])
        self.doc_block_size = int(row0[10])
        self.agg_id = int(row0[11])

        # Validate image dimensions
        if self.image.shape[0] != (self.h_meta + self.M) or self.image.shape[1] != self.D:
            raise ValueError("Image dimensions mismatch header.")

        # --- Row 1: Normalization flag ---
        self.norm_flag = int(self.image[1, 0, 0])

        # --- Parse Min/Max values if present ---
        if self.norm_flag == 1:
            self.min_vals = np.zeros(self.M, dtype=np.float64)
            self.max_vals = np.zeros(self.M, dtype=np.float64)
            
            # Each metric uses two pixels (min and max)
            for m in range(self.M):
                # MIN value (Q16.16 fixed-point)
                min_col = m * 2
                min_row = 2 + (min_col // self.D)
                min_col_in_row = min_col % self.D
                min_high = self.image[min_row, min_col_in_row, 0]
                min_low  = self.image[min_row, min_col_in_row, 1]
                self.min_vals[m] = ((int(min_high) << 16) | int(min_low)) / 65536.0

                # MAX value (next column)
                max_col = min_col + 1
                max_row = 2 + (max_col // self.D)
                max_col_in_row = max_col % self.D
                max_high = self.image[max_row, max_col_in_row, 0]
                max_low  = self.image[max_row, max_col_in_row, 1]
                self.max_vals[m] = ((int(max_high) << 16) | int(max_low)) / 65536.0

    # --- Accessors ---
    
    @property
    def height(self) -> int:
        """Total image height (pixels)"""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Total image width (pixels)"""
        return self.image.shape[1]

    def get_metric_row_raw(self, metric_idx: int) -> np.ndarray:
        """
        Get raw pixel row for a metric.
        
        Args:
            metric_idx: Metric index (0-based)
            
        Returns:
            uint16 array of shape (D, 3) - (R, G, B) values
        """
        if metric_idx < 0 or metric_idx >= self.M:
            raise IndexError(f"Metric index out of range [0, {self.M-1}]")
        row_idx = self.h_meta + metric_idx
        return self.image[row_idx]

    def get_metric_values(self, metric_idx: int) -> np.ndarray:
        """
        Get normalized [0,1] values for a metric.
        
        Uses R channel directly without denormalization.
        """
        row = self.get_metric_row_raw(metric_idx)[:, 0].astype(np.float64)
        return row / 65535.0

    def get_metric_values_original(self, metric_idx: int) -> np.ndarray:
        """
        Reconstruct original values using min/max if available.
        
        Applies reverse normalization if min/max were stored.
        """
        norm = self.get_metric_values(metric_idx)
        if self.norm_flag == 1 and self.min_vals is not None and self.max_vals is not None:
            lo = self.min_vals[metric_idx]
            hi = self.max_vals[metric_idx]
            span = hi - lo
            if span == 0.0:
                return np.full_like(norm, lo)
            return lo + norm * span
        return norm

    def get_percentiles(self, metric_idx: int) -> np.ndarray:
        """Get percentile ranks [0,1] for a metric (G channel)"""
        row = self.get_metric_row_raw(metric_idx)[:, 1].astype(np.float64)
        return row / 65535.0

    # --- Virtual Ordering ---
    
    def virtual_order(
        self,
        metric_idx: Optional[int] = None,
        weights: Optional[Dict[int, float]] = None,
        top_k: Optional[int] = None,
        descending: bool = True,
    ) -> np.ndarray:
        """
        Generate document permutation based on sorting criteria.
        
        Supports:
        - Single metric ordering (optimized using G channel)
        - Composite score ordering (weighted sum of metrics)
        - Top-K retrieval (efficient partial sort)
        
        Args:
            metric_idx: Single metric to sort by
            weights: Dictionary of {metric_idx: weight} for composite scores
            top_k: Return only top K documents
            descending: Sort descending (highest first)
            
        Returns:
            Document indices in sorted order
        """
        # Single metric ordering
        if metric_idx is not None:
            # Use R channel values for sorting
            v = self.get_metric_row_raw(metric_idx)[:, 0].astype(np.int32)
            
            # Efficient top-K retrieval
            if top_k is not None and top_k < self.D:
                # Partial sort: partition then sort top-K
                idx = np.argpartition(-v, top_k-1)[:top_k]
                order = np.argsort(-v[idx])
                perm = idx[order]
            else:
                # Full sort
                perm = np.argsort(-v)
            
            # Handle ascending order
            if not descending:
                perm = perm[::-1]
            return perm

        # Composite score ordering
        if weights:
            composite = np.zeros(self.D, dtype=np.float64)
            for m, w in weights.items():
                if w:
                    composite += w * self.get_metric_values(m)
            
            # Efficient top-K retrieval
            if top_k is not None and top_k < self.D:
                idx = np.argpartition(-composite, top_k-1)[:top_k]
                order = np.argsort(-composite[idx])
                return idx[order]
            
            return np.argsort(-composite)

        raise ValueError("Must specify either metric_idx or weights")

    # --- Virtual View Extraction ---
    
    def get_virtual_view(
        self,
        metric_idx: Optional[int] = None,
        weights: Optional[Dict[int, float]] = None,
        x: int = 0,
        y: int = 0,
        width: int = 8,
        height: int = 8,
        descending: bool = True,
    ) -> np.ndarray:
        """
        Extract a viewport from virtually ordered documents.
        
        This is the core "critical tile" operation that enables efficient
        visualization without modifying the original image.
        
        Args:
            metric_idx: Metric for ordering (None for composite)
            weights: Weights for composite ordering
            x: Horizontal start in virtual order
            y: Vertical start (metric row offset)
            width: Viewport width (documents)
            height: Viewport height (metrics)
            descending: Sort order
            
        Returns:
            Image tile (height, width, 3) from the virtual view
        """
        # Get document permutation
        perm = self.virtual_order(
            metric_idx=metric_idx,
            weights=weights,
            top_k=x+width,
            descending=descending
        )
        
        # Select columns in virtual order
        cols = perm[x: x + width]
        
        # Select rows (metrics)
        row_start = self.h_meta + y
        row_end = min(self.h_meta + y + height, self.h_meta + self.M)
        
        # Extract and return viewport
        return self.image[row_start:row_end, cols, :]

    def read_metadata_bytes(self) -> bytes:
        # Check marker
        if self.image[1,1,0] != ord('M') or self.image[1,2,0] != ord('E') \
        or self.image[1,3,0] != ord('T') or self.image[1,4,0] != ord('A'):
            return b""

        L = ((int(self.image[1,5,0]) << 16) | int(self.image[1,6,0]))
        out = bytearray(L)

        # unpack helper (reverse of pack2)
        def unpack(word: int) -> tuple[int,int]:
            hi = (word >> 8) & 0xFF
            lo = word & 0xFF
            return hi, lo

        # Row 1 from col 7
        row = 1
        col = 7
        idx = 0
        H, W, _ = self.image.shape

        while idx < L and col < W:
            w_g = int(self.image[row, col, 1])
            w_b = int(self.image[row, col, 2])
            for b in unpack(w_g) + unpack(w_b):
                if idx < L:
                    out[idx] = b
                    idx += 1
            col += 1

        # spill rows from row=2, col=0
        r = 2
        while idx < L and r < self.h_meta:
            c = 0
            while idx < L and c < W:
                w_g = int(self.image[r, c, 1])
                w_b = int(self.image[r, c, 2])
                for b in unpack(w_g) + unpack(w_b):
                    if idx < L:
                        out[idx] = b
                        idx += 1
                c += 1
            r += 1

        return bytes(out)


# --- Hierarchy Builder ---

def build_parent_level_png(
    child_reader: VPMImageReader,
    out_path: str,
    K: int = 8,
    agg_id: int = AGG_MAX,
    compression: int = 6,
    level: Optional[int] = None,
) -> None:
    """
    Build parent level from child image through document aggregation.
    
    Creates a coarser representation by grouping documents into blocks:
    - Each pixel in parent represents K documents in child
    - Metrics are preserved at original resolution
    
    Supported aggregations:
    - AGG_MAX: Store maximum value + position hint
    - AGG_MEAN: Store mean value
    
    Args:
        child_reader: Reader for child level image
        out_path: Output path for parent PNG
        K: Documents per block (aggregation factor)
        agg_id: Aggregation method (AGG_MAX or AGG_MEAN)
        compression: PNG compression level
        level: Override parent level (default: child level - 1)
    """
    assert K >= 1, "Aggregation factor must be ≥1"
    M, D = child_reader.M, child_reader.D
    _check_header_width(D)

    # Calculate parent dimensions
    P = (D + K - 1) // K  # ceil(D/K) documents in parent

    # Extract child data (R channel only)
    child_data = child_reader.image[child_reader.h_meta:, :, :]
    R_child = child_data[:, :, 0].astype(np.uint16)

    # Initialize parent arrays
    R_parent = np.zeros((M, P), dtype=np.uint16)
    B_parent = np.zeros((M, P), dtype=np.uint16)  # Aux channel

    # Process each block
    for p in range(P):
        lo = p * K
        hi = min(D, lo + K)
        block = R_child[:, lo:hi]  # (M, block_size)

        if agg_id == AGG_MAX:
            # Maximum aggregation
            vmax = block.max(axis=1)
            R_parent[:, p] = vmax
            
            # Store relative position of maximum
            argm = block.argmax(axis=1)
            if hi - lo > 1:
                B_parent[:, p] = _round_u16((argm / (hi - lo - 1)) * 65535.0)
            else:
                B_parent[:, p] = 0

        elif agg_id == AGG_MEAN:
            # Mean aggregation
            vmean = np.round(block.mean(axis=1))
            R_parent[:, p] = _u16_clip(vmean)
            B_parent[:, p] = 0  # Unused

        else:
            raise ValueError(f"Unsupported agg_id: {agg_id}")

    # Compute percentiles for parent
    if P == 1:
        G_parent = np.full((M, P), 32767, dtype=np.uint16)  # Midpoint
    else:
        ranks = np.argsort(np.argsort(R_parent, axis=1), axis=1)
        G_parent = _round_u16((ranks / (P - 1)) * 65535.0)

    # Create and configure parent writer
    store_minmax = (child_reader.norm_flag == 1)
    writer = VPMImageWriter(
        score_matrix=(R_parent / 65535.0),
        store_minmax=False,  # Parents don't store min/max
        compression=compression,
        level=child_reader.level - 1 if level is None else level,
        doc_block_size=child_reader.doc_block_size * K,
        agg_id=agg_id,
    )

    # Assemble metadata (parents use simplified metadata)
    h_meta = DEFAULT_H_META_BASE
    meta = writer._assemble_metadata(h_meta, None, None)
    
    # Combine with data
    data = np.stack([R_parent, G_parent, B_parent], axis=-1)
    full = np.vstack([meta, data])

    # Write to PNG
    rows = full.reshape(full.shape[0], -1)
    with open(out_path, "wb") as f:
        png.Writer(
            width=full.shape[1],
            height=full.shape[0],
            bitdepth=16,
            greyscale=False,      
            compression=compression,
            planes=3,
        ).write(f, rows.tolist())