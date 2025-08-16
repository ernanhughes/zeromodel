# zeromodel/hierarchical.py
"""
Hierarchical Visual Policy Map (HVPM) implementation for world-scale navigation.

This module provides the HierarchicalVPM class for creating a pyramid structure
where navigation time grows logarithmically with data size, enabling:
- Planet-scale navigation that feels flat (40 hops for 1 trillion documents)
- Edge-to-cloud symmetry (same artifact format at all levels)
- Visual reasoning through spatial organization
- Deterministic provenance with VPF embedding

The core insight: "When the answer is always 40 steps away, size becomes irrelevant."
"""

import logging
import json
import math
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from PIL import Image

from zeromodel.vpm.encoder import VPMEncoder
from zeromodel.images.metadata import VPF_FOOTER_MAGIC
from zeromodel.images import create_vpf, extract_vpf
from zeromodel.core import ZeroModel
from zeromodel.utils import to_png_bytes, png_to_gray_array

from zeromodel.storage.base import StorageBackend
from zeromodel.storage.in_memory import InMemoryStorage

logger = logging.getLogger(__name__)


class HierarchicalVPM:
    """
    Hierarchical Visual Policy Map (HVPM) implementation for world-scale navigation.

    This class creates a pyramid structure where navigation time grows logarithmically
    with data size, enabling planet-scale navigation that feels flat:

    - 1 million documents → ~20 hops
    - 1 trillion documents → ~40 hops
    - All-world data → ~50 hops

    The core innovation: "When the answer is always 40 steps away, size becomes irrelevant."

    Key features:
    - Storage-agnostic design (works with in-memory, S3, databases)
    - Lazy loading of tiles (only loads what's needed)
    - Spatial calculus for signal concentration I saw him instantly
    - Logarithmic navigation with consistent performance
    - Built-in provenance with VPF embedding
    """

    def __init__(
        self,
        metric_names: List[str],
        num_levels: int = 5,
        zoom_factor: int = 4,
        precision: Union[int, str] = 8,  
        storage_backend: Optional[StorageBackend] = None,
        tile_size: int = 256,
    ):
        """
        Initialize the hierarchical VPM system.

        Args:
            metric_names: Names of all metrics being tracked.
            num_levels: Number of hierarchical levels (default 5).
            zoom_factor: Zoom factor between levels (default 4).
            precision: Bit precision for encoding (4-16).
            storage_backend: Optional backend for storing VPM tiles.
            tile_size: Size of each tile in pixels (default 256).

        Raises:
            ValueError: If inputs are invalid.
        """
        logger.debug(
            f"Initializing HierarchicalVPM with metrics: {metric_names}, "
            f"levels: {num_levels}, zoom: {zoom_factor}, precision: {precision}"
        )

        # Validate inputs
        if num_levels <= 0:
            error_msg = f"num_levels must be positive, got {num_levels}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if zoom_factor <= 1:
            error_msg = f"zoom_factor must be greater than 1, got {zoom_factor}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not (4 <= precision <= 16):
            error_msg = f"precision must be between 4-16, got {precision}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize properties
        self.metric_names = list(metric_names)
        self.num_levels = num_levels
        self.zoom_factor = zoom_factor
        self.precision = str(precision)
        self.tile_size = tile_size
        self.storage = storage_backend or InMemoryStorage()

        # Level metadata - will be populated during processing
        self.levels: List[Optional[Dict[str, Any]]] = [None] * num_levels

        # System metadata
        self.metadata: Dict[str, Any] = {
            "version": "1.0",
            "temporal_axis": False,
            "levels": num_levels,
            "zoom_factor": zoom_factor,
            "metric_names": self.metric_names,
            "tile_size": tile_size,
            "total_documents": None,
            "task": None,
        }

        logger.info(
            f"HierarchicalVPM initialized with {num_levels} levels, "
            f"zoom factor {zoom_factor}, precision {precision} bits"
        )

    def process(
        self,
        data_source: Union[np.ndarray, Callable[[int, int], np.ndarray]],
        task: str,
        total_documents: Optional[int] = None,
    ) -> None:
        """
        Process data into hierarchical visual policy maps using streaming approach.

        Args:
            data_source: Either a full score matrix or a callable that fetches chunks
                         of data (for world-scale operation).
            task: SQL query defining the task (e.g., "ORDER BY uncertainty DESC").
            total_documents: Total number of documents (required for world-scale).

        Raises:
            ValueError: If inputs are invalid or processing fails.
        """
        logger.info(f"Starting hierarchical processing for task: '{task}'")
        self._task = task
        if isinstance(data_source, np.ndarray):
            if data_source.ndim != 2:
                raise ValueError(
                    f"data_source must be a 2D array (documents x metrics); "
                    f"got {data_source.ndim}D with shape {getattr(data_source, 'shape', None)}"
                )
            rows, cols = data_source.shape
            # NEW: disallow empty dimensions
            if rows == 0 or cols == 0:
                raise ValueError(
                    f"data_source must be non-empty 2D; got shape {data_source.shape}"
                )
        elif not callable(data_source):
            raise TypeError(
                "data_source must be either a 2D numpy array or a callable that returns a 2D numpy array per tile."
            )

        # Determine total documents if not provided
        if total_documents is None:
            if isinstance(data_source, np.ndarray):
                total_documents = data_source.shape[0]
            else:
                raise ValueError(
                    "total_documents must be provided when using callable data source"
                )

        # Update metadata
        self.metadata["task"] = task
        self.metadata["total_documents"] = total_documents
        logger.debug(
            f"Updated metadata: documents={total_documents}, metrics={len(self.metric_names)}"
        )

        # Clear existing levels
        self.levels = [None] * self.num_levels
        logger.debug("Cleared existing levels.")

        # Create base level (Level num_levels-1: Full detail)
        base_level = self.num_levels - 1
        self._create_base_level(data_source, task, total_documents, base_level)
        logger.info(f"Base level (L{base_level}) created")

        # Create higher levels incrementally
        for level in range(base_level - 1, -1, -1):
            self._create_summary_level(level, level + 1)
            logger.info(f"Summary level (L{level}) created")

        logger.info("Hierarchical VPM processing complete")

    def _create_base_level(
        self,
        data_source: Union[np.ndarray, Callable[[int, int], np.ndarray]],
        task: str,
        total_documents: int,
        level_index: int,
    ) -> None:
        """
        Create the base level (highest detail) using streaming processing.

        Args:
            data_source: Data source for the base level.
            task: SQL query defining the task.
            total_documents: Total number of documents.
            level_index: Index of this level in the hierarchy.
        """
        logger.debug(
            f"Creating base level (L{level_index}) with {total_documents} documents"
        )

        # Create level metadata
        level_data = {
            "level": level_index,
            "type": "base",
            "tile_size": self.tile_size,
            "num_tiles_x": math.ceil(total_documents / self.tile_size),
            "num_tiles_y": math.ceil(len(self.metric_names) / self.tile_size),
            "storage_key": f"level_{level_index}",
            "metadata": {
                "documents": total_documents,
                "metrics": len(self.metric_names),
                "task": task,
            },
        }

        # Create spatial index for navigation
        self.storage.create_index(level_index, "spatial")

        # Process in tiles to avoid memory issues
        for tile_x in range(level_data["num_tiles_x"]):
            for tile_y in range(level_data["num_tiles_y"]):
                self._create_base_tile(tile_x, tile_y, data_source, task, level_index)

        # Store level metadata
        self.levels[level_index] = level_data

    def _create_base_tile(
        self,
        tile_x: int,
        tile_y: int,
        data_source: Union[np.ndarray, Callable[[int, int], np.ndarray]],
        task: str,
        level_index: int,
    ) -> None:
        """
        Create a single base level tile from the data source.

        Args:
            tile_x: X coordinate of the tile.
            tile_y: Y coordinate of the tile.
            data_source: Data source for the base level.
            task: SQL query defining the task.
            level_index: Index of this level in the hierarchy.
        """

        # Calculate document and metric ranges
        doc_start = tile_x * self.tile_size
        doc_end = min((tile_x + 1) * self.tile_size, self.metadata["total_documents"])
        metric_start = tile_y * self.tile_size
        metric_end = min((tile_y + 1) * self.tile_size, len(self.metric_names))

        logger.debug(
            f"Creating base tile L{level_index}_X{tile_x}_Y{tile_y} "
            f"with docs {doc_start}-{doc_end - 1}, metrics {metric_start}-{metric_end - 1}"
        )

        # Fetch data chunk
        if isinstance(data_source, np.ndarray):
            # For small datasets, slice the array directly
            chunk = data_source[doc_start:doc_end, metric_start:metric_end]
        else:
            # For world-scale, use the callable to fetch just this chunk
            chunk = data_source(doc_start, doc_end, metric_start, metric_end)

        # Create a ZeroModel for this tile
        tile_metric_names = self.metric_names[metric_start:metric_end]
        zeromodel = ZeroModel(tile_metric_names)
        zeromodel.precision = self.precision

        # Prepare the tile with spatial organization
        zeromodel.prepare(chunk, task)
        # --- inside _create_base_tile, after zeromodel.prepare(chunk, task) ---

        # Top doc within this chunk after task ordering
        top_doc_chunk = int(zeromodel.doc_order[0]) if (
            zeromodel.doc_order is not None and len(zeromodel.doc_order) > 0
        ) else 0

        # Convert to global doc index
        top_doc_global = doc_start + top_doc_chunk

        # Encode as VPM
        vpm_image = VPMEncoder(self.precision).encode(zeromodel.sorted_matrix)

        # Create VPF for provenance
        vpf = create_vpf(
            pipeline={"graph_hash": "sha3:base-level", "step": "spatial-organization"},
            model={"id": "zero-1.0", "assets": {}},
            determinism={"seed": 0, "rng_backends": ["numpy"]},
            params={
                "tile": f"L{level_index}_X{tile_x}_Y{tile_y}",
                "doc_start": doc_start,
                "doc_end": doc_end,
            },
            inputs={"task": task},
            metrics={
                "documents": doc_end - doc_start,
                "metrics": metric_end - metric_start,
                "top_doc_global": top_doc_global,        # <-- add this
                "top_doc_chunk": top_doc_chunk,          # <-- optional, for debugging
            },
            lineage={"parents": []},
        )

        # Embed VPF in the VPM
        png_bytes = embed_vpf(vpm_image, vpf)

        # Store the tile
        tile_id = self.storage.store_tile(level_index, tile_x, tile_y, png_bytes)
        logger.debug(f"Stored base tile with ID: {tile_id}")

    def _create_summary_level(self, target_level: int, source_level: int) -> None:
        """
        Create a summary level from the level below it.

        Args:
            target_level: Level to create.
            source_level: Level to summarize from.
        """
        logger.debug(f"Creating summary level L{target_level} from L{source_level}")

        # Get source level metadata
        source_meta = self.levels[source_level]
        if source_meta is None:
            raise ValueError(f"Source level {source_level} not created yet")

        # Calculate target level dimensions
        num_tiles_x = max(1, math.ceil(source_meta["num_tiles_x"] / self.zoom_factor))
        num_tiles_y = max(1, math.ceil(source_meta["num_tiles_y"] / self.zoom_factor))

        # Create level metadata
        level_data = {
            "level": target_level,
            "type": "summary",
            "tile_size": source_meta["tile_size"],
            "num_tiles_x": num_tiles_x,
            "num_tiles_y": num_tiles_y,
            "storage_key": f"level_{target_level}",
            "source_level": source_level,
            "metadata": {
                "documents": source_meta["metadata"]["documents"],
                "metrics": source_meta["metadata"]["metrics"],
            },
        }

        # Create spatial index
        self.storage.create_index(target_level, "spatial")

        # Process in tiles
        for tile_x in range(num_tiles_x):
            for tile_y in range(num_tiles_y):
                self._create_summary_tile(
                    target_level, tile_x, tile_y, source_level, source_meta
                )

        # Store level metadata
        self.levels[target_level] = level_data

    def _create_summary_tile(
        self,
        target_level: int,
        target_x: int,
        target_y: int,
        source_level: int,
        source_meta: Dict[str, Any],
    ) -> None:
        """
        Create a single summary tile by aggregating source tiles.

        Args:
            target_level: Target level index.
            target_x: X coordinate in target level.
            target_y: Y coordinate in target level.
            source_level: Source level index.
            source_meta: Metadata of the source level.
        """
        # Calculate source region
        source_x_start = target_x * self.zoom_factor
        source_x_end = min(
            (target_x + 1) * self.zoom_factor, source_meta["num_tiles_x"]
        )
        source_y_start = target_y * self.zoom_factor
        source_y_end = min(
            (target_y + 1) * self.zoom_factor, source_meta["num_tiles_y"]
        )

        logger.debug(
            f"Creating summary tile L{target_level}_X{target_x}_Y{target_y} "
            f"from source region X{source_x_start}-{source_x_end - 1}, "
            f"Y{source_y_start}-{source_y_end - 1}"
        )

        # Fetch source tiles
        source_tiles = []
        for source_x in range(source_x_start, source_x_end):
            for source_y in range(source_y_start, source_y_end):
                tile_id = self.storage.get_tile_id(source_level, source_x, source_y)
                png_bytes = self.storage.load_tile(tile_id)
                if png_bytes is not None:
                    source_tiles.append((source_x, source_y, png_bytes))

        if not source_tiles:
            logger.warning(
                f"No source tiles found for L{target_level}_X{target_x}_Y{target_y}"
            )
            return

        # Aggregate the source tiles using spatial calculus
        aggregated_data = self._aggregate_tiles(source_tiles, source_meta)

        if not isinstance(aggregated_data, np.ndarray) or aggregated_data.ndim != 2:
            logger.error("Aggregated data must be a 2D numpy array; got %r", type(aggregated_data))
            return
        # Let ZeroModel handle float conversion; this just avoids uint8 surprises in stats
        if aggregated_data.dtype != np.float32:
            aggregated_data = aggregated_data.astype(np.float32, copy=False)


        # Create a ZeroModel for the aggregated data
        zeromodel = ZeroModel(self.metric_names)
        zeromodel.precision = self.precision

        zeromodel.prepare(aggregated_data, sql_query=None)  # identity (no reordering)

        # Encode as VPM Oh God what's going on
        vpm_image = VPMEncoder(self.precision).encode(
           zeromodel.sorted_matrix if zeromodel.sorted_matrix is not None
            else zeromodel.canonical_matrix
        )

        # Create VPF for provenance
        vpf = create_vpf(
            pipeline={"graph_hash": "sha3:summary-level", "step": "aggregation"},
            model={"id": "zero-1.0", "assets": {}},
            determinism={"seed": 0, "rng_backends": ["numpy"]},
            params={"tile": f"L{target_level}_X{target_x}_Y{target_y}"},
            inputs={
                "source_tiles": [
                    self.storage.get_tile_id(source_level, x, y)
                    for x, y, _ in source_tiles
                ]
            },
            metrics={"source_tiles": len(source_tiles)},
            lineage={
                "parents": [
                    self.storage.get_tile_id(source_level, x, y)
                    for x, y, _ in source_tiles
                ]
            },
        )

        # Embed VPF in the VPM
        png_bytes = embed_vpf(vpm_image, vpf)

        # Store the summary tile
        self.storage.store_tile(target_level, target_x, target_y, png_bytes)

    def _aggregate_tiles(
        self, source_tiles: List[Tuple[int, int, bytes]], source_meta: Dict[str, Any]
    ) -> np.ndarray:
        """
        Aggregate multiple tiles using spatial calculus to preserve decision signal.
        
        Fixed to handle small datasets where critical region may be smaller than tile size.
        """
        logger.debug(f"Aggregating {len(source_tiles)} tiles for summary")
        
        # Decode all source tiles
        decoded_tiles = []
        for _, _, png_bytes in source_tiles:
            try:
                # Extract just the critical region (top-left) for aggregation
                critical_region = extract_critical_region(png_bytes, size=min(8, self.tile_size))
                decoded_tiles.append(critical_region)
            except Exception as e:
                logger.warning(f"Failed to decode tile: {e}")
        
        if not decoded_tiles:
            logger.error("No valid tiles to aggregate")
            return np.zeros((self.tile_size, self.tile_size))
        
        # Stack the critical regions
        stacked = np.stack(decoded_tiles)
        
        # Get actual dimensions of critical region
        num_tiles, crit_h, crit_w = stacked.shape
        
        # Calculate metric importance (variance across tiles)
        metric_importance = np.var(stacked, axis=0)
        
        # Sort metrics by importance (flatten the 2D array)
        sorted_indices = np.argsort(-metric_importance, axis=None)
        
        # Create output matrix
        output = np.zeros((self.tile_size, self.tile_size))
        
        # Fill output with aggregated data
        for i in range(min(self.tile_size * self.tile_size, len(sorted_indices))):
            flat_idx = sorted_indices[i]
            # Unravel using ACTUAL critical region dimensions
            y_src, x_src = np.unravel_index(flat_idx, (crit_h, crit_w))
            
            # Map to output coordinates (scale up proportionally)
            y_dest = int(y_src * self.tile_size / crit_h)
            x_dest = int(x_src * self.tile_size / crit_w)
            
            # Take the maximum value (preserves strong signals)
            output[y_dest, x_dest] = np.max(stacked[:, y_src, x_src])
        
        return output

    def navigate(
        self,
        start_level: int = 0,
        start_x: int = 0,
        start_y: int = 0,
        max_hops: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Navigate from a given tile down to the most relevant decision.

        Args:
            start_level: Level to start navigation from (0 = most abstract).
            start_x: X coordinate of starting tile.
            start_y: Y coordinate of starting tile.
            max_hops: Maximum number of hops (defaults to full navigation).
        """
        logger.info(f"Starting navigation from level {start_level}, coords=({start_x},{start_y})")

        if max_hops is None:
            max_hops = self.num_levels - start_level

        path = []
        current_level = start_level
        current_x, current_y = start_x, start_y  # <-- now configurable

        for _ in range(max_hops):
            if current_level >= self.num_levels - 1:
                break  # Already at base level

            tile_id = self.storage.get_tile_id(current_level, current_x, current_y)
            png_bytes = self.storage.load_tile(tile_id)

            if png_bytes is None:
                logger.warning(f"Tile not found: {tile_id}")
                break

            # fix
            next_level, rel_x, rel_y, relevance = self._analyze_tile(png_bytes, current_level)
            abs_x = current_x * self.zoom_factor + rel_x
            abs_y = current_y * self.zoom_factor + rel_y

            path.append({
                "level": current_level,
                "tile": (current_x, current_y),
                "next_level": next_level,
                "next_tile": (abs_x, abs_y),
                "relevance": relevance,
            })

            current_level = next_level
            current_x, current_y = abs_x, abs_y

        # Final decision step (unchanged)
        if current_level == self.num_levels - 1:
            tile_id = self.storage.get_tile_id(current_level, current_x, current_y)
            png_bytes = self.storage.load_tile(tile_id)
            if png_bytes is not None:
                try:
                    vpf_info = extract_vpf(png_bytes)
                    vpf = vpf_info[0] if isinstance(vpf_info, tuple) else vpf_info
                    doc_idx, relevance = self._extract_decision(png_bytes)
                    path.append({
                        "level": current_level,
                        "tile": (current_x, current_y),
                        "decision": int(doc_idx),
                        "relevance": float(relevance),
                        "vpf": vpf,
                    })
                except Exception as e:
                    logger.error(f"Failed to extract decision: {e}")

        logger.info(f"Navigation completed with {len(path)} steps")
        return path

    def _analyze_tile(
        self, png_bytes: bytes, level: int
    ) -> Tuple[int, int, int, float]:
        """
        Analyze a tile to determine where to navigate next.

        Args:
            png_bytes: Tile data in PNG format with embedded VPF.
            level: Current level in the hierarchy.

        Returns:
            (next_level, next_x, next_y, relevance)
        """
        # 1) Extract top-left critical region (auto-clipped to image size)
        region = extract_critical_region(png_bytes, size=8)

        # 2) Ensure grayscale for consistent intensity math
        if region.ndim == 3:  # e.g., (H, W, 3) RGB
            region = (
                0.299 * region[:, :, 0]
                + 0.587 * region[:, :, 1]
                + 0.114 * region[:, :, 2]
            )

        # 3) Compute importances along doc/metric axes
        doc_importance = np.sum(region, axis=1)  # shape: (h,)
        metric_importance = np.sum(region, axis=0)  # shape: (w,)

        top_doc = int(np.argmax(doc_importance))
        top_metric = int(np.argmax(metric_importance))

        # 4) Relevance = hottest pixel in the critical region, normalized
        critical_value = float(np.max(region)) / 255.0

        # 5) Map the hottest row/col to child tile coordinates on a zoom_factor grid
        next_level = level + 1

        h, w = region.shape

        def _scale_to_child(idx: int, dim: int, z: int) -> int:
            if z <= 1 or dim <= 1:
                return 0
            # Use center-of-bin to avoid bias at edges
            pos = (idx + 0.5) / dim  # in (0,1]
            j = int(pos * z)
            # clamp into [0, z-1]
            return max(0, min(z - 1, j))

        next_x = _scale_to_child(top_doc, h, self.zoom_factor)
        next_y = _scale_to_child(top_metric, w, self.zoom_factor)

        return (next_level, next_x, next_y, critical_value)

    def _extract_decision(self, png_bytes: bytes) -> Tuple[int, float]:
        """
        Return (global_doc_idx, relevance in [0,1]) from a base-level tile.

        Prefer the VPF-embedded decision; fall back to pixel heuristic only if absent.
        """
        # Try provenance first (deterministic, task-aligned)
        try:
            vpf_info = extract_vpf(png_bytes)
            vpf = vpf_info[0] if isinstance(vpf_info, tuple) else vpf_info
            metrics = vpf.get("metrics", {}) if isinstance(vpf, dict) else {}
            if "top_doc_global" in metrics:
                # Use a simple tile saliency as relevance (or set 1.0); ordering is already decided
                gray = png_to_gray_array(png_bytes)
                rel = float(np.max(gray)) / 255.0 if gray.ndim == 2 and gray.size > 0 else 1.0
                return (int(metrics["top_doc_global"]), rel)
        except Exception:
            pass

        # Fallback: pixel heuristic (best-effort only)
        gray = png_to_gray_array(png_bytes)
        if gray.ndim != 2 or gray.size == 0:
            return (0, 0.0)

        y, x = np.unravel_index(np.argmax(gray), gray.shape)
        rel = float(gray[y, x]) / 255.0

        # If we have doc_start/documents, map to global; otherwise return tile-local index
        doc_start = 0
        docs_in_tile = None
        try:
            vpf_info = extract_vpf(png_bytes)
            vpf = vpf_info[0] if isinstance(vpf_info, tuple) else vpf_info
            params = vpf.get("params", {}) if isinstance(vpf, dict) else {}
            metrics = vpf.get("metrics", {}) if isinstance(vpf, dict) else {}
            doc_start = int(params.get("doc_start", 0))
            docs_in_tile = int(metrics.get("documents", 0)) or None
            vpf.setdefault("metrics", {})["decision_fallback_used"] = True
        except Exception:
            pass

        header_rows = 0
        if docs_in_tile:
            header_rows = max(0, gray.shape[0] - docs_in_tile)

        local_doc_idx = max(0, y - header_rows)
        global_doc_idx = doc_start + local_doc_idx
        return (int(global_doc_idx), rel)

    def get_tile(
        self,
        level_index: int,
        x: int = 0,
        y: int = 0,
        width: int = 16,
        height: int = 16,
    ) -> bytes:
        """
        Get a tile from storage without loading the entire level.

        Args:
            level_index: Level to get the tile from.
            x, y: Coordinates of the top-left corner of the region (in pixels).
            width, height: Dimensions of the region to extract (in pixels).

        Returns:
            Bytes representing the tile data.
        """
        if not (0 <= level_index < self.num_levels):
            raise ValueError(f"Invalid level index: {level_index}")

        level_meta = self.levels[level_index]
        if level_meta is None:
            raise ValueError(f"Level {level_index} not processed yet")

        # Calculate tile coordinates
        tile_x = x // self.tile_size
        tile_y = y // self.tile_size

        # Get tile ID and load from storage
        tile_id = self.storage.get_tile_id(level_index, tile_x, tile_y)
        png_bytes = self.storage.load_tile(tile_id)

        if png_bytes is None:
            raise ValueError(f"Tile not found: {tile_id}")

        # Extract the requested region
        region_x = x % self.tile_size
        region_y = y % self.tile_size
        region_width = min(width, self.tile_size - region_x)
        region_height = min(height, self.tile_size - region_y)

        # For a real implementation, we'd extract the region from the PNG
        # Here we just return the full tile for simplicity
        return png_bytes

    def get_decision(self, level_index: Optional[int] = None) -> Tuple[int, int, float]:
        """
        Get top decision using logarithmic navigation through the hierarchy.

        Returns:
            (level, doc_idx, relevance_score)
        """
        if level_index is None:
            level_index = 0

        path = self.navigate(start_level=level_index)
        if not path:
            return (0, 0, 0.0)

        final_step = path[-1]
        # If we reached base level and extracted a decision, use it
        if "decision" in final_step:
            return (final_step["level"], int(final_step["decision"]), float(final_step["relevance"]))

        # Otherwise, we didn't reach the decision tile; fall back to a safe default
        return (final_step["level"], 0, float(final_step["relevance"]))

    def get_metadata(self) -> Dict[str, Any]:
        """Get complete metadata for the hierarchical map."""
        # Ensure levels metadata is current
        level_info = []
        for i, level_data in enumerate(self.levels):
            if level_data is not None:
                level_info.append(
                    {
                        "level": i,
                        "type": level_data["type"],
                        "documents": level_data["metadata"]["documents"],
                        "metrics": level_data["metadata"]["metrics"],
                        "num_tiles": level_data["num_tiles_x"]
                        * level_data["num_tiles_y"],
                    }
                )

        return {**self.metadata, "level_details": level_info}

    def get_path_length(self, total_documents: int) -> int:
        """
        Get the typical path length for a given number of documents.

        Args:
            total_documents: Total number of documents in the dataset.

        Returns:
            Typical navigation path length in hops.
        """
        # Calculate how many levels we actually need
        docs_per_base = self.tile_size  # Documents per base tile
        base_tiles = math.ceil(total_documents / docs_per_base)

        # Calculate how many levels needed
        levels_needed = 1
        while base_tiles > 1:
            base_tiles = math.ceil(base_tiles / (self.zoom_factor**2))
            levels_needed += 1

        # Never exceed configured max levels
        return min(levels_needed, self.num_levels)


    def get_level(self, level_index: int) -> Dict[str, Any]:
        """
        Get data for a specific level in the hierarchy.
        
        This implements ZeroModel's "constant-feeling navigation" principle by
        providing direct access to any level in the pyramid structure.
        
        Args:
            level_index: The hierarchical level index (0 = most abstract).
        
        Returns:
            Dictionary containing level data.
            
        Raises:
            ValueError: If level_index is invalid.
        """
        logger.debug(f"Retrieving data for level {level_index}.")
        
        # Validate level index
        if not (0 <= level_index < len(self.levels)):
            error_msg = f"Level index must be between 0 and {len(self.levels) - 1}, got {level_index}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get the level data
        level_data = self.levels[level_index]
        
        if level_data is None:
            error_msg = f"Level {level_index} has not been processed yet."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Level {level_index} data retrieved successfully.")
        return level_data

# --- Helper functions ---


def embed_vpf(png_or_img: Union[np.ndarray, bytes, bytearray], vpf: Dict[str, Any]) -> bytes:
    """
    Embed VPF into a PNG footer (allowed trailing data after IEND; most decoders ignore it).
    Accepts either raw PNG bytes or a numpy image array.
    """
    # 1) ensure PNG bytes
    png_bytes = to_png_bytes(png_or_img)

    # 2) serialize + compress VPF
    json_data = json.dumps(vpf, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(json_data)

    # 3) append footer blob (magic + length + data)
    footer = VPF_FOOTER_MAGIC + struct.pack(">I", len(compressed)) + compressed
    return png_bytes + footer


def extract_critical_region(png_bytes: bytes, size: int):
    """Return a size×size crop from the top-left of the PNG (gray).

    - size must be a positive integer -> ValueError if <= 0
    - if size exceeds image dims, clip and warn
    """
    if not isinstance(size, int):
        raise TypeError(f"size must be an int, got {type(size).__name__}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    arr = png_to_gray_array(png_bytes)  # shape: (H, W)
    h, w = arr.shape[0], arr.shape[1]

    if size > h or size > w:
        logger.warning(
            "Image dimensions (%d, %d) are smaller than requested critical region size (%d). "
            "Clipping to (%d, %d).", w, h, size, min(size, h), min(size, w)
        )

    return arr[: min(size, h), : min(size, w)]

def region_max_intensity(region: np.ndarray) -> float:
    """Convert region to grayscale if needed and return max intensity [0,1]."""
    if region.ndim == 3:
        region = 0.299 * region[:, :, 0] + 0.587 * region[:, :, 1] + 0.114 * region[:, :, 2]
    return float(np.max(region)) / 255.0
