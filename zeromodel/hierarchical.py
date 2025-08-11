# zeromodel/hierarchical.py
"""
Hierarchical Visual Policy Map (HVPM) implementation with DuckDB SQL support.

This module provides the HierarchicalVPM class for creating multi-level
decision maps, enabling efficient processing across different resource
constraints (edge vs. cloud).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeromodel.config import get_config

# Import the core ZeroModel
# Make sure the path is correct based on your package structure
from .core import ZeroModel

# Create a logger for this module
logger = logging.getLogger(__name__)

class HierarchicalVPM:
    """
    Hierarchical Visual Policy Map (HVPM) implementation with DuckDB SQL support.

    This class creates a multi-level decision map system where:
    - Level 0: Strategic overview (small, low-res, for edge devices)
    - Level 1: Tactical view (medium resolution)
    - Level 2: Operational detail (full resolution)

    The hierarchical structure enables:
    - Efficient decision-making at multiple abstraction levels
    - Resource-adaptive processing (edge vs cloud)
    - Multi-stage decision pipelines
    - Visual exploration of policy landscapes
    """

    def __init__(self,
                 metric_names: List[str],
                 num_levels: int = 3,
                 zoom_factor: int = 3):
        """
        Initialize the hierarchical VPM system.

        Args:
            metric_names: Names of all metrics being tracked.
            num_levels: Number of hierarchical levels (default 3).
            zoom_factor: Zoom factor between levels (default 3).
            precision: Bit precision for encoding (4-16).
            
        Raises:
            ValueError: If inputs are invalid (e.g., num_levels <= 0).
        """
        logger.debug(f"Initializing HierarchicalVPM with metrics: {metric_names}, levels: {num_levels}, zoom: {zoom_factor}")
        if num_levels <= 0:
            error_msg = f"num_levels must be positive, got {num_levels}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if zoom_factor <= 1:
             error_msg = f"zoom_factor must be greater than 1, got {zoom_factor}."
             logger.error(error_msg)
             raise ValueError(error_msg)

        self.metric_names = list(metric_names)
        self.num_levels = num_levels
        self.zoom_factor = zoom_factor
        self.precision = get_config("core").get("precision", 8)
        self.levels: List[Dict[str, Any]] = [] # Store level data
        self.metadata: Dict[str, Any] = {
            "version": "1.0",
            "temporal_axis": False,
            "levels": num_levels,
            "zoom_factor": zoom_factor,
            "metric_names": self.metric_names # Add for reference
        }
        logger.info(f"HierarchicalVPM initialized with {num_levels} levels.")

    def process(self, score_matrix: np.ndarray, task: str):
        """
        Process score matrix into hierarchical visual policy maps using the new prepare() method.

        Args:
            score_matrix: 2D array of shape [documents × metrics].
            task: SQL query defining the task.

        Raises:
            ValueError: If inputs are invalid or processing fails.
        """
        logger.info(f"Processing hierarchical VPM for task: '{task}' with data shape {score_matrix.shape}")
        if score_matrix is None:
            error_msg = "score_matrix cannot be None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
            error_msg = f"score_matrix must be 2D, got shape {score_matrix.shape}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not task:
             logger.warning("Task string is empty. Proceeding with empty task.")
             # Consider if an empty task is valid or should raise an error
             # For now, let ZeroModel.prepare handle it. 

        # Update metadata
        self.metadata["task"] = task
        self.metadata["documents"] = score_matrix.shape[0]
        self.metadata["metrics"] = score_matrix.shape[1]
        logger.debug(f"Updated metadata: documents={score_matrix.shape[0]}, metrics={score_matrix.shape[1]}")

        # Clear existing levels
        self.levels = []
        logger.debug("Cleared existing levels.")

        # --- Level Creation using prepare() ---
        # Create ZeroModel instance for the base (highest detail) level
        logger.debug("Creating base ZeroModel instance.")
        base_zeromodel = ZeroModel(self.metric_names)
        base_zeromodel.precision = self.precision  # Set precision for the base model
        
        # --- Use prepare() instead of set_sql_task() + process() ---
        # Prepare the base level ZeroModel with data and task in one step
        try:
            base_zeromodel.prepare(score_matrix, task) # <-- CHANGED HERE
            logger.debug("Base ZeroModel prepared with data and task.")
        except Exception as e:
            logger.error(f"Failed to prepare base ZeroModel: {e}")
            raise ValueError(f"Error preparing base ZeroModel level: {e}") from e
        # --- End of change ---

        # Create base level (Level N-1: Full detail, where N is num_levels)
        base_level_index = self.num_levels - 1
        base_level = self._create_base_level(base_zeromodel, score_matrix) # Pass original data for metadata if needed
        # Store level data. Levels list will be ordered [Level 0, Level 1, ..., Level N-1]
        self.levels.append(base_level) 
        logger.debug(f"Base level ({base_level_index}) created and added.")

        # Create higher levels (Level N-2, Level N-3, ..., Level 0)
        current_data = score_matrix
        # Iterate from level 1 up to num_levels-1 (0-indexed levels)
        # Level 0 is the most abstract (smallest)
        for relative_level in range(1, self.num_levels): 
            level_index = self.num_levels - 1 - relative_level # Absolute level index (0 to N-2)
            logger.debug(f"Creating level {level_index} (relative {relative_level})")
            
            # Calculate target dimensions for clustering
            target_docs = max(1, int(np.ceil(current_data.shape[0] / self.zoom_factor)))
            target_metrics = max(1, int(np.ceil(current_data.shape[1] / self.zoom_factor)))
            logger.debug(f"Clustering to target size: docs={target_docs}, metrics={target_metrics}")

            # Perform clustering
            clustered_data = self._cluster_data(current_data, target_docs, target_metrics)
            logger.debug(f"Clustered data shape: {clustered_data.shape}")

            # Create the level data structure
            # Pass the clustered data and the base task
            level_data = self._create_level(clustered_data, task, level_index) # <-- CHANGED: Pass task, not base_task_config
            # Insert at the beginning of the list to maintain order [L0, L1, L2, ...]
            self.levels.insert(0, level_data) 
            logger.debug(f"Level {level_index} created and added.")

            # Update data for next iteration (cluster the clustered data)
            current_data = clustered_data

        logger.info("Hierarchical VPM processing complete.")


    def _cluster_data(self, data: np.ndarray, num_docs: int, num_metrics: int) -> np.ndarray:
        """
        Cluster data for higher-level (more abstract) views.

        Args:
             Input data matrix of shape [docs, metrics].
            num_docs: Target number of document clusters.
            num_metrics: Target number of metric clusters.

        Returns:
            Clustered data matrix of shape [num_docs, num_metrics].
        """
        logger.debug(f"Clustering data of shape {data.shape} to {num_docs} docs x {num_metrics} metrics.")
        docs, metrics = data.shape

        # Handle edge case where we have fewer items than target clusters
        num_docs = min(num_docs, docs)
        num_metrics = min(num_metrics, metrics)
        logger.debug(f"Adjusted clustering targets: docs={num_docs}, metrics={num_metrics}")

        if docs == 0 or metrics == 0:
            logger.warning("Input data for clustering is empty. Returning empty array.")
            return np.empty((num_docs, num_metrics)) # Or np.zeros?

        # --- Document Clustering ---
        doc_clusters = []
        for i in range(num_docs):
            # Calculate slice indices for this cluster
            start_idx = int(i * docs / num_docs) # Use int() or floor?
            end_idx = int((i + 1) * docs / num_docs)
            # Ensure start < end to avoid empty slices
            end_idx = max(start_idx + 1, end_idx) 
            # Ensure end_idx doesn't exceed the array
            end_idx = min(end_idx, docs)

            if start_idx < end_idx:
                # Average the rows (documents) in this slice
                cluster_mean = np.mean(data[start_idx:end_idx], axis=0)
                doc_clusters.append(cluster_mean)
                logger.debug(f"Document cluster {i}: rows {start_idx}-{end_idx-1} -> mean")
            else:
                # Fallback if slice is somehow invalid (should be rare with adjustments)
                logger.warning(f"Invalid document slice [{start_idx}:{end_idx}] for cluster {i}. Using first row.")
                fallback_idx = min(start_idx, docs - 1)
                doc_clusters.append(data[fallback_idx])

        if not doc_clusters:
             logger.error("No document clusters created. This should not happen.")
             # Return an array of zeros or handle error?
             clustered_docs = np.zeros((num_docs, metrics))
        else:
             clustered_docs = np.array(doc_clusters)
        logger.debug(f"Document clustering resulted in shape: {clustered_docs.shape}")

        # --- Metric Clustering ---
        # Note: The original code clustered metrics based on the *already clustered docs*.
        # This means metric clustering happens on the `clustered_docs` array.
        metric_clusters = []
        effective_metrics = clustered_docs.shape[1] # Use shape of clustered_docs
        for j in range(num_metrics):
            start_idx = int(j * effective_metrics / num_metrics)
            end_idx = int((j + 1) * effective_metrics / num_metrics)
            end_idx = max(start_idx + 1, end_idx)
            end_idx = min(end_idx, effective_metrics)

            if start_idx < end_idx:
                # Average the columns (metrics) in this slice
                # np.mean(..., axis=1) averages along columns, resulting in a 1D array per slice
                # np.column_stack then combines these 1D arrays into columns
                cluster_mean = np.mean(clustered_docs[:, start_idx:end_idx], axis=1)
                metric_clusters.append(cluster_mean)
                logger.debug(f"Metric cluster {j}: cols {start_idx}-{end_idx-1} -> mean")
            else:
                 logger.warning(f"Invalid metric slice [{start_idx}:{end_idx}] for cluster {j}. Using first col.")
                 fallback_idx = min(start_idx, effective_metrics - 1)
                 metric_clusters.append(clustered_docs[:, fallback_idx])

        if not metric_clusters:
             logger.error("No metric clusters created. This should not happen.")
             # Return an array matching the doc cluster shape
             final_clustered_data = np.zeros((clustered_docs.shape[0], num_metrics))
        else:
             # np.column_stack takes a sequence of 1-D arrays and stacks them as columns
             final_clustered_data = np.column_stack(metric_clusters)
        logger.debug(f"Metric clustering resulted in shape: {final_clustered_data.shape}")
        
        return final_clustered_data

    def _create_base_level(self, zeromodel: ZeroModel, score_matrix: np.ndarray) -> Dict[str, Any]:
        """Create the base level (highest detail) using the prepared ZeroModel."""
        level_index = self.num_levels - 1
        logger.debug(f"Creating base level data structure (Level {level_index}).")
        # The zeromodel passed in should already be prepared.
        # We can now safely call encode(), get_metadata() etc.
        try:
            vpm_image = zeromodel.encode() # Uses zeromodel.sorted_matrix
            logger.debug(f"Encoded base level VPM image of shape {vpm_image.shape}.")
        except Exception as e:
            logger.error(f"Failed to encode VPM for base level: {e}")
            # Depending on requirements, re-raise or handle?
            raise # Re-raise 

        base_level_data = {
            "level": level_index,
            "type": "base",
            "zeromodel": zeromodel, # Store the prepared ZeroModel instance
            "vpm": vpm_image,
            "metadata": {
                "documents": score_matrix.shape[0], # Use original shape for metadata
                "metrics": score_matrix.shape[1],
                "sorted_docs": zeromodel.doc_order.tolist() if zeromodel.doc_order is not None else [],
                "sorted_metrics": zeromodel.metric_order.tolist() if zeromodel.metric_order is not None else [],
                # Add wavelet info if applicable
                # "wavelet_level": 0 
            }
        }
        logger.debug(f"Base level data structure created.")
        return base_level_data

    def _create_level(self,
                      approx_data: np.ndarray, # Renamed from clustered_data for clarity
                      task: str,              # Pass the task string
                      level_index: int) -> Dict[str, Any]: # Absolute level index (0, 1, 2, ...)
        """Create a higher-level (more abstract) view using prepare()."""
        logger.debug(f"Creating level data structure (Level {level_index}).")
        num_metrics_in_clustered_data = approx_data.shape[1]
        # Create a simplified metric set for this level
        level_metrics = [
            f"cluster_{i}" for i in range(num_metrics_in_clustered_data)
        ]
        logger.debug(f"Generated level metric names: {level_metrics}")

        # --- Create and Prepare ZeroModel for this level ---
        # Process with simplified metrics using a new ZeroModel instance
        level_zeromodel = ZeroModel(level_metrics)
        level_zeromodel.precision = self.precision  # Set precision for this level
        logger.debug("Created ZeroModel instance for this level.")

        # --- Use prepare() for this level's ZeroModel ---
        # Prepare the level's ZeroModel with its clustered data and the task
        # The task might need adaptation (see notes below), but for now, pass it as is
        # or use a default sorting strategy for clustered levels.
        try:
            # Option 1: Use the same task (might not be ideal for clustered data)
            # level_zeromodel.prepare(approx_data, task) 
            
            # Option 2: Use a simple default task for higher levels (often better)
            # E.g., sort by the first "cluster" metric which represents the aggregated importance
            # of the original metrics that formed this cluster.
            default_level_task = f"SELECT * FROM virtual_index ORDER BY {level_metrics[0]} DESC"
            level_zeromodel.prepare(approx_data, default_level_task) # <-- CHANGED HERE
            logger.debug(f"Prepared level {level_index} ZeroModel with default task: {default_level_task}")
        except Exception as e:
             logger.error(f"Failed to prepare ZeroModel for level {level_index}: {e}")
             raise ValueError(f"Error preparing ZeroModel for level {level_index}: {e}") from e
        # --- End of change ---

        try:
            level_vpm = level_zeromodel.encode() # Uses level_zeromodel.sorted_matrix
            logger.debug(f"Encoded level {level_index} VPM image of shape {level_vpm.shape}.")
        except Exception as e:
            logger.error(f"Failed to encode VPM for level {level_index}: {e}")
            raise # Re-raise

        level_data = {
            "level": level_index,
            "type": "clustered", # Or "approximated"
            "zeromodel": level_zeromodel, # Store the prepared ZeroModel instance
            "vpm": level_vpm,
            "metadata": {
                "documents": approx_data.shape[0],
                "metrics": approx_data.shape[1],
                "sorted_docs": level_zeromodel.doc_order.tolist() if level_zeromodel.doc_order is not None else [],
                "sorted_metrics": level_zeromodel.metric_order.tolist() if level_zeromodel.metric_order is not None else [],
                # "wavelet_level": level_index # If wavelets are used
            }
        }
        logger.debug(f"Level {level_index} data structure created.")
        return level_data

    def get_level(self, level_index: int) -> Dict[str, Any]:
        """
        Get data for a specific level.

        Args:
            level_index: The hierarchical level index (0 = most abstract).

        Returns:
            Dictionary containing level data.
            
        Raises:
            ValueError: If level_index is invalid.
        """
        logger.debug(f"Retrieving data for level {level_index}.")
        if not (0 <= level_index < self.num_levels):
            error_msg = f"Level index must be between 0 and {self.num_levels - 1}, got {level_index}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # self.levels is ordered [Level 0, Level 1, ..., Level N-1]
        # So, index directly corresponds to level_index
        level_data = self.levels[level_index]
        logger.debug(f"Level {level_index} data retrieved.")
        return level_data

    def get_tile(self,
                 level_index: int,
                 x: int = 0,
                 y: int = 0,
                 width: int = 3,
                 height: int = 3) -> bytes:
        """
        Get a tile from a specific level for edge devices.

        Args:
            level_index: Hierarchical level (0 = most abstract).
            x, y: Top-left corner of tile (currently ignored, gets top-left).
            width, height: Dimensions of tile.

        Returns:
            Compact byte representation of the tile.
            
        Raises:
            ValueError: If level_index is invalid or tile parameters are bad.
        """
        logger.debug(f"Getting tile from level {level_index} (x={x}, y={y}, w={width}, h={height}).")
        # Note: x, y are currently ignored by ZeroModel.get_critical_tile, which always gets top-left.
        # Future enhancement could involve slicing the VPM image based on x,y.
        
        level_data = self.get_level(level_index)
        zeromodel_instance: ZeroModel = level_data["zeromodel"]
        
        # Determine tile size - use the larger of width/height, or a specific logic
        # The original code used max(width, height). Let's stick to that for now,
        # though it might be better to get a width x height tile.
        # get_critical_tile currently only takes one size parameter (square tile).
        tile_size = max(width, height) 
        logger.debug(f"Determined tile size for request: {tile_size}")

        # Get critical tile (currently always top-left)
        tile_bytes = zeromodel_instance.get_critical_tile(tile_size=tile_size)
        logger.info(f"Tile retrieved from level {level_index}. Size: {len(tile_bytes)} bytes.")
        return tile_bytes

    def get_decision(self, level_index: Optional[int] = None) -> Tuple[int, int, float]:
        """
        Get top decision from a specific level.

        Args:
            level_index: The level to get the decision from.
                         If None, gets the decision from the most detailed level (highest index).

        Returns:
            Tuple of (level_index, document_index, relevance_score).
            
        Raises:
            ValueError: If level_index is invalid.
        """
        if level_index is None:
            level_index = self.num_levels - 1 # Default to most detailed level
        logger.debug(f"Getting decision from level {level_index}.")

        level_data = self.get_level(level_index)
        zeromodel_instance: ZeroModel = level_data["zeromodel"]
        doc_idx, relevance = zeromodel_instance.get_decision()
        logger.info(f"Decision from level {level_index}: Doc {doc_idx}, Relevance {relevance:.4f}.")
        return (level_index, doc_idx, relevance)

    def zoom_in(self, level_index: int, doc_idx: int, metric_idx: int) -> int:
        """
        Determine the next level to zoom into based on current selection.

        Args:
            level_index: Current hierarchical level.
            doc_idx: Selected document index (relative to the current level's data).
            metric_idx: Selected metric index (relative to the current level's data).

        Returns:
            Next level index to zoom into (level_index+1, or same level if already at base).
        """
        logger.debug(f"Calculating zoom-in from level {level_index}, doc {doc_idx}, metric {metric_idx}.")
        if level_index >= self.num_levels - 1:
            logger.info("Already at the most detailed level. Staying at current level.")
            return level_index  # Already at most detailed level
        next_level = level_index + 1
        logger.debug(f"Zooming in to level {next_level}.")
        return next_level

    def get_metadata(self) -> Dict[str, Any]:
        """Get complete metadata for the hierarchical map."""
        logger.debug("Retrieving hierarchical VPM metadata.")
        # Ensure levels metadata is current
        level_info = []
        for level_data in self.levels:
            level_info.append({
                "level": level_data["level"],
                "type": level_data["type"],
                "documents": level_data["metadata"]["documents"],
                "metrics": level_data["metadata"]["metrics"]
            })
        self.metadata["level_details"] = level_info
        logger.debug("Hierarchical VPM metadata retrieved.")
        return self.metadata


    def hunt_target(self, 
                    tau: float = 0.75, 
                    max_steps: int = 6,
                    initial_level: int = 0) -> Tuple[Tuple[int, int], float, List[Dict[str, Any]]]:
        """
        Hunt for a target using the heat-seeking VPM tuner.
        
        Args:
            tau: Confidence threshold.
            max_steps: Maximum search steps.
            initial_level: Starting level for the hunt.
            
        Returns:
            Tuple of ((level, doc_index), confidence, audit_trail).
        """
        from zeromodel.vpm.hunter import \
            VPMHunter  # Import here to avoid circular imports if needed
        hunter = VPMHunter(self, tau=tau, max_steps=max_steps)
        return hunter.hunt(initial_level=initial_level)