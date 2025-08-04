"""
Zero-Model Intelligence Encoder/Decoder

This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps where the
intelligence is in the data structure itself, not in processing.

The zeromodel class:
- Encodes score matrices into visual policy maps (VPMs)
- Handles task-aware sorting of documents and metrics
- Provides critical tile extraction for edge devices
- Enables zero-model decision making

Example:
    # Initialize with metric names
    zeromodel = zeromodel(metric_names=["uncertainty", "size", "quality", ...])
    
    # Process score matrix (documents × metrics)
    zeromodel.process(score_matrix)
    
    # Get visual policy map
    vpm = zeromodel.encode()
    
    # For edge devices: get critical tile
    tile = zeromodel.get_critical_tile()
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .sorter import TaskSorter
from .normalizer import DynamicNormalizer
from .utils import quantize

class ZeroModel:
    """
    Zero-Model Intelligence encoder/decoder - completely standalone
    
    This class transforms high-dimensional policy evaluation data into
    spatially-optimized visual maps where:
    - Position = Importance (top-left = most relevant)
    - Color = Value (darker = higher priority)
    - Structure = Task logic
    
    The intelligence is in the data structure, not in processing.
    """
    
    def __init__(self, metric_names: List[str], precision: int = 8):
        """
        Initialize zeromodel encoder.
        
        Args:
            metric_names: Names of all metrics being tracked
            precision: Bit precision for encoding (4-16)
        """
        self.metric_names = metric_names
        self.precision = max(4, min(16, precision))
        self.sorter = TaskSorter(metric_names)
        self.normalizer = DynamicNormalizer(metric_names)
        self.sorted_matrix = None
        self.doc_order = None
        self.metric_order = None
        self.task = "default"
    
    def set_task(self, task_description: str, feedback: Optional[Dict[str, float]] = None):
        """
        Update the current task and sorting weights.
        
        Args:
            task_description: Natural language description of the task
            feedback: Optional feedback on previous decisions
        """
        self.task = task_description
        self.sorter.update_weights(task_description, feedback)
    
    def process(self, score_matrix: np.ndarray) -> None:
        """
        Process a score matrix to prepare for encoding.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
        """
        # Update normalizer with new data
        self.normalizer.update(score_matrix)
        
        # Normalize scores
        normalized = self.normalizer.normalize(score_matrix)
        
        # Sort by task relevance
        self.sorted_matrix, self.metric_order, self.doc_order = self.sorter.sort_matrix(normalized)
    
    def encode(self) -> np.ndarray:
        """
        Encode the processed data into a full visual policy map.
        
        Returns:
            RGB image array of shape [height, width, 3]
        """
        if self.sorted_matrix is None:
            raise ValueError("Data not processed yet. Call process() first.")
        
        n_docs, n_metrics = self.sorted_matrix.shape
        
        # Calculate required width (3 metrics per pixel)
        width = (n_metrics + 2) // 3  # Ceiling division
        
        # Create image array
        img = np.zeros((n_docs, width, 3), dtype=np.uint8)
        
        # Fill pixels with normalized scores (0-255)
        for i in range(n_docs):
            for j in range(n_metrics):
                pixel_x = j // 3
                channel = j % 3
                img[i, pixel_x, channel] = int(self.sorted_matrix[i, j] * 255)
        
        return img
    
    def get_critical_tile(self, tile_size: int = 3) -> bytes:
        """
        Get critical tile for edge devices (top-left section).
        
        Args:
            tile_size: Size of tile to extract (default 3x3)
        
        Returns:
            Compact byte representation of the tile
        """
        if self.sorted_matrix is None:
            raise ValueError("Data not processed yet. Call process() first.")
        
        # Get top-left section (most relevant documents & metrics)
        tile_data = self.sorted_matrix[:tile_size, :tile_size*3]
        
        # Convert to compact byte format
        tile_bytes = bytearray()
        tile_bytes.append(tile_size)  # Width
        tile_bytes.append(tile_size)  # Height
        tile_bytes.append(0)  # X offset
        tile_bytes.append(0)  # Y offset
        
        # Add pixel data (1 byte per channel)
        for i in range(tile_size):
            for j in range(tile_size * 3):  # 3 channels per pixel
                if i < tile_data.shape[0] and j < tile_data.shape[1]:
                    tile_bytes.append(int(tile_data[i, j] * 255))
                else:
                    tile_bytes.append(0)  # Padding
        
        return bytes(tile_bytes)
    
    def get_decision(self) -> Tuple[int, float]:
        """
        Get top decision from encoded data (for edge devices).
        
        Returns:
            (document_index, relevance_score)
        """
        if self.sorted_matrix is None:
            raise ValueError("Data not processed yet. Call process() first.")
        
        # Top document is first row in sorted matrix
        top_doc_idx = self.doc_order[0]
        
        # Relevance score is based on first metric
        relevance = self.sorted_matrix[0, 0]
        
        return top_doc_idx, relevance
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current encoding state"""
        return {
            "task": self.task,
            "metric_order": self.metric_order.tolist() if self.metric_order is not None else [],
            "doc_order": self.doc_order.tolist() if self.doc_order is not None else [],
            "metric_names": self.metric_names,
            "precision": self.precision
        }