"""
Zero-Model Intelligence Encoder/Decoder with Wavelet Integration
This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps where the
intelligence is in the data structure itself, not in processing.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pywt

from .config import get_config_value, load_config, validate_config
from .normalizer import DynamicNormalizer
from .sorter import TaskSorter


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
    
    def __init__(self, 
                 metric_names: List[str],
                 config_path: Optional[str] = None):
        """
        Initialize ZeroModel encoder.
        
        Args:
            metric_names: Names of all metrics being tracked
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Validate configuration
        if not validate_config(self.config):
            raise ValueError("Invalid configuration")
        
        # Get precision from config
        self.precision = get_config_value(
            self.config, 'zeromodel', 'precision', default=8
        )
        self.precision = max(4, min(16, self.precision))
        
        # Initialize components with config
        self.sorter = TaskSorter(
            metric_names,
            config=self.config
        )
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
    
    def get_decision(self, context_size: int = 3) -> Tuple[int, float]:
        """
        Get top decision with contextual understanding, handling limited metrics.
        """
        if self.sorted_matrix is None:
            raise ValueError("Data not processed yet. Call process() first.")
        
        n_metrics = self.sorted_matrix.shape[1]
        context_width = min(context_size * 3, n_metrics)

        context = self.sorted_matrix[:context_size, :context_width]

        weights = np.zeros_like(context)
        for i in range(context.shape[0]):
            for j in range(context.shape[1]):  # ← FIXED
                distance = np.sqrt(i**2 + (j / 3)**2)
                weights[i, j] = max(0, 1.0 - distance * 0.3)

        weighted_relevance = np.sum(context * weights) / np.sum(weights)
        top_doc_idx = self.doc_order[0] if len(self.doc_order) > 0 else 0

        return top_doc_idx, weighted_relevance
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current encoding state"""
        return {
            "task": self.task,
            "metric_order": self.metric_order.tolist() if self.metric_order is not None else [],
            "doc_order": self.doc_order.tolist() if self.doc_order is not None else [],
            "metric_names": self.metric_names,
            "precision": self.precision
        }
    
    def to_hierarchical(self, 
                       metric_names: List[str],
                       num_levels: int = 3,
                       zoom_factor: int = 3) -> 'HierarchicalVPM':
        """
        Convert this ZeroModel instance to a HierarchicalVPM.
        
        Args:
            metric_names: Names of all metrics
            num_levels: Number of hierarchical levels
            zoom_factor: Zoom factor between levels
        
        Returns:
            HierarchicalVPM instance
        """
        from .hierarchical import HierarchicalVPM

        # Create hierarchical VPM
        hvpm = HierarchicalVPM(
            metric_names=metric_names,
            num_levels=num_levels,
            zoom_factor=zoom_factor,
            precision=self.precision
        )
        
        # Process with the same data and task
        hvpm.process(
            self.sorted_matrix,
            self.task
        )
        
        return hvpm

# In zeromodel/core.py
class HierarchicalVPM:
    def __init__(self, 
                 metric_names: List[str],
                 config_path: Optional[str] = None,
                 num_levels: Optional[int] = None,
                 zoom_factor: Optional[int] = None,
                 wavelet: Optional[str] = None):
        """
        Initialize the hierarchical VPM system with configuration.
        
        Args:
            metric_names: Names of all metrics being tracked
            config_path: Path to configuration file (optional)
            num_levels: Override config value for number of hierarchical levels
            zoom_factor: Override config value for zoom factor
            wavelet: Override config value for wavelet type
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Validate configuration
        if not validate_config(self.config):
            raise ValueError("Invalid configuration")
        
        # Get hierarchical parameters (allow overrides)
        hierarchical = self.config.get('zeromodel', {}).get('hierarchical', {})
        self.num_levels = num_levels if num_levels is not None else hierarchical.get('num_levels', 3)
        self.zoom_factor = zoom_factor if zoom_factor is not None else hierarchical.get('zoom_factor', 3)
        self.wavelet = wavelet if wavelet is not None else hierarchical.get('wavelet', 'bior6.8')
        self.max_levels = hierarchical.get('max_levels', None)
        
        # Get precision
        self.precision = get_config_value(
            self.config, 'zeromodel', 'precision', default=8
        )
        
        self.metric_names = metric_names
        self.levels = []
        self.metadata = {
            "version": "2.0",
            "temporal_axis": False,
            "levels": self.num_levels,
            "zoom_factor": self.zoom_factor,
            "wavelet": self.wavelet,
            "encoding": "wavelet"
        }

    def process(self, 
                score_matrix: np.ndarray, 
                task: str,
                temporal_data: Optional[List[np.ndarray]] = None):
        """
        Process score matrix into hierarchical visual policy maps using wavelet decomposition.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
            task: Task description for sorting
            temporal_data: Optional list of score matrices over time
        """
        # Update metadata
        self.metadata["task"] = task
        self.metadata["documents"] = score_matrix.shape[0]
        self.metadata["metrics"] = score_matrix.shape[1]
        
        # Process temporal data if provided
        if temporal_data:
            self.metadata["temporal_axis"] = True
            self.metadata["time_points"] = len(temporal_data)
        
        # Clear existing levels
        self.levels = []
        
        # Create base level (Level 2: Full detail)
        base_level = self._create_base_level(score_matrix, task)
        self.levels.append(base_level)
        
        # Create higher levels using wavelet decomposition
        current_data = score_matrix
        for level in range(1, self.num_levels):
            # Apply wavelet transform for hierarchical representation
            approx = self._wavelet_approximation(current_data)
            
            # Create level
            level_data = self._create_level(approx, task, level)
            self.levels.insert(0, level_data)  # Insert at beginning (Level 0 first)
            
            # Use this level as basis for next higher level
            current_data = approx
    
    def _wavelet_approximation(self, matrix: np.ndarray) -> np.ndarray:
        """
        Create a wavelet-based approximation of the matrix for higher-level views.
        
        Args:
            matrix: Input data matrix
            
        Returns:
            Approximated matrix with reduced resolution
        """
        # Handle edge cases
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return matrix
        
        try:
            # Determine optimal decomposition level
            min_dim = min(matrix.shape[0], matrix.shape[1])
            max_possible_levels = pywt.dwt_max_level(min_dim, pywt.Wavelet(self.wavelet).dec_len)
            decomposition_level = min(self.max_levels or max_possible_levels, max_possible_levels)
            
            if decomposition_level < 1:
                return matrix
            
            # Apply 2D wavelet transform
            coeffs = pywt.wavedec2(matrix, self.wavelet, level=decomposition_level)
            cA, detail_coeffs = coeffs[0], coeffs[1:]
            
            # Information Bottleneck thresholding
            threshold = self._ib_threshold(cA)
            
            # Threshold approximation coefficients (soft thresholding)
            cA_thresholded = pywt.threshold(cA, value=threshold, mode='soft')
            
            # Reconstruct approximation
            approx = pywt.waverec2([cA_thresholded] + detail_coeffs, self.wavelet)
            
            # Handle potential shape mismatch due to boundary effects
            if approx.shape != matrix.shape:
                # Crop or pad to match original shape
                approx = self._match_shape(approx, matrix.shape)
            
            # Ensure values stay within [0,1] range
            approx = np.clip(approx, 0, 1)
            
            return approx
            
        except Exception as e:
            warnings.warn(f"Wavelet decomposition failed: {str(e)}. Falling back to simple clustering.")
            # Fallback to simple clustering if wavelet fails
            return self._cluster_data(matrix, 
                                     max(5, matrix.shape[0] // self.zoom_factor),
                                     max(3, matrix.shape[1] // self.zoom_factor))
    
    def _ib_threshold(self, coefficients: np.ndarray) -> float:
        """
        Calculate threshold based on Information Bottleneck principles.
        
        Args:
            coefficients: Wavelet coefficients
            
        Returns:
            Threshold value
        """
        # Estimate mutual information
        std = np.std(coefficients)
        if std == 0:
            return 0
        
        # IB threshold formula: retains coefficients with sufficient information
        # This is based on VisuShrink thresholding with Information Bottleneck interpretation
        return std * np.sqrt(2 * np.log(coefficients.size))
    
    def _match_shape(self, array: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Ensure array matches target shape by cropping or padding"""
        result = np.zeros(target_shape, dtype=array.dtype)
        
        # Calculate slice ranges
        h_slice = min(array.shape[0], target_shape[0])
        w_slice = min(array.shape[1], target_shape[1])
        
        # Copy data
        result[:h_slice, :w_slice] = array[:h_slice, :w_slice]
        
        return result
    
    def _cluster_data(self, 
                     data: np.ndarray, 
                     num_docs: int, 
                     num_metrics: int) -> np.ndarray:
        """
        Cluster data for higher-level views (fallback method).
        
        Args:
            data: Input data matrix
            num_docs: Target number of document clusters
            num_metrics: Target number of metric clusters
            
        Returns:
            Clustered data matrix
        """
        docs, metrics = data.shape
        
        # Handle edge case where we have fewer items than clusters
        num_docs = min(num_docs, docs)
        num_metrics = min(num_metrics, metrics)
        
        # Create document clusters
        doc_clusters = []
        for i in range(num_docs):
            start_idx = i * docs // num_docs
            end_idx = (i + 1) * docs // num_docs
            if start_idx < end_idx:  # Ensure we have data to average
                doc_clusters.append(np.mean(data[start_idx:end_idx], axis=0))
            else:
                doc_clusters.append(data[start_idx])
        
        clustered_docs = np.array(doc_clusters)
        
        # Create metric clusters
        metric_clusters = []
        for j in range(num_metrics):
            start_idx = j * metrics // num_metrics
            end_idx = (j + 1) * metrics // num_metrics
            if start_idx < end_idx:  # Ensure we have data to average
                metric_clusters.append(np.mean(clustered_docs[:, start_idx:end_idx], axis=1))
            else:
                metric_clusters.append(clustered_docs[:, start_idx])
        
        return np.column_stack(metric_clusters)
    
    def _create_base_level(self, score_matrix: np.ndarray, task: str) -> Dict[str, Any]:
        """Create the base level (highest detail)"""
        # Use standard ZeroModel for base level
        zeromodel = ZeroModel(self.metric_names)
        zeromodel.set_task(task)
        zeromodel.process(score_matrix)
        
        return {
            "level": self.num_levels - 1,
            "type": "base",
            "zeromodel": zeromodel,
            "vpm": zeromodel.encode(),
            "metadata": {
                "documents": score_matrix.shape[0],
                "metrics": score_matrix.shape[1],
                "sorted_docs": zeromodel.doc_order.tolist(),
                "sorted_metrics": zeromodel.metric_order.tolist(),
                "wavelet_level": 0
            }
        }
    
    def _create_level(self, 
                     approx_data: np.ndarray, 
                     task: str, 
                     level: int) -> Dict[str, Any]:
        """Create a higher-level (more abstract) view"""
        # Create a simplified metric set for this level
        level_metrics = [
            f"cluster_{i}" for i in range(approx_data.shape[1])
        ]
        
        # Process with simplified metrics
        zeromodel = ZeroModel(level_metrics)
        zeromodel.set_task(task)
        zeromodel.process(approx_data)
        
        return {
            "level": level,
            "type": "wavelet",
            "zeromodel": zeromodel,
            "vpm": zeromodel.encode(),
            "metadata": {
                "documents": approx_data.shape[0],
                "metrics": approx_data.shape[1],
                "sorted_docs": zeromodel.doc_order.tolist(),
                "sorted_metrics": zeromodel.metric_order.tolist(),
                "wavelet_level": level
            }
        }
    
    def get_level(self, level: int) -> Dict[str, Any]:
        """Get data for a specific level"""
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level must be between 0 and {self.num_levels-1}")
        return self.levels[level]
    
    def get_tile(self, 
                level: int, 
                x: int = 0, 
                y: int = 0, 
                width: int = 3, 
                height: int = 3) -> bytes:
        """
        Get a tile from a specific level for edge devices.
        
        Args:
            level: Hierarchical level (0 = most abstract)
            x, y: Top-left corner of tile
            width, height: Dimensions of tile
        
        Returns:
            Compact byte representation of the tile
        """
        level_data = self.get_level(level)
        zeromodel = level_data["zeromodel"]
        
        # Get critical tile
        return zeromodel.get_critical_tile(tile_size=max(width, height))
    
    def get_decision(self, level: int) -> Tuple[int, float, int]:
        """
        Get top decision from a specific level.
        
        Returns:
            (level, document_index, relevance_score)
        """
        level_data = self.get_level(level)
        doc_idx, relevance = level_data["zeromodel"].get_decision()
        return (level, doc_idx, relevance)
    
    def zoom_in(self, level: int, doc_idx: int, metric_idx: int) -> int:
        """
        Determine the next level to zoom into based on current selection.
        
        Args:
            level: Current hierarchical level
            doc_idx: Selected document index
            metric_idx: Selected metric index
            
        Returns:
            Next level to zoom into (level+1, or same level if already at base)
        """
        if level >= self.num_levels - 1:
            return level  # Already at most detailed level
        return level + 1
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get complete metadata for the hierarchical map"""
        return self.metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "metadata": self.metadata,
            "levels": [
                {
                    "level": lvl["level"],
                    "type": lvl["type"],
                    "vpm": lvl["vpm"].tolist(),
                    "metadata": lvl["metadata"]
                } for lvl in self.levels
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], metric_names: List[str]) -> 'HierarchicalVPM':
        """Create from serialized dictionary"""
        # Create instance
        hvp = cls(
            metric_names=metric_names,
            num_levels=len(data["levels"]),
            zoom_factor=data["metadata"].get("zoom_factor", 3),
            wavelet=data["metadata"].get("wavelet", "bior6.8")
        )
        
        # Set metadata
        hvp.metadata = data["metadata"]
        
        # Set levels
        hvp.levels = []
        for level_data in data["levels"]:
            # We can't fully reconstruct ZeroModel, so we store just the VPM
            hvp.levels.append({
                "level": level_data["level"],
                "type": level_data["type"],
                "vpm": np.array(level_data["vpm"]),
                "metadata": level_data["metadata"]
            })
        
        return hvp
    
    def validate_hierarchy(self) -> Dict[str, Any]:
        """
        Validate the hierarchical structure for information preservation.
        
        Returns:
            Validation metrics
        """
        if len(self.levels) < 2:
            return {"valid": False, "message": "Not enough levels for validation"}
        
        # Get base level (most detailed)
        base_level = self.levels[-1]
        base_data = base_level["zeromodel"].sorted_matrix
        
        results = []
        for i, level in enumerate(self.levels[:-1]):
            # Get current level data
            level_data = level["zeromodel"].sorted_matrix
            
            # Upsample to compare with base level
            upsampled = self._upsample(level_data, base_data.shape)
            
            # Calculate metrics
            mse = np.mean((base_data - upsampled) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10)) if mse > 0 else float('inf')
            ssim = self._calculate_ssim(base_data, upsampled)
            
            results.append({
                "level": i,
                "mse": float(mse),
                "psnr": float(psnr),
                "ssim": float(ssim)
            })
        
        # Check if hierarchy preserves critical information
        base_decision = base_level["zeromodel"].get_decision()
        preserved = True
        for i, level in enumerate(self.levels):
            level_decision = level["zeromodel"].get_decision()
            if level_decision[0] != base_decision[0]:
                preserved = False
                break
        
        return {
            "valid": preserved,
            "metrics": results,
            "decision_preserved": preserved
        }
    
    def _upsample(self, low_res: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Upsample low-resolution data to target shape"""
        # Simple nearest-neighbor upsample for validation
        h_ratio = target_shape[0] / low_res.shape[0]
        w_ratio = target_shape[1] / low_res.shape[1]
        
        upsampled = np.zeros(target_shape)
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                orig_i = int(i / h_ratio)
                orig_j = int(j / w_ratio)
                upsampled[i, j] = low_res[orig_i, orig_j]
        
        return upsampled
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray, 
                       window_size: int = 3, k1: float = 0.01, k2: float = 0.03) -> float:
        """Calculate Structural Similarity Index (SSIM) between two images"""
        # Simple implementation for validation purposes
        C1 = (k1 * 1.0) ** 2
        C2 = (k2 * 1.0) ** 2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = img1.var()
        sigma2_sq = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        return numerator / (denominator + 1e-10)