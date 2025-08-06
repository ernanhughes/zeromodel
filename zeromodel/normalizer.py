"""
Dynamic Range Adaptation

This module provides the DynamicNormalizer class which handles normalization
of scores to handle value drift over time. This is critical for long-term
viability of the zeromodel system as score distributions may change.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

# Create a logger for this module
logger = logging.getLogger(__name__)

class DynamicNormalizer:
    """
    Handles dynamic normalization of scores to handle value drift over time.
    
    This is critical because:
    - Score ranges may change as policies improve
    - New documents may have scores outside previous ranges
    - Normalization must be consistent across time
    
    The normalizer tracks min/max values for each metric and updates them
    incrementally as new data arrives using exponential smoothing.
    """
    
    def __init__(self, metric_names: List[str], alpha: float = 0.1):
        """
        Initialize the normalizer.
        
        Args:
            metric_names: Names of all metrics being tracked. Order is preserved.
            alpha: Smoothing factor for updating min/max (0.0-1.0).
                   Lower values mean slower adaptation to changes,
                   higher values mean faster adaptation.
                   
        Raises:
            ValueError: If metric_names is None/empty, or alpha is not between 0 and 1.
        """
        logger.debug(f"Initializing DynamicNormalizer with metrics: {metric_names}, alpha: {alpha}")
        if not metric_names:
            error_msg = "metric_names list cannot be None or empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not (0.0 <= alpha <= 1.0):
            error_msg = f"Alpha must be between 0.0 and 1.0, got {alpha}."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.metric_names = list(metric_names) # Ensure it's a list
        self.alpha = float(alpha)  # Smoothing factor, ensure float
        # Initialize with values that will be updated on first data
        self.min_vals = {m: float('inf') for m in self.metric_names}
        self.max_vals = {m: float('-inf') for m in self.metric_names}
        logger.info(f"DynamicNormalizer initialized for {len(self.metric_names)} metrics.")
    
    def update(self, score_matrix: np.ndarray) -> None:
        """
        Update min/max values based on new data using exponential smoothing.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics].
                          Metrics must correspond to self.metric_names in order.
                          
        Raises:
            ValueError: If score_matrix is None, not 2D, or columns don't match metric_names.
        """
        if score_matrix is None:
            error_msg = "score_matrix cannot be None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
             error_msg = f"score_matrix must be 2D, got {score_matrix.ndim}D shape {score_matrix.shape}."
             logger.error(error_msg)
             raise ValueError(error_msg)
        if score_matrix.shape[1] != len(self.metric_names):
            error_msg = (f"score_matrix column count ({score_matrix.shape[1]}) "
                         f"must match metric_names count ({len(self.metric_names)}).")
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.size == 0: # Check for empty array
             logger.warning("Received empty score_matrix. Skipping update.")
             return

        logger.debug(f"Updating normalizer with data shape: {score_matrix.shape}")
        num_docs = score_matrix.shape[0]
        
        for i, metric in enumerate(self.metric_names):
            col = score_matrix[:, i]
            # Handle potential NaNs or infs if present in data (optional, depends on requirements)
            # col = col[np.isfinite(col)] # Uncomment if you want to ignore NaN/inf
            if col.size == 0: # Check if column is empty after potential filtering
                 logger.warning(f"Column for metric '{metric}' is empty (after filtering). Skipping update for this metric.")
                 continue

            current_min = np.min(col)
            current_max = np.max(col)
            
            # Update with exponential smoothing
            if np.isinf(self.min_vals[metric]): # First update for this metric
                logger.debug(f"First update for metric '{metric}': min={current_min}, max={current_max}")
                self.min_vals[metric] = float(current_min)
                self.max_vals[metric] = float(current_max)
            else:
                # Exponential smoothing update
                old_min = self.min_vals[metric]
                old_max = self.max_vals[metric]
                self.min_vals[metric] = float((1 - self.alpha) * old_min + self.alpha * current_min)
                self.max_vals[metric] = float((1 - self.alpha) * old_max + self.alpha * current_max)
                logger.debug(f"Updated metric '{metric}': min {old_min:.6f}->{self.min_vals[metric]:.6f}, "
                             f"max {old_max:.6f}->{self.max_vals[metric]:.6f} (data min={current_min:.6f}, max={current_max:.6f})")
        logger.info(f"Normalizer updated successfully with {num_docs} documents.")

    
    def normalize(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0,1] range using current min/max.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics].
                          Metrics must correspond to self.metric_names in order.
        
        Returns:
            np.ndarray: Normalized score matrix of the same shape as input,
                        with values scaled to [0, 1].
                        
        Raises:
            ValueError: If score_matrix is None, not 2D, or columns don't match metric_names.
        """
        if score_matrix is None:
            error_msg = "score_matrix cannot be None for normalization."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
             error_msg = f"score_matrix must be 2D for normalization, got {score_matrix.ndim}D shape {score_matrix.shape}."
             logger.error(error_msg)
             raise ValueError(error_msg)
        if score_matrix.shape[1] != len(self.metric_names):
            error_msg = (f"score_matrix column count ({score_matrix.shape[1]}) "
                         f"must match metric_names count ({len(self.metric_names)}) for normalization.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Normalizing score matrix of shape: {score_matrix.shape}")
        normalized = np.zeros_like(score_matrix, dtype=np.float64) # Use float64 for precision during calculation
        num_docs = score_matrix.shape[0]
        
        for i, metric in enumerate(self.metric_names):
            min_val = self.min_vals[metric]
            max_val = self.max_vals[metric]
            range_val = max_val - min_val
            
            if range_val > 0.0: # Normal case: valid range
                normalized[:, i] = (score_matrix[:, i] - min_val) / range_val
                logger.debug(f"Normalized metric '{metric}' using range [{min_val:.6f}, {max_val:.6f}]")
            else:
                # Handle case where min == max (constant metric)
                # Assign a default value, typically 0.5 as in the original
                normalized[:, i] = 0.5 
                if np.isinf(min_val): # Truly uninitialized (shouldn't happen if update called first)
                     logger.warning(f"Metric '{metric}' appears uninitialized (inf range). Assigned 0.5.")
                else: # Genuine constant value
                     logger.debug(f"Metric '{metric}' has constant value ({min_val}). Assigned 0.5.")
        
        logger.info(f"Normalization completed for {num_docs} documents.")
        # Return as float32 or float64 depending on downstream needs, or let caller cast.
        # Assuming float64 is fine for precision. Can cast to float32 if needed.
        return normalized 
    
    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get current min/max ranges for all metrics.
        
        Returns:
            Dict[str, Tuple[float, float]]: A dictionary mapping metric names
            to their (min, max) tuples.
        """
        ranges = {m: (float(self.min_vals[m]), float(self.max_vals[m])) for m in self.metric_names}
        logger.debug(f"Retrieved current ranges: {ranges}")
        return ranges
