"""
Dynamic Range Adaptation

This module provides the DynamicNormalizer class which handles normalization
of scores to handle value drift over time. This is critical for long-term
viability of the zeromodel system as score distributions may change.
"""

from typing import Dict, List, Tuple

import numpy as np


class DynamicNormalizer:
    """
    Handles dynamic normalization of scores to handle value drift over time.
    
    This is critical because:
    - Score ranges may change as policies improve
    - New documents may have scores outside previous ranges
    - Normalization must be consistent across time
    
    The normalizer tracks min/max values for each metric and updates them
    incrementally as new data arrives.
    """
    
    def __init__(self, metric_names: List[str], alpha: float = 0.1):
        """
        Initialize the normalizer.
        
        Args:
            metric_names: Names of all metrics being tracked
            alpha: Smoothing factor for updating min/max (0-1)
        """
        self.metric_names = metric_names
        self.alpha = alpha  # Smoothing factor
        self.min_vals = {m: float('inf') for m in metric_names}
        self.max_vals = {m: float('-inf') for m in metric_names}
    
    def update(self, score_matrix: np.ndarray) -> None:
        """
        Update min/max values based on new data.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
        """
        for i, metric in enumerate(self.metric_names):
            col = score_matrix[:, i]
            current_min = np.min(col)
            current_max = np.max(col)
            
            # Update with smoothing
            if self.min_vals[metric] == float('inf'):
                self.min_vals[metric] = current_min
            else:
                self.min_vals[metric] = (1 - self.alpha) * self.min_vals[metric] + self.alpha * current_min
            
            if self.max_vals[metric] == float('-inf'):
                self.max_vals[metric] = current_max
            else:
                self.max_vals[metric] = (1 - self.alpha) * self.max_vals[metric] + self.alpha * current_max
    
    def normalize(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0,1] range using current min/max.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
        
        Returns:
            Normalized score matrix
        """
        normalized = np.zeros_like(score_matrix)
        for i, metric in enumerate(self.metric_names):
            min_val = self.min_vals[metric]
            max_val = self.max_vals[metric]
            if max_val > min_val:
                normalized[:, i] = (score_matrix[:, i] - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5  # Default for constant metrics
        return normalized
    
    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get current min/max ranges for all metrics"""
        return {m: (self.min_vals[m], self.max_vals[m]) for m in self.metric_names}