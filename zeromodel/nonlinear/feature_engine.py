# zeromodel/feature_engine.py
"""
Feature Engineering for ZeroModel to handle non-linear patterns.
"""

import logging
from typing import Any, Callable, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Handles non-linear feature transformations for ZeroModel.
    """
    def __init__(self):
        """
        Initializes the feature engine with a registry of transformation functions.
        """
        self.transform_registry: Dict[str, Callable] = {
            "XOR": self._xor_transform,
            "RADIAL": self._radial_transform,
            "PRODUCT": self._product_transform,
            "DIFFERENCE": self._difference_transform,
            # Add more as needed
        }
        self.applied_transforms: List[Dict[str, Any]] = [] # Track transformations
        self.engineered_metric_names: List[str] = []

    def transform(self, matrix: np.ndarray, pattern_type: str, original_metric_names: List[str]) -> np.ndarray:
        """
        Applies a registered transformation to the input matrix.

        Args:
            matrix: The input score matrix (normalized).
            pattern_type: The type of transformation to apply (key in registry).
            original_metric_names: The names of the original metrics.

        Returns:
            np.ndarray: The transformed matrix (original + new features).
        """
        logger.debug(f"Applying feature transformation: {pattern_type}")
        transform_fn = self.transform_registry.get(pattern_type)
        
        if transform_fn is None:
            logger.debug(f"No specific transform for '{pattern_type}', using identity.")
            transform_fn = self._identity_transform

        try:
            # Store original shape for metadata
            original_docs, original_metrics = matrix.shape
            
            # Apply transformation
            transformed_matrix = transform_fn(matrix, original_metric_names)
            
            # Record transformation details
            self.applied_transforms.append({
                "type": pattern_type,
                "input_shape": (original_docs, original_metrics),
                "output_shape": transformed_matrix.shape,
                "timestamp": np.datetime64('now')
            })
            
            logger.info(f"Feature transformation '{pattern_type}' applied. New shape: {transformed_matrix.shape}")
            return transformed_matrix
            
        except Exception as e:
            logger.error(f"Error applying transformation '{pattern_type}': {e}")
            # Fallback to identity transform
            identity_result = self._identity_transform(matrix, original_metric_names)
            self.applied_transforms.append({
                "type": "IDENTITY_FALLBACK",
                "input_shape": matrix.shape,
                "output_shape": identity_result.shape,
                "error": str(e),
                "timestamp": np.datetime64('now')
            })
            return identity_result

    def _identity_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Passes the matrix through unchanged."""
        self.engineered_metric_names = list(metric_names) # No new names
        logger.debug("Applied identity transformation.")
        return matrix

    def _xor_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Generates features helpful for XOR-like patterns."""
        if matrix.shape[1] < 2:
            logger.warning("XOR transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix
            
        # Assume first two metrics are the primary coordinates for XOR
        m1, m2 = matrix[:, 0], matrix[:, 1]
        
        # Standard XOR-separating features
        product = m1 * m2
        abs_diff = np.abs(m1 - m2)
        
        # Stack original and new features
        result = np.column_stack([matrix, product, abs_diff])
        
        # Update metric names
        self.engineered_metric_names = list(metric_names) + ["xor_product", "xor_abs_diff"]
        logger.debug("Applied XOR feature transformation.")
        return result

    def _radial_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Generates radial/distance-based features."""
        if matrix.shape[1] < 2:
            logger.warning("Radial transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix

        # Assume first two metrics are X, Y coordinates centered at 0.5
        x, y = matrix[:, 0], matrix[:, 1]
        center_x, center_y = 0.5, 0.5
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        angle = np.arctan2(y - center_y, x - center_x)
        
        result = np.column_stack([matrix, distance, angle])
        self.engineered_metric_names = list(metric_names) + ["radial_distance", "radial_angle"]
        logger.debug("Applied radial feature transformation.")
        return result

    def _product_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Adds pairwise products of metrics."""
        if matrix.shape[1] < 2:
            logger.warning("Product transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix
            
        cols = matrix.shape[1]
        new_features = []
        new_names = []
        # Simple pairwise products for first few metrics
        for i in range(min(3, cols)):
            for j in range(i+1, min(4, cols)):
                new_features.append(matrix[:, i] * matrix[:, j])
                new_names.append(f"product_{metric_names[i]}_{metric_names[j]}")
        
        if new_features:
            result = np.column_stack([matrix] + new_features)
            self.engineered_metric_names = list(metric_names) + new_names
        else:
            result = matrix
            self.engineered_metric_names = list(metric_names)
            
        logger.debug(f"Applied product feature transformation. Added {len(new_names)} features.")
        return result

    def _difference_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Adds pairwise absolute differences of metrics."""
        if matrix.shape[1] < 2:
            logger.warning("Difference transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix
            
        cols = matrix.shape[1]
        new_features = []
        new_names = []
        # Simple pairwise differences for first few metrics
        for i in range(min(3, cols)):
            for j in range(i+1, min(4, cols)):
                new_features.append(np.abs(matrix[:, i] - matrix[:, j]))
                new_names.append(f"abs_diff_{metric_names[i]}_{metric_names[j]}")
        
        if new_features:
            result = np.column_stack([matrix] + new_features)
            self.engineered_metric_names = list(metric_names) + new_names
        else:
            result = matrix
            self.engineered_metric_names = list(metric_names)
            
        logger.debug(f"Applied difference feature transformation. Added {len(new_names)} features.")
        return result

    def get_metric_names(self) -> List[str]:
        """Gets the names of the metrics after the last transformation."""
        return self.engineered_metric_names

    def get_transformation_log(self) -> List[Dict[str, Any]]:
        """Gets a log of all applied transformations."""
        return self.applied_transforms

    def clear_log(self):
        """Clears the transformation log."""
        self.applied_transforms.clear()
