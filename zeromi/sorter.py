"""
Task-Agnostic Sorting Framework

This module provides the TaskSorter class which handles the critical task
of sorting documents and metrics based on task relevance, with implementations
optimized for both cloud and edge environments.

The key innovation: sorting happens during encoding, not at decision time,
which enables zero-model intelligence at the edge.
"""

import numpy as np
from typing import List, Dict, Tuple

class TaskSorter:
    """
    Task-agnostic sorter for Zero-Model Intelligence.
    
    This class handles sorting documents and metrics based on task relevance
    with implementations optimized for both cloud and edge environments.
    
    For edge devices (<25KB memory), uses simple rule-based sorting.
    For cloud systems, can incorporate learning from feedback.
    """
    
    def __init__(self, metric_names: List[str], default_weights: Dict[str, float] = None):
        """
        Initialize the sorter with metric names.
        
        Args:
            metric_names: Names of all metrics being tracked
            default_weights: Optional default weights for metrics
        """
        self.metric_names = metric_names
        self.weights = default_weights or self._auto_weights()
    
    def _auto_weights(self) -> Dict[str, float]:
        """Generate reasonable default weights based on metric properties"""
        weights = {}
        for metric in self.metric_names:
            # Default weights based on common patterns
            if "uncertainty" in metric.lower():
                weights[metric] = 0.8
            elif "size" in metric.lower() or "length" in metric.lower():
                weights[metric] = 0.7
            elif "quality" in metric.lower() or "score" in metric.lower():
                weights[metric] = 0.9
            elif "novelty" in metric.lower() or "diversity" in metric.lower():
                weights[metric] = 0.6
            else:
                weights[metric] = 0.5  # Neutral default
        return weights
    
    def update_weights(self, task_description: str, feedback: Dict[str, float] = None):
        """
        Update weights based on task description and optional feedback.
        
        Args:
            task_description: Natural language description of the task
            feedback: Optional feedback on previous decisions
        """
        # Simple rule-based approach (works on edge devices)
        task_lower = task_description.lower()
        
        # Reset to defaults
        self.weights = self._auto_weights()
        
        # Apply task-specific adjustments
        if "uncertainty" in task_lower or "confidence" in task_lower:
            for metric in self.metric_names:
                if "uncertainty" in metric.lower():
                    self.weights[metric] = 1.0
                elif "confidence" in metric.lower():
                    self.weights[metric] = 1.0
        
        if "large" in task_lower or "big" in task_lower or "size" in task_lower:
            for metric in self.metric_names:
                if "size" in metric.lower() or "length" in metric.lower():
                    self.weights[metric] = 0.8
        
        if "quality" in task_lower or "good" in task_lower or "best" in task_lower:
            for metric in self.metric_names:
                if "quality" in metric.lower() or "score" in metric.lower():
                    self.weights[metric] = 0.9
        
        if "novel" in task_lower or "new" in task_lower or "diverse" in task_lower:
            for metric in self.metric_names:
                if "novelty" in metric.lower() or "diversity" in metric.lower():
                    self.weights[metric] = 0.7
        
        # Incorporate feedback if available (cloud only)
        if feedback:
            for metric, score in feedback.items():
                if metric in self.weights:
                    # Adjust weight based on feedback (0-1 scale)
                    self.weights[metric] = 0.7 * self.weights[metric] + 0.3 * score
            
            # Normalize weights to 0-1 range
            max_weight = max(self.weights.values()) if self.weights else 1.0
            if max_weight > 0:
                for metric in self.weights:
                    self.weights[metric] /= max_weight
    
    def sort_matrix(self, score_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sort documents and metrics by task relevance.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
        
        Returns:
            (sorted_matrix, metric_order, doc_order)
        """
        # Calculate metric importance
        metric_importance = np.array([self.weights.get(m, 0) for m in self.metric_names])
        metric_order = np.argsort(metric_importance)[::-1]  # Most important first
        
        # Sort metrics
        sorted_by_metric = score_matrix[:, metric_order]
        
        # Calculate document relevance (weighted sum)
        doc_relevance = np.zeros(score_matrix.shape[0])
        for i in range(len(metric_importance)):
            doc_relevance += metric_importance[i] * sorted_by_metric[:, i]
        
        # Sort documents
        doc_order = np.argsort(doc_relevance)[::-1]  # Most relevant first
        
        # Final sorted matrix
        sorted_matrix = sorted_by_metric[doc_order]
        
        return sorted_matrix, metric_order, doc_order
    
    def get_weights(self) -> Dict[str, float]:
        """Get current metric weights"""
        return self.weights.copy()