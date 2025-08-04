"""
Task-Agnostic Sorting Framework

This module provides the TaskSorter class which handles the critical task
of sorting documents and metrics based on task relevance, with implementations
optimized for both cloud and edge environments.

The key innovation: sorting happens during encoding, not at decision time,
which enables zero-model intelligence at the edge.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


class TaskSorter:
    def __init__(self, metric_names: List[str], config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            metric_names: Names of all metrics being tracked
            config: Configuration dictionary
        """
        self.metric_names = metric_names
        self.config = config
        
        # Get task sorter parameters
        sorter_config = config.get('zeromodel', {}).get('task_sorter', {})
        self.ib_threshold = sorter_config.get('ib_threshold', 0.7)
        self.adaptive_threshold = sorter_config.get('adaptive_threshold', True)
        self.semantic_groups = sorter_config.get('semantic_groups', {
            "uncertainty": ["uncertainty", "confidence", "ambiguity", "doubt"],
            "size": ["size", "length", "scale", "magnitude"],
            "quality": ["quality", "score", "rating", "value"],
            "novelty": ["novelty", "diversity", "originality", "innovation"]
        })
        
        self.metric_embeddings = self._generate_metric_embeddings()
        self.task_history = []
    
    def _generate_metric_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate semantic embeddings with configuration-based approach"""
        embeddings = {}
        
        # Use semantic groups from config
        for metric in self.metric_names:
            # Find semantic group
            group_id = 0
            for i, (group_name, keywords) in enumerate(self.semantic_groups.items()):
                if any(keyword in metric.lower() for keyword in keywords):
                    group_id = i + 1
                    break
            
            # Create embedding based on semantic group
            embeddings[metric] = np.array([
                group_id / len(self.semantic_groups),  # Semantic group
                len(metric) / 20.0,               # Metric length (proxy for complexity)
                hash(metric) % 256 / 255.0,       # Unique identifier
                1.0                               # Constant dimension
            ])
        
        # Normalize embeddings
        for metric in embeddings:
            norm = np.linalg.norm(embeddings[metric])
            if norm > 0:
                embeddings[metric] = embeddings[metric] / norm
        
        return embeddings
    
    def update_weights(self, task_description: str, feedback: Dict[str, float] = None):
        """
        Update weights using Information Bottleneck principles with config options.
        """
        # Generate task embedding
        task_embedding = self._generate_task_embedding(task_description)
        
        # Calculate information retention for each metric
        weights = {}
        for metric, embedding in self.metric_embeddings.items():
            # Calculate similarity (mutual information proxy)
            similarity = np.dot(task_embedding, embedding) / (
                np.linalg.norm(task_embedding) * np.linalg.norm(embedding) + 1e-10
            )
            
            # Apply thresholding based on config
            if self.config.get('zeromodel', {}).get('task_sorter', {}).get('soft_thresholding', True):
                # Soft thresholding (sigmoid function)
                threshold = self.ib_threshold
                weights[metric] = 1.0 / (1.0 + np.exp(-10.0 * (similarity - threshold)))
            else:
                # Hard thresholding
                weights[metric] = similarity if similarity > self.ib_threshold else 0.0
        
        # Normalize weights
        max_weight = max(weights.values()) if weights else 1.0
        if max_weight > 0:
            for metric in weights:
                weights[metric] /= max_weight
        
        # Incorporate feedback if available
        if feedback:
            for metric, score in feedback.items():
                if metric in weights:
                    weights[metric] = 0.7 * weights[metric] + 0.3 * score
        
        self.weights = weights
        self.task_history.append((task_description, weights.copy()))
    
    def _generate_task_embedding(self, task_description: str) -> np.ndarray:
        """Generate lightweight embedding for task description"""
        # Simple hash-based embedding for edge compatibility
        hash_val = hash(task_description) % (2**32)
        return np.array([
            (hash_val >> 24) & 0xFF,
            (hash_val >> 16) & 0xFF,
            (hash_val >> 8) & 0xFF,
            hash_val & 0xFF
        ]) / 255.0
        
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