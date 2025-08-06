import numpy as np
from typing import List, Tuple

def generate_synthetic_data(num_docs: int = 100, num_metrics: int = 50) -> Tuple[np.ndarray, List[str]]:
    """Generate synthetic score data for demonstration"""
    # Create realistic score distributions
    scores = np.zeros((num_docs, num_metrics))
    
    # Uncertainty: higher for early documents
    scores[:, 0] = np.linspace(0.9, 0.1, num_docs)
    
    # Size: random but correlated with uncertainty
    scores[:, 1] = 0.5 + 0.5 * np.random.rand(num_docs) - 0.3 * scores[:, 0]
    
    # Quality: higher for later documents
    scores[:, 2] = np.linspace(0.2, 0.9, num_docs)
    
    # Novelty: random
    scores[:, 3] = np.random.rand(num_docs)
    
    # Coherence: correlated with quality
    scores[:, 4] = scores[:, 2] * 0.7 + 0.3 * np.random.rand(num_docs)
    
    # Fill remaining metrics with random values
    for i in range(5, num_metrics):
        scores[:, i] = np.random.rand(num_docs)
    
    # Ensure values are in [0,1] range
    scores = np.clip(scores, 0, 1)
    
    # Create metric names
    metric_names = [
        "uncertainty", "size", "quality", "novelty", "coherence",
        "relevance", "diversity", "complexity", "readability", "accuracy"
    ]
    # Add numbered metrics for the rest
    for i in range(10, num_metrics):
        metric_names.append(f"metric_{i}")
    
    return scores[:num_docs, :num_metrics], metric_names[:num_metrics]
