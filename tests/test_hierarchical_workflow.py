# tests/test_hierarchical_workflow.py
import numpy as np
from typing import Tuple, List
from zeromodel.core import HierarchicalVPM
from zeromodel.hierarchical_edge import HierarchicalEdgeProtocol

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


def test_hierarchical_vpm_workflow():
    """Test the hierarchical workflow demonstrated in the demo script"""
    # 1. Generate synthetic data
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)
    
    # 2. Create hierarchical VPM
    hvpm = HierarchicalVPM(metric_names=metric_names)
    hvpm.process(score_matrix, "Find uncertain large documents")
    
    # Verify hierarchical structure
    assert len(hvpm.levels) == 3
    for i in range(len(hvpm.levels) - 1):
        current = hvpm.get_level(i)
        next_level = hvpm.get_level(i+1)
        assert current["metadata"]["documents"] <= next_level["metadata"]["documents"]
        assert current["metadata"]["metrics"] <= next_level["metadata"]["metrics"]
    
    # Verify hierarchical decision process
    level, doc_idx, relevance = hvpm.get_decision(0)
    assert 0 <= level < 3
    assert 0 <= doc_idx < hvpm.get_level(level)["metadata"]["documents"]
    assert 0 <= relevance <= 1.0
    
    # Verify zoom behavior
    next_level = hvpm.zoom_in(level, doc_idx, 0)
    assert next_level > level or next_level == level  # Can stay at base level
    
    # Verify edge device hierarchical interaction
    tile0 = hvpm.get_tile(0)
    decision = HierarchicalEdgeProtocol.make_decision(tile0)
    assert isinstance(decision[3], bool)  # is_relevant