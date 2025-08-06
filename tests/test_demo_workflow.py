# examples/basic_demo.py
import numpy as np
from typing import Tuple, List
from zeromodel.core import ZeroModel
from zeromodel.edge import EdgeProtocol

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


# tests/test_demo_workflow.py
def test_complete_zeromodel_workflow():
    """Test the complete workflow demonstrated in the demo script"""
    # 1. Generate synthetic data
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)
    
    # 2. Process with zeromodel
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    zeromodel.process(score_matrix)
    
    # Verify processing completed correctly
    assert zeromodel.sorted_matrix is not None
    assert zeromodel.doc_order is not None
    assert zeromodel.metric_order is not None
    
    # 3. Verify encoding
    vpm = zeromodel.encode()
    assert vpm.shape[0] == score_matrix.shape[0]
    assert vpm.shape[1] == (score_matrix.shape[1] + 2) // 3
    assert vpm.shape[2] == 3  # RGB channels
    
    # 4. Verify edge device interaction
    tile = zeromodel.get_critical_tile()
    assert len(tile) > 0
    decision = EdgeProtocol.make_decision(tile)
    assert isinstance(decision[2], bool)  # is_relevant
    
    # 5. Verify top decision
    doc_idx, relevance = zeromodel.get_decision()
    assert 0 <= doc_idx < score_matrix.shape[0]
    assert 0 <= relevance <= 1.0