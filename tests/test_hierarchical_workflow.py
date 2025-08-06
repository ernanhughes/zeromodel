# tests/test_hierarchical_workflow.py
from zeromodel.core import HierarchicalVPM
from zeromodel.hierarchical_edge import HierarchicalEdgeProtocol
from tests.utils import generate_synthetic_data


def test_hierarchical_vpm_workflow():
    """Test the hierarchical workflow demonstrated in the demo script"""
    # 1. Generate synthetic data
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)
    
    # 2. Create hierarchical VPM
    hvpm = HierarchicalVPM(metric_names=metric_names)
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    
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
    print(f"Decision from edge device: {decision}")