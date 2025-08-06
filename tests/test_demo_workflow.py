# tests/test_demo_workflow.py

from zeromodel.core import ZeroModel
from zeromodel.edge import EdgeProtocol
from tests.utils import generate_synthetic_data

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
    
    # 5. Verify top decision
    doc_idx, relevance = zeromodel.get_decision()
    assert 0 <= doc_idx < score_matrix.shape[0]
    assert 0 <= relevance <= 1.0