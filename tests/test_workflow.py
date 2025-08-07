# tests/test_demo_workflow.py

from zeromodel.core import ZeroModel
from tests.utils import generate_synthetic_data
import numpy as np


# In tests/test_demo_workflow.py
def test_complete_zeromodel_workflow():
    """Test the complete workflow demonstrated in the demo script"""
    # 1. Generate synthetic data
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)

    # 2. Process with zeromodel
    zeromodel = ZeroModel(metric_names)
    # Set task
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")

    # 3. Encode as visual policy map
    vpm = zeromodel.encode(output_precision='uint8')
    assert vpm is not None
    assert vpm.shape[0] == 100  # Should match number of documents
    assert vpm.dtype == np.uint8

    # 4. Test critical tile extraction
    tile = zeromodel.get_critical_tile()
    assert tile is not None
    assert len(tile) > 0

    # 5. Test decision making
    doc_idx, relevance = zeromodel.get_decision()
    assert 0 <= doc_idx < 100
    assert 0 <= relevance <= 1.0
