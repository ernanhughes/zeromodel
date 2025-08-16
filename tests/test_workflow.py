# tests/test_demo_workflow.py

import numpy as np

from tests.utils import generate_synthetic_data
from zeromodel.core import ZeroModel
from zeromodel.vpm.encoder import VPMEncoder


# In tests/test_demo_workflow.py
def test_complete_zeromodel_workflow():
    """Test the complete workflow demonstrated in the demo script"""
    # 1. Generate synthetic data
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)

    # 2. Process with zeromodel
    zeromodel = ZeroModel(metric_names)
    # Set task
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")

    # 3. Encode as visual policy map (using encoder on sorted_matrix)
    vpm = VPMEncoder('uint8').encode(zeromodel.sorted_matrix, output_precision='uint8')
    assert vpm is not None
    assert vpm.shape[0] == 100  # Should match number of documents
    assert vpm.dtype == np.uint8

    # 4. Test critical tile extraction (encoder-based)
    tile = VPMEncoder('float32').get_critical_tile(zeromodel.sorted_matrix)
    assert tile is not None
    assert len(tile) > 0

    # 5. Test decision making (metric 0)
    doc_idx, relevance = zeromodel.get_decision_by_metric(0)
    assert 0 <= doc_idx < 100
    assert 0 <= relevance <= 1.0
