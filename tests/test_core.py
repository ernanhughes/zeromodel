import numpy as np
from zeromodel import ZeroModel

def test_zeromodel_initialization():
    metric_names = ["uncertainty", "size", "quality"]
    zeromodel = zeromodel(metric_names)
    assert zeromodel.metric_names == metric_names
    assert zeromodel.precision == 8

def test_zeromodel_processing():
    metric_names = ["uncertainty", "size", "quality"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9],
        [0.6, 0.7, 0.3],
        [0.2, 0.9, 0.5]
    ])
    
    zeromodel = zeromodel(metric_names)
    zeromodel.set_task("Find uncertain large documents")
    zeromodel.process(score_matrix)
    
    vpm = zeromodel.encode()
    assert vpm.shape == (3, 1, 3)  # 3 docs, 1 pixel width, 3 channels
    
    tile = zeromodel.get_critical_tile()
    assert len(tile) == 16  # 4 header bytes + 9 pixel bytes + 3 padding
    
    doc_idx, relevance = zeromodel.get_decision()
    assert 0 <= doc_idx < 3
    assert 0 <= relevance <= 1.0