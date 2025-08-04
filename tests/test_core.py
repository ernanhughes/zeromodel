import numpy as np
from zeromi import ZeroMI

def test_zeromi_initialization():
    metric_names = ["uncertainty", "size", "quality"]
    zeromi = ZeroMI(metric_names)
    assert zeromi.metric_names == metric_names
    assert zeromi.precision == 8

def test_zeromi_processing():
    metric_names = ["uncertainty", "size", "quality"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9],
        [0.6, 0.7, 0.3],
        [0.2, 0.9, 0.5]
    ])
    
    zeromi = ZeroMI(metric_names)
    zeromi.set_task("Find uncertain large documents")
    zeromi.process(score_matrix)
    
    vpm = zeromi.encode()
    assert vpm.shape == (3, 1, 3)  # 3 docs, 1 pixel width, 3 channels
    
    tile = zeromi.get_critical_tile()
    assert len(tile) == 16  # 4 header bytes + 9 pixel bytes + 3 padding
    
    doc_idx, relevance = zeromi.get_decision()
    assert 0 <= doc_idx < 3
    assert 0 <= relevance <= 1.0