# tests/test_hunter.py
from zeromodel import HierarchicalVPM, ZeroModel
from zeromodel.vpm.hunter import VPMHunter
import numpy as np

def test_vpm_hunter_hierarchical_basic():
    """Test basic VPM hunter functionality with HierarchicalVPM."""
    metric_names = ["m1", "m2", "m3"]
    score_matrix = np.random.rand(100, 3)
    
    hvpm = HierarchicalVPM(metric_names, num_levels=3)
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY m1 DESC")
    
    hunter = VPMHunter(hvpm, tau=0.9, max_steps=5)
    target_id, confidence, audit_trail = hunter.hunt()
    
    assert isinstance(target_id, tuple)
    assert len(target_id) == 2 # (level, doc_idx)
    assert 0 <= confidence <= 1.0
    assert len(audit_trail) > 0
    assert all("step" in step for step in audit_trail)
    print("Hierarchical VPM Hunter test passed.")

def test_vpm_hunter_base_zeromodel():
    """Test VPM hunter functionality with base ZeroModel."""
    metric_names = ["m1", "m2", "m3"]
    score_matrix = np.random.rand(10, 3)
    
    zm = ZeroModel(metric_names)
    zm.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY m1 DESC")
    
    hunter = VPMHunter(zm, tau=0.5, max_steps=4, aoi_size_sequence=(5, 3, 1))
    target_id, confidence, audit_trail = hunter.hunt()
    
    assert isinstance(target_id, int) # doc_idx for base ZM
    assert 0 <= confidence <= 1.0
    assert len(audit_trail) > 0
    print("Base ZeroModel Hunter test passed.")