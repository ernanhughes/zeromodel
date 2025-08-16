import time

import numpy as np
import pytest

from zeromodel import HierarchicalVPM, ZeroModel
from zeromodel.vpm.encoder import VPMEncoder


@pytest.mark.skip(reason="This test is for performance benchmarking and takes a long time to run")
def test_performance_scalability():
    """Test performance with large datasets and measure scalability"""
    # Test with medium dataset (1,000 documents × 20 metrics)
    metric_names = [f"metric_{i}" for i in range(20)]
    medium_matrix = np.random.rand(1000, 20)

    start = time.time()
    zeromodel = ZeroModel(metric_names)
    zeromodel.prepare(medium_matrix, "SELECT * FROM virtual_index ORDER BY metric_0 DESC")

    medium_time = time.time() - start

    # Verify processing completed
    assert zeromodel.sorted_matrix is not None
    assert zeromodel.doc_order is not None
  
    # Test with large dataset (10,000 documents × 50 metrics)
    metric_names = [f"metric_{i}" for i in range(50)]
    large_matrix = np.random.rand(10000, 50)
    
    start = time.time()
    zeromodel = ZeroModel(metric_names)
    zeromodel.prepare(large_matrix, "SELECT * FROM virtual_index ORDER BY metric_0 DESC, metric_1 ASC")
    large_time = time.time() - start
    
    # Verify processing completed
    assert zeromodel.sorted_matrix is not None
    assert zeromodel.doc_order is not None
    assert zeromodel.metric_order is not None
    
    # Test hierarchical processing with large dataset
    start = time.time()
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=5
    )
    hvpm.process(large_matrix, "SELECT * FROM virtual_index ORDER BY metric_0 DESC")
    hierarchical_time = time.time() - start
    
    # Verify hierarchical processing completed
    assert len(hvpm.levels) == 3
    
    # Test encoding performance
    start = time.time()
    vpm = VPMEncoder('float32').encode(zeromodel.sorted_matrix)
    encode_time = time.time() - start
    
    # Verify encoding completed
    assert vpm is not None
    assert vpm.shape[0] == large_matrix.shape[0]
    assert vpm.shape[1] == (large_matrix.shape[1] + 2) // 3
    
    # Test critical tile extraction
    start = time.time()
    tile = VPMEncoder('float32').get_critical_tile(zeromodel.sorted_matrix)
    tile_time = time.time() - start
    
    # Verify tile extraction completed
    assert tile is not None
    assert len(tile) > 0
    
    # Test decision making performance
    start = time.time()
    doc_idx, relevance = zeromodel.get_decision_by_metric(0)
    decision_time = time.time() - start
    
    # Verify decision making completed
    assert 0 <= doc_idx < large_matrix.shape[0]
    assert 0 <= relevance <= 1.0
    
    # Print performance metrics (for informational purposes)
    print("\nPerformance Metrics:")
    print(f"Medium dataset (1,000×20) processing: {medium_time:.4f} seconds")
    print(f"Large dataset (10,000×50) processing: {large_time:.4f} seconds")
    print(f"Hierarchical processing: {hierarchical_time:.4f} seconds")
    print(f"Encoding: {encode_time:.4f} seconds")
    print(f"Critical tile extraction: {tile_time:.4f} seconds")
    print(f"Decision making: {decision_time:.4f} seconds")
    
    # Verify reasonable performance (adjust thresholds as needed for your system)
    assert medium_time < 0.7   # Should process medium dataset quickly
    assert large_time < 10.0   # Should process large dataset in reasonable time
    assert hierarchical_time < 12.0  # Hierarchical processing should be efficient
    # assert encode_time < 0.15   # Encoding should be very fast
    assert tile_time < 0.01    # Tile extraction should be extremely fast
    assert decision_time < 0.01  # Decision making should be extremely fast
    
    # Test with extremely large dataset (100,000 documents × 100 metrics)
    # This might be too large for some systems, so we'll skip if it takes too long
    try:
        metric_names = [f"metric_{i}" for i in range(100)]
        huge_matrix = np.random.rand(100000, 100)
        
        start = time.time()
        zeromodel = ZeroModel(metric_names)
        zeromodel.prepare(huge_matrix, "SELECT * FROM virtual_index ORDER BY metric_0 DESC")
        huge_time = time.time() - start
        
        # Verify processing completed
        assert zeromodel.sorted_matrix is not None
        
        print(f"Huge dataset (100,000×100) processing: {huge_time:.4f} seconds")
        # This might be slow, but should complete within a reasonable timeframe
        assert huge_time < 180.0  # Should complete within 1 minute
    except MemoryError:
        pytest.skip("System doesn't have enough memory for huge dataset test")
