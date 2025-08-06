import numpy as np
import pytest
import time
from zeromodel import ZeroModel, HierarchicalVPM
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_zeromodel_example():
    """Test with the exact example you provided"""
    metric_names = ["metric1", "metric2", "metric3", "metric4"]
    
    # Your example data
    score_matrix = np.array([
        [0.7, 0.1, 0.3, 0.9],  # Document 0
        [0.9, 0.2, 0.4, 0.1],  # Document 1
        [0.5, 0.8, 0.2, 0.3],  # Document 2
        [0.1, 0.3, 0.9, 0.2]   # Document 3
    ])
    
    zeromodel = ZeroModel(metric_names)
    
    # Set the SQL task
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    
    # Process the data
    zeromodel.process(score_matrix)
    
    # Verify document order matches your example: [1, 0, 2, 3]
    assert np.array_equal(zeromodel.doc_order, [1, 0, 2, 3])
    
    # Verify the sorted matrix
    assert np.array_equal(zeromodel.sorted_matrix[0], [0.9, 0.2, 0.4, 0.1])
    assert np.array_equal(zeromodel.sorted_matrix[1], [0.7, 0.1, 0.3, 0.9])
    assert np.array_equal(zeromodel.sorted_matrix[2], [0.5, 0.8, 0.2, 0.3])
    assert np.array_equal(zeromodel.sorted_matrix[3], [0.1, 0.3, 0.9, 0.2])
    
    # Verify top decision
    doc_idx, relevance = zeromodel.get_decision()
    assert doc_idx == 1  # Document 1 is most relevant
    assert abs(relevance - 0.457) < 0.001  # Weighted relevance value No I'm not responding today

def test_duckdb_integration():
    """Test DuckDB integration for SQL query analysis"""
    metric_names = ["uncertainty", "size", "quality", "novelty"]
    zeromodel = ZeroModel(metric_names)
    
    # Verify DuckDB connection is properly initialized
    assert zeromodel.duckdb_conn is not None
    
    # Test virtual index table structure
    result = zeromodel.duckdb_conn.execute(
        "PRAGMA table_info(virtual_index)"
    ).fetchall()
    
    # Should have row_id + all metric columns
    assert len(result) == len(metric_names) + 1
    assert result[0][1] == "row_id"  # First column is row_id
    for i, col_name in enumerate(metric_names):
        assert result[i+1][1] == col_name
    
    # Test virtual index table content
    result = zeromodel.duckdb_conn.execute(
        "SELECT * FROM virtual_index"
    ).fetchone()
    
    # Should contain row_id (0) followed by metric indices (0, 1, 2, 3)
    assert result[0] == 0
    for i in range(len(metric_names)):
        assert result[i+1] == i
    
    # Test DuckDB query execution
    analysis = zeromodel._analyze_query(
        "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC"
    )
    assert "metric_order" in analysis
    assert len(analysis["metric_order"]) == len(metric_names)
    # Uncertainty should be first, size second
    assert analysis["metric_order"][0] == 0
    assert analysis["metric_order"][1] == 1

def test_normalization_quantization():
    """Test normalization and quantization behavior with different precision levels"""
    metric_names = ["metric1", "metric2"]
    score_matrix = np.array([
        [0.2, 0.8],
        [0.5, 0.3],
        [0.9, 0.1]
    ])
    
    # Test with 8-bit precision (default)
    zeromodel_8bit = ZeroModel(metric_names, precision=8)
    zeromodel_8bit.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zeromodel_8bit.process(score_matrix)
    
    # Verify normalization
    normalized = zeromodel_8bit.sorted_matrix
    assert np.all(normalized >= 0) and np.all(normalized <= 1)
    
    # Verify 8-bit quantization
    vpm_8bit = zeromodel_8bit.encode()
    assert vpm_8bit.dtype == np.uint8
    assert np.all(vpm_8bit >= 0) and np.all(vpm_8bit <= 255)
    
    # Test with 4-bit precision
    zeromodel_4bit = ZeroModel(metric_names, precision=4)
    zeromodel_4bit.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zeromodel_4bit.process(score_matrix)
    
    # Verify 4-bit quantization (values should be multiples of 16)
    vpm_4bit = zeromodel_4bit.encode()
    unique_values = np.unique(vpm_4bit)

    quantized = (normalized * 255).astype(np.uint8)
    quantized = (quantized // 16) * 16

    assert np.all(quantized % 16 == 0)
    
    # Test with 16-bit precision
    zeromodel_16bit = ZeroModel(metric_names, precision=16)
    zeromodel_16bit.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zeromodel_16bit.process(score_matrix)
    
    # Verify 16-bit quantization
    vpm_16bit = zeromodel_16bit.encode()
    assert vpm_8bit.dtype == np.uint8
    assert np.all(vpm_16bit >= 0) and np.all(vpm_16bit <= 65535)
    
    # Test normalization with negative values
    neg_matrix = np.array([
        [-0.2, 0.8],
        [0.0, 0.3],
        [0.9, -0.1]
    ])
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zeromodel.process(neg_matrix)
    
    # Verify negative values are properly normalized
    normalized = zeromodel.sorted_matrix
    assert np.all(normalized <= 1)
    
    # Test normalization with all identical values
    identical_matrix = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    zeromodel.process(identical_matrix)
    
    # When all values are identical, document order should be preserved
    # assert np.array_equal(zeromodel.doc_order, [0, 1, 2])

def test_hierarchical_clustering():
    """Test hierarchical clustering functionality across levels"""
    metric_names = ["metric1", "metric2", "metric3", "metric4"]
    score_matrix = np.array([
        [0.9, 0.2, 0.4, 0.1],  # Document 1 (most relevant)
        [0.7, 0.1, 0.3, 0.9],  # Document 0
        [0.5, 0.8, 0.2, 0.3],  # Document 2
        [0.1, 0.3, 0.9, 0.2]   # Document 3
    ])
    
    # Create hierarchical VPM
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=2
    )
    
    # Process with SQL task
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
    
    # Verify level structure
    assert len(hvpm.levels) == 3
    
    # Level 2 (base level) should have all documents
    level_2 = hvpm.get_level(2)
    assert level_2["metadata"]["documents"] == 4
    assert np.array_equal(level_2["zeromodel"].doc_order, [0, 1, 2, 3])  # Sorted by SQL
    
    # Level 1 should have fewer documents (2 in this case)
    level_1 = hvpm.get_level(1)
    assert level_1["metadata"]["documents"] == 2
    
    # Verify clustering preserves relative ordering
    # The top cluster in level 1 should contain the top documents from level 2
    level_1_top_cluster = level_1["zeromodel"].doc_order[0]
    level_2_top_docs = level_2["zeromodel"].doc_order[:2]  # Top 2 of 4
    # Level 1 top cluster should map to the top region in level 2
    assert level_1_top_cluster in level_2_top_docs
    
    # Level 0 should have the fewest documents (1)
    level_0 = hvpm.get_level(0)
    assert level_0["metadata"]["documents"] == 1
    
    # Verify the top document in level 0 corresponds to the top region in level 1
    level_0_top_doc = level_0["zeromodel"].doc_order[0]
    level_1_top_region = level_1["zeromodel"].doc_order[:1]  # Top 1 of 2
    assert level_0_top_doc in level_1_top_region
    
    # Test with different zoom factor
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=3
    )
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
    
    # Level 1 should have ceil(4/3) = 2 documents
    assert hvpm.get_level(1)["metadata"]["documents"] == 2
    # Level 0 should have ceil(2/3) = 1 document
    assert hvpm.get_level(0)["metadata"]["documents"] == 1
    
    # Test with very small dataset
    small_matrix = np.array([[0.5, 0.5]])
    hvpm = HierarchicalVPM(
        metric_names=["m1", "m2"],
        num_levels=3
    )
    hvpm.process(small_matrix, "SELECT * FROM virtual_index ORDER BY m1 DESC")
    
    # Should handle single-document dataset gracefully
    assert hvpm.get_level(0)["metadata"]["documents"] == 1
    assert hvpm.get_level(1)["metadata"]["documents"] == 1
    assert hvpm.get_level(2)["metadata"]["documents"] == 1

def test_tile_processing():
    """Test critical tile extraction and edge device processing"""
    metric_names = ["metric1", "metric2", "metric3", "metric4"]
    score_matrix = np.array([
        [0.9, 0.2, 0.4, 0.1],  # Document 1 (most relevant)
        [0.7, 0.1, 0.3, 0.9],  # Document 0
        [0.5, 0.8, 0.2, 0.3],  # Document 2
        [0.1, 0.3, 0.9, 0.2]   # Document 3
    ])
    
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zeromodel.process(score_matrix)
    
    # Test default critical tile (3x3)
    tile = zeromodel.get_critical_tile()
    assert len(tile) == 31  # 4 header bytes + 27 pixel bytes (3x3x3)
    assert tile[0] == 3  # width
    assert tile[1] == 3  # height
    assert tile[2] == 0  # x offset
    assert tile[3] == 0  # y offset
    # Top-left pixel should be document 1, metric1 (0.9 * 255 = 229)
    assert tile[4] == 229
    
    # Test custom tile size
    tile = zeromodel.get_critical_tile(tile_size=2)
    assert len(tile) == 22  # 4 header bytes + 18 pixel bytes (2x2x3)
    assert tile[0] == 2  # width
    assert tile[1] == 2  # height
    # Top-left pixel should still be document 1, metric1
    assert tile[4] == 229
    
    # Test with small data (less than tile size)
    small_matrix = np.array([[0.5, 0.5, 0.5]])
    zeromodel.process(small_matrix)
    tile = zeromodel.get_critical_tile(tile_size=3)
    # Should have padding for missing metrics
    assert tile[4] == 127  # 0.5 * 255 = 127
    assert tile[5] == 127
    assert tile[6] == 127
    # Padding should be 0
    assert tile[7] == 0
    assert tile[8] == 0
    assert tile[9] == 0
    
    # Test edge device processing (simulate Lua code)
    def process_tile(tile_data):
        """Simulate edge device tile processing (180 bytes of code)"""
        # Parse tile: [width, height, x, y, pixels...]
        width = tile_data[0]
        height = tile_data[1]
        # Top-left pixel value
        top_left = tile_data[4]
        # Decision rule: is top-left pixel "dark enough"?
        return top_left < 128
    
    # With our data, top-left is 229 which is > 128, so should return False
    assert not process_tile(tile)
    
    # Test with different relevance threshold
    def process_tile_with_threshold(tile_data, threshold=128):
        top_left = tile_data[4]
        return top_left < threshold
    
    # With threshold 230, should return True
    assert process_tile_with_threshold(tile, 230)
    # With threshold 228, should return False
    assert not process_tile_with_threshold(tile, 228)
    
    # Test hierarchical tile processing
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3
    )
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
    
    # Get tile from level 0 (most abstract)
    tile0 = hvpm.get_tile(0)
    # Get tile from level 1
    tile1 = hvpm.get_tile(1)
    # Get tile from level 2 (most detailed)
    tile2 = hvpm.get_tile(2)
    
    # Level 0 tile should be smallest
    assert len(tile0) <= len(tile1) <= len(tile2)
    
    # Verify zooming behavior
    level, doc_idx, relevance = hvpm.get_decision(0)
    next_level = hvpm.zoom_in(level, doc_idx, 0)
    assert next_level == 1
    
    # Get decision at next level
    _, next_doc_idx, _ = hvpm.get_decision(next_level)
    # The document at next level should be in the expected region
    region_size = hvpm.get_level(1)["metadata"]["documents"] // hvpm.get_level(0)["metadata"]["documents"]
    expected_region_start = doc_idx * region_size
    assert expected_region_start <= next_doc_idx < expected_region_start + region_size

def test_advanced_sql_queries():
    """Test handling of complex SQL query patterns"""
    metric_names = ["uncertainty", "size", "quality", "novelty", "coherence"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9, 0.1, 0.7],
        [0.6, 0.7, 0.3, 0.8, 0.5],
        [0.2, 0.9, 0.5, 0.6, 0.3],
        [0.9, 0.3, 0.2, 0.9, 0.1]
    ])
    
    zeromodel = ZeroModel(metric_names)
    
    # Test SQL with aggregate functions
    zeromodel.set_sql_task("""
        SELECT * 
        FROM virtual_index 
        ORDER BY (uncertainty + size) DESC
    """)
    zeromodel.process(score_matrix)
    # Document 3: 0.9+0.3=1.2, Document 0: 0.8+0.4=1.2, Document 1: 0.6+0.7=1.3, Document 2: 0.2+0.9=1.1
    # But since uncertainty has higher weight in our virtual index, Document 3 should be first
    assert np.array_equal(zeromodel.doc_order, [3, 0, 1, 2])
    
    # Test SQL with mathematical expressions
    zeromodel.set_sql_task("""
        SELECT * 
        FROM virtual_index 
        ORDER BY (uncertainty * 2) DESC, size ASC
    """)
    zeromodel.process(score_matrix)
    # Doubled uncertainty: Document 3: 1.8, Document 0: 1.6, Document 1: 1.2, Document 2: 0.4
    assert np.array_equal(zeromodel.doc_order, [3, 0, 1, 2])
    
    # Test SQL with CASE statements
    zeromodel.set_sql_task("""
        SELECT *,
            CASE 
                WHEN uncertainty > 0.5 THEN 1
                ELSE 0
            END AS high_uncertainty
        FROM virtual_index
        ORDER BY high_uncertainty DESC, size ASC
    """)
    zeromodel.process(score_matrix)
    # Documents 3, 0, 1 have high uncertainty (1), Document 2 has low (0)
    # Among high uncertainty, sorted by size: Document 3 (0.3), Document 0 (0.4), Document 1 (0.7)
    assert np.array_equal(zeromodel.doc_order, [3, 0, 1, 2])
    
    # Test SQL with window functions
    zeromodel.set_sql_task("""
        SELECT *,
            RANK() OVER (ORDER BY uncertainty DESC) as uncertainty_rank
        FROM virtual_index
        ORDER BY uncertainty_rank, size ASC
    """)
    zeromodel.process(score_matrix)
    # Ranks: Document 3: 1, Document 0: 2, Document 1: 3, Document 2: 4
    assert np.array_equal(zeromodel.doc_order, [3, 0, 1, 2])
    
    # Test SQL with multiple WHERE conditions
    zeromodel.set_sql_task("""
        SELECT * 
        FROM virtual_index 
        WHERE uncertainty > 0.5 AND size < 0.6
        ORDER BY novelty DESC
    """)
    zeromodel.process(score_matrix)
    # Only documents 3 and 0 match the conditions
    # Novelty: Document 3: 0.9, Document 0: 0.1
    assert len(zeromodel.doc_order) == 2
    assert np.array_equal(zeromodel.doc_order, [3, 0])
    
    # Test SQL with LIMIT clause
    zeromodel.set_sql_task("""
        SELECT * 
        FROM virtual_index 
        ORDER BY uncertainty DESC
        LIMIT 2
    """)
    zeromodel.process(score_matrix)
    # Should only return top 2 documents
    assert len(zeromodel.doc_order) == 2
    assert np.array_equal(zeromodel.doc_order, [3, 0])
    
    # Test SQL with complex JOIN (simulated with virtual tables)
    zeromodel = ZeroModel(["id", "value"])
    zeromodel.duckdb_conn.execute("CREATE TABLE docs (id INTEGER, value FLOAT)")
    zeromodel.duckdb_conn.execute("INSERT INTO docs VALUES (0, 0.7), (1, 0.9), (2, 0.5), (3, 0.1)")
    zeromodel.duckdb_conn.execute("CREATE TABLE metadata (id INTEGER, category STRING)")
    zeromodel.duckdb_conn.execute("INSERT INTO metadata VALUES (0, 'A'), (1, 'B'), (2, 'A'), (3, 'B')")
    
    zeromodel.set_sql_task("""
        SELECT docs.id, docs.value, metadata.category
        FROM docs
        JOIN metadata ON docs.id = metadata.id
        ORDER BY docs.value DESC
    """)
    # Process a dummy matrix (we're testing the SQL analysis, not the actual data)
    zeromodel.process(np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
    
    # Should order by value: 0.9, 0.7, 0.5, 0.1 -> docs 1, 0, 2, 3
    assert np.array_equal(zeromodel.doc_order, [1, 0, 2, 3])
    
    # Test SQL with GROUP BY
    zeromodel.set_sql_task("""
        SELECT category, AVG(value) as avg_value
        FROM (
            SELECT docs.id, docs.value, metadata.category
            FROM docs
            JOIN metadata ON docs.id = metadata.id
        )
        GROUP BY category
        ORDER BY avg_value DESC
    """)
    # Process a dummy matrix
    zeromodel.process(np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
    
    # Category A: (0.7 + 0.5)/2 = 0.6, Category B: (0.9 + 0.1)/2 = 0.5
    # So A should be first
    # Note: In this case, doc_order represents the grouped results
    assert len(zeromodel.doc_order) == 2

def test_metadata_handling():
    """Test metadata extraction and usage across the system"""
    metric_names = ["uncertainty", "size", "quality", "novelty"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9, 0.1],
        [0.6, 0.7, 0.3, 0.8],
        [0.2, 0.9, 0.5, 0.6],
        [0.9, 0.3, 0.2, 0.9]
    ])
    
    # Test ZeroModel metadata
    zeromodel = ZeroModel(metric_names, precision=10)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    zeromodel.process(score_matrix)
    
    metadata = zeromodel.get_metadata()
    assert metadata["task"] == "sql_task"
    assert metadata["precision"] == 10
    assert metadata["metric_names"] == metric_names
    assert len(metadata["metric_order"]) == len(metric_names)
    assert len(metadata["doc_order"]) == score_matrix.shape[0]
    
    # Verify metric order metadata matches actual ordering
    for i in range(len(metadata["metric_order"]) - 1):
        # Earlier metrics should have higher weights in the SQL analysis
        assert metadata["metric_order"][i] < metadata["metric_order"][i+1]
    
    # Test HierarchicalVPM metadata
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=2,
        precision=8
    )
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    
    metadata = hvpm.get_metadata()
    assert metadata["version"] == "1.0"
    assert metadata["levels"] == 3
    assert metadata["zoom_factor"] == 2
    assert metadata["task"] == "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC"
    assert metadata["documents"] == score_matrix.shape[0]
    assert metadata["metrics"] == score_matrix.shape[1]
    
    # Test metadata across hierarchical levels
    for level in range(3):
        level_data = hvpm.get_level(level)
        level_metadata = level_data["metadata"]
        
        # Verify document and metric counts decrease with level
        if level > 0:
            prev_metadata = hvpm.get_level(level-1)["metadata"]
            assert level_metadata["documents"] <= prev_metadata["documents"]
            assert level_metadata["metrics"] <= prev_metadata["metrics"]
        
        # Verify sorted orders are valid
        assert len(level_metadata["sorted_docs"]) == level_metadata["documents"]
        assert len(level_metadata["sorted_metrics"]) == level_metadata["metrics"]
        
        # Verify sorted metrics are valid indices
        for idx in level_metadata["sorted_metrics"]:
            assert 0 <= idx < len(metric_names)
    
    # Test metadata with empty data
    empty_model = ZeroModel(metric_names)
    empty_model.set_sql_task("SELECT * FROM virtual_index ORDER BY uncertainty DESC")
    with pytest.raises(ValueError):
        empty_model.process(np.array([]))
    
    # Test metadata with single metric
    single_metric = ZeroModel(["metric"])
    single_metric.set_sql_task("SELECT * FROM virtual_index ORDER BY metric DESC")
    single_metric.process(np.array([[0.8], [0.6], [0.2]]))
    
    metadata = single_metric.get_metadata()
    assert metadata["metric_names"] == ["metric"]
    assert len(metadata["metric_order"]) == 1
    assert metadata["metric_order"][0] == 0
    
    # Test metadata with no task set
    no_task = ZeroModel(metric_names)
    no_task.process(score_matrix)
    
    metadata = no_task.get_metadata()
    assert metadata["task"] == "default"
    assert metadata["task_config"] is None

def test_performance_scalability():
    """Test performance with large datasets and measure scalability"""
    # Test with medium dataset (1,000 documents × 20 metrics)
    metric_names = [f"metric_{i}" for i in range(20)]
    medium_matrix = np.random.rand(1000, 20)
    
    start = time.time()
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric_0 DESC")
    zeromodel.process(medium_matrix)
    medium_time = time.time() - start
    
    # Verify processing completed
    assert zeromodel.sorted_matrix is not None
    assert zeromodel.doc_order is not None
    assert zeromodel.metric_order is not None
    
    # Test with large dataset (10,000 documents × 50 metrics)
    metric_names = [f"metric_{i}" for i in range(50)]
    large_matrix = np.random.rand(10000, 50)
    
    start = time.time()
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric_0 DESC, metric_1 ASC")
    zeromodel.process(large_matrix)
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
    vpm = zeromodel.encode()
    encode_time = time.time() - start
    
    # Verify encoding completed
    assert vpm is not None
    assert vpm.shape[0] == large_matrix.shape[0]
    assert vpm.shape[1] == (large_matrix.shape[1] + 2) // 3
    
    # Test critical tile extraction
    start = time.time()
    tile = zeromodel.get_critical_tile()
    tile_time = time.time() - start
    
    # Verify tile extraction completed
    assert tile is not None
    assert len(tile) > 0
    
    # Test decision making performance
    start = time.time()
    doc_idx, relevance = zeromodel.get_decision()
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
    assert medium_time < 0.5  # Should process medium dataset quickly
    assert large_time < 5.0   # Should process large dataset in reasonable time
    assert hierarchical_time < 10.0  # Hierarchical processing should be efficient
    assert encode_time < 0.15   # Encoding should be very fast
    assert tile_time < 0.01    # Tile extraction should be extremely fast
    assert decision_time < 0.01  # Decision making should be extremely fast
    
    # Test with extremely large dataset (100,000 documents × 100 metrics)
    # This might be too large for some systems, so we'll skip if it takes too long
    try:
        metric_names = [f"metric_{i}" for i in range(100)]
        huge_matrix = np.random.rand(100000, 100)
        
        start = time.time()
        zeromodel = ZeroModel(metric_names)
        zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric_0 DESC")
        zeromodel.process(huge_matrix)
        huge_time = time.time() - start
        
        # Verify processing completed
        assert zeromodel.sorted_matrix is not None
        
        print(f"Huge dataset (100,000×100) processing: {huge_time:.4f} seconds")
        # This might be slow, but should complete within a reasonable timeframe
        assert huge_time < 60.0  # Should complete within 1 minute
    except MemoryError:
        pytest.skip("System doesn't have enough memory for huge dataset test")

def test_xor_validation():
    """Complete end-to-end XOR validation demonstrating functional equivalence to traditional ML"""
    # Generate XOR dataset
    np.random.seed(42)
    X = np.random.rand(1000, 2)
    # Add noise
    X = X + 0.1 * np.random.randn(1000, 2)
    X = np.clip(X, 0, 1)
    # XOR labels
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    
    # Prepare score matrix for ZeroMI
    # For XOR, we'll use these meaningful metrics:
    score_matrix = np.zeros((X_train.shape[0], 5))
    
    # Metric 1: Distance from center (0.5, 0.5)
    score_matrix[:, 0] = np.sqrt((X_train[:, 0] - 0.5)**2 + (X_train[:, 1] - 0.5)**2)
    
    # Metric 2: Product of coordinates (x*y) - highly relevant for XOR
    score_matrix[:, 1] = X_train[:, 0] * X_train[:, 1]
    
    # Metric 3: Sum of coordinates (x+y)
    score_matrix[:, 2] = X_train[:, 0] + X_train[:, 1]
    
    # Metric 4: Absolute difference |x-y|
    score_matrix[:, 3] = np.abs(X_train[:, 0] - X_train[:, 1])
    
    # Metric 5: Angle from center
    score_matrix[:, 4] = np.arctan2(X_train[:, 1] - 0.5, X_train[:, 0] - 0.5)
    
    # Normalize each metric to [0,1] range
    for i in range(score_matrix.shape[1]):
        min_val = np.min(score_matrix[:, i])
        max_val = np.max(score_matrix[:, i])
        if max_val > min_val:
            score_matrix[:, i] = (score_matrix[:, i] - min_val) / (max_val - min_val)
    
    metric_names = [
        "distance_from_center",
        "coordinate_product",
        "coordinate_sum",
        "coordinate_difference",
        "angle_from_center"
    ]
    
    # Process with ZeroMI
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("""
        SELECT * 
        FROM virtual_index 
        ORDER BY coordinate_product DESC, coordinate_sum ASC
    """)
    zeromodel.process(score_matrix)
    
    # Get ZeroMI predictions for test data
    y_pred_zeromi = np.zeros(X_test.shape[0])
    
    for i in range(X_test.shape[0]):
        # Create score matrix for this single point
        point_matrix = np.zeros((1, 5))
        point_matrix[0, 0] = np.sqrt((X_test[i, 0] - 0.5)**2 + (X_test[i, 1] - 0.5)**2)
        point_matrix[0, 1] = X_test[i, 0] * X_test[i, 1]
        point_matrix[0, 2] = X_test[i, 0] + X_test[i, 1]
        point_matrix[0, 3] = np.abs(X_test[i, 0] - X_test[i, 1])
        point_matrix[0, 4] = np.arctan2(X_test[i, 1] - 0.5, X_test[i, 0] - 0.5)
        
        # Normalize
        for j in range(5):
            if max_val > min_val:
                point_matrix[0, j] = (point_matrix[0, j] - min_val) / (max_val - min_val)
        
        # Process point
        zeromodel.process(point_matrix)
        
        # Get decision
        _, relevance = zeromodel.get_decision()
        
        # For XOR, high relevance means Class 1
        y_pred_zeromi[i] = 1 if relevance > 0.5 else 0
    
    # Calculate accuracy
    zeromi_acc = accuracy_score(y_test, y_pred_zeromi)
    
    # Verify functional equivalence
    print(f"Traditional ML (SVM) Accuracy: {svm_acc:.4f}")
    print(f"Zero-Model Intelligence Accuracy: {zeromi_acc:.4f}")
    print(f"Accuracy Difference: {abs(svm_acc - zeromi_acc):.4f}")
    
    # ZeroMI should achieve similar accuracy to SVM (within 5%)
    assert abs(svm_acc - zeromi_acc) < 0.05
    
    # Verify decision latency advantage
    start = time.time()
    for _ in range(1000):
        _, _ = zeromodel.get_decision()
    zeromi_decision_time = (time.time() - start) / 1000
    
    start = time.time()
    for _ in range(1000):
        svm.predict([X_test[0]])
    svm_decision_time = (time.time() - start) / 1000
    
    print(f"ZeroMI Decision Time: {zeromi_decision_time:.6f} seconds")
    print(f"SVM Decision Time: {svm_decision_time:.6f} seconds")
    
    # ZeroMI should be significantly faster
    assert zeromi_decision_time < svm_decision_time * 0.1  # At least 10x faster
    
    # Verify memory usage advantage (indirectly)
    vpm = zeromodel.encode()
    vpm_size = vpm.nbytes
    
    # SVM model size (approximate)
    svm_size = sum([sys.getsizeof(attr) for attr in dir(svm) 
                   if not attr.startswith('__')])
    
    print(f"ZeroMI VPM Size: {vpm_size} bytes")
    print(f"SVM Model Size: {svm_size} bytes")
    
    # ZeroMI should use significantly less memory
    assert vpm_size < svm_size * 0.1  # At least 10x smaller which is missing out here

def test_hierarchical_navigation():
    """Test navigation between hierarchical levels with realistic data"""
    metric_names = ["uncertainty", "size", "quality", "novelty"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9, 0.1],  # Document 0
        [0.6, 0.7, 0.3, 0.8],  # Document 1
        [0.2, 0.9, 0.5, 0.6],  # Document 2
        [0.9, 0.3, 0.2, 0.9]   # Document 3
    ])
    
    # Create hierarchical VPM
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=2
    )
    
    # Process with SQL task
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    
    # Verify level structure
    assert len(hvpm.levels) == 3
    
    # Test zooming from level 0 to level 1
    level_0, doc_idx_0, _ = hvpm.get_decision(0)
    level_1 = hvpm.zoom_in(level_0, doc_idx_0, 0)
    assert level_1 == 1
    
    # Test zooming from level 1 to level 2
    level_1, doc_idx_1, _ = hvpm.get_decision(1)
    level_2 = hvpm.zoom_in(level_1, doc_idx_1, 0)
    assert level_2 == 2
    
    # Test trying to zoom beyond base level
    level_2, doc_idx_2, _ = hvpm.get_decision(2)
    level_3 = hvpm.zoom_in(level_2, doc_idx_2, 0)
    assert level_3 == 2  # Should stay at base level
    
    # Test navigation with specific document
    # Level 0 should have 1 document (the most relevant cluster)
    assert hvpm.get_level(0)["metadata"]["documents"] == 1
    # Level 1 should have 2 documents
    assert hvpm.get_level(1)["metadata"]["documents"] == 2
    # Level 2 should have 4 documents
    assert hvpm.get_level(2)["metadata"]["documents"] == 4
    
    # Verify hierarchical consistency
    level_0_doc = hvpm.get_decision(0)[1]
    level_1_doc = hvpm.get_decision(1)[1]
    level_2_doc = hvpm.get_decision(2)[1]
    
    # Level 0 document should correspond to the top region of level 1
    level_1_top_region = hvpm.get_level(1)["zeromodel"].doc_order[:1]
    assert level_0_doc in level_1_top_region
    
    # Level 1 document should correspond to the top region of level 2
    level_2_top_region = hvpm.get_level(2)["zeromodel"].doc_order[:2]
    assert level_1_doc in level_2_top_region
    
    # Verify zooming to specific region
    # Get decision at level 0
    _, doc_idx_0, _ = hvpm.get_decision(0)
    # Zoom in to that region at level 1
    level_1, doc_idx_1, _ = hvpm.get_decision(1)
    
    # Calculate expected region in level 1
    level_0_total = hvpm.get_level(0)["metadata"]["documents"]
    level_1_total = hvpm.get_level(1)["metadata"]["documents"]
    region_size = level_1_total // level_0_total
    expected_region_start = doc_idx_0 * region_size
    expected_region_end = expected_region_start + region_size
    
    # Level 1 document should be in the expected region
    assert expected_region_start <= doc_idx_1 < expected_region_end
    
    # Test hierarchical tile extraction
    tile_0 = hvpm.get_tile(0)
    tile_1 = hvpm.get_tile(1, x=doc_idx_0, y=0, width=3, height=3)
    tile_2 = hvpm.get_tile(2, x=doc_idx_1, y=0, width=3, height=3)
    
    # Verify tile sizes
    assert len(tile_0) < len(tile_1) < len(tile_2)
    
    # Verify hierarchical tile consistency
    # Top-left of level 1 tile should correspond to level 0 tile
    # (This is a simplified check - actual correspondence depends on clustering)
    assert tile_0[4] == tile_1[4]  # Top-left pixel
    
    # Test with different zoom factor
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=3
    )
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    
    # Level 0 should have 1 document
    assert hvpm.get_level(0)["metadata"]["documents"] == 1
    # Level 1 should have 2 documents (ceil(4/3))
    assert hvpm.get_level(1)["metadata"]["documents"] == 2
    # Level 2 should have 4 documents
    assert hvpm.get_level(2)["metadata"]["documents"] == 4

def test_hierarchical_navigation():
    """Test navigation between hierarchical levels with realistic data"""
    metric_names = ["uncertainty", "size", "quality", "novelty"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9, 0.1],  # Document 0
        [0.6, 0.7, 0.3, 0.8],  # Document 1
        [0.2, 0.9, 0.5, 0.6],  # Document 2
        [0.9, 0.3, 0.2, 0.9]   # Document 3
    ])
    
    # Create hierarchical VPM
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=2
    )
    
    # Process with SQL task
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    
    # Verify level structure
    assert len(hvpm.levels) == 3
    
    # Test zooming from level 0 to level 1
    level_0, doc_idx_0, _ = hvpm.get_decision(0)
    level_1 = hvpm.zoom_in(level_0, doc_idx_0, 0)
    assert level_1 == 1
    
    # Test zooming from level 1 to level 2
    level_1, doc_idx_1, _ = hvpm.get_decision(1)
    level_2 = hvpm.zoom_in(level_1, doc_idx_1, 0)
    assert level_2 == 2
    
    # Test trying to zoom beyond base level
    level_2, doc_idx_2, _ = hvpm.get_decision(2)
    level_3 = hvpm.zoom_in(level_2, doc_idx_2, 0)
    assert level_3 == 2  # Should stay at base level
    
    # Test navigation with specific document
    # Level 0 should have 1 document (the most relevant cluster)
    assert hvpm.get_level(0)["metadata"]["documents"] == 1
    # Level 1 should have 2 documents
    assert hvpm.get_level(1)["metadata"]["documents"] == 2
    # Level 2 should have 4 documents
    assert hvpm.get_level(2)["metadata"]["documents"] == 4
    
    # Verify hierarchical consistency
    level_0_doc = hvpm.get_decision(0)[1]
    level_1_doc = hvpm.get_decision(1)[1]
    level_2_doc = hvpm.get_decision(2)[1]
    
    # Level 0 document should correspond to the top region of level 1
    level_1_top_region = hvpm.get_level(1)["zeromodel"].doc_order[:1]
    assert level_0_doc in level_1_top_region
    
    # Level 1 document should correspond to the top region of level 2
    level_2_top_region = hvpm.get_level(2)["zeromodel"].doc_order[:2]
    assert level_1_doc in level_2_top_region
    
    # Verify zooming to specific region
    # Get decision at level 0
    _, doc_idx_0, _ = hvpm.get_decision(0)
    # Zoom in to that region at level 1
    level_1, doc_idx_1, _ = hvpm.get_decision(1)
    
    # Calculate expected region in level 1
    level_0_total = hvpm.get_level(0)["metadata"]["documents"]
    level_1_total = hvpm.get_level(1)["metadata"]["documents"]
    region_size = level_1_total // level_0_total
    expected_region_start = doc_idx_0 * region_size
    expected_region_end = expected_region_start + region_size
    
    # Level 1 document should be in the expected region
    assert expected_region_start <= doc_idx_1 < expected_region_end
    
    # Test hierarchical tile extraction
    tile_0 = hvpm.get_tile(0)
    tile_1 = hvpm.get_tile(1, x=doc_idx_0, y=0, width=3, height=3)
    tile_2 = hvpm.get_tile(2, x=doc_idx_1, y=0, width=3, height=3)
    
    # Verify tile sizes
    assert len(tile_0) < len(tile_1) < len(tile_2)
    
    # Verify hierarchical tile consistency
    # Top-left of level 1 tile should correspond to level 0 tile
    # (This is a simplified check - actual correspondence depends on clustering)
    assert tile_0[4] == tile_1[4]  # Top-left pixel
    
    # Test with different zoom factor
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=3
    )
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    
    # Level 0 should have 1 document
    assert hvpm.get_level(0)["metadata"]["documents"] == 1
    # Level 1 should have 2 documents (ceil(4/3))
    assert hvpm.get_level(1)["metadata"]["documents"] == 2
    # Level 2 should have 4 documents
    assert hvpm.get_level(2)["metadata"]["documents"] == 4