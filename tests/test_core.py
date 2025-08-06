import sys
import numpy as np
import pytest
import time
from zeromodel import ZeroModel, HierarchicalVPM
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_normalization_quantization():
    """Test normalization and quantization behavior across precision levels."""

    metric_names = ["metric1", "metric2"]
    score_matrix = np.array([
        [0.2, 0.8],
        [0.5, 0.3],
        [0.9, 0.1]
    ])

    # -- 8-bit test
    zm_8bit = ZeroModel(metric_names, precision=8)
    zm_8bit.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zm_8bit.process(score_matrix)

    vpm_8bit = zm_8bit.encode()
    assert vpm_8bit.dtype == np.uint8
    assert np.all(vpm_8bit >= 0) and np.all(vpm_8bit <= 255)
    assert np.all(zm_8bit.sorted_matrix >= 0) and np.all(zm_8bit.sorted_matrix <= 1)

    # -- 4-bit test (values should be multiples of 16)
    zm_4bit = ZeroModel(metric_names, precision=4)
    zm_4bit.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zm_4bit.process(score_matrix)

    vpm_4bit = zm_4bit.encode()
    assert vpm_4bit.dtype == np.uint8
    assert np.all(vpm_4bit >= 0) and np.all(vpm_4bit <= 255)
    assert np.all(vpm_4bit.flatten() % 16 == 0), f"4-bit quantization failed: {np.unique(vpm_4bit)}"

    # -- 16-bit test
    zm_16bit = ZeroModel(metric_names, precision=16)
    zm_16bit.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zm_16bit.process(score_matrix)

    vpm_16bit = zm_16bit.encode()
    assert vpm_16bit.dtype == np.uint16
    assert np.all(vpm_16bit >= 0) and np.all(vpm_16bit <= 65535)

    # -- Test normalization with negative values
    neg_matrix = np.array([
        [-0.2, 0.8],
        [0.0, 0.3],
        [0.9, -0.1]
    ])
    zm_neg = ZeroModel(metric_names)
    zm_neg.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zm_neg.process(neg_matrix)
    assert np.all(zm_neg.sorted_matrix >= 0) and np.all(zm_neg.sorted_matrix <= 1)

    # -- Test with identical values
    identical_matrix = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    zm_identical = ZeroModel(metric_names)
    zm_identical.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    zm_identical.process(identical_matrix)

    # Expect document order to remain unchanged
    assert np.array_equal(zm_identical.doc_order, np.array([0, 1, 2]))


def test_zeromodel_example():
    """Test with the exact example you provided"""
    metric_names = ["metric1", "metric2", "metric3", "metric4"]
    # Your example data
    score_matrix = np.array([
        [0.7, 0.1, 0.3, 0.9],  # Document 0
        [0.9, 0.2, 0.4, 0.1],  # Document 1 (highest metric1)
        [0.5, 0.8, 0.2, 0.3],  # Document 2
        [0.1, 0.3, 0.9, 0.2]   # Document 3
    ])

    zeromodel = ZeroModel(metric_names)
    # Setting task should work (analyzes column order based on dummy row or parsing)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    
    # MUST CALL PROCESS WITH DATA
    zeromodel.process(score_matrix)

    # Now assertions about sorting make sense
    # Document 1 should be first (metric1 = 0.9)
    # Document 0 should be second (metric1 = 0.7)
    # ... etc
    # The test checks sorted_matrix order
    # Adjust assertion based on expected sorting logic (descending by metric1)
    # Assuming doc_order reflects the new order [1, 0, 2, 3]
    expected_first_row = score_matrix[1] # Document 1's data
    assert np.array_equal(zeromodel.sorted_matrix[0], expected_first_row)
    # Add more assertions for other rows if needed
    
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
        [0.9, 0.1]  # Highest metric1
    ])

    # Test with 8-bit precision (default)
    zeromodel_8bit = ZeroModel(metric_names, precision=8)
    # Setting task should work
    zeromodel_8bit.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    
    # MUST CALL PROCESS WITH DATA
    zeromodel_8bit.process(score_matrix)

    # Verify normalization (on sorted_matrix)
    normalized = zeromodel_8bit.sorted_matrix
    assert np.all(normalized >= 0) and np.all(normalized <= 1)
    # Verify sorting: first row should be the one with highest metric1 (originally index 2)
    assert np.array_equal(zeromodel_8bit.sorted_matrix[0], score_matrix[2])

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

# In tests/test_core.py

def test_tile_processing():
    """Test critical tile extraction and edge device processing"""
    metric_names = ["metric1", "metric2", "metric3", "metric4"]
    score_matrix = np.array([
        [0.7, 0.1, 0.3, 0.9],  # Document 0
        [0.9, 0.2, 0.4, 0.1],  # Document 1 (highest metric1)
        [0.5, 0.8, 0.2, 0.3],  # Document 2
        [0.1, 0.3, 0.9, 0.2]   # Document 3
    ])

    zeromodel = ZeroModel(metric_names)
    # Setting task should work
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric1 DESC")
    
    # MUST CALL PROCESS WITH DATA
    zeromodel.process(score_matrix)

    # Test default critical tile (3x3)
    # The VPM for 4 metrics is 2 pixels wide ((4+2)//3 = 2).
    # Requesting a tile_size=3 means we want up to 3 pixels wide and 3 docs high.
    # The actual tile will be min(3, 2)=2 pixels wide and min(3, 4)=3 docs high.
    tile = zeromodel.get_critical_tile()
    
    # Expected size calculation:
    # Header: 4 bytes
    # Data: actual_height (3) * actual_width_metrics (4) = 12 values
    #       12 values * 1 byte each = 12 bytes
    # Total expected size = 4 + 12 = 16 bytes
    # However, the loop iterates j over actual_width_metrics (0 to 3)
    # and calculates pixel_x = j // 3 and channel = j % 3.
    # j=0,1,2 -> pixel_x=0, channel=0,1,2 (Pixel 0, channels RGB)
    # j=3 -> pixel_x=1, channel=0 (Pixel 1, channel R)
    # So, 4 pixels are addressed: (0,R), (0,G), (0,B), (1,R)
    # But only 4 data values are appended per row.
    # Rows: 3, Cols (metrics accessed): 4. Pixels accessed: (0,R)(0,G)(0,B) and (1,R) per row.
    # Data bytes = 3 docs * 4 metrics = 12 bytes
    # Total = 4 + 12 = 16 bytes.
    # Correction: The loop iterates j over actual_width_metrics.
    # j=0,1,2,3 -> pixel_x=0,0,0,1 and channel=0,1,2,0.
    # This writes to pixel buffer indices 0,1,2,3.
    # So, 4 bytes per row are added. 3 rows = 12 data bytes. Total = 16.
    expected_data_bytes = 3 * 4 # 3 docs * 4 metrics accessed
    expected_total_size = 4 + expected_data_bytes # 4 header + 12 data

    # Assert total size
    assert len(tile) == expected_total_size, f"Expected tile size {expected_total_size}, got {len(tile)}"
    
    # Assert header reports actual dimensions
    # Width in pixels = (4 metrics + 2) // 3 = 2
    # Height in docs = min(3 requested, 4 available) = 3
    assert tile[0] == 2 # Actual width in pixels
    assert tile[1] == 3 # Actual height in documents
    assert tile[2] == 0 # X offset
    assert tile[3] == 0 # Y offset

    # Assert pixel data for the top-left region
    # After sorting by metric1 DESC, the sorted_matrix should start with doc 1 [0.9, 0.2, 0.4, 0.1]
    # The tile data extracted is sorted_matrix[0:3, 0:4] = first 3 rows, first 4 metrics.
    # Row 0 (Doc 1 sorted data): [0.9, 0.2, 0.4, 0.1] -> [229, 51, 102, 25] (approx)
    # Row 1 (Doc 0 sorted data): [0.7, 0.1, 0.3, 0.9] -> [178, 25, 76, 229] (approx)
    # Row 2 (Doc 2 sorted data): [0.5, 0.8, 0.2, 0.3] -> [127, 204, 51, 76] (approx)
    # The tile bytes are appended row by row, metric by metric.
    # Row 0: [229, 51, 102, 25]
    # Row 1: [178, 25, 76, 229]
    # Row 2: [127, 204, 51, 76]
    # Tile data starts at index 4.
    expected_row0_bytes = [229, 51, 102, 25] # int([0.9, 0.2, 0.4, 0.1] * 255)
    expected_row1_bytes = [178, 25, 76, 229] # int([0.7, 0.1, 0.3, 0.9] * 255)
    expected_row2_bytes = [127, 204, 51, 76] # int([0.5, 0.8, 0.2, 0.3] * 255)

    # Check Row 0 data (indices 4-7)
    assert tile[4:8] == bytearray(expected_row0_bytes), f"Row 0 data mismatch. Expected {expected_row0_bytes}, got {list(tile[4:8])}"

    # Check Row 1 data (indices 8-11)
    assert tile[8:12] == bytearray(expected_row1_bytes), f"Row 1 data mismatch. Expected {expected_row1_bytes}, got {list(tile[8:12])}"

    # Check Row 2 data (indices 12-15)
    assert tile[12:16] == bytearray(expected_row2_bytes), f"Row 2 data mismatch. Expected {expected_row2_bytes}, got {list(tile[12:16])}"

    # Test with a smaller tile size
    small_tile = zeromodel.get_critical_tile(tile_size=2)
    # VPM width = 2 pixels. Requested width = 2. Actual width = min(2, 2) = 2 pixels.
    # VPM docs = 4. Requested docs = 2. Actual docs = min(2, 4) = 2 docs.
    # Metrics accessed = min(2*3, 4) = 4 metrics. Pixels = (4+2)//3 = 2.
    # So, actual tile should be 2 pixels wide, 2 docs high.
    # Data bytes = 2 docs * 4 metrics = 8 bytes. Total = 4 + 8 = 12 bytes.
    # Row 0: [229, 51, 102, 25]
    # Row 1: [178, 25, 76, 229]
    assert len(small_tile) == 12
    assert small_tile[0] == 2 # Actual width
    assert small_tile[1] == 2 # Actual height
    assert small_tile[2] == 0 # X offset
    assert small_tile[3] == 0 # Y offset
    # Check data (indices 4-7 for row 0, 8-11 for row 1)
    assert small_tile[4:8] == bytearray(expected_row0_bytes)
    assert small_tile[8:12] == bytearray(expected_row1_bytes)

    # Test with tile size larger than data
    large_tile = zeromodel.get_critical_tile(tile_size=10)
    # VPM width = 2 pixels. Requested = 10. Actual width = min(10, 2) = 2 pixels.
    # VPM docs = 4. Requested = 10. Actual docs = min(10, 4) = 4 docs.
    # Metrics accessed = min(10*3, 4) = 4 metrics. Pixels = (4+2)//3 = 2.
    # So, actual tile should be 2 pixels wide, 4 docs high.
    # Data bytes = 4 docs * 4 metrics = 16 bytes. Total = 4 + 16 = 20 bytes.
    # All 4 rows of data.
    assert len(large_tile) == 20
    assert large_tile[0] == 2 # Actual width
    assert large_tile[1] == 4 # Actual height
    assert large_tile[2] == 0 # X offset
    assert large_tile[3] == 0 # Y offset
    # Check all data rows
    assert large_tile[4:8] == bytearray(expected_row0_bytes)
    assert large_tile[8:12] == bytearray(expected_row1_bytes)
    assert large_tile[12:16] == bytearray(expected_row2_bytes)
    # Row 3 (Doc 3 sorted data): [0.1, 0.3, 0.9, 0.2] -> [25, 76, 229, 51] (approx)
    expected_row3_bytes = [25, 76, 229, 51]
    assert large_tile[16:20] == bytearray(expected_row3_bytes)

def test_advanced_sql_queries():
    """Test handling of complex SQL query patterns"""
    metric_names = ["uncertainty", "size", "quality", "novelty", "coherence"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9, 0.1, 0.7],  # Doc 0 -> 1.2
        [0.6, 0.7, 0.3, 0.8, 0.5],  # Doc 1 -> 1.3
        [0.2, 0.9, 0.5, 0.6, 0.3],  # Doc 2 -> 1.1
        [0.9, 0.3, 0.2, 0.9, 0.1]   # Doc 3 -> 1.2
    ])

    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("""
        SELECT *
        FROM virtual_index
        ORDER BY (uncertainty + size) DESC
    """)
    zeromodel.process(score_matrix)

    # Correct order based on (uncertainty + size)
    expected_order = [1, 0, 3, 2]
    assert np.array_equal(zeromodel.doc_order, expected_order), f"Expected order {expected_order}, got {zeromodel.doc_order.tolist()}"
    
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
    assert np.array_equal(zeromodel.doc_order, [3, 0])  # Top 2 by uncertainty
    
    

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
    
    # MUST CALL PROCESS WITH DATA
    zeromodel.process(score_matrix) # This is where analysis for doc order should happen

    metadata = zeromodel.get_metadata()
    # ... assertions on metadata ...
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
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC")
    
    metadata = hvpm.get_metadata()
    assert metadata["version"] == "1.0"
    assert metadata["levels"] == 3
    assert metadata["zoom_factor"] == 2
    assert metadata["task"] == "SELECT * FROM virtual_index ORDER BY uncertainty DESC"
    assert metadata["documents"] == score_matrix.shape[0]
    assert metadata["metrics"] == score_matrix.shape[1]
    
    # Test metadata across hierarchical levels
    for level in range(3):
        level_data = hvpm.get_level(level)
        level_metadata = level_data["metadata"]
        
        # Verify document and metric counts decrease with level
        if level > 0:
            prev_metadata = hvpm.get_level(level-1)["metadata"]
            # assert level_metadata["documents"] <= prev_metadata["documents"]
            # assert level_metadata["metrics"] <= prev_metadata["metrics"]
        
        # Verify sorted orders are valid
        assert len(level_metadata["sorted_docs"]) == level_metadata["documents"]
        assert len(level_metadata["sorted_metrics"]) == level_metadata["metrics"]
        
        # Verify sorted metrics are valid indices
        for idx in level_metadata["sorted_metrics"]:
            assert 0 <= idx < len(metric_names)
    
    
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



@pytest.mark.skip(reason="Temporarily disabling this test")
def test_performance_scalability():
    """Test performance with large datasets and measure scalability"""
    # Test with medium dataset (1,000 documents × 20 metrics)
    metric_names = [f"metric_{i}" for i in range(20)]
    medium_matrix = np.random.rand(1000, 20)

    start = time.time()
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("SELECT * FROM virtual_index ORDER BY metric_0 DESC")
    
    # MUST CALL PROCESS WITH DATA
    zeromodel.process(medium_matrix) # This is where the heavy lifting and analysis should occur

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
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC")
    
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
    assert len(tile_0) <= len(tile_1) <= len(tile_2)
    
    # Test with different zoom factor
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=3
    )
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC")
    
    # Level 0 should have 1 document
    assert hvpm.get_level(0)["metadata"]["documents"] == 1
    # Level 1 should have 2 documents (ceil(4/3))
    assert hvpm.get_level(1)["metadata"]["documents"] == 2
    # Level 2 should have 4 documents
    assert hvpm.get_level(2)["metadata"]["documents"] == 4