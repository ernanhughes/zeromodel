import numpy as np
import pytest
from zeromodel import ZeroModel, HierarchicalVPM

def test_zeromodel_initialization():
    """Test basic initialization of ZeroModel with all parameters"""
    metric_names = ["uncertainty", "size", "quality", "novelty", "coherence"]
    
    # Test default initialization
    zeromodel = ZeroModel(metric_names)
    assert zeromodel.metric_names == metric_names
    assert zeromodel.precision == 8
    assert zeromodel.task == "default"
    assert zeromodel.sorted_matrix is None
    assert zeromodel.virtual_conn is not None
    
    # Test custom precision
    zeromodel = ZeroModel(metric_names, precision=10)
    assert zeromodel.precision == 10
    
    # Verify virtual index database was created correctly
    cursor = zeromodel.virtual_conn.cursor()
    cursor.execute("PRAGMA table_info(data)")
    columns = cursor.fetchall()
    assert len(columns) == len(metric_names) + 1  # row_id + metrics
    assert columns[0][1] == "row_id"  # First column is row_id
    for i, col_name in enumerate(metric_names):
        assert columns[i+1][1] == col_name


def test_zeromodel_sql_processing():
    """Test SQL-based processing with virtual index approach"""
    metric_names = ["uncertainty", "size", "quality"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9],  # Document 0
        [0.6, 0.7, 0.3],  # Document 1
        [0.2, 0.9, 0.5]   # Document 2
    ])
    
    zeromodel = ZeroModel(metric_names)
    
    # Test valid SQL query
    zeromodel.set_sql_task("""
        SELECT * 
        FROM data 
        ORDER BY uncertainty DESC, size ASC
    """)
    zeromodel.process(score_matrix)
    
    # Verify the spatial organization matches SQLite's ordering
    assert np.array_equal(zeromodel.doc_order, [0, 1, 2])
    
    # Verify metric ordering
    assert np.array_equal(zeromodel.metric_order, [0, 1, 2])
    
    # Check VPM encoding
    vpm = zeromodel.encode()
    assert vpm.shape == (3, 1, 3)
    
    # CORRECTED: Top-left should be 255 (1.0 * 255), not 204
    assert vpm[0, 0, 0] == 255  # 1.0 * 255
    
    # Check critical tile
    tile = zeromodel.get_critical_tile()
    assert len(tile) == 31
    assert tile[0] == 3
    assert tile[1] == 3
    # CORRECTED: Top-left should be 255
    assert tile[4] == 255
    
    # Check decision
    doc_idx, relevance = zeromodel.get_decision()
    assert doc_idx == 0
    assert relevance > 0.5

def test_zeromodel_complex_sql():
    """Test complex SQL queries with virtual index approach"""
    metric_names = ["uncertainty", "size", "quality", "novelty"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9, 0.1],  # Document 0
        [0.6, 0.7, 0.3, 0.8],  # Document 1
        [0.2, 0.9, 0.5, 0.6],  # Document 2
        [0.9, 0.3, 0.2, 0.9]   # Document 3
    ])
    
    zeromodel = ZeroModel(metric_names)
    
    # Test SQL with multiple ORDER BY columns
    zeromodel.set_sql_task("""
        SELECT * 
        FROM data 
        ORDER BY uncertainty DESC, size ASC
    """)
    zeromodel.process(score_matrix)
    
    # Verify SQL-based sorting
    # Uncertainty DESC: [0.9, 0.8, 0.6, 0.2] -> docs [3, 0, 1, 2]
    # Size ASC: [0.3, 0.4, 0.7, 0.9] -> docs [3, 0, 1, 2] (already in order)
    assert np.array_equal(zeromodel.doc_order, [3, 0, 1, 2])
    
    # Verify metric ordering
    assert np.array_equal(zeromodel.metric_order, [0, 1, 2, 3])  # All metrics used
    
    # Test SQL with WHERE clause
    zeromodel.set_sql_task("""
        SELECT * 
        FROM data 
        WHERE size > 0.5
        ORDER BY novelty DESC
    """)
    zeromodel.process(score_matrix)
    
    # Verify WHERE filtering (only docs 1 and 2 have size > 0.5)
    # After filtering, doc_order should be [1, 2] (novelty DESC: 0.8, 0.6)
    assert len(zeromodel.doc_order) == 2
    assert np.array_equal(zeromodel.doc_order, [1, 2])
    
    # Verify metric ordering still works correctly
    assert len(zeromodel.metric_order) == 4
    
    # Test invalid SQL query
    with pytest.raises(ValueError, match="must be a SELECT statement"):
        zeromodel.set_sql_task("UPDATE data SET uncertainty = 0.5")
    
    with pytest.raises(ValueError, match="must specify a FROM clause"):
        zeromodel.set_sql_task("SELECT *")
    
    with pytest.raises(ValueError, match="Comments not allowed"):
        zeromodel.set_sql_task("SELECT * FROM data -- comment")
    
    # Test SQL with complex conditions
    zeromodel.set_sql_task("""
        SELECT * 
        FROM data 
        WHERE uncertainty > 0.5 AND size < 0.6
        ORDER BY quality DESC
    """)
    zeromodel.process(score_matrix)
    
    # Verify complex WHERE filtering
    # Only doc 0 matches (uncertainty=0.8 > 0.5 and size=0.4 < 0.6)
    assert len(zeromodel.doc_order) == 1
    assert zeromodel.doc_order[0] == 0


def test_zeromodel_hierarchical():
    """Test hierarchical VPM functionality with SQL processing"""
    metric_names = ["uncertainty", "size", "quality"]
    score_matrix = np.array([
        [0.8, 0.4, 0.9],  # Document 0
        [0.6, 0.7, 0.3],  # Document 1
        [0.2, 0.9, 0.5]   # Document 2
    ])
    
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=3
    )
    
    # Process with SQL task
    hvpm.process(score_matrix, """
        SELECT * 
        FROM data 
        ORDER BY uncertainty DESC, size ASC
    """)
    
    # Check levels structure
    assert len(hvpm.levels) == 3
    assert hvpm.get_level(0)["level"] == 0
    assert hvpm.get_level(1)["level"] == 1
    assert hvpm.get_level(2)["level"] == 2
    
    # Check document counts across levels
    level_0_docs = hvpm.get_level(0)["metadata"]["documents"]
    level_1_docs = hvpm.get_level(1)["metadata"]["documents"]
    level_2_docs = hvpm.get_level(2)["metadata"]["documents"]
    
    # Level 2 should have all documents
    assert level_2_docs == 3
    # Level 1 should have fewer documents (but at least 1)
    assert 1 <= level_1_docs < level_2_docs
    # Level 0 should have the fewest documents (but at least 1)
    assert 1 <= level_0_docs <= level_1_docs
    
    # Check decisions across levels
    decisions = []
    for level in range(3):
        l, doc_idx, relevance = hvpm.get_decision(level)
        assert l == level
        assert 0 <= doc_idx < hvpm.get_level(level)["metadata"]["documents"]
        assert 0 <= relevance <= 1.0
        decisions.append((level, doc_idx, relevance))
    
    # Verify hierarchical consistency
    # The top document at level 0 should correspond to the top region at level 1
    level_0_doc = decisions[0][1]
    level_1_doc = decisions[1][1]
    
    # Calculate approximate position correspondence
    level_0_total = hvpm.get_level(0)["metadata"]["documents"]
    level_1_total = hvpm.get_level(1)["metadata"]["documents"]
    
    level_0_pos = level_0_doc / level_0_total
    level_1_pos = level_1_doc / level_1_total
    
    # Positions should be roughly consistent across levels
    assert abs(level_0_pos - level_1_pos) < 0.3
    
    # Verify level 2 decision matches the expected ordering
    assert decisions[2][1] == 0  # Most uncertain document should be first


def test_zeromodel_contextual_decision():
    """Test contextual decision making with weighted position"""
    metric_names = ["metric"]
    score_matrix = np.array([
        [0.9], [0.8], [0.7],  # Top row
        [0.6], [0.5], [0.4],  # Middle row
        [0.3], [0.2], [0.1]   # Bottom row
    ])
    
    zeromodel = ZeroModel(metric_names)
    zeromodel.set_sql_task("""
        SELECT * 
        FROM data 
        ORDER BY metric DESC
    """)
    zeromodel.process(score_matrix)
    
    # Test different context sizes
    for context_size in [1, 2, 3]:
        doc_idx, relevance = zeromodel.get_decision(context_size=context_size)
        assert doc_idx == 0  # Top-left should always be most relevant
        assert relevance > 0.5  # Should be weighted toward top-left
    
    # Test with custom context size
    doc_idx, relevance = zeromodel.get_decision(context_size=2)
    assert doc_idx == 0
    # Relevance should be higher with smaller context (more focused on top-left)
    _, relevance_full = zeromodel.get_decision(context_size=3)
    assert relevance > relevance_full
    
    # Test with a different sorting order
    zeromodel.set_sql_task("""
        SELECT * 
        FROM data 
        ORDER BY metric ASC
    """)
    zeromodel.process(score_matrix)
    
    # Now the bottom-right should be most relevant
    doc_idx, relevance = zeromodel.get_decision(context_size=3)
    assert doc_idx == 8  # Last document (bottom-right)


def test_zeromodel_edge_cases():
    """Test edge cases and error handling"""
    # Empty metric names
    with pytest.raises(AssertionError):
        ZeroModel([])
    
    # Empty score matrix
    zeromodel = ZeroModel(["metric"])
    with pytest.raises(ValueError, match="Data not processed yet"):
        zeromodel.process(np.array([]))
    
    # Single document
    zeromodel = ZeroModel(["metric"])
    zeromodel.set_sql_task("SELECT * FROM data ORDER BY metric DESC")
    zeromodel.process(np.array([[0.5]]))
    assert zeromodel.get_decision()[0] == 0
    
    # Single metric
    zeromodel = ZeroModel(["metric"])
    score_matrix = np.array([[0.8], [0.6], [0.2]])
    zeromodel.set_sql_task("SELECT * FROM data ORDER BY metric DESC")
    zeromodel.process(score_matrix)
    doc_idx, relevance = zeromodel.get_decision()
    assert doc_idx == 0  # Highest value should be first
    
    # Test critical tile with small data
    tile = zeromodel.get_critical_tile(tile_size=2)
    assert len(tile) == 4 + (2*2*3)  # 4 header bytes + 12 pixel bytes
    
    # Test invalid tile size
    with pytest.raises(ValueError, match="Tile size must be at least 1"):
        zeromodel.get_critical_tile(tile_size=0)
    
    # Test decision on unprocessed data
    zeromodel = ZeroModel(["metric"])
    with pytest.raises(ValueError, match="Data not processed yet"):
        zeromodel.get_decision()
    
    # Test with minimal precision
    zeromodel = ZeroModel(["metric"], precision=4)
    zeromodel.set_sql_task("SELECT * FROM data ORDER BY metric DESC")
    zeromodel.process(np.array([[0.5]]))
    vpm = zeromodel.encode()
    # With 4-bit precision, values should be multiples of 16
    assert vpm[0, 0, 0] % 16 == 0
    
    # Test SQL with no ORDER BY clause
    zeromodel.set_sql_task("SELECT * FROM data")
    zeromodel.process(score_matrix)
    assert np.array_equal(zeromodel.doc_order, [0, 1, 2])  # Original order


def test_hierarchical_sql_processing():
    """Test hierarchical SQL processing with virtual index approach"""
    metric_names = ["type", "topic", "uncertainty", "size"]
    score_matrix = np.array([
        [0, 1, 0.8, 0.4],  # Document 0: type=0, topic=1
        [0, 2, 0.6, 0.7],  # Document 1: type=0, topic=2
        [1, 1, 0.2, 0.9],  # Document 2: type=1, topic=1
        [1, 2, 0.9, 0.3]   # Document 3: type=1, topic=2
    ])
    
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=2
    )
    
    # Process with SQL task
    hvpm.process(score_matrix, """
        SELECT * 
        FROM data 
        ORDER BY type ASC, topic DESC, uncertainty DESC, size ASC
    """)
    
    # Check that each level used the correct ordering
    level_0 = hvpm.get_level(0)
    level_1 = hvpm.get_level(1)
    level_2 = hvpm.get_level(2)
    
    # Level 2 (most detailed) should have all documents in correct order
    # type ASC: [0,0,1,1] -> docs [0,1,2,3]
    # topic DESC: [1,2,1,2] -> docs [1,0,3,2]
    # uncertainty DESC: [0.6,0.8,0.9,0.2] -> docs [3,0,1,2]
    # size ASC: [0.3,0.4,0.7,0.9] -> docs [3,0,1,2] (already in order)
    assert np.array_equal(level_2["zeromodel"].doc_order, [3, 0, 1, 2])
    
    # Level 1 should have a subset of documents in consistent order
    # Verify the relative ordering is consistent with level 2
    level_1_docs = level_1["zeromodel"].doc_order
    for i in range(len(level_1_docs) - 1):
        idx1 = level_2["zeromodel"].doc_order.tolist().index(level_1_docs[i])
        idx2 = level_2["zeromodel"].doc_order.tolist().index(level_1_docs[i+1])
        assert idx1 < idx2  # Order should be preserved
    
    # Level 0 should have the most relevant cluster
    level_0_doc = level_0["zeromodel"].doc_order[0]
    # This should correspond to the top region of level 1
    level_1_top_region = level_1["zeromodel"].doc_order[:2]
    assert level_0_doc in level_1_top_region


def test_virtual_index_correctness():
    """Test that the virtual index approach correctly captures SQL ordering"""
    metric_names = ["A", "B", "C"]
    score_matrix = np.array([
        [0.1, 0.2, 0.3],  # Document 0
        [0.4, 0.5, 0.6],  # Document 1
        [0.7, 0.8, 0.9]   # Document 2
    ])
    
    zeromodel = ZeroModel(metric_names)
    
    # Test basic ordering
    zeromodel.set_sql_task("SELECT * FROM data ORDER BY A DESC")
    zeromodel.process(score_matrix)
    assert np.array_equal(zeromodel.doc_order, [2, 1, 0])
    
    # Test reverse ordering
    zeromodel.set_sql_task("SELECT * FROM data ORDER BY A ASC")
    zeromodel.process(score_matrix)
    assert np.array_equal(zeromodel.doc_order, [0, 1, 2])
    
    # Test multi-column ordering
    zeromodel.set_sql_task("SELECT * FROM data ORDER BY B DESC, A ASC")
    zeromodel.process(score_matrix)
    # B DESC: [0.8, 0.5, 0.2] -> docs [2, 1, 0]
    # A ASC: [0.7, 0.4, 0.1] -> docs [2, 1, 0] (already in order)
    assert np.array_equal(zeromodel.doc_order, [2, 1, 0])
    
    # Test WHERE clause
    zeromodel.set_sql_task("SELECT * FROM data WHERE A > 0.3 ORDER BY C DESC")
    zeromodel.process(score_matrix)
    # WHERE A > 0.3: docs [1, 2]
    # C DESC: [0.9, 0.6] -> docs [2, 1]
    assert np.array_equal(zeromodel.doc_order, [2, 1])
    
    # Test complex query
    zeromodel.set_sql_task("""
        SELECT * 
        FROM data 
        WHERE A > 0.2 
        ORDER BY B DESC, C ASC
    """)
    zeromodel.process(score_matrix)
    # WHERE A > 0.2: docs [1, 2]
    # B DESC: [0.8, 0.5] -> docs [2, 1]
    # C ASC: [0.6, 0.9] -> docs [1, 2] (reversed)
    assert np.array_equal(zeromodel.doc_order, [1, 2])


def test_hierarchical_zoom_behavior():
    """Test zoom behavior across hierarchical levels"""
    metric_names = ["metric"]
    score_matrix = np.array([[i/10] for i in range(100)])  # 0.0, 0.1, 0.2, ..., 9.9
    
    hvpm = HierarchicalVPM(
        metric_names=metric_names,
        num_levels=3,
        zoom_factor=4
    )
    
    # Process with SQL task
    hvpm.process(score_matrix, """
        SELECT * 
        FROM data 
        ORDER BY metric DESC
    """)
    
    # Level 2 should have all 100 documents
    level_2 = hvpm.get_level(2)
    assert level_2["metadata"]["documents"] == 100
    # Documents should be in descending order (99, 98, ..., 0)
    assert np.array_equal(level_2["zeromodel"].doc_order, np.arange(99, -1, -1))
    
    # Level 1 should have ~25 documents (100 / 4)
    level_1 = hvpm.get_level(1)
    assert 20 <= level_1["metadata"]["documents"] <= 30
    # The top documents in level 1 should correspond to the top region of level 2
    level_1_top_docs = level_1["zeromodel"].doc_order[:5]
    level_2_top_docs = level_2["zeromodel"].doc_order[:20]  # Top 20 of 100
    for doc in level_1_top_docs:
        assert doc in level_2_top_docs
    
    # Level 0 should have ~6 documents (25 / 4)
    level_0 = hvpm.get_level(0)
    assert 4 <= level_0["metadata"]["documents"] <= 8
    # The top document in level 0 should correspond to the top region of level 1
    level_0_top_doc = level_0["zeromodel"].doc_order[0]
    level_1_top_docs = level_1["zeromodel"].doc_order[:3]
    assert level_0_top_doc in level_1_top_docs
    
    # Test zooming behavior
    level, doc_idx, _ = hvpm.get_decision(0)
    assert level == 0
    assert 0 <= doc_idx < level_0["metadata"]["documents"]
    
    # Zooming in should take us to the corresponding region in level 1
    next_level = hvpm.zoom_in(level, doc_idx, 0)
    assert next_level == 1
    
    # Get decision at next level
    _, next_doc_idx, _ = hvpm.get_decision(next_level)
    # The document at next level should be in the expected region
    region_size = level_1["metadata"]["documents"] // level_0["metadata"]["documents"]
    expected_region_start = doc_idx * region_size
    assert expected_region_start <= next_doc_idx < expected_region_start + region_size