# tests/test_hierarchical_workflow.py
from zeromodel import HierarchicalVPM
from tests.utils import generate_synthetic_data


# In tests/test_hierarchical_workflow.py
def test_hierarchical_vpm_workflow():
    """Test the hierarchical workflow demonstrated in the demo script"""
    # 1. Generate synthetic data
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)

    # 2. Create hierarchical VPM
    hvpm = HierarchicalVPM(metric_names=metric_names)

    # 3. Process with task - Ensure the task is valid for the metric_names
    # The error showed "ORDER BY uncertainty DESC, size ASC" but the generated names might be different
    # Use metric names that exist, or adjust the task.
    # Example fix: Use the first two generated metric names
    if len(metric_names) >= 2:
        task_query = f"SELECT * FROM virtual_index ORDER BY {metric_names[0]} DESC, {metric_names[1]} ASC"
    else:
        task_query = f"SELECT * FROM virtual_index ORDER BY {metric_names[0]} DESC"

    # Process - this internally calls ZeroModel.process
    hvpm.process(score_matrix, task_query) # <-- This line should work if the task is valid

    # 4. Test level access
    base_level = hvpm.get_level(2)  # Base level (most detailed)
    assert base_level is not None
    assert base_level["level"] == 2
    assert "vpm" in base_level

    # 5. Test tile extraction
    tile = hvpm.get_tile(2, 0, 0, 3, 3)  # Get tile from base level
    assert tile is not None
    assert len(tile) > 0

    # 6. Test decision making
    level, doc_idx, relevance = hvpm.get_decision()
    assert level == 2  # Should default to base level
    assert 0 <= doc_idx < 100
    assert 0 <= relevance <= 1.0

    # 7. Test zoom navigation
    next_level = hvpm.zoom_in(level, doc_idx, 0)
    assert next_level == 2  # Already at base level, should stay

    # Test metadata
    metadata = hvpm.get_metadata()
    assert metadata is not None
    assert metadata["levels"] == 3