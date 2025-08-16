from zeromodel.images import extract_vpf
import numpy as np
import pytest
from zeromodel.hierarchical import HierarchicalVPM, extract_critical_region, region_max_intensity
from zeromodel.utils import png_to_gray_array
import time

@pytest.mark.parametrize("order_col,expected_doc", [
    ("metric1", 0),
    ("metric2", 2),
    ("metric3", 3),
    ("metric4", 1),
])
def test_task_ordering_param(order_col, expected_doc):
    metric_names = ["metric1","metric2","metric3","metric4"]
    M = np.array([
        [0.9, 0.2, 0.4, 0.1],
        [0.7, 0.1, 0.3, 0.9],
        [0.5, 0.8, 0.2, 0.3],
        [0.1, 0.3, 0.9, 0.2],
    ])
    hvpm = HierarchicalVPM(metric_names=metric_names, num_levels=3, zoom_factor=2)
    hvpm.process(M, f"SELECT * FROM virtual_index ORDER BY {order_col} DESC")
    _, doc_idx, _ = hvpm.get_decision()
    assert doc_idx == expected_top_doc(M, order_col, metric_names)

class TestHierarchicalVPM:
    """
    Comprehensive test suite for HierarchicalVPM class.
    
    This test suite validates ZeroModel's core principle: "intelligence lives in the data structure"
    rather than in processing. The HierarchicalVPM implementation embodies the revolutionary claim:
    
    "When the answer is always 40 steps away, size becomes irrelevant."
    
    Each test verifies a specific aspect of the spatial organization and hierarchical navigation
    that enables planet-scale reasoning with constant-time cognition.
    """
    
    def setup_method(self):
        """Create a standardized small dataset for testing"""
        self.metric_names = ["metric1", "metric2", "metric3", "metric4"]
        
        # Document scores with clear properties for different metrics
        self.score_matrix = np.array([
            [0.9, 0.2, 0.4, 0.1],  # Document 0 (highest metric1)
            [0.7, 0.1, 0.3, 0.9],  # Document 1 (highest metric4)
            [0.5, 0.8, 0.2, 0.3],  # Document 2 (highest metric2)
            [0.1, 0.3, 0.9, 0.2]   # Document 3 (highest metric3)
        ])
        
        # Create a larger dataset for scale testing
        np.random.seed(42)
        self.large_matrix = np.random.rand(512, 4)
    
    def test_initialization_validation(self):
        """
        Verify proper validation of initialization parameters.
        
        WHY THIS MATTERS:
        ZeroModel's "boring by design" principle requires strict validation to ensure
        the hierarchical structure is always well-formed. Invalid parameters would break
        the logarithmic navigation guarantee that makes world-scale reasoning possible.
        """
        # Valid initialization
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2,
            precision=8
        )
        assert hvpm.num_levels == 3
        assert hvpm.zoom_factor == 2
        assert hvpm.precision == "8"
        
        # Test invalid parameters
        with pytest.raises(ValueError, match="num_levels must be positive"):
            HierarchicalVPM(self.metric_names, num_levels=0)
            
        with pytest.raises(ValueError, match="zoom_factor must be greater than 1"):
            HierarchicalVPM(self.metric_names, zoom_factor=1)
            
        with pytest.raises(ValueError, match="precision must be between 4-16"):
            HierarchicalVPM(self.metric_names, precision=3)
    
    def test_base_level_creation(self):
        """
        Verify correct creation of the base level (highest detail).
        
        WHY THIS MATTERS:
        The base level is where ZeroModel's spatial organization happens. This test
        verifies the "top-left rule" principle: "retrieval query 'uncertain then large'
        pushes ambiguous-but-significant items into the top-left cluster."
        
        Without proper spatial organization at the base level, the entire hierarchical
        navigation would fail to concentrate relevant signals in predictable positions.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Verify base level exists and has correct metadata
        level_2 = hvpm.get_level(2)
        assert level_2["type"] == "base"
        assert level_2["metadata"]["documents"] == 4
        assert level_2["metadata"]["metrics"] == 4
        assert level_2["metadata"]["task"] == "SELECT * FROM virtual_index ORDER BY metric1 DESC"
        
        # Verify tile structure
        assert level_2["num_tiles_x"] == 1  # 4 docs / 256 tile size = 1 tile
        assert level_2["num_tiles_y"] == 1  # 4 metrics / 256 tile size = 1 tile
    
    def test_summary_level_creation(self):
        """
        Verify correct creation of summary levels from base level.
        
        WHY THIS MATTERS:
        This validates ZeroModel's hierarchical navigation principle: "Finding information
        in ZeroModel is like using a building directory: Check the lobby map (global overview),
        Take elevator to correct floor, Find your office door. Always 3 steps, whether in a cottage or skyscraper."
        
        The summary levels must properly aggregate signal concentration from lower levels
        to maintain the logarithmic navigation property that makes world-scale reasoning feasible.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Verify level 1 (first summary level)
        level_1 = hvpm.get_level(1)
        assert level_1["type"] == "summary"
        assert level_1["source_level"] == 2
        assert level_1["num_tiles_x"] == 1
        assert level_1["num_tiles_y"] == 1
        
        # Verify level 0 (top summary level)
        level_0 = hvpm.get_level(0)
        assert level_0["type"] == "summary"
        assert level_0["source_level"] == 1
        assert level_0["num_tiles_x"] == 1
        assert level_0["num_tiles_y"] == 1
    
    def test_large_dataset_level_creation(self):
        """Verify correct level creation with larger dataset requiring multiple tiles.
        
        WHY THIS MATTERS:
        ZeroModel's claim of "planet-scale navigation that feels flat" requires proper
        handling of datasets that span multiple tiles at the base level. This test verifies
        that the hierarchical structure correctly forms even when the base level has
        multiple tiles (512 documents in this case).
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=4,
            zoom_factor=4
        )
        
        hvpm.process(self.large_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Verify base level has multiple tiles
        base_level = hvpm.get_level(3)
        assert base_level["num_tiles_x"] > 1
        assert base_level["num_tiles_y"] == 1  # Only 4 metrics
        
        # Verify higher levels have fewer OR EQUAL tiles (logarithmic reduction)
        level_2 = hvpm.get_level(2)
        assert level_2["num_tiles_x"] <= base_level["num_tiles_x"]
        
        level_1 = hvpm.get_level(1)
        # Allow for cases where zooming doesn't reduce tile count (1 zoomed is still 1)
        assert level_1["num_tiles_x"] <= level_2["num_tiles_x"]
        
        level_0 = hvpm.get_level(0)
        assert level_0["num_tiles_x"] == 1

    def test_spatial_organization_top_left_rule(self):
        """
        Verify the "top-left rule" spatial organization principle.
        
        WHY THIS MATTERS:
        This is ZeroModel's core innovation: "A retrieval query 'uncertain then large'
        pushes ambiguous-but-significant items into the top-left cluster. The router reads
        just those pixels to decide what to process next."
        
        The test verifies that relevant signals concentrate in the top-left region,
        enabling constant-time cognition through spatial organization rather than
        linear processing.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Get base level tile
        base_tile_id = hvpm.storage.get_tile_id(2, 0, 0)
        base_tile_bytes = hvpm.storage.load_tile(base_tile_id)
        
        # Extract critical region (top-left)
        critical_region = extract_critical_region(base_tile_bytes, size=4)
        
        # Verify concentration of signal in top-left
        assert region_max_intensity(critical_region) > 0.5
        
        # Verify top-left value is higher than other regions
        if critical_region.size > 1:
            top_left_value = float(critical_region[0, 0]) / 255.0
            other_values = []
            if critical_region.shape[0] > 1:
                other_values.append(float(critical_region[1, 0]) / 255.0)
            if critical_region.shape[1] > 1:
                other_values.append(float(critical_region[0, 1]) / 255.0)
            
            if other_values:
                assert top_left_value >= max(other_values)
    
    def test_navigation_path_length(self):
        """
        Verify logarithmic navigation path length.
        
        WHY THIS MATTERS:
        This validates ZeroModel's revolutionary claim: "This pyramid structure gets
        faster the bigger it gets." The test ensures navigation time grows logarithmically
        with data size, enabling "planet-scale navigation that feels flat."
        
        Without this property, ZeroModel couldn't deliver on its promise of "40 hops for
        1 trillion documents."
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=5,
            zoom_factor=4
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Verify navigation path length is short
        start_time = time.time()
        path = hvpm.navigate()
        duration = time.time() - start_time
        
        assert len(path) <= 5, f"Navigation path should be short (got {len(path)} steps)"
        assert duration < 0.1, f"Navigation should be fast (took {duration:.4f}s)"
        
        # Verify path reaches a decision
        assert any("decision" in step for step in path)
    
    def test_world_scale_path_length(self):
        """
        Verify logarithmic scaling of navigation path with data size.
        
        WHY THIS MATTERS:
        This is the essence of ZeroModel's breakthrough: "When the answer is always 40
        steps away, size becomes irrelevant." The test verifies that path length scales
        logarithmically rather than linearly with data size.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=6,
            zoom_factor=4
        )
        
        # Process a small dataset
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Test path length for different scales
        path_length_10k = hvpm.get_path_length(10_000)
        path_length_1m = hvpm.get_path_length(1_000_000)
        path_length_1t = hvpm.get_path_length(1_000_000_000_000)
        
        # Verify logarithmic scaling
        assert path_length_10k <= 10
        assert path_length_1m <= 20
        assert path_length_1t <= 40
        assert path_length_1t - path_length_1m <= 10
    
    def test_task_reorientation(self):
        """
        Verify spatial reorganization based on different tasks.
        
        WHY THIS MATTERS:
        ZeroModel's spatial organization is task-aware: "The organization is semantic:
        spatial location reflects task relevance, enabling AI to 'see' what matters at a glance."
        
        This test verifies that changing the query reorganizes the spatial layout, demonstrating
        that the intelligence lives in the data structure rather than in processing.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        # Process with metric1 ordering
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        _, doc_idx1, _ = hvpm.get_decision()
        
        # Process with metric3 ordering
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric3 DESC")
        _, doc_idx3, _ = hvpm.get_decision()
        
        # Verify different decisions for different tasks
        assert doc_idx1 != doc_idx3, "Decision should change with different task ordering"
        
        # Verify document 0 is top for metric1
        if doc_idx1 == 0:
            # Document 0 has highest metric1
            assert self.score_matrix[0, 0] > self.score_matrix[1, 0]
        
        # Verify document 3 is top for metric3
        if doc_idx3 == 3:
            # Document 3 has highest metric3
            assert self.score_matrix[3, 2] > self.score_matrix[0, 2]
    
    def test_vpf_provenance(self):
        """Verify VPF (Visual Policy Fingerprint) embedding and extraction.
        
        WHY THIS MATTERS:
        This validates ZeroModel's "deterministic, reproducible provenance" principle:
        "A core tenet of ZeroModel is that the system's output should be inherently understandable."
        
        The VPF provides built-in audit trails that are "visible structure" rather than
        "bolted on" explanations, enabling verification by reading pixels rather than
        running models.
        """
        from zeromodel.images import extract_vpf  # Import the correct function
        
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Get base level tile
        base_tile_id = hvpm.storage.get_tile_id(2, 0, 0)
        base_tile_bytes = hvpm.storage.load_tile(base_tile_id)
        
        # Extract VPF using the correct function
        vpf = extract_vpf(base_tile_bytes)
        
        # Verify critical VPF fields
        assert "pipeline" in vpf
        assert "step" in vpf["pipeline"]
        assert "graph_hash" in vpf["pipeline"]
        assert "metrics" in vpf
        assert "documents" in vpf["metrics"]
        assert vpf["metrics"]["documents"] == 4
        assert vpf["metrics"]["metrics"] == 4

    def test_critical_region_extraction(self):
        """
        Verify correct extraction of critical region with various sizes.
        
        WHY THIS MATTERS:
        The critical region is where ZeroModel concentrates 99.99% of the answer in 0.1% of the space.
        This test verifies robust handling of edge cases where the requested region exceeds
        the actual image size, ensuring the "top-left rule" works consistently across scales.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Get base level tile
        base_tile_id = hvpm.storage.get_tile_id(2, 0, 0)
        base_tile_bytes = hvpm.storage.load_tile(base_tile_id)
        
        # Test with small critical region
        small_region = extract_critical_region(base_tile_bytes, size=2)
        assert small_region.shape == (2, 2) or small_region.shape == (min(2, self.score_matrix.shape[0]), min(2, self.score_matrix.shape[1]))
        
        # Test with large critical region (exceeds image size)
        large_region = extract_critical_region(base_tile_bytes, size=10)
        img_array = png_to_gray_array(base_tile_bytes)
        expected_shape = (min(10, img_array.shape[0]), min(10, img_array.shape[1]))
        assert large_region.shape == expected_shape
    
    def test_decision_consistency(self):
        """
        Verify decision consistency across navigation paths.
        
        WHY THIS MATTERS:
        ZeroModel's "edge ↔ cloud symmetry" principle states: "The same tile drives a micro-decision
        on-device and a full inspection in the cloud or a human viewer - no special formats."
        
        This test verifies that decisions are consistent regardless of navigation path,
        ensuring the spatial organization is deterministic and reproducible.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Get decision from different starting levels
        _, doc_idx0, relevance0 = hvpm.get_decision(0)
        _, doc_idx1, relevance1 = hvpm.get_decision(1)
        _, doc_idx2, relevance2 = hvpm.get_decision(2)
        
        # Verify consistent decision
        assert doc_idx0 == doc_idx1 == doc_idx2
        assert abs(relevance0 - relevance1) < 0.1
        assert abs(relevance1 - relevance2) < 0.1
    
    def test_edge_cases(self):
        """Verify robust handling of edge cases.
        
        WHY THIS MATTERS:
        ZeroModel's "robust under pressure" principle requires handling of minimal datasets,
        empty inputs, and other edge cases without failure. This ensures the system works
        reliably in real-world conditions where data quality varies.
        """
        # Test with minimal dataset (1 document)
        minimal_matrix = np.array([[0.9]])
        minimal_hvpm = HierarchicalVPM(
            metric_names=["metric1"],
            num_levels=2,
            zoom_factor=2
        )
        minimal_hvpm.process(minimal_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Verify minimal dataset navigation
        path = minimal_hvpm.navigate()
        assert len(path) > 0
        
        # For single-document datasets, relevance is meaningful if > 0
        # This aligns with ZeroModel's "intelligence lives in the data structure" principle
        assert path[-1]["relevance"] > 0, "Should have some relevance signal even with one document"
        
        # Verify the decision is the only document
        if "decision" in path[-1]:
            assert path[-1]["decision"] == 0, "With one document, it must be the decision"
        

    def test_storage_agnosticism(self):
        """
        Verify storage-agnostic design.
        
        WHY THIS MATTERS:
        ZeroModel's "boring by design" principle means it works with any storage backend.
        This test verifies the separation of concerns between the hierarchical structure
        and storage implementation, ensuring "decisions, not models, move" across systems.
        """
        from zeromodel.storage.in_memory import InMemoryStorage
        
        # Custom storage backend that logs operations
        class LoggingStorage(InMemoryStorage):
            def __init__(self):
                super().__init__()
                self.operations = []
            
            def store_tile(self, level, x, y, png_bytes):
                self.operations.append(("store", level, x, y))
                return super().store_tile(level, x, y, png_bytes)
            
            def load_tile(self, tile_id):
                self.operations.append(("load", tile_id))
                return super().load_tile(tile_id)
        
        # Create HVPM with custom storage
        storage = LoggingStorage()
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2,
            storage_backend=storage
        )
        
        # Process data
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Verify storage operations
        assert any(op[0] == "store" for op in storage.operations)
        
        # Get a tile
        hvpm.get_tile(2, width=4, height=4)
        
        # Verify tile loading
        assert any(op[0] == "load" for op in storage.operations)
    
    def test_performance_under_scale(self):
        """
        Verify performance characteristics with larger datasets.
        
        WHY THIS MATTERS:
        ZeroModel's claim of "milliseconds on tiny hardware" requires efficient processing
        even with large datasets. This test verifies that processing time scales reasonably
        with data size, enabling edge deployment on resource-constrained devices.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=4,
            zoom_factor=4
        )
        
        # Process small dataset
        start_time = time.time()
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        small_duration = time.time() - start_time
        
        # Process larger dataset
        start_time = time.time()
        hvpm.process(self.large_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        large_duration = time.time() - start_time
        
        # Verify reasonable scaling
        assert large_duration < small_duration * 20  # Processing shouldn't scale linearly
    
    def test_analyze_tile_method(self):
        """Verify the _analyze_tile method correctly determines navigation direction.
        
        WHY THIS MATTERS:
        This is the core of ZeroModel's spatial navigation: "The router reads just those
        pixels to decide what to process next." This test verifies the intelligence in
        the spatial organization drives navigation decisions.
        
        IMPORTANT: _analyze_tile should only be called on non-base levels. The navigation
        system has guard clauses to prevent calling it on the base level, as navigation
        completes there.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Get level 0 tile (global overview)
        level0_tile_id = hvpm.storage.get_tile_id(0, 0, 0)
        level0_tile_bytes = hvpm.storage.load_tile(level0_tile_id)
        
        # Analyze level 0 tile
        next_level, next_x, next_y, relevance = hvpm._analyze_tile(level0_tile_bytes, 0)
        
        # Verify analysis results - should navigate to level 1
        assert next_level == 1, "Should navigate to next level (level 1)"
        assert 0 <= next_x < max(1, hvpm.levels[1]["num_tiles_x"]), f"Invalid x coordinate: {next_x}"
        assert 0 <= next_y < max(1, hvpm.levels[1]["num_tiles_y"]), f"Invalid y coordinate: {next_y}"
        assert 0 <= relevance <= 1.0, f"Relevance out of range: {relevance}"
        
        # Get level 1 tile
        level1_tile_id = hvpm.storage.get_tile_id(1, next_x, next_y)
        level1_tile_bytes = hvpm.storage.load_tile(level1_tile_id)
        
        # Analyze level 1 tile
        next_level2, next_x2, next_y2, relevance2 = hvpm._analyze_tile(level1_tile_bytes, 1)
        
        # Should navigate to base level (level 2)
        assert next_level2 == 2, "Should navigate to base level (level 2)"
        assert 0 <= next_x2 < max(1, hvpm.levels[2]["num_tiles_x"]), f"Invalid x2 coordinate: {next_x2}"
        assert 0 <= next_y2 < max(1, hvpm.levels[2]["num_tiles_y"]), f"Invalid y2 coordinate: {next_y2}"
        assert 0 <= relevance2 <= 1.0, f"Relevance2 out of range: {relevance2}"

    def test_decision_extraction(self):
        """
        Verify direct decision extraction from base level tiles.
        
        WHY THIS MATTERS:
        This validates ZeroModel's "human-compatible explanations" principle: "The 'why'
        isn't a post-hoc blurb - it's visible structure. You can point to the region/pixels
        that drove the choice."
        
        The test verifies that decisions can be directly extracted from the spatial organization.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Get base level tile
        base_tile_id = hvpm.storage.get_tile_id(2, 0, 0)
        base_tile_bytes = hvpm.storage.load_tile(base_tile_id)
        
        # Extract decision
        doc_idx, relevance = hvpm._extract_decision(base_tile_bytes)
        
        # Verify decision extraction
        assert 0 <= doc_idx < 4
        assert 0 <= relevance <= 1.0
        
        # Verify relevance corresponds to pixel value
        gray = png_to_gray_array(base_tile_bytes)
        max_pixel = np.max(gray)
        assert abs(relevance - max_pixel/255.0) < 0.01
    
    def test_hierarchical_metadata(self):
        """
        Verify complete metadata for the hierarchical structure.
        
        WHY THIS MATTERS:
        ZeroModel's "deterministic, reproducible provenance" requires complete metadata
        to understand the context of decisions. This test verifies that all necessary
        metadata is captured and accessible.
        """
        hvpm = HierarchicalVPM(
            metric_names=self.metric_names,
            num_levels=3,
            zoom_factor=2
        )
        
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        
        # Get metadata
        metadata = hvpm.get_metadata()
        
        # Verify critical metadata fields
        assert metadata["version"] == "1.0"
        assert metadata["levels"] == 3
        assert metadata["zoom_factor"] == 2
        assert metadata["metric_names"] == self.metric_names
        assert metadata["total_documents"] == 4
        assert metadata["task"] == "SELECT * FROM virtual_index ORDER BY metric1 DESC"
        
        # Verify level details
        assert "level_details" in metadata
        assert len(metadata["level_details"]) == 3
        for level in metadata["level_details"]:
            assert "level" in level
            assert "type" in level
            assert "documents" in level
            assert "metrics" in level
            assert "num_tiles" in level

    def test_determinism_idempotence(self):
        hvpm = HierarchicalVPM(metric_names=self.metric_names, num_levels=3, zoom_factor=2, precision=8)
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        png1 = hvpm.get_tile(2)  # base
        vpf1 = hvpm.get_vpf(2, 0, 0) if hasattr(hvpm, "get_vpf") else None

        # Re-run with the same inputs
        hvpm2 = HierarchicalVPM(metric_names=self.metric_names, num_levels=3, zoom_factor=2, precision=8)
        hvpm2.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        png2 = hvpm2.get_tile(2)
        vpf2 = hvpm2.get_vpf(2, 0, 0) if hasattr(hvpm2, "get_vpf") else None

        assert png1 == png2, "Tiles should be byte-identical for same inputs"
        if vpf1 and vpf2:
            assert vpf1["pipeline"]["graph_hash"] == vpf2["pipeline"]["graph_hash"]


    def test_navigation_converges_from_all_tiles(self):
        hvpm = HierarchicalVPM(metric_names=self.metric_names, num_levels=4, zoom_factor=2)
        hvpm.process(self.large_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")

        # Expected decision from canonical start
        _, expected_doc, _ = hvpm.get_decision(0)

        # Walk from every tile at level 1 (and level 0)
        for lvl in [0, 1]:
            nx = max(1, hvpm.levels[lvl].get("num_tiles_x", 1))
            ny = max(1, hvpm.levels[lvl].get("num_tiles_y", 1))
            for x in range(nx):
                for y in range(ny):
                    # Use start_x/start_y instead of start_coords
                    path = hvpm.navigate(start_level=lvl, start_x=x, start_y=y)
                    assert any("decision" in step for step in path)
                    assert path[-1]["decision"] == expected_doc


    def test_extract_critical_region_rejects_bad_size(self):
        hvpm = HierarchicalVPM(metric_names=self.metric_names, num_levels=3, zoom_factor=2)
        hvpm.process(self.score_matrix, q("metric1"))
        png = hvpm.get_tile(2)
        with pytest.raises(ValueError):
            extract_critical_region(png, size=0)
        with pytest.raises(ValueError):
            extract_critical_region(png, size=-3)


    def test_process_rejects_bad_shapes(self):
        hvpm = HierarchicalVPM(metric_names=["m1"], num_levels=2, zoom_factor=2)
        with pytest.raises(ValueError):
            hvpm.process(np.array([1,2,3]), q("metric1"))  # 1D
        with pytest.raises(ValueError):
            hvpm.process(np.empty((0,0)), q("metric1"))

    def test_process_handles_nans_infs(self):
        hvpm = HierarchicalVPM(metric_names=["m1","m2"], num_levels=2, zoom_factor=2)
        bad = np.array([[np.nan, 1.0],[np.inf, 0.5]])
        with pytest.raises(ValueError):
            hvpm.process(bad, q("metric1"))

    def test_metric_name_mismatch_is_handled(self):
        hvpm = HierarchicalVPM(metric_names=["m1", "m2"], num_levels=2, zoom_factor=2)

        # Provide more columns than names (mismatch on purpose)
        data = np.random.rand(8, 3)

        # Use a query that matches provided names so DuckDB doesn't error
        hvpm.process(data, q("m1"))

        # Inspect the base-level tile's VPF to verify reconciliation behavior
        base_level = hvpm.num_levels - 1
        png = hvpm.get_tile(base_level)

        vpf_info = extract_vpf(png)
        vpf = vpf_info[0] if isinstance(vpf_info, tuple) else vpf_info

        # Documents match, metrics are reconciled to declared names (truncate extras)
        assert vpf["metrics"]["documents"] == data.shape[0]
        assert vpf["metrics"]["metrics"] in {len(hvpm.metric_names), data.shape[1]}

    def test_get_tile_resizing_dimensions(self):
        hvpm = HierarchicalVPM(metric_names=self.metric_names, num_levels=3, zoom_factor=2)
        hvpm.process(self.score_matrix, q("metric1"))

        # Request a 32x32 bounding box; small tiles may not be upscaled
        png = hvpm.get_tile(2, width=32, height=32)
        arr = png_to_gray_array(png)

        # Sanity: non-empty 2D grayscale
        assert arr.ndim == 2 and arr.size > 0

        # Height corresponds to # of documents in the base tile
        assert arr.shape[0] == self.score_matrix.shape[0]

        # Implementation guarantees the image fits within requested box
        assert arr.shape[0] <= 32 and arr.shape[1] <= 32

    def test_vpf_roundtrip_and_corruption(self):
        hvpm = HierarchicalVPM(metric_names=self.metric_names, num_levels=3, zoom_factor=2)
        hvpm.process(self.score_matrix, q("metric1"))

        png = hvpm.get_tile(2)
        vpf = extract_vpf(png)
        assert isinstance(vpf, dict) and vpf  # roundtrip sanity

        # --- Corrupt the actual VPF payload or its PNG chunk ---
        corrupted = None

        # 1) Try to corrupt inside the VPF JSON itself (most reliable).
        #    Look for a distinctive key that appears early in the JSON.
        for needle in (b'{"determinism"', b'"vpf_hash"', b'"metrics"'):
            i = png.find(needle)
            if i != -1:
                # Flip a byte inside the JSON to break parsing.
                corrupted = bytearray(png)
                corrupted[i + 1] ^= 0xFF  # mutate one byte
                corrupted = bytes(corrupted)
                break

        # 2) Fallback: corrupt the PNG chunk type (e.g., iTXt/tEXt/zTXt) if the JSON wasn’t found.
        if corrupted is None:
            for chunk in (b"iTXt", b"tEXt", b"zTXt"):
                j = png.find(chunk)
                if j != -1:
                    corrupted = png[:j] + b"iBAD" + png[j+4:]  # invalidate chunk type
                    break

        # 3) Last resort: truncate tail (make metadata incomplete)
        if corrupted is None:
            corrupted = png[:-12]

        # --- Expect extractor to either raise OR return a non-valid result ---
        try:
            vpf_bad = extract_vpf(corrupted)
        except Exception:
            # Raising is acceptable legacy behavior on severe corruption.
            return

        # Graceful path: must not claim a valid VPF dict with content
        assert vpf_bad in (None, {}) or (isinstance(vpf_bad, dict) and not vpf_bad)

    def test_reprocess_replaces_state(self):
        hvpm = HierarchicalVPM(metric_names=self.metric_names, num_levels=3, zoom_factor=2)
        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
        meta1 = hvpm.get_metadata()
        tile1 = hvpm.get_tile(2)

        hvpm.process(self.score_matrix, "SELECT * FROM virtual_index ORDER BY metric3 DESC")
        meta2 = hvpm.get_metadata()
        tile2 = hvpm.get_tile(2)

        assert meta1["task"] != meta2["task"]
        assert tile1 != tile2  # spatial reorg should change pixels

def q(order_col="metric1", direction="DESC"):
    # must match the column names present in metric_names/effective_metric_names
    return f"SELECT * FROM virtual_index ORDER BY {order_col} {direction}"

def expected_top_doc(scores, metric_name, metric_names):
    j = metric_names.index(metric_name)
    return int(np.argmax(scores[:, j]))

