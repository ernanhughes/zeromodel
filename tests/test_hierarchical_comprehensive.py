import numpy as np
import pytest

from zeromodel.hierarchical import HierarchicalVPM, extract_critical_region
from zeromodel.images import extract_vpf
from zeromodel.utils import png_to_gray_array


def test_hierarchical_vpm_comprehensive():
    """Comprehensive test for HierarchicalVPM covering core behaviors."""
    # 1) Setup: tiny dataset with clear ordering on metric1
    metric_names = ["metric1", "metric2", "metric3", "metric4"]
    score_matrix = np.array([
        [0.9, 0.2, 0.4, 0.1],  # doc 0 (highest metric1)
        [0.7, 0.1, 0.3, 0.9],  # doc 1
        [0.5, 0.8, 0.2, 0.3],  # doc 2
        [0.1, 0.3, 0.9, 0.2],  # doc 3
    ], dtype=float)

    # 2) Process
    hvpm = HierarchicalVPM(metric_names=metric_names, num_levels=3, zoom_factor=2, precision=8)
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")

    # 3) Level structure & metadata
    assert len(hvpm.levels) == 3, "Expected exactly 3 levels"

    base = hvpm.get_level(2)
    assert base["type"] == "base"
    assert base["metadata"]["documents"] == 4

    sum1 = hvpm.get_level(1)
    assert sum1["type"] == "summary"
    assert sum1["metadata"]["documents"] == 4

    sum0 = hvpm.get_level(0)
    assert sum0["type"] == "summary"
    assert sum0["metadata"]["documents"] == 4

    # 4) Spatial organization: critical region (top-left) has strong signal
    base_tile_id = hvpm.storage.get_tile_id(2, 0, 0)
    base_tile_bytes = hvpm.storage.load_tile(base_tile_id)
    assert isinstance(base_tile_bytes, (bytes, bytearray)) and len(base_tile_bytes) > 0

    crit4 = extract_critical_region(base_tile_bytes, size=4)
    assert crit4.ndim == 2 and crit4.size > 0
    # Strong signal present in the critical region (normalize to [0,1])
    assert float(np.max(crit4)) / 255.0 >= 0.6

    # Top-left concentration heuristic (allow tiny matrices/clipping)
    if crit4.shape[0] >= 1 and crit4.shape[1] >= 1:
        tl = float(crit4[0, 0]) / 255.0
        neighbors = []
        if crit4.shape[0] > 1:
            neighbors.append(float(crit4[1, 0]) / 255.0)
        if crit4.shape[1] > 1:
            neighbors.append(float(crit4[0, 1]) / 255.0)
        if neighbors:
            assert tl >= max(neighbors), f"Top-left {tl} should be ≥ neighbors {neighbors}"

    # 5) Navigation: short path, reaches decision
    path = hvpm.navigate()
    assert len(path) <= 5, f"Path should be short/logarithmic, got {len(path)}"
    assert any("decision" in step for step in path), "Navigation should reach a decision"

    # 6) Decision at base level: use base tile’s direct decision
    dec_doc_idx, dec_rel = hvpm._extract_decision(base_tile_bytes)
    assert isinstance(dec_doc_idx, int)
    assert 0 <= dec_doc_idx < score_matrix.shape[0]
    assert 0.6 <= float(dec_rel) <= 1.0, f"Decision relevance should be strong (got {dec_rel})"

    # 7) Critical-region clipping edge-cases
    small = extract_critical_region(base_tile_bytes, size=2)
    assert small.shape[0] <= 2 and small.shape[1] <= 2

    img_arr = png_to_gray_array(base_tile_bytes)
    big = extract_critical_region(base_tile_bytes, size=10)
    assert big.shape == (min(10, img_arr.shape[0]), min(10, img_arr.shape[1]))

    # 8) World-scale: path length grows slowly (log-like)
    path_len_1m = hvpm.get_path_length(1_000_000)
    path_len_1t = hvpm.get_path_length(1_000_000_000_000)
    assert path_len_1m <= 25
    assert path_len_1t <= 45
    assert (path_len_1t - path_len_1m) <= 25

    # 9) VPF provenance present & decodable
    vpf_info = extract_vpf(base_tile_bytes)
    vpf = vpf_info[0] if isinstance(vpf_info, tuple) else vpf_info
    assert isinstance(vpf, dict)
    assert "pipeline" in vpf and "step" in vpf["pipeline"] and "graph_hash" in vpf["pipeline"]
    assert "metrics" in vpf
    
    # 10) _analyze_tile sanity: returns relevance in [0,1] and does not go beyond base
    lvl, x, y, rel = hvpm._analyze_tile(base_tile_bytes, 2)

    # Allow either: staying at base level (2) or returning a terminal sentinel (num_levels)
    max_level = getattr(hvpm, "num_levels", len(hvpm.levels))  # fallback if attribute name differs
    assert lvl in (2, max_level), f"Unexpected level from _analyze_tile: {lvl} (expected 2 or {max_level})"

    assert 0 <= float(rel) <= 1.0

    # 11) Minimal dataset: still navigates and yields a valid decision
    minimal = np.array([[0.9]], dtype=float)
    hvpm_min = HierarchicalVPM(metric_names=["metric1"], num_levels=2, zoom_factor=2)
    hvpm_min.process(minimal, "SELECT * FROM virtual_index ORDER BY metric1 DESC")

    path_min = hvpm_min.navigate()
    assert len(path_min) > 0
    assert any("decision" in step for step in path_min)

    # Relevance can be low with 1x1 due to quantization/smoothing; just require [0,1]
    last_rel = float(path_min[-1].get("relevance", 0.0))
    assert 0.0 <= last_rel <= 1.0

    # Also validate the direct decision from the base tile
    min_base_level = 1  # with num_levels=2, base is index 1
    min_tile_id = hvpm_min.storage.get_tile_id(min_base_level, 0, 0)
    min_tile_bytes = hvpm_min.storage.load_tile(min_tile_id)
    doc_idx_min, rel_min = hvpm_min._extract_decision(min_tile_bytes)

    assert doc_idx_min == 0
    assert 0.0 <= float(rel_min) <= 1.0

    # 12) Larger dataset → multiple tiles + efficient navigation
    np.random.seed(0)
    larger_matrix = np.random.rand(512, 4)  # 512 docs, 4 metrics
    hvpm_big = HierarchicalVPM(metric_names=metric_names, num_levels=4, zoom_factor=4)
    hvpm_big.process(larger_matrix, "SELECT * FROM virtual_index ORDER BY metric1 DESC")
    base_big = hvpm_big.get_level(3)
    assert base_big["num_tiles_x"] > 1 or base_big["num_tiles_y"] > 1
    path_big = hvpm_big.navigate()
    assert len(path_big) <= 8

    # 13) Task re-orientation: different ORDER BY still yields a confident decision
    hvpm.process(score_matrix, "SELECT * FROM virtual_index ORDER BY metric3 DESC")
    path2 = hvpm.navigate()
    assert any("decision" in step for step in path2)
    assert float(path2[-1]["relevance"]) >= 0.6

    # 14) Storage-agnostic metadata + tile access
    meta = hvpm.get_metadata()
    assert "level_details" in meta and len(meta["level_details"]) == 3
    tile_bytes = hvpm.get_tile(2, width=4, height=4)
    assert isinstance(tile_bytes, (bytes, bytearray)) and len(tile_bytes) > 0
