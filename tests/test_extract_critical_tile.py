import math
import numpy as np
import pytest

from zeromodel import ZeroModel
from zeromodel.vpm.image import VPMImageReader
from zeromodel.vpm.metadata import VPMMetadata

def _effective_doc_count(reader: VPMImageReader) -> int:
    D_eff = reader.D
    try:
        mb = reader.read_metadata_bytes()
        if mb:
            md = VPMMetadata.from_bytes(mb)
            if md.doc_count:
                D_eff = int(md.doc_count)
    except Exception:
        pass
    return D_eff

def test_extract_basic_top_left_bright(tmp_path):
    # Small dataset with a clear gradient on metric 0 ("uncertainty")
    docs = 40
    metrics = ["uncertainty", "size", "quality"]
    X = np.zeros((docs, len(metrics)), dtype=np.float32)
    # Make uncertainty highest for small doc indices (so ORDER BY DESC brings them to top-left)
    X[:, 0] = np.linspace(1.0, 0.0, docs, dtype=np.float32)  # descending
    X[:, 1] = np.random.rand(docs).astype(np.float32)
    X[:, 2] = np.random.rand(docs).astype(np.float32)

    zm = ZeroModel(metric_names=metrics)
    out = tmp_path / "vpm.png"
    zm.prepare(
        score_matrix=X,
        sql_query="SELECT * FROM virtual_index ORDER BY uncertainty DESC",
        nonlinearity_hint=None,
        vpm_output_path=str(out),
    )

    # Extract a 12x12 top-left tile for metric 0
    size = 12
    tile = zm.extract_critical_tile(metric_idx=0, size=size)
    assert tile.dtype == np.uint16
    H, W, C = tile.shape
    assert C == 3
    # Width may be clamped if logical doc_count < requested size
    reader = VPMImageReader(str(out))
    D_eff = _effective_doc_count(reader)
    assert H == min(size, len(metrics)) or H == size  # some readers fix H to 'size'
    assert W == min(size, D_eff)

    # Check “top-left brighter” pattern on R channel
    r = tile[..., 0].astype(np.float32) / 65535.0
    # Compare mean of top-left quadrant vs bottom-right
    q = max(2, W // 3)
    tl = r[:min(q, r.shape[0]), :min(q, r.shape[1])].mean()
    br = r[-q:, -q:].mean()
    assert tl > br, f"Expected top-left mean {tl:.4f} > bottom-right {br:.4f}"

def test_extract_respects_logical_width_padding(tmp_path):
    # Force docs < META_MIN_COLS (usually 12) to trigger right-padding in the image writer
    docs = 10
    metrics = ["uncertainty", "size"]
    X = np.random.rand(docs, len(metrics)).astype(np.float32)

    zm = ZeroModel(metric_names=metrics)
    out = tmp_path / "vpm_pad.png"
    zm.prepare(
        score_matrix=X,
        sql_query="SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC",
        vpm_output_path=str(out),
    )

    reader = VPMImageReader(str(out))
    D_eff = _effective_doc_count(reader)   # should be 10, not the padded physical width (e.g., 12)
    assert D_eff == docs

    # Ask for a larger size than D_eff; expect clamping
    tile = zm.extract_critical_tile(metric_idx=0, size=32)
    H, W, C = tile.shape
    assert W == D_eff

def test_extract_with_weights(tmp_path):
    docs = 60
    metrics = ["uncertainty", "size", "quality"]
    X = np.random.rand(docs, len(metrics)).astype(np.float32)

    # Make metric 0 dominate on first half, metric 2 dominate on second half
    X[:docs//2, 0] += 2.0   # boost uncertainty for first half
    X[docs//2:, 2] += 2.0   # boost quality for second half

    zm = ZeroModel(metric_names=metrics)
    out = tmp_path / "vpm_weights.png"
    zm.prepare(
        score_matrix=X,
        sql_query="SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC",
        vpm_output_path=str(out),
    )

    # weights favor uncertainty more than quality
    weights = {0: 0.7, 2: 0.3}
    order = zm.compile_view(weights=weights, top_k=10)
    assert len(order) > 0
    # The top document should come from the first half more often than not
    assert int(order[0]) < docs//2

    # Extract tile by weights; just sanity check shape/dtype
    tile = zm.extract_critical_tile(weights=weights, size=8)
    assert tile.dtype == np.uint16
    assert tile.shape[2] == 3

def test_extract_metric_vs_weights_consistency(tmp_path):
    docs = 48
    metrics = ["uncertainty", "size"]
    X = np.random.rand(docs, len(metrics)).astype(np.float32)
    X[:, 0] += np.linspace(1.0, 0.0, docs)  # bias toward early docs on metric 0

    zm = ZeroModel(metric_names=metrics)
    out = tmp_path / "vpm_consistency.png"
    zm.prepare(
        score_matrix=X,
        sql_query="SELECT * FROM virtual_index ORDER BY uncertainty DESC",
        vpm_output_path=str(out),
    )

    ord_metric = zm.compile_view(metric_idx=0, top_k=5)
    ord_weight = zm.compile_view(weights={0: 1.0}, top_k=5)
    # Orders should be close when weights are 100% metric 0
    assert np.allclose(ord_metric, ord_weight)
