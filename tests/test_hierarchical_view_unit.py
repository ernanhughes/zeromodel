# tests/test_hierarchical_view_unit.py
import numpy as np
import pytest

from zeromodel.pipeline.organizer.hierarchical_view import HierarchicalView


def test_hierarchical_view_basic():
    N, M = 64, 32
    rng = np.random.default_rng(123)
    X = np.clip(rng.normal(0.5, 0.2, size=(N, M)), 0, 1).astype(np.float32)

    stage = HierarchicalView(levels=3, row_frac=0.5, col_frac=0.5)
    result, ctx = stage.process(X, {})

    # The result should be a submatrix (smaller or equal, never bigger)
    assert result.shape[0] <= X.shape[0]
    assert result.shape[1] <= X.shape[1]
    assert result.shape[0] > 0 and result.shape[1] > 0

    # Context checks
    assert "hierview" in ctx
    hierview = ctx["hierview"]
    assert "levels" in hierview
    assert "preview" in hierview

    # At least 1 level must be recorded
    assert len(hierview["levels"]) >= 1


@pytest.mark.parametrize("levels", [1, 2, 4])
def test_hierarchical_view_levels(levels):
    N, M = 32, 16
    X = np.linspace(0, 1, N * M, dtype=np.float32).reshape(N, M)

    stage = HierarchicalView(levels=levels, row_frac=0.5, col_frac=0.5)
    _, ctx = stage.process(X, {})

    hierview = ctx["hierview"]
    assert "levels" in hierview
    n_levels = len(hierview["levels"])

    # Because of early stopping, we might not hit the full requested depth
    assert 1 <= n_levels <= levels

def test_hierarchical_view_determinism():
    """
    Check determinism: same input should yield identical output.
    """

    N, M = 32, 16
    rng = np.random.default_rng(999)
    X = np.clip(rng.normal(0.5, 0.1, size=(N, M)), 0, 1).astype(np.float32)

    stage = HierarchicalView(levels=2)

    out1, ctx1 = stage.process(X, {})
    out2, ctx2 = stage.process(X, {})

    np.testing.assert_allclose(out1, out2, atol=1e-8)
    assert ctx1.keys() == ctx2.keys()
