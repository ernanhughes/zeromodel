# tests/test_hunter.py
import numpy as np

from zeromodel import ZeroModel, HierarchicalVPM
from zeromodel.vpm.hunter import VPMHunter

def test_hunter_zeromodel_ndarray_path(monkeypatch):
    metrics = ["m1", "m2"]
    rng = np.random.default_rng(123)
    X = rng.random((40, len(metrics)), dtype=np.float32)
    zm = ZeroModel(metrics)
    zm.prepare(X, "SELECT * FROM virtual_index ORDER BY m1 DESC")

    # Force a stable decision
    monkeypatch.setattr(zm, "get_decision_by_metric", lambda metric_idx=0, context_size=3: (5, 0.42))

    hunter = VPMHunter(zm, tau=0.9, max_steps=3, aoi_size_sequence=(9, 5, 3))
    target, conf, audit = hunter.hunt()

    assert isinstance(target, int)
    assert 0 <= conf <= 1
    assert len(audit) == 3
    # ndarray-only shapes
    for step in audit:
        assert "tile_shape" in step
        assert isinstance(step["tile_shape"], tuple)
        assert step["tile_shape"][2] == 3

def test_hunter_stops_on_tau_first_step_zm(monkeypatch):
    zm = ZeroModel(["m"])
    rng = np.random.default_rng(321)
    X = rng.random((10, 1), dtype=np.float32)
    zm.prepare(X, "SELECT * FROM virtual_index ORDER BY m DESC")
    monkeypatch.setattr(zm, "get_decision_by_metric", lambda metric_idx=0, context_size=3: (0, 0.95))

    hunter = VPMHunter(zm, tau=0.9, max_steps=5)
    target, conf, audit = hunter.hunt()

    assert conf >= 0.9
    assert len(audit) == 1

def test_hunter_hvpm_ndarray_path(monkeypatch):
    hvpm = HierarchicalVPM(["m1", "m2"], num_levels=2)

    class _DummyZM:
        def get_decision_by_metric(self, metric_idx=0, context_size=3):
            return (7, 0.2)

    # Provide the level API expected by hunter
    monkeypatch.setattr(
        hvpm, "get_level",
        lambda level: {"vpm": np.zeros((8, 8, 3), dtype=np.uint16), "zeromodel": _DummyZM()}
    )
    # Return an ndarray tile
    monkeypatch.setattr(hvpm, "get_tile", lambda level, width=3, height=3: np.zeros((8, 8, 3), dtype=np.uint16))

    hunter = VPMHunter(hvpm, tau=0.8, max_steps=2)
    target, conf, audit = hunter.hunt()

    assert isinstance(target, tuple) and len(target) == 2
    assert len(audit) == 2
