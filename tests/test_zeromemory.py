import numpy as np
import pytest

from zeromodel.memory import ZeroMemory

METRICS = ["loss", "val_loss", "accuracy", "lr", "grad_norm", "aux"]

def mk_steps(n, start=0):
    return list(range(start, start + n))

def log_series(zm: ZeroMemory, step, **values):
    zm.log(step, values)

def fill_linear(zm: ZeroMemory, name, start, end, steps):
    xs = np.linspace(start, end, steps)
    for i, v in enumerate(xs):
        zm.log(i, {name: float(v)})

def fill_pair(zm: ZeroMemory, steps, loss_fn, val_loss_fn, **extra):
    for i in range(steps):
        row = {
            "loss": float(loss_fn(i)),
            "val_loss": float(val_loss_fn(i)),
        }
        row.update({k: float(v(i)) for k, v in extra.items()})
        zm.log(i, row)

def test_init_validation():
    with pytest.raises(ValueError):
        ZeroMemory([], buffer_steps=16)

    with pytest.raises(ValueError):
        ZeroMemory(METRICS, buffer_steps=0)

    with pytest.raises(ValueError):
        ZeroMemory(METRICS, tile_size=0)

    # selection_k must be <= tile_size * 3
    with pytest.raises(ValueError):
        ZeroMemory(METRICS, tile_size=4, selection_k=13)  # 4*3 = 12 max

    zm = ZeroMemory(METRICS, tile_size=4, selection_k=12)
    assert zm.selection_k == 12

def test_logging_and_buffer():
    zm = ZeroMemory(METRICS, buffer_steps=8)
    assert zm.buffer_count == 0

    # log a couple rows
    zm.log(0, {"loss": 1.0, "val_loss": 1.2})
    zm.log(1, {"loss": 0.9, "val_loss": 1.1, "accuracy": 0.6})

    assert zm.buffer_count == 2
    # last row stored in ring buffer
    assert np.isfinite(zm.buffer_values).sum() >= 4  # at least the values we logged

def test_feature_ranking_stable():
    zm = ZeroMemory(METRICS, buffer_steps=64)
    # Make accuracy strongly increasing to rank high
    for i in range(50):
        zm.log(i, {"loss": 1.0, "val_loss": 1.0, "accuracy": i / 50.0, "lr": 1e-3})

    ranked = zm.get_feature_ranking(window_size=32, target_metric_name="loss")
    assert ranked.shape[0] == len(METRICS)
    # accuracy should be among top 3 (trend)
    top3 = set(ranked[:3].tolist())
    assert METRICS.index("accuracy") in top3

def test_snapshot_vpm_and_tile_shapes():
    zm = ZeroMemory(METRICS, buffer_steps=128, tile_size=5, selection_k=15)  # 5*3=15 channels
    # Fill enough steps so window >= tile_size
    for i in range(10):
        zm.log(i, {
            "loss": 1.0 - 0.05 * i,
            "val_loss": 1.0 + 0.02 * i,
            "accuracy": 0.5 + 0.04 * i,
            "lr": 1e-3,
            "grad_norm": 1.0,
            "aux": np.sin(i/3.0)
        })

    vpm = zm.snapshot_vpm(window_size=5, target_metric_name="loss")
    assert vpm.shape == (5, 5, 3)
    assert vpm.dtype == np.uint8

    tile = zm.snapshot_tile(tile_size=3, window_size=5)
    # header: 4 bytes + 3x3 pixels x 3 channels
    assert isinstance(tile, (bytes, bytearray))
    width, height, xoff, yoff = tile[0], tile[1], tile[2], tile[3]
    assert (width, height) == (3, 3)
    expected_len = 4 + width * height * 3
    assert len(tile) == expected_len

def test_alert_overfitting():
    zm = ZeroMemory(METRICS, buffer_steps=64)
    # train loss decreases, val loss increases
    fill_pair(
        zm, steps=40,
        loss_fn=lambda i: 1.0 - 0.02 * i,
        val_loss_fn=lambda i: 1.0 + 0.02 * i,
        accuracy=lambda i: 0.5 + 0.01 * i
    )
    alerts = zm.get_alerts(window_size=32)
    assert alerts["overfitting"] is True
    # sanity: not everything else at once
    assert alerts["underfitting"] is False

def test_alert_underfitting():
    zm = ZeroMemory(METRICS, buffer_steps=64)
    # high flat loss ~1.5
    fill_pair(
        zm, steps=40,
        loss_fn=lambda i: 1.5 + 0.01*np.sin(i),   # oscillate slightly, flat mean
        val_loss_fn=lambda i: 1.5 + 0.01*np.cos(i),
    )
    alerts = zm.get_alerts(window_size=32)
    assert alerts["underfitting"] is True

@pytest.mark.skip("Needs work")
def test_alert_saturation():
    zm = ZeroMemory(METRICS, buffer_steps=64)
    # many metrics constant (low variance)
    for i in range(30):
        zm.log(i, {"loss": 1.0, "val_loss": 1.0, "accuracy": 0.7, "lr": 1e-3, "grad_norm": 1.0, "aux": 0.0})
    alerts = zm.get_alerts(window_size=30)
    assert alerts["saturation"] is True

def test_alert_instability():
    zm = ZeroMemory(METRICS, buffer_steps=128)
    # loss with bursts: last quarter much noisier => short_std / long_std high
    vals = []
    for i in range(80):
        base = 1.0
        noise = (0.02 if i < 60 else 0.5) * np.sin(i * 0.5)
        vals.append(base + noise)
        zm.log(i, {"loss": vals[-1], "val_loss": vals[-1]})

    alerts = zm.get_alerts(window_size=80)
    assert alerts["instability"] is True

@pytest.mark.skip("Needs work")
def test_alert_drift():
    zm = ZeroMemory(METRICS, buffer_steps=128)
    # metric drifts upward in second half
    for i in range(60):
        val = 0.1 if i < 30 else 1.1
        zm.log(i, {"loss": 1.0, "val_loss": 1.0, "aux": val})
    alerts = zm.get_alerts(window_size=60)
    assert alerts["drift"] is True

def test_handles_nans_and_missing_metrics():
    zm = ZeroMemory(METRICS, buffer_steps=32)
    # mix finite and NaN
    for i in range(20):
        row = {"loss": 1.0 - 0.01*i}
        if i % 3 == 0:
            row["val_loss"] = np.nan
        zm.log(i, row)

    # should not crash; still produce VPM/tile
    vpm = zm.snapshot_vpm(window_size=10)
    assert vpm.shape == (zm.tile_size, zm.tile_size, 3)
    tile = zm.snapshot_tile(tile_size=3, window_size=10)
    assert len(tile) >= 4 + 3*3*3

def test_selection_k_caps_to_tile_channels():
    # selection_k cannot exceed tile_size*3
    zm = ZeroMemory(METRICS, tile_size=4, selection_k=12)  # ok
    for i in range(10):
        zm.log(i, {"loss": 1.0, "val_loss": 1.0})

    vpm = zm.snapshot_vpm(window_size=8)
    assert vpm.shape == (4, 4, 3)  # tile_size, tile_size, 3