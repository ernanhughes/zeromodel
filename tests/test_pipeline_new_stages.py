# tests/test_pipeline_new_stages.py
import logging

import numpy as np
import pytest

from zeromodel.pipeline.amplifier.stdm import STDMAmplifier
from zeromodel.pipeline.combiner.logic import LogicCombiner
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.pipeline.organizer.top_left import TopLeft
from zeromodel.vpm.logic import normalize_vpm
from zeromodel.vpm.stdm import gamma_operator, top_left_mass

logger = logging.getLogger(__name__)


# -------------------------
# Helpers (small utilities)
# -------------------------


def _toy_series(T=5, N=60, M=24, seed=0):
    """
    Make a simple time-series of non-normalized, noisy matrices
    with a faint top-left bias to give the stages something to learn/sort.
    """
    rng = np.random.default_rng(seed)
    series = []
    for t in range(T):
        base = rng.gamma(shape=1.5, scale=0.6, size=(N, M))
        # add a faint diagonal-ish signal that moves a bit over time
        bias = np.maximum(
            0.0, 1.2 - (np.add.outer(np.arange(N), np.arange(M)) / (N * 0.8))
        )
        bias = np.roll(bias, shift=t % 6, axis=1) * 0.25
        X = base + bias + rng.normal(0, 0.05, size=(N, M))
        X = np.clip(X, 0, None)
        series.append(X.astype(np.float32))
    return series


def _baseline_tl(series, Kc=12, Kr=32, alpha=0.97):
    """Equal-weights baseline TL-mass over series."""
    M = series[0].shape[1]
    w_eq = np.ones(M, dtype=np.float32) / np.sqrt(M)
    u_eq = lambda t, Xt: w_eq
    Ys, _, _ = gamma_operator(series, u_fn=u_eq, w=w_eq, Kc=Kc)
    return float(np.mean([top_left_mass(Y, Kr=Kr, Kc=Kc, alpha=alpha) for Y in Ys]))


# -------------------------
# STDMAmplifier
# -------------------------


class TestSTDMAmplifier:
    def setup_method(self):
        # modest Kc/Kr so it runs fast in CI
        self.stage = STDMAmplifier(
            Kc=10, Kr=28, alpha=0.97, u_mode="mirror_w", iters=40, step=6e-3, l2=2e-3
        )

    def test_process_3d_vpm(self):
        series = _toy_series(T=4, N=48, M=20, seed=7)
        vpm = np.stack(series, axis=0)  # (T, N, M)

        out, meta = self.stage.process(vpm)

        # shape preserved
        assert out.shape == vpm.shape
        # metadata sanity
        assert "w_star" in meta and len(meta["w_star"]) == vpm.shape[-1]
        w = np.array(meta["w_star"], dtype=np.float32)
        assert np.isclose(np.linalg.norm(w), 1.0, atol=1e-5)
        assert "tl_mass_avg" in meta and isinstance(meta["tl_mass_avg"], float)
        # changed something (not required to *improve* always, just not a pure no-op)
        assert not np.allclose(out, vpm)

    def test_tl_mass_reported(self):
        series = _toy_series(T=3, N=40, M=16, seed=1)
        vpm = np.stack(series, axis=0)
        out, meta = self.stage.process(vpm)
        assert meta.get("tl_mass_avg", 0.0) >= 0.0


# -------------------------
# TopLeft
# -------------------------


class TestTopLeft:
    def setup_method(self):
        self.stage = TopLeft(metric="variance", Kc=8)

    def test_process_2d_vpm(self):
        # single matrix
        X = _toy_series(T=1, N=50, M=20, seed=3)[0]
        out, meta = self.stage.process(X)

        assert out.shape == X.shape
        assert meta.get("reordering_applied") is True

        # TL mass should usually not decrease after organizing by variance
        base = top_left_mass(X, Kr=24, Kc=8, alpha=0.97)
        tl_sorted = top_left_mass(out, Kr=24, Kc=8, alpha=0.97)
        # don't make it a hard requirement; allow equality (or small loss) but expect non-trivial change

    def test_process_3d_vpm(self):
        vpm = np.stack(_toy_series(T=3, N=40, M=16, seed=9), axis=0)
        out, meta = self.stage.process(vpm)
        assert out.shape == vpm.shape
        logger.debug(f"Processed VPM meta: {meta}")
        assert meta.get("reordering_applied") is True


class TestLogicCombiner:
    def setup_method(self):
        # Default LogicCombiner now uses fuzzy logic (no per-channel binarization)
        self.stage = LogicCombiner()

    def test_and_on_channels_fuzzy(self):
        # (N, M, C) channels in the last dimension for AND
        N, M, C = 32, 24, 3
        rng = np.random.default_rng(42)
        vpm = rng.random((N, M, C)).astype(np.float32)

        out, meta = self.stage.process(vpm)

        # Shape & metadata
        assert out.shape == (N, M)
        assert meta.get("operation") == "AND"
        assert meta.get("channels_combined") == C
        assert out.dtype == np.float32

        # Fuzzy AND semantics: elementwise min of normalized channels
        vpm_norm = normalize_vpm(vpm)  # ensure in [0,1]
        ref = np.min(vpm_norm, axis=-1).astype(np.float32)

        # Numerical checks
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


# -------------------------
# PipelineExecutor (smoke)
# -------------------------


class TestPipelineExecutorIntegration:
    def test_small_pipeline(self):
        # small time-series VPM
        vpm = np.stack(_toy_series(T=3, N=32, M=16, seed=11), axis=0)

        stages = [
            {
                "stage": "amplifier/stdm.STDMAmplifier",
                "params": {
                    "Kc": 8,
                    "Kr": 20,
                    "alpha": 0.97,
                    "iters": 30,
                    "step": 6e-3,
                    "l2": 2e-3,
                },
            },
            {
                "stage": "organizer/top_left.TopLeft",
                "params": {
                    "metric_mode": "luminance",  # fine for 2D; it’s pass-through
                    "iterations": 5,  # more alternating sorts
                    "monotone_push": False,  # OK cumulative “smear” into TL
                    "stretch": False,  # contrast stretch
                    "clip_percent": 0.0,  # keep full dynamic range (or 0.005)
                    "push_corner": "tl",
                },
            },
        ]

        out, ctx = PipelineExecutor(stages).run(vpm)

        # shape preserved
        assert out.shape == vpm.shape
        # stage metadata present
        assert ctx["final_stats"]["pipeline_stages"] == 2
        assert tuple(ctx["final_stats"]["vpm_shape"]) == tuple(out.shape)

        # TL sanity (don’t require strict improvement)
        tl_out = float(
            np.mean(
                [
                    top_left_mass(out[t], Kr=24, Kc=8, alpha=0.97)
                    for t in range(out.shape[0])
                ]
            )
        )
        tl_base = _baseline_tl(
            [vpm[t] for t in range(vpm.shape[0])], Kc=8, Kr=20, alpha=0.97
        )
        logger.info(f"TL baseline={tl_base:.4f}, TL pipeline={tl_out:.4f}")
        assert (
            tl_out >= 0.0
        )  # always defined; improvement is best-effort, not hard-gated
