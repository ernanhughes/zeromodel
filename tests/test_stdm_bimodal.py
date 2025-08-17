# tests/test_stdm_bimodal.py
import numpy as np
import logging

from zeromodel.tools.stdm import (
    gamma_operator,
    learn_w,
    top_left_mass,
    curvature_over_time,  # not used here but handy to keep import consistent
)

logger = logging.getLogger(__name__)

def _make_bimodal_series(
    T=5, N=600, M=40, active=8, high_scale=2.3, low_scale=0.6,
    noise=0.35, drift=0.03, seed=123
):
    """
    Build an *unnormalized*, bimodal feature world:
    - A subset of `active` metrics are genuinely informative and scaled up.
    - The rest are distractors with lower mean/scale but still erratic.
    - Each time step, the 'true' weight drifts slightly (nonstationary).
    """
    rng = np.random.default_rng(seed)

    # choose active metrics
    act_idx = rng.choice(M, size=active, replace=False)
    w_true0 = np.zeros(M, dtype=np.float64)
    w_true0[act_idx] = rng.uniform(0.9, 1.4, size=active)
    w_true0 /= (np.linalg.norm(w_true0) + 1e-12)

    # base features: distractors have lower typical magnitude but are noisy
    # NOTE: *not* normalized – this is on purpose.
    series = []
    # create a persistent label signal tied to active features
    # (so that ranking precision is meaningful)
    base_signal = rng.normal(scale=0.7, size=N)
    y_latent = base_signal + rng.normal(scale=0.3, size=N)
    y = (y_latent > np.quantile(y_latent, 0.7)).astype(int)

    for t in range(T):
        # small drift in the true weight over time
        w_t = w_true0 + rng.normal(scale=drift, size=M)
        w_t = np.maximum(0, w_t)
        w_t /= (np.linalg.norm(w_t) + 1e-12)

        # build features: active columns get "high" mode, inactives "low" mode
        X_high = rng.normal(loc=1.2, scale=high_scale, size=(N, M))
        X_low  = rng.normal(loc=0.3, scale=low_scale,  size=(N, M))
        X_t = X_low
        X_t[:, act_idx] = X_high[:, act_idx]

        # inject label-correlated signal along w_t (to reward good ordering)
        signal = np.outer(y, w_t) * rng.uniform(0.9, 1.1)
        X_t = X_t + signal + rng.normal(scale=noise, size=(N, M))

        # clamp to nonnegative to mimic “activation” style matrices
        X_t = np.maximum(0.0, X_t)
        series.append(X_t)

    return series, y, act_idx


def _precision_at_k(row_order, y, K):
    top_k = row_order[:K]
    return float(y[top_k].mean())


def test_bimodal_unnormalized_data_improves_metrics():
    """
    On a clearly bimodal, unnormalized dataset, learned weights should
    discover the truly active metrics and improve both:
      - top-left mass (TL) at Kr×Kc,
      - Precision@K for the top rows.
    """
    series, y, act_idx = _make_bimodal_series(
        T=6, N=600, M=40, active=8,
        high_scale=2.6, low_scale=0.5,
        noise=0.28, drift=0.02, seed=123
    )
    N, M = series[0].shape
    Kc = 14
    Kr = 60
    alpha = 0.97
    K_prec = 60

    # ===== Baseline: equal weights =====
    w_eq = np.ones(M) / np.sqrt(M)
    u_eq = lambda t, Xt: w_eq
    Ys_eq, _, rows_eq = gamma_operator(series, u_fn=u_eq, w=w_eq, Kc=Kc)
    tl_eq = np.mean([top_left_mass(Y, Kr=Kr, Kc=Kc, alpha=alpha) for Y in Ys_eq])
    p_eq  = np.mean([_precision_at_k(r, y, K=K_prec) for r in rows_eq])
    logger.info(f"[Bimodal] Equal: TL={tl_eq:.4f}  P@{K_prec}={p_eq:.4f}")

    # ===== Learned weights =====
    w_star = learn_w(
        series=series,
        Kc=Kc,
        Kr=Kr,
        u_mode="mirror_w",   # mirrors learned metric importance
        alpha=alpha,
        l2=2e-3,
        iters=160,           # modest – SciPy path will be quick; fallback okay too
        step=6e-3,
        seed=0,
    )
    u_st = lambda t, Xt: w_star
    Ys_st, _, rows_st = gamma_operator(series, u_fn=u_st, w=w_star, Kc=Kc)
    tl_st = np.mean([top_left_mass(Y, Kr=Kr, Kc=Kc, alpha=alpha) for Y in Ys_st])
    p_st  = np.mean([_precision_at_k(r, y, K=K_prec) for r in rows_st])
    logger.info(f"[Bimodal] Learned: TL={tl_st:.4f}  P@{K_prec}={p_st:.4f}")

    tl_improve = (tl_st - tl_eq) / max(1e-12, tl_eq)
    p_improve  = p_st - p_eq
    logger.info(f"[Bimodal] ΔTL={tl_improve*100:.2f}%  ΔP@{K_prec}={p_improve*100:.2f}%")

    # Expect **clear** improvements on this construction
    assert tl_st > tl_eq * 1.05, f"TL should improve ≥5% (got {tl_st:.4f} vs {tl_eq:.4f})"
    assert p_st  > p_eq + 0.05,  f"Precision@{K_prec} should improve by ≥0.05 (got {p_st:.4f} vs {p_eq:.4f})"

    # sanity: learned weights emphasize a subset of metrics
    assert np.isclose(np.linalg.norm(w_star), 1.0, atol=1e-6)
    assert (w_star >= -1e-12).all()
