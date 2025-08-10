import numpy as np
from zeromodel.stdm import learn_w, gamma_operator, top_left_mass, curvature_over_time

def _generate_series(T=6, N=800, M=32, sparsity=6, noise=0.5, drift=0.08, seed=7):
    rng = np.random.default_rng(seed)
    w_true = np.zeros(M); active = rng.choice(M, sparsity, replace=False); w_true[active] = rng.uniform(0.8,1.2,sparsity)
    w_true /= (np.linalg.norm(w_true)+1e-12)
    base = rng.normal(size=(N,M)) * 0.4
    intensity = base @ w_true + rng.normal(scale=0.5, size=N)
    thresh = np.quantile(intensity, 0.7)
    y = (intensity >= thresh).astype(int)

    series = []
    for t in range(T):
        w_t = w_true + rng.normal(scale=drift, size=M)
        w_t = np.maximum(0, w_t); w_t /= (np.linalg.norm(w_t)+1e-12)
        signal = np.outer(y, w_t) * rng.uniform(0.9, 1.1)
        X_t = np.maximum(0.0, base + signal + rng.normal(scale=noise, size=(N,M)))
        series.append(X_t)
    return series, y

def _precision_at_k(row_order, y, K):
    return float(y[row_order[:K]].mean())

def test_temporal_vpm_calculus_improves_tl_and_precision():
    series, y = _generate_series()
    N, M = series[0].shape

    # baseline: equal weights
    w_eq = np.ones(M)/np.sqrt(M)
    u_eq = lambda t, Xt: w_eq
    Ys_eq, _, rows_eq = gamma_operator(series, u_fn=u_eq, w=w_eq, Kc=12)
    tl_eq = np.mean([top_left_mass(Y, Kr=48, Kc=12, alpha=0.97) for Y in Ys_eq])
    p_eq  = np.mean([_precision_at_k(r, y, K=48) for r in rows_eq])

    # learned weights
    w_star = learn_w(series, Kc=12, Kr=48, u_mode="mirror_w", alpha=0.97, l2=2e-3, iters=80, step=8e-3, seed=0)
    u_st = lambda t, Xt: w_star
    Ys_st, _, rows_st = gamma_operator(series, u_fn=u_st, w=w_star, Kc=12)
    tl_st = np.mean([top_left_mass(Y, Kr=48, Kc=12, alpha=0.97) for Y in Ys_st])
    p_st  = np.mean([_precision_at_k(r, y, K=48) for r in rows_st])

    assert tl_st > tl_eq * 1.05, "Top-left mass should improve by at least 5%."
    assert p_st  > p_eq  + 0.02, "Precision@K should improve by at least 2%."

def test_curvature_runs_without_error():
    series, _ = _generate_series()
    from zeromodel.stdm import gamma_operator
    w_eq = np.ones(series[0].shape[1])/np.sqrt(series[0].shape[1])
    Ys, _, _ = gamma_operator(series, u_fn=lambda t,Xt: w_eq, w=w_eq, Kc=8)
    curv = curvature_over_time(Ys)
    assert curv.shape[0] == len(series)
