import pytest
import logging
import os
from datetime import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

logger = logging.getLogger(__name__)

def _download_tsla_5m_60d():
    """Try to fetch ~60 days of 5m TSLA bars via yfinance. Returns df or None."""
    try:
        import yfinance as yf
        df = yf.download("TSLA", interval="5m", period="60d", progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns=str.lower)
        # ensure expected columns
        need = {"datetime","open","high","low","close","volume"}
        if not need.issubset(set(df.columns)):
            return None
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        logger.warning("yfinance fetch failed: %s", e)
        return None

def _market_sessions(start_date, end_date):
    return pd.date_range(start_date, end_date, freq="B")

def _synthesize_tsla_5m(start_date, end_date, seed=42):
    """
    Synthetic 5m bars for regular hours (09:30–16:00).
    Year-like behavior compressed into 60d window (up, then down), to force mixed signs.
    """
    rng = np.random.default_rng(seed)
    sessions = _market_sessions(start_date, end_date)
    rows = []
    price = 220.0
    for d in sessions:
        # 78 five-minute bars from 09:30 to 15:55
        idx = pd.date_range(pd.Timestamp.combine(d, time(9, 30)),
                            pd.Timestamp.combine(d, time(16, 0)),
                            freq="5min")[:-1]
        # drift & vol schedule across window
        t = (d - sessions[0]).days / max(1, (sessions[-1] - sessions[0]).days)
        drift = 0.00020 if t < 0.5 else -0.00015
        vol = 0.004 + 0.003 * (math.sin(2 * math.pi * t) * 0.5 + 0.5)

        prev = price * (1 + rng.normal(0, 0.003))
        for ts in idx:
            ret = drift + vol * rng.normal()
            px = max(5.0, prev * math.exp(ret))
            high = px * (1 + abs(rng.normal(0, 0.0015)))
            low  = px * (1 - abs(rng.normal(0, 0.0015)))
            op   = prev
            cl   = px
            volu = max(100, int(abs(rng.normal(0.0,1.0)) * 3000 + 1000))
            rows.append((ts, op, high, low, cl, volu))
            prev = px
        price = prev

    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
    return df

def _compute_daily_zero_model(df, bar_minutes=5, alpha=1.0, beta=0.5, gamma=0.5):
    """
    Per-day reset signals:
      R_t = (close_t / open_day) - 1
      G_t = EMA(sign(returns))   ~ directional consistency
      B_t = EMA(|returns|)       ~ volatility/entropy
      DZ  = alpha*R + beta*G - gamma*B
    Returns:
      curves_by_date: {date: DataFrame(datetime, close, DZ)}
      summary: DataFrame(date, dz_close, dz_max, dz_min, day_return)
    """
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    curves, rows = {}, []
    for d, g in df.groupby("date", sort=True):
        if len(g) < 10:
            continue
        g = g.sort_values("datetime").reset_index(drop=True)
        open_px = float(g.iloc[0]["open"])

        ret = g["close"].pct_change().fillna(0.0).to_numpy()
        R = (g["close"] / open_px) - 1.0

        k_fast = 2 / (max(2, int(20*5/bar_minutes)) + 1)  # ~20 min
        k_slow = 2 / (max(2, int(60*5/bar_minutes)) + 1)  # ~60 min

        sign_ret = np.sign(ret)
        G = np.zeros_like(sign_ret, dtype=float)
        for i, s in enumerate(sign_ret):
            G[i] = (1 - k_fast) * (G[i-1] if i else 0.0) + k_fast * s

        Abs = np.abs(ret)
        B = np.zeros_like(Abs, dtype=float)
        for i, a in enumerate(Abs):
            B[i] = (1 - k_slow) * (B[i-1] if i else 0.0) + k_slow * a

        DZ = alpha * R.to_numpy() + beta * G - gamma * B

        gg = g[["datetime","close"]].copy()
        gg["DZ"] = DZ
        curves[d] = gg

        rows.append({
            "date": pd.to_datetime(d),
            "dz_close": float(DZ[-1]),
            "dz_max": float(np.max(DZ)),
            "dz_min": float(np.min(DZ)),
            "day_return": float((g.iloc[-1]["close"]/open_px) - 1.0),
        })

    summary = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return curves, summary

@pytest.mark.slow
def test_finance_tesla_trading_demo_daily_zero_model(tmp_path):
    """
    Daily ZeroModel test (per-session reset) with two outputs:
      - dz_year_sawtooth.png : price (top) + daily DZ close values (bottom)
      - dz_intraday_examples.png : last 5 sessions' intraday DZ curves
    Uses TSLA 5m (last ~60d) if available; otherwise synthesizes a realistic dataset.
    """
    # 1) Data (prefer real; fallback to synthetic)
    df = _download_tsla_5m_60d()
    used_real = df is not None
    if df is None:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=60)
        df = _synthesize_tsla_5m(start, end)

    assert not df.empty
    assert {"datetime","open","high","low","close","volume"}.issubset(set(df.columns))

    # 2) Compute Daily ZeroModel with per-day resets
    curves, summary = _compute_daily_zero_model(df, alpha=1.0, beta=0.5, gamma=0.5)
    assert len(summary) >= 10  # at least two trading weeks

    # Sanity: across a 60d window we should have both positive and negative DZ closes
    pos_days = (summary["dz_close"] > 0).sum()
    neg_days = (summary["dz_close"] < 0).sum()
    assert pos_days > 0 and neg_days > 0

    # 3) Plot: Price + daily DZ sawtooth
    out1 = tmp_path / "dz_year_sawtooth.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios":[2,1]})
    ax1.plot(df["datetime"], df["close"], lw=1)
    ax1.set_title(f"TSLA Price (5m, ~60d){' [CSV]' if used_real else ' [synthetic]'}")
    ax1.set_ylabel("Price")

    ax2.plot(summary["date"], summary["dz_close"], lw=1, marker='o', ms=2)
    ax2.axhline(0, ls="--", lw=1, color='r')
    ax2.set_title("Daily ZeroModel Index (DZ) — Session Reset (Saw-tooth)")
    ax2.set_ylabel("DZ Close")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close(fig)

    # 4) Plot: last 5 intraday curves
    out2 = tmp_path / "dz_intraday_examples.png"
    last_days = summary["date"].tail(5).dt.date.tolist()
    if last_days:
        fig2, axes = plt.subplots(len(last_days), 1, figsize=(14, 10))
        if len(last_days) == 1:
            axes = [axes]
        for ax, d in zip(axes, last_days):
            gg = curves[d]
            ax.plot(gg["datetime"], gg["DZ"], lw=1.5)
            ax.axhline(0, ls="--", lw=1, color='r')
            ax.set_title(f"DZ Intraday — {d}")
            ax.set_ylabel("DZ")
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax.set_xlabel("Time")
        plt.tight_layout()
        plt.savefig(out2, dpi=150)
        plt.close(fig2)

    # 5) Assertions: files exist
    assert out1.exists()
    if last_days:
        assert out2.exists()

    logger.info("Saved plots:\n  %s\n  %s", out1, (out2 if last_days else "(no intraday plot)"))
