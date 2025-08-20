import pytest
import logging
import os
logger = logging.getLogger(__name__)

@pytest.mark.slow
def test_finance_tesla_trading_demo_year(tmp_path):
    import yfinance as yf
    import matplotlib.pyplot as plt
    import numpy as np
    from zeromodel.pipeline.executor import PipelineExecutor
    from zeromodel.vpm.stdm import top_left_mass

    # --- Step 1. Download Tesla 1-min for 1 year (yfinance only allows ~60d for 1m bars)
    # So instead, we use 5m bars for 1y
    df = yf.download("TSLA", interval="1d", period="1y")
    assert not df.empty

    # --- Step 2. Features ---
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Volatility"] = df["Return"].rolling(5).std().fillna(0)
    df["MA5"] = df["Close"].rolling(5).mean().bfill()
    df["MA20"] = df["Close"].rolling(20).mean().bfill()
    features = df[["Open","High","Low","Close","Volume","Return","Volatility","MA5","MA20"]]
    X = ((features - features.mean()) / (features.std() + 1e-9)).to_numpy().astype(np.float32)

    # --- Step 3. Run ZeroModel pipeline ---
    stages = [
        {"stage": "normalizer/normalize.NormalizeStage", "params": {}},
        {"stage": "organizer/hierarchical_view.HierarchicalView", "params": {
            "levels": 2,
            "row_frac": 0.5, "col_frac": 0.5,
            "window_mode": "quadrants",
            "norm_mode": "local_subset",
            "alpha": 0.97,
        }},
    ]
    result, ctx = PipelineExecutor(stages).run(X, context={})

    # --- Step 4. Compute ZeroModel signal ---
    signals = []
    for i in range(60, len(result)):
        window = result[i-60:i, :]
        s = top_left_mass(window, Kr=40, Kc=8, alpha=0.97)
        signals.append(s)
    signals = np.array(signals)
    sig_z = (signals - signals.mean()) / (signals.std() + 1e-9)
    sig_times = df.index[60:60+len(sig_z)]

    # --- Step 5. Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,8), sharex=True)
    ax1.plot(df.index, df["Close"], label="TSLA Close", color="black")
    ax1.set_title("Tesla Price (1yr, 5m bars)")
    ax1.legend()

    ax2.plot(sig_times, sig_z, label="ZeroModel Signal", color="blue")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_title("ZeroModel Signal (1yr)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "tesla_signal_year.png"))
    plt.close()
    logger.info("Saved plot to %s", os.path.join(os.getcwd(), "tesla_signal_year.png"))
    print("Saved plot to %s", os.path.join(os.getcwd(), "tesla_signal_year.png"))
