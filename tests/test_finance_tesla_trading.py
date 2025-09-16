from unittest import signals
import pytest
import numpy as np
import yfinance as yf
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.vpm.stdm import top_left_mass
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for headless testing

logger = logging.getLogger(__name__)

@pytest.mark.slow
def test_finance_tesla_trading(tmp_path):
    # --- Step 1. Download Tesla 1-min data ---
    df = yf.download("TSLA", interval="1m", period="5d")
    assert not df.empty

    # --- Step 2. Features ---
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Volatility"] = df["Return"].rolling(5).std().fillna(0)
    df["MA5"] = df["Close"].rolling(5).mean().bfill()
    df["MA20"] = df["Close"].rolling(20).mean().bfill()
    features = df[["Open","High","Low","Close","Volume","Return","Volatility","MA5","MA20"]]
    X = ((features - features.mean()) / (features.std() + 1e-9)).to_numpy().astype(np.float32)

    # --- Step 3. Run ZeroModel pipeline (1 level just for demo) ---
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
    for i in range(50, len(result)):   # rolling window
        window = result[i-50:i, :]
        s = top_left_mass(window, Kr=80, Kc=16, alpha=0.97)
        signals.append(s)
    signals = np.array(signals)
    sig_z = (signals - signals.mean()) / (signals.std() + 1e-9)

    # --- Step 5. Trading logic ---
    capital, position, trades = 10000, 0, []
    entry_price = None
    closes = df["Close"].to_numpy()

    for t, z in enumerate(sig_z, start=50):
        if position == 0 and z > 0.5:  # enter long
            position = 1
            entry_price = float(closes[t])
            trades.append(("BUY", df.index[t], entry_price))
        elif position == 1 and z < -0.5:  # exit
            position = 0
            exit_price = float(closes[t])
            pnl = (exit_price / entry_price - 1) * 100
            trades.append(("SELL", df.index[t], exit_price, pnl))
            capital *= exit_price / entry_price

    # Compare with Buy & Hold
    bh_return = float(closes[-1] / closes[0] - 1) * 100
    zm_return = float(capital / 10000 - 1) * 100

    logger.info("Trades:")
    for tr in trades: 
        logger.info(tr)
    logger.info(f"Buy & Hold return: {bh_return:.2f}%")
    logger.info(f"ZeroModel return: {zm_return:.2f}%")

    logger.debug(f"Buy & Hold return: {bh_return:.2f}%")
    logger.debug(f"ZeroModel return: {zm_return:.2f}%")

    # --- Step 7. Assertions ---
    assert len(trades) > 0
    assert isinstance(zm_return, float)

        # --- Step 6. Pretty-logger.info trade log ---
    logger.info("\nTrade Log:")
    logger.info("Trade Log:")
    logger.info(f"{'Idx':<4} {'Action':<6} {'Time':<20} {'Price':>10} {'PnL%':>8}")
    logger.info(f"{'Idx':<4} {'Action':<6} {'Time':<20} {'Price':>10} {'PnL%':>8}")
    logger.info("-" * 55)
    logger.info("-" * 55)

    for i, tr in enumerate(trades, start=1):
        if tr[0] == "BUY":
            _, ts, price = tr
            logger.info(f"{i:<4} BUY    {ts:%Y-%m-%d %H:%M} {price:>10.2f} {'-':>8}")
            logger.info(f"Trade {i}: BUY at {ts:%Y-%m-%d %H:%M} for {price:.2f}")
        elif tr[0] == "SELL":
            _, ts, price, pnl = tr
            logger.info(f"{i:<4} SELL   {ts:%Y-%m-%d %H:%M} {price:>10.2f} {pnl:>7.2f}")
            logger.info(f"Trade {i}: SELL at {ts:%Y-%m-%d %H:%M} for {price:.2f}, PnL: {pnl:.2f}%")


@pytest.mark.slow
def test_finance_tesla_trading_demo2(tmp_path):
    # --- Step 1. Download Tesla 1-min data ---
    df = yf.download("TSLA", interval="1m", period="1d")
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
    for i in range(30, len(result)):   # shorter rolling window
        window = result[i-30:i, :]
        s = top_left_mass(window, Kr=40, Kc=8, alpha=0.97)
        signals.append(s)
    signals = np.array(signals)
    sig_z = (signals - signals.mean()) / (signals.std() + 1e-9)

    # --- Step 5. Auto thresholds to ensure multiple trades ---
    high_th = np.percentile(sig_z, 70)   # enter on upper 30%
    low_th  = np.percentile(sig_z, 30)   # exit on lower 30%

    # --- Step 6. Trading logic ---
    capital, position, trades = 10000, 0, []
    entry_price = None
    closes = df["Close"].to_numpy()

    for t, z in enumerate(sig_z, start=30):
        if position == 0 and z > high_th:  # enter long
            position = 1
            entry_price = float(closes[t])
            trades.append(("BUY", df.index[t], entry_price))
        elif position == 1 and z < low_th:  # exit
            position = 0
            exit_price = float(closes[t])
            pnl = (exit_price / entry_price - 1) * 100
            trades.append(("SELL", df.index[t], exit_price, pnl))
            capital *= exit_price / entry_price

    # Compare with Buy & Hold
    bh_return = float(closes[-1] / closes[0] - 1) * 100
    zm_return = float(capital / 10000 - 1) * 100

    logger.info("\nTrades:")
    for tr in trades:
        logger.info(tr)
    logger.info(f"Buy & Hold return: {bh_return:.2f}%")
    logger.info(f"ZeroModel return: {zm_return:.2f}%")

    # --- Step 7. Plot signal with thresholds ---
    plt.figure(figsize=(10,5))
    plt.plot(sig_z, label="ZeroModel signal (z-score)")
    plt.axhline(high_th, color='g', linestyle='--', label=f'High threshold {high_th:.2f}')
    plt.axhline(low_th, color='r', linestyle='--', label=f'Low threshold {low_th:.2f}')
    plt.legend()
    plt.title("ZeroModel Tesla Trading Signal")
    plt.savefig(tmp_path / "tesla_signal.png")
    plt.close()

    logger.info(f"Result saved: {tmp_path}/tesla_signal.png")

    # --- Assertions ---
    assert len(trades) >= 2, "Should produce at least one buy/sell cycle"
    assert isinstance(zm_return, float)


@pytest.mark.slow
def test_finance_tesla_trading_demo3(tmp_path):
    # --- Step 1. Download Tesla 1-min data ---
    df = yf.download("TSLA", interval="1m", period="1d")
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
    for i in range(30, len(result)):   # shorter rolling window
        window = result[i-30:i, :]
        s = top_left_mass(window, Kr=40, Kc=8, alpha=0.97)
        signals.append(s)
    # build signals (already shorter than closes)
    signals = np.array(signals)
    sig_z = (signals - signals.mean()) / (signals.std() + 1e-9)

    # align timestamps
    sig_times = df.index[50:50+len(sig_z)]


    
    # --- Step 5. Auto thresholds to ensure multiple trades ---
    high_th = np.percentile(sig_z, 70)   # enter on upper 30%
    low_th  = np.percentile(sig_z, 30)   # exit on lower 30%

    # --- Step 6. Trading logic ---
    capital, position, trades = 10000, 0, []
    entry_price = None
    closes = df["Close"].to_numpy()
    buy_times, buy_prices, sell_times, sell_prices = [], [], [], []

    for t, z in enumerate(sig_z, start=30):
        if position == 0 and z > high_th:  # enter long
            position = 1
            entry_price = float(closes[t])
            trades.append(("BUY", df.index[t], entry_price))
            buy_times.append(df.index[t])
            buy_prices.append(entry_price)
        elif position == 1 and z < low_th:  # exit
            position = 0
            exit_price = float(closes[t])
            pnl = (exit_price / entry_price - 1) * 100
            trades.append(("SELL", df.index[t], exit_price, pnl))
            capital *= exit_price / entry_price
            sell_times.append(df.index[t])
            sell_prices.append(exit_price)

    # Compare with Buy & Hold
    bh_return = float(closes[-1] / closes[0] - 1) * 100
    zm_return = float(capital / 10000 - 1) * 100

    logger.info("\nTrades:")
    for tr in trades:
        logger.info(tr)
    logger.info(f"Buy & Hold return: {bh_return:.2f}%")
    logger.info(f"ZeroModel return: {zm_return:.2f}%")

    # --- Step 7. Plot price with trades + signal ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)

    sig_times = df.index[30:30+len(sig_z)]

    # Plot ZeroModel signal
    ax1.plot(sig_times, sig_z, label="ZeroModel Signal", color="blue")
    ax1.axhline(high_th, color="green", linestyle="--", label=f"Buy Th ({high_th:.2f})")
    ax1.axhline(low_th, color="red", linestyle="--", label=f"Sell Th ({low_th:.2f})")

    # mark trades on signal chart
    for tr in trades:
        if tr[0] == "BUY":
            idx = np.where(sig_times == tr[1])[0][0]
            ax1.scatter(tr[1], sig_z[idx], marker="^", color="green", s=100)
        elif tr[0] == "SELL":
            idx = np.where(sig_times == tr[1])[0][0]
            ax1.scatter(tr[1], sig_z[idx], marker="v", color="red", s=100)

    ax1.set_ylabel("Signal (z-score)")
    ax1.legend()
    ax1.set_title("ZeroModel Trading Signal")

    # Plot price with buy/sell markers
    ax2.plot(df.index, df["Close"], label="Close Price", color="black")

    ax2.scatter(buy_times, buy_prices, marker="^", color="green", s=100, label="BUY")
    ax2.scatter(sell_times, sell_prices, marker="v", color="red", s=100, label="SELL")

    ax2.set_ylabel("Price ($)")
    ax2.set_title("Tesla Price with Trades")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(tmp_path / "tesla_trades3.png")
    plt.close()
    logger.info(f"Result saved: {tmp_path}/tesla_trades3.png")




def get_one_month_1m(symbol="TSLA"):
    import pandas as pd
    data = []
    for days_back in range(0, 28, 7):  # fetch in 7-day chunks
        df = yf.download(symbol, interval="1m", period="7d", 
                         start=pd.Timestamp.today() - pd.Timedelta(days=days_back+7),
                         end=pd.Timestamp.today() - pd.Timedelta(days=days_back))
        data.append(df)
    return pd.concat(data).sort_index()

df = get_one_month_1m()


@pytest.mark.slow
def test_finance_tesla_trading_month(tmp_path):
    import matplotlib.pyplot as plt

    # --- Step 1. Download Tesla 1-min data (1 month) ---
    df = yf.download("TSLA", interval="5m", period="1mo")
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
    closes = df["Close"].to_numpy()

    # --- Step 5. Dynamic thresholds ---
    high_th = np.percentile(sig_z, 70)
    low_th  = np.percentile(sig_z, 30)

    # --- Step 6. Trading logic ---
    capital, position, trades = 10000, 0, []
    equity = []
    entry_price = None
    buy_times, buy_prices, sell_times, sell_prices = [], [], [], []

    for t, z in enumerate(sig_z, start=60):
        price = float(closes[t])
        if position == 0 and z > high_th:
            position, entry_price = 1, price
            trades.append(("BUY", df.index[t], price))
            buy_times.append(df.index[t]); buy_prices.append(price)
        elif position == 1 and z < low_th:
            position = 0
            exit_price = price
            pnl = (exit_price / entry_price - 1) * 100
            trades.append(("SELL", df.index[t], exit_price, pnl))
            capital *= exit_price / entry_price
            sell_times.append(df.index[t]); sell_prices.append(exit_price)

        # track equity each bar
        if position == 1:
            equity.append(capital * (price / entry_price))
        else:
            equity.append(capital)

    bh_return = float(closes[-1] / closes[0] - 1) * 100
    zm_return = float(capital / 10000 - 1) * 100

    # --- Step 7. Plots ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14,10), sharex=True)

    # Price + trades
    ax1.plot(df.index, df["Close"], label="TSLA Price", color="black")
    ax1.scatter(buy_times, buy_prices, marker="^", color="green", s=80, label="BUY")
    ax1.scatter(sell_times, sell_prices, marker="v", color="red", s=80, label="SELL")
    ax1.legend(); ax1.set_title("ZeroModel Trading Demo (TSLA 1mo)")

    # Signal
    ax2.plot(sig_times, sig_z, label="ZeroModel Signal", color="blue")
    ax2.axhline(high_th, color="green", linestyle="--", label="Buy Threshold")
    ax2.axhline(low_th, color="red", linestyle="--", label="Sell Threshold")
    ax2.legend()

    # Equity vs Buy&Hold
    ax3.plot(sig_times, equity, label="ZeroModel Equity", color="blue")
    ax3.plot(df.index, 10000 * closes / closes[0], label="Buy&Hold Equity", color="gray")
    ax3.legend(); ax3.set_ylabel("Equity ($)")

    plt.tight_layout()
    plt.savefig(tmp_path / "tesla_trading_month.png")
    plt.close()

    logger.info("Trades:")
    for tr in trades: logger.info(tr)
    logger.info(f"Buy & Hold return: {bh_return:.2f}%")
    logger.info(f"ZeroModel return: {zm_return:.2f}%")
    logger.info(f"Result saved: {tmp_path}/tesla_trading_month.png")


@pytest.mark.slow
def test_finance_tesla_trading_demo_year(tmp_path):
    import yfinance as yf
    import matplotlib.pyplot as plt
    import numpy as np
    from zeromodel.pipeline.executor import PipelineExecutor
    from zeromodel.vpm.stdm import top_left_mass

    # --- Step 1. Download Tesla intraday data ---
    # Yahoo often limits intraday (<=60d) for 5m bars; fall back and skip if unavailable.
    df = yf.download("TSLA", interval="5m", period="1y")
    if df.empty:
        df = yf.download("TSLA", interval="5m", period="60d")
    if df.empty:
        pytest.skip("TSLA intraday data unavailable (rate limit or Yahoo restriction)")

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
    ax1.set_title("Tesla Price (5m bars)")
    ax1.legend()

    ax2.plot(sig_times, sig_z, label="ZeroModel Signal", color="blue")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_title("ZeroModel Signal")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(tmp_path / "tesla_signal_year.png")
    plt.close()
    logger.info(f"Result saved: {tmp_path}/tesla_signal_year.png")
