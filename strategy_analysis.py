"""
strategy_analysis.py - Investigate WHY the EMA/RSI strategy loses money.

DEMO / EDUCATIONAL PROJECT ONLY:
- READ-ONLY: this script never places, modifies, or closes a trade
  (there is no mt5.order_send() anywhere in this file).
- It re-uses the SAME data and indicators as backtest.py and studies the
  signals statistically.

IMPORTANT - THIS IS NOT AN OPTIMIZER:
- It does NOT search for better parameters.
- Every threshold below (trend/volatility buckets, RSI ranges) is FIXED
  and chosen up front, before seeing any results.
- The goal is only to understand the weaknesses of the existing strategy.

WHAT "WIN" MEANS HERE:
- A BUY signal "wins" if the price is HIGHER 12 candles after the signal
  bar's close. A SELL signal "wins" if the price is LOWER 12 candles
  after the signal bar's close.
- This is a measurement of the OUTCOME (we are allowed to look at future
  closes to JUDGE a past signal - that is not look-ahead bias).
- Look-ahead bias would be using future data to DECIDE the signal. We
  never do that: the signal on bar i uses only bars 0..i (closed candles).

DISCLAIMER: Educational statistics only. Nothing here proves the
strategy is (or can be) profitable.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read historical candles (read-only usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # np.select() for the signal column

# ============================================================
# SECTION 2: Fixed settings (no searching/tuning - chosen up front)
# ============================================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 2000   # Same history size as backtest.py for a fair comparison

# --- Indicator periods (same as the existing strategy) ---
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
WARMUP = 50  # Bars needed before EMA50 (the slowest indicator) is valid

# --- Forward-return horizons to measure (in candles) ---
HORIZONS = [1, 3, 6, 12]
WIN_HORIZON = 12  # The horizon used to decide if a signal "won"

# --- Fixed regime buckets (NOT tuned; round educational values) ---
TREND_STRONG_ATR = 0.5   # |EMA20-EMA50| >= 0.5 x ATR  -> "strong trend"
VOL_HIGH_FACTOR = 1.2    # ATR >= 1.2 x its median       -> "high volatility"
VOL_LOW_FACTOR = 0.8     # ATR <= 0.8 x its median       -> "low volatility"

# ============================================================
# SECTION 3: Indicator functions (same math as backtest.py)
# ============================================================
def add_ema(df, period):
    """Exponential Moving Average of the close price."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def add_rsi(df, period=14):
    """Relative Strength Index (Wilder's classic calculation)."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df, period=14):
    """Average True Range - a simple volatility measure."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


def fmt_pct(value):
    """Small helper to print percentages neatly (handles NaN)."""
    if pd.isna(value):
        return "n/a"
    return f"{value:+.4f}%"


# ============================================================
# SECTION 4: Connect to MetaTrader 5 and download history
# ============================================================
print("Connecting to MetaTrader 5...")

if not mt5.initialize():
    print("ERROR: Could not connect to MetaTrader 5.")
    print("Make sure the MT5 desktop terminal is running and logged into your DEMO account.")
    print("Last error:", mt5.last_error())
    quit()

print("Connected to MetaTrader 5 successfully!")

try:
    if not mt5.symbol_select(SYMBOL, True):
        print(f"ERROR: Symbol {SYMBOL} is not available on this account.")
        print("Last error:", mt5.last_error())
        quit()

    print(f"\nDownloading the latest {NUM_CANDLES} {SYMBOL} candles (H1)...")

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)

    if rates is None or len(rates) == 0:
        print(f"ERROR: Could not retrieve {SYMBOL} data.")
        print("Last error:", mt5.last_error())
        quit()

    # ============================================================
    # SECTION 5: Build the DataFrame and DROP the forming candle
    # ============================================================
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # The LAST row is still forming right now -> drop it (closed data only).
    df = df.iloc[:-1].reset_index(drop=True)

    # ============================================================
    # SECTION 6: Indicators and the signal column (same strategy)
    # ============================================================
    df = add_ema(df, EMA_FAST)
    df = add_ema(df, EMA_SLOW)
    df = add_rsi(df, RSI_PERIOD)
    df = add_atr(df, ATR_PERIOD)

    buy_condition = (df["EMA20"] > df["EMA50"]) & (df["RSI"] >= 50) & (df["RSI"] <= 70)
    sell_condition = (df["EMA20"] < df["EMA50"]) & (df["RSI"] >= 30) & (df["RSI"] <= 50)
    df["signal"] = np.select([buy_condition, sell_condition], ["BUY", "SELL"], default="HOLD")

    # ============================================================
    # SECTION 7: Direction-adjusted forward returns
    #
    # For each bar we measure how the price moved AFTER the bar closed:
    #   raw return after N candles = (close[i+N] - close[i]) / close[i]
    # For a BUY signal a positive move is good; for a SELL signal a
    # NEGATIVE move is good. We flip the SELL sign so that "bigger =
    # better for the signal" in every statistic below.
    #
    # close.shift(-N) simply "looks N candles into the future" for the
    # MEASUREMENT. The signal itself never uses it (no look-ahead bias).
    # The last N rows have no future close yet -> they become NaN and are
    # excluded from the statistics automatically.
    # ============================================================
    for n in HORIZONS:
        raw_return = (df["close"].shift(-n) - df["close"]) / df["close"] * 100
        df[f"fwd_return_{n}"] = raw_return  # raw direction-adjusted below

    def adjusted(series, direction):
        """Flip the sign for SELL signals so positive = good for the signal."""
        return series * direction

    directions = np.where(df["signal"] == "SELL", -1.0, 1.0)
    for n in HORIZONS:
        df[f"ret_{n}"] = adjusted(df[f"fwd_return_{n}"], directions)

    # A signal bar is only usable if its signal and indicators are valid
    # (past warm-up) AND the longest forward return exists (not NaN).
    usable = (
        (df.index >= WARMUP)
        & df["signal"].isin(["BUY", "SELL", "HOLD"])
        & pd.notna(df[f"ret_{WIN_HORIZON}"])
        & pd.notna(df["ATR"])
    )
    study = df[usable].copy()

    # ============================================================
    # SECTION 8: Basic signal counts and win rates
    # ============================================================
    buys = study[study["signal"] == "BUY"]
    sells = study[study["signal"] == "SELL"]
    holds = study[study["signal"] == "HOLD"]

    buy_wins = (buys[f"ret_{WIN_HORIZON}"] > 0).sum()
    sell_wins = (sells[f"ret_{WIN_HORIZON}"] > 0).sum()

    print("\n" + "=" * 64)
    print("STRATEGY ANALYSIS - WHY DOES THE STRATEGY LOSE?")
    print("(virtual / educational only - no trades were placed)")
    print("=" * 64)
    print(f"Period analyzed : {study['time'].iloc[0]}  ->  {study['time'].iloc[-1]}")
    print(f"Bars studied    : {len(study)} (closed candles only, warm-up excluded)")
    print(f"\n1. Total BUY signals   : {len(buys)}")
    print(f"2. Total SELL signals  : {len(sells)}")
    print(f"3. Total HOLD periods  : {len(holds)} "
          f"({len(holds) / len(study) * 100:.1f}% of all bars)")
    print(f"\n4. BUY win rate  (after {WIN_HORIZON} candles): "
          f"{buy_wins}/{len(buys)} = {buy_wins / len(buys) * 100 if len(buys) else float('nan'):.1f}%")
    print(f"5. SELL win rate (after {WIN_HORIZON} candles): "
          f"{sell_wins}/{len(sells)} = {sell_wins / len(sells) * 100 if len(sells) else float('nan'):.1f}%")

    # ============================================================
    # SECTION 9: Average forward returns after 1 / 3 / 6 / 12 candles
    # (positive number = the market moved in the signal's favor,
    #  negative number = the market moved AGAINST the signal)
    # ============================================================
    print(f"\n6-9. Average direction-adjusted return after N candles:")
    header = "  Signal   " + "".join([f"{'after ' + str(n):>12}" for n in HORIZONS])
    print(header)
    for name, group in (("BUY", buys), ("SELL", sells)):
        row = f"  {name:<8} "
        for n in HORIZONS:
            row += f"{group[f'ret_{n}'].mean():>12.4f}%"
        print(row)

    # ============================================================
    # SECTION 10: Performance when EMA20 > EMA50 vs EMA20 < EMA50
    # (measured on ALL signals in that regime, using the WIN_HORIZON)
    # ============================================================
    print(f"\n10-11. Performance by trend direction (avg return after "
          f"{WIN_HORIZON} candles):")
    uptrend = study[study["EMA20"] > study["EMA50"]]
    downtrend = study[study["EMA20"] < study["EMA50"]]
    for name, group in (("EMA20 > EMA50 (uptrend bars)", uptrend),
                        ("EMA20 < EMA50 (downtrend bars)", downtrend)):
        if len(group) == 0:
            print(f"  {name:<32}: no bars")
            continue
        avg_ret = group[f"ret_{WIN_HORIZON}"].mean()
        wr = (group[f"ret_{WIN_HORIZON}"] > 0).mean() * 100
        print(f"  {name:<32}: {fmt_pct(avg_ret):>10}   win rate {wr:5.1f}%   "
              f"({len(group)} bars)")

    # ============================================================
    # SECTION 12: Performance for different RSI ranges
    # Fixed educational buckets: 0-30, 30-50, 50-70, 70-100.
    # ============================================================
    print(f"\n12. Performance by RSI range (avg direction-adjusted return after "
          f"{WIN_HORIZON} candles):")
    rsi_buckets = [(0, 30), (30, 50), (50, 70), (70, 100)]
    for low, high in rsi_buckets:
        mask = (study["RSI"] >= low) & (study["RSI"] <= high)
        group = study[mask]
        if len(group) == 0:
            print(f"  RSI {low:>3}-{high:<3}: no bars")
            continue
        avg_ret = group[f"ret_{WIN_HORIZON}"].mean()
        wr = (group[f"ret_{WIN_HORIZON}"] > 0).mean() * 100
        n_signals = (group["signal"] != "HOLD").sum()
        print(f"  RSI {low:>3}-{high:<3}: {fmt_pct(avg_ret):>10}   win rate {wr:5.1f}%   "
              f"({len(group)} bars, {n_signals} were BUY/SELL signals)")

    # ============================================================
    # SECTION 13-14: Average ATR during winning vs losing signals
    # Question: do winners appear in calmer or wilder markets?
    # ============================================================
    signals = study[study["signal"] != "HOLD"].copy()
    wins = signals[signals[f"ret_{WIN_HORIZON}"] > 0]
    losses = signals[signals[f"ret_{WIN_HORIZON}"] <= 0]

    print(f"\n13. Average ATR during WINNING signals  : "
          f"{wins['ATR'].mean():.5f}  ({len(wins)} signals)")
    print(f"14. Average ATR during LOSING signals   : "
          f"{losses['ATR'].mean():.5f}  ({len(losses)} signals)")

    # ============================================================
    # SECTION 15: Regime study - strong/weak trend, high/low volatility
    # Fixed thresholds defined in SECTION 2 (not tuned!).
    # ============================================================
    # Trend strength: how far apart the EMAs are, measured in ATRs.
    study["trend_strength"] = (study["EMA20"] - study["EMA50"]).abs() / study["ATR"]
    strong_trend = study["trend_strength"] >= TREND_STRONG_ATR

    # Volatility regime: ATR compared to its own median over the period.
    atr_median = study["ATR"].median()
    high_vol = study["ATR"] >= atr_median * VOL_HIGH_FACTOR
    low_vol = study["ATR"] <= atr_median * VOL_LOW_FACTOR

    def regime_report(label, mask):
        group = study[mask]
        if len(group) == 0:
            print(f"  {label:<34}: no bars")
            return
        avg_ret = group[f"ret_{WIN_HORIZON}"].mean()
        wr = (group[f"ret_{WIN_HORIZON}"] > 0).mean() * 100
        n_signals = (group["signal"] != "HOLD").sum()
        print(f"  {label:<34}: {fmt_pct(avg_ret):>10}   win rate {wr:5.1f}%   "
              f"({len(group)} bars, {n_signals} signals)")

    print(f"\nRegime study (avg direction-adjusted return after {WIN_HORIZON} candles):")
    print(f"  (ATR median over period: {atr_median:.5f})")
    regime_report("Strong trend (EMA gap >= 0.5 ATR)", strong_trend)
    regime_report("Weak trend   (EMA gap <  0.5 ATR)", ~strong_trend)
    regime_report("High volatility (ATR >= 1.2x med)", high_vol)
    regime_report("Low volatility  (ATR <= 0.8x med)", low_vol)

    # ============================================================
    # SECTION 16: Disclaimer
    # ============================================================
    print("\nDISCLAIMER: These are historical statistics, not predictions.")
    print("Nothing here proves the strategy is profitable, and no")
    print("parameters were optimized. Educational use only.")

finally:
    # ============================================================
    # SECTION 17: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
