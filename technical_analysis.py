"""
technical_analysis.py - Read EURUSD H1 candles and calculate basic indicators.

DEMO / EDUCATIONAL PROJECT ONLY:
- This script only READS market data.
- It never places, modifies, or closes any trade (no mt5.order_send()).
- The signals below are for LEARNING purposes only.
  This simple strategy is NOT tested and NOT proven to be profitable.
"""

# ============================================================
# SECTION 1: Imports
# ============================================================
import MetaTrader5 as mt5  # Official MetaTrader 5 Python package
import pandas as pd        # DataFrame + indicator math (ewm, rolling)
import numpy as np         # np.select() to build the signal column

SYMBOL = "EURUSD"


# ============================================================
# SECTION 2: Indicator functions (pure pandas, no extra libraries)
# ============================================================
def add_ema(df, period):
    """Exponential Moving Average of the close price.

    ewm(span=period) is pandas' built-in exponential weighting.
    min_periods makes the first (period - 1) rows NaN instead of
    showing unreliable 'half-calculated' values.
    """
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def add_rsi(df, period=14):
    """Relative Strength Index (Wilder's classic calculation).

    RSI = 100 - 100 / (1 + RS),  where RS = average gain / average loss
    over the last `period` bars. Wilder smoothing is an exponential
    average with alpha = 1/period.
    """
    delta = df["close"].diff()          # Change in close from one bar to the next
    gain = delta.clip(lower=0)          # Positive changes only (losses -> 0)
    loss = -delta.clip(upper=0)         # Negative changes made positive

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss            # If avg_loss is 0, rs = inf -> RSI = 100
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df, period=14):
    """Average True Range - a simple measure of volatility.

    True Range for one bar is the greatest of:
      - high - low
      - |high - previous close|
      - |low  - previous close|
    ATR is the smoothed average of the True Range.
    """
    prev_close = df["close"].shift(1)   # Previous bar's close
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)                       # Take the largest of the three

    df["ATR"] = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


# ============================================================
# SECTION 3: Connect to MetaTrader 5
# ============================================================
print("Connecting to MetaTrader 5...")

if not mt5.initialize():
    print("ERROR: Could not connect to MetaTrader 5.")
    print("Make sure the MT5 desktop terminal is running and logged into your DEMO account.")
    print("Last error:", mt5.last_error())
    quit()  # Nothing was opened, so no shutdown() call is needed here.

print("Connected to MetaTrader 5 successfully!")

try:
    # Make sure the symbol is enabled in Market Watch (data only, not trading).
    if not mt5.symbol_select(SYMBOL, True):
        print(f"ERROR: Symbol {SYMBOL} is not available on this account.")
        print("Last error:", mt5.last_error())
        quit()

    # ============================================================
    # SECTION 4: Retrieve 200 EURUSD H1 candles
    # ============================================================
    print(f"\nReading the latest 200 {SYMBOL} candles (H1 timeframe)...")

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 200)

    # --- Safety check 1: did we get any data at all? ---
    if rates is None or len(rates) == 0:
        print(f"ERROR: Could not retrieve {SYMBOL} data.")
        print("Last error:", mt5.last_error())
        quit()

    # --- Safety check 2: is there enough data for the indicators? ---
    # EMA50 needs 50 bars of warm-up, so we require at least 60.
    MIN_CANDLES = 60
    if len(rates) < MIN_CANDLES:
        print(f"ERROR: Not enough data. Got {len(rates)} candles, "
              f"but at least {MIN_CANDLES} are needed for the indicators.")
        quit()

    if len(rates) < 200:
        print(f"NOTE: Only {len(rates)} candles available (requested 200). Continuing anyway...")

    # ============================================================
    # SECTION 5: Convert to a pandas DataFrame
    # ============================================================
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")  # Unix time -> readable datetime

    # ============================================================
    # SECTION 6: Calculate the indicators (new DataFrame columns)
    # ============================================================
    df = add_ema(df, 20)   # -> column "EMA20"
    df = add_ema(df, 50)   # -> column "EMA50"
    df = add_rsi(df, 14)   # -> column "RSI"
    df = add_atr(df, 14)   # -> column "ATR"

    # ============================================================
    # SECTION 7: Simple educational signal (NOT a proven strategy!)
    #
    # BUY  when: EMA20 > EMA50  AND  50 <= RSI <= 70
    # SELL when: EMA20 < EMA50  AND  30 <= RSI <= 50
    # Otherwise: HOLD
    #
    # np.select() picks the first condition that is True.
    # Rows where indicators are still NaN compare as False,
    # so they naturally fall through to "HOLD".
    # ============================================================
    buy_condition = (df["EMA20"] > df["EMA50"]) & (df["RSI"] >= 50) & (df["RSI"] <= 70)
    sell_condition = (df["EMA20"] < df["EMA50"]) & (df["RSI"] >= 30) & (df["RSI"] <= 50)

    df["signal"] = np.select([buy_condition, sell_condition], ["BUY", "SELL"], default="HOLD")

    # ============================================================
    # SECTION 8: Display the latest 10 rows
    # ============================================================
    columns = ["time", "close", "EMA20", "EMA50", "RSI", "ATR", "signal"]

    print(f"\nLatest 10 {SYMBOL} candles with indicators:")

    # Round only the numeric columns (rounding the datetime column would
    # trigger a pandas warning and do nothing useful).
    display_df = df[columns].tail(10).copy()
    numeric_columns = ["close", "EMA20", "EMA50", "RSI", "ATR"]
    display_df[numeric_columns] = display_df[numeric_columns].round(5)
    print(display_df.to_string(index=False))

    print(f"\nTotal candles retrieved: {len(df)}")
    print(f"Latest signal: {df['signal'].iloc[-1]}")

    # IMPORTANT DISCLAIMER - this strategy is untested and NOT proven profitable.
    print("\nDISCLAIMER: These signals are for education only.")
    print("This simple rule-based strategy is NOT tested and NOT guaranteed to be profitable.")

finally:
    # ============================================================
    # SECTION 9: Always disconnect when the script finishes
    # ============================================================
    # `finally` guarantees this runs even if an error happened above.
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
