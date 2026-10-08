"""
mean_reversion_backtest.py - Backtest a mean-reversion hypothesis on EURUSD H1.

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: no mt5.order_send() anywhere, no live account.
- The script only READS historical candles from MetaTrader 5.

THE HYPOTHESIS BEING TESTED (the OPPOSITE of the old trend strategy):
- When price stretches too far from its average AND momentum is extreme,
  it tends to snap back toward the mean.
- LONG (BUY)  : candle closes BELOW the lower Bollinger Band AND RSI14 < 30
- SHORT (SELL): candle closes ABOVE the upper Bollinger Band AND RSI14 > 70
- Otherwise   : HOLD

WHY THE EXIT RULES ARE DIFFERENT FROM backtest.py:
- A mean-reversion trade expects a SHORT snap-back, so the take profit is
  set CLOSER (1.5 x ATR) than the stop loss (2.0 x ATR).
  That gives a high-win-rate / small-average-win profile.
  Breakeven win rate = SL / (SL + TP) = 2.0 / 3.5 = ~57% BEFORE costs.
- These two numbers were chosen UP FRONT as reasonable educational values.
  They are NOT the result of a parameter search (no optimization).

NO LOOK-AHEAD BIAS:
- The newest (still forming) candle is dropped.
- A signal on bar i uses only closed candles 0..i.
- The virtual entry happens at the OPEN of bar i+1 - the first moment a
  real trader could have acted.
- SL and TP are fixed at entry from the ATR of bar i.

DISCLAIMER: This is a HISTORICAL backtest with virtual money.
Past results do NOT predict future performance and nothing here is
proven to be profitable.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read historical candles (read-only usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # np.select() to build the signal column

# ============================================================
# SECTION 2: Settings - fixed up front, NOT optimized
# ============================================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 2000   # History to download (task requires >= 2000)
MIN_CANDLES = 1100   # Absolute minimum we accept (~60 warm-up + 1000 bars)

# --- Virtual account settings (same as backtest.py for comparability) ---
STARTING_BALANCE = 10_000.0
LOTS = 0.10                  # Fixed position size
CONTRACT_SIZE = 100_000      # 1.00 lot = 100,000 units of base currency
SPREAD = 0.0002              # 2.0 pips round-trip cost assumption

# --- Exit rules (ATR-based, deterministic) ---
SL_ATR_MULT = 2.0   # Stop loss distance   = 2.0 x ATR (room for the swing)
TP_ATR_MULT = 1.5   # Take profit distance = 1.5 x ATR (quick snap-back)

# --- Indicator / Bollinger settings ---
EMA_PERIOD = 20     # Required indicator; computed for reference (see note
                    # in SECTION 6: the entry rules below do not use it)
BB_PERIOD = 20      # Bollinger moving average period
BB_NUM_STD = 2      # Bollinger standard deviation width
RSI_PERIOD = 14
ATR_PERIOD = 14
WARMUP = 60         # Bars needed before all indicators are valid

# ============================================================
# SECTION 3: Indicator functions (same math as the earlier scripts)
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


def add_bollinger(df, period=20, num_std=2):
    """Bollinger Bands.

    Middle band = simple moving average of the close.
    Upper/lower = middle +/- num_std standard deviations.
    ddof=0 (population std) matches what most charting platforms show.
    """
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


# ============================================================
# SECTION 4: The backtest engine
# Same candle-by-candle logic as backtest.py, but parameterized so we can
# run it TWICE: without and with the RSI filter.
# ============================================================
def run_backtest(df, use_rsi_filter):
    """Run the virtual backtest on df. Returns (trades DataFrame, final balance)."""

    # --- Build the signal column for THIS variant ---
    # Base mean-reversion conditions (Bollinger close outside the bands)
    long_condition = df["close"] < df["BB_LOWER"]
    short_condition = df["close"] > df["BB_UPPER"]

    # Variant B adds the RSI extremes (oversold < 30 / overbought > 70)
    if use_rsi_filter:
        long_condition = long_condition & (df["RSI"] < 30)
        short_condition = short_condition & (df["RSI"] > 70)

    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []                     # Every virtual trade is recorded here
    balance = STARTING_BALANCE      # Virtual running balance
    spread_cost = SPREAD * CONTRACT_SIZE * LOTS  # Flat $ cost per trade

    i = WARMUP
    while i < len(df) - 1:          # Need bar i+1 to exist for the entry
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        # Only enter on a valid signal with a valid ATR value
        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1                         # Next candle
            entry_price = df["open"].iloc[entry_bar]  # Enter at its open

            # Fix SL and TP at entry, using the ATR known at signal time
            if signal == "BUY":
                stop_loss = entry_price - SL_ATR_MULT * atr
                take_profit = entry_price + TP_ATR_MULT * atr
            else:  # SELL
                stop_loss = entry_price + SL_ATR_MULT * atr
                take_profit = entry_price - TP_ATR_MULT * atr

            # Scan forward, bar by bar, for the exit.
            # If BOTH levels fall inside one candle we assume the STOP LOSS
            # was hit first - the conservative assumption.
            exit_index = None
            exit_price = None
            exit_reason = None
            for j in range(entry_bar, len(df)):
                bar_high = df["high"].iloc[j]
                bar_low = df["low"].iloc[j]

                if signal == "BUY":
                    if bar_low <= stop_loss:
                        exit_index, exit_price, exit_reason = j, stop_loss, "STOP_LOSS"
                        break
                    if bar_high >= take_profit:
                        exit_index, exit_price, exit_reason = j, take_profit, "TAKE_PROFIT"
                        break
                else:  # SELL
                    if bar_high >= stop_loss:
                        exit_index, exit_price, exit_reason = j, stop_loss, "STOP_LOSS"
                        break
                    if bar_low <= take_profit:
                        exit_index, exit_price, exit_reason = j, take_profit, "TAKE_PROFIT"
                        break

            # Trade still open at the end of the data -> force-close it
            if exit_index is None:
                exit_index = len(df) - 1
                exit_price = df["close"].iloc[exit_index]
                exit_reason = "END_OF_DATA"

            # --- Virtual profit/loss (EURUSD quote currency = USD) ---
            if signal == "BUY":
                gross_pnl = (exit_price - entry_price) * CONTRACT_SIZE * LOTS
            else:
                gross_pnl = (entry_price - exit_price) * CONTRACT_SIZE * LOTS

            net_pnl = gross_pnl - spread_cost
            balance += net_pnl

            trades.append(
                {
                    "trade": len(trades) + 1,
                    "direction": signal,
                    "entry_time": df["time"].iloc[entry_bar],
                    "entry_price": round(entry_price, 5),
                    "exit_time": df["time"].iloc[exit_index],
                    "exit_price": round(exit_price, 5),
                    "exit_reason": exit_reason,
                    "pnl": round(net_pnl, 2),
                    "balance_after": round(balance, 2),
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 5: Statistics printer (shared by both variants)
# ============================================================
def show_results(label, trades_df, final_balance):
    """Print all required statistics. Returns key numbers for the comparison."""
    print("\n" + "=" * 64)
    print(f"{label}  (HISTORICAL BACKTEST - VIRTUAL MONEY ONLY)")
    print("=" * 64)
    print(f"Rules: BUY when close < lower BB (RSI<30 if enabled), "
          f"SELL when close > upper BB (RSI>70 if enabled)")
    print(f"Exits: SL={SL_ATR_MULT}xATR, TP={TP_ATR_MULT}xATR, "
          f"{LOTS} lots, spread cost ${SPREAD * CONTRACT_SIZE * LOTS:.2f}/trade")

    if len(trades_df) == 0:
        print("\nNo virtual trades were generated in this variant.")
        return None

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    win_rate = len(wins) / len(trades_df) * 100
    total_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100

    # --- Maximum drawdown: biggest peak-to-trough drop of the balance ---
    balance_history = [STARTING_BALANCE] + trades_df["balance_after"].tolist()
    peak = balance_history[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for b in balance_history[1:]:
        if b > peak:
            peak = b
        drawdown = peak - b
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown / peak * 100

    print(f"\nStarting balance    : ${STARTING_BALANCE:,.2f}")
    print(f"Ending balance      : ${final_balance:,.2f}")
    print(f"Total return        : {total_return:+.2f}%")
    print(f"\nNumber of trades    : {len(trades_df)}")
    print(f"BUY trades          : {(trades_df['direction'] == 'BUY').sum()}")
    print(f"SELL trades         : {(trades_df['direction'] == 'SELL').sum()}")
    print(f"Winning trades      : {len(wins)}")
    print(f"Losing trades       : {len(losses)}")
    print(f"Win rate            : {win_rate:.1f}%")
    print(f"\nGross profit        : ${gross_profit:,.2f}")
    print(f"Gross loss          : ${gross_loss:,.2f}")
    if gross_loss > 0:
        print(f"Profit factor       : {gross_profit / gross_loss:.2f}")
    else:
        print("Profit factor       : infinite (no losing trades)")
    print(f"Average trade       : ${trades_df['pnl'].mean():,.2f}")
    print(f"\nMaximum drawdown    : ${max_dd:,.2f}  ({max_dd_pct:.2f}%)")

    # --- Show the last 10 virtual trades ---
    print("\nLast 10 virtual trades:")
    print(trades_df.tail(10).to_string(index=False))

    return {
        "return": total_return,
        "trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_dd_pct": max_dd_pct,
        "avg_trade": trades_df["pnl"].mean(),
    }


# ============================================================
# SECTION 6: Main program
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

    if len(rates) < MIN_CANDLES:
        print(f"ERROR: Not enough history. Got {len(rates)} candles, "
              f"but at least {MIN_CANDLES} are needed.")
        quit()

    if len(rates) < NUM_CANDLES:
        print(f"NOTE: Broker returned {len(rates)} candles (requested {NUM_CANDLES}). Continuing...")

    # --- Build the DataFrame; drop the still-forming last candle ---
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)   # closed candles only

    # --- Calculate all indicators ---
    df = add_ema(df, EMA_PERIOD)      # "EMA20" - computed for reference; the
                                      # entry rules currently do NOT use it,
                                      # it is kept for future experiments.
    df = add_rsi(df, RSI_PERIOD)      # "RSI"
    df = add_atr(df, ATR_PERIOD)      # "ATR"
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)  # "BB_UPPER"/"BB_MIDDLE"/"BB_LOWER"

    # ============================================================
    # SECTION 7: Run BOTH variants on the SAME data
    #   A) Bollinger bands only (no RSI filter)
    #   B) Bollinger bands + RSI filter
    # This directly answers: does RSI actually improve the idea?
    # ============================================================
    print(f"\nRunning variant A: Bollinger bands only (no RSI filter)...")
    trades_a, balance_a = run_backtest(df.copy(), use_rsi_filter=False)

    print(f"Running variant B: Bollinger bands + RSI filter...")
    trades_b, balance_b = run_backtest(df.copy(), use_rsi_filter=True)

    result_a = show_results("A) BOLLINGER ONLY (no RSI filter)", trades_a, balance_a)
    result_b = show_results("B) BOLLINGER + RSI FILTER", trades_b, balance_b)

    # ============================================================
    # SECTION 8: A vs B comparison (neutral facts, no optimization)
    # ============================================================
    if result_a is not None and result_b is not None:
        print("\n" + "=" * 64)
        print("COMPARISON: does the RSI filter improve the mean-reversion idea?")
        print("=" * 64)
        print(f"{'Metric':<16}{'A: BB only':>16}{'B: BB + RSI':>16}")
        print(f"{'Total return':<16}{result_a['return']:>15.2f}%{result_b['return']:>15.2f}%")
        print(f"{'Trades':<16}{result_a['trades']:>16}{result_b['trades']:>16}")
        print(f"{'Win rate':<16}{result_a['win_rate']:>15.1f}%{result_b['win_rate']:>15.1f}%")
        print(f"{'Profit factor':<16}{result_a['profit_factor']:>16.2f}{result_b['profit_factor']:>16.2f}")
        print(f"{'Max drawdown':<16}{result_a['max_dd_pct']:>15.2f}%{result_b['max_dd_pct']:>15.2f}%")
        print(f"{'Avg trade':<16}{result_a['avg_trade']:>15.2f}${result_b['avg_trade']:>15.2f}$")

        # Neutral, factual interpretation (computed, not hand-tuned):
        if result_b["profit_factor"] > result_a["profit_factor"]:
            verdict = "IMPROVED the profit factor"
        elif result_b["profit_factor"] < result_a["profit_factor"]:
            verdict = "WORSENED the profit factor"
        else:
            verdict = "left the profit factor unchanged"
        print(f"\nOn this historical data, the RSI filter {verdict} "
              f"({result_a['profit_factor']:.2f} -> {result_b['profit_factor']:.2f}) "
              f"and changed the number of trades "
              f"({result_a['trades']} -> {result_b['trades']}).")

    # ============================================================
    # SECTION 9: Disclaimer
    # ============================================================
    print("\nDISCLAIMER: This is a HISTORICAL BACKTEST with virtual money.")
    print("Results do NOT predict future performance, and nothing here")
    print("is proven to be profitable. No parameters were optimized.")
    print("Educational use only.")

finally:
    # ============================================================
    # SECTION 10: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
