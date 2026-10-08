"""
backtest.py - Historical backtest of the EMA/RSI strategy on EURUSD H1.

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: this script never places, modifies, or closes a real
  trade (there is no mt5.order_send() anywhere in this file).
- It only READS historical candles from MetaTrader 5.

HOW LOOK-AHEAD BIAS IS AVOIDED (very important):
- The newest candle returned by MT5 is still FORMING, so we drop it.
- A signal on bar "i" uses indicator values computed from bars 0..i only
  (all closed candles).
- The virtual trade is entered at the OPEN of bar i+1 - the first moment a
  real trader could have acted on the bar-i signal.

DISCLAIMER: This is a historical simulation. Past results do NOT predict
future performance, and this strategy is NOT proven to be profitable.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read historical candles (read-only usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # np.select() to build the signal column

# ============================================================
# SECTION 2: Settings - all the "knobs" in one place
# ============================================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 2000   # History to download (>= 1000 required by the task)
MIN_CANDLES = 1100   # Minimum we accept: ~50 warm-up + 1000 testable bars

# --- Virtual account settings ---
STARTING_BALANCE = 10_000.0  # Virtual starting balance in USD
LOTS = 0.10                  # Fixed position size: 0.10 lots = 10,000 EUR
CONTRACT_SIZE = 100_000      # 1.00 standard lot = 100,000 units of base currency
SPREAD = 0.0002              # Assumed cost: 2.0 pips round-trip (conservative;
                             # we saw 1.6-1.9 pips in the live spread column).

# --- Exit rules (deterministic, based on ATR at the signal bar) ---
SL_ATR_MULT = 1.5   # Stop loss distance   = 1.5 x ATR
TP_ATR_MULT = 3.0   # Take profit distance = 3.0 x ATR  (1:2 reward-to-risk)

# --- Indicator periods ---
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
WARMUP = 50  # Bars needed before EMA50 (the slowest indicator) is valid

# ============================================================
# SECTION 3: Indicator functions
# (Same math as technical_analysis.py. We copy them here instead of
# importing, because technical_analysis.py opens its own MT5 connection
# at import time - importing it here would connect twice.)
# ============================================================
def add_ema(df, period):
    """Exponential Moving Average of the close price."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def add_rsi(df, period=14):
    """Relative Strength Index (Wilder's classic calculation)."""
    delta = df["close"].diff()          # Bar-to-bar change in close
    gain = delta.clip(lower=0)          # Keep gains only
    loss = -delta.clip(upper=0)         # Losses as positive numbers

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

    # --- Safety check 1: did we get any data? ---
    if rates is None or len(rates) == 0:
        print(f"ERROR: Could not retrieve {SYMBOL} data.")
        print("Last error:", mt5.last_error())
        quit()

    if len(rates) < MIN_CANDLES:
        print(f"ERROR: Not enough history. Got {len(rates)} candles, "
              f"but at least {MIN_CANDLES} are needed for a meaningful backtest.")
        quit()

    if len(rates) < NUM_CANDLES:
        print(f"NOTE: Broker returned {len(rates)} candles (requested {NUM_CANDLES}). Continuing...")

    # ============================================================
    # SECTION 5: Build the DataFrame and DROP the forming candle
    # ============================================================
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # The LAST row is the candle that is still forming right now.
    # Using it would be look-ahead bias, so we remove it.
    df = df.iloc[:-1].reset_index(drop=True)

    # ============================================================
    # SECTION 6: Calculate indicators and the signal column
    # ============================================================
    df = add_ema(df, EMA_FAST)   # -> "EMA20"
    df = add_ema(df, EMA_SLOW)   # -> "EMA50"
    df = add_rsi(df, RSI_PERIOD) # -> "RSI"
    df = add_atr(df, ATR_PERIOD) # -> "ATR"

    # Same educational strategy as technical_analysis.py:
    # BUY  when EMA20 > EMA50 and RSI is between 50 and 70
    # SELL when EMA20 < EMA50 and RSI is between 30 and 50
    # Otherwise HOLD
    buy_condition = (df["EMA20"] > df["EMA50"]) & (df["RSI"] >= 50) & (df["RSI"] <= 70)
    sell_condition = (df["EMA20"] < df["EMA50"]) & (df["RSI"] >= 30) & (df["RSI"] <= 50)

    df["signal"] = np.select([buy_condition, sell_condition], ["BUY", "SELL"], default="HOLD")

    # ============================================================
    # SECTION 7: The backtest loop - one candle at a time
    #
    # Rules used below (all deterministic, all using only closed data):
    # 1. If flat, check the signal on bar i (a CLOSED candle).
    # 2. If BUY/SELL, enter virtually at the OPEN of bar i+1.
    # 3. Stop loss / take profit are fixed at entry using ATR from bar i.
    # 4. On every bar j >= entry bar, check exits against that bar's
    #    high/low. If BOTH SL and TP fall inside one bar, we assume the
    #    STOP LOSS was hit first (the conservative assumption, because a
    #    single H1 candle cannot tell us which came first).
    # 5. Fills are assumed to happen exactly at the SL/TP price
    #    (no slippage) and the spread is charged as a flat cost.
    # 6. A trade still open at the end of the data is force-closed at the
    #    last close price.
    # ============================================================
    trades = []                    # Every virtual trade is recorded here
    balance = STARTING_BALANCE     # Virtual running balance
    spread_cost = SPREAD * CONTRACT_SIZE * LOTS  # Flat $ cost per trade

    i = WARMUP  # Start after the indicator warm-up period
    while i < len(df) - 1:         # Need bar i+1 to exist for the entry
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        # Only trade on a valid signal with a valid ATR value
        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1                       # First bar after the signal
            entry_price = df["open"].iloc[entry_bar]  # Enter at the open

            # Fix SL and TP at entry, based on ATR known at signal time
            if signal == "BUY":
                stop_loss = entry_price - SL_ATR_MULT * atr
                take_profit = entry_price + TP_ATR_MULT * atr
            else:  # SELL
                stop_loss = entry_price + SL_ATR_MULT * atr
                take_profit = entry_price - TP_ATR_MULT * atr

            # Scan forward, bar by bar, for the exit
            exit_index = None
            exit_price = None
            exit_reason = None
            for j in range(entry_bar, len(df)):
                bar_high = df["high"].iloc[j]
                bar_low = df["low"].iloc[j]

                if signal == "BUY":
                    # Check STOP LOSS first (conservative)
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

            # If we reached the end of the data, force-close the trade
            if exit_index is None:
                exit_index = len(df) - 1
                exit_price = df["close"].iloc[exit_index]
                exit_reason = "END_OF_DATA"

            # --- Virtual profit/loss for this trade ---
            # For EURUSD the quote currency is USD, so the P&L below is
            # already in USD and needs no conversion.
            if signal == "BUY":
                gross_pnl = (exit_price - entry_price) * CONTRACT_SIZE * LOTS
            else:  # SELL profits when the price goes DOWN
                gross_pnl = (entry_price - exit_price) * CONTRACT_SIZE * LOTS

            net_pnl = gross_pnl - spread_cost  # Pay the assumed spread cost
            balance += net_pnl

            # Record every virtual trade
            trades.append(
                {
                    "trade": len(trades) + 1,
                    "direction": signal,
                    "entry_time": df["time"].iloc[entry_bar],
                    "entry_price": entry_price,
                    "exit_time": df["time"].iloc[exit_index],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": net_pnl,
                    "balance_after": balance,
                }
            )

            # Skip ahead: the next signal search starts on the bar AFTER
            # the exit bar (keeps the loop simple; slightly conservative).
            i = exit_index

        i += 1

    # ============================================================
    # SECTION 8: Performance statistics
    # ============================================================
    trades_df = pd.DataFrame(trades)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (VIRTUAL MONEY - HISTORICAL SIMULATION)")
    print("=" * 60)
    print(f"Symbol / timeframe : {SYMBOL} H1")
    print(f"Period tested      : {df['time'].iloc[WARMUP]}  ->  {df['time'].iloc[-1]}")
    print(f"Assumptions        : {LOTS} lots, spread {SPREAD} "
          f"(${spread_cost:.2f}/trade), SL={SL_ATR_MULT}xATR, TP={TP_ATR_MULT}xATR")

    if len(trades_df) == 0:
        print("\nNo virtual trades were generated in this period.")
        print("(The market never met the BUY/SELL conditions.)")
    else:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        breakeven = len(trades_df) - len(wins) - len(losses)

        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        win_rate = len(wins) / len(trades_df) * 100
        total_return = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
        avg_trade = trades_df["pnl"].mean()

        # --- Maximum drawdown: biggest peak-to-trough drop of the balance ---
        balance_history = [STARTING_BALANCE] + trades_df["balance_after"].tolist()
        peak = balance_history[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        for b in balance_history[1:]:
            if b > peak:
                peak = b              # New highest balance
            drawdown = peak - b       # How far we are below the peak
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown / peak * 100

        print(f"\nStarting balance   : ${STARTING_BALANCE:,.2f}")
        print(f"Ending balance     : ${balance:,.2f}")
        print(f"Total return       : {total_return:+.2f}%")
        print(f"\nNumber of trades   : {len(trades_df)}")
        print(f"Winning trades     : {len(wins)}")
        print(f"Losing trades      : {len(losses)}")
        if breakeven:
            print(f"Breakeven trades   : {breakeven}")
        print(f"Win rate           : {win_rate:.1f}%")
        print(f"\nGross profit       : ${gross_profit:,.2f}")
        print(f"Gross loss         : ${gross_loss:,.2f}")
        if gross_loss > 0:
            print(f"Profit factor      : {gross_profit / gross_loss:.2f}")
        else:
            print("Profit factor      : infinite (no losing trades)")
        print(f"Average trade      : ${avg_trade:,.2f}")
        print(f"\nMaximum drawdown   : ${max_dd:,.2f}  ({max_dd_pct:.2f}%)")

        # --- Show the last 10 virtual trades ---
        print("\nLast 10 virtual trades:")
        display = trades_df.tail(10).copy()
        for col in ("entry_price", "exit_price"):
            display[col] = display[col].round(5)
        for col in ("pnl", "balance_after"):
            display[col] = display[col].round(2)
        print(display.to_string(index=False))

    # ============================================================
    # SECTION 9: Disclaimer
    # ============================================================
    print("\nDISCLAIMER: This is a HISTORICAL BACKTEST with virtual money.")
    print("Past results do NOT predict future performance, and this")
    print("strategy is NOT proven to be profitable. Educational use only.")

finally:
    # ============================================================
    # SECTION 10: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
