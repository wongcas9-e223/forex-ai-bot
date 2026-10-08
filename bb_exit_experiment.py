"""
bb_exit_experiment.py - Test different EXIT rules for the same Bollinger entry.

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: no mt5.order_send() anywhere, no live account.
- The script only READS historical candles from MetaTrader 5.

THE QUESTION BEING ASKED:
The Bollinger mean-reversion ENTRY (close outside the band) roughly broke
even with a fixed 1.5 x ATR take profit. But the hypothesis is really about
price returning to the MIDDLE of the band. So we keep the entry identical
and only change the EXIT:

  VARIANT A - fixed ATR exits (same as the previous experiment)
      LONG : SL = entry - 2.0 x ATR, TP = entry + 1.5 x ATR
      SHORT: SL = entry + 2.0 x ATR, TP = entry - 1.5 x ATR

  VARIANT B - LIVE middle band target
      Exit when price touches the Bollinger middle band. The band is
      MOVING: each bar we compare against the previous CLOSED candle's
      band value (BB_MIDDLE[j-1]), which is fully known when bar j opens
      (no look-ahead). Emergency stop = 2.0 x ATR (fixed at entry).
      No fixed take profit.

  VARIANT C - FROZEN middle band target
      Target = the middle band value at the SIGNAL bar (the last closed
      candle when the trade opens), frozen for the whole trade.
      Stop = 2.0 x ATR (fixed at entry).

ENTRY RULES - IDENTICAL FOR ALL VARIANTS:
  LONG : candle CLOSES below the lower Bollinger Band (20, 2.0)
  SHORT: candle CLOSES above the upper Bollinger Band
  Entry happens at the NEXT candle's open (first moment a real trader
  could act). RSI is deliberately NOT used.

CONSERVATIVE ASSUMPTIONS (documented, used in all variants):
  1. If BOTH the stop and the target are touched inside the SAME candle,
     the STOP is assumed to fill first. One H1 candle cannot tell us the
     intra-bar order, so we take the pessimistic outcome.
  2. Gap safety (variants B/C): if the entry open is ALREADY at/beyond
     the band target, the trade exits immediately at that open - exactly
     what a resting limit order at the band would do. This handles the
     "impossible direction" case without any look-ahead.
  3. Fills happen exactly at the stop/target/band level (no slippage);
     a flat spread cost is charged per trade instead.

DISCLAIMER: This is a HISTORICAL backtest with virtual money.
Past results do NOT predict future performance and nothing here is
proven to be profitable. No parameters were searched or optimized.
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
NUM_CANDLES = 2000   # Same history size as the previous experiments
MIN_CANDLES = 1100   # Absolute minimum we accept

# --- Virtual account settings (same as previous experiments) ---
STARTING_BALANCE = 10_000.0
LOTS = 0.10                  # Fixed position size
CONTRACT_SIZE = 100_000      # 1.00 lot = 100,000 units of base currency
SPREAD = 0.0002              # 2.0 pips round-trip cost = $2.00 per trade

# --- Exit distances (ATR multiples) ---
SL_ATR_MULT = 2.0   # Stop loss distance (used by ALL variants)
TP_ATR_MULT = 1.5   # Take profit distance (used by VARIANT A only)

# --- Bollinger Bands settings ---
BB_PERIOD = 20
BB_NUM_STD = 2
ATR_PERIOD = 14
WARMUP = 60         # Bars needed before all indicators are valid

# ============================================================
# SECTION 3: Indicator functions (BB + ATR only - no RSI this time)
# ============================================================
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
    """
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


# ============================================================
# SECTION 4: The backtest engine
# One function, three exit variants. The ENTRY logic is identical;
# only the exit rules inside the forward scan differ.
# ============================================================
def run_backtest(df, variant):
    """Run the virtual backtest. variant is "A", "B" or "C".
    Returns (trades DataFrame, final balance)."""

    # --- Entry signals: IDENTICAL for every variant ---
    long_condition = df["close"] < df["BB_LOWER"]   # Stretched below the band
    short_condition = df["close"] > df["BB_UPPER"]  # Stretched above the band
    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE
    spread_cost = SPREAD * CONTRACT_SIZE * LOTS  # Flat $2.00 per trade

    i = WARMUP
    while i < len(df) - 1:          # Need bar i+1 to exist for the entry
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        # Only enter on a valid signal with a valid ATR value
        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1                         # Next candle
            entry_price = df["open"].iloc[entry_bar]  # Enter at its open

            # The middle band of the last CLOSED candle (bar i). This is the
            # freshest band value a real trader could know at entry time.
            band_at_entry = df["BB_MIDDLE"].iloc[i]

            # Stop loss: identical for all variants, fixed at entry
            if signal == "BUY":
                stop_loss = entry_price - SL_ATR_MULT * atr
            else:
                stop_loss = entry_price + SL_ATR_MULT * atr

            exit_index = None
            exit_price = None
            exit_reason = None

            # --- Gap safety for B and C -------------------------------
            # If the entry open is ALREADY at/beyond the band target, a
            # resting limit order at the band would fill immediately.
            # Exit at the open - safe, realistic, no look-ahead.
            if variant in ("B", "C"):
                if signal == "BUY" and entry_price >= band_at_entry:
                    exit_index = entry_bar
                    exit_price = entry_price
                    exit_reason = "TARGET_AT_ENTRY"
                elif signal == "SELL" and entry_price <= band_at_entry:
                    exit_index = entry_bar
                    exit_price = entry_price
                    exit_reason = "TARGET_AT_ENTRY"

            # --- Forward scan: candle by candle, looking for the exit ---
            if exit_index is None:
                for j in range(entry_bar, len(df)):
                    bar_high = df["high"].iloc[j]
                    bar_low = df["low"].iloc[j]

                    # The STOP is always checked FIRST (conservative:
                    # if stop and target share one candle, stop fills first).
                    stop_hit = bar_low <= stop_loss if signal == "BUY" else bar_high >= stop_loss
                    if stop_hit:
                        exit_index, exit_price, exit_reason = j, stop_loss, "STOP_LOSS"
                        break

                    # --- The target check differs per variant ---
                    if variant == "A":
                        # Fixed take profit at 1.5 x ATR
                        target_hit = bar_high >= entry_price + TP_ATR_MULT * atr \
                            if signal == "BUY" else bar_low <= entry_price - TP_ATR_MULT * atr
                        if target_hit:
                            target = entry_price + TP_ATR_MULT * atr if signal == "BUY" \
                                else entry_price - TP_ATR_MULT * atr
                            exit_index, exit_price, exit_reason = j, target, "TAKE_PROFIT"
                            break

                    elif variant == "B":
                        # LIVE middle band: compare against the PREVIOUS
                        # closed candle's band value (known when bar j opens).
                        band_level = df["BB_MIDDLE"].iloc[j - 1]
                        if signal == "BUY" and bar_high >= band_level:
                            exit_index, exit_price, exit_reason = j, band_level, "BAND_TARGET"
                            break
                        if signal == "SELL" and bar_low <= band_level:
                            exit_index, exit_price, exit_reason = j, band_level, "BAND_TARGET"
                            break

                    else:  # variant == "C"
                        # FROZEN middle band snapshot from the signal bar
                        if signal == "BUY" and bar_high >= band_at_entry:
                            exit_index, exit_price, exit_reason = j, band_at_entry, "BAND_TARGET"
                            break
                        if signal == "SELL" and bar_low <= band_at_entry:
                            exit_index, exit_price, exit_reason = j, band_at_entry, "BAND_TARGET"
                            break

                # Data ran out while the trade was open -> force-close
                if exit_index is None:
                    exit_index = len(df) - 1
                    exit_price = df["close"].iloc[exit_index]
                    exit_reason = "END_OF_DATA"

            # --- Measurement: did price touch the (moving) middle band
            #     at any point during the trade? (Used for questions 3/4.)
            #     At the stop bar we skip the check: stop-first means the
            #     trade never legitimately "reached" the band there.
            reached_band = False
            for j in range(entry_bar, exit_index + 1):
                if j == exit_index and exit_reason == "STOP_LOSS":
                    continue
                band_level = df["BB_MIDDLE"].iloc[j - 1]
                if signal == "BUY" and df["high"].iloc[j] >= band_level:
                    reached_band = True
                    break
                if signal == "SELL" and df["low"].iloc[j] <= band_level:
                    reached_band = True
                    break

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
                    "hold_candles": exit_index - entry_bar + 1,
                    "reached_band": reached_band,
                    "pnl": round(net_pnl, 2),
                    "balance_after": round(balance, 2),
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 5: Statistics printer (shared by all variants)
# ============================================================
def show_results(label, trades_df, final_balance):
    """Print all statistics for one variant. Returns a metrics dict."""
    print("\n" + "=" * 64)
    print(f"{label}  (HISTORICAL BACKTEST - VIRTUAL MONEY ONLY)")
    print("=" * 64)
    print("Assumptions: stop-first when stop & target share a candle,")
    print(f"{LOTS} lots, spread cost ${SPREAD * CONTRACT_SIZE * LOTS:.2f}/trade, "
          f"SL = {SL_ATR_MULT} x ATR at entry")

    if len(trades_df) == 0:
        print("\nNo virtual trades were generated in this variant.")
        return None

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    breakeven = len(trades_df) - len(wins) - len(losses)

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

    stopped_before_band = int(((trades_df["exit_reason"] == "STOP_LOSS")
                               & (~trades_df["reached_band"])).sum())

    print(f"\nStarting balance    : ${STARTING_BALANCE:,.2f}")
    print(f"Ending balance      : ${final_balance:,.2f}")
    print(f"Total return        : {total_return:+.2f}%")
    print(f"\nNumber of trades    : {len(trades_df)}")
    print(f"BUY trades          : {(trades_df['direction'] == 'BUY').sum()}")
    print(f"SELL trades         : {(trades_df['direction'] == 'SELL').sum()}")
    print(f"Winning trades      : {len(wins)}")
    print(f"Losing trades       : {len(losses)}")
    if breakeven:
        print(f"Breakeven trades    : {breakeven}")
    print(f"Win rate            : {win_rate:.1f}%")
    print(f"\nGross profit        : ${gross_profit:,.2f}")
    print(f"Gross loss          : ${gross_loss:,.2f}")
    if gross_loss > 0:
        print(f"Profit factor       : {gross_profit / gross_loss:.2f}")
    else:
        print("Profit factor       : infinite (no losing trades)")
    print(f"Average trade       : ${trades_df['pnl'].mean():,.2f}")
    print(f"\nMaximum drawdown    : ${max_dd:,.2f}  ({max_dd_pct:.2f}%)")

    # --- Exit reason breakdown + band statistics ---
    print("\nExit reasons:")
    for reason, count in trades_df["exit_reason"].value_counts().items():
        print(f"  {reason:<16}: {count}")
    print(f"\nTrades that touched the moving middle band : {trades_df['reached_band'].sum()}")
    print(f"Stopped WITHOUT ever touching the band     : {stopped_before_band}")
    print(f"Average holding time                       : "
          f"{trades_df['hold_candles'].mean():.1f} candles")

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
        "avg_hold": trades_df["hold_candles"].mean(),
        "reached_band": int(trades_df["reached_band"].sum()),
        "stopped_before_band": stopped_before_band,
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

    # --- Calculate the indicators (BB + ATR only) ---
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)

    # ============================================================
    # SECTION 7: Run all THREE variants on the SAME data
    # NOTE: the entry RULE is identical, but because exits happen at
    # different times, each variant becomes free for its next entry at a
    # different moment - so the entry sequences naturally diverge a bit.
    # That is part of what is being tested (faster exits = sooner ready).
    # ============================================================
    results = {}
    for variant, label in (("A", "A) FIXED ATR EXIT (SL 2.0x / TP 1.5x ATR)"),
                           ("B", "B) LIVE MIDDLE BAND EXIT (emergency SL 2.0x ATR)"),
                           ("C", "C) FROZEN MIDDLE BAND TARGET (SL 2.0x ATR)")):
        print(f"\nRunning variant {variant}...")
        trades_df, final_balance = run_backtest(df.copy(), variant)
        results[variant] = show_results(label, trades_df, final_balance)

    # ============================================================
    # SECTION 8: Comparison - answering the five questions
    # ============================================================
    valid = {v: r for v, r in results.items() if r is not None}
    if len(valid) == 3:
        print("\n" + "=" * 64)
        print("EXIT COMPARISON (same entry, different exits)")
        print("=" * 64)
        print(f"{'Metric':<22}{'A: ATR exit':>14}{'B: live band':>14}{'C: frozen band':>15}")
        print(f"{'Total return':<22}{valid['A']['return']:>13.2f}%"
              f"{valid['B']['return']:>13.2f}%{valid['C']['return']:>14.2f}%")
        print(f"{'Trades':<22}{valid['A']['trades']:>14}"
              f"{valid['B']['trades']:>14}{valid['C']['trades']:>15}")
        print(f"{'Win rate':<22}{valid['A']['win_rate']:>13.1f}%"
              f"{valid['B']['win_rate']:>13.1f}%{valid['C']['win_rate']:>14.1f}%")
        print(f"{'Profit factor':<22}{valid['A']['profit_factor']:>14.2f}"
              f"{valid['B']['profit_factor']:>14.2f}{valid['C']['profit_factor']:>15.2f}")
        print(f"{'Max drawdown':<22}{valid['A']['max_dd_pct']:>13.2f}%"
              f"{valid['B']['max_dd_pct']:>13.2f}%{valid['C']['max_dd_pct']:>14.2f}%")
        print(f"{'Avg trade':<22}{valid['A']['avg_trade']:>13.2f}$"
              f"{valid['B']['avg_trade']:>13.2f}${valid['C']['avg_trade']:>14.2f}$")

        # Question 1: which variant generated the MOST trades?
        q1 = max(valid, key=lambda v: valid[v]["trades"])
        print(f"\n1. Most trades          : variant {q1} ({valid[q1]['trades']} trades)")

        # Question 2: which generated the HIGHEST average trade?
        q2 = max(valid, key=lambda v: valid[v]["avg_trade"])
        print(f"2. Highest average trade: variant {q2} (${valid[q2]['avg_trade']:.2f})")

        # Questions 3 & 4: middle band statistics per variant
        print("3. Trades that touched the moving middle band:")
        for v in ("A", "B", "C"):
            print(f"     variant {v}: {valid[v]['reached_band']} of {valid[v]['trades']}")
        print("4. Trades stopped WITHOUT ever touching the band:")
        for v in ("A", "B", "C"):
            print(f"     variant {v}: {valid[v]['stopped_before_band']} of {valid[v]['trades']}")

        # Question 5: average holding time per variant
        print("5. Average holding time in candles:")
        for v in ("A", "B", "C"):
            print(f"     variant {v}: {valid[v]['avg_hold']:.1f}")

        # Neutral, factual summary of what the band-target experiment shows
        print("\nObservation: variants B and C exit at the Bollinger middle")
        print("band instead of a fixed ATR distance. Compare their win")
        print("rates, average trades and holding times against variant A")
        print("to see whether the mean-reversion edge is actually tied to")
        print("price returning to the middle of the band.")

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
