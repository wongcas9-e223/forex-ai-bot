"""
bb_oos_validation.py - OUT-OF-SAMPLE VALIDATION of the Bollinger Band
mean-reversion strategy (baseline and ADX14 < 25 regime-filter version).

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: this script contains NO order-sending calls at all.
- The script only READS historical candles from MetaTrader 5.
- Never connects to a live trading account.

THE QUESTION BEING ASKED:
The ADX<25 regime-filtered Bollinger strategy was defined in a previous
experiment. Does it behave SIMILARLY on data it has never influenced?
We validate, we do NOT optimize: every parameter below was fixed in
advance and is copied from the earlier experiments unchanged.

THE STRATEGY UNDER TEST (identical in both variants):
  ENTRY (Bollinger Bands, period 20, 2.0 std dev):
    LONG : a closed candle closes BELOW the lower band
    SHORT: a closed candle closes ABOVE the upper band
    Variant B additionally requires ADX14 < 25 at the signal candle.
    Variant A (baseline) has no extra filter.
  EXITS (ATR14, fixed at entry from the signal candle):
    LONG : SL = entry - 2.0 x ATR, TP = entry + 1.5 x ATR
    SHORT: SL = entry + 2.0 x ATR, TP = entry - 1.5 x ATR

DATA SPLIT (chronological, NOT random):
  The closed candles are split in the middle by time:
    DEVELOPMENT (in-sample)  = first 50% of the candles
    OUT-OF-SAMPLE            = final 50% of the candles
  If the candle count is odd, the extra candle falls in OOS.
  Indicators are computed ONCE over the full closed series. This is safe:
  every indicator value at bar i uses only bars <= i (purely causal), so
  the OOS half only ever sees information from its own past.

  Start of scanning per period:
    Development: bar 60 (same warm-up convention as ALL previous
                 experiments - early bars have unstable indicator values).
    OOS        : bar 0 of the OOS slice. This is realistic, not lenient:
                 a trader at the OOS boundary already had the whole
                 development history behind them, so indicators there are
                 fully warmed up. Nothing about OOS bar 0 uses future data.

BACKTEST RULES (identical to the previous experiments):
  - Entry at the NEXT candle's OPEN after the signal candle closes.
  - Fixed 0.10 lots, starting virtual balance $10,000 PER PERIOD
    (the two periods are independent runs; balances are NOT chained).
  - Transaction cost $2.00 per completed trade (2.0 pip spread at 0.1 lot).
  - One position at a time.
  - If SL and TP are both touched inside the same candle, the STOP LOSS
    is assumed to fill first (conservative).
  - Trades still open at the end of a period are force-closed at the last
    close (exit reason END_OF_DATA).

HOW PROFITABILITY IS JUDGED (fixed rule, NOT a fitted threshold):
  PROFITABLE requires ALL of: total return > 0, profit factor > 1.00,
  AND average trade > 0. A high win rate alone does NOT count - win rate
  says nothing about the size of losses or transaction costs.

DISCLAIMER: This is a historical backtest with virtual money.
Out-of-sample results do not guarantee future performance.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read historical candles (read-only usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # np.select() to build the signal column

# ============================================================
# SECTION 2: Settings - fixed in advance, NOT optimized
# ============================================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 4000   # Requested history (task: at least 4000 if available)
MIN_CANDLES = 2000   # Absolute minimum (~1000 candles per half)

# --- Virtual account settings (same as previous experiments) ---
STARTING_BALANCE = 10_000.0
LOTS = 0.10                  # Fixed position size
CONTRACT_SIZE = 100_000      # 1.00 lot = 100,000 units of base currency
SPREAD = 0.0002              # 2.0 pips round-trip cost = $2.00 per trade

# --- Exit distances (ATR multiples), same as previous experiments ---
SL_ATR_MULT = 2.0   # Stop loss distance
TP_ATR_MULT = 1.5   # Take profit distance

# --- Indicator settings (copied unchanged from previous experiments) ---
BB_PERIOD = 20      # Bollinger moving average period
BB_NUM_STD = 2      # Bollinger standard deviation width
ADX_PERIOD = 14     # Variant B filter period
ADX_THRESHOLD = 25.0  # Variant B filter threshold - the ONLY one tested
ATR_PERIOD = 14
WARMUP = 60         # Development-period start (previous convention)

SPLIT_RATIO = 0.5   # First 50% development, final 50% out-of-sample


# ============================================================
# SECTION 3: Indicator functions (same math as the earlier scripts)
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


def add_adx(df, period=14):
    """Average Directional Index (Wilder's classic calculation).

    ADX measures trend STRENGTH, not direction:
    - rising/high ADX  = strong trend (either direction)
    - falling/low ADX  = weak trend / range conditions

    Steps (Wilder):
    1. +DM / -DM from consecutive highs/lows
    2. True Range (same as ATR)
    3. Wilder smoothing (EWM with alpha=1/period) -> +DI, -DI
    4. DX = 100 * |+DI - -DI| / (+DI + -DI)
    5. ADX = Wilder-smoothed DX
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1 / period
    atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return df


# ============================================================
# SECTION 4: The backtest engine
# Identical to the previous experiments, with two additions:
#   - a switch for the ADX<25 entry filter (on/off)
#   - a configurable start bar (dev period starts at WARMUP,
#     the OOS period starts at 0 - see the docstring for why)
# ============================================================
def run_backtest(df, use_adx_filter, start_bar):
    """Run the virtual backtest on df. Returns (trades DataFrame, final balance)."""

    # --- Raw Bollinger conditions (identical for both variants) ---
    long_condition = df["close"] < df["BB_LOWER"]    # Stretched below the band
    short_condition = df["close"] > df["BB_UPPER"]   # Stretched above the band

    # --- Variant B additionally requires the ADX14 < 25 regime ---
    if use_adx_filter:
        long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
        short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE
    spread_cost = SPREAD * CONTRACT_SIZE * LOTS  # Flat $2.00 per trade

    i = start_bar
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

            # --- Forward scan: candle by candle, looking for the exit ---
            # STOP is checked FIRST: if both levels share one candle,
            # we assume the stop filled (conservative).
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

            # Data ran out while the trade was open -> force-close
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
                    "hold_candles": exit_index - entry_bar + 1,
                    "pnl": round(net_pnl, 2),
                    "balance_after": round(balance, 2),
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 5: Statistics printer (shared by all four runs)
# ============================================================
def show_results(label, trades_df, final_balance):
    """Print the full statistics block for one run. Returns a metrics dict."""
    print("\n" + "=" * 64)
    print(f"{label}  (HISTORICAL BACKTEST - VIRTUAL MONEY ONLY)")
    print("=" * 64)
    print(f"Rules: BB(20,2) entry, SL={SL_ATR_MULT}xATR / TP={TP_ATR_MULT}xATR, "
          f"{LOTS} lots, cost ${SPREAD * CONTRACT_SIZE * LOTS:.2f}/trade")

    if len(trades_df) == 0:
        print("\nNo virtual trades were generated in this period.")
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

    tp_count = int((trades_df["exit_reason"] == "TAKE_PROFIT").sum())
    sl_count = int((trades_df["exit_reason"] == "STOP_LOSS").sum())
    eod_count = int((trades_df["exit_reason"] == "END_OF_DATA").sum())

    print(f"\nStarting balance    : ${STARTING_BALANCE:,.2f}")
    print(f"Ending balance      : ${final_balance:,.2f}")
    print(f"Total return        : {total_return:+.2f}%")
    print(f"\nNumber of trades    : {len(trades_df)}")
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
    print(f"Average holding time: {trades_df['hold_candles'].mean():.1f} candles")
    print(f"\nTake-profit exits   : {tp_count}")
    print(f"Stop-loss exits     : {sl_count}")
    if eod_count:
        print(f"END_OF_DATA exits   : {eod_count} (trade still open at period end)")

    # --- Show the last 5 virtual trades for sanity checking ---
    print("\nLast 5 virtual trades:")
    print(trades_df.tail(5).to_string(index=False))

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "end_balance": final_balance,
        "return": total_return,
        "trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": pf,
        "avg_trade": trades_df["pnl"].mean(),
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "avg_hold": trades_df["hold_candles"].mean(),
        "tp": tp_count,
        "sl": sl_count,
        "eod": eod_count,
    }


def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def profitability_verdict(metrics):
    """Fixed, pre-declared assessment rule (NOT a fitted threshold).

    PROFITABLE requires total return > 0 AND profit factor > 1.00 AND
    average trade > 0 together. Win rate alone is deliberately ignored.
    """
    if metrics is None:
        return "NO TRADES"
    if metrics["return"] > 0 and metrics["profit_factor"] > 1.0 and metrics["avg_trade"] > 0:
        return "PROFITABLE"
    if metrics["return"] == 0:
        return "BREAKEVEN"
    return "NOT PROFITABLE"


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
              f"but at least {MIN_CANDLES} are needed for a meaningful 50/50 split.")
        quit()

    if len(rates) < NUM_CANDLES:
        print(f"NOTE: Broker returned {len(rates)} candles (requested {NUM_CANDLES}). Continuing...")

    # --- Build the DataFrame; drop the still-forming last candle ---
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)   # closed candles only
    print(f"Closed candles used: {len(df)} "
          f"(from {df['time'].iloc[0]} to {df['time'].iloc[-1]})")

    # --- Calculate indicators ONCE over the full closed series ---
    # Safe / no look-ahead: every indicator value at bar i uses only
    # bars <= i, so slicing afterwards changes nothing historically.
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)

    # ============================================================
    # SECTION 7: Chronological 50/50 split (NOT random)
    # ============================================================
    split_index = len(df) // 2
    dev_df = df.iloc[:split_index].reset_index(drop=True)   # first half
    oos_df = df.iloc[split_index:].reset_index(drop=True)   # final half

    print("\n" + "=" * 64)
    print("CHRONOLOGICAL 50/50 DATA SPLIT")
    print("=" * 64)
    print(f"DEVELOPMENT (in-sample): {len(dev_df)} candles, "
          f"{dev_df['time'].iloc[0]} -> {dev_df['time'].iloc[-1]}")
    print(f"OUT-OF-SAMPLE          : {len(oos_df)} candles, "
          f"{oos_df['time'].iloc[0]} -> {oos_df['time'].iloc[-1]}")
    print("Each period runs as an INDEPENDENT virtual backtest starting")
    print("from $10,000 (balances are NOT chained across the split).")
    print(f"Development scanning starts at bar {WARMUP} (the warm-up")
    print("convention of all previous experiments); OOS scanning starts")
    print("at bar 0 because its indicators are already warmed by history.")

    # ============================================================
    # SECTION 8: Run all FOUR backtests
    #   A-dev, A-oos  : baseline (no regime filter)
    #   B-dev, B-oos  : ADX14 < 25 regime filter
    # ============================================================
    runs = {}
    for variant_key, use_filter, variant_label in (
        ("A", False, "A) BASELINE - NO REGIME FILTER"),
        ("B", True, "B) ADX14 < 25 REGIME FILTER"),
    ):
        for period_key, period_label, slice_df, start_bar in (
            ("dev", "DEVELOPMENT (in-sample)", dev_df, WARMUP),
            ("oos", "OUT-OF-SAMPLE (unseen data)", oos_df, 0),
        ):
            print(f"\nRunning variant {variant_key} on {period_key} data...")
            trades_df, final_balance = run_backtest(slice_df.copy(), use_filter, start_bar)
            runs[(variant_key, period_key)] = show_results(
                f"{variant_label} - {period_label}", trades_df, final_balance
            )

    # ============================================================
    # SECTION 9: Development vs out-of-sample comparison
    # ============================================================
    a_dev, a_oos = runs[("A", "dev")], runs[("A", "oos")]
    b_dev, b_oos = runs[("B", "dev")], runs[("B", "oos")]

    if all(m is not None for m in (a_dev, a_oos, b_dev, b_oos)):
        print("\n" + "=" * 64)
        print("1) DEVELOPMENT vs OUT-OF-SAMPLE COMPARISON")
        print("=" * 64)
        print(f"\n{'Metric':<22}{'A: Baseline':>22}{'B: ADX14<25':>22}")
        print(f"{'':<22}{'Dev':>11}{'OOS':>11}{'Dev':>11}{'OOS':>11}")
        print("-" * 78)
        print(f"{'Total return':<22}"
              f"{a_dev['return']:>10.2f}%{a_oos['return']:>10.2f}%"
              f"{b_dev['return']:>10.2f}%{b_oos['return']:>10.2f}%")
        print(f"{'Trades':<22}"
              f"{a_dev['trades']:>11}{a_oos['trades']:>11}"
              f"{b_dev['trades']:>11}{b_oos['trades']:>11}")
        print(f"{'Win rate':<22}"
              f"{a_dev['win_rate']:>10.1f}%{a_oos['win_rate']:>10.1f}%"
              f"{b_dev['win_rate']:>10.1f}%{b_oos['win_rate']:>10.1f}%")
        print(f"{'Profit factor':<22}"
              f"{fmt_pf(a_dev['profit_factor']):>11}{fmt_pf(a_oos['profit_factor']):>11}"
              f"{fmt_pf(b_dev['profit_factor']):>11}{fmt_pf(b_oos['profit_factor']):>11}")
        print(f"{'Max drawdown':<22}"
              f"{a_dev['max_dd_pct']:>10.2f}%{a_oos['max_dd_pct']:>10.2f}%"
              f"{b_dev['max_dd_pct']:>10.2f}%{b_oos['max_dd_pct']:>10.2f}%")
        print(f"{'Average trade':<22}"
              f"{a_dev['avg_trade']:>10.2f}${a_oos['avg_trade']:>10.2f}$"
              f"{b_dev['avg_trade']:>10.2f}${b_oos['avg_trade']:>10.2f}$")
        print(f"{'Avg hold (candles)':<22}"
              f"{a_dev['avg_hold']:>11.1f}{a_oos['avg_hold']:>11.1f}"
              f"{b_dev['avg_hold']:>11.1f}{b_oos['avg_hold']:>11.1f}")
        print(f"{'Take-profit count':<22}"
              f"{a_dev['tp']:>11}{a_oos['tp']:>11}"
              f"{b_dev['tp']:>11}{b_oos['tp']:>11}")
        print(f"{'Stop-loss count':<22}"
              f"{a_dev['sl']:>11}{a_oos['sl']:>11}"
              f"{b_dev['sl']:>11}{b_oos['sl']:>11}")

        # --- 2) and 3): stability differences (OOS minus Development) ---
        print("\n" + "=" * 64)
        print("2) STABILITY DIFFERENCES (out-of-sample MINUS development)")
        print("=" * 64)
        for name, dev_m, oos_m in (
            ("A (baseline)", a_dev, a_oos),
            ("B (ADX14 < 25)", b_dev, b_oos),
        ):
            if dev_m["profit_factor"] == float("inf") or oos_m["profit_factor"] == float("inf"):
                pf_diff_txt = "n/a (a period had zero losing trades)"
            else:
                pf_diff_txt = f"{oos_m['profit_factor'] - dev_m['profit_factor']:+.2f}"
            wr_diff = oos_m["win_rate"] - dev_m["win_rate"]
            print(f"  Variant {name}:")
            print(f"    Profit factor difference : {pf_diff_txt} "
                  f"({fmt_pf(dev_m['profit_factor'])} -> {fmt_pf(oos_m['profit_factor'])})")
            print(f"    Win-rate difference      : {wr_diff:+.1f} percentage points "
                  f"({dev_m['win_rate']:.1f}% -> {oos_m['win_rate']:.1f}%)")
            print(f"    Trade count              : {dev_m['trades']} -> {oos_m['trades']}")

        # --- 5) and 6): ADX<25 regime context per period ---------------
        print("\n" + "=" * 64)
        print("3) REGIME CONTEXT: ADX14 < 25 candles in each period")
        print("=" * 64)
        for period_name, slice_df in (("Development", dev_df), ("Out-of-sample", oos_df)):
            valid_adx = slice_df["ADX"].dropna()
            below = int((valid_adx < ADX_THRESHOLD).sum())
            pct = below / len(valid_adx) * 100 if len(valid_adx) else 0.0
            print(f"  {period_name:<14}: {below} of {len(valid_adx)} valid-ADX candles "
                  f"were ADX<25 ({pct:.1f}%)")

        # --- 7) and 8): profitability assessment -----------------------
        print("\n" + "=" * 64)
        print("4) DID EACH STRATEGY REMAIN PROFITABLE OUT-OF-SAMPLE?")
        print("=" * 64)
        print("  Rule (fixed in advance): PROFITABLE requires total return")
        print("  > 0 AND profit factor > 1.00 AND average trade > 0 together.")
        print("  A high win rate alone does NOT count as profitability.")
        for name, dev_m, oos_m in (
            ("A (baseline)", a_dev, a_oos),
            ("B (ADX14 < 25)", b_dev, b_oos),
        ):
            print(f"\n  Variant {name}:")
            print(f"    Development : {profitability_verdict(dev_m)} "
                  f"(return {dev_m['return']:+.2f}%, PF {fmt_pf(dev_m['profit_factor'])}, "
                  f"avg ${dev_m['avg_trade']:+.2f}, max DD {dev_m['max_dd_pct']:.2f}%)")
            print(f"    Out-of-sample: {profitability_verdict(oos_m)} "
                  f"(return {oos_m['return']:+.2f}%, PF {fmt_pf(oos_m['profit_factor'])}, "
                  f"avg ${oos_m['avg_trade']:+.2f}, max DD {oos_m['max_dd_pct']:.2f}%)")

        # --- Neutral closing observation --------------------------------
        print("\nHow to read this validation:")
        print("- Similar PF / win rate / avg trade in BOTH periods suggests")
        print("  the strategy's behavior is stable across time.")
        print("- A large drop from development to OOS is a classic warning")
        print("  sign that the earlier result may not generalize.")
        print("- Either way, one 50/50 split of one instrument is still a")
        print("  SINGLE observation - not proof of anything.")

    else:
        print("\nNOTE: at least one period produced no trades; "
              "comparison skipped.")

    # ============================================================
    # SECTION 10: Disclaimer (required wording)
    # ============================================================
    print("\nDISCLAIMER: This is a historical backtest with virtual money.")
    print("Out-of-sample results do not guarantee future performance.")
    print("No parameters were optimized: BB(20,2), ATR exits 2.0/1.5,")
    print("ADX14 < 25, and the 50/50 split were all fixed in advance.")
    print("Educational use only.")

finally:
    # ============================================================
    # SECTION 11: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
