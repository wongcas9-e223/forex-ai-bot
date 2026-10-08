"""
gold_risk_analysis.py - WHY did the frozen GOLD strategy produce a
maximum drawdown of $4,879.02 (34.81%)?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only analysis.
- This script NEVER places, modifies, or closes any trade and contains
  no order execution, order checking, position, or account functions.
- It only reads candles and symbol specifications from MetaTrader 5.

PURPOSE (descriptive, NOT prescriptive):
The frozen strategy already produced (previous experiment):
  Return +2.12%, 113 trades, win rate 54.9%, PF 1.01,
  average trade +$1.87, max drawdown $4,879.02 (34.81%).
Nothing here tries to improve or optimize the strategy. The exact same
frozen backtest is reproduced first, and then the resulting trades and
equity curve are DESCRIBED:
  1. Equity curve / drawdown analysis (max/min equity, the largest
     drawdown periods with start/trough/recovery, durations).
  2. Losing streak analysis (longest streak + top 5 streaks, with the
     total loss and balances around each streak).

STRATEGY (FROZEN - identical to gold_validation.py, nothing changed):
  Symbol     : GOLD (XM MT5), H1
  Entry      : BB(20, 2.0) - LONG when a closed candle closes below the
               lower band, SHORT when above the upper band
  Regime     : only when ADX14 < 25 (Wilder smoothing, only threshold)
  Exit       : ATR14 fixed at entry
               LONG : SL = entry - 2.0 x ATR, TP = entry + 1.5 x ATR
               SHORT: SL = entry + 2.0 x ATR, TP = entry - 1.5 x ATR
  Execution  : signal on closed candle, enter at NEXT candle open,
               one position at a time, stop-first when SL and TP share
               a candle, no look-ahead, $2.00 cost per completed trade
  Balance    : $10,000 starting virtual balance
  Volume     : 0.10 lots requested, validated DOWN to the broker's
               volume_min / volume_step, never up
  P/L        : from GOLD's actual tick specifications
               (tick_value / tick_size = USD per price-unit per lot)

DISCLAIMER: This is a historical backtest with virtual money.
Results do not guarantee future performance. Nothing was optimized.
The analysis only DESCRIBES what already happened; it does not claim
the drawdown will or will not repeat. Educational use only.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read-only usage in this script
import pandas as pd        # DataFrame + indicator math
import numpy as np         # numeric helpers

# ============================================================
# SECTION 2: Settings - frozen, NOT optimized
# ============================================================
SYMBOL = "GOLD"       # The confirmed exact XM MT5 gold symbol
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 4000    # Requested history (same as gold_validation.py)
MIN_CANDLES = 2000    # Minimum usable closed candles required

STARTING_BALANCE = 10_000.0
LOTS_REQUESTED = 0.10         # Validated against broker rules below
COST_PER_TRADE = 2.0          # $2 fixed per completed trade

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.5

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0   # The ONLY threshold used
ATR_PERIOD = 14
WARMUP = 60            # Bars skipped before scanning (previous convention)

TOP_N_DRAWDOWNS = 5    # Top 5 drawdown periods to report
TOP_N_STREAKS = 5      # Top 5 losing streaks to report


# ============================================================
# SECTION 3: Indicator functions (causal only - same math as the
# previous experiment scripts in this project)
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


def add_bollinger(df, period=20, num_std=2.0):
    """Bollinger Bands (middle, upper, lower)."""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


def add_adx(df, period=14):
    """Average Directional Index (Wilder's classic calculation)."""
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
# SECTION 4: Broker specification and volume validation
# (identical to gold_validation.py)
# ============================================================
def get_gold_spec(symbol_name):
    """Read the GOLD symbol specs from MT5 (read-only)."""
    info = mt5.symbol_info(symbol_name)
    if info is None:
        return None

    contract_size = float(info.trade_contract_size)
    point = float(info.point)
    tick_size = float(info.trade_tick_size) if info.trade_tick_size > 0 else point
    tick_value = float(info.trade_tick_value)

    fallback_used = tick_value <= 0
    if fallback_used:
        # Documented fallback (profit currency USD for GOLD):
        tick_value = contract_size * point
        print(f"WARNING: trade_tick_value missing/zero for {symbol_name};")
        print(f"         using fallback tick_value = contract_size x point "
              f"= {tick_value:g} (documented fallback)")

    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0

    return {
        "contract_size": contract_size,
        "point": point,
        "digits": int(info.digits),
        "tick_size": tick_size,
        "tick_value": tick_value,
        "value_per_unit": value_per_unit,
        "volume_min": float(info.volume_min),
        "volume_max": float(info.volume_max),
        "volume_step": float(info.volume_step) if info.volume_step > 0 else 0.01,
        "currency_profit": info.currency_profit or "USD",
        "tick_value_fallback": fallback_used,
    }


def validate_volume(spec):
    """Validate LOTS_REQUESTED down to the broker's volume rules."""
    vmin = spec["volume_min"]
    vmax = spec["volume_max"]
    vstep = spec["volume_step"]

    if LOTS_REQUESTED < vmin:
        return None
    if vmin > vmax:
        return None

    steps_below = int((LOTS_REQUESTED - vmin) / vstep + 1e-9)
    volume = round(vmin + steps_below * vstep, 8)
    if volume > vmax:
        return None
    return volume


# ============================================================
# SECTION 5: The backtest engine
# IDENTICAL to gold_validation.py - same frozen strategy, same
# execution model. Only descriptive fields were added to the log.
# ============================================================
def run_backtest(df, value_per_unit, volume):
    """Run the frozen virtual backtest. Returns (trades DataFrame, balance)."""

    long_condition = df["close"] < df["BB_LOWER"]    # Stretched below the band
    short_condition = df["close"] > df["BB_UPPER"]   # Stretched above the band

    # Regime filter: ADX14 < 25 (frozen - the ONLY threshold)
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE

    i = WARMUP
    while i < len(df) - 1:          # Need bar i+1 to exist for the entry
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1                         # NEXT candle
            entry_price = df["open"].iloc[entry_bar]  # Enter at its open

            if signal == "BUY":
                stop_loss = entry_price - SL_ATR_MULT * atr
                take_profit = entry_price + TP_ATR_MULT * atr
            else:  # SELL
                stop_loss = entry_price + SL_ATR_MULT * atr
                take_profit = entry_price - TP_ATR_MULT * atr

            # STOP checked FIRST (conservative, no look-ahead)
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

            if exit_index is None:
                exit_index = len(df) - 1
                exit_price = df["close"].iloc[exit_index]
                exit_reason = "END_OF_DATA"

            # P/L from GOLD's own specifications
            if signal == "BUY":
                gross_pnl = (exit_price - entry_price) * volume * value_per_unit
            else:
                gross_pnl = (entry_price - exit_price) * volume * value_per_unit

            net_pnl = gross_pnl - COST_PER_TRADE
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

            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 6: DESCRIPTIVE ANALYSIS 1 - equity curve and drawdowns
# ============================================================
def analyze_drawdowns(trades_df):
    """Describe the equity curve and its drawdown periods.

    A drawdown period runs from a new equity PEAK until the balance
    regains that peak (recovery). If the data ends before recovery,
    the period is reported as NOT RECOVERED.
    """
    print()
    print("=" * 64)
    print("ANALYSIS 1: EQUITY CURVE AND DRAWDOWN PERIODS")
    print("=" * 64)

    # --- Build the step-wise equity curve -------------------------------
    # balances[t] = balance after trade t (t=0 is the start).
    times = [None] + trades_df["exit_time"].tolist()
    balances = [STARTING_BALANCE] + trades_df["balance_after"].tolist()

    max_equity = max(balances)
    min_equity = min(balances)

    # --- Walk the curve once, collecting every drawdown period ----------
    peak_value = balances[0]
    peak_idx = 0
    periods = []          # finished drawdown periods
    current = None        # drawdown in progress

    for t in range(1, len(balances)):
        b = balances[t]
        if b >= peak_value:
            # Peak matched or exceeded: any open drawdown recovers here
            if current is not None:
                current["recovery_idx"] = t
                current["recovered"] = True
                periods.append(current)
                current = None
            if b > peak_value:
                peak_value = b
                peak_idx = t
        else:
            # Below the peak -> inside a drawdown
            if current is None:
                current = {
                    "peak_idx": peak_idx,
                    "peak_time": times[peak_idx],
                    "peak_value": peak_value,
                    "start_time": times[peak_idx],
                    "start_idx": peak_idx,
                    "trough_idx": t,
                    "trough_time": times[t],
                    "trough_value": b,
                    "recovery_idx": None,
                    "recovery_time": None,
                    "recovered": False,
                }
            elif b < current["trough_value"]:
                # Same drawdown period, but a NEW low point
                current["trough_idx"] = t
                current["trough_time"] = times[t]
                current["trough_value"] = b

    # Drawdown still open at the end of the data
    if current is not None:
        periods.append(current)

    # --- Enrich with dollars, percentages and durations ------------------
    last_trade_idx = len(balances) - 1
    for p in periods:
        p["dd_dollars"] = p["peak_value"] - p["trough_value"]
        p["dd_pct"] = p["dd_dollars"] / p["peak_value"] * 100 if p["peak_value"] > 0 else 0.0
        p["trades_to_trough"] = p["trough_idx"] - p["start_idx"]
        end_idx = p["recovery_idx"] if p["recovered"] else last_trade_idx
        p["trades_total"] = end_idx - p["start_idx"]
        if p["recovered"]:
            p["recovery_trades"] = p["recovery_idx"] - p["trough_idx"]
            p["recovery_time"] = times[p["recovery_idx"]]
            # Recovery duration measured from the TROUGH
            p["recovery_hours"] = ((times[p["recovery_idx"]]
                                    - times[p["trough_idx"]]).total_seconds() / 3600.0)
            # Full period duration measured from the PEAK
            p["duration_hours"] = ((times[p["recovery_idx"]]
                                    - times[p["start_idx"]]).total_seconds() / 3600.0)
        else:
            p["recovery_trades"] = None
            p["recovery_hours"] = None
            p["duration_hours"] = None

    # Sort by dollar drawdown, largest first
    periods.sort(key=lambda p: p["dd_dollars"], reverse=True)

    # --- Headline numbers -------------------------------------------------
    if periods:
        worst = periods[0]
        if worst["recovered"]:
            rec_hours = worst["recovery_hours"]
            rec_days = rec_hours / 24.0
            rec_txt = (f"{worst['recovery_time']}  "
                       f"({worst['recovery_trades']} trades, "
                       f"{rec_hours:.0f} h = {rec_days:.1f} days after the trough)")
        else:
            rec_txt = "NOT RECOVERED within this data"

        print()
        print(f"Starting balance    : ${STARTING_BALANCE:,.2f}")
        print(f"Ending balance      : ${balances[-1]:,.2f}")
        print(f"Maximum equity      : ${max_equity:,.2f}")
        print(f"Minimum equity      : ${min_equity:,.2f}")
        print()
        print(f"Maximum drawdown    : ${worst['dd_dollars']:,.2f} "
              f"({worst['dd_pct']:.2f}%)")
        print(f"  Drawdown began at : {worst['start_time']} "
              f"(equity peak ${worst['peak_value']:,.2f})")
        print(f"  Trough reached at : {worst['trough_time']} "
              f"(equity low ${worst['trough_value']:,.2f})")
        print(f"  Recovery          : {rec_txt}")
        print(f"  Trades peak->trough  : {worst['trades_to_trough']}")
        print(f"  Trades in whole period: {worst['trades_total']}")

        print()
        print(f"TOP {TOP_N_DRAWDOWNS} LARGEST DRAWDOWN PERIODS:")
        print(f"{'#':<3}{'Start (peak)':<21}{'Trough':<21}"
              f"{'DD $':>10}{'DD %':>8}{'Trades':>8}  Recovery")
        for rank, p in enumerate(periods[:TOP_N_DRAWDOWNS], 1):
            if p["recovered"]:
                rec = (f"{p['recovery_trades']} trades, "
                       f"{p['recovery_hours']:.0f} h after trough")
            else:
                rec = "not recovered"
            print(f"{rank:<3}{str(p['start_time'])[:19]:<21}"
                  f"{str(p['trough_time'])[:19]:<21}"
                  f"{p['dd_dollars']:>10,.2f}{p['dd_pct']:>7.2f}%"
                  f"{p['trades_total']:>8}  {rec}")
    else:
        print()
        print("No drawdown periods exist "
              "(the balance never fell below a previous peak).")

    print()
    print("Reading note: the drawdown periods are DESCRIBED, not fixed.")
    print("A high-win-rate strategy with SL > TP (2.0 vs 1.5 ATR) loses")
    print("more dollars per loss than it gains per win, so losing clusters")
    print("produce deep equity dips even when most trades win.")

    return periods


# ============================================================
# SECTION 7: DESCRIPTIVE ANALYSIS 2 - losing streaks
# ============================================================
def analyze_losing_streaks(trades_df):
    """Describe consecutive losing-trade streaks (pnl < 0)."""
    print()
    print("=" * 64)
    print("ANALYSIS 2: LOSING STREAKS")
    print("=" * 64)

    pnls = trades_df["pnl"].tolist()
    n = len(pnls)

    # --- Walk the sequence once, collecting every streak -----------------
    streaks = []
    start = None
    for t in range(n):
        if pnls[t] < 0:
            if start is None:
                start = t
        else:
            if start is not None:
                streaks.append({"start": start, "end": t - 1})
                start = None
    if start is not None:
        streaks.append({"start": start, "end": n - 1})  # into data end

    if not streaks:
        print()
        print("No losing streaks exist (no losing trades at all).")
        return []

    # --- Enrich streaks with balances and totals -------------------------
    balances_after = trades_df["balance_after"].tolist()
    times = trades_df["exit_time"].tolist()

    for s in streaks:
        s_start, s_end = s["start"], s["end"]
        s["length"] = s_end - s_start + 1
        s["total_loss"] = sum(pnls[s_start:s_end + 1])
        s["balance_before"] = (STARTING_BALANCE if s_start == 0
                               else balances_after[s_start - 1])
        s["balance_after"] = balances_after[s_end]
        s["first_time"] = times[s_start]
        s["last_time"] = times[s_end]
        s["hours"] = ((times[s_end] - times[s_start]).total_seconds() / 3600.0
                      if s["length"] > 1 else 0.0)

    streaks.sort(key=lambda s: s["total_loss"])   # biggest loss first

    # --- Longest streak (by trade count, independent of $ rank) ----------
    longest = max(streaks, key=lambda s: s["length"])

    print()
    print("LONGEST LOSING STREAK (by trade count):")
    print(f"  Consecutive losses : {longest['length']} trades")
    print(f"  From               : trade ending {longest['first_time']}")
    print(f"  To                 : trade ending {longest['last_time']}")
    print(f"  Total loss         : ${longest['total_loss']:,.2f}")
    print(f"  Starting balance   : ${longest['balance_before']:,.2f}")
    print(f"  Ending balance     : ${longest['balance_after']:,.2f}")

    # --- Top 5 streaks by total loss -------------------------------------
    print()
    print(f"TOP {TOP_N_STREAKS} LOSING STREAKS (by total dollar loss):")
    print(f"{'#':<3}{'Losses':>7}{'Total loss $':>14}{'Balance':>12}"
          f"{'->':>4}{'Balance':>12}{'Span':>12}  First exit time")
    for rank, s in enumerate(streaks[:TOP_N_STREAKS], 1):
        print(f"{rank:<3}{s['length']:>7}{s['total_loss']:>14,.2f}"
              f"{s['balance_before']:>12,.2f}"
              f"{'->':>4}{s['balance_after']:>12,.2f}"
              f"{s['hours']:>10.0f} h  {str(s['first_time'])[:19]}")

    print()
    print("Reading note: with SL (2.0 ATR) larger than TP (1.5 ATR), each")
    print("loss is bigger than each win, so streaks of only a few losses")
    print("can erase the gains of many wins. Streaks are a natural")
    print("property of any strategy with a win rate below 100% - they are")
    print("described here, not fixed.")

    return streaks


# ============================================================
# SECTION 8: Main program
# ============================================================
print("Connecting to MetaTrader 5...")

if not mt5.initialize():
    print("ERROR: Could not connect to MetaTrader 5.")
    print("Make sure the MT5 desktop terminal is installed, running,")
    print("and logged into your DEMO account.")
    print("Last error:", mt5.last_error())
    quit()

print("Connected to MetaTrader 5 successfully!")

try:
    # --- Enable the symbol and read its specs (READ-ONLY) ---------------
    if not mt5.symbol_select(SYMBOL, True):
        print(f"ERROR: Symbol '{SYMBOL}' could not be enabled in Market Watch.")
        print("Last error:", mt5.last_error())
        quit()

    spec = get_gold_spec(SYMBOL)
    if spec is None:
        print(f"ERROR: No symbol information available for '{SYMBOL}'.")
        quit()

    volume = validate_volume(spec)
    if volume is None:
        print(f"ERROR: No valid volume <= {LOTS_REQUESTED} lots for {SYMBOL} "
              f"(volume_min {spec['volume_min']}, step {spec['volume_step']}).")
        quit()

    print()
    print(f"MT5 symbol   : {SYMBOL}")
    print(f"Contract     : size {spec['contract_size']:g}, point {spec['point']:g}, "
          f"tick size {spec['tick_size']:g}, tick value {spec['tick_value']:g} "
          f"({spec['currency_profit']})")
    print(f"Volume used  : {volume} lots (requested {LOTS_REQUESTED})")

    # --- Download history (closed candles only) --------------------------
    print(f"Downloading up to {NUM_CANDLES} candles for {SYMBOL} (H1)...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)

    if rates is None or len(rates) == 0:
        print(f"ERROR: No historical data returned for {SYMBOL}.")
        quit()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)   # drop the forming candle

    if len(df) < MIN_CANDLES:
        print(f"ERROR: Only {len(df)} closed candles available "
              f"(minimum required: {MIN_CANDLES}).")
        quit()

    print(f"Closed candles used: {len(df)} "
          f"(from {df['time'].iloc[0]} to {df['time'].iloc[-1]})")

    # --- Indicators (causal only) ----------------------------------------
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)

    # ============================================================
    # SECTION 9: Reproduce the frozen backtest
    # ============================================================
    trades_df, final_balance = run_backtest(df, spec["value_per_unit"], volume)

    print()
    print("=" * 64)
    print("REPRODUCED FROZEN BACKTEST (must match gold_validation.py)")
    print("=" * 64)

    if len(trades_df) == 0:
        print("No virtual trades were generated - nothing to analyze.")
        quit()

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    win_rate = len(wins) / len(trades_df) * 100
    total_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"Starting balance    : ${STARTING_BALANCE:,.2f}")
    print(f"Ending balance      : ${final_balance:,.2f}")
    print(f"Total return        : {total_return:+.2f}%")
    print(f"Number of trades    : {len(trades_df)}")
    print(f"Win rate            : {win_rate:.1f}%")
    pf_txt = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"Profit factor       : {pf_txt}")
    print(f"Average trade       : ${trades_df['pnl'].mean():,.2f}")

    # ============================================================
    # SECTION 10: The descriptive analyses
    # ============================================================
    analyze_drawdowns(trades_df)
    analyze_losing_streaks(trades_df)

    # ============================================================
    # SECTION 11: Disclaimer
    # ============================================================
    print()
    print("DISCLAIMER: This is a historical backtest with virtual money.")
    print("The analyses above DESCRIBE what happened; they do not")
    print("guarantee anything about the future. No parameters were")
    print("optimized or changed. Educational use only.")

finally:
    # ============================================================
    # SECTION 12: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print()
    print("MetaTrader 5 connection closed.")
