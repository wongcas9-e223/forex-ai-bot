"""
gold_exit_structure_experiment.py - Is the large GOLD drawdown mainly
caused by the ASYMMETRIC EXIT STRUCTURE, or is the GOLD entry signal
itself unstable?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only research run.
- This script NEVER places, modifies, or closes any trade and contains
  no order execution, order checking, position, or account functions.
- It only reads candles and symbol specifications from MetaTrader 5.
- No machine learning, no parameter optimization, no random search,
  no genetic optimization. The four exit structures below were fixed
  in advance and no other combination was tested or considered.

THE RESEARCH DESIGN (one variable at a time):
The entry logic is FROZEN and identical in all four variants:
  Symbol     : GOLD (XM MT5), H1
  Entry      : BB(20, 2.0) - LONG when a closed candle closes below the
               lower band, SHORT when above the upper band
  Regime     : only when ADX14 < 25 (Wilder smoothing, only threshold)
  ATR        : ATR14 exactly as in gold_validation.py
  Execution  : signal on closed candle, enter at NEXT candle open,
               one position at a time, stop-first when SL and TP share
               a candle, no look-ahead, $2.00 cost per completed trade,
               $10,000 starting virtual balance, 0.10 lots requested
               (validated down to the broker's volume rules),
               P/L from GOLD's actual tick specifications
               (tick_value / tick_size = USD per price-unit per lot)

ONLY THE EXIT STRUCTURE DIFFERS (four predefined variants):
  Variant A - CURRENT BASELINE : SL = 2.0 x ATR, TP = 1.5 x ATR
  Variant B - EQUAL 1.5 ATR    : SL = 1.5 x ATR, TP = 1.5 x ATR
  Variant C - 1.5 / 2.0        : SL = 1.5 x ATR, TP = 2.0 x ATR
  Variant D - EQUAL 2.0 ATR    : SL = 2.0 x ATR, TP = 2.0 x ATR

BASELINE REPRODUCTION CHECK:
  Variant A must approximately reproduce the existing GOLD validation
  result (ending balance ~$10,211.77, return ~+2.12%, 113 trades,
  win rate ~54.9%, PF ~1.01, max DD ~$4,879.02 / 34.81%). If it does
  NOT, the script prints a clear warning and STOPS before the
  interpretation section, so no conclusions are drawn from a broken
  comparison. gold_validation.py and gold_risk_analysis.py are never
  modified by this script.

NO RANKING:
  No variant is called "best" and no winner ranking is produced. The
  comparison table is neutral measurement output only.

DISCLAIMER: Historical backtest only. Virtual money only. Results do
not guarantee future performance. No parameters were optimized.
Educational use only.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read-only usage in this script
import pandas as pd        # DataFrame + indicator math
import numpy as np         # numeric helpers

# ============================================================
# SECTION 2: Settings - frozen entry, four predefined exit variants
# ============================================================
SYMBOL = "GOLD"       # The confirmed exact XM MT5 gold symbol
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 4000    # Same as gold_validation.py
MIN_CANDLES = 2000    # Minimum usable closed candles required

STARTING_BALANCE = 10_000.0
LOTS_REQUESTED = 0.10         # Validated against broker rules below
COST_PER_TRADE = 2.0          # $2 fixed per completed trade

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0   # The ONLY threshold used
ATR_PERIOD = 14
WARMUP = 60            # Bars skipped before scanning (previous convention)

# The FOUR predefined exit structures (no others are tested)
VARIANTS = [
    ("A", 2.0, 1.5),   # Current baseline
    ("B", 1.5, 1.5),   # Equal 1.5 ATR
    ("C", 1.5, 2.0),   # 1.5 / 2.0
    ("D", 2.0, 2.0),   # Equal 2.0 ATR
]

# Previous GOLD validation numbers for the reproduction check
# (from the earlier experiment output)
PREVIOUS_RESULT = {
    "end_balance": 10_211.77,
    "return": 2.12,
    "trades": 113,
    "win_rate": 54.9,
    "profit_factor": 1.01,
    "max_dd": 4_879.02,
    "max_dd_pct": 34.81,
}
REPRO_TOLERANCES = {
    "end_balance": 150.0,   # $ tolerance on the ending balance
    "return": 1.5,          # percentage points
    "trades": 5,            # trade count
    "win_rate": 5.0,        # percentage points
    "profit_factor": 0.10,
    "max_dd": 750.0,        # $ tolerance on max drawdown
}


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
# Entry logic FROZEN (identical to gold_validation.py); the SL/TP
# multipliers are parameters so the four variants can share one engine.
# ============================================================
def run_backtest(df, value_per_unit, volume, sl_mult, tp_mult):
    """Run the frozen-entry backtest with the given exit multipliers.
    Returns (trades DataFrame, final balance)."""

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

            # Exit structure of THIS variant, fixed at entry
            if signal == "BUY":
                stop_loss = entry_price - sl_mult * atr
                take_profit = entry_price + tp_mult * atr
            else:  # SELL
                stop_loss = entry_price + sl_mult * atr
                take_profit = entry_price - tp_mult * atr

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
# SECTION 6: Metrics (all required numbers for one variant)
# ============================================================
def compute_metrics(trades_df, final_balance):
    """Compute the full required metric set for one variant run."""
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    win_rate = len(wins) / len(trades_df) * 100
    total_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # --- Max drawdown (peak-to-trough of the balance curve) --------------
    balances = [STARTING_BALANCE] + trades_df["balance_after"].tolist()
    peak = balances[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for b in balances[1:]:
        if b > peak:
            peak = b
        drawdown = peak - b
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown / peak * 100

    # --- Losing streaks (longest by trade count + its $ total) -----------
    pnls = trades_df["pnl"].tolist()
    longest_len = 0
    longest_loss = 0.0
    run_len = 0
    run_loss = 0.0
    for p in pnls:
        if p < 0:
            run_len += 1
            run_loss += p
            if run_len > longest_len:
                longest_len = run_len
                longest_loss = run_loss
        else:
            run_len = 0
            run_loss = 0.0

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
        "tp": int((trades_df["exit_reason"] == "TAKE_PROFIT").sum()),
        "sl": int((trades_df["exit_reason"] == "STOP_LOSS").sum()),
        "eod": int((trades_df["exit_reason"] == "END_OF_DATA").sum()),
        "longest_streak_len": longest_len,
        "longest_streak_loss": longest_loss,
        "max_equity": max(balances),
        "min_equity": min(balances),
        # Recovery question: does the balance regain its all-time peak
        # AFTER the max drawdown trough, before the data ends?
        "dd_recovered": _dd_recovered(balances),
        "dd_peak_to_trough_trades": _dd_peak_to_trough(balances),
    }


def _dd_recovered(balances):
    """True if the balance regains its running all-time peak after the
    max-drawdown trough, before the data ends."""
    peak = balances[0]
    worst_dd = 0.0
    trough_idx = 0
    for t, b in enumerate(balances):
        if b > peak:
            peak = b
        dd = peak - b
        if dd > worst_dd:
            worst_dd = dd
            trough_idx = t
    if worst_dd == 0.0:
        return True   # no drawdown at all
    peak_at_trough = max(balances[:trough_idx + 1])
    return any(b >= peak_at_trough for b in balances[trough_idx + 1:])


def _dd_peak_to_trough(balances):
    """Number of trades from the max-drawdown peak to its trough."""
    peak = balances[0]
    peak_idx = 0
    worst_dd = 0.0
    trough_idx = 0
    for t, b in enumerate(balances):
        if b > peak:
            peak = b
            peak_idx = t
        dd = peak - b
        if dd > worst_dd:
            worst_dd = dd
            trough_idx = t
    return trough_idx - peak_idx


def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


# ============================================================
# SECTION 7: Main program
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
    print()
    print("=" * 62)
    print("GOLD EXIT STRUCTURE EXPERIMENT")
    print("=" * 62)

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

    # --- Download history (closed candles only) --------------------------
    print()
    print("Data:")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: H1")

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

    print(f"Closed candles: {len(df)}")
    print(f"Date range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    print(f"Volume used: {volume} lots (requested {LOTS_REQUESTED})")
    print(f"Contract: size {spec['contract_size']:g}, tick size "
          f"{spec['tick_size']:g}, tick value {spec['tick_value']:g} "
          f"({spec['currency_profit']})")

    # --- Indicators (causal only, computed ONCE for all variants) --------
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)

    # ============================================================
    # SECTION 8: Run the FOUR predefined exit variants
    # ============================================================
    metrics = {}
    for code, sl_mult, tp_mult in VARIANTS:
        label = (f"{code} - SL {sl_mult} ATR / TP {tp_mult} ATR")
        print()
        print("-" * 62)
        print(f"## VARIANT {label}")
        print("-" * 62)

        trades_df, final_balance = run_backtest(df, spec["value_per_unit"], volume,
                                                sl_mult, tp_mult)
        m = compute_metrics(trades_df, final_balance)
        metrics[code] = m

        print(f"Starting balance    : ${STARTING_BALANCE:,.2f}")
        print(f"Ending balance      : ${m['end_balance']:,.2f}")
        print(f"Total return        : {m['return']:+.2f}%")
        print(f"Number of trades    : {m['trades']}")
        print(f"Wins / Losses       : {m['wins']} / {m['losses']}")
        print(f"Win rate            : {m['win_rate']:.1f}%")
        print(f"Gross profit        : ${m['gross_profit']:,.2f}")
        print(f"Gross loss          : ${m['gross_loss']:,.2f}")
        print(f"Profit factor       : {fmt_pf(m['profit_factor'])}")
        print(f"Average trade       : ${m['avg_trade']:,.2f}")
        print(f"Maximum drawdown    : ${m['max_dd']:,.2f}  ({m['max_dd_pct']:.2f}%)")
        print(f"Average holding time: {m['avg_hold']:.1f} candles")
        print(f"TP / SL / END exits : {m['tp']} / {m['sl']} / {m['eod']}")
        print(f"Longest losing streak: {m['longest_streak_len']} trades "
              f"(${m['longest_streak_loss']:,.2f})")
        print(f"Maximum equity      : ${m['max_equity']:,.2f}")
        print(f"Minimum equity      : ${m['min_equity']:,.2f}")
        rec = "yes" if m["dd_recovered"] else "NO - still open at data end"
        print(f"Max DD recovered    : {rec}")
        print(f"Peak-to-trough trades: {m['dd_peak_to_trough_trades']}")

    # ============================================================
    # SECTION 9: Compact neutral comparison table (NO ranking)
    # ============================================================
    print()
    print("-" * 62)
    print("COMPARISON TABLE (neutral measurements - no variant is 'best')")
    print("-" * 62)
    print(f"{'Var':<4}{'SL':>4}{'TP':>4}{'Return':>9}{'Trades':>8}"
          f"{'WinRate':>9}{'PF':>7}{'AvgTrade':>10}{'DD%':>8}{'DD$':>10}"
          f"{'Streak':>8}")
    for code, sl_mult, tp_mult in VARIANTS:
        m = metrics[code]
        print(f"{code:<4}{sl_mult:>4.1f}{tp_mult:>4.1f}"
              f"{m['return']:>8.2f}%{m['trades']:>8}"
              f"{m['win_rate']:>8.1f}%{fmt_pf(m['profit_factor']):>7}"
              f"{m['avg_trade']:>10.2f}{m['max_dd_pct']:>7.2f}%"
              f"{m['max_dd']:>10,.2f}{m['longest_streak_len']:>8}")

    # ============================================================
    # SECTION 10: BASELINE REPRODUCTION CHECK (mandatory gate)
    # ============================================================
    print()
    print("-" * 62)
    print("## BASELINE REPRODUCTION CHECK")
    print("-" * 62)
    prev = PREVIOUS_RESULT
    a = metrics["A"]

    checks = [
        ("Ending balance", a["end_balance"], prev["end_balance"],
         abs(a["end_balance"] - prev["end_balance"]) <= REPRO_TOLERANCES["end_balance"],
         f"${a['end_balance']:,.2f} vs ${prev['end_balance']:,.2f}"),
        ("Total return", a["return"], prev["return"],
         abs(a["return"] - prev["return"]) <= REPRO_TOLERANCES["return"],
         f"{a['return']:+.2f}% vs {prev['return']:+.2f}%"),
        ("Trades", a["trades"], prev["trades"],
         abs(a["trades"] - prev["trades"]) <= REPRO_TOLERANCES["trades"],
         f"{a['trades']} vs {prev['trades']}"),
        ("Win rate", a["win_rate"], prev["win_rate"],
         abs(a["win_rate"] - prev["win_rate"]) <= REPRO_TOLERANCES["win_rate"],
         f"{a['win_rate']:.1f}% vs {prev['win_rate']:.1f}%"),
        ("Profit factor", a["profit_factor"], prev["profit_factor"],
         abs(a["profit_factor"] - prev["profit_factor"]) <= REPRO_TOLERANCES["profit_factor"],
         f"{fmt_pf(a['profit_factor'])} vs {prev['profit_factor']:.2f}"),
        ("Max drawdown $", a["max_dd"], prev["max_dd"],
         abs(a["max_dd"] - prev["max_dd"]) <= REPRO_TOLERANCES["max_dd"],
         f"${a['max_dd']:,.2f} vs ${prev['max_dd']:,.2f}"),
        ("Max drawdown %", a["max_dd_pct"], prev["max_dd_pct"],
         abs(a["max_dd_pct"] - prev["max_dd_pct"]) <= REPRO_TOLERANCES["max_dd"] / 10,
         f"{a['max_dd_pct']:.2f}% vs {prev['max_dd_pct']:.2f}%"),
    ]

    all_ok = True
    for name, got, expected, ok, detail in checks:
        print(f"  {name:<16}: {detail}  ->  {'OK' if ok else 'MISMATCH'}")
        all_ok = all_ok and ok

    if not all_ok:
        print()
        print("STOP: Variant A does NOT reproduce the existing GOLD")
        print("validation result within tolerance. Diagnose the difference")
        print("(data period, broker specs, volume validation) before")
        print("drawing any conclusions. The interpretation section below")
        print("is deliberately SKIPPED so no conclusion rests on a")
        print("broken baseline.")
        quit()

    print()
    print("Baseline reproduced within tolerance - the exit comparison")
    print("is built on the same data, specs and execution model as the")
    print("existing GOLD validation.")

    # ============================================================
    # SECTION 11: FACTUAL OBSERVATIONS (five research questions)
    # ============================================================
    a, b, c, d = metrics["A"], metrics["B"], metrics["C"], metrics["D"]

    def pct_change(new, old):
        if old == 0:
            return "n/a"
        chg = (new - old) / abs(old) * 100
        return f"{chg:+.0f}%"

    print()
    print("-" * 62)
    print("## FACTUAL OBSERVATIONS (answers, not recommendations)")
    print("-" * 62)

    # Q1: Does changing only the exit structure materially change max DD?
    dd_values = [a["max_dd"], b["max_dd"], c["max_dd"], d["max_dd"]]
    dd_spread = max(dd_values) - min(dd_values)
    dd_rel = dd_spread / min(dd_values) * 100 if min(dd_values) > 0 else 0.0
    print()
    print("1. Maximum drawdown across exit structures:")
    print(f"   DD ranges from ${min(dd_values):,.2f} to ${max(dd_values):,.2f} "
          f"(spread ${dd_spread:,.2f} = {dd_rel:.0f}% of the smallest).")
    if dd_rel >= 50:
        print("   -> The exit structure MATERIALLY changes the drawdown.")
        print("      The same entry produces very different equity pain")
        print("      depending only on where SL/TP are placed.")
    else:
        print("   -> The exit structure changes the drawdown only")
        print("      moderately; a large part of the pain comes from")
        print("      the entry signal itself.")

    # Q2: Does reducing SL 2.0 -> 1.5 reduce losing streaks?
    print()
    print("2. Effect of reducing SL from 2.0 to 1.5 ATR:")
    print(f"   Longest losing streak: A {a['longest_streak_len']} trades "
          f"(${a['longest_streak_loss']:,.2f}) -> "
          f"B {b['longest_streak_len']} trades (${b['longest_streak_loss']:,.2f})")
    print(f"   Streak $ total change: {pct_change(b['longest_streak_loss'], a['longest_streak_loss'])}")
    print(f"   Win rate change     : {a['win_rate']:.1f}% -> {b['win_rate']:.1f}%")
    print("   (A tighter SL stops trades earlier: losses are smaller, but")
    print("   some trades that would have recovered now stop out, which")
    print("   can shorten OR lengthen streaks depending on the data.)")

    # Q3: Does increasing TP 1.5 -> 2.0 reduce or increase win rate?
    print()
    print("3. Effect of increasing TP from 1.5 to 2.0 ATR:")
    print(f"   A (TP 1.5): win rate {a['win_rate']:.1f}%  |  "
          f"C (TP 2.0, SL 1.5): {c['win_rate']:.1f}%  "
          f"({c['win_rate'] - a['win_rate']:+.1f} pp, note SL also changed)")
    print(f"   A (TP 1.5): win rate {a['win_rate']:.1f}%  |  "
          f"D (TP 2.0, SL 2.0): {d['win_rate']:.1f}%  "
          f"({d['win_rate'] - a['win_rate']:+.1f} pp, note SL also changed)")
    print("   (A farther TP is reached less often, so the win rate")
    print("   typically DROPS when TP increases - the numbers above show")
    print("   the actual measured effect on this data.)")

    # Q4: Does any structure improve PF AND reduce max DD simultaneously?
    print()
    print("4. Profit factor vs drawdown trade-off:")
    for code in ("B", "C", "D"):
        m = metrics[code]
        pf_up = m["profit_factor"] > a["profit_factor"]
        dd_down = m["max_dd"] < a["max_dd"]
        if pf_up and dd_down:
            print(f"   Variant {code}: PF {fmt_pf(a['profit_factor'])} -> "
                  f"{fmt_pf(m['profit_factor'])} (UP) and DD "
                  f"${a['max_dd']:,.2f} -> ${m['max_dd']:,.2f} (DOWN): "
                  f"improves BOTH on this data.")
        else:
            print(f"   Variant {code}: PF {fmt_pf(a['profit_factor'])} -> "
                  f"{fmt_pf(m['profit_factor'])} "
                  f"({'up' if pf_up else 'down'}) and DD "
                  f"${a['max_dd']:,.2f} -> ${m['max_dd']:,.2f} "
                  f"({'down' if dd_down else 'up'}): mixed effect.")

    # Q5: Unstable equity regardless of exit structure?
    print()
    print("5. Equity stability across ALL four structures:")
    for code in ("A", "B", "C", "D"):
        m = metrics[code]
        print(f"   {code}: DD {m['max_dd_pct']:.2f}%, "
              f"streak {m['longest_streak_len']} losses, "
              f"PF {fmt_pf(m['profit_factor'])}, "
              f"recovered: {'yes' if m['dd_recovered'] else 'no'}")
    all_dd_gt_20 = all(m["max_dd_pct"] > 20 for m in metrics.values())
    if all_dd_gt_20:
        print("   -> ALL structures show drawdowns above 20%: the instability")
        print("      persists regardless of exit structure, pointing at the")
        print("      ENTRY signal / market conditions rather than exits.")
    else:
        print("   -> At least one structure shows materially smaller")
        print("      drawdowns, so the exit structure contributes")
        print("      meaningfully to the observed instability.")

    print()
    print("These are measurements on ONE historical sample. They do not")
    print("recommend any variant and do not predict future behavior.")

    # ============================================================
    # SECTION 12: Disclaimer
    # ============================================================
    print()
    print("Historical backtest only.")
    print("Virtual money only.")
    print("No live/demo orders were executed.")
    print("No parameters were optimized.")

finally:
    # ============================================================
    # SECTION 13: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print()
    print("MetaTrader 5 connection closed.")
