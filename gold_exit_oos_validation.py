"""
gold_exit_oos_validation.py - Do the FOUR PREDEFINED GOLD exit structures
from gold_exit_structure_experiment.py behave SIMILARLY on unseen
chronological data?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only validation.
- This script NEVER places, modifies, or closes any trade and contains
  no order execution, order checking, position, or account functions.
- It only reads candles and symbol specifications from MetaTrader 5.
- No machine learning, no parameter optimization, no random search,
  no genetic optimization, and NO variant is selected based on the
  OOS result. The four exit structures were frozen in advance.

THE VALIDATION DESIGN:
  Entry logic FROZEN and identical for all variants:
    Symbol : GOLD (XM MT5), H1
    Entry  : BB(20, 2.0) - LONG when a closed candle closes below the
             lower band, SHORT when above the upper band
    Regime : only when ADX14 < 25 (Wilder smoothing, only threshold)
    ATR    : ATR14
  The four frozen exit structures (no others tested):
    A: SL = 2.0 x ATR, TP = 1.5 x ATR   (existing baseline)
    B: SL = 1.5 x ATR, TP = 1.5 x ATR
    C: SL = 1.5 x ATR, TP = 2.0 x ATR
    D: SL = 2.0 x ATR, TP = 2.0 x ATR

DATA HANDLING:
  - Up to 4000 closed GOLD H1 candles, forming candle dropped,
    at least 2000 required.
  - CHRONOLOGICAL 50/50 split: development = first half, OOS = second
    half. OOS occurs strictly AFTER development. Nothing is shuffled
    or mixed.
  - Indicators are computed ONCE over the full closed series (causal
    math: value at bar i uses only bars <= i). The OOS period therefore
    warms up on historical candles immediately preceding it - which is
    realistic (a trader at the boundary already had that history) and
    uses NO future information.
  - Signal at the close of candle i, entry at candle i+1 open,
    stop-first when SL and TP share a candle, one position at a time,
    $2 cost per trade, P/L from GOLD's actual broker tick specs
    (tick_value / tick_size = USD per price-unit per lot), volume
    validated down to the broker's volume rules.

BALANCES:
  Development and OOS each run INDEPENDENTLY from $10,000 - balances
  are NOT carried across the split, making the periods comparable.

BASELINE CONSISTENCY CHECK (mandatory, runs FIRST):
  The full 4000-candle dataset is first run through Variant A. It must
  approximately reproduce the previous GOLD result (ending balance
  ~$10,211.77, return ~+2.12%, 113 trades, win rate ~54.9%,
  PF ~1.01, max DD ~$4,879.02 / 34.81%). The 50/50 period results will
  naturally differ (separate periods); the FULL-dataset check verifies
  the engine/data connection. If it fails, the script STOPS and does
  not interpret any OOS results.

NO RANKING:
  No variant is called "best" and none is selected. A positive OOS
  result is NOT treated as proof of future profitability.

DISCLAIMER: Historical backtest only. Virtual money only.
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
NUM_CANDLES = 4000    # Same as the previous GOLD experiments
MIN_CANDLES = 2000    # Minimum usable closed candles required

STARTING_BALANCE = 10_000.0
LOTS_REQUESTED = 0.10         # Validated against broker rules below
COST_PER_TRADE = 2.0          # $2 fixed per completed trade

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0   # The ONLY threshold used
ATR_PERIOD = 14
WARMUP = 60            # Development-period start (previous convention)

# The FOUR frozen exit structures (no others are tested)
VARIANTS = [
    ("A", 2.0, 1.5),   # Existing baseline
    ("B", 1.5, 1.5),
    ("C", 1.5, 2.0),
    ("D", 2.0, 2.0),
]

SPLIT_RATIO = 0.5      # First 50% development, final 50% OOS

# Previous full-dataset GOLD result (Variant A baseline) for the
# consistency check - from the earlier GOLD experiment output.
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
# FROZEN entry + parameterized SL/TP multipliers (the four variants).
# Identical execution model to gold_validation.py.
# ============================================================
def run_backtest(df, value_per_unit, volume, sl_mult, tp_mult, start_bar=WARMUP):
    """Run the frozen-entry backtest with the given exit multipliers.
    start_bar: first candle scanned (WARMUP for the full/dev runs, 0 for
    OOS - whose indicators are already warmed by prior history).
    Returns (trades DataFrame, final balance)."""

    long_condition = df["close"] < df["BB_LOWER"]    # Stretched below the band
    short_condition = df["close"] > df["BB_UPPER"]   # Stretched above the band

    # Regime filter: ADX14 < 25 (frozen - the ONLY threshold)
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE

    i = start_bar
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
# SECTION 6: Metrics (identical set for BOTH periods)
# ============================================================
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


def compute_metrics(trades_df, final_balance):
    """Compute the full required metric set for one variant/period run."""
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    win_rate = len(wins) / len(trades_df) * 100
    total_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

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

    # Longest losing streak by trade count + its $ total
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
        "max_equity": max(balances),
        "min_equity": min(balances),
        "avg_hold": trades_df["hold_candles"].mean(),
        "tp": int((trades_df["exit_reason"] == "TAKE_PROFIT").sum()),
        "sl": int((trades_df["exit_reason"] == "STOP_LOSS").sum()),
        "eod": int((trades_df["exit_reason"] == "END_OF_DATA").sum()),
        "longest_streak_len": longest_len,
        "longest_streak_loss": longest_loss,
        "dd_recovered": _dd_recovered(balances),
        "dd_peak_to_trough_trades": _dd_peak_to_trough(balances),
    }


def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def print_variant_results(code, sl_mult, tp_mult, m):
    """Print the full metric block for one variant/period."""
    print(f"\nVariant {code} - SL {sl_mult} ATR / TP {tp_mult} ATR")
    print(f"  Starting balance     : ${STARTING_BALANCE:,.2f}")
    print(f"  Ending balance       : ${m['end_balance']:,.2f}")
    print(f"  Return               : {m['return']:+.2f}%")
    print(f"  Number of trades     : {m['trades']}")
    print(f"  Wins / Losses        : {m['wins']} / {m['losses']}")
    print(f"  Win rate             : {m['win_rate']:.1f}%")
    print(f"  Gross profit         : ${m['gross_profit']:,.2f}")
    print(f"  Gross loss           : ${m['gross_loss']:,.2f}")
    print(f"  Profit factor        : {fmt_pf(m['profit_factor'])}")
    print(f"  Average trade        : ${m['avg_trade']:,.2f}")
    print(f"  Maximum drawdown     : ${m['max_dd']:,.2f}  ({m['max_dd_pct']:.2f}%)")
    print(f"  Maximum equity       : ${m['max_equity']:,.2f}")
    print(f"  Minimum equity       : ${m['min_equity']:,.2f}")
    print(f"  Average holding time : {m['avg_hold']:.1f} candles")
    print(f"  TP / SL / END exits  : {m['tp']} / {m['sl']} / {m['eod']}")
    print(f"  Longest losing streak: {m['longest_streak_len']} trades "
          f"(${m['longest_streak_loss']:,.2f})")
    rec = "yes" if m["dd_recovered"] else "NO - still open at period end"
    print(f"  Max DD recovered     : {rec}")
    print(f"  Peak-to-trough trades: {m['dd_peak_to_trough_trades']}")


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
    print("GOLD EXIT STRUCTURE OOS VALIDATION")
    print("=" * 62)
    print()
    print("Symbol:")
    print(SYMBOL)
    print("Timeframe:")
    print("H1")

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

    # --- Indicators ONCE over the full closed series (causal math) -------
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)

    # ============================================================
    # SECTION 8: BASELINE CONSISTENCY CHECK (full dataset, Variant A)
    # Runs FIRST. The OOS analysis below only happens if this passes.
    # ============================================================
    print()
    print("=" * 62)
    print("BASELINE CONSISTENCY CHECK (full dataset, Variant A)")
    print("=" * 62)

    trades_full, full_balance = run_backtest(df, spec["value_per_unit"], volume,
                                             VARIANTS[0][1], VARIANTS[0][2],
                                             start_bar=WARMUP)
    full_m = compute_metrics(trades_full, full_balance)

    prev = PREVIOUS_RESULT
    a = full_m
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
        print("STOP: The full-dataset Variant A does NOT reproduce the")
        print("previous GOLD baseline within tolerance.")
        print()
        print("Possible causes to diagnose BEFORE interpreting anything:")
        print("- Different historical period: the terminal may now serve")
        print("  a shifted window of the last 4000 candles (new candles")
        print("  arrive constantly, so the window moves forward in time).")
        print("- Different broker specifications (tick value / tick size).")
        print("- Volume validation resolving to a different volume.")
        print("- A change in the engine or data handling.")
        print()
        print("The OOS analysis is deliberately SKIPPED: split results")
        print("from an engine/data mismatch would be meaningless.")
        quit()

    print()
    print("Full-dataset Variant A reproduces the previous baseline within")
    print("tolerance - the engine, broker specs and data connection are")
    print("consistent with the earlier GOLD experiment. Continuing with")
    print("the chronological 50/50 split.")

    # ============================================================
    # SECTION 9: CHRONOLOGICAL 50/50 SPLIT
    # ============================================================
    split_index = len(df) // 2
    dev_df = df.iloc[:split_index].reset_index(drop=True)   # first half
    oos_df = df.iloc[split_index:].reset_index(drop=True)   # final half

    print()
    print("=" * 62)
    print("CHRONOLOGICAL 50/50 SPLIT")
    print("=" * 62)
    print(f"Development: {dev_df['time'].iloc[0]} -> {dev_df['time'].iloc[-1]}")
    print(f"OOS        : {oos_df['time'].iloc[0]} -> {oos_df['time'].iloc[-1]}")
    print("OOS occurs strictly AFTER development; nothing was shuffled.")
    print("Indicators were computed causally ONCE over the full closed")
    print("series, so OOS warms up on the history immediately before it")
    print("and uses NO future information.")
    print("Each period starts from an INDEPENDENT $10,000 balance.")

    # ============================================================
    # SECTION 10: DEVELOPMENT RESULTS (all four variants)
    # ============================================================
    print()
    print("=" * 62)
    print("DEVELOPMENT RESULTS")
    print("=" * 62)

    dev_metrics = {}
    for code, sl_mult, tp_mult in VARIANTS:
        trades_df, final_balance = run_backtest(dev_df, spec["value_per_unit"],
                                                volume, sl_mult, tp_mult,
                                                start_bar=WARMUP)
        dev_metrics[code] = compute_metrics(trades_df, final_balance)
        print_variant_results(code, sl_mult, tp_mult, dev_metrics[code])

    # ============================================================
    # SECTION 11: OOS RESULTS (all four variants)
    # ============================================================
    print()
    print("=" * 62)
    print("OOS RESULTS")
    print("=" * 62)

    oos_metrics = {}
    for code, sl_mult, tp_mult in VARIANTS:
        trades_df, final_balance = run_backtest(oos_df, spec["value_per_unit"],
                                                volume, sl_mult, tp_mult,
                                                start_bar=0)
        oos_metrics[code] = compute_metrics(trades_df, final_balance)
        print_variant_results(code, sl_mult, tp_mult, oos_metrics[code])

    # ============================================================
    # SECTION 12: OOS COMPARISON TABLE (neutral, no ranking)
    # ============================================================
    print()
    print("=" * 62)
    print("OOS COMPARISON TABLE")
    print("=" * 62)
    print(f"{'Var':<4}{'SL':>4}{'TP':>4}{'Return':>9}{'Trades':>8}"
          f"{'WinRate':>9}{'PF':>7}{'AvgTrade':>10}{'DD%':>8}{'DD$':>10}"
          f"{'Streak':>8}")
    for code, sl_mult, tp_mult in VARIANTS:
        m = oos_metrics[code]
        print(f"{code:<4}{sl_mult:>4.1f}{tp_mult:>4.1f}"
              f"{m['return']:>8.2f}%{m['trades']:>8}"
              f"{m['win_rate']:>8.1f}%{fmt_pf(m['profit_factor']):>7}"
              f"{m['avg_trade']:>10.2f}{m['max_dd_pct']:>7.2f}%"
              f"{m['max_dd']:>10,.2f}{m['longest_streak_len']:>8}")
    print("Neutral measurements only - no variant is ranked or selected.")

    # ============================================================
    # SECTION 13: DEVELOPMENT vs OOS (factual deltas)
    # ============================================================
    print()
    print("=" * 62)
    print("DEVELOPMENT vs OOS (OOS minus Development)")
    print("=" * 62)
    for code, sl_mult, tp_mult in VARIANTS:
        dm, om = dev_metrics[code], oos_metrics[code]
        print(f"\nVariant {code}:")
        print(f"  Return    : {dm['return']:+.2f}% -> {om['return']:+.2f}% "
              f"(difference {om['return'] - dm['return']:+.2f} pp)")
        print(f"  PF        : {fmt_pf(dm['profit_factor'])} -> "
              f"{fmt_pf(om['profit_factor'])} "
              f"(difference {om['profit_factor'] - dm['profit_factor']:+.2f})")
        print(f"  Win rate  : {dm['win_rate']:.1f}% -> {om['win_rate']:.1f}% "
              f"(difference {om['win_rate'] - dm['win_rate']:+.1f} pp)")
        print(f"  Max DD %  : {dm['max_dd_pct']:.2f}% -> {om['max_dd_pct']:.2f}% "
              f"(difference {om['max_dd_pct'] - dm['max_dd_pct']:+.2f} pp)")

    # ============================================================
    # SECTION 14: RESEARCH INTERPRETATION (six factual answers)
    # ============================================================
    print()
    print("=" * 62)
    print("RESEARCH INTERPRETATION (neutral, factual)")
    print("=" * 62)

    # Q1: Does the exit-structure difference remain visible in OOS?
    print()
    print("1. Is the exit-structure difference still visible in OOS?")
    dev_returns = [dev_metrics[c]["return"] for c, _, _ in VARIANTS]
    oos_returns = [oos_metrics[c]["return"] for c, _, _ in VARIANTS]
    dev_spread = max(dev_returns) - min(dev_returns)
    oos_spread = max(oos_returns) - min(oos_returns)
    print(f"   Return spread across variants: development {dev_spread:.2f} pp, "
          f"OOS {oos_spread:.2f} pp.")
    if oos_spread >= dev_spread * 0.5:
        print("   -> YES: the variants still diverge in OOS, so the exit")
        print("      structure remains a meaningful differentiator.")
    else:
        print("   -> The OOS divergence is much smaller: differences seen")
        print("      in development did NOT carry over strongly.")

    # Q2: Which variants remain 'profitable' in OOS (predefined rule)?
    print()
    print("2. Variants meeting ALL THREE in OOS "
          "(return > 0, PF > 1, avg trade > 0):")
    for code, _, _ in VARIANTS:
        m = oos_metrics[code]
        ok = (m["return"] > 0 and m["profit_factor"] > 1.0 and m["avg_trade"] > 0)
        print(f"   {code}: return {m['return']:+.2f}%, "
              f"PF {fmt_pf(m['profit_factor'])}, "
              f"avg ${m['avg_trade']:+.2f}  ->  "
              f"{'YES' if ok else 'no'}")
    print("   (A high win rate alone does NOT count as profitability.)")

    # Q3: OOS drawdown vs development drawdown?
    print()
    print("3. OOS drawdown vs development drawdown (percentage points):")
    for code, _, _ in VARIANTS:
        dm, om = dev_metrics[code], oos_metrics[code]
        diff = om["max_dd_pct"] - dm["max_dd_pct"]
        direction = ("DEEPER" if diff > 1 else
                     "SHALLOWER" if diff < -1 else "similar")
        print(f"   {code}: {dm['max_dd_pct']:.2f}% -> {om['max_dd_pct']:.2f}% "
              f"({diff:+.2f} pp, {direction})")

    # Q4: Variant B's historical +47% result - similar, smaller, negative?
    print()
    print("4. Variant B's previously observed large development result:")
    bm = oos_metrics["B"]
    if bm["return"] > 0:
        if abs(bm["return"] - 47.0) <= 15:
            verdict = "remained SIMILAR"
        else:
            verdict = f"became SMALLER ({bm['return']:+.2f}% in OOS)"
    else:
        verdict = f"became NEGATIVE ({bm['return']:+.2f}% in OOS)"
    print(f"   -> In this OOS period it {verdict}.")
    print("   (One OOS period is a single observation - this does NOT")
    print("   predict which behavior future data will show.)")

    # Q5: Any large deterioration from development to OOS?
    print()
    print("5. Large deteriorations (return drop > 10 pp or DD increase > 5 pp):")
    found = False
    for code, _, _ in VARIANTS:
        dm, om = dev_metrics[code], oos_metrics[code]
        ret_drop = dm["return"] - om["return"]
        dd_rise = om["max_dd_pct"] - dm["max_dd_pct"]
        if ret_drop > 10 or dd_rise > 5:
            found = True
            print(f"   {code}: return drop {ret_drop:+.2f} pp, "
                  f"DD rise {dd_rise:+.2f} pp  ->  DETERIORATED")
    if not found:
        print("   No variant shows a large deterioration by these criteria.")

    # Q6: Instability even when the exit structure changed?
    print()
    print("6. Instability across ALL exit structures:")
    unstable = [c for c, _, _ in VARIANTS
                if oos_metrics[c]["max_dd_pct"] > 20
                or oos_metrics[c]["longest_streak_len"] >= 8]
    for code, _, _ in VARIANTS:
        m = oos_metrics[code]
        print(f"   {code}: OOS DD {m['max_dd_pct']:.2f}%, "
              f"streak {m['longest_streak_len']} losses, "
              f"recovered: {'yes' if m['dd_recovered'] else 'no'}")
    if len(unstable) >= 3:
        print("   -> Instability is present across MOST exit structures in")
        print("      OOS: changing the exits does not stabilize the GOLD")
        print("      strategy - the entry signal / market conditions remain")
        print("      the likely source.")
    else:
        print("   -> Only some structures show OOS instability; the exit")
        print("      structure does influence the stability on this data.")

    print()
    print("IMPORTANT: A positive OOS result is NOT proof of future")
    print("profitability. This is one historical sample, split once.")

    # ============================================================
    # SECTION 15: Disclaimer
    # ============================================================
    print()
    print("Historical backtest only.")
    print("Virtual money only.")
    print("No live/demo orders were executed.")
    print("No parameters were optimized; no variant was selected based")
    print("on the OOS result.")

finally:
    # ============================================================
    # SECTION 16: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print()
    print("MetaTrader 5 connection closed.")
