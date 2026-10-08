"""
multi_instrument_10000_history_test.py - RESEARCH/DIAGNOSTIC ONLY.

Extends the FROZEN BB(20, 2.0) + ADX14 < 25 mean-reversion strategy
(SL 2.0 x ATR / TP 1.5 x ATR, next-open entry, stop-first, 0.10 lots,
$2 cost) from approximately 4,000 H1 candles to up to 10,000 H1 CLOSED
candles per instrument. The ONLY purpose is to test whether the
previously observed behaviour persists across a longer historical
window.

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only diagnostic.
- This script NEVER places, modifies, or closes any order and contains
  no order execution, order checking, position, history, or account
  trading functions.
- Allowed MT5 operations (read-only): initialize, shutdown, symbols_get,
  symbol_info, symbol_info_tick, symbol_select, copy_rates_from_pos.

NO OPTIMIZATION OF ANY KIND:
- no parameter loops, no grid search, no random search, no walk-forward
  optimization, no parameter selection, no strategy selection, no
  ranking, no filtering based on performance. Only the frozen
  parameters below are used, unchanged, for every instrument:
    Indicators: BB period 20, BB std dev 2.0, ADX period 14, ATR period 14
    Entry     : LONG close < lower band AND ADX < 25
                SHORT close > upper band AND ADX < 25
                (signal from candle i, entry at candle i+1 OPEN)
    Exits     : SL = 2.0 x ATR, TP = 1.5 x ATR
                (if both occur inside one candle, SL is assumed first)
    Position  : 0.10 lots, one position at a time, no compounding,
                independent $10,000 starting balance
    Cost      : $2.00 per completed trade

DATA:
- Up to 10,000 H1 candles requested per instrument, CLOSED candles only
  (the still-forming candle is dropped). If fewer closed candles are
  available, all available closed candles are used and the actual count
  is reported. Nothing is shuffled or sampled.

INDICATOR RULE:
- Indicators are calculated ONCE over the complete chronological
  closed-candle series, which is causal: every indicator value uses
  only current and previous candles, never future ones.

P/L:
- Uses each symbol's actual MT5 contract/tick specifications (contract
  size, point, digits, tick size, tick value). GOLD-style P/L is never
  assumed for forex pairs.

STRUCTURE:
- For each instrument the available closed-candle history is divided
  CHRONOLOGICALLY into four approximately equal periods (no shuffling,
  no performance-based selection). Each period starts from its own
  independent $10,000 balance. One additional descriptive full-history
  run over the entire window is reported; it is NOT used to select or
  modify anything.

OUTPUT IS DESCRIPTIVE ONLY:
  Sections 1-5 are factual tables. Section 6 answers eight factual
  robustness questions, followed by a small factual comparison of
  period-to-period sign changes between a ~4,000-candle window and the
  up-to-10,000-candle window. There are NO rankings, NO scores, NO
  "good"/"bad"/"best"/"worst"/"robust" judgements, and NO parameter
  recommendations anywhere.

DISCLAIMER: Historical backtest with virtual money. Descriptive results
do not predict future performance and are not trading advice.
Educational use only.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read market data + symbol info (READ-ONLY usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # numeric helpers

from indicators import (
    add_adx,
    add_atr,
    add_bollinger,
)

# ============================================================
# SECTION 2: Settings - frozen, NOT optimized, IDENTICAL everywhere
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 10_000   # Requested history per instrument (up to 10,000 H1)
MIN_CANDLES = 400      # Absolute floor to attempt a four-period split;
                       # anything below 10,000 is still USED and REPORTED.
PREV_WINDOW = 4_000    # Candle count of the previous test's window, used
                       # ONLY for the descriptive comparison section.

INSTRUMENTS = ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]  # exactly these

STARTING_BALANCE = 10_000.0     # Independent for EVERY instrument AND period
LOTS_REQUESTED = 0.10
CONTRACT_SIZE_FALLBACK = 100_000
COST_PER_TRADE = 2.0            # $2.00 per completed trade (fixed)

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.5

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0
ATR_PERIOD = 14
WARMUP = 60              # Bars skipped before scanning (previous convention)

NUM_PERIODS = 4          # Chronological quartile split


# ============================================================
# SECTION 3: Indicator functions (causal only)
# The core indicators (add_atr, add_bollinger, add_adx) are imported
# from the shared, pandas/numpy-only indicators.py module (extracted
# verbatim; no formula, smoothing, min_periods, ddof, column-name or
# NaN behaviour change).
# ============================================================


# ============================================================
# SECTION 4: Symbol discovery (handles broker suffixes safely)
# The requested instrument is never silently substituted; if nothing
# matches, the instrument is reported NOT AVAILABLE and the run
# continues with the remaining instruments.
# ============================================================
def discover_symbol(requested):
    """Find the actual MT5 symbol for a requested instrument.

    1. Try the exact name first.
    2. Otherwise match symbols that START WITH the requested name and
       are followed only by a short separator suffix (e.g. GOLDm,
       EURUSD., GBPUSDmicro).
    Returns (actual_symbol_name, note) or (None, reason).
    """
    info = mt5.symbol_info(requested)
    if info is not None:
        return requested, "exact symbol name"

    matches = []
    all_syms = mt5.symbols_get() or []
    for sym in all_syms:
        name = sym.name
        if name.startswith(requested) and len(name) > len(requested):
            suffix = name[len(requested):]
            if len(suffix) <= 10 and suffix.replace(".", "").isalnum():
                matches.append(name)

    if not matches:
        return None, "no exact or suffix match found"

    if len(matches) > 1:
        # Deterministic, NOT performance-based: shortest name wins,
        # ties broken alphabetically. The choice is PRINTED, never hidden.
        matches.sort(key=lambda s: (len(s), s))

    chosen = matches[0]
    note = f"suffix match (available: {', '.join(matches[:5])})"
    return chosen, note


def get_contract_spec(symbol_name):
    """Read contract specs from MT5 (read-only) with DOCUMENTED fallbacks."""
    info = mt5.symbol_info(symbol_name)
    if info is None:
        return None

    contract_size = float(info.trade_contract_size) if info.trade_contract_size > 0 \
        else CONTRACT_SIZE_FALLBACK
    if info.trade_contract_size <= 0:
        print(f"    WARNING: trade_contract_size missing for {symbol_name}; "
              f"using fallback {CONTRACT_SIZE_FALLBACK} (documented fallback)")

    point = float(info.point)
    tick_size = float(info.trade_tick_size) if info.trade_tick_size > 0 else point
    tick_value = float(info.trade_tick_value)

    fallback_used = tick_value <= 0
    if fallback_used:
        # Documented fallback: quote-currency value of one tick per lot
        # = contract_size * point.
        tick_value = contract_size * point
        print(f"    WARNING: trade_tick_value missing/zero for {symbol_name}; "
              f"using fallback tick_value = contract_size x point = {tick_value:g}")

    # $ per 1.0 price-unit move per 1.0 lot (tick_value is $ per tick per
    # lot, tick_size is the price change per tick).
    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0

    quote_currency = getattr(info, "currency_profit", "") or "USD"

    # If the fallback was used AND the quote currency is not USD, try to
    # convert value_per_unit into USD using a <QUOTE>USD symbol bid
    # (documented fallback, printed - never hidden).
    converted = False
    if fallback_used and value_per_unit > 0 and quote_currency.upper() not in ("USD", ""):
        conv_symbol, _ = discover_symbol(f"{quote_currency.upper()}USD")
        if conv_symbol is not None:
            tick = mt5.symbol_info_tick(conv_symbol)
            if tick is not None and tick.bid > 0:
                value_per_unit = value_per_unit / tick.bid
                converted = True
                print(f"    Fallback P/L converted from {quote_currency.upper()} "
                      f"to USD via {conv_symbol} bid")

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
        "quote_currency": quote_currency,
        "tick_value_fallback": fallback_used,
        "fallback_converted": converted,
    }


def validate_volume(spec):
    """Return the largest valid volume <= LOTS_REQUESTED for this symbol, or None.

    The volume is never increased above 0.10 lots and never changed
    based on performance.
    """
    vmin = spec["volume_min"]
    vmax = spec["volume_max"]
    vstep = spec["volume_step"]

    if LOTS_REQUESTED < vmin:
        return None
    if vmin > vmax:
        return None

    steps_below = int((LOTS_REQUESTED - vmin) / vstep + 1e-9)  # tolerate float noise
    volume = round(vmin + steps_below * vstep, 8)
    if volume > vmax:
        return None
    return volume


# ============================================================
# SECTION 5: The frozen backtest engine - identical to the previous
# scripts (only value_per_unit and the volume come from each symbol's
# own specification)
# ============================================================
def run_backtest(df, value_per_unit, volume, start_bar=WARMUP):
    """Frozen-entry / frozen-exit virtual backtest.

    Signal on closed candle i, entry at the OPEN of candle i+1, SL/TP
    fixed from the signal candle's ATR, stop-first inside-candle
    handling, one position at a time, no overlapping positions, $2 cost
    per completed trade. The virtual balance always starts from
    STARTING_BALANCE (independent per call, i.e. per instrument AND per
    period). Returns (trades DataFrame, final balance).
    """

    long_condition = df["close"] < df["BB_LOWER"]
    short_condition = df["close"] > df["BB_UPPER"]
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df = df.copy()
    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE

    i = start_bar
    while i < len(df) - 1:
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1                          # Next candle
            entry_price = df["open"].iloc[entry_bar]   # Enter at its open

            if signal == "BUY":
                stop_loss = entry_price - SL_ATR_MULT * atr
                take_profit = entry_price + TP_ATR_MULT * atr
            else:
                stop_loss = entry_price + SL_ATR_MULT * atr
                take_profit = entry_price - TP_ATR_MULT * atr

            # Forward scan for the exit; STOP checked FIRST (conservative):
            # if both SL and TP occur inside the same candle, SL is assumed.
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
                else:
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

            # --- Virtual P/L from the INSTRUMENT's own specifications ---
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
                    "signal_time": df["time"].iloc[i],
                    "entry_time": df["time"].iloc[entry_bar],
                    "entry_price": entry_price,
                    "exit_time": df["time"].iloc[exit_index],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": net_pnl,
                    "balance_after": balance,
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 6: Period splitting and descriptive helpers (pure functions
# - no MT5, no optimization anywhere)
# ============================================================
def split_periods(df, num_periods=NUM_PERIODS):
    """Split a closed-candle DataFrame into N approximately equal
    CHRONOLOGICAL quartiles (simple slicing - no shuffling, no tuning).

    Returns (list of period DataFrames, list of boundary indices).
    """
    n = len(df)
    edges = [round(k * n / num_periods) for k in range(num_periods + 1)]
    periods = [df.iloc[edges[p]:edges[p + 1]].copy().reset_index(drop=True)
               for p in range(num_periods)]
    return periods, edges


def run_frozen_four_periods(df, value_per_unit, volume):
    """Run the frozen backtest on four chronological quartiles.

    Period 1 skips the indicator warm-up bars; later periods start at
    bar 0 (their indicators were already warmed up earlier in the
    series, and each period restarts from its own $10,000 balance).
    Returns (list of trade DataFrames, list of period frames).
    """
    periods, _ = split_periods(df)
    trades_list = []
    for p, sub in enumerate(periods, 1):
        tr, _ = run_backtest(sub, value_per_unit, volume,
                             start_bar=(WARMUP if p == 1 else 0))
        trades_list.append(tr)
    return trades_list, periods


def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def longest_losing_streak(trades):
    """Longest run of CONSECUTIVE losing trades (pnl < 0) in trade order."""
    if trades is None or len(trades) == 0:
        return 0
    streak = 0
    max_streak = 0
    for v in trades["pnl"]:
        if v < 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return max_streak


def period_stats(trades):
    """Descriptive stats for one period (or any trade list).

    Returns a dict; empty trade lists produce NaN fields (except the
    drawdown and streak, which are 0 by definition).
    """
    if trades is None or len(trades) == 0:
        return {"trades": 0, "return": float("nan"), "win_rate": float("nan"),
                "pf": float("nan"), "avg_trade": float("nan"),
                "total_pnl": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0,
                "ending_balance": STARTING_BALANCE, "lose_streak": 0}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())

    balances = [STARTING_BALANCE] + trades["balance_after"].tolist()
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

    return {
        "trades": len(trades),
        "return": (trades["balance_after"].iloc[-1] - STARTING_BALANCE) / STARTING_BALANCE * 100,
        "win_rate": len(wins) / len(trades) * 100,
        "pf": gp / gl if gl > 0 else float("inf"),
        "avg_trade": trades["pnl"].mean(),
        "total_pnl": trades["pnl"].sum(),
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "ending_balance": trades["balance_after"].iloc[-1],
        "lose_streak": longest_losing_streak(trades),
    }


def money(v):
    """Format a money value, handling 'no trades' (NaN)."""
    return "no trades" if (v != v) else f"${v:+.2f}"


def num(v, spec="{:.2f}"):
    """Format a number, printing n/a for NaN."""
    return "n/a" if (v != v) else spec.format(v)


def sign_changes(vals):
    """Period-to-period sign transitions in a list of values.

    Returns labels like 'P1->P2' for each transition where the sign
    flips between two non-NaN, non-zero values (factual counts only).
    """
    changes = []
    for p in range(len(vals) - 1):
        a, b = vals[p], vals[p + 1]
        if a == a and b == b and a != 0 and b != 0:
            if (a > 0) != (b > 0):
                changes.append(f"P{p + 1}->P{p + 2}")
    return changes


# ============================================================
# SECTION 7: Factual robustness questions (pure function - the answers
# are computed ONLY from the collected result rows; no rankings, no
# scores, no recommendations)
# ============================================================
def factual_answers(rows):
    """Compute the eight Section 6 answers from per-instrument rows.

    Each row: {instrument, returns[4], pfs[4], avg_trades[4], trades[4]}.
    """
    facts = {}

    # Q1: profitable in all four periods?
    q1 = {}
    for row in rows:
        rets = row["returns"]
        valid = [r for r in rets if r == r]
        if len(valid) == NUM_PERIODS and all(r > 0 for r in valid):
            ans = "YES - positive return in all four periods"
        elif len(valid) < NUM_PERIODS:
            ans = "INCOMPLETE - one or more periods had no trades"
        else:
            neg = [str(p + 1) for p, r in enumerate(rets) if r == r and r <= 0]
            ans = f"NO - non-positive return in period(s) {', '.join(neg)}"
        q1[row["instrument"]] = {"answer": ans, "returns": rets}
    facts["q1"] = q1

    # Q2: instruments with at least one negative period
    facts["q2"] = sum(
        1 for row in rows
        if any(r == r and r < 0 for r in row["returns"])
    )

    # Q3: instruments with negative Period 3
    facts["q3"] = sum(
        1 for row in rows
        if row["returns"][2] == row["returns"][2] and row["returns"][2] < 0
    )

    # Q4: instruments with negative Period 4
    facts["q4"] = sum(
        1 for row in rows
        if row["returns"][3] == row["returns"][3] and row["returns"][3] < 0
    )

    # Q5: instruments with a period-to-period return sign change
    facts["q5"] = sum(
        1 for row in rows if len(sign_changes(row["returns"])) > 0
    )

    # Q6: instruments with PF > 1 in all four periods
    q6_list = []
    for row in rows:
        pfs = row["pfs"]
        if all(pf == pf and pf > 1.0 for pf in pfs):
            q6_list.append(row["instrument"])
    facts["q6"] = q6_list

    # Q7: instruments with positive average $/trade in all four periods
    q7_list = []
    for row in rows:
        avgs = row["avg_trades"]
        if all(a == a and a > 0 for a in avgs):
            q7_list.append(row["instrument"])
    facts["q7"] = q7_list

    # Q8: factual observations about time instability in the longer window
    neg_counts = {}
    for row in rows:
        n_neg = sum(1 for r in row["returns"] if r == r and r < 0)
        neg_counts[row["instrument"]] = n_neg
    facts["q8"] = {
        "neg_period_distribution": {},
        "sign_changes": {row["instrument"]: sign_changes(row["returns"])
                         for row in rows},
        "neg_counts": neg_counts,
    }
    for n in (0, 1, 2, 3, 4):
        facts["q8"]["neg_period_distribution"][n] = \
            sum(1 for v in neg_counts.values() if v == n)

    return facts


def print_factual_answers(facts):
    """Print the eight factual robustness questions and their answers."""

    print()
    print("1. Does each instrument remain profitable in all four periods?")
    for name, item in facts["q1"].items():
        print(f"   {name:<12}: {item['answer']}")
        print(f"   {'':<12}  returns: "
              + " | ".join(num(r, "{:+.2f}%") for r in item["returns"]))

    print()
    print(f"2. Instruments with at least one negative period: {facts['q2']}")
    print(f"3. Instruments with negative Period 3 results:    {facts['q3']}")
    print(f"4. Instruments with negative Period 4 results:    {facts['q4']}")
    print(f"5. Instruments with a period-to-period return "
          f"sign change: {facts['q5']}")
    print(f"6. Instruments with PF > 1 in all four periods:   "
          f"{len(facts['q6'])}"
          + (f" ({', '.join(facts['q6'])})" if facts["q6"] else ""))
    print(f"7. Instruments with positive average $/trade in all "
          f"four periods: {len(facts['q7'])}"
          + (f" ({', '.join(facts['q7'])})" if facts["q7"] else ""))

    print()
    print("8. Does the longer historical window show the same type of time")
    print("   instability observed in the previous 4,000-candle test?")
    print("   Factual observations from THIS window only:")
    dist = facts["q8"]["neg_period_distribution"]
    for n in (0, 1, 2, 3, 4):
        print(f"   - instruments with {n} negative period(s): "
              f"{dist[n]}")
    for name, changes in facts["q8"]["sign_changes"].items():
        if changes:
            print(f"   - {name}: return sign changes at "
                  f"{', '.join(changes)}")
        else:
            print(f"   - {name}: no return sign change across periods")
    print("   (Whether this matches the previous test's pattern is a")
    print("    factual comparison - see the comparison section below.")
    print("    No stability judgement is made here.)")


# ============================================================
# SECTION 8: Main program (MT5 read-only)
# ============================================================
def main():
    print("Connecting to MetaTrader 5...")

    if not mt5.initialize():
        print("ERROR: Could not connect to MetaTrader 5.")
        print("Make sure the MT5 desktop terminal is installed, running,")
        print("and logged into your DEMO account.")
        print("Last error:", mt5.last_error())
        return

    print("Connected to MetaTrader 5 successfully!")

    try:
        print()
        print("=" * 70)
        print("MULTI-INSTRUMENT 10,000-CANDLE HISTORY TEST (descriptive only)")
        print("Frozen strategy, four chronological periods per instrument,")
        print("independent $10,000 per instrument AND per period, plus one")
        print("descriptive full-history run.")
        print("=" * 70)
        print()
        print("Timeframe: H1 | BB(20, 2.0) + ADX14 < 25 entry | "
              f"SL {SL_ATR_MULT} x ATR / TP {TP_ATR_MULT} x ATR exits")
        print(f"{LOTS_REQUESTED} lots (validated per symbol) | "
              f"${COST_PER_TRADE:.2f}/trade | next-open entry | stop-first")
        print("Nothing below is optimized, swept, selected or ranked.")

        rows = []             # per-instrument data for Sections 4-6
        unavailable = []      # instruments reported unavailable

        for requested in INSTRUMENTS:
            print()
            print("#" * 70)
            print(f"# INSTRUMENT: {requested}")
            print("#" * 70)

            try:
                # --- Symbol discovery (no substitution) ---------------
                actual_symbol, note = discover_symbol(requested)
                if actual_symbol is None:
                    print(f"STATUS: NOT AVAILABLE - {note}. "
                          f"Continuing with the remaining instruments.")
                    unavailable.append(requested)
                    continue

                print(f"Requested: {requested}  ->  using MT5 symbol: "
                      f"{actual_symbol}  ({note})")

                if not mt5.symbol_select(actual_symbol, True):
                    print(f"STATUS: NOT AVAILABLE - {actual_symbol} could not be "
                          f"enabled in Market Watch. Continuing with the "
                          f"remaining instruments.")
                    unavailable.append(requested)
                    continue

                # --- Broker specs + volume validation -----------------
                spec = get_contract_spec(actual_symbol)
                if spec is None:
                    print(f"STATUS: NOT AVAILABLE - no symbol_info for "
                          f"{actual_symbol}. Continuing with the remaining "
                          f"instruments.")
                    unavailable.append(requested)
                    continue

                volume = validate_volume(spec)
                if volume is None:
                    print(f"STATUS: SKIPPED - requested {LOTS_REQUESTED} lots is "
                          f"not a valid volume for {actual_symbol} "
                          f"(volume_min {spec['volume_min']}, volume_max "
                          f"{spec['volume_max']}, volume_step {spec['volume_step']}) "
                          f"and volumes above {LOTS_REQUESTED} lots are not "
                          f"allowed. Continuing with the remaining instruments.")
                    unavailable.append(requested)
                    continue

                # --- Download history (closed candles only) -----------
                print(f"Downloading up to {NUM_CANDLES} candles for "
                      f"{actual_symbol} (H1)...")
                rates = mt5.copy_rates_from_pos(actual_symbol, TIMEFRAME, 0, NUM_CANDLES)

                if rates is None or len(rates) == 0:
                    print(f"STATUS: NOT AVAILABLE - no historical data returned "
                          f"for {actual_symbol}. Continuing with the remaining "
                          f"instruments.")
                    unavailable.append(requested)
                    continue

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df = df.iloc[:-1].reset_index(drop=True)   # drop forming candle

                if len(df) < MIN_CANDLES:
                    print(f"STATUS: NOT AVAILABLE - only {len(df)} closed candles "
                          f"for {actual_symbol} (minimum {MIN_CANDLES}). "
                          f"Continuing with the remaining instruments.")
                    unavailable.append(requested)
                    continue

                # ======================================================
                # SECTION 1 - DATA
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 1 - DATA: {requested}")
                print("=" * 70)
                print(f"  symbol             : {actual_symbol} "
                      f"(requested '{requested}')")
                print(f"  actual closed candles: {len(df)}")
                if len(df) < NUM_CANDLES:
                    print(f"    NOTE: fewer than {NUM_CANDLES} closed candles "
                          f"available; using ALL {len(df)} available closed "
                          f"candles.")
                print(f"  first candle       : {df['time'].iloc[0]}")
                print(f"  last candle        : {df['time'].iloc[-1]}")
                print(f"  volume used        : {volume} lots (requested "
                      f"{LOTS_REQUESTED}; volume_min {spec['volume_min']}, "
                      f"volume_step {spec['volume_step']})")
                print(f"  contract size      : {spec['contract_size']:g}")
                print(f"  point              : {spec['point']:g}")
                print(f"  digits             : {spec['digits']}")
                print(f"  tick size          : {spec['tick_size']:g}")
                print(f"  tick value         : {spec['tick_value']:g}")
                pl_note = ""
                if spec["tick_value_fallback"]:
                    pl_note = " (documented fallback tick_value = contract x point"
                    pl_note += ", converted to USD" if spec["fallback_converted"] else ""
                    pl_note += ")"
                print(f"  P/L basis          : 1.0 price-unit move with 1.0 lot "
                      f"= {spec['value_per_unit']:.2f} "
                      f"(tick_value / tick_size, quote currency "
                      f"{spec['quote_currency']}){pl_note}")

                # --- Indicators ONCE over the full closed series -------
                # (causal: every value uses only current + previous candles)
                df = add_atr(df, ATR_PERIOD)
                df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
                df = add_adx(df, ADX_PERIOD)

                # ======================================================
                # SECTION 2 - FOUR PERIOD PERFORMANCE
                # ======================================================
                trades_by_period, periods_df = run_frozen_four_periods(
                    df, spec["value_per_unit"], volume)

                print()
                print("=" * 70)
                print(f"SECTION 2 - FOUR PERIOD PERFORMANCE: {requested}")
                print("(chronological quartiles, independent $10,000 each)")
                print("=" * 70)

                period_returns = []
                period_pfs = []
                period_avgs = []
                for p, sub in enumerate(periods_df, 1):
                    st = period_stats(trades_by_period[p - 1])
                    period_returns.append(st["return"])
                    period_pfs.append(st["pf"])
                    period_avgs.append(st["avg_trade"])

                    print(f"\nPeriod {p}:")
                    print(f"  period start          : {sub['time'].iloc[0]}")
                    print(f"  period end            : {sub['time'].iloc[-1]}")
                    print(f"  candle count          : {len(sub)}")
                    if st["trades"] == 0:
                        print("  trades                : 0 (no trades in "
                              "this period)")
                        continue
                    print(f"  trades                : {st['trades']}")
                    print(f"  return                : {st['return']:+.2f}%")
                    print(f"  win rate              : {st['win_rate']:.1f}%")
                    print(f"  profit factor         : {fmt_pf(st['pf'])}")
                    print(f"  average $/trade       : ${st['avg_trade']:+.2f}")
                    print(f"  total P/L             : ${st['total_pnl']:+,.2f}")
                    print(f"  maximum drawdown $    : ${st['max_dd']:,.2f}")
                    print(f"  maximum drawdown %    : {st['max_dd_pct']:.2f}%")
                    print(f"  longest losing streak : {st['lose_streak']} trades")

                # ======================================================
                # SECTION 3 - FULL-HISTORY PERFORMANCE (descriptive only)
                # ======================================================
                full_trades, full_balance = run_backtest(df, spec["value_per_unit"],
                                                         volume, start_bar=WARMUP)
                fst = period_stats(full_trades)

                print()
                print("=" * 70)
                print(f"SECTION 3 - FULL-HISTORY PERFORMANCE: {requested}")
                print("(ONE continuous descriptive run over the entire window;")
                print(" it is NOT used to select or modify the strategy)")
                print("=" * 70)
                print(f"  starting balance    : ${STARTING_BALANCE:,.2f}")
                print(f"  ending balance      : ${fst['ending_balance']:,.2f}")
                print(f"  total return        : {fst['return']:+.2f}%")
                if fst["trades"] == 0:
                    print("  trades              : 0 (no trades in the full "
                          "window)")
                else:
                    print(f"  trades              : {fst['trades']}")
                    print(f"  win rate            : {fst['win_rate']:.1f}%")
                    print(f"  profit factor       : {fmt_pf(fst['pf'])}")
                    print(f"  average $/trade     : ${fst['avg_trade']:+.2f}")
                    print(f"  maximum drawdown $  : ${fst['max_dd']:,.2f}")
                    print(f"  maximum drawdown %  : {fst['max_dd_pct']:.2f}%")
                print()
                print("  NOTE: the full-history run is one continuous backtest.")
                print("  Its trade list differs from the sum of the four period")
                print("  runs because the period runs force-close positions at")
                print("  their boundaries and restart from $10,000. Both views")
                print("  are descriptive only.")

                rows.append({
                    "instrument": requested,
                    "returns": period_returns,
                    "pfs": period_pfs,
                    "avg_trades": period_avgs,
                    "sign_changes": sign_changes(period_returns),
                    "_df": df,                   # for the previous-window recompute
                    "_vpu": spec["value_per_unit"],
                    "_volume": volume,
                })

            except Exception as exc:  # never let one instrument kill the run
                print(f"STATUS: ERROR while processing {requested}: {exc}")
                print("Continuing with the remaining instruments.")
                unavailable.append(requested)

        # ==========================================================
        # SECTION 4 - PERIOD RETURN TABLE
        # ==========================================================
        print()
        print("=" * 70)
        print("SECTION 4 - PERIOD RETURN TABLE (%)")
        print("=" * 70)
        if rows:
            hdr = (f"   {'Instrument':<12}" +
                   "".join(f"{'P' + str(p):>10}" for p in range(1, NUM_PERIODS + 1)))
            print(hdr)
            print("   " + "-" * (12 + 10 * NUM_PERIODS))
            for row in rows:
                line = f"   {row['instrument']:<12}"
                for r in row["returns"]:
                    line += f"{num(r, '{:+.2f}'):>10}"
                print(line)
            print("\n   'n/a' = no trades in that period. Each period ran on")
            print("   its own independent $10,000 balance.")
        else:
            print("   No instrument could be diagnosed.")

        # ==========================================================
        # SECTION 5 - PERIOD PROFIT FACTOR TABLE
        # ==========================================================
        print()
        print("=" * 70)
        print("SECTION 5 - PERIOD PROFIT FACTOR TABLE")
        print("=" * 70)
        if rows:
            hdr = (f"   {'Instrument':<12}" +
                   "".join(f"{'P' + str(p):>10}" for p in range(1, NUM_PERIODS + 1)))
            print(hdr)
            print("   " + "-" * (12 + 10 * NUM_PERIODS))
            for row in rows:
                line = f"   {row['instrument']:<12}"
                for pf in row["pfs"]:
                    line += f"{('n/a' if pf != pf else fmt_pf(pf)):>10}"
                print(line)
            print("\n   'inf' = no losing trades in that period; 'n/a' = no")
            print("   trades in that period.")
        else:
            print("   No instrument could be diagnosed.")

        # ==========================================================
        # SECTION 6 - FACTUAL ROBUSTNESS QUESTIONS
        # ==========================================================
        if rows:
            print()
            print("=" * 70)
            print("SECTION 6 - FACTUAL ROBUSTNESS QUESTIONS")
            print("(computed only from the tables above; no rankings, no scores,")
            print(" no 'good'/'bad'/'best'/'worst', no recommendations)")
            print("=" * 70)
            print_factual_answers(factual_answers(rows))

            # ------------------------------------------------------
            # COMPARISON WITH PREVIOUS TEST (factual counts only)
            # ------------------------------------------------------
            print()
            print("=" * 70)
            print("COMPARISON WITH PREVIOUS TEST - period-to-period return")
            print("sign changes only (factual counts, no quality judgement)")
            print("=" * 70)
            print("Previous-window recompute: the SAME frozen strategy on the")
            print(f"LATEST ~{PREV_WINDOW:,} closed candles of the current download")
            print("(the same window size the previous 4,000-candle test used),")
            print("split into four chronological periods with independent")
            print("$10,000 balances. Recomputed here so both numbers come from")
            print("identical code - nothing is copied or invented.")
            print()
            n_sc_prev = 0
            n_sc_new = 0
            for row in rows:
                # latest PREV_WINDOW closed candles (or all if fewer)
                sub_full = row.get("_df")
                if sub_full is None:   # defensive; should not happen
                    print(f"   {row['instrument']:<12}: window data unavailable")
                    continue
                window = sub_full.iloc[-PREV_WINDOW:] if len(sub_full) >= PREV_WINDOW \
                    else sub_full
                prev_trades, _ = run_frozen_four_periods(window,
                                                         row["_vpu"], row["_volume"])
                prev_returns = [period_stats(t)["return"] for t in prev_trades]
                sc_prev = sign_changes(prev_returns)
                sc_new = row["sign_changes"]
                if len(sc_prev):
                    n_sc_prev += 1
                if len(sc_new):
                    n_sc_new += 1
                print(f"   {row['instrument']:<12}")
                print(f"     ~{PREV_WINDOW:,}-candle window ({len(window):,} candles): "
                      f"{', '.join(sc_prev) if sc_prev else 'no sign change'}")
                print(f"     up-to-{NUM_CANDLES:,}-candle window ({len(row['_df']):,} candles): "
                      f"{', '.join(sc_new) if sc_new else 'no sign change'}")
            print()
            print(f"   Instruments with >=1 sign change in the 4,000-candle "
                  f"window : {n_sc_prev}")
            print(f"   Instruments with >=1 sign change in the 10,000-candle "
                  f"window: {n_sc_new}")
            print()
            print("   Factual counts only. The two windows overlap (the recent")
            print("   4,000 candles are part of the 10,000-candle window), so")
            print("   differences come from the ADDED older history and the")
            print("   different period boundaries - both are descriptive")
            print("   observations, not quality judgements.")

        if unavailable:
            seen = []
            for ins in INSTRUMENTS:
                if ins in unavailable and ins not in seen:
                    seen.append(ins)
            print("\nInstruments NOT diagnosed (reported unavailable/skipped,")
            print(f"no substitution made): {', '.join(seen)}")

        print()
        print("Historical backtest with virtual money. Descriptive results do")
        print("not predict future performance and are not trading advice.")
        print("No parameters were optimized, no instrument was ranked, and no")
        print("strategy change is recommended. Educational use only.")

    finally:
        # ==========================================================
        # Always disconnect when the script finishes
        # ==========================================================
        mt5.shutdown()
        print()
        print("MetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()
