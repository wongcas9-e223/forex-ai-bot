"""
multi_instrument_rolling_diagnostic.py - RESEARCH/DIAGNOSTIC ONLY.

Does the period-to-period instability observed on the frozen GOLD
mean-reversion strategy (BB(20, 2.0) + ADX14 < 25, SL 2.0 x ATR /
TP 1.5 x ATR) appear on OTHER instruments too, or is it specific
to GOLD?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only diagnostic.
- This script NEVER places, modifies, or closes any trade. It contains
  no order execution, order checking, position, or account functions.
- It only reads candles and symbol specifications from MetaTrader 5,
  using the same read-only calls as the previous scripts:
  initialize, shutdown, symbols_get, symbol_info, symbol_info_tick,
  copy_rates_from_pos (plus symbol_select to enable a symbol in
  Market Watch - also read-only, no trading action).

NO OPTIMIZATION OF ANY KIND:
- The strategy is FROZEN and IDENTICAL on every instrument:
    Timeframe : H1
    Indicators: Bollinger Bands (20, 2.0), ADX14, ATR14
    Entry     : LONG when close < lower band AND ADX < 25
                SHORT when close > upper band AND ADX < 25
                (signal on closed candle i, entry at open of candle i+1)
    Exits     : SL = 2.0 x ATR at entry, TP = 1.5 x ATR at entry
                (if both touched inside one candle, STOP is hit first)
    Position  : fixed 0.10 lots, $2.00 cost per completed trade
  Nothing is optimized, nothing is swept, nothing is selected, no
  randomization, no shuffling, no sampling. Every period starts from
  its own independent $10,000 virtual balance.

BROKER-SPEC P/L:
  P/L uses each MT5 symbol's own specifications (contract size, tick
  size, tick value, point, digits) - never a GOLD-style assumption for
  forex pairs. The strategy logic is identical; only the monetary value
  of one price-unit move comes from the broker per instrument.

INSTRUMENTS (exactly these five, in this order):
  GOLD, EURUSD, GBPUSD, USDJPY, AUDUSD
  If one is unavailable it is REPORTED as unavailable and the run
  continues with the remaining instruments. No substitution.

OUTPUT IS DESCRIPTIVE ONLY:
  Sections 1-8 are factual tables. Section 9 answers ten factual
  robustness questions computed from those tables. There are NO
  rankings, NO scores, NO "best"/"worst" instrument, NO recommendations,
  NO filters, NO parameter changes, and NO trading advice anywhere.

DISCLAIMER: Historical backtest with virtual money. Descriptive
diagnostics do not guarantee future performance and are not trading
advice. Educational use only.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read market data + symbol info (READ-ONLY usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # numeric helpers

# ============================================================
# SECTION 2: Settings - frozen, NOT optimized, IDENTICAL everywhere
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 4000     # Requested history per instrument (~4000 H1 candles)
MIN_CANDLES = 2000     # Minimum closed candles required per instrument

INSTRUMENTS = ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]  # exactly these

STARTING_BALANCE = 10_000.0     # Independent for EVERY instrument AND period
LOTS_REQUESTED = 0.10
CONTRACT_SIZE_FALLBACK = 100_000
COST_PER_TRADE = 2.0            # $2.00 per completed trade (fixed, not rescaled)

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.5

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0
ATR_PERIOD = 14
EMA_TREND_PERIOD = 50    # Diagnostic reference ONLY (Section 7), never a filter
WARMUP = 60              # Bars skipped before scanning (previous convention)

HOLD_WINDOW = 12         # Candles for forward return and MFE/MAE (Section 5/6)

NUM_PERIODS = 4          # Chronological quartile split
LOW_SAMPLE_N = 5         # Cells below this trade count get the LOW SAMPLE tag
LOW_SAMPLE_TAG = "LOW SAMPLE - DESCRIPTIVE ONLY"


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


def add_ema(df, period):
    """Exponential Moving Average of the close price."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


# ============================================================
# SECTION 4: Symbol discovery (handles broker suffixes safely)
# The requested instrument is never silently substituted by a
# different underlying symbol; if nothing matches, the instrument
# is reported NOT AVAILABLE and the run continues.
# ============================================================
def discover_symbol(requested):
    """Find the actual MT5 symbol for a requested instrument.

    1. Try the exact name first.
    2. Otherwise match symbols that START WITH the requested name and
       are followed only by a short separator suffix (e.g. GOLDm,
       EURUSD., GBPUSDmicro). Underlying-instrument changes are never
       auto-substituted.
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
    """Read contract specs from MT5 (read-only) with DOCUMENTED fallbacks.

    Returns a dict with contract size, tick size/value, point, digits,
    value_per_unit ($ per 1.0 price-unit move per 1.0 lot), volume rules
    and the profit (quote) currency.
    """
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

    # $ per 1.0 price-unit move per 1.0 lot (tick_value is $ per tick per lot,
    # tick_size is the price change per tick).
    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0

    quote_currency = getattr(info, "currency_profit", "") or "USD"

    # If the fallback was used AND the quote currency is not USD, try to
    # convert value_per_unit into USD using a <QUOTE>USD symbol bid, so
    # P/L stays comparable across instruments (documented fallback).
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
# SECTION 5: The frozen backtest engine with an attribute-rich trade
# log - identical for every instrument (only value_per_unit and the
# volume come from the broker's own symbol specification)
# ============================================================
def run_backtest(df, value_per_unit, volume, start_bar=WARMUP):
    """Frozen-entry / frozen-exit virtual backtest.

    Signal on closed candle i, entry at the open of candle i+1, SL/TP
    fixed from the signal candle's ATR, stop-first inside-bar handling,
    one position at a time, $2 cost per completed trade. The virtual
    balance always starts from STARTING_BALANCE (independent per call,
    i.e. per instrument AND per period).
    Returns (trades DataFrame, final balance).
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

            # Forward scan for the exit; STOP checked FIRST (conservative)
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

            # --- Signal-time attributes (known at the signal close) ---
            adx_sig = df["ADX"].iloc[i]
            atr_sig = df["ATR"].iloc[i]
            close_sig = df["close"].iloc[i]
            band_width = df["BB_UPPER"].iloc[i] - df["BB_LOWER"].iloc[i]
            excursion = (abs(close_sig - df["BB_MIDDLE"].iloc[i]) / band_width
                         if pd.notna(band_width) and band_width > 0 else np.nan)
            ema50 = df["EMA50"].iloc[i]
            trend_dist = (abs(close_sig - ema50) / atr_sig
                          if pd.notna(ema50) and atr_sig and atr_sig > 0 else np.nan)
            atr_pct = atr_sig / close_sig * 100 if close_sig > 0 else np.nan

            # --- 12-candle forward return (INDEPENDENT of the exit) ---
            fwd_idx = min(entry_bar + HOLD_WINDOW - 1, len(df) - 1)
            close_fwd = df["close"].iloc[fwd_idx]
            if signal == "BUY":
                fwd_ret = (close_fwd - entry_price) / entry_price
            else:
                fwd_ret = (entry_price - close_fwd) / entry_price

            # --- MFE / MAE over the next 12 candles from ENTRY price ---
            mfe = 0.0
            mae = 0.0
            for j in range(entry_bar, min(entry_bar + HOLD_WINDOW, len(df))):
                move_up = df["high"].iloc[j] - entry_price
                move_dn = entry_price - df["low"].iloc[j]
                mfe = max(mfe, move_up if signal == "BUY" else move_dn)
                mae = max(mae, move_dn if signal == "BUY" else move_up)
            mfe_atr = mfe / atr_sig if atr_sig > 0 else np.nan
            mae_atr = mae / atr_sig if atr_sig > 0 else np.nan

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
                    "ADX": adx_sig,
                    "ATR_pct": atr_pct,
                    "excursion": excursion,
                    "trend_dist": trend_dist,
                    "fwd_ret_12": fwd_ret,
                    "MFE_ATR": mfe_atr,
                    "MAE_ATR": mae_atr,
                    "pnl": net_pnl,
                    "balance_after": balance,
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 6: Period splitting, volatility edges and descriptive
# helpers (pure functions - no MT5, no optimization anywhere)
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


def compute_vol_edges(trades_by_period):
    """33rd/66th percentile ATR% edges from the FULL signal dataset.

    Computed ONCE from all periods combined - never per period, never
    optimized. Returns (v1, v2).
    """
    all_signals = pd.concat(trades_by_period, ignore_index=True)
    v1 = all_signals["ATR_pct"].quantile(1 / 3)
    v2 = all_signals["ATR_pct"].quantile(2 / 3)
    return v1, v2


def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def period_stats(trades):
    """Descriptive stats for one instrument-period. Returns a dict."""
    if trades is None or len(trades) == 0:
        return {"trades": 0, "return": float("nan"), "win_rate": float("nan"),
                "pf": float("nan"), "avg_trade": float("nan"),
                "total_pnl": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0}
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
    }


def cell_stats(sub):
    """Descriptive stats for one cell. Returns a dict or None."""
    if sub is None or len(sub) == 0:
        return None
    wins = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] < 0]
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    return {
        "n": len(sub),
        "win_rate": len(wins) / len(sub) * 100,
        "avg_trade": sub["pnl"].mean(),
        "total_pnl": sub["pnl"].sum(),
        "pf": gp / gl if gl > 0 else float("inf"),
    }


def print_cell(period_label, sub, with_pf=False):
    """Print one cell with its sample-size tag if needed."""
    s = cell_stats(sub)
    if s is None:
        print(f"      {period_label:<20} (no trades)")
        return
    tag = f"  [{LOW_SAMPLE_TAG}]" if s["n"] < LOW_SAMPLE_N else ""
    line = (f"      {period_label:<20} {s['n']:>3} trades | "
            f"win {s['win_rate']:>5.1f}% | avg ${s['avg_trade']:>8.2f} | "
            f"total ${s['total_pnl']:>9.2f}")
    if with_pf:
        line += f" | PF {fmt_pf(s['pf']):>5}"
    print(line + tag)


def fwd_line(sub):
    """Format one 12-candle forward-return line (with sample tag)."""
    if sub is None or len(sub) == 0:
        return "(no trades)"
    fwd = sub["fwd_ret_12"] * 100
    tag = f"  [{LOW_SAMPLE_TAG}]" if len(sub) < LOW_SAMPLE_N else ""
    return (f"n={len(sub):>3} | avg {fwd.mean():+.4f}% | "
            f"median {fwd.median():+.4f}% | "
            f"pos {(fwd > 0).mean() * 100:.1f}% / "
            f"neg {(fwd < 0).mean() * 100:.1f}%{tag}")


def mfe_mae_line(sub):
    """Format one MFE/MAE line (winners vs losers, with sample tag)."""
    if sub is None or len(sub) == 0:
        return "(no trades)"
    w = sub[sub["pnl"] > 0]
    l = sub[sub["pnl"] < 0]
    wtxt = (f"winners MFE {w['MFE_ATR'].mean():.2f} / MAE {w['MAE_ATR'].mean():.2f}"
            if len(w) else "winners (none)")
    ltxt = (f"losers MFE {l['MFE_ATR'].mean():.2f} / MAE {l['MAE_ATR'].mean():.2f}"
            if len(l) else "losers (none)")
    tag = f"  [{LOW_SAMPLE_TAG}]" if len(sub) < LOW_SAMPLE_N else ""
    return f"n={len(sub):>3} | {wtxt} | {ltxt}{tag}"


def money(v):
    """Format a money value, handling 'no trades' (NaN)."""
    return "no trades" if (v != v) else f"${v:+.2f}"


def num(v, spec="{:.2f}"):
    """Format a number, printing n/a for NaN."""
    return "n/a" if (v != v) else spec.format(v)


def condition_summary_rows(trades_by_period):
    """Section 7 rows: signal-time environment metrics per period.

    Returns a list of (metric_name, [4 values], float_format) tuples.
    Purely descriptive - never a filter.
    """
    def col(trades, fn):
        return fn(trades) if len(trades) else float("nan")

    rows = [
        ("Average ADX", lambda t: t["ADX"].mean(), "{:.1f}"),
        ("Median ADX", lambda t: t["ADX"].median(), "{:.1f}"),
        ("Average ATR%", lambda t: t["ATR_pct"].mean(), "{:.3f}"),
        ("Median ATR%", lambda t: t["ATR_pct"].median(), "{:.3f}"),
        ("Avg BB excursion", lambda t: t["excursion"].mean(), "{:.4f}"),
        ("Median BB excursion", lambda t: t["excursion"].median(), "{:.4f}"),
        ("Avg EMA50 dist (ATR)", lambda t: t["trend_dist"].mean(), "{:.2f}"),
        ("Median EMA50 dist", lambda t: t["trend_dist"].median(), "{:.2f}"),
        ("LONG %", lambda t: (t["direction"] == "BUY").mean() * 100, "{:.1f}%"),
        ("SHORT %", lambda t: (t["direction"] == "SELL").mean() * 100, "{:.1f}%"),
    ]
    out = []
    for name, fn, spec_fmt in rows:
        vals = [col(trades_by_period[p], fn) for p in range(NUM_PERIODS)]
        out.append((name, vals, spec_fmt))
    return out


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


def factual_answers(summary_rows, diagnostic_data):
    """Compute the ten Section 9 answers from the stored tables.

    Pure function: takes the per-instrument period rows (returns and
    PFs) and the per-instrument trade lists + tercile edges, and
    returns a dict with only factual counts/answers. No rankings, no
    scores, no recommendations.
    """
    facts = {}
    instruments_done = [row["instrument"] for row in summary_rows]
    trades_map = {name: d["trades_by_period"] for name, d in diagnostic_data.items()}
    edges_map = {name: d["vol_edges"] for name, d in diagnostic_data.items()}

    # Q1: profitable in all four periods?
    q1 = {}
    for row in summary_rows:
        rets = [row[f"return_p{p}"] for p in range(1, NUM_PERIODS + 1)]
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

    # Q2/Q3: instruments with negative Period 3 / Period 4 results
    def count_negative(p_idx):
        cnt = 0
        for row in summary_rows:
            r = row[f"return_p{p_idx}"]
            if r == r and r < 0:
                cnt += 1
        return cnt

    facts["q2"] = count_negative(3)
    facts["q3"] = count_negative(4)
    facts["q2_total"] = len(summary_rows)
    facts["q3_total"] = len(summary_rows)

    # Q4: high-volatility average trade positive in all four periods?
    q4 = {}
    for name in instruments_done:
        trades_list = trades_map[name]
        v2 = edges_map[name][1]
        vals = []
        for t in trades_list:
            hv = t[t["ATR_pct"] >= v2]
            vals.append(hv["pnl"].mean() if len(hv) else float("nan"))
        valid = [v for v in vals if v == v]
        if len(valid) == NUM_PERIODS and all(v > 0 for v in valid):
            ans = "YES"
        elif len(valid) < NUM_PERIODS:
            ans = "INCOMPLETE - one or more periods had no high-vol trades"
        else:
            neg = [str(p + 1) for p, v in enumerate(vals) if v == v and v <= 0]
            ans = f"NO - non-positive in period(s) {', '.join(neg)}"
        q4[name] = {"answer": ans, "vals": vals}
    facts["q4"] = q4

    # Q5: high-volatility 12-candle forward return positive in all periods?
    q5 = {}
    for name in instruments_done:
        trades_list = trades_map[name]
        v2 = edges_map[name][1]
        vals = []
        for t in trades_list:
            hv = t[t["ATR_pct"] >= v2]
            vals.append(hv["fwd_ret_12"].mean() * 100 if len(hv) else float("nan"))
        valid = [v for v in vals if v == v]
        if len(valid) == NUM_PERIODS and all(v > 0 for v in valid):
            ans = "YES"
        elif len(valid) < NUM_PERIODS:
            ans = "INCOMPLETE - one or more periods had no high-vol trades"
        else:
            neg = [str(p + 1) for p, v in enumerate(vals) if v == v and v <= 0]
            ans = f"NO - non-positive in period(s) {', '.join(neg)}"
        q5[name] = {"answer": ans, "vals": vals}
    facts["q5"] = q5

    # Q6: direction effect consistency across periods
    q6 = {}
    for name in instruments_done:
        trades_list = trades_map[name]
        long_avg = []
        short_avg = []
        for t in trades_list:
            ln = t[t["direction"] == "BUY"]
            sh = t[t["direction"] == "SELL"]
            long_avg.append(ln["pnl"].mean() if len(ln) else float("nan"))
            short_avg.append(sh["pnl"].mean() if len(sh) else float("nan"))
        # A direction "dominates" a period when its average trade is higher
        # than the other direction's (factual per-period fact).
        dominance = []
        for p in range(NUM_PERIODS):
            la, sa = long_avg[p], short_avg[p]
            if la != la or sa != sa:
                dominance.append("?")
            elif la > sa:
                dominance.append("LONG")
            elif sa > la:
                dominance.append("SHORT")
            else:
                dominance.append("EQUAL")
        signs = set(d for d in dominance if d not in ("?", "EQUAL"))
        if len(signs) <= 1:
            ans = "YES - the same direction dominates in every period"
            if len(signs) == 1:
                ans += f" ({signs.pop()})"
        else:
            ans = "NO - the dominant direction changes between periods"
        q6[name] = {"answer": ans, "long_avg": long_avg, "short_avg": short_avg}
    facts["q6"] = q6

    # Q7/Q8: period-to-period sign changes
    q7 = {}
    for row in summary_rows:
        rets = [row[f"return_p{p}"] for p in range(1, NUM_PERIODS + 1)]
        q7[row["instrument"]] = sign_changes(rets)
    facts["q7"] = q7

    q8 = {}
    for name in instruments_done:
        trades_list = trades_map[name]
        v2 = edges_map[name][1]
        vals = []
        for t in trades_list:
            hv = t[t["ATR_pct"] >= v2]
            vals.append(hv["pnl"].mean() if len(hv) else float("nan"))
        q8[name] = sign_changes(vals)
    facts["q8"] = q8

    # Q9: does the same instability appear across multiple instruments?
    n_ret_unstable = sum(1 for chg in q7.values() if len(chg) > 0)
    n_hv_unstable = sum(1 for chg in q8.values() if len(chg) > 0)
    facts["q9"] = {
        "n_ret_unstable": n_ret_unstable,
        "n_hv_unstable": n_hv_unstable,
        "n_neg_p3": facts["q2"],
        "n_neg_p4": facts["q3"],
        "n_total": len(summary_rows),
    }

    # Q10: instrument/period/volatility cells below the sample threshold
    low_cells = 0
    total_cells = 0
    for name in instruments_done:
        trades_list = trades_map[name]
        v1, v2 = edges_map[name]
        buckets = [(0.0, v1), (v1, v2), (v2, float("inf"))]
        for t in trades_list:
            for lo, hi in buckets:
                n_cell = len(t[(t["ATR_pct"] >= lo) & (t["ATR_pct"] < hi)])
                total_cells += 1
                if n_cell < LOW_SAMPLE_N:
                    low_cells += 1
    facts["q10"] = {"low_cells": low_cells, "total_cells": total_cells}

    return facts


def print_cross_instrument_summary(summary_rows, unavailable):
    """Section 8: factual cross-instrument table. No ranking, no scores."""
    print()
    print("=" * 70)
    print("SECTION 8 - CROSS-INSTRUMENT SUMMARY (factual table, no ranking)")
    print("=" * 70)

    if summary_rows:
        hdr = (f"   {'Instrument':<12}" +
               "".join(f"{'P' + str(p):>10}" for p in range(1, NUM_PERIODS + 1)))

        print("\nPeriod returns (%):")
        print(hdr)
        print("   " + "-" * (12 + 10 * NUM_PERIODS))
        for row in summary_rows:
            line = f"   {row['instrument']:<12}"
            for p in range(1, NUM_PERIODS + 1):
                line += f"{num(row[f'return_p{p}'], '{:+.2f}'):>10}"
            print(line)

        print("\nPeriod profit factors:")
        print(hdr)
        print("   " + "-" * (12 + 10 * NUM_PERIODS))
        for row in summary_rows:
            line = f"   {row['instrument']:<12}"
            for p in range(1, NUM_PERIODS + 1):
                pf = row[f"pf_p{p}"]
                line += f"{('n/a' if pf != pf else fmt_pf(pf)):>10}"
            print(line)

        print("\nEach period ran on its own independent $10,000 balance.")
        print("Instruments are deliberately NOT ranked; none is called")
        print("'best' or 'worst'. 'inf' PF means no losing trades in")
        print("that period; 'n/a' means no trades in that period.")
    else:
        print("\nNo instrument could be diagnosed (all unavailable, "
              "skipped, or produced no trades).")

    if unavailable:
        seen = []
        for ins in INSTRUMENTS:
            if ins in unavailable and ins not in seen:
                seen.append(ins)
        print("\nInstruments NOT diagnosed (reported unavailable/skipped,")
        print(f"no substitution made): {', '.join(seen)}")


# ============================================================
# SECTION 7: Printing of the per-question Section 9 output
# (formatting only - the facts come from factual_answers())
# ============================================================
def print_factual_answers(facts):
    """Print the ten factual robustness questions and their answers."""

    print()
    print("1. Does the strategy remain profitable in all four periods "
          "for each instrument?")
    for name, item in facts["q1"].items():
        print(f"   {name:<12}: {item['answer']}")
        print(f"   {'':<12}  returns: "
              + " | ".join(num(r, "{:+.2f}%") for r in item["returns"]))

    print()
    print(f"2. Instruments with negative results in Period 3: "
          f"{facts['q2']} of {facts['q2_total']}")
    print(f"3. Instruments with negative results in Period 4: "
          f"{facts['q3']} of {facts['q3_total']}")

    print()
    print("4. Does high-volatility average trade remain positive across "
          "all four periods for each instrument?")
    for name, item in facts["q4"].items():
        print(f"   {name:<12}: {item['answer']}")
        print(f"   {'':<12}  " + " | ".join(money(v) for v in item["vals"]))

    print()
    print("5. Does high-volatility 12-candle forward return remain "
          "positive across all four periods?")
    for name, item in facts["q5"].items():
        print(f"   {name:<12}: {item['answer']}")
        print(f"   {'':<12}  "
              + " | ".join("n/a" if v != v else f"{v:+.4f}%" for v in item["vals"]))

    print()
    print("6. Does the direction effect remain consistent across periods?")
    for name, item in facts["q6"].items():
        print(f"   {name:<12}: {item['answer']}")
        print(f"   {'':<12}  LONG  avg: " + " | ".join(money(v) for v in item["long_avg"]))
        print(f"   {'':<12}  SHORT avg: " + " | ".join(money(v) for v in item["short_avg"]))

    print()
    print("7. Which instruments show period-to-period sign changes in "
          "strategy return?")
    any_sc = False
    for name, changes in facts["q7"].items():
        if changes:
            any_sc = True
            print(f"   {name:<12}: {', '.join(changes)} "
                  f"({len(changes)} of {NUM_PERIODS - 1} transitions)")
        else:
            print(f"   {name:<12}: none")
    if not any_sc:
        print("   -> No instrument changes the sign of its period return.")

    print()
    print("8. Which instruments show period-to-period sign changes in "
          "high-volatility average trade?")
    any_sc = False
    for name, changes in facts["q8"].items():
        if changes:
            any_sc = True
            print(f"   {name:<12}: {', '.join(changes)} "
                  f"({len(changes)} of {NUM_PERIODS - 1} transitions)")
        else:
            print(f"   {name:<12}: none")
    if not any_sc:
        print("   -> No instrument changes the sign of its high-vol avg trade.")

    q9 = facts["q9"]
    print()
    print("9. Does the same instability appear across multiple instruments?")
    print(f"   Instruments with a return sign change across periods: "
          f"{q9['n_ret_unstable']} of {q9['n_total']}")
    print(f"   Instruments with a high-vol avg-trade sign change: "
          f"{q9['n_hv_unstable']} of {q9['n_total']}")
    print(f"   Instruments with negative Period 3 return: {q9['n_neg_p3']}")
    print(f"   Instruments with negative Period 4 return: {q9['n_neg_p4']}")
    if q9["n_ret_unstable"] > 1 or q9["n_hv_unstable"] > 1:
        print("   -> The sign-change instability appears on MORE THAN ONE")
        print("      instrument in this data window (factual count only).")
    elif q9["n_ret_unstable"] == 1 or q9["n_hv_unstable"] == 1:
        print("   -> The sign-change instability appears on exactly ONE")
        print("      instrument in this data window (factual count only).")
    else:
        print("   -> No instrument shows a return sign change in this")
        print("      data window (factual count only).")

    q10 = facts["q10"]
    print()
    print("10. How many instrument/period/volatility cells have fewer "
          "than 5 trades?")
    print(f"    {q10['low_cells']} of {q10['total_cells']} "
          f"instrument/period/volatility cells have fewer than "
          f"{LOW_SAMPLE_N} trades.")
    if q10["low_cells"]:
        print("    Every such cell is tagged LOW SAMPLE above and is")
        print("    descriptive only - small cells are NOT interpreted")
        print("    as meaningful evidence.")

    print()
    print("All ten answers above are counts and factual observations")
    print("from this single historical window. They do not recommend a")
    print("filter, a parameter change, or a strategy modification, and")
    print("they do not predict future performance.")


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
        print("MULTI-INSTRUMENT ROLLING DIAGNOSTIC (descriptive only)")
        print("Frozen strategy, four chronological periods per instrument,")
        print("independent $10,000 per instrument AND per period.")
        print("=" * 70)
        print()
        print("Timeframe: H1 | BB(20, 2.0) + ADX14 < 25 entry | "
              f"SL {SL_ATR_MULT} x ATR / TP {TP_ATR_MULT} x ATR exits")
        print(f"{LOTS_REQUESTED} lots (validated per symbol) | "
              f"${COST_PER_TRADE:.2f}/trade | next-open entry | stop-first")
        print("Nothing below is optimized, swept, selected or ranked.")

        # ------------------------------------------------------------
        # Data collection and per-instrument diagnostics
        # ------------------------------------------------------------
        summary_rows = []     # for Section 8 (per-instrument period returns/PFs)
        diagnostic_data = {}  # instrument -> trades/edges for Section 9
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

                print(f"Volume used: {volume} lots (requested {LOTS_REQUESTED}; "
                      f"volume_min {spec['volume_min']}, volume_step {spec['volume_step']})")
                print(f"Contract size: {spec['contract_size']:g} | point "
                      f"{spec['point']:g} | digits {spec['digits']} | "
                      f"tick size {spec['tick_size']:g} | tick value "
                      f"{spec['tick_value']:g} | quote currency "
                      f"{spec['quote_currency']}")
                print(f"P/L basis: 1.0 price-unit move with 1.0 lot = "
                      f"{spec['value_per_unit']:.2f} (from this symbol's own "
                      f"tick specs - NOT a GOLD-style assumption)")

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

                if len(rates) < NUM_CANDLES:
                    print(f"NOTE: broker returned {len(rates)} candles "
                          f"(requested {NUM_CANDLES}); using all {len(df)} "
                          f"closed ones.")

                print(f"Closed candles used: {len(df)} "
                      f"(from {df['time'].iloc[0]} to {df['time'].iloc[-1]})")

                # --- Indicators ONCE over the full closed series -------
                # (computed causally on the full history; each period then
                # reuses only the values of its own candles - no lookahead)
                df = add_atr(df, ATR_PERIOD)
                df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
                df = add_adx(df, ADX_PERIOD)
                df = add_ema(df, EMA_TREND_PERIOD)   # diagnostic reference only

                # --- Chronological quartile split (no tuning) ----------
                periods_df, _ = split_periods(df)
                print()
                print("-" * 70)
                print("CHRONOLOGICAL PERIODS (simple quartiles, independent "
                      "$10,000 each)")
                print("-" * 70)
                for p, sub in enumerate(periods_df, 1):
                    print(f"Period {p}: {sub['time'].iloc[0]} -> "
                          f"{sub['time'].iloc[-1]} ({len(sub)} candles)")

                # ======================================================
                # SECTION 1: OVERALL PERFORMANCE per period
                # ======================================================
                trades_by_period = []
                print()
                print("=" * 70)
                print(f"SECTION 1 - OVERALL PERFORMANCE: {requested}")
                print("=" * 70)
                for p, sub in enumerate(periods_df, 1):
                    trades, balance = run_backtest(sub, spec["value_per_unit"], volume,
                                                   start_bar=(WARMUP if p == 1 else 0))
                    trades_by_period.append(trades)

                    if len(trades) == 0:
                        print(f"\nPeriod {p}: NO TRADES in this period.")
                        continue

                    st = period_stats(trades)
                    print(f"\nPeriod {p}:")
                    print(f"  trades        : {st['trades']}")
                    print(f"  return        : {st['return']:+.2f}%")
                    print(f"  win rate      : {st['win_rate']:.1f}%")
                    print(f"  profit factor : {fmt_pf(st['pf'])}")
                    print(f"  avg $/trade   : ${st['avg_trade']:+.2f}")
                    print(f"  total P/L     : ${st['total_pnl']:+,.2f}")
                    print(f"  max DD        : ${st['max_dd']:,.2f} "
                          f"({st['max_dd_pct']:.2f}%)")

                # ======================================================
                # SECTION 2: VOLATILITY REGIMES (fixed tercile edges from
                # the FULL signal dataset of this instrument)
                # ======================================================
                if sum(len(t) for t in trades_by_period) == 0:
                    print(f"\nNOTE: no trades generated for {requested} in any "
                          f"period; instrument cannot be diagnosed further.")
                    unavailable.append(requested)
                    continue

                v1, v2 = compute_vol_edges(trades_by_period)
                vol_buckets = [(0.0, v1, "Low volatility"),
                               (v1, v2, "Medium volatility"),
                               (v2, float("inf"), "High volatility")]

                print()
                print("=" * 70)
                print(f"SECTION 2 - VOLATILITY REGIMES: {requested}")
                print("=" * 70)
                print(f"33rd percentile edge: {v1:.4f}%")
                print(f"66th percentile edge: {v2:.4f}%")
                print("(computed ONCE from the FULL signal dataset of all four")
                print(" periods combined - never per period, never optimized)")
                print("Every signal is classified LOW / MEDIUM / HIGH with these")
                print("fixed instrument-specific edges (descriptive grouping only):")
                all_signals = pd.concat(trades_by_period, ignore_index=True)
                print(f"   LOW    (ATR% <  {v1:.4f}): "
                      f"{int((all_signals['ATR_pct'] < v1).sum())} signals")
                print(f"   MEDIUM ({v1:.4f} - {v2:.4f}): "
                      f"{int(((all_signals['ATR_pct'] >= v1)
                              & (all_signals['ATR_pct'] < v2)).sum())} signals")
                print(f"   HIGH   (ATR% >= {v2:.4f}): "
                      f"{int((all_signals['ATR_pct'] >= v2).sum())} signals")

                # ======================================================
                # SECTION 3: PERIOD x VOLATILITY
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 3 - PERIOD x VOLATILITY: {requested}")
                print("=" * 70)
                for p, trades in enumerate(trades_by_period, 1):
                    print(f"\nPeriod {p}:")
                    for lo, hi, vlabel in vol_buckets:
                        print_cell(f"{vlabel}",
                                   trades[(trades["ATR_pct"] >= lo)
                                          & (trades["ATR_pct"] < hi)])

                # ======================================================
                # SECTION 4: PERIOD x DIRECTION
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 4 - PERIOD x DIRECTION: {requested}")
                print("=" * 70)
                for p, trades in enumerate(trades_by_period, 1):
                    print(f"\nPeriod {p}:")
                    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
                        print_cell(dname, trades[trades["direction"] == d])

                # ======================================================
                # SECTION 5: HIGH-VOLATILITY 12-CANDLE FORWARD RETURN
                # (exit-independent)
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 5 - HIGH-VOL {HOLD_WINDOW}-CANDLE FORWARD RETURN "
                      f"(exit-independent): {requested}")
                print("=" * 70)
                print(f"(high volatility = ATR% >= {v2:.4f}%, the fixed edge)")
                for p, trades in enumerate(trades_by_period, 1):
                    hv = trades[trades["ATR_pct"] >= v2]
                    print(f"\nPeriod {p}: {fwd_line(hv)}")
                    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
                        hv_d = hv[hv["direction"] == d]
                        print(f"   {dname:<6}: {fwd_line(hv_d)}")

                # ======================================================
                # SECTION 6: HIGH-VOLATILITY MFE / MAE (ATR-normalized)
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 6 - HIGH-VOL MFE / MAE (normalized by ATR at "
                      f"entry): {requested}")
                print("=" * 70)
                for p, trades in enumerate(trades_by_period, 1):
                    hv = trades[trades["ATR_pct"] >= v2]
                    print(f"\nPeriod {p}: {mfe_mae_line(hv)}")

                # ======================================================
                # SECTION 7: PERIOD CONDITION SUMMARY (environment only)
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 7 - PERIOD CONDITION SUMMARY: {requested}")
                print("=" * 70)

                header = (f"   {'Metric':<24}" +
                          "".join(f"{'P' + str(p):>10}"
                                  for p in range(1, NUM_PERIODS + 1)))
                print(header)

                def fmt_or_na(v, spec_fmt="{:.1f}"):
                    return "n/a" if (v != v) else spec_fmt.format(v)

                for name, vals, spec_fmt in condition_summary_rows(trades_by_period):
                    line = f"   {name:<24}"
                    for v in vals:
                        line += f"{fmt_or_na(v, spec_fmt):>10}"
                    print(line)

                print()
                print("(Signal-time environment metrics only - descriptive,")
                print(" never a filter recommendation.)")

                # ======================================================
                # Collect rows for Section 8 and Section 9
                # ======================================================
                row = {"instrument": requested}
                for p in range(NUM_PERIODS):
                    st = period_stats(trades_by_period[p])
                    row[f"return_p{p + 1}"] = st["return"]
                    row[f"pf_p{p + 1}"] = st["pf"]
                summary_rows.append(row)

                diagnostic_data[requested] = {
                    "trades_by_period": trades_by_period,
                    "vol_edges": (v1, v2),
                }

            except Exception as exc:  # never let one instrument kill the run
                print(f"STATUS: ERROR while processing {requested}: {exc}")
                print("Continuing with the remaining instruments.")
                unavailable.append(requested)

        # ==========================================================
        # SECTION 8: CROSS-INSTRUMENT SUMMARY (factual, no ranking)
        # ==========================================================
        print_cross_instrument_summary(summary_rows, unavailable)

        # ==========================================================
        # SECTION 10: FACTUAL ROBUSTNESS QUESTIONS (ten answers)
        # ==========================================================
        if summary_rows:
            print()
            print("=" * 70)
            print("SECTION 9 - FACTUAL ROBUSTNESS QUESTIONS")
            print("(computed only from the tables above; no rankings, no scores,")
            print(" no 'best'/'worst', no recommendations, no filters)")
            print("=" * 70)
            print_factual_answers(factual_answers(summary_rows, diagnostic_data))

    finally:
        # ==========================================================
        # Always disconnect when the script finishes
        # ==========================================================
        mt5.shutdown()
        print()
        print("MetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()
