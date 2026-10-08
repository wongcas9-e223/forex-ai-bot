"""
gold_rolling_regime_diagnostic.py - Is the previously observed GOLD
high-volatility deterioration STABLE across multiple chronological
periods, or specific to the Development/OOS split?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only diagnostic.
- This script NEVER places, modifies, or closes any trade and contains
  no order execution, order checking, position, or account functions.
- It only reads candles and symbol specifications from MetaTrader 5.
- NO optimization: the strategy, the period boundaries (simple
  chronological quartiles) and the volatility terciles (computed ONCE
  from the full signal dataset) are all fixed in advance. Nothing is
  selected, ranked, removed, or turned into a filter.

THE FROZEN STRATEGY (identical to gold_entry_interaction_diagnostic.py):
  Symbol : GOLD (XM MT5), H1, same data window as the current terminal
           provides (up to 4000 requested; forming candle dropped)
  Entry  : BB(20, 2.0); LONG when close < lower band AND ADX14 < 25,
           SHORT when close > upper band AND ADX14 < 25
  Exit   : SL = 2.0 x ATR14, TP = 1.5 x ATR14 (Variant A baseline)
  Exec   : next-open entry, one position at a time, stop-first when SL
           and TP share a candle, 0.10 lots (validated down to broker
           rules), $2 cost per trade, P/L from GOLD's actual tick specs
           (tick_value / tick_size = USD per price-unit per lot)

CHRONOLOGICAL PERIODS:
  The closed candles are divided into FOUR approximately equal
  chronological QUARTILES (no shuffling, no boundary tuning). Each
  period is backtested INDEPENDENTLY from its own $10,000 virtual
  balance - no compounding across periods.

VOLATILITY REGIMES (descriptive only):
  ATR% = ATR14 / close * 100. Tercile edges (Low / Medium / High) are
  computed ONCE from the FULL signal dataset of all four periods
  combined, exactly like the previous interaction diagnostic - never
  per-period, never optimized.

MEASUREMENTS PER PERIOD:
  - Overall: trades, return, win rate, PF, average trade, total P/L,
    max drawdown
  - Period x Volatility and Period x Direction tables (with LOW SAMPLE
    tags for < 5 trades)
  - High-volatility 12-candle forward return (exit-independent) and
    high-volatility MFE/MAE (normalized by ATR at entry)
  - Market-condition summary (ADX, ATR%, excursion, EMA50 distance,
    direction mix)

DISCLAIMER: Historical, diagnostic only. Nothing is optimized, nothing
is selected, no predictions and no recommendations are made.
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
SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 4000    # Same request as all previous GOLD experiments
MIN_CANDLES = 2000

STARTING_BALANCE = 10_000.0
LOTS_REQUESTED = 0.10
COST_PER_TRADE = 2.0

SL_ATR_MULT = 2.0   # Variant A baseline (frozen exit)
TP_ATR_MULT = 1.5

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0
ATR_PERIOD = 14
EMA_TREND_PERIOD = 50    # Diagnostic reference ONLY (never a filter)
WARMUP = 60

HOLD_WINDOW = 12         # Candles for MFE/MAE and forward returns

NUM_PERIODS = 4          # Chronological quartiles
LOW_SAMPLE_N = 5
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
# SECTION 4: Broker specification and volume validation
# (identical to the previous GOLD scripts)
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
# SECTION 5: The frozen backtest engine with an attribute-rich trade
# log - identical to gold_entry_interaction_diagnostic.py
# ============================================================
def run_backtest(df, value_per_unit, volume, start_bar=WARMUP):
    """Frozen-entry / frozen-exit virtual backtest.
    Returns (trades DataFrame, final balance)."""

    long_condition = df["close"] < df["BB_LOWER"]
    short_condition = df["close"] > df["BB_UPPER"]
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE

    i = start_bar
    while i < len(df) - 1:
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1
            entry_price = df["open"].iloc[entry_bar]

            if signal == "BUY":
                stop_loss = entry_price - SL_ATR_MULT * atr
                take_profit = entry_price + TP_ATR_MULT * atr
            else:
                stop_loss = entry_price + SL_ATR_MULT * atr
                take_profit = entry_price - TP_ATR_MULT * atr

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

            # --- Virtual P/L from GOLD's own specifications ---
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

            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 6: Diagnostic helpers
# ============================================================
def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


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
    """Print one cell with its sample-size warning if needed."""
    s = cell_stats(sub)
    if s is None:
        print(f"      {period_label:<26} (no trades)")
        return
    tag = f"  [{LOW_SAMPLE_TAG}]" if s["n"] < LOW_SAMPLE_N else ""
    line = (f"      {period_label:<26} {s['n']:>3} trades | "
            f"win {s['win_rate']:>5.1f}% | avg ${s['avg_trade']:>7.2f} | "
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
    print("GOLD ROLLING REGIME DIAGNOSTIC (four chronological periods)")
    print("=" * 62)
    print()
    print("Data:")
    print(f"Symbol: {SYMBOL}")
    print("Timeframe: H1")

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
        print(f"ERROR: No valid volume <= {LOTS_REQUESTED} lots for {SYMBOL}.")
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
    print(f"Start timestamp: {df['time'].iloc[0]}")
    print(f"End timestamp  : {df['time'].iloc[-1]}")
    print(f"Volume used: {volume} lots | Frozen exits: SL {SL_ATR_MULT} x ATR "
          f"/ TP {TP_ATR_MULT} x ATR | Cost ${COST_PER_TRADE:.2f}/trade")

    # --- Indicators ONCE over the full closed series (causal math) -------
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)
    df = add_ema(df, EMA_TREND_PERIOD)   # diagnostic reference ONLY

    # ============================================================
    # SECTION 8: Chronological quartile split (no tuning, no shuffling)
    # ============================================================
    n = len(df)
    edges = [round(k * n / NUM_PERIODS) for k in range(NUM_PERIODS + 1)]
    periods = []
    print()
    print("=" * 62)
    print("CHRONOLOGICAL PERIODS (simple quartiles, independent $10,000)")
    print("=" * 62)
    for p in range(NUM_PERIODS):
        sub = df.iloc[edges[p]:edges[p + 1]].reset_index(drop=True)
        periods.append(sub)
        print(f"Period {p + 1}: {sub['time'].iloc[0]} -> {sub['time'].iloc[-1]} "
              f"({len(sub)} candles)")

    # ============================================================
    # SECTION 9: Run the frozen backtest per period
    # ============================================================
    trades_by_period = []
    print()
    print("=" * 62)
    print("OVERALL STRATEGY PERFORMANCE PER PERIOD")
    print("=" * 62)
    for p, sub in enumerate(periods, 1):
        trades, balance = run_backtest(sub, spec["value_per_unit"], volume,
                                       start_bar=(WARMUP if p == 1 else 0))
        trades_by_period.append(trades)

        if len(trades) == 0:
            print(f"\nPeriod {p}: NO TRADES in this period.")
            continue

        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        gp = wins["pnl"].sum()
        gl = abs(losses["pnl"].sum())
        ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
        pf = gp / gl if gl > 0 else float("inf")

        # Max drawdown of this period's balance curve
        balances = [STARTING_BALANCE] + trades["balance_after"].tolist()
        peak = balances[0]
        max_dd = 0.0
        for b in balances[1:]:
            if b > peak:
                peak = b
            max_dd = max(max_dd, peak - b)

        print(f"\nPeriod {p}: {len(trades)} trades | return {ret:+.2f}% | "
              f"win rate {len(wins) / len(trades) * 100:.1f}% | "
              f"PF {fmt_pf(pf)} | avg ${trades['pnl'].mean():+.2f} | "
              f"total ${trades['pnl'].sum():+.2f} | max DD ${max_dd:,.2f}")

    # ============================================================
    # SECTION 10: Volatility tercile edges from the FULL signal dataset
    # (computed ONCE for all periods - never per-period, never optimized)
    # ============================================================
    all_signals = pd.concat(trades_by_period, ignore_index=True)
    if len(all_signals) == 0:
        print("\nERROR: No trades across all periods; cannot diagnose.")
        quit()

    v1 = all_signals["ATR_pct"].quantile(1 / 3)
    v2 = all_signals["ATR_pct"].quantile(2 / 3)
    vol_buckets = [(0.0, v1, "Low volatility"),
                   (v1, v2, "Medium volatility"),
                   (v2, float("inf"), "High volatility")]
    print()
    print("=" * 62)
    print("VOLATILITY REGIMES (terciles of the FULL signal dataset)")
    print("=" * 62)
    print(f"33rd percentile edge: {v1:.4f}%")
    print(f"66th percentile edge: {v2:.4f}%")
    print("(fixed for all four periods - descriptive grouping only)")

    # ============================================================
    # SECTION 11: PERIOD x VOLATILITY table
    # ============================================================
    print()
    print("=" * 62)
    print("PERIOD x VOLATILITY")
    print("=" * 62)
    for p, trades in enumerate(trades_by_period, 1):
        print(f"\nPeriod {p}:")
        for lo, hi, vlabel in vol_buckets:
            print_cell(f"{vlabel}", trades[(trades["ATR_pct"] >= lo)
                                           & (trades["ATR_pct"] < hi)])

    # ============================================================
    # SECTION 12: PERIOD x DIRECTION table
    # ============================================================
    print()
    print("=" * 62)
    print("PERIOD x DIRECTION")
    print("=" * 62)
    for p, trades in enumerate(trades_by_period, 1):
        print(f"\nPeriod {p}:")
        for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
            print_cell(f"{dname}", trades[trades["direction"] == d])

    # ============================================================
    # SECTION 13: HIGH VOLATILITY - exit-independent measurements
    # ============================================================
    print()
    print("=" * 62)
    print(f"HIGH VOLATILITY: {HOLD_WINDOW}-CANDLE FORWARD RETURN "
          f"(exit-independent)")
    print("=" * 62)
    hv_mask = all_signals["ATR_pct"] >= v2
    print(f"(high volatility = ATR% >= {v2:.4f}%, the fixed tercile edge)")
    for p, trades in enumerate(trades_by_period, 1):
        hv = trades[trades["ATR_pct"] >= v2]
        print(f"\nPeriod {p}: {fwd_line(hv)}")
        for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
            hv_d = hv[hv["direction"] == d]
            print(f"   {dname:<6}: {fwd_line(hv_d)}")

    print()
    print("=" * 62)
    print("HIGH VOLATILITY: MFE / MAE (normalized by ATR at entry)")
    print("=" * 62)
    for p, trades in enumerate(trades_by_period, 1):
        hv = trades[trades["ATR_pct"] >= v2]
        print(f"\nPeriod {p}: {mfe_mae_line(hv)}")

    # ============================================================
    # SECTION 14: PERIOD CONDITION SUMMARY (market environment)
    # ============================================================
    print()
    print("=" * 62)
    print("PERIOD CONDITION SUMMARY (signal-time environment)")
    print("=" * 62)

    def col(trades, fn):
        return fn(trades) if len(trades) else float("nan")

    header = (f"   {'Metric':<26}" +
              "".join(f"{'P' + str(p):>10}" for p in range(1, NUM_PERIODS + 1)))
    print(header)

    def fmt_or_na(v, spec="{:.1f}"):
        return "n/a" if v != v else spec.format(v)

    rows = [
        ("Average ADX", lambda t: t["ADX"].mean(), "{:.1f}"),
        ("Median ADX", lambda t: t["ADX"].median(), "{:.1f}"),
        ("Average ATR%", lambda t: t["ATR_pct"].mean(), "{:.3f}"),
        ("Median ATR%", lambda t: t["ATR_pct"].median(), "{:.3f}"),
        ("Average excursion", lambda t: t["excursion"].mean(), "{:.4f}"),
        ("Median excursion", lambda t: t["excursion"].median(), "{:.4f}"),
        ("Avg EMA50 dist (ATR)", lambda t: t["trend_dist"].mean(), "{:.2f}"),
        ("Median EMA50 dist", lambda t: t["trend_dist"].median(), "{:.2f}"),
        ("LONG %", lambda t: (t["direction"] == "BUY").mean() * 100, "{:.1f}%"),
        ("SHORT %", lambda t: (t["direction"] == "SELL").mean() * 100, "{:.1f}%"),
    ]
    for name, fn, spec_fmt in rows:
        vals = [col(trades_by_period[p], fn) for p in range(NUM_PERIODS)]
        line = f"   {name:<26}"
        for v in vals:
            line += f"{fmt_or_na(v, spec_fmt):>10}"
        print(line)

    print()
    print("If these environment metrics shifted materially across the")
    print("four periods, the market conditions the frozen signal faced")
    print("changed - that is an observation, not an optimization.")

    # ============================================================
    # SECTION 15: FINAL FACTUAL QUESTIONS (seven answers)
    # ============================================================
    print()
    print("=" * 62)
    print("FINAL FACTUAL QUESTIONS")
    print("=" * 62)

    def hv_avg(trades):
        sub = trades[trades["ATR_pct"] >= v2]
        return sub["pnl"].mean() if len(sub) else float("nan")

    def nonhv_avg(trades):
        sub = trades[trades["ATR_pct"] < v2]
        return sub["pnl"].mean() if len(sub) else float("nan")

    def money(v):
        return "no trades" if (v != v) else f"${v:+.2f}"

    def fwd_avg(trades):
        sub = trades[trades["ATR_pct"] >= v2]
        return sub["fwd_ret_12"].mean() * 100 if len(sub) else float("nan")

    # Q1: high-volatility performance across periods
    print()
    print("1. High-volatility average trade per period:")
    hv_vals = [hv_avg(trades_by_period[p]) for p in range(NUM_PERIODS)]
    for p in range(NUM_PERIODS):
        n_hv = len(trades_by_period[p][trades_by_period[p]["ATR_pct"] >= v2])
        print(f"   Period {p + 1}: {money(hv_vals[p])} (n={n_hv})")
    valid = [v for v in hv_vals if v == v]
    if len(valid) >= 2:
        all_neg = all(v < 0 for v in valid)
        if all_neg:
            print("   -> NEGATIVE in every period with trades: the")
            print("      deterioration is RECURRING across time.")
        else:
            print("   -> MIXED across periods: the deterioration is NOT")
            print("      uniform - it varies between periods.")

    # Q2: does the DEV/OOS-style deterioration appear in >1 transition?
    print()
    print("2. Period-to-period transitions of high-volatility avg trade:")
    transitions_negative = 0
    transitions_total = 0
    for p in range(NUM_PERIODS - 1):
        a, b = hv_vals[p], hv_vals[p + 1]
        if a == a and b == b and b < a:
            transitions_negative += 1
        if a == a and b == b:
            transitions_total += 1
            arrow = "worse" if b < a else "better"
            print(f"   Period {p + 1} -> {p + 2}: {money(a)} -> {money(b)} ({arrow})")
        else:
            print(f"   Period {p + 1} -> {p + 2}: incomplete data")
    if transitions_total:
        print(f"   -> {transitions_negative} of {transitions_total} transitions "
              f"moved negative (worse).")

    # Q3: high-vol forward return pattern
    print()
    print("3. High-volatility 12-candle forward return per period:")
    fwd_vals = [fwd_avg(trades_by_period[p]) for p in range(NUM_PERIODS)]
    for p in range(NUM_PERIODS):
        print(f"   Period {p + 1}: "
              f"{'n/a' if fwd_vals[p] != fwd_vals[p] else f'{fwd_vals[p]:+.4f}%'}")
    valid_f = [v for v in fwd_vals if v == v]
    if len(valid_f) >= 2:
        if all(v < 0 for v in valid_f):
            print("   -> SAME deterioration pattern in the exit-independent")
            print("      forward return: consistent with a signal-environment")
            print("      effect, not an exit artifact.")
        else:
            print("   -> Forward return does NOT show a uniform pattern;")
            print("      mixed evidence across periods.")

    # Q4: direction effect stability
    print()
    print("4. Direction effect per period (avg $/trade):")
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        vals = []
        for p in range(NUM_PERIODS):
            sub = trades_by_period[p][trades_by_period[p]["direction"] == d]
            vals.append(sub["pnl"].mean() if len(sub) else float("nan"))
        print(f"   {dname:<6}: " + " | ".join(money(v) for v in vals))

    # Q5: condition shifts across periods
    print()
    print("5. Market-condition measurements across periods")
    print("   (from the condition summary table above):")
    for name, fn, spec_fmt in (("ADX", lambda t: t["ADX"].mean(), 1),
                               ("ATR%", lambda t: t["ATR_pct"].mean(), 3),
                               ("EMA50 dist", lambda t: t["trend_dist"].mean(), 2)):
        vals = [col(trades_by_period[p], fn) for p in range(NUM_PERIODS)]
        vals_txt = ["n/a" if v != v else f"{v:.{spec_fmt}f}" for v in vals]
        print(f"   {name:<12}: " + " -> ".join(vals_txt))

    # Q6: low-sample limitations
    print()
    print("6. Sample-size limitations:")
    low_cells = 0
    total_cells = 0
    for p in range(NUM_PERIODS):
        trades = trades_by_period[p]
        for lo, hi, vlabel in vol_buckets:
            n_cell = len(trades[(trades["ATR_pct"] >= lo) & (trades["ATR_pct"] < hi)])
            total_cells += 1
            if n_cell < LOW_SAMPLE_N:
                low_cells += 1
        n_hv = len(trades[trades["ATR_pct"] >= v2])
        total_cells += 1
        if n_hv < LOW_SAMPLE_N:
            low_cells += 1
    print(f"   {low_cells} of {total_cells} period/volatility cells are below "
          f"{LOW_SAMPLE_N} trades (LOW SAMPLE tags above).")
    if low_cells >= total_cells / 3:
        print("   -> A large share of cells is low-sample: conclusions are")
        print("      materially limited by sample size.")
    else:
        print("   -> Most cells have at least 5 trades, but every small")
        print("      cell remains descriptive only.")

    # Q7: recurring vs inconsistent
    print()
    print("7. Recurring or inconsistent?")
    valid_hv = [v for v in hv_vals if v == v]
    if len(valid_hv) >= 3 and all(v < 0 for v in valid_hv):
        print("   -> High-volatility deterioration appears in EVERY period")
        print("      with trades: evidence the regime behavior RECURS across")
        print("      time, not an artifact of the Development/OOS split.")
    elif len(valid_hv) >= 2 and sum(v < 0 for v in valid_hv) >= max(1, len(valid_hv) - 1):
        print("   -> High-volatility deterioration appears in MOST periods:")
        print("      partially recurring, but not uniform across time.")
    else:
        print("   -> The high-volatility behavior is INCONSISTENT across the")
        print("      four periods: the earlier DEV/OOS deterioration may be")
        print("      specific to that particular split.")

    print()
    print("These are descriptive measurements on one historical window,")
    print("split four ways. They do not recommend a filter, a parameter")
    print("change, or a best period, and they do not predict the future.")

    # ============================================================
    # SECTION 16: Disclaimer
    # ============================================================
    print()
    print("Historical diagnostic only. Virtual money only. No orders were")
    print("executed. Nothing was optimized, selected or removed; period")
    print("boundaries are simple quartiles and volatility edges are fixed")
    print("terciles of the full signal dataset.")

finally:
    # ============================================================
    # SECTION 17: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print()
    print("MetaTrader 5 connection closed.")
