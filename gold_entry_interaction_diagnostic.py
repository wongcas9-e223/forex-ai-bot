"""
gold_entry_interaction_diagnostic.py - Are the previously observed GOLD
regime effects INDEPENDENT, or do they INTERACT with trade direction
and market conditions?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only diagnostic.
- This script NEVER places, modifies, or closes any trade and contains
  no order execution, order checking, position, or account functions.
- It only reads candles and symbol specifications from MetaTrader 5.
- NO optimization: the strategy, the buckets and the split boundaries
  are all FIXED in advance. Nothing is selected, ranked, removed, or
  turned into a filter.

THE FROZEN STRATEGY (identical to gold_entry_diagnostic.py):
  Symbol : GOLD (XM MT5), H1, up to 4000 closed candles (forming candle
           dropped, >= 2000 required)
  Entry  : BB(20, 2.0); LONG when close < lower band, SHORT when close
           > upper band, only when ADX14 < 25
  Exit   : Variant A - SL = 2.0 x ATR, TP = 1.5 x ATR
  Exec   : next-open entry, one position at a time, stop-first when SL
           and TP share a candle, 0.10 lots (validated down to broker
           rules), $2 cost per trade, P/L from GOLD's actual tick specs

CHRONOLOGICAL SPLIT (the exact boundaries returned by the previous
gold_entry_diagnostic.py run - candle timestamps, not invented dates):
  Development: 2026-01-26 04:00:00 -> 2026-05-28 05:00:00
  OOS        : 2026-05-28 06:00:00 -> 2026-09-28 14:00:00
  The split is applied BY TIMESTAMP so this diagnostic examines exactly
  the same two periods. If the terminal's current 4000-candle window no
  longer contains these boundaries, the script stops instead of
  silently analyzing different periods.

PREVIOUSLY OBSERVED (from gold_entry_diagnostic.py, to be cross-checked):
  High volatility : DEV +$176.12/trade  ->  OOS -$231.91/trade
  LONG            : DEV  -$12.16/trade  ->  OOS  -$73.17/trade
  ADX 20-25       : DEV +$175.54/trade  ->  OOS  -$35.84/trade
  EMA50 distance  : 1.95 ATR -> 2.26 ATR
This script determines whether these effects OVERLAP (interaction
analysis). It is descriptive research only.

INTERACTION TABLES (Development and OOS separately, every cell with
its trade count; cells with < 5 trades are tagged LOW SAMPLE):
  1. Direction x Volatility (ATR% terciles)
  2. Direction x ADX (<15, 15-20, 20-25)
  3. Direction x EMA50 distance (<1, 1-2, >2 ATR)
  4. Volatility x EMA50 distance (3x3)
  5. ADX x Volatility (3x3)
Plus, exit-independent measurements by interaction:
  6. 12-candle forward return (avg/median/pos%/neg%)
  7. MFE / MAE (winners vs losers)

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
NUM_CANDLES = 4000
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

# Frozen split boundaries (timestamps returned by gold_entry_diagnostic.py)
DEV_START = pd.Timestamp("2026-01-26 04:00:00")
DEV_END = pd.Timestamp("2026-05-28 05:00:00")
OOS_START = pd.Timestamp("2026-05-28 06:00:00")
OOS_END = pd.Timestamp("2026-09-28 14:00:00")

# Frozen descriptive buckets (same edges as the previous diagnostics)
ADX_BUCKETS = [(0.0, 15.0, "ADX <15"),
               (15.0, 20.0, "ADX 15-20"),
               (20.0, 25.0, "ADX 20-25")]
TREND_BUCKETS = [(0.0, 1.0, "<1 ATR"),
                 (1.0, 2.0, "1-2 ATR"),
                 (2.0, float("inf"), ">2 ATR")]
LOW_SAMPLE_N = 5     # Cells with fewer trades get the LOW SAMPLE tag
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
# SECTION 5: The frozen backtest engine (Variant A exits) with an
# attribute-rich trade log - identical to gold_entry_diagnostic.py
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
            hour = df["time"].iloc[i].hour

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
                    "entry_time": df["time"].iloc[entry_bar],
                    "exit_time": df["time"].iloc[exit_index],
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
# SECTION 6: Interaction-table helpers
# ============================================================
def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def cell_stats(sub):
    """Descriptive stats for one interaction cell. Returns a dict or None."""
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


def print_cell(period, label, sub, with_pf=True):
    """Print one interaction cell with its sample-size warning if needed."""
    s = cell_stats(sub)
    if s is None:
        print(f"      {period}  {label:<28} (no trades)")
        return
    tag = f"  [{LOW_SAMPLE_TAG}]" if s["n"] < LOW_SAMPLE_N else ""
    line = (f"      {period}  {label:<28} {s['n']:>3} trades | "
            f"win {s['win_rate']:>5.1f}% | avg ${s['avg_trade']:>7.2f} | "
            f"total ${s['total_pnl']:>9.2f}")
    if with_pf:
        line += f" | PF {fmt_pf(s['pf']):>5}"
    print(line + tag)


# ============================================================
# SECTION 7: The five interaction tables
# ============================================================
def table_direction_vol(trades_dev, trades_oos, vol_edges):
    """INTERACTION 1 - Direction x Volatility."""
    v1, v2 = vol_edges
    vol_buckets = [(0.0, v1, "Low volatility"),
                   (v1, v2, "Medium volatility"),
                   (v2, float("inf"), "High volatility")]
    print()
    print("=" * 70)
    print("INTERACTION 1: DIRECTION x VOLATILITY (ATR% terciles)")
    print("=" * 70)
    print(f"   (volatility edges from this dataset's signals: "
          f"33rd = {v1:.4f}%, 66th = {v2:.4f}%)")
    for direction, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        print(f"   {dname}:")
        for lo, hi, vlabel in vol_buckets:
            label = f"{dname} x {vlabel}"
            print_cell("DEV", label, trades_dev[(trades_dev["direction"] == direction)
                                                & (trades_dev["ATR_pct"] >= lo)
                                                & (trades_dev["ATR_pct"] < hi)])
            print_cell("OOS", label, trades_oos[(trades_oos["direction"] == direction)
                                                & (trades_oos["ATR_pct"] >= lo)
                                                & (trades_oos["ATR_pct"] < hi)])
    print()
    print("   Descriptive only: no cell is ranked, selected or filtered.")


def table_direction_adx(trades_dev, trades_oos):
    """INTERACTION 2 - Direction x ADX."""
    print()
    print("=" * 70)
    print("INTERACTION 2: DIRECTION x ADX (entry filter stays ADX < 25)")
    print("=" * 70)
    for direction, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        print(f"   {dname}:")
        for lo, hi, alabel in ADX_BUCKETS:
            label = f"{dname} x {alabel}"
            print_cell("DEV", label, trades_dev[(trades_dev["direction"] == direction)
                                                & (trades_dev["ADX"] >= lo)
                                                & (trades_dev["ADX"] < hi)])
            print_cell("OOS", label, trades_oos[(trades_oos["direction"] == direction)
                                                & (trades_oos["ADX"] >= lo)
                                                & (trades_oos["ADX"] < hi)])
    print()
    print("   Descriptive only: no cell is ranked, selected or filtered.")


def table_direction_trend(trades_dev, trades_oos):
    """INTERACTION 3 - Direction x EMA50 distance."""
    print()
    print("=" * 70)
    print("INTERACTION 3: DIRECTION x EMA50 DISTANCE (NOT a filter)")
    print("=" * 70)
    for direction, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        print(f"   {dname}:")
        for lo, hi, tlabel in TREND_BUCKETS:
            label = f"{dname} x {tlabel}"
            print_cell("DEV", label, trades_dev[(trades_dev["direction"] == direction)
                                                & (trades_dev["trend_dist"] >= lo)
                                                & (trades_dev["trend_dist"] < hi)])
            print_cell("OOS", label, trades_oos[(trades_oos["direction"] == direction)
                                                & (trades_oos["trend_dist"] >= lo)
                                                & (trades_oos["trend_dist"] < hi)])
    print()
    print("   Descriptive only: no cell is ranked, selected or filtered.")


def table_vol_trend(trades_dev, trades_oos, vol_edges):
    """INTERACTION 4 - Volatility x EMA50 distance (3x3)."""
    v1, v2 = vol_edges
    vol_buckets = [(0.0, v1, "Low volatility"),
                   (v1, v2, "Medium volatility"),
                   (v2, float("inf"), "High volatility")]
    print()
    print("=" * 70)
    print("INTERACTION 4: VOLATILITY x EMA50 DISTANCE (3x3 matrix)")
    print("=" * 70)
    for lo_v, hi_v, vlabel in vol_buckets:
        print(f"   {vlabel}:")
        for lo_t, hi_t, tlabel in TREND_BUCKETS:
            label = f"{vlabel} x {tlabel}"
            print_cell("DEV", label, trades_dev[(trades_dev["ATR_pct"] >= lo_v)
                                                & (trades_dev["ATR_pct"] < hi_v)
                                                & (trades_dev["trend_dist"] >= lo_t)
                                                & (trades_dev["trend_dist"] < hi_t)],
                       with_pf=False)
            print_cell("OOS", label, trades_oos[(trades_oos["ATR_pct"] >= lo_v)
                                                & (trades_oos["ATR_pct"] < hi_v)
                                                & (trades_oos["trend_dist"] >= lo_t)
                                                & (trades_oos["trend_dist"] < hi_t)],
                       with_pf=False)
    print()
    print("   Descriptive only: no cell is ranked, selected or filtered.")


def table_adx_vol(trades_dev, trades_oos, vol_edges):
    """INTERACTION 5 - ADX x Volatility (3x3)."""
    v1, v2 = vol_edges
    vol_buckets = [(0.0, v1, "Low volatility"),
                   (v1, v2, "Medium volatility"),
                   (v2, float("inf"), "High volatility")]
    print()
    print("=" * 70)
    print("INTERACTION 5: ADX x VOLATILITY (3x3 matrix)")
    print("=" * 70)
    for lo_a, hi_a, alabel in ADX_BUCKETS:
        print(f"   {alabel}:")
        for lo_v, hi_v, vlabel in vol_buckets:
            label = f"{alabel} x {vlabel}"
            print_cell("DEV", label, trades_dev[(trades_dev["ADX"] >= lo_a)
                                                & (trades_dev["ADX"] < hi_a)
                                                & (trades_dev["ATR_pct"] >= lo_v)
                                                & (trades_dev["ATR_pct"] < hi_v)],
                       with_pf=False)
            print_cell("OOS", label, trades_oos[(trades_oos["ADX"] >= lo_a)
                                                & (trades_oos["ADX"] < hi_a)
                                                & (trades_oos["ATR_pct"] >= lo_v)
                                                & (trades_oos["ATR_pct"] < hi_v)],
                       with_pf=False)
    print()
    print("   Descriptive only: no cell is ranked, selected or filtered.")


# ============================================================
# SECTION 8: Exit-independent measurements by interaction
# ============================================================
def forward_by_interaction(trades_dev, trades_oos, title, split_fn, groups):
    """Print 12-candle forward-return stats per interaction group."""
    print()
    print(f"   {title}")
    for glabel, gfn in groups:
        print(f"   {glabel}:")
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = gfn(trades)
            if len(sub) == 0:
                print(f"      {period}: (no trades)")
                continue
            fwd = sub["fwd_ret_12"] * 100
            tag = f"  [{LOW_SAMPLE_TAG}]" if len(sub) < LOW_SAMPLE_N else ""
            print(f"      {period}: n={len(sub):>3} | "
                  f"avg {fwd.mean():+.4f}% | median {fwd.median():+.4f}% | "
                  f"pos {(fwd > 0).mean() * 100:.1f}% / "
                  f"neg {(fwd < 0).mean() * 100:.1f}%{tag}")
    print("   (forward return is independent of the SL/TP exits)")


def mfe_mae_by_interaction(trades_dev, trades_oos, title, groups):
    """Print avg MFE/MAE (winners vs losers) per interaction group."""
    print()
    print(f"   {title}")
    for glabel, gfn in groups:
        print(f"   {glabel}:")
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = gfn(trades)
            if len(sub) == 0:
                print(f"      {period}: (no trades)")
                continue
            w = sub[sub["pnl"] > 0]
            l = sub[sub["pnl"] < 0]
            wtxt = (f"winners MFE {w['MFE_ATR'].mean():.2f} / "
                    f"MAE {w['MAE_ATR'].mean():.2f}" if len(w) else
                    "winners (none)")
            ltxt = (f"losers MFE {l['MFE_ATR'].mean():.2f} / "
                    f"MAE {l['MAE_ATR'].mean():.2f}" if len(l) else
                    "losers (none)")
            tag = f"  [{LOW_SAMPLE_TAG}]" if len(sub) < LOW_SAMPLE_N else ""
            print(f"      {period}: n={len(sub):>3} | {wtxt} | {ltxt}{tag}")


def interaction_measurements(trades_dev, trades_oos, vol_edges):
    """Sections 7 and 8: forward returns and MFE/MAE by interaction."""
    v1, v2 = vol_edges

    print()
    print("=" * 70)
    print("EXIT-INDEPENDENT MEASUREMENTS BY INTERACTION")
    print("=" * 70)

    # --- Section 7: forward returns --------------------------------------
    print()
    print("-" * 70)
    print("7. 12-CANDLE FORWARD RETURN BY INTERACTION")
    print("-" * 70)

    direction_groups = [(dname, lambda t, d=d: t[t["direction"] == d])
                        for d, dname in (("BUY", "LONG"), ("SELL", "SHORT"))]

    # A. Direction x volatility
    vol_split = [(dname, lambda t, d=d, lo=lo, hi=hi:
                  t[(t["direction"] == d) & (t["ATR_pct"] >= lo) & (t["ATR_pct"] < hi)])
                 for d, dname in (("BUY", "LONG"), ("SELL", "SHORT"))
                 for lo, hi, vlabel in
                 ((0.0, v1, "Low"), (v1, v2, "Medium"), (v2, float("inf"), "High"))]
    forward_by_interaction(
        trades_dev, trades_oos,
        "A. Direction x Volatility",
        None, vol_split)

    # B. Direction x EMA50 distance
    trend_split = [(f"{dname} x {tlabel}", lambda t, d=d, lo=lo, hi=hi:
                    t[(t["direction"] == d) & (t["trend_dist"] >= lo)
                      & (t["trend_dist"] < hi)])
                   for d, dname in (("BUY", "LONG"), ("SELL", "SHORT"))
                   for lo, hi, tlabel in TREND_BUCKETS]
    forward_by_interaction(
        trades_dev, trades_oos,
        "B. Direction x EMA50 distance",
        None, trend_split)

    # C. Direction x ADX
    adx_split = [(f"{dname} x {alabel}", lambda t, d=d, lo=lo, hi=hi:
                  t[(t["direction"] == d) & (t["ADX"] >= lo) & (t["ADX"] < hi)])
                 for d, dname in (("BUY", "LONG"), ("SELL", "SHORT"))
                 for lo, hi, alabel in ADX_BUCKETS]
    forward_by_interaction(
        trades_dev, trades_oos,
        "C. Direction x ADX",
        None, adx_split)

    # --- Section 8: MFE / MAE --------------------------------------------
    print()
    print("-" * 70)
    print("8. MFE / MAE BY INTERACTION (winners vs losers)")
    print("-" * 70)

    mfe_mae_by_interaction(
        trades_dev, trades_oos,
        "A. LONG vs SHORT",
        direction_groups)

    mfe_mae_by_interaction(
        trades_dev, trades_oos,
        "B. High volatility vs non-high volatility",
        [("High volatility", lambda t: t[t["ATR_pct"] >= v2]),
         ("Non-high volatility", lambda t: t[t["ATR_pct"] < v2])])

    mfe_mae_by_interaction(
        trades_dev, trades_oos,
        "C. EMA50 distance > 2 ATR vs <= 2 ATR",
        [("> 2 ATR", lambda t: t[t["trend_dist"] > 2.0]),
         ("<= 2 ATR", lambda t: t[t["trend_dist"] <= 2.0])])


# ============================================================
# SECTION 9: FINAL INTERPRETATION (seven factual answers)
# ============================================================
def final_interpretation(trades_dev, trades_oos, vol_edges):
    """Answer ONLY the seven factual questions, with measured numbers."""
    v1, v2 = vol_edges

    def money(v):
        """Format a per-trade average; empty cells print 'no trades'."""
        return "no trades" if (v != v) else f"${v:+.2f}/trade"
    print()
    print("=" * 70)
    print("FINAL INTERPRETATION (factual answers, no recommendations)")
    print("=" * 70)

    def cell(trades, **conds):
        sub = trades
        for col, (lo, hi) in conds.items():
            sub = sub[(sub[col] >= lo) & (sub[col] < hi)]
        return sub

    # Q1 + Q2: concentration of the high-vol and ADX 20-25 deterioration
    def avg_by(trades, direction=None, vol_high=None, adx_band=None):
        sub = trades
        if direction:
            sub = sub[sub["direction"] == direction]
        if vol_high is not None:
            sub = sub[sub["ATR_pct"] >= v2] if vol_high else sub[sub["ATR_pct"] < v2]
        if adx_band:
            lo, hi = adx_band
            sub = sub[(sub["ADX"] >= lo) & (sub["ADX"] < hi)]
        return sub["pnl"].mean() if len(sub) else float("nan")

    print()
    print("1. High-volatility deterioration - LONG, SHORT, or both?")
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        d_avg = avg_by(trades_dev, direction=d, vol_high=True)
        o_avg = avg_by(trades_oos, direction=d, vol_high=True)
        n_d = len(trades_dev[(trades_dev["direction"] == d)
                             & (trades_dev["ATR_pct"] >= v2)])
        n_o = len(trades_oos[(trades_oos["direction"] == d)
                             & (trades_oos["ATR_pct"] >= v2)])
        print(f"   {dname} x high vol: DEV {money(d_avg)} -> "
              f"OOS {money(o_avg)} (n_dev={n_d}/n_oos={n_o})")
    print("   (compare with the direction tables above; a cell with a big")
    print("   DEV->OOS swing shows WHERE the deterioration concentrates)")

    print()
    print("2. ADX 20-25 deterioration - LONG, SHORT, or both?")
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        d_avg = avg_by(trades_dev, direction=d, adx_band=(20.0, 25.0))
        o_avg = avg_by(trades_oos, direction=d, adx_band=(20.0, 25.0))
        n_d = len(trades_dev[(trades_dev["direction"] == d)
                             & (trades_dev["ADX"] >= 20.0) & (trades_dev["ADX"] < 25.0)])
        n_o = len(trades_oos[(trades_oos["direction"] == d)
                             & (trades_oos["ADX"] >= 20.0) & (trades_oos["ADX"] < 25.0)])
        print(f"   {dname} x ADX 20-25: DEV {money(d_avg)} -> "
              f"OOS {money(o_avg)} (n_dev={n_d}/n_oos={n_o})")

    print()
    print("3. Does increased EMA50 distance coincide with worse outcomes?")
    for lo, hi, tlabel in TREND_BUCKETS:
        d_avg = trades_dev[(trades_dev["trend_dist"] >= lo)
                           & (trades_dev["trend_dist"] < hi)]["pnl"].mean()
        o_avg = trades_oos[(trades_oos["trend_dist"] >= lo)
                           & (trades_oos["trend_dist"] < hi)]["pnl"].mean()
        print(f"   {tlabel}: DEV {money(d_avg)} -> OOS {money(o_avg)}")
    print("   (compare the >2 ATR row against the nearer rows; the EMA50")
    print("   distance metric itself moved 1.95 -> 2.26 ATR previously)")

    print()
    print("4. Is the volatility effect still visible after separating")
    print("   LONG and SHORT?")
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        d_low = avg_by(trades_dev, direction=d, vol_high=False)
        o_low = avg_by(trades_oos, direction=d, vol_high=False)
        d_high = avg_by(trades_dev, direction=d, vol_high=True)
        o_high = avg_by(trades_oos, direction=d, vol_high=True)
        print(f"   {dname}: low/mid vol DEV {money(d_low)} -> OOS {money(o_low)}; "
              f"high vol DEV {money(d_high)} -> OOS {money(o_high)}")

    print()
    print("5. Is the direction effect still visible after separating")
    print("   volatility?")
    for lo, hi, vlabel in ((0.0, v1, "Low"), (v1, v2, "Medium"),
                           (v2, float("inf"), "High")):
        ld = trades_dev[(trades_dev["ATR_pct"] >= lo) & (trades_dev["ATR_pct"] < hi)
                        & (trades_dev["direction"] == "BUY")]["pnl"].mean()
        sd = trades_dev[(trades_dev["ATR_pct"] >= lo) & (trades_dev["ATR_pct"] < hi)
                        & (trades_dev["direction"] == "SELL")]["pnl"].mean()
        lo_ = trades_oos[(trades_oos["ATR_pct"] >= lo) & (trades_oos["ATR_pct"] < hi)
                         & (trades_oos["direction"] == "BUY")]["pnl"].mean()
        so = trades_oos[(trades_oos["ATR_pct"] >= lo) & (trades_oos["ATR_pct"] < hi)
                        & (trades_oos["direction"] == "SELL")]["pnl"].mean()
        print(f"   {vlabel}: LONG DEV {money(ld)} -> OOS {money(lo_)}; "
              f"SHORT DEV {money(sd)} -> OOS {money(so)}")

    print()
    print("6. Substantially different interactions DEV -> OOS")
    print("   (|avg trade| change > $50/trade):")
    found = False
    combos = []
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        for lo, hi, vlabel in ((0.0, v1, "LowVol"), (v1, v2, "MedVol"),
                               (v2, float("inf"), "HighVol")):
            dv = trades_dev[(trades_dev["direction"] == d)
                            & (trades_dev["ATR_pct"] >= lo)
                            & (trades_dev["ATR_pct"] < hi)]["pnl"].mean()
            ov = trades_oos[(trades_oos["direction"] == d)
                            & (trades_oos["ATR_pct"] >= lo)
                            & (trades_oos["ATR_pct"] < hi)]["pnl"].mean()
            combos.append((abs(ov - dv), f"{dname} x {vlabel}", dv, ov))
        for lo, hi, tlabel in TREND_BUCKETS:
            dv = trades_dev[(trades_dev["direction"] == d)
                            & (trades_dev["trend_dist"] >= lo)
                            & (trades_dev["trend_dist"] < hi)]["pnl"].mean()
            ov = trades_oos[(trades_oos["direction"] == d)
                            & (trades_oos["trend_dist"] >= lo)
                            & (trades_oos["trend_dist"] < hi)]["pnl"].mean()
            combos.append((abs(ov - dv), f"{dname} x {tlabel}", dv, ov))
        for lo, hi, alabel in ADX_BUCKETS:
            dv = trades_dev[(trades_dev["direction"] == d)
                            & (trades_dev["ADX"] >= lo)
                            & (trades_dev["ADX"] < hi)]["pnl"].mean()
            ov = trades_oos[(trades_oos["direction"] == d)
                            & (trades_oos["ADX"] >= lo)
                            & (trades_oos["ADX"] < hi)]["pnl"].mean()
            combos.append((abs(ov - dv), f"{dname} x {alabel}", dv, ov))
    combos = [c for c in combos if c[0] == c[0]]  # drop cells lacking trades
    for change, name, dv, ov in sorted(combos, reverse=True):
        if change > 50:
            found = True
            print(f"   {name}: DEV {money(dv)} -> OOS {money(ov)} "
                  f"(change ${change:.2f}/trade)")
    if not found:
        print("   No interaction exceeds the $50/trade change threshold.")
    print("   (Listing a change is NOT a recommendation to act on it.)")

    # Q7: the single largest numerical DEV->OOS change among the
    # previously observed relationships
    print()
    print("7. Largest numerical DEV->OOS change among the observed")
    print("   relationships (avg $/trade, higher = bigger change):")
    ranked = sorted(combos, reverse=True)[:3]
    if not ranked:
        print("   (no cells have trades in both periods to compare)")
    for i, (change, name, dv, ov) in enumerate(ranked, 1):
        print(f"   {i}. {name}: change ${change:.2f}/trade "
              f"(DEV {money(dv)} -> OOS {money(ov)})")
    print("   This identifies the LARGEST CHANGE, not the 'best' or")
    print("   'worst' cell, and not a filter candidate.")

    print()
    print("These are descriptive measurements of one historical sample.")
    print("They do not say which filter to use, which parameter is")
    print("optimal, which strategy is best, or what trade to take.")


# ============================================================
# SECTION 10: Main program
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
    print("GOLD ENTRY INTERACTION DIAGNOSTIC")
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
    print(f"Available range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    # --- Validate the frozen split boundaries exist in this window ------
    if df["time"].iloc[0] > DEV_START or df["time"].iloc[-1] < OOS_END:
        print()
        print("STOP: The frozen split boundaries are not inside the")
        print("current terminal window:")
        print(f"  Requested DEV start : {DEV_START}")
        print(f"  Requested OOS end   : {OOS_END}")
        print(f"  Available range     : {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
        print("The terminal's 4000-candle window has shifted. Re-run")
        print("gold_entry_diagnostic.py to refresh the boundaries rather")
        print("than silently analyzing different periods.")
        quit()

    # --- Indicators ONCE over the full closed series (causal math) -------
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)
    df = add_ema(df, EMA_TREND_PERIOD)   # diagnostic reference ONLY

    # ============================================================
    # SECTION 11: Split by the FROZEN TIMESTAMP boundaries
    # ============================================================
    dev_df = df[(df["time"] >= DEV_START) & (df["time"] <= DEV_END)].reset_index(drop=True)
    oos_df = df[(df["time"] >= OOS_START) & (df["time"] <= OOS_END)].reset_index(drop=True)

    print()
    print(f"Development period: {dev_df['time'].iloc[0]} -> {dev_df['time'].iloc[-1]} "
          f"({len(dev_df)} candles)")
    print(f"OOS period        : {oos_df['time'].iloc[0]} -> {oos_df['time'].iloc[-1]} "
          f"({len(oos_df)} candles)")
    print("Chronological split by the frozen boundaries; nothing shuffled.")
    print(f"Indicators computed causally ONCE. Frozen exits: SL "
          f"{SL_ATR_MULT} x ATR / TP {TP_ATR_MULT} x ATR. Volume {volume} lots.")

    # ============================================================
    # SECTION 12: Run the frozen backtest per period
    # ============================================================
    trades_dev, bal_dev = run_backtest(dev_df, spec["value_per_unit"], volume,
                                       start_bar=WARMUP)
    trades_oos, bal_oos = run_backtest(oos_df, spec["value_per_unit"], volume,
                                       start_bar=0)

    if len(trades_dev) == 0 or len(trades_oos) == 0:
        print("ERROR: One of the periods produced no trades; "
              "the interaction diagnostic needs both. Stopping.")
        quit()

    for period, trades, bal in (("Development", trades_dev, bal_dev),
                                ("OOS", trades_oos, bal_oos)):
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        gp = wins["pnl"].sum()
        gl = abs(losses["pnl"].sum())
        ret = (bal - STARTING_BALANCE) / STARTING_BALANCE * 100
        pf = gp / gl if gl > 0 else float("inf")
        print(f"  {period}: {len(trades)} trades, return {ret:+.2f}%, "
              f"win rate {len(wins) / len(trades) * 100:.1f}%, "
              f"PF {fmt_pf(pf)}, avg ${trades['pnl'].mean():+.2f}/trade")

    # ============================================================
    # SECTION 13: Descriptive volatility edges (from this dataset's
    # signals, used ONLY for grouping - never for selection)
    # ============================================================
    all_signals = pd.concat([trades_dev, trades_oos], ignore_index=True)
    vol_e1 = all_signals["ATR_pct"].quantile(1 / 3)
    vol_e2 = all_signals["ATR_pct"].quantile(2 / 3)

    # ============================================================
    # SECTION 14: The five interaction tables + exit-independent
    # measurements + final interpretation
    # ============================================================
    table_direction_vol(trades_dev, trades_oos, (vol_e1, vol_e2))
    table_direction_adx(trades_dev, trades_oos)
    table_direction_trend(trades_dev, trades_oos)
    table_vol_trend(trades_dev, trades_oos, (vol_e1, vol_e2))
    table_adx_vol(trades_dev, trades_oos, (vol_e1, vol_e2))
    interaction_measurements(trades_dev, trades_oos, (vol_e1, vol_e2))
    final_interpretation(trades_dev, trades_oos, (vol_e1, vol_e2))

    # ============================================================
    # SECTION 15: Disclaimer
    # ============================================================
    print()
    print("Historical diagnostic only. Virtual money only. No orders were")
    print("executed. Nothing was optimized, selected or removed. Cells")
    print("with few trades are descriptive only (LOW SAMPLE tags).")

finally:
    # ============================================================
    # SECTION 16: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print()
    print("MetaTrader 5 connection closed.")
