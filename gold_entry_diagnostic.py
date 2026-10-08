"""
gold_entry_diagnostic.py - WHY did the frozen GOLD BB(20,2.0) + ADX<25
entry signal perform POSITIVELY during Development but NEGATIVELY
during the out-of-sample (OOS) period?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only diagnostic.
- This script NEVER places, modifies, or closes any trade and contains
  no order execution, order checking, position, or account functions.
- It only reads candles and symbol specifications from MetaTrader 5.
- NO optimization: the entry rules, exit rules and every bucket edge
  below are FIXED in advance (quantile edges are descriptive only and
  are computed once from the full signal dataset - they never select,
  remove, or rank anything).

THE FROZEN STRATEGY (identical to the previous GOLD experiments):
  Symbol  : GOLD (XM MT5), H1, up to 4000 closed candles (forming
            candle dropped, >= 2000 required)
  Entry   : BB(20, 2.0); LONG when close < lower band, SHORT when
            close > upper band, only when ADX14 < 25
  Exit    : Variant A baseline - SL = 2.0 x ATR, TP = 1.5 x ATR
  Exec    : signal on closed candle, entry at NEXT candle open, one
            position at a time, stop-first when SL and TP share a
            candle, 0.10 lots requested (validated down to broker
            rules), $2 cost per trade, P/L from GOLD's actual tick
            specifications (tick_value / tick_size = USD per
            price-unit per lot)

CHRONOLOGICAL SPLIT (same methodology as gold_exit_oos_validation.py):
  Indicators are computed ONCE over the full closed series (causal
  math only). The first half of the closed candles is Development,
  the second half is OOS. Each period's trade log starts from an
  INDEPENDENT $10,000 balance. Balances are NOT carried across.

DIAGNOSTIC DESIGN:
  Every executed trade carries its SIGNAL-TIME attributes (direction,
  ADX, ATR%, band excursion, EMA50 distance, hour). The nine sections
  then group those trades descriptively:
    A LONG vs SHORT          B ADX buckets
    C Bollinger excursion    D Volatility (ATR%)
    E Time of day            F Trend distance (EMA50/ATR)
    G MFE / MAE              H 12-candle forward outcome
    I Signal-quality comparison
  MFE/MAE and the 12-candle forward return are measured from the
  ENTRY PRICE over the next 12 H1 candles INDEPENDENTLY of the actual
  exit, so the entry signal's inherent behavior is visible even when
  the frozen exits truncate the move.

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

# Frozen descriptive bucket edges (never optimized, never selected on)
ADX_BUCKETS = [(0.0, 15.0, "ADX < 15"),
               (15.0, 20.0, "15 <= ADX < 20"),
               (20.0, 25.0, "20 <= ADX < 25")]
SESSIONS = [(0, 6, "00:00-05:59"), (6, 12, "06:00-11:59"),
            (12, 18, "12:00-17:59"), (18, 24, "18:00-23:59")]
TREND_BUCKETS = [(0.0, 1.0, "< 1 ATR"),
                 (1.0, 2.0, "1-2 ATR"),
                 (2.0, float("inf"), "> 2 ATR")]
EXCURSION_LABELS = ["small excursion", "medium excursion", "large excursion"]
VOL_LABELS = ["Low volatility", "Medium volatility", "High volatility"]


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
# attribute-rich trade log for the diagnostics
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
                    "signal_hour": hour,
                    "entry_time": df["time"].iloc[entry_bar],
                    "entry_price": round(entry_price, 5),
                    "exit_time": df["time"].iloc[exit_index],
                    "exit_price": round(exit_price, 5),
                    "exit_reason": exit_reason,
                    "hold_candles": exit_index - entry_bar + 1,
                    "ADX": round(adx_sig, 2) if pd.notna(adx_sig) else np.nan,
                    "ATR_pct": round(atr_pct, 4) if pd.notna(atr_pct) else np.nan,
                    "excursion": round(excursion, 4) if pd.notna(excursion) else np.nan,
                    "trend_dist": round(trend_dist, 4) if pd.notna(trend_dist) else np.nan,
                    "fwd_ret_12": fwd_ret,
                    "MFE_ATR": round(mfe_atr, 3) if pd.notna(mfe_atr) else np.nan,
                    "MAE_ATR": round(mae_atr, 3) if pd.notna(mae_atr) else np.nan,
                    "pnl": round(net_pnl, 2),
                    "balance_after": round(balance, 2),
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


def group_stats(sub, label):
    """Descriptive stats for one group of trades. Returns a dict or None."""
    if sub is None or len(sub) == 0:
        return None
    wins = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] < 0]
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    return {
        "label": label,
        "n": len(sub),
        "win_rate": len(wins) / len(sub) * 100,
        "avg_trade": sub["pnl"].mean(),
        "pf": gp / gl if gl > 0 else float("inf"),
        "total_pnl": sub["pnl"].sum(),
        "avg_fwd": sub["fwd_ret_12"].mean() * 100,   # in %
    }


def print_group_row(s, with_pf=True, with_fwd=True):
    """Print one group-statistics line."""
    if s is None:
        print("      (no trades)")
        return
    line = (f"      {s['label']:<22} {s['n']:>4} trades | "
            f"win {s['win_rate']:>5.1f}% | avg ${s['avg_trade']:>6.2f} | "
            f"total ${s['total_pnl']:>8.2f}")
    if with_pf:
        line += f" | PF {fmt_pf(s['pf']):>5}"
    if with_fwd:
        line += f" | fwd12 {s['avg_fwd']:>+6.3f}%"
    print(line)


def compare_periods(trades_dev, trades_oos, bucket_fn, title, with_pf=True,
                    with_fwd=True):
    """Print one diagnostic table for Development vs OOS."""
    print()
    print(f"   {title}")
    print(f"   {'':<22} {'Development':>28} | OOS")
    for label_dev, label_oos, key in _period_bucket_pairs(
            trades_dev, trades_oos, bucket_fn):
        s_dev = group_stats(trades_dev[bucket_fn(trades_dev, key)] if key is not None
                            else trades_dev, label_dev) if len(trades_dev) else None
        s_oos = group_stats(trades_oos[bucket_fn(trades_oos, key)] if key is not None
                            else trades_oos, label_oos) if len(trades_oos) else None
        # Print dev + OOS on separate lines for readability
        print(f"   DEV: ", end="")
        print_group_row(s_dev, with_pf=with_pf, with_fwd=with_fwd)
        print(f"   OOS: ", end="")
        print_group_row(s_oos, with_pf=with_pf, with_fwd=with_fwd)
        print()


# ============================================================
# SECTION 7: The nine diagnostic sections
# Each prints Development vs OOS with measured numbers only.
# ============================================================
def section_long_short(trades_dev, trades_oos):
    """ANALYSIS A - LONG vs SHORT."""
    print()
    print("-" * 70)
    print("1. LONG vs SHORT")
    print("-" * 70)
    for direction, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        print(f"   {dname} signals:")
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = trades[trades["direction"] == direction]
            s = group_stats(sub, dname)
            print(f"      {period}: ", end="")
            print_group_row(s)


def section_adx(trades_dev, trades_oos):
    """ANALYSIS B - ADX regime buckets (descriptive only)."""
    print()
    print("-" * 70)
    print("2. ADX DIAGNOSTIC (strategy still uses ADX < 25 - descriptive only)")
    print("-" * 70)
    for lo, hi, label in ADX_BUCKETS:
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = trades[(trades["ADX"] >= lo) & (trades["ADX"] < hi)]
            s = group_stats(sub, label)
            print(f"      {period}: ", end="")
            print_group_row(s)
        print()


def section_excursion(trades_dev, trades_oos, exc_edges):
    """ANALYSIS C - Bollinger band excursion terciles (descriptive)."""
    print()
    print("-" * 70)
    print("3. BOLLINGER EXCURSION DIAGNOSTIC (fixed 33rd/66th percentile)")
    print("-" * 70)
    e1, e2 = exc_edges
    buckets = [(0.0, e1, EXCURSION_LABELS[0]),
               (e1, e2, EXCURSION_LABELS[1]),
               (e2, float("inf"), EXCURSION_LABELS[2])]
    for lo, hi, label in buckets:
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = trades[(trades["excursion"] >= lo) & (trades["excursion"] < hi)]
            s = group_stats(sub, f"{label}")
            print(f"      {period}: ", end="")
            print_group_row(s)
        print()
    print(f"      (bucket edges from the full signal dataset: "
          f"33rd = {e1:.4f}, 66th = {e2:.4f} - descriptive only)")


def section_volatility(trades_dev, trades_oos, vol_edges):
    """ANALYSIS D - ATR% volatility terciles (descriptive)."""
    print()
    print("-" * 70)
    print("4. VOLATILITY DIAGNOSTIC (ATR% terciles from the full dataset)")
    print("-" * 70)
    v1, v2 = vol_edges
    buckets = [(0.0, v1, VOL_LABELS[0]),
               (v1, v2, VOL_LABELS[1]),
               (v2, float("inf"), VOL_LABELS[2])]
    for lo, hi, label in buckets:
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = trades[(trades["ATR_pct"] >= lo) & (trades["ATR_pct"] < hi)]
            s = group_stats(sub, label)
            print(f"      {period}: ", end="")
            print_group_row(s)
        print()
    print(f"      (bucket edges: 33rd = {v1:.4f}%, 66th = {v2:.4f}% - "
          f"descriptive only)")


def section_time_of_day(trades_dev, trades_oos):
    """ANALYSIS E - Time-of-day sessions (descriptive only)."""
    print()
    print("-" * 70)
    print("5. TIME-OF-DAY DIAGNOSTIC (fixed 6-hour sessions)")
    print("-" * 70)
    for lo, hi, label in SESSIONS:
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = trades[(trades["signal_hour"] >= lo)
                         & (trades["signal_hour"] < hi)]
            s = group_stats(sub, label)
            print(f"      {period}: ", end="")
            print_group_row(s, with_pf=False, with_fwd=False)  # spec: no PF/fwd
        print()
    print("      (no forward returns shown for sessions - spec asks for "
          "trades/win rate/avg/total only)")


def section_trend_distance(trades_dev, trades_oos):
    """ANALYSIS F - EMA50 distance buckets (diagnostic only)."""
    print()
    print("-" * 70)
    print("6. TREND-DISTANCE DIAGNOSTIC (|close - EMA50| / ATR; "
          "NOT a filter)")
    print("-" * 70)
    for lo, hi, label in TREND_BUCKETS:
        for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
            sub = trades[(trades["trend_dist"] >= lo) & (trades["trend_dist"] < hi)]
            s = group_stats(sub, label)
            print(f"      {period}: ", end="")
            print_group_row(s)
        print()


def section_mfe_mae(trades_dev, trades_oos):
    """ANALYSIS G - MFE/MAE of winners vs losers."""
    print()
    print("-" * 70)
    print(f"7. MFE / MAE DIAGNOSTIC (next {HOLD_WINDOW} candles, "
          f"normalized by ATR at entry)")
    print("-" * 70)
    for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
        winners = trades[trades["pnl"] > 0]
        losers = trades[trades["pnl"] < 0]
        print(f"      {period}:")
        for label, sub in (("winners", winners), ("losers", losers)):
            if len(sub) == 0:
                print(f"         {label:<8}: (no trades)")
                continue
            print(f"         {label:<8}: n={len(sub):>3} | "
                  f"avg MFE {sub['MFE_ATR'].mean():.2f} ATR | "
                  f"avg MAE {sub['MAE_ATR'].mean():.2f} ATR")
    print()
    print("      Reading: MFE = how far price moved IN FAVOR before/while")
    print("      the trade was on; MAE = how far against. If OOS losers")
    print("      show MAE similar to DEV but MFE near zero, price never")
    print("      reverted; if OOS losers show huge MAE, adverse shocks")
    print("      dominate (stop-first exits realize them).")


def section_forward(trades_dev, trades_oos):
    """ANALYSIS H - 12-candle forward outcome (independent of exits)."""
    print()
    print("-" * 70)
    print(f"8. {HOLD_WINDOW}-CANDLE FORWARD OUTCOME (independent of exits)")
    print("-" * 70)
    for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
        fwd = trades["fwd_ret_12"] * 100
        pos_pct = (fwd > 0).mean() * 100
        neg_pct = (fwd < 0).mean() * 100
        print(f"      {period}: ALL   n={len(fwd):>3} | "
              f"avg {fwd.mean():+.4f}% | median {fwd.median():+.4f}% | "
              f"pos {pos_pct:.1f}% / neg {neg_pct:.1f}%")
        for direction, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
            fsub = trades.loc[trades["direction"] == direction, "fwd_ret_12"] * 100
            if len(fsub) == 0:
                print(f"             {dname:<5}: (no trades)")
                continue
            print(f"             {dname:<5}: avg {fsub.mean():+.4f}% | "
                  f"median {fsub.median():+.4f}% | "
                  f"pos {(fsub > 0).mean() * 100:.1f}% / "
                  f"neg {(fsub < 0).mean() * 100:.1f}%")
    print()
    print("      This is the raw edge of the ENTRY SIGNAL without any")
    print("      exit interference. A positive DEV / negative OOS split")
    print("      here means the signal itself degraded.")


def section_signal_quality(trades_dev, trades_oos, dev_start, dev_end,
                           oos_start, oos_end):
    """ANALYSIS I - signal-quality comparison DEV vs OOS."""
    print()
    print("-" * 70)
    print("9. DEVELOPMENT vs OOS SIGNAL-QUALITY COMPARISON")
    print("-" * 70)
    print(f"   {'Metric':<28}{'Development':>16}{'OOS':>16}")

    def col(trades, fn):
        return fn(trades) if len(trades) else float("nan")

    rows = [
        ("Signal count",
         lambda t: f"{len(t)}",
         lambda t: f"{len(t)}"),
        ("LONG %",
         lambda t: f"{(t['direction'] == 'BUY').mean() * 100:.1f}%",
         lambda t: f"{(t['direction'] == 'BUY').mean() * 100:.1f}%"),
        ("SHORT %",
         lambda t: f"{(t['direction'] == 'SELL').mean() * 100:.1f}%",
         lambda t: f"{(t['direction'] == 'SELL').mean() * 100:.1f}%"),
        ("Average ADX",
         lambda t: f"{t['ADX'].mean():.1f}",
         lambda t: f"{t['ADX'].mean():.1f}"),
        ("Median ADX",
         lambda t: f"{t['ADX'].median():.1f}",
         lambda t: f"{t['ADX'].median():.1f}"),
        ("Average ATR%",
         lambda t: f"{t['ATR_pct'].mean():.3f}%",
         lambda t: f"{t['ATR_pct'].mean():.3f}%"),
        ("Median ATR%",
         lambda t: f"{t['ATR_pct'].median():.3f}%",
         lambda t: f"{t['ATR_pct'].median():.3f}%"),
        ("Average excursion",
         lambda t: f"{t['excursion'].mean():.4f}",
         lambda t: f"{t['excursion'].mean():.4f}"),
        ("Median excursion",
         lambda t: f"{t['excursion'].median():.4f}",
         lambda t: f"{t['excursion'].median():.4f}"),
        ("Avg EMA50 dist (ATR)",
         lambda t: f"{t['trend_dist'].mean():.2f}",
         lambda t: f"{t['trend_dist'].mean():.2f}"),
        ("Median EMA50 dist",
         lambda t: f"{t['trend_dist'].median():.2f}",
         lambda t: f"{t['trend_dist'].median():.2f}"),
    ]
    for name, fn_dev, fn_oos in rows:
        print(f"   {name:<28}{col(trades_dev, fn_dev):>16}"
              f"{col(trades_oos, fn_oos):>16}")

    print()
    print(f"   Period boundaries: DEV {dev_start} -> {dev_end}")
    print(f"                      OOS {oos_start} -> {oos_end}")
    print("   If the market-condition metrics above shifted materially,")
    print("   the conditions the frozen signal was tested in changed -")
    print("   that is a market-regime observation, NOT an optimization.")


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
    print()
    print("=" * 62)
    print("GOLD ENTRY SIGNAL DIAGNOSTIC")
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

    # --- Indicators ONCE over the full closed series (causal math) -------
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)
    df = add_ema(df, EMA_TREND_PERIOD)   # diagnostic reference ONLY

    # ============================================================
    # SECTION 9: CHRONOLOGICAL 50/50 SPLIT (same methodology as
    # gold_exit_oos_validation.py)
    # ============================================================
    split_index = len(df) // 2
    dev_df = df.iloc[:split_index].reset_index(drop=True)
    oos_df = df.iloc[split_index:].reset_index(drop=True)

    dev_start, dev_end = dev_df["time"].iloc[0], dev_df["time"].iloc[-1]
    oos_start, oos_end = oos_df["time"].iloc[0], oos_df["time"].iloc[-1]

    print(f"Development period: {dev_start} -> {dev_end}")
    print(f"OOS period        : {oos_start} -> {oos_end}")
    print("Chronological split, nothing shuffled or mixed. Indicators")
    print("computed causally ONCE; OOS warms up on preceding history only.")
    print(f"Frozen exits: SL {SL_ATR_MULT} x ATR / TP {TP_ATR_MULT} x ATR "
          f"(Variant A baseline). Volume {volume} lots.")

    # ============================================================
    # SECTION 10: Run the frozen backtest per period
    # ============================================================
    trades_dev, bal_dev = run_backtest(dev_df, spec["value_per_unit"], volume,
                                       start_bar=WARMUP)
    trades_oos, bal_oos = run_backtest(oos_df, spec["value_per_unit"], volume,
                                       start_bar=0)

    if len(trades_dev) == 0 or len(trades_oos) == 0:
        print("ERROR: One of the periods produced no trades; "
              "diagnostic needs both. Stopping.")
        quit()

    # --- Context line: what we are diagnosing ---------------------------
    def quick_stats(trades, balance):
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        gp = wins["pnl"].sum()
        gl = abs(losses["pnl"].sum())
        ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
        pf = gp / gl if gl > 0 else float("inf")
        win_rate = len(wins) / len(trades) * 100
        return len(trades), win_rate, ret, pf, trades["pnl"].mean()

    n_dev, wr_dev, ret_dev, pf_dev, avg_dev = quick_stats(trades_dev, bal_dev)
    n_oos, wr_oos, ret_oos, pf_oos, avg_oos = quick_stats(trades_oos, bal_oos)

    print()
    print("Reproduced frozen strategy per period (context for the")
    print("diagnosis - these numbers explain WHAT we are diagnosing):")
    print(f"  Development: {n_dev} trades, return {ret_dev:+.2f}%, "
          f"win rate {wr_dev:.1f}%, PF {fmt_pf(pf_dev)}, "
          f"avg ${avg_dev:+.2f}")
    print(f"  OOS        : {n_oos} trades, return {ret_oos:+.2f}%, "
          f"win rate {wr_oos:.1f}%, PF {fmt_pf(pf_oos)}, "
          f"avg ${avg_oos:+.2f}")

    # ============================================================
    # SECTION 11: Descriptive bucket edges (from the FULL SIGNAL
    # DATASET, i.e. both periods' signals COMBINED - descriptive only,
    # never used to select, remove or rank anything)
    # ============================================================
    all_signals = pd.concat([trades_dev, trades_oos], ignore_index=True)

    exc_e1 = all_signals["excursion"].quantile(1 / 3)
    exc_e2 = all_signals["excursion"].quantile(2 / 3)
    vol_e1 = all_signals["ATR_pct"].quantile(1 / 3)
    vol_e2 = all_signals["ATR_pct"].quantile(2 / 3)

    # ============================================================
    # SECTION 12: The nine diagnostic sections
    # ============================================================
    section_long_short(trades_dev, trades_oos)
    section_adx(trades_dev, trades_oos)
    section_excursion(trades_dev, trades_oos, (exc_e1, exc_e2))
    section_volatility(trades_dev, trades_oos, (vol_e1, vol_e2))
    section_time_of_day(trades_dev, trades_oos)
    section_trend_distance(trades_dev, trades_oos)
    section_mfe_mae(trades_dev, trades_oos)
    section_forward(trades_dev, trades_oos)
    section_signal_quality(trades_dev, trades_oos,
                           dev_start, dev_end, oos_start, oos_end)

    # ============================================================
    # SECTION 13: FINAL FACTUAL SUMMARY (measured numbers only)
    # ============================================================
    print()
    print("=" * 62)
    print("FINAL FACTUAL SUMMARY")
    print("=" * 62)

    # Q1: LONG vs SHORT behavior change
    print()
    print("1. LONG vs SHORT between periods:")
    for direction, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        d = group_stats(trades_dev[trades_dev["direction"] == direction], dname)
        o = group_stats(trades_oos[trades_oos["direction"] == direction], dname)
        if d is None or o is None:
            print(f"   {dname}: insufficient trades in one period.")
            continue
        print(f"   {dname}: avg trade ${d['avg_trade']:+.2f} -> "
              f"${o['avg_trade']:+.2f} "
              f"({o['avg_trade'] - d['avg_trade']:+.2f}); "
              f"total ${d['total_pnl']:+.2f} -> ${o['total_pnl']:+.2f}")

    # Q2: ADX distribution change
    print()
    print("2. ADX distribution of signals:")
    print(f"   average {trades_dev['ADX'].mean():.1f} -> "
          f"{trades_oos['ADX'].mean():.1f}, "
          f"median {trades_dev['ADX'].median():.1f} -> "
          f"{trades_oos['ADX'].median():.1f}")
    for lo, hi, label in ADX_BUCKETS:
        d_pct = ((trades_dev["ADX"] >= lo) & (trades_dev["ADX"] < hi)).mean() * 100
        o_pct = ((trades_oos["ADX"] >= lo) & (trades_oos["ADX"] < hi)).mean() * 100
        print(f"   {label:<16}: {d_pct:.1f}% of DEV signals -> "
              f"{o_pct:.1f}% of OOS signals")

    # Q3: volatility change
    print()
    print("3. Volatility (ATR%) of signals:")
    print(f"   average {trades_dev['ATR_pct'].mean():.3f}% -> "
          f"{trades_oos['ATR_pct'].mean():.3f}% "
          f"({trades_oos['ATR_pct'].mean() - trades_dev['ATR_pct'].mean():+.3f} pp)")

    # Q4: EMA50 distance change
    print()
    print("4. Distance from EMA50 (ATR units):")
    print(f"   average {trades_dev['trend_dist'].mean():.2f} -> "
          f"{trades_oos['trend_dist'].mean():.2f}, "
          f"median {trades_dev['trend_dist'].median():.2f} -> "
          f"{trades_oos['trend_dist'].median():.2f}")

    # Q5: excursion change
    print()
    print("5. Bollinger excursion of signals:")
    print(f"   average {trades_dev['excursion'].mean():.4f} -> "
          f"{trades_oos['excursion'].mean():.4f}")
    for lo, hi, label in ((0.0, exc_e1, EXCURSION_LABELS[0]),
                          (exc_e1, exc_e2, EXCURSION_LABELS[1]),
                          (exc_e2, float("inf"), EXCURSION_LABELS[2])):
        d_pct = ((trades_dev["excursion"] >= lo) & (trades_dev["excursion"] < hi)).mean() * 100
        o_pct = ((trades_oos["excursion"] >= lo) & (trades_oos["excursion"] < hi)).mean() * 100
        print(f"   {label:<18}: {d_pct:.1f}% of DEV signals -> "
              f"{o_pct:.1f}% of OOS signals")

    # Q6: MFE/MAE change
    print()
    print("6. MFE/MAE (winners vs losers):")
    for period, trades in (("DEV", trades_dev), ("OOS", trades_oos)):
        w = trades[trades["pnl"] > 0]
        l = trades[trades["pnl"] < 0]
        print(f"   {period}: winners MFE {w['MFE_ATR'].mean():.2f} / "
              f"MAE {w['MAE_ATR'].mean():.2f} ATR | "
              f"losers MFE {l['MFE_ATR'].mean():.2f} / "
              f"MAE {l['MAE_ATR'].mean():.2f} ATR")

    # Q7: 12-candle forward return change
    print()
    print("7. 12-candle forward return (raw signal edge, no exits):")
    print(f"   all signals: {trades_dev['fwd_ret_12'].mean() * 100:+.4f}% -> "
          f"{trades_oos['fwd_ret_12'].mean() * 100:+.4f}% "
          f"per trade on average")
    for direction, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        d = trades_dev.loc[trades_dev["direction"] == direction, "fwd_ret_12"] * 100
        o = trades_oos.loc[trades_oos["direction"] == direction, "fwd_ret_12"] * 100
        print(f"   {dname:<6}: {d.mean():+.4f}% -> {o.mean():+.4f}%")

    # Q8: conditions vs exits - evidence-based verdict
    print()
    print("8. Market conditions vs exit structure:")
    # Evidence 1: raw signal edge (exits play no role here)
    fwd_dev = trades_dev["fwd_ret_12"].mean()
    fwd_oos = trades_oos["fwd_ret_12"].mean()
    # Evidence 2: MFE of losers (did price EVER revert toward the trade?)
    lw_dev = trades_dev.loc[trades_dev["pnl"] < 0, "MFE_ATR"].mean()
    lw_oos = trades_oos.loc[trades_oos["pnl"] < 0, "MFE_ATR"].mean()
    # Evidence 3: condition shifts
    cond_shifts = []
    cond_shifts.append(("ATR%",
                        trades_dev["ATR_pct"].mean(),
                        trades_oos["ATR_pct"].mean()))
    cond_shifts.append(("ADX",
                        trades_dev["ADX"].mean(),
                        trades_oos["ADX"].mean()))
    cond_shifts.append(("EMA50 dist",
                        trades_dev["trend_dist"].mean(),
                        trades_oos["trend_dist"].mean()))

    print(f"   Evidence 1 - raw 12-candle forward edge: "
          f"DEV {fwd_dev * 100:+.4f}% vs OOS {fwd_oos * 100:+.4f}% "
          f"(independent of exits)")
    print(f"   Evidence 2 - MFE of losing trades: "
          f"DEV {lw_dev:.2f} ATR vs OOS {lw_oos:.2f} ATR "
          f"(how far losing trades moved in favor first)")
    print("   Evidence 3 - condition shifts at signal time:")
    for name, dv, ov in cond_shifts:
        rel = abs(ov - dv) / abs(dv) * 100 if dv else 0.0
        print(f"      {name:<12}: {dv:.3f} -> {ov:.3f} ({rel:.0f}% change)")
    print()
    print("   Reading: if the RAW forward edge flipped sign or shrank, the")
    print("   change comes from the SIGNAL ENVIRONMENT (market conditions),")
    print("   which no exit structure can repair. The condition shifts in")
    print("   Evidence 3 describe HOW the environment differed. This is a")
    print("   measured description of one historical sample - not a")
    print("   prediction, not a recommendation, and no bucket is 'best'.")

    # ============================================================
    # SECTION 14: Disclaimer
    # ============================================================
    print()
    print("Historical diagnostic only. Virtual money only. No orders were")
    print("executed. Nothing was optimized, selected or removed. The")
    print("buckets are descriptive groupings, not strategy components.")

finally:
    # ============================================================
    # SECTION 15: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print()
    print("MetaTrader 5 connection closed.")
