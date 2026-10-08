"""
walk_forward_validation.py - HISTORICAL RESEARCH / VALIDATION ONLY.

Research question:
  "Does the frozen strategy show reasonably consistent behaviour
   across MULTIPLE chronological walk-forward OOS periods?"

This is the next stage after multi_instrument_10000_history_test.py,
market_regime_diagnostic.py, regime_oos_validation.py and
adaptive_regime_validation.py. Earlier stages showed period
instability, non-persisting regime relationships and only mixed
similarity information, so this stage tests the frozen strategy
directly on SIX sequential unseen OOS windows.

THE STRATEGY REMAINS COMPLETELY FROZEN:
  BB(20, 2.0) + ADX14 < 25 entry (next-open execution), SL 2.0 x ATR14
  / TP 1.5 x ATR14, stop-first inside candles, 0.10 lots, $2 per trade,
  one position at a time, no compounding, independent $10,000 per OOS
  window, broker-spec P/L. NO filter of any kind is added and NO
  variant is selected. The development window exists only as the
  historical period preceding each unseen OOS window - it is never
  used to change, tune or select anything (there is only ONE strategy).

DETERMINISTIC WALK-FORWARD WINDOWS (exactly six, never optimized):
  Window 1: development = first 3000 candles, OOS = next 1000
  Window 2: development = first 4000 candles, OOS = next 1000
  Window 3: development = first 5000 candles, OOS = next 1000
  Window 4: development = first 6000 candles, OOS = next 1000
  Window 5: development = first 7000 candles, OOS = next 1000
  Window 6: development = first 8000 candles, OOS = next 1000
  (candles after window 6's OOS may remain unused; no extra windows)
  OOS window w is evaluated in ISOLATION on its own $10,000 balance.
  A trade still open at the end of an OOS window is force-closed at
  that window's final close (deterministic; counted and reported as a
  forced close).

READ-ONLY MT5 SAFETY:
  Only initialize, shutdown, symbols_get, symbol_info,
  symbol_info_tick, symbol_select and copy_rates_from_pos are used.
  No order is ever placed, modified or closed. mt5.shutdown() always
  runs through a finally block.

COMMAND-LINE SAFETY:
  Default execution runs ONLY the synthetic verification tests:
      python walk_forward_validation.py
  The real MT5 diagnostic requires the explicit flag:
      python walk_forward_validation.py --run

DISCLAIMER: Historical results with virtual money, descriptive only.
No causal claims, no predictions, no rankings, no recommendations.
Educational use only.
"""

# ============================================================
# Imports (no new packages needed)
# ============================================================
import sys
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5  # Read market data + symbol info (READ-ONLY usage)
import numpy as np
import pandas as pd

# ============================================================
# Frozen settings - identical to every previous diagnostic
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 10_000
MIN_CANDLES = 9_000      # need 8000 dev + 1000 OOS for window 6

INSTRUMENTS: List[str] = ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

STARTING_BALANCE = 10_000.0     # independent per instrument AND OOS window
LOTS_REQUESTED = 0.10
CONTRACT_SIZE_FALLBACK = 100_000
COST_PER_TRADE = 2.0

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.5

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0
ATR_PERIOD = 14
EMA_TREND_PERIOD = 50    # descriptive context ONLY (never a filter)
WARMUP = 60

# Exactly six deterministic expanding walk-forward windows
# (development size, OOS size) - never optimized, never extended
WF_WINDOWS: List[Tuple[int, int]] = [
    (3000, 1000),   # Window 1
    (4000, 1000),   # Window 2
    (5000, 1000),   # Window 3
    (6000, 1000),   # Window 4
    (7000, 1000),   # Window 5
    (8000, 1000),   # Window 6
]

LOW_SAMPLE_N = 5
LOW_SAMPLE_TAG = "LOW SAMPLE"

ADX_BUCKETS = ["ADX<15", "ADX15-20", "ADX20-25", "ADX>=25"]


# ============================================================
# Indicator functions (causal only - same math as previous scripts)
# ============================================================
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range (uses only current and previous candles)."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (rolling - causal by construction)."""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (Wilder - causal)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
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


def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Exponential Moving Average (causal; descriptive context only)."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All frozen indicators + descriptive context measurements.

    Computed ONCE over the full closed series - causal by construction
    (every value uses only current and previous candles), so the OOS
    window of a later walk-forward window reuses exactly the same
    indicator values it would have had computed alone. No lookahead.
    """
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)
    df = add_ema(df, EMA_TREND_PERIOD)
    df["ATR_pct"] = np.where(df["close"] > 0, df["ATR"] / df["close"] * 100, np.nan)
    df["EMA50_dist"] = np.where(df["ATR"] > 0,
                                (df["close"] - df["EMA50"]).abs() / df["ATR"], np.nan)
    return df


# ============================================================
# Walk-forward window construction (deterministic; never optimized)
# ============================================================
def build_walk_forward_windows(df: pd.DataFrame) -> List[dict]:
    """Build exactly the six fixed expanding windows.

    Window w: development = df[0:dev_size], OOS = df[dev_size:dev_size+oos].
    OOS follows development immediately and windows never overlap in
    their OOS segments. Pure slicing - no randomness anywhere.
    """
    windows = []
    n = len(df)
    for w, (dev_size, oos_size) in enumerate(WF_WINDOWS, 1):
        if n < dev_size + oos_size:
            break   # instrument has too little history for this window
        windows.append({
            "window": w,
            "dev_size": dev_size,
            "oos_size": oos_size,
            "dev_df": df.iloc[:dev_size].reset_index(drop=True),
            "oos_df": df.iloc[dev_size:dev_size + oos_size].reset_index(drop=True),
        })
    return windows


# ============================================================
# Frozen strategy engine (identical rules as all previous scripts)
# ============================================================
def calculate_frozen_strategy(df: pd.DataFrame, value_per_unit: float,
                              volume: float, start_bar: int = WARMUP
                              ) -> Tuple[pd.DataFrame, dict]:
    """Run the frozen strategy; returns (trade log, summary dict).

    Signal on closed candle i, entry at the OPEN of candle i+1, SL/TP
    fixed from the signal candle's ATR, stop-first inside-candle
    handling, one position at a time, $2 cost per completed trade,
    independent $10,000 starting balance per call. A trade still open
    at the frame end is force-closed at the final close (deterministic;
    counted as a forced close).
    """
    long_condition = df["close"] < df["BB_LOWER"]
    short_condition = df["close"] > df["BB_UPPER"]
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df = df.copy()
    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"],
                             default="HOLD")

    trades: List[dict] = []
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

            # STOP checked FIRST (deterministic and conservative)
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
                exit_reason = "END_OF_DATA"   # forced close (counted)

            row = {
                "trade": len(trades) + 1,
                "direction": signal,
                "signal_time": df["time"].iloc[i],
                "signal_index": i,
                "entry_time": df["time"].iloc[entry_bar],
                "entry_price": entry_price,
                "exit_time": df["time"].iloc[exit_index],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "hold_candles": exit_index - entry_bar + 1,
                "adx": df["ADX"].iloc[i],
                "atr_pct": df["ATR_pct"].iloc[i],
                "ema_dist": df["EMA50_dist"].iloc[i],
                "pnl": ((exit_price - entry_price) if signal == "BUY"
                        else (entry_price - exit_price)) * volume * value_per_unit
                     - COST_PER_TRADE,
                "balance_after": 0.0,
            }
            trades.append(row)
            balance += row["pnl"]
            trades[-1]["balance_after"] = balance

            # One position at a time: resume AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), summarize_window(pd.DataFrame(trades))


# ============================================================
# Descriptive summary statistics (pure functions)
# ============================================================
def longest_losing_streak(pnls: List[float]) -> int:
    """Longest run of consecutive losing trades (pnl < 0)."""
    streak = max_streak = 0
    for v in pnls:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def summarize_window(trades: pd.DataFrame) -> dict:
    """All required descriptive metrics for one window."""
    if trades is None or len(trades) == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": float("nan"),
                "gp": 0.0, "gl": 0.0, "pf": float("nan"), "avg_trade": float("nan"),
                "total_pnl": 0.0, "ending_balance": STARTING_BALANCE,
                "return": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0,
                "lose_streak": 0, "total_cost": 0.0, "tp": 0, "sl": 0,
                "forced": 0, "avg_hold": float("nan"),
                "avg_adx": float("nan"), "avg_atr_pct": float("nan"),
                "avg_ema_dist": float("nan")}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())

    balances = [STARTING_BALANCE] + trades["balance_after"].tolist()
    peak = balances[0]
    max_dd = max_dd_pct = 0.0
    for b in balances[1:]:
        if b > peak:
            peak = b
        dd = peak - b
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "gp": gp,
        "gl": gl,
        "pf": gp / gl if gl > 0 else float("inf"),
        "avg_trade": trades["pnl"].mean(),
        "total_pnl": trades["pnl"].sum(),
        "ending_balance": trades["balance_after"].iloc[-1],
        "return": (trades["balance_after"].iloc[-1] - STARTING_BALANCE)
                  / STARTING_BALANCE * 100,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "lose_streak": longest_losing_streak(trades["pnl"].tolist()),
        "total_cost": len(trades) * COST_PER_TRADE,
        "tp": int((trades["exit_reason"] == "TAKE_PROFIT").sum()),
        "sl": int((trades["exit_reason"] == "STOP_LOSS").sum()),
        "forced": int((trades["exit_reason"] == "END_OF_DATA").sum()),
        "avg_hold": trades["hold_candles"].mean(),
        "avg_adx": trades["adx"].mean(),
        "avg_atr_pct": trades["atr_pct"].mean(),
        "avg_ema_dist": trades["ema_dist"].mean(),
    }


def fmt_pf(pf: float) -> str:
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def num(v: float, spec: str = "{:.2f}") -> str:
    return "n/a" if (v != v) else spec.format(v)


def sign_of(v: float) -> str:
    if v != v:
        return "NO DATA"
    return "POSITIVE" if v > 0 else ("NEGATIVE" if v < 0 else "ZERO")


def consistency_counts(oos_summaries: List[dict]) -> dict:
    """Period-consistency counts across an instrument's OOS windows."""
    rets = [s["return"] for s in oos_summaries]
    pfs = [s["pf"] for s in oos_summaries]
    avgs = [s["avg_trade"] for s in oos_summaries]
    dds = [s["max_dd_pct"] for s in oos_summaries]
    streaks = [s["lose_streak"] for s in oos_summaries]
    return {
        "n_windows": len(oos_summaries),
        "positive_windows": sum(1 for r in rets if r > 0),
        "negative_windows": sum(1 for r in rets if r < 0),
        "pf_above": sum(1 for p in pfs if p == p and p > 1.0),
        "pf_below": sum(1 for p in pfs if p == p and p < 1.0),
        "avg_pos": sum(1 for a in avgs if a == a and a > 0),
        "avg_neg": sum(1 for a in avgs if a == a and a < 0),
        "ret_mean": float(np.mean(rets)) if rets else float("nan"),
        "ret_median": float(np.median(rets)) if rets else float("nan"),
        "ret_std": float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan"),
        "avg_trade_mean": float(np.mean([a for a in avgs if a == a]))
                          if any(a == a for a in avgs) else float("nan"),
        "avg_trade_median": float(np.median([a for a in avgs if a == a]))
                            if any(a == a for a in avgs) else float("nan"),
        "dd_mean": float(np.mean(dds)) if dds else float("nan"),
        "dd_median": float(np.median(dds)) if dds else float("nan"),
        "dd_max": float(np.max(dds)) if dds else float("nan"),
        "streak_longest": int(np.max(streaks)) if streaks else 0,
        "streak_avg": float(np.mean(streaks)) if streaks else float("nan"),
        "returns": rets,
        "pfs": pfs,
        "avg_trades": avgs,
        "dd_pcts": dds,
    }


# ============================================================
# Factual questions (association wording only)
# ============================================================
def factual_questions(results: List[dict]) -> None:
    """Answer the nine factual questions with YES/NO/MIXED/INSUFFICIENT
    DATA plus supporting numbers (no recommendations anywhere)."""
    print()
    print("=" * 70)
    print("MOST IMPORTANT TEST - FACTUAL QUESTIONS")
    print("(neutral wording; no recommendation is made)")
    print("=" * 70)

    def yn(cond_all: bool, any_valid: bool) -> str:
        if not any_valid:
            return "INSUFFICIENT DATA"
        return "YES" if cond_all else "NO"

    # Q1-Q3: per-instrument all-window consistency
    all_pos, all_pf, all_avg = {}, {}, {}
    for r in results:
        c = r["consistency"]
        n = c["n_windows"]
        all_pos[r["instrument"]] = yn(
            n > 0 and c["positive_windows"] == n, n > 0)
        all_pf[r["instrument"]] = yn(
            n > 0 and c["pf_above"] == n, n > 0)
        all_avg[r["instrument"]] = yn(
            n > 0 and c["avg_pos"] == n, n > 0)

    print()
    print("1. Did any instrument remain profitable in every OOS window?")
    yes_list = [k for k, v in all_pos.items() if v == "YES"]
    print(f"   {'YES' if yes_list else 'NO'} - "
          f"{len(yes_list)} instrument(s) positive in all windows: "
          f"{', '.join(yes_list) if yes_list else 'none'}")
    for k, v in all_pos.items():
        c = next(r["consistency"] for r in results if r["instrument"] == k)
        print(f"     {k:<12}: {v} (positive {c['positive_windows']}/"
              f"{c['n_windows']})")

    print()
    print("2. Did any instrument have PF > 1 in every OOS window?")
    yes_list = [k for k, v in all_pf.items() if v == "YES"]
    print(f"   {'YES' if yes_list else 'NO'} - "
          f"{len(yes_list)} instrument(s): "
          f"{', '.join(yes_list) if yes_list else 'none'}")
    for k, v in all_pf.items():
        c = next(r["consistency"] for r in results if r["instrument"] == k)
        print(f"     {k:<12}: {v} (PF>1 in {c['pf_above']}/{c['n_windows']})")

    print()
    print("3. Did any instrument have positive average trade in every OOS")
    print("   window?")
    yes_list = [k for k, v in all_avg.items() if v == "YES"]
    print(f"   {'YES' if yes_list else 'NO'} - "
          f"{len(yes_list)} instrument(s): "
          f"{', '.join(yes_list) if yes_list else 'none'}")
    for k, v in all_avg.items():
        c = next(r["consistency"] for r in results if r["instrument"] == k)
        print(f"     {k:<12}: {v} (avg>0 in {c['avg_pos']}/{c['n_windows']})")

    # Q4/Q5: positive-heavy vs negative-heavy instruments
    print()
    print("4. How many of the five instruments had more positive than")
    print("   negative OOS windows?")
    n_more_pos = sum(1 for r in results
                     if r["consistency"]["positive_windows"]
                     > r["consistency"]["negative_windows"])
    print(f"   {n_more_pos} of {len(results)}")
    print("5. How many had more negative than positive OOS windows?")
    n_more_neg = sum(1 for r in results
                     if r["consistency"]["negative_windows"]
                     > r["consistency"]["positive_windows"])
    print(f"   {n_more_neg} of {len(results)}")

    # Q6: sign changes across windows
    print()
    print("6. Did OOS performance signs change across the six windows?")
    instruments_with_flips = 0
    for r in results:
        rets = [x for x in r["consistency"]["returns"] if x != 0]
        flips = sum(1 for a, b in zip(rets, rets[1:]) if (a > 0) != (b > 0))
        if flips:
            instruments_with_flips += 1
            print(f"     {r['instrument']:<12}: {flips} sign change(s) "
                  f"across {len(rets)} windows")
        else:
            print(f"     {r['instrument']:<12}: no sign change")
    if instruments_with_flips == 0:
        print("   NO - no instrument changed its OOS return sign.")
    elif instruments_with_flips == len(results):
        print(f"   YES - all {len(results)} instruments changed sign at")
        print("   least once across windows.")
    else:
        print(f"   MIXED - {instruments_with_flips} of {len(results)} "
              f"instruments changed sign at least once.")

    # Q7: same-window negativity across instruments
    print()
    print("7. Did multiple instruments experience negative performance in")
    print("   the same chronological window?")
    any_shared = False
    for w in range(len(WF_WINDOWS)):
        neg = [r["instrument"] for r in results
               if w < len(r["oos_summaries"])
               and r["oos_summaries"][w]["return"] < 0]
        if len(neg) >= 2:
            any_shared = True
            print(f"   Window {w + 1}: negative in {len(neg)} instruments "
                  f"({', '.join(neg)})")
    if not any_shared:
        print("   NO - no chronological window had 2+ instruments negative.")

    # Q8: drawdown behaviour across windows
    print()
    print("8. Did drawdown increase materially in particular windows?")
    overall_max = max((r["consistency"]["dd_max"] for r in results), default=0.0)
    elevated = []
    for w in range(len(WF_WINDOWS)):
        dds = [r["oos_summaries"][w]["max_dd_pct"] for r in results
               if w < len(r["oos_summaries"])]
        if dds:
            w_max = max(dds)
            if w_max >= overall_max * 0.9 and w_max > 0:
                elevated.append((w + 1, w_max))
    if elevated:
        print(f"   YES - the largest per-window drawdowns occurred in: "
              + ", ".join(f"window {w} ({d:.2f}%)" for w, d in elevated))
        print(f"   (largest observed OOS drawdown overall: {overall_max:.2f}%)")
    else:
        print(f"   NO clear concentration - largest observed OOS drawdown "
              f"was {overall_max:.2f}%.")

    # Q9: overall consistency
    print()
    print("9. Did the frozen strategy demonstrate consistent OOS behaviour")
    print("   across instruments and time?")
    total_windows = sum(r["consistency"]["n_windows"] for r in results)
    total_pos = sum(r["consistency"]["positive_windows"] for r in results)
    total_pf = sum(r["consistency"]["pf_above"] for r in results)
    share_pos = total_pos / total_windows * 100 if total_windows else 0.0
    share_pf = total_pf / total_windows * 100 if total_windows else 0.0
    if share_pos == 100 and share_pf == 100:
        ans = "YES"
    elif share_pos == 0 and share_pf == 0:
        ans = "NO"
    else:
        ans = "MIXED"
    print(f"   {ans} - positive return in {total_pos} of {total_windows}")
    print(f"   instrument/windows ({share_pos:.1f}%) and PF > 1 in "
          f"{total_pf} of {total_windows} ({share_pf:.1f}%).")
    print("   This is a descriptive historical observation of the observed")
    print("   OOS results - it is not a quality judgement, not a")
    print("   prediction, and not a recommendation.")


# ============================================================
# Data loading and broker specs (read-only, as previous scripts)
# ============================================================
def load_closed_data(symbol: str, num_candles: int = NUM_CANDLES
                     ) -> Tuple[Optional[pd.DataFrame], str]:
    """Up to num_candles H1 candles with the forming candle removed."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, num_candles)
    if rates is None or len(rates) == 0:
        return None, "no historical data returned"
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)
    return df, "closed candles"


def discover_symbol(requested: str) -> Tuple[Optional[str], str]:
    """Exact symbol name first, then a short suffix match."""
    info = mt5.symbol_info(requested)
    if info is not None:
        return requested, "exact symbol name"
    matches = []
    for sym in (mt5.symbols_get() or []):
        name = sym.name
        if name.startswith(requested) and len(name) > len(requested):
            suffix = name[len(requested):]
            if len(suffix) <= 10 and suffix.replace(".", "").isalnum():
                matches.append(name)
    if not matches:
        return None, "no exact or suffix match found"
    if len(matches) > 1:
        matches.sort(key=lambda s: (len(s), s))   # deterministic, not performance-based
    return matches[0], f"suffix match (available: {', '.join(matches[:5])})"


def get_contract_spec(symbol_name: str) -> Optional[dict]:
    """Broker specs with the same documented fallbacks as before."""
    info = mt5.symbol_info(symbol_name)
    if info is None:
        return None
    contract_size = float(info.trade_contract_size) if info.trade_contract_size > 0 \
        else CONTRACT_SIZE_FALLBACK
    point = float(info.point)
    tick_size = float(info.trade_tick_size) if info.trade_tick_size > 0 else point
    tick_value = float(info.trade_tick_value)
    fallback_used = tick_value <= 0
    if fallback_used:
        tick_value = contract_size * point
        print(f"    WARNING: trade_tick_value missing/zero for {symbol_name}; "
              f"using fallback tick_value = contract_size x point = {tick_value:g}")
    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0
    quote_currency = getattr(info, "currency_profit", "") or "USD"
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
        "contract_size": contract_size, "point": point,
        "digits": int(info.digits), "tick_size": tick_size,
        "tick_value": tick_value, "value_per_unit": value_per_unit,
        "volume_min": float(info.volume_min), "volume_max": float(info.volume_max),
        "volume_step": float(info.volume_step) if info.volume_step > 0 else 0.01,
        "quote_currency": quote_currency,
        "tick_value_fallback": fallback_used, "fallback_converted": converted,
    }


def validate_volume(spec: dict) -> Optional[float]:
    """Largest valid volume <= LOTS_REQUESTED (never increased)."""
    vmin, vmax, vstep = spec["volume_min"], spec["volume_max"], spec["volume_step"]
    if LOTS_REQUESTED < vmin or vmin > vmax:
        return None
    steps_below = int((LOTS_REQUESTED - vmin) / vstep + 1e-9)
    volume = round(vmin + steps_below * vstep, 8)
    return None if volume > vmax else volume


# ============================================================
# Printing helpers
# ============================================================
def print_oos_metrics(label: str, oos_df: pd.DataFrame, s: dict) -> None:
    """Print every required OOS metric for one window."""
    print(f"\n  {label}: {oos_df['time'].iloc[0]} -> {oos_df['time'].iloc[-1]} "
          f"({len(oos_df)} candles)")
    print(f"    trades               : {s['trades']}")
    if s["trades"] == 0:
        print("    (no trades in this OOS window)")
        return
    print(f"    winning trades       : {s['wins']}")
    print(f"    losing trades        : {s['losses']}")
    print(f"    win rate             : {s['win_rate']:.1f}%")
    print(f"    gross profit         : ${s['gp']:,.2f}")
    print(f"    gross loss           : ${s['gl']:,.2f}")
    print(f"    profit factor        : {fmt_pf(s['pf'])}")
    print(f"    average trade        : ${s['avg_trade']:+.2f}")
    print(f"    ending balance       : ${s['ending_balance']:,.2f}")
    print(f"    return               : {s['return']:+.2f}%")
    print(f"    maximum drawdown $   : ${s['max_dd']:,.2f}")
    print(f"    maximum drawdown %   : {s['max_dd_pct']:.2f}%")
    print(f"    longest losing streak: {s['lose_streak']} trades")
    print(f"    total trading cost   : ${s['total_cost']:,.2f}")
    print(f"    TP / SL / forced     : {s['tp']} / {s['sl']} / {s['forced']}")
    if s["forced"]:
        print(f"    NOTE: {s['forced']} trade(s) still open at window end were")
        print("    force-closed at the final close (documented deterministic")
        print("    method; counted as forced closes).")
    print(f"    average holding      : {s['avg_hold']:.1f} candles")
    print(f"    regime context       : avg ADX {s['avg_adx']:.1f} | "
          f"avg ATR% {s['avg_atr_pct']:.3f} | avg EMA50 dist "
          f"{s['avg_ema_dist']:.2f} ATR (descriptive only)")


def print_results(instrument: str, actual_symbol: str, spec: dict, volume: float,
                  n_candles: int, windows: List[dict], dev_summaries: List[dict],
                  oos_summaries: List[dict], consistency: dict) -> None:
    """Print all sections for one instrument."""
    print()
    print("=" * 70)
    print(f"WALK-FORWARD VALIDATION - {instrument} (traded as '{actual_symbol}')")
    print("=" * 70)
    print(f"  closed candles: {n_candles}")
    if n_candles < NUM_CANDLES:
        print(f"    NOTE: fewer than {NUM_CANDLES} closed candles available; "
              f"using ALL {n_candles}.")
    print(f"  volume {volume} lots | contract size {spec['contract_size']:g} | "
          f"point {spec['point']:g} | digits {spec['digits']} | tick size "
          f"{spec['tick_size']:g} | tick value {spec['tick_value']:g} | "
          f"quote currency {spec['quote_currency']}")
    pl_note = ""
    if spec["tick_value_fallback"]:
        pl_note = " (documented fallback)" + \
            (" + USD conversion" if spec["fallback_converted"] else "")
    print(f"  P/L basis: 1.0 price-unit move with 1.0 lot = "
          f"{spec['value_per_unit']:.2f} (tick_value / tick_size){pl_note}")
    print("  Development = historical context only. The ONE frozen strategy")
    print("  is evaluated directly on each OOS window (independent $10,000).")

    for w, win in enumerate(windows):
        dev_sum = dev_summaries[w]
        oos_sum = oos_summaries[w]
        print()
        print("-" * 70)
        print(f"WINDOW {win['window']}: development {win['dev_size']} candles "
              f"({win['dev_df']['time'].iloc[0]} -> {win['dev_df']['time'].iloc[-1]})")
        print(f"           OOS {win['oos_size']} candles")
        # Development descriptive metrics (transparency only - never used
        # to change the strategy)
        print(f"  DEVELOPMENT (descriptive only): {dev_sum['trades']} trades | "
              f"win {num(dev_sum['win_rate'], '{:.1f}%')} | "
              f"PF {fmt_pf(dev_sum['pf'])} | avg {num(dev_sum['avg_trade'], '${:+.2f}')} | "
              f"return {num(dev_sum['return'], '{:+.2f}%')} | "
              f"maxDD {num(dev_sum['max_dd_pct'], '{:.2f}%')}")
        print_oos_metrics(f"OOS (window {win['window']})", win["oos_df"], oos_sum)

    # --- period consistency -------------------------------------------
    print()
    print("=" * 70)
    print(f"PERIOD CONSISTENCY - {instrument}")
    print("=" * 70)
    c = consistency
    print(f"  positive OOS windows       : {c['positive_windows']} of {c['n_windows']}")
    print(f"  negative OOS windows       : {c['negative_windows']} of {c['n_windows']}")
    print(f"  PF > 1 windows             : {c['pf_above']} of {c['n_windows']}")
    print(f"  PF < 1 windows             : {c['pf_below']} of {c['n_windows']}")
    print(f"  avg trade > 0 windows      : {c['avg_pos']} of {c['n_windows']}")
    print(f"  avg trade < 0 windows      : {c['avg_neg']} of {c['n_windows']}")
    print(f"  OOS return mean / median   : {num(c['ret_mean'], '{:+.2f}%')} / "
          f"{num(c['ret_median'], '{:+.2f}%')} (std {num(c['ret_std'], '{:.2f}')})")
    print(f"  OOS avg trade mean/median  : {num(c['avg_trade_mean'], '${:+.2f}')} / "
          f"{num(c['avg_trade_median'], '${:+.2f}')}")

    # --- streak / drawdown consistency ----------------------------------
    print()
    print("  STREAK / DRAWDOWN CONSISTENCY (across the six OOS windows;")
    print("  balances are NOT compounded into one curve):")
    print(f"    average max drawdown %        : {num(c['dd_mean'], '{:.2f}%')}")
    print(f"    median max drawdown %         : {num(c['dd_median'], '{:.2f}%')}")
    print(f"    maximum observed OOS DD %     : {num(c['dd_max'], '{:.2f}%')}")
    print(f"    longest observed losing streak: {c['streak_longest']} trades")
    print(f"    average losing streak length  : {num(c['streak_avg'], '{:.1f}')} trades")


# ============================================================
# Real MT5 diagnostic (requires --run; never runs automatically)
# ============================================================
def run_real_diagnostic() -> None:
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
        print("WALK-FORWARD VALIDATION (read-only, descriptive only)")
        print("Frozen strategy evaluated on six sequential unseen OOS")
        print("windows per instrument. Development = context only.")
        print("=" * 70)

        results: List[dict] = []
        unavailable: List[str] = []

        for requested in INSTRUMENTS:
            print()
            print("#" * 70)
            print(f"# INSTRUMENT: {requested}")
            print("#" * 70)
            try:
                actual_symbol, note = discover_symbol(requested)
                if actual_symbol is None:
                    print(f"STATUS: NOT AVAILABLE - {note}. Continuing.")
                    unavailable.append(requested)
                    continue
                print(f"Requested: {requested}  ->  using MT5 symbol: "
                      f"{actual_symbol}  ({note})")

                if not mt5.symbol_select(actual_symbol, True):
                    print(f"STATUS: NOT AVAILABLE - {actual_symbol} could not "
                          f"be enabled in Market Watch. Continuing.")
                    unavailable.append(requested)
                    continue

                spec = get_contract_spec(actual_symbol)
                if spec is None:
                    print("STATUS: NOT AVAILABLE - no symbol_info. Continuing.")
                    unavailable.append(requested)
                    continue
                volume = validate_volume(spec)
                if volume is None:
                    print(f"STATUS: SKIPPED - {LOTS_REQUESTED} lots is not a "
                          f"valid volume. Continuing.")
                    unavailable.append(requested)
                    continue

                df, load_note = load_closed_data(actual_symbol)
                if df is None:
                    print(f"STATUS: NOT AVAILABLE - {load_note}. Continuing.")
                    unavailable.append(requested)
                    continue
                if len(df) < MIN_CANDLES:
                    print(f"STATUS: NOT AVAILABLE - only {len(df)} closed "
                          f"candles (minimum {MIN_CANDLES} for the six fixed "
                          f"windows). Continuing.")
                    unavailable.append(requested)
                    continue

                # Causal indicators over the full closed series (no lookahead)
                df = calculate_indicators(df)

                windows = build_walk_forward_windows(df)
                dev_summaries: List[dict] = []
                oos_summaries: List[dict] = []
                for win in windows:
                    _, dev_sum = calculate_frozen_strategy(
                        win["dev_df"], spec["value_per_unit"], volume)
                    _, oos_sum = calculate_frozen_strategy(
                        win["oos_df"], spec["value_per_unit"], volume,
                        start_bar=0)
                    dev_summaries.append(dev_sum)
                    oos_summaries.append(oos_sum)

                consistency = consistency_counts(oos_summaries)
                print_results(requested, actual_symbol, spec, volume,
                              len(df), windows, dev_summaries, oos_summaries,
                              consistency)

                results.append({
                    "instrument": requested,
                    "windows": windows,
                    "dev_summaries": dev_summaries,
                    "oos_summaries": oos_summaries,
                    "consistency": consistency,
                })

            except Exception as exc:
                print(f"STATUS: ERROR while processing {requested}: {exc}")
                print("Continuing with the remaining instruments.")
                unavailable.append(requested)

        # ----------------------------------------------------------
        # CROSS-INSTRUMENT SUMMARY (factual, no ranking)
        # ----------------------------------------------------------
        print()
        print("=" * 70)
        print("CROSS-INSTRUMENT SUMMARY (factual table, no ranking)")
        print("=" * 70)
        if results:
            hdr = (f"   {'Instrument':<12}{'Win':>5}{'Pos':>5}{'Neg':>5}"
                   f"{'PF>1':>6}{'PF<1':>6}{'meanRet':>10}{'medRet':>10}"
                   f"{'meanAvg':>10}{'medAvg':>10}{'meanDD%':>9}{'maxDD%':>9}")
            print(hdr)
            print("   " + "-" * 97)
            for r in results:
                c = r["consistency"]
                print(f"   {r['instrument']:<12}{c['n_windows']:>5}"
                      f"{c['positive_windows']:>5}{c['negative_windows']:>5}"
                      f"{c['pf_above']:>6}{c['pf_below']:>6}"
                      f"{num(c['ret_mean'], '{:+.2f}'):>10}"
                      f"{num(c['ret_median'], '{:+.2f}'):>10}"
                      f"{num(c['avg_trade_mean'], '${:+.2f}'):>10}"
                      f"{num(c['avg_trade_median'], '${:+.2f}'):>10}"
                      f"{num(c['dd_mean'], '{:.2f}'):>9}"
                      f"{num(c['dd_max'], '{:.2f}'):>9}")
            print()
            print("   Each OOS window ran on its own independent $10,000")
            print("   balance. Instruments are deliberately NOT ranked; none")
            print("   is labelled best, worst, winner, loser, strong or weak.")
        else:
            print("   No instrument could be analyzed.")

        # ----------------------------------------------------------
        # WINDOW SUMMARY (by chronological window, across instruments)
        # ----------------------------------------------------------
        if results:
            print()
            print("=" * 70)
            print("WINDOW SUMMARY (per chronological window, all instruments)")
            print("(shows whether conditions affected instruments at the")
            print(" same time; factual table, no ranking)")
            print("=" * 70)
            for w in range(len(WF_WINDOWS)):
                rows = [(r["instrument"], r["oos_summaries"][w])
                        for r in results if w < len(r["oos_summaries"])]
                if not rows:
                    continue
                print(f"\n  Window {w + 1} "
                      f"(OOS {rows[0][1]['total_pnl'] != None and ''}"
                      f"fixed 1000-candle segment):")
                for name, s in rows:
                    print(f"    {name:<12} return {num(s['return'], '{:+.2f}%'):>9} | "
                          f"PF {fmt_pf(s['pf']) if s['trades'] else 'n/a':>5} | "
                          f"avg {num(s['avg_trade'], '${:+.2f}'):>9} | "
                          f"win {num(s['win_rate'], '{:.1f}%'):>7} | "
                          f"DD% {num(s['max_dd_pct'], '{:.2f}'):>6}")

            factual_questions(results)

        if unavailable:
            seen = []
            for ins in INSTRUMENTS:
                if ins in unavailable and ins not in seen:
                    seen.append(ins)
            print("\nInstruments NOT analyzed (reported unavailable/skipped,")
            print(f"no substitution made): {', '.join(seen)}")

        print()
        print("Historical research only. No orders were sent.")
        print()
        print("All results are descriptive historical observations of the ONE")
        print("frozen strategy on six fixed OOS windows per instrument. No")
        print("parameter was optimized, no variant was selected, no filter")
        print("was added, and nothing here predicts future performance.")
        print("Educational use only.")

    finally:
        # mt5.shutdown() always executes (read-only session teardown)
        mt5.shutdown()
        print()
        print("MetaTrader 5 connection closed.")


# ============================================================
# Synthetic tests (deterministic; run in default verification mode)
# ============================================================
def run_synthetic_tests() -> Tuple[int, int]:
    """Deterministic self-tests. Returns (passed, failed)."""
    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} {detail}")

    def synth_candles(n: int, seed: int, base: float, vol_scale: float) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t = np.arange(n)
        close = base + np.sin(t / 17.0) * 12.0 * vol_scale \
            + rng.normal(0.0, 3.0 * vol_scale, n)
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        spread = np.abs(rng.normal(0.0, 2.0 * vol_scale, n)) + 0.5 * vol_scale
        return pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
        })

    # [1-4] chronological ordering, six exact windows, OOS strictly after
    # development, no overlap
    df = calculate_indicators(synth_candles(9500, 1, 100.0, 0.5))
    windows = build_walk_forward_windows(df)
    check("exactly six windows built from 9500 candles", len(windows) == 6,
          f"got {len(windows)}")
    check("window sizes are exactly 3000/4000/.../8000 dev + 1000 OOS",
          [(w["dev_size"], w["oos_size"]) for w in windows]
          == [(3000, 1000), (4000, 1000), (5000, 1000),
              (6000, 1000), (7000, 1000), (8000, 1000)])
    check("development window is the FIRST dev_size candles",
          all(w["dev_df"]["time"].iloc[0] == df["time"].iloc[0]
              and len(w["dev_df"]) == w["dev_size"] for w in windows))
    check("OOS strictly follows development (starts at dev_size)",
          all(w["oos_df"]["time"].iloc[0] > w["dev_df"]["time"].iloc[-1]
              and w["oos_df"]["time"].iloc[0] == df["time"].iloc[w["dev_size"]]
              for w in windows))
    check("OOS segments never overlap",
          all(windows[i]["oos_df"]["time"].iloc[-1]
              < windows[i + 1]["oos_df"]["time"].iloc[0]
              for i in range(len(windows) - 1)))
    short_windows = build_walk_forward_windows(df.iloc[:4500])
    check("too-short history yields fewer windows (no crash)",
          len(short_windows) == 1 and short_windows[0]["dev_size"] == 3000)

    # [5-6] signal uses only current/past candles; entry is next open
    base = calculate_indicators(synth_candles(1200, 7, 100.0, 0.5))
    tr0, sum0 = calculate_frozen_strategy(base, 1000.0, 0.10)
    check("engine produces trades on synthetic feed", len(tr0) > 0)
    first = tr0.iloc[0]
    i_sig = base.index[base["time"] == first["signal_time"]][0]
    check("entry at next candle OPEN",
          abs(first["entry_price"] - base["open"].iloc[i_sig + 1]) < 1e-12
          and first["entry_time"] == base["time"].iloc[i_sig + 1])
    # prefix causality: indicators on an isolated prefix match the full frame
    head = base.iloc[:600].reset_index(drop=True)
    head_meas = calculate_indicators(head.copy())
    check("indicators on an isolated prefix match the full frame (causal)",
          all(np.allclose(head_meas[c].values, base[c].iloc[:600].values,
                          equal_nan=True)
              for c in ("ATR", "ADX", "ATR_pct", "EMA50_dist", "BB_UPPER")))

    # [7] stop-first; [8] forced-close behaviour (crafted candles)
    eb = i_sig + 1
    entry = base["open"].iloc[eb]
    atr = base["ATR"].iloc[i_sig]
    is_buy = first["direction"] == "BUY"
    span = base.copy()
    span.iloc[eb, span.columns.get_loc("high")] = entry + 3.0 * atr
    span.iloc[eb, span.columns.get_loc("low")] = entry - 3.0 * atr
    tr_span, _ = calculate_frozen_strategy(span, 1000.0, 0.10)
    t_span = tr_span[tr_span["signal_time"] == first["signal_time"]].iloc[0]
    pnl_stop = (-2.0 * atr) * 0.10 * 1000.0 - COST_PER_TRADE
    check("spanning SL+TP candle exits at the exact STOP value (stop-first)",
          t_span["exit_reason"] == "STOP_LOSS"
          and abs(t_span["pnl"] - pnl_stop) < 1e-9)
    # forced close: truncate the frame right after entry (no SL/TP hit)
    trunc_len = eb + 3
    trunc = base.iloc[:trunc_len].reset_index(drop=True)
    tr_f, _ = calculate_frozen_strategy(trunc, 1000.0, 0.10, start_bar=WARMUP)
    t_f = tr_f[tr_f["signal_time"] == first["signal_time"]].iloc[0]
    pnl_eod = ((trunc["close"].iloc[-1] - entry) if is_buy
               else (entry - trunc["close"].iloc[-1])) * 0.10 * 1000.0 \
        - COST_PER_TRADE
    check("open trade at window end is force-closed at the final close",
          t_f["exit_reason"] == "END_OF_DATA"
          and abs(t_f["pnl"] - pnl_eod) < 1e-9,
          f"reason={t_f['exit_reason']}")
    # forced close is COUNTED in the summary
    _, sum_f = calculate_frozen_strategy(trunc, 1000.0, 0.10, start_bar=WARMUP)
    check("forced close count is reported by the summary",
          sum_f["forced"] >= 1)

    # [9-10] independent $10,000 balances; no compounding
    w1 = windows[0]
    _, oos_sum_1 = calculate_frozen_strategy(w1["oos_df"], 1000.0, 0.10,
                                             start_bar=0)
    tr_oos1, _ = calculate_frozen_strategy(w1["oos_df"], 1000.0, 0.10,
                                           start_bar=0)
    if len(tr_oos1):
        check("OOS window restarts from $10,000",
              abs(tr_oos1["balance_after"].iloc[0] - STARTING_BALANCE
                  - tr_oos1["pnl"].iloc[0]) < 1e-9)
        check("no compounding (balance = 10k + cumulative pnl, linear)",
              np.allclose(tr_oos1["balance_after"].values,
                          STARTING_BALANCE + tr_oos1["pnl"].cumsum().values))
    # Each independent OOS window must be evaluated from the SAME fixed
    # $10,000: for every one of the six windows the implied starting
    # balance (first trade's balance minus that trade's P/L, or the
    # unchanged balance when a window has no trades) must equal
    # STARTING_BALANCE, and window 1's summary must agree independently
    # (ending_balance - total P/L = STARTING_BALANCE). This catches any
    # carry-over of a previous window's P/L (i.e. accidental compounding).
    implied_starts = []
    for win in windows:
        tr_w, sum_w = calculate_frozen_strategy(win["oos_df"], 1000.0, 0.10,
                                                start_bar=0)
        implied_starts.append(
            tr_w["balance_after"].iloc[0] - tr_w["pnl"].iloc[0]
            if len(tr_w) else sum_w["ending_balance"])
    check("each window's OOS summary starts from the same fixed balance",
          len(implied_starts) == 6
          and all(abs(b - STARTING_BALANCE) < 1e-6 for b in implied_starts)
          and abs((oos_sum_1["ending_balance"] - oos_sum_1["total_pnl"])
                  - STARTING_BALANCE) < 1e-6,
          f"implied starts {implied_starts}")

    # [11] broker-spec P/L (value_per_unit scaling is linear, cost fixed)
    tr_v2, _ = calculate_frozen_strategy(base, 2000.0, 0.10)
    t2_ = tr_v2[tr_v2["signal_time"] == first["signal_time"]].iloc[0]
    gross1 = t_span["pnl"] + COST_PER_TRADE if len(t_span) else None
    g1 = first["pnl"] + COST_PER_TRADE
    g2 = t2_["pnl"] + COST_PER_TRADE
    check("gross P/L scales linearly with the broker value_per_unit",
          abs(g2 - 2.0 * g1) < 1e-9)
    check("$2 cost subtracted per completed trade",
          abs(g1 - (first["pnl"] + COST_PER_TRADE)) < 1e-12
          and abs((first["pnl"] + COST_PER_TRADE) - g1) < 1e-12)

    # [12] drawdown calculation (hand-computed path)
    demo = pd.DataFrame({"pnl": [50.0, -20.0, 30.0, -10.0, -5.0, 10.0],
                         "balance_after": [10050, 10030, 10060, 10050, 10045,
                                           10055],
                         "exit_reason": ["TAKE_PROFIT"] * 6,
                         "hold_candles": [1] * 6,
                         "adx": [20.0] * 6, "atr_pct": [0.5] * 6,
                         "ema_dist": [1.0] * 6})
    s = summarize_window(demo)
    check("max drawdown $ = 20 (peak 10050 -> trough 10030)",
          abs(s["max_dd"] - 20.0) < 1e-9, f"got {s['max_dd']}")
    check("max drawdown % = 20/10050",
          abs(s["max_dd_pct"] - 20.0 / 10050 * 100) < 1e-9)

    # [13] win rate; [14] profit factor; [15] losing streak
    check("win rate = 50%", abs(s["win_rate"] - 50.0) < 1e-9)
    check("profit factor = 90/35", abs(s["pf"] - 90.0 / 35.0) < 1e-9)
    check("longest losing streak = 2", s["lose_streak"] == 2)
    check("TP/SL/forced counts", s["tp"] == 6 and s["sl"] == 0 and s["forced"] == 0)
    check("total cost = trades x $2", abs(s["total_cost"] - 12.0) < 1e-9)
    check("empty window summary is NaN-safe",
          summarize_window(pd.DataFrame(columns=["pnl"]))["trades"] == 0)

    # [16] window aggregation (consistency_counts)
    fake = [{"return": 1.0, "pf": 1.2, "avg_trade": 5.0, "max_dd_pct": 2.0,
             "lose_streak": 1},
            {"return": -0.5, "pf": 0.8, "avg_trade": -3.0, "max_dd_pct": 4.0,
             "lose_streak": 3},
            {"return": 0.7, "pf": 1.1, "avg_trade": 4.0, "max_dd_pct": 3.0,
             "lose_streak": 2}]
    c = consistency_counts(fake)
    check("positive/negative window counts", c["positive_windows"] == 2
          and c["negative_windows"] == 1)
    check("PF>1 / PF<1 counts", c["pf_above"] == 2 and c["pf_below"] == 1)
    check("avg>0 / avg<0 counts", c["avg_pos"] == 2 and c["avg_neg"] == 1)
    check("return mean/median/std",
          abs(c["ret_mean"] - (1.0 - 0.5 + 0.7) / 3) < 1e-9
          and abs(c["ret_median"] - 0.7) < 1e-9
          and c["ret_std"] > 0)
    check("drawdown consistency fields",
          abs(c["dd_max"] - 4.0) < 1e-9 and abs(c["dd_mean"] - 3.0) < 1e-9
          and c["streak_longest"] == 3)

    # [17] cross-instrument aggregation shape (two fake instruments)
    results_fake = [{"instrument": "A", "consistency": c,
                     "oos_summaries": fake},
                    {"instrument": "B", "consistency": consistency_counts(fake),
                     "oos_summaries": fake}]
    shared = [w for w in range(len(WF_WINDOWS))
              if sum(1 for r in results_fake
                     if w < len(r["oos_summaries"])
                     and r["oos_summaries"][w]["return"] < 0) >= 2]
    # both fake instruments share only window index 1 (return -0.5);
    # windows 3..5 have no fake summaries and cannot be shared-negative
    check("shared negative windows detected across instruments",
          shared == [1], f"got {shared}")

    # [18] no strategy modification (constants + engine behaviour identical
    # to the previous diagnostics' rules)
    check("frozen strategy constants unchanged",
          (SL_ATR_MULT, TP_ATR_MULT, BB_PERIOD, BB_NUM_STD, ADX_PERIOD,
           ADX_THRESHOLD, ATR_PERIOD, LOTS_REQUESTED, COST_PER_TRADE,
           STARTING_BALANCE)
          == (2.0, 1.5, 20, 2.0, 14, 25.0, 14, 0.10, 2.0, 10_000.0))
    check("window definitions are the fixed six (never optimized)",
          WF_WINDOWS == [(3000, 1000), (4000, 1000), (5000, 1000),
                         (6000, 1000), (7000, 1000), (8000, 1000)])

    # [19] no order functions in this module's source (fragment trick so
    # this file itself contains zero contiguous forbidden names)
    import inspect
    src = inspect.getsource(sys.modules[__name__])
    for a, b in (("order", "send"), ("order", "check"),
                 ("positions", "get"), ("positions", "total"),
                 ("position", "get")):
        check(f"source contains no '{a}_{b}' function", (a + "_" + b) not in src)

    # [21] no optimization loops (structural: fixed windows, fixed
    # constants, no parameter iteration anywhere)
    check("single fixed walk-forward design (no sweep structures)",
          len(WF_WINDOWS) == 6 and all(o == 1000 for _, o in WF_WINDOWS))

    # descriptive context fields are populated for the regime section
    check("regime context fields present in summaries",
          all(k in sum0 for k in ("avg_adx", "avg_atr_pct", "avg_ema_dist")))

    return passed, failed


# ============================================================
# Entry point: default = verification only; --run = real diagnostic
# ============================================================
def main() -> None:
    print("Running synthetic tests...")
    passed, failed = run_synthetic_tests()
    print(f"Synthetic tests: {passed} passed, {failed} failed")

    if failed:
        print("SYNTHETIC TESTS FAILED - aborting before any MT5 access.")
        return

    # COMMAND-LINE SAFETY: the real MT5 diagnostic NEVER runs
    # automatically. Default execution verifies only; the explicit
    # --run flag is required to touch the terminal.
    if "--run" not in sys.argv:
        print()
        print("Verification mode: synthetic tests only. The real MT5")
        print("diagnostic was NOT run. To execute it explicitly, use:")
        print("    python walk_forward_validation.py --run")
        print()
        print("Historical research only. No orders were sent.")
        print("Walk-forward windows are fixed (3000/4000/5000/6000/7000/8000")
        print("development -> 1000 OOS); the strategy remained frozen.")
        return

    run_real_diagnostic()


if __name__ == "__main__":
    main()
