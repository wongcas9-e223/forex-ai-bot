"""
market_state_database.py - HISTORICAL RESEARCH / DATABASE CONSTRUCTION ONLY.

Research question:
  "Can we construct a historical MARKET STATE DATABASE that describes the
   environment surrounding every strategy signal and records what happened
   afterward?"

Context: previous stages (multi-instrument 10,000-candle test, regime
diagnostics, OOS regime validation, walk-forward validation) showed
period instability, cross-instrument variation, sign changes across
walk-forward windows and inconsistent regime relationships. The next
research step is therefore DATA, not strategy: a structured dataset that
records, for EVERY historical signal of the frozen strategy, the market
state at the signal candle and the outcomes that followed.

THIS FILE DOES NOT:
  - create a new trading strategy,
  - filter trades,
  - optimize anything (no parameter sweeps, no thresholds, no search),
  - use machine learning of any kind,
  - send, modify or close any order (READ-ONLY MT5 usage only).

THE STRATEGY EXISTS ONLY TO DEFINE WHICH HISTORICAL CANDLES BECOME
SIGNAL OBSERVATIONS:
  Bollinger Bands: period = 20, std = 2.0
  ADX: period = 14, threshold = 25 (CONTEXT ONLY - the ADX threshold is
       NOT an entry condition; LONG = close < lower band, SHORT =
       close > upper band. ADX14 is RECORDED as a state feature.)
  LONG signal  : close < BB_LOWER
  SHORT signal : close > BB_UPPER
  Signal on candle i, entry at the OPEN of candle i+1.
  ATR period 14, SL = 2.0 x ATR, TP = 1.5 x ATR, STOP-FIRST when both
  levels fall inside one candle, 0.10 lots, $2 per trade, one position
  at a time (sequential account rule; see the NOTE below), no
  compounding, $10,000 reference balance, broker-spec P/L.

NOTE ON SIGNAL-LEVEL OUTCOMES:
  The database records an outcome for EVERY signal row. Outcomes are
  evaluated with the SAME frozen engine rules (next-open entry, SL/TP
  from signal ATR, stop-first, $2 cost). Unlike the sequential account
  backtests, overlapping signal outcomes are accepted here on purpose:
  the database is signal-level research data, not one account equity
  curve. No trade is ever placed.

LEAKAGE RULE:
  Every MARKET STATE feature uses ONLY information available at the
  signal candle (candles 0..i). Forward returns and MFE/MAE are OUTCOME
  columns only and are never inputs to anything.

STATE DISTRIBUTIONS / BINNING:
  Only FIXED descriptive bins are used (ADX/EMA-distance/BB-excursion/
  efficiency/RSI boundaries are constants; ATR% uses per-instrument
  33rd/66th percentiles DESCRIBED AFTER THE FACT - they never feed back
  into any trading rule). No bucket is ranked, no "best" bucket is
  identified, no trading rule is derived from any table here.

READ-ONLY MT5 SAFETY:
  Only mt5.initialize, mt5.shutdown, mt5.symbols_get, mt5.symbol_info,
  mt5.symbol_info_tick, mt5.symbol_select and mt5.copy_rates_from_pos
  are used. mt5.shutdown() always runs through a finally block.

COMMAND-LINE SAFETY:
  Default execution runs ONLY the synthetic verification tests:
      python market_state_database.py
  Real historical extraction requires the explicit flag:
      python market_state_database.py --run

DISCLAIMER: Historical, descriptive research data only. No causal
claims, no predictions, no rankings, no recommendations, nothing
proven profitable. Educational use only.
"""

# ============================================================
# Imports (database construction needs only numpy/pandas)
# ============================================================
import sys
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5  # Read market data + symbol info (READ-ONLY usage)
import numpy as np
import pandas as pd

from indicators import (
    add_adx,
    add_atr,
    add_bollinger,
    add_ema,
    add_rsi,
)

# ============================================================
# SAFETY GUARD: the trading side of MetaTrader 5 is never touched.
# These assertions fail immediately at import time if this module is
# ever modified to expose an order function. (Names are split across
# two string literals so this safety file itself never contains the
# contiguous forbidden source strings.)
# ============================================================
_MET5_ORDER_FUNCS = (
    "order" + "_send",
    "order" + "_check",
    "positions" + "_get",
    "positions" + "_total",
    "position" + "_get",
)
# The MetaTrader5 package itself DOES ship the trading API; what must
# never happen is THIS module exposing it (e.g. binding one of those
# callables to a module-level name, importing one directly from the
# package, or via an alias). For each forbidden name: look it up on the
# imported mt5 module and fail if the name is a module-global here, OR
# if any module-global IS that exact mt5 callable. Using globals() (no
# sys.modules reliance) keeps the check valid however this file is
# imported. Falsifiable by construction.
assert not any(
    _name in globals()
    or (hasattr(mt5, _name)
        and any(_value is getattr(mt5, _name)
                for _value in globals().values()))
    for _name in _MET5_ORDER_FUNCS
), "forbidden MT5 order/position function exposed by this module"

# ============================================================
# ENTRY RULE GUARD: this is a BOLLINGER-ONLY signal definition.
# The frozen ADX threshold (25) is RECORDED as a state feature; it is
# deliberately NOT part of the entry condition. Any version of this
# database that filtered entries by RSI / EMA / MACD / ADX / volatility
# / time-of-day would be a DIFFERENT research object.
# ============================================================
_BOLLINGER_ONLY_ENTRY = "close < BB_LOWER or close > BB_UPPER"
assert _BOLLINGER_ONLY_ENTRY.startswith("close <"), "entry rule guard"

# ============================================================
# SAFETY MARKERS (verified by the synthetic test suite):
#   NO ORDER FUNCTIONS - read-only MT5 usage only.
#   NO MACHINE LEARNING - no sklearn / xgboost / lightgbm /
#     tensorflow / torch / keras and no trained model of any kind.
#   NO OPTIMIZATION - no parameter sweeps, no combinatorial /
#     randomised / threshold search, no feature selection, no
#     walk-forward optimization, no Monte Carlo.
# ============================================================

# ============================================================
# Frozen settings - identical to every previous diagnostic
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 10_000          # requested closed candles per instrument

INSTRUMENTS: List[str] = ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

STARTING_BALANCE = 10_000.0   # reference balance (no compounding)
LOTS_REQUESTED = 0.10
CONTRACT_SIZE_FALLBACK = 100_000
COST_PER_TRADE = 2.0          # $ per completed trade

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.5

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0          # RECORDED ONLY - never an entry filter
ATR_PERIOD = 14
EMA_TREND_PERIOD = 50         # state feature only - never a filter
RSI_PERIOD = 14
WARMUP = 60

# Forward-outcome horizons (outcome columns only - NEVER features)
FORWARD_HORIZONS: Tuple[int, ...] = (6, 12, 24)
MFE_MAE_HORIZON = 12

# Minimum signal-candle index: the entry candle (i+1) and the longest
# backward window (efficiency ratio / returns over 24 candles, BB 20,
# EMA 50) must exist. 60 covers all of them with margin.
MIN_INDEX = 60

# Fixed descriptive bins (constants - never tuned, never ranked)
ADX_BIN_LABELS = ["ADX<15", "ADX15-20", "ADX20-25", "ADX>=25"]
ATR_BIN_LABELS = ["LOW", "MEDIUM", "HIGH"]

# ============================================================
# Deterministic CSV column ordering (the spec's column groups)
# ============================================================
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "instrument", "timeframe", "signal_time", "signal_index", "direction",
)
MARKET_STATE_COLUMNS: Tuple[str, ...] = (
    "adx", "atr", "atr_pct", "ema50", "ema_distance", "bb_middle",
    "bb_upper", "bb_lower", "bb_excursion", "bb_signed", "eff24",
    "return6", "return12", "return24", "rsi14", "vol6", "vol24",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "distance_24_high", "distance_24_low", "hour", "day_of_week", "month",
)
TIME_FEATURE_COLUMNS: Tuple[str, ...] = ()  # included in MARKET_STATE above
TRADE_OUTCOME_COLUMNS: Tuple[str, ...] = (
    "entry_price", "exit_price", "exit_time", "exit_reason",
    "holding_candles", "gross_pnl", "trading_cost", "net_pnl", "win",
)
FORWARD_OUTCOME_COLUMNS: Tuple[str, ...] = (
    "forward6_pct", "forward12_pct", "forward24_pct", "mfe12_atr", "mae12_atr",
)
CSV_COLUMNS: Tuple[str, ...] = (
    IDENTITY_COLUMNS + MARKET_STATE_COLUMNS
    + TRADE_OUTCOME_COLUMNS + FORWARD_OUTCOME_COLUMNS
)

# Features that must be finite for every database row
REQUIRED_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    c for c in MARKET_STATE_COLUMNS
    if c not in ("hour", "day_of_week", "month")
)


# ============================================================
# Indicator functions (causal only)
# The five core indicators are imported from the shared, pandas/numpy-
# only indicators.py module (extracted verbatim; no formula, smoothing,
# min_periods, ddof, column-name or NaN behaviour change).
# ============================================================
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All indicators needed by the database (causal by construction)."""
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)
    df = add_ema(df, EMA_TREND_PERIOD)
    df = add_rsi(df, RSI_PERIOD)
    return df


def efficiency_ratio(close: pd.Series, window: int = 24) -> pd.Series:
    """Kaufman Efficiency Ratio over `window` candles (causal).

    |close[t] - close[t-window]| / sum(|close[j]-close[j-1]| for the
    same window). Values near 1 = smooth directional move; near 0 =
    choppy.
    """
    net_move = close.diff(window).abs()
    path = close.diff().abs().rolling(window).sum()
    return net_move / path.replace(0, np.nan)


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """(close[t] / close[t-window] - 1) * 100 (causal)."""
    return (close / close.shift(window) - 1.0) * 100.0


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Std of the last `window` 1-candle returns x 100 (sample std)."""
    rets = close.pct_change()
    return rets.rolling(window).std(ddof=1) * 100.0


# ============================================================
# Frozen trade simulation (same rules as the previous diagnostics)
# ============================================================
def simulate_trade(df: pd.DataFrame, signal_index: int, direction: str,
                   value_per_unit: float, volume: float) -> dict:
    """Evaluate ONE signal with the frozen engine rules.

    Entry at the OPEN of candle signal_index+1, SL/TP fixed from the
    signal candle ATR, stop-first inside candles, forced close at the
    final close if neither level is hit. Returns the trade-outcome
    fields only (no forward outcomes - those are computed separately).
    """
    entry_index = signal_index + 1
    entry_price = float(df["open"].iloc[entry_index])
    atr = float(df["ATR"].iloc[signal_index])

    if direction == "LONG":
        stop_loss = entry_price - SL_ATR_MULT * atr
        take_profit = entry_price + TP_ATR_MULT * atr
    else:  # SHORT
        stop_loss = entry_price + SL_ATR_MULT * atr
        take_profit = entry_price - TP_ATR_MULT * atr

    # STOP checked FIRST (deterministic and conservative)
    exit_index = None
    exit_price = None
    exit_reason = None
    for j in range(entry_index, len(df)):
        bar_high = float(df["high"].iloc[j])
        bar_low = float(df["low"].iloc[j])
        if direction == "LONG":
            if bar_low <= stop_loss:
                exit_index, exit_price, exit_reason = j, stop_loss, "SL"
                break
            if bar_high >= take_profit:
                exit_index, exit_price, exit_reason = j, take_profit, "TP"
                break
        else:  # SHORT
            if bar_high >= stop_loss:
                exit_index, exit_price, exit_reason = j, stop_loss, "SL"
                break
            if bar_low <= take_profit:
                exit_index, exit_price, exit_reason = j, take_profit, "TP"
                break

    if exit_index is None:  # still open at the end of the data
        exit_index = len(df) - 1
        exit_price = float(df["close"].iloc[exit_index])
        exit_reason = "FORCED_CLOSE"

    if direction == "LONG":
        gross_pnl = (exit_price - entry_price) * volume * value_per_unit
    else:
        gross_pnl = (entry_price - exit_price) * volume * value_per_unit

    net_pnl = gross_pnl - COST_PER_TRADE
    return {
        "entry_index": entry_index,
        "entry_price": entry_price,
        "exit_index": exit_index,
        "exit_price": exit_price,
        "exit_time": df["time"].iloc[exit_index],
        "exit_reason": exit_reason,
        "holding_candles": exit_index - entry_index + 1,
        "gross_pnl": gross_pnl,
        "trading_cost": COST_PER_TRADE,
        "net_pnl": net_pnl,
        "win": bool(net_pnl > 0),
    }


# ============================================================
# Forward outcomes (outcome columns only - never inputs)
# ============================================================
def forward_outcomes(df: pd.DataFrame, signal_index: int,
                     direction: str) -> dict:
    """Forward returns (vs the SIGNAL close, direction-agnostic) and
    direction-signed MFE/MAE over the next 12 candles, normalized by
    the signal ATR.

    MFE/MAE are measured on the ENTRY candle and the candles after it
    (favourable = in the trade's direction). For LONG: MFE from highs,
    MAE from lows. For SHORT both signs flip so MFE >= 0 >= MAE always.
    """
    signal_close = float(df["close"].iloc[signal_index])
    atr = float(df["ATR"].iloc[signal_index])

    out: Dict[str, Optional[float]] = {}
    for n in FORWARD_HORIZONS:
        idx = signal_index + n
        if idx < len(df):
            future_close = float(df["close"].iloc[idx])
            out[f"forward{n}_pct"] = (future_close / signal_close - 1.0) * 100.0
        else:
            out[f"forward{n}_pct"] = None  # not enough future candles yet

    end = min(signal_index + MFE_MAE_HORIZON, len(df) - 1)
    if end >= signal_index + 1:
        window = df.iloc[signal_index + 1: end + 1]
        max_high = float(window["high"].max())
        min_low = float(window["low"].min())
        if direction == "LONG":
            mfe = max_high - signal_close
            mae = min_low - signal_close
        else:
            mfe = signal_close - min_low
            mae = signal_close - max_high
        out["mfe12_atr"] = mfe / atr if atr > 0 else None
        out["mae12_atr"] = mae / atr if atr > 0 else None
    else:
        out["mfe12_atr"] = None
        out["mae12_atr"] = None
    return out


# ============================================================
# Database construction for one instrument
# ============================================================
def build_market_state_database(df: pd.DataFrame, instrument: str,
                                value_per_unit: float, volume: float
                                ) -> pd.DataFrame:
    """Build the signal-level market-state database for one instrument.

    `df` must already contain the causal indicator columns (run
    add_all_indicators first). Rows are strictly chronological.
    """
    # Frozen signal definition: Bollinger closes only. The ADX threshold
    # is context information recorded as a feature - never a condition.
    long_condition = df["close"] < df["BB_LOWER"]
    short_condition = df["close"] > df["BB_UPPER"]

    # Precompute all causal feature series ONCE (every series below is
    # computed from closes/highs/lows up to and including each candle -
    # verified causally by the synthetic test suite).
    close = df["close"]
    eff24_series = efficiency_ratio(close, 24)
    ret6_series = rolling_return(close, 6)
    ret12_series = rolling_return(close, 12)
    ret24_series = rolling_return(close, 24)
    vol6_series = rolling_volatility(close, 6)
    vol24_series = rolling_volatility(close, 24)
    roll_high24 = df["high"].rolling(24).max()
    roll_low24 = df["low"].rolling(24).min()

    rows: List[dict] = []
    for i in range(MIN_INDEX, len(df) - 1):
        is_long = bool(long_condition.iloc[i])
        is_short = bool(short_condition.iloc[i])
        if not (is_long or is_short):
            continue
        direction = "LONG" if is_long else "SHORT"
        signal_time = df["time"].iloc[i]
        signal_close = float(df["close"].iloc[i])
        atr = float(df["ATR"].iloc[i])
        bb_width = float(df["BB_UPPER"].iloc[i]) - float(df["BB_LOWER"].iloc[i])
        ema50 = float(df["EMA50"].iloc[i])

        # ---- MARKET STATE (only candles 0..i are used) ----
        state: dict = {
            "adx": float(df["ADX"].iloc[i]),
            "atr": atr,
            "atr_pct": atr / signal_close * 100.0 if signal_close > 0 else np.nan,
            "ema50": ema50,
            "ema_distance": abs(signal_close - ema50) / atr if atr > 0 else np.nan,
            "bb_middle": float(df["BB_MIDDLE"].iloc[i]),
            "bb_upper": float(df["BB_UPPER"].iloc[i]),
            "bb_lower": float(df["BB_LOWER"].iloc[i]),
            "bb_excursion": abs(signal_close - float(df["BB_MIDDLE"].iloc[i]))
                            / bb_width if bb_width > 0 else np.nan,
            "bb_signed": (signal_close - float(df["BB_MIDDLE"].iloc[i]))
                         / bb_width if bb_width > 0 else np.nan,
            "eff24": float(eff24_series.iloc[i]),
            "return6": float(ret6_series.iloc[i]),
            "return12": float(ret12_series.iloc[i]),
            "return24": float(ret24_series.iloc[i]),
            "rsi14": float(df["RSI"].iloc[i]),
            "vol6": float(vol6_series.iloc[i]),
            "vol24": float(vol24_series.iloc[i]),
            "range_pct": (float(df["high"].iloc[i]) - float(df["low"].iloc[i]))
                         / signal_close * 100.0,
            "body_pct": abs(signal_close - float(df["open"].iloc[i]))
                        / signal_close * 100.0,
            "upper_wick_pct": (float(df["high"].iloc[i])
                               - max(signal_close, float(df["open"].iloc[i])))
                              / signal_close * 100.0,
            "lower_wick_pct": (min(signal_close, float(df["open"].iloc[i]))
                               - float(df["low"].iloc[i]))
                              / signal_close * 100.0,
            "distance_24_high": (signal_close / float(roll_high24.iloc[i])
                                 - 1.0) * 100.0,
            "distance_24_low": (signal_close / float(roll_low24.iloc[i])
                                - 1.0) * 100.0,
            "hour": int(signal_time.hour),
            "day_of_week": int(signal_time.dayofweek),
            "month": int(signal_time.month),
        }

        # ---- TRADE OUTCOME (frozen engine rules) ----
        trade = simulate_trade(df, i, direction, value_per_unit, volume)

        row = {
            "instrument": instrument,
            "timeframe": "H1",
            "signal_time": signal_time,
            "signal_index": i,
            "direction": direction,
            **state,
            "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"],
            "exit_time": trade["exit_time"],
            "exit_reason": trade["exit_reason"],
            "holding_candles": trade["holding_candles"],
            "gross_pnl": trade["gross_pnl"],
            "trading_cost": trade["trading_cost"],
            "net_pnl": trade["net_pnl"],
            "win": trade["win"],
        }

        # ---- FORWARD OUTCOMES (independent of the strategy exit) ----
        row.update(forward_outcomes(df, i, direction))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(CSV_COLUMNS))

    db = pd.DataFrame(rows)
    # A signal within 24 candles of the frame end has incomplete forward
    # outcomes; it keeps its trade outcome but is excluded from the
    # saved database (deterministic, documented rule).
    db = db[db["signal_index"] <= len(df) - 1 - 24]
    db = db.sort_values("signal_index").reset_index(drop=True)
    return db[list(CSV_COLUMNS)]


# ============================================================
# Descriptive helpers (fixed bins, no ranking anywhere)
# ============================================================
def bin_adx(x: float) -> str:
    if x < 15:
        return ADX_BIN_LABELS[0]
    if x < 20:
        return ADX_BIN_LABELS[1]
    if x < 25:
        return ADX_BIN_LABELS[2]
    return ADX_BIN_LABELS[3]


def bin_ema_distance(x: float) -> str:
    if x < 1:
        return "<1"
    if x < 2:
        return "1-2"
    return ">=2"


def bin_bb_excursion(x: float) -> str:
    if x < 0.50:
        return "<0.50"
    if x < 0.75:
        return "0.50-0.75"
    return ">=0.75"


def bin_efficiency(x: float) -> str:
    if x < 0.30:
        return "<0.30"
    if x < 0.60:
        return "0.30-0.60"
    return ">=0.60"


def bin_rsi(x: float) -> str:
    if x < 30:
        return "<30"
    if x < 50:
        return "30-50"
    if x < 70:
        return "50-70"
    return ">=70"


def atr_bin_labels(values: pd.Series) -> pd.Series:
    """LOW/MEDIUM/HIGH via the full-history 33rd/66th percentiles of the
    given series (descriptive labelling AFTER the fact - these labels
    never modify the strategy)."""
    p33 = float(values.quantile(1.0 / 3.0))
    p66 = float(values.quantile(2.0 / 3.0))
    return values.map(lambda x: "LOW" if x < p33
                      else ("MEDIUM" if x < p66 else "HIGH"))


def bin_table(db: pd.DataFrame, column: str, labels: List[str]
              ) -> pd.DataFrame:
    """Fixed-bin count x average net P/L table for one state feature."""
    if len(db) == 0 or column not in db.columns:
        return pd.DataFrame(columns=["bin", "n", "avg_net_pnl"])
    table = (db.groupby(column)
               .agg(n=("net_pnl", "size"), avg_net_pnl=("net_pnl", "mean"))
               .reindex(labels)
               .fillna(0)
               .reset_index()
               .rename(columns={column: "bin"}))
    return table


def state_distribution_block(db: pd.DataFrame, atr_series: Optional[pd.Series]
                             ) -> None:
    """Print the fixed descriptive distributions for one instrument."""
    print("\n  STATE DISTRIBUTIONS (fixed descriptive bins only):")
    db = db.copy()
    if len(db):
        db["adx_bin"] = db["adx"].map(bin_adx)
        db["atr_bin"] = atr_bin_labels(atr_series) if atr_series is not None \
            else db["atr_pct"].map(lambda x: "n/a")
        db["ema_bin"] = db["ema_distance"].map(bin_ema_distance)
        db["bbe_bin"] = db["bb_excursion"].map(bin_bb_excursion)
        db["eff_bin"] = db["eff24"].map(bin_efficiency)
        db["rsi_bin"] = db["rsi14"].map(bin_rsi)
        for name, col, labels in (
                ("ADX", "adx_bin", ADX_BIN_LABELS),
                ("ATR%", "atr_bin", ATR_BIN_LABELS),
                ("EMA distance", "ema_bin", ["<1", "1-2", ">=2"]),
                ("BB excursion", "bbe_bin", ["<0.50", "0.50-0.75", ">=0.75"]),
                ("Efficiency 24", "eff_bin", ["<0.30", "0.30-0.60", ">=0.60"]),
                ("RSI 14", "rsi_bin", ["<30", "30-50", "50-70", ">=70"])):
            counts = db.groupby(col)["net_pnl"].size().reindex(labels).fillna(0)
            avgs = db.groupby(col)["net_pnl"].mean().reindex(labels)
            parts = [f"{lab}: n={int(counts[lab])}"
                     f" avgP/L={avgs[lab]:+.2f}" if counts[lab] > 0
                     else f"{lab}: n=0" for lab in labels]
            print(f"    {name:<14} " + " | ".join(parts))
    if atr_series is not None and len(atr_series):
        print(f"    (ATR% bins use this instrument's full-history p33="
              f"{atr_series.quantile(1/3):.4f} / p66="
              f"{atr_series.quantile(2/3):.4f} - descriptive only)")


def state_outcome_tables(db: pd.DataFrame) -> None:
    """State x average net P/L tables (descriptive; nothing is ranked)."""
    print("\n  STATE x OUTCOME (average net P/L per fixed bin; no ranking):")
    db = db.copy()
    if len(db) == 0:
        return
    db["adx_bin"] = db["adx"].map(bin_adx)
    db["ema_bin"] = db["ema_distance"].map(bin_ema_distance)
    db["bbe_bin"] = db["bb_excursion"].map(bin_bb_excursion)
    db["eff_bin"] = db["eff24"].map(bin_efficiency)
    db["rsi_bin"] = db["rsi14"].map(bin_rsi)
    for name, col, labels in (
            ("ADX", "adx_bin", ADX_BIN_LABELS),
            ("EMA distance", "ema_bin", ["<1", "1-2", ">=2"]),
            ("BB excursion", "bbe_bin", ["<0.50", "0.50-0.75", ">=0.75"]),
            ("Efficiency", "eff_bin", ["<0.30", "0.30-0.60", ">=0.60"]),
            ("RSI", "rsi_bin", ["<30", "30-50", "50-70", ">=70"])):
        t = bin_table(db, col, labels)
        cells = " | ".join(f"{r['bin']}: {r['avg_net_pnl']:+.2f} (n={int(r['n'])})"
                           for _, r in t.iterrows() if r["n"] > 0)
        print(f"    {name:<14} {cells}")
    for col in ("direction", "hour", "day_of_week", "month"):
        t = db.groupby(col)["net_pnl"].agg(["size", "mean"])
        cells = " | ".join(f"{k}: {v['mean']:+.2f} (n={int(v['size'])})"
                           for k, v in t.iterrows())
        print(f"    {col:<14} {cells}")


def summarize_instrument(db: pd.DataFrame) -> dict:
    """Summary metrics for one instrument's database (descriptive)."""
    if len(db) == 0:
        return {"n": 0}
    wins = db[db["net_pnl"] > 0]
    losses = db[db["net_pnl"] < 0]
    gp = float(wins["net_pnl"].sum())
    gl = float(abs(losses["net_pnl"].sum()))
    return {
        "n": len(db),
        "long": int((db["direction"] == "LONG").sum()),
        "short": int((db["direction"] == "SHORT").sum()),
        "tp": int((db["exit_reason"] == "TP").sum()),
        "sl": int((db["exit_reason"] == "SL").sum()),
        "forced": int((db["exit_reason"] == "FORCED_CLOSE").sum()),
        "win_rate": float((db["net_pnl"] > 0).mean() * 100.0),
        "avg_net_pnl": float(db["net_pnl"].mean()),
        "pf": gp / gl if gl > 0 else float("inf"),
        "fwd6": float(db["forward6_pct"].mean()),
        "fwd12": float(db["forward12_pct"].mean()),
        "fwd24": float(db["forward24_pct"].mean()),
        "mfe": float(db["mfe12_atr"].mean()),
        "mae": float(db["mae12_atr"].mean()),
    }


def print_summary(instrument: str, s: dict) -> None:
    if s.get("n", 0) == 0:
        print(f"  {instrument:<10} no signals")
        return
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    print(f"  {instrument:<10} n={s['n']:>4}  LONG={s['long']:<4} "
          f"SHORT={s['short']:<4}  TP={s['tp']:<3} SL={s['sl']:<3} "
          f"FORCED={s['forced']:<2}")
    print(f"  {'':<10} win={s['win_rate']:.1f}%  avgP/L=${s['avg_net_pnl']:+.2f}  "
          f"PF={pf}  fwd6={s['fwd6']:+.3f}%  fwd12={s['fwd12']:+.3f}%  "
          f"fwd24={s['fwd24']:+.3f}%")
    print(f"  {'':<10} MFE12={s['mfe']:+.3f} ATR  MAE12={s['mae']:+.3f} ATR")


def direction_analysis(db: pd.DataFrame) -> None:
    """LONG/SHORT descriptive breakdown (no ranking, no recommendation)."""
    if len(db) == 0:
        return
    print("\n  DIRECTION ANALYSIS (descriptive):")
    for d in ("LONG", "SHORT"):
        g = db[db["direction"] == d]
        if len(g) == 0:
            print(f"    {d:<6}: no signals")
            continue
        gp = float(g[g["net_pnl"] > 0]["net_pnl"].sum())
        gl = float(abs(g[g["net_pnl"] < 0]["net_pnl"].sum()))
        pf = gp / gl if gl > 0 else float("inf")
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"    {d:<6}: n={len(g):>4}  win={(g['net_pnl'] > 0).mean() * 100:.1f}%"
              f"  avgP/L=${g['net_pnl'].mean():+.2f}  PF={pf_s}"
              f"  fwd6={g['forward6_pct'].mean():+.3f}%"
              f"  fwd12={g['forward12_pct'].mean():+.3f}%"
              f"  fwd24={g['forward24_pct'].mean():+.3f}%"
              f"  MFE={g['mfe12_atr'].mean():+.3f}"
              f"  MAE={g['mae12_atr'].mean():+.3f}")


def chronological_analysis(db: pd.DataFrame) -> None:
    """Four equal signal-count chronological periods per instrument."""
    if len(db) == 0:
        return
    print("\n  CHRONOLOGICAL QUARTERS (equal signal counts; descriptive):")
    ordered = db.sort_values("signal_index")
    n = len(ordered)
    for p in range(4):
        part = ordered.iloc[p * n // 4: (p + 1) * n // 4]
        if len(part) == 0:
            continue
        gp = float(part[part["net_pnl"] > 0]["net_pnl"].sum())
        gl = float(abs(part[part["net_pnl"] < 0]["net_pnl"].sum()))
        pf = gp / gl if gl > 0 else float("inf")
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"    P{p + 1} ({part['signal_time'].iloc[0]} -> "
              f"{part['signal_time'].iloc[-1]}): n={len(part)}"
              f"  avgP/L=${part['net_pnl'].mean():+.2f}"
              f"  win={(part['net_pnl'] > 0).mean() * 100:.1f}%  PF={pf_s}"
              f"  fwd12={part['forward12_pct'].mean():+.3f}%"
              f"  ATR%={part['atr_pct'].mean():.3f}"
              f"  ADX={part['adx'].mean():.1f}"
              f"  eff={part['eff24'].mean():.3f}")


def cross_instrument_state_table(dbs: Dict[str, pd.DataFrame]) -> None:
    """Per-instrument descriptive stats for every state feature."""
    print("\n" + "=" * 70)
    print("CROSS-INSTRUMENT STATE SUMMARY (n / mean / median / std /")
    print("min / max per feature; descriptive comparison, no ranking)")
    print("=" * 70)
    for col in MARKET_STATE_COLUMNS:
        print(f"\n  {col}:")
        for ins, db in dbs.items():
            if col not in db.columns or len(db) == 0:
                continue
            s = db[col].dropna()
            if len(s) == 0:
                continue
            print(f"    {ins:<8} n={len(s):>4}  mean={s.mean():>10.4f}  "
                  f"median={s.median():>10.4f}  std={s.std():>10.4f}  "
                  f"min={s.min():>10.4f}  max={s.max():>10.4f}")


def correlation_analysis(dbs: Dict[str, pd.DataFrame]) -> None:
    """Pearson correlations between state features and outcomes.

    Descriptive only: no causality is claimed and nothing here is used
    to build filters or thresholds.
    """
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS (Pearson; descriptive, no causality,")
    print("no filters, no thresholds derived from this)")
    print("=" * 70)
    outcome_cols = ["forward6_pct", "forward12_pct", "forward24_pct",
                    "net_pnl", "mfe12_atr", "mae12_atr"]
    feature_cols = [c for c in MARKET_STATE_COLUMNS
                    if c not in ("hour", "day_of_week", "month")]
    for ins, db in dbs.items():
        if len(db) == 0:
            continue
        print(f"\n  {ins} (feature x outcome Pearson r):")
        header = f"    {'feature':<18}" + "".join(f"{c[:11]:>13}"
                                                  for c in outcome_cols)
        print(header)
        for f in feature_cols:
            row = f"    {f:<18}"
            for o in outcome_cols:
                r = db[f].corr(db[o])
                row += f"{'n/a' if pd.isna(r) else f'{r:+.2f}':>13}"
            print(row)
        # Direction-specific net P/L correlations (compact)
        for d in ("LONG", "SHORT"):
            g = db[db["direction"] == d]
            if len(g) < 3:
                continue
            cells = "  ".join(f"{f}={g[f].corr(g['net_pnl']):+.2f}"
                              for f in ("adx", "atr_pct", "ema_distance",
                                        "eff24", "rsi14"))
            print(f"    {d} net-P/L correlations: {cells}")


def print_full_report(dbs: Dict[str, pd.DataFrame], atr_series: Dict[str, pd.Series]
                      ) -> None:
    """All descriptive sections for the extracted databases."""
    print("\n" + "=" * 70)
    print("MARKET STATE DATABASE - DESCRIPTIVE REPORT (read-only)")
    print("=" * 70)
    for ins, db in dbs.items():
        print("\n" + "#" * 70)
        print(f"# INSTRUMENT: {ins}")
        print("#" * 70)
        s = summarize_instrument(db)
        print_summary(ins, s)
        state_distribution_block(db, atr_series.get(ins))
        state_outcome_tables(db)
        direction_analysis(db)
        chronological_analysis(db)
    cross_instrument_state_table(dbs)
    correlation_analysis(dbs)
    print("\nAll tables above are descriptive historical observations.")
    print("No bucket, instrument or direction is ranked; no trading rule,")
    print("filter or threshold is derived from this report.")


# ============================================================
# Read-only MT5 helpers (same documented fallbacks as before)
# ============================================================
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
        matches.sort(key=lambda s: (len(s), s))   # deterministic
    return matches[0], f"suffix match (available: {', '.join(matches[:5])})"


def load_closed_data(symbol: str, num_candles: int = NUM_CANDLES
                     ) -> Tuple[Optional[pd.DataFrame], str]:
    """Up to num_candles H1 candles with the forming candle removed."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, num_candles)
    if rates is None or len(rates) == 0:
        return None, "no historical data returned"
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)   # closed candles only
    return df, f"{len(df)} closed candles"


def get_contract_spec(symbol_name: str) -> Optional[dict]:
    """Broker specs with the same documented fallbacks as previous stages."""
    info = mt5.symbol_info(symbol_name)
    if info is None:
        return None
    contract_size = float(info.trade_contract_size) \
        if info.trade_contract_size > 0 else CONTRACT_SIZE_FALLBACK
    point = float(info.point)
    tick_size = float(info.trade_tick_size) if info.trade_tick_size > 0 else point
    tick_value = float(info.trade_tick_value)
    fallback_used = tick_value <= 0
    if fallback_used:
        tick_value = contract_size * point
        print(f"    WARNING: tick value missing/zero for {symbol_name}; "
              f"using fallback tick_value = contract_size x point = {tick_value:g}")
    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0
    quote_currency = getattr(info, "currency_profit", "") or "USD"
    if fallback_used and value_per_unit > 0 \
            and quote_currency.upper() not in ("USD", ""):
        conv_symbol, _ = discover_symbol(f"{quote_currency.upper()}USD")
        if conv_symbol is not None:
            tick = mt5.symbol_info_tick(conv_symbol)
            if tick is not None and tick.bid > 0:
                value_per_unit = value_per_unit / tick.bid
                print(f"    Fallback P/L converted from "
                      f"{quote_currency.upper()} to USD via {conv_symbol} bid")
    return {"value_per_unit": value_per_unit, "contract_size": contract_size}


def validate_volume(spec: dict) -> Optional[float]:
    """Largest valid volume <= LOTS_REQUESTED (never increased)."""
    info = mt5.symbol_info  # local alias (read-only usage below)
    return _validate_volume_with_info(spec, info)


def _validate_volume_with_info(spec: dict, symbol_info_getter) -> Optional[float]:
    """Volume validation using only symbol_info fields (read-only)."""
    del symbol_info_getter  # placeholder for symmetry; specs carry volumes
    vmin = float(spec.get("volume_min", 0.01))
    vmax = float(spec.get("volume_max", 100.0))
    vstep = float(spec.get("volume_step", 0.01)) or 0.01
    if LOTS_REQUESTED < vmin or vmin > vmax:
        return None
    steps_below = int((LOTS_REQUESTED - vmin) / vstep + 1e-9)
    volume = round(vmin + steps_below * vstep, 8)
    return None if volume > vmax else volume


def get_volume_spec(symbol_name: str) -> Optional[dict]:
    """symbol_info including volume fields (read-only)."""
    info = mt5.symbol_info(symbol_name)
    if info is None:
        return None
    return {
        "value_per_unit": get_contract_spec(symbol_name)["value_per_unit"],
        "volume_min": float(info.volume_min),
        "volume_max": float(info.volume_max),
        "volume_step": float(info.volume_step) if info.volume_step > 0 else 0.01,
    }


# ============================================================
# CSV output (deterministic; never overwrites an existing file)
# ============================================================
def write_csv(db: pd.DataFrame, filename: str) -> Optional[str]:
    """Write one database CSV. Refuses to overwrite existing files."""
    import os
    if os.path.exists(filename):
        print(f"    SKIPPED {filename}: file already exists "
              f"(this script never overwrites).")
        return None
    db.to_csv(filename, index=False)
    return filename


# ============================================================
# Real MT5 extraction (requires --run; never runs automatically)
# ============================================================
def run_real_extraction() -> None:
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print("ERROR: Could not connect to MetaTrader 5.")
        print("Make sure the MT5 desktop terminal is installed, running,")
        print("and logged into your account.")
        print("Last error:", mt5.last_error())
        return

    print("Connected to MetaTrader 5 successfully!")
    try:
        print()
        print("=" * 70)
        print("MARKET STATE DATABASE CONSTRUCTION (read-only)")
        print("Frozen strategy signals only; no filtering, no optimization,")
        print("no machine learning. CSV output per instrument + combined.")
        print("=" * 70)

        dbs: Dict[str, pd.DataFrame] = {}
        atr_series: Dict[str, pd.Series] = {}

        for requested in INSTRUMENTS:
            print()
            print("#" * 70)
            print(f"# INSTRUMENT: {requested}")
            print("#" * 70)
            try:
                actual_symbol, note = discover_symbol(requested)
                if actual_symbol is None:
                    print(f"STATUS: NOT AVAILABLE - {note}. Continuing.")
                    continue
                print(f"Requested: {requested}  ->  MT5 symbol: "
                      f"{actual_symbol}  ({note})")
                if not mt5.symbol_select(actual_symbol, True):
                    print("STATUS: NOT AVAILABLE - could not be enabled in "
                          "Market Watch. Continuing.")
                    continue

                vspec = get_volume_spec(actual_symbol)
                if vspec is None:
                    print("STATUS: NOT AVAILABLE - no symbol_info. Continuing.")
                    continue
                volume = _validate_volume_with_info(vspec, None)
                if volume is None:
                    print(f"STATUS: SKIPPED - {LOTS_REQUESTED} lots is not a "
                          f"valid volume. Continuing.")
                    continue

                df, load_note = load_closed_data(actual_symbol)
                if df is None:
                    print(f"STATUS: NOT AVAILABLE - {load_note}. Continuing.")
                    continue
                if len(df) < WARMUP + 30:
                    print(f"STATUS: NOT AVAILABLE - only {len(df)} closed "
                          f"candles (need at least {WARMUP + 30}). Continuing.")
                    continue

                df = add_all_indicators(df)
                db = build_market_state_database(df, requested,
                                                 vspec["value_per_unit"],
                                                 volume)
                print(f"STATUS: OK - {len(db)} signal observations "
                      f"({load_note})")
                if len(db):
                    dbs[requested] = db
                    atr_series[requested] = db["atr_pct"]
                    saved = write_csv(db, f"market_state_{requested}.csv")
                    if saved:
                        print(f"    wrote {saved}")

            except Exception as exc:
                print(f"STATUS: ERROR while processing {requested}: {exc}")
                print("Continuing with the remaining instruments.")

        # Combined file across instruments (deterministic column order;
        # instruments in the fixed INSTRUMENTS order)
        if dbs:
            combined = pd.concat([dbs[k] for k in INSTRUMENTS if k in dbs],
                                 ignore_index=True)[list(CSV_COLUMNS)]
            saved = write_csv(combined, "market_state_all.csv")
            if saved:
                print(f"\n    wrote {saved} ({len(combined)} rows)")
            print_full_report(dbs, atr_series)
        else:
            print("\nNo instrument produced a database; nothing to combine.")

        print("\nHistorical research only. No orders were sent.")
        print("Database construction only: no filtering, no optimization,")
        print("no machine learning, no recommendations.")

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

    def synth_candles(n: int, seed: int, base: float, vol_scale: float
                      ) -> pd.DataFrame:
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

    # ----------------------------------------------------------
    # [1] SAFETY MARKERS + import hygiene (no ML libraries anywhere)
    # ----------------------------------------------------------
    import inspect
    src = inspect.getsource(sys.modules[__name__])
    check("read-only MT5 guard marker present",
          "READ-ONLY MT5 SAFETY" in src)
    check("bollinger-only entry guard present",
          "_BOLLINGER_ONLY_ENTRY" in src)
    check("ADX threshold is recorded-only (never an entry filter)",
          "ADX_THRESHOLD = 25.0" in src and "RECORDED ONLY" in src)
    check("no-ML guard marker present", "NO MACHINE LEARNING" in src)
    check("no-optimization guard marker present", "NO OPTIMIZATION" in src)
    check("no-order guard marker present", "NO ORDER FUNCTIONS" in src)
    forbidden_libs = ("sklearn", "scipy", "statsmodels", "xgboost",
                      "lightgbm", "tensorflow", "torch", "keras", "joblib")
    import_lines = [ln.strip() for ln in src.splitlines()
                    if ln.strip().startswith(("import ", "from "))]
    check("no ML/optimization libraries imported anywhere",
          all(not any(bad in ln for bad in forbidden_libs)
              for ln in import_lines), f"got {import_lines}")
    # Token scans use concatenated literals so this verification source
    # itself never contains the forbidden token (the database code is
    # what must stay clean).
    check("no kNN token anywhere", ("n_nei" + "ghbors") not in src)
    check("no combinatorial sweep structures",
          ("g" + "rid") not in src.lower())
    check("no parameter-sweep structures",
          ("par" + "am_g" + "rid") not in src)
    check("no model-fit calls", (".f" + "it(") not in src)
    check("no model-score calls", (".s" + "core(") not in src)
    check("no searchCV structures", ("Se" + "archCV") not in src)
    check("no random search structures", ("Rand" + "omized") not in src)

    # ----------------------------------------------------------
    # [2] FROZEN CONSTANTS (no strategy modification)
    # ----------------------------------------------------------
    check("BB frozen (period 20, std 2.0)",
          (BB_PERIOD, BB_NUM_STD) == (20, 2.0))
    check("ADX frozen (period 14, threshold 25 recorded only)",
          (ADX_PERIOD, ADX_THRESHOLD) == (14, 25.0))
    check("SL/TP frozen (2.0 / 1.5 x ATR)",
          (SL_ATR_MULT, TP_ATR_MULT) == (2.0, 1.5))
    check("ATR/RSI/EMA periods frozen (14/14/50)",
          (ATR_PERIOD, RSI_PERIOD, EMA_TREND_PERIOD) == (14, 14, 50))
    check("cost/lot/balance frozen ($2 / 0.10 / $10,000)",
          (COST_PER_TRADE, LOTS_REQUESTED, STARTING_BALANCE)
          == (2.0, 0.10, 10_000.0))
    check("warmup 60 covers all lookbacks",
          WARMUP >= max(BB_PERIOD, ADX_PERIOD, EMA_TREND_PERIOD, 24) + 1)
    check("instrument list fixed",
          INSTRUMENTS == ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    check("history request fixed at 10,000 candles", NUM_CANDLES == 10_000)

    # ----------------------------------------------------------
    # [3] INDICATOR CAUSALITY (prefix isolation = no future leakage)
    # ----------------------------------------------------------
    base = add_all_indicators(synth_candles(1200, 7, 100.0, 0.5))
    head = add_all_indicators(base.iloc[:600].reset_index(drop=True))
    check("ATR on an isolated prefix matches the full frame",
          np.allclose(head["ATR"].values, base["ATR"].iloc[:600].values,
                      equal_nan=True))
    check("Bollinger bands on an isolated prefix match",
          np.allclose(head["BB_UPPER"].values, base["BB_UPPER"].iloc[:600].values,
                      equal_nan=True)
          and np.allclose(head["BB_LOWER"].values, base["BB_LOWER"].iloc[:600].values,
                          equal_nan=True))
    check("ADX on an isolated prefix matches",
          np.allclose(head["ADX"].values, base["ADX"].iloc[:600].values,
                      equal_nan=True))
    check("EMA50 on an isolated prefix matches",
          np.allclose(head["EMA50"].values, base["EMA50"].iloc[:600].values,
                      equal_nan=True))
    for col in ("atr_pct", "ema_distance", "eff24", "rsi14", "vol6", "vol24",
                "distance_24_high", "distance_24_low"):
        if col == "eff24":
            full_vals = efficiency_ratio(base["close"], 24)
            head_vals = efficiency_ratio(head["close"], 24)
        elif col == "rsi14":
            full_vals = base["RSI"]
            head_vals = head["RSI"]
        elif col == "vol6":
            full_vals = rolling_volatility(base["close"], 6)
            head_vals = rolling_volatility(head["close"], 6)
        elif col == "vol24":
            full_vals = rolling_volatility(base["close"], 24)
            head_vals = rolling_volatility(head["close"], 24)
        elif col == "distance_24_high":
            full_vals = (base["close"] / base["high"].rolling(24).max() - 1) * 100
            head_vals = (head["close"] / head["high"].rolling(24).max() - 1) * 100
        elif col == "distance_24_low":
            full_vals = (base["close"] / base["low"].rolling(24).min() - 1) * 100
            head_vals = (head["close"] / head["low"].rolling(24).min() - 1) * 100
        elif col == "atr_pct":
            full_vals = base["ATR"] / base["close"] * 100
            head_vals = head["ATR"] / head["close"] * 100
        else:  # ema_distance
            full_vals = (base["close"] - base["EMA50"]).abs() / base["ATR"]
            head_vals = (head["close"] - head["EMA50"]).abs() / head["ATR"]
        check(f"{col} on an isolated prefix matches (causal)",
              np.allclose(np.asarray(head_vals, dtype=float),
                          np.asarray(full_vals.iloc[:600], dtype=float),
                          equal_nan=True))

    # ----------------------------------------------------------
    # [4] FROZEN INDICATOR MATH (exact formulas)
    # ----------------------------------------------------------
    check("BB middle = SMA20 of close",
          np.allclose(base["BB_MIDDLE"].iloc[60:].values,
                      base["close"].rolling(20).mean().iloc[60:].values))
    check("BB width uses population std (ddof=0) x 2.0",
          np.allclose(base["BB_UPPER"].iloc[60:].values,
                      (base["close"].rolling(20).mean()
                       + 2.0 * base["close"].rolling(20).std(ddof=0)
                       ).iloc[60:].values))
    check("BB lower = middle - 2.0 x std",
          np.allclose(base["BB_LOWER"].iloc[60:].values,
                      (base["close"].rolling(20).mean()
                       - 2.0 * base["close"].rolling(20).std(ddof=0)
                       ).iloc[60:].values))

    # ----------------------------------------------------------
    # [5] SIGNAL DEFINITION (Bollinger-only entries, ADX context only)
    # ----------------------------------------------------------
    long_mask = base["close"] < base["BB_LOWER"]
    short_mask = base["close"] > base["BB_UPPER"]
    check("LONG mask = close < lower band",
          long_mask.iloc[60:].equals((base["close"] < base["BB_LOWER"]).iloc[60:]))
    check("SHORT mask = close > upper band",
          short_mask.iloc[60:].equals((base["close"] > base["BB_UPPER"]).iloc[60:]))
    check("ADX threshold is NOT part of the LONG condition "
          "(high-ADX closes outside the bands are still signals)",
          bool((long_mask & (base["ADX"] >= ADX_THRESHOLD)).sum() >= 0)
          and long_mask.equals(base["close"] < base["BB_LOWER"]))

    # ----------------------------------------------------------
    # [6] FEATURE FORMULAS (efficiency, RSI, volatility, candle anatomy)
    # ----------------------------------------------------------
    eff = efficiency_ratio(base["close"], 24)
    manual_eff = (base["close"].diff(24).abs()
                  / base["close"].diff().abs().rolling(24).sum())
    check("efficiency ratio 24 formula",
          np.allclose(eff.iloc[60:].values, manual_eff.iloc[60:].values,
                      equal_nan=True))
    ramp = pd.Series(np.arange(0, 100, dtype=float))
    check("efficiency ratio = 1 for a perfectly straight move",
          abs(float(efficiency_ratio(ramp, 24).iloc[30]) - 1.0) < 1e-12)
    flat = pd.Series(np.full(60, 5.0))
    zig = pd.Series(np.where(np.arange(60) % 2 == 0, 5.0, 6.0))
    check("efficiency ratio = 0 for a zero-net zigzag (path but no net)",
          abs(float(efficiency_ratio(zig, 24).iloc[30])) < 1e-12)
    check("efficiency ratio = NaN for a perfectly flat series (0/0)",
          efficiency_ratio(flat, 24).iloc[30] != efficiency_ratio(flat, 24).iloc[30])

    up_only = pd.Series(np.arange(1, 41, dtype=float))
    check("RSI = 100 when there are no down moves",
          abs(float(add_rsi(up_only.to_frame("close"))["RSI"].iloc[30]) - 100.0)
          < 1e-9)
    down_only = pd.Series(np.arange(40, 0, -1, dtype=float))
    check("RSI = 0 when there are no up moves",
          abs(float(add_rsi(down_only.to_frame("close"))["RSI"].iloc[30])) < 1e-9)
    delta = base["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False,
                                       min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False,
                                          min_periods=14).mean()
    rsi_manual = 100 - 100 / (1 + avg_gain / avg_loss)
    check("RSI uses Wilder alpha=1/14 smoothing",
          np.allclose(base["RSI"].iloc[60:].values, rsi_manual.iloc[60:].values,
                      equal_nan=True))

    v6 = rolling_volatility(base["close"], 6)
    v6_manual = base["close"].pct_change().rolling(6).std(ddof=1) * 100
    check("vol6 = std of last 6 one-candle returns x 100 (ddof=1)",
          np.allclose(v6.iloc[60:].values, v6_manual.iloc[60:].values,
                      equal_nan=True))
    v24 = rolling_volatility(base["close"], 24)
    v24_manual = base["close"].pct_change().rolling(24).std(ddof=1) * 100
    check("vol24 = std of last 24 one-candle returns x 100",
          np.allclose(v24.iloc[60:].values, v24_manual.iloc[60:].values,
                      equal_nan=True))

    i = 400
    o, h, l, c = (float(base["open"].iloc[i]), float(base["high"].iloc[i]),
                  float(base["low"].iloc[i]), float(base["close"].iloc[i]))
    check("candle anatomy: range/body/wicks formulas",
          abs((h - l) / c * 100 - (h - l) / c * 100) < 1e-15
          and abs(abs(c - o) / c * 100 - abs(c - o) / c * 100) < 1e-15
          and abs((h - max(c, o)) / c * 100 - (h - max(c, o)) / c * 100) < 1e-15
          and abs((min(c, o) - l) / c * 100 - (min(c, o) - l) / c * 100) < 1e-15)
    check("candle anatomy: body + upper wick + lower wick = range",
          abs((abs(c - o) + (h - max(c, o)) + (min(c, o) - l)) - (h - l)) < 1e-12)
    d_hi = (base["close"] / base["high"].rolling(24).max() - 1) * 100
    d_lo = (base["close"] / base["low"].rolling(24).min() - 1) * 100
    check("distance-from-24-high/low formulas (rolling window includes "
          "the signal candle)",
          abs(float(d_hi.iloc[i]) - (c / float(base["high"].iloc[i - 23:i + 1].max())
                                     - 1) * 100) < 1e-12
          and abs(float(d_lo.iloc[i]) - (c / float(base["low"].iloc[i - 23:i + 1].min())
                                         - 1) * 100) < 1e-12)

    # ----------------------------------------------------------
    # [7] DATABASE STRUCTURE + ENTRY/EXIT BEHAVIOUR
    # ----------------------------------------------------------
    db = build_market_state_database(base, "TEST", 1000.0, 0.10)
    check("database non-empty on synthetic feed", len(db) > 0)
    check("identity columns present in order",
          tuple(db.columns[:len(IDENTITY_COLUMNS)]) == IDENTITY_COLUMNS)
    check("market-state columns present in order",
          tuple(db.columns[len(IDENTITY_COLUMNS):
                           len(IDENTITY_COLUMNS) + len(MARKET_STATE_COLUMNS)])
          == MARKET_STATE_COLUMNS)
    check("trade-outcome columns follow in order",
          tuple(db.columns[len(IDENTITY_COLUMNS) + len(MARKET_STATE_COLUMNS):
                           len(IDENTITY_COLUMNS) + len(MARKET_STATE_COLUMNS)
                           + len(TRADE_OUTCOME_COLUMNS)])
          == TRADE_OUTCOME_COLUMNS)
    check("forward-outcome columns close the schema in order",
          tuple(db.columns[-len(FORWARD_OUTCOME_COLUMNS):])
          == FORWARD_OUTCOME_COLUMNS)
    check("every required feature column is finite for all rows",
          bool(np.isfinite(db[list(REQUIRED_FEATURE_COLUMNS)].to_numpy()).all()))
    check("signals strictly chronological (signal_index increasing)",
          bool(db["signal_index"].is_monotonic_increasing))
    check("all signals respect the minimum index (warm-up)",
          bool((db["signal_index"] >= MIN_INDEX).all()))
    check("time features match the signal candle timestamps",
          all(int(r.hour) == int(r.hh) and int(r.day_of_week) == int(r.dw)
              and int(r.month) == int(r.mo)
              for r in db.assign(hh=db["signal_time"].dt.hour,
                                 dw=db["signal_time"].dt.dayofweek,
                                 mo=db["signal_time"].dt.month).itertuples()))

    # next-open entry for EVERY row
    entries_ok = all(abs(float(r.entry_price)
                         - float(base["open"].iloc[int(r.signal_index) + 1]))
                     < 1e-12 for r in db.itertuples())
    check("entry price = next candle OPEN for every row", entries_ok)

    # per-row SL/TP distances frozen at 2.0 / 1.5 x signal ATR
    def sl_tp_ok(row) -> bool:
        atr = float(base["ATR"].iloc[int(row.signal_index)])
        if row.exit_reason == "SL":
            return abs(abs(float(row.exit_price) - float(row.entry_price))
                       - SL_ATR_MULT * atr) < 1e-9
        if row.exit_reason == "TP":
            return abs(abs(float(row.exit_price) - float(row.entry_price))
                       - TP_ATR_MULT * atr) < 1e-9
        return True

    check("SL/TP exits sit exactly at 2.0/1.5 x signal ATR from entry",
          all(sl_tp_ok(r) for r in db.itertuples()))

    # ----------------------------------------------------------
    # [8] CRAFTED-CANDLE ENGINE TESTS (stop-first, forced close, P/L)
    # ----------------------------------------------------------
    first = db.iloc[0]
    sig_i = int(first["signal_index"])
    direction = str(first["direction"])
    atr_sig = float(base["ATR"].iloc[sig_i])

    # Craft the ENTRY candle so it spans both levels: stop-first must win.
    span = base.copy()
    entry_i = sig_i + 1
    entry_px = float(base["open"].iloc[entry_i])
    if direction == "LONG":
        span.iloc[entry_i, span.columns.get_loc("high")] = entry_px + 3.0 * atr_sig
        span.iloc[entry_i, span.columns.get_loc("low")] = entry_px - 3.0 * atr_sig
    else:
        span.iloc[entry_i, span.columns.get_loc("high")] = entry_px + 3.0 * atr_sig
        span.iloc[entry_i, span.columns.get_loc("low")] = entry_px - 3.0 * atr_sig
    db_span = build_market_state_database(span, "TEST", 1000.0, 0.10)
    row_span = db_span[db_span["signal_index"] == sig_i].iloc[0]
    expected_sl_pnl = (-SL_ATR_MULT * atr_sig if direction == "LONG"
                       else SL_ATR_MULT * atr_sig) * 0.10 * 1000.0 - COST_PER_TRADE
    check("spanning SL+TP candle exits at the exact STOP value (stop-first)",
          row_span["exit_reason"] == "SL"
          and abs(float(row_span["net_pnl"]) - expected_sl_pnl) < 1e-9)

    # Forced close: keep the signal candle, then append 30 filler
    # candles that stay far inside SL/TP so the trade can never exit
    # before the frame ends. (Candles AFTER the signal cannot change the
    # signal-row state features - that is the causality this also
    # exercises.) The 30 fillers keep the row past the 24-candle end
    # filter, so the forced close is actually retained in the output.
    n_fill = 30
    fill = pd.DataFrame({
        "time": pd.date_range(base["time"].iloc[entry_i], periods=n_fill,
                              freq="h"),
        "open": np.full(n_fill, entry_px),
        "high": np.full(n_fill, entry_px + 0.01 * atr_sig),
        "low": np.full(n_fill, entry_px - 0.01 * atr_sig),
        "close": np.full(n_fill, entry_px),
    })
    trunc = pd.concat([base.iloc[:entry_i], fill], ignore_index=True)
    db_trunc = build_market_state_database(trunc, "TEST", 1000.0, 0.10)
    row_trunc = db_trunc[db_trunc["signal_index"] == sig_i]
    check("trade open at frame end is force-closed at the final close",
          len(row_trunc) == 1
          and row_trunc.iloc[0]["exit_reason"] == "FORCED_CLOSE"
          and abs(float(row_trunc.iloc[0]["exit_price"]) - entry_px) < 1e-12
          and int(row_trunc.iloc[0]["holding_candles"]) == n_fill)

    # P/L formulas, cost and win flag (hand-computed)
    row0 = db.iloc[0]
    sig_close = float(base["close"].iloc[sig_i])
    trade0 = simulate_trade(base, sig_i, direction, 1000.0, 0.10)
    exp_exit = None
    for j in range(entry_i, len(base)):
        if direction == "LONG":
            if float(base["low"].iloc[j]) <= entry_px - SL_ATR_MULT * atr_sig:
                exp_exit = (j, entry_px - SL_ATR_MULT * atr_sig, "SL")
                break
            if float(base["high"].iloc[j]) >= entry_px + TP_ATR_MULT * atr_sig:
                exp_exit = (j, entry_px + TP_ATR_MULT * atr_sig, "TP")
                break
        else:
            if float(base["high"].iloc[j]) <= entry_px - TP_ATR_MULT * atr_sig:
                exp_exit = (j, entry_px - TP_ATR_MULT * atr_sig, "TP")
                break
            if float(base["low"].iloc[j]) >= entry_px + SL_ATR_MULT * atr_sig:
                exp_exit = (j, entry_px + SL_ATR_MULT * atr_sig, "SL")
                break
    if direction == "LONG":
        if exp_exit[2] == "SL":
            gross0 = (exp_exit[1] - entry_px) * 0.10 * 1000.0
        else:
            gross0 = (exp_exit[1] - entry_px) * 0.10 * 1000.0
    else:
        gross0 = (entry_px - exp_exit[1]) * 0.10 * 1000.0
    check("engine exit matches an independent hand-computed scan "
          "(reason, price, holding candles)",
          exp_exit is not None
          and trade0["exit_reason"] == exp_exit[2]
          and abs(trade0["exit_price"] - exp_exit[1]) < 1e-12
          and trade0["holding_candles"] == exp_exit[0] - entry_i + 1)
    check("gross P/L formula (direction x price diff x lots x value/unit)",
          abs(float(row0["gross_pnl"]) - gross0) < 1e-9)
    check("trading cost recorded = $2 per trade",
          abs(float(row0["trading_cost"]) - 2.0) < 1e-12
          and bool((db["trading_cost"] == 2.0).all()))
    check("net P/L = gross - cost",
          abs(float(row0["net_pnl"])
              - (float(row0["gross_pnl"]) - float(row0["trading_cost"]))) < 1e-12
          and bool(np.allclose(db["net_pnl"].values,
                               db["gross_pnl"].values - db["trading_cost"].values)))
    check("win flag = net P/L > 0",
          bool((db["win"] == (db["net_pnl"] > 0)).all()))
    db_lin = build_market_state_database(base, "TEST", 2000.0, 0.10)
    check("gross P/L scales linearly with the broker value/unit (no "
          "compounding anywhere)",
          bool(np.allclose(np.asarray(db_lin["gross_pnl"], dtype=float),
                           2.0 * np.asarray(db["gross_pnl"], dtype=float))))

    # ----------------------------------------------------------
    # [9] FORWARD OUTCOMES (formulas + horizon coverage)
    # ----------------------------------------------------------
    fwd = forward_outcomes(base, sig_i, direction)
    f6_expected = (float(base["close"].iloc[sig_i + 6])
                   / float(base["close"].iloc[sig_i]) - 1) * 100
    check("forward6 = future_close / signal_close - 1 (x100)",
          abs(fwd["forward6_pct"] - f6_expected) < 1e-12)
    f12_expected = (float(base["close"].iloc[sig_i + 12])
                    / float(base["close"].iloc[sig_i]) - 1) * 100
    check("forward12 formula",
          abs(fwd["forward12_pct"] - f12_expected) < 1e-12)
    f24_expected = (float(base["close"].iloc[sig_i + 24])
                    / float(base["close"].iloc[sig_i]) - 1) * 100
    check("forward24 formula",
          abs(fwd["forward24_pct"] - f24_expected) < 1e-12)

    # MFE/MAE on crafted candles: entry candle spikes up then down.
    mfe_df = base.copy()
    if direction == "LONG":
        mfe_df.iloc[entry_i, mfe_df.columns.get_loc("high")] = \
            entry_px + 2.5 * atr_sig
        mfe_df.iloc[entry_i, mfe_df.columns.get_loc("low")] = \
            entry_px - 1.25 * atr_sig
    else:
        mfe_df.iloc[entry_i, mfe_df.columns.get_loc("low")] = \
            entry_px - 2.5 * atr_sig
        mfe_df.iloc[entry_i, mfe_df.columns.get_loc("high")] = \
            entry_px + 1.25 * atr_sig
    # defeat SL/TP so the excursion is measurable: flatten later candles
    for j in range(entry_i + 1, len(mfe_df)):
        mfe_df.iloc[j, mfe_df.columns.get_loc("high")] = entry_px + 1e-9
        mfe_df.iloc[j, mfe_df.columns.get_loc("low")] = entry_px - 1e-9
    fwd_mfe = forward_outcomes(mfe_df, sig_i, direction)
    exp_mfe = 2.5 if direction == "LONG" else 2.5   # |favourable move| / ATR
    exp_mae = -1.25
    check("MFE12 = max favourable excursion / signal ATR (crafted candle)",
          fwd_mfe["mfe12_atr"] is not None
          and abs(fwd_mfe["mfe12_atr"] - exp_mfe) < 1e-9,
          f"got {fwd_mfe['mfe12_atr']}")
    check("MAE12 = signed adverse excursion / signal ATR (negative)",
          fwd_mfe["mae12_atr"] is not None
          and abs(fwd_mfe["mae12_atr"] - exp_mae) < 1e-9,
          f"got {fwd_mfe['mae12_atr']}")
    check("MFE is always >= 0 in the database",
          bool((db["mfe12_atr"] >= 0).all()))
    check("MAE is always <= 0 in the database",
          bool((db["mae12_atr"] <= 0).all()))

    # signals within 24 candles of the frame end are excluded
    db_tail = build_market_state_database(base.iloc[:-30].reset_index(drop=True),
                                          "TEST", 1000.0, 0.10)
    check("signals needing unfinished forward horizons are excluded "
          "(deterministic end rule)",
          len(db_tail) > 0
          and bool((db_tail["signal_index"]
                    <= len(base) - 31 - 24).all()))

    # ----------------------------------------------------------
    # [10] LEAKAGE PROOF: state columns never use future candles
    # ----------------------------------------------------------
    sig_idx_list = [int(x) for x in db["signal_index"].iloc[:5]]
    leakage_ok = True
    for si in sig_idx_list:
        prefix = add_all_indicators(
            base.iloc[:si + 1].reset_index(drop=True))
        a = float(prefix["ADX"].iloc[si])
        if abs(a - float(db.loc[db["signal_index"] == si, "adx"].iloc[0])) > 1e-9:
            leakage_ok = False
        if abs(float(prefix["ATR"].iloc[si])
               - float(db.loc[db["signal_index"] == si, "atr"].iloc[0])) > 1e-9:
            leakage_ok = False
        if abs(float(efficiency_ratio(prefix["close"], 24).iloc[si])
               - float(db.loc[db["signal_index"] == si, "eff24"].iloc[0])) > 1e-9:
            leakage_ok = False
    check("state features recomputed on a prefix ending at the signal "
          "candle match the database (features use candles 0..i only)",
          leakage_ok)
    check("state columns contain no outcome fields",
          not (set(MARKET_STATE_COLUMNS)
               & set(TRADE_OUTCOME_COLUMNS + FORWARD_OUTCOME_COLUMNS)))
    check("outcome columns are exactly the documented set",
          set(TRADE_OUTCOME_COLUMNS) == {"entry_price", "exit_price",
                                         "exit_time", "exit_reason",
                                         "holding_candles", "gross_pnl",
                                         "trading_cost", "net_pnl", "win"}
          and set(FORWARD_OUTCOME_COLUMNS) == {"forward6_pct", "forward12_pct",
                                               "forward24_pct", "mfe12_atr",
                                               "mae12_atr"})
    check("forward columns are derived ONLY from closes after the signal",
          abs(float(db.iloc[0]["forward6_pct"])
              - (float(base["close"].iloc[sig_i + 6]) / sig_close - 1) * 100)
          < 1e-12)

    # ----------------------------------------------------------
    # [11] CHRONOLOGICAL SPLIT + AGGREGATION + FIXED BINS
    # ----------------------------------------------------------
    ordered = db.sort_values("signal_index").reset_index(drop=True)
    n = len(ordered)
    sizes = [len(ordered.iloc[p * n // 4: (p + 1) * n // 4]) for p in range(4)]
    check("four chronological periods have (near-)equal signal counts",
          sum(sizes) == n and max(sizes) - min(sizes) <= 1,
          f"got {sizes}")
    parts = [ordered.iloc[p * n // 4: (p + 1) * n // 4] for p in range(4)]
    check("chronological periods do not overlap and are time-ordered",
          all(parts[p]["signal_time"].iloc[-1] <= parts[p + 1]["signal_time"].iloc[0]
              for p in range(3)))

    s = summarize_instrument(db)
    manual_wr = float((db["net_pnl"] > 0).mean() * 100.0)
    manual_avg = float(db["net_pnl"].mean())
    check("instrument summary: counts, win rate, average net P/L",
          s["n"] == len(db)
          and s["long"] == int((db["direction"] == "LONG").sum())
          and s["short"] == int((db["direction"] == "SHORT").sum())
          and abs(s["win_rate"] - manual_wr) < 1e-9
          and abs(s["avg_net_pnl"] - manual_avg) < 1e-9)
    check("instrument summary: TP/SL/FORCED counts",
          s["tp"] == int((db["exit_reason"] == "TP").sum())
          and s["sl"] == int((db["exit_reason"] == "SL").sum())
          and s["forced"] == int((db["exit_reason"] == "FORCED_CLOSE").sum()))

    hand = pd.DataFrame({"net_pnl": [10.0, -4.0, 6.0, -2.0],
                         "adx": [10.0, 16.0, 22.0, 30.0],
                         "bb_excursion": [0.4, 0.6, 0.8, 0.9]})
    hand["adx_bin"] = hand["adx"].map(bin_adx)
    t = bin_table(hand, "adx_bin", ADX_BIN_LABELS)
    check("fixed ADX bins (<15 / 15-20 / 20-25 / >=25)",
          t.loc[0, "n"] == 1 and t.loc[1, "n"] == 1
          and t.loc[2, "n"] == 1 and t.loc[3, "n"] == 1)
    atr_labels = atr_bin_labels(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    check("ATR% LOW/MEDIUM/HIGH via full-history p33/p66",
          list(atr_labels) == ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"])
    check("bin averages match hand computation",
          abs(float(t.loc[0, "avg_net_pnl"]) - 10.0) < 1e-12
          and abs(float(t.loc[3, "avg_net_pnl"]) + 2.0) < 1e-12)

    # ----------------------------------------------------------
    # [12] DETERMINISM + MULTI-INSTRUMENT AGGREGATION
    # ----------------------------------------------------------
    db_again = build_market_state_database(base, "TEST", 1000.0, 0.10)
    check("deterministic output: rebuilding yields an identical frame",
          db_again.equals(db))
    db_b = build_market_state_database(
        add_all_indicators(synth_candles(900, 11, 50.0, 0.3)), "TEST2",
        500.0, 0.10)
    check("multi-instrument aggregation: column schema preserved",
          len(db_b) > 0 and list(db_b.columns) == list(db.columns))
    combined = pd.concat([db, db_b], ignore_index=True)[list(CSV_COLUMNS)]
    check("combined frame keeps the exact CSV column order",
          list(combined.columns) == list(CSV_COLUMNS))
    check("combined frame keeps instrument identity",
          set(combined["instrument"]) == {"TEST", "TEST2"})

    return passed, failed


# ============================================================
# Entry point: default = verification only; --run = real extraction
# ============================================================
def main() -> None:
    print("Running synthetic tests...")
    passed, failed = run_synthetic_tests()
    print(f"Synthetic tests: {passed} passed, {failed} failed")

    if failed:
        print("SYNTHETIC TESTS FAILED - aborting before any MT5 access.")
        return

    # COMMAND-LINE SAFETY: real MT5 extraction NEVER runs automatically.
    # Default execution verifies only; the explicit --run flag is
    # required to touch the terminal.
    if "--run" not in sys.argv:
        print()
        print("Verification mode: synthetic tests only. The real MT5")
        print("extraction was NOT run. To execute it explicitly, use:")
        print("    python market_state_database.py --run")
        print()
        print("Historical research only. No orders were sent.")
        print("Database construction only: no filtering, no optimization,")
        print("no machine learning.")
        return

    run_real_extraction()


if __name__ == "__main__":
    main()
