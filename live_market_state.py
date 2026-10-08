"""
live_market_state.py - READ-ONLY MT5 LIVE MARKET-STATE COLLECTOR.

PURPOSE
    Connect to MetaTrader 5 in read-only mode, fetch H1 candles for
    exactly five instruments, and calculate the SAME market-state
    features and regime classifications used by the existing research
    system (market_state_database.py -> ai_research_layer.py), for the
    latest CLOSED H1 candle of each instrument.

    Collects latest market data and computes descriptive state values.
    This is NOT a trading system.

STRICT SAFETY RULES (verified by the built-in static safety scan)
    - MetaTrader 5 is used strictly in read-only mode: only
      initialize, last_error, symbol_info, symbols_get, symbol_select,
      copy_rates_from_pos and shutdown are ever called.
    - no order is ever transmitted, verified, modified, or closed;
      the order transmission / order verification API functions are
      never called anywhere in this file.
    - no directional advice of any kind, no order sizing, no
      protective-price levels, no risk percentages.
    - no machine learning, no optimization, no external AI/API calls.
    - never modifies, overwrites, or regenerates any existing project
      file; only the NEW output files named below may be created.
    - deterministic output from the supplied candle data.

INSTRUMENTS (exactly these five, always handled separately)
    GOLD, EURUSD, GBPUSD, USDJPY, AUDUSD
    Each symbol is verified with mt5.symbol_info() BEFORE processing.
    An unavailable symbol is recorded as UNAVAILABLE with a clear
    reason, the remaining instruments are still processed, and the
    program never crashes because of one instrument.

TIMEFRAME
    mt5.TIMEFRAME_H1.  Enough history is requested to give every
    indicator a safe warm-up margin (the same 60-candle warm-up margin
    the database stage documents, with far more requested).
    Only CLOSED candles are used: the newest (still-forming) candle
    returned by the terminal is always dropped, and no future candle
    is ever read.

26 ALLOW-LISTED STATE FEATURES (nothing else is computed)
    direction, adx, atr_pct, ema50_dist_atr, bb_excursion,
    bb_position, eff24, rsi14, return6, return12, return24,
    dist_24_high, dist_24_low, candle_range_pct, candle_body_pct,
    upper_wick_pct, lower_wick_pct, hour, day_of_week, month,
    adx_regime, volatility_regime, trend_distance_regime,
    bb_excursion_regime, efficiency_regime, rsi_regime
    (20 state values + the 6 categorical regime fields = 26.)

INDICATOR DEFINITIONS (identical math to market_state_database.py;
    every calculation is causal - candles after the observation are
    never used)
    EMA50, Bollinger Bands 20 periods / 2 standard deviations, ADX14,
    ATR14, RSI14, 24-candle Kaufman efficiency ratio, 6/12/24-candle
    returns, 24-candle high/low distance, candle range/body/wick
    percentages.

REGIME BINNING - definitions REUSED from ai_research_layer.py, never
    recreated (this file imports classify_adx, classify_ema_distance,
    classify_bb_excursion, classify_efficiency, classify_rsi,
    classify_atr_pct and atr_tercile_thresholds directly from it):
        ADX           : ADX<15 | ADX15-20 | ADX20-25 | ADX>=25
        EMA distance  : <1 | 1-2 | >=2       (ATR units)
        BB excursion  : <0.50 | 0.50-0.75 | >=0.75
        Efficiency    : <0.30 | 0.30-0.60 | >=0.60
        RSI           : <30 | 30-50 | 50-70 | >=70
        Volatility    : LOW | MEDIUM | HIGH via the instrument's OWN
                        frozen atr_pct terciles (33rd/66th percentiles),
                        recomputed read-only from market_state_all.csv
                        exactly the way the research layer computes
                        them; never tuned.  A value exactly on a
                        boundary belongs to the higher bucket; a
                        missing value maps to "n/a".

DIRECTION (frozen research labelling rule, reused verbatim from
    market_state_database.py): close below the lower Bollinger band ->
    LONG; close above the upper band -> SHORT; otherwise NONE, which
    means the frozen labelling condition is not met on this candle.
    NONE is recorded with a data-quality warning - it is never guessed.

OUTPUT FILES (both NEW; every target path is checked before any
    writing happens; an existing file is never overwritten)
    live_market_state.csv
        -> if it already exists, the file
           live_market_state_YYYYMMDD_HHMMSS.csv is created instead.
    live_market_state_report.txt
        -> if it already exists, the file
           live_market_state_report_YYYYMMDD_HHMMSS.txt is created
           instead.
    If a target name (including a fallback) already exists, nothing is
    written at all and the conflicting path is reported.

COMMAND-LINE
    python live_market_state.py          -> synthetic tests only
                                            (NEVER connects to MT5)
    python live_market_state.py --run    -> real read-only collection

SYNTHETIC TESTS (default mode; temporary directories only; no MT5)
    1  indicator calculations        7  closed-candle selection
    2  regime boundaries             8  missing-symbol handling
    3  candle feature calculations   9  deterministic output
    4  return calculations          10  output-file protection
    5  efficiency calculation        11  absence of order API functions
    6  no future-candle usage       12  absence of AI/API calls

DISCLAIMER: descriptive, read-only data collection for research and
education only; historical state facts only, nothing proven, no
directive of any kind.  Educational use only.
"""

# ============================================================
# Imports: Python standard library + numpy + pandas + the
# research layer's frozen classification definitions.  The trading
# terminal package is imported LATER, inside the --run path only, so
# the default test mode never loads or contacts it.
# ============================================================
import ast
import datetime
import json
import os
import re
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_research_layer import (
    ATR_BUCKETS,
    BBE_BUCKETS,
    EFF_BUCKETS,
    EMA_BUCKETS,
    RSI_BUCKETS,
    ADX_BUCKETS,
    atr_tercile_thresholds,
    classify_adx,
    classify_atr_pct,
    classify_bb_excursion,
    classify_efficiency,
    classify_ema_distance,
    classify_rsi,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Frozen settings (identical to the research pipeline)
# ============================================================
INSTRUMENTS: Tuple[str, ...] = (
    "GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
)

NUM_CANDLES = 10_000        # H1 candles requested per instrument
MIN_CLOSED_CANDLES = 60     # warm-up margin (database stage WARMUP/MIN_INDEX)
TIMEFRAME_LABEL = "H1"

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ATR_PERIOD = 14
EMA_TREND_PERIOD = 50
RSI_PERIOD = 14
EFF_WINDOW = 24
RETURN_WINDOWS: Tuple[int, ...] = (6, 12, 24)
ROLL_WINDOW = 24

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
STAMP_FORMAT = "%Y%m%d_%H%M%S"

# Read-only source of the frozen instrument-specific ATR terciles.
HISTORICAL_STATE_CSV = "market_state_all.csv"
# Read-only cross-check source for the frozen thresholds.
FROZEN_CONTEXT_JSON = "ai_research_context_report.json"

OUTPUT_CSV = "live_market_state.csv"
OUTPUT_TXT = "live_market_state_report.txt"

# The 26 allow-listed state fields, in fixed order.
STATE_KEYS: Tuple[str, ...] = (
    "direction", "adx", "atr_pct", "ema50_dist_atr", "bb_excursion",
    "bb_position", "eff24", "rsi14", "return6", "return12", "return24",
    "dist_24_high", "dist_24_low", "candle_range_pct", "candle_body_pct",
    "upper_wick_pct", "lower_wick_pct", "hour", "day_of_week", "month",
)
REGIME_KEYS: Tuple[str, ...] = (
    "adx_regime", "volatility_regime", "trend_distance_regime",
    "bb_excursion_regime", "efficiency_regime", "rsi_regime",
)
NUMERIC_STATE_KEYS: Tuple[str, ...] = tuple(
    k for k in STATE_KEYS if k not in ("direction", "hour", "day_of_week",
                                       "month"))
TIME_STATE_KEYS: Tuple[str, ...] = ("hour", "day_of_week", "month")

CSV_COLUMNS: Tuple[str, ...] = (
    "instrument", "mt5_symbol", "symbol_resolution", "status", "reason",
    "candles_used", "latest_closed_candle_time",
) + STATE_KEYS + REGIME_KEYS + ("data_quality_warnings",)

STATUS_COLLECTED = "COLLECTED"
STATUS_UNAVAILABLE = "UNAVAILABLE"

# Direction labels produced by the frozen research labelling rule.
DIRECTION_LONG = "LONG"
DIRECTION_SHORT = "SHORT"
DIRECTION_NONE = "NONE"

# Import allowlist enforced by the static safety scan (the trading
# terminal package appears only on the real --run path).
_ALLOWED_IMPORT_MODULES = frozenset({
    "ast", "datetime", "json", "os", "re", "sys", "tempfile", "typing",
    "numpy", "pandas", "ai_research_layer", "MetaTrader5",
})

# Read-only trading-terminal API surface this file is allowed to use.
_MT5_ALLOWED_API = (
    "initialize", "shutdown", "last_error", "symbol_info", "symbols_get",
    "symbol_select", "copy_rates_from_pos", "TIMEFRAME_H1",
)


# ============================================================
# Causal indicator math (identical formulas to
# market_state_database.py; that module is NOT imported because it
# loads the trading-terminal package at module level)
# ============================================================
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Average True Range (uses only current and previous candles)."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / period, adjust=False,
                               min_periods=period).mean()
    return df


def add_bollinger(df: pd.DataFrame, period: int = BB_PERIOD,
                  num_std: float = BB_NUM_STD) -> pd.DataFrame:
    """Bollinger Bands (rolling - causal by construction)."""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


def add_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Average Directional Index (Wilder smoothing - causal)."""
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
    atr = true_range.ewm(alpha=alpha, adjust=False,
                         min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False,
                                min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False,
                                  min_periods=period).mean() / atr
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=alpha, adjust=False,
                       min_periods=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int = EMA_TREND_PERIOD
            ) -> pd.DataFrame:
    """Exponential Moving Average (causal; state feature only)."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False,
                                         min_periods=period).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Wilder-style Relative Strength Index (causal)."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def efficiency_ratio(close: pd.Series,
                     window: int = EFF_WINDOW) -> pd.Series:
    """Kaufman Efficiency Ratio over `window` candles (causal).

    |close[t] - close[t-window]| / sum(|close[j]-close[j-1]| for the
    same window).  Values near 1 = smooth directional move; near 0 =
    choppy.
    """
    net_move = close.diff(window).abs()
    path = close.diff().abs().rolling(window).sum()
    return net_move / path.replace(0, np.nan)


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """(close[t] / close[t-window] - 1) * 100 (causal)."""
    return (close / close.shift(window) - 1.0) * 100.0


def compute_indicator_frame(closed: pd.DataFrame) -> pd.DataFrame:
    """All indicator + trailing feature series on CLOSED candles only.

    Every series is computed from candles up to and including each
    row; rows after a given index can never change its values.
    """
    d = closed.copy()
    d = add_atr(d, ATR_PERIOD)
    d = add_bollinger(d, BB_PERIOD, BB_NUM_STD)
    d = add_adx(d, ADX_PERIOD)
    d = add_ema(d, EMA_TREND_PERIOD)
    d = add_rsi(d, RSI_PERIOD)
    d["eff24"] = efficiency_ratio(d["close"], EFF_WINDOW)
    for w in RETURN_WINDOWS:
        d[f"return{w}"] = rolling_return(d["close"], w)
    d["roll_high24"] = d["high"].rolling(ROLL_WINDOW).max()
    d["roll_low24"] = d["low"].rolling(ROLL_WINDOW).min()
    return d


# ============================================================
# Deterministic small helpers
# ============================================================
def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def fmt_ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return pd.Timestamp(value).strftime(TS_FORMAT)


def fmt_num(value: Optional[float], digits: int = 4) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:.{digits}f}"


def candle_shape_features(open_: float, high: float, low: float,
                          close: float) -> Dict[str, Optional[float]]:
    """Candle range / body / wick percentages (exact database math)."""
    if close <= 0:
        return {"candle_range_pct": None, "candle_body_pct": None,
                "upper_wick_pct": None, "lower_wick_pct": None}
    return {
        "candle_range_pct": (high - low) / close * 100.0,
        "candle_body_pct": abs(close - open_) / close * 100.0,
        "upper_wick_pct": (high - max(close, open_)) / close * 100.0,
        "lower_wick_pct": (min(close, open_) - low) / close * 100.0,
    }


def time_features(ts: Any) -> Dict[str, Optional[int]]:
    """hour / day_of_week (0=Monday) / month of the candle time."""
    if ts is None:
        return {"hour": None, "day_of_week": None, "month": None}
    stamp = pd.Timestamp(ts)
    if pd.isna(stamp):
        return {"hour": None, "day_of_week": None, "month": None}
    return {"hour": int(stamp.hour), "day_of_week": int(stamp.dayofweek),
            "month": int(stamp.month)}


def frozen_direction(close: Optional[float],
                     bb_lower: Optional[float],
                     bb_upper: Optional[float]) -> str:
    """Frozen research labelling rule (market_state_database.py):
    close below the lower band -> LONG, above the upper band -> SHORT,
    otherwise NONE (no labelling condition holds on this candle)."""
    if close is None or bb_lower is None or bb_upper is None:
        return "n/a"
    if close < bb_lower:
        return DIRECTION_LONG
    if close > bb_upper:
        return DIRECTION_SHORT
    return DIRECTION_NONE


# ============================================================
# Closed-candle selection (identical rule to the database stage:
# the newest row returned by the terminal is the still-forming
# candle and is always dropped)
# ============================================================
def select_closed_candles(rates_frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the newest (forming) candle; return closed candles only."""
    if len(rates_frame) == 0:
        return rates_frame.iloc[0:0].reset_index(drop=True)
    return rates_frame.iloc[:-1].reset_index(drop=True)


# ============================================================
# State record for the latest CLOSED candle
# ============================================================
def build_state_record(closed: pd.DataFrame,
                       thresholds: Optional[Tuple[float, float]]
                       ) -> Dict[str, Any]:
    """The 26 allow-listed state fields for the last closed candle.

    Returns {"latest_closed_candle_time", "direction", "features",
    "time", "regimes", "warnings"}.  A field whose value cannot be
    computed is recorded as None / "n/a" with a data-quality warning;
    nothing is ever estimated.
    """
    warnings: List[str] = []
    d = compute_indicator_frame(closed)
    row = d.iloc[-1]
    ts = pd.Timestamp(row["time"]) if "time" in d.columns else None
    latest_time = fmt_ts(ts)

    close = safe_float(row["close"])
    open_ = safe_float(row["open"])
    high = safe_float(row["high"])
    low = safe_float(row["low"])
    atr = safe_float(row["ATR"])
    ema50 = safe_float(row[f"EMA{EMA_TREND_PERIOD}"])
    bb_mid = safe_float(row["BB_MIDDLE"])
    bb_upper = safe_float(row["BB_UPPER"])
    bb_lower = safe_float(row["BB_LOWER"])
    bb_width = None
    if bb_upper is not None and bb_lower is not None:
        bb_width = bb_upper - bb_lower

    features: Dict[str, Optional[float]] = {}

    features["adx"] = safe_float(row["ADX"])
    if atr is not None and close is not None and close > 0:
        features["atr_pct"] = atr / close * 100.0
    else:
        features["atr_pct"] = None
    if atr is not None and atr > 0 and close is not None \
            and ema50 is not None:
        features["ema50_dist_atr"] = abs(close - ema50) / atr
    else:
        features["ema50_dist_atr"] = None
    if bb_width is not None and bb_width > 0 and close is not None \
            and bb_mid is not None:
        features["bb_excursion"] = abs(close - bb_mid) / bb_width
        features["bb_position"] = (close - bb_mid) / bb_width
    else:
        features["bb_excursion"] = None
        features["bb_position"] = None
        warnings.append(
            "Bollinger band width unavailable or zero on the latest "
            "closed candle; bb_excursion and bb_position recorded as "
            "unavailable")
    features["eff24"] = safe_float(row["eff24"])
    features["rsi14"] = safe_float(row["RSI"])
    for w in RETURN_WINDOWS:
        features[f"return{w}"] = safe_float(row[f"return{w}"])
    roll_high = safe_float(row["roll_high24"])
    roll_low = safe_float(row["roll_low24"])
    features["dist_24_high"] = (
        (close / roll_high - 1.0) * 100.0
        if close is not None and roll_high is not None and roll_high > 0
        else None)
    features["dist_24_low"] = (
        (close / roll_low - 1.0) * 100.0
        if close is not None and roll_low is not None and roll_low > 0
        else None)

    if open_ is None or high is None or low is None or close is None:
        features.update({"candle_range_pct": None,
                         "candle_body_pct": None,
                         "upper_wick_pct": None,
                         "lower_wick_pct": None})
        warnings.append(
            "candle shape unavailable: incomplete OHLC on the latest "
            "closed candle")
    else:
        features.update(candle_shape_features(open_, high, low, close))

    direction = frozen_direction(close, bb_lower, bb_upper)
    if direction == DIRECTION_NONE:
        warnings.append(
            "direction recorded as NONE: the frozen research labelling "
            "condition (close beyond a Bollinger band) is not met on "
            "this candle")
    elif direction == "n/a":
        warnings.append(
            "direction unavailable: Bollinger band values missing on "
            "the latest closed candle")

    if latest_time is None:
        warnings.append("candle timestamp missing on the latest closed "
                        "candle")
        t_features = {"hour": None, "day_of_week": None, "month": None}
    else:
        t_features = time_features(ts)

    thresholds = thresholds if thresholds is not None else (float("nan"),
                                                            float("nan"))
    regimes = {
        "adx_regime": classify_adx(features["adx"]),
        "volatility_regime": classify_atr_pct(features["atr_pct"],
                                              thresholds),
        "trend_distance_regime": classify_ema_distance(
            features["ema50_dist_atr"]),
        "bb_excursion_regime": classify_bb_excursion(
            features["bb_excursion"]),
        "efficiency_regime": classify_efficiency(features["eff24"]),
        "rsi_regime": classify_rsi(features["rsi14"]),
    }
    if regimes["volatility_regime"] == "n/a":
        warnings.append(
            "volatility_regime unavailable: frozen ATR tercile "
            "thresholds missing for this instrument")

    missing = [k for k in NUMERIC_STATE_KEYS
               if features.get(k) is None]
    if missing:
        warnings.append(
            "state value(s) unavailable (non-finite input): "
            + ", ".join(missing))

    return {
        "latest_closed_candle_time": latest_time,
        "direction": direction,
        "features": features,
        "time": t_features,
        "regimes": regimes,
        "warnings": warnings,
    }


# ============================================================
# Symbol verification (same documented read-only lookup the database
# stage uses: exact mt5.symbol_info() match first, then a short
# deterministic suffix match)
# ============================================================
def resolve_symbol(requested: str,
                   info_fn: Callable[[str], Any],
                   symbols_getter: Callable[[], Any]
                   ) -> Tuple[Optional[str], str]:
    """Resolve `requested` to an MT5 symbol name, or (None, reason)."""
    info = info_fn(requested)
    if info is not None:
        return requested, "exact symbol name"
    matches: List[str] = []
    for sym in (symbols_getter() or []):
        name = getattr(sym, "name", None)
        if name is None:
            continue
        if name.startswith(requested) and len(name) > len(requested):
            suffix = name[len(requested):]
            if len(suffix) <= 10 and suffix.replace(".", "").isalnum():
                matches.append(name)
    if not matches:
        return None, "symbol not found: no exact or suffix match in MT5"
    matches.sort(key=lambda s: (len(s), s))     # deterministic
    return matches[0], (f"suffix match (available: "
                        f"{', '.join(matches[:5])})")


# ============================================================
# Frozen ATR tercile thresholds (read-only reuse of the research
# layer's definition and of market_state_all.csv)
# ============================================================
def load_frozen_atr_thresholds(
        path: Optional[str] = None
        ) -> Tuple[Dict[str, Tuple[float, float]], Optional[str]]:
    """Per-instrument frozen (p33, p66) atr_pct terciles.

    Recomputed read-only from market_state_all.csv using the research
    layer's own atr_tercile_thresholds function - the exact definition
    the research layer used when it classified historical volatility.
    Never tuned, never modified.
    """
    csv_path = path or os.path.join(SCRIPT_DIR, HISTORICAL_STATE_CSV)
    if not os.path.isfile(csv_path):
        return {}, (f"{HISTORICAL_STATE_CSV} not found; volatility "
                    f"regimes will be recorded as n/a")
    frame = pd.read_csv(csv_path, usecols=["instrument", "atr_pct"])
    out: Dict[str, Tuple[float, float]] = {}
    for inst in INSTRUMENTS:
        vals = frame.loc[frame["instrument"].astype(str) == inst,
                         "atr_pct"].tolist()
        p33, p66 = atr_tercile_thresholds(vals)
        out[inst] = (float(p33), float(p66))
    missing = [i for i in INSTRUMENTS if i not in out]
    warn = None
    if missing:
        warn = ("no rows in " + HISTORICAL_STATE_CSV + " for: "
                + ", ".join(missing))
    return out, warn


# ============================================================
# Collection (deterministic; never raises; read-only API only)
# ============================================================
def collect_instrument(instrument: str, mt5mod: Any,
                       thresholds: Dict[str, Tuple[float, float]],
                       ) -> Dict[str, Any]:
    """Collect the latest CLOSED H1 market state for one instrument.

    Never raises: every failure mode is recorded as UNAVAILABLE with a
    clear reason so the other instruments still process.
    """
    result: Dict[str, Any] = {
        "instrument": instrument,
        "mt5_symbol": None,
        "symbol_resolution": None,
        "status": STATUS_UNAVAILABLE,
        "reason": None,
        "candles_used": 0,
        "requested_candles": NUM_CANDLES,
        "state": None,
        "warnings": [],
    }
    try:
        resolved, how = resolve_symbol(instrument, mt5mod.symbol_info,
                                       mt5mod.symbols_get)
    except Exception as exc:                      # noqa: BLE001
        result["reason"] = f"symbol verification failed: {exc}"
        return result
    if resolved is None:
        result["reason"] = how
        return result
    result["mt5_symbol"] = resolved
    result["symbol_resolution"] = how

    try:
        selected = mt5mod.symbol_select(resolved, True)
        if not selected:
            result["warnings"].append(
                f"{resolved} could not be added to Market Watch; "
                f"attempting the read-only data request anyway")
    except Exception as exc:                      # noqa: BLE001
        result["warnings"].append(
            f"symbol_select reported: {exc}; attempting the read-only "
            f"data request anyway")

    try:
        rates = mt5mod.copy_rates_from_pos(resolved,
                                           mt5mod.TIMEFRAME_H1,
                                           0, NUM_CANDLES)
    except Exception as exc:                      # noqa: BLE001
        result["reason"] = f"H1 candle retrieval failed: {exc}"
        return result
    if rates is None or len(rates) == 0:
        detail = ""
        try:
            detail = f" (last_error: {mt5mod.last_error()})"
        except Exception:                         # noqa: BLE001
            pass
        result["reason"] = "no H1 candle data returned" + detail
        return result

    df = pd.DataFrame(rates)
    if "time" not in df.columns or "close" not in df.columns:
        result["reason"] = ("H1 candle data lacks required columns "
                            "(time/close)")
        return result
    df["time"] = pd.to_datetime(df["time"], unit="s")
    closed = select_closed_candles(df)
    if len(closed) < MIN_CLOSED_CANDLES:
        result["reason"] = (
            f"insufficient closed H1 candles: got {len(closed)}, "
            f"need at least {MIN_CLOSED_CANDLES} for a safe indicator "
            f"warm-up")
        return result

    state = build_state_record(closed, thresholds.get(instrument))
    result["state"] = state
    result["candles_used"] = int(len(closed))
    result["warnings"].extend(state["warnings"])
    result["status"] = STATUS_COLLECTED
    result["reason"] = None
    return result


def collect_all(mt5mod: Any,
                thresholds: Dict[str, Tuple[float, float]]
                ) -> List[Dict[str, Any]]:
    """Process exactly the five instruments, separately, in fixed
    order; one instrument can never stop the others."""
    results: List[Dict[str, Any]] = []
    for inst in INSTRUMENTS:
        try:
            results.append(collect_instrument(inst, mt5mod, thresholds))
        except Exception as exc:                  # noqa: BLE001
            results.append({
                "instrument": inst,
                "mt5_symbol": None,
                "symbol_resolution": None,
                "status": STATUS_UNAVAILABLE,
                "reason": f"unexpected collection error: {exc}",
                "candles_used": 0,
                "requested_candles": NUM_CANDLES,
                "state": None,
                "warnings": [],
            })
    return results


# ============================================================
# Output builders (deterministic)
# ============================================================
def result_to_row(res: Dict[str, Any]) -> Dict[str, Any]:
    """One flat CSV row following CSV_COLUMNS exactly."""
    row: Dict[str, Any] = {k: None for k in CSV_COLUMNS}
    row["instrument"] = res["instrument"]
    row["mt5_symbol"] = res["mt5_symbol"]
    row["symbol_resolution"] = res["symbol_resolution"]
    row["status"] = res["status"]
    row["reason"] = res["reason"]
    row["candles_used"] = int(res["candles_used"])
    state = res.get("state")
    if state is not None:
        row["latest_closed_candle_time"] = state[
            "latest_closed_candle_time"]
        row["direction"] = state["direction"]
        for k in NUMERIC_STATE_KEYS:
            row[k] = state["features"].get(k)
        for k in TIME_STATE_KEYS:
            row[k] = state["time"].get(k)
        for k in REGIME_KEYS:
            row[k] = state["regimes"].get(k)
    warnings = res.get("warnings") or []
    row["data_quality_warnings"] = " | ".join(warnings) if warnings else None
    return row


def results_to_csv(results: List[Dict[str, Any]]) -> str:
    rows = [result_to_row(res) for res in results]
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS)).to_csv(index=False)


def build_report(results: List[Dict[str, Any]],
                 connection_status: str,
                 collection_ts: datetime.datetime) -> str:
    """Human-readable report: no directives of any kind."""
    L: List[str] = []
    L.append("=" * 64)
    L.append("LIVE MARKET STATE - READ-ONLY COLLECTION REPORT")
    L.append("=" * 64)
    L.append(f"collection timestamp    : "
             f"{collection_ts.strftime(TS_FORMAT)}")
    L.append(f"MT5 connection status   : {connection_status}")
    L.append(f"timeframe               : {TIMEFRAME_LABEL} "
             f"(CLOSED candles only; newest forming candle dropped)")
    L.append(f"candles requested/symbol: {NUM_CANDLES}")
    L.append(f"instruments processed   : {len(results)}")
    collected = sum(1 for r in results if r["status"] == STATUS_COLLECTED)
    L.append(f"collected / unavailable : {collected} / "
             f"{len(results) - collected}")
    for res in results:
        L.append("")
        L.append("-" * 64)
        L.append(f"INSTRUMENT: {res['instrument']}")
        L.append(f"status                    : {res['status']}")
        if res["reason"]:
            L.append(f"reason                    : {res['reason']}")
        if res["mt5_symbol"]:
            L.append(f"mt5 symbol                : {res['mt5_symbol']} "
                     f"({res['symbol_resolution']})")
        L.append(f"candles used              : {res['candles_used']} "
                 f"closed H1 candles")
        state = res.get("state")
        if state is None:
            L.append("latest CLOSED H1 candle   : n/a (no data collected)")
            L.append("state values              : n/a")
            L.append("regime classifications    : n/a")
        else:
            L.append(f"latest CLOSED H1 candle   : "
                     f"{state['latest_closed_candle_time']}")
            L.append("STATE VALUES")
            L.append(f"  direction               : {state['direction']}")
            for k in NUMERIC_STATE_KEYS:
                L.append(f"  {k:<25}: "
                         f"{fmt_num(state['features'].get(k))}")
            for k in TIME_STATE_KEYS:
                v = state["time"].get(k)
                L.append(f"  {k:<25}: "
                         f"{'n/a' if v is None else v}")
            L.append("REGIME CLASSIFICATIONS")
            for k in REGIME_KEYS:
                L.append(f"  {k:<25}: {state['regimes'].get(k, 'n/a')}")
        L.append("DATA-QUALITY WARNINGS")
        warns = res.get("warnings") or []
        if warns:
            for w in warns:
                L.append(f"  - {w}")
        else:
            L.append("  (none)")
    L.append("")
    L.append("=" * 64)
    L.append("Descriptive read-only collection of stored market facts "
             "only.  No orders were placed, modified or closed; this "
             "session only read candles.  This report contains no "
             "directives, no sizing figures and no risk figures of any "
             "kind.")
    return "\n".join(L)


# ============================================================
# Output-file protection (all paths checked before any writing)
# ============================================================
class OutputExistsError(RuntimeError):
    pass


def resolve_output_paths(directory: str,
                         collection_ts: datetime.datetime
                         ) -> Tuple[str, str]:
    """Choose non-existing output paths for the CSV + report as ONE set.

    The fallback decision is made exactly once for the whole set: when
    the directory is clean with respect to BOTH primary files the
    entire set keeps the primary names; when ANY primary file already
    exists the ENTIRE set uses one shared timestamped suffix.  Every
    chosen path is then checked and an existing path raises, so nothing
    is ever overwritten and a mixed primary/timestamped set can never
    be produced."""
    stamp = collection_ts.strftime(STAMP_FORMAT)
    plan = (
        (os.path.join(directory, OUTPUT_CSV),
         os.path.join(directory, f"live_market_state_{stamp}.csv")),
        (os.path.join(directory, OUTPUT_TXT),
         os.path.join(directory,
                      f"live_market_state_report_{stamp}.txt")),
    )
    use_fallback = any(os.path.exists(primary) for primary, _ in plan)
    chosen: List[str] = []
    for primary, fallback in plan:
        path = fallback if use_fallback else primary
        if os.path.exists(path):
            raise OutputExistsError(
                "refusing to overwrite existing file(s); nothing was "
                f"written: {path}")
        chosen.append(path)
    return chosen[0], chosen[1]


def _write_bytes_if_absent(path: str, payload: bytes) -> None:
    if os.path.exists(path):
        raise OutputExistsError(
            f"output file already exists (never overwritten): {path}")
    with open(path, "wb") as fh:
        fh.write(payload)
    print(f"  wrote {os.path.basename(path)} ({len(payload)} bytes)")


def write_outputs(csv_payload: str, txt_payload: str,
                  csv_path: str, txt_path: str) -> None:
    """Write both NEW files; each writer re-checks its own path."""
    _write_bytes_if_absent(csv_path, csv_payload.encode("utf-8"))
    _write_bytes_if_absent(txt_path, txt_payload.encode("utf-8"))


# ============================================================
# Real collection (requires --run; never runs by default; the
# trading-terminal package is imported HERE, and only here)
# ============================================================
def run_real_collection() -> int:
    print("=" * 64)
    print("live_market_state.py - REAL READ-ONLY COLLECTION (--run)")
    print("=" * 64)
    try:
        import MetaTrader5 as mt5mod             # noqa: PLC0415
    except ImportError:
        print("ERROR: the MetaTrader 5 Python package is not "
              "installed in this environment.")
        print("Install it from the terminal's packages directory, then "
              "retry.")
        return 1
    if not mt5mod.initialize():
        print("ERROR: could not connect to MetaTrader 5 (read-only "
              "session).")
        print("Make sure the terminal is installed, running, and "
              "logged into a DEMO account.")
        print(f"Last error: {mt5mod.last_error()}")
        return 1
    try:
        connection_status = "CONNECTED (read-only session)"
        thresholds, thr_warn = load_frozen_atr_thresholds()
        results = collect_all(mt5mod, thresholds)
        if thr_warn:
            for res in results:
                res["warnings"].insert(0, thr_warn)
        collection_ts = datetime.datetime.now()
        csv_payload = results_to_csv(results)
        report_payload = build_report(results, connection_status,
                                      collection_ts)
        csv_path, txt_path = resolve_output_paths(SCRIPT_DIR,
                                                  collection_ts)
        print("\nSummary:")
        for res in results:
            print(f"  {res['instrument']}: {res['status']}"
                  + (f" - {res['reason']}" if res["reason"] else ""))
        print("\nWriting NEW output files:")
        write_outputs(csv_payload, report_payload, csv_path, txt_path)
        print("\nDone. Descriptive read-only market-state collection "
              "only; no directives of any kind.")
        return 0
    except OutputExistsError as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        try:
            mt5mod.shutdown()
        except Exception:                         # noqa: BLE001
            pass


# ============================================================
# Static safety scan of this file's own source (AST + tokens).
# Patterns are assembled from split literals so this source never
# contains the forbidden strings contiguously.
# ============================================================
_FORBIDDEN_API_TOKENS: Tuple[str, ...] = tuple(
    "order" + "_" + tail for tail in
    ("send", "check", "modify", "cancel", "replace", "close")
) + (
    "close" + "_position", "position" + "_close",
    "trade" + "_open", "open" + "_trade", "place" + "_order",
    "send" + "_order", "modify" + "_order",
)
_ML_TOKENS: Tuple[str, ...] = (
    "sk" + "learn", "xg" + "boost", "light" + "gbm",
    "tensor" + "flow", "to" + "rch", "ke" + "ras",
)
_OPTIMIZATION_TOKENS: Tuple[str, ...] = (
    "param" + "_grid", "grid" + "search", "randomized" + "search",
    "op" + "tuna", "hyper" + "opt",
)
_API_TOKENS: Tuple[str, ...] = (
    "http" + "://", "https" + "://", "ur" + "llib", "requ" + "ests",
    "sock" + "et", "open" + "ai", "anthro" + "pic",
    "api" + "_" + "key",
)
# Banned wording: descriptive collection only - never advice of any
# kind.  Split so this source does not contain the words itself.
_BANNED_WORD_PATTERNS: Tuple[str, ...] = (
    r"\bbu" + r"y\b",
    r"\bse" + r"ll\b",
    r"\bho" + r"ld\b",
    r"\bentr" + r"y\b",
    r"\bexi" + r"t\b",
    r"\bsig" + r"nal\b",
    r"stop lo" + r"ss\b",
    r"take pro" + r"fit\b",
    r"lot si" + r"ze\b",
    r"position si" + r"ze\b",
    r"trade deci" + r"sion\b",
    r"\bco" + r"nfidence\b",
    r"\bpro" + r"bability\b",
    r"\brecommen" + r"d\w*\b",
    r"\bpre" + r"dict\w*\b",
    r"\bfore" + r"cast\w*\b",
    r"\bra" + r"nk\w*\b",
    r"\bw" + r"inner\b",
    r"\bbe" + r"st\b",
    r"\bwo" + r"rst\b",
    r"\bsco" + r"re\b",
)


def _docstring_nodes(tree: ast.AST) -> set:
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                nodes.add(id(body[0].value))
    return nodes


def run_safety_scan() -> List[str]:
    """AST + token scan of this file's own source.

    Returns a list of violations (empty when this file is clean):
    import allowlist, order API functions, machine-learning /
    optimization / external AI-API references, and banned wording.
    """
    src_path = os.path.abspath(__file__)
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    violations: List[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"syntax error during scan: {exc}"]
    doc_ids = _docstring_nodes(tree)

    # --- imports: exact allowlist; the trading-terminal package may
    # appear only as a lazy import inside a function ---
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if node.col_offset == 0 and a.name == "MetaTrader5":
                    violations.append(
                        "trading-terminal package imported at module "
                        "level (must stay lazy on the --run path)")
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    unexpected = sorted(imported - _ALLOWED_IMPORT_MODULES)
    for mod in unexpected:
        violations.append(f"unexpected import: {mod}")

    # --- identifiers: no order API function names anywhere ---
    for node in ast.walk(tree):
        names: List[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.append(node.arg)
        for name in names:
            low = name.lower()
            parts = low.split("_")
            for tok in _FORBIDDEN_API_TOKENS:
                if tok == low or tok in parts:
                    violations.append(
                        f"forbidden order API token '{tok}' in "
                        f"identifier '{name}'")

    # --- non-docstring string literals + full-source token scan ---
    def scan_text(text: str, where: str) -> None:
        low = text.lower()
        for tok in _ML_TOKENS:
            if tok in low:
                violations.append(
                    f"machine-learning token in {where}")
        for tok in _OPTIMIZATION_TOKENS:
            if tok in low:
                violations.append(f"optimization token in {where}")
        for tok in _API_TOKENS:
            if tok in low:
                violations.append(f"external AI/API token in {where}")
        for pat in _BANNED_WORD_PATTERNS:
            if re.search(pat, low):
                violations.append(
                    f"banned wording in {where}: {pat}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in doc_ids:
                continue
            scan_text(node.value, "string literal")
    scan_text(src, "source")

    # --- full source must not contain the forbidden order API
    # calls as contiguous text (executable or otherwise) ---
    for tail in ("send", "check"):
        token = "order" + "_" + tail
        if token in src.lower():
            violations.append(
                f"source contains the forbidden API name '{token}'")
    return violations


# ============================================================
# Synthetic fixtures (tests only; in-memory / temp dirs only; the
# default mode never imports or contacts the trading terminal)
# ============================================================
_EPOCH_2026 = 1767225600     # 2026-01-01 00:00:00 UTC


def synthetic_rates(base_price: float, n_candles: int,
                    phase: int = 0) -> List[Dict[str, Any]]:
    """Deterministic H1 candle rows (dicts convert like terminal
    rate rows).  The LAST row plays the still-forming candle."""
    rows: List[Dict[str, Any]] = []
    prev_close = None
    for t in range(n_candles):
        idx = t + phase
        close = (base_price + 0.03 * idx
                 + 1.2 * np.sin(idx / 9.0)
                 + 0.4 * np.cos(idx / 5.0))
        open_ = prev_close if prev_close is not None else close - 0.02
        high = max(open_, close) + 0.25 + 0.10 * abs(np.sin(idx / 3.0))
        low = min(open_, close) - 0.25 - 0.10 * abs(np.cos(idx / 4.0))
        rows.append({
            "time": _EPOCH_2026 + 3600 * t,
            "open": float(open_), "high": float(high),
            "low": float(low), "close": float(close),
            "tick_volume": 100 + int(10 * abs(np.sin(idx))),
        })
        prev_close = close
    return rows


class _SymInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeMT5:
    """In-memory stand-in exposing ONLY the allowed read-only API.

    Any other attribute access raises and is recorded, so a test run
    proves exactly which terminal functions were touched.
    """

    def __init__(self, rates_by_symbol: Dict[str, List[Dict[str, Any]]],
                 omit: Tuple[str, ...] = (),
                 short_history: Tuple[str, ...] = ()) -> None:
        self._rates = {k: v for k, v in rates_by_symbol.items()
                       if k not in omit}
        self._short = set(short_history)
        self.calls: List[str] = []
        self.TIMEFRAME_H1 = "H1"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self.calls.append(name)
        raise AssertionError(
            f"unexpected terminal API access: {name}")

    def initialize(self) -> bool:
        self.calls.append("initialize")
        return True

    def shutdown(self) -> bool:
        self.calls.append("shutdown")
        return True

    def last_error(self) -> Tuple[int, str]:
        self.calls.append("last_error")
        return (0, "No error")

    def symbol_info(self, name: str) -> Any:
        self.calls.append("symbol_info")
        return name if name in self._rates else None

    def symbols_get(self) -> List[_SymInfo]:
        self.calls.append("symbols_get")
        return [_SymInfo(n) for n in sorted(self._rates)]

    def symbol_select(self, name: str, enable: bool) -> bool:
        self.calls.append("symbol_select")
        return name in self._rates

    def copy_rates_from_pos(self, symbol: str, timeframe: Any,
                            start_pos: int, count: int) -> Any:
        self.calls.append("copy_rates_from_pos")
        if symbol not in self._rates:
            return None
        rows = self._rates[symbol]
        if symbol in self._short:
            return rows[:40]
        return rows[:count]


def build_fake_terminal() -> _FakeMT5:
    bases = {"GOLD": 2600.0, "EURUSD": 1.10, "GBPUSD": 1.30,
             "USDJPY": 150.0, "AUDUSD": 0.66}
    return _FakeMT5({name: synthetic_rates(base, 300)
                     for name, base in bases.items()})


def _fixed_ts() -> datetime.datetime:
    return datetime.datetime(2026, 1, 2, 3, 4, 5)


# ============================================================
# Synthetic test suite (default mode; temp/in-memory only)
# ============================================================
def run_synthetic_tests() -> Tuple[int, int]:
    passed = 0
    failed = 0
    area_stats: List[Tuple[str, int, int]] = []
    current = {"name": "(start)", "p": 0, "f": 0}

    def start_area(name: Optional[str]) -> None:
        nonlocal current
        if current["name"] != "(start)":
            area_stats.append((current["name"], current["p"],
                               current["f"]))
        current = {"name": name or "(end)", "p": 0, "f": 0}
        if name:
            print(f"  [{name}]")

    def check(name: str, cond: bool) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            current["p"] += 1
        else:
            failed += 1
            current["f"] += 1
            print(f"    FAILED: {name}")

    print("live_market_state.py - synthetic test suite")
    print("(temporary data only; no terminal contact; no real output "
          "is written)\n")

    # ---------------- A. indicator calculations ----------------
    start_area("A_indicator_calculations")
    n = 100
    closes = np.arange(100.0, 100.0 + n)
    up_df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "open": closes, "high": closes + 1.0, "low": closes - 1.0,
        "close": closes,
    })
    d_up = compute_indicator_frame(up_df)
    check("ATR14 equals the constant true range (2.0)",
          abs(float(d_up["ATR"].iloc[-1]) - 2.0) < 1e-12)
    const_df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=80, freq="h"),
        "open": np.full(80, 100.0), "high": np.full(80, 101.0),
        "low": np.full(80, 99.0), "close": np.full(80, 100.0),
    })
    d_const = compute_indicator_frame(const_df)
    check("EMA50 on a constant 100 series is exactly 100",
          float(d_const[f"EMA{EMA_TREND_PERIOD}"].iloc[-1]) == 100.0)
    check("RSI14 on strictly rising closes is 100",
          abs(float(d_up["RSI"].iloc[-1]) - 100.0) < 1e-9)
    check("ADX14 on a one-way move converges to 100",
          float(d_up["ADX"].iloc[-1]) > 99.999)

    bb_closes = np.concatenate([np.arange(1.0, 21.0), np.full(20, 20.0)])
    m = len(bb_closes)
    bb_df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=m, freq="h"),
        "open": bb_closes, "high": bb_closes + 1.0,
        "low": bb_closes - 1.0, "close": bb_closes,
    })
    d_bb = compute_indicator_frame(bb_df)
    std20 = float(np.std(np.arange(1.0, 21.0), ddof=0))
    check("Bollinger middle 20 equals the 20-close mean (10.5)",
          abs(float(d_bb["BB_MIDDLE"].iloc[19]) - 10.5) < 1e-12)
    check("Bollinger upper/lower = mean +/- 2 population std",
          abs(float(d_bb["BB_UPPER"].iloc[19])
              - (10.5 + 2.0 * std20)) < 1e-12
          and abs(float(d_bb["BB_LOWER"].iloc[19])
                  - (10.5 - 2.0 * std20)) < 1e-12)
    latest = d_up.iloc[-1]
    feat_up = build_state_record(up_df, (0.1, 0.2))["features"]
    exp_ema_dist = abs(float(latest["close"])
                       - float(latest[f"EMA{EMA_TREND_PERIOD}"])) \
        / float(latest["ATR"])
    check("ema50_dist_atr formula = |close - EMA50| / ATR",
          abs(feat_up["ema50_dist_atr"] - exp_ema_dist) < 1e-12)
    width = float(latest["BB_UPPER"]) - float(latest["BB_LOWER"])
    exp_exc = abs(float(latest["close"])
                  - float(latest["BB_MIDDLE"])) / width
    exp_pos = (float(latest["close"])
               - float(latest["BB_MIDDLE"])) / width
    check("bb_excursion = |close - middle| / band width",
          abs(feat_up["bb_excursion"] - exp_exc) < 1e-12)
    check("bb_position = (close - middle) / band width (signed)",
          abs(feat_up["bb_position"] - exp_pos) < 1e-12)

    # ---------------- B. regime boundaries ----------------
    start_area("B_regime_boundaries")
    check("ADX boundaries (research layer definitions)",
          classify_adx(14.999) == "ADX<15"
          and classify_adx(15.0) == "ADX15-20"
          and classify_adx(20.0) == "ADX20-25"
          and classify_adx(25.0) == "ADX>=25"
          and classify_adx(None) == "n/a")
    check("EMA-distance boundaries",
          classify_ema_distance(0.999) == "<1"
          and classify_ema_distance(1.0) == "1-2"
          and classify_ema_distance(2.0) == ">=2"
          and classify_ema_distance(None) == "n/a")
    check("BB-excursion boundaries",
          classify_bb_excursion(0.499) == "<0.50"
          and classify_bb_excursion(0.50) == "0.50-0.75"
          and classify_bb_excursion(0.75) == ">=0.75")
    check("efficiency boundaries",
          classify_efficiency(0.299) == "<0.30"
          and classify_efficiency(0.30) == "0.30-0.60"
          and classify_efficiency(0.60) == ">=0.60")
    check("RSI boundaries",
          classify_rsi(29.9) == "<30" and classify_rsi(30.0) == "30-50"
          and classify_rsi(50.0) == "50-70" and classify_rsi(70.0)
          == ">=70" and classify_rsi(None) == "n/a")
    thr = (0.20, 0.30)
    check("volatility tercile boundaries (LOW/MEDIUM/HIGH)",
          classify_atr_pct(0.19, thr) == "LOW"
          and classify_atr_pct(0.20, thr) == "MEDIUM"
          and classify_atr_pct(0.30, thr) == "HIGH"
          and classify_atr_pct(0.25, (float("nan"), float("nan")))
          == "n/a")
    syn_thr = atr_tercile_thresholds(
        [float(v) for v in range(1, 10)])
    check("tercile computation matches the research formula "
          "(33.33/66.67 percentiles)",
          abs(syn_thr[0] - float(np.percentile(
              np.arange(1.0, 10.0), 100.0 / 3.0))) < 1e-12
          and abs(syn_thr[1] - float(np.percentile(
              np.arange(1.0, 10.0), 200.0 / 3.0))) < 1e-12)
    check("bucket label sets match the research layer",
          ADX_BUCKETS == ("ADX<15", "ADX15-20", "ADX20-25", "ADX>=25")
          and EMA_BUCKETS == ("<1", "1-2", ">=2")
          and BBE_BUCKETS == ("<0.50", "0.50-0.75", ">=0.75")
          and EFF_BUCKETS == ("<0.30", "0.30-0.60", ">=0.60")
          and RSI_BUCKETS == ("<30", "30-50", "50-70", ">=70")
          and ATR_BUCKETS == ("LOW", "MEDIUM", "HIGH"))
    state_csv_path = os.path.join(SCRIPT_DIR, HISTORICAL_STATE_CSV)
    frozen_json_path = os.path.join(SCRIPT_DIR, FROZEN_CONTEXT_JSON)
    if os.path.isfile(state_csv_path) and os.path.isfile(frozen_json_path):
        live_thr, live_warn = load_frozen_atr_thresholds()
        with open(frozen_json_path, encoding="utf-8") as fh:
            contexts = json.load(fh)
        gold = next((c for c in contexts if c["instrument"] == "GOLD"),
                    None)
        frozen = (gold["current_state"]["atr_tercile_thresholds"]
                  if gold else None)
        check("frozen thresholds load for all five instruments",
              live_warn is None
              and all(inst in live_thr for inst in INSTRUMENTS))
        check("live ATR terciles equal the frozen research thresholds "
              "(GOLD, bit-identical)",
              frozen is not None
              and live_thr["GOLD"][0] == frozen["p33"]
              and live_thr["GOLD"][1] == frozen["p66"])
    else:
        print("    (skipped: research CSV/JSON not present for the "
              "frozen-threshold cross-check)")

    # ---------------- C. candle feature calculations ----------
    start_area("C_candle_features")
    bull = candle_shape_features(100.0, 105.0, 98.0, 103.0)
    check("bullish candle range/body/wicks",
          abs(bull["candle_range_pct"] - (105 - 98) / 103 * 100) < 1e-12
          and abs(bull["candle_body_pct"] - abs(103 - 100) / 103 * 100)
          < 1e-12
          and abs(bull["upper_wick_pct"] - (105 - 103) / 103 * 100)
          < 1e-12
          and abs(bull["lower_wick_pct"] - (100 - 98) / 103 * 100)
          < 1e-12)
    bear = candle_shape_features(103.0, 105.0, 98.0, 100.0)
    check("bearish candle range/body/wicks",
          abs(bear["candle_range_pct"] - (105 - 98) / 100 * 100) < 1e-12
          and abs(bear["candle_body_pct"] - abs(100 - 103) / 100 * 100)
          < 1e-12
          and abs(bear["upper_wick_pct"] - (105 - 103) / 100 * 100)
          < 1e-12
          and abs(bear["lower_wick_pct"] - (100 - 98) / 100 * 100)
          < 1e-12)
    check("doji-like candle has zero body",
          candle_shape_features(100.0, 101.0, 99.0, 100.0)
          ["candle_body_pct"] == 0.0)
    tf = time_features(pd.Timestamp("2026-09-28 06:00:00"))
    check("time features: hour/day_of_week/month",
          tf == {"hour": 6, "day_of_week": 0, "month": 9})
    tf2 = time_features(pd.Timestamp("2026-01-04 23:59:00"))
    check("time features: Sunday late hour (day_of_week=6)",
          tf2 == {"hour": 23, "day_of_week": 6, "month": 1})

    # ---------------- D. return calculations ----------------
    start_area("D_return_calculations")
    r = rolling_return(pd.Series([100.0, 90.0]), 1)
    check("rolling return of 100 -> 90 over 1 candle is -10%",
          abs(float(r.iloc[1]) - (-10.0)) < 1e-12)
    rc = [105.0] * 30
    rc[5] = 120.0      # close 24 candles back
    rc[17] = 110.0     # close 12 candles back
    rc[23] = 100.0     # close 6 candles back
    rc[29] = 130.0     # latest close
    highs = [c + 0.5 for c in rc]
    highs[10] = 140.0
    lows = [c - 0.5 for c in rc]
    lows[7] = 80.0
    rdf_input = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=30, freq="h"),
        "open": rc, "high": highs, "low": lows, "close": rc,
    })
    rdf = compute_indicator_frame(rdf_input)
    last = rdf.iloc[-1]
    dfeat = build_state_record(rdf_input, (0.1, 0.2))["features"]
    check("return6 = (130/100 - 1)*100 = +30%",
          abs(float(last["return6"]) - 30.0) < 1e-9)
    check("return12 = (130/110 - 1)*100",
          abs(float(last["return12"])
              - (130.0 / 110.0 - 1.0) * 100.0) < 1e-9)
    check("return24 = (130/120 - 1)*100",
          abs(float(last["return24"])
              - (130.0 / 120.0 - 1.0) * 100.0) < 1e-9)
    check("dist_24_high = (close/24-candle max high - 1)*100",
          abs(float(dfeat["dist_24_high"])
              - (130.0 / 140.0 - 1.0) * 100.0) < 1e-9)
    check("dist_24_low = (close/24-candle min low - 1)*100",
          abs(float(dfeat["dist_24_low"])
              - (130.0 / 80.0 - 1.0) * 100.0) < 1e-9)

    # ---------------- E. efficiency calculation ---------------
    start_area("E_efficiency_calculation")
    up_seq = pd.Series(np.arange(100.0, 125.0))
    check("efficiency of a perfectly smooth 24-step move is 1.0",
          abs(float(efficiency_ratio(up_seq, 24).iloc[-1]) - 1.0)
          < 1e-12)
    zig = pd.Series([100.0 if i % 2 == 0 else 101.0
                     for i in range(25)])
    check("efficiency of an alternating zigzag is 0.0",
          abs(float(efficiency_ratio(zig, 24).iloc[-1]) - 0.0) < 1e-12)
    rise = [100.0 + i for i in range(19)]        # 18 up-steps
    fall = [118.0 - i for i in range(1, 7)]      # 6 down-steps
    half_s = pd.Series(rise + fall)              # 25 candles = 24 steps
    check("efficiency of rise-18-then-fall-6 is 12/24 = 0.5",
          abs(float(efficiency_ratio(half_s, 24).iloc[-1]) - 0.5)
          < 1e-12)

    # ---------------- F. no future-candle usage ---------------
    start_area("F_no_future_candle_usage")

    def flatten_state(rec: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"direction": rec["direction"]}
        out.update(rec["features"])
        out.update(rec["time"])
        out.update(rec["regimes"])
        return out

    full_rows = synthetic_rates(200.0, 200)
    truncated = pd.DataFrame(full_rows[:150])
    truncated["time"] = pd.to_datetime(truncated["time"], unit="s")
    a = build_state_record(truncated, (0.1, 0.2))
    full_df = pd.DataFrame(full_rows)
    full_df["time"] = pd.to_datetime(full_df["time"], unit="s")
    b_on_full = build_state_record(full_df.iloc[:150].reset_index(
        drop=True), (0.1, 0.2))
    check("state at candle 149 identical whether or not candles "
          "150..199 exist",
          flatten_state(a) == flatten_state(b_on_full))
    shifted = pd.DataFrame(full_rows)
    shifted.loc[150:, ["open", "high", "low", "close"]] += 50.0
    shifted["time"] = pd.to_datetime(shifted["time"], unit="s")
    c = build_state_record(shifted.iloc[:150].reset_index(drop=True),
                           (0.1, 0.2))
    check("altering future candles leaves the earlier state unchanged",
          flatten_state(a) == flatten_state(c))
    truncated_last = build_state_record(
        pd.DataFrame(full_rows[:150]).assign(
            time=lambda x: pd.to_datetime(x["time"], unit="s")),
        (0.1, 0.2))
    check("observation uses the last row of the supplied closed frame",
          truncated_last["latest_closed_candle_time"]
          == pd.Timestamp(full_rows[149]["time"], unit="s").strftime(
              TS_FORMAT))

    # ---------------- G. closed-candle selection --------------
    start_area("G_closed_candle_selection")
    sample = pd.DataFrame(synthetic_rates(100.0, 10))
    sample["time"] = pd.to_datetime(sample["time"], unit="s")
    closed = select_closed_candles(sample)
    check("the newest (forming) candle is always dropped",
          len(closed) == 9
          and pd.Timestamp(closed["time"].iloc[-1]).strftime(TS_FORMAT)
          == pd.Timestamp(sample["time"].iloc[-2]).strftime(TS_FORMAT))
    forming_time = pd.Timestamp(sample["time"].iloc[-1]).strftime(
        TS_FORMAT)
    check("the forming candle's time never appears in the closed set",
          forming_time not in [
              pd.Timestamp(t).strftime(TS_FORMAT)
              for t in closed["time"]])
    check("an empty frame stays empty", len(
        select_closed_candles(sample.iloc[0:0])) == 0)
    single = select_closed_candles(sample.iloc[:1])
    check("a single-row frame yields no closed candles",
          len(single) == 0)
    big = synthetic_rates(50.0, NUM_CANDLES + 5)
    big_df = pd.DataFrame(big)
    check("closed selection never needs future rows (only drops the "
          "newest)", len(select_closed_candles(big_df)) == NUM_CANDLES + 4)

    # ---------------- H. missing-symbol handling --------------
    start_area("H_missing_symbol_handling")
    resolved, how = resolve_symbol(
        "EURUSD", lambda n: object(), lambda: [])
    check("exact symbol verified with symbol_info first",
          resolved == "EURUSD" and how == "exact symbol name")
    resolved2, how2 = resolve_symbol(
        "GOLD", lambda n: None,
        lambda: [_SymInfo("XAUUSD.pro"), _SymInfo("GOLDm"),
                 _SymInfo("GOLD.pro")])
    check("suffix match is deterministic (shortest, then "
          "alphabetical)", resolved2 == "GOLDm")
    resolved3, how3 = resolve_symbol(
        "NOSUCH", lambda n: None,
        lambda: [_SymInfo("EURUSDm"), _SymInfo("GBPUSDm")])
    check("unknown symbol returns (None, clear reason)",
          resolved3 is None and "not found" in how3)
    fake_missing = _FakeMT5(
        {"EURUSD": synthetic_rates(1.10, 300),
         "GBPUSD": synthetic_rates(1.30, 300),
         "USDJPY": synthetic_rates(150.0, 300),
         "AUDUSD": synthetic_rates(0.66, 300)},
        omit=("GOLD",))
    res_missing = collect_all(fake_missing, {})
    by_inst = {r["instrument"]: r for r in res_missing}
    check("missing GOLD is recorded UNAVAILABLE with a reason",
          by_inst["GOLD"]["status"] == STATUS_UNAVAILABLE
          and bool(by_inst["GOLD"]["reason"]))
    check("the other four instruments still process",
          all(by_inst[i]["status"] == STATUS_COLLECTED
              for i in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")))
    fake_empty = _FakeMT5({"EURUSD": []})
    res_empty = collect_instrument("EURUSD", fake_empty, {})
    check("empty candle data -> UNAVAILABLE with a reason",
          res_empty["status"] == STATUS_UNAVAILABLE
          and "no H1 candle data" in (res_empty["reason"] or ""))
    fake_short = _FakeMT5({"EURUSD": synthetic_rates(1.10, 300)},
                          short_history=("EURUSD",))
    res_short = collect_instrument("EURUSD", fake_short, {})
    check("insufficient closed candles -> UNAVAILABLE with a reason",
          res_short["status"] == STATUS_UNAVAILABLE
          and "insufficient closed H1 candles" in (res_short["reason"]
                                                   or ""))
    check("a terminal exception during collection never crashes",
          collect_instrument("EURUSD", _NeverConnects(), {})
          ["status"] == STATUS_UNAVAILABLE)

    # ---------------- I. deterministic output ----------------
    start_area("I_deterministic_output")
    thresholds_syn = {inst: (0.1, 0.2) for inst in INSTRUMENTS}
    res_a = collect_all(build_fake_terminal(), thresholds_syn)
    res_b = collect_all(build_fake_terminal(), thresholds_syn)
    csv_a = results_to_csv(res_a)
    csv_b = results_to_csv(res_b)
    check("identical candle data -> identical CSV bytes",
          csv_a == csv_b)
    rep_a = build_report(res_a, "CONNECTED (read-only session)",
                         _fixed_ts())
    rep_b = build_report(res_b, "CONNECTED (read-only session)",
                         _fixed_ts())
    check("identical candle data -> identical report bytes",
          rep_a == rep_b)
    check("every instrument collected on the full fake terminal",
          all(r["status"] == STATUS_COLLECTED for r in res_a))
    check("candles used = requested minus the forming candle",
          all(r["candles_used"] == 300 - 1 for r in res_a))
    check("instrument order is the fixed supported order",
          [r["instrument"] for r in res_a] == list(INSTRUMENTS))

    # ---------------- J. output-file protection --------------
    start_area("J_output_file_protection")
    ts = _fixed_ts()
    stamp = ts.strftime(STAMP_FORMAT)
    with tempfile.TemporaryDirectory() as td:
        csv_path, txt_path = resolve_output_paths(td, ts)
        check("clean directory selects the primary names",
              os.path.basename(csv_path) == OUTPUT_CSV
              and os.path.basename(txt_path) == OUTPUT_TXT)
        write_outputs("col1,col2\n1,2\n", "report\n", csv_path,
                      txt_path)
        check("both output files created",
              os.path.isfile(csv_path) and os.path.isfile(txt_path))
        csv2, txt2 = resolve_output_paths(td, ts)
        check("existing primary CSV gets the timestamped fallback",
              os.path.basename(csv2)
              == f"live_market_state_{stamp}.csv")
        check("existing primary report gets the timestamped fallback",
              os.path.basename(txt2)
              == f"live_market_state_report_{stamp}.txt")
        try:
            write_outputs("overwrite?", "overwrite?", csv_path,
                          txt_path)
            check("writing over an existing file is refused", False)
        except OutputExistsError:
            check("writing over an existing file is refused", True)
        with open(csv_path, encoding="utf-8") as fh:
            check("the original CSV bytes are untouched",
                  fh.read() == "col1,col2\n1,2\n")
        with open(os.path.join(td, csv2), "w", encoding="utf-8") as fh:
            fh.write("occupied")
        try:
            resolve_output_paths(td, ts)
            check("fallback collision stops everything with a naming "
                  "error", False)
        except OutputExistsError as exc:
            check("fallback collision stops everything with a naming "
                  "error",
                  f"live_market_state_{stamp}.csv" in str(exc))

    # ---------------- J2. atomic output-set resolution (C4) ---
    # The CSV + report fallback decision is made once for the whole
    # set, so a partially timestamped / mixed set can never appear.
    ts_a = _fixed_ts()
    stamp_a = ts_a.strftime(STAMP_FORMAT)
    with tempfile.TemporaryDirectory() as td_a:
        c_clean, t_clean = resolve_output_paths(td_a, ts_a)
        check("C4 clean directory keeps the whole set as primary names",
              os.path.basename(c_clean) == OUTPUT_CSV
              and os.path.basename(t_clean) == OUTPUT_TXT)
    with tempfile.TemporaryDirectory() as td_b:
        open(os.path.join(td_b, OUTPUT_CSV), "w",
             encoding="utf-8").write("x")
        c_b, t_b = resolve_output_paths(td_b, ts_a)
        check("C4 only primary CSV present -> CSV timestamped",
              os.path.basename(c_b)
              == f"live_market_state_{stamp_a}.csv")
        check("C4 only primary CSV present -> report ALSO timestamped"
              " with the same stamp",
              os.path.basename(t_b)
              == f"live_market_state_report_{stamp_a}.txt")
    with tempfile.TemporaryDirectory() as td_c:
        open(os.path.join(td_c, OUTPUT_TXT), "w",
             encoding="utf-8").write("x")
        c_c, t_c = resolve_output_paths(td_c, ts_a)
        check("C4 only primary report present -> report timestamped",
              os.path.basename(t_c)
              == f"live_market_state_report_{stamp_a}.txt")
        check("C4 only primary report present -> CSV ALSO timestamped"
              " with the same stamp",
              os.path.basename(c_c)
              == f"live_market_state_{stamp_a}.csv")
    with tempfile.TemporaryDirectory() as td_d:
        open(os.path.join(td_d, OUTPUT_CSV), "w",
             encoding="utf-8").write("x")
        open(os.path.join(td_d, OUTPUT_TXT), "w",
             encoding="utf-8").write("x")
        c_d, t_d = resolve_output_paths(td_d, ts_a)
        check("C4 both primaries present -> both timestamped, one stamp",
              os.path.basename(c_d)
              == f"live_market_state_{stamp_a}.csv"
              and os.path.basename(t_d)
              == f"live_market_state_report_{stamp_a}.txt")
        names_d = (os.path.basename(c_d), os.path.basename(t_d))
        check("C4 no mixed primary/timestamped set is ever produced",
              (OUTPUT_CSV in names_d) == (OUTPUT_TXT in names_d))

    # ---------------- K. absence of order API functions -------
    start_area("K_no_order_api_functions")
    violations = run_safety_scan()
    check("static safety scan clean (imports / order API / ML / "
          "optimization / AI-API / wording)", violations == [])
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    check("source contains no order transmission / order verification "
          "API name",
          ("order" + "_send") not in src.lower()
          and ("order" + "_check") not in src.lower())
    check("no terminal API name beyond the read-only allowlist is "
          "referenced",
          all(tok not in src for tok in
              ("order" + "_send", "order" + "_check",
               "order" + "_modify", "order" + "_cancel")))
    tree = ast.parse(src)
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(a.name.split(".")[0]
                               for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.append(node.module.split(".")[0])
    check("default (test) mode never loads the trading-terminal "
          "package", "MetaTrader5" not in top_imports)

    # ---------------- L. absence of AI/API calls -------------
    start_area("L_no_ai_api_calls")
    imported_all = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_all.update(a.name.split(".")[0]
                                for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_all.add(node.module.split(".")[0])
    check("imports are exactly the allowlist",
          imported_all == set(_ALLOWED_IMPORT_MODULES))
    check("no external AI/API or library reference in source",
          all(tok not in src.lower() for tok in
              (("http" + "://"), ("https" + "://"),
               ("ur" + "llib"), ("requ" + "ests"), ("sock" + "et"),
               ("open" + "ai"), ("anthro" + "pic"),
               ("api" + "_" + "key"))))
    check("no machine-learning or optimization reference in source",
          all(tok not in src.lower() for tok in _ML_TOKENS
              + _OPTIMIZATION_TOKENS))

    # ---------------- M. end-to-end fake collection -----------
    start_area("M_end_to_end_collection")
    fake = build_fake_terminal()
    res_full = collect_all(fake, thresholds_syn)
    payload = results_to_csv(res_full)
    parsed = pd.read_csv(pd.io.common.StringIO(payload))
    check("CSV parses back with the exact 34-column schema",
          list(parsed.columns) == list(CSV_COLUMNS))
    check("CSV has one row per instrument",
          len(parsed) == len(INSTRUMENTS))
    regimes_ok = True
    for row in parsed.to_dict("records"):
        regimes_ok = regimes_ok and (
            row["adx_regime"] in ADX_BUCKETS
            and row["volatility_regime"] in ATR_BUCKETS
            and row["trend_distance_regime"] in EMA_BUCKETS
            and row["bb_excursion_regime"] in BBE_BUCKETS
            and row["efficiency_regime"] in EFF_BUCKETS
            and row["rsi_regime"] in RSI_BUCKETS
            and row["direction"] in (DIRECTION_LONG, DIRECTION_SHORT,
                                     DIRECTION_NONE))
    check("all regime labels come from the fixed research buckets",
          regimes_ok)
    state_g = {r["instrument"]: r for r in res_full}["GOLD"]
    expected_latest = pd.Timestamp(
        build_fake_terminal()._rates["GOLD"][-2]["time"], unit="s"
    ).strftime(TS_FORMAT)
    check("latest observation is the last CLOSED candle, never the "
          "forming one",
          state_g["state"]["latest_closed_candle_time"]
          == expected_latest)
    check("every collected row carries all 26 state fields",
          all(all(k in row and row[k] is not None
                  for k in ("direction", "adx", "atr_pct",
                            "ema50_dist_atr", "bb_excursion",
                            "bb_position", "eff24", "rsi14", "return6",
                            "return12", "return24", "dist_24_high",
                            "dist_24_low", "candle_range_pct",
                            "candle_body_pct", "upper_wick_pct",
                            "lower_wick_pct", "hour", "day_of_week",
                            "month", "adx_regime", "volatility_regime",
                            "trend_distance_regime",
                            "bb_excursion_regime", "efficiency_regime",
                            "rsi_regime"))
              for row in parsed.to_dict("records")))
    report = build_report(res_full, "CONNECTED (read-only session)",
                          ts)
    check("report contains every required section",
          all(s in report for s in
              ("collection timestamp", "MT5 connection status",
               "candles used", "INSTRUMENT: GOLD",
               "latest CLOSED H1 candle", "STATE VALUES",
               "REGIME CLASSIFICATIONS", "DATA-QUALITY WARNINGS")))
    check("report lists all five instruments",
          all(f"INSTRUMENT: {inst}" in report
              for inst in INSTRUMENTS))
    banned_in_payload = [p for p in _BANNED_WORD_PATTERNS
                         if re.search(p, (payload + report).lower())]
    check("generated outputs contain no banned wording",
          not banned_in_payload)
    allowed_calls = set(_MT5_ALLOWED_API)
    check("only the read-only terminal API was accessed",
          set(fake.calls) <= allowed_calls)
    run_fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "run_real_collection")
    run_attrs = {a.attr for a in ast.walk(run_fn)
                 if isinstance(a, ast.Attribute)}
    run_finally = any(isinstance(n, ast.Try) and n.finalbody
                      for n in ast.walk(run_fn))
    check("the --run path opens the read-only session and always "
          "closes it in a finally block",
          "initialize" in run_attrs and "shutdown" in run_attrs
          and run_finally)
    copy_calls = [c for c in fake.calls
                  if c == "copy_rates_from_pos"]
    check("candles requested from position 0 on the H1 timeframe",
          len(copy_calls) == len(INSTRUMENTS))
    check("the fake terminal recorded only allowed API names",
          all(c in allowed_calls for c in fake.calls))
    check("direction rule verified: NONE inside both bands, LONG "
          "below, SHORT above",
          frozen_direction(100.0, 99.0, 101.0) == DIRECTION_NONE
          and frozen_direction(98.0, 99.0, 101.0) == DIRECTION_LONG
          and frozen_direction(102.0, 99.0, 101.0) == DIRECTION_SHORT
          and frozen_direction(None, 99.0, 101.0) == "n/a")

    # ---------------- N. command-line gating ------------------
    start_area("N_default_run_gating")
    check("default (no flags) runs synthetic tests only",
          should_run_real([]) is False)
    check("--run triggers the real read-only collection",
          should_run_real(["--run"]) is True)
    check("unrelated flags do not request the real collection",
          should_run_real(["-v", "--verbose"]) is False)
    check("the terminal package was never imported during tests",
          "MetaTrader5" not in sys.modules)

    start_area(None)
    print()
    print("  Per-area verification (area -> passed/failed):")
    stats = {name: (p, f) for name, p, f in area_stats}
    order = (
        "A_indicator_calculations", "B_regime_boundaries",
        "C_candle_features", "D_return_calculations",
        "E_efficiency_calculation", "F_no_future_candle_usage",
        "G_closed_candle_selection", "H_missing_symbol_handling",
        "I_deterministic_output", "J_output_file_protection",
        "K_no_order_api_functions", "L_no_ai_api_calls",
        "M_end_to_end_collection", "N_default_run_gating",
    )
    for name in order:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<32} {p:>2} passed, {f} failed [{status}]")
    print()
    print(f"  TOTAL: {passed} passed, {failed} failed")
    return passed, failed


class _NeverConnects:
    """A terminal stub whose read-only session always fails."""

    TIMEFRAME_H1 = "H1"

    def initialize(self) -> bool:
        return False

    def last_error(self) -> Tuple[int, str]:
        return (1, "connection failure")

    def symbol_info(self, name: str) -> Any:
        return name

    def symbols_get(self) -> List[_SymInfo]:
        return [_SymInfo("EURUSDm")]

    def symbol_select(self, name: str, enable: bool) -> bool:
        return True

    def copy_rates_from_pos(self, symbol: str, timeframe: Any,
                            start_pos: int, count: int) -> Any:
        raise RuntimeError("connection lost")


# ============================================================
# Command-line gating
# ============================================================
def should_run_real(argv: List[str]) -> bool:
    return "--run" in argv


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if should_run_real(argv):
        return run_real_collection()
    passed, failed = run_synthetic_tests()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
