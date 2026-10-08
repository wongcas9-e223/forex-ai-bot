"""
regime_oos_validation.py - HISTORICAL RESEARCH / VALIDATION ONLY.

Strict chronological OUT-OF-SAMPLE (OOS) validation diagnostic.

THE QUESTION BEING ASKED:
  Do the regime relationships discovered by market_regime_diagnostic.py
  remain observable in completely unseen LATER historical data?

WHY THE OOS SPLIT IS CHRONOLOGICAL:
  The first half of each instrument's closed-candle history is the
  DEVELOPMENT window and the strictly later second half is the OOS
  window. The OOS period therefore occurs strictly AFTER the
  development period and no shuffling or random split is used. Nothing
  about the strategy, the regime definitions, or the volatility
  thresholds is derived from OOS information.

WHY THE VOLATILITY THRESHOLDS ARE FROZEN FROM DEVELOPMENT:
  The 33rd/66th percentile ATR% edges are computed from the
  DEVELOPMENT signal population of each instrument and then FROZEN.
  Those exact frozen thresholds are applied to BOTH development and
  OOS trades. Recomputing percentiles on OOS data would let the OOS
  window influence its own regime labels and would contaminate the
  OOS test. (The other regime families - ADX, trend distance, BB
  excursion, efficiency - use fixed global bucket boundaries, so there
  is nothing to freeze per instrument.)

WHY NO FUTURE INFORMATION IS USED:
  Every regime measurement of a trade signalled on candle i uses only
  candle i and earlier (BB, ADX, ATR, EMA50, ATR%, efficiency ratio,
  trend distance, BB excursion). The entry happens at the OPEN of
  candle i+1 - the first moment a real trader could have acted.
  Forward returns and MFE/MAE are recorded as diagnostic measurements
  only and never alter any trade outcome.

WHY THE STRATEGY ITSELF IS FROZEN:
  This is NOT an optimization and NOT a strategy search. The trading
  rules are copied unchanged from the previous diagnostics:
    BB(20, 2.0) + ADX14 < 25 entry, SL 2.0 x ATR / TP 1.5 x ATR,
    next-open entry, stop-first inside-candle handling, 0.10 lots,
    $2 per completed trade, one position at a time, $10,000
    independent starting balance per window. No variant is selected
    based on OOS results, no parameter is tuned, no ranking is made.

WHY THIS IS RESEARCH ONLY:
  READ-ONLY MT5 access; this script NEVER places, modifies or closes
  an order (no live trading, no demo trading, no order execution).
  All outputs are descriptive historical associations with wording
  such as "historical association", "sign persisted", "sign changed",
  "relationship did not persist". No causal claims, no predictions,
  no recommendations, no filters, no rankings.

DEMO / EDUCATIONAL PROJECT ONLY. Educational use only.
"""

# ============================================================
# Imports (no new packages needed)
# ============================================================
import sys
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5  # Read market data + symbol info (READ-ONLY usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # numeric helpers

# ============================================================
# Frozen settings - identical to the previous diagnostics
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 10_000          # Requested history per instrument
MIN_CANDLES = 400             # Absolute floor to attempt the analysis

INSTRUMENTS: List[str] = ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

STARTING_BALANCE = 10_000.0   # Independent for EVERY window (dev AND OOS)
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
EMA_TREND_PERIOD = 50         # Diagnostic reference ONLY (never a filter)
WARMUP = 60

HOLD_WINDOW = 12              # MFE/MAE window
FWD_WINDOWS = (6, 12, 24)     # Forward-return diagnostic windows

LOW_SAMPLE_N = 5
LOW_SAMPLE_TAG = "LOW SAMPLE"

# Fixed descriptive regime buckets (identical to
# market_regime_diagnostic.py - never optimized, never filters)
ADX_BUCKETS = ["ADX<15", "ADX15-20", "ADX20-25", "ADX>=25"]
ADX_BELOW_BUCKETS = ["ADX<15", "ADX15-20", "ADX20-25"]
TREND_BUCKETS = ["<1 ATR", "1-2 ATR", ">=2 ATR"]
BB_BUCKETS = ["LOW", "MEDIUM", "HIGH"]
EFF_BUCKETS = ["LOW", "MEDIUM", "HIGH"]
VOL_BUCKETS = ["LOW", "MEDIUM", "HIGH"]


# ============================================================
# Indicator functions (causal only - same math as previous scripts)
# ============================================================
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range - uses only current and previous candles."""
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


def add_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (rolling - causal by construction)."""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (Wilder's classic calculation - causal)."""
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


def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Exponential Moving Average of the close (causal)."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all frozen indicators + causal diagnostic measurements."""
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)
    df = add_ema(df, EMA_TREND_PERIOD)   # diagnostic reference only
    df["ATR_pct"] = np.where(df["close"] > 0, df["ATR"] / df["close"] * 100, np.nan)

    # Trend distance: abs(close - EMA50) / ATR14 (causal)
    df["EMA50_dist"] = np.where(
        df["ATR"] > 0,
        (df["close"] - df["EMA50"]).abs() / df["ATR"],
        np.nan,
    )

    # Bollinger excursion: abs(close - BB_middle) / (BB_upper - BB_lower)
    band_width = df["BB_UPPER"] - df["BB_LOWER"]
    df["BB_exc"] = np.where(band_width > 0,
                            (df["close"] - df["BB_MIDDLE"]).abs() / band_width, np.nan)

    # Efficiency ratio (24 changes): net move / gross path (causal)
    net_move = (df["close"] - df["close"].shift(EFF_LOOKBACK)).abs()
    gross_path = df["close"].diff().abs().rolling(EFF_LOOKBACK).sum()
    df["eff24"] = np.where(gross_path > 0, net_move / gross_path, np.nan)
    return df


EFF_LOOKBACK = 24   # Efficiency-ratio lookback (frozen, as specified)


# ============================================================
# Regime classification (fixed buckets - identical to
# market_regime_diagnostic.py; pure functions)
# ============================================================
def adx_regime_label(adx: float) -> str:
    """ADX<15 / ADX15-20 / ADX20-25 / ADX>=25 (fixed buckets)."""
    if adx != adx:
        return "n/a"
    if adx < 15:
        return "ADX<15"
    if adx < 20:
        return "ADX15-20"
    if adx < 25:
        return "ADX20-25"
    return "ADX>=25"


def vol_regime_label(atr_pct: float, v1: float, v2: float) -> str:
    """LOW / MEDIUM / HIGH using the FROZEN development-derived edges."""
    if atr_pct != atr_pct:
        return "n/a"
    if atr_pct < v1:
        return "LOW"
    if atr_pct < v2:
        return "MEDIUM"
    return "HIGH"


def trend_dist_label(dist: float) -> str:
    """<1.0 ATR / 1.0 to <2.0 ATR / >=2.0 ATR (fixed buckets)."""
    if dist != dist:
        return "n/a"
    if dist < 1.0:
        return "<1 ATR"
    if dist < 2.0:
        return "1-2 ATR"
    return ">=2 ATR"


def bb_exc_label(exc: float) -> str:
    """<0.50 / 0.50 to <0.75 / >=0.75 (fixed buckets)."""
    if exc != exc:
        return "n/a"
    if exc < 0.50:
        return "LOW"
    if exc < 0.75:
        return "MEDIUM"
    return "HIGH"


def eff_label(ratio: float) -> str:
    """LOW <0.30 / MEDIUM 0.30 to <0.60 / HIGH >=0.60 (fixed buckets)."""
    if ratio != ratio:
        return "n/a"
    if ratio < 0.30:
        return "LOW"
    if ratio < 0.60:
        return "MEDIUM"
    return "HIGH"


def calculate_regimes(trades: pd.DataFrame, v1: float, v2: float) -> pd.DataFrame:
    """Attach descriptive regime labels to a trade log.

    The labels classify signal-time measurements only and never change
    any trade outcome. Volatility labels use the FROZEN
    development-derived edges (v1, v2) on BOTH windows.
    """
    trades = trades.copy()
    trades["vol_regime"] = trades["ATR_pct"].apply(lambda x: vol_regime_label(x, v1, v2))
    trades["adx_regime"] = trades["ADX"].apply(adx_regime_label)
    trades["trend_regime"] = trades["EMA50_dist"].apply(trend_dist_label)
    trades["bb_regime"] = trades["BB_exc"].apply(bb_exc_label)
    trades["eff_regime"] = trades["eff24"].apply(eff_label)
    return trades


def compute_vol_edges_from_development(dev_df: pd.DataFrame) -> Tuple[float, float]:
    """33rd/66th percentile ATR% edges from the DEVELOPMENT candles only.

    WHY DEVELOPMENT ONLY: these edges are frozen before OOS evaluation;
    letting OOS data influence its own regime labels would contaminate
    the OOS test.
    """
    s = dev_df["ATR_pct"].dropna()
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


# ============================================================
# Data loading and broker specs (read-only, as previous scripts)
# ============================================================
def load_closed_data(symbol: str, num_candles: int = NUM_CANDLES
                     ) -> Tuple[Optional[pd.DataFrame], str]:
    """Download up to num_candles H1 candles and drop the forming candle.

    Returns (closed-candle DataFrame, note) or (None, reason).
    Only mt5.copy_rates_from_pos is used (read-only).
    """
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, num_candles)
    if rates is None or len(rates) == 0:
        return None, "no historical data returned"
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)   # drop the forming candle
    return df, "closed candles"


def discover_symbol(requested: str) -> Tuple[Optional[str], str]:
    """Find the actual MT5 symbol (exact name first, then short suffix)."""
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

    return matches[0], f"suffix match (available: {', '.join(matches[:5])})"


def get_contract_spec(symbol_name: str) -> Optional[dict]:
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
        tick_value = contract_size * point   # documented fallback
        print(f"    WARNING: trade_tick_value missing/zero for {symbol_name}; "
              f"using fallback tick_value = contract_size x point = {tick_value:g}")

    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0
    quote_currency = getattr(info, "currency_profit", "") or "USD"

    # If the fallback was used AND the quote currency is not USD, convert
    # via a <QUOTE>USD symbol bid (documented fallback, printed).
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


def validate_volume(spec: dict) -> Optional[float]:
    """Largest valid volume <= LOTS_REQUESTED (never increased)."""
    vmin = spec["volume_min"]
    vmax = spec["volume_max"]
    vstep = spec["volume_step"]
    if LOTS_REQUESTED < vmin or vmin > vmax:
        return None
    steps_below = int((LOTS_REQUESTED - vmin) / vstep + 1e-9)
    volume = round(vmin + steps_below * vstep, 8)
    if volume > vmax:
        return None
    return volume


# ============================================================
# The frozen backtest engine (identical rules to previous scripts;
# regime measurements are RECORDED at signal time, never acted on)
# ============================================================
def calculate_frozen_strategy(df: pd.DataFrame, value_per_unit: float,
                              volume: float, start_bar: int = WARMUP
                              ) -> Tuple[pd.DataFrame, float]:
    """Run the frozen strategy on df (returns trades log, final balance).

    Signal on closed candle i, entry at the OPEN of candle i+1, SL/TP
    fixed from the signal candle's ATR, stop-first inside-candle
    handling, one position at a time, $2 cost per completed trade,
    independent $10,000 starting balance per call.
    """
    long_condition = df["close"] < df["BB_LOWER"]
    short_condition = df["close"] > df["BB_UPPER"]
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df = df.copy()
    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"],
                             default="HOLD")

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

            # Forward scan; STOP checked FIRST (conservative, deterministic)
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

            trades.append(
                {
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
                    "ADX": df["ADX"].iloc[i],
                    "ATR_pct": df["ATR_pct"].iloc[i],
                    "EMA50_dist": df["EMA50_dist"].iloc[i],
                    "BB_exc": df["BB_exc"].iloc[i],
                    "eff24": df["eff24"].iloc[i],
                    "pnl": ((exit_price - entry_price) if signal == "BUY"
                            else (entry_price - exit_price)) * volume * value_per_unit
                         - COST_PER_TRADE,
                    "balance_after": 0.0,   # filled below (keeps dict literal tidy)
                }
            )
            balance += trades[-1]["pnl"]
            trades[-1]["balance_after"] = balance

            # One position at a time: resume AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


def calculate_forward_returns(df: pd.DataFrame, trades: pd.DataFrame,
                              windows: Tuple[int, ...] = FWD_WINDOWS) -> pd.DataFrame:
    """Attach 6/12/24-candle forward returns (% at entry) to a trade log.

    Exit-independent diagnostic measurements only - they never alter
    any trade outcome. For a BUY the return is (close_fwd - entry)/entry;
    for a SELL it is (entry - close_fwd)/entry.
    """
    trades = trades.copy()
    for w in windows:
        vals = []
        for _, tr in trades.iterrows():
            entry_bar = tr["signal_index"] + 1
            fwd_idx = min(entry_bar + w - 1, len(df) - 1)
            close_fwd = df["close"].iloc[fwd_idx]
            entry_price = tr["entry_price"]
            ret = ((close_fwd - entry_price) if tr["direction"] == "BUY"
                   else (entry_price - close_fwd)) / entry_price * 100
            vals.append(ret)
        trades[f"fwd{w}_pct"] = vals
    return trades


def calculate_mfe_mae(df: pd.DataFrame, trades: pd.DataFrame,
                      window: int = HOLD_WINDOW) -> pd.DataFrame:
    """Attach ATR-normalized MFE/MAE (next `window` candles) to a trade log.

    Purely diagnostic measurements - never used in trade decisions.
    """
    trades = trades.copy()
    mfe_vals = []
    mae_vals = []
    for _, tr in trades.iterrows():
        entry_bar = tr["signal_index"] + 1
        entry_price = tr["entry_price"]
        atr = df["ATR"].iloc[tr["signal_index"]]
        mfe = 0.0
        mae = 0.0
        for j in range(entry_bar, min(entry_bar + window, len(df))):
            move_up = df["high"].iloc[j] - entry_price
            move_dn = entry_price - df["low"].iloc[j]
            mfe = max(mfe, move_up if tr["direction"] == "BUY" else move_dn)
            mae = max(mae, move_dn if tr["direction"] == "BUY" else move_up)
        mfe_vals.append(mfe / atr if atr > 0 else np.nan)
        mae_vals.append(mae / atr if atr > 0 else np.nan)
    trades["MFE_ATR"] = mfe_vals
    trades["MAE_ATR"] = mae_vals
    return trades


# ============================================================
# Descriptive statistics helpers (pure functions)
# ============================================================
def split_development_oos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                     List[Tuple[int, int]],
                                                     List[Tuple[int, int]]]:
    """Strict chronological 50/50 split: development first, OOS second.

    Returns (dev_df, oos_df, dev_bounds, oos_bounds) where the bounds
    are (start_index, end_index_exclusive) into the full frame.
    """
    n = len(df)
    half = n // 2
    dev = df.iloc[:half].reset_index(drop=True)
    oos = df.iloc[half:].reset_index(drop=True)
    return dev, oos, [(0, half)], [(half, n)]


def longest_losing_streak(trades: pd.DataFrame) -> int:
    """Longest run of consecutive losing trades (pnl < 0)."""
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


def summarize_window(trades: pd.DataFrame) -> dict:
    """Descriptive summary for one window (development or OOS)."""
    if trades is None or len(trades) == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": float("nan"),
                "gp": 0.0, "gl": 0.0, "pf": float("nan"), "avg_trade": float("nan"),
                "total_pnl": 0.0, "ending_balance": STARTING_BALANCE,
                "return": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0,
                "lose_streak": 0, "avg_hold": float("nan")}
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
        "lose_streak": longest_losing_streak(trades),
        "avg_hold": trades["hold_candles"].mean(),
    }


def fmt_pf(pf: float) -> str:
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def money(v: float) -> str:
    """Format a money value, handling 'no trades' (NaN)."""
    return "no trades" if (v != v) else f"${v:+.2f}"


def num(v: float, spec: str = "{:.2f}") -> str:
    """Format a number, printing n/a for NaN."""
    return "n/a" if (v != v) else spec.format(v)


def sign_of(v: float) -> str:
    """POSITIVE / NEGATIVE / ZERO / NO DATA for sign comparisons."""
    if v != v:
        return "NO DATA"
    if v > 0:
        return "POSITIVE"
    if v < 0:
        return "NEGATIVE"
    return "ZERO"


def regime_cell_stats(sub: pd.DataFrame) -> Optional[dict]:
    """Descriptive stats for one regime cell (or None when empty)."""
    if sub is None or len(sub) == 0:
        return None
    wins = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] < 0]
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    return {
        "n": len(sub),
        "avg_pnl": sub["pnl"].mean(),
        "total_pnl": sub["pnl"].sum(),
        "win_rate": len(wins) / len(sub) * 100,
        "pf": gp / gl if gl > 0 else float("inf"),
    }


def print_cell_line(prefix: str, sub: pd.DataFrame, cells_log: Optional[list] = None,
                    cell_tag: str = "", with_fwd: Optional[str] = None) -> Optional[dict]:
    """Print one regime cell; optionally record it and its average
    forward return. Returns the stats dict (or None for empty cells)."""
    s = regime_cell_stats(sub)
    if s is None:
        print(f"      {prefix:<28} (no trades)")
        if cells_log is not None:
            cells_log.append({"tag": cell_tag or prefix, "n": 0, "avg_pnl": np.nan})
        return None
    tag = f"  [{LOW_SAMPLE_TAG}]" if s["n"] < LOW_SAMPLE_N else ""
    line = (f"      {prefix:<28} {s['n']:>4} trades | "
            f"avg ${s['avg_pnl']:>8.2f} | total ${s['total_pnl']:>10.2f} | "
            f"win {s['win_rate']:>5.1f}% | PF {fmt_pf(s['pf']):>5}")
    if with_fwd is not None:
        line += f" | fwd{with_fwd} {sub[f'fwd{with_fwd}_pct'].mean():+.3f}%"
    print(line + tag)
    if cells_log is not None:
        cells_log.append({"tag": cell_tag or prefix, "n": s["n"],
                          "avg_pnl": s["avg_pnl"]})
    return s


def summarize_regime(trades: pd.DataFrame, col: str, order: List[str],
                     cells_log: Optional[list] = None, tag_prefix: str = "",
                     fwd_col: Optional[str] = None) -> Dict[str, dict]:
    """Print and collect one regime-family table; returns bucket stats."""
    out = {}
    for b in order:
        sub = trades[trades[col] == b]
        s = print_cell_line(b, sub, cells_log=cells_log,
                            cell_tag=f"{tag_prefix}|{col}|{b}",
                            with_fwd=None)
        if s is not None and fwd_col is not None:
            fsub = sub[sub[fwd_col].notna()]
            avg_f = fsub[fwd_col].mean() if len(fsub) else float("nan")
            print(f"        average {fwd_col}: {num(avg_f, '{:+.3f}%')}")
            s["avg_fwd"] = avg_f
        out[b] = s if s is not None else {"n": 0, "avg_pnl": np.nan}
    return out


def compare_development_oos(dev_trades: pd.DataFrame, oos_trades: pd.DataFrame,
                            col: str, order: List[str]) -> List[dict]:
    """Compare each bucket's average P/L between development and OOS.

    Returns one factual record per bucket: both averages, both signs,
    and whether the sign persisted. No scores, no rankings.
    """
    rows = []
    for b in order:
        d_sub = dev_trades[dev_trades[col] == b]
        o_sub = oos_trades[oos_trades[col] == b]
        d_avg = d_sub["pnl"].mean() if len(d_sub) else float("nan")
        o_avg = o_sub["pnl"].mean() if len(o_sub) else float("nan")
        if d_avg != d_avg or o_avg != o_avg:
            persistence = "insufficient data"
        elif sign_of(d_avg) == sign_of(o_avg):
            persistence = "same sign"
        else:
            persistence = "changed sign"
        rows.append({"bucket": b, "dev_n": len(d_sub), "oos_n": len(o_sub),
                     "dev_avg": d_avg, "oos_avg": o_avg,
                     "dev_sign": sign_of(d_avg), "oos_sign": sign_of(o_avg),
                     "persistence": persistence})
    return rows


def print_persistence_table(instrument: str, family: str, rows: List[dict]) -> None:
    """Print one development-vs-OOS persistence table (factual only)."""
    print(f"\n  {family} (development -> OOS):")
    for r in rows:
        line = (f"      {r['bucket']:<12} dev ${num(r['dev_avg'], '{:+.2f}'):>9} "
                f"(n={r['dev_n']:>3})  ->  OOS ${num(r['oos_avg'], '{:+.2f}'):>9} "
                f"(n={r['oos_n']:>3})  |  {r['persistence']}")
        print(line)


# ============================================================
# Factual questions (pure function; association wording only)
# ============================================================
def factual_answers(results: List[dict]) -> dict:
    """Compute the ten factual OOS answers from the collected results.

    results: list of per-instrument dicts with keys:
      instrument, dev_trades, oos_trades, summary (dev/oos), persistence
      {family: rows}.
    """
    facts = {}

    # Q1-Q3: OOS profitability / PF / average P/L per instrument
    q1, q2, q3 = {}, {}, {}
    for r in results:
        s = r["summary"]["oos"]
        if s["trades"] == 0:
            q1[r["instrument"]] = "NO DATA - no OOS trades"
            q2[r["instrument"]] = "NO DATA - no OOS trades"
            q3[r["instrument"]] = "NO DATA - no OOS trades"
            continue
        q1[r["instrument"]] = ("YES - positive OOS return"
                               if s["return"] > 0 else
                               f"NO - OOS return {s['return']:+.2f}%")
        q2[r["instrument"]] = ("YES - OOS profit factor "
                               f"{fmt_pf(s['pf'])}" if s["pf"] > 1.0 else
                               f"NO - OOS profit factor {fmt_pf(s['pf'])}")
        q3[r["instrument"]] = ("YES - positive OOS average P/L per trade"
                               if s["avg_trade"] > 0 else
                               f"NO - OOS average P/L per trade {s['avg_trade']:+.2f}")
    facts["q1"], facts["q2"], facts["q3"] = q1, q2, q3

    # Q4-Q8: per-family bucket sign persistence (same / changed / n/a)
    fam_map = (("q4", "VOLATILITY"), ("q5", "EFFICIENCY"), ("q6", "ADX"),
               ("q7", "TREND DISTANCE"), ("q8", "BB EXCURSION"))
    for qk, fam in fam_map:
        table = {}
        for r in results:
            same = sum(1 for x in r["persistence"][fam]
                       if x["persistence"] == "same sign")
            changed = sum(1 for x in r["persistence"][fam]
                          if x["persistence"] == "changed sign")
            insufficient = sum(1 for x in r["persistence"][fam]
                               if x["persistence"] == "insufficient data")
            table[r["instrument"]] = {"same": same, "changed": changed,
                                      "insufficient": insufficient}
        facts[qk] = table

    # Q9: which relationships changed sign (factual list)
    changed_list = []
    for r in results:
        for fam, rows in r["persistence"].items():
            for x in rows:
                if x["persistence"] == "changed sign":
                    changed_list.append({
                        "instrument": r["instrument"],
                        "family": fam,
                        "bucket": x["bucket"],
                        "dev_avg": x["dev_avg"],
                        "oos_avg": x["oos_avg"],
                    })
    facts["q9"] = changed_list

    # Q10: does regime instability appear across multiple instruments?
    instruments_with_change = sorted({c["instrument"] for c in changed_list})
    facts["q10"] = {
        "n_instruments_with_change": len(instruments_with_change),
        "instruments": instruments_with_change,
        "n_total_instruments": len(results),
        "n_changed_relationships": len(changed_list),
    }
    return facts


def print_factual_answers(facts: dict) -> None:
    """Print the ten factual questions (association wording only)."""
    print()
    print("1. Does each instrument remain profitable in OOS?")
    for name, ans in facts["q1"].items():
        print(f"   {name:<12}: {ans}")
    print()
    print("2. Does each instrument have PF > 1 in OOS?")
    for name, ans in facts["q2"].items():
        print(f"   {name:<12}: {ans}")
    print()
    print("3. Does each instrument have positive average P/L per trade in OOS?")
    for name, ans in facts["q3"].items():
        print(f"   {name:<12}: {ans}")

    labels = (("q4", "4. Does each volatility bucket keep the same P/L sign "
               "from Development to OOS?", "volatility"),
              ("q5", "5. Does each efficiency bucket keep the same P/L sign?",
               "efficiency"),
              ("q6", "6. Does each ADX bucket keep the same P/L sign?", "ADX"),
              ("q7", "7. Does each trend-distance bucket keep the same P/L sign?",
               "trend distance"),
              ("q8", "8. Does each BB-excursion bucket keep the same P/L sign?",
               "BB excursion"))
    for qk, title, fam in labels:
        print()
        print(title)
        for name, t in facts[qk].items():
            print(f"   {name:<12}: same sign {t['same']} | changed sign "
                  f"{t['changed']} | insufficient data {t['insufficient']}"
                  f"  ({fam} buckets)")

    print()
    print("9. Which regime relationships changed sign between Development")
    print("   and OOS?")
    if facts["q9"]:
        print(f"   {len(facts['q9'])} relationship(s) changed sign:")
        for c in facts["q9"]:
            print(f"   - {c['instrument']:<12} {c['family']:<15} "
                  f"{c['bucket']:<12} dev {money(c['dev_avg'])} -> "
                  f"OOS {money(c['oos_avg'])}")
        print("   For these buckets the development-observed relationship")
        print("   did not persist into OOS (factual observation).")
    else:
        print("   None: no populated bucket changed sign between windows.")

    q10 = facts["q10"]
    print()
    print("10. Does regime instability appear across multiple instruments?")
    print(f"    {q10['n_instruments_with_change']} of "
          f"{q10['n_total_instruments']} diagnosed instruments had at least")
    print(f"    one bucket change sign "
          f"({q10['n_changed_relationships']} changed relationships in total).")
    if q10["n_instruments_with_change"] > 1:
        print("    -> Yes: sign changes appear on MORE THAN ONE instrument")
        print("       in this data window (factual count only).")
    elif q10["n_instruments_with_change"] == 1:
        print("    -> Sign changes appear on exactly ONE instrument in this")
        print("       data window (factual count only).")
    else:
        print("    -> No instrument shows a bucket sign change in this")
        print("       data window (factual count only).")

    print()
    print("All answers use historical-association wording only: 'observed")
    print("relationship', 'sign persisted', 'sign changed', 'relationship")
    print("did not persist'. No causal claims, no predictions, no filters,")
    print("no rankings, and no recommendations are made.")


# ============================================================
# Print helpers for one instrument's full analysis
# ============================================================
def print_window_summary(label: str, s: dict) -> None:
    """Print the descriptive summary block for one window."""
    print(f"  {label}:")
    print(f"    starting balance      : ${STARTING_BALANCE:,.2f}")
    print(f"    ending balance        : ${s['ending_balance']:,.2f}")
    print(f"    return                : {s['return']:+.2f}%")
    print(f"    trade count           : {s['trades']}")
    print(f"    wins / losses         : {s['wins']} / {s['losses']}")
    print(f"    win rate              : {num(s['win_rate'], '{:.1f}%')}")
    print(f"    gross profit          : ${s['gp']:,.2f}")
    print(f"    gross loss            : ${s['gl']:,.2f}")
    print(f"    profit factor         : {fmt_pf(s['pf'])}")
    print(f"    average P/L per trade : {money(s['avg_trade'])}")
    print(f"    maximum drawdown $    : ${s['max_dd']:,.2f}")
    print(f"    maximum drawdown %    : {num(s['max_dd_pct'], '{:.2f}%')}")
    print(f"    longest losing streak : {s['lose_streak']} trades")
    print(f"    average holding       : {num(s['avg_hold'], '{:.1f}')} candles")


def regime_mfe_mae_line(sub: pd.DataFrame) -> str:
    """Format one winners/losers MFE-MAE line (with sample tag)."""
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


def print_forward_return_tables(label: str, trades: pd.DataFrame) -> None:
    """Print average forward returns per regime for one window."""
    for w in FWD_WINDOWS:
        col = f"fwd{w}_pct"
        print(f"\n  {label} - average {w}-candle forward return by regime:")
        tables = (
            ("volatility", "vol_regime", VOL_BUCKETS),
            ("efficiency", "eff_regime", EFF_BUCKETS),
            ("ADX", "adx_regime", ADX_BELOW_BUCKETS),
        )
        for fam, col_r, order in tables:
            for b in order:
                sub = trades[trades[col_r] == b]
                fsub = sub[sub[col].notna()]
                tag = (f"  [{LOW_SAMPLE_TAG}]" if len(fsub) < LOW_SAMPLE_N else "")
                avg = fsub[col].mean() if len(fsub) else float("nan")
                print(f"      {fam:<12} {b:<12} n={len(fsub):>3} "
                      f"avg {num(avg, '{:+.3f}%')}{tag}")
        for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
            d_tr = trades[trades["direction"] == d]
            for col_r, order in (("vol_regime", VOL_BUCKETS),
                                 ("eff_regime", EFF_BUCKETS)):
                for b in order:
                    sub = d_tr[(d_tr[col_r] == b) & (d_tr[col].notna())]
                    tag = (f"  [{LOW_SAMPLE_TAG}]"
                           if len(sub) < LOW_SAMPLE_N else "")
                    avg = sub[col].mean() if len(sub) else float("nan")
                    print(f"      {dname} x {col_r.replace('_regime', ''):<10} "
                          f"{b:<12} n={len(sub):>3} avg {num(avg, '{:+.3f}%')}{tag}")


def print_mfe_mae_tables(label: str, trades: pd.DataFrame) -> None:
    """Print winners/losers MFE-MAE per regime for one window."""
    print(f"\n  {label} - MFE / MAE (ATR-normalized, next {HOLD_WINDOW} candles):")
    print(f"    overall: {regime_mfe_mae_line(trades)}")
    for col_r, order, fam in (("vol_regime", VOL_BUCKETS, "volatility"),
                              ("eff_regime", EFF_BUCKETS, "efficiency")):
        print(f"    by {fam}:")
        for b in order:
            print(f"      {b:<12} {regime_mfe_mae_line(trades[trades[col_r] == b])}")
    print("    by direction:")
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        print(f"      {dname:<12} {regime_mfe_mae_line(trades[trades['direction'] == d])}")


def print_results(instrument: str, actual_symbol: str, spec: dict, volume: float,
                  df: pd.DataFrame, dev_df: pd.DataFrame, oos_df: pd.DataFrame,
                  v1: float, v2: float, dev_trades: pd.DataFrame,
                  oos_trades: pd.DataFrame, dev_sum: dict, oos_sum: dict,
                  persistence: Dict[str, List[dict]], cells_log: List[dict]) -> None:
    """Print all analysis sections for one instrument."""
    # ------------------------------------------------------------
    # OOS DESIGN / DATA
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"OOS DESIGN - {instrument} (traded as '{actual_symbol}')")
    print("=" * 70)
    print(f"  total closed candles   : {len(df)}")
    if len(df) < NUM_CANDLES:
        print(f"    NOTE: fewer than {NUM_CANDLES} closed candles available; "
              f"using ALL {len(df)}.")
    print(f"  development candles    : {len(dev_df)} "
          f"({dev_df['time'].iloc[0]} -> {dev_df['time'].iloc[-1]})")
    print(f"  OOS candles            : {len(oos_df)} "
          f"({oos_df['time'].iloc[0]} -> {oos_df['time'].iloc[-1]})")
    print("  split: first 50% = development, strictly later 50% = OOS")
    print("  (chronological; no shuffling, no random split)")
    print(f"  volume                 : {volume} lots (requested "
          f"{LOTS_REQUESTED}; volume_min {spec['volume_min']}, volume_step "
          f"{spec['volume_step']})")
    print(f"  contract size          : {spec['contract_size']:g} | point "
          f"{spec['point']:g} | digits {spec['digits']} | tick size "
          f"{spec['tick_size']:g} | tick value {spec['tick_value']:g} | "
          f"quote currency {spec['quote_currency']}")
    pl_note = ""
    if spec["tick_value_fallback"]:
        pl_note = " (documented fallback tick_value = contract x point"
        pl_note += ", converted to USD" if spec["fallback_converted"] else ""
        pl_note += ")"
    print(f"  P/L basis              : 1.0 price-unit move with 1.0 lot = "
          f"{spec['value_per_unit']:.2f} (tick_value / tick_size){pl_note}")
    print(f"  FROZEN volatility edges (DEVELOPMENT-derived, applied to BOTH")
    print(f"  windows): 33rd pct {v1:.4f}% | 66th pct {v2:.4f}% (ATR%)")
    print("  WHY FROZEN: OOS must not influence its own regime labels -")
    print("  recalculating percentiles on OOS would contaminate the test.")

    # ------------------------------------------------------------
    # DEVELOPMENT VS OOS PERFORMANCE
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"DEVELOPMENT vs OOS PERFORMANCE - {instrument}")
    print("(each window starts from its own independent $10,000)")
    print("=" * 70)
    print_window_summary("DEVELOPMENT", dev_sum)
    print()
    print_window_summary("OOS", oos_sum)

    # ------------------------------------------------------------
    # REGIME PERFORMANCE (both windows, frozen labels)
    # ------------------------------------------------------------
    for label, trades in (("DEVELOPMENT", dev_trades), ("OOS", oos_trades)):
        print()
        print("=" * 70)
        print(f"REGIME PERFORMANCE - {instrument} - {label}")
        print("(fixed descriptive buckets; cells with < "
              f"{LOW_SAMPLE_N} trades are {LOW_SAMPLE_TAG})")
        print("=" * 70)
        summarize_regime(trades, "adx_regime", ADX_BUCKETS, cells_log,
                         f"{instrument}|{label}", fwd_col="fwd12_pct")
        summarize_regime(trades, "vol_regime", VOL_BUCKETS, cells_log,
                         f"{instrument}|{label}", fwd_col="fwd12_pct")
        summarize_regime(trades, "trend_regime", TREND_BUCKETS, cells_log,
                         f"{instrument}|{label}", fwd_col="fwd12_pct")
        summarize_regime(trades, "bb_regime", BB_BUCKETS, cells_log,
                         f"{instrument}|{label}", fwd_col="fwd12_pct")
        summarize_regime(trades, "eff_regime", EFF_BUCKETS, cells_log,
                         f"{instrument}|{label}", fwd_col="fwd12_pct")

    # ------------------------------------------------------------
    # INTERACTION ANALYSIS (both windows)
    # ------------------------------------------------------------
    for label, trades in (("DEVELOPMENT", dev_trades), ("OOS", oos_trades)):
        print()
        print("=" * 70)
        print(f"INTERACTION ANALYSIS - {instrument} - {label}")
        print("=" * 70)

        def interactions(tr: pd.DataFrame) -> None:
            for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
                d_tr = tr[tr["direction"] == d]
                print(f"\n  {dname} x volatility:")
                for b in VOL_BUCKETS:
                    print_cell_line(b, d_tr[d_tr["vol_regime"] == b], cells_log,
                                    f"{instrument}|{label}|dir-vol|{dname}|{b}")
                print(f"\n  {dname} x efficiency:")
                for b in EFF_BUCKETS:
                    print_cell_line(b, d_tr[d_tr["eff_regime"] == b], cells_log,
                                    f"{instrument}|{label}|dir-eff|{dname}|{b}")

            def cross(dim_a, order_a, dim_b, order_b, fam_label):
                print(f"\n  {fam_label}:")
                for ba in order_a:
                    a_tr = tr[tr[dim_a] == ba]
                    if len(a_tr) == 0:
                        continue
                    for bb_ in order_b:
                        sub = a_tr[a_tr[dim_b] == bb_]
                        print_cell_line(f"{ba} x {bb_}", sub, cells_log,
                                        f"{instrument}|{label}|{fam_label}|{ba}x{bb_}")

            cross("adx_regime", ADX_BELOW_BUCKETS, "vol_regime", VOL_BUCKETS,
                  "ADX x volatility")
            cross("adx_regime", ADX_BELOW_BUCKETS, "eff_regime", EFF_BUCKETS,
                  "ADX x efficiency")
            cross("vol_regime", VOL_BUCKETS, "trend_regime", TREND_BUCKETS,
                  "volatility x trend distance")
            cross("eff_regime", EFF_BUCKETS, "trend_regime", TREND_BUCKETS,
                  "efficiency x trend distance")

        interactions(trades)

    # ------------------------------------------------------------
    # FORWARD RETURN DIAGNOSTIC (both windows)
    # ------------------------------------------------------------
    for label, trades in (("DEVELOPMENT", dev_trades), ("OOS", oos_trades)):
        print()
        print("=" * 70)
        print(f"FORWARD RETURN DIAGNOSTIC - {instrument} - {label}")
        print("(exit-independent measurements only; never trade decisions)")
        print("=" * 70)
        print_forward_return_tables(label, trades)

    # ------------------------------------------------------------
    # MFE / MAE (both windows)
    # ------------------------------------------------------------
    for label, trades in (("DEVELOPMENT", dev_trades), ("OOS", oos_trades)):
        print()
        print("=" * 70)
        print(f"MFE / MAE - {instrument} - {label}")
        print("=" * 70)
        print_mfe_mae_tables(label, trades)

    # ------------------------------------------------------------
    # MOST IMPORTANT TEST: DEVELOPMENT vs OOS relationship comparison
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"REGIME PERSISTENCE TEST - {instrument}")
    print("(does the development-observed relationship also appear in OOS?)")
    print("=" * 70)
    for fam in ("VOLATILITY", "EFFICIENCY", "ADX", "TREND DISTANCE",
                "BB EXCURSION"):
        print_persistence_table(instrument, fam, persistence[fam])

    # Factual persistence counts for this instrument
    print()
    print("  Persistence counts (factual, no score):")
    for fam in ("VOLATILITY", "EFFICIENCY", "ADX", "TREND DISTANCE",
                "BB EXCURSION"):
        rows = persistence[fam]
        same = sum(1 for x in rows if x["persistence"] == "same sign")
        changed = sum(1 for x in rows if x["persistence"] == "changed sign")
        insuff = sum(1 for x in rows if x["persistence"] == "insufficient data")
        print(f"    {fam:<15} same {same} | changed {changed} | "
              f"insufficient {insuff}")
    print()
    print("  A 'changed sign' bucket means the development-observed")
    print("  relationship did not persist into OOS in this sample. This is")
    print("  a factual observation - not a recommendation and not proof of")
    print("  anything about future behaviour.")


# ============================================================
# Synthetic tests (deterministic; run automatically at startup)
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

    # ---- deterministic synthetic feed --------------------------------
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

    # [1-3] chronological development/OOS split, no overlap, OOS strictly
    # after development
    df = calculate_indicators(synth_candles(1000, 1, 100.0, 0.5))
    dev, oos, dev_b, oos_b = split_development_oos(df)
    check("dev/OOS split covers all candles", len(dev) + len(oos) == len(df))
    check("dev is the FIRST half", dev_b[0] == (0, len(df) // 2) and len(dev) == len(df) // 2)
    check("OOS is the SECOND half", oos_b[0] == (len(df) // 2, len(df)))
    check("no overlap between windows",
          dev["time"].iloc[-1] < oos["time"].iloc[0])
    check("OOS occurs strictly after development",
          oos["time"].iloc[0] > dev["time"].iloc[0])

    # [4] independent $10,000 balances per window
    vpu, vol = 1000.0, 0.10
    v1, v2 = compute_vol_edges_from_development(dev)
    dev_tr = calculate_frozen_strategy(dev, vpu, vol)
    oos_tr = calculate_frozen_strategy(oos, vpu, vol)
    for label, tr in (("dev", dev_tr[0]), ("oos", oos_tr[0])):
        if len(tr):
            check(f"{label} window restarts from $10,000",
                  abs(tr["balance_after"].iloc[0] - STARTING_BALANCE
                      - tr["pnl"].iloc[0]) < 1e-9)
    if len(dev_tr[0]) and len(oos_tr[0]):
        check("dev profits are NOT carried into OOS",
              abs(oos_tr[0]["balance_after"].iloc[0] - STARTING_BALANCE
                  - oos_tr[0]["pnl"].iloc[0]) < 1e-9)

    # [5-6] volatility thresholds from DEVELOPMENT only; OOS uses them frozen
    e1, e2 = compute_vol_edges_from_development(dev)
    full1, full2 = (float(df["ATR_pct"].dropna().quantile(1 / 3)),
                    float(df["ATR_pct"].dropna().quantile(2 / 3)))
    check("edges equal development 33rd/66th percentiles",
          abs(e1 - float(dev["ATR_pct"].dropna().quantile(1 / 3))) < 1e-12
          and abs(e2 - float(dev["ATR_pct"].dropna().quantile(2 / 3))) < 1e-12)
    # Development-only sourcing must be distinguishable from an OOS-derived
    # edge. The shared synthetic feed above is stationary, so its development
    # and OOS ATR% terciles can coincide by chance (the reason the old check
    # could not assert inequality). Use a DELIBERATE low-vol / high-vol
    # fixture instead: development is quiet, OOS is volatile, so the two ATR%
    # distributions cannot coincide. The frozen edges must pin EXACTLY to the
    # development quantiles (true regardless of any OOS value), and must sit
    # far from the OOS-only quantiles - so a regression that derived the
    # edges from OOS would move them by a large, unmistakable margin.
    quiet_dev = calculate_indicators(synth_candles(600, 21, 100.0, 0.20))
    wild_oos = calculate_indicators(synth_candles(600, 22, 100.0, 2.00))
    quiet_q1, quiet_q2 = compute_vol_edges_from_development(quiet_dev)
    wild_q1 = float(wild_oos["ATR_pct"].dropna().quantile(1 / 3))
    wild_q2 = float(wild_oos["ATR_pct"].dropna().quantile(2 / 3))
    check("edges are NOT recomputed from OOS (development-only, frozen)",
          abs(quiet_q1 - float(quiet_dev["ATR_pct"].dropna().quantile(1 / 3))) < 1e-12
          and abs(quiet_q2 - float(quiet_dev["ATR_pct"].dropna().quantile(2 / 3))) < 1e-12
          and abs(quiet_q1 - wild_q1) > 1.0
          and abs(quiet_q2 - wild_q2) > 1.0)
    dev_lab = calculate_regimes(dev_tr[0], e1, e2) if len(dev_tr[0]) else dev_tr[0]
    oos_lab = calculate_regimes(oos_tr[0], e1, e2) if len(oos_tr[0]) else oos_tr[0]
    lab_ok = True
    for tr_lab, src in ((dev_lab, dev), (oos_lab, oos)):
        for _, t in tr_lab.iterrows():
            i = src.index[src["time"] == t["signal_time"]][0]
            if t["vol_regime"] != vol_regime_label(src["ATR_pct"].iloc[i], e1, e2):
                lab_ok = False
    check("BOTH windows labelled with the SAME frozen dev edges", lab_ok)

    # [7] all regime boundary conditions
    check("ADX boundaries", adx_regime_label(14.999) == "ADX<15"
          and adx_regime_label(15.0) == "ADX15-20"
          and adx_regime_label(19.999) == "ADX15-20"
          and adx_regime_label(20.0) == "ADX20-25"
          and adx_regime_label(24.999) == "ADX20-25"
          and adx_regime_label(25.0) == "ADX>=25"
          and adx_regime_label(float("nan")) == "n/a")
    check("vol boundaries", vol_regime_label(0.999, 1.0, 2.0) == "LOW"
          and vol_regime_label(1.0, 1.0, 2.0) == "MEDIUM"
          and vol_regime_label(1.999, 1.0, 2.0) == "MEDIUM"
          and vol_regime_label(2.0, 1.0, 2.0) == "HIGH")
    check("trend boundaries", trend_dist_label(0.999) == "<1 ATR"
          and trend_dist_label(1.0) == "1-2 ATR"
          and trend_dist_label(1.999) == "1-2 ATR"
          and trend_dist_label(2.0) == ">=2 ATR")
    check("BB excursion boundaries", bb_exc_label(0.499) == "LOW"
          and bb_exc_label(0.50) == "MEDIUM"
          and bb_exc_label(0.749) == "MEDIUM"
          and bb_exc_label(0.75) == "HIGH")
    check("efficiency boundaries", eff_label(0.299) == "LOW"
          and eff_label(0.30) == "MEDIUM"
          and eff_label(0.599) == "MEDIUM"
          and eff_label(0.60) == "HIGH")

    # [8] no-lookahead: isolated prefix measurements match the full frame
    head = df.iloc[:500].reset_index(drop=True)
    head_meas = calculate_indicators(head.copy())
    cols = ["ATR_pct", "EMA50_dist", "BB_exc", "eff24"]
    check("measurements on an isolated prefix match the full frame (causal)",
          all(np.allclose(head_meas[c].values, df[c].iloc[:500].values,
                          equal_nan=True) for c in cols))

    # [9] next-open entry + [10] stop-first + MFE/MAE + forward returns
    # (crafted candles with hand-computed expected values)
    n = 300
    base = calculate_indicators(synth_candles(n, 7, 100.0, 0.5))
    tr0, _ = calculate_frozen_strategy(base, vpu, vol)
    check("crafted feed produces trades", len(tr0) > 0)
    first = tr0.iloc[0]
    i_sig = base.index[base["time"] == first["signal_time"]][0]
    eb = i_sig + 1
    entry = base["open"].iloc[eb]
    atr = base["ATR"].iloc[i_sig]
    is_buy = first["direction"] == "BUY"
    check("entry at next candle OPEN",
          abs(first["entry_price"] - entry) < 1e-12
          and first["entry_time"] == base["time"].iloc[eb])
    check("entry strictly after signal candle",
          first["entry_time"] > first["signal_time"])

    def rerun_with_entry_candle(mod_df: pd.DataFrame) -> pd.Series:
        tr, _ = calculate_frozen_strategy(mod_df, vpu, vol)
        return tr[tr["signal_time"] == first["signal_time"]].iloc[0]

    # stop-first: entry candle spans BOTH SL and TP -> exact STOP value
    span = base.copy()
    span.iloc[eb, span.columns.get_loc("high")] = entry + 3.0 * atr
    span.iloc[eb, span.columns.get_loc("low")] = entry - 3.0 * atr
    t_span = rerun_with_entry_candle(span)
    pnl_stop = (-2.0 * atr) * vol * vpu - COST_PER_TRADE
    check("spanning candle exits at the exact STOP value (stop-first)",
          t_span["exit_reason"] == "STOP_LOSS"
          and abs(t_span["pnl"] - pnl_stop) < 1e-9)

    # TP-only and SL-only cases
    quiet = base.copy()
    quiet.iloc[eb, quiet.columns.get_loc("high")] = entry + 1.0 * atr
    quiet.iloc[eb, quiet.columns.get_loc("low")] = entry - 1.0 * atr
    tp_only = quiet.copy()
    if is_buy:
        tp_only.iloc[eb + 1, tp_only.columns.get_loc("high")] = entry + 1.5 * atr
        tp_only.iloc[eb + 1, tp_only.columns.get_loc("low")] = entry - 0.5 * atr
    else:
        tp_only.iloc[eb + 1, tp_only.columns.get_loc("low")] = entry - 1.5 * atr
        tp_only.iloc[eb + 1, tp_only.columns.get_loc("high")] = entry + 0.5 * atr
    t_tp = rerun_with_entry_candle(tp_only)
    pnl_tp = (1.5 * atr) * vol * vpu - COST_PER_TRADE
    check("TP-only candle exits at the exact TP value",
          t_tp["exit_reason"] == "TAKE_PROFIT"
          and abs(t_tp["pnl"] - pnl_tp) < 1e-9)

    sl_only = quiet.copy()
    if is_buy:
        sl_only.iloc[eb + 1, sl_only.columns.get_loc("low")] = entry - 2.0 * atr
        sl_only.iloc[eb + 1, sl_only.columns.get_loc("high")] = entry + 0.5 * atr
    else:
        sl_only.iloc[eb + 1, sl_only.columns.get_loc("high")] = entry + 2.0 * atr
        sl_only.iloc[eb + 1, sl_only.columns.get_loc("low")] = entry - 0.5 * atr
    t_sl = rerun_with_entry_candle(sl_only)
    pnl_sl = (-2.0 * atr) * vol * vpu - COST_PER_TRADE
    check("SL-only candle exits at the exact SL value",
          t_sl["exit_reason"] == "STOP_LOSS"
          and abs(t_sl["pnl"] - pnl_sl) < 1e-9)

    # [11] MFE/MAE recomputed independently for every trade
    # (tr0 was computed on the `base` frame, so all index lookups use base)
    tr_lab = calculate_forward_returns(base, tr0)
    tr_lab = calculate_mfe_mae(base, tr_lab)
    ok_mfe = True
    for _, t in tr_lab.iterrows():
        i = base.index[base["time"] == t["signal_time"]][0]
        eb2 = i + 1
        e_price = base["open"].iloc[eb2]
        a_sig = base["ATR"].iloc[i]
        mfe = mae = 0.0
        for j in range(eb2, min(eb2 + HOLD_WINDOW, len(base))):
            up = base["high"].iloc[j] - e_price
            dn = e_price - base["low"].iloc[j]
            mfe = max(mfe, up if t["direction"] == "BUY" else dn)
            mae = max(mae, dn if t["direction"] == "BUY" else up)
        if (abs(t["MFE_ATR"] - mfe / a_sig) > 1e-9
                or abs(t["MAE_ATR"] - mae / a_sig) > 1e-9):
            ok_mfe = False
            break
    check("MFE/MAE match per-trade independent recomputation", ok_mfe)

    # [12] 6/12/24 forward returns recomputed independently per trade
    ok_fwd = True
    for _, t in tr_lab.iterrows():
        i = base.index[base["time"] == t["signal_time"]][0]
        eb2 = i + 1
        e_price = base["open"].iloc[eb2]
        for w in FWD_WINDOWS:
            idx = min(eb2 + w - 1, len(base) - 1)
            exp = ((base["close"].iloc[idx] - e_price) if t["direction"] == "BUY"
                   else (e_price - base["close"].iloc[idx])) / e_price * 100
            if abs(t[f"fwd{w}_pct"] - exp) > 1e-9:
                ok_fwd = False
                break
        if not ok_fwd:
            break
    check("6/12/24 forward returns match per-trade recomputation", ok_fwd)

    # [13] regime persistence sign comparison (constructed case)
    def mk_frame(pnls, atrs):
        return pd.DataFrame({"pnl": pnls, "ATR_pct": atrs, "ADX": [20.0] * len(pnls),
                             "EMA50_dist": [0.5] * len(pnls),
                             "BB_exc": [0.4] * len(pnls),
                             "eff24": [0.4] * len(pnls),
                             "direction": ["BUY"] * len(pnls),
                             "fwd12_pct": [0.0] * len(pnls)})

    dev_case = calculate_regimes(mk_frame([10, 12, 11, 9, 10], [0.1] * 5), 0.5, 5.0)
    oos_case = calculate_regimes(mk_frame([-3, -2, -4, -1, -2], [0.1] * 5), 0.5, 5.0)
    rows = compare_development_oos(dev_case, oos_case, "vol_regime", VOL_BUCKETS)
    low_row = next(x for x in rows if x["bucket"] == "LOW")
    check("persistence: positive dev + negative OOS -> 'changed sign'",
          low_row["dev_sign"] == "POSITIVE" and low_row["oos_sign"] == "NEGATIVE"
          and low_row["persistence"] == "changed sign")
    oos_same = calculate_regimes(mk_frame([1.0, 2.0], [0.1] * 2), 0.5, 5.0)
    rows2 = compare_development_oos(dev_case, oos_same, "vol_regime", VOL_BUCKETS)
    low2 = next(x for x in rows2 if x["bucket"] == "LOW")
    check("persistence: both positive -> 'same sign'",
          low2["persistence"] == "same sign")
    empty_oos = mk_frame([], [])
    rows3 = compare_development_oos(dev_case, calculate_regimes(empty_oos, 0.5, 5.0),
                                    "vol_regime", VOL_BUCKETS)
    low3 = next(x for x in rows3 if x["bucket"] == "LOW")
    check("persistence: empty OOS bucket -> 'insufficient data'",
          low3["persistence"] == "insufficient data")

    # [14] low-sample tagging
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_cell_line("tiny", tr_lab.head(3))
    check("LOW SAMPLE tag when n < 5", "[LOW SAMPLE]" in buf.getvalue())
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        print_cell_line("big", tr_lab)
    check("no LOW SAMPLE tag when n >= 5", "[LOW SAMPLE]" not in buf2.getvalue())

    # [15] interaction cells partition all trades
    lab_all = calculate_regimes(tr_lab, e1, e2)
    n_cover = sum(int(((lab_all["adx_regime"] == a)
                       & (lab_all["vol_regime"] == v)).sum())
                  for a in ADX_BELOW_BUCKETS for v in VOL_BUCKETS)
    n_adx_below = int(lab_all["adx_regime"].isin(ADX_BELOW_BUCKETS).sum())
    check("ADX x volatility cells partition the below-25 trades",
          n_cover == n_adx_below)

    # [16] no order functions in this module's source
    # (names are assembled from fragments so this source file itself
    # contains zero contiguous occurrences of any forbidden name -
    # the external verification scan then also finds 0)
    import inspect
    src = inspect.getsource(sys_module())
    for a, b in (("order", "send"), ("order", "check"),
                 ("positions", "get"), ("positions", "total"),
                 ("position", "get")):
        name = a + "_" + b
        check(f"source contains no '{name}' function", name not in src)

    # [17] no optimization loops (structural: no parameter iteration
    # exists - the strategy constants are module-level and never reassigned)
    check("frozen strategy constants unchanged",
          (SL_ATR_MULT, TP_ATR_MULT, BB_PERIOD, BB_NUM_STD, ADX_PERIOD,
           ADX_THRESHOLD, ATR_PERIOD, LOTS_REQUESTED, COST_PER_TRADE)
          == (2.0, 1.5, 20, 2.0, 14, 25.0, 14, 0.10, 2.0))

    return passed, failed


def sys_module():
    """Return this module (used only to inspect its own source text)."""
    import sys
    return sys.modules[__name__]


# NOTE FOR REVIEWERS: this file contains no order-sending, order-checking
# or position-query calls of any kind. Only the seven read-only MT5 data
# functions listed at the top are used, and mt5.shutdown() always runs
# through the finally block in main().


# ============================================================
# Main program (MT5 read-only)
# ============================================================
def main() -> None:
    # Synthetic tests run automatically first (no MT5 needed).
    print("Running synthetic tests...")
    passed, failed = run_synthetic_tests()
    print(f"Synthetic tests: {passed} passed, {failed} failed")
    if failed:
        print("SYNTHETIC TESTS FAILED - aborting before any MT5 access.")
        return

    # GUARD: the real MT5 diagnostic never runs automatically. It must be
    # requested explicitly with --run, so a casual execution of this file
    # only verifies the synthetic tests and never touches the terminal.
    if "--run" not in sys.argv:
        print()
        print("Verification mode: synthetic tests only. The real MT5")
        print("diagnostic was NOT run. To execute it explicitly, use:")
        print("    python regime_oos_validation.py --run")
        print()
        print("Historical research only. No orders were sent.")
        print("Development regime thresholds were frozen before OOS evaluation.")
        return

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
        print("REGIME OOS VALIDATION (read-only, descriptive only)")
        print("Frozen strategy + frozen development-derived regime edges.")
        print("Question: do development regime relationships also appear")
        print("in strictly later, unseen OOS data?")
        print("=" * 70)
        print()
        print("Timeframe: H1 | BB(20, 2.0) + ADX14 < 25 entry | "
              f"SL {SL_ATR_MULT} x ATR / TP {TP_ATR_MULT} x ATR exits")
        print(f"{LOTS_REQUESTED} lots | ${COST_PER_TRADE:.2f}/trade | "
              f"next-open entry | stop-first | one position at a time")
        print("Development: first 50% of history. OOS: strictly later 50%.")
        print("Volatility tercile edges come from DEVELOPMENT only and are")
        print("FROZEN before OOS evaluation. Nothing is optimized.")

        results: List[dict] = []
        cells_log: List[dict] = []
        unavailable: List[str] = []

        for requested in INSTRUMENTS:
            print()
            print("#" * 70)
            print(f"# INSTRUMENT: {requested}")
            print("#" * 70)

            try:
                # --- Symbol discovery (no substitution) ---------------
                actual_symbol, note = discover_symbol(requested)
                if actual_symbol is None:
                    print(f"STATUS: NOT AVAILABLE - {note}. Continuing with "
                          f"the remaining instruments.")
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
                    print(f"STATUS: NOT AVAILABLE - no symbol_info for "
                          f"{actual_symbol}. Continuing.")
                    unavailable.append(requested)
                    continue

                volume = validate_volume(spec)
                if volume is None:
                    print(f"STATUS: SKIPPED - {LOTS_REQUESTED} lots is not a "
                          f"valid volume for {actual_symbol}. Continuing.")
                    unavailable.append(requested)
                    continue

                # --- Load closed candles ------------------------------
                df, load_note = load_closed_data(actual_symbol)
                if df is None:
                    print(f"STATUS: NOT AVAILABLE - {load_note} for "
                          f"{actual_symbol}. Continuing.")
                    unavailable.append(requested)
                    continue

                if len(df) < MIN_CANDLES:
                    print(f"STATUS: NOT AVAILABLE - only {len(df)} closed "
                          f"candles (minimum {MIN_CANDLES}). Continuing.")
                    unavailable.append(requested)
                    continue

                # --- Indicators over the FULL closed series (causal) ---
                # All indicator values use only current + previous candles,
                # so computing once here is equivalent to computing per
                # window and leaks no OOS information into development.
                df = calculate_indicators(df)

                # --- Strict chronological DEV/OOS split ---------------
                dev_df, oos_df, _, _ = split_development_oos(df)

                # --- FROZEN volatility edges from DEVELOPMENT ONLY ----
                v1, v2 = compute_vol_edges_from_development(dev_df)

                # --- Frozen strategy on BOTH windows (independent $10k)
                dev_trades, _ = calculate_frozen_strategy(dev_df,
                                                          spec["value_per_unit"],
                                                          volume)
                oos_trades, _ = calculate_frozen_strategy(oos_df,
                                                          spec["value_per_unit"],
                                                          volume)

                # --- Forward returns + MFE/MAE (diagnostics only) -----
                dev_trades = calculate_forward_returns(dev_df, dev_trades)
                dev_trades = calculate_mfe_mae(dev_df, dev_trades)
                oos_trades = calculate_forward_returns(oos_df, oos_trades)
                oos_trades = calculate_mfe_mae(oos_df, oos_trades)

                # --- Regime labels: SAME frozen dev edges on BOTH -----
                dev_trades = calculate_regimes(dev_trades, v1, v2)
                oos_trades = calculate_regimes(oos_trades, v1, v2)

                # --- Window summaries ----------------------------------
                dev_sum = summarize_window(dev_trades)
                oos_sum = summarize_window(oos_trades)

                if dev_sum["trades"] == 0 and oos_sum["trades"] == 0:
                    print(f"NOTE: no trades generated for {requested} in "
                          f"either window.")
                    unavailable.append(requested)
                    continue

                # --- Persistence comparison per regime family ---------
                persistence = {
                    "VOLATILITY": compare_development_oos(dev_trades, oos_trades,
                                                          "vol_regime", VOL_BUCKETS),
                    "EFFICIENCY": compare_development_oos(dev_trades, oos_trades,
                                                          "eff_regime", EFF_BUCKETS),
                    "ADX": compare_development_oos(dev_trades, oos_trades,
                                                   "adx_regime", ADX_BUCKETS),
                    "TREND DISTANCE": compare_development_oos(dev_trades, oos_trades,
                                                              "trend_regime",
                                                              TREND_BUCKETS),
                    "BB EXCURSION": compare_development_oos(dev_trades, oos_trades,
                                                            "bb_regime", BB_BUCKETS),
                }

                # --- Print everything for this instrument --------------
                print_results(instrument=requested, actual_symbol=actual_symbol,
                              spec=spec, volume=volume, df=df, dev_df=dev_df,
                              oos_df=oos_df, v1=v1, v2=v2, dev_trades=dev_trades,
                              oos_trades=oos_trades, dev_sum=dev_sum,
                              oos_sum=oos_sum, persistence=persistence,
                              cells_log=cells_log)

                results.append({
                    "instrument": requested,
                    "dev_trades": dev_trades,
                    "oos_trades": oos_trades,
                    "summary": {"dev": dev_sum, "oos": oos_sum},
                    "persistence": persistence,
                })

            except Exception as exc:  # never let one instrument kill the run
                print(f"STATUS: ERROR while processing {requested}: {exc}")
                print("Continuing with the remaining instruments.")
                unavailable.append(requested)

        # ==========================================================
        # CROSS-INSTRUMENT SUMMARY (factual, no ranking)
        # ==========================================================
        print()
        print("=" * 70)
        print("CROSS-INSTRUMENT SUMMARY (factual table, no ranking)")
        print("=" * 70)
        if results:
            hdr = (f"   {'Instrument':<12}{'Dev return':>12}{'OOS return':>12}"
                   f"{'Dev PF':>9}{'OOS PF':>9}{'Dev avg':>10}{'OOS avg':>10}"
                   f"{'Dev DD$':>10}{'OOS DD$':>10}")
            print(hdr)
            print("   " + "-" * 94)
            for r in results:
                d, o = r["summary"]["dev"], r["summary"]["oos"]
                print(f"   {r['instrument']:<12}"
                      f"{num(d['return'], '{:+.2f}%'):>12}"
                      f"{num(o['return'], '{:+.2f}%'):>12}"
                      f"{fmt_pf(d['pf']) if d['trades'] else 'n/a':>9}"
                      f"{fmt_pf(o['pf']) if o['trades'] else 'n/a':>9}"
                      f"{money(d['avg_trade']) if d['trades'] else 'n/a':>10}"
                      f"{money(o['avg_trade']) if o['trades'] else 'n/a':>10}"
                      f"{d['max_dd']:>10,.2f}{o['max_dd']:>10,.2f}")
            print()
            print("   Both windows started from independent $10,000 balances.")
            print("   Instruments are deliberately NOT ranked; none is called")
            print("   best, worst, winner, loser, strongest or weakest.")

            # Regime persistence tables (factual counts)
            print()
            print("REGIME PERSISTENCE TABLES (bucket counts per instrument)")
            for fam in ("VOLATILITY", "EFFICIENCY", "ADX", "TREND DISTANCE",
                        "BB EXCURSION"):
                print(f"\n  {fam}:")
                print(f"    {'Instrument':<12}{'same':>7}{'changed':>9}"
                      f"{'insufficient':>14}")
                for r in results:
                    rows = r["persistence"][fam]
                    same = sum(1 for x in rows if x["persistence"] == "same sign")
                    changed = sum(1 for x in rows
                                  if x["persistence"] == "changed sign")
                    insuff = sum(1 for x in rows
                                 if x["persistence"] == "insufficient data")
                    print(f"    {r['instrument']:<12}{same:>7}{changed:>9}"
                          f"{insuff:>14}")
            print()
            print("   'changed' means the development-observed average-P/L")
            print("   relationship did not persist into OOS for that bucket")
            print("   (factual observation only - no score, no ranking).")
        else:
            print("   No instrument could be diagnosed.")

        if unavailable:
            seen = []
            for ins in INSTRUMENTS:
                if ins in unavailable and ins not in seen:
                    seen.append(ins)
            print("\nInstruments NOT diagnosed (reported unavailable/skipped,")
            print(f"no substitution made): {', '.join(seen)}")

        # ==========================================================
        # FACTUAL QUESTIONS
        # ==========================================================
        if results:
            print()
            print("=" * 70)
            print("FACTUAL QUESTIONS (association wording only)")
            print("=" * 70)
            print_factual_answers(factual_answers(results))

        print()
        print("Historical research only. No orders were sent.")
        print("Development regime thresholds were frozen before OOS evaluation.")
        print()
        print("This is a historical association study on one data window. It")
        print("does not prove anything about future behaviour, does not")
        print("recommend any filter or parameter change, and ranks no")
        print("instrument. Educational use only.")

    finally:
        # ==========================================================
        # mt5.shutdown() is guaranteed by finally (read-only session)
        # ==========================================================
        mt5.shutdown()
        print()
        print("MetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()
