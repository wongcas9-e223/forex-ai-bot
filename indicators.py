"""
indicators.py - SHARED, PURE PANDAS/NUMPY CORE INDICATOR FUNCTIONS.

WHAT THIS IS
    The single home for the five core indicator calculators that were
    previously duplicated (verbatim, and verified runtime-identical) as
    local functions across the project's live / research / diagnostic
    scripts:
        add_atr(df, period=14)
        add_bollinger(df, period=20, num_std=2.0)
        add_adx(df, period=14)
        add_ema(df, period)
        add_rsi(df, period=14)

WHAT THIS IS NOT
    - not a trading component, not an execution layer,
    - contains NO trading-terminal import, NO terminal initialisation,
      NO order/execution API, NO trading logic, NO directional/BUY-SELL
      logic, NO command-line / --run behaviour,
    - NO filesystem side effects: every function takes a DataFrame and
      returns the same DataFrame with indicator columns added,
    - deterministic and pandas/numpy-only.

FIDELITY
    The formulas here are the frozen, verified ones.  Smoothing, warm-up
    (min_periods), rolling windows, std ddof, multipliers, output column
    names and NaN behaviour are unchanged.  These functions are causal:
    each value uses only the current and previous rows.

    (Established by Technical Debt Audit 2: all 83 duplicated copies of
    these five functions were runtime-identical on a deterministic
    240-row fixture, including NaN masks, dtypes and column names.)

DISCLAIMER: descriptive indicator math only, for research and education.
No directive of any kind.  Educational use only.
"""

# ============================================================
# Imports: pandas + numpy only (numpy is required by add_adx for
# np.nan in the zero-DI-sum guard).
# ============================================================
import numpy as np
import pandas as pd


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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


def add_bollinger(df: pd.DataFrame, period: int = 20,
                  num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (rolling - causal by construction)."""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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


def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Exponential Moving Average (causal; state feature only)."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False,
                                         min_periods=period).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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
