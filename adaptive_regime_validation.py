"""
adaptive_regime_validation.py - HISTORICAL RESEARCH / VALIDATION ONLY.

Research question:
  "Can the current market environment be compared with historically
   similar environments, and does that similarity contain useful
   OUT-OF-SAMPLE information about subsequent signal outcomes?"

This is the next stage after multi_instrument_10000_history_test.py,
market_regime_diagnostic.py and regime_oos_validation.py. Earlier
results indicated time instability and development-to-OOS regime
relationship changes, so a FIXED regime filter may not be robust.
This script therefore tests the SIMILARITY idea deterministically.

NOTHING ABOUT THE TRADING RULE CHANGES:
  The signal is the frozen BB(20, 2.0) + ADX14 < 25 mean-reversion
  entry (next-open execution) with the frozen SL 2.0 x ATR14 /
  TP 1.5 x ATR exit model, stop-first inside candles, 0.10 lots, $2
  per trade, one position at a time, independent $10,000 per window,
  broker-spec P/L. Every original frozen signal stays in the
  evaluation: NO filter, NO position-size change, NO SL/TP/entry
  change, and NO trading rule of the form "if neighbor average > 0
  then trade". The similarity analysis is descriptive only.

NO AI / NO MACHINE LEARNING:
  No LLM, no neural networks, no sklearn, no XGBoost, no random
  forests, no clustering trained on OOS, no reinforcement learning.
  The similarity model is a deterministic, transparent nearest-
  neighbour computation in a standardized feature space.

STRICT CHRONOLOGICAL OOS DESIGN:
  Per instrument: first ~50% of closed candles = DEVELOPMENT,
  strictly later ~50% = OOS. No shuffling, no random split.

WHY NORMALIZATION COMES FROM DEVELOPMENT ONLY:
  Per-feature mean and standard deviation are computed from the
  DEVELOPMENT signal population, FROZEN, and applied to BOTH
  windows. Recomputing them on OOS would let OOS data influence its
  own distances and would contaminate the OOS test. Zero or missing
  standard deviations are handled safely (the feature then
  contributes nothing to any distance - no crash, documented).

WHY NEIGHBORS COME FROM DEVELOPMENT ONLY:
  Every OOS signal is compared ONLY against historical DEVELOPMENT
  signals - never against other OOS signals, and never using future
  OOS outcomes. Neighbor outcomes (12/24-candle forward returns and
  actual frozen-strategy trade P/L) are computed INSIDE the
  development window only: a neighbor's forward window is clipped at
  the development boundary so no OOS-period candle can leak into the
  historical reference. Late-development signals therefore have
  clipped forward windows (deterministic and conservative).

WHY K = 10:
  K = 10 is a fixed research parameter. No other K is tested and K
  is not optimized.

DISTANCE BUCKETS:
  OOS signals are descriptively grouped by nearest-neighbour
  distance into LOW / MEDIUM / HIGH. The 33rd/66th percentile
  thresholds come from DEVELOPMENT nearest-neighbour distances only
  (each development signal's distance to its closest OTHER
  development signal), are frozen, and are applied to OOS. They are
  never optimized. If development nearest-neighbour distances cannot
  be computed cleanly (e.g. fewer than K+1 development signals), a
  clearly reported deterministic fallback is used and the buckets are
  reported as n/a.

READ-ONLY MT5 SAFETY:
  Only initialize, shutdown, symbols_get, symbol_info,
  symbol_info_tick, symbol_select and copy_rates_from_pos are used.
  No order is ever placed, modified or closed. mt5.shutdown() always
  runs through a finally block.

COMMAND-LINE SAFETY:
  Default execution runs ONLY the synthetic verification tests:
      python adaptive_regime_validation.py
  The real MT5 diagnostic requires the explicit flag:
      python adaptive_regime_validation.py --run

DISCLAIMER: Historical association study on one data window. No
causal claims, no predictions, no rankings, no recommendations.
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

from indicators import (
    add_adx,
    add_atr,
    add_bollinger,
    add_ema,
    add_rsi,
)

# ============================================================
# Frozen settings (identical to the previous diagnostics)
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 10_000
MIN_CANDLES = 400

INSTRUMENTS: List[str] = ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

STARTING_BALANCE = 10_000.0
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
RSI_PERIOD = 14
EMA_TREND_PERIOD = 50
WARMUP = 60

FWD_WINDOWS = (12, 24)        # Neighbor-outcome forward-return windows
HOLD_WINDOW = 12              # (frozen exit model horizon reference)

NEIGHBORS_K = 10              # Fixed research parameter - NOT optimized

# Fixed regime buckets (identical to the previous diagnostics)
ADX_BUCKETS = ["ADX<15", "ADX15-20", "ADX20-25", "ADX>=25"]
VOL_BUCKETS = ["LOW", "MEDIUM", "HIGH"]
EFF_BUCKETS = ["LOW", "MEDIUM", "HIGH"]
TREND_BUCKETS = ["<1 ATR", "1-2 ATR", ">=2 ATR"]

LOW_SAMPLE_N = 5
LOW_SAMPLE_TAG = "LOW SAMPLE"

DIST_BUCKETS = ["LOW DISTANCE", "MEDIUM DISTANCE", "HIGH DISTANCE"]

# The 12 environment features (fixed, in this order, with the required
# canonical names). Every value uses only the signal candle and earlier -
# no future data. Mapping to the underlying measurements:
#   adx             <- ADX14
#   atr_pct         <- ATR14 / close * 100
#   ema_dist        <- abs(close - EMA50) / ATR14
#   bb_excursion    <- abs(close - BB_middle) / (BB_upper - BB_lower)
#   efficiency      <- 24-candle efficiency ratio
#   return_6/12/24  <- 6/12/24-candle return %
#   rsi             <- RSI14
#   bb_mid_distance <- (close - BB_middle) / (BB_upper - BB_lower) [signed]
#   volatility_6/24 <- std of 1-candle returns over the last 6 / 24 candles
FEATURES: List[str] = [
    "adx",              # 1. ADX14
    "atr_pct",          # 2. ATR14 / close * 100
    "ema_dist",         # 3. abs(close - EMA50) / ATR14
    "bb_excursion",     # 4. abs(close - BB_middle) / (BB_upper - BB_lower)
    "efficiency",       # 5. 24-candle efficiency ratio
    "return_6",         # 6. 6-candle return %
    "return_12",        # 7. 12-candle return %
    "return_24",        # 8. 24-candle return %
    "rsi",              # 9. RSI14
    "bb_mid_distance",  # 10. signed distance from the BB middle (band width)
    "volatility_6",     # 11. recent 6-candle volatility
    "volatility_24",    # 12. recent 24-candle volatility
]


# ============================================================
# Indicator functions (causal only)
# The five core indicators are imported from the shared, pandas/numpy-
# only indicators.py module (extracted verbatim; no formula, smoothing,
# min_periods, ddof, column-name or NaN behaviour change).
# ============================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All frozen indicators + causal environment measurements."""
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)
    df = add_ema(df, EMA_TREND_PERIOD)
    df = add_rsi(df, RSI_PERIOD)

    df["ATR_pct"] = np.where(df["close"] > 0, df["ATR"] / df["close"] * 100, np.nan)
    df["EMA50_dist"] = np.where(df["ATR"] > 0,
                                (df["close"] - df["EMA50"]).abs() / df["ATR"], np.nan)
    band_width = df["BB_UPPER"] - df["BB_LOWER"]
    df["bb_signed"] = np.where(band_width > 0,
                               (df["close"] - df["BB_MIDDLE"]) / band_width, np.nan)
    df["bb_exc"] = df["bb_signed"].abs()
    net_move = (df["close"] - df["close"].shift(24)).abs()
    gross_path = df["close"].diff().abs().rolling(24).sum()
    df["eff24"] = np.where(gross_path > 0, net_move / gross_path, np.nan)
    df["ret6"] = df["close"].pct_change(6) * 100
    df["ret12"] = df["close"].pct_change(12) * 100
    df["ret24"] = df["close"].pct_change(24) * 100
    ret1 = df["close"].pct_change(1) * 100
    df["vol6"] = ret1.rolling(6).std(ddof=0)
    df["vol24"] = ret1.rolling(24).std(ddof=0)
    return df


# ============================================================
# Frozen signal + frozen exit engine (identical rules as before)
# ============================================================
def calculate_frozen_strategy(df: pd.DataFrame, value_per_unit: float,
                              volume: float, start_bar: int = WARMUP
                              ) -> Tuple[pd.DataFrame, float]:
    """Run the frozen strategy; one row per signal/trade.

    Signal on closed candle i, entry at the OPEN of candle i+1, SL/TP
    fixed from the signal candle's ATR, stop-first inside-candle
    handling, one position at a time, $2 cost per completed trade,
    independent $10,000 starting balance per call. The environment
    vector is RECORDED at signal time and never alters any decision.
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
                exit_reason = "END_OF_DATA"

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
                "adx": df["ADX"].iloc[i],
                "atr_pct": df["ATR_pct"].iloc[i],
                "ema_dist": df["EMA50_dist"].iloc[i],
                "bb_excursion": df["bb_exc"].iloc[i],
                "bb_mid_distance": df["bb_signed"].iloc[i],
                "efficiency": df["eff24"].iloc[i],
                "return_6": df["ret6"].iloc[i],
                "return_12": df["ret12"].iloc[i],
                "return_24": df["ret24"].iloc[i],
                "rsi": df["RSI"].iloc[i],
                "volatility_6": df["vol6"].iloc[i],
                "volatility_24": df["vol24"].iloc[i],
                # forward returns filled later (exit-independent)
                "fwd12_pct": np.nan,
                "fwd24_pct": np.nan,
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

    return pd.DataFrame(trades), balance


def calculate_forward_returns(df: pd.DataFrame, trades: pd.DataFrame,
                              windows: Tuple[int, ...] = FWD_WINDOWS) -> pd.DataFrame:
    """Attach forward returns (% at entry) to a trade log.

    Exit-independent diagnostic measurements only. Computed WITHIN the
    given frame: for development trades this clips at the development
    boundary, which guarantees no OOS candle can leak into the
    historical neighbour reference (see module docstring).
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


# ============================================================
# Environment vectors and DEVELOPMENT-only normalization
# ============================================================
def feature_matrix(trades: pd.DataFrame) -> np.ndarray:
    """Extract the 12-feature environment matrix from a trade log."""
    return trades[FEATURES].to_numpy(dtype=float)


def normalize_features(matrix: np.ndarray,
                       mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize a feature matrix with FROZEN mean/std vectors.

    WHY FROZEN: the mean/std come from DEVELOPMENT signals only and are
    applied unchanged to both windows; OOS data never influences its
    own normalization. NaN features and zero standard deviations are
    handled safely: those feature pairs simply contribute 0 to any
    distance (no crash, documented deterministic behaviour).
    """
    z = np.full(matrix.shape, 0.0, dtype=float)
    for f in range(matrix.shape[1]):
        col = matrix[:, f]
        s = std[f]
        m = mean[f]
        if not np.isfinite(s) or s <= 0 or not np.isfinite(m):
            continue                      # zero/NaN std -> feature contributes 0
        valid = np.isfinite(col)
        z[valid, f] = (col[valid] - m) / s
        # NaN entries stay 0.0 (documented: they contribute nothing)
    return z


def normalization_stats(dev_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std from DEVELOPMENT signals ONLY."""
    mean = np.zeros(dev_matrix.shape[1])
    std = np.zeros(dev_matrix.shape[1])
    for f in range(dev_matrix.shape[1]):
        col = dev_matrix[:, f]
        valid = col[np.isfinite(col)]
        if len(valid) == 0:
            mean[f], std[f] = np.nan, np.nan
            continue
        mean[f] = valid.mean()
        std[f] = valid.std(ddof=0)        # population std (frozen choice)
    return mean, std


# ============================================================
# Nearest-neighbour similarity (K = 10, fixed)
# ============================================================
def nearest_neighbor_indices(oos_z: np.ndarray, dev_z: np.ndarray,
                             k: int = NEIGHBORS_K) -> Tuple[np.ndarray, np.ndarray]:
    """For each OOS signal, the k nearest DEVELOPMENT signals.

    Neighbors come from the DEVELOPMENT database ONLY - an OOS signal
    is never compared against another OOS signal, and no future OOS
    outcome is used. Returns (indices [n_oos x k], distances [n_oos x k])
    sorted by distance. If the development database has fewer than k
    signals, all available development signals are used (deterministic
    fallback, reported by the caller).
    """
    # squared distances via the expansion trick; NaNs cannot occur in z
    # (normalize_features emits 0.0 for invalid entries)
    diff = oos_z[:, None, :] - dev_z[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    k_eff = min(k, dev_z.shape[0])
    idx = np.argsort(d2, axis=1)[:, :k_eff]
    dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
    return idx, dist


def neighbor_outcomes(dev_trades: pd.DataFrame,
                      idx: np.ndarray) -> List[dict]:
    """Aggregate the ACTUAL subsequent outcomes of each OOS signal's
    development neighbours.

    The outcomes are the development trades' own records (12/24-candle
    forward returns computed inside the development window and actual
    frozen-strategy trade P/L) - data strictly AFTER each historical
    signal, with no lookahead relative to it and no OOS information.
    """
    out = []
    for row in idx:
        nb = dev_trades.iloc[row]
        f12 = nb["fwd12_pct"].astype(float)
        f24 = nb["fwd24_pct"].astype(float)
        pnl = nb["pnl"].astype(float)
        out.append({
            "pred_12r": f12.mean(),
            "pred_24r": f24.mean(),
            "pred_pnl": pnl.mean(),
            "pos_12r_pct": (f12 > 0).mean() * 100 if len(f12) else np.nan,
            "pos_24r_pct": (f24 > 0).mean() * 100 if len(f24) else np.nan,
            "profitable_pct": (pnl > 0).mean() * 100 if len(pnl) else np.nan,
            "nn_distance": float("nan"),   # filled by caller from dist matrix
        })
    return out


def development_nn_distances(dev_z: np.ndarray,
                             k: int = NEIGHBORS_K) -> np.ndarray:
    """Each DEVELOPMENT signal's distance to its closest OTHER
    development signal (self excluded) - used ONLY to derive the
    descriptive distance-bucket thresholds (33rd/66th percentiles,
    frozen before OOS evaluation).
    """
    d2 = np.einsum("ijk,ijk->ij",
                   dev_z[:, None, :] - dev_z[None, :, :],
                   dev_z[:, None, :] - dev_z[None, :, :])
    np.fill_diagonal(d2, np.inf)          # exclude self
    nearest = np.sqrt(d2.min(axis=1))
    nearest[np.isinf(nearest)] = np.nan   # a lone signal has no neighbor
    return nearest


def distance_bucket_thresholds(dev_nn: np.ndarray) -> Tuple[float, float, str]:
    """33rd/66th percentiles of DEVELOPMENT nearest-neighbour distances.

    Returns (t1, t2, note). If development distances cannot be computed
    cleanly (fewer than NEIGHBORS_K + 1 valid values), a documented
    deterministic fallback is returned: NaN thresholds and buckets
    reported as n/a.
    """
    valid = dev_nn[np.isfinite(dev_nn)]
    if len(valid) < NEIGHBORS_K + 1:
        return (float("nan"), float("nan"),
                "fallback: fewer than K+1 development nearest-neighbour "
                "distances available; distance buckets reported as n/a")
    return (float(np.quantile(valid, 1 / 3)),
            float(np.quantile(valid, 2 / 3)),
            "development-only 33rd/66th percentiles (frozen)")


def distance_bucket_label(dist: float, t1: float, t2: float) -> str:
    """LOW / MEDIUM / HIGH DISTANCE using the frozen dev-only thresholds."""
    if dist != dist or t1 != t1 or t2 != t2:
        return "n/a"
    if dist < t1:
        return "LOW DISTANCE"
    if dist < t2:
        return "MEDIUM DISTANCE"
    return "HIGH DISTANCE"


# ============================================================
# Regime labels (fixed buckets, as in the previous diagnostics)
# ============================================================
def adx_regime_label(adx: float) -> str:
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
    if atr_pct != atr_pct:
        return "n/a"
    if atr_pct < v1:
        return "LOW"
    if atr_pct < v2:
        return "MEDIUM"
    return "HIGH"


def eff_label(ratio: float) -> str:
    if ratio != ratio:
        return "n/a"
    if ratio < 0.30:
        return "LOW"
    if ratio < 0.60:
        return "MEDIUM"
    return "HIGH"


def calculate_regimes(trades: pd.DataFrame, v1: float, v2: float) -> pd.DataFrame:
    """Attach the fixed descriptive regime labels (never filters)."""
    trades = trades.copy()
    trades["adx_regime"] = trades["adx"].apply(adx_regime_label)
    trades["vol_regime"] = trades["atr_pct"].apply(lambda x: vol_regime_label(x, v1, v2))
    trades["eff_regime"] = trades["efficiency"].apply(eff_label)
    return trades


def compute_vol_edges_from_development(dev_df: pd.DataFrame) -> Tuple[float, float]:
    """33rd/66th percentile ATR% edges from DEVELOPMENT candles only."""
    s = dev_df["ATR_pct"].dropna()
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


# ============================================================
# Statistics helpers
# ============================================================
def correlation_safe(x: pd.Series, y: pd.Series) -> Tuple[float, str]:
    """Pearson correlation with an explicit 'insufficient variation' path.

    Returns (value_or_nan, note). No value is invented: if either side
    has fewer than 3 valid pairs or zero variance, the note says
    'insufficient variation'.
    """
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return float("nan"), "insufficient variation (fewer than 3 pairs)"
    if pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
        return float("nan"), "insufficient variation (zero variance)"
    return float(pair.corr().iloc[0, 1]), "ok"


def sign_agreement(pred: pd.Series, actual: pd.Series) -> dict:
    """2x2 sign table + agreement rate (descriptive only)."""
    pair = pd.concat([pred.rename("p"), actual.rename("a")], axis=1).dropna()
    pp_an = int(((pair["p"] > 0) & (pair["a"] < 0)).sum())
    pn_ap = int(((pair["p"] < 0) & (pair["a"] > 0)).sum())
    pp_ap = int(((pair["p"] > 0) & (pair["a"] > 0)).sum())
    pn_an = int(((pair["p"] < 0) & (pair["a"] < 0)).sum())
    n = pp_ap + pp_an + pn_ap + pn_an
    agree = pp_ap + pn_an
    return {
        "n": n,
        "pred_pos_actual_pos": pp_ap,
        "pred_pos_actual_neg": pp_an,
        "pred_neg_actual_pos": pn_ap,
        "pred_neg_actual_neg": pn_an,
        "agreement_rate": (agree / n * 100) if n else float("nan"),
    }


def split_development_oos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological 50/50 split (development first, OOS later)."""
    half = len(df) // 2
    return df.iloc[:half].reset_index(drop=True), df.iloc[half:].reset_index(drop=True)


def fmt_pf(pf: float) -> str:
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def money(v: float) -> str:
    return "no trades" if (v != v) else f"${v:+.2f}"


def num(v: float, spec: str = "{:.2f}") -> str:
    return "n/a" if (v != v) else spec.format(v)


def longest_losing_streak(trades: pd.DataFrame) -> int:
    if trades is None or len(trades) == 0:
        return 0
    streak = max_streak = 0
    for v in trades["pnl"]:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def summarize_window(trades: pd.DataFrame) -> dict:
    """Descriptive window summary (both windows start from $10,000)."""
    if trades is None or len(trades) == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": float("nan"),
                "gp": 0.0, "gl": 0.0, "pf": float("nan"), "avg_trade": float("nan"),
                "total_pnl": 0.0, "ending_balance": STARTING_BALANCE,
                "return": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0,
                "lose_streak": 0}
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
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100, "gp": gp, "gl": gl,
        "pf": gp / gl if gl > 0 else float("inf"),
        "avg_trade": trades["pnl"].mean(), "total_pnl": trades["pnl"].sum(),
        "ending_balance": trades["balance_after"].iloc[-1],
        "return": (trades["balance_after"].iloc[-1] - STARTING_BALANCE)
                  / STARTING_BALANCE * 100,
        "max_dd": max_dd, "max_dd_pct": max_dd_pct,
        "lose_streak": longest_losing_streak(trades),
    }


def consensus_answer(values: List[str]) -> str:
    """Collapse per-instrument factual answers into one factual answer."""
    valid = [v for v in values if v in ("YES", "NO", "MIXED")]
    if not valid:
        return "INSUFFICIENT DATA"
    if all(v == "YES" for v in valid):
        return "YES"
    if all(v == "NO" for v in valid):
        return "NO"
    return "MIXED"


def corr_answer(corrs: List[float]) -> str:
    """YES = every computable correlation is positive, NO = every one
    negative, MIXED = both signs present (factual, no magnitude
    judgement)."""
    valid = [c for c in corrs if c == c]
    if not valid:
        return "INSUFFICIENT DATA"
    if all(c > 0 for c in valid):
        return "YES"
    if all(c < 0 for c in valid):
        return "NO"
    return "MIXED"


# ============================================================
# Per-instrument similarity analysis
# ============================================================
def analyze_instrument(df: pd.DataFrame, dev_df: pd.DataFrame,
                       oos_df: pd.DataFrame, value_per_unit: float,
                       volume: float) -> Optional[dict]:
    """Run the full similarity analysis for one instrument.

    Returns a result dict (or None when there is nothing to compare).
    All normalization, neighbors and thresholds come from DEVELOPMENT
    only; the OOS side is evaluated with those frozen references.
    """
    # --- frozen strategy on both windows (independent $10,000) --------
    dev_trades, _ = calculate_frozen_strategy(dev_df, value_per_unit, volume)
    oos_trades, _ = calculate_frozen_strategy(oos_df, value_per_unit, volume)
    if len(dev_trades) == 0 or len(oos_trades) == 0:
        return None

    # --- forward returns (development clipped at its own boundary) ----
    dev_trades = calculate_forward_returns(dev_df, dev_trades)
    oos_trades = calculate_forward_returns(oos_df, oos_trades)

    # --- DEVELOPMENT-only normalization, frozen -----------------------
    dev_matrix = feature_matrix(dev_trades)
    oos_matrix = feature_matrix(oos_trades)
    mean, std = normalization_stats(dev_matrix)
    dev_z = normalize_features(dev_matrix, mean, std)
    oos_z = normalize_features(oos_matrix, mean, std)

    # --- K=10 nearest DEVELOPMENT neighbours per OOS signal -----------
    idx, dist = nearest_neighbor_indices(oos_z, dev_z)
    nb = neighbor_outcomes(dev_trades, idx)
    nb_df = pd.DataFrame(nb)
    nb_df["nn_distance"] = dist[:, 0]           # closest neighbour distance
    nb_df["nn_mean_distance"] = dist.mean(axis=1)

    # --- descriptive distance buckets (frozen dev-only thresholds) ----
    dev_nn = development_nn_distances(dev_z)
    t1, t2, thr_note = distance_bucket_thresholds(dev_nn)
    nb_df["dist_bucket"] = [distance_bucket_label(d, t1, t2) for d in nb_df["nn_distance"]]

    # --- regime labels (fixed buckets; vol edges from DEVELOPMENT) ----
    v1, v2 = compute_vol_edges_from_development(dev_df)
    oos_trades = calculate_regimes(oos_trades, v1, v2)
    dev_trades = calculate_regimes(dev_trades, v1, v2)

    # --- combine OOS actuals with neighbour expectations ---------------
    oos_eval = pd.concat([
        oos_trades.reset_index(drop=True),
        nb_df.reset_index(drop=True),
    ], axis=1)

    # --- correlations (predicted vs actual) ----------------------------
    corr12, note12 = correlation_safe(oos_eval["pred_12r"], oos_eval["fwd12_pct"])
    corr24, note24 = correlation_safe(oos_eval["pred_24r"], oos_eval["fwd24_pct"])
    corr_pnl, note_pnl = correlation_safe(oos_eval["pred_pnl"], oos_eval["pnl"])

    # --- sign agreement -------------------------------------------------
    agree12 = sign_agreement(oos_eval["pred_12r"], oos_eval["fwd12_pct"])
    agree24 = sign_agreement(oos_eval["pred_24r"], oos_eval["fwd24_pct"])
    agree_pnl = sign_agreement(oos_eval["pred_pnl"], oos_eval["pnl"])

    return {
        "instrument": None,   # filled by caller
        "dev_trades": dev_trades,
        "oos_trades": oos_trades,
        "oos_eval": oos_eval,
        "mean": mean,
        "std": std,
        "thresholds": (t1, t2),
        "threshold_note": thr_note,
        "vol_edges": (v1, v2),
        "corr": {"12r": (corr12, note12), "24r": (corr24, note24),
                 "pnl": (corr_pnl, note_pnl)},
        "agreement": {"12r": agree12, "24r": agree24, "pnl": agree_pnl},
        "dev_summary": summarize_window(dev_trades),
        "oos_summary": summarize_window(oos_trades),
    }


# ============================================================
# Printing
# ============================================================
def print_norm_stats(name: str, mean: np.ndarray, std: np.ndarray) -> None:
    """Print the FROZEN development normalization statistics."""
    print(f"\n  {name} (FROZEN, development signals only):")
    print(f"    {'feature':<12}{'mean':>14}{'std':>14}")
    for f, feat in enumerate(FEATURES):
        m = mean[f] if np.isfinite(mean[f]) else float("nan")
        s = std[f] if np.isfinite(std[f]) else float("nan")
        zero_note = "  (zero std -> feature contributes 0 to distances)" \
            if (np.isfinite(s) and s == 0) else ""
        print(f"    {feat:<12}{m:>14.4f}{s:>14.4f}{zero_note}")


def cell_line(prefix: str, sub: pd.DataFrame, with_corr: bool = False) -> None:
    """Print one descriptive cell with LOW SAMPLE tagging (no ranking)."""
    if sub is None or len(sub) == 0:
        print(f"      {prefix:<34} (no trades)")
        return
    tag = f"  [{LOW_SAMPLE_TAG}]" if len(sub) < LOW_SAMPLE_N else ""
    wins = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] < 0]
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    line = (f"      {prefix:<34} {len(sub):>4} trades | "
            f"avg P/L ${sub['pnl'].mean():>8.2f} | "
            f"win {(len(wins) / len(sub) * 100):>5.1f}% | PF {fmt_pf(pf):>5}")
    if with_corr and "pred_12r" in sub:
        line += (f" | pred12R {sub['pred_12r'].mean():+.3f}%"
                 f" | act12R {sub['fwd12_pct'].mean():+.3f}%")
    print(line + tag)


def print_instrument_analysis(res: dict, instrument: str) -> None:
    """Print the full per-instrument similarity analysis."""
    oos_eval = res["oos_eval"]

    print()
    print("=" * 70)
    print(f"SIMILARITY ANALYSIS - {instrument}")
    print("(neighbors = DEVELOPMENT signals only, K = "
          f"{NEIGHBORS_K}, standardized Euclidean distance)")
    print("=" * 70)
    print(f"  development signal count : {len(res['dev_trades'])}")
    print(f"  OOS signal count         : {len(oos_eval)}")
    print(f"  distance thresholds      : t1 {num(res['thresholds'][0], '{:.4f}')} | "
          f"t2 {num(res['thresholds'][1], '{:.4f}')}")
    print(f"    ({res['threshold_note']})")

    print_norm_stats("Development feature normalization", res["mean"], res["std"])

    # --- OOS similarity statistics ------------------------------------
    print()
    print("  OOS similarity statistics:")
    print(f"    average nearest distance : {num(oos_eval['nn_distance'].mean(), '{:.4f}')}")
    print(f"    median nearest distance  : {num(oos_eval['nn_distance'].median(), '{:.4f}')}")
    print(f"    avg neighbor predicted 12R : {num(oos_eval['pred_12r'].mean(), '{:+.3f}%')}")
    print(f"    avg actual 12R             : {num(oos_eval['fwd12_pct'].mean(), '{:+.3f}%')}")
    print(f"    avg neighbor predicted 24R : {num(oos_eval['pred_24r'].mean(), '{:+.3f}%')}")
    print(f"    avg actual 24R             : {num(oos_eval['fwd24_pct'].mean(), '{:+.3f}%')}")
    print(f"    avg neighbor predicted P/L : {money(oos_eval['pred_pnl'].mean())}")
    print(f"    avg actual trade P/L       : {money(oos_eval['pnl'].mean())}")

    # --- correlations --------------------------------------------------
    print()
    print("  Correlations (neighbor expectation vs actual OOS outcome):")
    for key, label in (("12r", "12-candle return"), ("24r", "24-candle return"),
                       ("pnl", "trade P/L")):
        c, note = res["corr"][key]
        c_txt = "n/a" if c != c else f"{c:+.4f}"
        print(f"    {label:<18}: {c_txt}  ({note})")

    # --- sign agreement -------------------------------------------------
    print()
    print("  Sign agreement (predicted vs actual):")
    for key, label in (("12r", "12-candle return"), ("24r", "24-candle return"),
                       ("pnl", "trade P/L")):
        a = res["agreement"][key]
        if a["n"] == 0:
            print(f"    {label:<18}: no comparable pairs")
            continue
        print(f"    {label:<18}: n={a['n']} | +/+ {a['pred_pos_actual_pos']} "
              f"| +/- {a['pred_pos_actual_neg']} | -/+ {a['pred_neg_actual_pos']} "
              f"| -/- {a['pred_neg_actual_neg']} | agreement "
              f"{a['agreement_rate']:.1f}%")

    # --- distance bucket analysis ---------------------------------------
    print()
    print("  DISTANCE BUCKET ANALYSIS (descriptive only, no ranking):")
    for b in DIST_BUCKETS + ["n/a"]:
        sub = oos_eval[oos_eval["dist_bucket"] == b]
        if len(sub) == 0 and b == "n/a":
            continue
        cell_line(f"{b} (pred vs act)", sub, with_corr=True)

    # --- direction analysis ---------------------------------------------
    print()
    print("  DIRECTION ANALYSIS (same frozen reference for both):")
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        sub = oos_eval[oos_eval["direction"] == d]
        if len(sub) == 0:
            print(f"    {dname}: no OOS signals")
            continue
        c12, n12 = correlation_safe(sub["pred_12r"], sub["fwd12_pct"])
        cp, np_ = correlation_safe(sub["pred_pnl"], sub["pnl"])
        a12 = sign_agreement(sub["pred_12r"], sub["fwd12_pct"])
        print(f"    {dname}: n={len(sub)} | corr12 "
              f"{'n/a' if c12 != c12 else f'{c12:+.4f}'} ({n12}) | "
              f"corrPL {'n/a' if cp != cp else f'{cp:+.4f}'} ({np_}) | "
              f"12R agreement {num(a12['agreement_rate'], '{:.1f}%')} "
              f"(n={a12['n']})")
        print(f"         avg pred P/L {money(sub['pred_pnl'].mean())} | "
              f"avg actual P/L {money(sub['pnl'].mean())} | "
              f"actual win rate "
              f"{(sub['pnl'] > 0).mean() * 100:.1f}%")

    # --- direction x distance ---------------------------------------------
    print()
    print("  DIRECTION x DISTANCE (descriptive, no ranking):")
    for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
        for b in DIST_BUCKETS:
            sub = oos_eval[(oos_eval["direction"] == d)
                           & (oos_eval["dist_bucket"] == b)]
            cell_line(f"{dname} x {b}", sub)

    # --- regime x similarity ----------------------------------------------
    print()
    print("  REGIME x SIMILARITY (descriptive, no filters created):")
    for col, order in (("adx_regime", ADX_BUCKETS),
                       ("vol_regime", VOL_BUCKETS),
                       ("eff_regime", EFF_BUCKETS)):
        print(f"    {col.replace('_regime', '')} regime:")
        for b in order:
            sub = oos_eval[oos_eval[col] == b]
            if len(sub) == 0:
                continue
            cell_line(b, sub)

    # --- window summaries ---------------------------------------------------
    print()
    print("  Window summaries (independent $10,000 each; the similarity")
    print("  layer changed nothing about these outcomes):")
    for label, s in (("DEVELOPMENT", res["dev_summary"]),
                     ("OOS", res["oos_summary"])):
        print(f"    {label}: {s['trades']} trades | return {s['return']:+.2f}% | "
              f"PF {fmt_pf(s['pf'])} | avg {money(s['avg_trade'])} | "
              f"max DD ${s['max_dd']:,.2f} | streak {s['lose_streak']}")


# ============================================================
# Factual questions
# ============================================================
def factual_questions(results: List[dict]) -> None:
    """Answer the nine factual questions with YES/NO/MIXED/INSUFFICIENT
    DATA plus the supporting numbers (association wording only)."""
    print()
    print("=" * 70)
    print("MOST IMPORTANT OUT-OF-SAMPLE TEST - FACTUAL QUESTIONS")
    print("(association wording only; no recommendation is made)")
    print("=" * 70)

    corr12 = [r["corr"]["12r"][0] for r in results]
    corr24 = [r["corr"]["24r"][0] for r in results]
    corr_pnl = [r["corr"]["pnl"][0] for r in results]

    # Q1-Q3: correlations
    a1 = corr_answer(corr12)
    a2 = corr_answer(corr24)
    a3 = corr_answer(corr_pnl)
    print()
    print("1. Does historical similarity correlate with actual OOS 12-candle")
    print("   returns?")
    print(f"   {a1}")
    for r, c in zip(results, corr12):
        note = r["corr"]["12r"][1]
        print(f"     {r['instrument']:<12}: "
              f"{'n/a' if c != c else f'{c:+.4f}'} ({note})")
    print()
    print("2. Does historical similarity correlate with actual OOS 24-candle")
    print("   returns?")
    print(f"   {a2}")
    for r, c in zip(results, corr24):
        note = r["corr"]["24r"][1]
        print(f"     {r['instrument']:<12}: "
              f"{'n/a' if c != c else f'{c:+.4f}'} ({note})")
    print()
    print("3. Does historical similarity correlate with actual OOS trade P/L?")
    print(f"   {a3}")
    for r, c in zip(results, corr_pnl):
        note = r["corr"]["pnl"][1]
        print(f"     {r['instrument']:<12}: "
              f"{'n/a' if c != c else f'{c:+.4f}'} ({note})")

    # Q4: sign agreement above 50%?
    print()
    print("4. Is neighbor sign agreement above 50%?")
    answers = []
    for key, label in (("12r", "12R"), ("24r", "24R"), ("pnl", "P/L")):
        rates = [r["agreement"][key]["agreement_rate"] for r in results
                 if r["agreement"][key]["n"] > 0]
        rates_valid = [x for x in rates if x == x]
        if not rates_valid:
            ans = "INSUFFICIENT DATA"
        elif all(x > 50 for x in rates_valid):
            ans = "YES"
        elif all(x <= 50 for x in rates_valid):
            ans = "NO"
        else:
            ans = "MIXED"
        answers.append(ans)
        txt = ", ".join(f"{r['instrument']} "
                        f"{num(r['agreement'][key]['agreement_rate'], '{:.1f}%')}"
                        f" (n={r['agreement'][key]['n']})" for r in results)
        print(f"   {label:<4}: {ans}   [{txt}]")

    # Q5: consistent across instruments?
    print()
    print("5. Does similarity behavior remain consistent across all five")
    print("   instruments?")
    cons = [a1, a2, a3]
    overall = consensus_answer(cons)
    print(f"   {overall} - correlation answers per metric: "
          f"12R {a1}, 24R {a2}, P/L {a3}")
    print("   (factual consistency of the correlation SIGN across")
    print("    instruments; no magnitude or quality judgement)")

    # Q6: consistent between LONG and SHORT?
    print()
    print("6. Does similarity behavior remain consistent between LONG and")
    print("   SHORT?")
    long_rates, short_rates = [], []
    for r in results:
        sub = r["oos_eval"]
        for d, store in (("BUY", long_rates), ("SELL", short_rates)):
            s = sub[sub["direction"] == d]
            if len(s):
                a = sign_agreement(s["pred_pnl"], s["pnl"])
                if a["n"]:
                    store.append(a["agreement_rate"])
    if long_rates and short_rates:
        lm, sm = float(np.mean(long_rates)), float(np.mean(short_rates))
        both_above = lm > 50 and sm > 50
        both_below = lm <= 50 and sm <= 50
        ans = "YES" if both_above else ("NO" if both_below else "MIXED")
        print(f"   {ans} - average P/L sign agreement: LONG {lm:.1f}% "
              f"(across {len(long_rates)} instruments), SHORT {sm:.1f}% "
              f"(across {len(short_rates)})")
    else:
        print("   INSUFFICIENT DATA - one direction has no comparable pairs")

    # Q7: behavior across distance buckets?
    print()
    print("7. Does similarity behavior change across LOW/MEDIUM/HIGH")
    print("   distance?")
    for b in DIST_BUCKETS:
        rows = []
        for r in results:
            sub = r["oos_eval"][r["oos_eval"]["dist_bucket"] == b]
            if len(sub):
                c12, note = correlation_safe(sub["pred_12r"], sub["fwd12_pct"])
                rows.append((r["instrument"], len(sub),
                             "n/a" if c12 != c12 else f"{c12:+.4f}"))
        if rows:
            print(f"   {b:<16}: " +
                  "; ".join(f"{i} n={n} corr12 {c}" for i, n, c in rows))
        else:
            print(f"   {b:<16}: no OOS signals")
    print("   (Per-bucket correlations are reported factually; a difference")
    print("    between buckets is an observed relationship, not a filter.)")

    # Q8: does a relationship in one instrument appear in others?
    print()
    print("8. Does a relationship observed in one instrument also appear in")
    print("   the others?")
    pos12 = sum(1 for c in corr12 if c == c and c > 0)
    neg12 = sum(1 for c in corr12 if c == c and c < 0)
    computed12 = pos12 + neg12
    if computed12 == 0:
        ans8 = "INSUFFICIENT DATA"
    elif pos12 == computed12 or neg12 == computed12:
        ans8 = "YES"
    else:
        ans8 = "MIXED"
    print(f"   {ans8} - 12R correlation sign: positive in {pos12}, negative in "
          f"{neg12} of {computed12} computable instruments")
    posp = sum(1 for c in corr_pnl if c == c and c > 0)
    negp = sum(1 for c in corr_pnl if c == c and c < 0)
    computedp = posp + negp
    if computedp == 0:
        ansp = "INSUFFICIENT DATA"
    elif posp == computedp or negp == computedp:
        ansp = "YES"
    else:
        ansp = "MIXED"
    print(f"         P/L correlation sign: positive in {posp}, negative in "
          f"{negp} of {computedp} computable instruments")

    # Q9: overall descriptive answer
    print()
    print("9. Does historical similarity provide descriptive OOS information")
    print("   without changing the strategy?")
    informative = sum(1 for c in (corr12 + corr24 + corr_pnl) if c == c and abs(c) > 0.1)
    computable = sum(1 for c in (corr12 + corr24 + corr_pnl) if c == c)
    if computable == 0:
        ans9 = "INSUFFICIENT DATA"
    elif informative == 0:
        ans9 = "NO"
    elif informative == computable:
        ans9 = "YES"
    else:
        ans9 = "MIXED"
    print(f"   {ans9} - {informative} of {computable} computable")
    print("   instrument/metric correlations were non-trivial (|r| > 0.1).")
    print("   The frozen strategy was NOT modified in any way by this")
    print("   analysis: no filter, no size change, no rule was created.")
    print("   Any correlation observed here is a historical association in")
    print("   one data window - it does not predict future outcomes and it")
    print("   is not a recommendation.")


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
    """Exact symbol name first, then a short suffix match (no
    substitution of a different underlying instrument)."""
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
        print("ADAPTIVE REGIME VALIDATION (read-only, descriptive only)")
        print("Frozen signal + deterministic K=10 development-neighbour")
        print("similarity. NO machine learning, NO optimization, NO filters.")
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
                    print(f"STATUS: NOT AVAILABLE - no symbol_info. Continuing.")
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
                          f"candles (minimum {MIN_CANDLES}). Continuing.")
                    unavailable.append(requested)
                    continue

                print(f"Closed candles: {len(df)} "
                      f"({df['time'].iloc[0]} -> {df['time'].iloc[-1]})")
                print(f"Volume {volume} lots | contract size "
                      f"{spec['contract_size']:g} | point {spec['point']:g} | "
                      f"digits {spec['digits']} | tick size {spec['tick_size']:g} "
                      f"| tick value {spec['tick_value']:g}")

                # Causal indicators over the full closed series (each value
                # uses only current + previous candles, so this is
                # equivalent to per-window computation and leaks nothing).
                df = calculate_indicators(df)

                dev_df, oos_df = split_development_oos(df)
                print(f"DEVELOPMENT: {len(dev_df)} candles "
                      f"({dev_df['time'].iloc[0]} -> {dev_df['time'].iloc[-1]})")
                print(f"OOS        : {len(oos_df)} candles "
                      f"({oos_df['time'].iloc[0]} -> {oos_df['time'].iloc[-1]})")
                print("  (strict chronological split; OOS occurs strictly "
                      "after development)")

                res = analyze_instrument(df, dev_df, oos_df,
                                         spec["value_per_unit"], volume)
                if res is None:
                    print("NOTE: no comparable signals in one of the windows; "
                          "instrument cannot be analyzed.")
                    unavailable.append(requested)
                    continue
                res["instrument"] = requested
                print_instrument_analysis(res, requested)
                results.append(res)

            except Exception as exc:
                print(f"STATUS: ERROR while processing {requested}: {exc}")
                print("Continuing with the remaining instruments.")
                unavailable.append(requested)

        # ----------------------------------------------------------
        # Cross-instrument summary (factual, no ranking)
        # ----------------------------------------------------------
        print()
        print("=" * 70)
        print("CROSS-INSTRUMENT SUMMARY (factual table, no ranking)")
        print("=" * 70)
        if results:
            hdr = (f"   {'Instrument':<12}{'DevSig':>8}{'OOSsig':>8}"
                   f"{'AvgNNd':>10}{'corr12':>9}{'corr24':>9}"
                   f"{'corrPL':>9}{'agr12':>8}{'agr24':>8}{'agrPL':>8}")
            print(hdr)
            print("   " + "-" * 89)
            for r in results:
                c12 = r["corr"]["12r"][0]
                c24 = r["corr"]["24r"][0]
                cp = r["corr"]["pnl"][0]
                a12 = r["agreement"]["12r"]["agreement_rate"]
                a24 = r["agreement"]["24r"]["agreement_rate"]
                ap = r["agreement"]["pnl"]["agreement_rate"]
                print(f"   {r['instrument']:<12}{len(r['dev_trades']):>8}"
                      f"{len(r['oos_eval']):>8}"
                      f"{num(r['oos_eval']['nn_distance'].mean(), '{:.3f}'):>10}"
                      f"{('n/a' if c12 != c12 else f'{c12:+.3f}'):>9}"
                      f"{('n/a' if c24 != c24 else f'{c24:+.3f}'):>9}"
                      f"{('n/a' if cp != cp else f'{cp:+.3f}'):>9}"
                      f"{num(a12, '{:.0f}%'):>8}{num(a24, '{:.0f}%'):>8}"
                      f"{num(ap, '{:.0f}%'):>8}")
            print()
            print("   corrXX = Pearson correlation between neighbour")
            print("   expectation and actual OOS outcome; agrXX = sign")
            print("   agreement rate. Factual observations only - no")
            print("   instrument is ranked and no bucket is 'best'.")
        else:
            print("   No instrument could be analyzed.")

        if unavailable:
            seen = []
            for ins in INSTRUMENTS:
                if ins in unavailable and ins not in seen:
                    seen.append(ins)
            print("\nInstruments NOT analyzed (reported unavailable/skipped,")
            print(f"no substitution made): {', '.join(seen)}")

        if results:
            factual_questions(results)

        print()
        print("Historical research only. No orders were sent.")
        print("Development-derived normalization, neighbours and distance")
        print("thresholds were frozen before OOS evaluation.")
        print()
        print("This is a descriptive association study on one historical")
        print("window. It does not predict future outcomes, does not")
        print("recommend any filter or rule, and modifies nothing.")
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

    # [1-3] chronological 50/50 split, OOS strictly after, no overlap
    df = calculate_indicators(synth_candles(1000, 1, 100.0, 0.5))
    dev, oos = split_development_oos(df)
    check("split covers all candles", len(dev) + len(oos) == len(df))
    check("development is the FIRST half", len(dev) == len(df) // 2)
    check("no overlap and OOS strictly after development",
          dev["time"].iloc[-1] < oos["time"].iloc[0])

    # [4-5] development-only normalization; OOS uses frozen dev mean/std
    dev_m = np.array([[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]])
    mean, std = normalization_stats(dev_m)
    check("mean from development only",
          np.allclose(mean, [2.0, 20.0]))
    check("std from development only (population)",
          np.allclose(std, [np.sqrt(2 / 3), np.sqrt(200 / 3)]))
    oos_m = np.array([[4.0, 40.0]])
    z_oos = normalize_features(oos_m, mean, std)
    check("OOS z-scores use the FROZEN development mean/std",
          abs(z_oos[0, 0] - (4.0 - 2.0) / std[0]) < 1e-12
          and abs(z_oos[0, 1] - (40.0 - 20.0) / std[1]) < 1e-12)
    z_zero = normalize_features(np.array([[7.0, 7.0]]),
                                np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    check("zero std handled safely (feature contributes 0)",
          np.allclose(z_zero, 0.0))
    z_nan = normalize_features(np.array([[np.nan, 1.0]]),
                               np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    check("NaN features handled safely (contribute 0)",
          np.allclose(z_nan[0, 0], 0.0) and abs(z_nan[0, 1] - 1.0) < 1e-12)

    # [6] nearest-neighbour distance calculation (hand-computed)
    dev_z = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    oos_z = np.array([[1.0, 1.0], [0.0, 0.0]])
    idx, dist = nearest_neighbor_indices(oos_z, dev_z, k=2)
    # oos[0]: distances to dev = sqrt(2), 5, 0 -> nearest dev[2], then dev[0]
    check("NN distance values (standardized Euclidean)",
          abs(dist[0, 0] - 0.0) < 1e-12 and abs(dist[0, 1] - np.sqrt(2)) < 1e-12)
    check("NN ordering by ascending distance",
          idx[0, 0] == 2 and idx[0, 1] == 0 and idx[1, 0] == 0 and idx[1, 1] == 2)

    # [7] K=10 exactly (fixed parameter)
    check("K is exactly 10 and nothing else is tested", NEIGHBORS_K == 10)
    dev_z_big = np.arange(50.0).reshape(25, 2) / 25.0
    idx_big, dist_big = nearest_neighbor_indices(dev_z_big[:3], dev_z_big, k=NEIGHBORS_K)
    check("returns exactly K neighbours", idx_big.shape[1] == NEIGHBORS_K
          and dist_big.shape[1] == NEIGHBORS_K)

    # [8-9] development-only neighbours; no OOS-to-OOS neighbours
    check("neighbor search space is the development matrix only "
          "(signature takes neighbours as an argument)",
          nearest_neighbor_indices.__code__.co_argcount == 3)
    check("neighbor indices always point into the development database",
          idx_big.max() < dev_z_big.shape[0])

    # [10-13] engine outcomes: neighbor P/L = actual dev trade P/L;
    # forward returns; frozen P/L with stop-first; entry timing
    n = 400
    base = calculate_indicators(synth_candles(n, 7, 100.0, 0.5))
    tr0, _ = calculate_frozen_strategy(base, 1000.0, 0.10)
    check("crafted feed produces trades", len(tr0) > 0)
    tr0 = calculate_forward_returns(base, tr0)
    first = tr0.iloc[0]
    i_sig = base.index[base["time"] == first["signal_time"]][0]
    eb = i_sig + 1
    entry = base["open"].iloc[eb]
    atr = base["ATR"].iloc[i_sig]
    check("entry at next candle OPEN",
          abs(first["entry_price"] - entry) < 1e-12
          and first["entry_time"] == base["time"].iloc[eb])
    # forward returns recomputed independently
    ok_fwd = True
    for _, t in tr0.iterrows():
        i2 = base.index[base["time"] == t["signal_time"]][0]
        e2 = base["open"].iloc[i2 + 1]
        for w in FWD_WINDOWS:
            f_idx = min(i2 + 1 + w - 1, len(base) - 1)
            exp = ((base["close"].iloc[f_idx] - e2) if t["direction"] == "BUY"
                   else (e2 - base["close"].iloc[f_idx])) / e2 * 100
            if abs(t[f"fwd{w}_pct"] - exp) > 1e-9:
                ok_fwd = False
    check("12/24 forward returns match independent recomputation", ok_fwd)
    # stop-first: spanning candle -> exact STOP pnl (frozen trade P/L)
    span = base.copy()
    span.iloc[eb, span.columns.get_loc("high")] = entry + 3.0 * atr
    span.iloc[eb, span.columns.get_loc("low")] = entry - 3.0 * atr
    tr_span, _ = calculate_frozen_strategy(span, 1000.0, 0.10)
    t_span = tr_span[tr_span["signal_time"] == first["signal_time"]].iloc[0]
    pnl_stop = (-2.0 * atr) * 0.10 * 1000.0 - COST_PER_TRADE
    check("frozen trade P/L exact and stop-first on spanning candles",
          t_span["exit_reason"] == "STOP_LOSS"
          and abs(t_span["pnl"] - pnl_stop) < 1e-9)
    # neighbor outcomes equal the development trade records themselves
    dev_z_all = np.zeros((len(tr0), 2))
    one_oos = np.zeros((1, 2))
    idx_o, _ = nearest_neighbor_indices(one_oos, dev_z_all, k=min(NEIGHBORS_K, len(tr0)))
    nb = neighbor_outcomes(tr0, idx_o)
    picked = tr0.iloc[idx_o[0]]
    check("neighbor outcomes are the neighbours' OWN dev-window records",
          abs(nb[0]["pred_pnl"] - picked["pnl"].mean()) < 1e-12
          and abs(nb[0]["pred_12r"] - picked["fwd12_pct"].mean()) < 1e-12)

    # [11] no-lookahead: isolated-prefix measurements match full frame
    head = base.iloc[:300].reset_index(drop=True)
    head_meas = calculate_indicators(head.copy())
    cols = ["ATR_pct", "EMA50_dist", "bb_exc", "bb_signed", "eff24",
            "ret6", "ret12", "ret24", "RSI", "vol6", "vol24"]
    check("measurements on an isolated prefix match the full frame (causal)",
          all(np.allclose(head_meas[c].values, base[c].iloc[:300].values,
                          equal_nan=True) for c in cols))
    # environment vector equals the signal-candle values
    ok_env = True
    for _, t in tr0.iterrows():
        i2 = base.index[base["time"] == t["signal_time"]][0]
        for f_frame, f_trade in (("ADX", "adx"), ("ATR_pct", "atr_pct"),
                             ("EMA50_dist", "ema_dist"),
                             ("bb_exc", "bb_excursion"),
                             ("bb_signed", "bb_mid_distance"),
                             ("eff24", "efficiency"), ("ret6", "return_6"),
                             ("ret12", "return_12"), ("ret24", "return_24"),
                             ("RSI", "rsi"), ("vol6", "volatility_6"),
                             ("vol24", "volatility_24")):
            if abs(t[f_trade] - base[f_frame].iloc[i2]) > 1e-12:
                ok_env = False
    check("every environment feature is its SIGNAL-candle value "
          "(no entry-candle or future data)", ok_env)

    # [14] distance percentile thresholds from development only
    # 90 values (30x0.5, 30x1.0, 30x2.0): with linear interpolation the
    # 1/3 quantile sits at position 29.67 -> 0.5+0.667*0.5 = 0.8333, and
    # the 2/3 quantile at position 59.33 -> 1.0+0.333*1.0 = 1.3333.
    dev_nn = np.array([0.5] * 30 + [1.0] * 30 + [2.0] * 30)
    t1, t2, note = distance_bucket_thresholds(dev_nn)
    check("thresholds = development 33rd/66th percentiles",
          abs(t1 - 5.0 / 6.0) < 1e-9 and abs(t2 - 4.0 / 3.0) < 1e-9,
          f"t1={t1} t2={t2}")
    check("bucket labels at the frozen thresholds",
          distance_bucket_label(0.8, t1, t2) == "LOW DISTANCE"
          and distance_bucket_label(1.0, t1, t2) == "MEDIUM DISTANCE"
          and distance_bucket_label(2.0, t1, t2) == "HIGH DISTANCE")
    t1f, t2f, note_f = distance_bucket_thresholds(np.array([np.nan] * 5))
    check("documented deterministic fallback when dev distances are unusable",
          t1f != t1f and t2f != t2f and "fallback" in note_f)

    # [15] low-sample tagging
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        cell_line("tiny", tr0.head(3))
    check("LOW SAMPLE tag when n < 5", "[LOW SAMPLE]" in buf.getvalue())
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        cell_line("big", tr0)
    check("no LOW SAMPLE tag when n >= 5", "[LOW SAMPLE]" not in buf2.getvalue())

    # [16] sign agreement (crafted case)
    sa = sign_agreement(pd.Series([1, 2, -1, -2, 5]),
                        pd.Series([3, -1, -2, -3, 4]))
    check("sign agreement table",
          sa["pred_pos_actual_pos"] == 2 and sa["pred_pos_actual_neg"] == 1
          and sa["pred_neg_actual_pos"] == 0 and sa["pred_neg_actual_neg"] == 2
          and sa["n"] == 5 and abs(sa["agreement_rate"] - 80.0) < 1e-9)

    # [17] correlation handling
    c_ok, note_ok = correlation_safe(pd.Series([1, 2, 3, 4]),
                                     pd.Series([2, 4, 6, 8]))
    check("perfect correlation detected", abs(c_ok - 1.0) < 1e-12 and note_ok == "ok")
    c_const, note_const = correlation_safe(pd.Series([1, 1, 1, 1]),
                                           pd.Series([1, 2, 3, 4]))
    check("zero variance -> 'insufficient variation' (no invented value)",
          c_const != c_const and "insufficient variation" in note_const)
    c_short, note_short = correlation_safe(pd.Series([1.0, 2.0]),
                                           pd.Series([1.0, 2.0]))
    check("fewer than 3 pairs -> 'insufficient variation'",
          c_short != c_short and "insufficient variation" in note_short)

    # [18] LONG/SHORT separation
    sub_long = tr0[tr0["direction"] == "BUY"]
    sub_short = tr0[tr0["direction"] == "SELL"]
    check("LONG/SHORT subsets partition the OOS signals",
          len(sub_long) + len(sub_short) == len(tr0))

    # [19] no order functions in this module's source (fragment trick so
    # this file itself contains zero contiguous forbidden names)
    import inspect
    src = inspect.getsource(sys.modules[__name__])
    for a, b in (("order", "send"), ("order", "check"),
                 ("positions", "get"), ("positions", "total"),
                 ("position", "get")):
        check(f"source contains no '{a}_{b}' function", (a + "_" + b) not in src)

    # [20] frozen strategy constants unchanged
    check("frozen strategy constants unchanged",
          (SL_ATR_MULT, TP_ATR_MULT, BB_PERIOD, BB_NUM_STD, ADX_PERIOD,
           ADX_THRESHOLD, ATR_PERIOD, RSI_PERIOD, LOTS_REQUESTED,
           COST_PER_TRADE, STARTING_BALANCE)
          == (2.0, 1.5, 20, 2.0, 14, 25.0, 14, 14, 0.10, 2.0, 10_000.0))

    # [21] no optimization loops (structural: single fixed K, single
    # feature set, single distance formula, no parameter iteration)
    check("single fixed K and fixed feature set (no parameter search)",
          NEIGHBORS_K == 10 and len(FEATURES) == 12
          and FWD_WINDOWS == (12, 24))

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
        print("    python adaptive_regime_validation.py --run")
        print()
        print("Historical research only. No orders were sent.")
        print("Normalization, neighbours and distance thresholds come from")
        print("DEVELOPMENT signals only and were frozen before OOS evaluation.")
        return

    run_real_diagnostic()


if __name__ == "__main__":
    main()
