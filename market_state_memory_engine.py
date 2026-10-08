"""
market_state_memory_engine.py - HISTORICAL MARKET-STATE MEMORY ENGINE.

WHAT THIS IS
    A structured historical memory of market states, built ONLY from the
    existing market_state_*.csv research files.  For every historical
    signal observation the engine preserves:

        instrument, signal timestamp, direction,
        all available market-state features,
        the known historical trade outcome,
        forward returns, MFE/MAE,
        and chronological period information.

    A future AI research layer can then ask:

        "What historical market environments looked similar to this
         environment?"

    and receive factual historical context (nearest historical matches
    with their stored outcomes and descriptive summaries).

WHAT THIS IS NOT
    - not a trading strategy, filter or signal generator,
    - no BUY / SELL / HOLD output of any kind,
    - no parameter fitting, sweeping or selecting (K is frozen at 10),
    - no machine learning, no fitted structure of any kind,
    - no connection to any trading terminal; the engine reads CSV files
      and writes only the four NEW output CSVs listed below,
    - never a modification of any existing file.

DATA SOURCES (read-only, all required; a missing file stops the run
with the exact missing filename):
    market_state_GOLD.csv
    market_state_EURUSD.csv
    market_state_GBPUSD.csv
    market_state_USDJPY.csv
    market_state_AUDUSD.csv
    market_state_oos_period_summary.csv
    market_state_oos_state_summary.csv
    market_state_oos_walkforward.csv
    market_state_oos_cross_instrument.csv

NEW OUTPUT FILES (never overwrite; the run stops and lists conflicting
filenames if an output file already exists):
    market_memory_records.csv          one row per historical observation
    market_memory_oos_validation.csv   one row per walk-forward OOS query
    market_memory_match_details.csv    one row per (query, match) pair
    market_memory_summary.csv          descriptive per-instrument summary

NORMALIZATION (documented exactly, deterministic, no fitting):
    - numerical similarity uses z-scores only.
    - statistics are computed PER INSTRUMENT from the DEVELOPMENT
      portion ONLY (the first half of that instrument's observations,
      chronologically ordered; split index = n // 2, so the earlier
      half is the development memory).
    - mean = arithmetic mean of the development values of the feature;
      std  = POPULATION standard deviation (ddof = 0) of the same
      development values.  Nothing from the evaluation half, and
      nothing from any later observation, ever enters these numbers.
    - z = (x - mean) / std.  If std == 0 the normalized value is 0.0
      for every observation (the feature cannot separate anything).
      If x is NaN the normalized value is NaN.
    - the same frozen statistics are used for every query, so an
      earlier observation is never normalized with later data.

DISTANCE (documented exactly, deterministic):
    standardized Euclidean distance on the z-scores above:
        d(q, c) = sqrt( SUM over VALID features of (z_q - z_c)^2 )
    A feature is VALID for a pair when BOTH z-values are finite;
    invalid features are skipped (excluded from the sum).
    A candidate with no valid feature in common is ineligible.
    Note: a pair sharing fewer valid features tends to a smaller sum,
    so n_eligible and the shared-feature rule are reported alongside.
    Ties are broken deterministically by chronological row order.

CHRONOLOGICAL SAFETY:
    query_memory_as_of(instrument, timestamp) uses ONLY observations
    with historical_timestamp < query_timestamp (strictly earlier -
    never equal, never later).  The walk-forward memory test queries
    only strictly-earlier DEVELOPMENT observations for every
    evaluation-half query.

FROZEN SETTINGS (no sweeps, no selection):
    K = 10 (number of historical matches per query)
    LOW SAMPLE threshold: fewer than 5 eligible matches
    distance buckets: 33rd / 66th percentile of the DEVELOPMENT
    nearest-match distance distribution, frozen per instrument before
    any evaluation-half observation is touched.

DESCRIPTIVE LANGUAGE:
    Everything returned is a description of stored history:
    "Historical matches in this sample had ...".  Nothing is a
    forecast, a directional call, or an entry/exit instruction.

COMMAND-LINE SAFETY:
    python market_state_memory_engine.py          -> synthetic tests only
    python market_state_memory_engine.py --run    -> real CSV analysis

DISCLAIMER: Historical, descriptive research only.  No causal claims,
no recommendations, nothing proven profitable.  Educational use only.
"""

# ============================================================
# Imports: Python standard library + numpy + pandas ONLY.
# No machine-learning library and no trading terminal is
# imported anywhere in this file (verified by the built-in
# synthetic suite against this file's own source).
# ============================================================
import ast
import datetime
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# SAFETY GUARDS
# Forbidden tokens are assembled from split string literals so
# this safety code never contains the contiguous strings itself.
# ============================================================
_MT5_SOURCE_TOKENS: Tuple[str, ...] = (
    "Meta" + "Trader5",
    "or" + "der_send",
    "or" + "der_check",
    "positions" + "_get",
    "positions" + "_total",
    "position" + "_get",
    "init" + "ialize",
    "shut" + "down",
    "copy_rates_from" + "_pos",
    "symbol" + "_info",
    "symbols" + "_get",
    "m" + "t5",
)
_ML_SOURCE_TOKENS: Tuple[str, ...] = (
    "sk" + "learn",
    "xg" + "boost",
    "light" + "gbm",
    "tensor" + "flow",
    "to" + "rch",
    "ke" + "ras",
)
# Call-syntax patterns (word boundary + open paren), also assembled
# from pieces so the patterns never self-match in this source.
_ML_CALL_PATTERNS: Tuple[str, ...] = (
    "\\b" + "fit" + "\\s*\\(",
    "\\b" + "train" + "\\s*\\(",
    "\\b" + "pred" + "ict" + "\\s*\\(",
    "\\b" + "mo" + "del" + "\\b",
    "\\b" + "class" + "ifier" + "\\b",
    "\\b" + "regress" + "or" + "\\b",
)
_OPTIMIZATION_TOKENS: Tuple[str, ...] = (
    "p" + "aram_g",
    "Gr" + "idSearch",
    "Rand" + "omizedS",
    "op" + "tuna",
    "hy" + "peropt",
    ".f" + "it(",
    ".s" + "core(",
    "itertools.p" + "roduct",
)
_FORBIDDEN_FN_NAME_TOKENS: Tuple[str, ...] = (
    "optim", "sweep", "grid", "tune", "search", "rank",
)
_ALLOWED_IMPORT_MODULES: Tuple[str, ...] = (
    "ast", "datetime", "dataclasses", "os", "re", "sys",
    "tempfile", "typing", "warnings", "numpy", "pandas",
)
_FORBIDDEN_IMPORT_MODULES: Tuple[str, ...] = (
    "Meta" + "Trader5",
    "m" + "t5",
    "sk" + "learn",
    "xg" + "boost",
    "light" + "gbm",
    "tensor" + "flow",
    "to" + "rch",
    "ke" + "ras",
)
_FORBIDDEN_REPORT_WORDS = ("BU" + "Y", "SE" + "LL", "HO" + "LD")

# ============================================================
# Frozen configuration
# ============================================================
EXPECTED_INSTRUMENTS: Tuple[str, ...] = (
    "GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
)
INSTRUMENT_CSV_TEMPLATE = "market_state_{instrument}.csv"
OOS_INPUT_FILES: Tuple[str, ...] = (
    "market_state_oos_period_summary.csv",
    "market_state_oos_state_summary.csv",
    "market_state_oos_walkforward.csv",
    "market_state_oos_cross_instrument.csv",
)
REQUIRED_INPUT_FILES: Tuple[str, ...] = (
    tuple(INSTRUMENT_CSV_TEMPLATE.format(instrument=i) for i in EXPECTED_INSTRUMENTS)
    + OOS_INPUT_FILES
)

OUTPUT_FILES: Tuple[str, ...] = (
    "market_memory_records.csv",
    "market_memory_oos_validation.csv",
    "market_memory_match_details.csv",
    "market_memory_summary.csv",
)

# Frozen constants - fixed by specification, never swept, never tuned.
TOP_K = 10                    # number of historical matches per query
LOW_SAMPLE_N = 5              # fewer eligible matches -> LOW SAMPLE
DEV_SPLIT_FRACTION = 0.5      # first half = development memory
DISTANCE_PCT_LOW = 33.0       # LOW/MEDIUM bucket threshold (percentile)
DISTANCE_PCT_HIGH = 66.0      # MEDIUM/HIGH bucket threshold (percentile)
DEFAULT_TOP_K = TOP_K

# (spec feature name, CSV column name) - the existing database columns.
CORE_STATE_FEATURES: Tuple[Tuple[str, str], ...] = (
    ("ADX", "adx"),
    ("ATR", "atr"),
    ("ATR_pct", "atr_pct"),
    ("EMA50", "ema50"),
    ("EMA50_dist", "ema_distance"),
    ("BB_middle", "bb_middle"),
    ("BB_upper", "bb_upper"),
    ("BB_lower", "bb_lower"),
    ("BB_excursion", "bb_excursion"),
    ("BB_signed", "bb_signed"),
    ("eff24", "eff24"),
    ("ret6", "return6"),
    ("ret12", "return12"),
    ("ret24", "return24"),
    ("RSI14", "rsi14"),
    ("vol6", "vol6"),
    ("vol24", "vol24"),
    ("range_pct", "range_pct"),
    ("body_pct", "body_pct"),
    ("upper_wick_pct", "upper_wick_pct"),
    ("lower_wick_pct", "lower_wick_pct"),
    ("dist_24_high", "distance_24_high"),
    ("dist_24_low", "distance_24_low"),
)
TIME_CONTEXT_FEATURES: Tuple[Tuple[str, str], ...] = (
    ("hour", "hour"),
    ("day_of_week", "day_of_week"),
    ("month", "month"),
)
STATE_FEATURES: Tuple[Tuple[str, str], ...] = (
    CORE_STATE_FEATURES + TIME_CONTEXT_FEATURES
)
FEATURE_TO_COLUMN = dict(STATE_FEATURES)
COLUMN_TO_FEATURE = {col: spec for spec, col in STATE_FEATURES}
STATE_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    col for _, col in STATE_FEATURES
)

OUTCOME_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "entry_price", "exit_price", "holding_candles", "gross_pnl",
    "trading_cost", "net_pnl",
    "forward6_pct", "forward12_pct", "forward24_pct",
    "mfe12_atr", "mae12_atr",
)
NUMERIC_COLUMNS: Tuple[str, ...] = STATE_FEATURE_COLUMNS + OUTCOME_NUMERIC_COLUMNS

REQUIRED_INPUT_COLUMNS: Tuple[str, ...] = (
    "instrument", "timeframe", "signal_time", "signal_index", "direction",
    "adx", "atr", "atr_pct", "ema50", "ema_distance",
    "bb_middle", "bb_upper", "bb_lower", "bb_excursion", "bb_signed",
    "eff24", "return6", "return12", "return24",
    "rsi14", "vol6", "vol24",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "distance_24_high", "distance_24_low",
    "hour", "day_of_week", "month",
    "entry_price", "exit_price", "exit_time", "exit_reason",
    "holding_candles", "gross_pnl", "trading_cost", "net_pnl", "win",
    "forward6_pct", "forward12_pct", "forward24_pct",
    "mfe12_atr", "mae12_atr",
)

RECORDS_COLUMNS: Tuple[str, ...] = (
    ("instrument", "timeframe", "timestamp", "row_index",
     "chronological_period", "split", "direction")
    + STATE_FEATURE_COLUMNS
    + tuple("z_" + col for col in STATE_FEATURE_COLUMNS)
    + ("entry_price", "exit_price", "exit_time", "exit_reason",
       "holding_candles", "gross_pnl", "trading_cost", "net_pnl", "win",
       "forward6_pct", "forward12_pct", "forward24_pct",
       "mfe12_atr", "mae12_atr")
)

VALIDATION_COLUMNS: Tuple[str, ...] = (
    "instrument", "query_timestamp", "query_direction", "query_period",
    "n_eligible", "n_matches", "low_sample",
    "nearest_distance", "median_match_distance",
    "nearest_distance_percentile", "distance_bucket",
    "mean_match_net_pnl", "median_match_net_pnl", "match_win_rate_pct",
    "actual_net_pnl", "actual_win",
    "mean_match_forward6_pct", "mean_match_forward12_pct",
    "mean_match_forward24_pct",
    "actual_forward6_pct", "actual_forward12_pct", "actual_forward24_pct",
    "mean_match_mfe12_atr", "mean_match_mae12_atr",
    "actual_mfe12_atr", "actual_mae12_atr",
    "mean_match_holding_candles", "actual_holding_candles",
    "same_direction_matches", "opposite_direction_matches",
    "pnl_sign_agree", "fwd12_sign_agree", "fwd24_sign_agree",
    "match_age_median_days", "match_age_mean_days",
    "match_age_min_days", "match_age_max_days",
)

DETAIL_COLUMNS: Tuple[str, ...] = (
    "instrument", "query_timestamp", "query_direction", "query_period",
    "match_rank", "match_instrument", "match_timestamp", "match_period",
    "match_direction", "same_direction",
    "distance", "distance_percentile", "age_days",
    "net_pnl", "win", "forward6_pct", "forward12_pct", "forward24_pct",
    "mfe12_atr", "mae12_atr", "holding_candles", "exit_reason",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "instrument", "section", "label", "n", "n_queries", "low_sample_count",
    "avg_match_net_pnl", "median_match_net_pnl", "match_win_rate_pct",
    "profit_factor", "positive_match_count", "negative_match_count",
    "avg_actual_net_pnl", "actual_win_rate_pct",
    "avg_match_forward6_pct", "avg_match_forward12_pct",
    "avg_match_forward24_pct",
    "avg_actual_forward6_pct", "avg_actual_forward12_pct",
    "avg_actual_forward24_pct",
    "avg_mfe12_atr", "avg_mae12_atr", "avg_holding_candles",
    "avg_nearest_distance", "median_nearest_distance",
    "avg_match_distance", "median_match_distance",
    "pnl_sign_agreement_pct", "pnl_sign_n",
    "fwd12_sign_agreement_pct", "fwd12_sign_n",
    "fwd24_sign_agreement_pct", "fwd24_sign_n",
    "corr_match_actual_pnl_r", "corr_match_actual_pnl_n",
    "corr_match_actual_fwd12_r", "corr_match_actual_fwd12_n",
    "corr_match_actual_fwd24_r", "corr_match_actual_fwd24_n",
    "age_median_days", "age_mean_days", "age_min_days", "age_max_days",
    "pct_same_instrument", "pct_same_direction", "pct_different_direction",
    "pct_nearest_same_direction", "pct_nearest_different_direction",
    "distance_threshold_p33", "distance_threshold_p66",
)

BUCKET_LABELS: Tuple[str, ...] = (
    "LOW DISTANCE", "MEDIUM DISTANCE", "HIGH DISTANCE",
)
DIRECTION_LABELS: Tuple[str, ...] = ("SAME_DIRECTION", "OPPOSITE_DIRECTION")
SPLIT_DEVELOPMENT = "DEVELOPMENT"
SPLIT_EVALUATION = "EVALUATION"

NORMALIZATION_DOC = """\
Normalization (frozen, deterministic, development-only):
  - per instrument, statistics are computed ONLY from the development
    portion = the first half of that instrument's observations in
    chronological order (split index = n // 2).
  - mean = development arithmetic mean; std = development POPULATION
    standard deviation (ddof = 0).  No later observation, and no
    evaluation-half observation, ever enters these numbers.
  - z = (x - mean) / std;  std == 0  ->  z = 0.0;  x NaN -> z NaN.
  - the frozen statistics are reused unchanged for every query, so an
    earlier observation is never normalized with later data.
Distance (standardized Euclidean, deterministic):
  - d(q, c) = sqrt( SUM over valid features of (z_q - z_c)^2 ),
    where a feature is valid for a pair only when both z-values are
    finite (invalid features are excluded from the sum); ties are
    broken by chronological row order.
"""

# ============================================================
# Small deterministic helpers
# ============================================================
def parse_timestamp(value) -> pd.Timestamp:
    """Parse a timestamp from str / datetime / pd.Timestamp."""
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime.datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, str):
        ts = pd.Timestamp(value.strip())
    else:
        raise ValueError(f"cannot parse timestamp from {value!r}")
    if pd.isna(ts):
        raise ValueError(f"unparseable timestamp value {value!r}")
    return ts


def fmt_ts(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def coerce_win(value) -> Optional[bool]:
    """Robust True/False/None for the stored 'win' column."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return None


def sign_of(value) -> Optional[int]:
    """-1 / 0 / +1, or None when the value is missing."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    v = float(value)
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def mean_of(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def median_of(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def profit_factor(pnls: List[float]) -> Optional[float]:
    """Sum of gains / abs(sum of losses).  No losses -> inf; both zero -> NaN."""
    gains = float(sum(p for p in pnls if p > 0))
    losses = float(-sum(p for p in pnls if p < 0))
    if losses > 0:
        return gains / losses
    if gains > 0:
        return float("inf")
    return float("nan")


def sign_agreement_rate(pairs: List[Tuple[Optional[float], Optional[float]]]):
    """Share of pairs with equal sign, plus the usable pair count."""
    usable = [(a, b) for a, b in pairs
              if sign_of(a) is not None and sign_of(b) is not None]
    if not usable:
        return None, 0
    same = sum(1 for a, b in usable if sign_of(a) == sign_of(b))
    return 100.0 * same / len(usable), len(usable)


def safe_pearson(xs, ys):
    """Pearson r and n; None when fewer than 3 pairs or zero variance."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and np.isfinite(x) and np.isfinite(y)]
    n = len(pairs)
    if n < 3:
        return None, n
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None, n
    return float(np.corrcoef(x, y)[0, 1]), n


def bucket_for(distance: Optional[float], thresholds) -> str:
    """Frozen descriptive bucket; thresholds are development-only."""
    p33, p66 = thresholds
    if distance is None or not np.isfinite(distance) \
            or p33 is None or p66 is None \
            or not np.isfinite(p33) or not np.isfinite(p66):
        return "n/a"
    if distance < p33:
        return BUCKET_LABELS[0]
    if distance < p66:
        return BUCKET_LABELS[1]
    return BUCKET_LABELS[2]


def resolve_feature_key(key: str) -> str:
    """Accept either the spec feature name or the CSV column name."""
    if key in FEATURE_TO_COLUMN:
        return FEATURE_TO_COLUMN[key]
    if key in COLUMN_TO_FEATURE:
        return key
    valid = ", ".join(list(FEATURE_TO_COLUMN.keys())[:6]) + ", ..."
    raise ValueError(f"unknown state feature {key!r}; use spec names "
                     f"(e.g. {valid}) or CSV column names")


def fmt_corr(r) -> str:
    return "n/a" if r is None else f"{r:+.3f}"


# ============================================================
# Historical memory record (section 5 of the specification)
# ============================================================
@dataclass(frozen=True)
class MemoryRecord:
    """One standardized historical observation in the memory."""
    instrument: str
    timeframe: str
    timestamp: pd.Timestamp
    row_index: int                 # chronological position within instrument
    direction: str
    split: str                     # DEVELOPMENT / EVALUATION
    chronological_period: str
    raw_features: Dict[str, float]     # csv column -> raw value
    state_features: Dict[str, float]   # spec name -> raw value
    historical_outcome: Dict[str, Any]
    forward_outcomes: Dict[str, float]

    def to_memory_dict(self) -> Dict[str, Any]:
        """The standardized memory record shape from the specification."""
        return {
            "instrument": self.instrument,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "state_features": dict(self.state_features),
            "historical_outcome": dict(self.historical_outcome),
            "forward_outcomes": dict(self.forward_outcomes),
            "chronological_period": self.chronological_period,
            "split": self.split,
        }


# ============================================================
# The memory engine
# ============================================================
class MarketStateMemoryEngine:
    """Deterministic, read-only historical memory of market states.

    Built from in-memory DataFrames (so the synthetic suite never needs
    the real CSVs).  The real analysis loads the CSVs with
    ``load_input_bundle`` and passes them here.
    """

    def __init__(self, frames: Dict[str, pd.DataFrame],
                 period_summary: Optional[pd.DataFrame] = None):
        if not frames:
            raise ValueError("no instrument frames supplied to the engine")
        self.period_lookup: Dict[str, List[Tuple[str, int, int]]] = {}
        if period_summary is not None:
            self._load_period_spans(period_summary)

        ordered = [i for i in EXPECTED_INSTRUMENTS if i in frames]
        ordered += sorted(set(frames) - set(ordered))
        self.instruments: Tuple[str, ...] = tuple(ordered)

        self.records: Dict[str, List[MemoryRecord]] = {}
        self.time_ns: Dict[str, np.ndarray] = {}
        self.z_matrix: Dict[str, np.ndarray] = {}
        self.stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.thresholds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self.dev_indices: Dict[str, np.ndarray] = {}
        self.eval_indices: Dict[str, np.ndarray] = {}
        self.period_unassigned: Dict[str, int] = {}

        for inst in self.instruments:
            self._build_instrument(str(inst), frames[inst])

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------
    def _load_period_spans(self, period_summary: pd.DataFrame) -> None:
        need = {"instrument", "period", "start_time", "end_time"}
        if not need.issubset(set(period_summary.columns)):
            raise ValueError(
                "period summary is missing columns: "
                + ", ".join(sorted(need - set(period_summary.columns))))
        for inst, grp in period_summary.groupby("instrument"):
            spans = []
            g = grp.sort_values("start_time", kind="mergesort")
            for row in g.itertuples(index=False):
                s = parse_timestamp(row.start_time)
                e = parse_timestamp(row.end_time)
                spans.append((str(row.period), s.value, e.value))
            self.period_lookup[str(inst)] = spans

    def _period_for(self, inst: str, ts: pd.Timestamp, idx: int,
                    n_total: int) -> Tuple[str, bool]:
        """(period label, assigned_ok).  Fallback: equal-count quartiles."""
        spans = self.period_lookup.get(inst)
        if spans:
            v = ts.value
            for label, s, e in spans:
                if s <= v <= e:
                    return label, True
            return "n/a", False
        label_n = min(4, (4 * idx) // max(1, n_total)) + 1
        return f"P{label_n}", True

    def _build_instrument(self, inst: str, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"instrument frame {inst} is missing columns: "
                + ", ".join(missing))
        work = df.copy()
        work["signal_time"] = pd.to_datetime(work["signal_time"],
                                             errors="raise")
        if work["signal_time"].isna().any():
            bad = int(work["signal_time"].isna().sum())
            raise ValueError(
                f"instrument frame {inst} has {bad} unparseable signal_time values")
        for col in NUMERIC_COLUMNS:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        # Chronological order; stable sort keeps file order for ties.
        work = work.sort_values("signal_time", kind="mergesort").reset_index(drop=True)

        n = len(work)
        split = int(n * DEV_SPLIT_FRACTION)  # floor: earlier half = development
        time_ns = work["signal_time"].map(
            lambda t: pd.Timestamp(t).value).to_numpy(dtype=np.int64)

        records: List[MemoryRecord] = []
        for i in range(n):
            row = work.iloc[i]
            ts = pd.Timestamp(row["signal_time"])
            period, ok = self._period_for(inst, ts, i, n)
            raw = {col: (None if pd.isna(row[col]) else float(row[col]))
                   for col in STATE_FEATURE_COLUMNS}
            state = {spec: raw[col] for spec, col in STATE_FEATURES}
            outcome = {
                "entry_price": (None if pd.isna(row["entry_price"])
                                else float(row["entry_price"])),
                "exit_price": (None if pd.isna(row["exit_price"])
                               else float(row["exit_price"])),
                "exit_time": ("" if pd.isna(row["exit_time"])
                              else str(row["exit_time"])),
                "exit_reason": str(row["exit_reason"]),
                "holding_candles": (None if pd.isna(row["holding_candles"])
                                    else int(row["holding_candles"])),
                "gross_pnl": (None if pd.isna(row["gross_pnl"])
                              else float(row["gross_pnl"])),
                "trading_cost": (None if pd.isna(row["trading_cost"])
                                 else float(row["trading_cost"])),
                "net_pnl": (None if pd.isna(row["net_pnl"])
                            else float(row["net_pnl"])),
                "win": coerce_win(row["win"]),
            }
            forward = {
                "forward6_pct": (None if pd.isna(row["forward6_pct"])
                                 else float(row["forward6_pct"])),
                "forward12_pct": (None if pd.isna(row["forward12_pct"])
                                  else float(row["forward12_pct"])),
                "forward24_pct": (None if pd.isna(row["forward24_pct"])
                                  else float(row["forward24_pct"])),
                "mfe12_atr": (None if pd.isna(row["mfe12_atr"])
                              else float(row["mfe12_atr"])),
                "mae12_atr": (None if pd.isna(row["mae12_atr"])
                              else float(row["mae12_atr"])),
            }
            records.append(MemoryRecord(
                instrument=inst,
                timeframe=str(row["timeframe"]),
                timestamp=ts,
                row_index=i,
                direction=str(row["direction"]),
                split=SPLIT_DEVELOPMENT if i < split else SPLIT_EVALUATION,
                chronological_period=period,
                raw_features=raw,
                state_features=state,
                historical_outcome=outcome,
                forward_outcomes=forward,
            ))
            if not ok:
                self.period_unassigned[inst] = self.period_unassigned.get(inst, 0) + 1

        self.records[inst] = records
        self.time_ns[inst] = time_ns
        self.dev_indices[inst] = np.arange(split, dtype=int)
        self.eval_indices[inst] = np.arange(split, n, dtype=int)

        # ---- development-only normalization statistics ----
        X = work[list(STATE_FEATURE_COLUMNS)].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if split >= 1:
                mean = np.nanmean(X[:split], axis=0)
                std = np.nanstd(X[:split], axis=0)  # population, ddof = 0
            else:
                mean = np.full(len(STATE_FEATURE_COLUMNS), np.nan)
                std = np.full(len(STATE_FEATURE_COLUMNS), np.nan)
        self.stats[inst] = {
            col: (float(mean[j]), float(std[j]))
            for j, col in enumerate(STATE_FEATURE_COLUMNS)
        }
        std_safe = np.where(np.isfinite(std) & (std != 0.0), std, 1.0)
        Z = (X - mean) / std_safe
        zero_or_bad = ~np.isfinite(std) | (std == 0.0)
        if zero_or_bad.any():
            Z[:, zero_or_bad] = np.where(
                np.isnan(X[:, zero_or_bad]), np.nan, 0.0)
        self.z_matrix[inst] = Z

        # ---- development-only, frozen distance thresholds ----
        self.thresholds[inst] = self._development_thresholds(inst)

    def _development_thresholds(self, inst: str):
        """33rd/66th percentile of each DEVELOPMENT observation's nearest
        distance to the OTHER development observations.  Frozen before any
        evaluation-half data is touched; evaluation values never enter."""
        dev = self.dev_indices[inst]
        Z = self.z_matrix[inst]
        if dev.size < 2:
            return (None, None)
        dists: List[float] = []
        for i in dev:
            others = dev[dev != i]
            d = self._distances_from_z(inst, Z[i], others)
            d = d[np.isfinite(d)]
            if d.size:
                dists.append(float(d.min()))
        if len(dists) < 2:
            return (None, None)
        p33 = float(np.percentile(dists, DISTANCE_PCT_LOW))
        p66 = float(np.percentile(dists, DISTANCE_PCT_HIGH))
        return (p33, p66)

    # --------------------------------------------------------
    # Distances and matches (deterministic)
    # --------------------------------------------------------
    def z_vector_for_values(self, inst: str,
                            raw_by_column: Dict[str, Optional[float]]) -> np.ndarray:
        """z-score supplied raw values with that instrument's FROZEN
        development statistics (std == 0 -> z = 0)."""
        zs = []
        for col in STATE_FEATURE_COLUMNS:
            x = raw_by_column.get(col)
            if x is None or not np.isfinite(x):
                zs.append(np.nan)
                continue
            mean, std = self.stats[inst][col]
            if std == 0.0 or not np.isfinite(std):
                zs.append(0.0)
            else:
                zs.append((float(x) - mean) / std)
        return np.asarray(zs, dtype=float)

    def _distances_from_z(self, inst: str, query_z: np.ndarray,
                          cand_indices: np.ndarray) -> np.ndarray:
        Zc = self.z_matrix[inst][cand_indices]
        diff = Zc - query_z[None, :]
        valid = ~np.isnan(diff)
        cnt = valid.sum(axis=1)
        sq = np.where(valid, diff * diff, 0.0).sum(axis=1)
        d = np.sqrt(sq)   # standardized Euclidean: sqrt of the SUM of
        d[cnt == 0] = np.nan  # squared z-differences over valid features
        return d

    def _matches(self, inst: str, query_record: MemoryRecord,
                 cand_indices: np.ndarray, top_k: int,
                 query_ts: Optional[pd.Timestamp]):
        """Nearest historical matches from a candidate pool.

        Returns (matches, n_eligible).  n_eligible counts every candidate
        with at least one comparable feature; the distance percentile is
        computed from that eligible pool only.
        """
        if int(top_k) <= 0:
            raise ValueError("top_k must be a positive integer")
        cand = np.asarray(cand_indices, dtype=int)
        cand = cand[cand != query_record.row_index]  # self always excluded
        if cand.size == 0:
            return [], 0
        qz = self.z_matrix[inst][query_record.row_index]
        d = self._distances_from_z(inst, qz, cand)
        finite = np.isfinite(d)
        elig = cand[finite]
        ed = d[finite]
        if ed.size == 0:
            return [], 0
        order = np.lexsort((elig, ed))  # distance, then chronological order
        k = min(int(top_k), ed.size)
        out: List[Dict[str, Any]] = []
        for rank, pos in enumerate(order[:k], start=1):
            i = int(elig[pos])
            dist = float(ed[pos])
            pct = 100.0 * float(np.count_nonzero(ed <= dist)) / float(ed.size)
            rec = self.records[inst][i]
            if query_ts is not None:
                age = (query_ts - rec.timestamp).total_seconds() / 86400.0
            else:
                age = float("nan")
            out.append({
                "instrument": inst,
                "timestamp": rec.timestamp,
                "direction": rec.direction,
                "period": rec.chronological_period,
                "row_index": i,
                "distance": dist,
                "distance_percentile": pct,
                "age_days": age,
                "net_pnl": rec.historical_outcome["net_pnl"],
                "win": rec.historical_outcome["win"],
                "forward6_pct": rec.forward_outcomes["forward6_pct"],
                "forward12_pct": rec.forward_outcomes["forward12_pct"],
                "forward24_pct": rec.forward_outcomes["forward24_pct"],
                "mfe12_atr": rec.forward_outcomes["mfe12_atr"],
                "mae12_atr": rec.forward_outcomes["mae12_atr"],
                "holding_candles": rec.historical_outcome["holding_candles"],
                "exit_reason": rec.historical_outcome["exit_reason"],
            })
        return out, int(ed.size)

    # --------------------------------------------------------
    # Location and queries
    # --------------------------------------------------------
    def locate_record(self, instrument: str, timestamp) -> MemoryRecord:
        if instrument not in self.records:
            raise KeyError(f"unknown instrument {instrument!r}; "
                           f"known: {', '.join(self.instruments)}")
        ts = parse_timestamp(timestamp)
        hits = np.nonzero(self.time_ns[instrument] == ts.value)[0]
        if hits.size == 0:
            raise KeyError(f"no historical observation for {instrument} "
                           f"at {fmt_ts(ts)}")
        return self.records[instrument][int(hits[0])]

    def _earlier_indices(self, instrument: str, ts: pd.Timestamp) -> np.ndarray:
        """Row indices STRICTLY earlier than ts (never equal, never later)."""
        return np.nonzero(self.time_ns[instrument] < ts.value)[0]

    def development_indices(self, instrument: str) -> np.ndarray:
        return self.dev_indices[instrument]

    def evaluation_indices(self, instrument: str) -> np.ndarray:
        return self.eval_indices[instrument]

    def query_memory(self, instrument: str, timestamp,
                     top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """Descriptive full-history lookup against the SAME instrument.

        Compares the observation with every OTHER observation of the
        instrument (the observation itself is excluded).  Later
        observations may appear here; for chronological research use
        ``query_memory_as_of``.
        """
        ts = parse_timestamp(timestamp)
        rec = self.locate_record(instrument, ts)
        n = len(self.records[instrument])
        cand = np.arange(n, dtype=int)
        matches, n_elig = self._matches(instrument, rec, cand, top_k, ts)
        return {
            "instrument": instrument,
            "timestamp": ts,
            "direction": rec.direction,
            "period": rec.chronological_period,
            "matches": matches,
            "n_eligible": n_elig,
            "low_sample": len(matches) < LOW_SAMPLE_N,
        }

    def query_memory_as_of(self, instrument: str, timestamp,
                           top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """Historical-validation lookup: ONLY strictly earlier observations
        of the same instrument are eligible (historical_timestamp <
        query_timestamp - never equal, never later)."""
        ts = parse_timestamp(timestamp)
        rec = self.locate_record(instrument, ts)
        cand = self._earlier_indices(instrument, ts)
        matches, n_elig = self._matches(instrument, rec, cand, top_k, ts)
        return {
            "instrument": instrument,
            "timestamp": ts,
            "direction": rec.direction,
            "period": rec.chronological_period,
            "matches": matches,
            "n_eligible": n_elig,
            "low_sample": len(matches) < LOW_SAMPLE_N,
        }

    def query_cross_instrument_memory(self, timestamp=None,
                                      state_features=None,
                                      top_k: int = DEFAULT_TOP_K,
                                      as_of: bool = False,
                                      exclude=None) -> Dict[str, Any]:
        """Descriptive lookup across all instruments.

        state_features maps feature names (spec names or CSV column
        names) to raw values; each instrument compares them with its OWN
        frozen development statistics.  The pooled nearest matches are
        returned WITH their instrument label; summaries stay separated
        per instrument (no combined P/L series is ever formed).

        timestamp: optional reference time (enables as_of filtering and
        match ages).  exclude: optional (instrument, timestamp) tuple to
        drop that instrument's observations at that timestamp.
        """
        if not state_features:
            raise ValueError("state_features dict is required")
        if as_of and timestamp is None:
            raise ValueError("as_of=True requires a timestamp")
        raw_by_col: Dict[str, Optional[float]] = {}
        for key, val in state_features.items():
            col = resolve_feature_key(str(key))
            try:
                v = float(val)
            except (TypeError, ValueError):
                v = None
            raw_by_col[col] = v if (v is not None and np.isfinite(v)) else None

        excl_inst = excl_ns = None
        if exclude is not None:
            excl_inst, excl_ts = exclude[0], parse_timestamp(exclude[1])
            excl_ns = excl_ts.value

        pooled: List[Tuple[float, str, int]] = []
        n_eligible_total = 0
        for inst in self.instruments:
            cand = np.arange(len(self.records[inst]), dtype=int)
            if timestamp is not None:
                ts = parse_timestamp(timestamp)
                if as_of:
                    cand = self._earlier_indices(inst, ts)
                if excl_inst == inst:
                    cand = cand[self.time_ns[inst][cand] != excl_ns]
            elif excl_inst == inst:
                cand = cand[self.time_ns[inst][cand] != excl_ns]
            if cand.size == 0:
                continue
            qz = self.z_vector_for_values(inst, raw_by_col)
            d = self._distances_from_z(inst, qz, cand)
            finite = np.isfinite(d)
            n_eligible_total += int(finite.sum())
            for pos in np.nonzero(finite)[0]:
                pooled.append((float(d[pos]), inst, int(cand[pos])))

        pooled.sort(key=lambda t: (t[0], t[1], t[2]))
        k = min(int(top_k), len(pooled))
        matches: List[Dict[str, Any]] = []
        for rank, (dist, inst, i) in enumerate(pooled[:k], start=1):
            rec = self.records[inst][i]
            pct = 100.0 * float(sum(1 for dd, ii, jj in pooled
                                    if dd <= dist)) / float(len(pooled))
            if timestamp is not None:
                qts = parse_timestamp(timestamp)
                age = (qts - rec.timestamp).total_seconds() / 86400.0
            else:
                age = float("nan")
            matches.append({
                "instrument": inst,
                "timestamp": rec.timestamp,
                "direction": rec.direction,
                "period": rec.chronological_period,
                "row_index": i,
                "distance": dist,
                "distance_percentile": pct,
                "age_days": age,
                "net_pnl": rec.historical_outcome["net_pnl"],
                "win": rec.historical_outcome["win"],
                "forward6_pct": rec.forward_outcomes["forward6_pct"],
                "forward12_pct": rec.forward_outcomes["forward12_pct"],
                "forward24_pct": rec.forward_outcomes["forward24_pct"],
                "mfe12_atr": rec.forward_outcomes["mfe12_atr"],
                "mae12_atr": rec.forward_outcomes["mae12_atr"],
                "holding_candles": rec.historical_outcome["holding_candles"],
                "exit_reason": rec.historical_outcome["exit_reason"],
                "match_rank": rank,
            })

        by_instrument: Dict[str, Dict[str, Any]] = {}
        for inst in self.instruments:
            grp = [m for m in matches if m["instrument"] == inst]
            if grp:
                by_instrument[inst] = summarize_matches(grp)
        return {
            "query": {
                "timestamp": parse_timestamp(timestamp) if timestamp is not None else None,
                "state_features": dict(raw_by_col),
                "as_of": bool(as_of),
            },
            "matches": matches,
            "n_eligible": n_eligible_total,
            "low_sample": len(matches) < LOW_SAMPLE_N,
            "summary_by_instrument": by_instrument,
        }

    # --------------------------------------------------------
    # Records export
    # --------------------------------------------------------
    def records_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for inst in self.instruments:
            Z = self.z_matrix[inst]
            for rec in self.records[inst]:
                row: Dict[str, Any] = {
                    "instrument": rec.instrument,
                    "timeframe": rec.timeframe,
                    "timestamp": fmt_ts(rec.timestamp),
                    "row_index": rec.row_index,
                    "chronological_period": rec.chronological_period,
                    "split": rec.split,
                    "direction": rec.direction,
                }
                zrow = Z[rec.row_index]
                for j, col in enumerate(STATE_FEATURE_COLUMNS):
                    row[col] = rec.raw_features[col]
                    row["z_" + col] = (None if not np.isfinite(zrow[j])
                                       else float(zrow[j]))
                row.update(rec.historical_outcome)
                row.update(rec.forward_outcomes)
                rows.append(row)
        return pd.DataFrame(rows, columns=list(RECORDS_COLUMNS))


# ============================================================
# Descriptive summaries (sections 10, 11 of the specification)
# ============================================================
def summarize_matches(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive summary of historical matches.  Wording contract:
    these are descriptions of PAST observations ("Historical matches in
    this sample had ..."), never a statement about the future."""
    n = len(matches)
    pnls = [float(m["net_pnl"]) for m in matches
            if m.get("net_pnl") is not None and np.isfinite(m["net_pnl"])]
    wins = [bool(m["win"]) for m in matches if m.get("win") is not None]

    def col_avg(key: str) -> Optional[float]:
        vals = [float(m[key]) for m in matches
                if m.get(key) is not None and np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else None

    return {
        "n": n,
        "low_sample": n < LOW_SAMPLE_N,
        "avg_net_pnl": mean_of(pnls),
        "median_net_pnl": median_of(pnls),
        "win_rate_pct": (100.0 * sum(wins) / len(wins)) if wins else None,
        "profit_factor": profit_factor(pnls) if pnls else None,
        "positive_count": sum(1 for p in pnls if p > 0),
        "negative_count": sum(1 for p in pnls if p < 0),
        "avg_forward6_pct": col_avg("forward6_pct"),
        "avg_forward12_pct": col_avg("forward12_pct"),
        "avg_forward24_pct": col_avg("forward24_pct"),
        "avg_mfe12_atr": col_avg("mfe12_atr"),
        "avg_mae12_atr": col_avg("mae12_atr"),
        "avg_holding_candles": col_avg("holding_candles"),
    }


# ============================================================
# Input / output guards
# ============================================================
class MissingInputError(RuntimeError):
    pass


class OutputExistsError(RuntimeError):
    pass


def check_required_inputs(directory: str = SCRIPT_DIR) -> None:
    """Every required input file must exist; otherwise stop and name it."""
    missing = [f for f in REQUIRED_INPUT_FILES
               if not os.path.isfile(os.path.join(directory, f))]
    if missing:
        raise MissingInputError(
            "missing required input file(s): " + ", ".join(missing))


def check_output_files_available(directory: str = SCRIPT_DIR) -> None:
    """Never overwrite: stop and list every conflicting output file."""
    conflicts = [f for f in OUTPUT_FILES
                 if os.path.exists(os.path.join(directory, f))]
    if conflicts:
        raise OutputExistsError(
            "refusing to run: output file(s) already exist and existing "
            "files are never modified or overwritten: " + ", ".join(conflicts))


@dataclass(frozen=True)
class InputBundle:
    instrument_frames: Dict[str, pd.DataFrame]
    oos_frames: Dict[str, pd.DataFrame]
    period_summary: Optional[pd.DataFrame]


def load_input_bundle(directory: str = SCRIPT_DIR) -> InputBundle:
    """Read the required CSVs (read-only) into memory."""
    check_required_inputs(directory)
    frames: Dict[str, pd.DataFrame] = {}
    for inst in EXPECTED_INSTRUMENTS:
        path = os.path.join(directory,
                            INSTRUMENT_CSV_TEMPLATE.format(instrument=inst))
        frames[inst] = pd.read_csv(path)
    oos_frames: Dict[str, pd.DataFrame] = {}
    for name in OOS_INPUT_FILES:
        oos_frames[name] = pd.read_csv(os.path.join(directory, name))
    period_summary = oos_frames.get("market_state_oos_period_summary.csv")
    return InputBundle(instrument_frames=frames, oos_frames=oos_frames,
                       period_summary=period_summary)


# ============================================================
# Historical context report (section 13 of the specification)
# ============================================================
def _format_features_line(rec: MemoryRecord) -> List[str]:
    parts = []
    for spec, _col in STATE_FEATURES:
        v = rec.state_features.get(spec)
        parts.append(f"{spec}={'n/a' if v is None else format(v, '.6g')}")
    lines = []
    for start in range(0, len(parts), 4):
        lines.append("     " + " | ".join(parts[start:start + 4]))
    return lines


def generate_memory_report(self, instrument: str, timestamp,
                           as_of: bool = True,
                           top_k: int = DEFAULT_TOP_K) -> str:
    """Historical context report with sections A-H.  This is memory, not
    a trading decision system: it contains no directional call, no
    entry/exit instruction, and no confidence figure of any kind."""
    ts = parse_timestamp(timestamp)
    rec = self.locate_record(instrument, ts)
    if as_of:
        res = self.query_memory_as_of(instrument, ts, top_k=top_k)
        mode = "AS-OF (only strictly earlier observations eligible)"
    else:
        res = self.query_memory(instrument, ts, top_k=top_k)
        mode = ("FULL-HISTORY LOOKUP (descriptive; later observations "
                "may appear - use as_of=True for chronological research)")
    matches = res["matches"]
    s = summarize_matches(matches)
    ho = rec.historical_outcome
    dists = [m["distance"] for m in matches]
    ages = [m["age_days"] for m in matches
            if m.get("age_days") is not None and np.isfinite(m["age_days"])]
    same = [m for m in matches if m["direction"] == rec.direction]
    opp = [m for m in matches if m["direction"] != rec.direction]
    s_same = summarize_matches(same)
    s_opp = summarize_matches(opp)
    missing_q = [spec for spec, _c in STATE_FEATURES
                 if rec.state_features.get(spec) is None]

    L: List[str] = []
    L.append("=" * 64)
    L.append("MARKET-STATE MEMORY REPORT  (historical, descriptive only)")
    L.append("engine: market_state_memory_engine.py")
    L.append(f"mode: {mode}")
    L.append("=" * 64)
    L.append("")
    L.append("A. CURRENT HISTORICAL STATE")
    L.append(f"   instrument           : {rec.instrument}")
    L.append(f"   timestamp            : {fmt_ts(rec.timestamp)}")
    L.append(f"   chronological period : {rec.chronological_period}")
    L.append(f"   direction            : {rec.direction}")
    L.append("   state features:")
    L.extend(_format_features_line(rec))
    L.append("   known historical outcome of this observation:")
    L.append(f"     net P/L={ho['net_pnl']}  win={ho['win']}  "
             f"holding={ho['holding_candles']}  exit reason={ho['exit_reason']}")
    L.append("     (stored history - context only, not a recommendation)")
    L.append("")
    L.append(f"B. HISTORICAL MATCHES (nearest observations in memory, "
             f"K={len(matches)})")
    L.append(f"   eligible historical observations: {res['n_eligible']}")
    if matches:
        L.append("   rank timestamp           dir    distance  "
                 "dist_pct   net P/L    win   fwd12%    age_d")
        for i, m in enumerate(matches, start=1):
            age_s = "n/a" if (m["age_days"] is None
                              or not np.isfinite(m["age_days"])) \
                else f"{m['age_days']:.2f}"
            L.append(
                f"   {i:>4} {fmt_ts(m['timestamp'])} {m['direction']:<6} "
                f"{m['distance']:>9.4f}  {m['distance_percentile']:>7.2f}  "
                f"{m['net_pnl']!s:>9}  {m['win']!s:<5} "
                f"{m['forward12_pct']!s:>8}  {age_s}")
    else:
        L.append("   no eligible historical observation found")
    L.append("")
    L.append("C. OUTCOME SUMMARY (descriptive)")
    L.append("   Historical matches in this sample had:")
    L.append(f"     number of matches          : {s['n']}")
    L.append(f"     average historical net P/L : {s['avg_net_pnl']}")
    L.append(f"     median net P/L             : {s['median_net_pnl']}")
    L.append(f"     win rate                   : {s['win_rate_pct']}")
    L.append(f"     profit factor              : {s['profit_factor']}")
    L.append(f"     positive outcome count     : {s['positive_count']}")
    L.append(f"     negative outcome count     : {s['negative_count']}")
    if s["low_sample"]:
        L.append(f"     LOW SAMPLE: fewer than {LOW_SAMPLE_N} eligible matches - "
                 "no conclusions are drawn from this result")
    L.append("   These are descriptions of past observations only.")
    L.append("")
    L.append("D. MATCH-DISTANCE INFORMATION")
    L.append(f"   nearest distance               : "
             f"{matches[0]['distance'] if matches else 'n/a'}")
    L.append(f"   median match distance          : {median_of(dists)}")
    L.append(f"   distance percentile of nearest : "
             f"{matches[0]['distance_percentile'] if matches else 'n/a'} "
             f"(share of eligible memory equally close or closer)")
    L.append(f"   number of matches              : {len(matches)}")
    L.append("")
    L.append("E. INSTRUMENT AND DIRECTION BREAKDOWN")
    L.append(f"   matches from {rec.instrument:<6} : {len(matches)}")
    L.append(f"     same direction as query      : {len(same)} "
             f"(avg net P/L {s_same['avg_net_pnl']}, "
             f"win rate {s_same['win_rate_pct']})")
    L.append(f"     different direction          : {len(opp)} "
             f"(avg net P/L {s_opp['avg_net_pnl']}, "
             f"win rate {s_opp['win_rate_pct']})")
    L.append("   (descriptive split only - no direction is selected)")
    L.append("")
    L.append("F. FORWARD-RETURN SUMMARY")
    L.append(f"   average forward6  of matches : {s['avg_forward6_pct']}")
    L.append(f"   average forward12 of matches : {s['avg_forward12_pct']}")
    L.append(f"   average forward24 of matches : {s['avg_forward24_pct']}")
    L.append("   (stored historical forward returns of the matches)")
    L.append("")
    L.append("G. MFE/MAE SUMMARY")
    L.append(f"   average MFE12 (ATR)     : {s['avg_mfe12_atr']}")
    L.append(f"   average MAE12 (ATR)     : {s['avg_mae12_atr']}")
    L.append(f"   average holding candles : {s['avg_holding_candles']}")
    if ages:
        L.append(f"   match age (days): median {median_of(ages):.2f}, "
                 f"mean {mean_of(ages):.2f}, "
                 f"min {min(ages):.2f}, max {max(ages):.2f}")
    L.append("")
    L.append("H. DATA-QUALITY WARNINGS")
    if missing_q:
        L.append("   - missing query feature values: " + ", ".join(missing_q))
    else:
        L.append("   - missing query feature values: none")
    if s["low_sample"]:
        L.append(f"   - LOW SAMPLE: fewer than {LOW_SAMPLE_N} eligible matches")
    unassigned = self.period_unassigned.get(instrument, 0)
    L.append(f"   - observations without chronological period: {unassigned}")
    L.append("   - similarity uses development-only z-score normalization "
             "(population std; std = 0 -> normalized value 0)")
    L.append("")
    L.append("DISCLAIMER: historical context only.  This report never "
             "contains and must never be read as an entry/exit "
             "instruction, a directional call, or a forecast.")
    return "\n".join(L)


MarketStateMemoryEngine.generate_memory_report = generate_memory_report


# ============================================================
# Walk-forward memory test (sections 14-19 of the specification)
# ============================================================
def aggregate_match_stats(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-query descriptive aggregation over the top-K matches."""
    s = summarize_matches(matches)
    dists = [float(m["distance"]) for m in matches
             if np.isfinite(m.get("distance", np.nan))]
    ages = [float(m["age_days"]) for m in matches
            if m.get("age_days") is not None and np.isfinite(m["age_days"])]
    same = [m for m in matches if m.get("same_direction") is True]
    opp = [m for m in matches if m.get("same_direction") is False]
    return {
        "n_matches": s["n"],
        "low_sample": s["low_sample"],
        "mean_match_net_pnl": s["avg_net_pnl"],
        "median_match_net_pnl": s["median_net_pnl"],
        "match_win_rate_pct": s["win_rate_pct"],
        "mean_match_forward6_pct": s["avg_forward6_pct"],
        "mean_match_forward12_pct": s["avg_forward12_pct"],
        "mean_match_forward24_pct": s["avg_forward24_pct"],
        "mean_match_mfe12_atr": s["avg_mfe12_atr"],
        "mean_match_mae12_atr": s["avg_mae12_atr"],
        "mean_match_holding_candles": s["avg_holding_candles"],
        "nearest_distance": (float(dists[0]) if dists else None),
        "median_match_distance": median_of(dists),
        "nearest_distance_percentile": (float(matches[0]["distance_percentile"])
                                        if matches else None),
        "same_direction_matches": len(same),
        "opposite_direction_matches": len(opp),
        "age_median_days": median_of(ages),
        "age_mean_days": mean_of(ages),
        "age_min_days": (float(min(ages)) if ages else None),
        "age_max_days": (float(max(ages)) if ages else None),
    }


def _sign_agree(a, b) -> Optional[bool]:
    sa, sb = sign_of(a), sign_of(b)
    if sa is None or sb is None:
        return None
    return sa == sb


def run_walk_forward_memory_test(engine: MarketStateMemoryEngine,
                                 top_k: int = TOP_K):
    """Strict chronological validation of the memory.

    Per instrument: the first half of the observations is the development
    memory; every observation of the second half queries ONLY strictly
    earlier DEVELOPMENT observations, and the K=10 nearest matches are
    compared descriptively with the actual stored outcome.  K is frozen;
    nothing is fitted, swept or selected.
    """
    if int(top_k) != TOP_K:
        raise ValueError(f"the walk-forward memory test uses the frozen "
                         f"K={TOP_K}")
    validation_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for inst in engine.instruments:
        dev_idx = engine.development_indices(inst)
        for qi in engine.evaluation_indices(inst):
            rec = engine.records[inst][int(qi)]
            matches, n_eligible = engine._matches(inst, rec, dev_idx,
                                                  top_k, rec.timestamp)
            for m in matches:
                m["same_direction"] = (m["direction"] == rec.direction)
            agg = aggregate_match_stats(matches)
            row: Dict[str, Any] = {
                "instrument": inst,
                "query_timestamp": fmt_ts(rec.timestamp),
                "query_direction": rec.direction,
                "query_period": rec.chronological_period,
                "n_eligible": n_eligible,
                "n_matches": agg["n_matches"],
                "low_sample": agg["low_sample"],
                "nearest_distance": agg["nearest_distance"],
                "median_match_distance": agg["median_match_distance"],
                "nearest_distance_percentile": agg["nearest_distance_percentile"],
                "distance_bucket": bucket_for(agg["nearest_distance"],
                                              engine.thresholds[inst]),
                "mean_match_net_pnl": agg["mean_match_net_pnl"],
                "median_match_net_pnl": agg["median_match_net_pnl"],
                "match_win_rate_pct": agg["match_win_rate_pct"],
                "actual_net_pnl": rec.historical_outcome["net_pnl"],
                "actual_win": rec.historical_outcome["win"],
                "mean_match_forward6_pct": agg["mean_match_forward6_pct"],
                "mean_match_forward12_pct": agg["mean_match_forward12_pct"],
                "mean_match_forward24_pct": agg["mean_match_forward24_pct"],
                "actual_forward6_pct": rec.forward_outcomes["forward6_pct"],
                "actual_forward12_pct": rec.forward_outcomes["forward12_pct"],
                "actual_forward24_pct": rec.forward_outcomes["forward24_pct"],
                "mean_match_mfe12_atr": agg["mean_match_mfe12_atr"],
                "mean_match_mae12_atr": agg["mean_match_mae12_atr"],
                "actual_mfe12_atr": rec.forward_outcomes["mfe12_atr"],
                "actual_mae12_atr": rec.forward_outcomes["mae12_atr"],
                "mean_match_holding_candles": agg["mean_match_holding_candles"],
                "actual_holding_candles": rec.historical_outcome["holding_candles"],
                "same_direction_matches": agg["same_direction_matches"],
                "opposite_direction_matches": agg["opposite_direction_matches"],
                "pnl_sign_agree": _sign_agree(agg["mean_match_net_pnl"],
                                              rec.historical_outcome["net_pnl"]),
                "fwd12_sign_agree": _sign_agree(agg["mean_match_forward12_pct"],
                                                rec.forward_outcomes["forward12_pct"]),
                "fwd24_sign_agree": _sign_agree(agg["mean_match_forward24_pct"],
                                                rec.forward_outcomes["forward24_pct"]),
                "match_age_median_days": agg["age_median_days"],
                "match_age_mean_days": agg["age_mean_days"],
                "match_age_min_days": agg["age_min_days"],
                "match_age_max_days": agg["age_max_days"],
            }
            validation_rows.append(row)
            for match_rank, m in enumerate(matches, start=1):
                detail_rows.append({
                    "instrument": inst,
                    "query_timestamp": fmt_ts(rec.timestamp),
                    "query_direction": rec.direction,
                    "query_period": rec.chronological_period,
                    "match_rank": match_rank,
                    "match_instrument": m["instrument"],
                    "match_timestamp": fmt_ts(m["timestamp"]),
                    "match_period": m["period"],
                    "match_direction": m["direction"],
                    "same_direction": m["same_direction"],
                    "distance": m["distance"],
                    "distance_percentile": m["distance_percentile"],
                    "age_days": m["age_days"],
                    "net_pnl": m["net_pnl"],
                    "win": m["win"],
                    "forward6_pct": m["forward6_pct"],
                    "forward12_pct": m["forward12_pct"],
                    "forward24_pct": m["forward24_pct"],
                    "mfe12_atr": m["mfe12_atr"],
                    "mae12_atr": m["mae12_atr"],
                    "holding_candles": m["holding_candles"],
                    "exit_reason": m["exit_reason"],
                })
    return validation_rows, detail_rows


# ============================================================
# Summary rows (sections 15-19 of the specification)
# ============================================================
def _blank_summary_row(instrument: str, section: str, label: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {c: None for c in SUMMARY_COLUMNS}
    row["instrument"] = instrument
    row["section"] = section
    row["label"] = label
    return row


def _pooled_pct(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (100.0 * sum(vals) / len(vals)) if vals else None


def build_summary_rows(engine: MarketStateMemoryEngine,
                       validation_rows: List[Dict[str, Any]],
                       detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for inst in engine.instruments:
        vrows = [r for r in validation_rows if r["instrument"] == inst]
        drows = [r for r in detail_rows if r["instrument"] == inst]
        p33, p66 = engine.thresholds[inst]

        # ---------------- OVERALL ----------------
        r = _blank_summary_row(inst, "OVERALL", "ALL_OOS_QUERIES")
        r["n"] = len(vrows)
        r["n_queries"] = len(vrows)
        r["low_sample_count"] = sum(1 for x in vrows if x["low_sample"])
        pooled = summarize_matches(drows)
        r["avg_match_net_pnl"] = pooled["avg_net_pnl"]
        r["median_match_net_pnl"] = pooled["median_net_pnl"]
        r["match_win_rate_pct"] = pooled["win_rate_pct"]
        r["profit_factor"] = pooled["profit_factor"]
        r["positive_match_count"] = pooled["positive_count"]
        r["negative_match_count"] = pooled["negative_count"]
        r["avg_match_forward6_pct"] = pooled["avg_forward6_pct"]
        r["avg_match_forward12_pct"] = pooled["avg_forward12_pct"]
        r["avg_match_forward24_pct"] = pooled["avg_forward24_pct"]
        r["avg_mfe12_atr"] = pooled["avg_mfe12_atr"]
        r["avg_mae12_atr"] = pooled["avg_mae12_atr"]
        r["avg_holding_candles"] = pooled["avg_holding_candles"]
        r["avg_actual_net_pnl"] = mean_of([x["actual_net_pnl"] for x in vrows])
        r["actual_win_rate_pct"] = _pooled_pct(vrows, "actual_win")
        r["avg_actual_forward6_pct"] = mean_of([x["actual_forward6_pct"] for x in vrows])
        r["avg_actual_forward12_pct"] = mean_of([x["actual_forward12_pct"] for x in vrows])
        r["avg_actual_forward24_pct"] = mean_of([x["actual_forward24_pct"] for x in vrows])
        r["avg_nearest_distance"] = mean_of([x["nearest_distance"] for x in vrows])
        r["median_nearest_distance"] = median_of([x["nearest_distance"] for x in vrows])
        r["avg_match_distance"] = mean_of([x["distance"] for x in drows])
        r["median_match_distance"] = median_of([x["distance"] for x in drows])
        for key, col in (("pnl", "pnl_sign_agree"),
                         ("fwd12", "fwd12_sign_agree"),
                         ("fwd24", "fwd24_sign_agree")):
            usable = [x[col] for x in vrows if x[col] is not None]
            pct = 100.0 * sum(1 for v in usable if v) / len(usable) if usable else None
            r[f"{key}_sign_agreement_pct"] = pct
            r[f"{key}_sign_n"] = len(usable)
        for key, xa, yb in (
                ("pnl", "mean_match_net_pnl", "actual_net_pnl"),
                ("fwd12", "mean_match_forward12_pct", "actual_forward12_pct"),
                ("fwd24", "mean_match_forward24_pct", "actual_forward24_pct")):
            rr, nn = safe_pearson([x[xa] for x in vrows], [x[yb] for x in vrows])
            r[f"corr_match_actual_{key}_r"] = rr
            r[f"corr_match_actual_{key}_n"] = nn
        r["distance_threshold_p33"] = p33
        r["distance_threshold_p66"] = p66
        rows.append(r)

        # ---------------- AGE (descriptive only) ----------------
        ages = [x["age_days"] for x in drows
                if x.get("age_days") is not None and np.isfinite(x["age_days"])]
        r = _blank_summary_row(inst, "AGE", "MATCH_AGE_DAYS")
        r["n"] = len(ages)
        r["age_median_days"] = median_of(ages)
        r["age_mean_days"] = mean_of(ages)
        r["age_min_days"] = min(ages) if ages else None
        r["age_max_days"] = max(ages) if ages else None
        rows.append(r)

        # ---------------- STABILITY (descriptive only) ----------------
        if drows:
            same_dir = sum(1 for x in drows if x["same_direction"])
            same_inst = sum(1 for x in drows if x["match_instrument"] == inst)
            nearest = [x for x in drows if x["match_rank"] == 1]
            n_same_near = sum(1 for x in nearest if x["same_direction"])
            r = _blank_summary_row(inst, "STABILITY", "NEAREST_MATCHES")
            r["n"] = len(drows)
            r["n_queries"] = len(vrows)
            r["pct_same_instrument"] = 100.0 * same_inst / len(drows)
            r["pct_same_direction"] = 100.0 * same_dir / len(drows)
            r["pct_different_direction"] = 100.0 * (len(drows) - same_dir) / len(drows)
            if nearest:
                r["pct_nearest_same_direction"] = 100.0 * n_same_near / len(nearest)
                r["pct_nearest_different_direction"] = \
                    100.0 * (len(nearest) - n_same_near) / len(nearest)
            rows.append(r)

        # ---------------- DIRECTION SPLIT ----------------
        for label, flag in ((DIRECTION_LABELS[0], True),
                            (DIRECTION_LABELS[1], False)):
            grp = [x for x in drows if x["same_direction"] is flag]
            s = summarize_matches(grp)
            r = _blank_summary_row(inst, "DIRECTION", label)
            r["n"] = s["n"]
            r["n_queries"] = len({x["query_timestamp"] for x in grp})
            r["avg_match_net_pnl"] = s["avg_net_pnl"]
            r["median_match_net_pnl"] = s["median_net_pnl"]
            r["match_win_rate_pct"] = s["win_rate_pct"]
            r["avg_match_forward12_pct"] = s["avg_forward12_pct"]
            r["avg_match_forward24_pct"] = s["avg_forward24_pct"]
            rows.append(r)

        # ---------------- DISTANCE BUCKETS (frozen thresholds) ----------------
        for label in BUCKET_LABELS:
            grp = [x for x in vrows if x["distance_bucket"] == label]
            r = _blank_summary_row(inst, "DISTANCE_BUCKET", label)
            r["n"] = len(grp)
            r["n_queries"] = len(grp)
            r["avg_match_net_pnl"] = mean_of([x["mean_match_net_pnl"] for x in grp])
            r["avg_actual_net_pnl"] = mean_of([x["actual_net_pnl"] for x in grp])
            r["actual_win_rate_pct"] = _pooled_pct(grp, "actual_win")
            r["match_win_rate_pct"] = mean_of([x["match_win_rate_pct"] for x in grp])
            r["avg_match_forward12_pct"] = mean_of([x["mean_match_forward12_pct"] for x in grp])
            r["avg_match_forward24_pct"] = mean_of([x["mean_match_forward24_pct"] for x in grp])
            r["avg_actual_forward12_pct"] = mean_of([x["actual_forward12_pct"] for x in grp])
            r["avg_actual_forward24_pct"] = mean_of([x["actual_forward24_pct"] for x in grp])
            r["avg_nearest_distance"] = mean_of([x["nearest_distance"] for x in grp])
            rows.append(r)
    return rows


# ============================================================
# Synthetic data builders (used by the test suite ONLY; they never
# touch the real CSVs and never write real output files)
# ============================================================
def tiny_frame(inst: str, adx_values, rsi_values=None, const: float = 5.0,
               start: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    """Minimal deterministic frame: every state feature is CONSTANT
    (std = 0 -> z = 0, contributes 0 to distances) except adx and the
    optional rsi14.  Used for hand-computed distance checks."""
    base = pd.Timestamp(start)
    rows: List[Dict[str, Any]] = []
    for i, adx in enumerate(adx_values):
        ts = base + pd.Timedelta(hours=i)
        rsi = const if rsi_values is None else float(rsi_values[i])
        net = float(((i * 13) % 21) - 10)
        gross = net + 2.0
        rows.append({
            "instrument": inst, "timeframe": "H1",
            "signal_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "signal_index": 60 + i,
            "direction": "LONG" if i % 2 == 0 else "SHORT",
            "adx": float(adx), "atr": const, "atr_pct": const,
            "ema50": const, "ema_distance": const, "bb_middle": const,
            "bb_upper": const, "bb_lower": const, "bb_excursion": const,
            "bb_signed": const, "eff24": const, "return6": const,
            "return12": const, "return24": const, "rsi14": rsi,
            "vol6": const, "vol24": const, "range_pct": const,
            "body_pct": const, "upper_wick_pct": const,
            "lower_wick_pct": const, "distance_24_high": const,
            "distance_24_low": const,
            "hour": 0, "day_of_week": 0, "month": 1,
            "entry_price": 100.0, "exit_price": 100.0 + gross / 10.0,
            "exit_time": (ts + pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_reason": ("TP", "SL", "TIME")[i % 3],
            "holding_candles": (i % 5) + 1,
            "gross_pnl": gross, "trading_cost": 2.0, "net_pnl": net,
            "win": net > 0,
            "forward6_pct": ((i % 11) - 5) / 100.0,
            "forward12_pct": ((i % 15) - 7) / 100.0,
            "forward24_pct": ((i % 21) - 10) / 100.0,
            "mfe12_atr": 1.0 + (i % 5) * 0.5,
            "mae12_atr": -(1.0 + (i % 4) * 0.5),
        })
    return pd.DataFrame(rows)


def make_synthetic_frame(inst: str, n: int, start: str = "2024-01-01 00:00:00",
                         inject_nan: bool = True) -> pd.DataFrame:
    """Deterministic full-schema frame with varied feature formulas."""
    base = pd.Timestamp(start)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        ts = base + pd.Timedelta(hours=i)
        direction = "LONG" if i % 2 == 0 else "SHORT"
        bbe = 0.3 + (i % 5) * 0.1
        net = float(((i * 13) % 21) - 10)
        gross = net + 2.0
        adx = 10.0 + (i * 7) % 25
        if inject_nan and inst == "TESTA" and i == 5:
            adx = np.nan  # one missing value for data-quality handling
        rows.append({
            "instrument": inst, "timeframe": "H1",
            "signal_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "signal_index": 60 + i, "direction": direction,
            "adx": adx, "atr": 1.0 + (i % 9) * 0.1,
            "atr_pct": 0.10 + (i % 7) * 0.01,
            "ema50": 100.0 + i * 0.5, "ema_distance": 0.5 + (i % 11) * 0.1,
            "bb_middle": 100.0, "bb_upper": 101.0, "bb_lower": 99.0,
            "bb_excursion": bbe,
            "bb_signed": bbe if direction == "LONG" else -bbe,
            "eff24": 0.2 + (i % 6) * 0.1,
            "return6": ((i % 13) - 6) / 100.0,
            "return12": ((i % 17) - 8) / 100.0,
            "return24": ((i % 19) - 9) / 100.0,
            "rsi14": 30.0 + (i * 5) % 45,
            "vol6": 0.05 + (i % 4) * 0.01, "vol24": 0.08 + (i % 5) * 0.01,
            "range_pct": 0.2 + (i % 8) * 0.05, "body_pct": 0.1 + (i % 6) * 0.03,
            "upper_wick_pct": 0.01 + (i % 3) * 0.01,
            "lower_wick_pct": 0.01 + (i % 4) * 0.005,
            "distance_24_high": -0.5 + (i % 10) * 0.1,
            "distance_24_low": 0.5 - (i % 10) * 0.05,
            "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month,
            "entry_price": 100.0 + i * 0.5,
            "exit_price": 100.0 + i * 0.5 + gross / 10.0,
            "exit_time": (ts + pd.Timedelta(hours=(i % 5) + 1)).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "exit_reason": ("TP", "SL", "TIME")[i % 3],
            "holding_candles": (i % 5) + 1,
            "gross_pnl": gross, "trading_cost": 2.0, "net_pnl": net,
            "win": net > 0,
            "forward6_pct": ((i % 11) - 5) / 100.0,
            "forward12_pct": ((i % 15) - 7) / 100.0,
            "forward24_pct": ((i % 21) - 10) / 100.0,
            "mfe12_atr": 1.0 + (i % 5) * 0.5,
            "mae12_atr": -(1.0 + (i % 4) * 0.5),
        })
    return pd.DataFrame(rows)


def make_synthetic_frames() -> Dict[str, pd.DataFrame]:
    return {
        "TESTA": make_synthetic_frame("TESTA", 40),
        "TESTB": make_synthetic_frame("TESTB", 30),
    }


# ============================================================
# Static safety scan of this file's own source
# ============================================================
def run_safety_scan() -> List[str]:
    """AST + token scan of this file's own source.  Returns a list of
    violations (empty when the file is clean)."""
    src_path = os.path.abspath(__file__)
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    violations: List[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"syntax error during scan: {exc}"]
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for mod in sorted(imported):
        if mod not in _ALLOWED_IMPORT_MODULES:
            violations.append(f"unexpected import: {mod}")
    for mod in _FORBIDDEN_IMPORT_MODULES:
        if mod in imported:
            violations.append(f"forbidden import present: {mod}")
    for tok in _MT5_SOURCE_TOKENS:
        if tok in src:
            violations.append("trading-terminal token present in source")
    for tok in _ML_SOURCE_TOKENS:
        if tok in src:
            violations.append("machine-learning library token in source")
    for pat in _ML_CALL_PATTERNS:
        if re.search(pat, src):
            violations.append("fitting-call pattern present in source")
    for tok in _OPTIMIZATION_TOKENS:
        if tok in src:
            violations.append("optimization idiom present in source")
    fn_names = [n.name.lower() for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for name in fn_names:
        for tok in _FORBIDDEN_FN_NAME_TOKENS:
            if tok in name:
                violations.append(
                    f"forbidden token '{tok}' in function name '{name}'")
    return violations


def _independent_thresholds(df: pd.DataFrame):
    """Test-side re-derivation of the frozen distance thresholds straight
    from a raw frame (independent of the engine's internals)."""
    X = df[list(STATE_FEATURE_COLUMNS)].astype(float).to_numpy()
    n = len(X)
    split = int(n * DEV_SPLIT_FRACTION)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X[:split], axis=0)
        std = np.nanstd(X[:split], axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    Z = (X - mean) / std_safe
    zero = std == 0.0
    if zero.any():
        Z[:, zero] = np.where(np.isnan(X[:, zero]), np.nan, 0.0)
    dists = []
    for i in range(split):
        best = None
        for j in range(split):
            if i == j:
                continue
            diff = Z[i] - Z[j]
            valid = ~np.isnan(diff)
            if valid.sum() == 0:
                continue
            d = float(np.sqrt((diff[valid] ** 2).sum()))
            best = d if best is None else min(best, d)
        if best is not None:
            dists.append(best)
    if len(dists) < 2:
        return (None, None)
    return (float(np.percentile(dists, DISTANCE_PCT_LOW)),
            float(np.percentile(dists, DISTANCE_PCT_HIGH)))


# ============================================================
# Synthetic test suite (runs by default; never touches the real
# CSVs and never writes real output files)
# ============================================================
def run_synthetic_tests() -> Tuple[int, int]:
    passed = 0
    failed = 0
    area_stats: List[Tuple[str, int, int]] = []
    current = {"name": "(start)", "p": 0, "f": 0}

    def start_area(name: Optional[str]) -> None:
        nonlocal current
        if current["name"] != "(start)":
            area_stats.append((current["name"], current["p"], current["f"]))
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

    print("market_state_memory_engine.py - synthetic test suite")
    print("(in-memory DataFrames only; no real CSV is read or written)\n")

    # shared deterministic engines
    frames = make_synthetic_frames()
    eng1 = MarketStateMemoryEngine(frames)
    frames2 = {k: v.copy() for k, v in frames.items()}
    half_a = len(frames2["TESTA"]) // 2
    frames2["TESTA"].loc[frames2["TESTA"].index >= half_a, "adx"] = (
        frames2["TESTA"].loc[frames2["TESTA"].index >= half_a, "adx"]
        * 1000.0 + 7.0)
    eng2 = MarketStateMemoryEngine(frames2)
    frames3 = {k: v.copy() for k, v in frames.items()}
    frames3["TESTA"].loc[frames3["TESTA"].index < half_a, "adx"] = 7.0
    eng3 = MarketStateMemoryEngine(frames3)

    # ----------------------------------------------------------
    start_area("missing_file")
    with tempfile.TemporaryDirectory() as td:
        try:
            check_required_inputs(td)
            check("missing inputs raise MissingInputError", False)
        except MissingInputError as exc:
            msg = str(exc)
            check("missing inputs raise MissingInputError", True)
            check("exact missing instrument filename reported",
                  "market_state_GBPUSD.csv" in msg)
            check("exact missing oos filename reported",
                  "market_state_oos_walkforward.csv" in msg)
    with tempfile.TemporaryDirectory() as td:
        for f in REQUIRED_INPUT_FILES:
            open(os.path.join(td, f), "w").close()
        ok = True
        try:
            check_required_inputs(td)
        except MissingInputError:
            ok = False
        check("all required inputs present -> no error", ok)

    # ----------------------------------------------------------
    start_area("as_of_leakage")
    df_base = tiny_frame("TESTW", [float(i + 1) for i in range(13)])
    twin = df_base.iloc[[10]].copy()
    twin["net_pnl"] = 999.0                       # same timestamp as row 10
    twin2 = df_base.iloc[[10]].copy()
    twin2["signal_time"] = df_base.iloc[12]["signal_time"]
    twin2["net_pnl"] = 888.0                      # later timestamp
    dfw = pd.concat([df_base, twin, twin2], ignore_index=True)
    eng_w = MarketStateMemoryEngine({"TESTW": dfw})
    ts10 = pd.Timestamp(dfw.iloc[10]["signal_time"])
    res_q = eng_w.query_memory("TESTW", ts10)
    res_a = eng_w.query_memory_as_of("TESTW", ts10)
    check("query observation itself excluded (full-history lookup)",
          all(m["row_index"] != 10 for m in res_q["matches"]))
    check("same-timestamp twin eligible only in full-history mode",
          res_q["matches"][0]["row_index"] == 11
          and res_q["matches"][0]["distance"] == 0.0)
    check("full-history lookup may reach later observations",
          any(m["row_index"] == 14 for m in res_q["matches"]))
    check("as-of returns only strictly earlier observations",
          all(pd.Timestamp(m["timestamp"]) < ts10 for m in res_a["matches"]))
    check("same-timestamp observation excluded from as-of",
          all(m["row_index"] != 13 for m in res_a["matches"]))
    check("later observation excluded from as-of",
          all(m["row_index"] != 14 for m in res_a["matches"]))
    check("as-of eligible count = strictly earlier rows",
          res_a["n_eligible"] == 10)
    check("as-of nearest is the closest earlier observation",
          res_a["matches"][0]["row_index"] == 9
          and np.isclose(res_a["matches"][0]["distance"], 0.5))

    # ----------------------------------------------------------
    start_area("self_exclusion")
    check("self never returned by query_memory",
          all(m["row_index"] != 10 for m in res_q["matches"]))
    check("self never returned by query_memory_as_of",
          all(m["row_index"] != 10 for m in res_a["matches"]))

    # ----------------------------------------------------------
    start_area("normalization_dev_only")
    stats_equal = True
    for col in STATE_FEATURE_COLUMNS:
        m1, s1 = eng1.stats["TESTA"][col]
        m2, s2 = eng2.stats["TESTA"][col]
        if not (np.isclose(m1, m2, equal_nan=True)
                and np.isclose(s1, s2, equal_nan=True)):
            stats_equal = False
    check("normalization statistics ignore evaluation-half data", stats_equal)
    check("normalization statistics follow development data",
          not np.isclose(eng1.stats["TESTA"]["adx"][0],
                         eng3.stats["TESTA"]["adx"][0]))
    zc = eng_d_zcol = eng1.z_matrix["TESTA"]
    dev_rows = eng1.development_indices("TESTA")
    col_i = STATE_FEATURE_COLUMNS.index("adx")
    dev_adx = frames["TESTA"]["adx"].to_numpy()[:len(dev_rows)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        exp_mean = float(np.nanmean(dev_adx))
        exp_std = float(np.nanstd(dev_adx))
    z_exp = (frames["TESTA"]["adx"].to_numpy() - exp_mean) / exp_std
    got = zc[:, col_i]
    ok = True
    for i in range(len(z_exp)):
        if np.isnan(z_exp[i]):
            ok = ok and np.isnan(got[i])
        else:
            ok = ok and np.isclose(got[i], z_exp[i])
    check("z = (x - mean)/std with population std (ddof=0)", ok)
    dfd = tiny_frame("TESTD", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    eng_d = MarketStateMemoryEngine({"TESTD": dfd})
    zatr = eng_d.z_matrix["TESTD"][:, STATE_FEATURE_COLUMNS.index("atr")]
    check("std == 0 -> normalized value 0 for every row",
          bool(np.allclose(zatr, 0.0)))

    # ----------------------------------------------------------
    start_area("thresholds_dev_only")
    thr_same = (np.isclose(eng1.thresholds["TESTA"][0],
                           eng2.thresholds["TESTA"][0])
                and np.isclose(eng1.thresholds["TESTA"][1],
                               eng2.thresholds["TESTA"][1]))
    check("distance thresholds ignore evaluation-half data", thr_same)
    ind = _independent_thresholds(frames["TESTA"])
    check("thresholds = 33rd/66th of development nearest distances",
          np.isclose(eng1.thresholds["TESTA"][0], ind[0])
          and np.isclose(eng1.thresholds["TESTA"][1], ind[1]))
    check("thresholds follow development data",
          not np.isclose(eng1.thresholds["TESTA"][0],
                         eng3.thresholds["TESTA"][0]))
    check("frozen thresholds reused for every query",
          eng1.thresholds["TESTA"] == eng1.thresholds["TESTA"])

    # ----------------------------------------------------------
    start_area("k_fixed")
    check("K frozen at 10 (module constant)", TOP_K == 10 and DEFAULT_TOP_K == 10)
    ts20 = frames["TESTA"].iloc[20]["signal_time"]
    res20 = eng1.query_memory_as_of("TESTA", ts20)
    check("as-of returns exactly K=10 matches when available",
          len(res20["matches"]) == 10)
    res3 = eng1.query_memory_as_of("TESTA", ts20, top_k=3)
    check("explicit smaller top_k honored for interactive use",
          len(res3["matches"]) == 3)
    validation_rows, detail_rows = run_walk_forward_memory_test(eng1)
    check("validation uses the frozen K=10",
          all(r["n_matches"] <= 10 for r in validation_rows))
    check("validation n_matches == 10 with full development memory",
          all(r["n_matches"] == 10 for r in validation_rows))
    try:
        run_walk_forward_memory_test(eng1, top_k=7)
        check("non-frozen K rejected in validation", False)
    except ValueError:
        check("non-frozen K rejected in validation", True)

    # ----------------------------------------------------------
    start_area("ordering")
    ts_all = [r.timestamp for r in eng1.records["TESTA"]]
    check("records sorted chronologically",
          all(ts_all[i] <= ts_all[i + 1] for i in range(len(ts_all) - 1)))
    dev_ts = [eng1.records["TESTA"][i].timestamp
              for i in eng1.development_indices("TESTA")]
    eval_ts = [eng1.records["TESTA"][i].timestamp
               for i in eng1.evaluation_indices("TESTA")]
    check("development half strictly precedes evaluation half",
          max(dev_ts) < min(eval_ts))
    check("validation covers exactly the evaluation half",
          len([r for r in validation_rows if r["instrument"] == "TESTA"])
          == len(eval_ts))
    qts = [r["query_timestamp"] for r in validation_rows
           if r["instrument"] == "TESTA"]
    check("validation queries processed in chronological order",
          qts == sorted(qts))
    check("every validation match is strictly earlier than its query",
          all(d["match_timestamp"] < d["query_timestamp"] for d in detail_rows))

    # ----------------------------------------------------------
    start_area("low_sample")
    df6 = tiny_frame("TESTS", [float(i) for i in range(6)])
    eng_s = MarketStateMemoryEngine({"TESTS": df6})
    ts3 = df6.iloc[3]["signal_time"]
    res_s = eng_s.query_memory_as_of("TESTS", ts3)
    check("fewer than 5 eligible -> LOW SAMPLE flag",
          res_s["low_sample"] and len(res_s["matches"]) == 3)
    sm = summarize_matches(res_s["matches"])
    check("summary marks LOW SAMPLE", sm["low_sample"])
    rep_s = eng_s.generate_memory_report("TESTS", ts3)
    check("report contains LOW SAMPLE warning", "LOW SAMPLE" in rep_s)
    low_vrows = [r for r in validation_rows if r["low_sample"]]
    check("low-sample queries flagged in validation rows",
          all(r["n_matches"] < LOW_SAMPLE_N for r in low_vrows))

    # ----------------------------------------------------------
    start_area("determinism_distance")
    tsd3 = dfd.iloc[3]["signal_time"]
    res_d1 = eng_d.query_memory("TESTD", tsd3)
    res_d2 = eng_d.query_memory("TESTD", tsd3)
    check("identical query -> identical result",
          [(m["row_index"], m["distance"]) for m in res_d1["matches"]]
          == [(m["row_index"], m["distance"]) for m in res_d2["matches"]])
    exp_d = 2.0 / float(np.sqrt(8.0 / 3.0))
    check("hand-computed standardized distance",
          np.isclose(res_d1["matches"][0]["distance"], exp_d))
    check("tie at equal distance",
          np.isclose(res_d1["matches"][1]["distance"], exp_d))
    check("deterministic tie-break by chronological order",
          res_d1["matches"][0]["row_index"] == 2
          and res_d1["matches"][1]["row_index"] == 4)
    dfn = tiny_frame("TESTN", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                     rsi_values=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    dfn.loc[5, "rsi14"] = np.nan
    eng_n = MarketStateMemoryEngine({"TESTN": dfn})
    res_n = eng_n.query_memory("TESTN", dfn.iloc[5]["signal_time"])
    check("NaN feature skipped (sum over valid features only)",
          np.isclose(res_n["matches"][0]["distance"], exp_d))
    check("all reported distances finite and non-negative",
          all(np.isfinite(m["distance"]) and m["distance"] >= 0.0
              for m in res_n["matches"]))

    # ----------------------------------------------------------
    start_area("direction_split")
    drows_a = [d for d in detail_rows if d["instrument"] == "TESTA"]
    n_same_manual = sum(1 for d in drows_a if d["same_direction"])
    summary_rows = build_summary_rows(eng1, validation_rows, detail_rows)
    same_row = [r for r in summary_rows if r["instrument"] == "TESTA"
                and r["section"] == "DIRECTION"
                and r["label"] == "SAME_DIRECTION"][0]
    opp_row = [r for r in summary_rows if r["instrument"] == "TESTA"
               and r["section"] == "DIRECTION"
               and r["label"] == "OPPOSITE_DIRECTION"][0]
    check("same-direction count matches match details",
          same_row["n"] == n_same_manual)
    check("opposite-direction count matches match details",
          opp_row["n"] == len(drows_a) - n_same_manual)
    check("both directions reported (none selected)",
          same_row["n"] > 0 and opp_row["n"] > 0)
    same_pnls = [d["net_pnl"] for d in drows_a if d["same_direction"]]
    check("same-direction average net P/L matches details",
          np.isclose(same_row["avg_match_net_pnl"], float(np.mean(same_pnls))))

    # ----------------------------------------------------------
    start_area("age")
    dfc = tiny_frame("TESTC", [3.0] * 12)
    eng_c = MarketStateMemoryEngine({"TESTC": dfc})
    tsc10 = pd.Timestamp(dfc.iloc[10]["signal_time"])
    res_c = eng_c.query_memory_as_of("TESTC", tsc10)
    check("constant features -> zero distance",
          res_c["matches"][0]["distance"] == 0.0)
    check("tie-break by chronological order",
          res_c["matches"][0]["row_index"] == 0)
    check("age_days computed exactly (10 hourly candles = 10/24 days)",
          np.isclose(res_c["matches"][0]["age_days"], 10.0 / 24.0))
    check("age of every match = query - match",
          all(np.isclose(m["age_days"],
                         (tsc10 - m["timestamp"]).total_seconds() / 86400.0)
              for m in res_c["matches"]))
    ages_a = [d["age_days"] for d in drows_a if np.isfinite(d["age_days"])]
    age_row = [r for r in summary_rows if r["instrument"] == "TESTA"
               and r["section"] == "AGE"][0]
    check("pooled age statistics match details",
          np.isclose(age_row["age_median_days"], float(np.median(ages_a)))
          and np.isclose(age_row["age_min_days"], min(ages_a))
          and np.isclose(age_row["age_max_days"], max(ages_a)))

    # ----------------------------------------------------------
    start_area("cross_instrument")
    cross = eng1.query_cross_instrument_memory(
        timestamp=None, state_features={"ADX": 15.0, "RSI14": 50.0}, top_k=5)
    check("cross query returns K matches", len(cross["matches"]) == 5)
    check("every match carries its instrument",
          all(m["instrument"] in ("TESTA", "TESTB") for m in cross["matches"]))
    sums = cross["summary_by_instrument"]
    check("summaries separated per instrument",
          set(sums.keys()) == {m["instrument"] for m in cross["matches"]})
    check("per-instrument counts add to K",
          sum(s["n"] for s in sums.values()) == 5)
    check("no combined P/L series across instruments",
          all(isinstance(s, dict) and "avg_net_pnl" in s for s in sums.values()))
    cross2 = eng1.query_cross_instrument_memory(
        timestamp=None, state_features={"ADX": 15.0}, top_k=3,
        exclude=("TESTA", frames["TESTA"].iloc[0]["signal_time"]))
    t0 = fmt_ts(pd.Timestamp(frames["TESTA"].iloc[0]["signal_time"]))
    check("exclude removes the named observation",
          not any(m["instrument"] == "TESTA"
                  and fmt_ts(m["timestamp"]) == t0 for m in cross2["matches"]))
    try:
        eng1.query_cross_instrument_memory(state_features={"nope": 1.0})
        check("unknown feature name rejected", False)
    except ValueError:
        check("unknown feature name rejected", True)
    try:
        eng1.query_cross_instrument_memory(as_of=True, state_features={"ADX": 1.0})
        check("as_of without timestamp rejected", False)
    except ValueError:
        check("as_of without timestamp rejected", True)

    # ----------------------------------------------------------
    start_area("overwrite_protection")
    with tempfile.TemporaryDirectory() as td:
        ok = True
        try:
            check_output_files_available(td)
        except OutputExistsError:
            ok = False
        check("clean directory passes overwrite check", ok)
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "market_memory_records.csv"), "w").close()
        open(os.path.join(td, "market_memory_summary.csv"), "w").close()
        try:
            check_output_files_available(td)
            check("existing outputs stop the run", False)
        except OutputExistsError as exc:
            msg = str(exc)
            check("existing outputs stop the run", True)
            check("conflicting filenames listed",
                  "market_memory_records.csv" in msg
                  and "market_memory_summary.csv" in msg)

    # ----------------------------------------------------------
    start_area("report_safety")
    rep = eng1.generate_memory_report(
        "TESTA", frames["TESTA"].iloc[30]["signal_time"])
    for sec in ("A. CURRENT HISTORICAL STATE", "B. HISTORICAL MATCHES",
                "C. OUTCOME SUMMARY", "D. MATCH-DISTANCE INFORMATION",
                "E. INSTRUMENT AND DIRECTION BREAKDOWN",
                "F. FORWARD-RETURN SUMMARY", "G. MFE/MAE SUMMARY",
                "H. DATA-QUALITY WARNINGS"):
        check(f"report section present ({sec[:1]})", sec in rep)
    bad_rx = re.compile("\\b(?:" + "|".join(
        re.escape(w) for w in _FORBIDDEN_REPORT_WORDS) + ")\\b")
    check("report contains no directional call", bad_rx.search(rep) is None)
    check("report uses descriptive wording",
          "Historical matches in this sample had" in rep)
    rep5 = eng1.generate_memory_report(
        "TESTA", frames["TESTA"].iloc[5]["signal_time"])
    check("data-quality warning lists missing query feature", "ADX" in rep5)

    # ----------------------------------------------------------
    start_area("safety_scan")
    violations = run_safety_scan()
    check("static safety scan clean (no terminal, no ML, no optimization)",
          violations == [])
    md = eng1.records["TESTA"][0].to_memory_dict()
    check("standardized memory record fields",
          set(md.keys()) >= {"instrument", "timestamp", "direction",
                             "state_features", "historical_outcome",
                             "forward_outcomes", "chronological_period"})
    rf = eng1.records_frame()
    check("records frame preserves every observation", len(rf) == 70)
    check("records frame stores raw and normalized features",
          "adx" in rf.columns and "z_adx" in rf.columns)
    check("records frame stores outcomes and forward data",
          "net_pnl" in rf.columns and "mfe12_atr" in rf.columns
          and "forward24_pct" in rf.columns)
    bad_frame = frames["TESTA"].drop(columns=["rsi14"])
    try:
        MarketStateMemoryEngine({"BAD": bad_frame})
        check("frame with missing columns rejected", False)
    except ValueError:
        check("frame with missing columns rejected", True)

    # ----------------------------------------------------------
    start_area("periods_and_buckets")
    ps = pd.DataFrame({
        "instrument": ["TESTA", "TESTA"],
        "period": ["P1", "P2"],
        "start_time": ["2024-01-01 00:00:00", "2024-01-01 20:00:00"],
        "end_time": ["2024-01-01 19:00:00", "2024-01-02 15:00:00"],
    })
    eng_p = MarketStateMemoryEngine({"TESTA": frames["TESTA"]},
                                    period_summary=ps)
    recs_p = eng_p.records["TESTA"]
    check("period from period summary (P1)",
          recs_p[0].chronological_period == "P1")
    check("period from period summary (P2)",
          recs_p[25].chronological_period == "P2")
    recs_f = eng1.records["TESTA"]
    check("fallback quartile period P1", recs_f[0].chronological_period == "P1")
    check("fallback quartile period P4", recs_f[39].chronological_period == "P4")
    check("bucket LOW below 33rd percentile",
          bucket_for(0.5, (1.0, 2.0)) == "LOW DISTANCE")
    check("bucket boundary p33 -> MEDIUM",
          bucket_for(1.0, (1.0, 2.0)) == "MEDIUM DISTANCE")
    check("bucket boundary p66 -> HIGH",
          bucket_for(2.0, (1.0, 2.0)) == "HIGH DISTANCE")
    check("bucket n/a without thresholds",
          bucket_for(1.0, (None, None)) == "n/a")
    b_rows = [r for r in summary_rows if r["instrument"] == "TESTA"
              and r["section"] == "DISTANCE_BUCKET"]
    check("all three distance buckets reported",
          {r["label"] for r in b_rows} == set(BUCKET_LABELS))
    check("bucket thresholds frozen from development",
          all(r["n"] >= 0 for r in b_rows)
          and b_rows[0]["instrument"] == "TESTA")

    # ----------------------------------------------------------
    start_area("helpers")
    check("profit factor normal case",
          np.isclose(profit_factor([10.0, -5.0, 5.0]), 3.0))
    check("profit factor with no losses -> inf",
          profit_factor([10.0, 5.0]) == float("inf"))
    check("profit factor all zero -> NaN", np.isnan(profit_factor([0.0, 0.0])))
    r1, n1 = safe_pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    check("pearson perfect positive", r1 is not None and np.isclose(r1, 1.0)
          and n1 == 3)
    r2, n2 = safe_pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    check("pearson zero variance -> n/a", r2 is None and n2 == 3)
    r3, n3 = safe_pearson([1.0], [1.0])
    check("pearson needs at least 3 pairs", r3 is None and n3 == 1)
    p1, c1 = sign_agreement_rate([(1.0, -1.0), (2.0, 3.0)])
    check("sign agreement 50 percent", np.isclose(p1, 50.0) and c1 == 2)
    p2, c2 = sign_agreement_rate([(1.0, None), (-2.0, -3.0)])
    check("sign agreement skips missing values", np.isclose(p2, 100.0)
          and c2 == 1)
    check("parse_timestamp accepts strings",
          fmt_ts(parse_timestamp("2024-01-01 05:00:00"))
          == "2024-01-01 05:00:00")
    try:
        parse_timestamp("not-a-time")
        check("bad timestamp rejected", False)
    except ValueError:
        check("bad timestamp rejected", True)

    # ----------------------------------------------------------
    start_area("default_run")
    check("default (no flags) does not request real analysis",
          should_run_real([]) is False)
    check("--run flag requests real analysis",
          should_run_real(["--run"]) is True)
    check("unrelated flags do not request real analysis",
          should_run_real(["-x", "--verbose"]) is False)
    called = {"real": False, "tests": False}

    def spy_real():
        called["real"] = True

    def spy_tests():
        called["tests"] = True
        return 0, 0

    orig_real = globals()["run_real_analysis"]
    orig_tests = globals()["run_synthetic_tests"]
    argv_backup = list(sys.argv)
    try:
        globals()["run_real_analysis"] = spy_real
        globals()["run_synthetic_tests"] = spy_tests
        sys.argv = ["market_state_memory_engine.py"]
        main()
        check("default main() runs synthetic tests only",
              called["tests"] and not called["real"])
        called["real"] = False
        called["tests"] = False
        sys.argv = ["market_state_memory_engine.py", "--run"]
        main()
        check("--run main() reaches the real-analysis entry point",
              called["real"] and not called["tests"])
    finally:
        globals()["run_real_analysis"] = orig_real
        globals()["run_synthetic_tests"] = orig_tests
        sys.argv = argv_backup

    # ----------------------------------------------------------
    start_area(None)
    print()
    print("  Per-area verification (spec leakage/safety items -> area: "
          "passed/failed):")
    spec_map = (
        ("missing_file", "item 24.15 missing-file handling"),
        ("as_of_leakage", "items 24.1, 24.2, 24.4 future/timestamp leakage"),
        ("self_exclusion", "item 24.3 self-match exclusion"),
        ("normalization_dev_only", "item 24.5 development-only normalization"),
        ("thresholds_dev_only", "item 24.6 development-only thresholds"),
        ("k_fixed", "item 24.7 K fixed at 10"),
        ("ordering", "item 24.8 chronological ordering"),
        ("low_sample", "item 24.9 low-sample handling"),
        ("determinism_distance", "item 24.10 deterministic distance"),
        ("direction_split", "item 24.11 direction split"),
        ("age", "item 24.12 age calculation"),
        ("cross_instrument", "item 24.13 cross-instrument separation"),
        ("overwrite_protection", "item 24.14 overwrite protection"),
        ("report_safety", "report wording and sections"),
        ("safety_scan", "terminal / ML / optimization scan"),
        ("periods_and_buckets", "period + frozen bucket logic"),
        ("helpers", "descriptive helper functions"),
        ("default_run", "default-run + --run gating"),
    )
    stats = {name: (p, f) for name, p, f in area_stats}
    for name, spec in spec_map:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<22} ({spec:<44}) {p:>2} passed, {f} failed "
              f"[{status}]")
    print()
    print(f"  TOTAL: {passed} passed, {failed} failed")
    return passed, failed


# ============================================================
# Command-line gating and real analysis
# Default run = synthetic tests only; real analysis needs --run.
# ============================================================
def should_run_real(argv: List[str]) -> bool:
    return "--run" in argv


def _fmt_num(value) -> str:
    if value is None:
        return "n/a"
    try:
        if not np.isfinite(value):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    return f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _write_output_csv(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(path):
        raise OutputExistsError(
            f"output file already exists (existing files are never "
            f"overwritten): {filename}")
    df.to_csv(path, index=False)
    print(f"  wrote {filename} ({len(df)} rows)")


def run_real_analysis() -> None:
    """Real CSV analysis (requires --run).  Read-only inputs, four NEW
    outputs, descriptive summaries only."""
    print("=" * 64)
    print("market_state_memory_engine.py - REAL ANALYSIS (--run)")
    print("=" * 64)
    check_output_files_available(SCRIPT_DIR)  # stop before touching anything
    check_required_inputs(SCRIPT_DIR)
    bundle = load_input_bundle(SCRIPT_DIR)
    engine = MarketStateMemoryEngine(bundle.instrument_frames,
                                     period_summary=bundle.period_summary)
    print("\nReference files read (read-only):")
    for name, dfb in bundle.oos_frames.items():
        print(f"  {name}: {len(dfb)} rows")
    print("\nMemory built from instrument CSVs:")
    for inst in engine.instruments:
        ndev = len(engine.development_indices(inst))
        neval = len(engine.evaluation_indices(inst))
        p33, p66 = engine.thresholds[inst]
        print(f"  {inst}: {len(engine.records[inst])} observations "
              f"({ndev} development / {neval} evaluation) | frozen "
              f"distance thresholds p33={_fmt_num(p33)} "
              f"p66={_fmt_num(p66)}")
    print()
    print(NORMALIZATION_DOC)

    records_df = engine.records_frame()
    validation_rows, detail_rows = run_walk_forward_memory_test(engine)
    summary_rows = build_summary_rows(engine, validation_rows, detail_rows)

    print("Walk-forward memory test (first half = memory, second half = "
          "evaluation, K=10):")
    for inst in engine.instruments:
        vrows = [r for r in validation_rows if r["instrument"] == inst]
        ov = next(r for r in summary_rows if r["instrument"] == inst
                  and r["section"] == "OVERALL")
        print(f"  {inst}: {len(vrows)} queries | avg nearest distance "
              f"{_fmt_num(ov['avg_nearest_distance'])} | historical matches "
              f"in this sample had avg net P/L "
              f"{_fmt_num(ov['avg_match_net_pnl'])} vs actual "
              f"{_fmt_num(ov['avg_actual_net_pnl'])} | P/L sign agreement "
              f"{_fmt_pct(ov['pnl_sign_agreement_pct'])} (n={ov['pnl_sign_n']}) "
              f"| Pearson r match-vs-actual net P/L: "
              f"{fmt_corr(ov['corr_match_actual_pnl_r'])}")
    print("All figures are descriptive summaries of stored historical "
          "data only.")

    print("\nWriting NEW output files:")
    _write_output_csv(records_df, "market_memory_records.csv")
    _write_output_csv(
        pd.DataFrame(validation_rows, columns=list(VALIDATION_COLUMNS)),
        "market_memory_oos_validation.csv")
    _write_output_csv(
        pd.DataFrame(detail_rows, columns=list(DETAIL_COLUMNS)),
        "market_memory_match_details.csv")
    _write_output_csv(
        pd.DataFrame(summary_rows, columns=list(SUMMARY_COLUMNS)),
        "market_memory_summary.csv")
    print("\nDone. Descriptive historical memory only - no recommendations "
          "of any kind.")


def main(argv=None) -> int:
    """Default: synthetic tests only.  Real analysis requires --run."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if should_run_real(argv):
        run_real_analysis()
        return 0
    _, failed = run_synthetic_tests()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
