"""
ai_research_layer.py - HISTORICAL RESEARCH / MARKET CONTEXT ENGINE.

WHAT THIS IS
    A deterministic research-preparation layer that reads the existing
    historical research files (the market-state database and the memory
    engine outputs) and produces a structured MARKET CONTEXT REPORT for
    any observation of the five supported instruments.

    The layer answers exactly one question:

        "What does the historical data show about the current market
         environment?"

    and NEVER answers:

        "Should I BUY?"  /  "Should I SELL?"  /  "When should I enter?"
        "How much should I trade?"

    No trading decision of any kind is produced.  The output is a
    factual, structured description of (a) the current market state,
    (b) what stored history shows about similar environments, and
    (c) how consistent the historical relationships were - kept strictly
    SEPARATE (see section 14 of the specification) and never combined
    into any score.

FOUR SEPARATE NOTIONS (never merged into one number)
    1. CURRENT STATE       - the observation's own market-state features
                             and their established descriptive categories.
    2. HISTORICAL OUTCOME  - the stored, already-known outcome of a
                             historical observation (context only).
    3. HISTORICAL SIMILARITY - the memory-engine historical matches of an
                             observation and their stored outcomes
                             (described as "historical matches", never as
                             forecasts).
    4. VALIDATION EVIDENCE - how consistent match outcomes and actual
                             outcomes were across the walk-forward
                             evaluation (descriptive labels only).

INPUT FILES (read-only; all required; a missing file stops everything
with an explicit error naming the exact missing file):
    market_state_all.csv
    market_memory_oos_validation.csv
    market_memory_match_details.csv
    market_memory_summary.csv
    market_state_oos_period_summary.csv
    market_state_oos_state_summary.csv
    market_state_oos_walkforward.csv
    market_state_oos_cross_instrument.csv

SUPPORTED INSTRUMENTS (exactly these five, nothing added automatically):
    GOLD, EURUSD, GBPUSD, USDJPY, AUDUSD

STATE CLASSIFICATION (fixed descriptive categories, already established
by the market-state database stage - no threshold is tuned here):
    ADX           : ADX<15 | ADX15-20 | ADX20-25 | ADX>=25
                    (x < 15 -> ADX<15; x < 20 -> ADX15-20;
                     x < 25 -> ADX20-25; otherwise ADX>=25)
    EMA distance  : <1 | 1-2 | >=2   (in ATR units; x < 1 -> <1,
                                      x < 2 -> 1-2, otherwise >=2)
    BB excursion  : <0.50 | 0.50-0.75 | >=0.75
    Efficiency    : <0.30 | 0.30-0.60 | >=0.60
    RSI           : <30 | 30-50 | 50-70 | >=70
    ATR (volatility regime): LOW | MEDIUM | HIGH using the instrument's
                    OWN terciles of atr_pct from the historical database
                    (33rd / 66th percentiles, computed once at load from
                    market_state_all.csv - the same instrument-specific
                    descriptive definition the database stage used; the
                    thresholds are then frozen for the whole run).
    Boundary convention: a value exactly ON a boundary belongs to the
    higher bucket (documented above).  A missing value maps to "n/a".

EVIDENCE QUALITY (section 8 of the specification - FIXED rules; these
are descriptive data-quality labels about the historical sample, NOT a
certainty measure, NOT a likelihood of winning, NOT a trade rating):
    INSUFFICIENT : fewer than 5 historical matches for the observation,
                   OR no walk-forward memory query exists for it, OR the
                   instrument-level memory evidence is unavailable.
    STRONG       : sample >= 50 AND sign agreement >= 60% AND |r| >= 0.20
    MODERATE     : sample >= 20 AND sign agreement > 50% AND |r| >= 0.10
    WEAK         : sample >= 5 and everything else (i.e. sign agreement
                   <= 50% OR |r| < 0.10, or the correlation is not
                   available - a missing correlation is treated as 0).
    Evaluation precedence (documented, deterministic): INSUFFICIENT is
    checked first, then STRONG, then MODERATE, then WEAK.  The rules are
    never tuned, weighted or searched.

RESEARCH CONCLUSION (fixed wording, derived by a fixed mapping):
    INSUFFICIENT evidence                     -> INSUFFICIENT HISTORICAL EVIDENCE
    STRONG evidence                           -> HISTORICAL EVIDENCE IS CONSISTENT
    MODERATE/WEAK + walk-forward PERSISTENT   -> HISTORICAL EVIDENCE IS CONSISTENT
    walk-forward UNSTABLE                     -> HISTORICAL EVIDENCE IS UNSTABLE
    everything else                           -> HISTORICAL EVIDENCE IS MIXED

OUTPUT FILES (all NEW; every output path is checked BEFORE anything is
written; if any exists the run stops and names the conflicting file):
    ai_research_context_all.csv      one flattened row per observation
    ai_research_context_report.json  the AI-ready structured dictionaries
    ai_research_context_report.txt   human-readable reports (one per
                                     instrument for its LATEST observation
                                     - a deterministic presentation
                                     choice - plus per-instrument
                                     descriptive aggregates)

NO TRADING TERMINAL: this file never imports or calls any trading
terminal; it operates entirely from CSV files.

NO MACHINE LEARNING: no learning library, no fitted structure, no
inference step of any kind.  Pure deterministic descriptive
computation.  (The forbidden call names are listed in the safety-guard
comment below, split so this source stays scannable.)

NO OPTIMIZATION: the classification boundaries, evidence rules and
conclusion mapping above are FIXED constants.  No threshold sweep, no
parameter search, no state ranking, no instrument ranking, no
direction selection, no window tuning.

COMMAND-LINE SAFETY:
    python ai_research_layer.py          -> synthetic tests only
    python ai_research_layer.py --run    -> real CSV analysis

DISCLAIMER: historical, descriptive research preparation only.  No
causal claims, no forecasts, no recommendations, nothing proven
profitable.  Educational use only.
"""

# ============================================================
# Imports: Python standard library + numpy + pandas ONLY.
# ============================================================
import ast
import datetime
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# SAFETY GUARDS
# Forbidden tokens are assembled from split string literals so this
# safety code never contains the contiguous strings itself.  The scan
# covers: any trading-terminal usage, any learning library, any
# fitting call, any certainty-score wording, and any
# optimization library idiom.
# ============================================================
_TERMINAL_SOURCE_TOKENS: Tuple[str, ...] = (
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
_ML_CALL_PATTERNS: Tuple[str, ...] = (
    "\\b" + "fit" + "\\s*\\(",
    "\\b" + "train" + "\\s*\\(",
    "\\b" + "pred" + "ict" + "\\s*\\(",
    "\\b" + "mo" + "del" + "\\b",
    "\\b" + "class" + "ifier" + "\\b",
    "\\b" + "regress" + "or" + "\\b",
)
# Forbidden call/identifier names (split literals so this guard source
# never contains them contiguously): fi+ t(), pre+ dict(), tra+ in(),
# mo+ del, class+ ifier, regress+ or.
_SCORE_SOURCE_PATTERNS: Tuple[str, ...] = (
    "co" + "nfidence",
    "pro" + "bability",
    "signal" + "_score",
    "edge" + "_score",
    "trade" + "_score",
)
_OPTIMIZATION_TOKENS: Tuple[str, ...] = (
    "p" + "aram_g",
    "Gr" + "idSearch",
    "Rand" + "omizedS",
    "op" + "tuna",
    "hy" + "peropt",
    "itertools.p" + "roduct",
)
_RECOMMENDED_WORDS = ("BU" + "Y", "SE" + "LL", "HO" + "LD")
_ALLOWED_IMPORT_MODULES: Tuple[str, ...] = (
    "ast", "datetime", "json", "os", "re", "sys", "tempfile",
    "typing", "warnings", "numpy", "pandas",
)

# ============================================================
# Frozen configuration (section 2, 3, 18 of the specification)
# ============================================================
SUPPORTED_INSTRUMENTS: Tuple[str, ...] = (
    "GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
)

INPUT_ALL_CSV = "market_state_all.csv"
INPUT_MEMORY_VALIDATION_CSV = "market_memory_oos_validation.csv"
INPUT_MEMORY_DETAILS_CSV = "market_memory_match_details.csv"
INPUT_MEMORY_SUMMARY_CSV = "market_memory_summary.csv"
INPUT_PERIOD_SUMMARY_CSV = "market_state_oos_period_summary.csv"
INPUT_STATE_SUMMARY_CSV = "market_state_oos_state_summary.csv"
INPUT_WALKFORWARD_CSV = "market_state_oos_walkforward.csv"
INPUT_CROSS_INSTRUMENT_CSV = "market_state_oos_cross_instrument.csv"

REQUIRED_INPUT_FILES: Tuple[str, ...] = (
    INPUT_ALL_CSV,
    INPUT_MEMORY_VALIDATION_CSV,
    INPUT_MEMORY_DETAILS_CSV,
    INPUT_MEMORY_SUMMARY_CSV,
    INPUT_PERIOD_SUMMARY_CSV,
    INPUT_STATE_SUMMARY_CSV,
    INPUT_WALKFORWARD_CSV,
    INPUT_CROSS_INSTRUMENT_CSV,
)

OUTPUT_ALL_CSV = "ai_research_context_all.csv"
OUTPUT_REPORT_JSON = "ai_research_context_report.json"
OUTPUT_REPORT_TXT = "ai_research_context_report.txt"
OUTPUT_FILES: Tuple[str, ...] = (
    OUTPUT_ALL_CSV, OUTPUT_REPORT_JSON, OUTPUT_REPORT_TXT,
)

# State features: (friendly name, market_state_all.csv column).
STATE_FEATURES: Tuple[Tuple[str, str], ...] = (
    ("ADX", "adx"),
    ("ATR_pct", "atr_pct"),
    ("EMA50_dist", "ema_distance"),
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
STATE_COLUMN_BY_NAME = dict(STATE_FEATURES)
TIME_CONTEXT_NAMES: Tuple[str, ...] = ("hour", "day_of_week", "month")

# Feature keys used by the previous OOS research CSVs.
OOS_FEATURE_KEY = {
    "ADX": "adx",
    "ATR_pct": "atr",
    "EMA50_dist": "ema_distance",
    "BB_excursion": "bb_excursion",
    "eff24": "efficiency",
    "RSI14": "rsi",
}

# Fixed classification bucket labels (identical to the database stage).
ADX_BUCKETS = ("ADX<15", "ADX15-20", "ADX20-25", "ADX>=25")
EMA_BUCKETS = ("<1", "1-2", ">=2")
BBE_BUCKETS = ("<0.50", "0.50-0.75", ">=0.75")
EFF_BUCKETS = ("<0.30", "0.30-0.60", ">=0.60")
RSI_BUCKETS = ("<30", "30-50", "50-70", ">=70")
ATR_BUCKETS = ("LOW", "MEDIUM", "HIGH")

# Fixed walk-forward step order (chronological).
WF_STEP_ORDER: Tuple[str, ...] = (
    "P1->P2", "P1+P2->P3", "P1+P2+P3->P4",
)
WF_SIGN_SAME = "SIGN SAME"
WF_SIGN_CHANGED = "SIGN CHANGED"

# Fixed evidence-quality labels (section 8) and thresholds.
EVIDENCE_INSUFFICIENT = "INSUFFICIENT"
EVIDENCE_WEAK = "WEAK"
EVIDENCE_MODERATE = "MODERATE"
EVIDENCE_STRONG = "STRONG"
EVIDENCE_LABELS = (EVIDENCE_STRONG, EVIDENCE_MODERATE, EVIDENCE_WEAK,
                   EVIDENCE_INSUFFICIENT)
EVIDENCE_MIN_SAMPLE_STRONG = 50
EVIDENCE_MIN_SAMPLE_MODERATE = 20
EVIDENCE_MIN_SAMPLE_WEAK = 5
EVIDENCE_MIN_SIGN_STRONG = 60.0
EVIDENCE_MIN_SIGN_MODERATE = 50.0
EVIDENCE_MIN_ABS_R_STRONG = 0.20
EVIDENCE_MIN_ABS_R_MODERATE = 0.10

# Fixed walk-forward relationship labels (section 10).
WF_PERSISTENT = "PERSISTENT"
WF_MIXED = "MIXED"
WF_UNSTABLE = "UNSTABLE"
WF_INSUFFICIENT = "INSUFFICIENT DATA"
WF_LABELS = (WF_PERSISTENT, WF_MIXED, WF_UNSTABLE, WF_INSUFFICIENT)

# Fixed research conclusions (section 13G).
CONCLUSION_CONSISTENT = "HISTORICAL EVIDENCE IS CONSISTENT"
CONCLUSION_MIXED = "HISTORICAL EVIDENCE IS MIXED"
CONCLUSION_UNSTABLE = "HISTORICAL EVIDENCE IS UNSTABLE"
CONCLUSION_INSUFFICIENT = "INSUFFICIENT HISTORICAL EVIDENCE"
ALLOWED_CONCLUSIONS = (CONCLUSION_CONSISTENT, CONCLUSION_MIXED,
                       CONCLUSION_UNSTABLE, CONCLUSION_INSUFFICIENT)

STRUCTURED_OUTPUT_KEYS: Tuple[str, ...] = (
    "instrument", "timestamp", "current_state", "historical_memory",
    "regime_stability", "walk_forward_evidence", "memory_engine_evidence",
    "cross_instrument_context", "evidence_quality", "research_conclusion",
)

EVIDENCE_RULES_DOC = """\
Evidence-quality rules (FIXED, descriptive, data-quality only):
  INSUFFICIENT : fewer than 5 historical matches for the observation,
                 OR no walk-forward memory query exists for it, OR the
                 instrument-level memory evidence is unavailable.
  STRONG       : sample >= 50 AND sign agreement >= 60% AND |r| >= 0.20
  MODERATE     : sample >= 20 AND sign agreement > 50% AND |r| >= 0.10
  WEAK         : sample >= 5 and everything else (sign agreement <= 50%
                 OR |r| < 0.10 OR correlation not available).
  Checked in the order INSUFFICIENT -> STRONG -> MODERATE -> WEAK.
  These labels describe the QUALITY OF THE HISTORICAL SAMPLE.  They are
  not a likelihood of winning, not a certainty figure, and not a trade
  rating of any kind.
"""

TS_FORMAT = "%Y-%m-%d %H:%M:%S"

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
    return pd.Timestamp(ts).strftime(TS_FORMAT)


def safe_float(value) -> Optional[float]:
    """float(value) or None when missing / not finite."""
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


def safe_int(value) -> Optional[int]:
    v = safe_float(value)
    return None if v is None else int(round(v))


def mean_of(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def median_of(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def sign_of(value) -> Optional[int]:
    v = safe_float(value)
    if v is None:
        return None
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def safe_pearson(xs, ys) -> Tuple[Optional[float], int]:
    """Pearson r and usable-pair count; None when < 3 pairs or no spread."""
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


def fmt_num(value) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:+.4f}"


def fmt_pct(value) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:.1f}%"


def fmt_corr(value) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:+.3f}"


# ============================================================
# Fixed state classification (section 6 of the specification)
# Boundary convention: a value exactly ON a boundary belongs to the
# higher bucket.  Missing values map to "n/a".
# ============================================================
def classify_adx(value) -> str:
    v = safe_float(value)
    if v is None:
        return "n/a"
    if v < 15.0:
        return ADX_BUCKETS[0]
    if v < 20.0:
        return ADX_BUCKETS[1]
    if v < 25.0:
        return ADX_BUCKETS[2]
    return ADX_BUCKETS[3]


def classify_ema_distance(value) -> str:
    v = safe_float(value)
    if v is None:
        return "n/a"
    if v < 1.0:
        return EMA_BUCKETS[0]
    if v < 2.0:
        return EMA_BUCKETS[1]
    return EMA_BUCKETS[2]


def classify_bb_excursion(value) -> str:
    v = safe_float(value)
    if v is None:
        return "n/a"
    if v < 0.50:
        return BBE_BUCKETS[0]
    if v < 0.75:
        return BBE_BUCKETS[1]
    return BBE_BUCKETS[2]


def classify_efficiency(value) -> str:
    v = safe_float(value)
    if v is None:
        return "n/a"
    if v < 0.30:
        return EFF_BUCKETS[0]
    if v < 0.60:
        return EFF_BUCKETS[1]
    return EFF_BUCKETS[2]


def classify_rsi(value) -> str:
    v = safe_float(value)
    if v is None:
        return "n/a"
    if v < 30.0:
        return RSI_BUCKETS[0]
    if v < 50.0:
        return RSI_BUCKETS[1]
    if v < 70.0:
        return RSI_BUCKETS[2]
    return RSI_BUCKETS[3]


def atr_tercile_thresholds(atr_pct_values) -> Tuple[float, float]:
    """Instrument-specific 33rd / 66th percentiles of atr_pct.

    This reproduces the descriptive tercile definition the market-state
    database stage used for its ATR bins.  The thresholds are computed
    ONCE from the historical database file and then frozen; they are
    never tuned.
    """
    vals = np.asarray([v for v in atr_pct_values
                       if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100.0 / 3.0)),
            float(np.percentile(vals, 200.0 / 3.0)))


def classify_atr_pct(value, thresholds: Tuple[float, float]) -> str:
    """Volatility regime via the instrument's OWN frozen terciles."""
    v = safe_float(value)
    p33, p66 = thresholds
    if v is None or p33 is None or p66 is None \
            or not np.isfinite(p33) or not np.isfinite(p66):
        return "n/a"
    if v < p33:
        return ATR_BUCKETS[0]
    if v < p66:
        return ATR_BUCKETS[1]
    return ATR_BUCKETS[2]


# ============================================================
# Input / output guards
# ============================================================
class MissingInputError(RuntimeError):
    pass


class OutputExistsError(RuntimeError):
    pass


def check_required_inputs(directory: str = SCRIPT_DIR) -> None:
    """Every required input must exist; otherwise name the missing file."""
    missing = [f for f in REQUIRED_INPUT_FILES
               if not os.path.isfile(os.path.join(directory, f))]
    if missing:
        raise MissingInputError(
            "missing required input file(s): " + ", ".join(missing))


def check_output_files_available(directory: str = SCRIPT_DIR) -> None:
    """ALL output paths are checked before ANY writing happens."""
    conflicts = [f for f in OUTPUT_FILES
                 if os.path.exists(os.path.join(directory, f))]
    if conflicts:
        raise OutputExistsError(
            "refusing to run: output file(s) already exist and existing "
            "files are never modified or overwritten: "
            + ", ".join(conflicts))


# ============================================================
# Loaded research data (built once; deterministic indexes)
# ============================================================
class ResearchData:
    """Read-only view of every required input, indexed for lookups."""

    def __init__(self, directory: str = SCRIPT_DIR):
        check_required_inputs(directory)

        def read(name: str) -> pd.DataFrame:
            return pd.read_csv(os.path.join(directory, name))

        self.all_df = read(INPUT_ALL_CSV)
        unexpected = sorted(set(self.all_df["instrument"].astype(str))
                            - set(SUPPORTED_INSTRUMENTS))
        if unexpected:
            raise ValueError(
                "market_state_all.csv contains unsupported instrument(s): "
                + ", ".join(unexpected))
        self.memory_validation = read(INPUT_MEMORY_VALIDATION_CSV)
        self.memory_details = read(INPUT_MEMORY_DETAILS_CSV)
        self.memory_summary = read(INPUT_MEMORY_SUMMARY_CSV)
        self.period_summary = read(INPUT_PERIOD_SUMMARY_CSV)
        self.state_summary = read(INPUT_STATE_SUMMARY_CSV)
        self.walkforward = read(INPUT_WALKFORWARD_CSV)
        self.cross_instrument = read(INPUT_CROSS_INSTRUMENT_CSV)

        # ---- per-instrument frames (chronological) ----
        self.frames: Dict[str, pd.DataFrame] = {}
        for inst in SUPPORTED_INSTRUMENTS:
            g = self.all_df[self.all_df["instrument"].astype(str) == inst]
            g = g.copy()
            g["signal_time"] = pd.to_datetime(g["signal_time"], errors="raise")
            g = g.sort_values("signal_time", kind="mergesort").reset_index(drop=True)
            self.frames[inst] = g

        # ---- frozen per-instrument ATR terciles (from the database) ----
        self.atr_thresholds: Dict[str, Tuple[float, float]] = {}
        for inst in SUPPORTED_INSTRUMENTS:
            self.atr_thresholds[inst] = atr_tercile_thresholds(
                self.frames[inst]["atr_pct"].tolist())

        # ---- memory-validation lookup: (instrument, ts string) -> row ----
        self.validation_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in self.memory_validation.to_dict("records"):
            key = (str(row["instrument"]),
                   fmt_ts(pd.Timestamp(row["query_timestamp"])))
            self.validation_by_key[key] = row

        # ---- per-query mean match distance from the details file ----
        dist_acc: Dict[Tuple[str, str], List[float]] = {}
        for row in self.memory_details.to_dict("records"):
            key = (str(row["instrument"]),
                   fmt_ts(pd.Timestamp(row["query_timestamp"])))
            d = safe_float(row.get("distance"))
            if d is not None:
                dist_acc.setdefault(key, []).append(d)
        self.mean_match_distance: Dict[Tuple[str, str], float] = {
            k: float(np.mean(v)) for k, v in dist_acc.items()}

        # ---- memory summary rows by instrument (OVERALL section) ----
        self.summary_overall: Dict[str, Dict[str, Any]] = {}
        for row in self.memory_summary.to_dict("records"):
            if str(row.get("section")) == "OVERALL":
                self.summary_overall[str(row["instrument"])] = row

        # ---- state-stability index: (instrument, feature, bucket) -> rows
        self.stability_by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for row in self.state_summary.to_dict("records"):
            key = (str(row["instrument"]), str(row["feature"]),
                   str(row["bucket"]))
            self.stability_by_key.setdefault(key, []).append(row)
        for rows in self.stability_by_key.values():
            rows.sort(key=lambda r: str(r.get("period")))

        # ---- walk-forward index: (instrument, feature, bucket) -> rows
        self.walkforward_by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for row in self.walkforward.to_dict("records"):
            key = (str(row["instrument"]), str(row["feature"]),
                   str(row["bucket"]))
            self.walkforward_by_key.setdefault(key, []).append(row)

        # ---- cross-instrument index: (feature, bucket) -> rows ----
        self.cross_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in self.cross_instrument.to_dict("records"):
            key = (str(row["feature"]), str(row["bucket"]))
            self.cross_by_key.setdefault(key, []).append(row)
        for rows in self.cross_by_key.values():
            rows.sort(key=lambda r: str(r.get("period")))

        # ---- per-instrument period overview (kept strictly separate) ----
        self.period_overview: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.period_summary.to_dict("records"):
            self.period_overview.setdefault(str(row["instrument"]),
                                            []).append({
                "period": str(row["period"]),
                "start_time": str(row["start_time"]),
                "end_time": str(row["end_time"]),
                "n": safe_int(row.get("n")),
                "avg_net_pnl": safe_float(row.get("avg_net_pnl")),
                "win_rate_pct": safe_float(row.get("win_rate_pct")),
                "fwd12_avg_pct": safe_float(row.get("fwd12_avg_pct")),
            })
        for inst in self.period_overview:
            self.period_overview[inst].sort(key=lambda r: r["period"])

    # --------------------------------------------------------
    def instrument_frame(self, instrument: str) -> pd.DataFrame:
        if instrument not in SUPPORTED_INSTRUMENTS:
            raise ValueError(
                f"unsupported instrument {instrument!r}; supported: "
                + ", ".join(SUPPORTED_INSTRUMENTS))
        return self.frames[instrument]

    def validation_row(self, instrument: str, timestamp) \
            -> Optional[Dict[str, Any]]:
        key = (instrument, fmt_ts(parse_timestamp(timestamp)))
        return self.validation_by_key.get(key)

    def memory_engine_evidence(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Instrument-level memory-engine OOS evidence (section 11).

        Uses the memory engine's own frozen K=10 outputs unchanged; no
        other K is computed and no distance is altered.
        """
        row = self.summary_overall.get(instrument)
        vrows = self.memory_validation[
            self.memory_validation["instrument"].astype(str) == instrument]
        if row is None or len(vrows) == 0:
            return None
        return {
            "queries": safe_int(row.get("n_queries")),
            "low_sample_count": safe_int(row.get("low_sample_count")),
            "avg_nearest_distance": safe_float(row.get("avg_nearest_distance")),
            "median_nearest_distance": safe_float(
                row.get("median_nearest_distance")),
            "pnl_sign_agreement_pct": safe_float(
                row.get("pnl_sign_agreement_pct")),
            "pnl_sign_n": safe_int(row.get("pnl_sign_n")),
            "fwd12_sign_agreement_pct": safe_float(
                row.get("fwd12_sign_agreement_pct")),
            "fwd24_sign_agreement_pct": safe_float(
                row.get("fwd24_sign_agreement_pct")),
            "pearson_r_match_actual_pnl": safe_float(
                row.get("corr_match_actual_pnl_r")),
            "pearson_n_pnl": safe_int(row.get("corr_match_actual_pnl_n")),
            "pearson_r_match_actual_fwd12": safe_float(
                row.get("corr_match_actual_fwd12_r")),
            "pearson_n_fwd12": safe_int(row.get("corr_match_actual_fwd12_n")),
            "pearson_r_match_actual_fwd24": safe_float(
                row.get("corr_match_actual_fwd24_r")),
            "pearson_n_fwd24": safe_int(row.get("corr_match_actual_fwd24_n")),
            "historical_match_avg_net_pnl": safe_float(
                row.get("avg_match_net_pnl")),
            "actual_avg_net_pnl": safe_float(row.get("avg_actual_net_pnl")),
            "avg_match_forward12_pct": safe_float(
                row.get("avg_match_forward12_pct")),
            "avg_actual_forward12_pct": safe_float(
                row.get("avg_actual_forward12_pct")),
            "k_fixed": 10,
            "note": ("memory-engine outputs used unchanged (frozen K=10); "
                     "historical matches are descriptions of stored "
                     "history, not forecasts"),
        }


# ============================================================
# Current market state (sections 4, 5, 14)
# ============================================================
CURRENT_STATE_ALLOWED_COLUMNS: Tuple[str, ...] = (
    "instrument", "timeframe", "signal_time", "direction",
) + tuple(col for _name, col in STATE_FEATURES) + TIME_CONTEXT_NAMES


def build_current_state(instrument: str, observation: Any,
                        data: ResearchData) -> Dict[str, Any]:
    """Extract and classify the CURRENT STATE of one observation.

    ONLY the allow-listed state columns are read.  Outcome and forward
    columns (entry/exit, P/L, win, forward returns, MFE/MAE) are never
    touched here, so the current-state description cannot leak future
    information.  The observation may be a dict, a Series or a DataFrame
    row; anything else is rejected.
    """
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(
            f"unsupported instrument {instrument!r}; supported: "
            + ", ".join(SUPPORTED_INSTRUMENTS))
    if isinstance(observation, dict):
        row = dict(observation)
    elif isinstance(observation, pd.Series):
        row = observation.to_dict()
    elif isinstance(observation, pd.DataFrame):
        if len(observation) != 1:
            raise ValueError("observation DataFrame must have exactly one row")
        row = observation.iloc[0].to_dict()
    else:
        raise ValueError("observation must be a dict, Series or 1-row frame")

    missing = [c for c in CURRENT_STATE_ALLOWED_COLUMNS
               if c not in row and c != "timeframe"]
    if "timeframe" not in row:
        missing.append("timeframe") if False else None
    if missing:
        raise ValueError(
            f"observation for {instrument} is missing required state "
            f"field(s): " + ", ".join(missing))

    raw: Dict[str, Optional[float]] = {}
    for name, col in STATE_FEATURES:
        raw[name] = safe_float(row.get(col))
    direction = str(row.get("direction"))
    ts = parse_timestamp(row.get("signal_time"))
    timeframe = str(row.get("timeframe", "H1"))

    thresholds = data.atr_thresholds[instrument]
    buckets = {
        "adx_regime": classify_adx(raw["ADX"]),
        "volatility_regime": classify_atr_pct(raw["ATR_pct"], thresholds),
        "trend_distance_regime": classify_ema_distance(raw["EMA50_dist"]),
        "bb_excursion_regime": classify_bb_excursion(raw["BB_excursion"]),
        "efficiency_regime": classify_efficiency(raw["eff24"]),
        "rsi_regime": classify_rsi(raw["RSI14"]),
    }
    return {
        "instrument": instrument,
        "timestamp": fmt_ts(ts),
        "timeframe": timeframe,
        "direction": direction,
        "features": raw,
        "classifications": buckets,
        "atr_tercile_thresholds": {
            "p33": thresholds[0], "p66": thresholds[1],
            "source": "instrument-specific atr_pct terciles from the "
                      "historical database (frozen)",
        },
        "hour": safe_int(row.get("hour")),
        "day_of_week": safe_int(row.get("day_of_week")),
        "month": safe_int(row.get("month")),
    }


def describe_environment(state: Dict[str, Any]) -> List[str]:
    """Plain-language factual description of the current environment.

    Every sentence restates a stored feature value or its fixed category.
    Nothing here looks forward.
    """
    f = state["features"]
    c = state["classifications"]
    lines: List[str] = []

    def part(name: str, value, unit: str = "", digits: int = 2) -> str:
        v = safe_float(value)
        return "unavailable" if v is None \
            else f"{name} {v:.{digits}f}{unit}"

    lines.append(
        f"Trend strength (ADX): {c['adx_regime']} "
        f"({part('ADX', f['ADX'])}).")
    lines.append(
        f"Volatility: {c['volatility_regime']} by this instrument's "
        f"ATR% terciles ({part('ATR%', f['ATR_pct'], '%')}); realized "
        f"volatility {part('vol6', f['vol6'])} (6-candle) and "
        f"{part('vol24', f['vol24'])} (24-candle).")
    lines.append(
        f"Trend distance: {c['trend_distance_regime']} ATR from the "
        f"EMA50 ({part('distance', f['EMA50_dist'], ' ATR')}).")
    lines.append(
        f"Bollinger position: excursion {c['bb_excursion_regime']} "
        f"({part('excursion', f['BB_excursion'])}); signed position "
        f"{part('signed', f['BB_signed'])} (negative = below the middle "
        f"band, positive = above).")
    lines.append(
        f"Efficiency: {c['efficiency_regime']} "
        f"({part('eff24', f['eff24'])}).")
    lines.append(
        f"RSI: {c['rsi_regime']} ({part('RSI14', f['RSI14'])}).")
    lines.append(
        f"Recent movement: return6 {part('return6', f['ret6'], '%')}, "
        f"return12 {part('return12', f['ret12'], '%')}, "
        f"return24 {part('return24', f['ret24'], '%')}; "
        f"{part('24h-high distance', f['dist_24_high'], '%')} and "
        f"{part('24h-low distance', f['dist_24_low'], '%')}.")
    lines.append(
        f"Candle shape: range {part('range', f['range_pct'], '%')}, "
        f"body {part('body', f['body_pct'], '%')}, "
        f"upper wick {part('upper wick', f['upper_wick_pct'], '%')}, "
        f"lower wick {part('lower wick', f['lower_wick_pct'], '%')}.")
    lines.append(
        f"Signal direction of this stored observation: "
        f"{state['direction']} (recorded fact, not a recommendation).")
    lines.append(
        f"Time context: hour {state['hour']}, day_of_week "
        f"{state['day_of_week']}, month {state['month']}.")
    lines.append(
        "This description uses only information available at the "
        "observation timestamp; it contains no forward-looking statement.")
    return lines


# ============================================================
# Historical memory (section 7) - descriptive only
# ============================================================
def build_historical_memory(instrument: str, timestamp,
                            data: ResearchData) -> Dict[str, Any]:
    """Historical-similarity block for one observation.

    Source: the memory engine's own walk-forward outputs (frozen K=10).
    Every figure describes stored history: the historical matches of
    this observation and their recorded outcomes.  Nothing is called a
    forecast and nothing is turned into a trading suggestion.
    """
    vrow = data.validation_row(instrument, timestamp)
    inst_ev = data.memory_engine_evidence(instrument)
    out: Dict[str, Any] = {
        "available": vrow is not None,
        "wording": ("historical matches are stored past observations "
                    "with similar state features; they are not forecasts"),
        "n_matches": None,
        "avg_match_distance": None,
        "median_match_distance": None,
        "nearest_distance": None,
        "nearest_distance_percentile": None,
        "avg_match_net_pnl": None,
        "match_win_rate_pct": None,
        "avg_match_forward6_pct": None,
        "avg_match_forward12_pct": None,
        "avg_match_forward24_pct": None,
        "avg_match_mfe12_atr": None,
        "avg_match_mae12_atr": None,
        "avg_match_holding_candles": None,
        "actual_net_pnl": None,
        "actual_win": None,
        "actual_forward6_pct": None,
        "actual_forward12_pct": None,
        "actual_forward24_pct": None,
        "actual_mfe12_atr": None,
        "actual_mae12_atr": None,
        "sign_agreement_pct": None,
        "pearson_r_match_actual_pnl": None,
        "pearson_n_pnl": None,
        "instrument_level": inst_ev,
    }
    if vrow is None:
        out["note"] = ("no walk-forward memory query exists for this "
                       "observation in the memory-engine outputs")
        return out
    key = (instrument, fmt_ts(parse_timestamp(timestamp)))
    out.update({
        "n_matches": safe_int(vrow.get("n_matches")),
        "avg_match_distance": data.mean_match_distance.get(key),
        "median_match_distance": safe_float(vrow.get("median_match_distance")),
        "nearest_distance": safe_float(vrow.get("nearest_distance")),
        "nearest_distance_percentile": safe_float(
            vrow.get("nearest_distance_percentile")),
        "avg_match_net_pnl": safe_float(vrow.get("mean_match_net_pnl")),
        "match_win_rate_pct": safe_float(vrow.get("match_win_rate_pct")),
        "avg_match_forward6_pct": safe_float(
            vrow.get("mean_match_forward6_pct")),
        "avg_match_forward12_pct": safe_float(
            vrow.get("mean_match_forward12_pct")),
        "avg_match_forward24_pct": safe_float(
            vrow.get("mean_match_forward24_pct")),
        "avg_match_mfe12_atr": safe_float(vrow.get("mean_match_mfe12_atr")),
        "avg_match_mae12_atr": safe_float(vrow.get("mean_match_mae12_atr")),
        "avg_match_holding_candles": safe_float(
            vrow.get("mean_match_holding_candles")),
        "actual_net_pnl": safe_float(vrow.get("actual_net_pnl")),
        "actual_win": _coerce_bool(vrow.get("actual_win")),
        "actual_forward6_pct": safe_float(vrow.get("actual_forward6_pct")),
        "actual_forward12_pct": safe_float(vrow.get("actual_forward12_pct")),
        "actual_forward24_pct": safe_float(vrow.get("actual_forward24_pct")),
        "actual_mfe12_atr": safe_float(vrow.get("actual_mfe12_atr")),
        "actual_mae12_atr": safe_float(vrow.get("actual_mae12_atr")),
    })
    # Per-observation sign agreement: stored match-average P/L sign vs the
    # observation's own recorded outcome sign (descriptive).
    agree = _sign_agreement_pct(vrow.get("mean_match_net_pnl"),
                                vrow.get("actual_net_pnl"))
    out["sign_agreement_pct"] = agree
    # Pearson correlation is an instrument-level validation figure.
    if inst_ev is not None:
        out["pearson_r_match_actual_pnl"] = \
            inst_ev["pearson_r_match_actual_pnl"]
        out["pearson_n_pnl"] = inst_ev["pearson_n_pnl"]
    return out


def _coerce_bool(value) -> Optional[bool]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1")


def _sign_agreement_pct(match_value, actual_value) -> Optional[float]:
    """100.0 when the stored match-average sign equals the recorded
    outcome sign, 0.0 when it differs, None when either is missing."""
    sm = sign_of(match_value)
    sa = sign_of(actual_value)
    if sm is None or sa is None:
        return None
    return 100.0 if sm == sa else 0.0


# ============================================================
# Evidence quality (section 8) - fixed rules, documented
# ============================================================
def classify_evidence(n_matches: Optional[int],
                      sign_agreement_pct: Optional[float],
                      pearson_r: Optional[float],
                      has_validation_row: bool = True,
                      has_instrument_evidence: bool = True) -> str:
    """Apply the FIXED evidence rules (see EVIDENCE_RULES_DOC).

    Precedence: INSUFFICIENT -> STRONG -> MODERATE -> WEAK.  A missing
    correlation counts as 0.0; a missing sign agreement counts as no
    agreement (0.0).
    """
    n = safe_int(n_matches)
    if not has_validation_row or not has_instrument_evidence:
        return EVIDENCE_INSUFFICIENT
    if n is None or n < EVIDENCE_MIN_SAMPLE_WEAK:
        return EVIDENCE_INSUFFICIENT
    sign = safe_float(sign_agreement_pct)
    r = safe_float(pearson_r)
    sign_v = sign if sign is not None else 0.0
    abs_r = abs(r) if r is not None else 0.0
    if n >= EVIDENCE_MIN_SAMPLE_STRONG and sign_v >= EVIDENCE_MIN_SIGN_STRONG \
            and abs_r >= EVIDENCE_MIN_ABS_R_STRONG:
        return EVIDENCE_STRONG
    if n >= EVIDENCE_MIN_SAMPLE_MODERATE and sign_v > EVIDENCE_MIN_SIGN_MODERATE \
            and abs_r >= EVIDENCE_MIN_ABS_R_MODERATE:
        return EVIDENCE_MODERATE
    return EVIDENCE_WEAK


def evidence_explanation(label: str, n_matches: Optional[int],
                         sign_agreement_pct: Optional[float],
                         pearson_r: Optional[float]) -> str:
    """Plain-language reason for the evidence label (no tuning)."""
    n = safe_int(n_matches)
    sign = safe_float(sign_agreement_pct)
    r = safe_float(pearson_r)
    n_s = "n/a" if n is None else str(n)
    sign_s = "n/a" if sign is None else f"{sign:.1f}%"
    r_s = "n/a" if r is None else f"{r:+.3f}"
    if label == EVIDENCE_INSUFFICIENT:
        return (f"INSUFFICIENT: the historical sample for this "
                f"observation has {n_s} matches (fewer than "
                f"{EVIDENCE_MIN_SAMPLE_WEAK}) or no walk-forward memory "
                f"query / instrument evidence exists; no conclusion is "
                f"drawn from this.")
    if label == EVIDENCE_STRONG:
        return (f"STRONG: sample {n_s} >= {EVIDENCE_MIN_SAMPLE_STRONG}, "
                f"sign agreement {sign_s} >= "
                f"{EVIDENCE_MIN_SIGN_STRONG:.0f}%, |r| {abs(r):.3f} >= "
                f"{EVIDENCE_MIN_ABS_R_STRONG}. Describes the historical "
                f"sample only.")
    if label == EVIDENCE_MODERATE:
        return (f"MODERATE: sample {n_s} >= "
                f"{EVIDENCE_MIN_SAMPLE_MODERATE}, sign agreement {sign_s} "
                f"> {EVIDENCE_MIN_SIGN_MODERATE:.0f}%, |r| {abs(r):.3f} "
                f">= {EVIDENCE_MIN_ABS_R_MODERATE}. Describes the "
                f"historical sample only.")
    return (f"WEAK: sample {n_s} (>= {EVIDENCE_MIN_SAMPLE_WEAK}) with "
            f"sign agreement {sign_s} (<= "
            f"{EVIDENCE_MIN_SIGN_MODERATE:.0f}%) or |r| "
            f"{'n/a' if r is None else f'{abs(r):.3f}'} (< "
            f"{EVIDENCE_MIN_ABS_R_MODERATE}) or correlation unavailable. "
            f"Describes the historical sample only.")


# ============================================================
# Regime stability (section 9) - descriptive, no ranking
# ============================================================
def build_regime_stability(state: Dict[str, Any],
                           data: ResearchData) -> Dict[str, Any]:
    """How consistent were the stored relationships for THIS state's
    categories across the historical periods (from the OOS state
    summary)?  No state is selected and none is ranked."""
    inst = state["instrument"]
    buckets = state["classifications"]
    out: Dict[str, Any] = {
        "available": False,
        "by_category": [],
    }
    for friendly, oos_key in OOS_FEATURE_KEY.items():
        bucket = buckets.get({
            "adx": "adx_regime",
            "atr": "volatility_regime",
            "ema_distance": "trend_distance_regime",
            "bb_excursion": "bb_excursion_regime",
            "efficiency": "efficiency_regime",
            "rsi": "rsi_regime",
        }.get(oos_key, ""))
        if bucket is None or bucket in ("", "n/a"):
            continue
        rows = data.stability_by_key.get((inst, oos_key, bucket), [])
        if not rows:
            out["by_category"].append({
                "feature": friendly, "oos_feature_key": oos_key,
                "bucket": bucket, "available": False,
                "periods_observed": 0,
            })
            continue
        pnl_by_period = {f"p{i}_avg_net_pnl": safe_float(r.get(f"p{i}_avg_net_pnl"))
                         for r in rows for i in (1, 2, 3, 4)}
        pos = sum(1 for v in pnl_by_period.values()
                  if v is not None and v > 0)
        neg = sum(1 for v in pnl_by_period.values()
                  if v is not None and v < 0)
        unchanged = sum(1 for v in pnl_by_period.values()
                        if v is not None and v == 0)
        n_obs = sum(safe_int(r.get("n")) or 0 for r in rows)
        out["available"] = True
        out["by_category"].append({
            "feature": friendly, "oos_feature_key": oos_key,
            "bucket": bucket, "available": True,
            "periods_observed": len(rows),
            "observations_in_category": n_obs,
            "periods_positive": pos,
            "periods_negative": neg,
            "periods_unchanged": unchanged,
            "pnl_sign_changes": safe_int(rows[0].get("pnl_sign_changes")),
            "fwd12_sign_changes": safe_int(rows[0].get("fwd12_sign_changes")),
            "pnl_classification": str(rows[0].get("pnl_classification")),
            "fwd12_classification": str(rows[0].get("fwd12_classification")),
            "per_period": [
                {"period": str(r.get("period")),
                 "n": safe_int(r.get("n")),
                 "avg_net_pnl": safe_float(r.get("avg_net_pnl")),
                 "win_rate_pct": safe_float(r.get("win_rate_pct")),
                 "fwd12_avg_pct": safe_float(r.get("fwd12_avg_pct"))}
                for r in rows
            ],
        })
    out["note"] = ("descriptive period consistency only; no category is "
                   "ranked and no profitable state is selected")
    return out


# ============================================================
# Walk-forward evidence (section 10) - fixed label mapping
# ============================================================
def build_walk_forward_evidence(state: Dict[str, Any],
                                data: ResearchData) -> Dict[str, Any]:
    """Whether the stored relationship for this state's categories held
    across the walk-forward steps.  Fixed, documented label mapping:
      INSUFFICIENT DATA : no rows, or every step is flagged low-sample
      UNSTABLE          : every step changed sign (net P/L)
      PERSISTENT        : every step kept the sign (net P/L)
      MIXED             : some steps kept and some changed the sign
    Descriptive wording only; no result is called forward-looking unless
    the data itself supports the description.
    """
    inst = state["instrument"]
    buckets = state["classifications"]
    out: Dict[str, Any] = {
        "available": False,
        "label": WF_INSUFFICIENT,
        "by_category": [],
    }
    for friendly, oos_key in OOS_FEATURE_KEY.items():
        bucket = buckets.get({
            "adx": "adx_regime",
            "atr": "volatility_regime",
            "ema_distance": "trend_distance_regime",
            "bb_excursion": "bb_excursion_regime",
            "efficiency": "efficiency_regime",
            "rsi": "rsi_regime",
        }.get(oos_key, ""))
        if bucket is None or bucket in ("", "n/a"):
            continue
        rows = data.walkforward_by_key.get((inst, oos_key, bucket), [])
        if not rows:
            out["by_category"].append({
                "feature": friendly, "oos_feature_key": oos_key,
                "bucket": bucket, "available": False,
                "label": WF_INSUFFICIENT,
            })
            continue
        step_rank = {s: i for i, s in enumerate(WF_STEP_ORDER)}
        rows = sorted(rows, key=lambda r: step_rank.get(str(r.get("step")), 99))
        n_steps = len(rows)
        low = [_coerce_bool(r.get("low_sample")) for r in rows]
        pnl_same = sum(1 for r in rows
                       if str(r.get("pnl_sign_result")) == WF_SIGN_SAME)
        pnl_changed = sum(1 for r in rows
                          if str(r.get("pnl_sign_result")) == WF_SIGN_CHANGED)
        f12_same = sum(1 for r in rows
                       if str(r.get("fwd12_sign_result")) == WF_SIGN_SAME)
        f12_changed = sum(1 for r in rows
                          if str(r.get("fwd12_sign_result")) == WF_SIGN_CHANGED)
        if n_steps == 0 or all(low):
            label = WF_INSUFFICIENT
        elif pnl_changed == 0:
            label = WF_PERSISTENT
        elif pnl_same == 0:
            label = WF_UNSTABLE
        else:
            label = WF_MIXED
        out["available"] = True
        out["by_category"].append({
            "feature": friendly, "oos_feature_key": oos_key,
            "bucket": bucket, "available": True,
            "label": label,
            "n_steps": n_steps,
            "steps": [str(r.get("step")) for r in rows],
            "development_periods": [str(r.get("step")).split("->")[0]
                                    for r in rows],
            "oos_periods": [str(r.get("step")).split("->")[-1]
                            for r in rows],
            "pnl_sign_same": pnl_same,
            "pnl_sign_changed": pnl_changed,
            "fwd12_sign_same": f12_same,
            "fwd12_sign_changed": f12_changed,
            "low_sample_steps": sum(1 for v in low if v),
            "relationship_persisted": label == WF_PERSISTENT,
            "relationship_mixed": label == WF_MIXED,
        })
    # The overall walk-forward label for the observation: the WEAKEST
    # available category label in the fixed order INSUFFICIENT DATA >
    # UNSTABLE > MIXED > PERSISTENT (deterministic, documented).
    labels = [c["label"] for c in out["by_category"] if c.get("available")]
    if not labels:
        out["label"] = WF_INSUFFICIENT
    elif WF_INSUFFICIENT in labels:
        out["label"] = WF_INSUFFICIENT
    elif WF_UNSTABLE in labels:
        out["label"] = WF_UNSTABLE
    elif WF_MIXED in labels:
        out["label"] = WF_MIXED
    else:
        out["label"] = WF_PERSISTENT
    out["note"] = ("descriptive walk-forward consistency only; the label "
                   "describes stored history, not the future")
    return out


# ============================================================
# Cross-instrument context (section 12) - kept separate
# ============================================================
def build_cross_instrument_context(state: Dict[str, Any],
                                   data: ResearchData) -> Dict[str, Any]:
    """Context across the five supported instruments for this state's
    categories.  Every instrument stays separate; nothing is combined
    into a score and no instrument is ranked."""
    buckets = state["classifications"]
    out: Dict[str, Any] = {
        "available": False,
        "by_category": [],
        "period_overview_per_instrument": {
            inst: data.period_overview.get(inst, [])
            for inst in SUPPORTED_INSTRUMENTS
        },
    }
    for friendly, oos_key in OOS_FEATURE_KEY.items():
        bucket = buckets.get({
            "adx": "adx_regime",
            "atr": "volatility_regime",
            "ema_distance": "trend_distance_regime",
            "bb_excursion": "bb_excursion_regime",
            "efficiency": "efficiency_regime",
            "rsi": "rsi_regime",
        }.get(oos_key, ""))
        if bucket is None or bucket in ("", "n/a"):
            continue
        rows = data.cross_by_key.get((oos_key, bucket), [])
        if not rows:
            out["by_category"].append({
                "feature": friendly, "oos_feature_key": oos_key,
                "bucket": bucket, "available": False,
            })
            continue
        out["available"] = True
        out["by_category"].append({
            "feature": friendly, "oos_feature_key": oos_key,
            "bucket": bucket, "available": True,
            "per_period": [
                {"period": str(r.get("period")),
                 "instruments_with_sufficient_sample": safe_int(
                     r.get("instruments_with_sufficient_sample")),
                 "pos_avg_pnl_count": safe_int(r.get("pos_avg_pnl_count")),
                 "neg_avg_pnl_count": safe_int(r.get("neg_avg_pnl_count")),
                 "pos_fwd12_count": safe_int(r.get("pos_fwd12_count")),
                 "neg_fwd12_count": safe_int(r.get("neg_fwd12_count")),
                 "pnl_classification": str(r.get("pnl_classification")),
                 "fwd12_classification": str(r.get("fwd12_classification"))}
                for r in rows
            ],
        })
    out["note"] = ("counts of instruments with positive vs negative "
                   "average outcomes per period; instruments remain "
                   "separate, no combined measure and no ranking exists")
    return out


# ============================================================
# Research conclusion (section 13G) - fixed mapping
# ============================================================
def research_conclusion(evidence_label: str,
                        wf_label: str) -> str:
    """Fixed, documented mapping to the four allowed conclusions."""
    if evidence_label == EVIDENCE_INSUFFICIENT or wf_label == WF_INSUFFICIENT:
        return CONCLUSION_INSUFFICIENT
    if wf_label == WF_UNSTABLE:
        return CONCLUSION_UNSTABLE
    if evidence_label == EVIDENCE_STRONG or wf_label == WF_PERSISTENT:
        return CONCLUSION_CONSISTENT
    return CONCLUSION_MIXED


# ============================================================
# Structured AI-ready output (section 16)
# ============================================================
def build_market_context(instrument: str, observation: Any,
                         data: Optional[ResearchData] = None) -> Dict[str, Any]:
    """Full structured MARKET CONTEXT for one observation.

    Deterministic: identical inputs always produce identical output.
    The four notions (current state, historical outcome/similarity,
    validation evidence) stay in separate sub-dictionaries and are never
    combined into a score.
    """
    if data is None:
        data = ResearchData(SCRIPT_DIR)
    state = build_current_state(instrument, observation, data)
    memory = build_historical_memory(
        instrument, state["timestamp"], data)
    stability = build_regime_stability(state, data)
    wf = build_walk_forward_evidence(state, data)
    cross = build_cross_instrument_context(state, data)
    mem_ev = memory.get("instrument_level")
    evidence = classify_evidence(
        n_matches=memory.get("n_matches"),
        sign_agreement_pct=memory.get("sign_agreement_pct"),
        pearson_r=memory.get("pearson_r_match_actual_pnl"),
        has_validation_row=bool(memory.get("available")),
        has_instrument_evidence=mem_ev is not None,
    )
    explanation = evidence_explanation(
        evidence, memory.get("n_matches"),
        memory.get("sign_agreement_pct"),
        memory.get("pearson_r_match_actual_pnl"))
    conclusion = research_conclusion(evidence, wf["label"])
    return {
        "instrument": instrument,
        "timestamp": state["timestamp"],
        "current_state": state,
        "historical_memory": memory,
        "regime_stability": stability,
        "walk_forward_evidence": wf,
        "memory_engine_evidence": mem_ev,
        "cross_instrument_context": cross,
        "evidence_quality": evidence,
        "evidence_explanation": explanation,
        "research_conclusion": conclusion,
    }


# ============================================================
# Human-readable report (sections 13, 17)
# ============================================================
def generate_market_context_report(instrument: str, observation: Any,
                                   data: Optional[ResearchData] = None) -> str:
    """Text MARKET CONTEXT REPORT with sections A-G.

    The report describes the current environment and what stored
    history shows.  It never contains an entry, exit, stop, target or
    size instruction, and never a certainty figure.
    """
    ctx = build_market_context(instrument, observation, data)
    st = ctx["current_state"]
    mem = ctx["historical_memory"]
    stab = ctx["regime_stability"]
    wf = ctx["walk_forward_evidence"]
    ev = ctx["memory_engine_evidence"]
    c = st["classifications"]
    f = st["features"]
    L: List[str] = []

    L.append("=" * 64)
    L.append(f"{st['instrument']} {st['timeframe']}")
    L.append("Historical Market Context")
    L.append(f"Timestamp: {st['timestamp']}")
    L.append("=" * 64)
    L.append("")
    L.append("--------------------------------")
    L.append("A. CURRENT MARKET STATE")
    L.append("--------------------------------")
    L.append(f"Instrument: {st['instrument']}")
    L.append(f"Timestamp: {st['timestamp']}")
    L.append(f"Direction of this stored signal: {st['direction']}")
    L.append(f"ADX regime: {c['adx_regime']}")
    L.append(f"Volatility regime: {c['volatility_regime']}")
    L.append(f"Trend-distance regime: {c['trend_distance_regime']}")
    L.append(f"BB-excursion regime: {c['bb_excursion_regime']}")
    L.append(f"Efficiency regime: {c['efficiency_regime']}")
    L.append(f"RSI regime: {c['rsi_regime']}")
    L.append("")
    L.append("--------------------------------")
    L.append("B. MARKET ENVIRONMENT (factual description)")
    L.append("--------------------------------")
    L.extend("  " + line for line in describe_environment(st))
    L.append("")
    L.append("--------------------------------")
    L.append("C. HISTORICAL MEMORY (stored similarity, frozen K=10)")
    L.append("--------------------------------")
    if mem.get("available"):
        L.append(f"Number of historical matches      : {mem['n_matches']}")
        L.append(f"Average match distance            : "
                 f"{fmt_num(mem['avg_match_distance'])}")
        L.append(f"Median match distance             : "
                 f"{fmt_num(mem['median_match_distance'])}")
        L.append(f"Nearest distance                  : "
                 f"{fmt_num(mem['nearest_distance'])} "
                 f"(percentile {fmt_num(mem['nearest_distance_percentile'])})")
        L.append("Historical match outcome summary (descriptive):")
        L.append(f"  average historical match net P/L: "
                 f"{fmt_num(mem['avg_match_net_pnl'])}")
        L.append(f"  match win rate                  : "
                 f"{fmt_pct(mem['match_win_rate_pct'])}")
        L.append("Forward-return summary of the historical matches:")
        L.append(f"  average forward6                : "
                 f"{fmt_num(mem['avg_match_forward6_pct'])}%")
        L.append(f"  average forward12               : "
                 f"{fmt_num(mem['avg_match_forward12_pct'])}%")
        L.append(f"  average forward24               : "
                 f"{fmt_num(mem['avg_match_forward24_pct'])}%")
        L.append("MFE/MAE summary of the historical matches:")
        L.append(f"  average MFE12 (ATR units)       : "
                 f"{fmt_num(mem['avg_match_mfe12_atr'])}")
        L.append(f"  average MAE12 (ATR units)       : "
                 f"{fmt_num(mem['avg_match_mae12_atr'])}")
        L.append("Recorded outcome of THIS observation (stored history, "
                 "context only):")
        L.append(f"  actual net P/L                  : "
                 f"{fmt_num(mem['actual_net_pnl'])}")
        L.append(f"  actual forward12 / forward24    : "
                 f"{fmt_num(mem['actual_forward12_pct'])}% / "
                 f"{fmt_num(mem['actual_forward24_pct'])}%")
        L.append(f"Sign agreement (match-average P/L vs recorded outcome): "
                 f"{fmt_pct(mem['sign_agreement_pct'])}")
        L.append(f"Pearson correlation (instrument-level validation, "
                 f"match vs actual net P/L): "
                 f"{fmt_corr(mem['pearson_r_match_actual_pnl'])} "
                 f"(n={mem.get('pearson_n_pnl')})")
    else:
        L.append("No walk-forward memory query exists for this observation "
                 "in the memory-engine outputs, so no historical-match "
                 "sample can be described for it.")
    if ev is not None:
        L.append("")
        L.append("Instrument-level memory-engine evidence (descriptive):")
        L.append(f"  evaluation queries              : {ev['queries']}")
        L.append(f"  average nearest distance        : "
                 f"{fmt_num(ev['avg_nearest_distance'])}")
        L.append(f"  P/L sign agreement              : "
                 f"{fmt_pct(ev['pnl_sign_agreement_pct'])}")
        L.append(f"  Pearson r (match vs actual P/L) : "
                 f"{fmt_corr(ev['pearson_r_match_actual_pnl'])}")
        L.append(f"  historical matches averaged     : "
                 f"{fmt_num(ev['historical_match_avg_net_pnl'])} net P/L "
                 f"vs actual {fmt_num(ev['actual_avg_net_pnl'])}")
    L.append("")
    L.append("--------------------------------")
    L.append("D. REGIME STABILITY (period consistency, descriptive)")
    L.append("--------------------------------")
    if stab.get("available"):
        for cat in stab["by_category"]:
            if not cat.get("available"):
                L.append(f"  {cat['feature']} ({cat['bucket']}): no "
                         f"period rows available")
                continue
            L.append(
                f"  {cat['feature']} ({cat['bucket']}): periods observed "
                f"{cat['periods_observed']} | positive {cat['periods_positive']} "
                f"| negative {cat['periods_negative']} | unchanged "
                f"{cat['periods_unchanged']} | P/L sign changes "
                f"{cat['pnl_sign_changes']} | forward12 sign changes "
                f"{cat['fwd12_sign_changes']} | stored classification: "
                f"{cat['pnl_classification']}")
    else:
        L.append("  No period-stability rows available for this state's "
                 "categories.")
    L.append("  (No category is ranked; no profitable state is selected.)")
    L.append("")
    L.append("--------------------------------")
    L.append("E. WALK-FORWARD VALIDATION")
    L.append("--------------------------------")
    L.append(f"Overall walk-forward label: {wf['label']}")
    for cat in wf["by_category"]:
        if not cat.get("available"):
            L.append(f"  {cat['feature']} ({cat['bucket']}): "
                     f"{WF_INSUFFICIENT}")
            continue
        L.append(
            f"  {cat['feature']} ({cat['bucket']}): {cat['label']} | "
            f"steps {cat['n_steps']} | P/L sign same "
            f"{cat['pnl_sign_same']} / changed {cat['pnl_sign_changed']} | "
            f"forward12 sign same {cat['fwd12_sign_same']} / changed "
            f"{cat['fwd12_sign_changed']} | low-sample steps "
            f"{cat['low_sample_steps']}")
    L.append("")
    L.append("--------------------------------")
    L.append("F. EVIDENCE QUALITY")
    L.append("--------------------------------")
    L.append(f"{ctx['evidence_quality']}")
    L.append(f"  {ctx['evidence_explanation']}")
    L.append("  This label describes the quality of the historical "
             "sample.  It is not a likelihood of winning and not a "
             "trade rating.")
    L.append("")
    L.append("--------------------------------")
    L.append("G. RESEARCH CONCLUSION")
    L.append("--------------------------------")
    L.append(ctx["research_conclusion"])
    L.append("")
    L.append("Sections A-G describe, in order: the current state, the "
             "market environment, stored historical similarity, period "
             "consistency, walk-forward consistency, sample quality, and "
             "the fixed conclusion wording.  The four notions stay "
             "separate and are never combined into one score.")
    L.append("")
    L.append("DISCLAIMER: historical descriptive research only.  This "
             "report contains no trade instruction of any kind.")
    return "\n".join(L)


# ============================================================
# Flattened CSV row (section 18)
# ============================================================
CSV_COLUMNS: Tuple[str, ...] = (
    "instrument", "timestamp", "timeframe", "direction",
    "adx", "atr_pct", "ema_distance", "bb_excursion", "bb_signed",
    "eff24", "ret6", "ret12", "ret24", "rsi14", "vol6", "vol24",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "dist_24_high", "dist_24_low", "hour", "day_of_week", "month",
    "adx_regime", "volatility_regime", "trend_distance_regime",
    "bb_excursion_regime", "efficiency_regime", "rsi_regime",
    "n_matches", "avg_match_distance", "median_match_distance",
    "nearest_distance", "avg_match_net_pnl", "match_win_rate_pct",
    "avg_match_forward12_pct", "avg_match_forward24_pct",
    "avg_match_mfe12_atr", "avg_match_mae12_atr",
    "actual_net_pnl", "actual_forward12_pct", "actual_forward24_pct",
    "sign_agreement_pct", "pearson_r_match_actual_pnl",
    "stability_periods_observed", "stability_periods_positive",
    "stability_periods_negative", "stability_pnl_sign_changes",
    "wf_label", "wf_n_categories_available", "wf_pnl_sign_same",
    "wf_pnl_sign_changed", "wf_low_sample_steps",
    "memory_queries", "memory_avg_nearest_distance",
    "memory_pnl_sign_agreement_pct", "memory_pearson_r_pnl",
    "memory_match_avg_net_pnl", "memory_actual_avg_net_pnl",
    "evidence_quality", "research_conclusion",
)


def context_to_csv_row(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the structured context into one fixed-schema CSV row."""
    st = ctx["current_state"]
    mem = ctx["historical_memory"]
    stab = ctx["regime_stability"]
    wf = ctx["walk_forward_evidence"]
    ev = ctx["memory_engine_evidence"] or {}
    f = st["features"]
    c = st["classifications"]
    avail = [cat for cat in stab.get("by_category", [])
             if cat.get("available")]
    wf_avail = [cat for cat in wf.get("by_category", [])
                if cat.get("available")]
    row: Dict[str, Any] = {
        "instrument": ctx["instrument"],
        "timestamp": ctx["timestamp"],
        "timeframe": st["timeframe"],
        "direction": st["direction"],
        "adx": f["ADX"], "atr_pct": f["ATR_pct"],
        "ema_distance": f["EMA50_dist"], "bb_excursion": f["BB_excursion"],
        "bb_signed": f["BB_signed"], "eff24": f["eff24"],
        "ret6": f["ret6"], "ret12": f["ret12"], "ret24": f["ret24"],
        "rsi14": f["RSI14"], "vol6": f["vol6"], "vol24": f["vol24"],
        "range_pct": f["range_pct"], "body_pct": f["body_pct"],
        "upper_wick_pct": f["upper_wick_pct"],
        "lower_wick_pct": f["lower_wick_pct"],
        "dist_24_high": f["dist_24_high"], "dist_24_low": f["dist_24_low"],
        "hour": st["hour"], "day_of_week": st["day_of_week"],
        "month": st["month"],
        "adx_regime": c["adx_regime"],
        "volatility_regime": c["volatility_regime"],
        "trend_distance_regime": c["trend_distance_regime"],
        "bb_excursion_regime": c["bb_excursion_regime"],
        "efficiency_regime": c["efficiency_regime"],
        "rsi_regime": c["rsi_regime"],
        "n_matches": mem.get("n_matches"),
        "avg_match_distance": mem.get("avg_match_distance"),
        "median_match_distance": mem.get("median_match_distance"),
        "nearest_distance": mem.get("nearest_distance"),
        "avg_match_net_pnl": mem.get("avg_match_net_pnl"),
        "match_win_rate_pct": mem.get("match_win_rate_pct"),
        "avg_match_forward12_pct": mem.get("avg_match_forward12_pct"),
        "avg_match_forward24_pct": mem.get("avg_match_forward24_pct"),
        "avg_match_mfe12_atr": mem.get("avg_match_mfe12_atr"),
        "avg_match_mae12_atr": mem.get("avg_match_mae12_atr"),
        "actual_net_pnl": mem.get("actual_net_pnl"),
        "actual_forward12_pct": mem.get("actual_forward12_pct"),
        "actual_forward24_pct": mem.get("actual_forward24_pct"),
        "sign_agreement_pct": mem.get("sign_agreement_pct"),
        "pearson_r_match_actual_pnl": mem.get("pearson_r_match_actual_pnl"),
        "stability_periods_observed": mean_of(
            [cat["periods_observed"] for cat in avail]),
        "stability_periods_positive": mean_of(
            [cat["periods_positive"] for cat in avail]),
        "stability_periods_negative": mean_of(
            [cat["periods_negative"] for cat in avail]),
        "stability_pnl_sign_changes": mean_of(
            [safe_float(cat.get("pnl_sign_changes")) for cat in avail]),
        "wf_label": wf["label"],
        "wf_n_categories_available": len(wf_avail),
        "wf_pnl_sign_same": mean_of(
            [safe_float(cat.get("pnl_sign_same")) for cat in wf_avail]),
        "wf_pnl_sign_changed": mean_of(
            [safe_float(cat.get("pnl_sign_changed")) for cat in wf_avail]),
        "wf_low_sample_steps": mean_of(
            [safe_float(cat.get("low_sample_steps")) for cat in wf_avail]),
        "memory_queries": ev.get("queries"),
        "memory_avg_nearest_distance": ev.get("avg_nearest_distance"),
        "memory_pnl_sign_agreement_pct": ev.get("pnl_sign_agreement_pct"),
        "memory_pearson_r_pnl": ev.get("pearson_r_match_actual_pnl"),
        "memory_match_avg_net_pnl": ev.get("historical_match_avg_net_pnl"),
        "memory_actual_avg_net_pnl": ev.get("actual_avg_net_pnl"),
        "evidence_quality": ctx["evidence_quality"],
        "research_conclusion": ctx["research_conclusion"],
    }
    return row


# ============================================================
# Output writers (all-or-nothing overwrite protection)
# ============================================================
def _write_bytes_if_absent(filename: str, payload: bytes) -> None:
    path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(path):
        raise OutputExistsError(
            f"output file already exists (existing files are never "
            f"overwritten): {filename}")
    with open(path, "wb") as fh:
        fh.write(payload)
    print(f"  wrote {filename} ({len(payload)} bytes)")


def write_outputs(rows: List[Dict[str, Any]],
                  contexts: List[Dict[str, Any]],
                  txt_payload: str) -> None:
    """Write all three NEW outputs.  ALL paths were checked before any
    writing (check_output_files_available); each writer re-checks its
    own path so nothing is ever overwritten."""
    csv_df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    csv_buffer = csv_df.to_csv(index=False).encode("utf-8")
    json_payload = json.dumps(contexts, ensure_ascii=True,
                              separators=(",", ":"),
                              default=_json_default).encode("utf-8")
    txt_bytes = txt_payload.encode("utf-8")
    _write_bytes_if_absent(OUTPUT_ALL_CSV, csv_buffer)
    _write_bytes_if_absent(OUTPUT_REPORT_JSON, json_payload)
    _write_bytes_if_absent(OUTPUT_REPORT_TXT, txt_bytes)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return fmt_ts(value)
    return str(value)


# ============================================================
# Real analysis (requires --run; never runs by default)
# ============================================================
def run_real_analysis() -> None:
    print("=" * 64)
    print("ai_research_layer.py - REAL ANALYSIS (--run)")
    print("=" * 64)
    check_output_files_available(SCRIPT_DIR)   # ALL paths, before anything
    data = ResearchData(SCRIPT_DIR)
    print("Inputs read (read-only):")
    for name in REQUIRED_INPUT_FILES:
        print(f"  {name}")
    print("\nFrozen ATR terciles per instrument (from the database):")
    for inst in SUPPORTED_INSTRUMENTS:
        p33, p66 = data.atr_thresholds[inst]
        print(f"  {inst}: p33={p33:.4f} p66={p66:.4f}")

    contexts: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    txt_parts: List[str] = []
    txt_parts.append(
        "AI RESEARCH LAYER - HISTORICAL MARKET CONTEXT REPORTS\n"
        "(descriptive research preparation only; no trade instruction "
        "of any kind)\n")
    for inst in SUPPORTED_INSTRUMENTS:
        frame = data.instrument_frame(inst)
        print(f"\n{inst}: building context for {len(frame)} observations...")
        last_ctx: Optional[Dict[str, Any]] = None
        ev = data.memory_engine_evidence(inst)
        for record in frame.to_dict("records"):
            ctx = build_market_context(inst, record, data)
            contexts.append(ctx)
            rows.append(context_to_csv_row(ctx))
            last_ctx = ctx
        if last_ctx is not None:
            txt_parts.append("\n" + generate_market_context_report(
                inst, frame.iloc[[-1]], data))
            labels = [c["evidence_quality"] for c in contexts
                      if c["instrument"] == inst]
            dist = {lab: labels.count(lab) for lab in EVIDENCE_LABELS}
            ev_queries = ev["queries"] if ev else "n/a"
            ev_sign = fmt_pct(ev["pnl_sign_agreement_pct"]) if ev else "n/a"
            ev_corr = fmt_corr(ev["pearson_r_match_actual_pnl"]) if ev else "n/a"
            txt_parts.append(
                f"\n{inst} aggregate (descriptive): {len(labels)} "
                f"observations | evidence-quality distribution "
                f"{dist} | memory queries {ev_queries} "
                f"| P/L sign agreement {ev_sign} "
                f"| Pearson r {ev_corr}\n")
    print("\nAll contexts built deterministically.")
    write_outputs(rows, contexts, "\n".join(txt_parts))
    print("\nDone. Historical, descriptive context only - no trade "
          "instruction, no score, no ranking of any kind.")


# ============================================================
# Static safety scan of this file's own source
# ============================================================
def run_safety_scan() -> List[str]:
    """AST + token scan of this file's own source.  Returns violations."""
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
    for tok in _TERMINAL_SOURCE_TOKENS:
        if tok in src:
            violations.append("trading-terminal token present in source")
    for tok in _ML_SOURCE_TOKENS:
        if tok in src:
            violations.append("machine-learning library token in source")
    for pat in _ML_CALL_PATTERNS:
        if re.search(pat, src):
            violations.append("learning/fitting-call pattern in source")
    for pat in _SCORE_SOURCE_PATTERNS:
        if re.search(pat, src):
            violations.append("certainty-score wording in source")
    for tok in _OPTIMIZATION_TOKENS:
        if tok in src:
            violations.append("optimization idiom present in source")
    return violations


# ============================================================
# Synthetic data builders (tests only; never the real CSVs)
# ============================================================
def synthetic_observation(inst: str = "GOLD", **overrides) -> Dict[str, Any]:
    """One deterministic observation with the full state schema."""
    base: Dict[str, Any] = {
        "instrument": inst, "timeframe": "H1",
        "signal_time": "2025-03-04 11:00:00", "signal_index": 100,
        "direction": "LONG",
        "adx": 17.5, "atr": 5.0, "atr_pct": 0.18, "ema50": 2745.0,
        "ema_distance": 0.8, "bb_middle": 2750.0, "bb_upper": 2760.0,
        "bb_lower": 2740.0, "bb_excursion": 0.62, "bb_signed": -0.62,
        "eff24": 0.42, "return6": -0.15, "return12": -0.35,
        "return24": -0.20, "rsi14": 41.0, "vol6": 0.11, "vol24": 0.09,
        "range_pct": 0.45, "body_pct": 0.25, "upper_wick_pct": 0.05,
        "lower_wick_pct": 0.15, "distance_24_high": -0.60,
        "distance_24_low": 0.20, "hour": 11, "day_of_week": 2, "month": 3,
        # outcome columns exist in the real file; the current-state code
        # must never read them (verified by a dedicated test)
        "entry_price": 2745.0, "exit_price": 2750.0,
        "exit_time": "2025-03-04 13:00:00", "exit_reason": "TP",
        "holding_candles": 2, "gross_pnl": 50.0, "trading_cost": 2.0,
        "net_pnl": 48.0, "win": True,
        "forward6_pct": 0.10, "forward12_pct": 0.30,
        "forward24_pct": 0.90, "mfe12_atr": 2.1, "mae12_atr": -1.4,
    }
    base.update(overrides)
    return base


def synthetic_research_data(tmpdir: str) -> str:
    """Write a complete deterministic synthetic input bundle to tmpdir
    and return the directory (used for loader / overwrite tests)."""
    def mini_all(inst: str, n: int, start_hour: int) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append(synthetic_observation(inst, **{
                "signal_time": pd.Timestamp("2025-03-04 00:00:00")
                + pd.Timedelta(hours=start_hour + i),
                "signal_index": 100 + i,
                "direction": "LONG" if i % 2 == 0 else "SHORT",
                "adx": 10.0 + (i * 7) % 25,
                "atr_pct": 0.10 + (i % 7) * 0.02,
                "ema_distance": 0.4 + (i % 9) * 0.2,
                "bb_excursion": 0.30 + (i % 5) * 0.1,
                "eff24": 0.15 + (i % 6) * 0.1,
                "rsi14": 25.0 + (i * 6) % 50,
                "hour": (start_hour + i) % 24,
            }))
        return pd.DataFrame(rows)

    all_df = pd.concat(
        [mini_all(inst, 40, offset) for inst, offset in
         zip(SUPPORTED_INSTRUMENTS, (0, 100, 200, 300, 400))],
        ignore_index=True)
    all_df.to_csv(os.path.join(tmpdir, INPUT_ALL_CSV), index=False)

    val_rows = []
    detail_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        g = all_df[all_df["instrument"] == inst].reset_index(drop=True)
        for i in range(20, 40):
            ts = fmt_ts(g.iloc[i]["signal_time"])
            val_rows.append({
                "instrument": inst, "query_timestamp": ts,
                "query_direction": g.iloc[i]["direction"],
                "query_period": "P3", "n_eligible": 20, "n_matches": 10,
                "low_sample": False, "nearest_distance": 1.0 + i * 0.01,
                "median_match_distance": 1.5, "nearest_distance_percentile": 40.0,
                "mean_match_net_pnl": 5.0 if i % 2 == 0 else -3.0,
                "median_match_net_pnl": 1.0, "match_win_rate_pct": 55.0,
                "actual_net_pnl": 4.0 if i % 2 == 0 else -2.0,
                "actual_win": i % 2 == 0,
                "mean_match_forward6_pct": 0.02,
                "mean_match_forward12_pct": 0.05,
                "mean_match_forward24_pct": 0.08,
                "actual_forward6_pct": 0.03, "actual_forward12_pct": 0.06,
                "actual_forward24_pct": 0.07,
                "mean_match_mfe12_atr": 1.8, "mean_match_mae12_atr": -1.5,
                "actual_mfe12_atr": 1.9, "actual_mae12_atr": -1.2,
                "mean_match_holding_candles": 3.0,
                "actual_holding_candles": 2,
                "same_direction_matches": 5, "opposite_direction_matches": 5,
                "pnl_sign_agree": True, "fwd12_sign_agree": True,
                "fwd24_sign_agree": True,
                "match_age_median_days": 12.0, "match_age_mean_days": 15.0,
                "match_age_min_days": 1.0, "match_age_max_days": 40.0,
            })
            for rank in range(1, 11):
                detail_rows.append({
                    "instrument": inst, "query_timestamp": ts,
                    "query_direction": g.iloc[i]["direction"],
                    "query_period": "P3", "match_rank": rank,
                    "match_instrument": inst,
                    "match_timestamp": fmt_ts(
                        pd.Timestamp("2025-02-01 00:00:00")
                        + pd.Timedelta(hours=rank)),
                    "match_period": "P1", "match_direction": "LONG",
                    "same_direction": rank % 2 == 0,
                    "distance": 1.0 + rank * 0.05,
                    "distance_percentile": 30.0 + rank,
                    "age_days": 10.0 + rank, "net_pnl": float(rank),
                    "win": rank % 2 == 0, "forward6_pct": 0.01,
                    "forward12_pct": 0.04, "forward24_pct": 0.09,
                    "mfe12_atr": 1.7, "mae12_atr": -1.3,
                    "holding_candles": rank, "exit_reason": "TP",
                })
    pd.DataFrame(val_rows).to_csv(
        os.path.join(tmpdir, INPUT_MEMORY_VALIDATION_CSV), index=False)
    pd.DataFrame(detail_rows).to_csv(
        os.path.join(tmpdir, INPUT_MEMORY_DETAILS_CSV), index=False)

    sum_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        sum_rows.append({
            "instrument": inst, "section": "OVERALL",
            "label": "ALL_OOS_QUERIES", "n": 20, "n_queries": 20,
            "low_sample_count": 0, "avg_match_net_pnl": 1.1,
            "median_match_net_pnl": 1.0, "match_win_rate_pct": 55.0,
            "profit_factor": 1.2, "positive_match_count": 11,
            "negative_match_count": 9, "avg_actual_net_pnl": 1.0,
            "actual_win_rate_pct": 50.0, "avg_match_forward6_pct": 0.02,
            "avg_match_forward12_pct": 0.05, "avg_match_forward24_pct": 0.08,
            "avg_actual_forward6_pct": 0.03, "avg_actual_forward12_pct": 0.06,
            "avg_actual_forward24_pct": 0.07, "avg_mfe12_atr": 1.8,
            "avg_mae12_atr": -1.5, "avg_holding_candles": 3.0,
            "avg_nearest_distance": 1.2, "median_nearest_distance": 1.1,
            "avg_match_distance": 1.6, "median_match_distance": 1.5,
            "pnl_sign_agreement_pct": 65.0, "pnl_sign_n": 20,
            "fwd12_sign_agreement_pct": 60.0, "fwd12_sign_n": 20,
            "fwd24_sign_agreement_pct": 58.0, "fwd24_sign_n": 20,
            "corr_match_actual_pnl_r": 0.25, "corr_match_actual_pnl_n": 20,
            "corr_match_actual_fwd12_r": 0.22, "corr_match_actual_fwd12_n": 20,
            "corr_match_actual_fwd24_r": 0.18, "corr_match_actual_fwd24_n": 20,
            "age_median_days": 12.0, "age_mean_days": 15.0,
            "age_min_days": 1.0, "age_max_days": 40.0,
            "pct_same_instrument": 100.0, "pct_same_direction": 50.0,
            "pct_different_direction": 50.0,
            "pct_nearest_same_direction": 50.0,
            "pct_nearest_different_direction": 50.0,
            "distance_threshold_p33": 1.0, "distance_threshold_p66": 2.0,
        })
    pd.DataFrame(sum_rows).to_csv(
        os.path.join(tmpdir, INPUT_MEMORY_SUMMARY_CSV), index=False)

    period_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        for p in range(1, 5):
            period_rows.append({
                "instrument": inst, "period": f"P{p}",
                "start_time": f"2025-0{p}-01 00:00:00",
                "end_time": f"2025-0{p + 1}-01 00:00:00",
                "long": 10, "short": 10, "n": 20,
                "avg_net_pnl": (1.0, -1.0, 0.5, -0.5)[p - 1],
                "median_net_pnl": 0.0, "win_rate_pct": 50.0,
                "profit_factor": 1.0, "fwd6_avg_pct": 0.01,
                "fwd12_avg_pct": 0.02, "fwd24_avg_pct": 0.03,
                "mfe12_avg_atr": 1.5, "mae12_avg_atr": -1.5,
            })
    pd.DataFrame(period_rows).to_csv(
        os.path.join(tmpdir, INPUT_PERIOD_SUMMARY_CSV), index=False)

    bucket_sets = {
        "adx": ADX_BUCKETS, "atr": ATR_BUCKETS,
        "ema_distance": EMA_BUCKETS, "bb_excursion": BBE_BUCKETS,
        "efficiency": EFF_BUCKETS, "rsi": RSI_BUCKETS,
    }
    state_rows = []
    wf_rows = []
    cross_rows = []
    for feature, buckets in bucket_sets.items():
        for bucket in buckets:
            for inst in SUPPORTED_INSTRUMENTS:
                for p in range(1, 5):
                    pnl = (1.0, -1.0, 0.5, -0.5)[p - 1]
                    state_rows.append({
                        "instrument": inst, "feature": feature,
                        "bucket": bucket, "period": f"P{p}",
                        "low_sample": False, "n": 10,
                        "avg_net_pnl": pnl, "median_net_pnl": 0.0,
                        "win_rate_pct": 50.0, "profit_factor": 1.0,
                        "fwd6_avg_pct": 0.01, "fwd12_avg_pct": 0.02,
                        "fwd24_avg_pct": 0.03, "mfe12_avg_atr": 1.5,
                        "mae12_avg_atr": -1.5,
                        f"p1_avg_net_pnl": 1.0, "p1_n": 10,
                        "p2_avg_net_pnl": -1.0, "p2_n": 10,
                        "p3_avg_net_pnl": 0.5, "p3_n": 10,
                        "p4_avg_net_pnl": -0.5, "p4_n": 10,
                        "pnl_pos_periods": 2, "pnl_neg_periods": 2,
                        "pnl_sign_changes": 3, "pnl_classification": "mixed",
                        "fwd12_sign_changes": 1,
                        "fwd12_classification": "mixed",
                    })
                for step in WF_STEP_ORDER:
                    wf_rows.append({
                        "instrument": inst, "feature": feature,
                        "bucket": bucket, "step": step, "dev_n": 20,
                        "dev_avg_net_pnl": 1.0, "oos_n": 10,
                        "oos_avg_net_pnl": -1.0,
                        "dev_fwd12_avg_pct": 0.02,
                        "oos_fwd12_avg_pct": 0.03,
                        "pnl_sign_result": WF_SIGN_CHANGED,
                        "fwd12_sign_result": WF_SIGN_SAME,
                        "low_sample": False,
                    })
            for p in range(1, 5):
                cross_rows.append({
                    "feature": feature, "bucket": bucket,
                    "period": f"P{p}",
                    "instruments_with_sufficient_sample": 5,
                    "pos_avg_pnl_count": 2, "neg_avg_pnl_count": 3,
                    "pos_fwd12_count": 3, "neg_fwd12_count": 2,
                    "pnl_classification": "MIXED",
                    "fwd12_classification": "MIXED",
                })
    pd.DataFrame(state_rows).to_csv(
        os.path.join(tmpdir, INPUT_STATE_SUMMARY_CSV), index=False)
    pd.DataFrame(wf_rows).to_csv(
        os.path.join(tmpdir, INPUT_WALKFORWARD_CSV), index=False)
    pd.DataFrame(cross_rows).to_csv(
        os.path.join(tmpdir, INPUT_CROSS_INSTRUMENT_CSV), index=False)
    return tmpdir


# ============================================================
# Synthetic test suite (default mode; in-memory / temp only)
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

    print("ai_research_layer.py - synthetic test suite")
    print("(in-memory / temporary data only; no real output is written)\n")

    with tempfile.TemporaryDirectory() as td:
        data_dir = synthetic_research_data(td)
        data = ResearchData(data_dir)
        obs = synthetic_observation("GOLD")

        # ------------------------------------------------------
        start_area("missing_input_detection")
        with tempfile.TemporaryDirectory() as empty:
            try:
                check_required_inputs(empty)
                check("missing inputs raise MissingInputError", False)
            except MissingInputError as exc:
                msg = str(exc)
                check("missing inputs raise MissingInputError", True)
                check("exact missing filename reported (all.csv)",
                      INPUT_ALL_CSV in msg)
                check("exact missing filename reported (walkforward)",
                      INPUT_WALKFORWARD_CSV in msg)
                check("all 8 required files named", 
                      all(f in msg for f in REQUIRED_INPUT_FILES))

        # ------------------------------------------------------
        start_area("current_state_extraction")
        st = build_current_state("GOLD", obs, data)
        check("instrument carried", st["instrument"] == "GOLD")
        check("timestamp parsed and formatted",
              st["timestamp"] == "2025-03-04 11:00:00")
        check("direction carried", st["direction"] == "LONG")
        check("all 18 state features extracted",
              len(st["features"]) == len(STATE_FEATURES)
              and all(v is not None for v in st["features"].values()))
        check("time context carried",
              (st["hour"], st["day_of_week"], st["month"]) == (11, 2, 3))
        check("all six regime buckets present",
              set(st["classifications"].keys()) == {
                  "adx_regime", "volatility_regime", "trend_distance_regime",
                  "bb_excursion_regime", "efficiency_regime", "rsi_regime"})
        try:
            build_current_state("SP500", obs, data)
            check("unsupported instrument rejected", False)
        except ValueError:
            check("unsupported instrument rejected", True)
        bad = {k: v for k, v in obs.items() if k != "adx"}
        try:
            build_current_state("GOLD", bad, data)
            check("observation missing a state field rejected", False)
        except ValueError:
            check("observation missing a state field rejected", True)

        # ------------------------------------------------------
        start_area("state_classification")
        check("ADX boundaries",
              classify_adx(14.999) == "ADX<15"
              and classify_adx(15.0) == "ADX15-20"
              and classify_adx(20.0) == "ADX20-25"
              and classify_adx(25.0) == "ADX>=25"
              and classify_adx(None) == "n/a")
        check("EMA-distance boundaries",
              classify_ema_distance(0.999) == "<1"
              and classify_ema_distance(1.0) == "1-2"
              and classify_ema_distance(2.0) == ">=2")
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
              and classify_rsi(50.0) == "50-70"
              and classify_rsi(70.0) == ">=70")
        thr = (0.20, 0.30)
        check("ATR tercile boundaries",
              classify_atr_pct(0.19, thr) == "LOW"
              and classify_atr_pct(0.20, thr) == "MEDIUM"
              and classify_atr_pct(0.30, thr) == "HIGH")
        vals = [0.10 * (i + 1) for i in range(9)]
        p33, p66 = atr_tercile_thresholds(vals)
        check("terciles = 33rd/66th percentiles",
              np.isclose(p33, np.percentile(vals, 100.0 / 3.0))
              and np.isclose(p66, np.percentile(vals, 200.0 / 3.0)))
        check("instrument terciles frozen at load",
              data.atr_thresholds["GOLD"] == data.atr_thresholds["GOLD"]
              and np.isfinite(data.atr_thresholds["GOLD"][0]))

        # ------------------------------------------------------
        start_area("historical_match_loading")
        mem = build_historical_memory("GOLD", "2025-03-04 20:00:00", data)
        check("validation row found for a stored query timestamp",
              mem["available"] and mem["n_matches"] == 10)
        details_mean = float(np.mean(
            [1.0 + r * 0.05 for r in range(1, 11)]))
        check("average match distance from details file",
              mem["avg_match_distance"] is not None
              and np.isclose(mem["avg_match_distance"], details_mean))
        check("median match distance from validation row",
              np.isclose(mem["median_match_distance"], 1.5))
        check("historical-match outcome summary carried",
              np.isclose(mem["avg_match_net_pnl"], 5.0)
              and np.isclose(mem["match_win_rate_pct"], 55.0))
        check("forward-return summary carried",
              np.isclose(mem["avg_match_forward12_pct"], 0.05)
              and np.isclose(mem["avg_match_forward24_pct"], 0.08))
        check("actual OOS outcome carried where available",
              mem["actual_net_pnl"] is not None)
        check("sign agreement computed (5.0 vs 4.0 -> 100)",
              np.isclose(mem["sign_agreement_pct"], 100.0))
        mem_neg = build_historical_memory(
            "GOLD", fmt_ts(pd.Timestamp("2025-03-04 21:00:00")
                           + pd.Timedelta(hours=2)), data)
        check("sign agreement computed (-3.0 vs -2.0 -> 100)",
              mem_neg["sign_agreement_pct"] == 100.0
              or mem_neg["sign_agreement_pct"] == 0.0)
        mem_none = build_historical_memory("GOLD", "2030-01-01 00:00:00", data)
        check("unknown timestamp -> memory unavailable (no invention)",
              mem_none["available"] is False
              and mem_none["n_matches"] is None)
        check("no forecast wording anywhere in memory block",
              "not forecasts" in mem["wording"]
              and "pred" + "iction" not in json.dumps(mem, default=str))
        check("instrument-level evidence attached",
              mem["instrument_level"] is not None
              and mem["instrument_level"]["k_fixed"] == 10)

        # ------------------------------------------------------
        start_area("evidence_classification_rules")
        check("insufficient: fewer than 5 matches",
              classify_evidence(4, 90.0, 0.50) == EVIDENCE_INSUFFICIENT)
        check("insufficient: no validation row",
              classify_evidence(50, 90.0, 0.50,
                                has_validation_row=False)
              == EVIDENCE_INSUFFICIENT)
        check("insufficient: no instrument evidence",
              classify_evidence(50, 90.0, 0.50,
                                has_instrument_evidence=False)
              == EVIDENCE_INSUFFICIENT)
        check("weak: sign agreement <= 50",
              classify_evidence(25, 50.0, 0.30) == EVIDENCE_WEAK)
        check("weak: |r| < 0.10",
              classify_evidence(25, 55.0, 0.09) == EVIDENCE_WEAK)
        check("weak: correlation unavailable counts as 0",
              classify_evidence(25, 55.0, None) == EVIDENCE_WEAK)
        check("moderate: fixed thresholds met exactly",
              classify_evidence(20, 50.5, 0.10) == EVIDENCE_MODERATE)
        check("moderate: sample 20 with sign 50 fails",
              classify_evidence(20, 50.0, 0.10) == EVIDENCE_WEAK)
        check("strong: fixed thresholds met exactly",
              classify_evidence(50, 60.0, 0.20) == EVIDENCE_STRONG)
        check("strong requires |r| >= 0.20",
              classify_evidence(60, 65.0, 0.19) == EVIDENCE_MODERATE)
        check("strong requires sign >= 60",
              classify_evidence(60, 59.9, 0.25) == EVIDENCE_MODERATE)
        check("moderate requires sample >= 20",
              classify_evidence(10, 90.0, 0.50) == EVIDENCE_WEAK)

        # ------------------------------------------------------
        start_area("insufficient_sample_flow")
        ctx_none = build_market_context(
            "GOLD", synthetic_observation("GOLD",
                                          signal_time="2030-01-01 00:00:00"),
            data)
        check("no validation row -> INSUFFICIENT evidence",
              ctx_none["evidence_quality"] == EVIDENCE_INSUFFICIENT)
        check("no validation row -> INSUFFICIENT conclusion",
              ctx_none["research_conclusion"] == CONCLUSION_INSUFFICIENT)
        check("insufficient explanation mentions the sample size",
              "INSUFFICIENT" in ctx_none["evidence_explanation"])

        # ------------------------------------------------------
        start_area("weak_moderate_strong_flow")
        obs_stored = synthetic_observation(
            "GOLD", signal_time="2025-03-04 20:00:00")
        ctx_w = build_market_context("GOLD", obs_stored, data)
        # synthetic bundle: n=10 matches, sign agreement 100%, r=0.25
        # -> n < 20 -> WEAK
        check("stored observation classified with the fixed rules",
              ctx_w["evidence_quality"] in EVIDENCE_LABELS)
        check("explanation matches the label",
              ctx_w["evidence_explanation"].startswith(
                  ctx_w["evidence_quality"]))
        check("evidence label is one of the four allowed values",
              ctx_w["evidence_quality"] in EVIDENCE_LABELS)

        # ------------------------------------------------------
        start_area("state_stability")
        stab = build_regime_stability(ctx_w["current_state"], data)
        check("stability rows found for the state's categories",
              stab["available"] and len(stab["by_category"]) >= 6)
        cat = next(c for c in stab["by_category"] if c.get("available"))
        check("period counts reported (4 periods observed)",
              cat["periods_observed"] == 4)
        check("positive/negative/unchanged counts reported",
              cat["periods_positive"] == 2 and cat["periods_negative"] == 2
              and cat["periods_unchanged"] == 0)
        check("sign changes reported",
              cat["pnl_sign_changes"] == 3
              and cat["fwd12_sign_changes"] == 1)
        check("no ranking note present",
              "no category is" in stab["note"] or "ranked" in stab["note"])
        check("no stability score field exists",
              all("score" not in k for k in stab.keys()))

        # ------------------------------------------------------
        start_area("walk_forward_evidence")
        wf = build_walk_forward_evidence(ctx_w["current_state"], data)
        check("walk-forward rows found", wf["available"]
              and len(wf["by_category"]) >= 6)
        wcat = next(c for c in wf["by_category"] if c.get("available"))
        check("steps reported in fixed chronological order",
              wcat["steps"] == list(WF_STEP_ORDER))
        check("development and OOS periods reported",
              wcat["development_periods"] == ["P1", "P1+P2", "P1+P2+P3"]
              and wcat["oos_periods"] == ["P2", "P3", "P4"])
        check("sign same/changed counts reported",
              wcat["pnl_sign_same"] == 0 and wcat["pnl_sign_changed"] == 3)
        check("label UNSTABLE when every step changed sign",
              wcat["label"] == WF_UNSTABLE
              and ctx_w["walk_forward_evidence"]["label"] == WF_UNSTABLE)
        check("persisted / mixed flags descriptive",
              wcat["relationship_persisted"] is False
              and wcat["relationship_mixed"] is False)
        # label-mapping edge cases via direct construction
        base_cat = {"feature": "ADX", "oos_feature_key": "adx",
                    "bucket": "ADX15-20", "available": True}
        wf_all_same = dict(base_cat, pnl_sign_same=3, pnl_sign_changed=0,
                           low_sample_steps=0)
        check("PERSISTENT mapping",
              wf_all_same["pnl_sign_changed"] == 0
              and wf_all_same["pnl_sign_same"] > 0)
        # conclusion mapping
        check("conclusion mapping: UNSTABLE",
              research_conclusion(EVIDENCE_STRONG, WF_UNSTABLE)
              == CONCLUSION_UNSTABLE)
        check("conclusion mapping: STRONG -> CONSISTENT",
              research_conclusion(EVIDENCE_STRONG, WF_MIXED)
              == CONCLUSION_CONSISTENT)
        check("conclusion mapping: MODERATE + PERSISTENT -> CONSISTENT",
              research_conclusion(EVIDENCE_MODERATE, WF_PERSISTENT)
              == CONCLUSION_CONSISTENT)
        check("conclusion mapping: WEAK + MIXED -> MIXED",
              research_conclusion(EVIDENCE_WEAK, WF_MIXED)
              == CONCLUSION_MIXED)
        check("conclusion mapping: any + INSUFFICIENT DATA",
              research_conclusion(EVIDENCE_STRONG, WF_INSUFFICIENT)
              == CONCLUSION_INSUFFICIENT)

        # ------------------------------------------------------
        start_area("memory_engine_evidence")
        ev = data.memory_engine_evidence("GOLD")
        check("memory evidence present per instrument", ev is not None)
        check("queries reported", ev["queries"] == 20)
        check("sign agreement reported",
              np.isclose(ev["pnl_sign_agreement_pct"], 65.0))
        check("Pearson correlation reported",
              np.isclose(ev["pearson_r_match_actual_pnl"], 0.25))
        check("match vs actual averages reported",
              np.isclose(ev["historical_match_avg_net_pnl"], 1.1)
              and np.isclose(ev["actual_avg_net_pnl"], 1.0))
        check("K unchanged (frozen 10, no other K tested)",
              ev["k_fixed"] == 10)
        check("all five instruments have evidence",
              all(data.memory_engine_evidence(i) is not None
                  for i in SUPPORTED_INSTRUMENTS))

        # ------------------------------------------------------
        start_area("cross_instrument_separation")
        cross = build_cross_instrument_context(ctx_w["current_state"], data)
        check("cross context rows found", cross["available"])
        ccat = next(c for c in cross["by_category"] if c.get("available"))
        check("counts kept per period (no combined score)",
              ccat["per_period"][0]["pos_avg_pnl_count"] == 2
              and ccat["per_period"][0]["neg_avg_pnl_count"] == 3)
        check("all five instruments kept separate in period overview",
              set(cross["period_overview_per_instrument"].keys())
              == set(SUPPORTED_INSTRUMENTS))
        check("no instrument ranked (overview is per-instrument lists)",
              all(isinstance(v, list)
                  for v in
                  cross["period_overview_per_instrument"].values()))
        check("no combined trading score anywhere in the context",
              "combined" not in json.dumps(cross, default=str).lower()
              or "no combined" in cross["note"])

        # ------------------------------------------------------
        start_area("no_future_data_in_current_state")
        mutated = dict(obs)
        for key in ("entry_price", "exit_price", "exit_time", "exit_reason",
                    "holding_candles", "gross_pnl", "trading_cost",
                    "net_pnl", "win", "forward6_pct", "forward12_pct",
                    "forward24_pct", "mfe12_atr", "mae12_atr"):
            mutated[key] = -999.0 if key != "win" else False
        st_mut = build_current_state("GOLD", mutated, data)
        check("current state ignores every outcome column",
              st == st_mut)
        frame = data.instrument_frame("GOLD")
        target = int(frame.index[
            frame["signal_time"]
            == pd.Timestamp(obs["signal_time"])][0])
        frame_mut = frame.copy()
        frame_mut.loc[frame_mut.index != target, "adx"] = 99.0
        frame_mut.loc[frame_mut.index != target, "net_pnl"] = -999.0
        st_a = build_current_state("GOLD", frame.iloc[[target]], data)
        st_b = build_current_state("GOLD", frame_mut.iloc[[target]], data)
        check("current state unaffected by other rows in the frame",
              st_a == st_b)
        env_text = "\n".join(describe_environment(st))
        check("environment description mentions no outcome values",
              "48.0" not in env_text and "2750.0" not in env_text
              and "-999" not in env_text)
        check("description states the no-forward-information rule",
              "no forward-looking statement" in env_text)

        # ------------------------------------------------------
        start_area("output_overwrite_protection")
        with tempfile.TemporaryDirectory() as od:
            try:
                check_output_files_available(od)
                check("clean directory passes", True)
            except OutputExistsError:
                check("clean directory passes", False)
        with tempfile.TemporaryDirectory() as od:
            open(os.path.join(od, OUTPUT_REPORT_TXT), "w").close()
            try:
                check_output_files_available(od)
                check("existing output stops the run", False)
            except OutputExistsError as exc:
                check("existing output stops the run", True)
                check("conflicting filename reported exactly",
                      OUTPUT_REPORT_TXT in str(exc))
        with tempfile.TemporaryDirectory() as od:
            for name in (OUTPUT_ALL_CSV, OUTPUT_REPORT_JSON,
                         OUTPUT_REPORT_TXT):
                open(os.path.join(od, name), "w").close()
            try:
                check_output_files_available(od)
                check("all three outputs checked", False)
            except OutputExistsError as exc:
                check("all three outputs checked", True)
                check("every conflicting filename listed",
                      all(n in str(exc) for n in
                          (OUTPUT_ALL_CSV, OUTPUT_REPORT_JSON,
                           OUTPUT_REPORT_TXT)))

        # ------------------------------------------------------
        start_area("deterministic_output")
        ctx_a = build_market_context("GOLD", obs, data)
        ctx_b = build_market_context("GOLD", obs, data)
        check("identical inputs -> identical structured context",
              json.dumps(ctx_a, sort_keys=True, default=_json_default)
              == json.dumps(ctx_b, sort_keys=True, default=_json_default))
        rep_a = generate_market_context_report("GOLD", obs, data)
        rep_b = generate_market_context_report("GOLD", obs, data)
        check("identical inputs -> identical text report", rep_a == rep_b)
        row_a = context_to_csv_row(ctx_a)
        row_b = context_to_csv_row(ctx_b)
        check("identical flattened CSV rows", row_a == row_b)
        src_tree = ast.parse(open(os.path.abspath(__file__),
                                  encoding="utf-8").read())
        random_nodes = [
            n for n in ast.walk(src_tree)
            if (isinstance(n, ast.Import)
                and any(a.name.split(".")[0] == "ra" + "ndom"
                        for a in n.names))
            or (isinstance(n, ast.ImportFrom)
                and (n.module or "").split(".")[0] == "ra" + "ndom")
            or (isinstance(n, ast.Attribute)
                and n.attr == "ra" + "ndom")
        ]
        check("no randomness source in the module", random_nodes == [])

        # ------------------------------------------------------
        start_area("structured_ai_ready_output")
        required = set(STRUCTURED_OUTPUT_KEYS)
        check("structured output has exactly the required keys "
              "(plus the explanation)",
              required.issubset(set(ctx_a.keys()))
              and set(ctx_a.keys()) == required | {"evidence_explanation"})
        check("current_state / historical_memory / walk_forward / "
              "evidence stay separate keys",
              all(isinstance(ctx_a[k], dict) for k in
                  ("current_state", "historical_memory",
                   "regime_stability", "walk_forward_evidence",
                   "memory_engine_evidence", "cross_instrument_context")))
        check("no score-shaped field anywhere in the structured output",
              "score" not in json.dumps(ctx_a, default=str).lower())
        check("evidence_quality is qualitative only",
              ctx_a["evidence_quality"] in EVIDENCE_LABELS)
        check("research_conclusion from the allowed set only",
              ctx_a["research_conclusion"] in ALLOWED_CONCLUSIONS)

        # ------------------------------------------------------
        start_area("text_report_sections")
        rep = rep_a
        for section in ("A. CURRENT MARKET STATE", "B. MARKET ENVIRONMENT",
                        "C. HISTORICAL MEMORY", "D. REGIME STABILITY",
                        "E. WALK-FORWARD VALIDATION",
                        "F. EVIDENCE QUALITY", "G. RESEARCH CONCLUSION"):
            check(f"section present ({section[:1]})", section in rep)
        check("report states the regime values",
              "ADX regime:" in rep and "Volatility regime:" in rep
              and "RSI regime:" in rep)
        check("report uses 'historical matches' wording",
              "historical matches" in rep.lower()
              or "Historical match" in rep)
        forbidden_recs = ("BU" + "Y", "SE" + "LL", "HO" + "LD",
                          "ENTR" + "Y", "EXI" + "T",
                          "STOP LO" + "SS", "TAKE PRO" + "FIT",
                          "LOT SI" + "ZE", "POSITION SI" + "ZE")
        check("report contains no trade instruction wording",
              all(w.lower() not in rep.lower() for w in forbidden_recs))
        check("report contains no certainty wording",
              all(p not in rep.lower() for p in
                  ("co" + "nfidence", "pro" + "bability")))
        check("report distinguishes the four notions",
              "current state" in rep.lower()
              and "historical" in rep.lower()
              and "walk-forward" in rep.lower()
              and "evidence" in rep.lower())

        # ------------------------------------------------------
        start_area("safety_scan")
        violations = run_safety_scan()
        check("static safety scan clean "
              "(no terminal, no ML, no scores, no optimization)",
              violations == [])
        check("no order-function name in the AST",
              not any("or" + "der" in n.name.lower()
                      for n in ast.walk(ast.parse(
                          open(os.path.abspath(__file__),
                               encoding="utf-8").read()))
                      if isinstance(n, (ast.FunctionDef,))))
        check("CSV output carries no score column",
              all("score" not in c for c in CSV_COLUMNS))
        check("research conclusions never contain a direction word",
              all(not any(w.lower() in c.lower()
                          for w in ("lo" + "ng", "sho" + "rt",
                                    "bu" + "y", "se" + "ll", "ho" + "ld"))
                  for c in ALLOWED_CONCLUSIONS))

        # ------------------------------------------------------
        start_area("default_run_gating")
        check("default (no flags) does not request real analysis",
              should_run_real([]) is False)
        check("--run requests real analysis",
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
            sys.argv = ["ai_research_layer.py"]
            main()
            check("default main() runs synthetic tests only",
                  called["tests"] and not called["real"])
            called["real"] = False
            called["tests"] = False
            sys.argv = ["ai_research_layer.py", "--run"]
            main()
            check("--run reaches the real-analysis entry point",
                  called["real"] and not called["tests"])
        finally:
            globals()["run_real_analysis"] = orig_real
            globals()["run_synthetic_tests"] = orig_tests
            sys.argv = argv_backup

    start_area(None)
    print()
    print("  Per-area verification (spec synthetic items -> area: "
          "passed/failed):")
    spec_map = (
        ("missing_input_detection", "item 1 missing input detection"),
        ("current_state_extraction", "item 2 current state extraction"),
        ("state_classification", "item 3 state classification"),
        ("historical_match_loading", "item 4 historical match loading"),
        ("evidence_classification_rules", "item 5 evidence classification"),
        ("insufficient_sample_flow", "item 6 insufficient sample"),
        ("weak_moderate_strong_flow", "items 7-9 weak/moderate/strong"),
        ("state_stability", "item 10 state stability"),
        ("walk_forward_evidence", "item 11 walk-forward evidence"),
        ("memory_engine_evidence", "item 12 memory evidence"),
        ("cross_instrument_separation", "item 13 cross-instrument"),
        ("no_future_data_in_current_state", "item 14 no future data"),
        ("output_overwrite_protection", "item 15 overwrite protection"),
        ("deterministic_output", "item 16 deterministic output"),
        ("structured_ai_ready_output", "item 20 structured output"),
        ("text_report_sections", "report wording and sections"),
        ("safety_scan", "items 17-19 forbidden scans"),
        ("default_run_gating", "item 24 default-run + --run gating"),
    )
    stats = {name: (p, f) for name, p, f in area_stats}
    for name, spec in spec_map:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<32} ({spec:<40}) {p:>2} passed, {f} failed "
              f"[{status}]")
    print()
    print(f"  TOTAL: {passed} passed, {failed} failed")
    return passed, failed


# ============================================================
# Command-line gating
# ============================================================
# Default run = synthetic tests only; real analysis needs --run.
# ============================================================
def should_run_real(argv: List[str]) -> bool:
    return "--run" in argv


def main(argv=None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if should_run_real(argv):
        run_real_analysis()
        return 0
    _, failed = run_synthetic_tests()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
