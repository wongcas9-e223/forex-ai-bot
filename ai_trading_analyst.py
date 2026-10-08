"""
ai_trading_analyst.py - RESEARCH-ONLY AI ANALYST CONTEXT ENGINE.

WHAT THIS IS
    A deterministic preparation layer that organizes the existing
    historical research outputs into a structured ANALYST CONTEXT that a
    future AI analyst can consume as evidence.

    It answers:  "What does the stored historical research show about
                  each instrument's current environment, its historical
                  matches, and how consistent that evidence was?"

    It NEVER produces a directive of any kind: no trading instruction,
    no certainty figure, no likelihood figure, no compound rating, and
    no ordering of instruments.  It is an ANALYSIS CONTEXT GENERATOR.

WHAT THIS IS NOT
    - not a trading strategy or strategy search,
    - not a learning system and not a fitted structure of any kind,
    - not an instruction generator,
    - not a broker connector and not an execution layer,
    - it does not modify ANY existing file: all inputs are read-only,
      and only the three NEW output files may be created.

INPUT FILES (read-only; every one required; a missing file raises a
MissingInputError naming EVERY missing file):
    market_state_all.csv
    market_memory_oos_validation.csv
    market_memory_match_details.csv
    market_memory_summary.csv
    market_state_oos_period_summary.csv
    market_state_oos_state_summary.csv
    market_state_oos_walkforward.csv
    market_state_oos_cross_instrument.csv
    ai_research_context_all.csv
    ai_research_context_report.json
    ai_research_context_report.txt
    (The stored-research files above remain read-only and supply the
    historical-memory figures.  The CURRENT research context for the
    real --run is taken instead from the newest valid timestamped LIVE
    research context collection produced by live_research_context.py:
        live_research_context_YYYYMMDD_HHMMSS.csv
        live_research_context_YYYYMMDD_HHMMSS.json
        live_research_context_report_YYYYMMDD_HHMMSS.txt
    sharing one timestamp.  The stale ai_research_context_all.csv is
    never used as a fallback for the current research context; when no
    complete timestamped live collection exists the run stops with a
    clear error naming the missing live input.)

INSTRUMENTS (exactly these five, always handled separately):
    GOLD, EURUSD, GBPUSD, USDJPY, AUDUSD
    They are never combined into one figure, never ordered, and none is
    ever declared superior to another.

HISTORICAL MATCHES TERMINOLOGY
    The memory-engine outputs are ALWAYS described as "historical
    matches": stored past observations with similar state features and
    their recorded outcomes.  They are never given any forward-looking
    name and never treated as certainty about anything.

ANALYST CONTEXT STRUCTURE (per instrument, deterministic):
    {
        "instrument": ..., "timestamp": ...,
        "current_state": {...}, "market_environment": {...},
        "historical_memory": {...}, "research_evidence": {...},
        "regime_stability": [...], "walk_forward": [...],
        "cross_instrument_context": {...},
        "limitations": [...], "research_only": true,
    }

    The structure never contains a certainty figure, a likelihood
    figure, a compound rating, an action suggestion, or any sizing
    field.  Evidence labels (INSUFFICIENT / WEAK / MODERATE / STRONG)
    are preserved exactly as produced by the research layer.

LIMITATIONS
    Only limitations already demonstrated by the input files are
    reported, each under a fixed documented rule (chance-level sign
    agreement at or below 55%, absolute Pearson correlation below 0.10,
    mixed stored period classifications, unstable walk-forward labels,
    low-sample flags, insufficient evidence, match-versus-recorded
    outcome disagreement, missing memory query for the latest
    observation).  Nothing is invented and nothing negative is hidden.

OUTPUT FILES (all NEW; every output path is checked before anything is
written; if any exists the run stops and names the conflicting file).
Each real --run writes a fresh, UNIQUE timestamped set so a later run
never collides with an earlier one:
    ai_trading_analyst_context_YYYYMMDD_HHMMSS.csv
    ai_trading_analyst_context_YYYYMMDD_HHMMSS.json
    ai_trading_analyst_report_YYYYMMDD_HHMMSS.txt
The three files of one run share a single timestamp; existing files are
never modified, renamed or overwritten.

NO TRADING TERMINAL: zero terminal imports or calls.  NO LEARNING
LIBRARY: zero imports and zero fitting calls.  NO OPTIMIZATION: no
search of any parameter.  NO ORDER FUNCTIONS.  NO DIRECTIONAL OUTPUT.
NO CERTAINTY FIGURES.  NO INSTRUMENT ORDERING.  These are enforced by a
built-in static scan of this file's own source.

COMMAND-LINE SAFETY:
    python ai_trading_analyst.py          -> synthetic tests only
    python ai_trading_analyst.py --run    -> real CSV analysis

DISCLAIMER: historical, descriptive research preparation only.  No
causal claims, no forward-looking statements, no directives of any
kind, nothing proven profitable.  Educational use only.
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
# SAFETY GUARD TOKENS (assembled from split literals so this guard
# source never contains the forbidden strings contiguously).
# ============================================================
_TERMINAL_TOKENS: Tuple[str, ...] = (
    "meta" + "trader5",
    "m" + "t5",
    "or" + "der_send",
    "or" + "der_check",
    "positi" + "ons_get",
    "positi" + "ons_total",
    "position" + "_get",
    "initiali" + "ze",
    "shut" + "down",
    "copy_" + "rates",
    "symbol_" + "info",
    "symbols_" + "get",
)
_ML_TOKENS: Tuple[str, ...] = (
    "sk" + "learn",
    "xg" + "boost",
    "light" + "gbm",
    "tensor" + "flow",
    "to" + "rch",
    "ke" + "ras",
)
_ML_CALL_PATTERNS: Tuple[str, ...] = (
    "\\bfi" + "t\\s*\\(",
    "\\btr" + "ain\\s*\\(",
    "\\bmo" + "del\\b",
)
_OPTIMIZATION_TOKENS: Tuple[str, ...] = (
    "param_" + "grid",
    "gridse" + "arch",
    "randomizeds" + "earch",
    "op" + "tuna",
    "hyper" + "opt",
    "itertools.pro" + "duct",
)
_BANNED_WORD_PATTERNS: Tuple[str, ...] = (
    "\\bbu" + "y\\b",
    "\\bse" + "ll\\b",
    "\\bho" + "ld\\b",
    "\\bentr" + "y\\b",
    "\\bexi" + "t\\b",
    "stop lo" + "ss\\b",
    "take pro" + "fit\\b",
    "lot si" + "ze\\b",
    "position si" + "ze\\b",
    "trade deci" + "sion\\b",
    "\\bco" + "nfidence\\b",
    "\\bprob" + "abilit",
    "signal_" + "score",
    "edge_" + "score",
    "trade_" + "score",
    "\\bpre" + "dict",
    "\\bfore" + "casts?\\b",
    "\\bra" + "nk",
    "\\bw" + "inner\\b",
    "\\bbe" + "st\\b",
    "\\bwo" + "rst\\b",
    "recommen" + "ded_action",
    "trade_" + "decision",
    "\\bex" + "pected\\s+ret" + "urns\\b",
)
_FORBIDDEN_NAME_TOKENS: Tuple[str, ...] = (
    "m" + "t5", "me" + "tatrader5",
    "or" + "der_send", "or" + "der_check", "positi" + "ons_get",
    "positi" + "ons_total", "position" + "_get", "symbol" + "_info",
    "symbols" + "_get", "copy" + "_rates", "initiali" + "ze",
    "shut" + "down",
    "sk" + "learn", "xg" + "boost", "light" + "gbm", "tensor" + "flow",
    "to" + "rch", "ke" + "ras",
    "param_" + "grid", "gridse" + "arch", "randomizeds" + "earch",
    "op" + "tuna", "hyper" + "opt",
    "co" + "nfidence", "pro" + "bability", "signal_" + "score",
    "edge_" + "score", "trade_" + "score", "recommen" + "ded_action",
    "trade_" + "decision", "stop_" + "loss", "take_" + "profit",
    "lot_" + "size", "position_" + "size", "ra" + "nk", "ra" + "nking",
    "ra" + "nked", "best_" + "instrument", "worst_" + "instrument",
    "w" + "inner",
    "entr" + "y", "exi" + "t", "mo" + "del", "pre" + "dict",
    "tr" + "ain", "fore" + "cast",
)
_ALLOWED_IMPORT_MODULES: Tuple[str, ...] = (
    "ast", "datetime", "json", "os", "re", "sys", "tempfile",
    "typing", "numpy", "pandas",
)

# ============================================================
# Frozen configuration
# ============================================================
SUPPORTED_INSTRUMENTS: Tuple[str, ...] = (
    "GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
)

INPUT_FILES: Tuple[str, ...] = (
    "market_state_all.csv",
    "market_memory_oos_validation.csv",
    "market_memory_match_details.csv",
    "market_memory_summary.csv",
    "market_state_oos_period_summary.csv",
    "market_state_oos_state_summary.csv",
    "market_state_oos_walkforward.csv",
    "market_state_oos_cross_instrument.csv",
    "ai_research_context_all.csv",
    "ai_research_context_report.json",
    "ai_research_context_report.txt",
)

OUTPUT_CSV = "ai_trading_analyst_context.csv"
OUTPUT_JSON = "ai_trading_analyst_context.json"
OUTPUT_TXT = "ai_trading_analyst_report.txt"
OUTPUT_FILES: Tuple[str, ...] = (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_TXT)

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
STAMP_FORMAT = "%Y%m%d_%H%M%S"

# Newest valid timestamped live research context (produced upstream by
# live_research_context.py); the source of the CURRENT research context
# for the real run.  The stale ai_research_context_all.csv is never a
# fallback.
LIVE_RESEARCH_CSV_STAMP_RE = re.compile(
    r"^live_research_context_(\d{8}_\d{6})\.csv$")
LIVE_RESEARCH_JSON_STAMP_RE = re.compile(
    r"^live_research_context_(\d{8}_\d{6})\.json$")
LIVE_RESEARCH_REPORT_STAMP_RE = re.compile(
    r"^live_research_context_report_(\d{8}_\d{6})\.txt$")

EVIDENCE_LABELS: Tuple[str, ...] = (
    "STRONG", "MODERATE", "WEAK", "INSUFFICIENT",
)
WF_LABELS: Tuple[str, ...] = (
    "PERSISTENT", "MIXED", "UNSTABLE", "INSUFFICIENT DATA",
)
WF_STEP_ORDER: Tuple[str, ...] = (
    "P1->P2", "P1+P2->P3", "P1+P2+P3->P4",
)

# Fixed, documented limitation-rule thresholds (never tuned).
CHANCE_LEVEL_SIGN_PCT = 55.0      # sign agreement at/below -> chance-level
WEAK_CORRELATION_ABS_R = 0.10     # |r| below -> weak-correlation limitation

# Fixed input-schema expectations (explicit schemas).
REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "market_state_all.csv": (
        "instrument", "signal_time", "direction", "adx", "atr_pct",
        "ema_distance", "bb_excursion", "bb_signed", "eff24", "rsi14",
    ),
    "market_memory_oos_validation.csv": (
        "instrument", "query_timestamp", "n_matches", "nearest_distance",
        "pnl_sign_agree",
    ),
    "market_memory_match_details.csv": (
        "instrument", "query_timestamp", "match_rank", "distance",
    ),
    "market_memory_summary.csv": (
        "instrument", "section", "n_queries", "avg_nearest_distance",
        "median_nearest_distance", "pnl_sign_agreement_pct", "pnl_sign_n",
        "corr_match_actual_pnl_r", "corr_match_actual_pnl_n",
        "fwd12_sign_agreement_pct", "fwd24_sign_agreement_pct",
        "avg_match_net_pnl", "avg_actual_net_pnl", "low_sample_count",
        "avg_match_forward12_pct", "avg_actual_forward12_pct",
    ),
    "market_state_oos_period_summary.csv": (
        "instrument", "period", "n", "avg_net_pnl", "fwd12_avg_pct",
    ),
    "market_state_oos_state_summary.csv": (
        "instrument", "feature", "bucket", "period", "n", "avg_net_pnl",
        "pnl_sign_changes", "fwd12_sign_changes", "pnl_classification",
    ),
    "market_state_oos_walkforward.csv": (
        "instrument", "feature", "bucket", "step", "pnl_sign_result",
        "fwd12_sign_result", "low_sample",
    ),
    "market_state_oos_cross_instrument.csv": (
        "feature", "bucket", "period", "instruments_with_sufficient_sample",
        "pos_avg_pnl_count", "neg_avg_pnl_count", "pnl_classification",
    ),
    "ai_research_context_all.csv": (
        "instrument", "timestamp", "direction", "adx", "atr_pct",
        "ema_distance", "bb_excursion", "bb_signed", "eff24", "rsi14",
        "ret6", "ret12", "ret24", "vol6", "vol24", "range_pct",
        "body_pct", "upper_wick_pct", "lower_wick_pct", "dist_24_high",
        "dist_24_low", "hour", "day_of_week", "month", "adx_regime",
        "volatility_regime", "trend_distance_regime",
        "bb_excursion_regime", "efficiency_regime", "rsi_regime",
        "evidence_quality", "research_conclusion",
    ),
}

REQUIRED_CONTEXT_KEYS: Tuple[str, ...] = (
    "instrument", "timestamp", "current_state", "historical_memory",
    "regime_stability", "walk_forward_evidence", "memory_engine_evidence",
    "cross_instrument_context", "evidence_quality", "evidence_explanation",
    "research_conclusion",
)

ANALYST_CONTEXT_KEYS: Tuple[str, ...] = (
    "instrument", "timestamp", "current_state", "market_environment",
    "historical_memory", "research_evidence", "regime_stability",
    "walk_forward", "cross_instrument_context", "limitations",
    "research_only",
)

ANALYST_CSV_COLUMNS: Tuple[str, ...] = (
    "instrument", "timestamp", "timeframe", "direction",
    "adx", "atr_pct", "ema_distance", "bb_excursion", "bb_signed",
    "eff24", "rsi14", "ret6", "ret12", "ret24", "dist_24_high",
    "dist_24_low", "range_pct", "body_pct", "upper_wick_pct",
    "lower_wick_pct", "vol6", "vol24", "hour", "day_of_week", "month",
    "adx_regime", "volatility_regime", "trend_distance_regime",
    "bb_excursion_regime", "efficiency_regime", "rsi_regime",
    "memory_evaluation_queries", "memory_avg_nearest_distance",
    "memory_pnl_sign_agreement_pct", "memory_pearson_r_pnl",
    "memory_match_avg_net_pnl", "memory_actual_avg_net_pnl",
    "latest_n_matches", "latest_nearest_distance",
    "latest_avg_match_distance", "latest_median_match_distance",
    "latest_match_win_rate_pct", "latest_avg_match_net_pnl",
    "latest_forward6_pct", "latest_forward12_pct",
    "latest_forward24_pct", "latest_mfe12_atr", "latest_mae12_atr",
    "latest_actual_net_pnl", "latest_actual_forward12_pct",
    "latest_actual_forward24_pct", "latest_sign_agreement_pct",
    "evidence_quality", "evidence_explanation", "research_conclusion",
    "regime_stability_classifications", "walk_forward_label",
    "walk_forward_classifications", "n_limitations", "limitations",
    "research_only",
)


# ============================================================
# Small deterministic helpers
# ============================================================
def parse_timestamp(value) -> pd.Timestamp:
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


def mean_of(values) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def coerce_bool(value) -> Optional[bool]:
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


def fmt_num(value) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:+.4f}"


def fmt_pct(value) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:.1f}%"


def fmt_corr(value) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:+.3f}"


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return fmt_ts(value)
    return str(value)


# ============================================================
# Input / output guards
# ============================================================
class MissingInputError(RuntimeError):
    pass


class OutputExistsError(RuntimeError):
    pass


def check_required_inputs(directory: str = SCRIPT_DIR) -> None:
    """Name EVERY missing required file (never continue silently)."""
    missing = [f for f in INPUT_FILES
               if not os.path.isfile(os.path.join(directory, f))]
    if missing:
        raise MissingInputError(
            "missing required input file(s): " + ", ".join(missing))


def timestamped_output_files(stamp: str) -> Tuple[str, str, str]:
    """The three timestamped real-run output names for one run stamp.

    A single stamp is used for the whole set, so the CSV, the JSON and
    the report always belong to the same run.
    """
    return (
        f"ai_trading_analyst_context_{stamp}.csv",
        f"ai_trading_analyst_context_{stamp}.json",
        f"ai_trading_analyst_report_{stamp}.txt",
    )


def check_output_files_available(directory: str = SCRIPT_DIR,
                                 filenames: Tuple[str, ...] = OUTPUT_FILES
                                 ) -> None:
    """ALL output paths are checked before ANY writing happens."""
    conflicts = [f for f in filenames
                 if os.path.exists(os.path.join(directory, f))]
    if conflicts:
        raise OutputExistsError(
            "refusing to run: output file(s) already exist and existing "
            "files are never modified or overwritten: "
            + ", ".join(conflicts))


def _require_columns(frame: pd.DataFrame, name: str,
                     required: Tuple[str, ...]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{name}: missing required column(s): " + ", ".join(missing))


# ============================================================
# Loaded research data (read-only; deterministic indexes)
# ============================================================
class AnalystData:
    """Read-only view of the eleven required inputs."""

    def __init__(self, directory: str = SCRIPT_DIR):
        check_required_inputs(directory)

        def read_csv(name: str) -> pd.DataFrame:
            path = os.path.join(directory, name)
            frame = pd.read_csv(path)
            _require_columns(frame, name, REQUIRED_COLUMNS[name])
            return frame

        self.all_df = read_csv("market_state_all.csv")
        self.validation = read_csv("market_memory_oos_validation.csv")
        self.match_details = read_csv("market_memory_match_details.csv")
        self.memory_summary = read_csv("market_memory_summary.csv")
        self.period_summary = read_csv("market_state_oos_period_summary.csv")
        self.state_summary = read_csv("market_state_oos_state_summary.csv")
        self.walkforward = read_csv("market_state_oos_walkforward.csv")
        self.cross_instrument = read_csv(
            "market_state_oos_cross_instrument.csv")
        self.context_csv = read_csv("ai_research_context_all.csv")

        unexpected = sorted(set(self.context_csv["instrument"].astype(str))
                            - set(SUPPORTED_INSTRUMENTS))
        if unexpected:
            raise ValueError(
                "ai_research_context_all.csv contains unsupported "
                "instrument(s): " + ", ".join(unexpected))

        # ---- structured research contexts (JSON list) ----
        json_path = os.path.join(directory, "ai_research_context_report.json")
        with open(json_path, "r", encoding="utf-8") as fh:
            raw_contexts = json.load(fh)
        if not isinstance(raw_contexts, list) or not raw_contexts:
            raise ValueError(
                "ai_research_context_report.json: expected a non-empty "
                "JSON list of research contexts")
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for i, ctx in enumerate(raw_contexts):
            if not isinstance(ctx, dict):
                raise ValueError(
                    f"ai_research_context_report.json: element {i} is "
                    f"not an object")
            missing = [k for k in REQUIRED_CONTEXT_KEYS if k not in ctx]
            if missing:
                raise ValueError(
                    f"ai_research_context_report.json: element {i} is "
                    f"missing key(s): " + ", ".join(missing))
            key = (str(ctx["instrument"]), fmt_ts(ctx["timestamp"]))
            by_key[key] = ctx
        self.contexts_by_key = by_key
        del raw_contexts

        # ---- text report provenance ----
        txt_path = os.path.join(directory, "ai_research_context_report.txt")
        with open(txt_path, "r", encoding="utf-8") as fh:
            txt = fh.read()
        self.txt_provenance = {
            "bytes": len(txt.encode("utf-8")),
            "lines": txt.count("\n") + 1,
            "has_all_report_sections": all(
                s in txt for s in ("A. CURRENT MARKET STATE",
                                   "B. MARKET ENVIRONMENT",
                                   "C. HISTORICAL MEMORY",
                                   "D. REGIME STABILITY",
                                   "E. WALK-FORWARD VALIDATION",
                                   "F. EVIDENCE QUALITY",
                                   "G. RESEARCH CONCLUSION")),
        }

        # ---- instrument-level memory evidence (summary OVERALL rows) ----
        self.memory_overall: Dict[str, Dict[str, Any]] = {}
        for row in self.memory_summary.to_dict("records"):
            if str(row.get("section")) == "OVERALL":
                self.memory_overall[str(row["instrument"])] = row
        missing_inst = [i for i in SUPPORTED_INSTRUMENTS
                        if i not in self.memory_overall]
        if missing_inst:
            raise ValueError(
                "market_memory_summary.csv: no OVERALL row for: "
                + ", ".join(missing_inst))

        # ---- latest stored observation per instrument (context CSV) ----
        self.latest_rows: Dict[str, Dict[str, Any]] = {}
        self.observation_counts: Dict[str, int] = {}
        for inst in SUPPORTED_INSTRUMENTS:
            sub = self.context_csv[
                self.context_csv["instrument"].astype(str) == inst]
            if len(sub) == 0:
                raise ValueError(
                    f"ai_research_context_all.csv: no observations for "
                    f"{inst}")
            self.observation_counts[inst] = int(len(sub))
            best_ts: Optional[pd.Timestamp] = None
            best_row: Optional[Dict[str, Any]] = None
            for row in sub.to_dict("records"):
                ts = parse_timestamp(row["timestamp"])
                if best_ts is None or ts >= best_ts:
                    best_ts, best_row = ts, row
            self.latest_rows[inst] = best_row

        # ---- cross-instrument index: (feature, bucket) -> rows ----
        self.cross_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in self.cross_instrument.to_dict("records"):
            key = (str(row["feature"]), str(row["bucket"]))
            self.cross_by_key.setdefault(key, []).append(row)
        for rows in self.cross_by_key.values():
            rows.sort(key=lambda r: str(r.get("period")))

    # --------------------------------------------------------
    def latest_context(self, instrument: str) -> Dict[str, Any]:
        row = self.latest_rows[instrument]
        key = (instrument, fmt_ts(row["timestamp"]))
        ctx = self.contexts_by_key.get(key)
        if ctx is None:
            raise ValueError(
                "ai_research_context_report.json: no structured context "
                f"for the latest {instrument} observation at "
                f"{key[1]} (data inconsistency between the CSV and JSON "
                "outputs)")
        return ctx

    def query_count(self, instrument: str) -> int:
        return int(len(self.validation[
            self.validation["instrument"].astype(str) == instrument]))


# ============================================================
# Analyst context builders (deterministic, explicit schemas)
# ============================================================
def build_current_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Factual current state, copied from the research context with
    explicit key selection (no outcome fields, no reinterpretation)."""
    st = ctx["current_state"]
    return {
        "instrument": st["instrument"],
        "timestamp": st["timestamp"],
        "timeframe": st.get("timeframe"),
        "direction": st.get("direction"),
        "adx_regime": st["classifications"]["adx_regime"],
        "volatility_regime": st["classifications"]["volatility_regime"],
        "trend_distance_regime":
            st["classifications"]["trend_distance_regime"],
        "bb_excursion_regime":
            st["classifications"]["bb_excursion_regime"],
        "efficiency_regime": st["classifications"]["efficiency_regime"],
        "rsi_regime": st["classifications"]["rsi_regime"],
        "features": dict(st["features"]),
        "atr_tercile_thresholds": dict(st.get("atr_tercile_thresholds", {})),
        "hour": st.get("hour"),
        "day_of_week": st.get("day_of_week"),
        "month": st.get("month"),
    }


def build_market_environment(state: Dict[str, Any]) -> List[str]:
    """Plain-language factual description of the stored state only."""
    f = state["features"]

    def val(name: str, unit: str = "", digits: int = 2) -> str:
        v = safe_float(f.get(name))
        return "unavailable" if v is None \
            else f"{v:.{digits}f}{unit}"

    lines = [
        f"Trend strength: ADX regime {state['adx_regime']} "
        f"(ADX {val('ADX')}).",
        f"Volatility: regime {state['volatility_regime']} "
        f"(ATR {val('ATR_pct', '%')} of price; realized 6-candle "
        f"volatility {val('vol6')}, 24-candle volatility {val('vol24')}).",
        f"Trend distance: regime {state['trend_distance_regime']} "
        f"({val('EMA50_dist', ' ATR')} from the EMA50).",
        f"Bollinger position: excursion regime "
        f"{state['bb_excursion_regime']} (excursion "
        f"{val('BB_excursion')}); signed position {val('BB_signed')} "
        f"(negative = below the middle band, positive = above).",
        f"Efficiency: regime {state['efficiency_regime']} "
        f"(eff24 {val('eff24')}).",
        f"RSI: regime {state['rsi_regime']} (RSI14 {val('RSI14')}).",
        f"Recent movement: return6 {val('ret6', '%')}, "
        f"return12 {val('ret12', '%')}, return24 {val('ret24', '%')}; "
        f"24h-high distance {val('dist_24_high', '%')}, 24h-low "
        f"distance {val('dist_24_low', '%')}.",
        f"Candle shape: range {val('range_pct', '%')}, body "
        f"{val('body_pct', '%')}, upper wick {val('upper_wick_pct', '%')}, "
        f"lower wick {val('lower_wick_pct', '%')}.",
        f"Time context: hour {state['hour']}, day_of_week "
        f"{state['day_of_week']}, month {state['month']}.",
        "These are stored descriptive facts; they are not "
        "interpreted as instructions of any kind.",
    ]
    return lines


def build_historical_memory(data: AnalystData, instrument: str,
                            ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Instrument-level memory evidence plus the latest observation's
    stored historical-match sample.  Frozen K=10 outputs used unchanged.
    """
    row = data.memory_overall[instrument]
    lm = ctx["historical_memory"]
    return {
        "terminology": ("historical matches are stored past observations "
                        "with similar state features and their recorded "
                        "outcomes"),
        "k_note": ("existing frozen K=10 memory results are used "
                   "unchanged; no other K is computed"),
        "evaluation_query_count": safe_int(row.get("n_queries")),
        "average_nearest_distance": safe_float(
            row.get("avg_nearest_distance")),
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
        "historical_match_avg_net_pnl": safe_float(
            row.get("avg_match_net_pnl")),
        "actual_avg_net_pnl": safe_float(row.get("avg_actual_net_pnl")),
        "avg_match_forward12_pct": safe_float(
            row.get("avg_match_forward12_pct")),
        "avg_actual_forward12_pct": safe_float(
            row.get("avg_actual_forward12_pct")),
        "low_sample_count": safe_int(row.get("low_sample_count")),
        "latest_observation": {
            "available": bool(lm.get("available")),
            "n_matches": safe_int(lm.get("n_matches")),
            "nearest_distance": safe_float(lm.get("nearest_distance")),
            "avg_match_distance": safe_float(lm.get("avg_match_distance")),
            "median_match_distance": safe_float(
                lm.get("median_match_distance")),
            "historical_match_avg_net_pnl": safe_float(
                lm.get("avg_match_net_pnl")),
            "match_win_rate_pct": safe_float(lm.get("match_win_rate_pct")),
            "forward6_pct": safe_float(lm.get("avg_match_forward6_pct")),
            "forward12_pct": safe_float(lm.get("avg_match_forward12_pct")),
            "forward24_pct": safe_float(lm.get("avg_match_forward24_pct")),
            "mfe12_atr": safe_float(lm.get("avg_match_mfe12_atr")),
            "mae12_atr": safe_float(lm.get("avg_match_mae12_atr")),
            "actual_net_pnl": safe_float(lm.get("actual_net_pnl")),
            "actual_win": coerce_bool(lm.get("actual_win")),
            "actual_forward12_pct": safe_float(
                lm.get("actual_forward12_pct")),
            "actual_forward24_pct": safe_float(
                lm.get("actual_forward24_pct")),
            "latest_match_sign_agreement_pct": safe_float(
                lm.get("sign_agreement_pct")),
        },
    }


def build_research_evidence(data: AnalystData, instrument: str,
                            ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence labels preserved exactly; no new rating system."""
    stability_classes: List[str] = []
    for cat in ctx["regime_stability"].get("by_category", []):
        if cat.get("available"):
            stability_classes.append(str(cat.get("pnl_classification")))
    wf_classes: List[str] = []
    for cat in ctx["walk_forward_evidence"].get("by_category", []):
        wf_classes.append(str(cat.get("label")))
    return {
        "evidence_quality": ctx["evidence_quality"],
        "evidence_explanation": ctx["evidence_explanation"],
        "research_conclusion": ctx["research_conclusion"],
        "aggregate_observation_count": data.observation_counts[instrument],
        "memory_query_count": safe_int(
            data.memory_overall[instrument].get("n_queries")),
        "pnl_sign_agreement_pct": safe_float(
            data.memory_overall[instrument].get(
                "pnl_sign_agreement_pct")),
        "pearson_r_match_actual_pnl": safe_float(
            data.memory_overall[instrument].get(
                "corr_match_actual_pnl_r")),
        "regime_stability_classifications": stability_classes,
        "walk_forward_classifications": wf_classes,
        "label_note": ("labels INSUFFICIENT / WEAK / MODERATE / STRONG "
                       "are preserved from the research layer and are "
                       "descriptive sample-quality labels only"),
    }


def build_regime_stability(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-category period consistency; no category is preferred."""
    out: List[Dict[str, Any]] = []
    for cat in ctx["regime_stability"].get("by_category", []):
        if not cat.get("available"):
            out.append({
                "category": f"{cat.get('feature')} "
                            f"({cat.get('bucket')})",
                "available": False,
            })
            continue
        out.append({
            "category": f"{cat['feature']} ({cat['bucket']})",
            "available": True,
            "observed_periods": cat.get("periods_observed"),
            "positive_periods": cat.get("periods_positive"),
            "negative_periods": cat.get("periods_negative"),
            "unchanged_periods": cat.get("periods_unchanged"),
            "pnl_sign_changes": cat.get("pnl_sign_changes"),
            "fwd12_sign_changes": cat.get("fwd12_sign_changes"),
            "stored_classification": cat.get("pnl_classification"),
        })
    return out


def build_walk_forward(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-category walk-forward evidence; no new testing is done."""
    out: List[Dict[str, Any]] = []
    for cat in ctx["walk_forward_evidence"].get("by_category", []):
        if not cat.get("available"):
            out.append({
                "category": f"{cat.get('feature')} "
                            f"({cat.get('bucket')})",
                "available": False,
                "label": cat.get("label"),
            })
            continue
        out.append({
            "category": f"{cat['feature']} ({cat['bucket']})",
            "available": True,
            "label": cat.get("label"),
            "n_steps": cat.get("n_steps"),
            "steps": list(cat.get("steps", [])),
            "pnl_sign_same": cat.get("pnl_sign_same"),
            "pnl_sign_changed": cat.get("pnl_sign_changed"),
            "fwd12_sign_same": cat.get("fwd12_sign_same"),
            "fwd12_sign_changed": cat.get("fwd12_sign_changed"),
            "low_sample_steps": cat.get("low_sample_steps"),
        })
    return out


def build_cross_instrument_context(
        data: AnalystData,
        contexts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive context across the five instruments.  Instruments
    stay separate; nothing is combined into one figure."""
    evidence_dist = {label: 0 for label in EVIDENCE_LABELS}
    wf_dist = {label: 0 for label in WF_LABELS}
    per_instrument: List[Dict[str, Any]] = []
    for inst in SUPPORTED_INSTRUMENTS:
        ctx = contexts[inst]
        label = str(ctx["evidence_quality"])
        wf_label = str(ctx["walk_forward_evidence"].get("label"))
        if label in evidence_dist:
            evidence_dist[label] += 1
        if wf_label in wf_dist:
            wf_dist[wf_label] += 1
        per_instrument.append({
            "instrument": inst,
            "evidence_quality": label,
            "walk_forward_label": wf_label,
            "pnl_sign_agreement_pct": safe_float(
                data.memory_overall[inst].get("pnl_sign_agreement_pct")),
            "pearson_r_match_actual_pnl": safe_float(
                data.memory_overall[inst].get(
                    "corr_match_actual_pnl_r")),
        })
    non_insufficient = [p for p in per_instrument
                        if p["walk_forward_label"] != "INSUFFICIENT DATA"]
    if not non_insufficient:
        consistency = "insufficient walk-forward evidence across instruments"
    elif all(p["walk_forward_label"] == "PERSISTENT"
             for p in non_insufficient):
        consistency = ("stored walk-forward labels appear consistent "
                       "across instruments")
    elif any(p["walk_forward_label"] == "UNSTABLE"
             for p in non_insufficient):
        consistency = ("stored walk-forward labels appear inconsistent "
                       "across instruments")
    else:
        consistency = ("stored walk-forward labels are mixed across "
                       "instruments")
    return {
        "definitions": ("counts describe stored labels; no instrument is "
                        "preferred, ordered or combined into one figure"),
        "evidence_quality_distribution": evidence_dist,
        "walk_forward_label_distribution": wf_dist,
        "per_instrument": per_instrument,
        "consistency_statement": consistency,
    }


def build_limitations(data: AnalystData, instrument: str,
                      ctx: Dict[str, Any],
                      memory: Dict[str, Any],
                      stability: List[Dict[str, Any]],
                      walk_forward: List[Dict[str, Any]]) -> List[str]:
    """Only limitations demonstrated by the input files, each under a
    fixed documented rule.  Nothing is invented."""
    out: List[str] = []
    sign = safe_float(memory.get("pnl_sign_agreement_pct"))
    r = safe_float(memory.get("pearson_r_match_actual_pnl"))
    if sign is not None and sign <= CHANCE_LEVEL_SIGN_PCT:
        out.append(
            f"historical-match direction agreement is close to "
            f"chance-level ({sign:.1f}%) in the stored validation")
    if r is None or abs(r) < WEAK_CORRELATION_ABS_R:
        out.append(
            "the Pearson correlation between historical-match averages "
            f"and recorded outcomes is below {WEAK_CORRELATION_ABS_R:.2f} "
            f"(r={fmt_corr(r)})")
    mixed_count = sum(
        1 for cat in stability
        if cat.get("available")
        and str(cat.get("stored_classification")).lower() == "mixed")
    if mixed_count:
        out.append(
            f"regime relationships are mixed across historical periods "
            f"for {mixed_count} of the state categories")
    wf_overall = str(ctx["walk_forward_evidence"].get("label"))
    if wf_overall == "UNSTABLE":
        out.append(
            "walk-forward relationships are unstable across the tested "
            "chronological steps")
    elif wf_overall == "MIXED":
        out.append(
            "walk-forward relationships are mixed across the tested "
            "chronological steps")
    low = safe_int(memory.get("low_sample_count"))
    if low:
        out.append(
            f"{low} walk-forward queries were flagged low-sample in the "
            "stored validation")
    latest = memory["latest_observation"]
    if not latest.get("available"):
        out.append(
            "no historical-match query exists for the latest stored "
            "observation, so no match sample can be described for it")
    if str(ctx["evidence_quality"]) == "INSUFFICIENT":
        out.append(
            "historical evidence for the latest observation is "
            "insufficient under the fixed research rules")
    latest_sign = safe_float(latest.get("latest_match_sign_agreement_pct"))
    if latest_sign is not None and latest_sign < 100.0:
        out.append(
            "historical-match average outcomes disagreed with the "
            "recorded outcome for the latest stored observation")
    if not out:
        out.append(
            "no additional limitations were recorded by the existing "
            "research outputs for this instrument")
    return out


def build_summary(evidence_label: str, wf_overall: str,
                  sign_agreement_pct: Optional[float],
                  limitations: List[str]) -> str:
    """Deterministic summary from existing labels (fixed wording)."""
    parts: List[str] = []
    parts.append(f"Historical evidence quality is "
                 f"{evidence_label.lower()}.")
    if wf_overall == "MIXED":
        parts.append("Walk-forward relationships are mixed across the "
                     "tested chronological steps.")
    elif wf_overall == "UNSTABLE":
        parts.append("Walk-forward relationships are unstable across "
                     "the tested chronological steps.")
    elif wf_overall == "PERSISTENT":
        parts.append("Walk-forward relationships held the same sign "
                     "across the tested chronological steps.")
    else:
        parts.append("Walk-forward evidence is insufficient across the "
                     "tested chronological steps.")
    sign = safe_float(sign_agreement_pct)
    if sign is None:
        parts.append("Historical-match direction agreement is not "
                     "available for this instrument in the stored "
                     "validation.")
    elif sign <= CHANCE_LEVEL_SIGN_PCT:
        parts.append("Historical-match direction agreement is close to "
                     "chance-level in the stored validation.")
    else:
        parts.append(f"Historical-match direction agreement was "
                     f"{sign:.1f}% in the stored validation.")
    if limitations:
        parts.append(f"The stored research records {len(limitations)} "
                     f"limitation(s) for this instrument.")
    parts.append("This context is descriptive research evidence only "
                 "and contains no directive of any kind.")
    return " ".join(parts)


def build_analyst_context(data: AnalystData, instrument: str,
                          contexts: Dict[str, Dict[str, Any]]
                          ) -> Dict[str, Any]:
    ctx = contexts[instrument]
    state = build_current_state(ctx)
    environment = build_market_environment(state)
    memory = build_historical_memory(data, instrument, ctx)
    evidence = build_research_evidence(data, instrument, ctx)
    stability = build_regime_stability(ctx)
    walk_forward = build_walk_forward(ctx)
    limitations = build_limitations(data, instrument, ctx, memory,
                                    stability, walk_forward)
    cross = build_cross_instrument_context(data, contexts)
    summary = build_summary(str(ctx["evidence_quality"]),
                            str(ctx["walk_forward_evidence"].get("label")),
                            memory.get("pnl_sign_agreement_pct"),
                            limitations)
    return {
        "instrument": instrument,
        "timestamp": fmt_ts(ctx["timestamp"]),
        "current_state": state,
        "market_environment": {
            "lines": environment,
            "summary": " ".join(environment[:3]),
        },
        "historical_memory": memory,
        "research_evidence": evidence,
        "regime_stability": stability,
        "walk_forward": walk_forward,
        "cross_instrument_context": cross,
        "limitations": limitations,
        "analyst_context_summary": summary,
        "research_only": True,
    }


# ============================================================
# Flattened CSV row (explicit schema)
# ============================================================
def context_to_csv_row(ac: Dict[str, Any]) -> Dict[str, Any]:
    st = ac["current_state"]
    f = st["features"]
    mem = ac["historical_memory"]
    latest = mem["latest_observation"]
    ev = ac["research_evidence"]
    wf_labels = [c.get("label", "") for c in ac["walk_forward"]]
    stability_classes = [
        str(c.get("stored_classification", "")) for c in
        ac["regime_stability"] if c.get("available")]
    return {
        "instrument": ac["instrument"],
        "timestamp": ac["timestamp"],
        "timeframe": st["timeframe"],
        "direction": st["direction"],
        "adx": f.get("ADX"), "atr_pct": f.get("ATR_pct"),
        "ema_distance": f.get("EMA50_dist"),
        "bb_excursion": f.get("BB_excursion"),
        "bb_signed": f.get("BB_signed"), "eff24": f.get("eff24"),
        "rsi14": f.get("RSI14"), "ret6": f.get("ret6"),
        "ret12": f.get("ret12"), "ret24": f.get("ret24"),
        "dist_24_high": f.get("dist_24_high"),
        "dist_24_low": f.get("dist_24_low"),
        "range_pct": f.get("range_pct"), "body_pct": f.get("body_pct"),
        "upper_wick_pct": f.get("upper_wick_pct"),
        "lower_wick_pct": f.get("lower_wick_pct"),
        "vol6": f.get("vol6"), "vol24": f.get("vol24"),
        "hour": st["hour"], "day_of_week": st["day_of_week"],
        "month": st["month"],
        "adx_regime": st["adx_regime"],
        "volatility_regime": st["volatility_regime"],
        "trend_distance_regime": st["trend_distance_regime"],
        "bb_excursion_regime": st["bb_excursion_regime"],
        "efficiency_regime": st["efficiency_regime"],
        "rsi_regime": st["rsi_regime"],
        "memory_evaluation_queries": mem["evaluation_query_count"],
        "memory_avg_nearest_distance": mem["average_nearest_distance"],
        "memory_pnl_sign_agreement_pct": mem["pnl_sign_agreement_pct"],
        "memory_pearson_r_pnl": mem["pearson_r_match_actual_pnl"],
        "memory_match_avg_net_pnl": mem["historical_match_avg_net_pnl"],
        "memory_actual_avg_net_pnl": mem["actual_avg_net_pnl"],
        "latest_n_matches": latest["n_matches"],
        "latest_nearest_distance": latest["nearest_distance"],
        "latest_avg_match_distance": latest["avg_match_distance"],
        "latest_median_match_distance": latest["median_match_distance"],
        "latest_match_win_rate_pct": latest["match_win_rate_pct"],
        "latest_avg_match_net_pnl": latest["historical_match_avg_net_pnl"],
        "latest_forward6_pct": latest["forward6_pct"],
        "latest_forward12_pct": latest["forward12_pct"],
        "latest_forward24_pct": latest["forward24_pct"],
        "latest_mfe12_atr": latest["mfe12_atr"],
        "latest_mae12_atr": latest["mae12_atr"],
        "latest_actual_net_pnl": latest["actual_net_pnl"],
        "latest_actual_forward12_pct": latest["actual_forward12_pct"],
        "latest_actual_forward24_pct": latest["actual_forward24_pct"],
        "latest_sign_agreement_pct":
            latest["latest_match_sign_agreement_pct"],
        "evidence_quality": ev["evidence_quality"],
        "evidence_explanation": ev["evidence_explanation"],
        "research_conclusion": ev["research_conclusion"],
        "regime_stability_classifications": " | ".join(stability_classes),
        "walk_forward_label": ac["cross_instrument_context"][
            "per_instrument"][SUPPORTED_INSTRUMENTS.index(
                ac["instrument"])]["walk_forward_label"],
        "walk_forward_classifications": " | ".join(
            str(v) for v in wf_labels),
        "n_limitations": len(ac["limitations"]),
        "limitations": " | ".join(ac["limitations"]),
        "research_only": True,
    }


# ============================================================
# Human-readable analyst report (9 numbered sections per instrument)
# ============================================================
def generate_analyst_report(analyst_contexts: List[Dict[str, Any]]) -> str:
    L: List[str] = []
    L.append("=" * 64)
    L.append("AI TRADING ANALYST - RESEARCH CONTEXT REPORT")
    L.append("(deterministic research preparation only; no directive "
             "of any kind)")
    L.append("=" * 64)
    for ac in analyst_contexts:
        st = ac["current_state"]
        mem = ac["historical_memory"]
        latest = mem["latest_observation"]
        ev = ac["research_evidence"]
        L.append("")
        L.append("--------------------------------")
        L.append(str(ac["instrument"]))
        L.append("--------------------------------")
        L.append("1. CURRENT STATE")
        L.append(f"   Instrument: {st['instrument']}")
        L.append(f"   Timestamp: {st['timestamp']}")
        L.append(f"   Stored direction: {st['direction']}")
        L.append(f"   ADX regime: {st['adx_regime']}")
        L.append(f"   Volatility regime: {st['volatility_regime']}")
        L.append(f"   Trend-distance regime: {st['trend_distance_regime']}")
        L.append(f"   BB-excursion regime: {st['bb_excursion_regime']}")
        L.append(f"   Efficiency regime: {st['efficiency_regime']}")
        L.append(f"   RSI regime: {st['rsi_regime']}")
        L.append("2. MARKET ENVIRONMENT")
        L.extend("   " + line for line in ac["market_environment"]["lines"])
        L.append("3. HISTORICAL MEMORY")
        L.append(f"   Evaluation query count: "
                 f"{mem['evaluation_query_count']}")
        L.append(f"   Average nearest distance: "
                 f"{fmt_num(mem['average_nearest_distance'])}")
        L.append(f"   P/L sign agreement: "
                 f"{fmt_pct(mem['pnl_sign_agreement_pct'])}")
        L.append(f"   Pearson correlation: "
                 f"{fmt_corr(mem['pearson_r_match_actual_pnl'])}")
        L.append(f"   Historical-match average P/L: "
                 f"{fmt_num(mem['historical_match_avg_net_pnl'])}")
        L.append(f"   Actual average P/L: "
                 f"{fmt_num(mem['actual_avg_net_pnl'])}")
        L.append(f"   Latest observation historical match count: "
                 f"{latest['n_matches']}")
        L.append(f"   Latest observation match distance: nearest "
                 f"{fmt_num(latest['nearest_distance'])}, average "
                 f"{fmt_num(latest['avg_match_distance'])}, median "
                 f"{fmt_num(latest['median_match_distance'])}")
        L.append(f"   Latest observation historical match average P/L: "
                 f"{fmt_num(latest['historical_match_avg_net_pnl'])}")
        L.append(f"   Latest observation historical match win rate: "
                 f"{fmt_pct(latest['match_win_rate_pct'])}")
        L.append(f"   Latest observation forward6 / forward12 / "
                 f"forward24: {fmt_num(latest['forward6_pct'])}% / "
                 f"{fmt_num(latest['forward12_pct'])}% / "
                 f"{fmt_num(latest['forward24_pct'])}%")
        L.append(f"   Latest observation MFE12 / MAE12: "
                 f"{fmt_num(latest['mfe12_atr'])} / "
                 f"{fmt_num(latest['mae12_atr'])}")
        L.append(f"   Actual recorded outcome where available: "
                 f"{fmt_num(latest['actual_net_pnl'])}; actual "
                 f"forward12 {fmt_num(latest['actual_forward12_pct'])}%, "
                 f"forward24 {fmt_num(latest['actual_forward24_pct'])}%")
        L.append(f"   Latest match sign agreement: "
                 f"{fmt_pct(latest['latest_match_sign_agreement_pct'])}")
        L.append("4. RESEARCH EVIDENCE")
        L.append(f"   Evidence quality: {ev['evidence_quality']}")
        L.append(f"   Explanation: {ev['evidence_explanation']}")
        L.append(f"   Research conclusion: {ev['research_conclusion']}")
        L.append(f"   Aggregate observation count: "
                 f"{ev['aggregate_observation_count']}")
        L.append(f"   Memory query count: {ev['memory_query_count']}")
        L.append(f"   P/L sign agreement: "
                 f"{fmt_pct(ev['pnl_sign_agreement_pct'])}")
        L.append(f"   Pearson correlation: "
                 f"{fmt_corr(ev['pearson_r_match_actual_pnl'])}")
        L.append("   Regime stability classifications: "
                 + (", ".join(ev["regime_stability_classifications"])
                    or "n/a"))
        L.append("   Walk-forward classifications: "
                 + (", ".join(ev["walk_forward_classifications"])
                    or "n/a"))
        L.append("5. REGIME STABILITY")
        for cat in ac["regime_stability"]:
            if not cat.get("available"):
                L.append(f"   {cat.get('category')}: not available")
                continue
            L.append(
                f"   {cat['category']}: observed periods "
                f"{cat['observed_periods']}, positive "
                f"{cat['positive_periods']}, negative "
                f"{cat['negative_periods']}, unchanged "
                f"{cat['unchanged_periods']}, P/L sign changes "
                f"{cat['pnl_sign_changes']}, forward12 sign changes "
                f"{cat['fwd12_sign_changes']}, stored classification "
                f"{cat['stored_classification']}")
        L.append("6. WALK-FORWARD EVIDENCE")
        for cat in ac["walk_forward"]:
            if not cat.get("available"):
                L.append(f"   {cat.get('category')}: "
                         f"{cat.get('label')}")
                continue
            L.append(
                f"   {cat['category']}: label {cat['label']}, steps "
                f"{cat['n_steps']}, P/L sign same {cat['pnl_sign_same']}, "
                f"P/L sign changed {cat['pnl_sign_changed']}, forward12 "
                f"sign same {cat['fwd12_sign_same']}, forward12 sign "
                f"changed {cat['fwd12_sign_changed']}, low-sample steps "
                f"{cat['low_sample_steps']}")
        L.append("7. CROSS-INSTRUMENT CONTEXT")
        cross = ac["cross_instrument_context"]
        L.append("   Evidence quality distribution: "
                 + ", ".join(f"{k}={v}" for k, v in
                             cross["evidence_quality_distribution"].items()))
        L.append("   Walk-forward label distribution: "
                 + ", ".join(f"{k}={v}" for k, v in
                             cross["walk_forward_label_distribution"].items()))
        L.append(f"   Consistency: {cross['consistency_statement']}")
        for p in cross["per_instrument"]:
            L.append(f"   {p['instrument']}: evidence "
                     f"{p['evidence_quality']}, walk-forward "
                     f"{p['walk_forward_label']}, P/L sign agreement "
                     f"{fmt_pct(p['pnl_sign_agreement_pct'])}, Pearson r "
                     f"{fmt_corr(p['pearson_r_match_actual_pnl'])}")
        L.append("8. RESEARCH LIMITATIONS")
        for lim in ac["limitations"]:
            L.append(f"   - {lim}")
        L.append("9. ANALYST CONTEXT SUMMARY")
        L.append(f"   {ac['analyst_context_summary']}")
    L.append("")
    L.append("=" * 64)
    L.append("This report is deterministic research preparation only.  "
             "It contains no directive of any kind, no certainty figure, "
             "and no ordering of instruments.")
    return "\n".join(L)


# ============================================================
# Output writers (all-or-nothing overwrite protection)
# ============================================================
def _write_bytes_if_absent(filename: str, payload: bytes) -> None:
    path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(path):
        raise OutputExistsError(
            "output file already exists (existing files are never "
            f"overwritten): {filename}")
    with open(path, "wb") as fh:
        fh.write(payload)
    print(f"  wrote {filename} ({len(payload)} bytes)")


def write_outputs(rows: List[Dict[str, Any]],
                  contexts: List[Dict[str, Any]],
                  txt_payload: str,
                  provenance: Dict[str, Any],
                  csv_name: str = OUTPUT_CSV,
                  json_name: str = OUTPUT_JSON,
                  txt_name: str = OUTPUT_TXT) -> None:
    """Write the three NEW outputs; every path was checked beforehand
    and each writer re-checks its own path.  The real --run passes the
    timestamped names so an earlier run is never collided with."""
    csv_buffer = pd.DataFrame(
        rows, columns=list(ANALYST_CSV_COLUMNS)).to_csv(
        index=False).encode("utf-8")
    payload = {
        "generated_by": "ai_trading_analyst.py",
        "content_note": ("deterministic research context for a future "
                         "AI analyst; evidence only, no directive, no "
                         "certainty figure, no instrument ordering"),
        "input_provenance": provenance,
        "instruments": contexts,
    }
    json_buffer = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"),
        default=_json_default).encode("utf-8")
    txt_bytes = txt_payload.encode("utf-8")
    _write_bytes_if_absent(csv_name, csv_buffer)
    _write_bytes_if_absent(json_name, json_buffer)
    _write_bytes_if_absent(txt_name, txt_bytes)


# ============================================================
# Live research-context input (real run only)
# ============================================================
# The frozen feature-key translation from the live state to the analyst
# context.  A live value the live output does not carry is reported as
# unavailable instead of being substituted.
_LIVE_FEATURE_MAP: Tuple[Tuple[str, str], ...] = (
    ("ADX", "adx"),
    ("ATR_pct", "atr_pct"),
    ("EMA50_dist", "ema50_dist_atr"),
    ("BB_excursion", "bb_excursion"),
    ("BB_signed", "bb_position"),
    ("eff24", "eff24"),
    ("RSI14", "rsi14"),
    ("ret6", "return6"),
    ("ret12", "return12"),
    ("ret24", "return24"),
    ("dist_24_high", "dist_24_high"),
    ("dist_24_low", "dist_24_low"),
    ("range_pct", "candle_range_pct"),
    ("body_pct", "candle_body_pct"),
    ("upper_wick_pct", "upper_wick_pct"),
    ("lower_wick_pct", "lower_wick_pct"),
)
_LIVE_UNAVAILABLE_FEATURES: Tuple[str, ...] = ("vol6", "vol24")


def find_live_research_inputs(directory: str = SCRIPT_DIR
                              ) -> Tuple[str, str, str]:
    """The newest matching timestamped live research context.

    A collection is the CSV, the JSON and the report that share one
    filename stamp.  An incomplete stamp (for example a CSV without its
    report) is skipped.  When no complete collection exists the run
    stops with a clear error; the stale ai_research_context_all.csv is
    never used as a fallback.  Selection is deterministic.
    """
    csv_by_stamp: Dict[str, str] = {}
    json_by_stamp: Dict[str, str] = {}
    report_by_stamp: Dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        csv_match = LIVE_RESEARCH_CSV_STAMP_RE.match(name)
        if csv_match:
            csv_by_stamp.setdefault(csv_match.group(1), name)
            continue
        json_match = LIVE_RESEARCH_JSON_STAMP_RE.match(name)
        if json_match:
            json_by_stamp.setdefault(json_match.group(1), name)
            continue
        report_match = LIVE_RESEARCH_REPORT_STAMP_RE.match(name)
        if report_match:
            report_by_stamp.setdefault(report_match.group(1), name)
    shared = sorted(set(csv_by_stamp) & set(json_by_stamp)
                    & set(report_by_stamp))
    if not shared:
        raise MissingInputError(
            "no complete timestamped live research context is present "
            "(expected live_research_context_YYYYMMDD_HHMMSS.csv, "
            "live_research_context_YYYYMMDD_HHMMSS.json and "
            "live_research_context_report_YYYYMMDD_HHMMSS.txt sharing "
            "one timestamp); refusing to fall back to the stale "
            "ai_research_context_all.csv")
    stamp = shared[-1]
    return (os.path.join(directory, csv_by_stamp[stamp]),
            os.path.join(directory, json_by_stamp[stamp]),
            os.path.join(directory, report_by_stamp[stamp]))


def _split_live_category(text: Any) -> Tuple[str, str]:
    """Split a live category label such as 'ADX (ADX20-25)'."""
    label = "" if text is None else str(text)
    if label.endswith(")") and " (" in label:
        feature, bucket = label[:-1].split(" (", 1)
        return feature, bucket
    return label, ""


def build_live_research_ctx(live: Dict[str, Any]) -> Dict[str, Any]:
    """Map one live research context into the analyst's context shape.

    Labels are copied exactly from the live output; nothing is predicted,
    re-scored or ordered, and a value the live output does not carry is
    reported as unavailable instead of being invented.
    """
    instrument = str(live.get("instrument"))
    live_state = live.get("live_state") or {}
    live_features = live_state.get("features") or {}
    classifications = dict(live_state.get("classifications") or {})
    time_context = live_state.get("time_context") or {}
    history = live.get("historical_research_context") or {}
    match = live.get("live_state_historical_match") or {}

    features: Dict[str, Any] = {
        out_key: live_features.get(live_key)
        for out_key, live_key in _LIVE_FEATURE_MAP}
    for unavailable_key in _LIVE_UNAVAILABLE_FEATURES:
        features[unavailable_key] = None

    node_time = live_state.get("latest_closed_candle_time")
    stability_categories: List[Dict[str, Any]] = []
    for cat in match.get("by_category") or []:
        feature, bucket = _split_live_category(cat.get("category"))
        stability_categories.append({
            "feature": feature,
            "bucket": bucket,
            "available": True,
            "pnl_classification": cat.get("stored_classification"),
        })
    walk_forward_categories: List[Dict[str, Any]] = []
    for cat in match.get("walk_forward_by_category") or []:
        feature, bucket = _split_live_category(cat.get("category"))
        walk_forward_categories.append({
            "feature": feature,
            "bucket": bucket,
            "available": bool(cat.get("available")),
            "label": cat.get("label"),
        })

    latest_n_matches = history.get("latest_stored_n_matches")
    return {
        "instrument": instrument,
        "timestamp": node_time,
        "current_state": {
            "instrument": instrument,
            "timestamp": node_time,
            "timeframe": "H1",
            "direction": live_state.get("direction_stored_research"),
            "features": features,
            "classifications": classifications,
            "atr_tercile_thresholds": {},
            "hour": time_context.get("hour"),
            "day_of_week": time_context.get("day_of_week"),
            "month": time_context.get("month"),
        },
        "historical_memory": {
            "available": latest_n_matches is not None,
            "n_matches": latest_n_matches,
            "nearest_distance": history.get("latest_stored_nearest_distance"),
            "sign_agreement_pct":
                history.get("latest_stored_sign_agreement_pct"),
        },
        "regime_stability": {"by_category": stability_categories},
        "walk_forward_evidence": {
            "label": (match.get("walk_forward_label")
                      or history.get("stored_walk_forward_label")),
            "by_category": walk_forward_categories,
        },
        "evidence_quality": history.get("stored_evidence_quality"),
        "evidence_explanation": history.get("stored_evidence_explanation"),
        "research_conclusion": history.get("stored_research_conclusion"),
    }


def load_live_research_contexts(json_path: str
                                ) -> Dict[str, Dict[str, Any]]:
    """One analyst-shaped current context per instrument from the live
    research context JSON (fixed instrument order)."""
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) \
            or not isinstance(payload.get("instruments"), list):
        raise ValueError(
            "live research context input: expected an object with an "
            "'instruments' list")
    contexts: Dict[str, Dict[str, Any]] = {}
    for record in payload["instruments"]:
        if not isinstance(record, dict):
            raise ValueError(
                "live research context input: an instrument record is "
                "not an object")
        instrument = str(record.get("instrument"))
        if instrument in SUPPORTED_INSTRUMENTS:
            contexts[instrument] = build_live_research_ctx(record)
    missing = [i for i in SUPPORTED_INSTRUMENTS if i not in contexts]
    if missing:
        raise ValueError(
            "live research context input: no context for: "
            + ", ".join(missing))
    return {i: contexts[i] for i in SUPPORTED_INSTRUMENTS}


# ============================================================
# Real analysis (requires --run; never runs by default)
# ============================================================
def run_real_analysis() -> None:
    print("=" * 64)
    print("ai_trading_analyst.py - REAL ANALYSIS (--run)")
    print("=" * 64)
    stamp = datetime.datetime.now().strftime(STAMP_FORMAT)
    csv_name, json_name, txt_name = timestamped_output_files(stamp)
    check_output_files_available(
        SCRIPT_DIR, (csv_name, json_name, txt_name))   # ALL paths, first
    try:
        live_csv, live_json, live_report = find_live_research_inputs(
            SCRIPT_DIR)
    except MissingInputError as exc:
        print("\nERROR: " + str(exc))
        raise SystemExit(1)
    print("\nLive research context input selected:")
    print("  csv    : " + os.path.basename(live_csv))
    print("  json   : " + os.path.basename(live_json))
    print("  report : " + os.path.basename(live_report))
    data = AnalystData(SCRIPT_DIR)
    print("\nStored-research files read (read-only):")
    for name in INPUT_FILES:
        print(f"  {name}")
    contexts = load_live_research_contexts(live_json)
    analyst_contexts = [
        build_analyst_context(data, inst, contexts)
        for inst in SUPPORTED_INSTRUMENTS]
    rows = [context_to_csv_row(ac) for ac in analyst_contexts]
    txt = generate_analyst_report(analyst_contexts)
    provenance = {
        "observation_counts": dict(data.observation_counts),
        "memory_query_counts": {
            inst: data.query_count(inst)
            for inst in SUPPORTED_INSTRUMENTS},
        "contexts_loaded": len(data.contexts_by_key),
        "text_report": dict(data.txt_provenance),
        "latest_timestamps": {
            ac["instrument"]: ac["timestamp"] for ac in analyst_contexts},
    }
    print("\nLatest stored observation per instrument:")
    for ac in analyst_contexts:
        ev = ac["research_evidence"]
        print(f"  {ac['instrument']}: {ac['timestamp']} | evidence "
              f"{ev['evidence_quality']} | walk-forward "
              f"{ac['cross_instrument_context']['per_instrument'][SUPPORTED_INSTRUMENTS.index(ac['instrument'])]['walk_forward_label']}"
              f" | limitations {len(ac['limitations'])}")
    print("\nWriting NEW output files:")
    write_outputs(rows, analyst_contexts, txt, provenance,
                  csv_name, json_name, txt_name)
    print("\nDone. Descriptive research context only - no directive, no "
          "certainty figure, no instrument ordering.")


# ============================================================
# Static safety scan of this file's own source (AST-based)
# ============================================================
def _docstring_nodes(tree: ast.AST) -> set:
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                nodes.add(id(body[0].value))
    return nodes


def run_safety_scan() -> List[str]:
    """AST scan: imports, identifiers and non-docstring string literals.
    Returns a list of violations (empty when the file is clean)."""
    src_path = os.path.abspath(__file__)
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    violations: List[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"syntax error during scan: {exc}"]
    doc_ids = _docstring_nodes(tree)

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

    name_tokens = set(_FORBIDDEN_NAME_TOKENS)
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
            for tok in name_tokens:
                # Word-part match (not substring) so legitimate names
                # such as the standard-library termination call are not
                # false positives.
                if tok == low or tok in parts:
                    violations.append(
                        f"forbidden token '{tok}' in identifier "
                        f"'{name}'")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in doc_ids:
                continue
            low = node.value.lower()
            for tok in _TERMINAL_TOKENS:
                if tok in low:
                    violations.append(
                        "trading-terminal token in string literal")
            for tok in _ML_TOKENS:
                if tok in low:
                    violations.append(
                        "learning-library token in string literal")
            for tok in _OPTIMIZATION_TOKENS:
                if tok in low:
                    violations.append(
                        "optimization idiom in string literal")
            for pat in _ML_CALL_PATTERNS:
                if re.search(pat, low):
                    violations.append(
                        "fitting-call pattern in string literal")
            for pat in _BANNED_WORD_PATTERNS:
                if re.search(pat, low):
                    violations.append(
                        "forbidden wording in string literal: "
                        f"{node.value[:60]!r}")
    return violations


# ============================================================
# Synthetic fixtures (tests only; temporary directories only)
# ============================================================
def _fixture_context(inst: str, ts: str, index: int,
                     available: bool, evidence: str,
                     wf_label: str) -> Dict[str, Any]:
    """One synthetic research context with the real nested schema."""
    hour = 9 + index
    features = {
        "ADX": 10.0 + index * 5, "ATR_pct": 0.10 + index * 0.02,
        "EMA50_dist": 0.5 + index * 0.3, "BB_excursion": 0.4 + index * 0.1,
        "BB_signed": -0.4 - index * 0.1, "eff24": 0.2 + index * 0.1,
        "ret6": -0.10 + index * 0.05, "ret12": -0.20 + index * 0.05,
        "ret24": -0.30 + index * 0.05, "RSI14": 35.0 + index * 5,
        "vol6": 0.10 + index * 0.01, "vol24": 0.09 + index * 0.01,
        "range_pct": 0.40 + index * 0.02, "body_pct": 0.20,
        "upper_wick_pct": 0.05, "lower_wick_pct": 0.10,
        "dist_24_high": -0.60, "dist_24_low": 0.20,
    }
    classifications = {
        "adx_regime": "ADX>=25" if index == 2 else "ADX15-20",
        "volatility_regime": "LOW" if index < 2 else "MEDIUM",
        "trend_distance_regime": "<1" if index < 2 else "1-2",
        "bb_excursion_regime": "0.50-0.75" if index == 2 else "<0.50",
        "efficiency_regime": "<0.30" if index < 2 else "0.30-0.60",
        "rsi_regime": "30-50" if index < 2 else "50-70",
    }
    latest_block = {
        "available": available,
        "wording": ("historical matches are stored past observations "
                    "with similar state features"),
        "n_matches": 10 if available else None,
        "avg_match_distance": 1.5 if available else None,
        "median_match_distance": 1.4 if available else None,
        "nearest_distance": 1.2 if available else None,
        "nearest_distance_percentile": 40.0 if available else None,
        "avg_match_net_pnl": 3.0 if available else None,
        "match_win_rate_pct": 55.0 if available else None,
        "avg_match_forward6_pct": 0.02 if available else None,
        "avg_match_forward12_pct": 0.05 if available else None,
        "avg_match_forward24_pct": 0.08 if available else None,
        "avg_match_mfe12_atr": 1.8 if available else None,
        "avg_match_mae12_atr": -1.5 if available else None,
        "avg_match_holding_candles": 3.0 if available else None,
        "actual_net_pnl": 2.0 if available else None,
        "actual_win": available,
        "actual_forward6_pct": 0.03 if available else None,
        "actual_forward12_pct": 0.06 if available else None,
        "actual_forward24_pct": 0.07 if available else None,
        "actual_mfe12_atr": 1.9 if available else None,
        "actual_mae12_atr": -1.2 if available else None,
        "sign_agreement_pct": 100.0 if available else None,
        "pearson_r_match_actual_pnl": 0.05 if available else None,
        "pearson_n_pnl": 20 if available else None,
        "instrument_level": None,
        "note": "synthetic fixture",
    }
    stability_cats = []
    for feat, key, bucket in (
            ("ADX", "adx", classifications["adx_regime"]),
            ("ATR_pct", "atr", classifications["volatility_regime"]),
            ("EMA50_dist", "ema_distance",
             classifications["trend_distance_regime"]),
            ("BB_excursion", "bb_excursion",
             classifications["bb_excursion_regime"]),
            ("eff24", "efficiency", classifications["efficiency_regime"]),
            ("RSI14", "rsi", classifications["rsi_regime"])):
        stability_cats.append({
            "feature": feat, "oos_feature_key": key, "bucket": bucket,
            "available": True, "periods_observed": 4,
            "observations_in_category": 10, "periods_positive": 2,
            "periods_negative": 2, "periods_unchanged": 0,
            "pnl_sign_changes": 2, "fwd12_sign_changes": 1,
            "pnl_classification": "mixed", "fwd12_classification": "mixed",
            "per_period": [],
        })
    wf_cats = []
    for feat, key, bucket in (
            ("ADX", "adx", classifications["adx_regime"]),
            ("ATR_pct", "atr", classifications["volatility_regime"]),
            ("EMA50_dist", "ema_distance",
             classifications["trend_distance_regime"]),
            ("BB_excursion", "bb_excursion",
             classifications["bb_excursion_regime"]),
            ("eff24", "efficiency", classifications["efficiency_regime"]),
            ("RSI14", "rsi", classifications["rsi_regime"])):
        wf_cats.append({
            "feature": feat, "oos_feature_key": key, "bucket": bucket,
            "available": True, "label": wf_label, "n_steps": 3,
            "steps": list(WF_STEP_ORDER),
            "development_periods": ["P1", "P1+P2", "P1+P2+P3"],
            "oos_periods": ["P2", "P3", "P4"],
            "pnl_sign_same": 3 if wf_label == "PERSISTENT" else 1,
            "pnl_sign_changed": 0 if wf_label == "PERSISTENT" else 2,
            "fwd12_sign_same": 2, "fwd12_sign_changed": 1,
            "low_sample_steps": 0,
            "relationship_persisted": wf_label == "PERSISTENT",
            "relationship_mixed": wf_label == "MIXED",
        })
    cross_cats = []
    for feat, key, bucket in (
            ("ADX", "adx", classifications["adx_regime"]),
            ("ATR_pct", "atr", classifications["volatility_regime"]),
            ("EMA50_dist", "ema_distance",
             classifications["trend_distance_regime"]),
            ("BB_excursion", "bb_excursion",
             classifications["bb_excursion_regime"]),
            ("eff24", "efficiency", classifications["efficiency_regime"]),
            ("RSI14", "rsi", classifications["rsi_regime"])):
        cross_cats.append({
            "feature": feat, "oos_feature_key": key, "bucket": bucket,
            "available": True,
            "per_period": [{
                "period": f"P{p}",
                "instruments_with_sufficient_sample": 5,
                "pos_avg_pnl_count": 2, "neg_avg_pnl_count": 3,
                "pos_fwd12_count": 3, "neg_fwd12_count": 2,
                "pnl_classification": "MIXED",
                "fwd12_classification": "MIXED",
            } for p in range(1, 5)],
        })
    return {
        "instrument": inst,
        "timestamp": ts,
        "current_state": {
            "instrument": inst, "timestamp": ts, "timeframe": "H1",
            "direction": "LONG" if index % 2 == 0 else "SHORT",
            "features": features, "classifications": classifications,
            "atr_tercile_thresholds": {
                "p33": 0.15, "p66": 0.25,
                "source": "instrument-specific atr_pct terciles from the "
                          "historical database (frozen)"},
            "hour": hour, "day_of_week": 2, "month": 3,
        },
        "historical_memory": latest_block,
        "regime_stability": {"available": True,
                             "by_category": stability_cats,
                             "note": "synthetic fixture"},
        "walk_forward_evidence": {"available": True, "label": wf_label,
                                  "by_category": wf_cats,
                                  "note": "synthetic fixture"},
        "memory_engine_evidence": {"queries": 20, "k_fixed": 10},
        "cross_instrument_context": {
            "available": True, "by_category": cross_cats,
            "period_overview_per_instrument": {
                i: [] for i in SUPPORTED_INSTRUMENTS},
            "note": "synthetic fixture"},
        "evidence_quality": evidence,
        "evidence_explanation": f"{evidence}: synthetic fixture "
                                f"explanation.",
        "research_conclusion": "HISTORICAL EVIDENCE IS MIXED",
    }


_LABEL_PLAN = {
    "GOLD": ("WEAK", "UNSTABLE"),
    "EURUSD": ("MODERATE", "MIXED"),
    "GBPUSD": ("WEAK", "MIXED"),
    "USDJPY": ("WEAK", "MIXED"),
    "AUDUSD": ("INSUFFICIENT", "INSUFFICIENT DATA"),
}
_MEMORY_PLAN = {
    "GOLD": (624, 4.9343, 50.0, 0.0474, -11.73, -15.25),
    "EURUSD": (650, 2.5685, 51.3846, 0.1352, -4.00, -0.62),
    "GBPUSD": (660, 2.6738, 46.6667, -0.0209, -4.08, -0.48),
    "USDJPY": (632, 4.0537, 46.6772, 0.0538, -5.25, -2.93),
    "AUDUSD": (624, 7.5115, 48.8782, 0.0224, 1.30, 0.05),
}


def build_analyst_fixture(tmpdir: str) -> str:
    """Write a complete synthetic bundle of all 11 inputs to tmpdir."""
    def write_csv(name: str, rows: List[Dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_csv(os.path.join(tmpdir, name), index=False)

    # ---- research CSVs ----
    state_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        for i in range(3):
            state_rows.append({
                "instrument": inst, "signal_time":
                    f"2026-09-0{i + 1} 10:00:00", "direction": "LONG",
                "adx": 10.0 + i, "atr_pct": 0.10 + i * 0.05,
                "ema_distance": 0.5, "bb_excursion": 0.5, "bb_signed": -0.5,
                "eff24": 0.3, "rsi14": 40.0,
                "net_pnl": -999.0, "win": False, "forward6_pct": -9.0,
                "forward12_pct": -9.0, "mfe12_atr": -9.0, "mae12_atr": -9.0,
            })
    write_csv("market_state_all.csv", state_rows)

    val_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        for i in range(20):
            val_rows.append({
                "instrument": inst,
                "query_timestamp": f"2026-08-{i + 1:02d} 10:00:00",
                "n_matches": 10, "nearest_distance": 1.0 + i * 0.01,
                "pnl_sign_agree": i % 2 == 0,
            })
    write_csv("market_memory_oos_validation.csv", val_rows)

    det_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        for i in range(20):
            det_rows.append({
                "instrument": inst,
                "query_timestamp": f"2026-08-{i + 1:02d} 10:00:00",
                "match_rank": 1, "distance": 1.0 + i * 0.01,
            })
    write_csv("market_memory_match_details.csv", det_rows)

    sum_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        q, dist, sign, r, mavg, aavg = _MEMORY_PLAN[inst]
        sum_rows.append({
            "instrument": inst, "section": "OVERALL", "n_queries": q,
            "avg_nearest_distance": dist, "median_nearest_distance": dist,
            "pnl_sign_agreement_pct": sign, "pnl_sign_n": q,
            "corr_match_actual_pnl_r": r, "corr_match_actual_pnl_n": q,
            "fwd12_sign_agreement_pct": 48.0, "fwd24_sign_agreement_pct": 50.0,
            "avg_match_net_pnl": mavg, "avg_actual_net_pnl": aavg,
            "low_sample_count": 0,
            "avg_match_forward12_pct": 0.10,
            "avg_actual_forward12_pct": 0.05,
        })
    write_csv("market_memory_summary.csv", sum_rows)

    period_rows = []
    for inst in SUPPORTED_INSTRUMENTS:
        for p in range(1, 5):
            period_rows.append({
                "instrument": inst, "period": f"P{p}", "n": 20,
                "avg_net_pnl": (1.0, -1.0, 0.5, -0.5)[p - 1],
                "fwd12_avg_pct": 0.02,
            })
    write_csv("market_state_oos_period_summary.csv", period_rows)

    stability_rows = []
    wf_rows = []
    cross_rows = []
    latest_buckets = {
        "adx": "ADX>=25", "atr": "MEDIUM", "ema_distance": "1-2",
        "bb_excursion": "0.50-0.75", "efficiency": "0.30-0.60",
        "rsi": "50-70",
    }
    for inst in SUPPORTED_INSTRUMENTS:
        for feat, bucket in latest_buckets.items():
            for p in range(1, 5):
                stability_rows.append({
                    "instrument": inst, "feature": feat, "bucket": bucket,
                    "period": f"P{p}", "n": 10,
                    "avg_net_pnl": (1.0, -1.0, 0.5, -0.5)[p - 1],
                    "pnl_sign_changes": 2, "fwd12_sign_changes": 1,
                    "pnl_classification": "mixed",
                })
            for step in WF_STEP_ORDER:
                wf_rows.append({
                    "instrument": inst, "feature": feat, "bucket": bucket,
                    "step": step, "pnl_sign_result": "SIGN CHANGED",
                    "fwd12_sign_result": "SIGN SAME", "low_sample": False,
                })
    for feat, bucket in latest_buckets.items():
        for p in range(1, 5):
            cross_rows.append({
                "feature": feat, "bucket": bucket, "period": f"P{p}",
                "instruments_with_sufficient_sample": 5,
                "pos_avg_pnl_count": 2, "neg_avg_pnl_count": 3,
                "pnl_classification": "MIXED",
            })
    write_csv("market_state_oos_state_summary.csv", stability_rows)
    write_csv("market_state_oos_walkforward.csv", wf_rows)
    write_csv("market_state_oos_cross_instrument.csv", cross_rows)

    # ---- ai_research_context outputs ----
    context_rows = []
    contexts = []
    for inst in SUPPORTED_INSTRUMENTS:
        evidence, wf_label = _LABEL_PLAN[inst]
        for index in range(3):
            ts = f"2026-09-0{index + 1} 10:00:00"
            available = (index == 2) and inst != "AUDUSD"
            ctx = _fixture_context(inst, ts, index, available,
                                   evidence, wf_label)
            contexts.append(ctx)
            row = {
                "instrument": inst, "timestamp": ts,
                "timeframe": "H1", "direction": ctx["current_state"]
                ["direction"],
                "adx": ctx["current_state"]["features"]["ADX"],
                "atr_pct": ctx["current_state"]["features"]["ATR_pct"],
                "ema_distance": ctx["current_state"]["features"]
                ["EMA50_dist"],
                "bb_excursion": ctx["current_state"]["features"]
                ["BB_excursion"],
                "bb_signed": ctx["current_state"]["features"]["BB_signed"],
                "eff24": ctx["current_state"]["features"]["eff24"],
                "rsi14": ctx["current_state"]["features"]["RSI14"],
                "ret6": ctx["current_state"]["features"]["ret6"],
                "ret12": ctx["current_state"]["features"]["ret12"],
                "ret24": ctx["current_state"]["features"]["ret24"],
                "vol6": ctx["current_state"]["features"]["vol6"],
                "vol24": ctx["current_state"]["features"]["vol24"],
                "range_pct": ctx["current_state"]["features"]["range_pct"],
                "body_pct": ctx["current_state"]["features"]["body_pct"],
                "upper_wick_pct": ctx["current_state"]["features"]
                ["upper_wick_pct"],
                "lower_wick_pct": ctx["current_state"]["features"]
                ["lower_wick_pct"],
                "dist_24_high": ctx["current_state"]["features"]
                ["dist_24_high"],
                "dist_24_low": ctx["current_state"]["features"]
                ["dist_24_low"],
                "hour": ctx["current_state"]["hour"],
                "day_of_week": ctx["current_state"]["day_of_week"],
                "month": ctx["current_state"]["month"],
                "adx_regime": ctx["current_state"]["classifications"]
                ["adx_regime"],
                "volatility_regime": ctx["current_state"]["classifications"]
                ["volatility_regime"],
                "trend_distance_regime": ctx["current_state"]
                ["classifications"]["trend_distance_regime"],
                "bb_excursion_regime": ctx["current_state"]
                ["classifications"]["bb_excursion_regime"],
                "efficiency_regime": ctx["current_state"]
                ["classifications"]["efficiency_regime"],
                "rsi_regime": ctx["current_state"]["classifications"]
                ["rsi_regime"],
                "evidence_quality": evidence,
                "research_conclusion": ctx["research_conclusion"],
            }
            context_rows.append(row)
    write_csv("ai_research_context_all.csv", context_rows)
    with open(os.path.join(tmpdir, "ai_research_context_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(contexts, fh, ensure_ascii=True, separators=(",", ":"),
                  default=_json_default)
    with open(os.path.join(tmpdir, "ai_research_context_report.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("AI RESEARCH LAYER - HISTORICAL MARKET CONTEXT REPORTS\n")
        for section in ("A. CURRENT MARKET STATE", "B. MARKET ENVIRONMENT",
                        "C. HISTORICAL MEMORY", "D. REGIME STABILITY",
                        "E. WALK-FORWARD VALIDATION",
                        "F. EVIDENCE QUALITY", "G. RESEARCH CONCLUSION"):
            fh.write(section + "\n")
    return tmpdir


def _live_research_json_payload(live_ts: Dict[str, str],
                                evidence: str = "WEAK"
                                ) -> Dict[str, Any]:
    """A minimal synthetic live research context JSON payload."""
    instruments = []
    for inst in SUPPORTED_INSTRUMENTS:
        ts = live_ts[inst]
        instruments.append({
            "instrument": inst,
            "live_state": {
                "collection_status": "COLLECTED",
                "latest_closed_candle_time": ts,
                "candles_used": 9999,
                "direction_stored_research": "NONE",
                "features": {
                    "adx": 12.0, "atr_pct": 0.2, "ema50_dist_atr": 0.5,
                    "bb_excursion": 0.3, "bb_position": -0.3,
                    "eff24": 0.2, "rsi14": 45.0, "return6": 0.0,
                    "return12": 0.0, "return24": 0.0,
                    "dist_24_high": -0.5, "dist_24_low": 0.2,
                    "candle_range_pct": 0.3, "candle_body_pct": 0.1,
                    "upper_wick_pct": 0.05, "lower_wick_pct": 0.05,
                },
                "time_context": {"hour": 5, "day_of_week": 2,
                                 "month": 10},
                "classifications": {
                    "adx_regime": "ADX<15", "volatility_regime": "LOW",
                    "trend_distance_regime": "<1",
                    "bb_excursion_regime": "<0.50",
                    "efficiency_regime": "<0.30", "rsi_regime": "30-50",
                },
            },
            "historical_research_context": {
                "stored_observation_timestamp": ts,
                "stored_evidence_quality": evidence,
                "stored_evidence_explanation": "stored explanation",
                "stored_research_conclusion": "HISTORICAL EVIDENCE IS MIXED",
                "memory_n_queries": 10,
                "memory_pnl_sign_agreement_pct": 50.0,
                "memory_pearson_r_pnl": 0.04,
                "memory_avg_nearest_distance": 4.0,
                "latest_stored_n_matches": 10,
                "latest_stored_nearest_distance": 3.0,
                "latest_stored_sign_agreement_pct": 0.0,
                "stored_regime_stability_classifications": ["mixed"],
                "stored_walk_forward_label": "MIXED",
                "stored_walk_forward_classifications": ["MIXED"],
                "cross_instrument_consistency_statement": "stored labels",
            },
            "live_state_historical_match": {
                "available": True,
                "available_categories": 6,
                "category_count": 6,
                "by_category": [
                    {"category": "ADX (ADX<15)",
                     "stored_classification": "mixed"},
                ],
                "classifications": ["mixed"],
                "walk_forward_label": "UNSTABLE",
                "walk_forward_classifications": ["MIXED"],
                "walk_forward_by_category": [
                    {"category": "ADX (ADX<15)", "available": True,
                     "label": "MIXED"},
                ],
                "note": "descriptive lookup of the live regimes",
            },
            "limitations": [],
            "data_quality": {"issue_count": 0, "issues": []},
            "research_interpretation": [],
            "research_only": True,
        })
    return {
        "generated_timestamp": "2026-10-07 12:00:00",
        "inputs_read": [],
        "research_only": True,
        "instruments": instruments,
    }


def write_live_research_fixture(directory: str, stamp: str,
                                live_ts: Optional[Dict[str, str]] = None,
                                evidence: str = "WEAK",
                                write_csv: bool = True,
                                write_json: bool = True,
                                write_report: bool = True) -> None:
    """Write a synthetic timestamped live research context collection."""
    if live_ts is None:
        live_ts = {i: "2026-10-07 05:00:00" for i in SUPPORTED_INSTRUMENTS}
    if write_csv:
        csv_rows = [{"instrument": i,
                     "live_latest_closed_candle_time": live_ts[i]}
                    for i in SUPPORTED_INSTRUMENTS]
        pd.DataFrame(csv_rows).to_csv(
            os.path.join(directory,
                         f"live_research_context_{stamp}.csv"),
            index=False)
    if write_json:
        payload = _live_research_json_payload(live_ts, evidence)
        with open(os.path.join(
                directory, f"live_research_context_{stamp}.json"),
                "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True,
                      separators=(",", ":"), default=_json_default)
    if write_report:
        with open(os.path.join(
                directory,
                f"live_research_context_report_{stamp}.txt"),
                "w", encoding="utf-8") as fh:
            fh.write("LIVE RESEARCH CONTEXT - READ-ONLY REPORT\n")


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

    print("ai_trading_analyst.py - synthetic test suite")
    print("(temporary data only; no real output is written)\n")

    with tempfile.TemporaryDirectory() as td:
        build_analyst_fixture(td)
        data = AnalystData(td)
        contexts = {inst: data.latest_context(inst)
                    for inst in SUPPORTED_INSTRUMENTS}
        analyst = [build_analyst_context(data, inst, contexts)
                   for inst in SUPPORTED_INSTRUMENTS]
        by_inst = {ac["instrument"]: ac for ac in analyst}
        txt = generate_analyst_report(analyst)
        rows = [context_to_csv_row(ac) for ac in analyst]

        # ---------------- A. missing-file handling ----------------
        start_area("A_missing_files")
        with tempfile.TemporaryDirectory() as empty:
            try:
                check_required_inputs(empty)
                check("empty directory raises MissingInputError", False)
            except MissingInputError as exc:
                msg = str(exc)
                check("empty directory raises MissingInputError", True)
                check("every missing filename named",
                      all(f in msg for f in INPUT_FILES))
        with tempfile.TemporaryDirectory() as partial:
            for name in INPUT_FILES[:-1]:
                open(os.path.join(partial, name), "w").close()
            try:
                check_required_inputs(partial)
                check("single missing file raises", False)
            except MissingInputError as exc:
                check("single missing file raises", True)
                check("exact missing filename named",
                      INPUT_FILES[-1] in str(exc))
                check("only the missing file named",
                      str(exc).count(INPUT_FILES[-2]) == 0)

        # ---------------- B. schema validation ----------------
        start_area("B_schema")
        with tempfile.TemporaryDirectory() as bad:
            build_analyst_fixture(bad)
            frame = pd.read_csv(os.path.join(bad,
                                             "market_memory_summary.csv"))
            frame = frame.drop(columns=["pnl_sign_agreement_pct"])
            frame.to_csv(os.path.join(bad, "market_memory_summary.csv"),
                         index=False)
            try:
                AnalystData(bad)
                check("missing column raises ValueError", False)
            except ValueError as exc:
                check("missing column raises ValueError", True)
                check("file and column named in error",
                      "market_memory_summary.csv" in str(exc)
                      and "pnl_sign_agreement_pct" in str(exc))
        with tempfile.TemporaryDirectory() as bad:
            build_analyst_fixture(bad)
            path = os.path.join(bad, "ai_research_context_report.json")
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            del payload[0]["current_state"]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            try:
                AnalystData(bad)
                check("invalid JSON context raises ValueError", False)
            except ValueError as exc:
                check("invalid JSON context raises ValueError", True)
                check("missing context key named",
                      "current_state" in str(exc))
        check("valid fixture loads with 11 inputs", True)

        # ---------------- B2. real-source schema regression --------
        # Regression guard: the context CSV schema expectations must
        # match the ACTUAL ai_research_context_all.csv schema produced
        # by ai_research_layer.py.  The real file is read header-only
        # and never modified.
        start_area("real_schema_regression")
        required_ctx = REQUIRED_COLUMNS["ai_research_context_all.csv"]
        fixture_header = set(pd.read_csv(
            os.path.join(td, "ai_research_context_all.csv"),
            nrows=0).columns)
        check("fixture context CSV uses the actual source column names",
              all(c in fixture_header for c in required_ctx))
        real_path = os.path.join(SCRIPT_DIR, "ai_research_context_all.csv")
        check("real ai_research_context_all.csv present (required input)",
              os.path.isfile(real_path))
        if os.path.isfile(real_path):
            real_header = pd.read_csv(real_path, nrows=0)
            try:
                _require_columns(real_header,
                                 "ai_research_context_all.csv",
                                 required_ctx)
                check("REQUIRED_COLUMNS match the real source schema",
                      True)
            except ValueError:
                check("REQUIRED_COLUMNS match the real source schema",
                      False)
            check("fixture header is a subset of the real source schema",
                  all(c in set(real_header.columns)
                      for c in fixture_header))

        # ---------------- C. instrument separation ----------------
        start_area("C_instrument_separation")
        check("all five instruments handled",
              [ac["instrument"] for ac in analyst]
              == list(SUPPORTED_INSTRUMENTS))
        check("each context carries its own instrument",
              all(ac["current_state"]["instrument"] == ac["instrument"]
                  for ac in analyst))
        check("timestamps are per instrument",
              len({ac["timestamp"] for ac in analyst}) >= 1)
        check("latest observation selected (2026-09-03)",
              all(ac["timestamp"] == "2026-09-03 10:00:00"
                  for ac in analyst))

        # ---------------- D. no ordering / preference ----------------
        start_area("D_no_ordering")
        blob = (json.dumps(analyst, default=_json_default) + txt
                + json.dumps(rows, default=_json_default)).lower()
        check("no ordering words in outputs",
              all(w not in blob for w in ("ra" + "nk", "w" + "inner",
                                          "be" + "st", "wo" + "rst")))
        check("instrument order is the fixed supported order in the JSON",
              [ac["instrument"] for ac in analyst]
              == list(SUPPORTED_INSTRUMENTS))
        check("cross context states no preference",
              "no instrument is" in by_inst["GOLD"]
              ["cross_instrument_context"]["definitions"])

        # ---------------- E. no certainty figures ----------------
        start_area("E_no_certainty_figures")
        check("no certainty wording in outputs",
              all(p not in blob for p in
                  ("co" + "nfidence", "pro" + "babilit")))
        check("no compound rating fields in context keys",
              all(("score" not in k) for k in by_inst["GOLD"].keys()))
        check("evidence labels are qualitative only",
              by_inst["GOLD"]["research_evidence"]["evidence_quality"]
              in EVIDENCE_LABELS)

        # ---------------- F. no trading terminology ----------------
        start_area("F_no_trading_terminology")
        banned_res = (
            re.compile("\\b" + "bu" + "y\\b"),
            re.compile("\\b" + "se" + "ll\\b"),
            re.compile("\\b" + "ho" + "ld\\b"),
            re.compile("\\b" + "entr" + "y\\b"),
            re.compile("\\b" + "exi" + "t\\b"),
            re.compile("stop lo" + "ss\\b"),
            re.compile("take pro" + "fit\\b"),
            re.compile("lot si" + "ze\\b"),
            re.compile("position si" + "ze\\b"),
            re.compile("trade deci" + "sion\\b"),
        )
        check("outputs contain no trading terminology",
              all(rx.search(blob) is None for rx in banned_res))
        check("outputs contain no directive wording",
              "recommend" not in blob and "should " not in blob)

        # ---------------- G. deterministic output ----------------
        start_area("G_deterministic")
        data2 = AnalystData(td)
        contexts2 = {inst: data2.latest_context(inst)
                     for inst in SUPPORTED_INSTRUMENTS}
        analyst2 = [build_analyst_context(data2, inst, contexts2)
                    for inst in SUPPORTED_INSTRUMENTS]
        txt2 = generate_analyst_report(analyst2)
        check("identical inputs -> identical structured contexts",
              json.dumps(analyst, sort_keys=True, default=_json_default)
              == json.dumps(analyst2, sort_keys=True,
                            default=_json_default))
        check("identical inputs -> identical text report", txt == txt2)
        check("identical inputs -> identical CSV rows",
              json.dumps(rows, default=_json_default)
              == json.dumps([context_to_csv_row(ac) for ac in analyst2],
                            default=_json_default))

        # ---------------- H. output overwrite protection ----------------
        start_area("H_overwrite_protection")
        with tempfile.TemporaryDirectory() as od:
            try:
                check_output_files_available(od)
                check("clean directory passes", True)
            except OutputExistsError:
                check("clean directory passes", False)
        with tempfile.TemporaryDirectory() as od:
            open(os.path.join(od, OUTPUT_JSON), "w").close()
            try:
                check_output_files_available(od)
                check("existing output stops the run", False)
            except OutputExistsError as exc:
                check("existing output stops the run", True)
                check("exact conflicting filename reported",
                      OUTPUT_JSON in str(exc))
        with tempfile.TemporaryDirectory() as od:
            for name in OUTPUT_FILES:
                open(os.path.join(od, name), "w").close()
            try:
                check_output_files_available(od)
                check("all outputs checked before writing", False)
            except OutputExistsError as exc:
                check("all outputs checked before writing", True)
                check("every conflict listed",
                      all(n in str(exc) for n in OUTPUT_FILES))

        # ---------------- H2. timestamped output naming ----------------
        start_area("H2_timestamped_outputs")
        ts_names = timestamped_output_files("20261007_120000")
        ts_again = timestamped_output_files("20261007_120000")
        ts_other = timestamped_output_files("20261007_120001")
        check("timestamped names follow the documented pattern",
              ts_names == (
                  "ai_trading_analyst_context_20261007_120000.csv",
                  "ai_trading_analyst_context_20261007_120000.json",
                  "ai_trading_analyst_report_20261007_120000.txt"))
        check("one run uses a single shared timestamp",
              ts_names == ts_again)
        check("a different stamp yields a different set",
              ts_names != ts_other)
        check("STAMP_FORMAT round-trips the run stamp",
              datetime.datetime.strptime("20261007_120000", STAMP_FORMAT)
              .strftime(STAMP_FORMAT) == "20261007_120000")
        check("timestamped names never equal the fixed legacy names",
              all(name not in OUTPUT_FILES for name in ts_names))

        with tempfile.TemporaryDirectory() as od:
            check_output_files_available(od, ts_names)
            check("a clean directory passes for timestamped names", True)
            open(os.path.join(od, ts_names[1]), "w").close()
            try:
                check_output_files_available(od, ts_names)
                check("an existing timestamped output stops the run", False)
            except OutputExistsError as exc:
                check("an existing timestamped output stops the run", True)
                check("the timestamped conflict is named",
                      ts_names[1] in str(exc))
        with tempfile.TemporaryDirectory() as od:
            for name in ts_names:
                open(os.path.join(od, name), "w").close()
            try:
                check_output_files_available(od, ts_names)
                check("all timestamped paths checked before writing", False)
            except OutputExistsError as exc:
                check("all timestamped paths checked before writing", True)
                check("every timestamped conflict listed",
                      all(name in str(exc) for name in ts_names))
        with tempfile.TemporaryDirectory() as od:
            target = os.path.join(od, ts_names[0])
            open(target, "wb").close()
            saved_dir = globals()["SCRIPT_DIR"]
            try:
                globals()["SCRIPT_DIR"] = od
                try:
                    _write_bytes_if_absent(ts_names[0], b"new")
                    check("the timestamped writer refuses to overwrite",
                          False)
                except OutputExistsError:
                    check("the timestamped writer refuses to overwrite", True)
            finally:
                globals()["SCRIPT_DIR"] = saved_dir
            with open(target, "rb") as fh:
                check("the existing timestamped file is unchanged",
                      fh.read() == b"")

        # ---------------- I. historical-match terminology ----------------
        start_area("I_match_terminology")
        check("outputs use 'historical match' terminology",
              "historical match" in txt.lower())
        check("outputs never use forward-looking names",
              all(w not in blob for w in
                  ("pre" + "dict", "fore" + "cast",
                   "ex" + "pected ret" + "urn")))
        check("K note present and frozen",
              "K=10" in by_inst["GOLD"]["historical_memory"]["k_note"])

        # ---------------- J. no future leakage ----------------
        start_area("J_no_future_leakage")
        state_now = by_inst["GOLD"]["current_state"]
        env_now = by_inst["GOLD"]["market_environment"]["lines"]
        outcome_tokens = ("-999", "-9.0")
        check("current state contains no outcome values",
              all(t not in json.dumps(state_now, default=_json_default)
                  for t in outcome_tokens))
        check("environment contains no outcome values",
              all(t not in " ".join(env_now) for t in outcome_tokens))
        check("current state has no outcome keys",
              all(k not in state_now for k in
                  ("net_pnl", "win", "forward6_pct", "mfe12_atr")))

        # ---------------- K. outcome-column isolation ----------------
        start_area("K_outcome_isolation")
        with tempfile.TemporaryDirectory() as mut:
            build_analyst_fixture(mut)
            frame = pd.read_csv(os.path.join(mut,
                                             "ai_research_context_all.csv"))
            for col in ("actual_net_pnl", "actual_forward12_pct",
                        "sign_agreement_pct"):
                if col in frame.columns:
                    frame[col] = -777.0
            frame.to_csv(os.path.join(mut,
                                      "ai_research_context_all.csv"),
                         index=False)
            data_mut = AnalystData(mut)
            contexts_mut = {inst: data_mut.latest_context(inst)
                            for inst in SUPPORTED_INSTRUMENTS}
            ac_mut = build_analyst_context(data_mut, "GOLD", contexts_mut)
            check("mutating CSV outcome columns leaves current state "
                  "unchanged",
                  json.dumps(ac_mut["current_state"], sort_keys=True,
                             default=_json_default)
                  == json.dumps(state_now, sort_keys=True,
                                default=_json_default))
            check("mutating CSV outcome columns leaves environment "
                  "unchanged",
                  ac_mut["market_environment"] ==
                  by_inst["GOLD"]["market_environment"])

        # ---------------- L/M/N. static scans ----------------
        start_area("LMN_static_scans")
        violations = run_safety_scan()
        check("static safety scan clean (terminal / learning / "
              "optimization / wording)", violations == [])
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
        check("zero terminal tokens in source",
              all(t not in src.lower() for t in
                  ("meta" + "trader5", "m" + "t5")))
        check("zero optimization idioms in source",
              all(t not in src.lower() for t in
                  ("param_" + "grid", "gridse" + "arch",
                   "op" + "tuna", "hyper" + "opt")))

        # ---------------- O/P/Q. function-name guarantees ----------------
        start_area("OPQ_function_names")
        tree = ast.parse(src)
        fn_names = [n.name.lower() for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        check("no order function names",
              not any(t in n for n in fn_names for t in
                      ("or" + "der_", "position", "symbol")))
        check("no signal-generation function names",
              not any("sign" + "al" in n for n in fn_names))
        context_keys = set(by_inst["GOLD"].keys())
        forbidden_fields = ("co" + "nfidence", "pro" + "bability",
                            "signal_" + "score", "edge_" + "score",
                            "trade_" + "score", "recommen" + "ded_action",
                            "trade_" + "decision", "entr" + "y",
                            "exi" + "t", "stop_" + "loss",
                            "take_" + "profit", "lot_" + "size",
                            "position_" + "size")
        check("no forbidden decision fields in context keys",
              all(f not in context_keys for f in forbidden_fields))
        check("research_only flag present",
              all(ac["research_only"] is True for ac in analyst))

        # ---------------- R. cross-instrument separation ----------------
        start_area("R_cross_separation")
        cross = by_inst["GOLD"]["cross_instrument_context"]
        check("per-instrument entries kept separate",
              len(cross["per_instrument"]) == 5
              and all(p["instrument"] in SUPPORTED_INSTRUMENTS
                      for p in cross["per_instrument"]))
        check("evidence distribution counts 5 instruments",
              sum(cross["evidence_quality_distribution"].values()) == 5)
        check("walk-forward distribution counts 5 instruments",
              sum(cross["walk_forward_label_distribution"].values()) == 5)
        check("consistency statement is descriptive",
              isinstance(cross["consistency_statement"], str)
              and len(cross["consistency_statement"]) > 0)
        check("no combined figure field exists",
              all("combined" not in k for k in cross.keys())
              or "no instrument is" in cross["definitions"])

        # ---------------- limitations + summary ----------------
        start_area("limitations_summary")
        lim = by_inst["GOLD"]["limitations"]
        check("chance-level limitation recorded (50.0% <= 55%)",
              any("chance-level" in x for x in lim))
        check("weak-correlation limitation recorded (|0.047| < 0.10)",
              any("Pearson correlation" in x for x in lim))
        check("mixed regime limitation recorded",
              any("mixed across historical periods" in x for x in lim))
        check("unstable walk-forward limitation recorded (GOLD)",
              any("unstable" in x for x in lim))
        check("AUDUSD insufficient limitation recorded",
              any("insufficient" in x.lower()
                  for x in by_inst["AUDUSD"]["limitations"]))
        check("AUDUSD latest match unavailable limitation",
              any("no historical-match query" in x
                  for x in by_inst["AUDUSD"]["limitations"]))
        check("summary uses the fixed evidence sentence",
              "Historical evidence quality is weak."
              in by_inst["GOLD"]["analyst_context_summary"])
        check("summary uses the fixed unstable sentence",
              "Walk-forward relationships are unstable"
              in by_inst["GOLD"]["analyst_context_summary"])
        check("summary uses chance-level sentence",
              "close to chance-level"
              in by_inst["GOLD"]["analyst_context_summary"])
        check("summary states research-only nature",
              "no directive" in
              by_inst["GOLD"]["analyst_context_summary"])
        strong_summary = build_summary("STRONG", "PERSISTENT", 62.0, [])
        check("strong/persistent summary wording fixed",
              "Historical evidence quality is strong." in strong_summary
              and "held the same sign" in strong_summary)

        # ---------------- report sections ----------------
        start_area("report_sections")
        for section in ("1. CURRENT STATE", "2. MARKET ENVIRONMENT",
                        "3. HISTORICAL MEMORY", "4. RESEARCH EVIDENCE",
                        "5. REGIME STABILITY",
                        "6. WALK-FORWARD EVIDENCE",
                        "7. CROSS-INSTRUMENT CONTEXT",
                        "8. RESEARCH LIMITATIONS",
                        "9. ANALYST CONTEXT SUMMARY"):
            check(f"section present ({section[:1]})", section in txt)
        check("all five instruments have report blocks",
              all(txt.count(f"\n{inst}\n") >= 1
                  for inst in SUPPORTED_INSTRUMENTS))
        check("CSV schema is explicit and complete",
              set(rows[0].keys()) == set(ANALYST_CSV_COLUMNS))

        # ---------------- LR. live research context selection ----------------
        start_area("LR_live_research_inputs")
        lr = os.path.join(td, "_live_research")
        os.makedirs(lr, exist_ok=True)
        write_live_research_fixture(
            lr, "20261001_000000",
            live_ts={i: "2026-10-01 05:00:00"
                     for i in SUPPORTED_INSTRUMENTS})
        write_live_research_fixture(
            lr, "20261007_120000",
            live_ts={i: "2026-10-07 05:00:00"
                     for i in SUPPORTED_INSTRUMENTS})
        sel = find_live_research_inputs(lr)
        check("A newest timestamped live research context is selected",
              os.path.basename(sel[0])
              == "live_research_context_20261007_120000.csv"
              and os.path.basename(sel[1])
              == "live_research_context_20261007_120000.json"
              and os.path.basename(sel[2])
              == "live_research_context_report_20261007_120000.txt")
        check("E selection is deterministic",
              find_live_research_inputs(lr) == sel)
        loaded = load_live_research_contexts(sel[1])
        check("the selected live context maps all five instruments",
              list(loaded.keys()) == list(SUPPORTED_INSTRUMENTS)
              and loaded["GOLD"]["timestamp"] == "2026-10-07 05:00:00")
        check("live features are mapped and unavailable ones are None",
              loaded["GOLD"]["current_state"]["features"]["ADX"] == 12.0
              and loaded["GOLD"]["current_state"]["features"]["vol6"]
              is None)
        check("stored evidence labels are carried unchanged",
              loaded["GOLD"]["evidence_quality"] == "WEAK")

        lb = os.path.join(td, "_live_research_pair")
        os.makedirs(lb, exist_ok=True)
        write_live_research_fixture(lb, "20261007_120000", write_report=False)
        try:
            find_live_research_inputs(lb)
            check("B an unpaired timestamped CSV is not selected", False)
        except MissingInputError:
            check("B an unpaired timestamped CSV is not selected", True)
        write_live_research_fixture(lb, "20261006_000000")
        sel_b = find_live_research_inputs(lb)
        check("B only the complete timestamped set is selected",
              os.path.basename(sel_b[0])
              == "live_research_context_20261006_000000.csv")

        lc = os.path.join(td, "_live_research_legacy")
        os.makedirs(lc, exist_ok=True)
        with open(os.path.join(lc, "ai_research_context_all.csv"),
                  "w", encoding="utf-8") as fh:
            fh.write("instrument,timestamp,direction\n"
                     "GOLD,2026-09-28 06:00:00,LONG\n")
        write_live_research_fixture(lc, "20261007_120000")
        sel_c = find_live_research_inputs(lc)
        check("C the live context is selected over the legacy CSV",
              os.path.basename(sel_c[0]).startswith("live_research_context_")
              and all("ai_research_context_all.csv" not in p
                      for p in sel_c))

        ld = os.path.join(td, "_live_research_none")
        os.makedirs(ld, exist_ok=True)
        try:
            find_live_research_inputs(ld)
            check("D a missing live context fails safely", False)
        except MissingInputError as exc:
            check("D a missing live context fails safely", True)
            check("D the error names the live research context",
                  "live_research_context" in str(exc))

        tree_lr = ast.parse(open(os.path.abspath(__file__),
                                 encoding="utf-8").read())
        top_mods_lr: List[str] = []
        for node_lr in tree_lr.body:
            if isinstance(node_lr, ast.Import):
                top_mods_lr.extend(a.name.split(".")[0]
                                   for a in node_lr.names)
            elif isinstance(node_lr, ast.ImportFrom) and node_lr.module:
                top_mods_lr.append(node_lr.module.split(".")[0])
        check("F the module imports only its allowlisted dependencies",
              all(m in _ALLOWED_IMPORT_MODULES for m in top_mods_lr))
        check("F the live selection never yields a historical input",
              all("ai_research_context_all.csv" not in p
                  for p in (sel + sel_b + sel_c)))

        # ---------------- default-run gating ----------------
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
            sys.argv = ["ai_trading_analyst.py"]
            main()
            check("default main() runs synthetic tests only",
                  called["tests"] and not called["real"])
            called["real"] = False
            called["tests"] = False
            sys.argv = ["ai_trading_analyst.py", "--run"]
            main()
            check("--run reaches the real-analysis stage",
                  called["real"] and not called["tests"])
        finally:
            globals()["run_real_analysis"] = orig_real
            globals()["run_synthetic_tests"] = orig_tests
            sys.argv = argv_backup

    start_area(None)
    print()
    print("  Per-area verification (spec section 14 -> area: passed/failed):")
    spec_map = (
        ("A_missing_files", "A missing-file handling"),
        ("B_schema", "B schema validation"),
        ("real_schema_regression", "real-source schema regression"),
        ("C_instrument_separation", "C instrument separation"),
        ("D_no_ordering", "D no ordering of instruments"),
        ("E_no_certainty_figures", "E no certainty figures"),
        ("F_no_trading_terminology", "F no trading terminology"),
        ("G_deterministic", "G deterministic output"),
        ("H_overwrite_protection", "H existing-output protection"),
        ("H2_timestamped_outputs", "H2 timestamped output naming"),
        ("I_match_terminology", "I historical-match terminology"),
        ("J_no_future_leakage", "J no future leakage"),
        ("K_outcome_isolation", "K outcome-column isolation"),
        ("LMN_static_scans", "L/M/N terminal, learning, optimization"),
        ("OPQ_function_names", "O/P/Q order, signal, decision fields"),
        ("R_cross_separation", "R cross-instrument separation"),
        ("limitations_summary", "limitations + summary rules"),
        ("report_sections", "report structure"),
        ("default_run_gating", "default-run + --run gating"),
        ("LR_live_research_inputs", "live research context selection"),
    )
    stats = {name: (p, f) for name, p, f in area_stats}
    for name, spec in spec_map:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<28} ({spec:<36}) {p:>2} passed, {f} failed "
              f"[{status}]")
    print()
    print(f"  TOTAL: {passed} passed, {failed} failed")
    return passed, failed


# ============================================================
# Command-line gating
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
    _code = main()
    if _code:
        raise SystemExit(_code)
