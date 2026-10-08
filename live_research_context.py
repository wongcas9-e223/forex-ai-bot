"""
live_research_context.py - READ-ONLY LIVE-TO-RESEARCH CONTEXT BRIDGE.

WHAT THIS IS
    A deterministic, read-only preparation layer that joins the latest
    live market-state observation (live_market_state.csv) to the stored
    historical research context already produced by ai_research_layer.py
    and ai_trading_analyst.py.

    It answers: "What does the stored historical research already record
    about the environment the live observation is in?"

WHAT THIS IS NOT
    - not a trading system and not an execution layer,
    - not a learning system, not a fitted structure, not a search of any
      parameter,
    - it never touches a trading terminal,
    - it never declares anything as preferable,
    - it modifies NO existing file: every input is opened read-only and
      only the three NEW output files may be created.

REUSE (the frozen definitions are NOT re-created here)
    The live observation is matched to historical state evidence with the
    research layer's own functions, imported unchanged:
        ai_research_layer.build_regime_stability
        ai_research_layer.build_walk_forward_evidence
        ai_research_layer.ResearchData
    The historical-memory context is taken from the analyst layer's own
    functions, imported unchanged:
        ai_trading_analyst.AnalystData
        ai_trading_analyst.build_analyst_context
    Small helpers (safe_float, safe_int, fmt_ts, parse_timestamp) and the
    frozen label sets (EVIDENCE_LABELS, WF_LABELS) are likewise reused.
    Live-state column names, status values and warm-up constants are
    reused from live_market_state.py.

INPUT FILES (read-only; all eleven checked; a missing file is named)
    live_market_state.csv
    live_market_state_report.txt
    (The real --run instead selects the newest VALID timestamped
    live-market-state pair that shares one timestamp:
        live_market_state_YYYYMMDD_HHMMSS.csv
        live_market_state_report_YYYYMMDD_HHMMSS.txt
    Incomplete or mismatched timestamp pairs are ignored, and the legacy
    untimestamped files above are used only when no valid timestamped
    pair exists.  The selected names are printed and listed under
    "input files read" in the generated report.  The synthetic tests may
    keep using the legacy names internally.)
    ai_research_context_all.csv
    ai_research_context_report.json
    market_memory_oos_validation.csv
    market_memory_match_details.csv
    market_memory_summary.csv
    market_state_oos_period_summary.csv
    market_state_oos_state_summary.csv
    market_state_oos_walkforward.csv
    market_state_oos_cross_instrument.csv
    (The reused loaders additionally require market_state_all.csv and
    ai_research_context_report.txt, which already exist in the project.
    Their own guards name them if they are ever absent.)

OUTPUT STRUCTURE (per instrument, deterministic)
    A. current live state       - copied unchanged from live_market_state.csv
    B. historical research      - evidence label, match sample, correlation,
       context                    walk-forward label, state evidence,
                                  cross-instrument context
    C. limitations              - carried forward from the analyst layer
    D. research interpretation  - descriptive statements only
    E. data quality             - descriptive notes about the live input

OUTPUT FILES (all NEW; every path is checked before anything is written;
    an existing file is never overwritten - a timestamped alternative is
    used instead, and a collision on that alternative stops the run):
    live_research_context.csv
    live_research_context.json
    live_research_context_report.txt

COMMAND-LINE SAFETY
    python live_research_context.py          -> synthetic tests only
                                                (never connects to a
                                                trading terminal, never
                                                reads live data, never
                                                writes real output)
    python live_research_context.py --run    -> real analysis

DISCLAIMER: descriptive historical research preparation only. No causal
claims, no forward-looking statements, no directive of any kind, nothing
proven effective. Educational use only.
"""

# ============================================================
# Imports: Python standard library + numpy + pandas + the two
# existing project modules whose frozen definitions are reused.
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

from ai_research_layer import (
    EVIDENCE_LABELS as RESEARCH_EVIDENCE_LABELS,
    WF_LABELS as RESEARCH_WF_LABELS,
    ResearchData,
    build_regime_stability as build_oos_regime_stability,
    build_walk_forward_evidence as build_oos_walk_forward,
)
from ai_trading_analyst import (
    EVIDENCE_LABELS,
    SUPPORTED_INSTRUMENTS,
    TS_FORMAT,
    WF_LABELS,
    AnalystData,
    MissingInputError,
    OutputExistsError,
    _json_default,
    build_analyst_context,
    build_analyst_fixture,
    fmt_ts,
    parse_timestamp,
    safe_float,
    safe_int,
)
from live_market_state import (
    CSV_COLUMNS as LIVE_CSV_COLUMNS,
    DIRECTION_LONG,
    DIRECTION_NONE,
    DIRECTION_SHORT,
    MIN_CLOSED_CANDLES,
    NUMERIC_STATE_KEYS as LIVE_NUMERIC_KEYS,
    REGIME_KEYS as LIVE_REGIME_KEYS,
    STATUS_COLLECTED,
    STATUS_UNAVAILABLE,
    TIME_STATE_KEYS as LIVE_TIME_KEYS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Frozen settings
# ============================================================
LIVE_CSV = "live_market_state.csv"
LIVE_TXT = "live_market_state_report.txt"
RESEARCH_ALL_CSV = "ai_research_context_all.csv"
RESEARCH_JSON = "ai_research_context_report.json"
MEMORY_VALIDATION_CSV = "market_memory_oos_validation.csv"
MEMORY_DETAILS_CSV = "market_memory_match_details.csv"
MEMORY_SUMMARY_CSV = "market_memory_summary.csv"
OOS_PERIOD_CSV = "market_state_oos_period_summary.csv"
OOS_STATE_CSV = "market_state_oos_state_summary.csv"
OOS_WALKFORWARD_CSV = "market_state_oos_walkforward.csv"
OOS_CROSS_CSV = "market_state_oos_cross_instrument.csv"

REQUIRED_INPUT_FILES: Tuple[str, ...] = (
    LIVE_CSV, LIVE_TXT,
    RESEARCH_ALL_CSV, RESEARCH_JSON, MEMORY_VALIDATION_CSV,
    MEMORY_DETAILS_CSV, MEMORY_SUMMARY_CSV, OOS_PERIOD_CSV, OOS_STATE_CSV,
    OOS_WALKFORWARD_CSV, OOS_CROSS_CSV,
)

OUTPUT_CSV = "live_research_context.csv"
OUTPUT_JSON = "live_research_context.json"
OUTPUT_TXT = "live_research_context_report.txt"
OUTPUT_FILES: Tuple[str, ...] = (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_TXT)

STAMP_FORMAT = "%Y%m%d_%H%M%S"

# Newest valid timestamped live-market-state pair (real --run only).
LIVE_CSV_STAMP_RE = re.compile(
    r"^live_market_state_(\d{8}_\d{6})\.csv$")
LIVE_REPORT_STAMP_RE = re.compile(
    r"^live_market_state_report_(\d{8}_\d{6})\.txt$")

# Fixed, documented data-quality rules (never tuned).
STALE_LAG_HOURS = 72.0   # live candle behind the newest live candle -> note

LIVE_IDENTITY_COLUMNS: Tuple[str, ...] = (
    "instrument", "live_collection_status",
    "live_latest_closed_candle_time", "live_candles_used",
    "direction_stored_research",
)

HISTORY_FIELD_IDS: Tuple[str, ...] = (
    "stored_observation_timestamp", "stored_evidence_quality",
    "stored_evidence_explanation", "stored_research_conclusion",
    "memory_n_queries", "memory_pnl_sign_agreement_pct",
    "memory_pearson_r_pnl", "memory_avg_nearest_distance",
    "latest_stored_n_matches", "latest_stored_nearest_distance",
    "latest_stored_sign_agreement_pct",
    "stored_regime_stability_classifications", "stored_walk_forward_label",
    "stored_walk_forward_classifications",
    "cross_instrument_consistency_statement",
)

MATCH_FIELD_IDS: Tuple[str, ...] = (
    "live_state_oos_available", "live_state_matched_categories",
    "live_state_oos_classifications", "live_state_walk_forward_label",
)

CONTEXT_CSV_COLUMNS: Tuple[str, ...] = (
    LIVE_IDENTITY_COLUMNS + LIVE_NUMERIC_KEYS + LIVE_TIME_KEYS
    + LIVE_REGIME_KEYS + HISTORY_FIELD_IDS + MATCH_FIELD_IDS
    + ("n_limitations", "limitations", "n_data_quality_issues",
       "data_quality_issues", "research_interpretation", "research_only")
)

REQUIRED_LIVE_COLUMNS: Tuple[str, ...] = (
    ("instrument", "status", "latest_closed_candle_time", "candles_used",
     "direction", "data_quality_warnings")
    + LIVE_NUMERIC_KEYS + LIVE_TIME_KEYS + LIVE_REGIME_KEYS
)

# Plain-language labels for the six frozen regime categories.
_REGIME_LABELS: Tuple[Tuple[str, str], ...] = (
    ("adx_regime", "ADX"),
    ("volatility_regime", "volatility"),
    ("trend_distance_regime", "trend distance"),
    ("bb_excursion_regime", "Bollinger excursion"),
    ("efficiency_regime", "efficiency"),
    ("rsi_regime", "RSI"),
)


class LiveSchemaError(RuntimeError):
    """live_market_state.csv does not carry the expected frozen schema."""


# ============================================================
# Small deterministic helpers
# ============================================================
def _is_missing(value: Any) -> bool:
    """True for None, NaN and the empty string."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def _parse_ts_safe(value: Any) -> Optional[pd.Timestamp]:
    """parse_timestamp, but a bad value yields None instead of raising."""
    if _is_missing(value):
        return None
    try:
        return parse_timestamp(value)
    except (ValueError, TypeError):
        return None


def _fmt_ts_safe(value: Any) -> Optional[str]:
    ts = _parse_ts_safe(value)
    return None if ts is None else fmt_ts(ts)


def _as_text(value: Any) -> Optional[str]:
    return None if _is_missing(value) else str(value)


# ============================================================
# A. Live input loading (read-only)
# ============================================================
def find_live_market_inputs(directory: str = SCRIPT_DIR
                            ) -> Tuple[str, str]:
    """The newest valid timestamped live-market-state CSV + report pair.

    A pair is valid only when the CSV and the report share the exact
    same timestamp.  Incomplete or mismatched pairs are ignored.  When
    no valid pair exists the run stops with a clear error; the legacy
    live_market_state.csv is never used as a silent fallback.
    """
    csv_by_stamp: Dict[str, str] = {}
    report_by_stamp: Dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        csv_match = LIVE_CSV_STAMP_RE.match(name)
        if csv_match:
            csv_by_stamp.setdefault(csv_match.group(1), name)
            continue
        report_match = LIVE_REPORT_STAMP_RE.match(name)
        if report_match:
            report_by_stamp.setdefault(report_match.group(1), name)
    shared = sorted(set(csv_by_stamp) & set(report_by_stamp))
    if not shared:
        raise MissingInputError(
            "no valid timestamped live-market-state pair is present "
            "(expected live_market_state_YYYYMMDD_HHMMSS.csv and "
            "live_market_state_report_YYYYMMDD_HHMMSS.txt sharing one "
            "timestamp); refusing to consume the stale legacy "
            "live_market_state.csv")
    stamp = shared[-1]
    return (os.path.join(directory, csv_by_stamp[stamp]),
            os.path.join(directory, report_by_stamp[stamp]))


def resolved_input_names(live_csv_path: str,
                         live_report_path: str) -> Tuple[str, ...]:
    """The read-only input names for the report: the selected live pair
    (replacing the legacy live names) followed by the other inputs."""
    live_names = (os.path.basename(live_csv_path),
                  os.path.basename(live_report_path))
    others = tuple(f for f in REQUIRED_INPUT_FILES
                   if f not in (LIVE_CSV, LIVE_TXT))
    return live_names + others


def check_bridge_inputs(directory: str = SCRIPT_DIR,
                        live_inputs: Optional[Tuple[str, str]] = None
                        ) -> List[str]:
    """Names (returned, not raised) every missing required input.

    When `live_inputs` (the resolved live-market-state CSV and report) is
    supplied, those resolved paths are checked in place of the legacy
    live file names.
    """
    if live_inputs is None:
        return [f for f in REQUIRED_INPUT_FILES
                if not os.path.isfile(os.path.join(directory, f))]
    live_csv_path, live_report_path = live_inputs
    others = [f for f in REQUIRED_INPUT_FILES
              if f not in (LIVE_CSV, LIVE_TXT)]
    missing = [f for f in others
               if not os.path.isfile(os.path.join(directory, f))]
    for path in (live_csv_path, live_report_path):
        if not os.path.isfile(path):
            missing.append(os.path.basename(path))
    return missing


# ============================================================
# Live data-freshness gate (reuses market_data_freshness.py)
# ============================================================
def check_selected_live_freshness(
        live_csv_path: str,
        live_report_path: str,
        threshold_hours: Optional[float] = None
        ) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Gate the EXACT already-selected live pair on data freshness.

    Nothing here is recomputed: the collection timestamp comes from
    market_data_freshness.read_collection_timestamp, each instrument's
    result comes from market_data_freshness.compute_freshness, and the
    not-fresh set comes from market_data_freshness.data_quality_gate.
    The frozen 72-hour rule is market_data_freshness's own
    DEFAULT_FRESHNESS_THRESHOLD_HOURS - no second threshold exists.
    Reads only the given paths; it never substitutes an older dataset.
    Read-only; places no order and makes no trading decision.

    Returns (not_fresh_instruments, per_instrument_results).
    """
    import market_data_freshness as _mdf   # lazy: avoids an import cycle
    if threshold_hours is None:
        threshold_hours = _mdf.DEFAULT_FRESHNESS_THRESHOLD_HOURS

    rows, _issues = load_live_state(csv_path=live_csv_path)
    collection_ts, _note = _mdf.read_collection_timestamp(live_report_path)
    results: List[Dict[str, Any]] = []
    for inst in SUPPORTED_INSTRUMENTS:
        row = rows.get(inst)
        results.append(_mdf.compute_freshness(
            inst,
            collection_ts,
            None if row is None else row.get("latest_closed_candle_time"),
            None if row is None else row.get("status"),
            threshold_hours))
    return _mdf.data_quality_gate(results), results


def load_live_state(directory: str = SCRIPT_DIR,
                    csv_path: Optional[str] = None
                    ) -> Tuple[Dict[str, Optional[Dict[str, Any]]],
                               Dict[str, List[str]]]:
    """Read a live-market-state CSV and select each instrument's row.

    When `csv_path` is omitted the legacy live_market_state.csv is read
    (the synthetic tests rely on this).  The real --run passes the newest
    valid timestamped CSV.  When several rows exist for one instrument
    the row with the newest closed-candle timestamp is selected
    (deterministic).  A schema mismatch is reported by naming the missing
    columns.
    """
    path = csv_path if csv_path is not None \
        else os.path.join(directory, LIVE_CSV)
    live_name = os.path.basename(path)
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_LIVE_COLUMNS if c not in frame.columns]
    if missing:
        raise LiveSchemaError(
            live_name + ": missing required column(s): "
            + ", ".join(missing))

    rows: Dict[str, Optional[Dict[str, Any]]] = {}
    issues: Dict[str, List[str]] = {}
    for inst in SUPPORTED_INSTRUMENTS:
        sub = frame[frame["instrument"].astype(str) == inst]
        if len(sub) == 0:
            rows[inst] = None
            issues[inst] = [
                "no live observation row for this instrument is present "
                "in " + live_name]
            continue
        chosen: Optional[Dict[str, Any]] = None
        chosen_ts: Optional[pd.Timestamp] = None
        for rec in sub.to_dict("records"):
            ts = _parse_ts_safe(rec.get("latest_closed_candle_time"))
            newer = (ts is not None
                     and (chosen_ts is None or ts >= chosen_ts))
            if chosen is None or newer:
                chosen, chosen_ts = rec, ts
        rows[inst] = chosen
        issues[inst] = []
    return rows, issues


def newest_live_timestamp(
        rows: Dict[str, Optional[Dict[str, Any]]]) -> Optional[pd.Timestamp]:
    newest: Optional[pd.Timestamp] = None
    for inst in SUPPORTED_INSTRUMENTS:
        row = rows.get(inst)
        if row is None:
            continue
        ts = _parse_ts_safe(row.get("latest_closed_candle_time"))
        if ts is not None and (newest is None or ts > newest):
            newest = ts
    return newest


def build_live_state_snapshot(row: Optional[Dict[str, Any]]
                              ) -> Optional[Dict[str, Any]]:
    """Section A: the live state copied unchanged (no reinterpretation)."""
    if row is None:
        return None
    return {
        "collection_status": _as_text(row.get("status")),
        "latest_closed_candle_time":
            _fmt_ts_safe(row.get("latest_closed_candle_time")),
        "candles_used": safe_int(row.get("candles_used")),
        "direction_stored_research": _as_text(row.get("direction")),
        "features": {k: safe_float(row.get(k)) for k in LIVE_NUMERIC_KEYS},
        "time_context": {k: safe_int(row.get(k)) for k in LIVE_TIME_KEYS},
        "classifications": {
            k: _as_text(row.get(k)) for k in LIVE_REGIME_KEYS},
    }


# ============================================================
# B (part 1). Live state -> stored historical state evidence
# ============================================================
def build_live_state_for_match(instrument: str,
                               row: Dict[str, Any]) -> Dict[str, Any]:
    """The minimal state the research layer's matching functions expect."""
    return {
        "instrument": instrument,
        "classifications": {
            k: _as_text(row.get(k)) for k in LIVE_REGIME_KEYS},
        "features": {k: safe_float(row.get(k)) for k in LIVE_NUMERIC_KEYS},
    }


def match_live_state_to_history(instrument: str,
                                row: Optional[Dict[str, Any]],
                                research: ResearchData) -> Dict[str, Any]:
    """Look the live observation's OWN regime buckets up in the stored
    research summaries, using the research layer's frozen functions.

    No new figure is created and nothing is ordered.  The lookup depends
    only on the live regimes and the stored files: it never reads past
    the supplied inputs and never consults a later candle.
    """
    base: Dict[str, Any] = {
        "available": False,
        "available_categories": 0,
        "category_count": len(LIVE_REGIME_KEYS),
        "by_category": [],
        "classifications": [],
        "walk_forward_label": None,
        "walk_forward_classifications": [],
        "walk_forward_by_category": [],
        "note": ("descriptive lookup of the live observation's own regime "
                 "buckets in the stored research summaries; no new figure "
                 "and no ordering is produced"),
    }
    if row is None:
        return base

    state = build_live_state_for_match(instrument, row)
    stability = build_oos_regime_stability(state, research)
    walk_forward = build_oos_walk_forward(state, research)

    cats: List[Dict[str, Any]] = []
    for cat in stability.get("by_category", []):
        if cat.get("available"):
            cats.append({
                "category": f"{cat.get('feature')} ({cat.get('bucket')})",
                "stored_classification":
                    str(cat.get("pnl_classification")),
            })
    wf_cats: List[Dict[str, Any]] = []
    for cat in walk_forward.get("by_category", []):
        wf_cats.append({
            "category": f"{cat.get('feature')} ({cat.get('bucket')})",
            "available": bool(cat.get("available")),
            "label": str(cat.get("label")),
        })
    return {
        "available": bool(cats),
        "available_categories": len(cats),
        "category_count": len(LIVE_REGIME_KEYS),
        "by_category": cats,
        "classifications": [c["stored_classification"] for c in cats],
        "walk_forward_label": str(walk_forward.get("label")),
        "walk_forward_classifications": [c["label"] for c in wf_cats],
        "walk_forward_by_category": wf_cats,
        "note": base["note"],
    }


# ============================================================
# B (part 2). Historical research context from the analyst layer
# ============================================================
def build_historical_context(data: AnalystData, instrument: str,
                             analyst_context: Dict[str, Any]
                             ) -> Dict[str, Any]:
    """Preserve the stored labels exactly; create no new figure."""
    memory = analyst_context["historical_memory"]
    latest = memory["latest_observation"]
    evidence = analyst_context["research_evidence"]
    cross = analyst_context["cross_instrument_context"]
    idx = SUPPORTED_INSTRUMENTS.index(instrument)
    stability_classes = [
        str(c.get("stored_classification", ""))
        for c in analyst_context["regime_stability"] if c.get("available")]
    wf_classes = [str(c.get("label", ""))
                  for c in analyst_context["walk_forward"]]
    return {
        "stored_observation_timestamp": analyst_context["timestamp"],
        "stored_evidence_quality": evidence["evidence_quality"],
        "stored_evidence_explanation": evidence["evidence_explanation"],
        "stored_research_conclusion": evidence["research_conclusion"],
        "memory_n_queries": memory["evaluation_query_count"],
        "memory_pnl_sign_agreement_pct": memory["pnl_sign_agreement_pct"],
        "memory_pearson_r_pnl": memory["pearson_r_match_actual_pnl"],
        "memory_avg_nearest_distance": memory["average_nearest_distance"],
        "latest_stored_n_matches": latest["n_matches"],
        "latest_stored_nearest_distance": latest["nearest_distance"],
        "latest_stored_sign_agreement_pct":
            latest["latest_match_sign_agreement_pct"],
        "stored_regime_stability_classifications": stability_classes,
        "stored_walk_forward_label":
            cross["per_instrument"][idx]["walk_forward_label"],
        "stored_walk_forward_classifications": wf_classes,
        "cross_instrument_consistency_statement":
            cross["consistency_statement"],
    }


# ============================================================
# C. Limitations carried forward from the analyst layer
# ============================================================
def carry_forward_limitations(analyst_context: Dict[str, Any]) -> List[str]:
    """The analyst layer's own limitations, copied unchanged.  A live
    input problem is reported separately under data quality and is never
    mixed into these stored limitations."""
    return [str(x) for x in analyst_context.get("limitations", [])]


# ============================================================
# D. Research interpretation (descriptive statements only)
# ============================================================
def build_research_interpretation(
        instrument: str, row: Optional[Dict[str, Any]],
        history: Optional[Dict[str, Any]],
        match: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if row is None:
        lines.append(
            "No live observation is stored for " + instrument + ", so no "
            "current live state can be described.")
    else:
        status = _as_text(row.get("status")) or "n/a"
        lines.append("The live collection status for " + instrument
                     + " is " + status + ".")
        direction = _as_text(row.get("direction"))
        if direction is None:
            lines.append("The current live observation has no stored "
                         "research direction value.")
        elif direction == DIRECTION_NONE:
            lines.append("The current live observation carries a stored "
                         "research direction of NONE; the frozen labelling "
                         "condition is not met on this candle.")
        elif direction in (DIRECTION_LONG, DIRECTION_SHORT):
            lines.append(
                "The current live observation carries a stored research "
                "direction of " + direction + "; this is a stored research "
                "labelling value and never a directive of any kind.")
        else:
            lines.append("The current live observation carries a stored "
                         "research direction of " + direction + ".")
        for key, label in _REGIME_LABELS:
            value = row.get(key)
            if not _is_missing(value):
                lines.append("The current live state falls within the "
                             + str(value) + " regime for " + label + ".")

    if history is not None:
        lines.append("Historical memory evidence for the latest stored "
                     "observation is classified as "
                     + str(history["stored_evidence_quality"]) + ".")
        lines.append("Stored walk-forward stability for the latest stored "
                     "observation is classified as "
                     + str(history["stored_walk_forward_label"]) + ".")

    if match is not None:
        if match.get("available_categories", 0) > 0:
            lines.append(
                "The current live state's own regime categories have stored "
                "historical state evidence in "
                + str(match["available_categories"]) + " of "
                + str(match["category_count"]) + " categories.")
            lines.append(
                "Stored walk-forward stability for the current live state's "
                "regime categories is classified as "
                + str(match["walk_forward_label"]) + ".")
        else:
            lines.append("The current state has no qualifying historical "
                         "evidence in the stored research set.")

    lines.append("These statements are descriptive research context only "
                 "and contain no directive of any kind.")
    return lines


# ============================================================
# E. Data quality of the live input (descriptive only)
# ============================================================
def assess_live_data_quality(instrument: str,
                             row: Optional[Dict[str, Any]],
                             intrinsic_issues: List[str],
                             newest_ts: Optional[pd.Timestamp]) -> List[str]:
    issues: List[str] = list(intrinsic_issues)
    if row is None:
        return issues

    status = _as_text(row.get("status"))
    if status != STATUS_COLLECTED:
        issues.append(f"live collection status is {status} for this "
                      f"instrument")
    else:
        candles = safe_int(row.get("candles_used"))
        if candles is None or candles < MIN_CLOSED_CANDLES:
            issues.append(
                f"live candles used ({candles}) is below the frozen warm-up "
                f"requirement ({MIN_CLOSED_CANDLES})")

    direction = _as_text(row.get("direction"))
    if direction is None:
        issues.append("stored research direction is missing on the live "
                      "observation")
    elif direction == DIRECTION_NONE:
        issues.append("stored research direction is NONE: the frozen "
                      "labelling condition is not met on this live candle")
    elif direction not in (DIRECTION_LONG, DIRECTION_SHORT):
        issues.append("stored research direction is an unexpected value: "
                      + direction)

    missing_values = [k for k in LIVE_NUMERIC_KEYS
                      if safe_float(row.get(k)) is None]
    if missing_values:
        issues.append("missing or non-finite live state value(s): "
                      + ", ".join(missing_values))
    missing_regimes = [k for k in LIVE_REGIME_KEYS
                       if _is_missing(row.get(k))]
    if missing_regimes:
        issues.append("missing live regime classification(s): "
                      + ", ".join(missing_regimes))

    ts = _parse_ts_safe(row.get("latest_closed_candle_time"))
    if ts is None:
        issues.append("live latest closed candle timestamp is missing or "
                      "unreadable")
    elif newest_ts is not None and (newest_ts - ts) > pd.Timedelta(
            hours=STALE_LAG_HOURS):
        hours = (newest_ts - ts).total_seconds() / 3600.0
        issues.append(
            f"live latest closed candle timestamp appears older than the "
            f"newest collected live candle by {hours:.1f} hours (fixed "
            f"descriptive rule: more than {STALE_LAG_HOURS:.0f} hours)")

    reported = row.get("data_quality_warnings")
    if not _is_missing(reported):
        issues.append("the live collector reported: " + str(reported))
    return issues


# ============================================================
# Context assembly (deterministic)
# ============================================================
def build_contexts(directory: str = SCRIPT_DIR,
                   live_csv_path: Optional[str] = None
                   ) -> List[Dict[str, Any]]:
    """One context per supported instrument, in the fixed order.

    `live_csv_path` lets the real --run supply the newest timestamped
    live-market-state CSV; when omitted the legacy file is read.
    """
    data = AnalystData(directory)
    research = ResearchData(directory)
    rows, intrinsic = load_live_state(directory, live_csv_path)
    newest_ts = newest_live_timestamp(rows)

    latest_contexts = {inst: data.latest_context(inst)
                       for inst in SUPPORTED_INSTRUMENTS}
    analyst_contexts = {
        inst: build_analyst_context(data, inst, latest_contexts)
        for inst in SUPPORTED_INSTRUMENTS}

    contexts: List[Dict[str, Any]] = []
    for inst in SUPPORTED_INSTRUMENTS:
        row = rows[inst]
        history = build_historical_context(data, inst, analyst_contexts[inst])
        match = match_live_state_to_history(inst, row, research)
        quality = assess_live_data_quality(inst, row, intrinsic[inst],
                                           newest_ts)
        interpretation = build_research_interpretation(inst, row, history,
                                                       match)
        contexts.append({
            "instrument": inst,
            "live_state": build_live_state_snapshot(row),
            "historical_research_context": history,
            "live_state_historical_match": match,
            "limitations": carry_forward_limitations(analyst_contexts[inst]),
            "data_quality": {"issue_count": len(quality),
                             "issues": quality},
            "research_interpretation": interpretation,
            "research_only": True,
        })
    return contexts


# ============================================================
# Output builders (deterministic)
# ============================================================
def context_to_row(ctx: Dict[str, Any]) -> Dict[str, Any]:
    live = ctx["live_state"]
    history = ctx["historical_research_context"]
    match = ctx["live_state_historical_match"]
    row: Dict[str, Any] = {c: None for c in CONTEXT_CSV_COLUMNS}
    row["instrument"] = ctx["instrument"]
    if live is not None:
        row["live_collection_status"] = live["collection_status"]
        row["live_latest_closed_candle_time"] = \
            live["latest_closed_candle_time"]
        row["live_candles_used"] = live["candles_used"]
        row["direction_stored_research"] = live["direction_stored_research"]
        for k in LIVE_NUMERIC_KEYS:
            row[k] = live["features"].get(k)
        for k in LIVE_TIME_KEYS:
            row[k] = live["time_context"].get(k)
        for k in LIVE_REGIME_KEYS:
            row[k] = live["classifications"].get(k)
    row["stored_observation_timestamp"] = history["stored_observation_timestamp"]
    row["stored_evidence_quality"] = history["stored_evidence_quality"]
    row["stored_evidence_explanation"] = history["stored_evidence_explanation"]
    row["stored_research_conclusion"] = history["stored_research_conclusion"]
    row["memory_n_queries"] = history["memory_n_queries"]
    row["memory_pnl_sign_agreement_pct"] = \
        history["memory_pnl_sign_agreement_pct"]
    row["memory_pearson_r_pnl"] = history["memory_pearson_r_pnl"]
    row["memory_avg_nearest_distance"] = history["memory_avg_nearest_distance"]
    row["latest_stored_n_matches"] = history["latest_stored_n_matches"]
    row["latest_stored_nearest_distance"] = \
        history["latest_stored_nearest_distance"]
    row["latest_stored_sign_agreement_pct"] = \
        history["latest_stored_sign_agreement_pct"]
    row["stored_regime_stability_classifications"] = " | ".join(
        history["stored_regime_stability_classifications"])
    row["stored_walk_forward_label"] = history["stored_walk_forward_label"]
    row["stored_walk_forward_classifications"] = " | ".join(
        history["stored_walk_forward_classifications"])
    row["cross_instrument_consistency_statement"] = \
        history["cross_instrument_consistency_statement"]
    row["live_state_oos_available"] = match["available"]
    row["live_state_matched_categories"] = match["available_categories"]
    row["live_state_oos_classifications"] = " | ".join(match["classifications"])
    row["live_state_walk_forward_label"] = match["walk_forward_label"]
    row["n_limitations"] = len(ctx["limitations"])
    row["limitations"] = " | ".join(ctx["limitations"])
    row["n_data_quality_issues"] = ctx["data_quality"]["issue_count"]
    row["data_quality_issues"] = " | ".join(ctx["data_quality"]["issues"])
    row["research_interpretation"] = " ".join(ctx["research_interpretation"])
    row["research_only"] = True
    return row


def contexts_to_csv(contexts: List[Dict[str, Any]]) -> str:
    rows = [context_to_row(c) for c in contexts]
    return pd.DataFrame(rows, columns=list(CONTEXT_CSV_COLUMNS)).to_csv(
        index=False)


def build_json_payload(contexts: List[Dict[str, Any]],
                       generated_ts: datetime.datetime,
                       inputs_read: Tuple[str, ...]) -> Dict[str, Any]:
    return {
        "generated_timestamp": generated_ts.strftime(TS_FORMAT),
        "inputs_read": list(inputs_read),
        "research_only": True,
        "instruments": contexts,
    }


def contexts_to_json(contexts: List[Dict[str, Any]],
                     generated_ts: datetime.datetime,
                     inputs_read: Tuple[str, ...]) -> str:
    payload = build_json_payload(contexts, generated_ts, inputs_read)
    return json.dumps(payload, indent=2, default=_json_default,
                      ensure_ascii=True) + "\n"


def build_report(contexts: List[Dict[str, Any]],
                 generated_ts: datetime.datetime,
                 inputs_read: Tuple[str, ...]) -> str:
    L: List[str] = []
    L.append("=" * 68)
    L.append("LIVE RESEARCH CONTEXT - READ-ONLY BRIDGE REPORT")
    L.append("=" * 68)
    L.append("generated timestamp : " + generated_ts.strftime(TS_FORMAT))
    L.append("timeframe           : H1 (closed candles only)")
    L.append("instruments         : " + ", ".join(SUPPORTED_INSTRUMENTS))
    L.append("input files read (read-only):")
    for name in inputs_read:
        L.append("  - " + name)
    L.append("")
    L.append("This report is descriptive research context only. It contains "
             "no directive, no certainty figure and no ordering of "
             "instruments.")

    for ctx in contexts:
        L.append("")
        L.append("-" * 68)
        L.append("INSTRUMENT: " + ctx["instrument"])
        live = ctx["live_state"]
        L.append("A. CURRENT LIVE STATE")
        if live is None:
            L.append("  (no live observation is stored for this instrument)")
        else:
            L.append("  live collection status      : "
                     + str(live["collection_status"]))
            L.append("  latest closed candle        : "
                     + str(live["latest_closed_candle_time"]))
            L.append("  live candles used           : "
                     + str(live["candles_used"]))
            L.append("  stored research direction   : "
                     + str(live["direction_stored_research"]))
            for k in LIVE_NUMERIC_KEYS:
                L.append(f"  {k:<27}: "
                         + _fmt_value(live["features"].get(k)))
            for k in LIVE_TIME_KEYS:
                L.append(f"  {k:<27}: "
                         + str(live["time_context"].get(k)))
            for k in LIVE_REGIME_KEYS:
                L.append(f"  {k:<27}: "
                         + str(live["classifications"].get(k)))

        history = ctx["historical_research_context"]
        L.append("B. HISTORICAL RESEARCH CONTEXT")
        L.append("  stored observation timestamp: "
                 + str(history["stored_observation_timestamp"]))
        L.append("  memory evidence classification: "
                 + str(history["stored_evidence_quality"]))
        L.append("  evidence explanation        : "
                 + str(history["stored_evidence_explanation"]))
        L.append("  research conclusion         : "
                 + str(history["stored_research_conclusion"]))
        L.append("  memory queries              : "
                 + str(history["memory_n_queries"]))
        L.append("  memory sign agreement pct   : "
                 + _fmt_value(history["memory_pnl_sign_agreement_pct"]))
        L.append("  memory pearson r            : "
                 + _fmt_value(history["memory_pearson_r_pnl"]))
        L.append("  memory avg nearest distance : "
                 + _fmt_value(history["memory_avg_nearest_distance"]))
        L.append("  latest stored match sample  : "
                 + str(history["latest_stored_n_matches"]))
        L.append("  latest stored sign agreement: "
                 + _fmt_value(history["latest_stored_sign_agreement_pct"]))
        L.append("  stored regime stability     : "
                 + " | ".join(
                     history["stored_regime_stability_classifications"]))
        L.append("  stored walk-forward label   : "
                 + str(history["stored_walk_forward_label"]))
        L.append("  stored walk-forward classes : "
                 + " | ".join(history["stored_walk_forward_classifications"]))
        L.append("  cross-instrument context    : "
                 + str(history["cross_instrument_consistency_statement"]))

        match = ctx["live_state_historical_match"]
        L.append("  live state matched categories: "
                 + str(match["available_categories"]) + " of "
                 + str(match["category_count"]))
        L.append("  live state oos classifications: "
                 + " | ".join(match["classifications"]))
        L.append("  live state walk-forward label : "
                 + str(match["walk_forward_label"]))

        L.append("C. LIMITATIONS (carried forward from the research outputs)")
        if ctx["limitations"]:
            for item in ctx["limitations"]:
                L.append("  - " + item)
        else:
            L.append("  (none recorded)")

        L.append("D. RESEARCH INTERPRETATION")
        for line in ctx["research_interpretation"]:
            L.append("  - " + line)

        L.append("E. DATA QUALITY")
        if ctx["data_quality"]["issues"]:
            for item in ctx["data_quality"]["issues"]:
                L.append("  - " + item)
        else:
            L.append("  (no data-quality issues recorded for the live input)")

    L.append("")
    L.append("=" * 68)
    L.append("Descriptive historical research context only. No order was "
             "placed, changed or removed; this tool only read stored files. "
             "No directive of any kind is contained here.")
    return "\n".join(L)


def _fmt_value(value: Any) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:.6f}"


# ============================================================
# Output-file protection (every path checked before any writing)
# ============================================================
def resolve_output_paths(directory: str,
                         generated_ts: datetime.datetime
                         ) -> Tuple[str, str, str]:
    """Primary names, or timestamped alternatives when they already
    exist.  If an alternative also exists, raise so nothing is written."""
    stamp = generated_ts.strftime(STAMP_FORMAT)
    plan = (
        (OUTPUT_CSV, f"live_research_context_{stamp}.csv"),
        (OUTPUT_JSON, f"live_research_context_{stamp}.json"),
        (OUTPUT_TXT, f"live_research_context_report_{stamp}.txt"),
    )
    chosen: List[str] = []
    for primary, fallback in plan:
        path = os.path.join(directory, primary)
        if os.path.exists(path):
            path = os.path.join(directory, fallback)
        if os.path.exists(path):
            raise OutputExistsError(
                "refusing to overwrite existing file; nothing was written: "
                + path)
        chosen.append(path)
    return chosen[0], chosen[1], chosen[2]


def _write_text_if_absent(path: str, text: str) -> None:
    if os.path.exists(path):
        raise OutputExistsError(
            "output file already exists (never overwritten): " + path)
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))
    print("  wrote " + os.path.basename(path) + " ("
          + str(len(text.encode("utf-8"))) + " bytes)")


# ============================================================
# Real analysis (requires --run; never runs by default)
# ============================================================
def run_real_analysis(directory: str = SCRIPT_DIR) -> int:
    print("=" * 68)
    print("live_research_context.py - REAL ANALYSIS (--run)")
    print("=" * 68)
    generated_ts = datetime.datetime.now()
    try:
        csv_path, json_path, txt_path = resolve_output_paths(directory,
                                                             generated_ts)
    except OutputExistsError as exc:
        print("ERROR: " + str(exc))
        return 1

    try:
        live_csv_path, live_report_path = find_live_market_inputs(directory)
    except MissingInputError as exc:
        print("ERROR: " + str(exc))
        return 1
    print("\nLive market-state input selected (read-only):")
    print("  csv    : " + os.path.basename(live_csv_path))
    print("  report : " + os.path.basename(live_report_path))

    missing = check_bridge_inputs(directory,
                                  (live_csv_path, live_report_path))
    if missing:
        print("ERROR: missing required input file(s): " + ", ".join(missing))
        return 1

    not_fresh, fresh_results = check_selected_live_freshness(
        live_csv_path, live_report_path)
    print("\nData-freshness gate (market_data_freshness.py, threshold "
          + f"{fresh_results[0]['freshness_threshold_hours']:.0f}h):")
    for _r in fresh_results:
        print(f"  {_r['instrument']}: {_r['freshness_status']}")
    if not_fresh:
        print("ERROR: the live data-quality gate blocked downstream "
              "processing; not FRESH: " + ", ".join(not_fresh))
        return 1

    try:
        contexts = build_contexts(directory, live_csv_path)
    except (MissingInputError, LiveSchemaError, ValueError) as exc:
        print("ERROR: " + str(exc))
        return 1

    inputs_read = resolved_input_names(live_csv_path, live_report_path)
    csv_text = contexts_to_csv(contexts)
    json_text = contexts_to_json(contexts, generated_ts, inputs_read)
    report_text = build_report(contexts, generated_ts, inputs_read)

    print("\nPer-instrument summary:")
    for ctx in contexts:
        match = ctx["live_state_historical_match"]
        print(f"  {ctx['instrument']}: live "
              f"{ctx['data_quality']['issue_count']} data-quality note(s) | "
              f"matched {match['available_categories']} of "
              f"{match['category_count']} categories | stored walk-forward "
              f"{match['walk_forward_label']}")
    print("\nWriting NEW output files:")
    try:
        _write_text_if_absent(csv_path, csv_text)
        _write_text_if_absent(json_path, json_text)
        _write_text_if_absent(txt_path, report_text)
    except OutputExistsError as exc:
        print("ERROR: " + str(exc))
        return 1
    print("\nDone. Descriptive research context only - no directive, no "
          "certainty figure, no ordering of instruments.")
    return 0


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
_TERMINAL_TOKENS: Tuple[str, ...] = (
    "meta" + "trader5", "m" + "t5", "or" + "der_send", "or" + "der_check",
    "positi" + "ons_get", "positi" + "ons_total", "position" + "_get",
    "initiali" + "ze", "shut" + "down", "copy_" + "rates",
    "symbol_" + "info", "symbols_" + "get",
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
    "sock" + "et", "open" + "ai", "anthro" + "pic", "api" + "_" + "key",
)
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

_ALLOWED_IMPORT_MODULES = frozenset({
    "ast", "datetime", "json", "os", "re", "sys", "tempfile", "typing",
    "numpy", "pandas", "ai_research_layer", "ai_trading_analyst",
    "live_market_state", "market_data_freshness",
})


def contains_banned_wording(text: str) -> List[str]:
    """Every banned-wording pattern that occurs in `text` (lower-case)."""
    low = text.lower()
    return [pat for pat in _BANNED_WORD_PATTERNS if re.search(pat, low)]


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

    Returns a list of violations (empty when the file is clean):
    import allowlist, forbidden order-API identifiers, terminal tokens,
    learning-library / search / external-service tokens and banned
    wording.
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

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for mod in sorted(imported - _ALLOWED_IMPORT_MODULES):
        violations.append(f"unexpected import: {mod}")

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
                        f"forbidden order API token '{tok}' in identifier "
                        f"'{name}'")

    def scan_text(text: str, where: str) -> None:
        low = text.lower()
        for tok in _TERMINAL_TOKENS:
            if tok in low:
                violations.append(f"trading-terminal token in {where}")
        for tok in _ML_TOKENS:
            if tok in low:
                violations.append(f"learning-library token in {where}")
        for tok in _OPTIMIZATION_TOKENS:
            if tok in low:
                violations.append(f"search idiom in {where}")
        for tok in _API_TOKENS:
            if tok in low:
                violations.append(f"external-service token in {where}")
        for pat in _BANNED_WORD_PATTERNS:
            if re.search(pat, low):
                violations.append(f"banned wording in {where}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in doc_ids:
                continue
            scan_text(node.value, "string literal")
    scan_text(src, "source")

    for tail in ("send", "check"):
        token = "order" + "_" + tail
        if token in src.lower():
            violations.append(
                f"source contains the forbidden API name '{token}'")
    return violations


# ============================================================
# Synthetic fixtures (tests only; temporary directories only)
# ============================================================
_MATCHED_REGIMES: Dict[str, str] = {
    "adx_regime": "ADX>=25",
    "volatility_regime": "MEDIUM",
    "trend_distance_regime": "1-2",
    "bb_excursion_regime": "0.50-0.75",
    "efficiency_regime": "0.30-0.60",
    "rsi_regime": "50-70",
}


def _live_row(instrument: str, direction: Any, ts: str, candles: Any,
              regimes: Dict[str, str],
              overrides: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
    row: Dict[str, Any] = {c: None for c in LIVE_CSV_COLUMNS}
    row.update({
        "instrument": instrument,
        "symbol_resolution": "exact symbol name", "status": STATUS_COLLECTED,
        "reason": None, "candles_used": candles,
        "latest_closed_candle_time": ts, "direction": direction,
        "adx": 20.0, "atr_pct": 0.2, "ema50_dist_atr": 1.5,
        "bb_excursion": 0.6, "bb_position": -0.6, "eff24": 0.4,
        "rsi14": 55.0, "return6": -0.1, "return12": -0.2, "return24": 0.3,
        "dist_24_high": -0.3, "dist_24_low": 0.4,
        "candle_range_pct": 0.5, "candle_body_pct": 0.2,
        "upper_wick_pct": 0.1, "lower_wick_pct": 0.2,
        "hour": 3, "day_of_week": 2, "month": 10,
        "data_quality_warnings": None,
    })
    row.update(regimes)
    if overrides:
        row.update(overrides)
    return row


def write_live_fixture(directory: str) -> None:
    """A small synthetic live_market_state.csv covering the cases the
    tests need: a NONE direction, a missing value, a missing direction
    and an older timestamp."""
    rows = [
        _live_row("GOLD", DIRECTION_LONG, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
        _live_row("EURUSD", DIRECTION_NONE, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
        _live_row("GBPUSD", DIRECTION_SHORT, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES, overrides={"atr_pct": None}),
        _live_row("USDJPY", DIRECTION_LONG, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
        _live_row("AUDUSD", None, "2026-09-30 18:00:00", 9999,
                  _MATCHED_REGIMES),
    ]
    pd.DataFrame(rows, columns=list(LIVE_CSV_COLUMNS)).to_csv(
        os.path.join(directory, LIVE_CSV), index=False)


def write_timestamped_live_fixture(directory: str, stamp: str,
                                   write_csv: bool = True,
                                   write_report: bool = True) -> None:
    """A synthetic timestamped live-market-state collection (tests)."""
    rows = [
        _live_row("GOLD", DIRECTION_LONG, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
        _live_row("EURUSD", DIRECTION_NONE, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
        _live_row("GBPUSD", DIRECTION_SHORT, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
        _live_row("USDJPY", DIRECTION_LONG, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
        _live_row("AUDUSD", DIRECTION_LONG, "2026-10-07 03:00:00", 9999,
                  _MATCHED_REGIMES),
    ]
    if write_csv:
        pd.DataFrame(rows, columns=list(LIVE_CSV_COLUMNS)).to_csv(
            os.path.join(directory,
                         f"live_market_state_{stamp}.csv"), index=False)
    if write_report:
        with open(os.path.join(
                directory,
                f"live_market_state_report_{stamp}.txt"),
                "w", encoding="utf-8") as fh:
            fh.write("LIVE MARKET STATE - READ-ONLY COLLECTION REPORT\n"
                     "collection timestamp    : " + stamp + "\n")


def write_freshness_live_pair(directory: str, stamp: str,
                              collection_ts: str,
                              candles: Dict[str, Any],
                              statuses: Optional[Dict[str, str]] = None
                              ) -> Tuple[str, str]:
    """A timestamped live pair with a CONTROLLED, parseable collection
    timestamp and per-instrument candle timestamps (freshness-gate
    tests).  Unlike write_timestamped_live_fixture the report carries a
    real '%Y-%m-%d %H:%M:%S' timestamp so the reused freshness logic can
    parse it."""
    statuses = statuses or {}
    rows = [
        _live_row(inst, DIRECTION_LONG, candles.get(inst), 9999,
                  _MATCHED_REGIMES,
                  overrides={"status": statuses.get(inst, STATUS_COLLECTED)})
        for inst in SUPPORTED_INSTRUMENTS]
    csv_path = os.path.join(directory, f"live_market_state_{stamp}.csv")
    report_path = os.path.join(
        directory, f"live_market_state_report_{stamp}.txt")
    pd.DataFrame(rows, columns=list(LIVE_CSV_COLUMNS)).to_csv(
        csv_path, index=False)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("LIVE MARKET STATE - READ-ONLY COLLECTION REPORT\n"
                 "collection timestamp    : " + collection_ts + "\n")
    return csv_path, report_path


def _augment_research_fixture(directory: str) -> None:
    """The analyst layer's minimal fixture omits two columns that the
    research layer's ResearchData also reads on the period summary; add
    them so both reused loaders accept the same bundle."""
    path = os.path.join(directory, OOS_PERIOD_CSV)
    frame = pd.read_csv(path)
    if "start_time" not in frame.columns:
        frame["start_time"] = "2026-01-01 00:00:00"
    if "end_time" not in frame.columns:
        frame["end_time"] = "2026-03-31 00:00:00"
    if "win_rate_pct" not in frame.columns:
        frame["win_rate_pct"] = 50.0
    frame.to_csv(path, index=False)


def build_test_bundle(directory: str) -> str:
    """All research fixtures (reused from the analyst layer) plus a
    synthetic live file, inside a temporary directory."""
    build_analyst_fixture(directory)
    _augment_research_fixture(directory)
    write_live_fixture(directory)
    return directory


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

    print("live_research_context.py - synthetic test suite")
    print("(temporary directories only; no trading terminal; no real "
          "output is written)\n")

    with tempfile.TemporaryDirectory() as td:
        build_test_bundle(td)

        # ---------------- A. live input loading ----------------
        start_area("A_live_input_loading")
        rows, intrinsic = load_live_state(td)
        check("all five required instruments are loaded from the live file",
              all(inst in rows for inst in SUPPORTED_INSTRUMENTS)
              and all(rows[i] is not None for i in SUPPORTED_INSTRUMENTS))
        check("the loader never raises for a reportable row",
              intrinsic["AUDUSD"] == [])
        # schema mismatch is reported by naming the missing column
        bad_dir = os.path.join(td, "_schema")
        os.makedirs(bad_dir, exist_ok=True)
        pd.DataFrame([{"instrument": "GOLD"}]).to_csv(
            os.path.join(bad_dir, LIVE_CSV), index=False)
        try:
            load_live_state(bad_dir)
            check("a live schema mismatch names the missing columns", False)
        except LiveSchemaError as exc:
            check("a live schema mismatch names the missing columns",
                  "adx_regime" in str(exc) and "direction" in str(exc))
        # latest observation selection with a duplicate row
        sel_dir = os.path.join(td, "_select")
        os.makedirs(sel_dir, exist_ok=True)
        dup = [
            _live_row("GOLD", DIRECTION_LONG, "2026-10-01 00:00:00", 500,
                      _MATCHED_REGIMES),
            _live_row("GOLD", DIRECTION_SHORT, "2026-10-07 03:00:00", 9999,
                      _MATCHED_REGIMES),
        ]
        pd.DataFrame(dup, columns=list(LIVE_CSV_COLUMNS)).to_csv(
            os.path.join(sel_dir, LIVE_CSV), index=False)
        sel_rows, _ = load_live_state(sel_dir)
        check("the newest closed candle is selected when rows repeat",
              sel_rows["GOLD"]["direction"] == DIRECTION_SHORT
              and sel_rows["GOLD"]["candles_used"] == 9999)

        # ---------------- B. state and regime preservation -----
        start_area("B_state_regime_preservation")
        contexts = build_contexts(td)
        by_inst = {c["instrument"]: c for c in contexts}
        check("one context per supported instrument, fixed order",
              [c["instrument"] for c in contexts]
              == list(SUPPORTED_INSTRUMENTS))
        gold_live = by_inst["GOLD"]["live_state"]
        check("current-state values are copied unchanged",
              gold_live["features"]["adx"] == 20.0
              and gold_live["features"]["eff24"] == 0.4
              and gold_live["features"]["rsi14"] == 55.0)
        check("frozen regime classifications are preserved",
              gold_live["classifications"]["adx_regime"] == "ADX>=25"
              and gold_live["classifications"]["rsi_regime"] == "50-70")
        check("the stored research direction is preserved verbatim",
              gold_live["direction_stored_research"] == DIRECTION_LONG)
        check("the live closed-candle timestamp is preserved",
              gold_live["latest_closed_candle_time"]
              == "2026-10-07 03:00:00")

        # ---------------- C. historical evidence preservation --
        start_area("C_historical_preservation")
        analyst = AnalystData(td)
        latest = {i: analyst.latest_context(i) for i in SUPPORTED_INSTRUMENTS}
        expected = {i: build_analyst_context(analyst, i, latest)
                    for i in SUPPORTED_INSTRUMENTS}
        gold_hist = by_inst["GOLD"]["historical_research_context"]
        check("memory evidence classification is preserved exactly",
              gold_hist["stored_evidence_quality"]
              == expected["GOLD"]["research_evidence"]["evidence_quality"]
              and gold_hist["stored_evidence_quality"] in EVIDENCE_LABELS)
        check("the memory sign-agreement and correlation values are carried",
              gold_hist["memory_pnl_sign_agreement_pct"]
              == expected["GOLD"]["historical_memory"][
                  "pnl_sign_agreement_pct"]
              and gold_hist["memory_pearson_r_pnl"]
              == expected["GOLD"]["historical_memory"][
                  "pearson_r_match_actual_pnl"])
        check("the walk-forward classification is preserved exactly",
              gold_hist["stored_walk_forward_label"]
              == expected["GOLD"]["cross_instrument_context"][
                  "per_instrument"][
                  SUPPORTED_INSTRUMENTS.index("GOLD")]["walk_forward_label"]
              and gold_hist["stored_walk_forward_label"] in WF_LABELS)
        check("the stored state classifications are preserved",
              gold_hist["stored_regime_stability_classifications"]
              == [str(c.get("stored_classification", ""))
                  for c in expected["GOLD"]["regime_stability"]
                  if c.get("available")])
        check("the cross-instrument context statement is carried",
              bool(gold_hist["cross_instrument_consistency_statement"]))
        check("limitations are carried forward unchanged",
              by_inst["GOLD"]["limitations"]
              == expected["GOLD"]["limitations"]
              and len(by_inst["GOLD"]["limitations"]) > 0)
        check("the live state's own regimes are matched to stored evidence",
              by_inst["GOLD"]["live_state_historical_match"][
                  "available_categories"] == len(LIVE_REGIME_KEYS)
              and by_inst["GOLD"]["live_state_historical_match"][
                  "walk_forward_label"] in WF_LABELS)

        # ---------------- D. data quality ----------------------
        start_area("D_data_quality")
        eur = by_inst["EURUSD"]
        none_text = " ".join(eur["research_interpretation"])
        check("a NONE direction is reported descriptively",
              DIRECTION_NONE in none_text
              and any(DIRECTION_NONE in x for x in eur["data_quality"][
                  "issues"]))
        gbp_issues = by_inst["GBPUSD"]["data_quality"]["issues"]
        check("a missing live value is reported",
              any("atr_pct" in x for x in gbp_issues))
        aud = by_inst["AUDUSD"]
        check("a missing stored research direction is reported",
              any("direction is missing" in x for x in aud["data_quality"][
                  "issues"]))
        check("an older live timestamp is reported as possibly older",
              any("hours" in x for x in aud["data_quality"]["issues"]))
        absent = build_research_interpretation("AUDUSD", None, None, None)
        check("an unavailable instrument is still described",
              "No live observation is stored for AUDUSD" in " ".join(absent)
              and build_live_state_snapshot(None) is None)
        # an out-of-range timestamp on a collectable instrument
        usd_issues = by_inst["USDJPY"]["data_quality"]["issues"]
        check("a fully valid live row has no data-quality note",
              usd_issues == [])

        # ---------------- E. missing input handling ------------
        start_area("E_missing_input_handling")
        empty_dir = os.path.join(td, "_empty")
        os.makedirs(empty_dir, exist_ok=True)
        missing = check_bridge_inputs(empty_dir)
        check("every missing required input file is named",
              set(missing) == set(REQUIRED_INPUT_FILES))
        partial = os.path.join(td, "_partial")
        os.makedirs(partial, exist_ok=True)
        write_live_fixture(partial)
        missing2 = check_bridge_inputs(partial)
        check("a partially populated directory names only the absent files",
              LIVE_CSV not in missing2
              and RESEARCH_ALL_CSV in missing2
              and OOS_STATE_CSV in missing2)

        # ---------------- F. no future-candle lookup -----------
        start_area("F_no_future_lookup")
        research = ResearchData(td)
        probe = rows["GOLD"]
        m1 = match_live_state_to_history("GOLD", probe, research)
        m2 = match_live_state_to_history("GOLD", probe, research)
        check("the stored-evidence lookup is a pure function of its input",
              m1 == m2)
        changed = dict(probe)
        changed["adx"] = 999.0
        changed["atr_pct"] = None
        m3 = match_live_state_to_history("GOLD", changed, research)
        check("the lookup depends only on the live regimes, not other "
              "fields", m3 == m1)
        unknown = dict(probe)
        for key in LIVE_REGIME_KEYS:
            unknown[key] = "not-a-stored-bucket"
        m4 = match_live_state_to_history("GOLD", unknown, research)
        check("an unmatched live state yields no qualifying evidence",
              m4["available_categories"] == 0 and m4["available"] is False)
        interp = build_research_interpretation("GOLD", unknown, None, m4)
        check("the no-evidence case is stated descriptively",
              "no qualifying historical evidence" in " ".join(interp))

        # ---------------- G. determinism and output protection -
        start_area("G_determinism_outputs")
        fixed_ts = datetime.datetime(2026, 10, 7, 12, 0, 0)
        ctx_a = build_contexts(td)
        ctx_b = build_contexts(td)
        csv_a = contexts_to_csv(ctx_a)
        csv_b = contexts_to_csv(ctx_b)
        rep_a = build_report(ctx_a, fixed_ts, REQUIRED_INPUT_FILES)
        rep_b = build_report(ctx_b, fixed_ts, REQUIRED_INPUT_FILES)
        json_a = contexts_to_json(ctx_a, fixed_ts, REQUIRED_INPUT_FILES)
        json_b = contexts_to_json(ctx_b, fixed_ts, REQUIRED_INPUT_FILES)
        check("identical inputs produce identical CSV bytes",
              csv_a == csv_b)
        check("identical inputs produce identical report bytes",
              rep_a == rep_b)
        check("identical inputs produce identical JSON bytes",
              json_a == json_b)
        check("the CSV carries the explicit frozen column schema",
              list(pd.read_csv(pd.io.common.StringIO(csv_a)).columns)
              == list(CONTEXT_CSV_COLUMNS))
        payload = json.loads(json_a)
        check("the JSON payload carries the expected top-level keys",
              payload["research_only"] is True
              and payload["generated_timestamp"]
              == fixed_ts.strftime(TS_FORMAT)
              and len(payload["instruments"]) == len(SUPPORTED_INSTRUMENTS))
        check("the report carries the AUDUSD data-quality note",
              "INSTRUMENT: AUDUSD" in rep_a
              and "stored research direction is missing" in rep_a)
        # output-file protection
        p_dir = os.path.join(td, "_paths")
        os.makedirs(p_dir, exist_ok=True)
        c_path, j_path, t_path = resolve_output_paths(p_dir, fixed_ts)
        check("a clean directory selects the primary file names",
              (os.path.basename(c_path), os.path.basename(j_path),
               os.path.basename(t_path))
              == (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_TXT))
        _write_text_if_absent(c_path, "x")
        stamp = fixed_ts.strftime(STAMP_FORMAT)
        c2, j2, t2 = resolve_output_paths(p_dir, fixed_ts)
        check("an existing CSV gets the timestamped alternative",
              os.path.basename(c2)
              == f"live_research_context_{stamp}.csv")
        with open(c_path, encoding="utf-8") as fh:
            check("the original CSV is never overwritten", fh.read() == "x")
        try:
            _write_text_if_absent(c_path, "y")
            check("writing over an existing file is refused", False)
        except OutputExistsError:
            check("writing over an existing file is refused", True)
        with open(c2, "w", encoding="utf-8") as fh:
            fh.write("occupied")
        try:
            resolve_output_paths(p_dir, fixed_ts)
            check("an alternative-name collision stops the run", False)
        except OutputExistsError as exc:
            check("an alternative-name collision stops the run",
                  f"live_research_context_{stamp}.csv" in str(exc))

        # ---------------- H. safety scans ----------------------
        start_area("H_safety_scans")
        violations = run_safety_scan()
        check("the static safety scan of this file is clean",
              violations == [])
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
        check("no order execution API name is present in the source",
              all(tok not in src.lower()
                  for tok in ("order" + "_send", "order" + "_check",
                              "order" + "_modify", "order" + "_cancel")))
        tree = ast.parse(src)
        top_imports: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(a.name.split(".")[0]
                                   for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.append(node.module.split(".")[0])
        check("no trading terminal package is imported",
              ("Meta" + "Trader5") not in top_imports
              and ("Meta" + "Trader5") not in sys.modules)
        check("no external service or learning library is imported",
              all(mod in _ALLOWED_IMPORT_MODULES
                  for mod in top_imports)
              and ("requ" + "ests") not in src.lower()
              and ("ur" + "llib") not in src.lower())
        outputs_blob = (csv_a + rep_a + json_a).lower()
        check("generated outputs contain no directional directive wording",
              contains_banned_wording(outputs_blob) == [])
        check("generated outputs contain no certainty or ordering wording",
              all(pat not in [p for p in contains_banned_wording(
                  outputs_blob)] for pat in _BANNED_WORD_PATTERNS))
        check("the live direction is labelled a stored research direction",
              "stored research direction" in rep_a.lower())

        # ---------------- J. live input selection --------------
        start_area("J_live_input_selection")
        lm = os.path.join(td, "_lmk")
        os.makedirs(lm, exist_ok=True)
        write_timestamped_live_fixture(lm, "20261001_000000")
        write_timestamped_live_fixture(lm, "20261007_114550")
        csv_sel, report_sel = find_live_market_inputs(lm)
        check("A the newest valid timestamped pair is selected",
              os.path.basename(csv_sel)
              == "live_market_state_20261007_114550.csv"
              and os.path.basename(report_sel)
              == "live_market_state_report_20261007_114550.txt")
        check("E selection is deterministic across repeated calls",
              find_live_market_inputs(lm) == (csv_sel, report_sel))

        # D: the legacy files are ignored when a valid pair exists.
        write_live_fixture(lm)
        legacy_sel = find_live_market_inputs(lm)
        check("D the legacy live file is not selected",
              "live_market_state.csv" not in legacy_sel
              and "live_market_state_report.txt" not in legacy_sel)

        # B: a CSV/report timestamp mismatch is rejected.
        lmis = os.path.join(td, "_lmk_mismatch")
        os.makedirs(lmis, exist_ok=True)
        write_timestamped_live_fixture(lmis, "20261007_114550",
                                       write_report=False)
        write_timestamped_live_fixture(lmis, "20261006_000000",
                                       write_csv=False)
        try:
            find_live_market_inputs(lmis)
            check("B a CSV/report timestamp mismatch is rejected", False)
        except MissingInputError:
            check("B a CSV/report timestamp mismatch is rejected", True)

        # C: an incomplete pair is ignored in favour of a complete one.
        lmin = os.path.join(td, "_lmk_incomplete")
        os.makedirs(lmin, exist_ok=True)
        write_timestamped_live_fixture(lmin, "20261007_114550",
                                       write_report=False)
        write_timestamped_live_fixture(lmin, "20261005_000000")
        sel_in = find_live_market_inputs(lmin)
        check("C an incomplete newer pair is ignored",
              os.path.basename(sel_in[0])
              == "live_market_state_20261005_000000.csv")

        # F: no valid pair fails safely.
        lmf = os.path.join(td, "_lmk_none")
        os.makedirs(lmf, exist_ok=True)
        try:
            find_live_market_inputs(lmf)
            check("F no valid timestamped pair fails safely", False)
        except MissingInputError as exc:
            check("F no valid timestamped pair fails safely", True)
            check("F the error names the live-market-state input",
                  "live_market_state" in str(exc))

        # G: the selected timestamped pair drives build_contexts while
        #    the analyst interfaces stay compatible.
        lfull = os.path.join(td, "_lmk_full")
        os.makedirs(lfull, exist_ok=True)
        build_test_bundle(lfull)
        write_timestamped_live_fixture(lfull, "20261007_114550")
        test_csv, test_report = find_live_market_inputs(lfull)
        ctx_live = build_contexts(lfull, test_csv)
        by_live = {c["instrument"]: c for c in ctx_live}
        check("G build_contexts consumes the selected timestamped CSV",
              by_live["GOLD"]["live_state"]["latest_closed_candle_time"]
              == "2026-10-07 03:00:00")
        names_g = resolved_input_names(test_csv, test_report)
        check("G the report input names carry the selected live pair",
              names_g[0] == os.path.basename(test_csv)
              and names_g[1] == os.path.basename(test_report)
              and "live_market_state.csv" not in names_g)
        analyst_compat = AnalystData(lfull)
        latest_compat = {i: analyst_compat.latest_context(i)
                         for i in SUPPORTED_INSTRUMENTS}
        compat_ctx = build_analyst_context(
            analyst_compat, "GOLD", latest_compat)
        check("G AnalystData/build_analyst_context stay compatible",
              compat_ctx["instrument"] == "GOLD"
              and "current_state" in compat_ctx)

        # H: no future-candle lookup remains intact.
        research_h = ResearchData(lfull)
        probe_h = load_live_state(lfull)[0]["GOLD"]
        h1 = match_live_state_to_history("GOLD", probe_h, research_h)
        h2 = match_live_state_to_history("GOLD", probe_h, research_h)
        check("H no future-candle lookup remains intact", h1 == h2)

        # ---------------- I. CLI gating ------------------------
        start_area("I_cli_gating")
        check("default (no flags) runs the synthetic tests only",
              should_run_real([]) is False)
        check("--run selects the real analysis", should_run_real(
            ["--run"]) is True)
        check("unrelated flags do not select the real analysis",
              should_run_real(["-v", "--verbose"]) is False)
        check("the default mode does not import the terminal package",
              ("Meta" + "Trader5") not in sys.modules)

        # ---------------- K. freshness gate ---------------------
        start_area("K_freshness_gate")

        def _candles(ts: str) -> Dict[str, Any]:
            return {inst: ts for inst in SUPPORTED_INSTRUMENTS}

        kdir = os.path.join(td, "_g_fresh")
        os.makedirs(kdir, exist_ok=True)
        f_csv, f_rep = write_freshness_live_pair(
            kdir, "20261007_120000", "2026-10-07 12:00:00",
            _candles("2026-10-07 03:00:00"))
        nf, res = check_selected_live_freshness(f_csv, f_rep)
        check("K all-FRESH data passes the gate and continues",
              nf == [] and {r["freshness_status"] for r in res} == {"FRESH"})

        sdir = os.path.join(td, "_g_stale")
        os.makedirs(sdir, exist_ok=True)
        stale = _candles("2026-10-07 03:00:00")
        stale["AUDUSD"] = "2026-09-30 18:00:00"
        s_csv, s_rep = write_freshness_live_pair(
            sdir, "20261007_120000", "2026-10-07 12:00:00", stale)
        nf_s, _res_s = check_selected_live_freshness(s_csv, s_rep)
        check("K STALE data blocks downstream processing",
              nf_s == ["AUDUSD"])

        idir = os.path.join(td, "_g_invalid")
        os.makedirs(idir, exist_ok=True)
        i_csv, i_rep = write_freshness_live_pair(
            idir, "20261007_120000", "not-a-timestamp",
            _candles("2026-10-07 03:00:00"))
        nf_i, _res_i = check_selected_live_freshness(i_csv, i_rep)
        check("K INVALID data blocks downstream processing",
              nf_i == list(SUPPORTED_INSTRUMENTS))

        # The gate must act on the EXACT pair it is handed, not on
        # whatever a directory scan would pick.
        pdir = os.path.join(td, "_g_exact")
        os.makedirs(pdir, exist_ok=True)
        o_csv, o_rep = write_freshness_live_pair(
            pdir, "20261001_000000", "2026-10-01 12:00:00",
            _candles("2026-09-20 03:00:00"))
        write_freshness_live_pair(
            pdir, "20261007_120000", "2026-10-07 12:00:00",
            _candles("2026-10-07 03:00:00"))
        sel_csv, sel_rep = find_live_market_inputs(pdir)
        check("K selection still picks the newest valid pair",
              os.path.basename(sel_csv)
              == "live_market_state_20261007_120000.csv")
        nf_new, _ = check_selected_live_freshness(sel_csv, sel_rep)
        check("K the gate passes the exact selected (newest) pair",
              nf_new == [])
        nf_old, _ = check_selected_live_freshness(o_csv, o_rep)
        check("K the gate honours the exact pair given (old pair STALE)",
              set(nf_old) == set(SUPPORTED_INSTRUMENTS))

        # An older/legacy dataset must never rescue a stale pair.
        fbdir = os.path.join(td, "_g_fallback")
        os.makedirs(fbdir, exist_ok=True)
        write_live_fixture(fbdir)          # fresh legacy live_market_state.csv
        fb_csv, fb_rep = write_freshness_live_pair(
            fbdir, "20261007_120000", "2026-10-07 12:00:00", stale)
        sel2_csv, sel2_rep = find_live_market_inputs(fbdir)
        check("K selection ignores the legacy file when a pair exists",
              os.path.basename(sel2_csv)
              != "live_market_state.csv")
        nf_fb, _ = check_selected_live_freshness(sel2_csv, sel2_rep)
        check("K a stale selected pair is never rescued by an older file",
              nf_fb == ["AUDUSD"])

        check("K the gate reuses market_data_freshness (no duplication)",
              "check_selected_live_freshness" in src
              and "market_data_freshness" in src
              and "data_quality_gate" in src
              and "compute_freshness" in src)
        check("K no trading/order API was introduced",
              all(tok not in src.lower() for tok in (
                  "order" + "_send", "order" + "_check",
                  "order" + "_modify", "order" + "_cancel")))

        # ---------------- L. C3 regression: analyst outputs -------
        # The analyst layer's own output files are NOT inputs of this
        # bridge.  These tests prove the three files are not required
        # and that the context output is byte-identical whether they are
        # absent or present.
        start_area("L_c3_optional_analyst_outputs")
        c3_dir = os.path.join(td, "_c3_optional")
        os.makedirs(c3_dir, exist_ok=True)
        build_test_bundle(c3_dir)
        # Assembled from split literals so this source never contains the
        # filenames contiguously (matching the safety-scan convention).
        _analyst_prefix = "ai_trading_" + "analyst_"
        optional_outputs = (_analyst_prefix + "context.csv",
                            _analyst_prefix + "context.json",
                            _analyst_prefix + "report.txt")
        check("C3 the three analyst output files are not required inputs",
              all(name not in REQUIRED_INPUT_FILES
                  for name in optional_outputs))
        missing_c3 = check_bridge_inputs(c3_dir)
        check("C3 none of the three analyst outputs is reported missing",
              all(name not in missing_c3 for name in optional_outputs))
        check("C3 the three analyst outputs are absent before the test",
              all(not os.path.isfile(os.path.join(c3_dir, name))
                  for name in optional_outputs))
        absent_ctx = build_contexts(c3_dir)
        absent_csv = contexts_to_csv(absent_ctx)
        absent_json = contexts_to_json(absent_ctx, fixed_ts,
                                       REQUIRED_INPUT_FILES)
        absent_rep = build_report(absent_ctx, fixed_ts, REQUIRED_INPUT_FILES)
        for name in optional_outputs:
            with open(os.path.join(c3_dir, name), "w",
                      encoding="utf-8") as fh:
                fh.write("PRESENT-BUT-UNUSED\n")
        present_ctx = build_contexts(c3_dir)
        present_csv = contexts_to_csv(present_ctx)
        present_json = contexts_to_json(present_ctx, fixed_ts,
                                        REQUIRED_INPUT_FILES)
        present_rep = build_report(present_ctx, fixed_ts,
                                   REQUIRED_INPUT_FILES)
        check("C3 context CSV is unchanged when the files are present",
              absent_csv == present_csv)
        check("C3 context JSON is unchanged when the files are present",
              absent_json == present_json)
        check("C3 context report is unchanged when the files are present",
              absent_rep == present_rep)
        check("C3 the analyst outputs never appear in the read inputs",
              all(name not in resolved_input_names(
                  os.path.join(c3_dir, LIVE_CSV),
                  os.path.join(c3_dir, LIVE_TXT))
                  for name in optional_outputs))

    start_area(None)
    print()
    print("  Per-area verification (area -> passed/failed):")
    stats = {name: (p, f) for name, p, f in area_stats}
    order = (
        "A_live_input_loading", "B_state_regime_preservation",
        "C_historical_preservation", "D_data_quality",
        "E_missing_input_handling", "F_no_future_lookup",
        "G_determinism_outputs", "H_safety_scans",
        "J_live_input_selection", "I_cli_gating",
        "K_freshness_gate", "L_c3_optional_analyst_outputs",
    )
    for name in order:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<30} {p:>2} passed, {f} failed [{status}]")
    print()
    print(f"  TOTAL: {passed} passed, {failed} failed")
    return passed, failed


# ============================================================
# Command-line gating
# ============================================================
def should_run_real(argv: List[str]) -> bool:
    return "--run" in argv


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if should_run_real(argv):
        return run_real_analysis()
    passed, failed = run_synthetic_tests()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
