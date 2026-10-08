"""
market_data_freshness.py - READ-ONLY DATA-FRESHNESS SAFETY GATE.

WHAT THIS IS
    A deterministic, read-only data-quality component for the existing
    live market-state pipeline.  For each supported instrument it
    reports whether the latest CLOSED H1 observation is fresh enough for
    downstream research processing.

    It is a DATA-QUALITY GATE, never a trading filter.

WHAT THIS IS NOT
    - not a trading decision component of any kind,
    - it never touches a trading terminal,
    - it never declares anything preferable and never orders the
      instruments,
    - it modifies NO existing file: the live inputs are opened read-only
      and only the three NEW output files may be created,
    - it never replaces a stored timestamp with another value.

REUSED DEFINITIONS (nothing is re-created here)
    live_research_context.STALE_LAG_HOURS      the existing 72-hour rule
    live_market_state.INSTRUMENTS              the five instruments, order
    live_market_state.CSV_COLUMNS              the live-state schema
    live_market_state.STATUS_COLLECTED         the live source status term
    ai_trading_analyst.TS_FORMAT               the project timestamp format
    The live observation and its closed-candle timestamps are read from
    the existing live outputs; no alternative timestamp field is invented.

LIVE INPUT SELECTION (newest matching timestamped pair first)
    The live inputs are located automatically.  The NEWEST matching
    timestamped collection pair
        live_market_state_YYYYMMDD_HHMMSS.csv
        live_market_state_report_YYYYMMDD_HHMMSS.txt
    is used when one exists; the legacy untimestamped files
        live_market_state.csv
        live_market_state_report.txt
    are used only when no matching timestamped pair is available.  A
    timestamped CSV and a timestamped report are paired by their shared
    filename stamp; a timestamped file whose counterpart stamp is absent
    is skipped.

TIMESTAMP SOURCES (existing fields only)
    collection timestamp            : the chosen live collection report
                                      ("collection timestamp : ...")
    latest closed H1 candle         : the chosen live market-state CSV
                                      ("latest_closed_candle_time")
    The collection timestamp is always read from the report CONTENT; a
    filesystem modification time is never used as the freshness
    timestamp.
    The newer live_research_context outputs carry the live closed-candle
    timestamps too but no collection timestamp, so the live collection
    report remains the collection-time source.

TIMESTAMP SAFETY
    Every timestamp is normalised to a single reference frame (UTC, naive)
    before any age is calculated, so an aware timestamp is never compared
    incorrectly against a naive one.  A timestamp that is missing or
    unreadable yields the INVALID status with a descriptive reason; no
    value is ever substituted and an older timestamp is never treated as
    current.

FRESHNESS RULE (the existing 72-hour rule, preserved exactly)
    data_age_hours = collection_timestamp - latest_closed_candle_timestamp
    FRESH   : data age <= freshness_threshold_hours (default 72 hours)
    STALE   : data age >  freshness_threshold_hours
    INVALID : a timestamp is missing or unreadable

OUTPUT FILES (all NEW; every path is checked before anything is written;
    an existing file is never overwritten - a timestamped alternative is
    used instead, and a collision on that alternative stops the run):
    market_data_freshness.csv
    market_data_freshness.json
    market_data_freshness_report.txt

COMMAND-LINE SAFETY
    python market_data_freshness.py          -> synthetic tests only
                                                (no terminal contact, no
                                                external service, no real
                                                output written)
    python market_data_freshness.py --run    -> real freshness analysis

DISCLAIMER: descriptive data-quality information only. No causal claims,
no forward-looking statements, no directive of any kind. Educational use
only.
"""

# ============================================================
# Imports: Python standard library + pandas + the existing project
# modules whose frozen definitions are reused (no terminal access).
# ============================================================
import ast
import datetime
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai_trading_analyst import TS_FORMAT
from live_market_state import (
    CSV_COLUMNS as LIVE_CSV_COLUMNS,
    INSTRUMENTS,
    NUMERIC_STATE_KEYS,
    REGIME_KEYS,
    STATUS_COLLECTED,
    TIME_STATE_KEYS,
)
from live_research_context import STALE_LAG_HOURS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Frozen settings
# ============================================================
LIVE_CSV = "live_market_state.csv"
LIVE_REPORT = "live_market_state_report.txt"

# Timestamped collection names produced by live_market_state.py when the
# legacy primary name is already occupied.  A CSV and a report form one
# collection when their filename stamps are identical.
LIVE_CSV_STAMP_RE = re.compile(r"^live_market_state_(\d{8}_\d{6})\.csv$")
LIVE_REPORT_STAMP_RE = re.compile(
    r"^live_market_state_report_(\d{8}_\d{6})\.txt$")

OUTPUT_CSV = "market_data_freshness.csv"
OUTPUT_JSON = "market_data_freshness.json"
OUTPUT_TXT = "market_data_freshness_report.txt"
OUTPUT_FILES: Tuple[str, ...] = (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_TXT)

STAMP_FORMAT = "%Y%m%d_%H%M%S"

# The existing 72-hour rule, reused verbatim from the research bridge.
DEFAULT_FRESHNESS_THRESHOLD_HOURS: float = float(STALE_LAG_HOURS)

FRESH = "FRESH"
STALE = "STALE"
INVALID = "INVALID"
FRESHNESS_STATUSES: Tuple[str, ...] = (FRESH, STALE, INVALID)

# Fixed, documented classification text (no new threshold system).
FRESHNESS_RULE_TEXT = (
    "FRESH when the data age is at or below the freshness threshold; "
    "STALE when the data age is above it; INVALID when a timestamp is "
    "missing or unreadable")

REQUIRED_LIVE_COLUMNS: Tuple[str, ...] = (
    "instrument", "status", "latest_closed_candle_time")

CSV_COLUMNS: Tuple[str, ...] = (
    "instrument",
    "collection_timestamp",
    "latest_closed_candle_timestamp",
    "data_age_hours",
    "freshness_threshold_hours",
    "freshness_status",
    "source_status",
    "data_quality_note",
)


class MissingInputError(RuntimeError):
    """A required read-only input file is absent."""


class LiveSchemaError(RuntimeError):
    """live_market_state.csv does not carry the expected frozen schema."""


class OutputExistsError(RuntimeError):
    """An output path is already occupied; nothing may be written."""


# ============================================================
# Timestamp normalisation (timezone-safe)
# ============================================================
def normalise_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """Parse `value` into a single reference frame (UTC, naive).

    A timezone-aware value is converted to UTC and its zone dropped, so
    an aware value and its naive UTC equivalent compare equal.  A
    missing or unreadable value returns None (never a substitute).
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _fmt(ts: Optional[pd.Timestamp]) -> Optional[str]:
    return None if ts is None else ts.strftime(TS_FORMAT)


# ============================================================
# Read-only inputs
# ============================================================
def _scan_timestamped_inputs(directory: str
                             ) -> Tuple[Dict[str, str], Dict[str, str],
                                        List[str]]:
    """Map filename stamps to names for live CSVs and reports.

    Returns (csv_by_stamp, report_by_stamp, shared_stamps_oldest_first).
    A stamp present for only one of the two file kinds never appears in
    the shared list, so an unpaired file is skipped by the caller.
    """
    csv_by_stamp: Dict[str, str] = {}
    report_by_stamp: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return csv_by_stamp, report_by_stamp, []
    for name in sorted(os.listdir(directory)):
        csv_match = LIVE_CSV_STAMP_RE.match(name)
        if csv_match:
            csv_by_stamp.setdefault(csv_match.group(1), name)
            continue
        report_match = LIVE_REPORT_STAMP_RE.match(name)
        if report_match:
            report_by_stamp.setdefault(report_match.group(1), name)
    shared = sorted(set(csv_by_stamp) & set(report_by_stamp))
    return csv_by_stamp, report_by_stamp, shared


def find_live_inputs(directory: str = SCRIPT_DIR
                     ) -> Tuple[Optional[str], Optional[str]]:
    """Locate the live market-state CSV and collection report to read.

    The newest matching timestamped pair is used when one exists; the
    legacy untimestamped files are used only when no matching pair is
    present.  Each returned path is None when that file is absent.
    """
    csv_by_stamp, report_by_stamp, shared = _scan_timestamped_inputs(
        directory)
    if shared:
        stamp = shared[-1]
        return (os.path.join(directory, csv_by_stamp[stamp]),
                os.path.join(directory, report_by_stamp[stamp]))
    legacy_csv = os.path.join(directory, LIVE_CSV)
    legacy_report = os.path.join(directory, LIVE_REPORT)
    return (legacy_csv if os.path.isfile(legacy_csv) else None,
            legacy_report if os.path.isfile(legacy_report) else None)


def check_required_inputs(directory: str = SCRIPT_DIR) -> List[str]:
    """Names the missing live input file(s) (returned, not raised)."""
    csv_path, report_path = find_live_inputs(directory)
    missing: List[str] = []
    if csv_path is None:
        missing.append(LIVE_CSV)
    if report_path is None:
        missing.append(LIVE_REPORT)
    return missing


def read_collection_timestamp(report_path: str
                              ) -> Tuple[Optional[pd.Timestamp],
                                         Optional[str]]:
    """The live collection timestamp from the existing collection report."""
    if not os.path.isfile(report_path):
        return None, ("live collection report is absent, so the collection "
                      "timestamp could not be read")
    with open(report_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r"^\s*collection timestamp\s*:\s*(.+?)\s*$",
                      text, re.MULTILINE)
    if not match:
        return None, ("the collection timestamp line is absent from the "
                      "live collection report")
    ts = normalise_timestamp(match.group(1))
    if ts is None:
        return None, ("the collection timestamp in the live collection "
                      "report is unreadable")
    return ts, None


# ============================================================
# Freshness calculation (pure, deterministic)
# ============================================================
def compute_freshness(instrument: str,
                      collection_norm: Optional[pd.Timestamp],
                      candle_value: Any,
                      source_status: Any,
                      threshold_hours: float
                      ) -> Dict[str, Any]:
    """One instrument's freshness result from the two stored timestamps."""
    candle_norm = normalise_timestamp(candle_value)
    notes: List[str] = []
    age: Optional[float] = None

    if collection_norm is None:
        status = INVALID
        notes.append("collection timestamp is missing or unreadable, so "
                     "the data age cannot be determined")
    elif candle_norm is None:
        status = INVALID
        notes.append("latest closed candle timestamp is missing or "
                     "unreadable")
    else:
        age = (collection_norm - candle_norm).total_seconds() / 3600.0
        status = FRESH if age <= float(threshold_hours) else STALE
        notes.append(f"latest closed candle is {age:.1f} hours older than "
                     f"the collection timestamp")
        if age < 0:
            notes.append("latest closed candle timestamp is after the "
                         "collection timestamp (negative age); the stored "
                         "values are reported unchanged")

    if source_status is not None and not _text_is_missing(source_status) \
            and str(source_status) != STATUS_COLLECTED:
        notes.append("live collector status is " + str(source_status))

    return {
        "instrument": instrument,
        "collection_timestamp": _fmt(collection_norm),
        "latest_closed_candle_timestamp": _fmt(candle_norm),
        "data_age_hours": None if age is None else round(age, 6),
        "freshness_threshold_hours": float(threshold_hours),
        "freshness_status": status,
        "source_status": (None if _text_is_missing(source_status)
                          else str(source_status)),
        "data_quality_note": "; ".join(notes) if notes else None,
    }


def _text_is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def build_results(directory: str = SCRIPT_DIR,
                  threshold_hours: float = DEFAULT_FRESHNESS_THRESHOLD_HOURS,
                  instruments: Tuple[str, ...] = INSTRUMENTS
                  ) -> Tuple[List[Dict[str, Any]], Optional[pd.Timestamp],
                             Optional[str]]:
    """One freshness result per instrument, in the fixed order."""
    csv_path, report_path = find_live_inputs(directory)
    if csv_path is None or not os.path.isfile(csv_path):
        raise MissingInputError("missing required input file: " + LIVE_CSV)
    frame = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_LIVE_COLUMNS if c not in frame.columns]
    if missing:
        raise LiveSchemaError(
            LIVE_CSV + ": missing required column(s): " + ", ".join(missing))

    collection_norm, collection_note = read_collection_timestamp(
        report_path if report_path is not None
        else os.path.join(directory, LIVE_REPORT))

    results: List[Dict[str, Any]] = []
    for instrument in instruments:
        sub = frame[frame["instrument"].astype(str) == instrument]
        if len(sub) == 0:
            results.append(compute_freshness(
                instrument, collection_norm, None, None, threshold_hours))
            continue
        chosen: Optional[Dict[str, Any]] = None
        chosen_ts: Optional[pd.Timestamp] = None
        for rec in sub.to_dict("records"):
            ts = normalise_timestamp(rec.get("latest_closed_candle_time"))
            newer = ts is not None and (chosen_ts is None or ts >= chosen_ts)
            if chosen is None or newer:
                chosen, chosen_ts = rec, ts
        results.append(compute_freshness(
            instrument, collection_norm,
            chosen.get("latest_closed_candle_time"),
            chosen.get("status"), threshold_hours))
    return results, collection_norm, collection_note


# ============================================================
# Downstream data-quality gate (never a trading filter)
# ============================================================
def is_fresh_for_downstream_processing(result: Any) -> bool:
    """True only for the FRESH status; False for STALE and INVALID.

    Accepts a per-instrument result mapping or a status string.  Places
    no order and makes no trading decision.
    """
    status = (result.get("freshness_status")
              if isinstance(result, dict) else result)
    return status == FRESH


def data_quality_gate(results: List[Dict[str, Any]]) -> List[str]:
    """The instruments that are NOT fresh, in the given order.

    A data-quality gate: it reports which observations did not pass the
    freshness requirement.  It is not a trading filter.
    """
    return [str(r.get("instrument")) for r in results
            if not is_fresh_for_downstream_processing(r)]


# ============================================================
# Output builders (deterministic)
# ============================================================
def _counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {FRESH: 0, STALE: 0, INVALID: 0}
    for r in results:
        status = r.get("freshness_status")
        if status in counts:
            counts[status] += 1
    return counts


def _summary_statement(counts: Dict[str, int], total: int,
                       threshold_hours: float) -> str:
    return (f"{counts[FRESH]} of {total} instruments are FRESH, "
            f"{counts[STALE]} are STALE and {counts[INVALID]} are INVALID "
            f"under the fixed {threshold_hours:.0f}-hour rule")


def results_to_csv(results: List[Dict[str, Any]]) -> str:
    return pd.DataFrame(results, columns=list(CSV_COLUMNS)).to_csv(
        index=False)


def build_json_payload(results: List[Dict[str, Any]],
                       generated_ts: datetime.datetime,
                       threshold_hours: float,
                       collection_norm: Optional[pd.Timestamp],
                       collection_note: Optional[str]) -> Dict[str, Any]:
    counts = _counts(results)
    return {
        "generated_timestamp": generated_ts.strftime(TS_FORMAT),
        "freshness_threshold_hours": float(threshold_hours),
        "collection_timestamp": _fmt(collection_norm),
        "collection_timestamp_note": collection_note,
        "instruments": results,
        "overall_data_quality_summary": {
            "total": len(results),
            "fresh_count": counts[FRESH],
            "stale_count": counts[STALE],
            "invalid_count": counts[INVALID],
            "statement": _summary_statement(counts, len(results),
                                            threshold_hours),
        },
    }


def contexts_to_json(results: List[Dict[str, Any]],
                     generated_ts: datetime.datetime,
                     threshold_hours: float,
                     collection_norm: Optional[pd.Timestamp],
                     collection_note: Optional[str]) -> str:
    payload = build_json_payload(results, generated_ts, threshold_hours,
                                 collection_norm, collection_note)
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def build_report(results: List[Dict[str, Any]],
                 generated_ts: datetime.datetime,
                 threshold_hours: float,
                 collection_norm: Optional[pd.Timestamp],
                 collection_note: Optional[str]) -> str:
    counts = _counts(results)
    L: List[str] = []
    L.append("=" * 68)
    L.append("MARKET DATA FRESHNESS - READ-ONLY DATA-QUALITY REPORT")
    L.append("=" * 68)
    L.append("generated timestamp       : " + generated_ts.strftime(TS_FORMAT))
    L.append(f"freshness threshold hours : {threshold_hours:.0f}")
    L.append("collection timestamp      : "
             + (collection_norm.strftime(TS_FORMAT)
                if collection_norm is not None else "n/a"))
    if collection_note:
        L.append("collection timestamp note : " + collection_note)
    L.append("freshness rule            : " + FRESHNESS_RULE_TEXT)
    L.append("")
    L.append("This report is descriptive data-quality information only. It "
             "contains no directive and no ordering of instruments.")

    for r in results:
        L.append("")
        L.append("-" * 68)
        L.append("INSTRUMENT: " + str(r["instrument"]))
        L.append("  freshness status          : " + str(r["freshness_status"]))
        L.append("  source status             : " + str(r["source_status"]))
        L.append("  collection timestamp      : "
                 + str(r["collection_timestamp"]))
        L.append("  latest closed candle      : "
                 + str(r["latest_closed_candle_timestamp"]))
        age = r["data_age_hours"]
        L.append("  data age hours            : "
                 + ("n/a" if age is None else f"{age:.1f}"))
        age_text = ("latest closed candle age is unavailable"
                    if age is None else
                    f"latest closed candle is {age:.1f} hours older than "
                    f"the collection timestamp")
        L.append("  " + str(r["instrument"]) + ": "
                 + str(r["freshness_status"]) + " — " + age_text + ".")
        L.append("  data-quality note         : "
                 + str(r["data_quality_note"]))

    L.append("")
    L.append("=" * 68)
    L.append("OVERALL")
    L.append(f"  FRESH   : {counts[FRESH]}")
    L.append(f"  STALE   : {counts[STALE]}")
    L.append(f"  INVALID : {counts[INVALID]}")
    L.append(f"  total   : {len(results)}")
    L.append("")
    L.append("Descriptive data-quality information only; no directive of any "
             "kind.")
    return "\n".join(L)


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
        (OUTPUT_CSV, f"market_data_freshness_{stamp}.csv"),
        (OUTPUT_JSON, f"market_data_freshness_{stamp}.json"),
        (OUTPUT_TXT, f"market_data_freshness_report_{stamp}.txt"),
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
def run_real_analysis(directory: str = SCRIPT_DIR,
                      threshold_hours: float = DEFAULT_FRESHNESS_THRESHOLD_HOURS
                      ) -> int:
    print("=" * 68)
    print("market_data_freshness.py - REAL ANALYSIS (--run)")
    print("=" * 68)

    generated_ts = datetime.datetime.now()
    try:
        csv_path, json_path, txt_path = resolve_output_paths(directory,
                                                             generated_ts)
    except OutputExistsError as exc:
        print("ERROR: " + str(exc))
        return 1

    input_csv, input_report = find_live_inputs(directory)
    print("\nLive inputs selected:")
    print("  csv    : " + (os.path.basename(input_csv) if input_csv
                            else "absent"))
    print("  report : " + (os.path.basename(input_report) if input_report
                             else "absent"))

    missing = check_required_inputs(directory)
    if missing:
        print("ERROR: missing required input file(s): " + ", ".join(missing))
        return 1

    try:
        results, collection_norm, collection_note = build_results(
            directory, threshold_hours)
    except (MissingInputError, LiveSchemaError, ValueError) as exc:
        print("ERROR: " + str(exc))
        return 1

    csv_text = results_to_csv(results)
    json_text = contexts_to_json(results, generated_ts, threshold_hours,
                                 collection_norm, collection_note)
    report_text = build_report(results, generated_ts, threshold_hours,
                               collection_norm, collection_note)

    counts = _counts(results)
    print("\nFreshness summary:")
    for r in results:
        print(f"  {r['instrument']}: {r['freshness_status']}")
    print(f"  FRESH {counts[FRESH]} | STALE {counts[STALE]} | INVALID "
          f"{counts[INVALID]}")
    not_fresh = data_quality_gate(results)
    print("  data-quality gate (not fresh): "
          + (", ".join(not_fresh) if not_fresh else "none"))

    print("\nWriting NEW output files:")
    try:
        _write_text_if_absent(csv_path, csv_text)
        _write_text_if_absent(json_path, json_text)
        _write_text_if_absent(txt_path, report_text)
    except OutputExistsError as exc:
        print("ERROR: " + str(exc))
        return 1
    print("\nDone. Descriptive data-quality information only - no "
          "directive, no certainty figure, no ordering of instruments.")
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
    "pandas", "ai_trading_analyst", "live_market_state",
    "live_research_context",
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
    """AST + token scan of this file's own source (empty when clean)."""
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
_FIXTURE_COLLECTION = "2026-10-07 12:00:00"
_FIXTURE_CANDLES: Dict[str, str] = {
    "GOLD": "2026-10-04 12:00:00",     # exactly 72 hours -> FRESH
    "EURUSD": "2026-10-07 03:00:00",   # fresh
    "GBPUSD": "2026-10-07 03:00:00",   # fresh
    "USDJPY": "2026-10-06 00:00:00",   # fresh
    "AUDUSD": "2026-09-30 18:00:00",   # stale
}


def _live_row(instrument: str, candle: Any, status: str,
              direction: Any = "LONG") -> Dict[str, Any]:
    row: Dict[str, Any] = {c: None for c in LIVE_CSV_COLUMNS}
    row.update({
        "instrument": instrument,
        "symbol_resolution": "exact symbol name",
        "status": status,
        "candles_used": 9999,
        "latest_closed_candle_time": candle,
        "direction": direction,
        "data_quality_warnings": None,
    })
    for k in NUMERIC_STATE_KEYS:
        row[k] = 20.0
    for k in TIME_STATE_KEYS:
        row[k] = 1
    for k in REGIME_KEYS:
        row[k] = "ADX20-25"
    return row


def write_live_fixture(directory: str, collection_text: str = _FIXTURE_COLLECTION,
                       candles: Optional[Dict[str, Any]] = None,
                       statuses: Optional[Dict[str, str]] = None,
                       directions: Optional[Dict[str, Any]] = None,
                       csv_name: Optional[str] = LIVE_CSV,
                       report_name: Optional[str] = LIVE_REPORT
                       ) -> None:
    candles = _FIXTURE_CANDLES if candles is None else candles
    statuses = statuses or {}
    directions = directions or {}
    rows = [
        _live_row(inst,
                  candles.get(inst),
                  statuses.get(inst, STATUS_COLLECTED),
                  directions.get(inst, "LONG"))
        for inst in INSTRUMENTS]
    if csv_name is not None:
        pd.DataFrame(rows, columns=list(LIVE_CSV_COLUMNS)).to_csv(
            os.path.join(directory, csv_name), index=False)
    report = "\n".join([
        "=" * 64,
        "LIVE MARKET STATE - READ-ONLY COLLECTION REPORT",
        "=" * 64,
        "collection timestamp    : " + collection_text,
        "timeframe               : H1 (CLOSED candles only)",
        "instruments processed   : 5",
        "",
    ])
    if report_name is not None:
        with open(os.path.join(directory, report_name), "w",
                  encoding="utf-8") as fh:
            fh.write(report)


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

    fixed_ts = datetime.datetime(2026, 10, 7, 13, 0, 0)
    threshold = DEFAULT_FRESHNESS_THRESHOLD_HOURS

    print("market_data_freshness.py - synthetic test suite")
    print("(temporary directories only; no trading terminal; no real "
          "output is written)\n")

    with tempfile.TemporaryDirectory() as td:
        write_live_fixture(td)

        # ---------------- A. five-instrument validation -------
        start_area("A_instrument_validation")
        results, collection_norm, note = build_results(td)
        check("exactly the five required instruments are present",
              [r["instrument"] for r in results] == list(INSTRUMENTS))
        check("the instrument order matches the live state order",
              [r["instrument"] for r in results]
              == ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
        check("the collection timestamp is read from the live report",
              collection_norm is not None and note is None
              and collection_norm.strftime(TS_FORMAT) == _FIXTURE_COLLECTION)

        # ---------------- B. timestamp parsing -----------------
        start_area("B_timestamp_parsing")
        parsed = normalise_timestamp("2026-10-07 03:00:00")
        check("a normal timestamp string parses",
              parsed is not None
              and parsed.strftime(TS_FORMAT) == "2026-10-07 03:00:00")
        check("a datetime object parses",
              normalise_timestamp(datetime.datetime(2026, 10, 7, 3, 0, 0))
              is not None)
        check("an empty string returns None",
              normalise_timestamp("") is None)

        # ---------------- C. timezone-safe handling -----------
        start_area("C_timezone_safe")
        aware = normalise_timestamp("2026-10-07 05:00:00+02:00")
        naive_utc = normalise_timestamp("2026-10-07 03:00:00")
        check("an aware timestamp is normalised to the same instant as its "
              "UTC form", aware is not None and aware == naive_utc)
        aware2 = normalise_timestamp("2026-10-07 03:00:00+00:00")
        check("an explicit UTC timestamp keeps its instant",
              aware2 == naive_utc)
        mixed = compute_freshness("GOLD", aware, "2026-10-06 03:00:00",
                                  STATUS_COLLECTED, threshold)
        check("mixing an aware collection with a naive candle does not "
              "raise and yields a finite age",
              mixed["data_age_hours"] is not None
              and abs(mixed["data_age_hours"] - 24.0) < 1e-9)

        # ---------------- D. freshness calculation ------------
        start_area("D_freshness_calculation")
        fresh = compute_freshness("EURUSD", normalise_timestamp(
            "2026-10-07 12:00:00"), "2026-10-07 03:00:00",
            STATUS_COLLECTED, threshold)
        check("the data age equals the difference in hours",
              abs(fresh["data_age_hours"] - 9.0) < 1e-9)
        check("a small age is FRESH", fresh["freshness_status"] == FRESH)
        stale = compute_freshness("AUDUSD", normalise_timestamp(
            "2026-10-07 12:00:00"), "2026-09-30 18:00:00",
            STATUS_COLLECTED, threshold)
        check("a large age is STALE", stale["freshness_status"] == STALE)
        check("the age for the stale case is exactly 162.0 hours",
              abs(stale["data_age_hours"] - 162.0) < 1e-9)

        # ---------------- E. 72-hour boundary -----------------
        start_area("E_boundary")
        exact = compute_freshness("GOLD", normalise_timestamp(
            "2026-10-07 12:00:00"), "2026-10-04 12:00:00",
            STATUS_COLLECTED, threshold)
        check("exactly 72 hours is FRESH",
              exact["data_age_hours"] == 72.0
              and exact["freshness_status"] == FRESH)
        over = compute_freshness("GOLD", normalise_timestamp(
            "2026-10-07 12:00:00"), "2026-10-04 11:59:59",
            STATUS_COLLECTED, threshold)
        check("just over 72 hours is STALE",
              over["data_age_hours"] > 72.0
              and over["freshness_status"] == STALE)
        custom = compute_freshness("GOLD", normalise_timestamp(
            "2026-10-07 12:00:00"), "2026-10-04 12:00:00",
            STATUS_COLLECTED, 48.0)
        check("the threshold is configurable (48h makes 72h STALE)",
              custom["freshness_status"] == STALE
              and custom["freshness_threshold_hours"] == 48.0)

        # ---------------- F. missing collection timestamp -----
        start_area("F_missing_collection_timestamp")
        missing_coll = compute_freshness("GOLD", None,
                                         "2026-10-07 03:00:00",
                                         STATUS_COLLECTED, threshold)
        check("a missing collection timestamp is INVALID",
              missing_coll["freshness_status"] == INVALID
              and missing_coll["data_age_hours"] is None)
        check("a missing collection timestamp gives a descriptive reason",
              "collection timestamp is missing"
              in str(missing_coll["data_quality_note"]))
        bad_dir = os.path.join(td, "_no_report")
        os.makedirs(bad_dir, exist_ok=True)
        write_live_fixture(bad_dir)
        os.remove(os.path.join(bad_dir, LIVE_REPORT))
        res_no_report, coll_none, note_none = build_results(bad_dir)
        check("an absent live report yields INVALID for every instrument",
              coll_none is None and note_none is not None
              and all(r["freshness_status"] == INVALID
                      for r in res_no_report))

        # ---------------- G. missing candle timestamp ---------
        start_area("G_missing_candle_timestamp")
        missing_candle = compute_freshness("GOLD", normalise_timestamp(
            "2026-10-07 12:00:00"), None, STATUS_COLLECTED, threshold)
        check("a missing closed-candle timestamp is INVALID",
              missing_candle["freshness_status"] == INVALID
              and missing_candle["latest_closed_candle_timestamp"] is None)
        check("a missing closed-candle timestamp is described",
              "candle timestamp is missing"
              in str(missing_candle["data_quality_note"]))

        # ---------------- H. invalid timestamp ----------------
        start_area("H_invalid_timestamp")
        bad = compute_freshness("GOLD", normalise_timestamp(
            "2026-10-07 12:00:00"), "not-a-timestamp", STATUS_COLLECTED,
            threshold)
        check("an unreadable closed-candle timestamp is INVALID",
              bad["freshness_status"] == INVALID)
        bad_coll = compute_freshness("GOLD", normalise_timestamp("nonsense"),
                                     "2026-10-07 03:00:00",
                                     STATUS_COLLECTED, threshold)
        check("an unreadable collection timestamp is INVALID",
              bad_coll["freshness_status"] == INVALID)

        # ---------------- I. negative age ---------------------
        start_area("I_negative_age")
        neg = compute_freshness("GOLD", normalise_timestamp(
            "2026-10-07 12:00:00"), "2026-10-08 12:00:00",
            STATUS_COLLECTED, threshold)
        check("a negative age is finite and reported",
              neg["data_age_hours"] is not None
              and neg["data_age_hours"] < 0)
        check("a negative age carries a descriptive ordering note",
              "after the collection timestamp"
              in str(neg["data_quality_note"]))

        # ---------------- J. NONE direction is not stale ------
        start_area("J_none_direction_not_stale")
        none_dir = os.path.join(td, "_none")
        os.makedirs(none_dir, exist_ok=True)
        write_live_fixture(none_dir,
                           directions={i: "NONE" for i in INSTRUMENTS})
        res_none, _, _ = build_results(none_dir)
        by_inst = {r["instrument"]: r for r in res_none}
        check("a NONE direction is FRESH when the candle is fresh",
              by_inst["EURUSD"]["freshness_status"] == FRESH
              and by_inst["GOLD"]["freshness_status"] == FRESH)
        check("a NONE direction is STALE only when the candle is old",
              by_inst["AUDUSD"]["freshness_status"] == STALE)
        check("the freshness result carries no direction field",
              "direction" not in by_inst["EURUSD"])

        # ---------------- K. existing schema compatibility ----
        start_area("K_schema_compatibility")
        check("the required live columns are a subset of the live schema",
              set(REQUIRED_LIVE_COLUMNS) <= set(LIVE_CSV_COLUMNS))
        check("the module reports the existing timestamp field names",
              "latest_closed_candle_time" in REQUIRED_LIVE_COLUMNS
              and all(c in CSV_COLUMNS for c in (
                  "instrument", "collection_timestamp",
                  "latest_closed_candle_timestamp", "data_age_hours",
                  "freshness_threshold_hours", "freshness_status",
                  "source_status", "data_quality_note")))
        bad_schema = os.path.join(td, "_schema")
        os.makedirs(bad_schema, exist_ok=True)
        pd.DataFrame([{"instrument": "GOLD"}]).to_csv(
            os.path.join(bad_schema, LIVE_CSV), index=False)
        try:
            build_results(bad_schema)
            check("a live schema mismatch names the missing columns", False)
        except LiveSchemaError as exc:
            check("a live schema mismatch names the missing columns",
                  "latest_closed_candle_time" in str(exc)
                  and "status" in str(exc))

        # ---------------- S/T. gate behaviour + coverage ------
        start_area("S_data_quality_gate")
        check("the gate returns true only for FRESH",
              is_fresh_for_downstream_processing({"freshness_status": FRESH})
              is True
              and is_fresh_for_downstream_processing(
                  {"freshness_status": STALE}) is False
              and is_fresh_for_downstream_processing(
                  {"freshness_status": INVALID}) is False)
        check("the gate accepts a plain status string",
              is_fresh_for_downstream_processing(FRESH) is True
              and is_fresh_for_downstream_processing(STALE) is False)
        not_fresh = data_quality_gate(results)
        expected_not_fresh = [
            r["instrument"] for r in results
            if r["freshness_status"] != FRESH]
        check("the data-quality gate lists exactly the not-fresh "
              "instruments", not_fresh == expected_not_fresh)
        check("all five instruments are represented in the results",
              len(results) == 5
              and {r["instrument"] for r in results} == set(INSTRUMENTS))

        # ---------------- L/W/V/U. outputs --------------------
        start_area("L_deterministic_outputs")
        res_a, coll_a, note_a = build_results(td)
        res_b, coll_b, note_b = build_results(td)
        csv_a = results_to_csv(res_a)
        csv_b = results_to_csv(res_b)
        rep_a = build_report(res_a, fixed_ts, threshold, coll_a, note_a)
        rep_b = build_report(res_b, fixed_ts, threshold, coll_b, note_b)
        json_a = contexts_to_json(res_a, fixed_ts, threshold, coll_a, note_a)
        json_b = contexts_to_json(res_b, fixed_ts, threshold, coll_b, note_b)
        check("identical inputs produce identical CSV bytes", csv_a == csv_b)
        check("identical inputs produce identical report bytes",
              rep_a == rep_b)
        check("identical inputs produce identical JSON bytes",
              json_a == json_b)
        check("the CSV carries the exact field set",
              list(pd.read_csv(pd.io.common.StringIO(csv_a)).columns)
              == list(CSV_COLUMNS))
        payload = json.loads(json_a)
        check("the JSON payload carries the expected keys",
              payload["freshness_threshold_hours"] == threshold
              and payload["generated_timestamp"]
              == fixed_ts.strftime(TS_FORMAT)
              and len(payload["instruments"]) == len(INSTRUMENTS)
              and "overall_data_quality_summary"
              in payload)
        check("the JSON summary counts are consistent",
              payload["overall_data_quality_summary"]["fresh_count"]
              == sum(1 for r in res_a if r["freshness_status"] == FRESH))
        check("the report carries the freshness sections",
              "MARKET DATA FRESHNESS" in rep_a
              and "OVERALL" in rep_a
              and "FRESH   :" in rep_a and "STALE   :" in rep_a
              and "INVALID :" in rep_a)
        check("the report states each instrument and its age",
              all(inst in rep_a for inst in INSTRUMENTS)
              and "hours older than the collection timestamp" in rep_a)
        check("the report preserves the 72-hour threshold in its header",
              "freshness threshold hours : 72" in rep_a)

        # ---------------- M. output-file protection -----------
        start_area("M_output_file_protection")
        p_dir = os.path.join(td, "_paths")
        os.makedirs(p_dir, exist_ok=True)
        c_path, j_path, t_path = resolve_output_paths(p_dir, fixed_ts)
        check("a clean directory selects the primary names",
              (os.path.basename(c_path), os.path.basename(j_path),
               os.path.basename(t_path))
              == (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_TXT))
        _write_text_if_absent(c_path, "x")
        _write_text_if_absent(j_path, "x")
        _write_text_if_absent(t_path, "x")
        stamp = fixed_ts.strftime(STAMP_FORMAT)
        c2, j2, t2 = resolve_output_paths(p_dir, fixed_ts)
        check("existing files get the timestamped alternatives",
              (os.path.basename(c2), os.path.basename(j2),
               os.path.basename(t2))
              == (f"market_data_freshness_{stamp}.csv",
                  f"market_data_freshness_{stamp}.json",
                  f"market_data_freshness_report_{stamp}.txt"))
        with open(c_path, encoding="utf-8") as fh:
            check("the existing file is never overwritten", fh.read() == "x")
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
                  f"market_data_freshness_{stamp}.csv" in str(exc))

        # ---------------- N/O/P/Q/R. safety -------------------
        start_area("N_safety_scans")
        violations = run_safety_scan()
        check("the static safety scan of this file is clean",
              violations == [])
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
        terminal = ("Meta" + "Trader5")
        check("no trading terminal package is imported or loaded",
              terminal not in src
              and terminal not in sys.modules)
        tree = ast.parse(src)
        top_imports: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(a.name.split(".")[0]
                                   for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.append(node.module.split(".")[0])
        check("every import is on the allowlist (no external service)",
              all(mod in _ALLOWED_IMPORT_MODULES for mod in top_imports)
              and ("ur" + "llib") not in src.lower()
              and ("requ" + "ests") not in src.lower()
              and ("sock" + "et") not in src.lower())
        check("no order execution API name is present",
              all(tok not in src.lower()
                  for tok in ("order" + "_send", "order" + "_check",
                              "order" + "_modify", "order" + "_cancel")))
        blob = (csv_a + rep_a + json_a).lower()
        check("generated outputs contain no banned wording",
              contains_banned_wording(blob) == [])
        check("generated outputs carry no certainty or ordering wording",
              all(pat not in contains_banned_wording(blob)
                  for pat in _BANNED_WORD_PATTERNS))

        # ---------------- X. missing input file ---------------
        start_area("X_missing_input")
        empty_dir = os.path.join(td, "_empty")
        os.makedirs(empty_dir, exist_ok=True)
        missing = check_required_inputs(empty_dir)
        check("both required live inputs are named when absent",
              set(missing) == {LIVE_CSV, LIVE_REPORT})
        try:
            build_results(empty_dir)
            check("a missing live input file raises a named error", False)
        except MissingInputError as exc:
            check("a missing live input file raises a named error",
                  LIVE_CSV in str(exc))

        # ---------------- Y. no future-data substitution ------
        start_area("Y_no_substitution")
        y_dir = os.path.join(td, "_subst")
        os.makedirs(y_dir, exist_ok=True)
        write_live_fixture(
            y_dir,
            candles={"GOLD": "2026-09-30 18:00:00", "EURUSD": None,
                     "GBPUSD": "2026-10-07 03:00:00",
                     "USDJPY": "2026-10-07 03:00:00",
                     "AUDUSD": "2026-09-30 18:00:00"})
        res_subst, _, _ = build_results(y_dir)
        sub_by = {r["instrument"]: r for r in res_subst}
        check("an old stored candle stays STALE and is not refreshed",
              sub_by["GOLD"]["freshness_status"] == STALE
              and sub_by["GOLD"]["latest_closed_candle_timestamp"]
              == "2026-09-30 18:00:00")
        check("a missing candle stays INVALID and is not filled in",
              sub_by["EURUSD"]["freshness_status"] == INVALID
              and sub_by["EURUSD"]["latest_closed_candle_timestamp"] is None
              and sub_by["EURUSD"]["data_age_hours"] is None)
        check("stored closed-candle timestamps are reported exactly",
              sub_by["GOLD"]["latest_closed_candle_timestamp"]
              == "2026-09-30 18:00:00"
              and sub_by["GBPUSD"]["latest_closed_candle_timestamp"]
              == "2026-10-07 03:00:00"
              and sub_by["EURUSD"]["latest_closed_candle_timestamp"]
              is None)

        # ---------------- T. timestamped input selection ------
        start_area("T_timestamped_inputs")
        stale_candles = {i: "2026-09-30 18:00:00" for i in INSTRUMENTS}
        fresh_candles = {i: "2026-10-07 05:00:00" for i in INSTRUMENTS}

        # T1: the newest matching pair wins over an older pair and over
        #     the legacy files.
        t1 = os.path.join(td, "_ts_newest")
        os.makedirs(t1, exist_ok=True)
        write_live_fixture(
            t1, collection_text="2026-10-05 12:00:00",
            candles=stale_candles,
            csv_name="live_market_state_20261005_120000.csv",
            report_name="live_market_state_report_20261005_120000.txt")
        write_live_fixture(
            t1, collection_text="2026-10-07 12:00:00",
            candles=fresh_candles,
            csv_name="live_market_state_20261007_120000.csv",
            report_name="live_market_state_report_20261007_120000.txt")
        write_live_fixture(t1, collection_text="2026-09-20 00:00:00",
                           candles=stale_candles)
        sel_csv, sel_report = find_live_inputs(t1)
        check("the newest matching timestamped pair is selected",
              os.path.basename(sel_csv)
              == "live_market_state_20261007_120000.csv"
              and os.path.basename(sel_report)
              == "live_market_state_report_20261007_120000.txt")
        res_ts, coll_ts, note_ts = build_results(t1)
        by_ts = {r["instrument"]: r for r in res_ts}
        check("the newest pair's collection timestamp is used",
              coll_ts is not None and note_ts is None
              and coll_ts.strftime(TS_FORMAT) == "2026-10-07 12:00:00")
        check("the newest pair's fresh candle is used, not the legacy one",
              by_ts["GOLD"]["freshness_status"] == FRESH
              and by_ts["GOLD"]["latest_closed_candle_timestamp"]
              == "2026-10-07 05:00:00")

        # T2: a timestamped CSV with no matching timestamped report stamp
        #     is skipped; with no fallback the run is refused.
        t2 = os.path.join(td, "_ts_mismatch")
        os.makedirs(t2, exist_ok=True)
        write_live_fixture(
            t2, collection_text="2026-10-07 12:00:00",
            candles=fresh_candles,
            csv_name="live_market_state_20261007_120000.csv",
            report_name=None)
        write_live_fixture(
            t2, collection_text="2026-10-07 09:00:00",
            candles=stale_candles, csv_name=None,
            report_name="live_market_state_report_20261007_090000.txt")
        sel2_csv, sel2_report = find_live_inputs(t2)
        check("an unpaired timestamped CSV is not used",
              sel2_csv is None and sel2_report is None)
        try:
            build_results(t2)
            check("a mismatched timestamped pair is rejected", False)
        except MissingInputError:
            check("a mismatched timestamped pair is rejected", True)

        # T2b: the same mismatch falls back to the legacy files when they
        #      exist.
        write_live_fixture(t2, collection_text="2026-09-20 00:00:00",
                           candles=stale_candles)
        sel2b_csv, sel2b_report = find_live_inputs(t2)
        check("a mismatched pair is skipped for the legacy files",
              os.path.basename(sel2b_csv) == LIVE_CSV
              and os.path.basename(sel2b_report) == LIVE_REPORT)
        res2, coll2, _ = build_results(t2)
        check("the legacy collection timestamp is used when no pair matches",
              coll2 is not None
              and coll2.strftime(TS_FORMAT) == "2026-09-20 00:00:00")

        # T3: legacy files are used only when no timestamped pair exists.
        t3 = os.path.join(td, "_legacy_only")
        os.makedirs(t3, exist_ok=True)
        write_live_fixture(t3)
        sel3_csv, sel3_report = find_live_inputs(t3)
        check("legacy files are used when no timestamped pair exists",
              os.path.basename(sel3_csv) == LIVE_CSV
              and os.path.basename(sel3_report) == LIVE_REPORT)

        # T4: the collection timestamp comes from the report CONTENT, not
        #     from the filename stamp or a filesystem modification time.
        t4 = os.path.join(td, "_content_timestamp")
        os.makedirs(t4, exist_ok=True)
        write_live_fixture(
            t4, collection_text="2026-10-07 08:30:00",
            candles=fresh_candles,
            csv_name="live_market_state_20261007_120000.csv",
            report_name="live_market_state_report_20261007_120000.txt")
        _, coll4, _ = build_results(t4)
        check("the collection timestamp is read from report content",
              coll4 is not None
              and coll4.strftime(TS_FORMAT) == "2026-10-07 08:30:00")

        # T5: identical timestamped inputs stay deterministic.
        res_ts_a, coll_ts_a, note_ts_a = build_results(t1)
        res_ts_b, coll_ts_b, note_ts_b = build_results(t1)
        check("identical timestamped inputs produce identical CSV bytes",
              results_to_csv(res_ts_a) == results_to_csv(res_ts_b))
        check("identical timestamped inputs produce identical report bytes",
              build_report(res_ts_a, fixed_ts, threshold, coll_ts_a,
                           note_ts_a)
              == build_report(res_ts_b, fixed_ts, threshold, coll_ts_b,
                              note_ts_b))

        # ---------------- CLI gating --------------------------
        start_area("Z_cli_gating")
        check("default (no flags) runs the synthetic tests only",
              should_run_real([]) is False)
        check("--run selects the real analysis",
              should_run_real(["--run"]) is True)
        check("unrelated flags do not select the real analysis",
              should_run_real(["-v", "--verbose"]) is False)
        check("the terminal package stays unloaded in test mode",
              ("Meta" + "Trader5") not in sys.modules)

    start_area(None)
    print()
    print("  Per-area verification (area -> passed/failed):")
    stats = {name: (p, f) for name, p, f in area_stats}
    order = (
        "A_instrument_validation", "B_timestamp_parsing",
        "C_timezone_safe", "D_freshness_calculation", "E_boundary",
        "F_missing_collection_timestamp", "G_missing_candle_timestamp",
        "H_invalid_timestamp", "I_negative_age",
        "J_none_direction_not_stale", "K_schema_compatibility",
        "S_data_quality_gate", "L_deterministic_outputs",
        "M_output_file_protection", "N_safety_scans",
        "X_missing_input", "Y_no_substitution",
        "T_timestamped_inputs", "Z_cli_gating",
    )
    for name in order:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<32} {p:>2} passed, {f} failed [{status}]")
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
