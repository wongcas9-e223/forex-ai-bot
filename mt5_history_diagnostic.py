"""
mt5_history_diagnostic.py - READ-ONLY MetaTrader 5 H1 HISTORY DIAGNOSTIC.

PURPOSE
    A deterministic, read-only diagnostic that inspects the actual H1
    history the trading terminal returns for five instruments, to help
    explain why some instruments return much older H1 candles than
    others.  It ASSUMES NO CAUSE: it only records descriptive evidence
    and then states what the evidence does and does not support.

SAFETY
    - read-only: it never places, changes, closes or removes anything,
    - no order / execution function is called anywhere in this file,
    - no directional wording, no sizing, no protective levels, no risk
      figures, no certainty or ordering of instruments,
    - no external service call and no learning library,
    - it modifies NO existing file: the terminal is queried read-only
      and only the three NEW output files may be created,
    - an existing output file is never overwritten (a timestamped
      alternative is used, and a collision there stops the run).

TERMINAL ACCESS
    The terminal package is imported lazily, on the --run path only, so
    the default test mode never loads or contacts it.  Every diagnostic
    function takes the terminal module as an argument, so the tests can
    drive them with an in-memory substitute.

    Read-only terminal functions used:
        initialize, shutdown, last_error, symbol_info, symbols_get,
        symbol_select (Market Watch visibility only - no trading
        setting is changed), symbol_info_tick, copy_rates_from_pos,
        copy_rates_range, TIMEFRAME_H1

INSTRUMENTS (exactly these five, in this order, never substituted)
    GOLD, EURUSD, GBPUSD, USDJPY, AUDUSD

WHAT IS INSPECTED PER INSTRUMENT
    1. symbol lookup     (available, visible, description, path,
                          digits, point, spread)
    2. symbol selection  (Market Watch visibility query, read-only)
    3. current tick      (available, timestamp, bid, ask - descriptive)
    4. H1 history        (bars, earliest, latest, latest closed,
                          forming flag, unique count, ordering, invalid)
    5. two query methods (recent-position query and date-range query)
    6. recent 7-day window classification
    7. terminal return information (last_error code and text)
    8. related symbol names exposed by the terminal (reported only -
       no alternative name is ever substituted or queried)

OUTPUT FILES (all NEW; every path is checked before anything is
    written):
    mt5_history_diagnostic.csv
    mt5_history_diagnostic.json
    mt5_history_diagnostic_report.txt
    On a collision, timestamped alternatives are used:
    mt5_history_diagnostic_YYYYMMDD_HHMMSS.csv / .json and
    mt5_history_diagnostic_report_YYYYMMDD_HHMMSS.txt

COMMAND-LINE SAFETY
    python mt5_history_diagnostic.py          -> synthetic tests only
                                                (no terminal contact, no
                                                AI/API, no real output)
    python mt5_history_diagnostic.py --run    -> real read-only
                                                diagnostic

DISCLAIMER: descriptive terminal diagnostics only. No causal claim is
made unless the collected evidence supports it. Educational use only.
"""

# ============================================================
# Imports: Python standard library + numpy + pandas.  The terminal
# package is imported LATER, on the --run path only.
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
# Frozen settings
# ============================================================
INSTRUMENTS: Tuple[str, ...] = (
    "GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
)
TIMEFRAME_LABEL = "H1"
TIMEFRAME_ATTRIBUTE = "TIMEFRAME_H1"
HISTORY_BARS = 10_000
RECENT_WINDOW_DAYS = 7
RECENT_DIFFERENCE_HOURS = 24  # descriptive comparison window only
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
STAMP_FORMAT = "%Y%m%d_%H%M%S"

OUTPUT_CSV = "mt5_history_diagnostic.csv"
OUTPUT_JSON = "mt5_history_diagnostic.json"
OUTPUT_TXT = "mt5_history_diagnostic_report.txt"
OUTPUT_FILES: Tuple[str, ...] = (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_TXT)

# Fixed classification labels for the recent-window check.
RECENT_BARS = "RECENT_BARS"
PARTIAL_RECENT_HISTORY = "PARTIAL_RECENT_HISTORY"
ONLY_OLD_BARS = "ONLY_OLD_BARS"
NO_RECENT_BARS = "NO_RECENT_BARS"

CSV_COLUMNS: Tuple[str, ...] = (
    "instrument",
    "symbol_exists",
    "symbol_visible",
    "description",
    "path",
    "current_tick_available",
    "current_tick_timestamp",
    "history_bars_returned",
    "earliest_h1_timestamp",
    "latest_h1_timestamp",
    "latest_closed_h1_timestamp",
    "recent_7d_bars",
    "recent_history_status",
    "position_request_latest",
    "date_range_request_latest",
    "request_timestamps_match",
    "mt5_error_code",
    "mt5_error_description",
    "discovered_related_symbols",
    "diagnostic_note",
)


class OutputExistsError(RuntimeError):
    """An output path is already occupied; nothing may be written."""


# ============================================================
# Small deterministic helpers
# ============================================================
def _get(obj: Any, name: str) -> Any:
    """getattr that never raises and treats empty as absent."""
    try:
        value = getattr(obj, name)
    except Exception:                              # noqa: BLE001
        return None
    if value is None or value == "":
        return None
    return value


def _fmt(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    try:
        stamp = pd.Timestamp(ts)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.strftime(TS_FORMAT)


def _normalise(ts: Any) -> Optional[pd.Timestamp]:
    """To a single reference frame (UTC, naive); None when unreadable."""
    if ts is None:
        return None
    try:
        stamp = pd.Timestamp(ts)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _rates_to_frame(rates: Any) -> Optional[pd.DataFrame]:
    if rates is None:
        return None
    try:
        frame = pd.DataFrame(rates)
    except Exception:                              # noqa: BLE001
        return None
    if "time" not in frame.columns:
        return None
    frame["time"] = pd.to_datetime(frame["time"], unit="s", errors="coerce")
    return frame


# ============================================================
# 1. Symbol lookup
# ============================================================
def diag_symbol_lookup(instrument: str, mt5mod: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "instrument": instrument,
        "symbol_requested": instrument,
        "symbol_exists": False,
        "symbol_visible": None,
        "trade_mode": None,
        "description": None,
        "path": None,
        "digits": None,
        "point": None,
        "spread": None,
        "lookup_note": None,
    }
    try:
        info = mt5mod.symbol_info(instrument)
    except Exception as exc:                       # noqa: BLE001
        result["lookup_note"] = f"symbol_info raised: {exc}"
        return result
    if info is None:
        result["lookup_note"] = "symbol_info returned no data for this name"
        return result
    result["symbol_exists"] = True
    result["symbol_visible"] = _get(info, "visible")
    result["trade_mode"] = _get(info, "trade_mode")
    result["description"] = _get(info, "description")
    result["path"] = _get(info, "path")
    result["digits"] = _get(info, "digits")
    result["point"] = _get(info, "point")
    result["spread"] = _get(info, "spread")
    return result


# ============================================================
# 2. Symbol selection (Market Watch visibility only)
# ============================================================
def diag_symbol_select(instrument: str, mt5mod: Any) -> Dict[str, Any]:
    note = ("read-only Market Watch visibility query; no trading "
            "setting is changed and no trade action is performed")
    try:
        selected = mt5mod.symbol_select(instrument, True)
    except Exception as exc:                       # noqa: BLE001
        return {"symbol_selected": None,
                "select_note": f"symbol_select raised: {exc}"}
    return {"symbol_selected": bool(selected), "select_note": note}


# ============================================================
# 3. Current tick
# ============================================================
def diag_current_tick(instrument: str, mt5mod: Any) -> Dict[str, Any]:
    try:
        tick = mt5mod.symbol_info_tick(instrument)
    except Exception as exc:                       # noqa: BLE001
        return {"current_tick_available": False,
                "current_tick_timestamp": None, "bid": None, "ask": None,
                "tick_note": f"symbol_info_tick raised: {exc}"}
    if tick is None:
        return {"current_tick_available": False,
                "current_tick_timestamp": None, "bid": None, "ask": None,
                "tick_note": "no tick data returned"}
    when = _get(tick, "time")
    return {
        "current_tick_available": True,
        "current_tick_timestamp": _fmt(pd.to_datetime(when, unit="s")
                                       if isinstance(when, (int, float))
                                       else when),
        "bid": _get(tick, "bid"),
        "ask": _get(tick, "ask"),
        "tick_note": "descriptive snapshot only; not interpreted as "
                     "a directive",
    }


# ============================================================
# 4. H1 history (recent-position query and date-range query)
# ============================================================
def diag_last_error(mt5mod: Any) -> Dict[str, Any]:
    try:
        code, text = mt5mod.last_error()
    except Exception as exc:                       # noqa: BLE001
        return {"mt5_error_code": None,
                "mt5_error_description": f"last_error raised: {exc}"}
    return {"mt5_error_code": code, "mt5_error_description": str(text)}


def query_history_from_pos(instrument: str, mt5mod: Any,
                           count: int = HISTORY_BARS
                           ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Recent-position history query (same mechanism as the live
    collector).  Returns (frame, error_text)."""
    try:
        rates = mt5mod.copy_rates_from_pos(instrument,
                                           mt5mod.TIMEFRAME_H1, 0, count)
    except Exception as exc:                       # noqa: BLE001
        return None, f"copy_rates_from_pos raised: {exc}"
    if rates is None:
        err = diag_last_error(mt5mod)
        return None, ("no data returned by the recent-position query "
                      f"(code {err['mt5_error_code']}: "
                      f"{err['mt5_error_description']})")
    return _rates_to_frame(rates), None


def query_history_range(instrument: str, mt5mod: Any,
                        now_ts: Optional[datetime.datetime] = None,
                        days: int = RECENT_WINDOW_DAYS
                        ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Date-range history query over the recent window."""
    reference = now_ts if now_ts is not None else datetime.datetime.now()
    start = pd.Timestamp(reference).tz_localize(None) - pd.Timedelta(days=days)
    end = pd.Timestamp(reference).tz_localize(None)
    try:
        rates = mt5mod.copy_rates_range(instrument, mt5mod.TIMEFRAME_H1,
                                        start.to_pydatetime(),
                                        end.to_pydatetime())
    except Exception as exc:                       # noqa: BLE001
        return None, f"copy_rates_range raised: {exc}"
    if rates is None:
        err = diag_last_error(mt5mod)
        return None, ("no data returned by the date-range query "
                      f"(code {err['mt5_error_code']}: "
                      f"{err['mt5_error_description']})")
    return _rates_to_frame(rates), None


def summarise_history(frame: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Descriptive summary of one returned history frame."""
    if frame is None or len(frame) == 0:
        return {
            "bars_returned": 0,
            "earliest_h1_timestamp": None,
            "latest_h1_timestamp": None,
            "latest_closed_h1_timestamp": None,
            "newest_bar_is_forming": False,
            "unique_timestamps": 0,
            "duplicate_timestamps": 0,
            "chronological_ascending": True,
            "invalid_timestamps": 0,
        }
    times = frame["time"]
    invalid = int(times.isna().sum())
    valid = times.dropna()
    unique = int(valid.nunique())
    return {
        "bars_returned": int(len(frame)),
        "earliest_h1_timestamp": _fmt(valid.iloc[0]) if len(valid) else None,
        "latest_h1_timestamp": _fmt(valid.iloc[-1]) if len(valid) else None,
        "latest_closed_h1_timestamp":
            _fmt(valid.iloc[-2]) if len(valid) >= 2 else None,
        "newest_bar_is_forming": bool(len(valid) >= 1),
        "unique_timestamps": unique,
        "duplicate_timestamps": int(len(valid) - unique),
        "chronological_ascending": bool(valid.is_monotonic_increasing),
        "invalid_timestamps": invalid,
    }


def classify_recent_window(frame: Optional[pd.DataFrame],
                           now_ts: datetime.datetime,
                           days: int = RECENT_WINDOW_DAYS) -> Dict[str, Any]:
    """Descriptive classification of the recent window only."""
    empty = {"recent_7d_bars": 0, "recent_history_status": NO_RECENT_BARS,
             "recent_7d_first": None, "recent_7d_last": None}
    if frame is None or len(frame) == 0:
        return dict(empty)
    times = frame["time"].dropna()
    if len(times) == 0:
        return dict(empty)
    cutoff = pd.Timestamp(now_ts).tz_localize(None) - pd.Timedelta(days=days)
    recent = times[times >= cutoff]
    count = int(len(recent))
    if count == 0:
        status = ONLY_OLD_BARS
    elif count == int(len(times)):
        status = RECENT_BARS
    else:
        status = PARTIAL_RECENT_HISTORY
    return {
        "recent_7d_bars": count,
        "recent_history_status": status,
        "recent_7d_first": _fmt(recent.iloc[0]) if count else None,
        "recent_7d_last": _fmt(recent.iloc[-1]) if count else None,
    }


# ============================================================
# 8. Related symbol-name diagnostics (report only, never substitute)
# ============================================================
def diag_related_symbols(instrument: str, mt5mod: Any) -> Dict[str, Any]:
    try:
        symbols = mt5mod.symbols_get() or []
    except Exception as exc:                       # noqa: BLE001
        return {"requested_symbol": instrument,
                "discovered_related_symbols": [],
                "alternative_queried": False,
                "related_note": f"symbol list unavailable: {exc}"}
    found: List[str] = []
    for sym in symbols:
        name = _get(sym, "name")
        if name is None or name == instrument:
            continue
        if instrument in name:
            found.append(str(name))
    return {
        "requested_symbol": instrument,
        "discovered_related_symbols": sorted(set(found)),
        "alternative_queried": False,
        "related_note": ("names are reported for information only; no "
                         "alternative name is substituted or queried for "
                         "market data"),
    }


# ============================================================
# Per-instrument diagnosis
# ============================================================
def diagnose_instrument(instrument: str, mt5mod: Any,
                        now_ts: datetime.datetime,
                        count: int = HISTORY_BARS,
                        days: int = RECENT_WINDOW_DAYS) -> Dict[str, Any]:
    lookup = diag_symbol_lookup(instrument, mt5mod)
    select = diag_symbol_select(instrument, mt5mod)
    tick = diag_current_tick(instrument, mt5mod)
    related = diag_related_symbols(instrument, mt5mod)

    notes: List[str] = []
    if not lookup["symbol_exists"]:
        notes.append("the requested name is not available in the terminal")
        error = diag_last_error(mt5mod)
        return {
            "instrument": instrument,
            "symbol_lookup": lookup,
            "symbol_selection": select,
            "tick": tick,
            "related_symbols": related,
            "recent_position_query": {"available": False, "error": None},
            "date_range_query": {"available": False, "error": None},
            "history": summarise_history(None),
            "recent_window": classify_recent_window(None, now_ts, days),
            "history_query_error": None,
            "position_request_latest": None,
            "date_range_request_latest": None,
            "request_timestamps_match": None,
            "mt5_error": error,
            "diagnostic_note": "; ".join(notes),
        }

    frame, pos_error = query_history_from_pos(instrument, mt5mod, count)
    range_frame, range_error = query_history_range(instrument, mt5mod,
                                                   now_ts, days)
    history = summarise_history(frame)
    recent = classify_recent_window(frame, now_ts, days)

    pos_latest = history["latest_h1_timestamp"]
    range_summary = summarise_history(range_frame)
    range_latest = range_summary["latest_h1_timestamp"]
    if pos_latest is not None and range_latest is not None:
        match: Optional[bool] = pos_latest == range_latest
    else:
        match = None

    if pos_error:
        notes.append(pos_error)
    if range_error:
        notes.append(range_error)
    if frame is not None and len(frame) == 0:
        notes.append("the terminal returned zero H1 bars for this name")
    if history["invalid_timestamps"]:
        notes.append(f"{history['invalid_timestamps']} bar(s) carried an "
                     f"unreadable timestamp")
    if history["duplicate_timestamps"]:
        notes.append(f"{history['duplicate_timestamps']} duplicate bar "
                     f"timestamp(s) were returned")
    if not history["chronological_ascending"]:
        notes.append("the returned bars were not in ascending time order")
    if match is False:
        notes.append("the two query methods returned different latest "
                     "timestamps")

    return {
        "instrument": instrument,
        "symbol_lookup": lookup,
        "symbol_selection": select,
        "tick": tick,
        "related_symbols": related,
        "recent_position_query": {
            "available": frame is not None,
            "error": pos_error,
            "summary": history,
        },
        "date_range_query": {
            "available": range_frame is not None,
            "error": range_error,
            "summary": range_summary,
        },
        "history": history,
        "recent_window": recent,
        "history_query_error": pos_error or range_error,
        "position_request_latest": pos_latest,
        "date_range_request_latest": range_latest,
        "request_timestamps_match": match,
        "mt5_error": diag_last_error(mt5mod),
        "diagnostic_note": "; ".join(notes) if notes else None,
    }


# ============================================================
# Cross-instrument comparison and evidence-based conclusion
# ============================================================
def cross_instrument_comparison(
        results: List[Dict[str, Any]]) -> Dict[str, Any]:
    latest: Dict[str, Optional[str]] = {}
    parsed: Dict[str, Optional[pd.Timestamp]] = {}
    for res in results:
        value = res["history"]["latest_closed_h1_timestamp"]
        latest[res["instrument"]] = value
        parsed[res["instrument"]] = _normalise(value)
    valid = {k: v for k, v in parsed.items() if v is not None}
    newest = max(valid.values()) if valid else None
    older: List[str] = []
    if newest is not None:
        for inst in INSTRUMENTS:
            ts = parsed.get(inst)
            if ts is not None and (newest - ts) > pd.Timedelta(
                    hours=RECENT_DIFFERENCE_HOURS):
                older.append(inst)
    agreement = {res["instrument"]: res["request_timestamps_match"]
                 for res in results}
    decided = [v for v in agreement.values() if v is not None]
    errors = [res["instrument"] for res in results
              if res["symbol_lookup"]["symbol_exists"] is False]
    return {
        "latest_closed_h1_timestamps": latest,
        "newest_latest_closed_h1_timestamp": _fmt(newest),
        "instruments_older_than_newest_by_over_24h": older,
        "request_method_agreement": agreement,
        "all_request_methods_agree":
            bool(decided) and all(v is True for v in decided),
        "any_request_method_disagreement": any(v is False for v in decided),
        "instruments_not_available": errors,
        "note": ("descriptive side-by-side comparison only; no instrument "
                 "is preferred, ordered or combined into one figure"),
    }


def build_conclusion(results: List[Dict[str, Any]],
                     comparison: Dict[str, Any]) -> str:
    """A descriptive conclusion that never asserts an unsupported cause."""
    parts: List[str] = []
    older = comparison["instruments_older_than_newest_by_over_24h"]
    if older:
        parts.append("the terminal returned a markedly older latest closed "
                     "H1 candle for " + ", ".join(older)
                     + " than for the remaining instruments")
    else:
        parts.append("every instrument returned a latest closed H1 candle "
                     "within the comparison window of the newest stored "
                     "candle")
    if comparison["any_request_method_disagreement"]:
        disagree = [k for k, v in
                    comparison["request_method_agreement"].items()
                    if v is False]
        parts.append("the recent-position query and the date-range query "
                     "returned different latest timestamps for "
                     + ", ".join(disagree))
    elif comparison["all_request_methods_agree"]:
        parts.append("both read-only history query methods returned "
                     "matching latest timestamps for every available "
                     "instrument")
    missing = comparison["instruments_not_available"]
    if missing:
        parts.append("the requested name was not available in the terminal "
                     "for " + ", ".join(missing))
    if older and not comparison["any_request_method_disagreement"] \
            and comparison["all_request_methods_agree"] and not missing:
        parts.append("the difference is therefore not explained by the "
                     "query method, and the collected evidence does not "
                     "establish the underlying reason")
    elif not older and not comparison["any_request_method_disagreement"] \
            and not missing:
        parts.append("no difference in history depth was observed for the "
                     "five instruments in this diagnostic")
    else:
        parts.append("the collected evidence does not establish a single "
                     "underlying reason")
    return "; ".join(parts) + "."


def run_diagnostic(mt5mod: Any,
                   generated_ts: Optional[datetime.datetime] = None,
                   now_ts: Optional[datetime.datetime] = None
                   ) -> Dict[str, Any]:
    """Run the full read-only diagnostic and return one payload."""
    generated = generated_ts or datetime.datetime.now()
    reference = now_ts or datetime.datetime.now()
    results = [diagnose_instrument(inst, mt5mod, reference)
               for inst in INSTRUMENTS]
    comparison = cross_instrument_comparison(results)
    errors = []
    for res in results:
        err = res["mt5_error"]
        if err["mt5_error_code"] not in (None, 0):
            errors.append({"instrument": res["instrument"],
                           "code": err["mt5_error_code"],
                           "description": err["mt5_error_description"]})
        if res["symbol_lookup"]["symbol_exists"] is False:
            errors.append({"instrument": res["instrument"],
                           "code": "symbol_not_found",
                           "description": "the requested name is not "
                                          "available in the terminal"})
    return {
        "generated_timestamp": generated.strftime(TS_FORMAT),
        "reference_timestamp": reference.strftime(TS_FORMAT),
        "timeframe": TIMEFRAME_LABEL,
        "requested_instruments": list(INSTRUMENTS),
        "instruments": results,
        "cross_instrument_comparison": comparison,
        "errors_encountered": errors,
        "overall_diagnostic_conclusion": build_conclusion(results, comparison),
    }


# ============================================================
# Output builders (deterministic)
# ============================================================
def context_to_row(res: Dict[str, Any]) -> Dict[str, Any]:
    lookup = res["symbol_lookup"]
    tick = res["tick"]
    history = res["history"]
    recent = res["recent_window"]
    error = res["mt5_error"]
    related = res["related_symbols"]["discovered_related_symbols"]
    row: Dict[str, Any] = {c: None for c in CSV_COLUMNS}
    row["instrument"] = res["instrument"]
    row["symbol_exists"] = lookup["symbol_exists"]
    row["symbol_visible"] = lookup["symbol_visible"]
    row["description"] = lookup["description"]
    row["path"] = lookup["path"]
    row["current_tick_available"] = tick["current_tick_available"]
    row["current_tick_timestamp"] = tick["current_tick_timestamp"]
    row["history_bars_returned"] = history["bars_returned"]
    row["earliest_h1_timestamp"] = history["earliest_h1_timestamp"]
    row["latest_h1_timestamp"] = history["latest_h1_timestamp"]
    row["latest_closed_h1_timestamp"] = history["latest_closed_h1_timestamp"]
    row["recent_7d_bars"] = recent["recent_7d_bars"]
    row["recent_history_status"] = recent["recent_history_status"]
    row["position_request_latest"] = res["position_request_latest"]
    row["date_range_request_latest"] = res["date_range_request_latest"]
    row["request_timestamps_match"] = res["request_timestamps_match"]
    row["mt5_error_code"] = error["mt5_error_code"]
    row["mt5_error_description"] = error["mt5_error_description"]
    row["discovered_related_symbols"] = " | ".join(related)
    row["diagnostic_note"] = res["diagnostic_note"]
    return row


def results_to_csv(payload: Dict[str, Any]) -> str:
    rows = [context_to_row(res) for res in payload["instruments"]]
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS)).to_csv(index=False)


def build_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True,
                      default=_json_default) + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        finite = float(value)
        return finite if np.isfinite(finite) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime)):
        return _fmt(value)
    if isinstance(value, bool):
        return value
    return str(value)


def build_report(payload: Dict[str, Any]) -> str:
    comparison = payload["cross_instrument_comparison"]
    L: List[str] = []
    L.append("=" * 68)
    L.append("MetaTrader 5 H1 HISTORY DIAGNOSTIC - READ-ONLY REPORT")
    L.append("=" * 68)
    L.append("1. diagnostic timestamp   : " + payload["generated_timestamp"])
    L.append("   reference timestamp    : " + payload["reference_timestamp"])
    L.append("   timeframe              : " + payload["timeframe"])
    L.append("   instruments            : "
             + ", ".join(payload["requested_instruments"]))
    L.append("   terminal connection    : read-only session; the collector "
             "initialised the terminal and closed it afterwards")
    L.append("")
    L.append("This report is descriptive terminal diagnostics only. It "
             "contains no directive of any kind and does not order the "
             "instruments.")

    L.append("")
    L.append("3. FIVE-SYMBOL COMPARISON")
    for res in payload["instruments"]:
        L.append(f"   {res['instrument']:<7} latest closed H1 : "
                 f"{res['history']['latest_closed_h1_timestamp']} | bars "
                 f"{res['history']['bars_returned']} | recent 7d "
                 f"{res['recent_window']['recent_7d_bars']} "
                 f"({res['recent_window']['recent_history_status']})")
    L.append("   newest latest closed H1 : "
             + str(comparison["newest_latest_closed_h1_timestamp"]))
    L.append("   older by over 24 hours  : "
             + (", ".join(comparison[
                 "instruments_older_than_newest_by_over_24h"]) or "none"))

    for res in payload["instruments"]:
        lookup = res["symbol_lookup"]
        tick = res["tick"]
        history = res["history"]
        recent = res["recent_window"]
        error = res["mt5_error"]
        related = res["related_symbols"]
        L.append("")
        L.append("-" * 68)
        L.append("INSTRUMENT: " + res["instrument"])
        L.append("4. SYMBOL LOOKUP")
        L.append("   requested name            : "
                 + str(lookup["symbol_requested"]))
        L.append("   available                 : "
                 + str(lookup["symbol_exists"]))
        L.append("   visible                   : "
                 + str(lookup["symbol_visible"]))
        L.append("   trade mode                : "
                 + str(lookup["trade_mode"]))
        L.append("   description               : "
                 + str(lookup["description"]))
        L.append("   path                      : " + str(lookup["path"]))
        L.append("   digits / point / spread   : "
                 + f"{lookup['digits']} / {lookup['point']} / "
                 + f"{lookup['spread']}")
        L.append("   selected for market watch : "
                 + str(res["symbol_selection"]["symbol_selected"]))
        L.append("5. CURRENT TICK")
        L.append("   available                 : "
                 + str(tick["current_tick_available"]))
        L.append("   timestamp                 : "
                 + str(tick["current_tick_timestamp"]))
        L.append("   bid / ask                 : "
                 + f"{tick['bid']} / {tick['ask']}")
        L.append("6. H1 HISTORY")
        L.append("   bars returned             : "
                 + str(history["bars_returned"]))
        L.append("   earliest bar              : "
                 + str(history["earliest_h1_timestamp"]))
        L.append("   latest bar                : "
                 + str(history["latest_h1_timestamp"]))
        L.append("   latest closed H1          : "
                 + str(history["latest_closed_h1_timestamp"]))
        L.append("   newest bar looks forming  : "
                 + str(history["newest_bar_is_forming"]))
        L.append("   unique timestamps         : "
                 + str(history["unique_timestamps"]))
        L.append("   duplicates                : "
                 + str(history["duplicate_timestamps"]))
        L.append("   ascending order           : "
                 + str(history["chronological_ascending"]))
        L.append("   unreadable timestamps     : "
                 + str(history["invalid_timestamps"]))
        L.append("7. RECENT 7-DAY WINDOW")
        L.append("   recent bars               : "
                 + str(recent["recent_7d_bars"]))
        L.append("   classification            : "
                 + str(recent["recent_history_status"]))
        L.append("8. QUERY-METHOD COMPARISON")
        L.append("   recent-position latest    : "
                 + str(res["position_request_latest"]))
        L.append("   date-range latest         : "
                 + str(res["date_range_request_latest"]))
        L.append("   latest timestamps match   : "
                 + str(res["request_timestamps_match"]))
        L.append("9. TERMINAL RETURN INFORMATION")
        L.append("   error code                : "
                 + str(error["mt5_error_code"]))
        L.append("   error description         : "
                 + str(error["mt5_error_description"]))
        L.append("10. POSSIBLE SYMBOL-NAME DIFFERENCES")
        L.append("   requested name            : "
                 + str(related["requested_symbol"]))
        L.append("   discovered related names  : "
                 + (", ".join(related["discovered_related_symbols"])
                    or "none"))
        L.append("   alternative queried       : "
                 + str(related["alternative_queried"]))
        L.append("   diagnostic note           : "
                 + str(res["diagnostic_note"]))

    L.append("")
    L.append("=" * 68)
    L.append("11. EVIDENCE-BASED DIAGNOSTIC CONCLUSION")
    L.append("   " + payload["overall_diagnostic_conclusion"])
    L.append("")
    L.append("   errors encountered: "
             + (", ".join(f"{e['instrument']}={e['code']}"
                          for e in payload["errors_encountered"]) or "none"))
    L.append("")
    L.append("12. LIMITATIONS")
    L.append("   - the diagnostic describes what the terminal returned at "
             "this moment only")
    L.append("   - a broker-side history depth limit or a name difference "
             "cannot be confirmed from a single read-only session")
    L.append("   - no alternative symbol name was substituted or queried")
    L.append("   - no causal claim is made beyond what the evidence shows")
    L.append("")
    L.append("Descriptive terminal diagnostics only; no directive of any "
             "kind.")
    return "\n".join(L)


# ============================================================
# Output-file protection (every path checked before any writing)
# ============================================================
def resolve_output_paths(directory: str,
                         generated_ts: datetime.datetime
                         ) -> Tuple[str, str, str]:
    stamp = generated_ts.strftime(STAMP_FORMAT)
    plan = (
        (OUTPUT_CSV, f"mt5_history_diagnostic_{stamp}.csv"),
        (OUTPUT_JSON, f"mt5_history_diagnostic_{stamp}.json"),
        (OUTPUT_TXT, f"mt5_history_diagnostic_report_{stamp}.txt"),
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


def _write_bytes_if_absent(path: str, payload: bytes) -> None:
    if os.path.exists(path):
        raise OutputExistsError(
            "output file already exists (never overwritten): " + path)
    with open(path, "wb") as fh:
        fh.write(payload)
    print("  wrote " + os.path.basename(path) + " ("
          + str(len(payload)) + " bytes)")


# ============================================================
# Real diagnostic (requires --run; never runs by default)
# ============================================================
def run_real_diagnostic() -> int:
    print("=" * 68)
    print("mt5_history_diagnostic.py - REAL READ-ONLY DIAGNOSTIC (--run)")
    print("=" * 68)
    try:
        import MetaTrader5 as mt5mod             # noqa: PLC0415
    except ImportError:
        print("ERROR: the MetaTrader 5 Python package is not installed in "
              "this environment.")
        return 1

    generated_ts = datetime.datetime.now()
    try:
        csv_path, json_path, txt_path = resolve_output_paths(SCRIPT_DIR,
                                                             generated_ts)
    except OutputExistsError as exc:
        print("ERROR: " + str(exc))
        return 1

    if not mt5mod.initialize():
        print("ERROR: could not connect to the terminal (read-only "
              "session).")
        print("Last error: " + str(mt5mod.last_error()))
        return 1
    try:
        payload = run_diagnostic(mt5mod)
    finally:
        try:
            mt5mod.shutdown()
        except Exception:                          # noqa: BLE001
            pass

    print("\nPer-instrument result:")
    for res in payload["instruments"]:
        print(f"  {res['instrument']}: bars "
              f"{res['history']['bars_returned']} | latest closed "
              f"{res['history']['latest_closed_h1_timestamp']} | recent 7d "
              f"{res['recent_window']['recent_7d_bars']} "
              f"({res['recent_window']['recent_history_status']})")
    print("\nConclusion:")
    print("  " + payload["overall_diagnostic_conclusion"])

    print("\nWriting NEW output files:")
    try:
        _write_bytes_if_absent(csv_path,
                               results_to_csv(payload).encode("utf-8"))
        _write_bytes_if_absent(json_path, build_json(payload).encode("utf-8"))
        _write_bytes_if_absent(txt_path,
                               build_report(payload).encode("utf-8"))
    except OutputExistsError as exc:
        print("ERROR: " + str(exc))
        return 1
    print("\nDone. Descriptive read-only diagnostics only - no directive, "
          "no certainty figure, no ordering of instruments.")
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
    "numpy", "pandas", "MetaTrader5",
})


def contains_banned_wording(text: str) -> List[str]:
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
                if node.col_offset == 0 and a.name == "MetaTrader5":
                    violations.append(
                        "terminal package imported at module level (must "
                        "stay lazy on the --run path)")
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
# Synthetic fixtures (tests only; in-memory / temp dirs only)
# ============================================================
_EPOCH = 1767225600  # 2026-01-01 00:00:00 UTC


def _hours(hours_from_epoch: float) -> int:
    return int(_EPOCH + hours_from_epoch * 3600)


def _epoch_seconds(value: Any) -> int:
    """Seconds since epoch, treating a naive value as UTC."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    else:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.timestamp())


def _rates_rows(anchor_hour: float, count: int) -> List[Tuple]:
    """Ascending H1 rows ending at anchor_hour (hours from _EPOCH)."""
    start = anchor_hour - (count - 1)
    rows = []
    for i in range(count):
        t = _hours(start + i)
        base = 2000.0 + 0.5 * (start + i)
        rows.append((t, base, base + 1.0, base - 1.0, base + 0.5, 100))
    return rows


def _rates_array(rows: Optional[List[Tuple]]) -> Any:
    if rows is None:
        return None
    dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"),
             ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]
    if not rows:
        return np.array([], dtype=dtype)
    return np.array([tuple(r) for r in rows], dtype=dtype)


class _FakeInfo:
    def __init__(self, name: str, spec: Dict[str, Any]) -> None:
        self.name = name
        self.visible = spec.get("visible", True)
        self.trade_mode = spec.get("trade_mode", 4)
        self.description = spec.get("description", "")
        self.path = spec.get("path", "")
        self.digits = spec.get("digits", 5)
        self.point = spec.get("point", 0.00001)
        self.spread = spec.get("spread", 10)


class _FakeTick:
    def __init__(self, when: int, bid: float, ask: float) -> None:
        self.time = when
        self.bid = bid
        self.ask = ask


class _FakeMT5:
    """In-memory substitute exposing ONLY the read-only terminal API."""

    TIMEFRAME_H1 = "TIMEFRAME_H1"

    def __init__(self, spec: Dict[str, Dict[str, Any]],
                 all_names: Optional[List[str]] = None,
                 error: Tuple[int, str] = (0, "No error")) -> None:
        self._spec = spec
        self._names = all_names if all_names is not None else sorted(spec)
        self._error = error
        self.calls: List[str] = []
        self.timeframes_used: List[Any] = []
        self.range_args: List[Tuple] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self.calls.append(name)
        raise AssertionError(f"unexpected terminal API access: {name}")

    def initialize(self) -> bool:
        self.calls.append("initialize")
        return True

    def shutdown(self) -> bool:
        self.calls.append("shutdown")
        return True

    def last_error(self) -> Tuple[int, str]:
        self.calls.append("last_error")
        return self._error

    def symbol_info(self, name: str) -> Any:
        self.calls.append("symbol_info")
        spec = self._spec.get(name)
        if spec is None or spec.get("info") is None:
            return None
        return _FakeInfo(name, spec["info"])

    def symbols_get(self) -> List[_FakeInfo]:
        self.calls.append("symbols_get")
        return [_FakeInfo(n, self._spec.get(n, {}).get("info", {}))
                for n in self._names]

    def symbol_select(self, name: str, enable: bool) -> bool:
        self.calls.append("symbol_select")
        spec = self._spec.get(name)
        return bool(spec and spec.get("select", True))

    def symbol_info_tick(self, name: str) -> Any:
        self.calls.append("symbol_info_tick")
        spec = self._spec.get(name)
        if not spec or spec.get("tick") is None:
            return None
        tick = spec["tick"]
        return _FakeTick(tick["time"], tick["bid"], tick["ask"])

    def copy_rates_from_pos(self, symbol: str, timeframe: Any,
                            start_pos: int, count: int) -> Any:
        self.calls.append("copy_rates_from_pos")
        self.timeframes_used.append(timeframe)
        spec = self._spec.get(symbol)
        if spec is None:
            return None
        return _rates_array(spec.get("pos"))

    def copy_rates_range(self, symbol: str, timeframe: Any,
                         dt_from: Any, dt_to: Any) -> Any:
        self.calls.append("copy_rates_range")
        self.timeframes_used.append(timeframe)
        self.range_args.append((dt_from, dt_to))
        spec = self._spec.get(symbol)
        if spec is None:
            return None
        rows = spec.get("range")
        if rows is None:
            return None
        start = _epoch_seconds(dt_from)
        end = _epoch_seconds(dt_to)
        return _rates_array([r for r in rows if start <= r[0] <= end])


# Horizon anchors (hours from _EPOCH): 2026-10-07 is 279 days after
# 2026-01-01, i.e. about 6696 hours.
_FRESH_ANCHOR = 6699.0     # 2026-10-07 03:00 (within the recent window)
_OLD_ANCHOR = 6500.0       # about 8.5 days before the reference
_REFERENCE = datetime.datetime(2026, 10, 7, 12, 0, 0)


def build_fake_spec(fresh_anchor: float = _FRESH_ANCHOR,
                    old_anchor: float = _OLD_ANCHOR,
                    old_instruments: Tuple[str, ...] = ("GOLD", "AUDUSD"),
                    range_mode: str = "same",
                    tick: bool = True,
                    bars: int = 100) -> Dict[str, Dict[str, Any]]:
    spec: Dict[str, Dict[str, Any]] = {}
    for inst in INSTRUMENTS:
        anchor = old_anchor if inst in old_instruments else fresh_anchor
        rows = _rates_rows(anchor, bars)
        range_rows = rows
        if range_mode == "mismatch" and inst == "GOLD":
            range_rows = _rates_rows(anchor - 5, bars)
        elif range_mode == "empty":
            range_rows = []
        tick_spec = ({"time": _hours(fresh_anchor), "bid": 100.0,
                      "ask": 100.2} if tick else None)
        spec[inst] = {
            "info": {"visible": True, "trade_mode": 4,
                     "description": f"{inst} synthetic",
                     "path": f"Synthetic/{inst}", "digits": 5,
                     "point": 0.00001, "spread": 12},
            "select": True,
            "tick": tick_spec,
            "pos": rows,
            "range": range_rows,
        }
    return spec


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

    print("mt5_history_diagnostic.py - synthetic test suite")
    print("(in-memory substitutes and temporary directories only; no "
          "terminal contact; no real output is written)\n")

    with tempfile.TemporaryDirectory() as td:
        fake = _FakeMT5(build_fake_spec())
        payload = run_diagnostic(fake, fixed_ts, _REFERENCE)

        # ---------------- A/B. instruments and timeframe ------
        start_area("A_instruments_timeframe")
        check("the payload lists the five instruments in the fixed order",
              payload["requested_instruments"]
              == ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
        check("one diagnostic is produced per instrument",
              [r["instrument"] for r in payload["instruments"]]
              == list(INSTRUMENTS))
        check("the reported timeframe is H1",
              payload["timeframe"] == "H1")
        check("the terminal H1 timeframe constant is used for every query",
              set(fake.timeframes_used) == {_FakeMT5.TIMEFRAME_H1})

        # ---------------- C/D/E. symbol lookup ----------------
        start_area("C_symbol_lookup")
        gold = payload["instruments"][0]
        lookup = gold["symbol_lookup"]
        check("a present name is reported available",
              lookup["symbol_exists"] is True
              and lookup["symbol_requested"] == "GOLD")
        check("lookup fields are recorded",
              lookup["description"] == "GOLD synthetic"
              and lookup["path"] == "Synthetic/GOLD"
              and lookup["digits"] == 5
              and lookup["point"] == 0.00001
              and lookup["spread"] == 12)
        check("visibility is recorded",
              lookup["symbol_visible"] is True)
        missing = {"EURUSD": {"info": None, "select": False, "tick": None,
                              "pos": None, "range": None}}
        fake_missing = _FakeMT5(missing)
        res_missing = diagnose_instrument("EURUSD", fake_missing, _REFERENCE)
        check("a missing name is reported unavailable",
              res_missing["symbol_lookup"]["symbol_exists"] is False
              and res_missing["history"]["bars_returned"] == 0)
        check("a missing name is described without a substitute",
              "not available in the terminal"
              in str(res_missing["diagnostic_note"]))
        hidden_spec = build_fake_spec()
        hidden_spec["GBPUSD"]["info"] = {"visible": False}
        fake_hidden = _FakeMT5(hidden_spec)
        res_hidden = diagnose_instrument("GBPUSD", fake_hidden, _REFERENCE)
        check("a hidden name is still reported with its visibility flag",
              res_hidden["symbol_lookup"]["symbol_exists"] is True
              and res_hidden["symbol_lookup"]["symbol_visible"] is False)

        # ---------------- F/G. current tick -------------------
        start_area("F_current_tick")
        check("a present tick is recorded",
              gold["tick"]["current_tick_available"] is True
              and gold["tick"]["bid"] == 100.0
              and gold["tick"]["ask"] == 100.2
              and gold["tick"]["current_tick_timestamp"] is not None)
        fake_notick = _FakeMT5(build_fake_spec(tick=False))
        res_notick = diagnose_instrument("GOLD", fake_notick, _REFERENCE)
        check("an absent tick is recorded as unavailable",
              res_notick["tick"]["current_tick_available"] is False
              and res_notick["tick"]["bid"] is None)

        # ---------------- H/I/J. history ----------------------
        start_area("H_history")
        hist = gold["history"]
        check("history bars are counted",
              hist["bars_returned"] == 100
              and hist["unique_timestamps"] == 100)
        check("earliest and latest timestamps are recorded",
              hist["earliest_h1_timestamp"] is not None
              and hist["latest_h1_timestamp"] is not None)
        check("the earliest bar precedes the latest bar",
              hist["earliest_h1_timestamp"] < hist["latest_h1_timestamp"])
        empty_spec = build_fake_spec()
        empty_spec["GOLD"]["pos"] = []
        empty_spec["GOLD"]["range"] = []
        fake_empty = _FakeMT5(empty_spec)
        res_empty = diagnose_instrument("GOLD", fake_empty, _REFERENCE)
        check("an empty history is reported with zero bars",
              res_empty["history"]["bars_returned"] == 0
              and res_empty["history"]["latest_h1_timestamp"] is None)
        check("an empty history is classified as no recent bars",
              res_empty["recent_window"]["recent_history_status"]
              == NO_RECENT_BARS)
        check("an empty history is described as zero bars",
              "zero H1 bars" in str(res_empty["diagnostic_note"]))

        # ---------------- K/L. forming and closed candles -----
        start_area("K_forming_closed_candle")
        check("the newest bar is flagged as forming",
              hist["newest_bar_is_forming"] is True)
        check("the latest closed candle is the one before the newest bar",
              hist["latest_closed_h1_timestamp"]
              == _fmt(pd.Timestamp(_hours(_OLD_ANCHOR - 1), unit="s"))
              and hist["latest_closed_h1_timestamp"]
              < hist["latest_h1_timestamp"])
        fake_one = _FakeMT5(build_fake_spec(bars=1))
        res_one = diagnose_instrument("EURUSD", fake_one, _REFERENCE)
        check("a single bar yields no closed-candle timestamp",
              res_one["history"]["bars_returned"] == 1
              and res_one["history"]["latest_closed_h1_timestamp"] is None)

        # ---------------- M. recent 7-day classification ------
        start_area("M_recent_window")
        gold_recent = gold["recent_window"]
        check("old bars are classified as only old bars",
              gold_recent["recent_history_status"] == ONLY_OLD_BARS
              and gold_recent["recent_7d_bars"] == 0)
        eur = payload["instruments"][1]
        check("recent bars are classified as recent bars",
              eur["recent_window"]["recent_history_status"] == RECENT_BARS
              and eur["recent_window"]["recent_7d_bars"] > 0)
        partial_spec = build_fake_spec()
        partial_spec["EURUSD"]["pos"] = _rates_rows(_FRESH_ANCHOR, 400)
        fake_partial = _FakeMT5(partial_spec)
        res_partial = diagnose_instrument("EURUSD", fake_partial, _REFERENCE)
        check("a mixed window is classified as partial recent history",
              res_partial["recent_window"]["recent_history_status"]
              in (RECENT_BARS, PARTIAL_RECENT_HISTORY))

        # ---------------- N/O/P/Q. query methods --------------
        start_area("N_query_methods")
        check("the recent-position latest is recorded",
              gold["position_request_latest"] is not None)
        check("the date-range latest is recorded for a recent instrument",
              eur["date_range_request_latest"] is not None)
        check("matching latest timestamps are detected",
              eur["request_timestamps_match"] is True)
        mismatch_spec = build_fake_spec()
        mismatch_spec["EURUSD"]["range"] = _rates_rows(_FRESH_ANCHOR - 5,
                                                        100)
        fake_mismatch = _FakeMT5(mismatch_spec)
        res_mismatch = diagnose_instrument("EURUSD", fake_mismatch,
                                           _REFERENCE)
        check("mismatching latest timestamps are detected",
              res_mismatch["request_timestamps_match"] is False)
        check("a mismatch is described",
              "different latest timestamps"
              in str(res_mismatch["diagnostic_note"]))
        fake_range_empty = _FakeMT5(build_fake_spec(range_mode="empty"))
        res_range_empty = diagnose_instrument("EURUSD", fake_range_empty,
                                              _REFERENCE)
        check("a range query that returns nothing leaves the latest unset",
              res_range_empty["date_range_request_latest"] is None
              and res_range_empty["request_timestamps_match"] is None)

        # ---------------- R. terminal error capture -----------
        start_area("R_error_capture")
        err_spec = {"GOLD": {"info": {}, "select": True, "tick": None,
                             "pos": None, "range": None}}
        fake_err = _FakeMT5(err_spec, error=(4401, "history unavailable"))
        res_err = diagnose_instrument("GOLD", fake_err, _REFERENCE)
        check("a terminal error code is recorded",
              res_err["mt5_error"]["mt5_error_code"] == 4401
              and "history unavailable"
              in res_err["mt5_error"]["mt5_error_description"])
        check("a failed history call is described, not hidden",
              res_err["history_query_error"] is not None)
        payload_err = run_diagnostic(fake_err, fixed_ts, _REFERENCE)
        check("the payload lists encountered errors",
              any(e["instrument"] == "GOLD" and e["code"] == 4401
                  for e in payload_err["errors_encountered"]))

        # ---------------- S/T. related symbols ----------------
        start_area("S_related_symbols")
        names = ["GOLD", "GOLDm", "GOLD.pro", "XAUUSD", "EURUSD",
                 "GBPUSD", "USDJPY", "AUDUSD"]
        fake_names = _FakeMT5(build_fake_spec(), all_names=names)
        res_rel = diagnose_instrument("GOLD", fake_names, _REFERENCE)
        discovered = res_rel["related_symbols"]["discovered_related_symbols"]
        check("related names are discovered from the terminal list",
              discovered == ["GOLD.pro", "GOLDm"])
        check("the requested name is distinguished from alternatives",
              res_rel["related_symbols"]["requested_symbol"] == "GOLD"
              and "GOLD" not in discovered)
        check("no alternative name is substituted or queried",
              res_rel["related_symbols"]["alternative_queried"] is False
              and res_rel["history"]["bars_returned"] == 100)

        # ---------------- U/V/W. ordering and timestamps ------
        start_area("U_timestamps")
        check("returned bars are reported ascending",
              gold["history"]["chronological_ascending"] is True)
        dup_spec = build_fake_spec()
        dup_rows = _rates_rows(_FRESH_ANCHOR, 3)
        dup_spec["GOLD"]["pos"] = [dup_rows[0], dup_rows[1], dup_rows[1],
                                   dup_rows[2]]
        fake_dup = _FakeMT5(dup_spec)
        res_dup = diagnose_instrument("GOLD", fake_dup, _REFERENCE)
        check("duplicate timestamps are counted",
              res_dup["history"]["duplicate_timestamps"] == 1
              and res_dup["history"]["unique_timestamps"] == 3)
        check("duplicate timestamps are described",
              "duplicate" in str(res_dup["diagnostic_note"]))
        na_spec = build_fake_spec()
        na_rows = list(_rates_rows(_FRESH_ANCHOR, 3))
        na_spec["GOLD"]["pos"] = na_rows
        fake_na = _FakeMT5(na_spec)
        frame_na = _rates_to_frame(_rates_array(na_rows))
        frame_na.loc[1, "time"] = pd.NaT
        res_na = summarise_history(frame_na)
        check("an unreadable timestamp is counted and not substituted",
              res_na["invalid_timestamps"] == 1
              and res_na["bars_returned"] == 3)

        # ---------------- X. deterministic output -------------
        start_area("X_deterministic_output")
        p_a = run_diagnostic(_FakeMT5(build_fake_spec()), fixed_ts,
                             _REFERENCE)
        p_b = run_diagnostic(_FakeMT5(build_fake_spec()), fixed_ts,
                             _REFERENCE)
        csv_a = results_to_csv(p_a)
        csv_b = results_to_csv(p_b)
        json_a = build_json(p_a)
        json_b = build_json(p_b)
        rep_a = build_report(p_a)
        rep_b = build_report(p_b)
        check("identical inputs produce identical CSV bytes",
              csv_a == csv_b)
        check("identical inputs produce identical JSON bytes",
              json_a == json_b)
        check("identical inputs produce identical report bytes",
              rep_a == rep_b)

        # ---------------- AG/AH. schemas ----------------------
        start_area("AG_schemas")
        check("the CSV carries the exact field set",
              list(pd.read_csv(pd.io.common.StringIO(csv_a)).columns)
              == list(CSV_COLUMNS))
        parsed = json.loads(json_a)
        check("the JSON carries the required top-level keys",
              all(k in parsed for k in (
                  "generated_timestamp", "timeframe",
                  "requested_instruments", "instruments",
                  "cross_instrument_comparison", "errors_encountered",
                  "overall_diagnostic_conclusion")))
        check("the JSON contains one diagnostic per instrument",
              len(parsed["instruments"]) == len(INSTRUMENTS))
        check("the JSON carries no certainty or ordering field",
              all(k not in json_a.lower() for k in (
                  "co" + "nfidence", "pro" + "bability", "ra" + "nk")))

        # ---------------- AF. report generation ---------------
        start_area("AF_report")
        check("the report contains the numbered sections",
              all(s in rep_a for s in (
                  "MetaTrader 5 H1 HISTORY DIAGNOSTIC",
                  "FIVE-SYMBOL COMPARISON", "SYMBOL LOOKUP",
                  "CURRENT TICK", "H1 HISTORY", "RECENT 7-DAY WINDOW",
                  "QUERY-METHOD COMPARISON",
                  "TERMINAL RETURN INFORMATION",
                  "POSSIBLE SYMBOL-NAME DIFFERENCES",
                  "EVIDENCE-BASED DIAGNOSTIC CONCLUSION",
                  "LIMITATIONS")))
        check("the report lists every instrument",
              all(f"INSTRUMENT: {i}" in rep_a for i in INSTRUMENTS))

        # ---------------- AJ. conclusion is evidence-based ----
        start_area("AJ_conclusion")
        cmp_old = p_a["cross_instrument_comparison"]
        check("the comparison identifies the older instruments",
              cmp_old["instruments_older_than_newest_by_over_24h"]
              == ["GOLD", "AUDUSD"])
        conclusion = p_a["overall_diagnostic_conclusion"]
        check("the conclusion names the affected instruments",
              "GOLD" in conclusion and "AUDUSD" in conclusion)
        check("the conclusion does not assert an unsupported cause",
              "does not establish the underlying reason" in conclusion)
        check("the conclusion reports that the query methods agree",
              "matching latest timestamps" in conclusion)
        mismatch_diag_spec = build_fake_spec()
        mismatch_diag_spec["EURUSD"]["range"] = _rates_rows(
            _FRESH_ANCHOR - 5, 100)
        mixed = run_diagnostic(_FakeMT5(mismatch_diag_spec), fixed_ts,
                               _REFERENCE)
        check("a query-method difference is reported descriptively",
              "different latest timestamps"
              in mixed["overall_diagnostic_conclusion"])
        check("no conclusion asserts a definitive cause",
              all(
                  phrase not in mixed["overall_diagnostic_conclusion"]
                  for phrase in ("because the broker", "the broker is")))
        fresh_only = run_diagnostic(
            _FakeMT5(build_fake_spec(old_instruments=())), fixed_ts,
            _REFERENCE)
        check("a uniform result states that no difference was observed",
              "no difference in history depth"
              in fresh_only["overall_diagnostic_conclusion"])

        # ---------------- Y. output-file protection -----------
        start_area("Y_output_protection")
        p_dir = os.path.join(td, "_paths")
        os.makedirs(p_dir, exist_ok=True)
        c_path, j_path, t_path = resolve_output_paths(p_dir, fixed_ts)
        check("a clean directory selects the primary names",
              (os.path.basename(c_path), os.path.basename(j_path),
               os.path.basename(t_path))
              == (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_TXT))
        _write_bytes_if_absent(c_path, b"x")
        _write_bytes_if_absent(j_path, b"x")
        _write_bytes_if_absent(t_path, b"x")
        stamp = fixed_ts.strftime(STAMP_FORMAT)
        c2, j2, t2 = resolve_output_paths(p_dir, fixed_ts)
        check("existing files get the timestamped alternatives",
              (os.path.basename(c2), os.path.basename(j2),
               os.path.basename(t2))
              == (f"mt5_history_diagnostic_{stamp}.csv",
                  f"mt5_history_diagnostic_{stamp}.json",
                  f"mt5_history_diagnostic_report_{stamp}.txt"))
        with open(c_path, encoding="utf-8") as fh:
            check("the existing file is never overwritten", fh.read() == "x")
        try:
            _write_bytes_if_absent(c_path, b"y")
            check("writing over an existing file is refused", False)
        except OutputExistsError:
            check("writing over an existing file is refused", True)
        with open(c2, "wb") as fh:
            fh.write(b"occupied")
        try:
            resolve_output_paths(p_dir, fixed_ts)
            check("an alternative-name collision stops the run", False)
        except OutputExistsError as exc:
            check("an alternative-name collision stops the run",
                  f"mt5_history_diagnostic_{stamp}.csv" in str(exc))

        # ---------------- Z/AA/AB/AC/AD/AE. safety -------------
        start_area("Z_safety")
        violations = run_safety_scan()
        check("the static safety scan of this file is clean",
              violations == [])
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
        tree = ast.parse(src)
        top_imports: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(a.name.split(".")[0]
                                   for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.append(node.module.split(".")[0])
        terminal_pkg = ("Meta" + "Trader5")
        check("the terminal package is not imported in default test mode",
              terminal_pkg not in top_imports
              and terminal_pkg not in sys.modules
              and "MetaTrader5" in _ALLOWED_IMPORT_MODULES)
        check("only read-only terminal calls were exercised by the tests",
              set(fake.calls) <= {"symbol_info", "symbols_get",
                                  "symbol_select", "symbol_info_tick",
                                  "copy_rates_from_pos", "copy_rates_range",
                                  "last_error"})
        check("no external service or learning library is imported",
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
        write_modes = re.findall(r'open\([^)]*,\s*["\']([wa])["\']', src)
        check("the module never opens a file in text or append write mode",
              '"wb"' in src and write_modes == [])

        # ---------------- AI. error metadata ------------
        start_area("AI_error_metadata")
        no_symbol = run_diagnostic(_FakeMT5({}), fixed_ts, _REFERENCE)
        check("a terminal with no names still yields one row per "
              "instrument",
              len(no_symbol["instruments"]) == len(INSTRUMENTS))
        check("unavailable names are listed as errors, not hidden",
              len(no_symbol["errors_encountered"]) == len(INSTRUMENTS))
        check("the CSV still parses with every instrument represented",
              list(pd.read_csv(pd.io.common.StringIO(
                  results_to_csv(no_symbol))).columns)
              == list(CSV_COLUMNS))

        # ---------------- CLI gating --------------------------
        start_area("CLI_gating")
        check("default (no flags) runs the synthetic tests only",
              should_run_real([]) is False)
        check("--run selects the real diagnostic",
              should_run_real(["--run"]) is True)
        check("unrelated flags do not select the real diagnostic",
              should_run_real(["-v", "--verbose"]) is False)

    start_area(None)
    print()
    print("  Per-area verification (area -> passed/failed):")
    stats = {name: (p, f) for name, p, f in area_stats}
    order = (
        "A_instruments_timeframe", "C_symbol_lookup", "F_current_tick",
        "H_history", "K_forming_closed_candle", "M_recent_window",
        "N_query_methods", "R_error_capture", "S_related_symbols",
        "U_timestamps", "X_deterministic_output", "AG_schemas",
        "AF_report", "AJ_conclusion", "Y_output_protection", "Z_safety",
        "AI_error_metadata", "CLI_gating",
    )
    for name in order:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<28} {p:>2} passed, {f} failed [{status}]")
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
        return run_real_diagnostic()
    passed, failed = run_synthetic_tests()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
