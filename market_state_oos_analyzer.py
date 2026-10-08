"""
market_state_oos_analyzer.py - HISTORICAL RESEARCH / VALIDATION ONLY.

Research question:
    "Do the descriptive market-state relationships observed in earlier
     historical signal data persist in later, unseen historical data?"

This tool answers factual persistence questions only:
    - Does a state/outcome relationship persist across time?
    - Does the sign of average P/L stay the same?
    - Does the sign of forward returns stay the same?
    - Does a relationship disappear or reverse?
    - Are relationships consistent across instruments?
Nothing here is a trading rule.

THIS TOOL DOES NOT:
    - create any trading rule, filter or signal,
    - optimize anything (no parameter loops, no threshold sweeps),
    - rank instruments, regimes, buckets or signals,
    - use machine learning of any kind,
    - connect to a trading terminal or touch orders in any way
      (there is no terminal import anywhere in this file),
    - modify any existing file; it reads the existing CSVs and writes
      only the four NEW output CSVs listed below.

DATA SOURCE (read-only):
    market_state_GOLD.csv, market_state_EURUSD.csv,
    market_state_GBPUSD.csv, market_state_USDJPY.csv,
    market_state_AUDUSD.csv
    as written by market_state_database.py. If one is missing the run
    stops with a clear error naming the file.

CHRONOLOGICAL VALIDATION:
    Per instrument: signals are sorted by timestamp ascending (never
    shuffled) and cut into four equal-COUNT periods P1..P4 using the
    same boundaries as market_state_database.py.

FROZEN STATE DEFINITIONS (identical to market_state_database.py):
    ADX           <15 / 15-20 / 20-25 / >=25
    ATR%          LOW / MEDIUM / HIGH - the CSVs do not store a label
                  column, so the database's EXISTING labelling function
                  (per-instrument 33rd/66th percentiles of atr_pct) is
                  reproduced exactly; no new thresholds are invented.
    EMA distance  <1 / 1-2 / >=2 (ATR units)
    BB excursion  <0.50 / 0.50-0.75 / >=0.75
    Efficiency    <0.30 / 0.30-0.60 / >=0.60
    RSI           <30 / 30-50 / 50-70 / >=70

DATA INTEGRITY:
    - no shuffling anywhere (verified by the synthetic suite),
    - no future data defines any earlier-period statistic,
    - low-sample rule: any cell with n < 5 is marked LOW SAMPLE and is
      excluded from factual persistence conclusions,
    - the walk-forward section is descriptive sign observation only -
      no state is ever selected, fitted or filtered.

COMMAND-LINE SAFETY:
    python market_state_oos_analyzer.py          -> synthetic tests only
    python market_state_oos_analyzer.py --run    -> real CSV analysis

NEW OUTPUT CSV FILES (existing files are NEVER overwritten; the run
stops with a clear message if an output file already exists):
    market_state_oos_period_summary.csv
    market_state_oos_state_summary.csv
    market_state_oos_walkforward.csv
    market_state_oos_cross_instrument.csv

DISCLAIMER: Historical, descriptive research only. No causal claims,
no predictions, no rankings, no recommendations, nothing proven
profitable. Educational use only.
"""

# ============================================================
# Imports: Python standard library + numpy + pandas ONLY.
# No ML library and no trading terminal is imported anywhere.
# ============================================================
import ast
import inspect
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# SAFETY GUARDS (verified against this file's own source by the
# synthetic suite). Forbidden tokens are assembled from split string
# literals so this safety code never contains the contiguous strings.
# ============================================================
_FORBIDDEN_CALL_TOKENS: Tuple[str, ...] = (
    "order" + "_send",
    "order" + "_check",
    "positions" + "_get",
    "positions" + "_total",
    "position" + "_get",
)
_ML_LIBRARY_TOKENS: Tuple[str, ...] = (
    "sklearn", "xgboost", "lightgbm", "tensorflow", "torch", "keras",
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

# ============================================================
# Frozen configuration
# ============================================================
EXPECTED_INSTRUMENTS: Tuple[str, ...] = (
    "GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
)
INPUT_CSV_TEMPLATE = "market_state_{instrument}.csv"

OUTPUT_FILES: Tuple[str, ...] = (
    "market_state_oos_period_summary.csv",
    "market_state_oos_state_summary.csv",
    "market_state_oos_walkforward.csv",
    "market_state_oos_cross_instrument.csv",
)

PERIOD_LABELS: Tuple[str, ...] = ("P1", "P2", "P3", "P4")
LOW_SAMPLE_N = 5  # cells with n < LOW_SAMPLE_N are marked LOW SAMPLE

REQUIRED_INPUT_COLUMNS: Tuple[str, ...] = (
    "instrument", "timeframe", "signal_time", "signal_index", "direction",
    "adx", "atr_pct", "ema_distance", "bb_excursion", "eff24", "rsi14",
    "net_pnl", "win",
    "forward6_pct", "forward12_pct", "forward24_pct",
    "mfe12_atr", "mae12_atr",
)

# Frozen bins - identical labels and boundaries to market_state_database.py
ADX_BIN_LABELS: Tuple[str, ...] = ("ADX<15", "ADX15-20", "ADX20-25", "ADX>=25")
ATR_BIN_LABELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
EMA_BIN_LABELS: Tuple[str, ...] = ("<1", "1-2", ">=2")
BBE_BIN_LABELS: Tuple[str, ...] = ("<0.50", "0.50-0.75", ">=0.75")
EFF_BIN_LABELS: Tuple[str, ...] = ("<0.30", "0.30-0.60", ">=0.60")
RSI_BIN_LABELS: Tuple[str, ...] = ("<30", "30-50", "50-70", ">=70")

# (report name, bin column, ordered labels) - fixed definition order
STATE_FEATURES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("adx", "adx_bin", ADX_BIN_LABELS),
    ("atr", "atr_bin", ATR_BIN_LABELS),
    ("ema_distance", "ema_bin", EMA_BIN_LABELS),
    ("bb_excursion", "bbe_bin", BBE_BIN_LABELS),
    ("efficiency", "eff_bin", EFF_BIN_LABELS),
    ("rsi", "rsi_bin", RSI_BIN_LABELS),
)

# If a database version ever stores an ATR label column it is used
# verbatim; otherwise the database's existing percentile labels are
# reproduced (never re-derived thresholds).
ATR_LABEL_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "atr_regime", "atr_bin", "atr_label", "atr_regime_label",
)

CORRELATION_FEATURES: Tuple[str, ...] = (
    "adx", "atr_pct", "ema_distance", "bb_excursion", "eff24", "rsi14",
)

WALK_STEPS: Tuple[Tuple[str, str], ...] = (
    ("P1", "P2"), ("P1+P2", "P3"), ("P1+P2+P3", "P4"),
)

# ============================================================
# Frozen bin functions - copied logic-for-logic from
# market_state_database.py. No threshold here may be changed.
# ============================================================
def bin_adx(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if x < 15:
        return ADX_BIN_LABELS[0]
    if x < 20:
        return ADX_BIN_LABELS[1]
    if x < 25:
        return ADX_BIN_LABELS[2]
    return ADX_BIN_LABELS[3]


def bin_ema_distance(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if x < 1:
        return EMA_BIN_LABELS[0]
    if x < 2:
        return EMA_BIN_LABELS[1]
    return EMA_BIN_LABELS[2]


def bin_bb_excursion(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if x < 0.50:
        return BBE_BIN_LABELS[0]
    if x < 0.75:
        return BBE_BIN_LABELS[1]
    return BBE_BIN_LABELS[2]


def bin_efficiency(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if x < 0.30:
        return EFF_BIN_LABELS[0]
    if x < 0.60:
        return EFF_BIN_LABELS[1]
    return EFF_BIN_LABELS[2]


def bin_rsi(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if x < 30:
        return RSI_BIN_LABELS[0]
    if x < 50:
        return RSI_BIN_LABELS[1]
    if x < 70:
        return RSI_BIN_LABELS[2]
    return RSI_BIN_LABELS[3]


def atr_bin_labels(values: pd.Series) -> pd.Series:
    """The database's existing ATR% labelling: per-instrument 33rd/66th
    percentiles -> LOW/MEDIUM/HIGH (descriptive labels only)."""
    p33 = float(values.quantile(1.0 / 3.0))
    p66 = float(values.quantile(2.0 / 3.0))
    return values.map(
        lambda x: "n/a" if pd.isna(x)
        else ("LOW" if x < p33 else ("MEDIUM" if x < p66 else "HIGH"))
    )


# ============================================================
# Small descriptive helpers
# ============================================================
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


def same_sign(a, b) -> Optional[bool]:
    sa = sign_of(a)
    sb = sign_of(b)
    if sa is None or sb is None:
        return None
    return sa == sb


def sign_result(a, b) -> str:
    same = same_sign(a, b)
    if same is None:
        return "n/a"
    return "SIGN SAME" if same else "SIGN CHANGED"


def short_sign(label: str) -> str:
    return {"SIGN SAME": "SAME", "SIGN CHANGED": "CHANGED"}.get(label, label)


def count_sign_changes(values: List[Optional[float]]) -> int:
    """Consecutive-period sign changes; missing values are skipped."""
    signs = [sign_of(v) for v in values]
    changes = 0
    for a, b in zip(signs, signs[1:]):
        if a is None or b is None:
            continue
        if a != b:
            changes += 1
    return changes


def classify_persistence(values: List[Optional[float]], ns: List[int]) -> str:
    """Descriptive classification of a P1..P4 average series.
    Any period with n < LOW_SAMPLE_N forces 'insufficient sample'."""
    if any(n < LOW_SAMPLE_N for n in ns):
        return "insufficient sample"
    if any(v is None or pd.isna(v) for v in values):
        return "insufficient sample"
    signs = [sign_of(v) for v in values]
    if all(s == 1 for s in signs):
        return "persistent positive"
    if all(s == -1 for s in signs):
        return "persistent negative"
    return "mixed"


def classify_cross_instrument(pos: int, neg: int, sufficient: int) -> str:
    """Factual cross-instrument sign classification (no ranking)."""
    if sufficient < 2:
        return "INSUFFICIENT SAMPLE"
    if pos > 0 and neg > 0:
        return "MIXED"
    return "CONSISTENT"


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation on pairwise-complete observations (or nan)."""
    pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    if len(pair) < 2:
        return float("nan")
    # Zero variance -> undefined correlation; return nan explicitly
    # (avoids divide-by-zero warnings deep inside numpy).
    if float(pair["x"].std(ddof=0)) == 0.0 \
            or float(pair["y"].std(ddof=0)) == 0.0:
        return float("nan")
    return float(pair["x"].corr(pair["y"]))


def cell_stats(sub: pd.DataFrame) -> dict:
    """Descriptive outcome statistics for one group of signal rows."""
    if len(sub) == 0:
        return {
            "n": 0, "avg_net_pnl": float("nan"),
            "median_net_pnl": float("nan"), "win_rate_pct": float("nan"),
            "profit_factor": float("nan"), "fwd6_avg_pct": float("nan"),
            "fwd12_avg_pct": float("nan"), "fwd24_avg_pct": float("nan"),
            "mfe12_avg_atr": float("nan"), "mae12_avg_atr": float("nan"),
        }
    net = sub["net_pnl"].astype(float)
    gp = float(net[net > 0].sum())
    gl = float(abs(net[net < 0].sum()))
    if gl > 0:
        pf = gp / gl
    elif gp > 0:
        pf = float("inf")
    else:
        pf = float("nan")
    return {
        "n": int(len(sub)),
        "avg_net_pnl": float(net.mean()),
        "median_net_pnl": float(net.median()),
        "win_rate_pct": float((net > 0).mean() * 100.0),
        "profit_factor": pf,
        "fwd6_avg_pct": float(sub["forward6_pct"].mean()),
        "fwd12_avg_pct": float(sub["forward12_pct"].mean()),
        "fwd24_avg_pct": float(sub["forward24_pct"].mean()),
        "mfe12_avg_atr": float(sub["mfe12_atr"].mean()),
        "mae12_avg_atr": float(sub["mae12_atr"].mean()),
    }


# ============================================================
# Read-only CSV loading (the only data access in this tool)
# ============================================================
def validate_input_frame(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required column(s): {', '.join(missing)}")


def load_instrument_csv(instrument: str, directory: Optional[str] = None
                        ) -> pd.DataFrame:
    directory = directory if directory is not None else SCRIPT_DIR
    path = os.path.join(directory,
                        INPUT_CSV_TEMPLATE.format(instrument=instrument))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file is missing: {path} "
            f"(expected the historical database CSV for {instrument} as "
            f"written by market_state_database.py). Analysis stopped.")
    frame = pd.read_csv(path)
    validate_input_frame(frame, path)
    frame["signal_time"] = pd.to_datetime(frame["signal_time"])
    return frame


# ============================================================
# Preparation: sorting (never shuffled), frozen bins, periods
# ============================================================
def attach_state_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["adx_bin"] = out["adx"].map(bin_adx)
    out["ema_bin"] = out["ema_distance"].map(bin_ema_distance)
    out["bbe_bin"] = out["bb_excursion"].map(bin_bb_excursion)
    out["eff_bin"] = out["eff24"].map(bin_efficiency)
    out["rsi_bin"] = out["rsi14"].map(bin_rsi)
    label_col = next((c for c in ATR_LABEL_COLUMN_CANDIDATES
                      if c in out.columns), None)
    if label_col is not None:
        out["atr_bin"] = out[label_col].astype(str)
    else:
        out["atr_bin"] = atr_bin_labels(out["atr_pct"].astype(float))
    return out


def assign_periods(db: pd.DataFrame) -> pd.DataFrame:
    """Sort ascending by signal time (stable; never shuffled) and tag
    four equal-count periods using the same boundaries as the database
    code: rows [p*n//4, (p+1)*n//4) belong to period p+1."""
    out = db.copy()
    if len(out) == 0:
        out["period"] = pd.Series(dtype=object)
        return out
    ordered = out.sort_values(["signal_time", "signal_index"],
                              kind="mergesort")
    n = len(ordered)
    labels = np.empty(n, dtype=object)
    for p in range(4):
        labels[p * n // 4: (p + 1) * n // 4] = PERIOD_LABELS[p]
    ordered = ordered.copy()
    ordered["period"] = labels
    return ordered.reset_index(drop=True)


def atr_label_source(df: pd.DataFrame) -> str:
    found = next((c for c in ATR_LABEL_COLUMN_CANDIDATES
                  if c in df.columns), None)
    if found is not None:
        return f"stored CSV column '{found}' (used verbatim)"
    return ("no ATR label column stored in the CSV - the database's "
            "existing per-instrument p33/p66 percentile labels are "
            "reproduced exactly (no new thresholds)")


def prepare_databases(dbs: Dict[str, pd.DataFrame]
                      ) -> Dict[str, pd.DataFrame]:
    prepared: Dict[str, pd.DataFrame] = {}
    for ins, db in dbs.items():
        frame = db.copy()
        frame["signal_time"] = pd.to_datetime(frame["signal_time"])
        frame = attach_state_bins(frame)
        prepared[ins] = assign_periods(frame)
    return prepared


def ordered_instruments(prepared: Dict[str, pd.DataFrame]) -> List[str]:
    """EXPECTED_INSTRUMENTS order first, then any extra (synthetic test)
    instruments in their insertion order."""
    ordered = [ins for ins in EXPECTED_INSTRUMENTS if ins in prepared]
    ordered += [ins for ins in prepared if ins not in EXPECTED_INSTRUMENTS]
    return ordered


# ============================================================
# Section builders (all descriptive; nothing selects or filters)
# ============================================================
def build_period_summary(prepared: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for ins in ordered_instruments(prepared):
        df = prepared[ins]
        for p_label in PERIOD_LABELS:
            part = df[df["period"] == p_label]
            if len(part) == 0:
                continue
            rows.append({
                "instrument": ins,
                "period": p_label,
                "start_time": str(part["signal_time"].iloc[0]),
                "end_time": str(part["signal_time"].iloc[-1]),
                "long": int((part["direction"] == "LONG").sum()),
                "short": int((part["direction"] == "SHORT").sum()),
                **cell_stats(part),
            })
    return pd.DataFrame(rows)


def persistence_record(cells: List[dict]) -> dict:
    """P1..P4 sign-persistence facts for one instrument/bucket."""
    pnls = [c["avg_net_pnl"] for c in cells]
    ns = [c["n"] for c in cells]
    rec: dict = {}
    for idx, label in enumerate(PERIOD_LABELS, start=1):
        rec[f"p{idx}_avg_net_pnl"] = pnls[idx - 1]
        rec[f"p{idx}_n"] = ns[idx - 1]
    rec["pnl_pos_periods"] = sum(1 for v in pnls if sign_of(v) == 1)
    rec["pnl_neg_periods"] = sum(1 for v in pnls if sign_of(v) == -1)
    rec["pnl_sign_changes"] = count_sign_changes(pnls)
    rec["pnl_classification"] = classify_persistence(pnls, ns)
    for metric, key in (("fwd6", "fwd6_avg_pct"),
                        ("fwd12", "fwd12_avg_pct"),
                        ("fwd24", "fwd24_avg_pct")):
        vals = [c[key] for c in cells]
        for idx in range(4):
            rec[f"{metric}_p{idx + 1}"] = vals[idx]
        rec[f"{metric}_pos_periods"] = sum(1 for v in vals
                                           if sign_of(v) == 1)
        rec[f"{metric}_neg_periods"] = sum(1 for v in vals
                                           if sign_of(v) == -1)
        rec[f"{metric}_sign_changes"] = count_sign_changes(vals)
        rec[f"{metric}_classification"] = classify_persistence(vals, ns)
    return rec


def build_state_summary(prepared: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for ins in ordered_instruments(prepared):
        df = prepared[ins]
        for feature, col, labels in STATE_FEATURES:
            for bucket in labels:
                cells = []
                for p_label in PERIOD_LABELS:
                    sub = df[(df[col] == bucket)
                             & (df["period"] == p_label)]
                    cells.append(cell_stats(sub))
                persistence = persistence_record(cells)
                for p_idx, p_label in enumerate(PERIOD_LABELS):
                    rows.append({
                        "instrument": ins,
                        "feature": feature,
                        "bucket": bucket,
                        "period": p_label,
                        "low_sample": cells[p_idx]["n"] < LOW_SAMPLE_N,
                        **cells[p_idx],
                        **persistence,
                    })
    return pd.DataFrame(rows)


def build_direction_summary(prepared: Dict[str, pd.DataFrame]
                            ) -> pd.DataFrame:
    rows: List[dict] = []
    for ins in ordered_instruments(prepared):
        df = prepared[ins]
        for p_label in PERIOD_LABELS:
            for d in ("LONG", "SHORT"):
                sub = df[(df["direction"] == d) & (df["period"] == p_label)]
                if len(sub) == 0:
                    continue
                stats = cell_stats(sub)
                rows.append({
                    "instrument": ins,
                    "period": p_label,
                    "direction": d,
                    "n": stats["n"],
                    "avg_net_pnl": stats["avg_net_pnl"],
                    "win_rate_pct": stats["win_rate_pct"],
                    "profit_factor": stats["profit_factor"],
                    "fwd12_avg_pct": stats["fwd12_avg_pct"],
                })
    return pd.DataFrame(rows)


def build_cross_instrument_summary(prepared: Dict[str, pd.DataFrame]
                                   ) -> pd.DataFrame:
    rows: List[dict] = []
    for feature, col, labels in STATE_FEATURES:
        for bucket in labels:
            for p_label in PERIOD_LABELS:
                pos = neg = pos_f = neg_f = sufficient = 0
                for ins in ordered_instruments(prepared):
                    df = prepared[ins]
                    sub = df[(df[col] == bucket)
                             & (df["period"] == p_label)]
                    if len(sub) < LOW_SAMPLE_N:
                        continue  # low-sample instruments are not counted
                    sufficient += 1
                    pnl_sign = sign_of(float(sub["net_pnl"].mean()))
                    f12_sign = sign_of(float(sub["forward12_pct"].mean()))
                    if pnl_sign == 1:
                        pos += 1
                    elif pnl_sign == -1:
                        neg += 1
                    if f12_sign == 1:
                        pos_f += 1
                    elif f12_sign == -1:
                        neg_f += 1
                rows.append({
                    "feature": feature,
                    "bucket": bucket,
                    "period": p_label,
                    "instruments_with_sufficient_sample": sufficient,
                    "pos_avg_pnl_count": pos,
                    "neg_avg_pnl_count": neg,
                    "pos_fwd12_count": pos_f,
                    "neg_fwd12_count": neg_f,
                    "pnl_classification":
                        classify_cross_instrument(pos, neg, sufficient),
                    "fwd12_classification":
                        classify_cross_instrument(pos_f, neg_f, sufficient),
                })
    return pd.DataFrame(rows)


def build_correlation_summary(prepared: Dict[str, pd.DataFrame]
                              ) -> pd.DataFrame:
    rows: List[dict] = []
    for ins in ordered_instruments(prepared):
        df = prepared[ins]
        for p_label in PERIOD_LABELS:
            part = df[df["period"] == p_label]
            for feature in CORRELATION_FEATURES:
                if feature not in part.columns:
                    continue
                rows.append({
                    "instrument": ins,
                    "period": p_label,
                    "feature": feature,
                    "corr_net_pnl": pearson_corr(
                        part[feature].astype(float),
                        part["net_pnl"].astype(float)),
                    "corr_forward12": pearson_corr(
                        part[feature].astype(float),
                        part["forward12_pct"].astype(float)),
                    "corr_forward24": pearson_corr(
                        part[feature].astype(float),
                        part["forward24_pct"].astype(float)),
                })
    return pd.DataFrame(rows)


def build_walkforward_summary(prepared: Dict[str, pd.DataFrame]
                              ) -> pd.DataFrame:
    """Descriptive development->OOS sign observation. No state is
    selected, no threshold is fitted, nothing is optimized here."""
    rows: List[dict] = []
    for ins in ordered_instruments(prepared):
        df = prepared[ins]
        for feature, col, labels in STATE_FEATURES:
            for bucket in labels:
                bdf = df[df[col] == bucket]
                if len(bdf) == 0:
                    continue
                for dev_label, oos_label in WALK_STEPS:
                    dev = bdf[bdf["period"].isin(dev_label.split("+"))]
                    oos = bdf[bdf["period"] == oos_label]
                    dev_avg = (float(dev["net_pnl"].mean())
                               if len(dev) else float("nan"))
                    oos_avg = (float(oos["net_pnl"].mean())
                               if len(oos) else float("nan"))
                    dev_f12 = (float(dev["forward12_pct"].mean())
                               if len(dev) else float("nan"))
                    oos_f12 = (float(oos["forward12_pct"].mean())
                               if len(oos) else float("nan"))
                    rows.append({
                        "instrument": ins,
                        "feature": feature,
                        "bucket": bucket,
                        "step": f"{dev_label}->{oos_label}",
                        "dev_n": int(len(dev)),
                        "dev_avg_net_pnl": dev_avg,
                        "oos_n": int(len(oos)),
                        "oos_avg_net_pnl": oos_avg,
                        "dev_fwd12_avg_pct": dev_f12,
                        "oos_fwd12_avg_pct": oos_f12,
                        "pnl_sign_result": sign_result(dev_avg, oos_avg),
                        "fwd12_sign_result": sign_result(dev_f12, oos_f12),
                        "low_sample": bool(len(dev) < LOW_SAMPLE_N
                                           or len(oos) < LOW_SAMPLE_N),
                    })
    return pd.DataFrame(rows)


def build_overall_summary(prepared: Dict[str, pd.DataFrame]
                          ) -> pd.DataFrame:
    rows: List[dict] = []
    for ins in ordered_instruments(prepared):
        df = prepared[ins]
        if len(df) == 0:
            continue
        stats = cell_stats(df)
        rows.append({
            "instrument": ins,
            "total_signals": stats["n"],
            "avg_net_pnl": stats["avg_net_pnl"],
            "total_net_pnl": float(df["net_pnl"].astype(float).sum()),
            "win_rate_pct": stats["win_rate_pct"],
            "profit_factor": stats["profit_factor"],
            "fwd6_avg_pct": stats["fwd6_avg_pct"],
            "fwd12_avg_pct": stats["fwd12_avg_pct"],
            "fwd24_avg_pct": stats["fwd24_avg_pct"],
            "mfe12_avg_atr": stats["mfe12_avg_atr"],
            "mae12_avg_atr": stats["mae12_avg_atr"],
        })
    return pd.DataFrame(rows)


def build_compact_summary(period_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    present = list(period_summary["instrument"].drop_duplicates())
    ordered = ([ins for ins in EXPECTED_INSTRUMENTS if ins in present]
               + [ins for ins in present
                  if ins not in EXPECTED_INSTRUMENTS])
    for ins in ordered:
        sub = period_summary[period_summary["instrument"] == ins]
        if len(sub) == 0:
            continue
        avgs: List[float] = []
        pfs: List[float] = []
        for p_label in PERIOD_LABELS:
            part = sub[sub["period"] == p_label]
            if len(part) == 0:
                avgs.append(float("nan"))
                pfs.append(float("nan"))
            else:
                avgs.append(float(part["avg_net_pnl"].iloc[0]))
                pfs.append(float(part["profit_factor"].iloc[0]))
        pos_n = sum(1 for v in avgs if sign_of(v) == 1)
        neg_n = sum(1 for v in avgs if sign_of(v) == -1)
        rows.append({
            "instrument": ins,
            **{f"p{idx}_avg_net_pnl": avgs[idx - 1] for idx in (1, 2, 3, 4)},
            **{f"p{idx}_profit_factor": pfs[idx - 1] for idx in (1, 2, 3, 4)},
            "pos_periods": pos_n,
            "neg_periods": neg_n,
            "sign_changes": count_sign_changes(avgs),
            "positive_all_four_periods":
                bool(all(sign_of(v) == 1 for v in avgs)),
            "negative_all_four_periods":
                bool(all(sign_of(v) == -1 for v in avgs)),
            "at_least_one_sign_change": count_sign_changes(avgs) >= 1,
            "more_positive_than_negative_periods": pos_n > neg_n,
            "more_negative_than_positive_periods": neg_n > pos_n,
        })
    return pd.DataFrame(rows)


def compute_all(dbs: Dict[str, pd.DataFrame]) -> dict:
    """Run every descriptive section on in-memory databases."""
    prepared = prepare_databases(dbs)
    period = build_period_summary(prepared)
    return {
        "prepared": prepared,
        "period_summary": period,
        "state_summary": build_state_summary(prepared),
        "walkforward": build_walkforward_summary(prepared),
        "cross_instrument": build_cross_instrument_summary(prepared),
        "correlations": build_correlation_summary(prepared),
        "direction_summary": build_direction_summary(prepared),
        "overall": build_overall_summary(prepared),
        "compact": build_compact_summary(period),
    }


# ============================================================
# Formatting helpers for the terminal report
# ============================================================
def fmt_num(value, spec: str = "+.2f") -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return format(float(value), spec)


def fmt_pf(value) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    if float(value) == float("inf"):
        return "inf"
    return format(float(value), ".2f")


def print_section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def summarize_list(items: List[str], limit: int = 30) -> str:
    if not items:
        return "(none)"
    shown = items[:limit]
    extra = len(items) - len(shown)
    text = "; ".join(shown)
    if extra > 0:
        text += (f" ... (+{extra} more, see "
                 f"market_state_oos_state_summary.csv)")
    return text


def relationship_frame(state_summary: pd.DataFrame) -> pd.DataFrame:
    """One row per instrument x feature x bucket (persistence facts)."""
    cols = ["instrument", "feature", "bucket", "pnl_classification",
            "pnl_sign_changes", "pnl_pos_periods", "pnl_neg_periods",
            "p1_avg_net_pnl", "p2_avg_net_pnl", "p3_avg_net_pnl",
            "p4_avg_net_pnl",
            "fwd6_classification", "fwd12_classification",
            "fwd24_classification", "fwd12_sign_changes",
            "fwd12_p1", "fwd12_p2", "fwd12_p3", "fwd12_p4",
            "p1_n", "p2_n", "p3_n", "p4_n"]
    return state_summary[cols].drop_duplicates().reset_index(drop=True)


def direction_sign_tally(direction_summary: pd.DataFrame, direction: str
                         ) -> Tuple[int, int, int]:
    """(same sign in all four periods, sign changed, skipped)."""
    sub = direction_summary[direction_summary["direction"] == direction]
    same = changed = skipped = 0
    present = list(sub["instrument"].drop_duplicates())
    ordered = ([ins for ins in EXPECTED_INSTRUMENTS if ins in present]
               + [ins for ins in present
                  if ins not in EXPECTED_INSTRUMENTS])
    for ins in ordered:
        grp = sub[sub["instrument"] == ins].sort_values("period")
        if len(grp) == 0:
            continue
        if len(grp) < 4 or bool((grp["n"] < LOW_SAMPLE_N).any()):
            skipped += 1
            continue
        signs = [sign_of(v) for v in grp["avg_net_pnl"]]
        if all(s == signs[0] for s in signs):
            same += 1
        else:
            changed += 1
    return same, changed, skipped


# ============================================================
# Final questions Q1..Q10 (factual counts only - no recommendations)
# ============================================================
def final_questions(results: dict) -> List[Tuple[str, str]]:
    qa: List[Tuple[str, str]] = []
    rel = relationship_frame(results["state_summary"])
    period = results["period_summary"]
    cross = results["cross_instrument"]
    wf = results["walkforward"]
    compact = results["compact"]

    # Q1 - same P/L sign in all four periods
    persistent = rel[rel["pnl_classification"].isin(
        ("persistent positive", "persistent negative"))]
    qa.append(("Q1", f"{len(persistent)} of {len(rel)} state relationships "
               "keep the same P/L sign across all four periods "
               "(sufficient sample required)."))
    qa.append(("Q1 list", summarize_list(
        [f"{r.instrument}/{r.feature}/{r.bucket}: {r.pnl_classification}"
         for r in persistent.itertuples()])))

    # Q2 - same forward-12 sign in all four periods
    pers12 = rel[rel["fwd12_classification"].isin(
        ("persistent positive", "persistent negative"))]
    qa.append(("Q2", f"{len(pers12)} of {len(rel)} state relationships "
               "keep the same forward-12 sign across all four periods."))
    qa.append(("Q2 list", summarize_list(
        [f"{r.instrument}/{r.feature}/{r.bucket}: {r.fwd12_classification}"
         for r in pers12.itertuples()])))

    # Q3 - relationships that change P/L sign at least once
    sufficient = rel[rel["pnl_classification"] != "insufficient sample"]
    changed = sufficient[sufficient["pnl_sign_changes"] >= 1]
    qa.append(("Q3", f"{len(changed)} of {len(sufficient)} state "
               "relationships with sufficient sample change P/L sign at "
               f"least once ({len(rel) - len(sufficient)} low-sample "
               "relationships excluded from this count)."))

    # Q4 - cross-instrument consistency
    consistent = cross[cross["pnl_classification"] == "CONSISTENT"]
    mixed = cross[cross["pnl_classification"] == "MIXED"]
    insuff = cross[cross["pnl_classification"] == "INSUFFICIENT SAMPLE"]
    all5 = cross[(cross["pnl_classification"] == "CONSISTENT")
                 & (cross["instruments_with_sufficient_sample"] == 5)]
    qa.append(("Q4", f"P/L sign across instruments: {len(consistent)} "
               f"CONSISTENT cells, {len(mixed)} MIXED cells, "
               f"{len(insuff)} INSUFFICIENT SAMPLE cells (of {len(cross)} "
               f"feature/bucket/period cells). {len(all5)} cells are "
               "CONSISTENT across all five instruments with sufficient "
               "sample."))

    # Q5 - development -> OOS sign agreement
    wf_ok = wf[~wf["low_sample"]]
    pnl_same = int((wf_ok["pnl_sign_result"] == "SIGN SAME").sum())
    pnl_changed = int((wf_ok["pnl_sign_result"] == "SIGN CHANGED").sum())
    f12_same = int((wf_ok["fwd12_sign_result"] == "SIGN SAME").sum())
    f12_changed = int((wf_ok["fwd12_sign_result"] == "SIGN CHANGED").sum())
    qa.append(("Q5", "Development->OOS sign agreement (sufficient sample "
               f"only): P/L {pnl_same} SIGN SAME vs {pnl_changed} SIGN "
               f"CHANGED; forward-12 {f12_same} SIGN SAME vs "
               f"{f12_changed} SIGN CHANGED "
               f"({len(wf) - len(wf_ok)} low-sample steps excluded)."))

    # Q6 - same persistent sign in multiple instruments
    shared: List[str] = []
    for (feature, bucket), grp in rel.groupby(["feature", "bucket"]):
        for cls in ("persistent positive", "persistent negative"):
            instruments = sorted(
                grp[grp["pnl_classification"] == cls]["instrument"]
                .unique())
            if len(instruments) >= 2:
                shared.append(f"{feature}/{bucket} {cls}: "
                              + ", ".join(instruments))
    qa.append(("Q6", f"{len(shared)} state buckets have the same "
               "persistent P/L sign in at least two instruments."))
    qa.append(("Q6 list", summarize_list(shared)))

    # Q7 - direction consistency across periods
    for d in ("LONG", "SHORT"):
        same, changed_n, skipped = direction_sign_tally(
            results["direction_summary"], d)
        qa.append((f"Q7 {d}", f"{same} instrument(s) keep the same {d} "
                   f"P/L sign across all four periods, {changed_n} "
                   f"change, {skipped} not evaluable (low sample)."))

    # Q8 - overall frozen strategy across the four periods
    pos_all = compact[compact["positive_all_four_periods"]][
        "instrument"].tolist()
    neg_all = compact[compact["negative_all_four_periods"]][
        "instrument"].tolist()
    qa.append(("Q8", f"{len(pos_all)} instrument(s) have positive average "
               f"P/L in all four periods: {', '.join(pos_all) or 'none'}. "
               f"{len(neg_all)} instrument(s) have negative average P/L "
               f"in all four periods: {', '.join(neg_all) or 'none'}."))

    # Q9 / Q10 - counts over instrument/period combinations
    total = len(period)
    pos_count = int((period["avg_net_pnl"] > 0).sum())
    pf_count = int((period["profit_factor"] > 1).sum())
    qa.append(("Q9", f"{pos_count} of {total} instrument/period "
               "combinations have positive average P/L."))
    qa.append(("Q10", f"{pf_count} of {total} instrument/period "
               "combinations have PF > 1."))
    return qa


# ============================================================
# Terminal report (descriptive; nothing ranked or recommended)
# ============================================================
def print_report(results: dict) -> None:
    prepared = results["prepared"]
    period = results["period_summary"]
    state = results["state_summary"]
    rel = relationship_frame(state)
    direction = results["direction_summary"]
    cross = results["cross_instrument"]
    correlations = results["correlations"]
    wf = results["walkforward"]
    overall = results["overall"]
    compact = results["compact"]

    print()
    print("#" * 74)
    print("MARKET STATE OOS ANALYZER - HISTORICAL VALIDATION REPORT")
    print("(read-only, descriptive; no trading rules, no optimization,")
    print("no rankings, no recommendations)")
    print("#" * 74)

    print_section("DATA LOADED (read-only)")
    for ins in EXPECTED_INSTRUMENTS:
        if ins not in prepared or len(prepared[ins]) == 0:
            continue
        df = prepared[ins]
        print(f"  {ins:<8} {len(df):>5} signals  "
              f"{df['signal_time'].iloc[0]} -> {df['signal_time'].iloc[-1]}")
    print("  ATR regime labels:")
    for ins in EXPECTED_INSTRUMENTS:
        if ins in prepared:
            print(f"    {ins:<8} {atr_label_source(prepared[ins])}")

    print_section("SECTION 1 - FOUR CHRONOLOGICAL PERIODS PER INSTRUMENT "
                  "(equal signal counts)")
    for ins in EXPECTED_INSTRUMENTS:
        sub = period[period["instrument"] == ins]
        if len(sub) == 0:
            continue
        print(f"  {ins}")
        for r in sub.itertuples():
            print(f"    {r.period}  {r.start_time} -> {r.end_time}")
            print(f"        n={r.n}  LONG={r.long}  SHORT={r.short}")
            print(f"        avgP/L={fmt_num(r.avg_net_pnl)}  "
                  f"median={fmt_num(r.median_net_pnl)}  "
                  f"win={fmt_num(r.win_rate_pct, '.1f')}%  "
                  f"PF={fmt_pf(r.profit_factor)}")
            print(f"        fwd6={fmt_num(r.fwd6_avg_pct, '+.3f')}%  "
                  f"fwd12={fmt_num(r.fwd12_avg_pct, '+.3f')}%  "
                  f"fwd24={fmt_num(r.fwd24_avg_pct, '+.3f')}%")
            print(f"        MFE12={fmt_num(r.mfe12_avg_atr, '+.3f')} ATR  "
                  f"MAE12={fmt_num(r.mae12_avg_atr, '+.3f')} ATR")

    print_section("SECTION 2 - STATE x OUTCOME BY CHRONOLOGICAL PERIOD")
    print("  Per bucket and period: n / avg net P/L / profit factor.")
    print("  [LOW SAMPLE] = n < 5 (excluded from persistence conclusions;")
    print("  full cell statistics incl. median, win rate and forward 6/12/24")
    print("  are in market_state_oos_state_summary.csv)")
    for ins in EXPECTED_INSTRUMENTS:
        if ins not in prepared:
            continue
        for feature, col, labels in STATE_FEATURES:
            fsub = state[(state["instrument"] == ins)
                         & (state["feature"] == feature)]
            if len(fsub) == 0:
                continue
            print(f"  {ins} / {feature}")
            for bucket in labels:
                bsub = fsub[fsub["bucket"] == bucket]
                parts = []
                for r in bsub.itertuples():
                    tag = " [LOW SAMPLE]" if r.low_sample else ""
                    parts.append(f"{r.period}: n={r.n} "
                                 f"avg={fmt_num(r.avg_net_pnl)} "
                                 f"PF={fmt_pf(r.profit_factor)}{tag}")
                print("    " + f"{bucket:<10} " + " | ".join(parts))

    print_section("SECTION 3 - SIGN PERSISTENCE (P1 -> P2 -> P3 -> P4)")
    print("  Period averages; classification: persistent positive /")
    print("  persistent negative / mixed / insufficient sample.")
    print("  (forward-6 and forward-24 series are in the state summary CSV)")
    for ins in EXPECTED_INSTRUMENTS:
        rel_ins = rel[rel["instrument"] == ins]
        if len(rel_ins) == 0:
            continue
        print(f"  {ins}")
        for r in rel_ins.itertuples():
            pnl_vals = [r.p1_avg_net_pnl, r.p2_avg_net_pnl,
                        r.p3_avg_net_pnl, r.p4_avg_net_pnl]
            f12_vals = [r.fwd12_p1, r.fwd12_p2, r.fwd12_p3, r.fwd12_p4]
            print(f"    {r.feature:<12} {r.bucket:<10} "
                  f"P/L [{' '.join(fmt_num(v) for v in pnl_vals)}] "
                  f"chg={r.pnl_sign_changes} -> {r.pnl_classification}")
            print(f"    {'':<12} {'':<10} "
                  f"f12 [{' '.join(fmt_num(v, '+.3f') for v in f12_vals)}] "
                  f"chg={r.fwd12_sign_changes} "
                  f"-> {r.fwd12_classification}")

    print_section("SECTION 4 - DIRECTION ANALYSIS (descriptive only)")
    for ins in EXPECTED_INSTRUMENTS:
        dsub = direction[direction["instrument"] == ins]
        if len(dsub) == 0:
            continue
        print(f"  {ins}")
        for r in dsub.itertuples():
            print(f"    {r.period} {r.direction:<5} n={r.n:<4} "
                  f"avg={fmt_num(r.avg_net_pnl)}  "
                  f"win={fmt_num(r.win_rate_pct, '.1f')}%  "
                  f"PF={fmt_pf(r.profit_factor)}  "
                  f"fwd12={fmt_num(r.fwd12_avg_pct, '+.3f')}%")
    for d in ("LONG", "SHORT"):
        same, changed_n, skipped = direction_sign_tally(direction, d)
        print(f"  Direction sign across periods ({d}): {same} instrument(s) "
              f"same, {changed_n} changed, {skipped} not evaluable")

    print_section("SECTION 5 - CROSS-INSTRUMENT PERSISTENCE (per period)")
    print("  suf = instruments with sufficient sample (n >= 5); counts of")
    print("  positive/negative average P/L and positive/negative fwd12.")
    for r in cross.itertuples():
        print(f"    {r.feature:<12} {r.bucket:<10} {r.period}: "
              f"suf={r.instruments_with_sufficient_sample}  "
              f"P/L +{r.pos_avg_pnl_count}/-{r.neg_avg_pnl_count}  "
              f"f12 +{r.pos_fwd12_count}/-{r.neg_fwd12_count}  "
              f"-> P/L {r.pnl_classification} / "
              f"f12 {r.fwd12_classification}")

    print_section("SECTION 6 - PEARSON CORRELATIONS BY PERIOD (values only)")
    print("  No causality is claimed; no threshold is selected from these.")
    for ins in EXPECTED_INSTRUMENTS:
        csub = correlations[correlations["instrument"] == ins]
        if len(csub) == 0:
            continue
        print(f"  {ins}")
        for p_label in PERIOD_LABELS:
            psub = csub[csub["period"] == p_label]
            if len(psub) == 0:
                continue
            lookup = {r.feature: r for r in psub.itertuples()}
            for target, attr in (("net P/L", "corr_net_pnl"),
                                 ("fwd12  ", "corr_forward12"),
                                 ("fwd24  ", "corr_forward24")):
                cells = "  ".join(
                    f"{feat}={fmt_num(getattr(lookup[feat], attr), '+.3f')}"
                    for feat in CORRELATION_FEATURES if feat in lookup)
                print(f"    {p_label} vs {target}: {cells}")

    print_section("SECTION 7 - WALK-FORWARD STYLE SIGN OBSERVATION")
    print("  Descriptive only: does a development-period sign persist into")
    print("  the next OOS period? Nothing is selected, fitted or optimized.")
    print("  SAME = same sign, CHANGED = sign changed; f12 = forward-12.")
    for ins in EXPECTED_INSTRUMENTS:
        wsub = wf[wf["instrument"] == ins]
        if len(wsub) == 0:
            continue
        print(f"  {ins}")
        for feature, col, labels in STATE_FEATURES:
            fsub = wsub[wsub["feature"] == feature]
            if len(fsub) == 0:
                continue
            print(f"    {feature}")
            for bucket in labels:
                bsub = fsub[fsub["bucket"] == bucket]
                if len(bsub) == 0:
                    continue
                parts = []
                f12_flags = []
                low = False
                for r in bsub.itertuples():
                    if r.low_sample:
                        low = True
                    f12_flags.append(short_sign(r.fwd12_sign_result))
                    parts.append(
                        f"{r.step} {fmt_num(r.dev_avg_net_pnl)}->"
                        f"{fmt_num(r.oos_avg_net_pnl)} "
                        f"{short_sign(r.pnl_sign_result)}")
                tag = "  [LOW SAMPLE]" if low else ""
                print("    " + f"{bucket:<10} " + " | ".join(parts)
                      + f"   f12: {'/'.join(f12_flags)}{tag}")

    print_section("SECTION 8 - OVERALL STRATEGY OUTCOME (reference only)")
    for r in overall.itertuples():
        print(f"  {r.instrument:<8} n={r.total_signals}  "
              f"avgP/L={fmt_num(r.avg_net_pnl)}  "
              f"totalP/L={fmt_num(r.total_net_pnl)}  "
              f"win={fmt_num(r.win_rate_pct, '.1f')}%  "
              f"PF={fmt_pf(r.profit_factor)}")
        print(f"  {'':<8} fwd6={fmt_num(r.fwd6_avg_pct, '+.3f')}%  "
              f"fwd12={fmt_num(r.fwd12_avg_pct, '+.3f')}%  "
              f"fwd24={fmt_num(r.fwd24_avg_pct, '+.3f')}%  "
              f"MFE12={fmt_num(r.mfe12_avg_atr, '+.3f')}  "
              f"MAE12={fmt_num(r.mae12_avg_atr, '+.3f')}")

    print_section("SECTION 9 - CROSS-INSTRUMENT COMPACT SUMMARY")
    header = f"    {'Instrument':<10} | " + " ".join(
        f"{p:>8}" for p in PERIOD_LABELS)
    print("  Average net P/L by period:")
    print(header)
    for r in compact.itertuples():
        vals = [r.p1_avg_net_pnl, r.p2_avg_net_pnl,
                r.p3_avg_net_pnl, r.p4_avg_net_pnl]
        print("    " + f"{r.instrument:<10} | "
              + " ".join(f"{fmt_num(v):>8}" for v in vals))
    print("  Profit factor by period:")
    print(header)
    for r in compact.itertuples():
        vals = [r.p1_profit_factor, r.p2_profit_factor,
                r.p3_profit_factor, r.p4_profit_factor]
        print("    " + f"{r.instrument:<10} | "
              + " ".join(f"{fmt_pf(v):>8}" for v in vals))
    print("  Factual counts across instruments:")
    print(f"    positive in all 4 periods:            "
          f"{int(compact['positive_all_four_periods'].sum())}")
    print(f"    negative in all 4 periods:            "
          f"{int(compact['negative_all_four_periods'].sum())}")
    print(f"    at least one sign change:             "
          f"{int(compact['at_least_one_sign_change'].sum())}")
    print(f"    more positive than negative periods:  "
          f"{int(compact['more_positive_than_negative_periods'].sum())}")
    print(f"    more negative than positive periods:  "
          f"{int(compact['more_negative_than_positive_periods'].sum())}")

    print_section("SECTION 10 - FINAL QUESTIONS (factual counts only)")
    for tag, text in final_questions(results):
        print(f"  {tag}: {text}")
    print()
    print("All results are descriptive historical observations.")
    print("No state, instrument, direction or period is ranked; no trading")
    print("rule, filter, threshold or recommendation is derived from this")
    print("report. Low-sample cells (n < 5) are excluded from persistence")
    print("conclusions. Correlations do not imply causation.")


# ============================================================
# Output CSVs (new files only - never overwrite)
# ============================================================
def find_output_conflicts(directory: str) -> List[str]:
    return [name for name in OUTPUT_FILES
            if os.path.exists(os.path.join(directory, name))]


def write_output_csv(df: pd.DataFrame, path: str) -> Optional[str]:
    if os.path.exists(path):
        print(f"    SKIPPED {path}: file already exists "
              f"(this tool never overwrites).")
        return None
    df.to_csv(path, index=False)
    return path


# ============================================================
# Real CSV analysis (requires the explicit --run flag)
# ============================================================
def run_real_analysis(directory: Optional[str] = None) -> None:
    directory = directory if directory is not None else SCRIPT_DIR

    # 1) Never overwrite existing outputs: stop before anything else.
    conflicts = find_output_conflicts(directory)
    if conflicts:
        raise SystemExit(
            "ABORT: output file(s) already exist and this tool never "
            "overwrites:\n  " + "\n  ".join(conflicts)
            + "\nMove or rename them first, then re-run.")

    # 2) Load the existing database CSVs (clear error if one is missing).
    print("Loading market-state database CSVs (read-only)...")
    dbs: Dict[str, pd.DataFrame] = {}
    for ins in EXPECTED_INSTRUMENTS:
        dbs[ins] = load_instrument_csv(ins, directory)
        print(f"  {ins:<8} {len(dbs[ins])} signals")
    print()

    # 3) Compute all descriptive sections and print the report.
    results = compute_all(dbs)
    print_report(results)

    # 4) Write the four NEW output CSVs.
    print()
    print("Writing output CSVs...")
    written: List[str] = []
    for name, frame in (
        (OUTPUT_FILES[0], results["period_summary"]),
        (OUTPUT_FILES[1], results["state_summary"]),
        (OUTPUT_FILES[2], results["walkforward"]),
        (OUTPUT_FILES[3], results["cross_instrument"]),
    ):
        path = write_output_csv(frame, os.path.join(directory, name))
        if path:
            written.append(path)
    print(f"Wrote {len(written)} new file(s).")
    print()
    print("Historical research only. No orders were sent, no trading rule,")
    print("filter or signal was created; nothing was optimized or ranked.")


# ============================================================
# Command-line safety: real analysis ONLY with the explicit --run flag
# ============================================================
def should_run_real(argv: List[str]) -> bool:
    return "--run" in argv


def main() -> None:
    print("Running synthetic verification tests...")
    passed, failed = run_synthetic_tests()
    print(f"Synthetic tests: {passed} passed, {failed} failed")

    if failed:
        print("SYNTHETIC TESTS FAILED - real analysis will not start.")
        return

    if not should_run_real(sys.argv):
        print()
        print("Verification mode: synthetic tests only. The real CSV")
        print("analysis was NOT run. To execute it explicitly, use:")
        print("    python market_state_oos_analyzer.py --run")
        print()
        print("Historical research only. No orders were sent, no trading")
        print("rule was created, nothing was optimized or ranked.")
        return

    run_real_analysis()


# ============================================================
# Synthetic verification tests (in-memory frames + system temp dirs
# only; nothing is written into the project directory)
# ============================================================
def synth_frame(instrument: str, n: int, pnl=None, fwd6=None, fwd12=None,
                fwd24=None, adx=None, atr_pct=None, ema=None, bbe=None,
                eff=None, rsi=None, direction=None) -> pd.DataFrame:
    """Deterministic synthetic database frame (in memory only)."""
    def fill(values, default):
        if values is None:
            return [default(i) for i in range(n)]
        values = list(values)
        if len(values) != n:
            raise AssertionError("synthetic list length mismatch")
        return values
    frame = pd.DataFrame({
        "instrument": instrument,
        "timeframe": "H1",
        "signal_time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "signal_index": np.arange(n, dtype=int),
        "direction": fill(direction,
                          lambda i: "LONG" if i % 2 == 0 else "SHORT"),
        "adx": fill(adx, lambda i: (10.0, 16.0, 22.0, 30.0)[i % 4]),
        "atr_pct": fill(atr_pct, lambda i: 1.0 + i),
        "ema_distance": fill(ema, lambda i: (0.5, 1.5, 2.5)[i % 3]),
        "bb_excursion": fill(bbe, lambda i: (0.4, 0.6, 0.9)[i % 3]),
        "eff24": fill(eff, lambda i: (0.2, 0.45, 0.7)[i % 3]),
        "rsi14": fill(rsi, lambda i: (20.0, 40.0, 60.0, 80.0)[i % 4]),
        "net_pnl": fill(pnl, lambda i: 1.0),
        "win": True,
        "forward6_pct": fill(fwd6, lambda i: 0.1),
        "forward12_pct": fill(fwd12, lambda i: 0.2),
        "forward24_pct": fill(fwd24, lambda i: 0.3),
        "mfe12_atr": 1.0,
        "mae12_atr": -1.0,
    })
    frame["win"] = frame["net_pnl"] > 0
    return frame


def run_synthetic_tests() -> Tuple[int, int]:
    """Deterministic self-tests. Returns (passed, failed)."""
    passed = 0
    failed = 0
    area_stats: List[Tuple[str, int, int]] = []
    area = {"name": None, "passed": 0, "failed": 0}

    def start_area(name: str) -> None:
        if area["name"] is not None:
            area_stats.append((area["name"], area["passed"],
                               area["failed"]))
        area["name"] = name
        area["passed"] = 0
        area["failed"] = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            area["passed"] += 1
        else:
            failed += 1
            area["failed"] += 1
            print(f"  FAIL: {name} {detail}")

    src = inspect.getsource(sys.modules[__name__])

    # ----------------------------------------------------------
    # SPEC ITEM 14: missing-file handling
    # ----------------------------------------------------------
    start_area("missing_file")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_instrument_csv("GOLD", tmp)
            check("missing input CSV stops with a clear error", False)
        except FileNotFoundError as exc:
            check("missing input CSV raises FileNotFoundError",
                  "market_state_GOLD.csv" in str(exc) and "GOLD" in str(exc))
    try:
        validate_input_frame(
            synth_frame("T", 8).drop(columns=["net_pnl"]), "synthetic")
        check("missing required column is detected", False)
    except ValueError as exc:
        check("missing required column is detected", "net_pnl" in str(exc))
    check("complete synthetic frame passes validation",
          validate_input_frame(synth_frame("T", 8), "s") is None)
    with tempfile.TemporaryDirectory() as tmp:
        synth_frame("TST", 6).to_csv(
            os.path.join(tmp, "market_state_TST.csv"), index=False)
        loaded = load_instrument_csv("TST", tmp)
        check("loader round-trip parses CSV + timestamps",
              len(loaded) == 6
              and str(loaded["signal_time"].dtype).startswith("datetime"))

    # ----------------------------------------------------------
    # SPEC ITEMS 1, 2, 4: chronological sorting, equal counts, boundaries
    # ----------------------------------------------------------
    start_area("chronological_split")
    n = 8
    frame8 = synth_frame("T", n, pnl=[float(i + 1) for i in range(n)])
    assigned = assign_periods(frame8)
    sizes = [int((assigned["period"] == p).sum()) for p in PERIOD_LABELS]
    check("four equal-count periods (sum preserved)",
          sizes == [2, 2, 2, 2], f"got {sizes}")
    boundary_ok = True
    for p_label, expected in (("P1", 1.5), ("P2", 3.5), ("P3", 5.5),
                              ("P4", 7.5)):
        part = assigned[assigned["period"] == p_label]
        if abs(float(part["net_pnl"].mean()) - expected) > 1e-12:
            boundary_ok = False
    check("P1/P2/P3/P4 boundaries land on the crafted rows", boundary_ok)
    contiguous = all(
        assigned[assigned["period"] == PERIOD_LABELS[p]]["signal_time"]
        .iloc[-1]
        <= assigned[assigned["period"] == PERIOD_LABELS[p + 1]]
        ["signal_time"].iloc[0] for p in range(3))
    check("periods are contiguous and time-ordered (no overlap)",
          contiguous)
    sizes10 = [int((assign_periods(synth_frame("T", 10))["period"] == p)
                   .sum()) for p in PERIOD_LABELS]
    check("10 signals -> near-equal counts (2,3,2,3), sum preserved",
          sizes10 == [2, 3, 2, 3] and sum(sizes10) == 10, f"got {sizes10}")

    # ----------------------------------------------------------
    # SPEC ITEM 3: no shuffling / determinism
    # ----------------------------------------------------------
    start_area("no_shuffle")
    rev = frame8.iloc[::-1].reset_index(drop=True)
    a1 = assign_periods(frame8).sort_values("signal_index")
    a2 = assign_periods(rev).sort_values("signal_index")
    check("reversed input yields identical period assignment",
          list(a1["period"]) == list(a2["period"]))
    r1 = compute_all({"T": frame8})
    r2 = compute_all({"T": frame8})
    check("analysis is deterministic (identical outputs on re-run)",
          r1["period_summary"].equals(r2["period_summary"])
          and r1["state_summary"].equals(r2["state_summary"]))
    r3 = compute_all({"T": rev})
    check("shuffled input order cannot change the period summary",
          r1["period_summary"].equals(r3["period_summary"]))
    check("no shuffling calls anywhere in this file",
          (".shu" + "ffle(") not in src and ("np.rand" + "om") not in src
          and ("sample(" + "frac") not in src
          and ("perm" + "utation(") not in src)

    # ----------------------------------------------------------
    # SPEC ITEM 5: predefined frozen state labels
    # ----------------------------------------------------------
    start_area("state_labels")
    check("ADX bins: <15 / 15-20 / 20-25 / >=25",
          (bin_adx(10.0), bin_adx(15.0), bin_adx(16.0), bin_adx(20.0),
           bin_adx(22.0), bin_adx(25.0), bin_adx(30.0))
          == ("ADX<15", "ADX15-20", "ADX15-20", "ADX20-25", "ADX20-25",
              "ADX>=25", "ADX>=25"))
    check("EMA distance bins: <1 / 1-2 / >=2",
          (bin_ema_distance(0.5), bin_ema_distance(1.0),
           bin_ema_distance(1.5), bin_ema_distance(2.0),
           bin_ema_distance(2.5)) == ("<1", "1-2", "1-2", ">=2", ">=2"))
    check("BB excursion bins: <0.50 / 0.50-0.75 / >=0.75",
          (bin_bb_excursion(0.4), bin_bb_excursion(0.5),
           bin_bb_excursion(0.6), bin_bb_excursion(0.75),
           bin_bb_excursion(0.9))
          == ("<0.50", "0.50-0.75", "0.50-0.75", ">=0.75", ">=0.75"))
    check("efficiency bins: <0.30 / 0.30-0.60 / >=0.60",
          (bin_efficiency(0.2), bin_efficiency(0.3), bin_efficiency(0.45),
           bin_efficiency(0.6), bin_efficiency(0.7))
          == ("<0.30", "0.30-0.60", "0.30-0.60", ">=0.60", ">=0.60"))
    check("RSI bins: <30 / 30-50 / 50-70 / >=70",
          (bin_rsi(20.0), bin_rsi(30.0), bin_rsi(40.0), bin_rsi(50.0),
           bin_rsi(60.0), bin_rsi(70.0), bin_rsi(80.0))
          == ("<30", "30-50", "30-50", "50-70", "50-70", ">=70", ">=70"))
    atr_series = atr_bin_labels(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    check("ATR% LOW/MEDIUM/HIGH via the database's existing p33/p66",
          list(atr_series) == ["LOW", "LOW", "MEDIUM", "MEDIUM",
                               "HIGH", "HIGH"])
    check("missing values map to 'n/a' (never forced into a bin)",
          bin_adx(float("nan")) == "n/a" and bin_rsi(float("nan")) == "n/a")
    binned = attach_state_bins(synth_frame("T", 12))
    check("attached bins match the frozen functions row by row",
          all(bin_adx(r.adx) == r.adx_bin and bin_rsi(r.rsi14) == r.rsi_bin
              for r in binned.itertuples()))
    check("ATR bin column only uses LOW/MEDIUM/HIGH",
          set(binned["atr_bin"]).issubset({"LOW", "MEDIUM", "HIGH", "n/a"}))

    # ----------------------------------------------------------
    # SPEC ITEM 6: low-sample detection (n < 5)
    # ----------------------------------------------------------
    start_area("low_sample")
    tiny = synth_frame("TINY", 12, pnl=[1.0] * 12, adx=[10.0] * 12)
    state_tiny = build_state_summary(prepare_databases({"TINY": tiny}))
    tcell = state_tiny[(state_tiny["feature"] == "adx")
                       & (state_tiny["bucket"] == "ADX<15")]
    check("every period of a 12-row frame is LOW SAMPLE (n=3 < 5)",
          bool(tcell["low_sample"].all()))
    check("low-sample cells are classified 'insufficient sample'",
          set(tcell["pnl_classification"]) == {"insufficient sample"})
    edge = synth_frame("EDGE", 20, pnl=[1.0] * 20, adx=[10.0] * 20)
    state_edge = build_state_summary(prepare_databases({"EDGE": edge}))
    ecell = state_edge[(state_edge["feature"] == "adx")
                       & (state_edge["bucket"] == "ADX<15")]
    check("n = 5 per period is NOT low sample (boundary n < 5)",
          not bool(ecell["low_sample"].any()))
    check("sufficient positive cells are 'persistent positive'",
          set(ecell["pnl_classification"]) == {"persistent positive"})
    mixed_n = synth_frame("MIXN", 20, pnl=[1.0] * 16 + [-1.0] * 4,
                          adx=[10.0] * 16 + [30.0] * 4)
    state_mixed_n = build_state_summary(
        prepare_databases({"MIXN": mixed_n}))
    mcell = state_mixed_n[(state_mixed_n["feature"] == "adx")
                          & (state_mixed_n["bucket"] == "ADX>=25")]
    check("buckets with n < 5 in any period are flagged insufficient",
          bool(mcell["low_sample"].all())
          and set(mcell["pnl_classification"]) == {"insufficient sample"})

    # ----------------------------------------------------------
    # SPEC ITEMS 7, 8: sign persistence + sign-change detection
    # ----------------------------------------------------------
    start_area("sign_persistence")
    pos = synth_frame("POS", 20, pnl=[1.0] * 5 + [2.0] * 5 + [3.0] * 5
                      + [4.0] * 5, adx=[10.0] * 20)
    neg = synth_frame("NEG", 20, pnl=[-1.0] * 20, adx=[10.0] * 20)
    mix = synth_frame("MIX", 20, pnl=[1.0] * 5 + [-2.0] * 5 + [3.0] * 5
                      + [-4.0] * 5, adx=[10.0] * 20)
    state = build_state_summary(
        prepare_databases({"POS": pos, "NEG": neg, "MIX": mix}))

    def rel_of(frame, ins):
        sub = frame[(frame["instrument"] == ins)
                    & (frame["feature"] == "adx")
                    & (frame["bucket"] == "ADX<15")]
        return sub.drop_duplicates(
            subset=["instrument", "feature", "bucket"]).iloc[0]

    r_pos = rel_of(state, "POS")
    check("all-positive periods -> persistent positive (0 changes)",
          r_pos["pnl_classification"] == "persistent positive"
          and r_pos["pnl_pos_periods"] == 4
          and r_pos["pnl_neg_periods"] == 0
          and r_pos["pnl_sign_changes"] == 0)
    r_neg = rel_of(state, "NEG")
    check("all-negative periods -> persistent negative (0 changes)",
          r_neg["pnl_classification"] == "persistent negative"
          and r_neg["pnl_neg_periods"] == 4
          and r_neg["pnl_sign_changes"] == 0)
    r_mix = rel_of(state, "MIX")
    check("alternating signs -> mixed with 3 sign changes",
          r_mix["pnl_classification"] == "mixed"
          and r_mix["pnl_sign_changes"] == 3
          and r_mix["pnl_pos_periods"] == 2
          and r_mix["pnl_neg_periods"] == 2)
    f12mix = synth_frame("F12", 20, pnl=[1.0] * 20, adx=[10.0] * 20,
                         fwd12=[-0.1] * 5 + [0.2] * 5 + [0.3] * 5
                         + [0.4] * 5)
    r_f12 = rel_of(build_state_summary(
        prepare_databases({"F12": f12mix})), "F12")
    check("P/L and forward-12 classifications are independent",
          r_f12["pnl_classification"] == "persistent positive"
          and r_f12["fwd12_classification"] == "mixed"
          and r_f12["fwd12_sign_changes"] == 1)
    check("low-sample relationships are never called persistent",
          rel_of(state_tiny, "TINY")["pnl_classification"]
          == "insufficient sample")

    # ----------------------------------------------------------
    # SPEC ITEMS 9, 10, 11: walk-forward P1->P2 / P1+P2->P3 / +P3->P4
    # ----------------------------------------------------------
    start_area("walkforward")
    negpos = synth_frame("NEGPOS", 20, pnl=[-5.0] * 5 + [-6.0] * 5
                         + [7.0] * 5 + [8.0] * 5, adx=[30.0] * 20)
    wf = build_walkforward_summary(
        prepare_databases({"POS": pos, "NEGPOS": negpos}))

    def wf_row(ins, bucket, step):
        sub = wf[(wf["instrument"] == ins) & (wf["bucket"] == bucket)
                 & (wf["step"] == step)]
        return sub.iloc[0]

    w1 = wf_row("POS", "ADX<15", "P1->P2")
    check("step 1 uses dev=P1, oos=P2 with correct averages",
          w1["dev_n"] == 5 and w1["oos_n"] == 5
          and abs(w1["dev_avg_net_pnl"] - 1.0) < 1e-12
          and abs(w1["oos_avg_net_pnl"] - 2.0) < 1e-12
          and w1["pnl_sign_result"] == "SIGN SAME")
    w2 = wf_row("POS", "ADX<15", "P1+P2->P3")
    check("step 2 pools P1+P2 (mean of rows, not of means)",
          w2["dev_n"] == 10
          and abs(w2["dev_avg_net_pnl"] - 1.5) < 1e-12
          and abs(w2["oos_avg_net_pnl"] - 3.0) < 1e-12
          and w2["pnl_sign_result"] == "SIGN SAME")
    w3 = wf_row("POS", "ADX<15", "P1+P2+P3->P4")
    check("step 3 pools P1+P2+P3",
          w3["dev_n"] == 15
          and abs(w3["dev_avg_net_pnl"] - 2.0) < 1e-12
          and abs(w3["oos_avg_net_pnl"] - 4.0) < 1e-12
          and w3["pnl_sign_result"] == "SIGN SAME")
    n1 = wf_row("NEGPOS", "ADX>=25", "P1->P2")
    check("two negative signs count as SIGN SAME",
          n1["pnl_sign_result"] == "SIGN SAME"
          and abs(n1["dev_avg_net_pnl"] - (-5.0)) < 1e-12
          and abs(n1["oos_avg_net_pnl"] - (-6.0)) < 1e-12)
    n2 = wf_row("NEGPOS", "ADX>=25", "P1+P2->P3")
    n3 = wf_row("NEGPOS", "ADX>=25", "P1+P2+P3->P4")
    check("sign change is detected when the OOS sign flips",
          abs(n2["dev_avg_net_pnl"] - (-5.5)) < 1e-12
          and abs(n2["oos_avg_net_pnl"] - 7.0) < 1e-12
          and n2["pnl_sign_result"] == "SIGN CHANGED"
          and abs(n3["dev_avg_net_pnl"] - (-20.0 / 15.0)) < 1e-12
          and abs(n3["oos_avg_net_pnl"] - 8.0) < 1e-12
          and n3["pnl_sign_result"] == "SIGN CHANGED")
    wf_tiny = build_walkforward_summary(prepare_databases({"TINY": tiny}))
    wt = wf_tiny[(wf_tiny["instrument"] == "TINY")
                 & (wf_tiny["step"] == "P1->P2")]
    check("low-sample walk-forward steps are flagged",
          not bool(wf_row("POS", "ADX<15", "P1->P2")["low_sample"])
          and bool(wt["low_sample"].iloc[0]))

    # ----------------------------------------------------------
    # SPEC ITEM 12: cross-instrument aggregation
    # ----------------------------------------------------------
    start_area("cross_instrument")
    a_pos = synth_frame("A", 20, pnl=[1.0] * 20, adx=[10.0] * 20)
    b_neg = synth_frame("B", 20, pnl=[-1.0] * 20, adx=[10.0] * 20)
    c_pos = synth_frame("C", 20, pnl=[2.0] * 20, adx=[10.0] * 20)

    def xcell(dbs):
        frame = build_cross_instrument_summary(prepare_databases(dbs))
        return frame[(frame["feature"] == "adx")
                     & (frame["bucket"] == "ADX<15")
                     & (frame["period"] == "P1")].iloc[0]

    x = xcell({"A": a_pos, "C": c_pos})
    check("two positive instruments -> CONSISTENT with +2/-0",
          x["pnl_classification"] == "CONSISTENT"
          and x["pos_avg_pnl_count"] == 2
          and x["neg_avg_pnl_count"] == 0
          and x["instruments_with_sufficient_sample"] == 2)
    x = xcell({"A": a_pos, "B": b_neg})
    check("opposite signs -> MIXED with +1/-1",
          x["pnl_classification"] == "MIXED"
          and x["pos_avg_pnl_count"] == 1
          and x["neg_avg_pnl_count"] == 1)
    x = xcell({"A": a_pos})
    check("a single sufficient instrument is INSUFFICIENT SAMPLE",
          x["pnl_classification"] == "INSUFFICIENT SAMPLE"
          and x["instruments_with_sufficient_sample"] == 1)
    x = xcell({"A": a_pos, "TINY": tiny})
    check("low-sample instruments are excluded from the counts",
          x["instruments_with_sufficient_sample"] == 1
          and x["pnl_classification"] == "INSUFFICIENT SAMPLE")
    x = xcell({"A": a_pos, "B": b_neg, "C": c_pos})
    check("three instruments with mixed signs -> MIXED with +2/-1",
          x["pnl_classification"] == "MIXED"
          and x["pos_avg_pnl_count"] == 2
          and x["neg_avg_pnl_count"] == 1)

    # ----------------------------------------------------------
    # SPEC ITEM 13: correlation calculations
    # ----------------------------------------------------------
    start_area("correlations")
    check("pearson r = +1 for a perfect positive line",
          abs(pearson_corr(pd.Series([1., 2., 3., 4., 5.]),
                           pd.Series([2., 4., 6., 8., 10.])) - 1.0) < 1e-12)
    check("pearson r = -1 for a perfect negative line",
          abs(pearson_corr(pd.Series([1., 2., 3., 4., 5.]),
                           pd.Series([10., 8., 6., 4., 2.])) + 1.0) < 1e-12)
    check("pearson r matches a hand-computed value (0.8)",
          abs(pearson_corr(pd.Series([1., 2., 3., 4., 5.]),
                           pd.Series([1., 3., 2., 5., 4.])) - 0.8) < 1e-12)
    nan_val = pearson_corr(pd.Series([1., 2., 3.]), pd.Series([5., 5., 5.]))
    check("zero-variance series -> nan (not an error)", nan_val != nan_val)
    corr_frame = build_correlation_summary(prepare_databases(
        {"T": synth_frame("T", 8,
                          adx=[float(i + 1) for i in range(8)],
                          pnl=[float(2 * (i + 1)) for i in range(8)])}))
    p1 = corr_frame[(corr_frame["period"] == "P1")
                    & (corr_frame["feature"] == "adx")].iloc[0]
    check("period correlations use that period's rows only",
          abs(p1["corr_net_pnl"] - 1.0) < 1e-12)

    # ----------------------------------------------------------
    # SPEC ITEM 15: refusal to overwrite existing output files
    # ----------------------------------------------------------
    start_area("overwrite_protection")
    with tempfile.TemporaryDirectory() as tmp:
        victim = os.path.join(tmp, OUTPUT_FILES[0])
        with open(victim, "w") as fh:
            fh.write("DO NOT TOUCH")
        check("existing output files are detected as conflicts",
              find_output_conflicts(tmp) == [OUTPUT_FILES[0]])
        result = write_output_csv(pd.DataFrame({"a": [1, 2]}), victim)
        with open(victim) as fh:
            content = fh.read()
        check("write_output_csv refuses to overwrite",
              result is None and content == "DO NOT TOUCH")
    with tempfile.TemporaryDirectory() as tmp:
        for name in OUTPUT_FILES:
            with open(os.path.join(tmp, name), "w") as fh:
                fh.write("x")
        raised = False
        message = ""
        try:
            run_real_analysis(tmp)
        except SystemExit as exc:
            raised = True
            message = str(exc)
        check("real analysis aborts BEFORE reading inputs when outputs "
              "already exist",
              raised and "already exist" in message and "never" in message)

    # ----------------------------------------------------------
    # SPEC ITEMS 16, 17, 18: no MT5 calls, no ML imports, no
    # optimization loops (source scan of this very file)
    # ----------------------------------------------------------
    start_area("safety_scan")
    check("no trading-terminal import anywhere",
          ("import mt" + "5") not in src
          and ("MetaTrader" + "5") not in src)
    check("no order/trade call tokens anywhere",
          all(tok not in src for tok in _FORBIDDEN_CALL_TOKENS))
    check("no market-data/session call tokens",
          all(tok not in src for tok in (
              "copy_" + "rates", "symbol_" + "info", "symbol_" + "select",
              "initi" + "alize", "shut" + "down", "last_e" + "rror")))
    check("no ML libraries in any import line",
          all(not any(lib in ln for lib in _ML_LIBRARY_TOKENS)
              for ln in src.splitlines()
              if ln.strip().startswith(("import ", "from "))))
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0]
                            for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    check("imports are exactly stdlib + numpy + pandas",
          imported <= {"ast", "inspect", "os", "sys", "tempfile",
                       "typing", "numpy", "pandas"},
          f"got {sorted(imported)}")
    fn_names = [node.name.lower() for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef,
                                     ast.AsyncFunctionDef))]
    check("no optimization / search / ranking function names",
          not any(tok in name for name in fn_names
                  for tok in ("optim", "sweep", "grid", "tune",
                              "search", "rank")))
    check("no optimizer or model-fit idioms anywhere",
          all(tok not in src for tok in _OPTIMIZATION_TOKENS))

    # ----------------------------------------------------------
    # Default run behaviour: synthetic tests only unless --run
    # (main is called with patched stubs so nothing heavy runs)
    # ----------------------------------------------------------
    start_area("default_run")
    check("default (no flags) does not request real analysis",
          should_run_real([]) is False)
    check("--run flag requests real analysis",
          should_run_real(["--run"]) is True)
    check("unrelated flags do not request real analysis",
          should_run_real(["-x", "--verbose"]) is False)
    called = {"ran": False}

    def spy(directory=None):
        called["ran"] = True

    def spy_tests():
        return 0, 0

    original_real = globals()["run_real_analysis"]
    original_tests = globals()["run_synthetic_tests"]
    argv_backup = list(sys.argv)
    try:
        globals()["run_real_analysis"] = spy
        globals()["run_synthetic_tests"] = spy_tests
        sys.argv = ["market_state_oos_analyzer.py"]
        main()
        check("default main() never runs the real analysis",
              called["ran"] is False)
        called["ran"] = False
        sys.argv = ["market_state_oos_analyzer.py", "--run"]
        main()
        check("--run main() reaches the real-analysis entry point",
              called["ran"] is True)
    finally:
        globals()["run_real_analysis"] = original_real
        globals()["run_synthetic_tests"] = original_tests
        sys.argv = argv_backup

    # close the last area and print the per-area summary
    start_area(None)
    print()
    print("  Per-area verification (spec items -> area: passed/failed):")
    spec_map = (
        ("missing_file", "item 14"),
        ("chronological_split", "items 1, 2, 4"),
        ("no_shuffle", "item 3"),
        ("state_labels", "item 5"),
        ("low_sample", "item 6"),
        ("sign_persistence", "items 7, 8"),
        ("walkforward", "items 9, 10, 11"),
        ("cross_instrument", "item 12"),
        ("correlations", "item 13"),
        ("overwrite_protection", "item 15"),
        ("safety_scan", "items 16, 17, 18"),
        ("default_run", "default-run + --run gating"),
    )
    stats = {name: (p, f) for name, p, f in area_stats}
    for name, spec in spec_map:
        p, f = stats.get(name, (0, 0))
        status = "OK" if f == 0 else "FAILED"
        print(f"    {name:<22} ({spec:<28}) {p:>2} passed, {f} failed "
              f"[{status}]")
    return passed, failed


if __name__ == "__main__":
    main()

