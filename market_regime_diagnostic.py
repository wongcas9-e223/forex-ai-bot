"""
market_regime_diagnostic.py - RESEARCH/DIAGNOSTIC ONLY.

Investigates whether the performance instability of the FROZEN
BB(20, 2.0) + ADX14 < 25 mean-reversion strategy (SL 2.0 x ATR /
TP 1.5 x ATR, next-open entry, stop-first, 0.10 lots, $2 cost) is
ASSOCIATED with different market regimes.

THE QUESTION BEING ASKED:
  "In what market conditions does the frozen strategy behave
   differently?" - answered descriptively, with association wording
  only ("was associated with", "showed a higher/lower historical
  average"). NO causal claims, NO optimization, NO strategy changes,
  NO rankings, NO best/worst/good/bad/robust judgements.

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical, read-only diagnostic.
- This script NEVER places, modifies, or closes any order. Allowed
  MT5 calls (read-only): initialize, shutdown, symbols_get,
  symbol_info, symbol_info_tick, symbol_select, copy_rates_from_pos.
- mt5.shutdown() always executes (try/finally).

INSTRUMENTS AND WINDOW (same as multi_instrument_10000_history_test.py):
  GOLD, EURUSD, GBPUSD, USDJPY, AUDUSD - up to 10,000 H1 CLOSED candles
  each (forming candle dropped; if fewer are available, all available
  closed candles are used and the actual count is reported). No
  shuffling, no sampling.

FROZEN STRATEGY (identical to the previous diagnostics, unchanged):
  Indicators: BB period 20, BB std dev 2.0, ADX period 14, ATR period 14
  Entry     : LONG close < lower band AND ADX < 25
              SHORT close > upper band AND ADX < 25
              (signal on candle i, entry at candle i+1 OPEN)
  Exits     : SL = 2.0 x ATR, TP = 1.5 x ATR
              (SL assumed first if both occur inside one candle)
  Position  : 0.10 lots, one position at a time, no overlapping
              positions, no compounding, $10,000 independent starting
              balance per analysis run
  Cost      : $2.00 per completed trade
  P/L       : from each symbol's actual MT5 tick specifications
              (GOLD-style P/L is never assumed for forex pairs)

DESCRIPTIVE REGIME MEASUREMENTS (fixed classifications, never filters):
  1. ADX regime       : ADX<15 / 15-20 / 20-25 / >=25 (ADX14)
  2. Volatility regime: ATR% = ATR14/close*100; instrument-specific
                        LOW/MEDIUM/HIGH edges = 33rd/66th percentiles
                        computed ONCE from the FULL closed-candle
                        history (never per period, never optimized)
  3. Trend distance   : abs(close-EMA50)/ATR14 -> <1 / 1-2 / >2 ATR
  4. BB excursion     : abs(close-BB_middle)/(BB_upper-BB_lower)
                        -> LOW <0.50 / MEDIUM 0.50-0.75 / HIGH >0.75
  5. Efficiency       : abs(close[t]-close[t-24]) /
                        sum(abs(close[i]-close[i-1]), 24 changes)
                        -> LOW <0.30 / MEDIUM 0.30-0.60 / HIGH >0.60
  6. Recent movement  : 6-, 12- and 24-candle returns (diagnostic
                        only; never used in entry/exit decisions)

ANALYSIS SECTIONS:
  1 data, 2 overall frozen strategy, 3 ADX regime, 4 volatility regime,
  5 trend distance, 6 efficiency, 7 direction x regime, 8 regime
  interactions, 9 period x regime (four chronological quartiles),
  10 cross-instrument factual tables, 11 factual questions. Cells with
  fewer than 5 trades are tagged LOW SAMPLE and are never interpreted.

DISCLAIMER: Historical backtest with virtual money. Descriptive
associations in one historical window do not predict future
performance and are not trading advice. Educational use only.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read market data + symbol info (READ-ONLY usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # numeric helpers

# ============================================================
# SECTION 2: Settings - frozen, NOT optimized, IDENTICAL everywhere
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 10_000   # Requested history per instrument (up to 10,000 H1)
MIN_CANDLES = 400      # Absolute floor to attempt the analysis; anything
                       # below 10,000 is still USED and REPORTED.

INSTRUMENTS = ["GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]  # exactly these

STARTING_BALANCE = 10_000.0     # Independent per analysis run/period
LOTS_REQUESTED = 0.10
CONTRACT_SIZE_FALLBACK = 100_000
COST_PER_TRADE = 2.0            # $2.00 per completed trade (fixed)

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.5

BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0
ATR_PERIOD = 14
EMA_TREND_PERIOD = 50           # Diagnostic reference ONLY (never a filter)
WARMUP = 60

HOLD_WINDOW = 12                # MFE/MAE and forward-return window
EFF_WINDOW = 24                 # Efficiency-ratio and long-return window

NUM_PERIODS = 4                 # Chronological quartiles (Section 9)
LOW_SAMPLE_N = 5                # Cells below this get the LOW SAMPLE tag
LOW_SAMPLE_TAG = "LOW SAMPLE"

# Fixed descriptive buckets (NEVER optimized, NEVER used as filters)
ADX_BUCKETS = ["ADX<15", "ADX15-20", "ADX20-25", "ADX>=25"]
TREND_BUCKETS = ["<1 ATR", "1-2 ATR", ">2 ATR"]
BB_BUCKETS = ["LOW", "MEDIUM", "HIGH"]          # BB excursion
EFF_BUCKETS = ["LOW", "MEDIUM", "HIGH"]         # efficiency ratio
VOL_BUCKETS = ["LOW", "MEDIUM", "HIGH"]         # ATR% terciles


# ============================================================
# SECTION 3: Indicator functions (causal only - same math as the
# previous experiment scripts in this project)
# ============================================================
def add_atr(df, period=14):
    """Average True Range - a simple volatility measure."""
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


def add_bollinger(df, period=20, num_std=2.0):
    """Bollinger Bands (middle, upper, lower)."""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


def add_adx(df, period=14):
    """Average Directional Index (Wilder's classic calculation)."""
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


def add_ema(df, period):
    """Exponential Moving Average of the close price."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


# ============================================================
# SECTION 4: Descriptive regime measurements (causal, fixed buckets,
# never used to change any trading decision)
# ============================================================
def add_diagnostic_measurements(df):
    """Add causal regime measurements to a closed-candle frame.

    Adds: ATR_pct, EMA50_dist, BB_exc, eff24, ret6, ret12, ret24.
    Every value uses only current and previous candles (no lookahead).
    """
    df = df.copy()

    # 2. Volatility: ATR14 / close * 100
    df["ATR_pct"] = np.where(df["close"] > 0, df["ATR"] / df["close"] * 100, np.nan)

    # 3. Trend distance: abs(close - EMA50) / ATR14
    df["EMA50_dist"] = np.where(
        df["ATR"] > 0,
        (df["close"] - df["EMA50"]).abs() / df["ATR"],
        np.nan,
    )

    # 4. Bollinger excursion: abs(close - BB_middle) / (BB_upper - BB_lower)
    band_width = df["BB_UPPER"] - df["BB_LOWER"]
    df["BB_exc"] = np.where(band_width > 0, (df["close"] - df["BB_MIDDLE"]).abs() / band_width,
                            np.nan)

    # 5. Efficiency ratio (24 changes):
    #    abs(close[t] - close[t-24]) / sum(abs(close[i] - close[i-1]))
    net_move = (df["close"] - df["close"].shift(EFF_WINDOW)).abs()
    gross_path = df["close"].diff().abs().rolling(EFF_WINDOW).sum()
    df["eff24"] = np.where(gross_path > 0, net_move / gross_path, np.nan)

    # 6. Recent returns (percent, causal)
    df["ret6"] = df["close"].pct_change(6) * 100
    df["ret12"] = df["close"].pct_change(12) * 100
    df["ret24"] = df["close"].pct_change(24) * 100

    return df


def compute_vol_edges_from_candles(df):
    """33rd/66th percentile ATR% edges from the FULL closed-candle history.

    Computed ONCE per instrument from the entire dataset - never per
    period, never optimized. Returns (v1, v2).
    """
    s = df["ATR_pct"].dropna()
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


# --- Fixed descriptive classification functions (pure) -------------
def adx_regime_label(adx):
    """ADX<15 / ADX15-20 / ADX20-25 / ADX>=25 (buckets [15,20), [20,25))."""
    if adx != adx:
        return "n/a"
    if adx < 15:
        return "ADX<15"
    if adx < 20:
        return "ADX15-20"
    if adx < 25:
        return "ADX20-25"
    return "ADX>=25"


def vol_regime_label(atr_pct, v1, v2):
    """LOW / MEDIUM / HIGH using the fixed instrument-specific edges."""
    if atr_pct != atr_pct:
        return "n/a"
    if atr_pct < v1:
        return "LOW"
    if atr_pct < v2:
        return "MEDIUM"
    return "HIGH"


def trend_dist_label(dist):
    """<1 ATR / 1-2 ATR (inclusive) / >2 ATR."""
    if dist != dist:
        return "n/a"
    if dist < 1.0:
        return "<1 ATR"
    if dist <= 2.0:
        return "1-2 ATR"
    return ">2 ATR"


def bb_exc_label(exc):
    """LOW <0.50 / MEDIUM 0.50-0.75 (inclusive) / HIGH >0.75."""
    if exc != exc:
        return "n/a"
    if exc < 0.50:
        return "LOW"
    if exc <= 0.75:
        return "MEDIUM"
    return "HIGH"


def eff_label(ratio):
    """LOW <0.30 / MEDIUM 0.30-0.60 (inclusive) / HIGH >0.60."""
    if ratio != ratio:
        return "n/a"
    if ratio < 0.30:
        return "LOW"
    if ratio <= 0.60:
        return "MEDIUM"
    return "HIGH"


def assign_regimes(trades, v1, v2):
    """Attach descriptive regime labels to a trade log (pure function).

    The labels are classifications of the signal-time measurements only;
    they never change any trade outcome.
    """
    trades = trades.copy()
    trades["vol_regime"] = trades["ATR_pct"].apply(lambda x: vol_regime_label(x, v1, v2))
    trades["adx_regime"] = trades["ADX"].apply(adx_regime_label)
    trades["trend_regime"] = trades["EMA50_dist"].apply(trend_dist_label)
    trades["bb_regime"] = trades["BB_exc"].apply(bb_exc_label)
    trades["eff_regime"] = trades["eff24"].apply(eff_label)
    return trades


# ============================================================
# SECTION 5: Symbol discovery and broker specs (read-only, same as
# the previous scripts - never substitutes a different instrument)
# ============================================================
def discover_symbol(requested):
    """Find the actual MT5 symbol for a requested instrument.

    1. Try the exact name first.
    2. Otherwise match symbols that START WITH the requested name and
       are followed only by a short separator suffix (e.g. GOLDm,
       EURUSD.).
    Returns (actual_symbol_name, note) or (None, reason).
    """
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

    chosen = matches[0]
    note = f"suffix match (available: {', '.join(matches[:5])})"
    return chosen, note


def get_contract_spec(symbol_name):
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
        # Documented fallback: quote-currency value of one tick per lot
        # = contract_size * point.
        tick_value = contract_size * point
        print(f"    WARNING: trade_tick_value missing/zero for {symbol_name}; "
              f"using fallback tick_value = contract_size x point = {tick_value:g}")

    # $ per 1.0 price-unit move per 1.0 lot (tick_value is $ per tick per
    # lot, tick_size is the price change per tick).
    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0

    quote_currency = getattr(info, "currency_profit", "") or "USD"

    # If the fallback was used AND the quote currency is not USD, try to
    # convert value_per_unit into USD using a <QUOTE>USD symbol bid
    # (documented fallback, printed - never hidden).
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


def validate_volume(spec):
    """Return the largest valid volume <= LOTS_REQUESTED for this symbol, or None."""
    vmin = spec["volume_min"]
    vmax = spec["volume_max"]
    vstep = spec["volume_step"]

    if LOTS_REQUESTED < vmin:
        return None
    if vmin > vmax:
        return None

    steps_below = int((LOTS_REQUESTED - vmin) / vstep + 1e-9)  # tolerate float noise
    volume = round(vmin + steps_below * vstep, 8)
    if volume > vmax:
        return None
    return volume


# ============================================================
# SECTION 6: The frozen backtest engine with a regime-rich trade log
# - identical trading rules to the previous diagnostics; the regime
# measurements are RECORDED at signal time but never alter decisions
# ============================================================
def run_backtest(df, value_per_unit, volume, start_bar=WARMUP):
    """Frozen-entry / frozen-exit virtual backtest with regime attributes.

    Signal on closed candle i, entry at the OPEN of candle i+1, SL/TP
    fixed from the signal candle's ATR, stop-first inside-candle
    handling, one position at a time, no overlapping positions, $2 cost
    per completed trade, independent $10,000 starting balance per call.
    Returns (trades DataFrame, final balance).
    """

    long_condition = df["close"] < df["BB_LOWER"]
    short_condition = df["close"] > df["BB_UPPER"]
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df = df.copy()
    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

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

            # Forward scan for the exit; STOP checked FIRST (conservative):
            # if both SL and TP occur inside one candle, SL is assumed.
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

            # --- Signal-time regime measurements (RECORDED only) -----
            close_sig = df["close"].iloc[i]
            atr_pct = df["ATR_pct"].iloc[i] if "ATR_pct" in df else np.nan
            ema50_dist = df["EMA50_dist"].iloc[i] if "EMA50_dist" in df else np.nan
            bb_exc = df["BB_exc"].iloc[i] if "BB_exc" in df else np.nan
            eff24 = df["eff24"].iloc[i] if "eff24" in df else np.nan
            ret6 = df["ret6"].iloc[i] if "ret6" in df else np.nan
            ret12 = df["ret12"].iloc[i] if "ret12" in df else np.nan
            ret24 = df["ret24"].iloc[i] if "ret24" in df else np.nan

            # --- 12-candle forward return (exit-independent, percent) --
            fwd_idx = min(entry_bar + HOLD_WINDOW - 1, len(df) - 1)
            close_fwd = df["close"].iloc[fwd_idx]
            if signal == "BUY":
                fwd_ret = (close_fwd - entry_price) / entry_price * 100
            else:
                fwd_ret = (entry_price - close_fwd) / entry_price * 100

            # --- MFE / MAE over the next 12 candles from ENTRY price ---
            mfe = 0.0
            mae = 0.0
            for j in range(entry_bar, min(entry_bar + HOLD_WINDOW, len(df))):
                move_up = df["high"].iloc[j] - entry_price
                move_dn = entry_price - df["low"].iloc[j]
                mfe = max(mfe, move_up if signal == "BUY" else move_dn)
                mae = max(mae, move_dn if signal == "BUY" else move_up)
            mfe_atr = mfe / atr if atr > 0 else np.nan
            mae_atr = mae / atr if atr > 0 else np.nan

            # --- Virtual P/L from the INSTRUMENT's own specifications ---
            if signal == "BUY":
                gross_pnl = (exit_price - entry_price) * volume * value_per_unit
            else:
                gross_pnl = (entry_price - exit_price) * volume * value_per_unit

            net_pnl = gross_pnl - COST_PER_TRADE
            balance += net_pnl

            trades.append(
                {
                    "trade": len(trades) + 1,
                    "direction": signal,
                    "signal_time": df["time"].iloc[i],
                    "entry_time": df["time"].iloc[entry_bar],
                    "entry_price": entry_price,
                    "exit_time": df["time"].iloc[exit_index],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "ADX": df["ADX"].iloc[i],
                    "ATR_pct": atr_pct,
                    "EMA50_dist": ema50_dist,
                    "BB_exc": bb_exc,
                    "eff24": eff24,
                    "ret6": ret6,
                    "ret12": ret12,
                    "ret24": ret24,
                    "MFE_ATR": mfe_atr,
                    "MAE_ATR": mae_atr,
                    "fwd_ret_pct": fwd_ret,
                    "pnl": net_pnl,
                    "balance_after": balance,
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 7: Descriptive statistics and cell printers (pure helpers;
# no optimization, no ranking - cells are only described)
# ============================================================
def split_periods(df, num_periods=NUM_PERIODS):
    """Split a closed-candle DataFrame into N approximately equal
    CHRONOLOGICAL quartiles (simple slicing - no shuffling, no tuning).

    Returns (list of period DataFrames, list of boundary indices).
    """
    n = len(df)
    edges = [round(k * n / num_periods) for k in range(num_periods + 1)]
    periods = [df.iloc[edges[p]:edges[p + 1]].copy().reset_index(drop=True)
               for p in range(num_periods)]
    return periods, edges


def run_frozen_four_periods(df, value_per_unit, volume):
    """Run the frozen backtest on four chronological quartiles.

    Period 1 skips the indicator warm-up bars; later periods start at
    bar 0 (their indicators were already warmed up earlier in the
    series) and each period restarts from its own $10,000 balance.
    Returns (list of trade DataFrames, list of period frames).
    """
    periods, _ = split_periods(df)
    trades_list = []
    for p, sub in enumerate(periods, 1):
        tr, _ = run_backtest(sub, value_per_unit, volume,
                             start_bar=(WARMUP if p == 1 else 0))
        trades_list.append(tr)
    return trades_list, periods


def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def longest_losing_streak(trades):
    """Longest run of CONSECUTIVE losing trades (pnl < 0) in trade order."""
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


def period_stats(trades):
    """Descriptive stats for one period (or any trade list)."""
    if trades is None or len(trades) == 0:
        return {"trades": 0, "return": float("nan"), "win_rate": float("nan"),
                "pf": float("nan"), "avg_trade": float("nan"),
                "total_pnl": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0,
                "ending_balance": STARTING_BALANCE, "lose_streak": 0}
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
        drawdown = peak - b
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown / peak * 100

    return {
        "trades": len(trades),
        "return": (trades["balance_after"].iloc[-1] - STARTING_BALANCE) / STARTING_BALANCE * 100,
        "win_rate": len(wins) / len(trades) * 100,
        "pf": gp / gl if gl > 0 else float("inf"),
        "avg_trade": trades["pnl"].mean(),
        "total_pnl": trades["pnl"].sum(),
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "ending_balance": trades["balance_after"].iloc[-1],
        "lose_streak": longest_losing_streak(trades),
    }


def regime_cell_stats(sub):
    """Descriptive stats for one regime cell.

    Returns a dict or None when the cell has no trades:
      n, avg_pnl, total_pnl, win_rate, pf, avg_fwd (percent),
      avg_mfe, avg_mae (ATR units).
    """
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
        "avg_fwd": sub["fwd_ret_pct"].mean(),
        "avg_mfe": sub["MFE_ATR"].mean(),
        "avg_mae": sub["MAE_ATR"].mean(),
    }


def print_cell_line(prefix, sub, with_fwd=True, with_mfe=False, cells_log=None,
                    cell_tag=""):
    """Print one regime cell; optionally record it for the LOW SAMPLE count.

    Returns the stats dict (or None for an empty cell).
    """
    s = regime_cell_stats(sub)
    if s is None:
        print(f"      {prefix:<26} (no trades)")
        if cells_log is not None:
            cells_log.append({"tag": cell_tag or prefix, "n": 0})
        return None
    tag = f"  [{LOW_SAMPLE_TAG}]" if s["n"] < LOW_SAMPLE_N else ""
    line = (f"      {prefix:<26} {s['n']:>4} trades | "
            f"avg ${s['avg_pnl']:>8.2f} | total ${s['total_pnl']:>10.2f} | "
            f"win {s['win_rate']:>5.1f}% | PF {fmt_pf(s['pf']):>5}")
    if with_fwd:
        line += f" | fwd12 {s['avg_fwd']:+.3f}%"
    if with_mfe:
        line += (f" | MFE {s['avg_mfe']:.2f} | MAE {s['avg_mae']:.2f}"
                 f" (ATR)")
    print(line + tag)
    if cells_log is not None:
        cells_log.append({"tag": cell_tag or prefix, "n": s["n"]})
    return s


def money(v):
    """Format a money value, handling 'no trades' (NaN)."""
    return "no trades" if (v != v) else f"${v:+.2f}"


def num(v, spec="{:.2f}"):
    """Format a number, printing n/a for NaN."""
    return "n/a" if (v != v) else spec.format(v)


def sign_changes(vals):
    """Period-to-period sign transitions in a list of values.

    Returns labels like 'P1->P2' for each transition where the sign
    flips between two non-NaN, non-zero values (factual counts only).
    """
    changes = []
    for p in range(len(vals) - 1):
        a, b = vals[p], vals[p + 1]
        if a == a and b == b and a != 0 and b != 0:
            if (a > 0) != (b > 0):
                changes.append(f"P{p + 1}->P{p + 2}")
    return changes


# ============================================================
# SECTION 8: Factual questions (pure function; answers are computed
# ONLY from the collected data - descriptive association wording, no
# causal claims, no rankings, no recommendations)
# ============================================================
def factual_answers(instrument_data, cells_log):
    """Compute the eleven Section 11 answers.

    instrument_data: list of dicts {instrument, trades (labelled full-
    history log), period_trades (list of 4 labelled logs), edges}.
    cells_log: list of {"tag", "n"} records from every printed regime
    cell (used for the LOW SAMPLE count).
    """
    facts = {}

    def bucket_avgs(trades, col, order):
        out = {}
        for b in order:
            sub = trades[trades[col] == b]
            out[b] = sub["pnl"].mean() if len(sub) else float("nan")
        return out

    # --- Q1: different average P/L across volatility regimes? ---------
    q1 = {}
    for d in instrument_data:
        avgs = bucket_avgs(d["trades"], "vol_regime", VOL_BUCKETS)
        valid = [v for v in avgs.values() if v == v]
        q1[d["instrument"]] = {
            "avgs": avgs,
            "differs": len(valid) >= 2 and (max(valid) - min(valid)) > 1e-12,
        }
    facts["q1"] = q1

    # --- Q2: different average P/L across efficiency regimes? ---------
    q2 = {}
    for d in instrument_data:
        avgs = bucket_avgs(d["trades"], "eff_regime", EFF_BUCKETS)
        valid = [v for v in avgs.values() if v == v]
        q2[d["instrument"]] = {
            "avgs": avgs,
            "differs": len(valid) >= 2 and (max(valid) - min(valid)) > 1e-12,
        }
    facts["q2"] = q2

    # --- Q3: different average P/L across ADX regimes below 25? -------
    q3 = {}
    adx_below = ["ADX<15", "ADX15-20", "ADX20-25"]
    for d in instrument_data:
        avgs = bucket_avgs(d["trades"], "adx_regime", adx_below)
        valid = [v for v in avgs.values() if v == v]
        q3[d["instrument"]] = {
            "avgs": avgs,
            "differs": len(valid) >= 2 and (max(valid) - min(valid)) > 1e-12,
        }
    facts["q3"] = q3

    # --- Q4: is the volatility relationship similar across instruments?
    # Factual: for instruments whose bucket averages actually DIFFER,
    # which bucket showed the highest / lowest historical average
    # (counted - instruments are never ranked; instruments with
    # identical bucket averages contribute nothing to count).
    q4 = {"highest": {}, "lowest": {}}
    for d in instrument_data:
        item = facts["q1"][d["instrument"]]
        if not item["differs"]:
            continue
        valid = {k: v for k, v in item["avgs"].items() if v == v}
        if len(valid) >= 2:
            hi = max(valid, key=valid.get)
            lo = min(valid, key=valid.get)
            q4["highest"][hi] = q4["highest"].get(hi, 0) + 1
            q4["lowest"][lo] = q4["lowest"].get(lo, 0) + 1
    facts["q4"] = q4

    # --- Q5: is the efficiency relationship similar across instruments?
    # (same counting rule as Q4 - only instruments whose bucket
    # averages actually differ are counted)
    q5 = {"highest": {}, "lowest": {}}
    for d in instrument_data:
        item = facts["q2"][d["instrument"]]
        if not item["differs"]:
            continue
        valid = {k: v for k, v in item["avgs"].items() if v == v}
        if len(valid) >= 2:
            hi = max(valid, key=valid.get)
            lo = min(valid, key=valid.get)
            q5["highest"][hi] = q5["highest"].get(hi, 0) + 1
            q5["lowest"][lo] = q5["lowest"].get(lo, 0) + 1
    facts["q5"] = q5

    # --- Q6: is the relationship consistent across the four periods? ---
    # For each instrument and each main regime dimension, count the
    # bucket/period cells whose average P/L changes sign between
    # consecutive periods (factual counts only).
    q6 = {}
    dims = (("vol_regime", VOL_BUCKETS), ("eff_regime", EFF_BUCKETS),
            ("adx_regime", ADX_BUCKETS))
    for d in instrument_data:
        per_dim = {}
        for col, order in dims:
            flips = 0
            cells = 0
            for b in order:
                per_period_avg = []
                for pt in d["period_trades"]:
                    sub = pt[pt[col] == b]
                    per_period_avg.append(sub["pnl"].mean() if len(sub)
                                          else float("nan"))
                ch = sign_changes(per_period_avg)
                flips += len(ch)
                cells += sum(1 for v in per_period_avg if v == v)
            per_dim[col] = {"flips": flips, "cells": cells}
        q6[d["instrument"]] = per_dim
    facts["q6"] = q6

    # --- Q7: regime combinations that change sign between periods -----
    q7_list = []
    for d in instrument_data:
        for col, order in dims:
            for b in order:
                per_period_avg = []
                for pt in d["period_trades"]:
                    sub = pt[pt[col] == b]
                    per_period_avg.append(sub["pnl"].mean() if len(sub)
                                          else float("nan"))
                ch = sign_changes(per_period_avg)
                if ch:
                    q7_list.append({
                        "instrument": d["instrument"],
                        "regime": f"{col}={b}",
                        "changes": ch,
                    })
    facts["q7"] = q7_list

    # --- Q8: regime cells with fewer than 5 trades --------------------
    low = sum(1 for c in cells_log if c["n"] < LOW_SAMPLE_N)
    facts["q8"] = {"low": low, "total": len(cells_log)}

    # --- Q9: does performance vary with market regime? (descriptive) ---
    dims_with_diff = 0
    dims_total = 0
    for d in instrument_data:
        for q in (facts["q1"], facts["q2"], facts["q3"]):
            item = q[d["instrument"]]
            dims_total += 1
            if item["differs"]:
                dims_with_diff += 1
    facts["q9"] = {"dims_with_diff": dims_with_diff, "dims_total": dims_total}

    return facts


def print_factual_answers(facts):
    """Print the eleven factual questions with association wording."""

    print()
    print("1. Does the strategy show different average P/L across volatility")
    print("   regimes?")
    for name, item in facts["q1"].items():
        txt = " | ".join(f"{b}: {money(v)}" for b, v in item["avgs"].items())
        verdict = ("YES - the historical average P/L differed between buckets"
                   if item["differs"] else
                   "NO clear difference observed between buckets (or too few "
                   "populated buckets)")
        print(f"   {name:<12}: {verdict}")
        print(f"   {'':<12}  {txt}")

    print()
    print("2. Does the strategy show different average P/L across efficiency")
    print("   regimes?")
    for name, item in facts["q2"].items():
        txt = " | ".join(f"{b}: {money(v)}" for b, v in item["avgs"].items())
        verdict = ("YES - the historical average P/L differed between buckets"
                   if item["differs"] else
                   "NO clear difference observed between buckets (or too few "
                   "populated buckets)")
        print(f"   {name:<12}: {verdict}")
        print(f"   {'':<12}  {txt}")

    print()
    print("3. Does the strategy show different average P/L across ADX")
    print("   regimes below 25?")
    for name, item in facts["q3"].items():
        txt = " | ".join(f"{b}: {money(v)}" for b, v in item["avgs"].items())
        verdict = ("YES - the historical average P/L differed between buckets"
                   if item["differs"] else
                   "NO clear difference observed between buckets (or too few "
                   "populated buckets)")
        print(f"   {name:<12}: {verdict}")
        print(f"   {'':<12}  {txt}")

    print()
    print("4. Does the volatility relationship remain similar across all")
    print("   five instruments?")
    hi = facts["q4"]["highest"]
    lo = facts["q4"]["lowest"]
    hi_txt = ", ".join(f"{k} in {v}" for k, v in hi.items()) or "insufficient data"
    lo_txt = ", ".join(f"{k} in {v}" for k, v in lo.items()) or "insufficient data"
    print(f"   Highest historical average was in: {hi_txt} instrument(s)")
    print(f"   Lowest historical average was in:  {lo_txt} instrument(s)")
    print("   (Factual counts only - instruments are never ranked.)")

    print()
    print("5. Does the efficiency relationship remain similar across all")
    print("   five instruments?")
    hi = facts["q5"]["highest"]
    lo = facts["q5"]["lowest"]
    hi_txt = ", ".join(f"{k} in {v}" for k, v in hi.items()) or "insufficient data"
    lo_txt = ", ".join(f"{k} in {v}" for k, v in lo.items()) or "insufficient data"
    print(f"   Highest historical average was in: {hi_txt} instrument(s)")
    print(f"   Lowest historical average was in:  {lo_txt} instrument(s)")
    print("   (Factual counts only - instruments are never ranked.)")

    print()
    print("6. Does the relationship remain consistent across all four")
    print("   chronological periods?")
    for name, per_dim in facts["q6"].items():
        parts = []
        for col, flips, cells in (
            ("vol", per_dim["vol_regime"]["flips"], per_dim["vol_regime"]["cells"]),
            ("eff", per_dim["eff_regime"]["flips"], per_dim["eff_regime"]["cells"]),
            ("adx", per_dim["adx_regime"]["flips"], per_dim["adx_regime"]["cells"]),
        ):
            parts.append(f"{col}: {flips} sign change(s) across {cells} "
                         f"populated bucket/period cells")
        print(f"   {name:<12}: " + "; ".join(parts))

    print()
    print("7. Are there regime combinations where the trade outcome changes")
    print("   sign between periods?")
    if facts["q7"]:
        print(f"   {len(facts['q7'])} regime combination(s) changed sign "
              f"between consecutive periods:")
        for item in facts["q7"]:
            print(f"   - {item['instrument']:<12} {item['regime']:<24} "
                  f"at {', '.join(item['changes'])}")
    else:
        print("   None observed: no populated regime bucket changed the sign")
        print("   of its average P/L between consecutive periods.")
    print("   (A sign change between periods is a factual observation, not")
    print("    evidence of instability by itself.)")

    q8 = facts["q8"]
    print()
    print("8. Are there regime combinations with fewer than 5 trades that")
    print("   should be treated as low sample?")
    print(f"   Yes: {q8['low']} of {q8['total']} printed regime cells had "
          f"fewer than {LOW_SAMPLE_N} trades.")
    print("   Every such cell is tagged LOW SAMPLE above. Small cells are")
    print("   descriptive only and were NOT interpreted as evidence.")

    q9 = facts["q9"]
    print()
    print("9. Does the data provide evidence that performance varies with")
    print("   market regime? (descriptive answer)")
    print(f"   In this historical window, {q9['dims_with_diff']} of "
          f"{q9['dims_total']} instrument/regime-dimension combinations")
    print("   (volatility, efficiency, ADX) showed different historical")
    print("   average P/L between their descriptive buckets. Where a")
    print("   difference was observed, regime was ASSOCIATED with different")
    print("   historical outcomes in this sample. This is an association")
    print("   observed in one historical window with fixed descriptive")
    print("   buckets - it does NOT establish that any regime causes the")
    print("   performance, and it does not recommend any strategy change.")

    print()
    print("Interpretation rule applied throughout: wording uses 'was")
    print("associated with', 'showed a higher/lower historical average',")
    print("'changed between periods', 'was inconsistent' and 'sample size")
    print("was small'. No causal claims, no predictions, no rankings, and")
    print("no recommendations are made.")


# ============================================================
# SECTION 9: Main program (MT5 read-only)
# ============================================================
def main():
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
        print("MARKET REGIME DIAGNOSTIC (read-only, descriptive only)")
        print("Frozen strategy + fixed descriptive regime buckets.")
        print("Question: in what market conditions did the frozen strategy")
        print("behave differently? (association wording only)")
        print("=" * 70)
        print()
        print("Timeframe: H1 | BB(20, 2.0) + ADX14 < 25 entry | "
              f"SL {SL_ATR_MULT} x ATR / TP {TP_ATR_MULT} x ATR exits")
        print(f"{LOTS_REQUESTED} lots (validated per symbol) | "
              f"${COST_PER_TRADE:.2f}/trade | next-open entry | stop-first")
        print("No parameter below is optimized; regime buckets are fixed")
        print("descriptive classifications and NEVER act as filters.")

        instrument_data = []   # collected for Sections 10-11
        cells_log = []         # every printed regime cell (LOW SAMPLE count)
        unavailable = []

        for requested in INSTRUMENTS:
            print()
            print("#" * 70)
            print(f"# INSTRUMENT: {requested}")
            print("#" * 70)

            try:
                # --- Symbol discovery (no substitution) ---------------
                actual_symbol, note = discover_symbol(requested)
                if actual_symbol is None:
                    print(f"STATUS: NOT AVAILABLE - {note}. "
                          f"Continuing with the remaining instruments.")
                    unavailable.append(requested)
                    continue

                print(f"Requested: {requested}  ->  using MT5 symbol: "
                      f"{actual_symbol}  ({note})")

                if not mt5.symbol_select(actual_symbol, True):
                    print(f"STATUS: NOT AVAILABLE - {actual_symbol} could not be "
                          f"enabled in Market Watch. Continuing with the "
                          f"remaining instruments.")
                    unavailable.append(requested)
                    continue

                spec = get_contract_spec(actual_symbol)
                if spec is None:
                    print(f"STATUS: NOT AVAILABLE - no symbol_info for "
                          f"{actual_symbol}. Continuing with the remaining "
                          f"instruments.")
                    unavailable.append(requested)
                    continue

                volume = validate_volume(spec)
                if volume is None:
                    print(f"STATUS: SKIPPED - requested {LOTS_REQUESTED} lots is "
                          f"not a valid volume for {actual_symbol}. Continuing "
                          f"with the remaining instruments.")
                    unavailable.append(requested)
                    continue

                # --- Download history (closed candles only) -----------
                print(f"Downloading up to {NUM_CANDLES} candles for "
                      f"{actual_symbol} (H1)...")
                rates = mt5.copy_rates_from_pos(actual_symbol, TIMEFRAME, 0, NUM_CANDLES)

                if rates is None or len(rates) == 0:
                    print(f"STATUS: NOT AVAILABLE - no historical data returned "
                          f"for {actual_symbol}. Continuing with the remaining "
                          f"instruments.")
                    unavailable.append(requested)
                    continue

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df = df.iloc[:-1].reset_index(drop=True)   # drop forming candle

                if len(df) < MIN_CANDLES:
                    print(f"STATUS: NOT AVAILABLE - only {len(df)} closed candles "
                          f"for {actual_symbol} (minimum {MIN_CANDLES}). "
                          f"Continuing with the remaining instruments.")
                    unavailable.append(requested)
                    continue

                # ======================================================
                # SECTION 1 - DATA
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 1 - DATA: {requested}")
                print("=" * 70)
                print(f"  actual closed candles: {len(df)}")
                if len(df) < NUM_CANDLES:
                    print(f"    NOTE: fewer than {NUM_CANDLES} closed candles "
                          f"available; using ALL {len(df)} available closed "
                          f"candles.")
                print(f"  first timestamp      : {df['time'].iloc[0]}")
                print(f"  last timestamp       : {df['time'].iloc[-1]}")
                print(f"  broker specs         : contract size "
                      f"{spec['contract_size']:g} | point {spec['point']:g} | "
                      f"digits {spec['digits']} | tick size "
                      f"{spec['tick_size']:g} | tick value "
                      f"{spec['tick_value']:g} | quote currency "
                      f"{spec['quote_currency']}")
                pl_note = ""
                if spec["tick_value_fallback"]:
                    pl_note = " (documented fallback tick_value = contract x point"
                    pl_note += ", converted to USD" if spec["fallback_converted"] else ""
                    pl_note += ")"
                print(f"  P/L basis            : 1.0 price-unit move with 1.0 lot "
                      f"= {spec['value_per_unit']:.2f} "
                      f"(tick_value / tick_size){pl_note}")
                print(f"  volume               : {volume} lots (requested "
                      f"{LOTS_REQUESTED}; volume_min {spec['volume_min']}, "
                      f"volume_step {spec['volume_step']})")

                # --- Indicators ONCE over the full closed series -------
                # (causal: every value uses only current + previous candles)
                df = add_atr(df, ATR_PERIOD)
                df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
                df = add_adx(df, ADX_PERIOD)
                df = add_ema(df, EMA_TREND_PERIOD)   # diagnostic reference only
                df = add_diagnostic_measurements(df)

                # --- Volatility edges from the FULL candle history -----
                v1, v2 = compute_vol_edges_from_candles(df)
                print(f"  volatility thresholds: 33rd pct {v1:.4f}% | "
                      f"66th pct {v2:.4f}% (ATR%; computed ONCE from the FULL")
                print(f"                         closed-candle history of this "
                      f"instrument - never per period, never optimized)")

                # ======================================================
                # SECTION 2 - OVERALL FROZEN STRATEGY (full history)
                # ======================================================
                full_trades, full_balance = run_backtest(df, spec["value_per_unit"],
                                                         volume, start_bar=WARMUP)
                if len(full_trades) == 0:
                    print(f"\nNOTE: no trades generated for {requested}; "
                          f"regime tables will be empty.")
                    unavailable.append(requested)
                    continue

                full_trades = assign_regimes(full_trades, v1, v2)
                ost = period_stats(full_trades)

                print()
                print("=" * 70)
                print(f"SECTION 2 - OVERALL FROZEN STRATEGY: {requested}")
                print("(one continuous run over the full window, independent "
                      "$10,000)")
                print("=" * 70)
                print(f"  trades               : {ost['trades']}")
                print(f"  return               : {ost['return']:+.2f}%")
                print(f"  win rate             : {ost['win_rate']:.1f}%")
                print(f"  profit factor        : {fmt_pf(ost['pf'])}")
                print(f"  average $/trade      : ${ost['avg_trade']:+.2f}")
                print(f"  maximum drawdown $   : ${ost['max_dd']:,.2f}")
                print(f"  maximum drawdown %   : {ost['max_dd_pct']:.2f}%")
                print(f"  longest losing streak: {ost['lose_streak']} trades")

                # ======================================================
                # SECTION 3 - ADX REGIME
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 3 - ADX REGIME: {requested}")
                print("(descriptive classification only - the frozen strategy")
                print(" still enters only when ADX < 25, so ADX>=25 may")
                print(" legitimately contain zero strategy trades)")
                print("=" * 70)
                for b in ADX_BUCKETS:
                    print_cell_line(b, full_trades[full_trades["adx_regime"] == b],
                                    cells_log=cells_log,
                                    cell_tag=f"{requested}|adx|{b}")

                # ======================================================
                # SECTION 4 - VOLATILITY REGIME
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 4 - VOLATILITY REGIME: {requested}")
                print(f"(LOW < {v1:.4f}% <= MEDIUM < {v2:.4f}% <= HIGH)")
                print("=" * 70)
                for b in VOL_BUCKETS:
                    print_cell_line(b, full_trades[full_trades["vol_regime"] == b],
                                    with_mfe=True, cells_log=cells_log,
                                    cell_tag=f"{requested}|vol|{b}")

                # ======================================================
                # SECTION 5 - TREND DISTANCE
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 5 - TREND DISTANCE (EMA50 distance in ATR): "
                      f"{requested}")
                print("=" * 70)
                for b in TREND_BUCKETS:
                    print_cell_line(b, full_trades[full_trades["trend_regime"] == b],
                                    with_mfe=True, cells_log=cells_log,
                                    cell_tag=f"{requested}|trend|{b}")

                # ======================================================
                # SECTION 6 - MARKET EFFICIENCY
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 6 - MARKET EFFICIENCY (24-candle efficiency "
                      f"ratio): {requested}")
                print("=" * 70)
                for b in EFF_BUCKETS:
                    print_cell_line(b, full_trades[full_trades["eff_regime"] == b],
                                    cells_log=cells_log,
                                    cell_tag=f"{requested}|eff|{b}")

                # ======================================================
                # SECTION 7 - DIRECTION x REGIME
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 7 - DIRECTION x REGIME: {requested}")
                print("=" * 70)
                for d, dname in (("BUY", "LONG"), ("SELL", "SHORT")):
                    d_tr = full_trades[full_trades["direction"] == d]
                    print(f"\n  {dname} x volatility:")
                    for b in VOL_BUCKETS:
                        print_cell_line(b, d_tr[d_tr["vol_regime"] == b],
                                        cells_log=cells_log,
                                        cell_tag=f"{requested}|{dname}-vol|{b}")
                    print(f"\n  {dname} x efficiency:")
                    for b in EFF_BUCKETS:
                        print_cell_line(b, d_tr[d_tr["eff_regime"] == b],
                                        cells_log=cells_log,
                                        cell_tag=f"{requested}|{dname}-eff|{b}")

                # ======================================================
                # SECTION 8 - REGIME INTERACTIONS
                # ======================================================
                print()
                print("=" * 70)
                print(f"SECTION 8 - REGIME INTERACTIONS: {requested}")
                print("=" * 70)

                def print_interaction(dim_a, order_a, dim_b, order_b, label):
                    print(f"\n  {label}:")
                    for ba in order_a:
                        a_tr = full_trades[full_trades[dim_a] == ba]
                        if len(a_tr) == 0:
                            continue
                        for bb_ in order_b:
                            sub = a_tr[a_tr[dim_b] == bb_]
                            print_cell_line(f"{ba} x {bb_}", sub,
                                            cells_log=cells_log,
                                            cell_tag=f"{requested}|{label}|{ba}x{bb_}")

                print_interaction("adx_regime", ["ADX<15", "ADX15-20", "ADX20-25"],
                                  "vol_regime", VOL_BUCKETS, "ADX x volatility")
                print_interaction("adx_regime", ["ADX<15", "ADX15-20", "ADX20-25"],
                                  "eff_regime", EFF_BUCKETS, "ADX x efficiency")
                print_interaction("vol_regime", VOL_BUCKETS,
                                  "trend_regime", TREND_BUCKETS,
                                  "volatility x trend distance")
                print_interaction("eff_regime", EFF_BUCKETS,
                                  "trend_regime", TREND_BUCKETS,
                                  "efficiency x trend distance")

                # ======================================================
                # SECTION 9 - PERIOD x REGIME (four chronological quartiles)
                # ======================================================
                period_trades, periods_df = run_frozen_four_periods(
                    df, spec["value_per_unit"], volume)
                period_trades = [assign_regimes(t, v1, v2) if len(t) else t
                                 for t in period_trades]

                print()
                print("=" * 70)
                print(f"SECTION 9 - PERIOD x REGIME: {requested}")
                print("(four chronological quartiles, independent $10,000 each)")
                print("=" * 70)
                for p, sub in enumerate(periods_df, 1):
                    print(f"\nPeriod {p}: {sub['time'].iloc[0]} -> "
                          f"{sub['time'].iloc[-1]} ({len(sub)} candles)")

                def print_period_table(col, order, title):
                    print(f"\n  {title}:")
                    for p, pt in enumerate(period_trades, 1):
                        for b in order:
                            sub = pt[pt[col] == b] if len(pt) else pt
                            print_cell_line(f"P{p} {b}", sub,
                                            with_fwd=False,
                                            cells_log=cells_log,
                                            cell_tag=f"{requested}|P{p}|{title}|{b}")

                print_period_table("vol_regime", VOL_BUCKETS, "Period x volatility")
                print_period_table("eff_regime", EFF_BUCKETS, "Period x efficiency")
                print_period_table("adx_regime", ADX_BUCKETS, "Period x ADX")

                print()
                print("(Period x regime cells are descriptive; nothing is")
                print(" optimized or selected across periods.)")

                # --- Collect for Sections 10-11 ------------------------
                instrument_data.append({
                    "instrument": requested,
                    "trades": full_trades,
                    "period_trades": period_trades,
                    "edges": (v1, v2),
                })

            except Exception as exc:  # never let one instrument kill the run
                print(f"STATUS: ERROR while processing {requested}: {exc}")
                print("Continuing with the remaining instruments.")
                unavailable.append(requested)

        # ==========================================================
        # SECTION 10 - CROSS-INSTRUMENT SUMMARY (factual tables only)
        # ==========================================================
        print()
        print("=" * 70)
        print("SECTION 10 - CROSS-INSTRUMENT SUMMARY (average P/L per")
        print("descriptive regime bucket - factual tables, no ranking)")
        print("=" * 70)

        if instrument_data:
            def cross_table(col, order, title):
                print(f"\n{title}")
                hdr = (f"   {'Instrument':<12}" +
                       "".join(f"{b:>16}" for b in order))
                print(hdr)
                print("   " + "-" * (12 + 16 * len(order)))
                for d in instrument_data:
                    line = f"   {d['instrument']:<12}"
                    for b in order:
                        sub = d["trades"][d["trades"][col] == b]
                        s = regime_cell_stats(sub)
                        val = "n/a" if s is None else f"${s['avg_pnl']:+.2f}"
                        line += f"{val:>16}"
                    print(line)
                print("   (Average P/L per trade in each fixed bucket; 'n/a' =")
                print("   no trades in that bucket. Instruments are NOT ranked.)")

            cross_table("vol_regime", VOL_BUCKETS,
                        "Average P/L by volatility regime:")
            cross_table("eff_regime", EFF_BUCKETS,
                        "Average P/L by efficiency regime:")
            cross_table("adx_regime", ["ADX<15", "ADX15-20", "ADX20-25"],
                        "Average P/L by ADX regime (below 25 the strategy can trade):")
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
        # SECTION 11 - FACTUAL QUESTIONS
        # ==========================================================
        if instrument_data:
            print()
            print("=" * 70)
            print("SECTION 11 - FACTUAL QUESTIONS (association wording only)")
            print("=" * 70)
            print_factual_answers(factual_answers(instrument_data, cells_log))

        print()
        print("Historical backtest with virtual money. Descriptive")
        print("associations in one historical window do not predict future")
        print("performance, do not establish causation, and are not trading")
        print("advice. Nothing was optimized and no strategy change is")
        print("suggested. Educational use only.")

    finally:
        # ==========================================================
        # Always disconnect when the script finishes
        # ==========================================================
        mt5.shutdown()
        print()
        print("MetaTrader 5 connection closed.")


if __name__ == "__main__":
    main()
