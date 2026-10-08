"""
bb_regime_experiment.py - Does a MARKET REGIME filter change the Bollinger
mean-reversion result?

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: no mt5.order_send() anywhere in this file.
- The script only READS historical candles from MetaTrader 5.
- Never connects to a live trading account.

THE HYPOTHESIS BEING TESTED:
The Bollinger Band entry seems to capture a short-term bounce after an
extreme move (win rate ~62.5%, profit factor ~1.00 - breakeven). But some
signals may fire during STRONG DIRECTIONAL TRENDS, where fading the move
(mean reversion) keeps failing. So we keep the entry and exits IDENTICAL
and only change WHEN a Bollinger signal is allowed to trigger:

  VARIANT A - BASELINE (no regime filter)
      The exact previous experiment: BB entry + ATR exits. It must
      approximately reproduce: ~104 trades, ~+0.04% return, PF ~1.00,
      win rate ~62.5%, max DD ~1.19%. If it differs materially, the
      backtest implementation should be investigated FIRST.

  VARIANT B - EMA50 REGIME FILTER (fade WITH the broader trend)
      LONG : close < lower band  AND  close > EMA50
      SHORT: close > upper band  AND  close < EMA50
      Rationale: a long fade is 'safer' above the EMA50 (pullback in an
      uptrend) and a short fade below it. Uses exactly EMA50 (not tuned).

  VARIANT C - ADX14 REGIME FILTER (only fade in quiet/ranging markets)
      LONG : close < lower band  AND  ADX14 < 25
      SHORT: close > upper band  AND  ADX14 < 25
      Rationale: ADX measures trend STRENGTH (not direction); below 25 is
      conventionally 'no strong trend'. Uses exactly ADX14 and exactly
      the threshold 25 (no other values were tested - not optimized).

ENTRY RULES - IDENTICAL FOR ALL VARIANTS:
  LONG : candle CLOSES below the lower Bollinger Band (20, 2.0)
  SHORT: candle CLOSES above the upper Bollinger Band
  Entry happens at the NEXT candle's OPEN (first moment a real trader
  could act). Only one virtual position at a time.

EXIT RULES - IDENTICAL FOR ALL VARIANTS (same as the previous experiments):
  Stop loss   = entry -/+ 2.0 x ATR14 (fixed at entry, from the signal bar)
  Take profit = entry +/- 1.5 x ATR14
  If BOTH levels fall inside the SAME candle, the STOP LOSS is assumed to
  fill first (the conservative, pessimistic assumption).

NO LOOK-AHEAD BIAS:
- The newest (still forming) candle is dropped.
- A signal on bar i uses only closed candles 0..i (BB, EMA50, ADX, ATR
  are all computed from closed candles).
- The virtual entry happens at the OPEN of bar i+1.
- Because trades are held open, the trade sequences can drift apart
  between variants (one variant is still in a position while another is
  free) - the filter changes WHICH trades happen, exactly what we test.

DISCLAIMER: This is a HISTORICAL backtest with virtual money.
Past results do NOT predict future performance and nothing here is
proven to be profitable. No parameters were searched or optimized.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read historical candles (read-only usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # np.select() to build the signal column

# ============================================================
# SECTION 2: Settings - fixed up front, NOT optimized
# ============================================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 2000   # Same history size as the previous experiments
MIN_CANDLES = 1100   # Absolute minimum we accept

# --- Virtual account settings (same as previous experiments) ---
STARTING_BALANCE = 10_000.0
LOTS = 0.10                  # Fixed position size
CONTRACT_SIZE = 100_000      # 1.00 lot = 100,000 units of base currency
SPREAD = 0.0002              # 2.0 pips round-trip cost = $2.00 per trade

# --- Exit distances (ATR multiples), same as the previous experiments ---
SL_ATR_MULT = 2.0   # Stop loss distance
TP_ATR_MULT = 1.5   # Take profit distance

# --- Indicator settings ---
BB_PERIOD = 20      # Bollinger moving average period
BB_NUM_STD = 2      # Bollinger standard deviation width
EMA_REGIME_PERIOD = 50   # Variant B: exactly 50 (not tuned)
ADX_PERIOD = 14          # Variant C: exactly 14 (not tuned)
ADX_THRESHOLD = 25.0     # Variant C: exactly 25 - "no strong trend"
ATR_PERIOD = 14
WARMUP = 60              # Bars needed before BB/ATR/RSI-style indicators
                         # are valid (ADX/EMA50 use their own NaN checks)


# ============================================================
# SECTION 3: Indicator functions (same math as the earlier scripts)
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


def add_bollinger(df, period=20, num_std=2):
    """Bollinger Bands.

    Middle band = simple moving average of the close.
    Upper/lower = middle +/- num_std standard deviations.
    """
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    df["BB_MIDDLE"] = middle
    df["BB_UPPER"] = middle + num_std * std
    df["BB_LOWER"] = middle - num_std * std
    return df


def add_ema(df, period):
    """Exponential Moving Average of the close price."""
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def add_adx(df, period=14):
    """Average Directional Index (Wilder's classic calculation).

    ADX measures trend STRENGTH, not direction:
    - rising/high ADX  = strong trend (either direction)
    - falling/low ADX  = weak trend / range conditions

    Steps (Wilder):
    1. +DM / -DM from consecutive highs/lows
    2. True Range (same as ATR)
    3. Wilder smoothing (EWM with alpha=1/period) -> +DI, -DI
    4. DX = 100 * |+DI - -DI| / (+DI + -DI)
    5. ADX = Wilder-smoothed DX
    """
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


# ============================================================
# SECTION 4: The backtest engine
# One function, three regime variants. Entry/exit RULES are identical;
# only the filter that gates the signal differs.
# ============================================================
def run_backtest(df, variant):
    """Run the virtual backtest. variant is "A", "B" or "C".
    Returns (trades DataFrame, final balance)."""

    # --- Raw Bollinger conditions (identical for every variant) ---
    long_condition = df["close"] < df["BB_LOWER"]    # Stretched below the band
    short_condition = df["close"] > df["BB_UPPER"]   # Stretched above the band

    # --- Apply the REGIME FILTER for this variant (entry only) ---
    if variant == "B":
        # Fade pullbacks WITH the broader EMA50 trend:
        # LONG only above EMA50, SHORT only below EMA50.
        long_condition = long_condition & (df["close"] > df["EMA50"])
        short_condition = short_condition & (df["close"] < df["EMA50"])
    elif variant == "C":
        # Only fade when ADX says there is NO strong trend.
        long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
        short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)
    # variant "A": no filter (raw conditions)

    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE
    spread_cost = SPREAD * CONTRACT_SIZE * LOTS  # Flat $2.00 per trade

    i = WARMUP
    while i < len(df) - 1:          # Need bar i+1 to exist for the entry
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        # Only enter on a valid signal with a valid ATR value
        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1                         # Next candle
            entry_price = df["open"].iloc[entry_bar]  # Enter at its open

            # Fix SL and TP at entry, using the ATR known at signal time
            if signal == "BUY":
                stop_loss = entry_price - SL_ATR_MULT * atr
                take_profit = entry_price + TP_ATR_MULT * atr
            else:  # SELL
                stop_loss = entry_price + SL_ATR_MULT * atr
                take_profit = entry_price - TP_ATR_MULT * atr

            # --- Forward scan: candle by candle, looking for the exit ---
            # STOP is checked FIRST: if both levels share one candle,
            # we assume the stop filled (conservative).
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
                else:  # SELL
                    if bar_high >= stop_loss:
                        exit_index, exit_price, exit_reason = j, stop_loss, "STOP_LOSS"
                        break
                    if bar_low <= take_profit:
                        exit_index, exit_price, exit_reason = j, take_profit, "TAKE_PROFIT"
                        break

            # Data ran out while the trade was open -> force-close
            if exit_index is None:
                exit_index = len(df) - 1
                exit_price = df["close"].iloc[exit_index]
                exit_reason = "END_OF_DATA"

            # --- Record the REGIME at signal time (for the diagnostics) ---
            adx_at_signal = df["ADX"].iloc[i]
            ema50_at_signal = df["EMA50"].iloc[i]
            close_at_signal = df["close"].iloc[i]

            if pd.notna(adx_at_signal) and adx_at_signal >= ADX_THRESHOLD:
                regime = "trending"
            else:
                regime = "ranging_or_unknown"

            if pd.notna(ema50_at_signal):
                if close_at_signal > ema50_at_signal:
                    ema_regime = "above_EMA50"
                else:
                    ema_regime = "below_or_at_EMA50"
            else:
                ema_regime = "unknown"

            # --- Virtual profit/loss (EURUSD quote currency = USD) ---
            if signal == "BUY":
                gross_pnl = (exit_price - entry_price) * CONTRACT_SIZE * LOTS
            else:
                gross_pnl = (entry_price - exit_price) * CONTRACT_SIZE * LOTS

            net_pnl = gross_pnl - spread_cost
            balance += net_pnl

            trades.append(
                {
                    "trade": len(trades) + 1,
                    "direction": signal,
                    "entry_time": df["time"].iloc[entry_bar],
                    "entry_price": round(entry_price, 5),
                    "exit_time": df["time"].iloc[exit_index],
                    "exit_price": round(exit_price, 5),
                    "exit_reason": exit_reason,
                    "hold_candles": exit_index - entry_bar + 1,
                    "ADX_at_signal": round(adx_at_signal, 1) if pd.notna(adx_at_signal) else None,
                    "regime_adx": regime,
                    "ema_regime": ema_regime,
                    "pnl": round(net_pnl, 2),
                    "balance_after": round(balance, 2),
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 5: Statistics printer (shared by all variants)
# ============================================================
def show_results(label, trades_df, final_balance):
    """Print all statistics for one variant. Returns a metrics dict."""
    print("\n" + "=" * 64)
    print(f"{label}  (HISTORICAL BACKTEST - VIRTUAL MONEY ONLY)")
    print("=" * 64)
    print(f"Rules: BB(20,2) entry, SL={SL_ATR_MULT}xATR / TP={TP_ATR_MULT}xATR, "
          f"{LOTS} lots, cost ${SPREAD * CONTRACT_SIZE * LOTS:.2f}/trade")

    if len(trades_df) == 0:
        print("\nNo virtual trades were generated in this variant.")
        return None

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    win_rate = len(wins) / len(trades_df) * 100
    total_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100

    # --- Maximum drawdown: biggest peak-to-trough drop of the balance ---
    balance_history = [STARTING_BALANCE] + trades_df["balance_after"].tolist()
    peak = balance_history[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for b in balance_history[1:]:
        if b > peak:
            peak = b
        drawdown = peak - b
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown / peak * 100

    print(f"\nStarting balance    : ${STARTING_BALANCE:,.2f}")
    print(f"Ending balance      : ${final_balance:,.2f}")
    print(f"Total return        : {total_return:+.2f}%")
    print(f"\nNumber of trades    : {len(trades_df)}")
    print(f"BUY trades          : {(trades_df['direction'] == 'BUY').sum()}")
    print(f"SELL trades         : {(trades_df['direction'] == 'SELL').sum()}")
    print(f"Winning trades      : {len(wins)}")
    print(f"Losing trades       : {len(losses)}")
    print(f"Win rate            : {win_rate:.1f}%")
    print(f"\nGross profit        : ${gross_profit:,.2f}")
    print(f"Gross loss          : ${gross_loss:,.2f}")
    if gross_loss > 0:
        print(f"Profit factor       : {gross_profit / gross_loss:.2f}")
    else:
        print("Profit factor       : infinite (no losing trades)")
    print(f"Average trade       : ${trades_df['pnl'].mean():,.2f}")
    print(f"\nMaximum drawdown    : ${max_dd:,.2f}  ({max_dd_pct:.2f}%)")

    # --- Exit reason breakdown ---
    print("\nExit reasons:")
    for reason, count in trades_df["exit_reason"].value_counts().items():
        print(f"  {reason:<14}: {count}")
    print(f"Average holding time: {trades_df['hold_candles'].mean():.1f} candles")

    # --- Show the last 10 virtual trades ---
    print("\nLast 10 virtual trades:")
    print(trades_df.tail(10).to_string(index=False))

    return {
        "return": total_return,
        "trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_dd_pct": max_dd_pct,
        "avg_trade": trades_df["pnl"].mean(),
        "avg_hold": trades_df["hold_candles"].mean(),
    }


# ============================================================
# SECTION 6: Regime diagnostics - WHAT did the filters actually remove?
# ============================================================
def show_regime_diagnostics(trades_a, trades_b, trades_c):
    """Break variant A's baseline trades down by regime, and list which
    baseline trades each filter kept/removed. Answers: does the hypothesis
    hold (do regime-filtered trades perform differently)?"""

    if len(trades_a) == 0:
        return

    print("\n" + "=" * 64)
    print("REGIME DIAGNOSTICS (all numbers from VARIANT A baseline trades)")
    print("=" * 64)

    # --- 1. Baseline performance split by ADX regime ---------------------
    # NOTE: uses the ADX value recorded at signal time in the trade log,
    # regardless of which filter the trade came from.
    print("\nBaseline (A) trades split by ADX regime at signal time:")
    for regime in ("ranging_or_unknown", "trending"):
        subset = trades_a[trades_a["regime_adx"] == regime]
        if len(subset) == 0:
            print(f"  {regime:<20}: no trades")
            continue
        w = subset[subset["pnl"] > 0]
        gl = abs(subset[subset["pnl"] < 0]["pnl"].sum())
        gp = subset[subset["pnl"] > 0]["pnl"].sum()
        pf = gp / gl if gl > 0 else float("inf")
        print(f"  {regime:<20}: {len(subset):>3} trades, "
              f"win rate {len(w) / len(subset) * 100:>5.1f}%, "
              f"PF {pf:.2f}, "
              f"avg ${subset['pnl'].mean():>6.2f}, "
              f"sum ${subset['pnl'].sum():>8.2f}")

    # --- 2. Baseline performance split by EMA50 side ---------------------
    print("\nBaseline (A) trades split by EMA50 side at signal time:")
    for side in ("above_EMA50", "below_or_at_EMA50", "unknown"):
        subset = trades_a[trades_a["ema_regime"] == side]
        if len(subset) == 0:
            print(f"  {side:<20}: no trades")
            continue
        w = subset[subset["pnl"] > 0]
        gl = subset[subset["pnl"] < 0]["pnl"].sum()
        gp = subset[subset["pnl"] > 0]["pnl"].sum()
        pf = gp / abs(gl) if gl < 0 else float("inf")
        print(f"  {side:<20}: {len(subset):>3} trades, "
              f"win rate {len(w) / len(subset) * 100:>5.1f}%, "
              f"PF {pf:.2f}, "
              f"avg ${subset['pnl'].mean():>6.2f}, "
              f"sum ${subset['pnl'].sum():>8.2f}")

    # --- 3. Which baseline trades does each filter remove? ---------------
    print("\nWhich baseline trades would each filter have REMOVED "
          "(by exit reason)?")
    print(f"  {'exit reason':<14}{'A total':>10}{'B removed':>11}{'C removed':>11}")
    all_reasons = sorted(set(trades_a["exit_reason"]))
    for reason in all_reasons:
        total_a = (trades_a["exit_reason"] == reason).sum()

        def removed_count(b_df):
            if len(b_df) == 0:
                return total_a
            kept_times = set(zip(b_df["entry_time"], b_df["direction"]))
            a_subset = trades_a[trades_a["exit_reason"] == reason]
            return sum(
                1 for t, d in zip(a_subset["entry_time"], a_subset["direction"])
                if (t, d) not in kept_times
            )

        print(f"  {reason:<14}{total_a:>10}"
              f"{removed_count(trades_b):>11}{removed_count(trades_c):>11}")
    print("  (Note: removed counts are approximate for trades taken AFTER")
    print("   a filtered-out trade changed the position-free timeline.)")

    # --- 4. Average ADX of winners vs losers (baseline) ------------------
    winners_adx = trades_a.loc[trades_a["pnl"] > 0, "ADX_at_signal"].dropna().astype(float)
    losers_adx = trades_a.loc[trades_a["pnl"] < 0, "ADX_at_signal"].dropna().astype(float)
    print("\nAverage ADX14 at signal (variant A):")
    if len(winners_adx):
        print(f"  winning trades : {winners_adx.mean():.1f}")
    else:
        print("  winning trades : n/a")
    if len(losers_adx):
        print(f"  losing trades  : {losers_adx.mean():.1f}")
    else:
        print("  losing trades  : n/a")


# ============================================================
# SECTION 7: Main program
# ============================================================
print("Connecting to MetaTrader 5...")

if not mt5.initialize():
    print("ERROR: Could not connect to MetaTrader 5.")
    print("Make sure the MT5 desktop terminal is running and logged into your DEMO account.")
    print("Last error:", mt5.last_error())
    quit()

print("Connected to MetaTrader 5 successfully!")

try:
    if not mt5.symbol_select(SYMBOL, True):
        print(f"ERROR: Symbol {SYMBOL} is not available on this account.")
        print("Last error:", mt5.last_error())
        quit()

    print(f"\nDownloading the latest {NUM_CANDLES} {SYMBOL} candles (H1)...")

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)

    if rates is None or len(rates) == 0:
        print(f"ERROR: Could not retrieve {SYMBOL} data.")
        print("Last error:", mt5.last_error())
        quit()

    if len(rates) < MIN_CANDLES:
        print(f"ERROR: Not enough history. Got {len(rates)} candles, "
              f"but at least {MIN_CANDLES} are needed.")
        quit()

    if len(rates) < NUM_CANDLES:
        print(f"NOTE: Broker returned {len(rates)} candles (requested {NUM_CANDLES}). Continuing...")

    # --- Build the DataFrame; drop the still-forming last candle ---
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)   # closed candles only

    # --- Calculate the indicators ---
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_ema(df, EMA_REGIME_PERIOD)   # "EMA50" for variant B
    df = add_adx(df, ADX_PERIOD)          # "ADX"  for variant C

    # Quick regime overview of the dataset (facts, no trading)
    valid_adx = df["ADX"].dropna()
    print(f"\nDataset regime overview (ADX14, {len(valid_adx)} closed candles):")
    print(f"  candles with ADX < 25 (weak trend / range): "
          f"{(valid_adx < ADX_THRESHOLD).sum()} "
          f"({(valid_adx < ADX_THRESHOLD).mean() * 100:.1f}%)")
    print(f"  candles with ADX >= 25 (strong trend):      "
          f"{(valid_adx >= ADX_THRESHOLD).sum()} "
          f"({(valid_adx >= ADX_THRESHOLD).mean() * 100:.1f}%)")

    # ============================================================
    # SECTION 8: Run all THREE variants on the SAME data
    # ============================================================
    results = {}
    for variant, label in (("A", "A) BASELINE - NO REGIME FILTER"),
                           ("B", "B) EMA50 REGIME FILTER (fade with the trend)"),
                           ("C", "C) ADX14 < 25 REGIME FILTER (fade in ranges)")):
        print(f"\nRunning variant {variant}...")
        trades_df, final_balance = run_backtest(df.copy(), variant)
        results[variant] = (show_results(label, trades_df, final_balance), trades_df)

    metrics = {v: r for v, (r, _) in results.items() if r is not None}
    trades_a = results["A"][1]
    trades_b = results["B"][1]
    trades_c = results["C"][1]

    # ============================================================
    # SECTION 9: Regime diagnostics
    # ============================================================
    show_regime_diagnostics(trades_a, trades_b, trades_c)

    # ============================================================
    # SECTION 10: Comparison - does the regime filter change anything?
    # ============================================================
    if len(metrics) == 3:
        print("\n" + "=" * 64)
        print("REGIME FILTER COMPARISON (same entry + exits, different filters)")
        print("=" * 64)
        print(f"{'Metric':<22}{'A: no filter':>14}{'B: EMA50':>14}{'C: ADX<25':>14}")
        print(f"{'Total return':<22}{metrics['A']['return']:>13.2f}%"
              f"{metrics['B']['return']:>13.2f}%{metrics['C']['return']:>13.2f}%")
        print(f"{'Trades':<22}{metrics['A']['trades']:>14}"
              f"{metrics['B']['trades']:>14}{metrics['C']['trades']:>14}")
        print(f"{'Win rate':<22}{metrics['A']['win_rate']:>13.1f}%"
              f"{metrics['B']['win_rate']:>13.1f}%{metrics['C']['win_rate']:>13.1f}%")
        print(f"{'Profit factor':<22}{metrics['A']['profit_factor']:>14.2f}"
              f"{metrics['B']['profit_factor']:>14.2f}{metrics['C']['profit_factor']:>14.2f}")
        print(f"{'Max drawdown':<22}{metrics['A']['max_dd_pct']:>13.2f}%"
              f"{metrics['B']['max_dd_pct']:>13.2f}%{metrics['C']['max_dd_pct']:>13.2f}%")
        print(f"{'Avg trade':<22}{metrics['A']['avg_trade']:>13.2f}$"
              f"{metrics['B']['avg_trade']:>13.2f}${metrics['C']['avg_trade']:>13.2f}$")
        print(f"{'Avg hold (candles)':<22}{metrics['A']['avg_hold']:>14.1f}"
              f"{metrics['B']['avg_hold']:>14.1f}{metrics['C']['avg_hold']:>14.1f}")

        # Neutral, factual summary (computed, not hand-tuned)
        best_pf = max(metrics, key=lambda v: metrics[v]["profit_factor"])
        print(f"\nHighest profit factor on this data: variant {best_pf} "
              f"(PF {metrics[best_pf]['profit_factor']:.2f})")
        for v in ("B", "C"):
            pf_diff = metrics[v]["profit_factor"] - metrics["A"]["profit_factor"]
            print(f"  Variant {v} vs A: profit factor {pf_diff:+.2f}, "
                  f"trades {metrics[v]['trades']} vs {metrics['A']['trades']} "
                  f"({metrics['A']['trades'] - metrics[v]['trades']} filtered out)")
        print("\nInterpretation guide:")
        print("- If B and/or C raise PF while keeping most of the edge's")
        print("  trade count, the regime hypothesis has support on THIS data.")
        print("- If PF barely moves, the bounce behavior seems similar in")
        print("  both regimes and a regime filter adds nothing here.")
        print("- One 2000-candle sample is small: treat differences as")
        print("  observations, NOT proof.")

    # ============================================================
    # SECTION 11: Disclaimer
    # ============================================================
    print("\nDISCLAIMER: This is a HISTORICAL BACKTEST with virtual money.")
    print("Results do NOT predict future performance, and nothing here")
    print("is proven to be profitable. No parameters were searched or")
    print("optimized (EMA50 and ADX 25 were fixed in advance).")
    print("Educational use only.")

finally:
    # ============================================================
    # SECTION 12: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
