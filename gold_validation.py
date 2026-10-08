"""
gold_validation.py - Validate the FROZEN Bollinger + ADX<25 strategy on
the XM MetaTrader 5 GOLD symbol.

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: strictly a historical backtest.
- READ-ONLY: reads candles and symbol specifications only.
- This script NEVER places, modifies, or closes any trade and contains
  no order execution, order checking, position, or account functions.

CONTEXT:
The same frozen strategy was already tested on EURUSD, GBPUSD, USDJPY
and AUDUSD. The exact XM terminal symbol for gold has been confirmed as
"GOLD" (NOT "XAUUSD"). This script validates the identical strategy on
that symbol, changing NOTHING:

  STRATEGY (frozen - no parameter below may be changed):
    Instrument  : GOLD (XM MT5), H1 timeframe
    Entry       : Bollinger Bands (period 20, 2.0 std dev)
                  LONG : a closed candle closes BELOW the lower band
                  SHORT: a closed candle closes ABOVE the upper band
    Regime      : only take a signal when ADX14 < 25 (Wilder smoothing)
    Exit        : ATR14, fixed at entry from the signal candle
                  LONG : SL = entry - 2.0 x ATR, TP = entry + 1.5 x ATR
                  SHORT: SL = entry + 2.0 x ATR, TP = entry - 1.5 x ATR
    Execution   : signal on closed candle, enter at NEXT candle open,
                  one position at a time, stop-first if SL and TP share
                  a candle, no future information anywhere.

  BACKTEST (frozen):
    History     : up to 4000 H1 candles, at least 2000 usable required,
                  forming candle dropped, timestamps printed
    Balance     : $10,000 virtual
    Volume      : 0.10 lots REQUESTED - validated against the broker's
                  volume_min / volume_max / volume_step, adjusted DOWN
                  to the nearest valid volume, never up
    Cost        : $2.00 per completed trade (fixed, as in all previous
                  experiments)
    P/L         : from the GOLD symbol's own specifications - the money
                  value of a 1.0 price-unit move per 1.0 lot is
                  tick_value / tick_size (NOT a forex pip assumption).
                  GOLD's profit currency is USD, so results are in USD.

RESEARCH RULES:
  Nothing is optimized and no alternative parameter is tested. No RSI,
  no additional filter, no machine learning, no AI decisions.

DISCLAIMER: This is a historical backtest with virtual money.
Out-of-sample/cross-instrument results do not guarantee future
performance. No parameters were optimized. Educational use only.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read-only usage in this script
import pandas as pd        # DataFrame + indicator math
import numpy as np         # np.select() to build the signal column

# ============================================================
# SECTION 2: Settings - frozen, NOT optimized
# ============================================================
SYMBOL = "GOLD"       # The CONFIRMED exact XM MT5 gold symbol (not XAUUSD)
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 4000    # Requested history
MIN_CANDLES = 2000    # Minimum usable closed candles required

# --- Virtual account settings (same as previous experiments) ---
STARTING_BALANCE = 10_000.0
LOTS_REQUESTED = 0.10         # Requested volume - validated below
COST_PER_TRADE = 2.0          # $2 fixed per completed trade

# --- Exit distances (ATR multiples), frozen ---
SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.5

# --- Indicator settings (frozen, same as all previous experiments) ---
BB_PERIOD = 20
BB_NUM_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0   # The ONLY threshold used
ATR_PERIOD = 14
WARMUP = 60            # Bars skipped before scanning (previous convention)


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
# SECTION 4: Broker specification and volume validation
# (READ-ONLY reads of the GOLD symbol's contract specs)
# ============================================================
def get_gold_spec(symbol_name):
    """Read the GOLD symbol specs from MT5.

    Returns a dict or None if the symbol is unknown.
    value_per_unit = tick_value / tick_size = the account-currency
    value of a 1.0 price-unit move with 1.00 lot.
    """
    info = mt5.symbol_info(symbol_name)
    if info is None:
        return None

    contract_size = float(info.trade_contract_size)
    point = float(info.point)
    tick_size = float(info.trade_tick_size) if info.trade_tick_size > 0 else point
    tick_value = float(info.trade_tick_value)

    fallback_used = tick_value <= 0
    if fallback_used:
        # Documented fallback: quote/profit-currency value of one tick
        # per lot = contract_size * point. For GOLD (profit currency USD)
        # this stays in USD, so the results remain valid dollar figures.
        tick_value = contract_size * point
        print(f"WARNING: trade_tick_value missing/zero for {symbol_name};")
        print(f"         using fallback tick_value = contract_size x point "
              f"= {tick_value:g} (documented fallback)")

    # Money made by a 1.0 price-unit move with 1.00 lot, in the profit
    # currency (USD for GOLD): $ per tick per lot / price-units per tick.
    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0

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
        "currency_profit": info.currency_profit or "n/a",
        "tick_value_fallback": fallback_used,
    }


def validate_volume(spec):
    """Validate LOTS_REQUESTED against the broker's volume rules.

    Returns (volume_used, note). The volume is adjusted DOWN to the
    nearest valid volume at or below 0.10 lots - never up. Returns
    (None, reason) if no valid volume <= 0.10 lots exists.
    """
    vmin = spec["volume_min"]
    vmax = spec["volume_max"]
    vstep = spec["volume_step"]

    if LOTS_REQUESTED < vmin:
        return None, (f"requested {LOTS_REQUESTED} lots is below the symbol's "
                      f"volume_min {vmin}, and volumes above 0.10 lots are "
                      f"not allowed")
    if vmin > vmax:
        return None, "invalid broker volume range (volume_min > volume_max)"

    steps_below = int((LOTS_REQUESTED - vmin) / vstep + 1e-9)  # float tolerance
    volume = round(vmin + steps_below * vstep, 8)

    if volume > vmax:
        return None, "computed volume exceeds volume_max"

    if volume == LOTS_REQUESTED:
        note = "requested volume is exactly valid - used unchanged"
    else:
        note = (f"adjusted down from requested {LOTS_REQUESTED} lots to the "
                f"nearest valid step")
    return volume, note


# ============================================================
# SECTION 5: The backtest engine
# IDENTICAL to the previous experiments (EURUSD OOS validation and
# cross-instrument validation). Only the instrument's own monetary
# point value and validated volume differ.
# ============================================================
def run_backtest(df, value_per_unit, volume):
    """Run the virtual backtest on df. Returns (trades DataFrame, final balance)."""

    # --- Bollinger entry conditions (frozen strategy) ---
    long_condition = df["close"] < df["BB_LOWER"]    # Stretched below the band
    short_condition = df["close"] > df["BB_UPPER"]   # Stretched above the band

    # --- Regime filter: ADX14 < 25 (frozen - the ONLY threshold) ---
    long_condition = long_condition & (df["ADX"] < ADX_THRESHOLD)
    short_condition = short_condition & (df["ADX"] < ADX_THRESHOLD)

    df["signal"] = np.select([long_condition, short_condition], ["BUY", "SELL"], default="HOLD")

    trades = []
    balance = STARTING_BALANCE

    i = WARMUP
    while i < len(df) - 1:          # Need bar i+1 to exist for the entry
        signal = df["signal"].iloc[i]
        atr = df["ATR"].iloc[i]

        # Only enter on a valid signal with a valid ATR value
        if signal in ("BUY", "SELL") and pd.notna(atr):
            entry_bar = i + 1                         # NEXT candle
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
            # we assume the stop filled (conservative, no look-ahead).
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

            # --- Virtual P/L using GOLD's OWN specifications ---
            # value_per_unit = tick_value / tick_size = money made by a
            # 1.0 price-unit move with 1.00 lot (USD for GOLD).
            if signal == "BUY":
                gross_pnl = (exit_price - entry_price) * volume * value_per_unit
            else:
                gross_pnl = (entry_price - exit_price) * volume * value_per_unit

            # $2 fixed cost per completed trade (NOT rescaled)
            net_pnl = gross_pnl - COST_PER_TRADE
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
                    "pnl": round(net_pnl, 2),
                    "balance_after": round(balance, 2),
                }
            )

            # One position at a time: resume the signal search AFTER the exit
            i = exit_index

        i += 1

    return pd.DataFrame(trades), balance


# ============================================================
# SECTION 6: Main program
# ============================================================
print("Connecting to MetaTrader 5...")

if not mt5.initialize():
    print("ERROR: Could not connect to MetaTrader 5.")
    print("Make sure the MT5 desktop terminal is installed, running,")
    print("and logged into your DEMO account.")
    print("Last error:", mt5.last_error())
    quit()

print("Connected to MetaTrader 5 successfully!")

try:
    # ============================================================
    # SECTION 7: Enable the symbol and read its specs (READ-ONLY)
    # ============================================================
    if not mt5.symbol_select(SYMBOL, True):
        print(f"ERROR: Symbol '{SYMBOL}' could not be enabled in Market Watch.")
        print("Check that the symbol name is exactly right for this terminal")
        print("(run find_xau_symbol.py to list the actual gold symbols).")
        print("Last error:", mt5.last_error())
        quit()

    spec = get_gold_spec(SYMBOL)
    if spec is None:
        print(f"ERROR: No symbol information available for '{SYMBOL}'.")
        print("Last error:", mt5.last_error())
        quit()

    print(f"\nMT5 symbol           : {SYMBOL}")
    print(f"Contract size        : {spec['contract_size']:g}")
    print(f"Point                : {spec['point']:g}")
    print(f"Digits               : {spec['digits']}")
    print(f"Tick size            : {spec['tick_size']:g}")
    print(f"Tick value           : {spec['tick_value']:g} "
          f"({spec['currency_profit']})")
    print(f"Value of 1.0 price-unit move, 1.0 lot: "
          f"{spec['value_per_unit']:.2f} {spec['currency_profit']}")
    print(f"Volume min/max/step  : {spec['volume_min']} / "
          f"{spec['volume_max']} / {spec['volume_step']}")

    volume, volume_note = validate_volume(spec)
    if volume is None:
        print(f"\nERROR: Cannot backtest with a valid volume: {volume_note}.")
        quit()

    print(f"Volume used          : {volume} lots ({volume_note})")

    # ============================================================
    # SECTION 8: Download history (closed candles only)
    # ============================================================
    print(f"\nDownloading up to {NUM_CANDLES} candles for {SYMBOL} (H1)...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)

    if rates is None or len(rates) == 0:
        print(f"ERROR: No historical data returned for {SYMBOL}.")
        print("Last error:", mt5.last_error())
        quit()

    # Drop the still-forming candle; closed candles only
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.iloc[:-1].reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        print(f"ERROR: Only {len(df)} closed candles available "
              f"(minimum required: {MIN_CANDLES}).")
        quit()

    print(f"Candles requested    : {NUM_CANDLES}")
    print(f"Candles received     : {len(rates)}")
    print(f"Closed candles used  : {len(df)}")

    # --- Indicators: causal calculations only ---
    df = add_atr(df, ATR_PERIOD)
    df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
    df = add_adx(df, ADX_PERIOD)

    valid_adx = df["ADX"].dropna()
    adx_valid = len(valid_adx)
    adx_below = int((valid_adx < ADX_THRESHOLD).sum())
    adx_pct = adx_below / adx_valid * 100 if adx_valid else 0.0

    # ============================================================
    # SECTION 9: Run the frozen strategy
    # ============================================================
    trades_df, final_balance = run_backtest(df, spec["value_per_unit"], volume)

    # ============================================================
    # SECTION 10: Report
    # ============================================================
    print("\n" + "=" * 64)
    print(f"GOLD VALIDATION REPORT  (HISTORICAL BACKTEST - VIRTUAL MONEY)")
    print("=" * 64)
    print(f"Instrument          : GOLD")
    print(f"MT5 symbol          : {SYMBOL}")
    print(f"Number of candles   : {len(df)}")
    print(f"Date range          : {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    print(f"Volume used         : {volume} lots")
    print(f"Transaction cost    : ${COST_PER_TRADE:.2f} per completed trade")

    if len(trades_df) == 0:
        print("\nNo virtual trades were generated for GOLD "
              "with the frozen strategy on this data.")
    else:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]

        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        win_rate = len(wins) / len(trades_df) * 100
        total_return = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100

        # --- Maximum drawdown: biggest peak-to-trough balance drop ---
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

        tp_count = int((trades_df["exit_reason"] == "TAKE_PROFIT").sum())
        sl_count = int((trades_df["exit_reason"] == "STOP_LOSS").sum())
        eod_count = int((trades_df["exit_reason"] == "END_OF_DATA").sum())

        print(f"Starting balance    : ${STARTING_BALANCE:,.2f}")
        print(f"Ending balance      : ${final_balance:,.2f}")
        print(f"Total return        : {total_return:+.2f}%")
        print(f"\nNumber of trades    : {len(trades_df)}")
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
        print(f"Average holding time: {trades_df['hold_candles'].mean():.1f} candles")
        print(f"\nTake-profit exits   : {tp_count}")
        print(f"Stop-loss exits     : {sl_count}")
        print(f"END_OF_DATA exits   : {eod_count}")
        print(f"\nADX < 25 candles    : {adx_below} of {adx_valid} "
              f"valid-ADX candles ({adx_pct:.1f}%)")

        # --- Last 5 virtual trades with all requested fields ---
        print("\nLast 5 virtual trades:")
        print(trades_df.tail(5).to_string(index=False))

        # --- Brief neutral reading (fixed rule, no optimization) ---
        print("\nPredefined reading rule (fixed in advance):")
        profitable = (total_return > 0
                      and (gross_profit / gross_loss if gross_loss > 0 else float("inf")) > 1.0
                      and trades_df["pnl"].mean() > 0)
        if profitable:
            print("  GOLD result meets: return > 0 AND PF > 1.00 AND avg trade > 0.")
        else:
            print("  GOLD result does NOT meet all three criteria "
                  "(return > 0, PF > 1.00, avg trade > 0).")
        print("  This is a factual check on one historical sample - NOT a")
        print("  prediction and NOT a recommendation.")

    # ============================================================
    # SECTION 11: Disclaimer
    # ============================================================
    print("\nDISCLAIMER: This is a historical backtest with virtual money.")
    print("Results do not guarantee future performance. No parameters")
    print("were optimized or changed: BB(20, 2.0), ADX14 < 25, ATR exits")
    print("2.0/1.5, 0.10-lot request, $2 cost - all frozen as in the")
    print("previous EURUSD/GBPUSD/USDJPY/AUDUSD validations.")
    print("Educational use only.")

finally:
    # ============================================================
    # SECTION 12: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
