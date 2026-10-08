"""
multi_instrument_validation.py - CROSS-INSTRUMENT VALIDATION of the
already-defined Bollinger + ADX<25 strategy on four MT5 instruments.

DEMO / EDUCATIONAL PROJECT ONLY:
- VIRTUAL MONEY ONLY: this script contains NO order-sending calls at all.
- READ-ONLY: initialize, symbol discovery, symbol_info, copy_rates, shutdown.
- Never connects to a live trading account. No live/demo trade execution,
  no account modification, no position closing, no order placement.

THE QUESTION BEING ASKED:
The EURUSD backtests defined one specific strategy. Does the SAME
behavior show up on OTHER instruments when NOT A SINGLE PARAMETER is
changed? This is validation, NOT optimization: BB(20, 2.0), ADX14 < 25,
ATR exits 2.0/1.5, 0.10 lots, $2 cost and H1 are copied unchanged from
the previous experiments and are identical for every instrument.

THE STRATEGY UNDER TEST (identical on all instruments):
  ENTRY (Bollinger Bands, period 20, 2.0 std dev):
    LONG : a closed candle closes BELOW the lower band  AND ADX14 < 25
    SHORT: a closed candle closes ABOVE the upper band  AND ADX14 < 25
  EXITS (ATR14, fixed at entry from the signal candle):
    LONG : SL = entry - 2.0 x ATR, TP = entry + 1.5 x ATR
    SHORT: SL = entry + 2.0 x ATR, TP = entry - 1.5 x ATR

INSTRUMENTS (exactly these four, in this order):
  GBPUSD, USDJPY, AUDUSD, XAUUSD

SYMBOL HANDLING:
  Brokers use suffixes (GBPUSDm, GBPUSD., XAUUSDm, ...). For each
  requested instrument the script:
    1. tries the exact symbol first,
    2. otherwise searches available symbols for a suffix match
       (prefix + optional separator characters),
    3. prints which actual MT5 symbol was selected,
  and never silently substitutes a different underlying instrument.
  If nothing suitable exists, the instrument is reported as NOT
  AVAILABLE and the remaining instruments are still tested.

COST / PRICE HANDLING (instruments are NOT EURUSD):
  P/L is computed from the MT5 symbol specifications, not from an
  assumed EURUSD pip value:
    value_per_point = trade_tick_value * (point / trade_tick_size)
  - MT5's trade_tick_value is in the ACCOUNT (deposit) currency, so P/L
    comes out in the same currency the balance is denominated in.
  - If tick value/size are missing or zero, the documented FALLBACK is
    tick_value = contract_size * point (quote currency per tick per lot).
    If the quote currency is not USD and no "<QUOTE>USD" symbol exists to
    convert with, P/L stays in the quote currency and every result for
    that instrument is clearly marked NON-USD P/L instead of being hidden.
  - The $2.00 transaction cost stays FIXED per completed trade, exactly
    as in the previous experiments (it is NOT rescaled per instrument).
  - If 0.10 lots is not a valid volume (volume_min / volume_step), the
    nearest valid volume AT OR BELOW 0.10 lots is used and printed. The
    volume is never increased above 0.10 lots and the strategy itself is
    never changed based on performance.

CROSS-INSTRUMENT SUMMARY (predefined research rule, NOT a prediction):
  "Strategy passes basic cross-instrument check" only if at least 3 of
  the 4 AVAILABLE instruments have return > 0 AND PF > 1.00 AND average
  trade > 0. Instruments are never ranked and none is called "best".

DISCLAIMER: This is a historical backtest with virtual money.
Cross-instrument results do not guarantee future performance.
No parameters were optimized. Results may be affected by spread,
slippage, broker contract specifications, market regime and historical
data limitations. Educational use only.
"""

# ============================================================
# SECTION 1: Imports (no new packages needed)
# ============================================================
import MetaTrader5 as mt5  # Read market data + symbol info (READ-ONLY usage)
import pandas as pd        # DataFrame + indicator math
import numpy as np         # np.select() to build the signal column

# ============================================================
# SECTION 2: Settings - fixed in advance, NOT optimized,
# and IDENTICAL for every instrument
# ============================================================
TIMEFRAME = mt5.TIMEFRAME_H1
NUM_CANDLES = 4000    # Requested history per instrument
MIN_CANDLES = 2000    # Minimum closed candles required per instrument

INSTRUMENTS = ["GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]  # exactly these four

# --- Virtual account settings (same as previous experiments) ---
STARTING_BALANCE = 10_000.0
LOTS = 0.10                   # Requested fixed position size (validated
                              # against volume_min/volume_step per symbol)
CONTRACT_SIZE_FALLBACK = 100_000  # 1.00 lot = 100,000 units (fallback only)
COST_PER_TRADE = 2.0          # $2 fixed per completed trade (NOT rescaled)

# --- Exit distances (ATR multiples), same as previous experiments ---
SL_ATR_MULT = 2.0   # Stop loss distance
TP_ATR_MULT = 1.5   # Take profit distance

# --- Indicator settings (copied unchanged from previous experiments) ---
BB_PERIOD = 20      # Bollinger moving average period
BB_NUM_STD = 2.0    # Bollinger standard deviation width
ADX_PERIOD = 14     # Regime filter period (Wilder smoothing)
ADX_THRESHOLD = 25.0  # Regime filter threshold - the ONLY one tested
ATR_PERIOD = 14
WARMUP = 60         # Bars skipped before scanning (previous convention)


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


def add_bollinger(df, period=20, num_std=2.0):
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
# SECTION 4: Symbol discovery (handles broker suffixes safely)
# ============================================================
def discover_symbol(requested):
    """Find the actual MT5 symbol for a requested instrument.

    1. Try the exact name first.
    2. Otherwise match symbols that START WITH the requested name and
       are followed only by separator characters (letters/digits/dots)
       - e.g. GBPUSD -> GBPUSDm, GBPUSD., GBPUSDmicro.
    Returns (actual_symbol_name, note) or (None, reason).
    """
    info = mt5.symbol_info(requested)
    if info is not None:
        return requested, "exact symbol name"

    # Suffix search over all visible symbols
    matches = []
    for sym in mt5.symbols_get():
        name = sym.name
        if name.startswith(requested) and len(name) > len(requested):
            suffix = name[len(requested):]
            # Reasonable suffixes: short alphanumeric/dot tags, not a
            # different underlying instrument glued to the same prefix.
            if len(suffix) <= 10 and suffix.replace(".", "").isalnum():
                matches.append(name)

    if not matches:
        return None, "no exact or suffix match found"

    if len(matches) > 1:
        # Deterministic, not performance-based: shortest name wins,
        # ties broken alphabetically. The choice is PRINTED, never hidden.
        matches.sort(key=lambda s: (len(s), s))

    chosen = matches[0]
    note = f"suffix match (available: {', '.join(matches[:5])})"
    return chosen, note


def get_contract_spec(symbol_name):
    """Read contract specs from MT5 with DOCUMENTED fallbacks.

    Returns a dict:
      contract_size, value_per_point, volume_min, volume_step,
      quote_currency, tick_value_fallback (bool)
    """
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
        # = contract_size * point. (MT5 trade_tick_value would normally
        # be account-currency; this fallback is quote-currency.)
        tick_value = contract_size * point
        print(f"    WARNING: trade_tick_value missing/zero for {symbol_name}; "
              f"using fallback tick_value = contract_size x point = {tick_value:g} "
              f"(documented fallback)")

    # Monetary value of a ONE-UNIT price move per 1.00 lot.
    # tick_value = $ per tick per lot, tick_size = price change per tick,
    # so $ per 1.0 price-unit per lot = tick_value / tick_size.
    # (For GBPUSD: 1.0 / 0.00001 = 100,000 $/unit/lot - matches the
    # 100,000 contract. For XAUUSD: 1.0 / 0.01 = 100 $/unit/lot -
    # 100 oz per lot, so a $1 gold move = $100 per lot.)
    value_per_unit = tick_value / tick_size if tick_size > 0 else 0.0
    value_per_point = value_per_unit * point if point > 0 else 0.0

    quote_currency = getattr(info, "currency_profit", "") or ""

    # If we used the fallback AND the quote currency is not USD, try to
    # convert value_per_point into USD using a <QUOTE>USD symbol bid.
    # (Broker-provided tick values are already account currency.)
    converted = False
    if fallback_used and value_per_unit > 0 and quote_currency.upper() not in ("USD", ""):
        conv_symbol, _ = discover_symbol(f"{quote_currency.upper()}USD")
        if conv_symbol is not None:
            tick = mt5.symbol_info_tick(conv_symbol)
            if tick is not None and tick.bid > 0:
                value_per_unit = value_per_unit / tick.bid
                converted = True
                print(f"    Fallback P/L converted from {quote_currency.upper()} to USD "
                      f"via {conv_symbol} bid")

    return {
        "contract_size": contract_size,
        "value_per_unit": value_per_unit,   # $ per 1.0 price-unit per 1.0 lot
        "value_per_point": value_per_point,  # $ per 1 point per 1.0 lot
        "volume_min": float(info.volume_min),
        "volume_step": float(info.volume_step) if info.volume_step > 0 else 0.01,
        "quote_currency": quote_currency,
        "tick_value_fallback": fallback_used,
        "fallback_converted": converted,
    }


def validate_volume(spec):
    """Return the largest valid volume <= LOTS for this symbol, or None.

    0.10 lots is kept if the broker accepts it. Otherwise the nearest
    valid volume AT OR BELOW 0.10 lots is used - the volume is never
    increased above 0.10 lots.
    """
    vmin = spec["volume_min"]
    vstep = spec["volume_step"]

    if LOTS < vmin:
        return None  # cannot trade at or below 0.10 lots -> skip instrument

    steps_below = int((LOTS - vmin) / vstep + 1e-9)   # tolerate float noise
    volume = round(vmin + steps_below * vstep, 8)
    return min(volume, LOTS)


# ============================================================
# SECTION 5: The backtest engine
# IDENTICAL to the previous experiments. The ADX<25 filter is part of
# the strategy under test (same on every instrument). The only
# instrument-dependent input is the monetary value of one price point,
# which comes from the broker's own symbol specification.
# ============================================================
def run_backtest(df, value_per_unit, volume):
    """Run the virtual backtest on df. Returns (trades DataFrame, final balance)."""

    # --- Bollinger entry conditions (identical everywhere) ---
    long_condition = df["close"] < df["BB_LOWER"]    # Stretched below the band
    short_condition = df["close"] > df["BB_UPPER"]   # Stretched above the band

    # --- Regime filter: ADX14 < 25 (part of the strategy, not tuned) ---
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

            # --- Virtual P/L using the INSTRUMENT's own specs ---
            # value_per_unit = broker tick_value / tick_size = the money
            # made by a 1.0 price-unit move with 1.00 lot, so:
            if signal == "BUY":
                gross_pnl = (exit_price - entry_price) * volume * value_per_unit
            else:
                gross_pnl = (entry_price - exit_price) * volume * value_per_unit

            # $2 fixed cost per completed trade (NOT rescaled)
            net_pnl = gross_pnl - COST_PER_TRADE
            balance += net_pnl

            trades.append(
                {
                    "instrument": df["instrument"].iloc[0],
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
# SECTION 6: Statistics printer (same structure as previous scripts)
# ============================================================
def fmt_pf(pf):
    """Format a profit factor, handling the 'no losing trades' case."""
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def show_results(label, trades_df, final_balance, adx_valid_count, adx_below_count,
                 currency_note=""):
    """Print the full statistics block for one instrument.
    Returns a metrics dict (or None if no trades)."""
    print("\n" + "=" * 64)
    print(f"{label}  (HISTORICAL BACKTEST - VIRTUAL MONEY ONLY)")
    print("=" * 64)
    if currency_note:
        print(currency_note)

    if len(trades_df) == 0:
        print("\nNo virtual trades were generated for this instrument.")
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

    tp_count = int((trades_df["exit_reason"] == "TAKE_PROFIT").sum())
    sl_count = int((trades_df["exit_reason"] == "STOP_LOSS").sum())
    eod_count = int((trades_df["exit_reason"] == "END_OF_DATA").sum())
    adx_pct = adx_below_count / adx_valid_count * 100 if adx_valid_count else 0.0

    print(f"\nStarting balance    : ${STARTING_BALANCE:,.2f}")
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
    if eod_count:
        print(f"END_OF_DATA exits   : {eod_count} (trade still open at data end)")
    print(f"\nADX<25 candles      : {adx_below_count} of {adx_valid_count} "
          f"valid-ADX candles ({adx_pct:.1f}%)")

    # --- Show the last 5 virtual trades for sanity checking ---
    print("\nLast 5 virtual trades:")
    print(trades_df.tail(5).to_string(index=False))

    return {
        "return": total_return,
        "trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_trade": trades_df["pnl"].mean(),
        "max_dd_pct": max_dd_pct,
        "avg_hold": trades_df["hold_candles"].mean(),
        "tp": tp_count,
        "sl": sl_count,
    }


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
    print("\n" + "=" * 64)
    print("CROSS-INSTRUMENT VALIDATION - STRATEGY PARAMETERS ARE FROZEN")
    print("=" * 64)
    print(f"Strategy on EVERY instrument: BB({BB_PERIOD}, {BB_NUM_STD}) entry,")
    print(f"ADX{ADX_PERIOD} < {ADX_THRESHOLD:.0f} filter, SL/TP = {SL_ATR_MULT}/{TP_ATR_MULT} x ATR, "
          f"{LOTS} lots (validated per symbol), ${COST_PER_TRADE:.2f}/trade.")
    print("Nothing below is optimized and nothing is tuned per instrument.")

    results = {}      # instrument -> metrics dict (only completed runs)

    # ============================================================
    # SECTION 8: Per-instrument loop (each fully independent)
    # ============================================================
    for requested in INSTRUMENTS:
        print("\n" + "#" * 64)
        print(f"# INSTRUMENT: {requested}")
        print("#" * 64)

        try:
            # --- 8.1 Symbol discovery -------------------------------
            actual_symbol, note = discover_symbol(requested)
            if actual_symbol is None:
                print(f"STATUS: NOT AVAILABLE - {note}. "
                      f"Continuing with the remaining instruments.")
                continue

            print(f"Requested: {requested}  ->  using MT5 symbol: {actual_symbol}  ({note})")

            if not mt5.symbol_select(actual_symbol, True):
                print(f"STATUS: NOT AVAILABLE - {actual_symbol} could not be enabled "
                      f"in Market Watch. Continuing with the remaining instruments.")
                continue

            # --- 8.2 Contract specs + volume validation -------------
            spec = get_contract_spec(actual_symbol)
            if spec is None:
                print(f"STATUS: NOT AVAILABLE - no symbol_info for {actual_symbol}. "
                      f"Continuing with the remaining instruments.")
                continue

            volume = validate_volume(spec)
            if volume is None:
                print(f"STATUS: SKIPPED - requested {LOTS} lots is below this "
                      f"symbol's minimum volume ({spec['volume_min']}), and volumes "
                      f"above {LOTS} lots are not allowed. Continuing.")
                continue

            print(f"Volume used: {volume} lots (requested {LOTS}; "
                  f"volume_min {spec['volume_min']}, volume_step {spec['volume_step']})")
            print(f"Contract size: {spec['contract_size']:g} | "
                  f"1.0 price-unit move with 1.0 lot = {spec['value_per_unit']:.2f} "
                  f"({spec['value_per_point']:.5f} per point)")

            # --- 8.3 Download history -------------------------------
            print(f"Downloading up to {NUM_CANDLES} candles for {actual_symbol} (H1)...")
            rates = mt5.copy_rates_from_pos(actual_symbol, TIMEFRAME, 0, NUM_CANDLES)

            if rates is None or len(rates) == 0:
                print(f"STATUS: SKIPPED - no historical data returned for "
                      f"{actual_symbol}. Continuing with the remaining instruments.")
                continue

            # Drop the still-forming candle; closed candles only
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.iloc[:-1].reset_index(drop=True)

            if len(df) < MIN_CANDLES:
                print(f"STATUS: SKIPPED - only {len(df)} closed candles available "
                      f"(minimum {MIN_CANDLES}). Continuing with the remaining "
                      f"instruments.")
                continue

            if len(rates) < NUM_CANDLES:
                print(f"NOTE: broker returned {len(rates)} candles "
                      f"(requested {NUM_CANDLES}); using all {len(df)} closed ones.")

            print(f"Closed candles used: {len(df)} "
                  f"(from {df['time'].iloc[0]} to {df['time'].iloc[-1]})")

            # --- 8.4 Indicators (same math for every instrument) ----
            df["instrument"] = requested   # for the trade log
            df = add_atr(df, ATR_PERIOD)
            df = add_bollinger(df, BB_PERIOD, BB_NUM_STD)
            df = add_adx(df, ADX_PERIOD)

            valid_adx = df["ADX"].dropna()
            adx_valid = len(valid_adx)
            adx_below = int((valid_adx < ADX_THRESHOLD).sum())

            # --- 8.5 Backtest (independent $10,000 per instrument) ---
            trades_df, final_balance = run_backtest(df, spec["value_per_unit"], volume)

            currency_note = None
            if spec["tick_value_fallback"] and not spec["fallback_converted"] \
                    and spec["quote_currency"].upper() not in ("USD", ""):
                currency_note = (f"NOTE: P/L for this instrument is expressed in "
                                 f"{spec['quote_currency'].upper()} (quote currency), "
                                 f"NOT USD, because tick-value data was missing and no "
                                 f"{spec['quote_currency'].upper()}USD conversion symbol "
                                 f"was available. Treat its dollar figures accordingly.")

            metrics = show_results(
                f"{requested} (traded as '{actual_symbol}', {volume} lots)",
                trades_df, final_balance, adx_valid, adx_below, currency_note
            )
            if metrics is not None:
                metrics["candles"] = len(df)
                metrics["adx_below"] = adx_below
                metrics["adx_pct"] = adx_below / adx_valid * 100 if adx_valid else 0.0
                metrics["pl_in_usd"] = not (
                    spec["tick_value_fallback"]
                    and not spec["fallback_converted"]
                    and spec["quote_currency"].upper() not in ("USD", "")
                )
                results[requested] = metrics

        except Exception as exc:  # never let one instrument kill the run
            print(f"STATUS: ERROR while processing {requested}: {exc}")
            print("Continuing with the remaining instruments.")

    # ============================================================
    # SECTION 9: Cross-instrument summary
    # ============================================================
    if results:
        print("\n" + "=" * 64)
        print("CROSS-INSTRUMENT SUMMARY (identical strategy, no optimization)")
        print("=" * 64)
        print(f"\n{'Instrument':<11}{'Candles':>8}{'ADX<25%':>9}{'Trades':>8}"
              f"{'WinRate':>9}{'PF':>7}{'AvgTrade':>10}{'Return':>9}{'MaxDD':>8}")
        print("-" * 72)
        for requested, m in results.items():
            print(f"{requested:<11}{m['candles']:>8}{m['adx_pct']:>8.1f}%"
                  f"{m['trades']:>8}{m['win_rate']:>8.1f}%"
                  f"{fmt_pf(m['profit_factor']):>7}{m['avg_trade']:>9.2f}$"
                  f"{m['return']:>8.2f}%{m['max_dd_pct']:>7.2f}%")

        non_usd = [k for k, m in results.items() if not m["pl_in_usd"]]
        if non_usd:
            print(f"\nNOTE: P/L for {', '.join(non_usd)} is NOT in USD "
                  f"(see the instrument sections above).")

        # --- Predefined tallies (facts, no ranking) -------------------
        positive_return = sum(1 for m in results.values() if m["return"] > 0)
        pf_above_one = sum(1 for m in results.values() if m["profit_factor"] > 1.0)
        positive_avg = sum(1 for m in results.values() if m["avg_trade"] > 0)
        all_three = sum(
            1 for m in results.values()
            if m["return"] > 0 and m["profit_factor"] > 1.0 and m["avg_trade"] > 0
        )

        print(f"\nInstruments tested successfully : {len(results)} of {len(INSTRUMENTS)}")
        print(f"Instruments with return > 0     : {positive_return}")
        print(f"Instruments with PF > 1.00      : {pf_above_one}")
        print(f"Instruments with avg trade > 0  : {positive_avg}")
        print(f"Instruments meeting ALL THREE   : {all_three} "
              f"(return > 0 AND PF > 1.00 AND avg trade > 0)")

        # --- Predefined pass rule (NOT a prediction) ------------------
        print("\n" + "-" * 64)
        if all_three >= 3:
            print("Strategy passes basic cross-instrument check")
            print(f"({all_three} of {len(results)} available instruments met all "
                  f"three criteria; rule requires at least 3).")
        else:
            print("Strategy does not pass the basic cross-instrument check.")
            print(f"({all_three} of {len(results)} available instruments met all "
                  f"three criteria; rule requires at least 3.)")
        print("-" * 64)
        print("This is only a predefined research rule, NOT a prediction of")
        print("future profitability. Instruments are deliberately NOT ranked.")
    else:
        print("\nNo instrument could be tested "
              "(all NOT AVAILABLE, skipped, or produced no trades).")

    # ============================================================
    # SECTION 10: Disclaimer (required wording)
    # ============================================================
    print("\nThis is a historical backtest with virtual money. Cross-instrument")
    print("results do not guarantee future performance. No parameters were")
    print("optimized. Results may be affected by spread, slippage, broker")
    print("contract specifications, market regime and historical data")
    print("limitations. Educational use only.")

finally:
    # ============================================================
    # SECTION 11: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
