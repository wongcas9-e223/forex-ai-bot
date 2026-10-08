"""
get_data.py - Read the latest 100 EURUSD H1 candles from MetaTrader 5.

DEMO / EDUCATIONAL PROJECT ONLY:
- This script only READS market data.
- It never places, modifies, or closes any trade (no mt5.order_send()).
"""

# ============================================================
# SECTION 1: Imports
# ============================================================
import MetaTrader5 as mt5  # Official MetaTrader 5 Python package
import pandas as pd        # For the DataFrame and readable output


# ============================================================
# SECTION 2: Initialize the MetaTrader 5 connection
# ============================================================
print("Connecting to MetaTrader 5...")

if not mt5.initialize():
    # Initialization failed: show a clear message and exit safely.
    print("ERROR: Could not connect to MetaTrader 5.")
    print("Make sure the MT5 desktop terminal is installed and running,")
    print("and that you are logged into your DEMO account.")
    print("Last error:", mt5.last_error())
    quit()  # Nothing was opened, so no shutdown() call is needed here.

print("Connected to MetaTrader 5 successfully!")

SYMBOL = "EURUSD"  # The symbol we want to read

# Make sure the symbol is enabled in Market Watch (this is NOT a trade,
# it only tells MT5 we want data for this symbol).
if not mt5.symbol_select(SYMBOL, True):
    print(f"ERROR: Symbol {SYMBOL} is not available on this account.")
    print("Last error:", mt5.last_error())
    mt5.shutdown()
    quit()

try:
    # ============================================================
    # SECTION 3: Read the latest 100 EURUSD H1 candles
    # ============================================================
    print(f"\nReading the latest 100 {SYMBOL} candles (H1 timeframe)...")

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 100)

    # Handle the case where no data could be retrieved.
    if rates is None or len(rates) == 0:
        print(f"ERROR: Could not retrieve {SYMBOL} data.")
        print("Check that the symbol name is correct and that market data")
        print("is available in your MT5 terminal (Market Watch window).")
        print("Last error:", mt5.last_error())
        quit()

    # ============================================================
    # SECTION 4: Convert the data into a pandas DataFrame
    # ============================================================
    # `rates` is a NumPy structured array; pandas turns it into a table.
    df = pd.DataFrame(rates)

    # ============================================================
    # SECTION 5: Convert the Unix timestamp into a readable datetime
    # ============================================================
    # The "time" column is seconds since 1970-01-01 (Unix time).
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # ============================================================
    # SECTION 6: Display the results
    # ============================================================
    # Show the latest 10 candles (newest candle is the last row).
    print(f"\nLatest 10 {SYMBOL} candles (H1):")
    print(df.tail(10).to_string(index=False))

    # Show the total number of candles retrieved.
    print(f"\nTotal candles retrieved: {len(df)}")

    # Show the column names of the DataFrame.
    print(f"Columns: {list(df.columns)}")

finally:
    # ============================================================
    # SECTION 7: Always disconnect when the script finishes
    # ============================================================
    # `finally` guarantees this runs even if an error happened above.
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
