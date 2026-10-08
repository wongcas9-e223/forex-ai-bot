import MetaTrader5 as mt5

print("Starting MT5 test...")

if not mt5.initialize():
    print("MT5 connection failed")
    print("Error:", mt5.last_error())
    quit()

print("MT5 connection successful!")

symbol = "EURUSD"

info = mt5.symbol_info_tick(symbol)

if info is None:
    print(f"Could not find {symbol}")
else:
    print(f"{symbol}")
    print(f"Bid: {info.bid}")
    print(f"Ask: {info.ask}")

mt5.shutdown()
