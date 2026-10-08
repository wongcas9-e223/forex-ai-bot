"""
find_xau_symbol.py - Find the actual Gold/XAU symbol in the connected
XM MetaTrader 5 terminal.

DEMO / EDUCATIONAL PROJECT ONLY:
- READ-ONLY: this script only READS terminal information.
- It NEVER places, modifies, or closes any trade.
- It contains NO order execution, order checking, position,
  or account-modification functions of any kind.

PURPOSE:
Broker symbol names differ (XAUUSD, XAUUSDm, GOLD#, "Gold Micro", ...).
This script does NOT assume the symbol is called XAUUSD. It lists every
available symbol whose NAME or DESCRIPTION contains "XAU" or "GOLD"
(case-insensitive), prints the full contract specification of each match,
and then highlights which matches look like tradable gold symbols - so
the real symbol name can be copied into the other experiment scripts.

Compatible with Python 3.14 (standard library + MetaTrader5 package).
"""

# ============================================================
# SECTION 1: Imports
# ============================================================
import re                   # word-boundary matching for "GOLD"
import MetaTrader5 as mt5   # Read-only usage in this script


# ============================================================
# SECTION 2: Safe attribute reader
# (some brokers/fields can be missing - never crash on that)
# ============================================================
def field(obj, attr, default="n/a"):
    value = getattr(obj, attr, default)
    if value is None or value == "":
        return default
    return value


# ============================================================
# SECTION 3: Connect to the terminal
# ============================================================
print("Connecting to MetaTrader 5...")

if not mt5.initialize():
    # Nothing was opened, so no shutdown() is needed here
    # (same convention as the other scripts in this project).
    print("ERROR: Could not connect to MetaTrader 5.")
    print("Make sure the MT5 desktop terminal is installed, running,")
    print("and logged into your DEMO account.")
    print("Last error:", mt5.last_error())
    quit()

print("Connected to MetaTrader 5 successfully!")

try:
    # ============================================================
    # SECTION 4: Read all available symbols (READ-ONLY)
    # ============================================================
    symbols = mt5.symbols_get()

    if symbols is None:
        print("ERROR: mt5.symbols_get() returned None - no symbol list available.")
        print("Last error:", mt5.last_error())
        quit()

    print(f"Terminal reports {len(symbols)} available symbols. Scanning...")

    # ============================================================
    # SECTION 5: Case-insensitive search: name OR description
    # contains "XAU" or "GOLD"
    # ============================================================
    matches = []
    for sym in symbols:
        name = (sym.name or "").lower()
        description = (sym.description or "").lower()
        if "xau" in name or "xau" in description or "gold" in name or "gold" in description:
            matches.append(sym)

    # ============================================================
    # SECTION 6: Print the matches with full specifications
    # ============================================================
    if not matches:
        print("\nNO XAU/GOLD SYMBOLS FOUND.")
        print("Your XM terminal does not expose any symbol whose name or")
        print("description contains 'XAU' or 'GOLD'. Check that gold is")
        print("enabled for this account type (Market Watch -> right click")
        print("-> 'Show all'), or ask XM support which symbol covers gold.")
    else:
        print(f"\nFound {len(matches)} matching symbol(s):\n")
        for sym in matches:
            print("-" * 64)
            print(f"Symbol name        : {field(sym, 'name')}")
            print(f"Description        : {field(sym, 'description')}")
            print(f"Path               : {field(sym, 'path')}")
            print(f"Contract size      : {field(sym, 'trade_contract_size')}")
            print(f"Point              : {field(sym, 'point')}")
            print(f"Digits             : {field(sym, 'digits')}")
            print(f"Tick size          : {field(sym, 'trade_tick_size')}")
            print(f"Tick value         : {field(sym, 'trade_tick_value')}")
            print(f"Volume minimum     : {field(sym, 'volume_min')}")
            print(f"Volume maximum     : {field(sym, 'volume_max')}")
            print(f"Volume step        : {field(sym, 'volume_step')}")
            print(f"Currency base      : {field(sym, 'currency_base')}")
            print(f"Currency profit    : {field(sym, 'currency_profit')}")
            print(f"Currency margin    : {field(sym, 'currency_margin')}")
            print(f"Visible            : {field(sym, 'visible')}")
            print(f"Selected           : {field(sym, 'select')}")

        # ============================================================
        # SECTION 7: Identify the LIKELY gold trading symbols
        # (identification only - nothing is assumed or auto-selected)
        # ============================================================
        likely = []
        for sym in matches:
            name = (sym.name or "").upper()
            description = (sym.description or "").upper()
            base_ccy = (sym.currency_base or "").upper()
            description_words = set(re.split(r"[^A-Z]+", description))
            # Likely a tradable gold symbol when: the name starts with
            # XAU (XAUUSD, XAUUSDm, XAUEUR...), or starts with GOLD,
            # or the base currency is XAU, or the description contains
            # the standalone word GOLD.
            if (name.startswith("XAU")
                    or name.startswith("GOLD")
                    or base_ccy == "XAU"
                    or "GOLD" in description_words):
                likely.append(sym.name)

        print("-" * 64)
        if likely:
            print("\nLIKELY GOLD/XAU TRADING SYMBOLS (actual terminal names):")
            for n in likely:
                print(f"  -> {n}")
            print("\nUse one of these EXACT names (e.g. as SYMBOL in the")
            print("experiment scripts). Do not rename or shorten them.")
        else:
            print("\nNo match looks like a direct gold trading symbol")
            print("(they may be indices/funds that merely mention gold).")

finally:
    # ============================================================
    # SECTION 8: Always disconnect when the script finishes
    # ============================================================
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
