"""Quick check that your keys work. Run: python test_connection.py"""
from trader.client import BinanceTrader

t = BinanceTrader()
print("Mode:", "TESTNET (fake money)" if t.testnet else "LIVE (real money)")
print("Dry-run:", t.dry_run)
print("BTC price:", t.get_price("BTCUSDT"))
try:
    print("Spot balances:", t.spot_balances())
except Exception as e:
    print("Spot balances: (skipped —", e, ")")
try:
    print("Futures balance:", t.futures_balance())
except Exception as e:
    print("Futures balance: (skipped —", e, ")")
print("\nConnection OK ✅")
