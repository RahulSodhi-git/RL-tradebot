"""Binance MCP server — the tools your AI agent uses to trade (spot + futures).

Run standalone:   python mcp_server.py
Or connect it to an MCP host (e.g. Claude Desktop) — see README-trader.md.
"""
from mcp.server.fastmcp import FastMCP

from trader.client import BinanceTrader
from trader import guards

mcp = FastMCP("binance-trader")
_t = None


def trader():
    global _t
    if _t is None:
        _t = BinanceTrader()
    return _t


# ---------- READ (safe) ----------
@mcp.tool()
def get_price(symbol: str) -> str:
    """Current market price of a symbol, e.g. BTCUSDT."""
    return f"{symbol.upper()} = {trader().get_price(symbol)} USDT"


@mcp.tool()
def account_balance() -> dict:
    """Show spot and futures balances, and whether we are on TESTNET or LIVE."""
    t = trader()

    def safe(fn):  # one blank key set shouldn't break the whole call
        try:
            return fn()
        except Exception as e:
            return {"unavailable": str(e)}

    return {"mode": "TESTNET (fake money)" if t.testnet else "LIVE (real money)",
            "dry_run": t.dry_run, "spot": safe(t.spot_balances), "futures": safe(t.futures_balance)}


@mcp.tool()
def list_positions() -> list:
    """List open futures positions with entry price and unrealized profit/loss."""
    return trader().futures_positions()


# ---------- TRADE (guarded) ----------
@mcp.tool()
def spot_trade(symbol: str, side: str, usdt_amount: float) -> dict:
    """Place a SPOT market order. side = BUY or SELL. usdt_amount = USDT to spend."""
    guards.check_order(symbol, usdt_amount)
    return trader().spot_market_order(symbol, side, usdt_amount)


@mcp.tool()
def open_futures(symbol: str, side: str, usdt_amount: float, leverage: int = 1) -> dict:
    """Open a FUTURES position. side = BUY (long) or SELL (short). usdt_amount = margin in USDT."""
    guards.check_order(symbol, usdt_amount, leverage)
    return trader().futures_market_order(symbol, side, usdt_amount, leverage)


@mcp.tool()
def close_futures(symbol: str) -> dict:
    """Close an open futures position for a symbol, e.g. BTCUSDT."""
    return trader().close_futures(symbol)


if __name__ == "__main__":
    mcp.run()
