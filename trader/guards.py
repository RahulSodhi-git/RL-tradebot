"""Hard safety limits enforced in code — the AI cannot bypass these."""
import json
import os

_RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "risk_rules.json")


class RiskError(Exception):
    """Raised when an order breaks a safety rule."""


def _rules():
    with open(_RULES_PATH) as f:
        return json.load(f)


def check_order(symbol, usdt_amount, leverage=1):
    """Reject unsafe orders BEFORE they reach Binance. Returns True if allowed."""
    r = _rules()
    if not r.get("trading_enabled", True):
        raise RiskError("Kill switch is ON (trading_enabled=false in risk_rules.json).")
    allowed = r.get("allowed_symbols") or []
    if allowed and symbol.upper() not in [s.upper() for s in allowed]:
        raise RiskError(f"{symbol} is not in allowed_symbols {allowed}.")
    if usdt_amount > r.get("max_usdt_per_trade", 0):
        raise RiskError(f"{usdt_amount} USDT exceeds max_usdt_per_trade "
                        f"({r.get('max_usdt_per_trade')}).")
    if leverage > r.get("max_leverage", 1):
        raise RiskError(f"{leverage}x exceeds max_leverage ({r.get('max_leverage')}x).")
    return True
