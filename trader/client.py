"""Binance wrapper for spot + futures (testnet or live). One safe place that talks to Binance."""
import time

from binance.client import Client

from . import settings


class BinanceTrader:
    def __init__(self):
        if not settings.SPOT_KEY and not settings.FUT_KEY:
            raise RuntimeError("No API keys found. Fill in your .env file (copy from .env.example).")
        self.testnet = settings.USE_TESTNET
        self.dry_run = settings.DRY_RUN
        # Separate clients so spot & futures testnet keys can differ.
        self.spot = Client(settings.SPOT_KEY, settings.SPOT_SECRET, testnet=self.testnet)
        self.fut = Client(settings.FUT_KEY, settings.FUT_SECRET, testnet=self.testnet)
        self._sync_time()          # align to Binance clock -> avoids APIError(-1021) on drift

    def _sync_time(self):
        """Align each client's timestamp to Binance server time, so signed requests don't fail with
        APIError(-1021) 'Timestamp for this request is outside of the recvWindow' when the PC clock
        drifts. Uses the UNSIGNED server-time endpoints, so it works even while the clock is wrong."""
        for client, getter in ((self.spot, "get_server_time"), (self.fut, "futures_time")):
            try:
                server_ms = getattr(client, getter)()["serverTime"]
                client.timestamp_offset = server_ms - int(time.time() * 1000)
            except Exception:
                pass

    # ---------- READ ----------
    def get_price(self, symbol):
        return float(self.spot.get_symbol_ticker(symbol=symbol.upper())["price"])

    def spot_balances(self):
        acct = self.spot.get_account()
        return {b["asset"]: float(b["free"]) for b in acct["balances"] if float(b["free"]) > 0}

    def futures_balance(self):
        return {b["asset"]: float(b["balance"])
                for b in self.fut.futures_account_balance() if float(b["balance"]) > 0}

    def futures_available(self, asset="USDT"):
        """Free (available) margin for NEW trades — wallet minus margin already locked in open
        positions/orders. This is what a new order can actually use (walletBalance would over-state
        it and trigger -2019 'margin insufficient'). Falls back to total balance if unavailable."""
        try:
            for b in self.fut.futures_account_balance():
                if b["asset"] == asset:
                    return float(b.get("availableBalance") or b["balance"])
        except Exception:
            pass
        return 0.0

    def futures_positions(self):
        return [{"symbol": p["symbol"], "amount": float(p["positionAmt"]),
                 "entry": float(p["entryPrice"]), "pnl": float(p["unRealizedProfit"])}
                for p in self.fut.futures_position_information() if float(p["positionAmt"]) != 0]

    # ---------- TRADE ----------
    def spot_market_order(self, symbol, side, usdt_amount):
        if self.dry_run:
            return {"dry_run": True, "would": f"SPOT {side} {usdt_amount} USDT of {symbol}"}
        return self.spot.create_order(symbol=symbol.upper(), side=side.upper(),
                                      type="MARKET", quoteOrderQty=usdt_amount)

    def _fut_qty(self, symbol, usdt_amount, leverage):
        price = float(self.fut.futures_symbol_ticker(symbol=symbol.upper())["price"])
        prec = 3
        for s in self.fut.futures_exchange_info()["symbols"]:
            if s["symbol"] == symbol.upper():
                prec = s["quantityPrecision"]
                break
        return round((usdt_amount * leverage) / price, prec)

    def _set_isolated(self, symbol):
        """Force ISOLATED margin before opening, so a bad trade can only lose its OWN margin —
        never the whole wallet, as cross margin could. Harmless if already isolated (Binance -4046)."""
        try:
            self.fut.futures_change_margin_type(symbol=symbol.upper(), marginType="ISOLATED")
        except Exception:
            pass

    def futures_market_order(self, symbol, side, usdt_amount, leverage=1):
        """side = BUY (long) or SELL (short)."""
        if self.dry_run:
            return {"dry_run": True,
                    "would": f"FUTURES {side} {usdt_amount} USDT of {symbol} at {leverage}x"}
        self._set_isolated(symbol)                       # isolated margin: one trade can't drain the wallet
        self.fut.futures_change_leverage(symbol=symbol.upper(), leverage=leverage)
        qty = self._fut_qty(symbol, usdt_amount, leverage)
        return self.fut.futures_create_order(symbol=symbol.upper(), side=side.upper(),
                                             type="MARKET", quantity=qty)

    def close_futures(self, symbol):
        for p in self.fut.futures_position_information(symbol=symbol.upper()):
            amt = float(p["positionAmt"])
            if amt != 0:
                if self.dry_run:
                    return {"dry_run": True, "would": f"CLOSE {symbol} ({amt})"}
                side = "SELL" if amt > 0 else "BUY"
                return self.fut.futures_create_order(symbol=symbol.upper(), side=side,
                                                     type="MARKET", quantity=abs(amt), reduceOnly=True)
        return {"info": f"No open futures position for {symbol}."}

    def partial_close(self, symbol, fraction):
        """Market-close a FRACTION (0<f<=1) of the open futures position (reduceOnly).
        Used by special trades to bank 50%/75% profit while letting the rest run."""
        fraction = max(0.0, min(float(fraction), 1.0))
        for p in self.fut.futures_position_information(symbol=symbol.upper()):
            amt = float(p["positionAmt"])
            if amt == 0:
                continue
            if self.dry_run:
                return {"dry_run": True, "would": f"CLOSE {fraction*100:.0f}% of {symbol} ({amt})"}
            side = "SELL" if amt > 0 else "BUY"
            qty = round(abs(amt) * fraction, self._qty_precision(symbol))
            if qty <= 0:
                return {"info": f"{symbol}: fraction too small to close a valid quantity."}
            return self.fut.futures_create_order(symbol=symbol.upper(), side=side,
                                                 type="MARKET", quantity=qty, reduceOnly=True)
        return {"info": f"No open futures position for {symbol}."}

    def _qty_precision(self, symbol):
        for s in self.fut.futures_exchange_info()["symbols"]:
            if s["symbol"] == symbol.upper():
                return s["quantityPrecision"]
        return 3

    # ---------- helpers for the auto-bot ----------
    def futures_price(self, symbol):
        sym = symbol.upper()
        t = self.fut.futures_symbol_ticker(symbol=sym)
        if isinstance(t, list):                                  # some responses come as a list
            t = next((x for x in t if x.get("symbol") == sym), t[0] if t else {})
        if isinstance(t, dict) and "price" in t:
            return float(t["price"])
        m = self.fut.futures_mark_price(symbol=sym)              # fallback: mark price
        if isinstance(m, list):
            m = m[0] if m else {}
        return float(m["markPrice"])

    def futures_prices_all(self):
        """EVERY futures price in ONE request -> {symbol: price}.
        Used by the fast 1-second watcher: 10 open trades still cost a single call."""
        t = self.fut.futures_symbol_ticker()
        out = {}
        for x in (t if isinstance(t, list) else [t]):
            try:
                out[x["symbol"]] = float(x["price"])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _price_precision(self, symbol):
        for s in self.fut.futures_exchange_info()["symbols"]:
            if s["symbol"] == symbol.upper():
                return s["pricePrecision"]
        return 2

    def place_stops(self, symbol, position_side, tp_price, sl_price):
        """Put a REAL stop-loss (+ optional take-profit) on the exchange (fires even if the
        bot is off). Pass tp_price=None to place ONLY the stop-loss (trailing-stop mode).
        position_side: +1 long / -1 short."""
        if self.dry_run:
            return {"dry_run": True}
        close_side = "SELL" if position_side > 0 else "BUY"
        prec = self._price_precision(symbol)
        orders = [("STOP_MARKET", sl_price)]
        if tp_price is not None:
            orders.append(("TAKE_PROFIT_MARKET", tp_price))
        out = {}
        for kind, price in orders:
            try:
                self.fut.futures_create_order(symbol=symbol.upper(), side=close_side, type=kind,
                                              stopPrice=round(price, prec), closePosition=True)
                out[kind] = "placed"
            except Exception as e:
                out[kind] = f"failed: {e}"
        return out

    def cancel_open_orders(self, symbol):
        try:
            self.fut.futures_cancel_all_open_orders(symbol=symbol.upper())
        except Exception:
            pass
