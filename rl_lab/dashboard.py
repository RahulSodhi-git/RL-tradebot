"""Local, read-only Binance demo monitoring dashboard."""
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import ppo_crypto as lab

st.set_page_config(page_title="Tradebot", page_icon="📊", layout="wide")
st.title("Tradebot — Binance demo")
st.caption("One position • 10% allocation target • 1× leverage • Test funds only")
st.info("Monitoring dashboard. Automatic trading and exits are not enabled or verified yet.")

with st.sidebar:
    st.header("Connection")
    host = st.selectbox("Binance environment", [
        "https://demo-fapi.binance.com", "https://testnet.binancefuture.com"
    ])
    refresh = st.selectbox("Refresh every", [30, 60, 120], index=1)
    monitor = st.toggle("Monitor Binance account", value=False)
    st.caption("Keys are read from your laptop's environment variables. Never paste them into chat.")
    st.caption("Leave the terminal running and prevent laptop sleep. This page does not run a trading engine.")


def report(name):
    path = lab.REPORT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


@st.fragment(run_every=f"{refresh}s")
def live_account():
    st.subheader("Account and open trades")
    if not monitor:
        st.write("Account monitoring is paused. Enable it in the sidebar.")
        return
    if not all(os.environ.get(k) for k in ["BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_SECRET"]):
        st.warning("Connection not configured: set both testnet environment variables, then restart Streamlit.")
        return
    try:
        balances = lab.binance_api_request(host, "/fapi/v2/balance", signed=True)
        positions = lab.binance_api_request(host, "/fapi/v2/positionRisk", signed=True)
        orders = lab.binance_api_request(host, "/fapi/v1/openOrders", signed=True)
        balance = next(b for b in balances if b["asset"] == "USDT")
        opened = [p for p in positions if float(p["positionAmt"]) != 0]
        a, b, c, d = st.columns(4)
        a.metric("Wallet balance (USDT)", f"{float(balance['balance']):,.2f}")
        b.metric("Available (USDT)", f"{float(balance['availableBalance']):,.2f}")
        c.metric("10% of wallet", f"{float(balance['balance']) * .1:,.2f}")
        d.metric("Open positions", len(opened))
        st.success("Connected — last successful refresh: " + datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if len(opened) > 1:
            st.error("More than one account position is open. The one-position target is exceeded.")
        if opened:
            rows = [{
                "Coin": p["symbol"], "Direction": "Long" if float(p["positionAmt"]) > 0 else "Short",
                "Quantity": abs(float(p["positionAmt"])), "Entry price": float(p["entryPrice"]),
                "Mark price": float(p["markPrice"]), "Unrealized P&L (USDT)": float(p["unRealizedProfit"]),
                "Leverage": p.get("leverage", "Unknown"),
            } for p in opened]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.write("No open positions reported by Binance.")
        st.write(f"Pending exchange orders: {len(orders)}")
        if orders:
            st.dataframe(pd.DataFrame(orders)[["symbol", "side", "type", "status", "origQty"]], hide_index=True)
        st.caption("Account-wide positions are shown, including any opened manually.")
    except Exception:
        st.error("Account refresh failed. No current balance or position status can be confirmed. Check network access, key permissions, selected environment and laptop clock.")


live_account()

@st.fragment(run_every="60s")
def reports():
    scan_tab, results_tab, trades_tab, setup_tab = st.tabs([
        "Coins and signals", "Model results", "Simulated trades", "Setup / remaining work"
    ])
    with scan_tab:
        scan = report("latest_testnet_scan.csv")
        if scan.empty:
            st.write("No live scan is available yet. Historical model results are not live trade recommendations.")
        else:
            path = lab.REPORT_DIR / "latest_testnet_scan.csv"
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            st.caption(f"Saved scan: {modified.isoformat(timespec='seconds')} UTC")
            if (datetime.now(timezone.utc) - modified).total_seconds() > 3600:
                st.warning("This scan is older than one hour. Treat its signals as stale.")
            st.dataframe(scan, hide_index=True, use_container_width=True)
            st.caption("Confidence is an action probability, not a probability of profit. A signal is not an open trade.")
    with results_tab:
        summary = report("universal_summary_1h.csv")
        st.metric("Specialist models saved", len(list(lab.MODEL_DIR.glob('*_1h_specialist.zip'))))
        if summary.empty:
            st.write("Evaluation report is not ready yet.")
        else:
            a, b, c = st.columns(3)
            a.metric("Coins evaluated", len(summary))
            b.metric("Positive simulations", int((summary.return_pct > 0).sum()))
            c.metric("Average simulated return", f"{summary.return_pct.mean():.2f}%")
            st.caption("Separate $5,000 simulations per coin. These numbers are not the return of one shared wallet. The evaluation period has been used for model selection.")
            st.dataframe(summary, hide_index=True, use_container_width=True)
            st.download_button("Download results CSV", summary.to_csv(index=False), "model_results.csv")
    with trades_tab:
        trades = report("universal_trades_1h.csv")
        if trades.empty:
            st.write("Trade report is not available yet.")
        else:
            coin = st.selectbox("Filter coin", ["All", *sorted(trades.symbol.unique())])
            filtered = trades if coin == "All" else trades[trades.symbol == coin]
            st.caption("Historical simulated entries and exits, not Binance fills. Net P&L includes the simulator's trading fees.")
            st.dataframe(filtered.tail(1000), hide_index=True, use_container_width=True)
            st.download_button("Download trade CSV", filtered.to_csv(index=False), "simulated_trades.csv")
    with setup_tab:
        st.markdown("""
Your part:
1. Create a Binance Futures demo/testnet account and obtain test API credentials.
2. Add test USDT and enable futures API access for the selected demo environment.
3. Set the two environment variables locally as described in DASHBOARD.md.
4. Start this dashboard, enable monitoring, and keep the laptop awake.

Still required before automatic trading:
- Complete specialist training and connect validated specialists to selection.
- Verify closed-candle features, shared-wallet position accounting and exchange order handling.
- Implement and test the persistent hourly trading worker and automatic exit management.
- Verify connectivity and test order fills from your laptop.

The dashboard can monitor continuously while Streamlit runs. It currently sends no orders.
""")


reports()
