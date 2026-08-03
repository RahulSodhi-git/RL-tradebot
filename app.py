"""Crypto Strategy Analyzer — backtest long/short + leverage strategies, ranked & filterable."""
import numpy as np
import pandas as pd
import streamlit as st

from data import get_ohlcv, get_symbols
from strategies import STRATEGIES
from backtest import run

st.set_page_config(page_title="RL - Trade bot", page_icon="📊", layout="wide")

st.title("📊 RL - Trade bot")
st.caption("Backtest 28 long/short + leverage strategies on real Binance data — returns are NET (after ~0.05%/side Binance fees). "
           "See the 🤖 Auto Bot page to trade automatically.")


@st.cache_data(ttl=3600)
def load_coins():
    try:
        return get_symbols()
    except Exception:
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT"]


coins = load_coins()

# ---- Controls ----
c1, c2, c3, c4 = st.columns(4)
symbol = c1.selectbox("Coin", coins, index=coins.index("BTCUSDT") if "BTCUSDT" in coins else 0)
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
interval = c2.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1d"))
limit = c3.slider("History (candles)", 100, 1000, 500, step=100)
leverage = c4.select_slider("Leverage", options=[1, 2, 3, 5, 10, 20], value=1)

DIR = {1: "🟢 Long", -1: "🔴 Short", 0: "⚪ Flat"}

if st.button("▶ Run Backtest", type="primary"):
    try:
        with st.spinner("Fetching data and testing strategies…"):
            df = get_ohlcv(symbol, interval, limit)
            months = max((df["time"].iloc[-1] - df["time"].iloc[0]).days / 30, 0.1)
            rows, equities = [], {}
            for name, (fn, level) in STRATEGIES.items():
                pos = np.asarray(fn(df), float)
                res = run(df, pos, leverage=leverage)
                equities[name] = res["equity"].values
                n = res["trades"]
                rows.append({
                    "Strategy": name,
                    "Level": level,
                    "Now": DIR[int(pos[-1])],
                    "Trades": n,
                    "Trades/Month": round(n / months, 1),
                    "Reliable?": "✅ Good" if n >= 20 else ("🟡 OK" if n >= 10 else "🔴 Too few"),
                    "Profit Chance": res["win_rate"],
                    "Total Return %": res["return_pct"],
                    "Fees %": res["fees_pct"],
                    "Max Drawdown %": res["max_dd"],
                    "Liq?": "⚠️" if res["liquidated"] else "",
                })
        full = pd.DataFrame(rows)

        # ---- Backtest period ----
        start, end = df["time"].iloc[0], df["time"].iloc[-1]
        days = (end - start).days
        st.info(f"📅 **Backtest period:** {start:%d %b %Y} → {end:%d %b %Y}  ·  "
                f"{days} days (~{days/30:.1f} months, {days/365:.1f} years)  ·  {len(df)} candles")

        # ---- Price chart ----
        st.subheader(f"{symbol} · {interval} price")
        st.line_chart(df.set_index("time")["close"], height=220)

        # ---- Winner (best RELIABLE strategy by return) ----
        rel = full[full["Trades"] >= 10]
        best = (rel if not rel.empty else full).sort_values("Total Return %", ascending=False).iloc[0]
        st.subheader("🏆 Best trustworthy strategy (10+ trades, by return)")
        m = st.columns(6)
        m[0].metric("Strategy", best["Strategy"])
        m[1].metric("Signal Now", best["Now"])
        m[2].metric("Trades", f"{best['Trades']}")
        m[3].metric("Profit Chance", f"{best['Profit Chance']:.0f}%")
        m[4].metric("Total Return", f"{best['Total Return %']:.1f}%")
        m[5].metric("Max Drawdown", f"{best['Max Drawdown %']:.1f}%")

        # ---- Filter / sort controls ----
        st.subheader("All strategies ranked")
        f1, f2, f3 = st.columns([2, 2, 3])
        sort_by = f1.selectbox("Sort by", ["Total Return %", "Profit Chance", "Trades",
                                           "Trades/Month", "Max Drawdown %"])
        show = f2.selectbox("Show", ["All", "Top 10", "Top 20"])
        picked_levels = f3.multiselect("Level", ["Basic", "Medium", "Advanced"],
                                       default=["Basic", "Medium", "Advanced"])

        view = full[full["Level"].isin(picked_levels)].sort_values(sort_by, ascending=False)
        if show == "Top 10":
            view = view.head(10)
        elif show == "Top 20":
            view = view.head(20)
        view = view.reset_index(drop=True)

        def color_pnl(col):  # green profit / red loss, no matplotlib needed
            return [f"color: {'#178a3a' if v > 0 else '#c0392b'}; font-weight: 600" for v in col]

        st.dataframe(
            view.style.format({
                "Profit Chance": "{:.0f}%",
                "Total Return %": "{:.1f}%",
                "Max Drawdown %": "{:.1f}%",
            }).apply(color_pnl, subset=["Total Return %"]),
            use_container_width=True, hide_index=True,
        )

        # ---- Equity curves (only the strategies shown above) ----
        st.subheader(f"Growth of $1  ·  {leverage}x leverage  ·  {len(view)} strategies shown")
        st.line_chart(pd.DataFrame({n: equities[n] for n in view["Strategy"]}, index=df["time"]), height=280)

        if leverage > 1:
            st.warning(f"⚠️ At {leverage}x leverage, a ~{100/leverage:.0f}% adverse move wipes the position. "
                       "Rows marked ⚠️ hit liquidation (equity → 0).")
        st.info("Backtest = past performance, not a future guarantee. This tool only analyzes; it does not place trades.")
    except Exception as e:
        st.error(f"Could not run: {e}")
else:
    st.write("Pick a coin, timeframe, leverage — then press **Run Backtest**.")
