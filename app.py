import os, time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from bot_core import (
    SUPPORTED_EXCHANGES, ExchangeManager, generate_signal, suggest_pairs,
    format_signal_for_display, bitnodes_snapshot, send_telegram
)

# Secrets: Streamlit first, then env
def _secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

LOGIN_CODE = _secret("LOGIN_CODE", "Sam0316")
TELE_TOKEN = _secret("TELEGRAM_TOKEN", "")
TELE_CHAT  = _secret("TELEGRAM_CHAT_ID", "")

st.set_page_config(page_title="Trading Bot", layout="wide")
st.markdown("<h2 style='text-align:center'>🤖 Trading Bot — Live Signals</h2>", unsafe_allow_html=True)

# Auth
pwd = st.text_input("Enter access code", type="password")
if pwd != LOGIN_CODE:
    st.warning("Access denied. Enter the correct access code.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
exchange_name = st.sidebar.selectbox("Exchange", SUPPORTED_EXCHANGES, index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=1)
auto = st.sidebar.checkbox("Auto-refresh")
interval = st.sidebar.number_input("Refresh seconds (auto)", min_value=5, max_value=120, value=20, step=5)

ex = ExchangeManager(exchange_name)

# Pair select with suggestions
with st.expander("🔎 Suggest pairs (volume & volatility)"):
    if st.button("Scan & Suggest"):
        with st.spinner("Scanning pairs..."):
            suggestions = suggest_pairs(ex, max_pairs=10, timeframe="5m")
        if suggestions:
            st.success("Suggestions ready. Click to copy:")
            for s in suggestions:
                st.code(s)
        else:
            st.info("No suggestions available (mock mode or rate-limited).")

all_syms = ex.symbols_usdt(400)
pair = st.selectbox("Select pair", options=all_syms, index=0)

cols = st.columns(3)
send_to_telegram = cols[0].checkbox("Send Telegram", value=True)
run = cols[1].button("Get Signal Now")
if auto:
    st.experimental_singleton.clear() if False else None  # noop to satisfy linter
    st.experimental_rerun() if False else None
    st.autorefresh = st.experimental_rerun  # backward compat placeholder

if auto and not run:
    st.experimental_set_query_params(_=str(time.time()))
    st.experimental_rerun() if False else None  # no-op guard

# Main action
if run or auto:
    with st.spinner("Fetching data & analyzing..."):
        try:
            df = ex.fetch_ohlcv(pair, timeframe=timeframe)
            ob = None
            try:
                ob = ex.fetch_orderbook(pair, limit=25)
            except Exception:
                ob = None

            ob_imb = 0.5 if ob is None else (lambda ob_: (sum(v for _,v in ob_["bids"][:20]) /
                                                          max(1e-9, sum(v for _,v in ob_["bids"][:20]) + sum(v for _,v in ob_["asks"][:20]))))(ob)

            sig = generate_signal(df, ob_imb)
            sig["pair"] = pair
            sig["exchange"] = exchange_name

            st.subheader("Signal")
            st.markdown(format_signal_for_display(sig))

            # Small chart
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Candlestick(x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])])
            fig.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Telegram
            if send_to_telegram:
                ok = send_telegram(sig, TELE_TOKEN, TELE_CHAT)
                st.info("Telegram sent ✅" if ok else "Telegram failed ❌ (check secrets)")

            # Optional BTC network context
            if pair.startswith("BTC/"):
                snap = bitnodes_snapshot()
                if snap:
                    st.caption(f"Bitnodes: {snap['nodes']} nodes (as of {datetime.utcfromtimestamp(snap['timestamp']).isoformat()}Z)")

        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.caption("Indicators: RSI, MACD, EMA(21), ATR(14) + orderbook imbalance → confidence score. Use for research only; not financial advice.")

