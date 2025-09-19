# app.py — Streamlit UI for live signals, multi-exchange, real-time monitoring
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from bot_core import (
    SUPPORTED_EXCHANGES,
    ExchangeManager,
    analyze_and_signal,
    backtest,
    send_telegram,
)

st.set_page_config(page_title="AI Trading Bot — Live", layout="wide")
st.markdown("<h2 style='text-align:center'>🤖 AI Trading Bot — Live Signals</h2>", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def _secret(key, default=None):
    # Streamlit secrets first, then env
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


# ---------------------------
# Authentication (simple)
# ---------------------------
LOGIN_CODE = _secret("LOGIN_CODE", "Sam0316")
pwd = st.text_input("Enter access code", type="password")
if pwd != LOGIN_CODE:
    st.warning("Access denied. Enter the correct access code.")
    st.stop()

# ---------------------------
# Sidebar — Connection & Settings
# ---------------------------
st.sidebar.header("Connection")
exch_name = st.sidebar.selectbox("Exchange", SUPPORTED_EXCHANGES, index=0)

st.sidebar.markdown("**API Credentials (optional, required for live mode):**")
api_key = st.sidebar.text_input("API Key", type="password", value=_secret("API_KEY", ""))
api_secret = st.sidebar.text_input("API Secret", type="password", value=_secret("API_SECRET", ""))
api_password = st.sidebar.text_input("API Password (if required)", type="password", value=_secret("API_PASSWORD", ""))

mock_mode = st.sidebar.checkbox("Mock mode (no live API)", value=(exch_name == "mock" or not (api_key and api_secret)))

st.sidebar.header("Parameters")
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"],
    index=6,
)
limit = st.sidebar.number_input("Candles", min_value=100, max_value=2000, value=500, step=50)

# Auto-refresh (real-time monitoring)
st.sidebar.header("Real-time")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_sec = st.sidebar.slider("Refresh every (sec)", 5, 120, 20)

# Telegram
st.sidebar.header("Telegram")
TELE_TOKEN = st.sidebar.text_input("Bot Token", type="password", value=_secret("TELEGRAM_TOKEN", ""))
TELE_CHAT  = st.sidebar.text_input("Chat ID", value=_secret("TELEGRAM_CHAT_ID", ""))

# ---------------------------
# Exchange init & symbols
# ---------------------------
ex = ExchangeManager(exch_name, api_key=api_key or None, api_secret=api_secret or None, password=api_password or None, mock_mode=mock_mode)

with st.expander("🔎 Symbols"):
    try:
        syms = ex.symbols_usdt(600)
    except Exception:
        syms = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","EUR/USD","GBP/USD"]

    colA, colB = st.columns([3,1])
    default_pair = "BTC/USDT" if "BTC/USDT" in syms else syms[0]
    pair = colA.selectbox("Select pair", options=syms, index=syms.index(default_pair) if default_pair in syms else 0)
    custom = colB.text_input("Or type custom", value="")
    if custom.strip():
        pair = custom.strip().upper()

# Action buttons
cols = st.columns(4)
run_btn = cols[0].button("Get Signal Now")
bt_btn  = cols[1].button("Quick Backtest")
tg_btn  = cols[2].button("Send Last Signal to Telegram")
clear_btn = cols[3].button("Clear Cache")

if clear_btn:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared.")

# ---------------------------
# Main logic (one-shot or loop)
# ---------------------------
def run_once():
    try:
        with st.spinner("Fetching data & analyzing..."):
            df = ex.get_ohlcv(pair, timeframe=timeframe, limit=int(limit))
            ob = ex.get_orderbook(pair, limit=50)

            res = analyze_and_signal(df, orderbook=ob)
            sig_df = res["df"]
            side = res["side"]
            conf = res["confidence"]

            # Signal color logic
            if conf > 70:
                color = "🟢"
            elif conf >= 40:
                color = "🟡"
            else:
                color = "🔴"

            # top summary
            top_cols = st.columns(5)
            top_cols[0].metric("Signal", f"{color} {side}")
            top_cols[1].metric("Price", f"{res['price']:.6f}")
            top_cols[2].metric("Prob ↑", f"{res['prob_up']:.2%}")
            top_cols[3].metric("OB Imbalance", f"{res['orderbook_imbalance']:.2%}")
            top_cols[4].metric("Confidence", f"{conf}")

            # chart
            try:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Candlestick(
                    x=sig_df["timestamp"], open=sig_df["open"], high=sig_df["high"], low=sig_df["low"], close=sig_df["close"]
                )])
                fig.add_scatter(x=sig_df["timestamp"], y=sig_df["EMA_FAST"], mode="lines", name="EMA_FAST")
                fig.add_scatter(x=sig_df["timestamp"], y=sig_df["EMA_SLOW"], mode="lines", name="EMA_SLOW")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Plotly not available; skipping chart.")

            # table
            show_rows = st.slider("Rows to display", 10, 200, 40)
            display_cols = ["timestamp", "close", "EMA_FAST", "EMA_SLOW", "RSI", "MACD_HIST", "BB_PB", "PRICE_SLOPE", "prob_up" if "prob_up" in sig_df.columns else "RSI"]
            st.dataframe(sig_df.tail(show_rows)[display_cols])

            return res
    except Exception as e:
        st.error(f"Failed to get signals: {e}")
        return None

# one-shot actions
last_result = None
if run_btn:
    last_result = run_once()

if bt_btn:
    try:
        with st.spinner("Backtesting..."):
            df_bt = ex.get_ohlcv(pair, timeframe=timeframe, limit=int(limit))
            stats = backtest(df_bt)
            st.subheader("Backtest Summary")
            st.json(stats)
    except Exception as e:
        st.error(f"Backtest failed: {e}")

if tg_btn:
    try:
        res = last_result or run_once()
        if res:
            msg = (f"[{res['time']}] {pair} {timeframe}\n"
                   f"Signal: {res['side']} | ProbUp: {res['prob_up']:.2%} | Conf: {res['confidence']}\n"
                   f"Price: {res['price']:.6f} | OB Imb: {res['orderbook_imbalance']:.2%}")
            ok = send_telegram(msg, TELE_TOKEN, TELE_CHAT)
            st.info("Telegram sent ✅" if ok else "Telegram failed ❌ (check token/chat)")
        else:
            st.warning("No signal to send yet.")
    except Exception as e:
        st.error(f"Telegram send failed: {e}")

# auto-refresh loop (non-blocking pattern)
if auto_refresh:
    st.caption(f"🔁 Auto-refresh enabled — every {refresh_sec}s")
    _ = run_once()
    time.sleep(int(refresh_sec))
    st.experimental_rerun()

st.caption("Signals blend EMA/RSI/MACD/ATR/Bollinger + slopes/acceleration + orderbook imbalance → probability & confidence. Research use only; not financial advice.")
