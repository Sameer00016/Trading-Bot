# app.py - Streamlit UI for the probabilistic trading bot
import os
import time
import streamlit as st
import pandas as pd
from datetime import datetime

from bot_core import ExchangeManager, TradingStrategy, Backtester, TelegramNotifier

# page config
st.set_page_config(page_title="Trading Bot", layout="wide")
st.markdown("<h2 style='text-align:center'>🤖 Trading Bot — Probabilistic Signals</h2>", unsafe_allow_html=True)

# secrets helper
def _secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# auth (optional)
LOGIN_CODE = _secret("LOGIN_CODE", "Sam0316")
pwd = st.text_input("Enter access code", type="password")
if pwd != LOGIN_CODE:
    st.warning("Access denied. Enter the correct access code.")
    st.stop()

# sidebar controls
st.sidebar.header("Controls")
exchange_name = st.sidebar.selectbox("Exchange", ["binance", "mexc"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
mock_mode = st.sidebar.checkbox("Mock Mode (no API required)", value=True)
limit = st.sidebar.number_input("OHLCV candles", min_value=100, max_value=2000, value=300, step=50)

# API keys (optional)
api_key = st.sidebar.text_input("API Key", type="password")
api_secret = st.sidebar.text_input("API Secret", type="password")

# Orderbook toggle
use_orderbook = st.sidebar.checkbox("Use Order Book Imbalance", value=True)

# Telegram
TELE_TOKEN = st.sidebar.text_input("Telegram Bot Token", type="password")
TELE_CHAT  = st.sidebar.text_input("Telegram Chat ID")

# Pair input
pair = st.text_input("Trading pair (e.g. BTC/USDT)", value="BTC/USDT")

# Risk & display
st.sidebar.subheader("Display / Risk")
show_recent = st.sidebar.slider("Show recent rows", 5, 200, 30)

# Buttons
cols = st.columns(3)
run_btn = cols[0].button("Get Signals")
backtest_btn = cols[1].button("Quick Backtest")
send_telegram_btn = cols[2].button("Send Last Signal to Telegram")

# Initialize exchange manager and strategy
ex = ExchangeManager(exchange_name, api_key=api_key or None, api_secret=api_secret or None, mock_mode=mock_mode)
strat = TradingStrategy()
bt = Backtester(strat)
tg = TelegramNotifier(TELE_TOKEN, TELE_CHAT) if TELE_TOKEN and TELE_CHAT else None

# Action: get signals
if run_btn:
    with st.spinner("Fetching data & computing signals..."):
        try:
            df = ex.get_ohlcv(pair, timeframe=timeframe, limit=int(limit))
            ob = ex.get_orderbook(pair, limit=50) if use_orderbook else None

            sigs = strat.generate_signals(df, orderbook=ob)
            st.subheader("Recent Signals")
            display_cols = ["timestamp", "close", "EMA_FAST", "EMA_SLOW", "RSI", "MACD_HIST", "BB_PB", "prob_up", "confidence", "signal"]
            # ensure timestamp column is present
            if "timestamp" not in sigs.columns and hasattr(sigs.index, "to_series"):
                sigs = sigs.reset_index().rename(columns={"index": "timestamp"})
            st.dataframe(sigs.tail(show_recent)[display_cols])

            # show simple candlestick chart with plotly
            try:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Candlestick(
                    x=sigs["timestamp"], open=sigs["open"], high=sigs["high"], low=sigs["low"], close=sigs["close"]
                )])
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Failed to get signals: {e}")

# Action: backtest
if backtest_btn:
    with st.spinner("Running quick backtest..."):
        try:
            df = ex.get_ohlcv(pair, timeframe=timeframe, limit=int(limit))
            res = bt.run(df)
            st.subheader("Backtest Summary")
            st.json(res)
        except Exception as e:
            st.error(f"Backtest failed: {e}")

# Action: send last signal to telegram
if send_telegram_btn:
    try:
        df = ex.get_ohlcv(pair, timeframe=timeframe, limit=int(limit))
        ob = ex.get_orderbook(pair, limit=50) if use_orderbook else None
        sigs = strat.generate_signals(df, orderbook=ob)
        last = sigs.iloc[-1]
        text = (f"[{datetime.utcnow().isoformat()}Z] {pair} {timeframe}\n"
                f"Signal: {last['signal']} | prob_up: {last['prob_up']:.2f} | conf: {int(last['confidence'])}\n"
                f"Price: {float(last['close']):.6f}")
        if tg:
            ok = tg.send_message(text)
            st.info("Telegram sent ✅" if ok else "Telegram failed ❌")
        else:
            st.info("Telegram not configured (fill token/chat in sidebar)")
    except Exception as e:
        st.error(f"Telegram send failed: {e}")

# Footer
st.caption("Probabilistic signals use EMA/RSI/MACD/ATR/Bollinger + slope/acceleration + orderbook imbalance. Use for research only; not financial advice.")
