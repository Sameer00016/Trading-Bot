import os
import time
import streamlit as st
import pandas as pd
from datetime import datetime
from bot_core import ExchangeManager, generate_signal, scan_pairs, make_trade_plan, send_telegram_alert

# Secrets helper
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

# Initialize exchange
exchange_name = st.sidebar.selectbox("Exchange", ["binance", "mexc"], index=0)
ex = ExchangeManager(exchange_name, mock_mode=True)  # Start in mock mode for testing

# Main UI
pair = st.selectbox("Select pair", options=scan_pairs(ex, "USDT", 10))
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)

if st.button("Generate Signal"):
    df = ex.fetch_ohlcv(pair, timeframe)
    df = analyze_market(df)
    p_up = probability_model(df, orderbook_imbalance=0)  # Mock orderbook imbalance
    signal, atr = generate_signal(df, p_up, orderbook_imbalance=0)
    
    st.subheader(f"Signal: {signal}")
    st.write(f"ATR: {atr:.4f}")
    
    if signal != "WAIT":
        plan = make_trade_plan(signal, df["close"].iloc[-1], atr)
        st.json(plan)
    
    if TELE_TOKEN and TELE_CHAT:
        send_telegram_alert(TELE_TOKEN, TELE_CHAT, f"{signal} {pair} at {df['close'].iloc[-1]:.4f}")
