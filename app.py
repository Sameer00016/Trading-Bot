import os
import streamlit as st
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import datetime

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
LOGIN_CODE = os.getenv("LOGIN_CODE")

# Authentication
st.title("Trading Bot Dashboard")
code_input = st.text_input("Enter Access Code", type="password")
if code_input != LOGIN_CODE:
    st.warning("Please enter the correct access code.")
    st.stop()

st.success("Access Granted ✅")

# Simple signal generator (placeholder logic)
def get_market_signal(symbol="BTCUSDT"):
    price = np.random.uniform(20000, 70000)
    trend = np.random.choice(["UP", "DOWN"])
    target_time = datetime.datetime.now() + datetime.timedelta(hours=np.random.randint(1, 12))
    exit_price = price * (1.05 if trend == "UP" else 0.95)
    return {
        "symbol": symbol,
        "signal": trend,
        "current_price": round(price, 2),
        "target_time": target_time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_price": round(exit_price, 2)
    }

st.header("Get Trading Signal")
symbol = st.text_input("Enter Trading Pair (e.g., BTCUSDT)", "BTCUSDT")
if st.button("Get Signal"):
    signal = get_market_signal(symbol)
    st.write(signal)
    # Send to Telegram
    msg = f"📊 Signal for {signal['symbol']}\nTrend: {signal['signal']}\nPrice: {signal['current_price']}\nTarget Time: {signal['target_time']}\nExit Price: {signal['exit_price']}"
    requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}")
    st.success("Signal sent to Telegram ✅")
from bot_core import send_telegram_message, generate_trade_signal

st.title("📊 Trading Bot Dashboard")

if st.button("📤 Send Test Signal"):
    signal = generate_trade_signal()
    send_telegram_message(signal)
    st.success("✅ Signal sent to your Telegram!")
