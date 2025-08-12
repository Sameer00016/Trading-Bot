import os
import streamlit as st
import requests
from datetime import datetime

# Load secrets
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

def send_telegram_message(message):
    """Send a message to the configured Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    return response.json()

def generate_trade_signal():
    """
    Dummy function for now — replace with your AI logic later.
    Returns a sample BTC signal with time and TP/SL.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"📈 *BTC/USD Signal*\n\nAction: BUY\nEntry: $29,500\nTake Profit: $30,200\nStop Loss: $29,200\nTime: {now}"
