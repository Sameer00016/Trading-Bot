import os
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime
from functools import wraps
import time
import requests

# ------------------------------
# Retry Decorator
# ------------------------------
def backoff_retry(max_retries=5, delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            wait = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator


# ------------------------------
# Exchange Manager
# ------------------------------
class ExchangeManager:
    def __init__(self, exchange_name="binance", mock=True):
        self.mock = mock
        if not mock:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                "apiKey": os.getenv("API_KEY", ""),
                "secret": os.getenv("API_SECRET", ""),
                "enableRateLimit": True,
            })
        else:
            self.exchange = None

    @backoff_retry()
    def fetch_ohlcv(self, symbol="BTC/USDT", timeframe="1h", limit=100):
        if self.mock:
            now = datetime.utcnow()
            dates = pd.date_range(end=now, periods=limit, freq="H")
            prices = np.cumsum(np.random.randn(limit)) + 50000
            df = pd.DataFrame({
                "timestamp": dates,
                "open": prices + np.random.randn(limit),
                "high": prices + np.random.rand(limit) * 100,
                "low": prices - np.random.rand(limit) * 100,
                "close": prices,
                "volume": np.random.rand(limit) * 100,
            })
        else:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    @backoff_retry()
    def fetch_orderbook(self, symbol="BTC/USDT", limit=20):
        if self.mock:
            bids = [[50000 - i, 1 + np.random.rand()] for i in range(limit)]
            asks = [[50000 + i, 1 + np.random.rand()] for i in range(limit)]
            return {"bids": bids, "asks": asks}
        else:
            return self.exchange.fetch_order_book(symbol, limit=limit)


# ------------------------------
# Indicators
# ------------------------------
def calculate_indicators(df: pd.DataFrame):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["RSI14"] = compute_rsi(df["close"], 14)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["ATR14"] = compute_atr(df, 14)
    df["Slope"] = np.gradient(df["close"])  # calculus derivative

    # Bollinger Bands
    df["BB_MID"] = df["close"].rolling(20).mean()
    df["BB_STD"] = df["close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + (2 * df["BB_STD"])
    df["BB_LOWER"] = df["BB_MID"] - (2 * df["BB_STD"])

    # 🧹 Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


# ------------------------------
# Probability + Calculus Orderbook Logic
# ------------------------------
def analyze_orderbook(orderbook):
    bids = np.array(orderbook["bids"])
    asks = np.array(orderbook["asks"])

    bid_volume = bids[:, 1].sum()
    ask_volume = asks[:, 1].sum()
    total = bid_volume + ask_volume

    if total == 0:
        return 0.5, 0.5  # neutral

    prob_up = bid_volume / total
    prob_down = ask_volume / total
    return prob_up, prob_down


# ------------------------------
# Signal Generation
# ------------------------------
def generate_signal(df: pd.DataFrame, orderbook=None):
    latest = df.iloc[-1]

    # TA signals
    ta_buy = latest["EMA20"] > latest["EMA50"] and latest["RSI14"] > 50
    ta_sell = latest["EMA20"] < latest["EMA50"] and latest["RSI14"] < 50

    # Order book probabilities
    if orderbook:
        prob_up, prob_down = analyze_orderbook(orderbook)
    else:
        prob_up, prob_down = 0.5, 0.5

    # Calculus: slope of price
    slope = latest["Slope"]

    # Weighted decision
    score = 0
    if ta_buy: score += 1
    if ta_sell: score -= 1
    score += (prob_up - prob_down) * 2
    score += np.sign(slope) * 0.5

    if score > 1:
        return "BUY"
    elif score < -1:
        return "SELL"
    else:
        return "HOLD"


# ------------------------------
# Backtest
# ------------------------------
def backtest_strategy(df: pd.DataFrame):
    df = calculate_indicators(df.copy())
    signals = df.apply(lambda row: "BUY" if row["EMA20"] > row["EMA50"] and row["RSI14"] > 50
                       else "SELL" if row["EMA20"] < row["EMA50"] and row["RSI14"] < 50
                       else "HOLD", axis=1)
    return signals.value_counts().to_dict()


# ------------------------------
# Telegram Alerts
# ------------------------------
def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, json=payload)
        return True
    except Exception:
        return False
