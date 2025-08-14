import ccxt
import pandas as pd
import numpy as np
import time
import requests
from functools import wraps

# ===== Retry Decorator =====
def backoff_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        delay = 1
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
    return wrapper

# ===== Technical Indicators =====
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, length=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(length).mean()

def bollinger_bands(series, length=20, std_mult=2):
    sma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, sma, lower

# ===== Exchange Manager =====
class ExchangeManager:
    def __init__(self, exchange_name, api_key=None, api_secret=None, mock_mode=True):
        self.mock_mode = mock_mode
        if not mock_mode:
            exchange_class = getattr(ccxt, exchange_name.lower())
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret
            })
        else:
            self.exchange = None

    @backoff_retry
    def get_ohlcv(self, symbol, timeframe="1d", limit=100):
        if self.mock_mode:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=limit)
            data = pd.DataFrame({
                "timestamp": dates,
                "open": np.random.rand(limit) * 100,
                "high": np.random.rand(limit) * 100,
                "low": np.random.rand(limit) * 100,
                "close": np.random.rand(limit) * 100,
                "volume": np.random.rand(limit) * 10
            })
            return data
        else:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df

# ===== Trading Strategy =====
class TradingStrategy:
    def generate_signals(self, df):
        df = df.copy()
        df["EMA20"] = ema(df["close"], 20)
        df["EMA50"] = ema(df["close"], 50)
        df["RSI14"] = rsi(df["close"], 14)
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["close"])
        df["ATR14"] = atr(df, 14)
        df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger_bands(df["close"])

        df["signal"] = np.where(
            (df["EMA20"] > df["EMA50"]) & (df["RSI14"] > 50), "BUY",
            np.where((df["EMA20"] < df["EMA50"]) & (df["RSI14"] < 50), "SELL", "HOLD")
        )
        return df

# ===== Backtester =====
class Backtester:
    def __init__(self, strategy):
        self.strategy = strategy

    def run(self, df):
        signals = self.strategy.generate_signals(df)
        buy_signals = signals[signals["signal"] == "BUY"]
        sell_signals = signals[signals["signal"] == "SELL"]
        return {
            "total_trades": len(buy_signals) + len(sell_signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals)
        }

# ===== Telegram Notifier =====
class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    def send_message(self, text):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        requests.post(url, json=payload)
