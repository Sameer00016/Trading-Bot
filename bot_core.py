import os
import time
import functools
import pandas as pd
import numpy as np
import ccxt
import talib
import logging
import plotly.graph_objects as go
from io import BytesIO
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ========== LOGGER ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== BACKOFF RETRY DECORATOR ==========
def backoff_retry(max_retries=5, backoff_factor=2):
    """Retry a function with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = 1
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"{func.__name__} failed: {e} (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise
        return wrapper
    return decorator

# ========== EXCHANGE MANAGER ==========
class ExchangeManager:
    def __init__(self, exchange_name, api_key=None, api_secret=None, mock_mode=False):
        self.mock_mode = mock_mode
        if not mock_mode:
            if exchange_name.lower() == "binance":
                self.exchange = ccxt.binance({"apiKey": api_key, "secret": api_secret})
            elif exchange_name.lower() == "mexc":
                self.exchange = ccxt.mexc({"apiKey": api_key, "secret": api_secret})
            else:
                raise ValueError("Unsupported exchange")
            self.exchange.load_markets()
        else:
            self.exchange = None
            logging.info("Mock mode enabled — no real trades will be placed.")

    @backoff_retry()
    def fetch_ohlcv(self, symbol, timeframe="5m", limit=200):
        if self.mock_mode:
            # Return fake OHLCV data
            now = datetime.utcnow()
            times = [now - timedelta(minutes=5*i) for i in range(limit)]
            df = pd.DataFrame({
                "timestamp": times[::-1],
                "open": np.random.rand(limit) * 100,
                "high": np.random.rand(limit) * 100,
                "low": np.random.rand(limit) * 100,
                "close": np.random.rand(limit) * 100,
                "volume": np.random.rand(limit) * 1000
            })
            return df
        else:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df

    @backoff_retry()
    def fetch_orderbook(self, symbol, limit=20):
        if self.mock_mode:
            return {"bids": [[100, 5], [99, 4]], "asks": [[101, 5], [102, 4]]}
        else:
            return self.exchange.fetch_order_book(symbol, limit=limit)

# ========== MARKET ANALYSIS ==========
def fetch_historical_data(exchange_mgr, symbol, timeframe="5m", limit=200):
    return exchange_mgr.fetch_ohlcv(symbol, timeframe, limit)

def get_orderbook_imbalance(exchange_mgr, symbol):
    ob = exchange_mgr.fetch_orderbook(symbol)
    bids = sum([b[1] for b in ob["bids"]])
    asks = sum([a[1] for a in ob["asks"]])
    return (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0

def analyze_market(df):
    close = df["close"].values
    df["rsi"] = talib.RSI(close, timeperiod=14)
    macd, signal, hist = talib.MACD(close, 12, 26, 9)
    df["macd"] = macd - signal
    df["ema"] = talib.EMA(close, timeperiod=50)
    df["atr"] = talib.ATR(df["high"], df["low"], close, timeperiod=14)
    return df

# ========== ADAPTIVE PROBABILITY MODEL ==========
def probability_model(df, orderbook_imbalance):
    # Feature prep
    df = df.dropna()
    if len(df) < 50:
        # fallback to heuristic
        roc = (df["close"].iloc[-1] - df["close"].iloc[-4]) / df["close"].iloc[-4]
        trend = (df["ema"].iloc[-1] - df["ema"].iloc[-4]) / df["ema"].iloc[-4]
        return 0.4 * roc + 0.3 * trend + 0.3 * orderbook_imbalance

    X = df[["rsi", "macd", "ema", "atr"]].values
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)[:-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:-1])

    model = LogisticRegression()
    model.fit(X_scaled, y)
    latest_scaled = scaler.transform([X[-1]])
    p_up = model.predict_proba(latest_scaled)[0][1]
    return p_up

# ========== SIGNAL GENERATION ==========
def generate_signal(df, p_up, orderbook_imbalance):
    rsi = df["rsi"].iloc[-1]
    macd = df["macd"].iloc[-1]
    close = df["close"].iloc[-1]
    ema = df["ema"].iloc[-1]
    atr = df["atr"].iloc[-1]

    if rsi < 40 and macd > 0 and close > ema and p_up > 0.55:
        return "BUY", atr
    elif rsi > 60 and macd < 0 and close < ema and p_up < 0.45:
        return "SELL", atr
    else:
        return "WAIT", atr

# ========== TRADE PLANNING ==========
def make_trade_plan(signal, price, atr, balance=1000, risk_per_trade=0.01):
    risk_amount = balance * risk_per_trade
    if atr == 0:
        atr = price * 0.01
    position_size = risk_amount / atr
    if signal == "BUY":
        sl = price - 1.5 * atr
        tp = price + 3 * atr
    elif signal == "SELL":
        sl = price + 1.5 * atr
        tp = price - 3 * atr
    else:
        sl, tp = None, None
    return {"size": position_size, "sl": sl, "tp": tp}

# ========== PAIR SCANNER ==========
def scan_pairs(exchange_mgr, base="USDT", limit=10):
    if exchange_mgr.mock_mode:
        return [f"MOCK/{base}"] * limit
    markets = exchange_mgr.exchange.load_markets()
    pairs = [m for m in markets if m.endswith("/" + base)]
    ranked = []
    for pair in pairs:
        try:
            df = exchange_mgr.fetch_ohlcv(pair, "1h", limit=50)
            df = pd.DataFrame(df, columns=["timestamp", "open", "high", "low", "close", "volume"])
            volatility = df["close"].std()
            volume = df["volume"].mean()
            ranked.append((pair, volatility * volume))
        except Exception:
            continue
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in ranked[:limit]]

# ========== BACKTESTING ==========
def run_backtest(df):
    df = analyze_market(df)
    balance = 1000
    for i in range(50, len(df)):
        p_up = probability_model(df.iloc[:i], 0)
        signal, atr = generate_signal(df.iloc[:i], p_up, 0)
        if signal != "WAIT":
            plan = make_trade_plan(signal, df["close"].iloc[i], atr, balance)
            balance += (plan["tp"] - df["close"].iloc[i]) if signal == "BUY" else (df["close"].iloc[i] - plan["tp"])
    return balance

# ========== TELEGRAM ALERTS ==========
def send_telegram_alert(token, chat_id, message, df=None):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": message})

    # Optional chart snapshot
    if df is not None:
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"]
        )])
        fig.update_layout(title="Market Snapshot", xaxis_rangeslider_visible=False)
        img_bytes = fig.to_image(format="png", engine="kaleido")
        files = {"photo": BytesIO(img_bytes)}
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        requests.post(url, data={"chat_id": chat_id}, files=files)
