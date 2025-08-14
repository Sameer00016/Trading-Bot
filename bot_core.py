# bot_core.py
# Pure pandas/numpy indicators + probability model + multi-exchange mock-safe manager + telegram

import time
from functools import wraps
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import requests

# ccxt optional - fall back to mock mode if import fails
try:
    import ccxt  # type: ignore
    _CCXT_OK = True
except Exception:
    ccxt = None  # type: ignore
    _CCXT_OK = False


# ---------------------------
# retry decorator
# ---------------------------
def backoff_retry(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 30.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_retries - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            return fn(*args, **kwargs)
        return wrapped
    return deco


# ---------------------------
# core indicators (pandas/numpy)
# ---------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def bollinger_bands(series: pd.Series, length: int = 20, n_std: float = 2.0):
    mid = series.rolling(length, min_periods=length).mean()
    sd = series.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pb = (series - lower) / (upper - lower)
    return mid, upper, lower, width.fillna(0), pb.clip(0, 1).fillna(0.5)


def slope(series: pd.Series, window: int = 5) -> pd.Series:
    # rolling linear regression slope
    x = np.arange(window)
    def _linfit(y):
        if np.any(np.isnan(y)):
            return np.nan
        vx = x - x.mean()
        vy = y - y.mean()
        denom = (vx * vx).sum()
        if denom == 0:
            return 0.0
        return (vx * vy).sum() / denom
    return series.rolling(window, min_periods=window).apply(_linfit, raw=True)


def acceleration(series: pd.Series, window: int = 5) -> pd.Series:
    return slope(series, window=window).diff()


# ---------------------------
# orderbook helper
# ---------------------------
def orderbook_imbalance(ob: Optional[Dict[str, Any]], depth: int = 20) -> float:
    if not ob or "bids" not in ob or "asks" not in ob:
        return 0.5
    bids = ob["bids"][:depth]
    asks = ob["asks"][:depth]
    bid_vol = sum([v for _, v in bids]) or 1e-9
    ask_vol = sum([v for _, v in asks]) or 1e-9
    return float(bid_vol / (bid_vol + ask_vol))


def _tf_to_freq(tf: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1H", "4h": "4H", "1d": "1D"}
    return mapping.get(tf, "1D")


# ---------------------------
# ExchangeManager (multi-exchange + mock-safe)
# ---------------------------
class ExchangeManager:
    def __init__(self, exchange_name: str, api_key: Optional[str] = None, api_secret: Optional[str] = None, mock_mode: bool = True):
        self.mock_mode = mock_mode or not _CCXT_OK
        self.exchange = None
        if not self.mock_mode:
            try:
                klass = getattr(ccxt, exchange_name.lower(), None)
                if klass is None:
                    raise ValueError(f"Unsupported exchange: {exchange_name}")
                self.exchange = klass({"apiKey": api_key or "", "secret": api_secret or "", "enableRateLimit": True})
                if hasattr(self.exchange, "load_markets"):
                    self.exchange.load_markets()
            except Exception:
                # fallback to mock mode on any error
                self.exchange = None
                self.mock_mode = True

    @backoff_retry(max_retries=4)
    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
        if self.mock_mode or self.exchange is None:
            idx = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=limit, freq=_tf_to_freq(timeframe))
            base = np.cumsum(np.random.normal(0, 0.3, size=limit)) + 100.0
            drift = np.linspace(0, np.random.uniform(-3, 3), limit)
            close = base + drift
            high = close + np.abs(np.random.normal(0.4, 0.2, size=limit))
            low = close - np.abs(np.random.normal(0.4, 0.2, size=limit))
            open_ = close + np.random.normal(0, 0.2, size=limit)
            vol = np.abs(np.random.normal(10, 3, size=limit))
            df = pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
            return df
        data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # ccxt timestamps in ms
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        return df

    @backoff_retry(max_retries=3)
    def get_orderbook(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        if self.mock_mode or self.exchange is None:
            px = 100.0 + np.random.randn() * 0.5
            bids = [[px - i * 0.05, 1.0 + np.random.rand()] for i in range(limit)]
            asks = [[px + i * 0.05, 1.0 + np.random.rand()] for i in range(limit)]
            return {"bids": bids, "asks": asks}
        return self.exchange.fetch_order_book(symbol, limit=limit)


# ---------------------------
# TradingStrategy (probability model using algebra + calculus + orderbook)
# ---------------------------
class TradingStrategy:
    def __init__(self,
                 ema_fast: int = 20,
                 ema_slow: int = 50,
                 rsi_len: int = 14,
                 atr_len: int = 14,
                 bb_len: int = 20,
                 bb_n: float = 2.0,
                 slope_win: int = 5):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_len = rsi_len
        self.atr_len = atr_len
        self.bb_len = bb_len
        self.bb_n = bb_n
        self.slope_win = slope_win

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not df.index.is_monotonic_increasing:
            df = df.sort_values("timestamp")
        df["EMA_FAST"] = ema(df["close"], self.ema_fast)
        df["EMA_SLOW"] = ema(df["close"], self.ema_slow)
        df["RSI"] = rsi(df["close"], self.rsi_len)
        macd_line, macd_sig, macd_hist = macd(df["close"])
        df["MACD"] = macd_line
        df["MACD_SIGNAL"] = macd_sig
        df["MACD_HIST"] = macd_hist
        df["ATR"] = atr(df, self.atr_len)
        bb_mid, bb_up, bb_lo, bb_w, bb_pb = bollinger_bands(df["close"], self.bb_len, self.bb_n)
        df["BB_MID"] = bb_mid
        df["BB_UP"] = bb_up
        df["BB_LO"] = bb_lo
        df["BB_WIDTH"] = bb_w
        df["BB_PB"] = bb_pb
        df["PRICE_SLOPE"] = slope(df["close"], window=self.slope_win)
        df["PRICE_ACCEL"] = acceleration(df["close"], window=self.slope_win)
        df["MACD_SLOPE"] = slope(df["MACD_HIST"], window=self.slope_win)
        df["MACD_ACCEL"] = acceleration(df["MACD_HIST"], window=self.slope_win)
        return df

    def _logistic(self, x: float) -> float:
        x = float(np.clip(x, -12.0, 12.0))
        return 1.0 / (1.0 + np.exp(-x))

    def _prob_model_row(self, row: pd.Series, ob_imb: float) -> float:
        ema_trend = 1.0 if row.get("EMA_FAST", 0.0) > row.get("EMA_SLOW", 0.0) else 0.0
        rsi_dev = (row.get("RSI", 50.0) - 50.0) / 50.0
        bb_pos = (row.get("BB_PB", 0.5) - 0.5) * 2.0
        macd_m = np.tanh(float(row.get("MACD_HIST", 0.0)) * 3.0)
        price_slope = np.tanh(float(row.get("PRICE_SLOPE", 0.0)))
        price_accel = np.tanh(float(row.get("PRICE_ACCEL", 0.0)))
        macd_slope = np.tanh(float(row.get("MACD_SLOPE", 0.0)))
        macd_accel = np.tanh(float(row.get("MACD_ACCEL", 0.0)))
        ob_centered = (float(ob_imb) - 0.5) * 2.0

        z = (
            0.35 * ema_trend +
            0.25 * rsi_dev +
            0.20 * bb_pos +
            0.20 * macd_m +
            0.18 * price_slope +
            0.12 * price_accel +
            0.15 * macd_slope +
            0.10 * macd_accel +
            0.28 * ob_centered
        )

        width_penalty = float(np.clip(row.get("BB_WIDTH", 0.0), 0.0, 2.0))
        z *= (1.0 - 0.12 * width_penalty)

        return self._logistic(z)

    def generate_signals(self, df: pd.DataFrame, orderbook: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        df = self._prep(df)
        ob_imb = orderbook_imbalance(orderbook) if orderbook else 0.5

        probs = []
        for _, row in df.iterrows():
            try:
                p = self._prob_model_row(row, ob_imb)
            except Exception:
                p = 0.5
            probs.append(p)

        df["prob_up"] = np.clip(np.array(probs), 0.0, 1.0)
        df["confidence"] = (np.abs(df["prob_up"] - 0.5) * 200).astype(int).clip(0, 100)
        buy_mask = df["prob_up"] >= 0.58
        sell_mask = df["prob_up"] <= 0.42
        df["signal"] = "HOLD"
        df.loc[buy_mask, "signal"] = "BUY"
        df.loc[sell_mask, "signal"] = "SELL"

        needed = ["EMA_FAST", "EMA_SLOW", "RSI", "MACD_HIST", "BB_MID"]
        early_nan = df[needed].isna().any(axis=1)
        df.loc[early_nan, "signal"] = "HOLD"
        df.loc[early_nan, "prob_up"] = 0.5
        df.loc[early_nan, "confidence"] = 10

        return df


# ---------------------------
# Backtester (light)
# ---------------------------
class Backtester:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"total_trades": 0, "buy_signals": 0, "sell_signals": 0, "pnl_proxy": 0.0}
        sigs = self.strategy.generate_signals(df)
        buys = int((sigs["signal"] == "BUY").sum())
        sells = int((sigs["signal"] == "SELL").sum())
        total = buys + sells

        pnl = 0.0
        closes = sigs["close"].values
        signals = sigs["signal"].values
        for i in range(len(sigs) - 1):
            if signals[i] == "BUY":
                pnl += closes[i + 1] - closes[i]
            elif signals[i] == "SELL":
                pnl += closes[i] - closes[i + 1]

        return {"total_trades": total, "buy_signals": buys, "sell_signals": sells, "pnl_proxy": float(pnl)}


# ---------------------------
# Telegram notifier
# ---------------------------
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def send_message(self, text: str) -> bool:
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            r = requests.post(url, json={"chat_id": self.chat_id, "text": text}, timeout=10)
            return r.status_code == 200
        except Exception:
            return False
