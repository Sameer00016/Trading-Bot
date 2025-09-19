# bot_core.py — multi-exchange core + probability/calculus model (pure pandas/numpy)
import os
import time
from typing import Optional, Dict, Any, List
from functools import wraps

import numpy as np
import pandas as pd
import requests

# ccxt optional
try:
    import ccxt  # type: ignore
    _CCXT_OK = True
except Exception:
    ccxt = None  # type: ignore
    _CCXT_OK = False

SUPPORTED_EXCHANGES = ["binance", "mexc", "bybit", "okx", "kraken", "mock"]
DEFAULT_LIMIT = 500
SAFE_TIMEOUT = 20000


# ---------------------------
# Utils
# ---------------------------
def _secret_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)

def _now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def backoff_retry(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 12.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt >= max_retries - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
        return wrapped
    return deco


def _tf_to_freq(tf: str) -> str:
    mapper = {
        "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D"
    }
    return mapper.get(tf, "1H")


# ---------------------------
# Indicators
# ---------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    f = ema(series, fast)
    s = ema(series, slow)
    line = f - s
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

def bollinger(series: pd.Series, length: int = 20, n: float = 2.0):
    mid = series.rolling(length, min_periods=length).mean()
    sd = series.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + n * sd
    lower = mid - n * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pb = (series - lower) / (upper - lower)
    return mid, upper, lower, width.fillna(0), pb.clip(0, 1).fillna(0.5)

def slope(series: pd.Series, win: int = 5) -> pd.Series:
    x = np.arange(win)
    def _fit(y):
        if np.any(np.isnan(y)): return np.nan
        vx = x - x.mean(); vy = y - y.mean()
        den = (vx * vx).sum()
        if den == 0: return 0.0
        return float((vx * vy).sum() / den)
    return series.rolling(win, min_periods=win).apply(_fit, raw=True)

def acceleration(series: pd.Series, win: int = 5) -> pd.Series:
    return slope(series, win).diff()


# ---------------------------
# Orderbook analysis
# ---------------------------
def orderbook_imbalance(ob: Optional[Dict[str, Any]], depth: int = 20) -> float:
    if not ob or "bids" not in ob or "asks" not in ob:
        return 0.5
    bids = ob["bids"][:depth]
    asks = ob["asks"][:depth]
    bid_vol = sum(_safe_float(v, 0.0) for _, v in bids) or 1e-9
    ask_vol = sum(_safe_float(v, 0.0) for _, v in asks) or 1e-9
    return float(bid_vol / (bid_vol + ask_vol))


# ---------------------------
# Exchange Manager
# ---------------------------
class ExchangeManager:
    def __init__(self, exchange_name: str = "binance", api_key: Optional[str] = None,
                 api_secret: Optional[str] = None, password: Optional[str] = None,
                 mock_mode: Optional[bool] = None):
        name = (exchange_name or "mock").lower()
        self.name = name
        self.mock_mode = (mock_mode if mock_mode is not None else (name == "mock" or not _CCXT_OK))
        self.ex = None

        if not self.mock_mode:
            try:
                klass = getattr(ccxt, name)
                config = {
                    "apiKey": api_key or _secret_env("API_KEY", ""),
                    "secret": api_secret or _secret_env("API_SECRET", ""),
                    "password": password or _secret_env("API_PASSWORD", None),
                    "enableRateLimit": True,
                    "timeout": SAFE_TIMEOUT,
                }
                config = {k: v for k, v in config.items() if v is not None}
                self.ex = klass(config)
                if hasattr(self.ex, "load_markets"):
                    self.ex.load_markets()
            except Exception:
                self.ex = None
                self.mock_mode = True

    def symbols_usdt(self, max_items: int = 500) -> List[str]:
        majors_fx = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD"]
        if self.mock_mode or not self.ex:
            base = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT","TON/USDT","BNB/USDT"]
            return (base + majors_fx)[:max_items]
        syms = []
        try:
            for s in self.ex.symbols:
                if "/USDT" in s or s in majors_fx:
                    syms.append(s)
        except Exception:
            syms = []
        return syms[:max_items] if syms else (["BTC/USDT","ETH/USDT"] + majors_fx)[:max_items]

    @backoff_retry(max_retries=4)
    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
        if self.mock_mode or not self.ex:
            idx = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=limit, freq=_tf_to_freq(timeframe))
            base = np.cumsum(np.random.normal(0, 0.25, size=limit)) + 100.0
            drift = np.linspace(0, np.random.uniform(-2, 2), limit)
            close = base + drift
            high = close + np.abs(np.random.normal(0.35, 0.15, size=limit))
            low = close - np.abs(np.random.normal(0.35, 0.15, size=limit))
            open_ = close + np.random.normal(0, 0.15, size=limit)
            vol = np.abs(np.random.normal(12, 4, size=limit))
            df = pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
            return df
        data = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=int(limit))
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        return df

    @backoff_retry(max_retries=3)
    def get_orderbook(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        if self.mock_mode or not self.ex:
            px = 100.0 + np.random.randn() * 0.5
            bids = [[px - i*0.05, 1.0 + np.random.rand()] for i in range(limit)]
            asks = [[px + i*0.05, 1.0 + np.random.rand()] for i in range(limit)]
            return {"bids": bids, "asks": asks}
        return self.ex.fetch_order_book(symbol, limit=limit)

    def fetch_tickers(self) -> Dict[str, Any]:
        if self.mock_mode or not self.ex:
            return {}
        try:
            return self.ex.fetch_tickers()
        except Exception:
            return {}


# ---------------------------
# Analysis & Signal
# ---------------------------
def _prep_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ✅ Drop previous indicators to avoid duplicate column error
    drop_cols = [
        "EMA_FAST","EMA_SLOW","RSI","MACD","MACD_SIGNAL","MACD_HIST","ATR",
        "BB_MID","BB_UP","BB_LO","BB_WIDTH","BB_PB",
        "PRICE_SLOPE","PRICE_ACCEL","MACD_SLOPE","MACD_ACCEL"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "timestamp" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "timestamp"}, inplace=True)

    if not df.index.is_monotonic_increasing:
        df.sort_values("timestamp", inplace=True)

    # Recalculate indicators
    df["EMA_FAST"] = ema(df["close"], 20)
    df["EMA_SLOW"] = ema(df["close"], 50)
    df["RSI"] = rsi(df["close"], 14)
    macd_line, macd_sig, macd_hist = macd(df["close"])
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = macd_sig
    df["MACD_HIST"] = macd_hist
    df["ATR"] = atr(df, 14)
    bb_mid, bb_up, bb_lo, bb_w, bb_pb = bollinger(df["close"], 20, 2.0)
    df["BB_MID"], df["BB_UP"], df["BB_LO"], df["BB_WIDTH"], df["BB_PB"] = bb_mid, bb_up, bb_lo, bb_w, bb_pb
    df["PRICE_SLOPE"] = slope(df["close"], 5)
    df["PRICE_ACCEL"] = acceleration(df["close"], 5)
    df["MACD_SLOPE"] = slope(df["MACD_HIST"], 5)
    df["MACD_ACCEL"] = acceleration(df["MACD_HIST"], 5)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def _logistic(x: float) -> float:
    x = float(np.clip(x, -12.0, 12.0))
    return 1.0 / (1.0 + np.exp(-x))

def _prob_from_row(row: pd.Series, ob_imb: float) -> float:
    ema_trend = 1.0 if row["EMA_FAST"] > row["EMA_SLOW"] else 0.0
    rsi_dev = (row["RSI"] - 50.0) / 50.0
    bb_pos = (row["BB_PB"] - 0.5) * 2.0
    macd_m = np.tanh(float(row["MACD_HIST"]) * 3.0)
    p_slope = np.tanh(float(row["PRICE_SLOPE"]))
    p_accel = np.tanh(float(row["PRICE_ACCEL"]))
    m_slope = np.tanh(float(row["MACD_SLOPE"]))
    m_accel = np.tanh(float(row["MACD_ACCEL"]))
    ob_center = (float(ob_imb) - 0.5) * 2.0

    z = (
        0.35*ema_trend + 0.25*rsi_dev + 0.20*bb_pos + 0.20*macd_m +
        0.18*p_slope + 0.12*p_accel + 0.15*m_slope + 0.10*m_accel +
        0.28*ob_center
    )

    width_penalty = float(np.clip(row["BB_WIDTH"], 0.0, 2.0))
    z *= (1.0 - 0.12 * width_penalty)
    return _logistic(z)

def analyze_and_signal(df: pd.DataFrame, orderbook: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    df = _prep_indicators(df)
    ob_imb = orderbook_imbalance(orderbook) if orderbook else 0.5
    p_up = _prob_from_row(df.iloc[-1], ob_imb)
    conf = int(np.clip(abs(p_up - 0.5) * 200, 0, 100))

    r = df.iloc[-1]
    long_ok = (r["EMA_FAST"] > r["EMA_SLOW"]) and (r["RSI"] < 60) and (r["MACD_HIST"] > 0) and (p_up > 0.55)
    short_ok = (r["EMA_FAST"] < r["EMA_SLOW"]) and (r["RSI"] > 40) and (r["MACD_HIST"] < 0) and (p_up < 0.45)

    if long_ok and not short_ok:
        side = "BUY"
    elif short_ok and not long_ok:
        side = "SELL"
    else:
        side = "HOLD"

    return {
        "time": _now_iso(),
        "side": side,
        "price": float(r["close"]),
        "rsi": float(r["RSI"]),
        "macd_hist": float(r["MACD_HIST"]),
        "ema_fast": float(r["EMA_FAST"]),
        "ema_slow": float(r["EMA_SLOW"]),
        "atr": float(r["ATR"]),
        "prob_up": float(p_up),
        "orderbook_imbalance": float(ob_imb),
        "confidence": int(conf),
        "df": df,
    }


# ---------------------------
# Backtest
# ---------------------------
def backtest(df: pd.DataFrame) -> Dict[str, Any]:
    df = _prep_indicators(df)
    if df.empty or len(df) < 3:
        return {"total_trades": 0, "buy": 0, "sell": 0, "pnl_proxy": 0.0}

    ob_imb = 0.5
    probs = []
    for _, row in df.iterrows():
        probs.append(_prob_from_row(row, ob_imb))
    df["prob_up"] = np.clip(probs, 0.0, 1.0)
    df["signal"] = "HOLD"
    df.loc[df["prob_up"] >= 0.58, "signal"] = "BUY"
    df.loc[df["prob_up"] <= 0.42, "signal"] = "SELL"

    closes = df["close"].values
    sigs = df["signal"].values
    pnl = 0.0
    buy = sell = 0
    for i in range(len(df) - 1):
        if sigs[i] == "BUY":
            pnl += closes[i + 1] - closes[i]; buy += 1
        elif sigs[i] == "SELL":
            pnl += closes[i] - closes[i + 1]; sell += 1

    return {"total_trades": int(buy + sell), "buy": int(buy), "sell": int(sell), "pnl_proxy": float(pnl)}


# ---------------------------
# Telegram
# ---------------------------
def send_telegram(text: str, token: Optional[str], chat_id: Optional[str]) -> bool:
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False
