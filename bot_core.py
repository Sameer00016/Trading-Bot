\
import os, time, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False

TRADE_LOG = "trade_history.csv"

class TradingCore:
    def __init__(self, tele_token=None, tele_chat=None):
        self.tele_token = tele_token
        self.tele_chat = tele_chat

    def get_ohlcv(self, source, symbol, timeframe="1m", limit=200):
        # Try using ccxt if available for exchanges
        if CCXT_AVAILABLE and source != "mock":
            try:
                ex = getattr(ccxt, source)()
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                # fallback to mock
                print("ccxt fetch failed:", e)
        # Mock data generation (reproducible-ish)
        now = int(time.time()*1000)
        prices = np.cumsum(np.random.randn(limit)) + 100
        volumes = (np.random.rand(limit) + 0.1) * 10
        rows = []
        for i,(p,v) in enumerate(zip(prices, volumes)):
            ts = now - (limit-1-i)*60000
            rows.append([ts, p-1, p+1, p-2, p, v])
        df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def analyze(self, df):
        df = df.copy().reset_index(drop=True)
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd_diff()
        df['ema'] = EMAIndicator(df['close'], window=9).ema_indicator()
        return df

    def probability_of_move(self, df):
        # simple volume-based heuristic
        recent = df.tail(10)
        buy_vol = recent['volume'].sum() * 0.5 + recent['volume'].diff().clip(lower=0).sum()*0.1
        sell_vol = recent['volume'].sum() - buy_vol
        p_up = buy_vol / (buy_vol + sell_vol + 1e-9)
        return float(p_up)

    def ml_signal(self, df):
        # very simple trend heuristic: slope of 5-period MA
        ma_short = df['close'].rolling(5).mean()
        slope = ma_short.iloc[-1] - ma_short.iloc[-5] if len(ma_short) > 5 else 0
        return 'BUY' if slope>0 else 'SELL' if slope<0 else 'WAIT'

    def make_trade_plan(self, df, signal):
        price = float(df['close'].iloc[-1])
        atr = (df['high'] - df['low']).tail(14).mean() if 'high' in df.columns else 0.5
        sl = price - atr*1.5 if signal=='BUY' else price + atr*1.5 if signal=='SELL' else None
        tp = price + atr*3 if signal=='BUY' else price - atr*3 if signal=='SELL' else None
        next_entry = price + atr*0.5 if signal=='BUY' else price - atr*0.5 if signal=='SELL' else None
        # time window: suggest to close within next X minutes/hours
        suggested_close_time = (datetime.utcnow() + timedelta(minutes=30)).isoformat() + "Z"
        return {"price":price,"stop_loss":sl,"take_profit":tp,"next_entry":next_entry,"close_by":suggested_close_time}

    def generate_signal(self, df):
        df2 = self.analyze(df)
        rsi = df2['rsi'].iloc[-1]
        macd = df2['macd'].iloc[-1]
        ema = df2['ema'].iloc[-1]
        p_up = self.probability_of_move(df2)
        ml = self.ml_signal(df2)
        roc = (df2['close'].iloc[-1] - df2['close'].iloc[-2]) if len(df2)>=2 else 0

        # combine rules (configurable)
        if ml=='BUY' and rsi<40 and macd>0 and df2['close'].iloc[-1]>ema and p_up>0.55 and roc>0:
            side='BUY'
        elif ml=='SELL' and rsi>60 and macd<0 and df2['close'].iloc[-1]<ema and p_up<0.45 and roc<0:
            side='SELL'
        else:
            side='WAIT'

        plan = self.make_trade_plan(df2, side)
        signal = {
            "time": datetime.utcnow().isoformat() + "Z",
            "side": side,
            "price": float(df2['close'].iloc[-1]),
            "rsi": float(rsi),
            "macd": float(macd),
            "ema": float(ema),
            "prob_up": float(p_up),
            "roc": float(roc),
            "plan": plan
        }
        return signal

    def process_dataframe_and_signal(self, df, pair, exchange):
        sig = self.generate_signal(df)
        sig['pair'] = pair
        sig['exchange'] = exchange
        return sig

    def send_telegram(self, signal):
        if not self.tele_token or not self.tele_chat:
            # user may have values in self.tele_token/tele_chat or environment; try environment
            token = os.getenv("TELEGRAM_TOKEN")
            chat = os.getenv("TELEGRAM_CHAT_ID")
        else:
            token = self.tele_token
            chat = self.tele_chat
        if not token or not chat:
            print("Telegram credentials not set; cannot send.")
            return False
        text = f\"[{signal['time']}] {signal['pair']} {signal['side']} @ {signal['price']:.4f}\\nSL: {signal['plan']['stop_loss']:.4f} TP: {signal['plan']['take_profit']:.4f} CloseBy: {signal['plan']['close_by']}\"
        url = f\"https://api.telegram.org/bot{token}/sendMessage\"
        try:
            r = requests.post(url, data={\"chat_id\": chat, \"text\": text}, timeout=10)
            return r.status_code==200
        except Exception as e:
            print(\"Telegram error:\", e)
            return False

    def log_signal(self, signal):
        # append to CSV
        df = pd.DataFrame([signal])
        if not os.path.exists(TRADE_LOG):
            df.to_csv(TRADE_LOG, index=False)
        else:
            df.to_csv(TRADE_LOG, mode='a', header=False, index=False)

    def read_history(self, limit=100):
        if os.path.exists(TRADE_LOG):
            df = pd.read_csv(TRADE_LOG)
            return df.tail(limit)
        return pd.DataFrame(columns=[\"time\",\"side\",\"price\",\"rsi\",\"macd\",\"ema\",\"prob_up\",\"roc\",\"plan\",\"pair\",\"exchange\"])